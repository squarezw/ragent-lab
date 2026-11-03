"""Execution and validation independent of presentation and transport."""
import math
import os
import time
from contextlib import contextmanager
from threading import Lock

from fastapi import HTTPException
from ..chunking.registry import get_all_strategies
from ..embedding.registry import get_all_models, get_model
from .schemas import ChunkRequest, EmbeddingRequest

_EXECUTION_LOCK = Lock()
_HEAVY_MODELS = {'e5_mistral_7b_instruct', 'gme_qwen2_vl_2b_instruct'}
PARAM_BOUNDS = {
    'chunk_size': (1, 10000), 'chunk_overlap': (0, 9999),
    'window_size': (1, 100), 'stride': (1, 100), 'threshold': (0, 1),
    'text_length': (1, 50000), 'chunk_len': (1, 50000),
}


@contextmanager
def execution_slot():
    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise HTTPException(429, '实验正在运行，请稍后再试。')
    try:
        yield
    finally:
        _EXECUTION_LOCK.release()


def strategy_catalog():
    return [dict(id=name, name=name, description=info['desc'], rating=info['rating'],
                 params=[dict(p, **({'min': PARAM_BOUNDS[p['name']][0], 'max': PARAM_BOUNDS[p['name']][1]} if p['name'] in PARAM_BOUNDS else {})) for p in info['params']])
            for name, info in get_all_strategies().items()]


def model_catalog():
    heavy_enabled = os.getenv('ENABLE_HEAVY_MODELS', 'false').lower() == 'true'
    return [dict(id=name, name=info['model'].name, description=info['desc'],
                 dimension=info['model'].dimension, provider='aliyun' if name.startswith('aliyun') else 'openai' if name.startswith('openai') else 'local',
                 enabled=name not in _HEAVY_MODELS or heavy_enabled)
            for name, info in get_all_models().items()]


def chunk_text(request: ChunkRequest):
    info = get_all_strategies().get(request.strategy)
    if info is None:
        raise HTTPException(422, '未知分块策略')
    specs = {p['name']: p for p in info['params']}
    if set(request.params) - set(specs):
        raise HTTPException(422, '包含不支持的策略参数')
    params = {}
    for name, spec in specs.items():
        value = request.params.get(name, spec['default'])
        if spec['type'] == 'str':
            if not isinstance(value, str) or len(value) > 500:
                raise HTTPException(422, '关键词长度不能超过 500 个字符')
            params[name] = [v.strip() for v in value.split(',') if v.strip()]
        else:
            expected = int if spec['type'] == 'int' else (int, float)
            low, high = PARAM_BOUNDS[name]
            if isinstance(value, bool) or not isinstance(value, expected) or not low <= value <= high:
                raise HTTPException(422, f'{name} 必须在 {low}–{high} 之间')
            params[name] = value
    if 'chunk_overlap' in params and params['chunk_overlap'] >= params['chunk_size']:
        raise HTTPException(422, '重叠长度必须小于分块长度')
    size = params.get('chunk_size', 0)
    overlap = params.get('chunk_overlap', 0)
    if size and (math.ceil(len(request.text) / (size - overlap)) > 1000 or math.ceil(len(request.text) / (size - overlap)) * size > 300000):
        raise HTTPException(422, '预计分块输出过多，请增加分块长度或减少重叠。')
    if request.strategy == '滑动窗口分块' and len(request.text) * math.ceil(params['window_size'] / params['stride']) > 300000:
        raise HTTPException(422, '窗口重复内容过多，请缩小窗口或增加步长。')
    if request.strategy in {'语义分块', '混合分块'} and len(request.text) > 12000:
        raise HTTPException(422, '语义实验请使用不超过 12000 个字符的文本')
    start = time.perf_counter()
    with execution_slot():
        try:
            raw = info['func'](request.text, **params)
        except Exception:
            raise HTTPException(502, '分块执行失败，请检查模型缓存及服务配置。') from None
    chunks = [dict(content=c) if isinstance(c, str) else c for c in raw]
    lengths = [len(c['content']) for c in chunks]
    if len(chunks) > 1000 or sum(lengths) > 300000:
        raise HTTPException(422, '结果超过 1000 块或 300000 字符，请缩短输入或调整参数。')
    return {'strategy': request.strategy, 'chunks': chunks,
            'stats': {'count': len(chunks), 'total_chars': sum(lengths),
                      'average_chars': round(sum(lengths) / len(lengths), 1) if lengths else 0,
                      'elapsed_ms': round((time.perf_counter() - start) * 1000, 1)}}


def embed_texts(request: EmbeddingRequest):
    entry = next((m for m in model_catalog() if m['id'] == request.model), None)
    if not entry:
        raise HTTPException(422, '未知模型')
    if not entry['enabled']:
        raise HTTPException(422, '该模型需要额外算力，当前服务器尚未启用。')
    key = request.api_key.get_secret_value().strip()
    if entry['provider'] != 'local' and not key:
        if os.getenv('ALLOW_SERVER_API_KEYS', 'false').lower() == 'true':
            key = os.getenv('Aliyun_API_KEY' if entry['provider'] == 'aliyun' else 'OPENAI_API_KEY', '')
        if not key:
            raise HTTPException(422, '请填写此模型的 API Key，仅用于本次请求。')
    query = request.query.strip()
    # Encode the query in the same call. Rendering/comparison never calls the provider.
    inputs = [*request.texts, *([query] if query else [])]
    start = time.perf_counter()
    with execution_slot():
        try:
            model = get_model(request.model)
            result = model.embed(inputs, **({'api_key': key} if key else {}))
            if len(result.embeddings) != len(inputs) or result.dimension == 0:
                raise ValueError('provider returned inconsistent data')
            if any(not math.isfinite(n) for v in result.embeddings for n in v):
                raise ValueError('provider returned non-finite data')
        except Exception:
            raise HTTPException(502, '向量生成失败，请检查 API Key、模型可用性或服务日志。') from None
    rankings = []
    if query:
        q = result.embeddings[-1]
        qnorm = math.sqrt(sum(x*x for x in q))
        for index, (text, vector) in enumerate(zip(request.texts, result.embeddings)):
            denom = qnorm * math.sqrt(sum(x*x for x in vector))
            score = sum(a*b for a, b in zip(q, vector)) / denom if denom else 0.
            rankings.append({'index': index, 'text': text, 'score': round(max(-1., min(1., score)), 6)})
        rankings.sort(key=lambda item: item['score'], reverse=True)
    return {'model': request.model, 'model_name': entry['name'], 'texts': request.texts,
            'query': query, 'dimension': result.dimension, 'count': len(request.texts),
            'rankings': rankings, 'elapsed_ms': round((time.perf_counter() - start) * 1000, 1)}
