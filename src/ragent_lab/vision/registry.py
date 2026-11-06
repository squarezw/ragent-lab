"""Vision model registry (supported_parameters checked against OpenRouter /models)."""
from __future__ import annotations

import time

import httpx

from . import openrouter

MODELS_URL = 'https://openrouter.ai/api/v1/models'
_PRICE_TTL = 6 * 3600
_price_cache: dict = {'at': 0.0, 'data': None}


def _m(id, name, family, response_format, max_tokens=32000, temperature=0.2,
       reasoning=None, provider=None, notes=''):
    return {'id': id, 'name': name, 'family': family, 'response_format': response_format,
            'max_tokens': max_tokens, 'temperature': temperature, 'reasoning': reasoning,
            'provider': provider, 'notes': notes}


_SCHEMA = 'supported_parameters 含 structured_outputs'
_ANTHROPIC_ONLY = {'order': ['anthropic'], 'allow_fallbacks': False}
VISION_MODELS: list[dict] = [
    _m('anthropic/claude-opus-5.5', 'Claude Opus 5.5', 'anthropic', 'json_schema',
       temperature=None, provider=_ANTHROPIC_ONLY,
       notes=_SCHEMA + '；不支持 temperature；锁定 Anthropic 服务商'),
    _m('openai/gpt-6-astra', 'GPT-6 Astra', 'openai', 'json_schema',
       temperature=None, notes=_SCHEMA + '；不支持 temperature'),
    _m('google/gemini-4-argon', 'Gemini 4 Argon', 'google', 'json_schema', notes=_SCHEMA),
    _m('qwen/qwen3.8-max-prime', 'Qwen3.8 Max Prime', 'qwen', 'json_schema', notes=_SCHEMA),
    _m('z-ai/glm-5v-turbo', 'GLM-5V Turbo', 'zhipu', 'json_object',
       notes='supported_parameters 仅含 response_format，无 structured_outputs'),
    _m('moonshotai/kimi-k3.1-pro', 'Kimi K3.1 Pro', 'moonshot', 'json_schema',
       temperature=None, notes=_SCHEMA + '；思考模型，不指定 temperature'),
    _m('bytedance-seed/seed-2-1-turbo', 'Seed 2.1 Turbo', 'bytedance', 'json_schema', notes=_SCHEMA),
    _m('deepseek/deepseek-v4-flash-vision-exp', 'DeepSeek V4 Flash Vision', 'deepseek',
       'json_schema', notes=_SCHEMA),
    _m('minimax/minimax-m3', 'MiniMax M3', 'minimax', 'json_schema',
       reasoning={'enabled': False}, notes=_SCHEMA + '；关闭思考'),
]

JUDGE_MODEL: dict = _m(
    'anthropic/claude-opus-5.5', 'Claude Opus 5.5', 'anthropic', 'json_schema',
    temperature=None, provider=_ANTHROPIC_ONLY,
    notes='判定模型，锁定 Anthropic 服务商')


def get_model(model_id: str) -> dict:
    for m in VISION_MODELS:
        if m['id'] == model_id:
            return m
    raise KeyError(model_id)


async def model_prices(client_factory=None) -> dict[str, dict | None]:
    now = time.time()
    if _price_cache['data'] is not None and now - _price_cache['at'] < _PRICE_TTL:
        return _price_cache['data']
    factory = client_factory or openrouter.new_client
    try:
        async with factory() as client:
            resp = await client.get(MODELS_URL)
        resp.raise_for_status()
        by_id = {x.get('id'): x for x in resp.json()['data']}
        result: dict[str, dict | None] = {}
        for m in VISION_MODELS:
            p = (by_id.get(m['id']) or {}).get('pricing')
            result[m['id']] = ({'prompt': p.get('prompt'), 'completion': p.get('completion')}
                               if isinstance(p, dict) else None)
    except Exception:
        return {m['id']: None for m in VISION_MODELS}
    _price_cache.update(at=now, data=result)
    return result
