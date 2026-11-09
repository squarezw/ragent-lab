"""Vision model registry (supported_parameters checked against OpenRouter /models)."""
from __future__ import annotations

import time

import httpx

from . import openrouter

MODELS_URL = 'https://openrouter.ai/api/v1/models'
_PRICE_TTL = 6 * 3600
_price_cache: dict = {'at': 0.0, 'data': None}


GATEWAYS = ('openrouter', 'kie', 'aihubmix')


def _m(id, name, family, response_format, max_tokens=32000, temperature=0.2,
       reasoning=None, provider=None, notes='', kie=None):
    return {'id': id, 'name': name, 'family': family, 'response_format': response_format,
            'max_tokens': max_tokens, 'temperature': temperature, 'reasoning': reasoning,
            'provider': provider, 'notes': notes, 'kie': kie}


def _kie(api: str, path: str, model: str | None = None) -> dict:
    """Kie.ai route. api: 'anthropic' (Messages), 'responses' (OpenAI Responses) or 'chat'."""
    return {'api': api, 'path': path, 'model': model}


# Kie.ai docs: https://docs.kie.ai/llms.txt (Chat Models). Each family has its own endpoint.
_KIE_CLAUDE = _kie('anthropic', '/claude/v1/messages', 'claude-opus-5-5')


_SCHEMA = 'supported_parameters 含 structured_outputs'
_ANTHROPIC_ONLY = {'order': ['anthropic'], 'allow_fallbacks': False}
VISION_MODELS: list[dict] = [
    _m('anthropic/claude-opus-5.5', 'Claude Opus 5.5', 'anthropic', 'json_schema',
       temperature=None, provider=_ANTHROPIC_ONLY,
       notes=_SCHEMA + '；不支持 temperature；锁定 Anthropic 服务商', kie=_KIE_CLAUDE),
    _m('openai/gpt-6-astra', 'GPT-6 Astra', 'openai', 'json_schema',
       temperature=None, notes=_SCHEMA + '；不支持 temperature',
       kie=_kie('responses', '/codex/v1/responses', 'gpt-6-astra')),
    _m('google/gemini-4-argon', 'Gemini 4 Argon', 'google', 'json_schema', notes=_SCHEMA,
       kie=_kie('chat', '/gemini-4-argon-openai/v1/chat/completions')),
    _m('qwen/qwen3.8-max-prime', 'Qwen3.8 Max Prime', 'qwen', 'json_schema', notes=_SCHEMA),
    _m('z-ai/glm-5v-turbo', 'GLM-5V Turbo', 'zhipu', 'json_object',
       notes='supported_parameters 仅含 response_format，无 structured_outputs'),
    _m('moonshotai/kimi-k3.1-pro', 'Kimi K3.1 Pro', 'moonshot', 'json_schema',
       temperature=None, notes=_SCHEMA + '；思考模型，不指定 temperature',
       kie=_kie('responses', '/openai/v1/responses', 'kimi-k3-1-pro')),
    _m('bytedance-seed/seed-2-1-turbo', 'Seed 2.1 Turbo', 'bytedance', 'json_schema', notes=_SCHEMA),
    _m('minimax/minimax-m3', 'MiniMax M3', 'minimax', 'json_schema',
       reasoning={'enabled': False}, notes=_SCHEMA + '；关闭思考'),
]

JUDGE_MODEL: dict = _m(
    'anthropic/claude-opus-5.5', 'Claude Opus 5.5', 'anthropic', 'json_schema',
    temperature=None, provider=_ANTHROPIC_ONLY,
    notes='判定模型，锁定 Anthropic 服务商', kie=_KIE_CLAUDE)

# AIHubMix (https://aihubmix.com): one key for every family. Claude goes through its native
# Messages endpoint; the rest through Chat Completions with a JSON schema. Model names checked
# against /v1/models and tested on the sample drawings (2026-11-09). Qwen3.8 Max Prime is not
# offered there (only Qwen3.8 Max), so it stays OpenRouter-only.
AIHUBMIX = {
    'anthropic/claude-opus-5.5': {'api': 'anthropic', 'model': 'claude-opus-5-5', 'json': 'prompt'},
    'openai/gpt-6-astra': {'api': 'chat', 'model': 'gpt-6-astra', 'json': 'schema'},
    # Gemini's schema validator needs anyOf instead of oneOf and no length/count limits.
    'google/gemini-4-argon': {'api': 'chat', 'model': 'gemini-4-argon', 'json': 'schema',
                              'lean_schema': True},
    'z-ai/glm-5v-turbo': {'api': 'chat', 'model': 'glm-5v-turbo', 'json': 'schema'},
    'moonshotai/kimi-k3.1-pro': {'api': 'chat', 'model': 'kimi-k3.1-pro', 'json': 'schema'},
    'bytedance-seed/seed-2-1-turbo': {'api': 'chat', 'model': 'doubao-seed-2-1-turbo',
                                      'json': 'schema'},
    'minimax/minimax-m3': {'api': 'chat', 'model': 'minimax-m3', 'json': 'schema'},
}
for _model in [*VISION_MODELS, JUDGE_MODEL]:
    _model['aihubmix'] = AIHUBMIX.get(_model['id'])


def get_model(model_id: str) -> dict:
    for m in VISION_MODELS:
        if m['id'] == model_id:
            return m
    raise KeyError(model_id)


def gateways_of(model: dict) -> list[str]:
    return (['openrouter'] + (['kie'] if model.get('kie') else [])
            + (['aihubmix'] if model.get('aihubmix') else []))


def judge_model(model_id: str | None = None) -> dict:
    """Judge for scoring and consensus: Claude by default, or any registered model."""
    if model_id is None or model_id == JUDGE_MODEL['id']:
        return JUDGE_MODEL
    return get_model(model_id)


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
