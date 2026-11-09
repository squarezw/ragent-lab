"""Vision model catalog. Every model is served through AIHubMix with the server's key.

Model names were checked against https://aihubmix.com/v1/models and each model was tested on
the sample drawings (2026-11-09). Claude goes through the native Anthropic Messages endpoint,
the rest through Chat Completions with a JSON schema. Visitors never see which platform serves
a model; it shows up only in the run's meta (exported JSON) and the API log.
"""
from __future__ import annotations


def _m(id, name, family, route, max_tokens=32000, notes=''):
    return {'id': id, 'name': name, 'family': family, 'max_tokens': max_tokens,
            'temperature': None, 'reasoning': None, 'notes': notes, 'route': route}


def _chat(model: str, **extra) -> dict:
    return {'api': 'chat', 'model': model, 'json': 'schema', **extra}


CLAUDE_ROUTE = {'api': 'anthropic', 'model': 'claude-opus-5-5', 'json': 'prompt'}

VISION_MODELS: list[dict] = [
    _m('anthropic/claude-opus-5.5', 'Claude Opus 5.5', 'anthropic', CLAUDE_ROUTE),
    _m('openai/gpt-6-astra', 'GPT-6 Astra', 'openai', _chat('gpt-6-astra')),
    # Gemini's schema validator needs anyOf instead of oneOf and no length/count limits.
    _m('google/gemini-4-argon', 'Gemini 4 Argon', 'google',
       _chat('gemini-4-argon', lean_schema=True)),
    _m('z-ai/glm-5v-turbo', 'GLM-5V Turbo', 'zhipu', _chat('glm-5v-turbo')),
    # Thinking models: 32k output tokens ran out on the bracket drawing.
    _m('moonshotai/kimi-k3.1-pro', 'Kimi K3.1 Pro', 'moonshot', _chat('kimi-k3.1-pro'),
       max_tokens=64000),
    _m('bytedance-seed/seed-2-1-turbo', 'Seed 2.1 Turbo', 'bytedance',
       _chat('doubao-seed-2-1-turbo')),
    _m('minimax/minimax-m3', 'MiniMax M3', 'minimax', _chat('minimax-m3'), max_tokens=64000),
]

JUDGE_MODEL: dict = VISION_MODELS[0]


def get_model(model_id: str) -> dict:
    for m in VISION_MODELS:
        if m['id'] == model_id:
            return m
    raise KeyError(model_id)


def judge_model(model_id: str | None = None) -> dict:
    """Judge for scoring and consensus: Claude by default, or any catalog model."""
    if model_id is None:
        return JUDGE_MODEL
    return get_model(model_id)
