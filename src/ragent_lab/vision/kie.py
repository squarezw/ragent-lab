"""Kie.ai: one route per model family, all called through the generic compat client.

Kie.ai has no single chat endpoint: each family uses its own route and wire format (see the
`kie` entry in registry.py). Its Claude and Gemini routes take no usable schema option
(Gemini's validator reads our property named "type" as the JSON-schema keyword and answers
422), so those ask for JSON in the prompt; the Responses routes take the schema.
"""
from __future__ import annotations

from .compat import call_endpoint
from .openrouter import CallResult, OpenRouterError

BASE_URL = 'https://api.kie.ai'


def endpoint(route: dict) -> dict:
    ep = {'api': route['api'], 'url': BASE_URL + route['path'], 'model': route.get('model'),
          'json': 'schema' if route['api'] == 'responses' else 'prompt', 'auth': 'bearer',
          'label': 'Kie.ai'}
    if route['api'] == 'anthropic':
        ep['extra'] = {'thinkingFlag': False}
    return ep


async def chat_json(api_key: str, model: dict, messages: list[dict],
                    schema_name: str, schema: dict) -> CallResult:
    route = model.get('kie')
    if not route:
        raise OpenRouterError('upstream', f'{model["name"]} 不支持 Kie.ai。')
    result = await call_endpoint(api_key, endpoint(route), model, messages, schema_name, schema)
    result.provider = 'Kie.ai'
    return result
