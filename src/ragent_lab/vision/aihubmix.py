"""AIHubMix: one key for many model families, called through the generic compat client.

Routes live in registry.AIHUBMIX: Claude uses the native Anthropic Messages endpoint
(x-api-key), everything else OpenAI Chat Completions.
"""
from __future__ import annotations

from .compat import call_endpoint
from .openrouter import CallResult, OpenRouterError

BASE_URL = 'https://aihubmix.com'
PATHS = {'chat': '/v1/chat/completions', 'anthropic': '/v1/messages'}


def endpoint(route: dict) -> dict:
    anthropic = route['api'] == 'anthropic'
    return {'api': route['api'], 'url': BASE_URL + PATHS[route['api']], 'model': route['model'],
            'json': route.get('json', 'prompt' if anthropic else 'schema'),
            'auth': 'x-api-key' if anthropic else 'bearer', 'label': 'AIHubMix',
            'lean_schema': bool(route.get('lean_schema'))}


async def chat_json(api_key: str, model: dict, messages: list[dict],
                    schema_name: str, schema: dict) -> CallResult:
    route = model.get('aihubmix')
    if not route:
        raise OpenRouterError('upstream', f'{model["name"]} 不支持 AIHubMix。')
    result = await call_endpoint(api_key, endpoint(route), model, messages, schema_name, schema)
    result.provider = 'AIHubMix'
    return result
