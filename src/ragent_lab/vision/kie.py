"""Kie.ai client for vision experiments (JSON output).

Kie.ai has no single chat endpoint: each model family uses its own route and wire format
(see the `kie` entry in registry.py). Callers pass OpenAI chat-style messages, the same as
for OpenRouter; this module converts them and returns the same CallResult.
"""
from __future__ import annotations

import logging
import re
import time

import httpx

from .openrouter import CallResult, OpenRouterError, extract_json

log = logging.getLogger(__name__)

BASE_URL = 'https://api.kie.ai'
TIMEOUT_SECONDS = 600.0
JSON_ONLY = '\n\n只输出一个符合上述 Schema 的 JSON 对象，不要输出任何其他文字或代码块标记。'

# Tests inject httpx.MockTransport here.
transport: httpx.AsyncBaseTransport | None = None

_STATUS_ERRORS = {
    401: ('auth', 'Kie.ai Key 无效或已撤销'),
    402: ('credits', 'Kie.ai 积分不足'),
    429: ('rate_limit', '请求过于频繁或被 Kie.ai 限流，请稍后重试'),
}
_KEY_LIKE = re.compile(r'(sk-[A-Za-z0-9_-]{8,}|[A-Fa-f0-9]{32,})')
_DETAIL_MAX = 300


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT_SECONDS, transport=transport)


# ---- request conversion ---------------------------------------------------------------

def _parts(messages: list[dict]) -> list[dict]:
    """Flatten the single user message the service sends into its content parts."""
    parts: list[dict] = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, str):
            parts.append({'type': 'text', 'text': content})
        else:
            parts.extend(content or [])
    return parts


def _data_url(part: dict) -> str:
    return part['image_url']['url']


def _anthropic_body(route: dict, model: dict, messages: list[dict]) -> dict:
    content = []
    for p in _parts(messages):
        if p['type'] == 'image_url':
            header, _, data = _data_url(p).partition(',')
            media = header.removeprefix('data:').split(';', 1)[0] or 'image/png'
            content.append({'type': 'image',
                            'source': {'type': 'base64', 'media_type': media, 'data': data}})
        else:
            content.append({'type': 'text', 'text': p['text']})
    # No structured-output option on this route: the prompt carries the schema, ask for JSON only.
    content.append({'type': 'text', 'text': JSON_ONLY})
    return {'model': route['model'], 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': model['max_tokens'], 'stream': False, 'thinkingFlag': False}


def _responses_body(route: dict, messages: list[dict], schema_name: str, schema: dict) -> dict:
    content = [{'type': 'input_image', 'image_url': _data_url(p)} if p['type'] == 'image_url'
               else {'type': 'input_text', 'text': p['text']} for p in _parts(messages)]
    return {'model': route['model'], 'stream': False,
            'input': [{'role': 'user', 'content': content}],
            'text': {'format': {'type': 'json_schema', 'name': schema_name,
                                'strict': False, 'schema': schema}}}


def _chat_body(route: dict, model: dict, messages: list[dict], schema_name: str,
               schema: dict) -> dict:
    body: dict = {'messages': messages, 'stream': False, 'max_tokens': model['max_tokens'],
                  'response_format': {'type': 'json_schema', 'json_schema': {
                      'name': schema_name, 'strict': False, 'schema': schema}}}
    if route.get('model'):
        body['model'] = route['model']
    return body


# ---- response parsing -----------------------------------------------------------------

def _anthropic_text(data: dict) -> tuple[str, str | None, bool]:
    text = ''.join(b.get('text', '') for b in data.get('content') or []
                   if isinstance(b, dict) and b.get('type') == 'text')
    finish = data.get('stop_reason')
    return text, finish, finish == 'max_tokens'


def _responses_text(data: dict) -> tuple[str, str | None, bool]:
    text = ''
    for item in data.get('output') or []:
        if isinstance(item, dict) and item.get('type') == 'message':
            text += ''.join(c.get('text', '') for c in item.get('content') or []
                            if isinstance(c, dict) and c.get('type') == 'output_text')
    status = data.get('status')
    reason = (data.get('incomplete_details') or {}).get('reason')
    return text, status, status == 'incomplete' and reason in (None, 'max_output_tokens')


def _chat_text(data: dict) -> tuple[str, str | None, bool]:
    choice = data['choices'][0]
    text = (choice.get('message') or {}).get('content') or ''
    if isinstance(text, list):
        text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
    finish = choice.get('finish_reason')
    return text, finish, finish == 'length'


_PARSERS = {'anthropic': _anthropic_text, 'responses': _responses_text, 'chat': _chat_text}


# ---- errors ---------------------------------------------------------------------------

def _redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, '***')
    return _KEY_LIKE.sub('***', text)[:_DETAIL_MAX]


def _error_detail(data) -> str:
    if not isinstance(data, dict):
        return ''
    err = data.get('error')
    if isinstance(err, dict):
        return str(err.get('message') or err.get('type') or '')
    if isinstance(err, str):
        return err
    return str(data.get('msg') or data.get('message') or '')


def _status_error(code: int, detail: str, api_key: str, model_id: str) -> OpenRouterError:
    detail = _redact(detail.strip(), api_key)
    kind, msg = _STATUS_ERRORS.get(code, ('upstream', f'Kie.ai 拒绝了请求（{code}）'))
    log.warning('kie %s code %s: %s', model_id, code, detail or '-')
    return OpenRouterError(kind, f'{msg}：{detail}' if detail else f'{msg}。')


def _envelope_code(data: dict) -> int | None:
    """Kie.ai may answer HTTP 200 with {"code": 4xx, "msg": ...} instead of a result."""
    code = data.get('code')
    if isinstance(code, int) and code != 200 and not any(
            k in data for k in ('choices', 'output', 'content')):
        return code
    return None


# ---- call -----------------------------------------------------------------------------

async def chat_json(api_key: str, model: dict, messages: list[dict],
                    schema_name: str, schema: dict) -> CallResult:
    route = model.get('kie')
    if not route:
        raise OpenRouterError('upstream', f'{model["name"]} 不支持 Kie.ai。')
    api = route['api']
    if api == 'anthropic':
        body = _anthropic_body(route, model, messages)
    elif api == 'responses':
        body = _responses_body(route, messages, schema_name, schema)
    else:
        body = _chat_body(route, model, messages, schema_name, schema)

    started = time.perf_counter()
    try:
        async with new_client() as client:
            resp = await client.post(route['path'], json=body,
                                     headers={'Authorization': f'Bearer {api_key}'})
    except httpx.TimeoutException:
        raise OpenRouterError('timeout', '模型调用超时。') from None
    except httpx.HTTPError as exc:
        raise OpenRouterError('upstream', f'网络错误：{type(exc).__name__}。') from None
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        data = resp.json()
    except ValueError:
        data = None
    if resp.status_code >= 400:
        raise _status_error(resp.status_code, _error_detail(data), api_key, model['id'])
    if not isinstance(data, dict):
        raise OpenRouterError('upstream', 'Kie.ai 返回了无法识别的响应。')
    code = _envelope_code(data)
    if code is not None:
        raise _status_error(code, _error_detail(data), api_key, model['id'])

    try:
        text, finish, truncated = _PARSERS[api](data)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise OpenRouterError('upstream', 'Kie.ai 返回了无法识别的响应。') from None
    if truncated:
        raise OpenRouterError('truncated', '输出被截断（达到 max_tokens）。')
    try:
        content = extract_json(text)
    except ValueError:
        raise OpenRouterError('invalid_output', '模型输出无法解析为 JSON。') from None
    credits = data.get('credits_consumed')
    return CallResult(
        content=content, raw_text=text, model=data.get('model') or route.get('model') or model['id'],
        provider='Kie.ai', finish_reason=finish, usage=data.get('usage') or {},
        cost=None, elapsed_ms=elapsed_ms,
        credits=float(credits) if isinstance(credits, (int, float)) else None,
    )

