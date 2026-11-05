"""Minimal OpenRouter chat client for vision experiments (JSON output)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Literal

import httpx

URL = 'https://openrouter.ai/api/v1/chat/completions'
TIMEOUT_SECONDS = 600.0

# Tests inject httpx.MockTransport here.
transport: httpx.AsyncBaseTransport | None = None

ErrorKind = Literal['auth', 'credits', 'rate_limit', 'upstream', 'timeout', 'truncated', 'invalid_output']


class OpenRouterError(Exception):
    def __init__(self, kind: ErrorKind, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class CallResult:
    content: dict
    raw_text: str
    model: str
    provider: str | None
    finish_reason: str | None
    usage: dict
    cost: float | None
    elapsed_ms: float


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS, transport=transport)


def extract_json(text: str) -> dict:
    text = re.sub(r'```[a-zA-Z]*', '', text)
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object found')
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError('unbalanced JSON object')
    chunk = text[start:end + 1]
    try:
        value = json.loads(chunk)
    except json.JSONDecodeError:
        try:
            value = json.loads(re.sub(r',\s*([}\]])', r'\1', chunk))
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON: {exc.msg}') from None
    if not isinstance(value, dict):
        raise ValueError('JSON root is not an object')
    return value


def _body(model: dict, messages: list[dict], schema_name: str, schema: dict) -> dict:
    body: dict = {
        'model': model['id'],
        'messages': messages,
        'max_tokens': model['max_tokens'],
        'usage': {'include': True},
    }
    if model.get('temperature') is not None:
        body['temperature'] = model['temperature']
    if model.get('reasoning') is not None:
        body['reasoning'] = model['reasoning']
    if model.get('provider') is not None:
        body['provider'] = model['provider']
    fmt = model.get('response_format')
    if fmt == 'json_schema':
        body['response_format'] = {
            'type': 'json_schema',
            'json_schema': {'name': schema_name, 'strict': True, 'schema': schema},
        }
    elif fmt == 'json_object':
        body['response_format'] = {'type': 'json_object'}
    return body


_STATUS_ERRORS = {
    401: ('auth', 'OpenRouter Key 无效或已撤销。'),
    403: ('auth', 'OpenRouter Key 无效或已撤销。'),
    402: ('credits', 'OpenRouter 余额不足。'),
    429: ('rate_limit', '请求过于频繁或被上游限流，请稍后重试。'),
}


async def chat_json(api_key: str, model: dict, messages: list[dict],
                    schema_name: str, schema: dict) -> CallResult:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://lab.ragents.net',
        'X-Title': 'Ragent Lab',
    }
    started = time.perf_counter()
    try:
        async with new_client() as client:
            resp = await client.post(URL, headers=headers,
                                     json=_body(model, messages, schema_name, schema))
    except httpx.TimeoutException:
        raise OpenRouterError('timeout', '模型调用超时。') from None
    except httpx.HTTPError as exc:
        raise OpenRouterError('upstream', f'网络错误：{type(exc).__name__}。') from None
    elapsed_ms = (time.perf_counter() - started) * 1000

    if resp.status_code in _STATUS_ERRORS:
        kind, msg = _STATUS_ERRORS[resp.status_code]
        raise OpenRouterError(kind, msg)
    if resp.status_code >= 400:
        raise OpenRouterError('upstream', f'上游服务出错（HTTP {resp.status_code}）。')

    try:
        data = resp.json()
        choice = data['choices'][0]
    except (ValueError, KeyError, IndexError, TypeError):
        raise OpenRouterError('upstream', '上游返回了无法识别的响应。') from None
    if not isinstance(choice, dict):
        raise OpenRouterError('upstream', '上游返回了无法识别的响应。')
    finish = choice.get('finish_reason')
    text = (choice.get('message') or {}).get('content') or ''
    if isinstance(text, list):
        text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
    if finish == 'length':
        raise OpenRouterError('truncated', '输出被截断（达到 max_tokens）。')
    try:
        content = extract_json(text)
    except ValueError:
        raise OpenRouterError('invalid_output', '模型输出无法解析为 JSON。') from None
    usage = data.get('usage') or {}
    cost = usage.get('cost')
    return CallResult(
        content=content, raw_text=text, model=data.get('model') or model['id'],
        provider=data.get('provider'), finish_reason=finish, usage=usage,
        cost=float(cost) if isinstance(cost, (int, float)) else None,
        elapsed_ms=elapsed_ms,
    )
