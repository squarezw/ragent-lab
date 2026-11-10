"""Streaming JSON client for any endpoint speaking one of three wire formats.

Used for every model call: the catalog (AIHubMix), server built-ins and custom models. An endpoint is a
dict:

    api        'chat' (OpenAI Chat Completions) | 'responses' (OpenAI Responses)
               | 'anthropic' (Anthropic Messages)
    url        full https URL of the endpoint
    model      model name sent in the body (None to omit it, when the URL names the model)
    json       'schema' (send the JSON schema) | 'object' (chat json_object mode)
               | 'prompt' (only ask for JSON in the prompt)
    auth       'bearer' (Authorization: Bearer) | 'x-api-key' (Anthropic's own API)
    label      name shown in error messages, e.g. 'AIHubMix' or 'api.deepseek.com'
    extra      optional dict merged into the request body
    public_only  re-check before each request that url is public https (visitor URLs)

Callers pass OpenAI chat-style messages; every format comes back as the same CallResult.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import httpx

from . import netguard
from .calls import CallResult, ModelError, extract_json
from .schema import lean_schema

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 600.0
JSON_ONLY = '\n\n只输出一个符合上述 Schema 的 JSON 对象，不要输出任何其他文字或代码块标记。'
ANTHROPIC_VERSION = '2023-06-01'

# Tests inject httpx.MockTransport here.
transport: httpx.AsyncBaseTransport | None = None

# Long calls go quiet for minutes, and gateways such as Cloudflare drop silent connections
# (HTTP 524), so every request streams. Overload and gateway errors are retried.
RETRY_STATUS = {500, 502, 503, 504, 520, 521, 522, 523, 524}
RETRY_DELAYS = (3.0, 8.0)  # seconds before attempt 2 and 3
_KEY_LIKE = re.compile(r'(sk-[A-Za-z0-9_-]{8,}|[A-Fa-f0-9]{32,})')
_DETAIL_MAX = 300


def new_client() -> httpx.AsyncClient:
    # follow_redirects stays off: a redirect could point a custom endpoint at an internal host.
    return httpx.AsyncClient(timeout=TIMEOUT_SECONDS, transport=transport, follow_redirects=False)


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


def _json_only() -> dict:
    return {'type': 'text', 'text': JSON_ONLY}


def _anthropic_body(ep: dict, max_tokens: int, messages: list[dict]) -> dict:
    content = []
    for p in _parts(messages) + [_json_only()]:  # Messages has no schema option
        if p['type'] == 'image_url':
            header, _, data = _data_url(p).partition(',')
            media = header.removeprefix('data:').split(';', 1)[0] or 'image/png'
            content.append({'type': 'image',
                            'source': {'type': 'base64', 'media_type': media, 'data': data}})
        else:
            content.append({'type': 'text', 'text': p['text']})
    return {'model': ep['model'], 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens, 'stream': True}


def _responses_body(ep: dict, messages: list[dict], schema_name: str, schema: dict) -> dict:
    parts = _parts(messages) + ([] if ep['json'] == 'schema' else [_json_only()])
    content = [{'type': 'input_image', 'image_url': _data_url(p)} if p['type'] == 'image_url'
               else {'type': 'input_text', 'text': p['text']} for p in parts]
    body: dict = {'model': ep['model'], 'stream': True,
                  'input': [{'role': 'user', 'content': content}]}
    if ep['json'] == 'schema':
        body['text'] = {'format': {'type': 'json_schema', 'name': schema_name,
                                   'strict': False, 'schema': schema}}
    return body


def _chat_body(ep: dict, max_tokens: int, messages: list[dict], schema_name: str,
               schema: dict) -> dict:
    mode = ep['json']
    parts = _parts(messages) + ([] if mode == 'schema' else [_json_only()])
    body: dict = {'messages': [{'role': 'user', 'content': parts}], 'stream': True,
                  'max_tokens': max_tokens}
    if mode == 'schema':
        if ep.get('lean_schema'):
            schema = lean_schema(schema)
        body['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': schema_name, 'strict': False, 'schema': schema}}
    elif mode == 'object':
        body['response_format'] = {'type': 'json_object'}
    if ep.get('model'):
        body['model'] = ep['model']
    return body


def build_body(ep: dict, max_tokens: int, messages: list[dict], schema_name: str,
               schema: dict) -> dict:
    if ep['api'] == 'anthropic':
        body = _anthropic_body(ep, max_tokens, messages)
    elif ep['api'] == 'responses':
        body = _responses_body(ep, messages, schema_name, schema)
    else:
        body = _chat_body(ep, max_tokens, messages, schema_name, schema)
    body.update(ep.get('extra') or {})
    return body


def _headers(ep: dict, api_key: str) -> dict:
    if ep.get('auth') == 'x-api-key':
        return {'x-api-key': api_key, 'anthropic-version': ANTHROPIC_VERSION}
    return {'Authorization': f'Bearer {api_key}'}


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


def _status_error(ep: dict, code: int, detail: str, api_key: str,
                  model_id: str) -> ModelError:
    label = ep['label']
    detail = _redact(detail.strip(), api_key)
    kind, msg = {
        401: ('auth', f'{label} Key 无效或已撤销'),
        402: ('credits', f'{label} 余额或积分不足'),
        429: ('rate_limit', f'请求过于频繁或被 {label} 限流，请稍后重试'),
    }.get(code, ('upstream', f'{label} 拒绝了请求（{code}）'))
    log.warning('%s %s code %s: %s', label, model_id, code, detail or '-')
    return ModelError(kind, f'{msg}：{detail}' if detail else f'{msg}。')


def _envelope_code(data: dict) -> int | None:
    """Some platforms answer HTTP 200 with {"code": 4xx, "msg": ...} instead of a result."""
    code = data.get('code')
    if isinstance(code, int) and code != 200 and not any(
            k in data for k in ('choices', 'output', 'content')):
        return code
    return None


# ---- streaming ------------------------------------------------------------------------

class _Transient(Exception):
    """A failure worth retrying (overload, gateway timeout, stream cut off)."""

    def __init__(self, error: ModelError):
        super().__init__(error.message)
        self.error = error


class _Stream:
    """Collects SSE events back into the shape of the non-streaming response."""

    def __init__(self, api: str, label: str):
        self.api = api
        self.label = label
        self.text = ''
        self.done = False
        self.model: str | None = None
        self.finish: str | None = None
        self.usage: dict = {}
        self.credits = None
        self.final: dict | None = None  # Responses API: the response.completed payload

    def _broken(self, message) -> _Transient:
        return _Transient(ModelError('upstream', f'{self.label} 上游中断：{message}'))

    def feed(self, ev: dict) -> None:
        if 'credits_consumed' in ev:
            self.credits = ev['credits_consumed']
        kind = ev.get('type')
        if kind == 'error':
            err = ev.get('error') if isinstance(ev.get('error'), dict) else ev
            raise self._broken(err.get('message') or err.get('code') or 'error')
        if self.api == 'responses':
            if kind == 'response.output_text.delta':
                self.text += ev.get('delta') or ''
            elif kind in ('response.completed', 'response.incomplete', 'response.failed'):
                self.final = ev.get('response') or {}
                self.done = True
                if kind == 'response.failed':
                    raise self._broken((self.final.get('error') or {}).get('message')
                                       or 'response.failed')
        elif self.api == 'chat':
            self.model = ev.get('model') or self.model
            if ev.get('usage'):
                self.usage = ev['usage']
            for choice in ev.get('choices') or []:
                self.text += (choice.get('delta') or {}).get('content') or ''
                if choice.get('finish_reason'):
                    self.finish = choice['finish_reason']
        else:  # anthropic
            if kind == 'message_start':
                msg = ev.get('message') or {}
                self.model = msg.get('model') or self.model
                self.usage.update(msg.get('usage') or {})
            elif kind == 'content_block_delta':
                delta = ev.get('delta') or {}
                if delta.get('type') == 'text_delta':
                    self.text += delta.get('text') or ''
            elif kind == 'message_delta':
                self.finish = (ev.get('delta') or {}).get('stop_reason') or self.finish
                self.usage.update(ev.get('usage') or {})
            elif kind == 'message_stop':
                self.done = True

    def result(self) -> dict:
        finished = self.done or (self.api == 'chat' and self.finish is not None)
        if not finished:
            raise _Transient(ModelError('upstream', f'{self.label} 流式响应中途断开。'))
        if self.api == 'responses':
            data = dict(self.final or {})
            data['output'] = [{'type': 'message',
                               'content': [{'type': 'output_text', 'text': self.text}]}]
        elif self.api == 'chat':
            data = {'model': self.model, 'usage': self.usage, 'choices': [
                {'finish_reason': self.finish, 'message': {'content': self.text}}]}
        else:
            data = {'model': self.model, 'usage': self.usage, 'stop_reason': self.finish,
                    'content': [{'type': 'text', 'text': self.text}]}
        if self.credits is not None:
            data['credits_consumed'] = self.credits
        return data


async def _read_sse(resp: httpx.Response, ep: dict) -> dict:
    stream = _Stream(ep['api'], ep['label'])
    async for line in resp.aiter_lines():
        if not line.startswith('data:'):
            continue
        raw = line[5:].strip()
        if raw == '[DONE]':
            stream.done = True
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        if isinstance(ev, dict):
            stream.feed(ev)
    return stream.result()


async def _attempt(ep: dict, body: dict, api_key: str, model_id: str) -> dict:
    """One request. Returns the response as a non-streaming-shaped dict."""
    label = ep['label']
    try:
        async with new_client() as client:
            async with client.stream('POST', ep['url'], json=body,
                                     headers=_headers(ep, api_key)) as resp:
                if resp.status_code >= 400 or 300 <= resp.status_code < 400:
                    raw = await resp.aread()
                    try:
                        detail = _error_detail(json.loads(raw))
                    except ValueError:
                        detail = ''  # e.g. Cloudflare's HTML 524 page
                    err = _status_error(ep, resp.status_code, detail, api_key, model_id)
                    if resp.status_code in RETRY_STATUS:
                        raise _Transient(err)
                    raise err
                if 'text/event-stream' in resp.headers.get('content-type', ''):
                    return await _read_sse(resp, ep)
                raw = await resp.aread()
    except httpx.TimeoutException:
        raise ModelError('timeout', '模型调用超时。') from None
    except httpx.HTTPError as exc:
        raise _Transient(ModelError('upstream', f'网络错误：{type(exc).__name__}。')) from None
    # A plain JSON answer: a server that ignored stream, or a {"code", "msg"} error envelope.
    try:
        data = json.loads(raw)
    except ValueError:
        raise ModelError('upstream', f'{label} 返回了无法识别的响应。') from None
    if not isinstance(data, dict):
        raise ModelError('upstream', f'{label} 返回了无法识别的响应。')
    code = _envelope_code(data)
    if code is not None:
        err = _status_error(ep, code, _error_detail(data), api_key, model_id)
        if code in RETRY_STATUS:
            raise _Transient(err)
        raise err
    return data


# ---- call -----------------------------------------------------------------------------

async def call_endpoint(api_key: str, ep: dict, model: dict, messages: list[dict],
                        schema_name: str, schema: dict) -> CallResult:
    body = build_body(ep, model['max_tokens'], messages, schema_name, schema)
    label = ep['label']
    started = time.perf_counter()
    attempts = len(RETRY_DELAYS) + 1
    for n in range(attempts):
        if ep.get('public_only'):
            try:
                await asyncio.to_thread(netguard.check_public_https, ep['url'])
            except netguard.UnsafeURL as exc:
                raise ModelError('upstream', str(exc)) from None
        try:
            data = await _attempt(ep, body, api_key, model['id'])
            break
        except _Transient as exc:
            if n == attempts - 1:
                raise ModelError(exc.error.kind,
                                      f'{exc.error.message}（已自动重试 {n} 次）') from None
            log.warning('%s %s attempt %d failed, retrying: %s', label, model['id'], n + 1,
                        _redact(exc.error.message, api_key))
            await asyncio.sleep(RETRY_DELAYS[n])
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        text, finish, truncated = _PARSERS[ep['api']](data)
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ModelError('upstream', f'{label} 返回了无法识别的响应。') from None
    if truncated:
        raise ModelError('truncated', '输出被截断（达到 max_tokens）。')
    try:
        content = extract_json(text)
    except ValueError as exc:
        # Why it failed and how the reply ended (a short tail; never the key).
        log.warning('%s reply is not JSON (%s): %d chars, finish=%s, ends with %r',
                    label, exc, len(text or ''), finish, (text or '')[-80:])
        raise ModelError('invalid_output', '模型输出无法解析为 JSON。') from None
    usage = data.get('usage') or {}
    credits = data.get('credits_consumed')
    cost = usage.get('cost') if isinstance(usage, dict) else None
    return CallResult(
        content=content, raw_text=text, model=data.get('model') or ep.get('model') or model['id'],
        provider=label, finish_reason=finish, usage=usage,
        cost=float(cost) if isinstance(cost, (int, float)) else None, elapsed_ms=elapsed_ms,
        credits=float(credits) if isinstance(credits, (int, float)) else None,
    )
