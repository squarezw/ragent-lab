"""Server built-in models: configured in env, paid with the server's keys, unlocked by code.

    VISION_BUILTIN_MODELS  JSON list, one object per model:
        id         short slug, e.g. "deepseek"
        name       display name
        api        "chat" | "responses" | "anthropic"            (default "chat")
        url        full endpoint URL
        model      model name sent to the endpoint
        key_env    name of the env var holding the API key, e.g. "DEEPSEEK_API_KEY"
        json       "schema" | "object" | "prompt"                (default "prompt")
        auth       "bearer" | "x-api-key"                        (default "bearer")
        max_tokens                                               (default 32000)
        notes      optional text shown on the settings page
        extra      optional JSON object merged into the request body, e.g.
                   {"thinking": {"type": "disabled"}} for DeepSeek without thinking
    OPENROUTER_API_KEY, KIE_API_KEY, AIHUBMIX_API_KEY
                             the server's own platform keys: visitors with the access code can
                             call the catalog models through them without a key of their own
    LAB_ACCESS_CODE          visitors must enter it to use built-in models or the server's
                             platform keys; unset = disabled
    VISION_BUILTIN_PER_HOUR  model calls per visitor IP per hour (default 60)

Keys and endpoint URLs never leave the server: public_list() only returns id, name, notes,
and platforms() only the platform names.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque

log = logging.getLogger(__name__)

_ID = re.compile(r'^[a-z0-9][a-z0-9_.-]{0,39}$')
_APIS = ('chat', 'responses', 'anthropic')
_JSON = ('schema', 'object', 'prompt')
_AUTH = ('bearer', 'x-api-key')
# Body fields the client builds itself; extra may not replace them.
_RESERVED = {'model', 'messages', 'input', 'stream', 'system'}

PLATFORM_KEY_ENV = {'openrouter': 'OPENROUTER_API_KEY', 'kie': 'KIE_API_KEY',
                    'aihubmix': 'AIHUBMIX_API_KEY'}
WINDOW_SECONDS = 3600.0
FAILED_CODE_LIMIT = 10  # wrong access codes per IP per window before lock-out


def _parse(raw: str, env) -> dict[str, dict]:
    try:
        items = json.loads(raw) if raw.strip() else []
    except ValueError:
        log.warning('VISION_BUILTIN_MODELS is not valid JSON; no built-in models')
        return {}
    if not isinstance(items, list):
        log.warning('VISION_BUILTIN_MODELS must be a JSON list; no built-in models')
        return {}
    out: dict[str, dict] = {}
    for n, item in enumerate(items):
        try:
            mid = str(item['id'])
            api = item.get('api', 'chat')
            mode = item.get('json', 'prompt')
            auth = item.get('auth', 'bearer')
            if not _ID.match(mid) or api not in _APIS or mode not in _JSON or auth not in _AUTH:
                raise ValueError('bad id/api/json/auth')
            url = str(item['url'])
            if not url.startswith('https://'):
                raise ValueError('url must be https')
            extra = item.get('extra') or {}
            if not isinstance(extra, dict) or _RESERVED & set(extra):
                raise ValueError('extra must be an object without ' + '/'.join(sorted(_RESERVED)))
            key = env.get(str(item['key_env']), '')
            if not key:
                log.warning('built-in model %s skipped: %s is not set', mid, item['key_env'])
                continue
            out[mid] = {
                'id': f'builtin:{mid}', 'slug': mid, 'name': str(item.get('name') or mid),
                'notes': str(item.get('notes') or ''),
                'max_tokens': int(item.get('max_tokens', 32000)),
                'temperature': None, 'reasoning': None,
                'api_key': key,
                'endpoint': {'api': api, 'url': url, 'model': item.get('model'), 'json': mode,
                             'auth': auth, 'label': str(item.get('name') or mid),
                             'extra': extra},
            }
        except (KeyError, TypeError, ValueError) as exc:
            log.warning('VISION_BUILTIN_MODELS[%d] skipped: %s', n, exc)
    return out


class Builtins:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.models = _parse(env.get('VISION_BUILTIN_MODELS', ''), env)
        self._code = env.get('LAB_ACCESS_CODE', '')
        self.platform_keys = {p: env.get(name, '').strip()
                              for p, name in PLATFORM_KEY_ENV.items() if env.get(name, '').strip()}
        self.per_hour = int(env.get('VISION_BUILTIN_PER_HOUR', '60'))
        self._calls: dict[str, deque] = defaultdict(deque)
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._code) and bool(self.models or self.platform_keys)

    def platforms(self) -> list[str]:
        """Platforms the server has its own key for (offered only with an access code)."""
        return list(self.platform_keys) if self._code else []

    def public_list(self) -> list[dict]:
        if not self.enabled:
            return []
        return [{'id': m['slug'], 'name': m['name'], 'notes': m['notes']}
                for m in self.models.values()]

    @staticmethod
    def _recent(q: deque, now: float) -> deque:
        while q and now - q[0] > WINDOW_SECONDS:
            q.popleft()
        return q

    def check_code(self, code: str, ip: str) -> bool:
        """True if code is right. Too many wrong codes from one IP lock it out for an hour."""
        if not self._code:
            return False
        now = time.time()
        with self._lock:
            fails = self._recent(self._fails[ip], now)
            if len(fails) >= FAILED_CODE_LIMIT:
                return False
            ok = hmac.compare_digest(code.encode(), self._code.encode())
            if not ok:
                fails.append(now)
            return ok

    def take(self, ip: str) -> bool:
        """Count one model call for ip; False if it is over the hourly limit."""
        now = time.time()
        with self._lock:
            calls = self._recent(self._calls[ip], now)
            if len(calls) >= self.per_hour:
                return False
            calls.append(now)
            return True


builtins = Builtins()
