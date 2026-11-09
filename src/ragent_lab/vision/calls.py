"""Result and error types shared by every model call, and JSON extraction from replies."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

ErrorKind = Literal['auth', 'credits', 'rate_limit', 'upstream', 'timeout', 'truncated', 'invalid_output']


class ModelError(Exception):
    """A model call failed; message is user-facing (Chinese) and never contains a key."""

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
    credits: float | None = None  # some platforms bill in credits, not USD


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
