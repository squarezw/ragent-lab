"""Shared model client tests: wire formats, streaming, retries, errors (no network)."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from ragent_lab.vision import compat
from ragent_lab.vision.calls import ModelError, extract_json

KEY = 'sk-test-0123456789abcdef0123456789abcdef'
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}
B64 = 'iVBORw0KGgo='
MSGS = [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{B64}'}},
    {'type': 'text', 'text': '读图'}]}]
LABEL = 'TestHub'
MODEL = {'id': 'm', 'name': 'M', 'max_tokens': 8000}
# One endpoint per wire format.
CLAUDE = {'api': 'anthropic', 'url': 'https://hub.example.com/v1/messages', 'model': 'claude-x',
          'json': 'prompt', 'auth': 'x-api-key', 'label': LABEL}
GPT = {'api': 'responses', 'url': 'https://hub.example.com/v1/responses', 'model': 'gpt-x',
       'json': 'schema', 'auth': 'bearer', 'label': LABEL}
GEMINI = {'api': 'chat', 'url': 'https://hub.example.com/v1/chat/completions', 'model': 'gem-x',
          'json': 'prompt', 'auth': 'bearer', 'label': LABEL}
KIMI = GPT


def run_call(handler, ep):
    compat.transport = httpx.MockTransport(handler)
    delays, compat.RETRY_DELAYS = compat.RETRY_DELAYS, (0.0, 0.0)
    try:
        return asyncio.run(compat.call_endpoint(KEY, ep, MODEL, MSGS, 'out', SCHEMA))
    finally:
        compat.transport = None
        compat.RETRY_DELAYS = delays


def sse(*events, done=False):
    """An event-stream response; events are dicts (sent as data: lines) or raw strings."""
    lines = []
    for ev in events:
        if isinstance(ev, str):
            lines.append(ev)
        else:
            lines += [f"event: {ev.get('type', 'message')}", 'data: ' + json.dumps(ev), '']
    if done:
        lines += ['data: [DONE]', '']
    return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                          content='\n'.join(lines).encode())


def sequence(*responses):
    """Handler returning the given responses in order, recording each request."""
    seen, queue = [], list(responses)

    def handler(req):
        seen.append(req)
        return queue.pop(0)
    return seen, handler


def recorder(response):
    seen = []

    def handler(req):
        seen.append(req)
        return response
    return seen, handler


class CallTests(unittest.TestCase):
    def test_truncation_per_format(self):
        cases = [
            (CLAUDE, {'content': [{'type': 'text', 'text': '{"a":'}], 'stop_reason': 'max_tokens'}),
            (GPT, {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
                   'output': []}),
            (GEMINI, {'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}),
        ]
        for model, payload in cases:
            with self.subTest(model['api']):
                with self.assertRaises(ModelError) as cm:
                    run_call(lambda r, p=payload: httpx.Response(200, json=p), model)
                self.assertEqual(cm.exception.kind, 'truncated')

    def test_errors_are_mapped_and_redacted(self):
        # (response, kind, text in message, logged as an upstream refusal)
        cases = [
            (httpx.Response(401, json={'error': {'message': 'Unauthorized ' + KEY}}), 'auth',
             f'{LABEL} Key 无效', True),
            (httpx.Response(402, json={'code': 402, 'msg': 'Insufficient credits'}), 'credits',
             'Insufficient credits', True),
            (httpx.Response(200, json={'code': 402, 'msg': 'credits low'}), 'credits',
             'credits low', True),
            (httpx.Response(200, json={'code': 422, 'msg': 'bad model', 'data': None}), 'upstream',
             f'{LABEL} 拒绝了请求（422）', True),
            (httpx.Response(500, text='boom ' + KEY), 'upstream', '500', True),
            (httpx.Response(200, json={'content': [{'type': 'text', 'text': '抱歉'}],
                                       'stop_reason': 'end_turn'}), 'invalid_output', 'JSON', False),
        ]
        for resp, kind, text, logged in cases:
            with self.subTest(kind=kind, text=text):
                with self.assertRaises(ModelError) as cm:
                    if logged:
                        with self.assertLogs('ragent_lab.vision.compat', level='WARNING') as logs:
                            run_call(lambda r, x=resp: x, CLAUDE)
                        self.assertNotIn(KEY, '\n'.join(logs.output))
                    else:
                        run_call(lambda r, x=resp: x, CLAUDE)
                self.assertEqual(cm.exception.kind, kind)
                self.assertIn(text, cm.exception.message)
                self.assertNotIn(KEY, cm.exception.message)

    def test_timeout(self):
        def slow(req):
            raise httpx.ReadTimeout('slow', request=req)
        with self.assertRaises(ModelError) as cm:
            run_call(slow, GPT)
        self.assertEqual(cm.exception.kind, 'timeout')


class StreamTests(unittest.TestCase):
    def test_responses_stream(self):
        # Shapes recorded from a Responses-API stream.
        resp = sse(
            {'type': 'response.created'},
            {'type': 'response.reasoning_text.delta', 'delta': 'thinking'},
            {'type': 'response.output_text.delta', 'delta': '{"a": '},
            {'type': 'response.output_text.delta', 'delta': '5}'},
            {'type': 'response.completed', 'credits_consumed': 13.8, 'response': {
                'status': 'completed', 'model': 'kimi-k3-1-pro',
                'usage': {'output_tokens': 17126}}})
        r = run_call(lambda req: resp, KIMI)
        self.assertEqual((r.content, r.credits, r.model), ({'a': 5}, 13.8, 'kimi-k3-1-pro'))
        self.assertEqual(r.usage, {'output_tokens': 17126})

    def test_chat_stream(self):
        # Chat stream with credits and usage on the last chunk.
        resp = sse(
            {'object': 'chat.completion.chunk', 'model': 'gemini-4-argon',
             'choices': [{'delta': {'content': '{"a"'}, 'index': 0}]},
            {'object': 'chat.completion.chunk',
             'choices': [{'delta': {'content': ': 6}'}, 'finish_reason': 'stop', 'index': 0}]},
            {'object': 'chat.completion.chunk', 'choices': [], 'credits_consumed': 2.85,
             'usage': {'total_tokens': 7094}}, done=True)
        r = run_call(lambda req: resp, GEMINI)
        self.assertEqual((r.content, r.credits, r.usage), ({'a': 6}, 2.85, {'total_tokens': 7094}))

    def test_anthropic_stream(self):
        resp = sse(
            {'type': 'message_start', 'message': {'model': 'claude-opus-5-5',
                                                  'usage': {'input_tokens': 9}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}},
            {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '{"a": 7}'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'},
             'usage': {'output_tokens': 4}},
            {'type': 'message_stop', 'credits_consumed': 0.75})
        r = run_call(lambda req: resp, CLAUDE)
        self.assertEqual((r.content, r.credits, r.finish_reason), ({'a': 7}, 0.75, 'end_turn'))
        self.assertEqual(r.usage, {'input_tokens': 9, 'output_tokens': 4})

    def test_anthropic_stream_truncated(self):
        resp = sse({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '{"a":'}},
                   {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens'}},
                   {'type': 'message_stop'})
        with self.assertRaises(ModelError) as cm:
            run_call(lambda req: resp, CLAUDE)
        self.assertEqual(cm.exception.kind, 'truncated')

    def test_transient_failures_are_retried(self):
        ok = sse({'type': 'response.output_text.delta', 'delta': '{"a": 1}'},
                 {'type': 'response.completed', 'response': {'status': 'completed'}})
        cases = {
            'overloaded 503': httpx.Response(503, json={'type': 'error', 'error': {
                'type': 'api_error', 'message': 'Resources are tight, please try again later.'}}),
            'cloudflare 524': httpx.Response(524, text='<!DOCTYPE html><html>timeout</html>'),
            'mid-stream error': sse(
                {'type': 'response.output_text.delta', 'delta': '{"a"'},
                {'type': 'error', 'code': 'upstream_error',
                 'message': 'The server is currently being maintained, please try again later~'}),
            'stream cut off': sse({'type': 'response.output_text.delta', 'delta': '{"a"'}),
        }
        for label, first in cases.items():
            with self.subTest(label):
                seen, handler = sequence(first, ok)
                with self.assertLogs('ragent_lab.vision.compat', level='WARNING'):
                    r = run_call(handler, GPT)
                self.assertEqual(r.content, {'a': 1})
                self.assertEqual(len(seen), 2)

    def test_gives_up_after_three_attempts(self):
        busy = httpx.Response(503, json={'error': {'message': 'Resources are tight'}})
        seen, handler = sequence(busy, busy, busy)
        with self.assertLogs('ragent_lab.vision.compat', level='WARNING'), \
                self.assertRaises(ModelError) as cm:
            run_call(handler, CLAUDE)
        self.assertEqual(len(seen), 3)
        self.assertIn('Resources are tight', cm.exception.message)
        self.assertIn('已自动重试 2 次', cm.exception.message)

    def test_client_errors_are_not_retried(self):
        seen, handler = sequence(httpx.Response(401, json={'error': {'message': 'Unauthorized'}}))
        with self.assertLogs('ragent_lab.vision.compat', level='WARNING'), \
                self.assertRaises(ModelError) as cm:
            run_call(handler, CLAUDE)
        self.assertEqual((cm.exception.kind, len(seen)), ('auth', 1))


class ExtractJsonTests(unittest.TestCase):
    def test_nested_braces_in_strings_and_trailing_commas(self):
        self.assertEqual(extract_json('x {"a":"}{","b":{"c":[1,]}} y'), {'a': '}{', 'b': {'c': [1]}})
        self.assertEqual(extract_json('说明\n```json\n{"a":1,}\n```'), {'a': 1})
        with self.assertRaises(ValueError):
            extract_json('抱歉')


if __name__ == '__main__':
    unittest.main()
