"""Kie.ai client tests (MockTransport, no network)."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from ragent_lab.vision import compat, kie
from ragent_lab.vision.openrouter import OpenRouterError
from ragent_lab.vision.registry import gateways_of, get_model

KEY = 'kie-key-0123456789abcdef0123456789abcdef'
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}
B64 = 'iVBORw0KGgo='
MSGS = [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{B64}'}},
    {'type': 'text', 'text': '读图'}]}]
CLAUDE = get_model('anthropic/claude-opus-5.5')
GPT = get_model('openai/gpt-6-astra')
GEMINI = get_model('google/gemini-4-argon')
KIMI = get_model('moonshotai/kimi-k3.1-pro')


def run_call(handler, model):
    compat.transport = httpx.MockTransport(handler)
    delays, compat.RETRY_DELAYS = compat.RETRY_DELAYS, (0.0, 0.0)
    try:
        return asyncio.run(kie.chat_json(KEY, model, MSGS, 'out', SCHEMA))
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


class KieRegistryTests(unittest.TestCase):
    def test_only_mapped_models_offer_kie(self):
        self.assertEqual(gateways_of(CLAUDE), ['openrouter', 'kie'])
        kie_models = {m for m in ('anthropic/claude-opus-5.5', 'openai/gpt-6-astra',
                                  'google/gemini-4-argon', 'moonshotai/kimi-k3.1-pro')}
        for mid in ('qwen/qwen3.8-max-prime', 'z-ai/glm-5v-turbo', 'minimax/minimax-m3',
                    'bytedance-seed/seed-2-1-turbo'):
            self.assertEqual(gateways_of(get_model(mid)), ['openrouter'], mid)
        self.assertEqual(len(kie_models), 4)


class KieCallTests(unittest.TestCase):
    def test_claude_uses_anthropic_messages(self):
        seen, handler = recorder(httpx.Response(200, json={
            'type': 'message', 'model': 'claude-opus-5-5', 'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': '```json\n{"a": 1}\n```'}],
            'usage': {'input_tokens': 10, 'output_tokens': 5}, 'credits_consumed': 0.28}))
        r = run_call(handler, CLAUDE)
        req = seen[0]
        self.assertEqual(str(req.url), 'https://api.kie.ai/claude/v1/messages')
        self.assertEqual(req.headers['Authorization'], f'Bearer {KEY}')
        body = json.loads(req.content)
        self.assertEqual(body['model'], 'claude-opus-5-5')
        self.assertIs(body['stream'], True)
        self.assertEqual(body['max_tokens'], CLAUDE['max_tokens'])
        image, text, suffix = body['messages'][0]['content']
        self.assertEqual(image, {'type': 'image', 'source': {
            'type': 'base64', 'media_type': 'image/png', 'data': B64}})
        self.assertEqual(text, {'type': 'text', 'text': '读图'})
        self.assertIn('JSON', suffix['text'])
        self.assertEqual((r.content, r.credits, r.cost, r.provider), ({'a': 1}, 0.28, None, 'Kie.ai'))

    def test_gpt_and_kimi_use_responses(self):
        for model, path, wire in ((GPT, '/codex/v1/responses', 'gpt-6-astra'),
                                  (KIMI, '/openai/v1/responses', 'kimi-k3-1-pro')):
            with self.subTest(model['id']):
                seen, handler = recorder(httpx.Response(200, json={
                    'status': 'completed', 'model': wire, 'credits_consumed': 0.48,
                    'output': [{'type': 'reasoning', 'summary': []},
                               {'type': 'message', 'content': [
                                   {'type': 'output_text', 'text': '{"a": 2}'}]}]}))
                r = run_call(handler, model)
                self.assertEqual(seen[0].url.path, path)
                body = json.loads(seen[0].content)
                self.assertEqual(body['model'], wire)
                self.assertIs(body['stream'], True)
                self.assertEqual(body['input'][0]['content'], [
                    {'type': 'input_image', 'image_url': f'data:image/png;base64,{B64}'},
                    {'type': 'input_text', 'text': '读图'}])
                fmt = body['text']['format']
                self.assertEqual((fmt['type'], fmt['strict'], fmt['schema']),
                                 ('json_schema', False, SCHEMA))
                self.assertEqual((r.content, r.credits), ({'a': 2}, 0.48))

    def test_gemini_uses_chat_completions(self):
        seen, handler = recorder(httpx.Response(200, json={
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"a": 3}'}}],
            'usage': {'total_tokens': 9}, 'credits_consumed': 0.01}))
        r = run_call(handler, GEMINI)
        self.assertEqual(seen[0].url.path, '/gemini-4-argon-openai/v1/chat/completions')
        body = json.loads(seen[0].content)
        self.assertNotIn('model', body)
        # Kie.ai rejects schemas with a property named "type", so none is sent on this route.
        self.assertNotIn('response_format', body)
        content = body['messages'][0]['content']
        self.assertEqual(content[:2], MSGS[0]['content'])
        self.assertIn('JSON', content[2]['text'])
        self.assertEqual((r.content, r.credits, r.usage), ({'a': 3}, 0.01, {'total_tokens': 9}))

    def test_truncation_per_format(self):
        cases = [
            (CLAUDE, {'content': [{'type': 'text', 'text': '{"a":'}], 'stop_reason': 'max_tokens'}),
            (GPT, {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
                   'output': []}),
            (GEMINI, {'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}),
        ]
        for model, payload in cases:
            with self.subTest(model['id']):
                with self.assertRaises(OpenRouterError) as cm:
                    run_call(lambda r, p=payload: httpx.Response(200, json=p), model)
                self.assertEqual(cm.exception.kind, 'truncated')

    def test_errors_are_mapped_and_redacted(self):
        # (response, kind, text in message, logged as an upstream refusal)
        cases = [
            (httpx.Response(401, json={'error': {'message': 'Unauthorized ' + KEY}}), 'auth',
             'Kie.ai Key 无效', True),
            (httpx.Response(402, json={'code': 402, 'msg': 'Insufficient credits'}), 'credits',
             'Insufficient credits', True),
            (httpx.Response(200, json={'code': 402, 'msg': 'credits low'}), 'credits',
             'credits low', True),
            (httpx.Response(200, json={'code': 422, 'msg': 'bad model', 'data': None}), 'upstream',
             'Kie.ai 拒绝了请求（422）', True),
            (httpx.Response(500, text='boom ' + KEY), 'upstream', '500', True),
            (httpx.Response(200, json={'content': [{'type': 'text', 'text': '抱歉'}],
                                       'stop_reason': 'end_turn'}), 'invalid_output', 'JSON', False),
        ]
        for resp, kind, text, logged in cases:
            with self.subTest(kind=kind, text=text):
                with self.assertRaises(OpenRouterError) as cm:
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
        with self.assertRaises(OpenRouterError) as cm:
            run_call(slow, GPT)
        self.assertEqual(cm.exception.kind, 'timeout')

    def test_model_without_kie_route(self):
        with self.assertRaises(OpenRouterError):
            run_call(lambda r: httpx.Response(200, json={}), get_model('qwen/qwen3.8-max-prime'))



class KieStreamTests(unittest.TestCase):
    def test_responses_stream(self):
        # Shapes recorded from Kie.ai's Kimi / GPT routes.
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
        # Shapes recorded from Kie.ai's Gemini route: credits and usage on the last chunk.
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
        with self.assertRaises(OpenRouterError) as cm:
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
                self.assertRaises(OpenRouterError) as cm:
            run_call(handler, CLAUDE)
        self.assertEqual(len(seen), 3)
        self.assertIn('Resources are tight', cm.exception.message)
        self.assertIn('已自动重试 2 次', cm.exception.message)

    def test_client_errors_are_not_retried(self):
        seen, handler = sequence(httpx.Response(401, json={'error': {'message': 'Unauthorized'}}))
        with self.assertLogs('ragent_lab.vision.compat', level='WARNING'), \
                self.assertRaises(OpenRouterError) as cm:
            run_call(handler, CLAUDE)
        self.assertEqual((cm.exception.kind, len(seen)), ('auth', 1))


if __name__ == '__main__':
    unittest.main()
