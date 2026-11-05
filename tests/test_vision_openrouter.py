"""OpenRouter client and registry tests (MockTransport, no network)."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from ragent_lab.vision import openrouter, registry
from ragent_lab.vision.openrouter import OpenRouterError, chat_json
from ragent_lab.vision.registry import JUDGE_MODEL, VISION_MODELS, get_model

IDS = ['anthropic/claude-opus-5.5', 'openai/gpt-6-astra', 'google/gemini-4-argon',
       'qwen/qwen3.8-max-prime', 'z-ai/glm-5v-turbo', 'moonshotai/kimi-k3.1-pro',
       'bytedance-seed/seed-2-1-turbo', 'deepseek/deepseek-v4-flash-vision-exp',
       'minimax/minimax-m3']
KEY = 'sk-test'
MSGS = [{'role': 'user', 'content': 'hi'}]


def ok(content, finish='stop', **extra):
    return httpx.Response(200, json={
        'model': 'x/y', 'provider': 'Anthropic',
        'choices': [{'finish_reason': finish, 'message': {'content': content}}],
        'usage': {'cost': 0.0123, 'prompt_tokens': 1}, **extra})


def run_call(handler, model=None):
    openrouter.transport = httpx.MockTransport(handler)
    try:
        return asyncio.run(chat_json(KEY, model or VISION_MODELS[1], MSGS, 'out', {'type': 'object'}))
    finally:
        openrouter.transport = None


class OpenRouterTests(unittest.TestCase):
    def test_registry_has_exact_models(self):
        self.assertEqual([m['id'] for m in VISION_MODELS], IDS)
        self.assertFalse(any('stepfun' in m['id'] for m in VISION_MODELS))
        self.assertEqual(get_model('minimax/minimax-m3')['reasoning'], {'enabled': False})
        self.assertEqual(JUDGE_MODEL['provider'],
                         {'order': ['anthropic'], 'allow_fallbacks': False})
        with self.assertRaises(KeyError):
            get_model('nope')

    def test_request_body_and_headers(self):
        seen = []

        def handler(req):
            seen.append(req)
            return ok('{"a":1}')

        run_call(handler)
        req = seen[0]
        body = json.loads(req.content)
        self.assertEqual(req.headers['Authorization'], 'Bearer sk-test')
        self.assertEqual(req.headers['X-Title'], 'Ragent Lab')
        self.assertIs(body['response_format']['json_schema']['strict'], False)
        self.assertTrue(body['usage']['include'])
        self.assertNotIn('provider', body)
        run_call(handler, JUDGE_MODEL)
        self.assertIs(json.loads(seen[1].content)['provider']['allow_fallbacks'], False)
        run_call(handler, get_model('z-ai/glm-5v-turbo'))
        self.assertEqual(json.loads(seen[2].content)['response_format'], {'type': 'json_object'})
        run_call(handler, get_model('minimax/minimax-m3'))
        self.assertEqual(json.loads(seen[3].content)['reasoning'], {'enabled': False})

    def test_parses_fenced_json_and_usage(self):
        r = run_call(lambda req: ok('说明\n```json\n{"a":1,}\n```'))
        self.assertEqual(r.content, {'a': 1})
        self.assertEqual(r.cost, 0.0123)
        self.assertEqual(r.provider, 'Anthropic')
        self.assertEqual(r.finish_reason, 'stop')

    def test_extract_json_nested_braces_in_strings(self):
        self.assertEqual(openrouter.extract_json('x {"a":"}{","b":{"c":[1,]}} y'),
                         {'a': '}{', 'b': {'c': [1]}})
        with self.assertRaises(ValueError):
            openrouter.extract_json('抱歉')

    def test_error_mapping(self):
        def timeout(req):
            raise httpx.ReadTimeout('slow ' + KEY, request=req)

        cases = [
            (lambda r: httpx.Response(401, text='bad ' + KEY), 'auth'),
            (lambda r: httpx.Response(402, text='x'), 'credits'),
            (lambda r: httpx.Response(429, text='x'), 'rate_limit'),
            (lambda r: httpx.Response(500, text='x ' + KEY), 'upstream'),
            (timeout, 'timeout'),
            (lambda r: ok('{"a":', finish='length'), 'truncated'),
            (lambda r: ok('抱歉'), 'invalid_output'),
        ]
        for handler, kind in cases:
            with self.subTest(kind):
                with self.assertRaises(OpenRouterError) as cm:
                    run_call(handler)
                err = cm.exception
                self.assertEqual(err.kind, kind)
                self.assertNotIn(KEY, str(err))
                self.assertNotIn(KEY, err.message)

    def test_prices(self):
        registry._price_cache.update(at=0.0, data=None)
        openrouter.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={'data': [
            {'id': IDS[0], 'pricing': {'prompt': '0.000005', 'completion': '0.000025'}}]}))
        try:
            prices = asyncio.run(registry.model_prices())
        finally:
            openrouter.transport = None
            registry._price_cache.update(at=0.0, data=None)
        self.assertEqual(prices[IDS[0]], {'prompt': '0.000005', 'completion': '0.000025'})
        self.assertIsNone(prices[IDS[1]])

    def test_prices_failure_returns_none(self):
        registry._price_cache.update(at=0.0, data=None)
        openrouter.transport = httpx.MockTransport(lambda r: httpx.Response(500))
        try:
            prices = asyncio.run(registry.model_prices())
        finally:
            openrouter.transport = None
        self.assertEqual(set(prices), set(IDS))
        self.assertTrue(all(v is None for v in prices.values()))


if __name__ == '__main__':
    unittest.main()
