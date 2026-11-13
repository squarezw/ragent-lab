"""AIHubMix routes (MockTransport, no network)."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from ragent_lab.vision import aihubmix, compat
from ragent_lab.vision.calls import ModelError
from ragent_lab.vision.registry import JUDGE_MODEL, VISION_MODELS, get_model
from ragent_lab.vision.schema import Geometry, Understanding, lean_schema, model_json_schema_for

KEY = 'sk-ahm-0123456789'
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}}, 'required': ['a']}
MSGS = [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}},
    {'type': 'text', 'text': '读图'}]}]


def run_call(handler, model):
    compat.transport = httpx.MockTransport(handler)
    delays, compat.RETRY_DELAYS = compat.RETRY_DELAYS, (0.0, 0.0)
    try:
        return asyncio.run(aihubmix.chat_json(KEY, model, MSGS, 'out', SCHEMA))
    finally:
        compat.transport = None
        compat.RETRY_DELAYS = delays


def record(resp):
    seen = []
    return seen, lambda req: seen.append(req) or resp


class AIHubMixTests(unittest.TestCase):
    def test_every_catalog_model_has_a_route(self):
        self.assertEqual(len(VISION_MODELS), 8)
        for m in VISION_MODELS:
            with self.subTest(m['id']):
                self.assertIn(m['route']['api'], ('chat', 'anthropic'))
                self.assertTrue(m['route']['model'])

    def test_menu_starts_with_fable_then_opus_and_opus_stays_the_judge(self):
        self.assertEqual([m['id'] for m in VISION_MODELS[:2]],
                         ['anthropic/claude-fable-5.2', 'anthropic/claude-opus-5.5'])
        self.assertEqual(JUDGE_MODEL['id'], 'anthropic/claude-opus-5.5')

    def test_claude_uses_native_messages_with_x_api_key(self):
        for model_id, upstream in (('anthropic/claude-fable-5.2', 'claude-fable-5-2'),
                                   ('anthropic/claude-opus-5.5', 'claude-opus-5-5')):
            with self.subTest(model_id):
                seen, handler = record(httpx.Response(200, json={
                    'type': 'message', 'stop_reason': 'end_turn',
                    'content': [{'type': 'text', 'text': '{"a": 1}'}],
                    'usage': {'input_tokens': 10, 'output_tokens': 5}}))
                r = run_call(handler, get_model(model_id))
                req = seen[0]
                self.assertEqual(str(req.url), 'https://aihubmix.com/v1/messages')
                self.assertEqual(req.headers['x-api-key'], KEY)
                self.assertNotIn('authorization', req.headers)
                body = json.loads(req.content)
                self.assertEqual(body['model'], upstream)
                self.assertNotIn('thinkingFlag', body)
                self.assertEqual((r.content, r.provider), ({'a': 1}, 'AIHubMix'))

    def test_others_use_chat_with_schema(self):
        seen, handler = record(httpx.Response(200, json={
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"a": 2}'}}]}))
        r = run_call(handler, get_model('bytedance-seed/seed-2-1-turbo'))
        req = seen[0]
        self.assertEqual(str(req.url), 'https://aihubmix.com/v1/chat/completions')
        self.assertEqual(req.headers['Authorization'], f'Bearer {KEY}')
        body = json.loads(req.content)
        self.assertEqual(body['model'], 'doubao-seed-2-1-turbo')
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertEqual(r.content, {'a': 2})

    def test_model_without_route(self):
        with self.assertRaises(ModelError):
            run_call(lambda req: httpx.Response(200), {'id': 'x', 'name': 'X', 'max_tokens': 1})

    def test_gemini_gets_a_lean_schema(self):
        seen, handler = record(httpx.Response(200, json={
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"a": 2}'}}]}))
        compat.transport = httpx.MockTransport(handler)
        try:
            asyncio.run(aihubmix.chat_json(KEY, get_model('google/gemini-4-argon'), MSGS, 'geometry',
                                           model_json_schema_for(Geometry)))
        finally:
            compat.transport = None
        sent = json.dumps(json.loads(seen[0].content)['response_format'])
        for word in ('oneOf', 'maxItems', 'exclusiveMinimum', '"title"', '"default"'):
            self.assertNotIn(word, sent)
        self.assertIn('anyOf', sent)
        # A property that happens to be named like a dropped keyword survives.
        lean = lean_schema({'type': 'object', 'properties': {'title': {'type': 'string', 'title': 'T'}}})
        self.assertEqual(lean, {'type': 'object', 'properties': {'title': {'type': 'string'}}})

    def test_schemas_have_no_const(self):
        """Gemini (via AIHubMix) rejects "const"; it becomes a one-value enum."""
        geo = json.dumps(model_json_schema_for(Geometry))
        self.assertNotIn('"const"', geo)
        self.assertIn('"enum": ["line"]', geo)
        self.assertNotIn('"const"', json.dumps(model_json_schema_for(Understanding)))


if __name__ == '__main__':
    unittest.main()
