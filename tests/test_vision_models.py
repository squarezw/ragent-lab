"""Built-in models, custom endpoints and the URL guard (no network)."""
import asyncio
import json
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from ragent_lab.vision import compat, netguard
from ragent_lab.vision.builtin import FAILED_CODE_LIMIT, Builtins
from ragent_lab.vision.openrouter import OpenRouterError


def resolver(*ips):
    def resolve(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port)) for ip in ips]
    return resolve


class NetguardTests(unittest.TestCase):
    def test_public_https_allowed(self):
        self.assertEqual(netguard.check_public_https(
            'https://api.deepseek.com/chat/completions', resolver('104.18.1.1')),
            'api.deepseek.com')

    def test_rejected(self):
        cases = {
            'http://api.example.com/v1': 'https://',
            'https://localhost/v1': '内网',
            'https://intranet/v1': '内网',
            'https://svc.internal/v1': '内网',
            'https://user:pw@api.example.com/v1': '格式',
        }
        for url, text in cases.items():
            with self.subTest(url), self.assertRaises(netguard.UnsafeURL) as cm:
                netguard.check_public_https(url, resolver('104.18.1.1'))
            self.assertIn(text, str(cm.exception))

    def test_private_resolution_rejected(self):
        for ip in ('127.0.0.1', '10.0.0.5', '172.17.0.2', '192.168.1.1', '169.254.169.254',
                   '100.64.0.1', '::1', 'fd00::1', '::ffff:10.0.0.1', '0.0.0.0'):
            with self.subTest(ip), self.assertRaises(netguard.UnsafeURL):
                netguard.check_public_https('https://evil.example.com/v1', resolver(ip))
        # One private address among public ones is enough to refuse.
        with self.assertRaises(netguard.UnsafeURL):
            netguard.check_public_https('https://mixed.example.com/v1',
                                        resolver('104.18.1.1', '10.0.0.1'))

    def test_unresolvable(self):
        def fail(*a, **k):
            raise socket.gaierror('nope')
        with self.assertRaises(netguard.UnsafeURL) as cm:
            netguard.check_public_https('https://nope.example.com/v1', fail)
        self.assertIn('无法解析', str(cm.exception))


CONFIG = json.dumps([
    {'id': 'deepseek', 'name': 'DeepSeek（内置）', 'url': 'https://api.deepseek.com/chat/completions',
     'model': 'deepseek-chat', 'key_env': 'DS_KEY', 'json': 'object', 'notes': '服务器付费'},
    {'id': 'nokey', 'name': 'X', 'url': 'https://x.example.com', 'key_env': 'MISSING'},
    {'id': 'BAD ID', 'name': 'X', 'url': 'https://x.example.com', 'key_env': 'DS_KEY'},
    {'id': 'plain', 'name': 'X', 'url': 'http://x.example.com', 'key_env': 'DS_KEY'},
    {'id': 'fast', 'name': 'Fast', 'url': 'https://api.deepseek.com/chat/completions',
     'key_env': 'DS_KEY', 'extra': {'thinking': {'type': 'disabled'}}},
    {'id': 'hijack', 'name': 'X', 'url': 'https://x.example.com', 'key_env': 'DS_KEY',
     'extra': {'messages': []}},
    {'id': 'listy', 'name': 'X', 'url': 'https://x.example.com', 'key_env': 'DS_KEY',
     'extra': ['nope']},
])


class BuiltinTests(unittest.TestCase):
    def make(self, **env):
        base = {'VISION_BUILTIN_MODELS': CONFIG, 'DS_KEY': 'sk-server-secret',
                'LAB_ACCESS_CODE': 'open-sesame', 'VISION_BUILTIN_PER_HOUR': '2'}
        base.update(env)
        with self.assertLogs('ragent_lab.vision.builtin', level='WARNING'):
            return Builtins({k: v for k, v in base.items() if v is not None})

    def test_loads_valid_entries_only(self):
        b = self.make()
        self.assertEqual(list(b.models), ['deepseek', 'fast'])
        self.assertEqual(b.models['fast']['endpoint']['extra'], {'thinking': {'type': 'disabled'}})
        self.assertEqual(b.models['deepseek']['endpoint']['extra'], {})
        m = b.models['deepseek']
        self.assertEqual(m['endpoint']['json'], 'object')
        self.assertEqual(m['api_key'], 'sk-server-secret')
        self.assertEqual(b.public_list()[0], {'id': 'deepseek', 'name': 'DeepSeek（内置）',
                                              'notes': '服务器付费'})
        self.assertNotIn('thinking', json.dumps(b.public_list()))
        self.assertNotIn('sk-server-secret', json.dumps(b.public_list()))

    def test_disabled_without_access_code(self):
        b = self.make(LAB_ACCESS_CODE=None)
        self.assertFalse(b.enabled)
        self.assertEqual(b.public_list(), [])
        self.assertFalse(b.check_code('', '1.2.3.4'))

    def test_access_code_and_lockout(self):
        b = self.make()
        self.assertTrue(b.check_code('open-sesame', '1.1.1.1'))
        for _ in range(FAILED_CODE_LIMIT):
            self.assertFalse(b.check_code('guess', '2.2.2.2'))
        # Locked out even with the right code; other IPs unaffected.
        self.assertFalse(b.check_code('open-sesame', '2.2.2.2'))
        self.assertTrue(b.check_code('open-sesame', '3.3.3.3'))

    def test_hourly_limit_per_ip(self):
        b = self.make()
        self.assertTrue(b.take('1.1.1.1'))
        self.assertTrue(b.take('1.1.1.1'))
        self.assertFalse(b.take('1.1.1.1'))
        self.assertTrue(b.take('9.9.9.9'))

    def test_platform_keys_need_access_code(self):
        b = self.make(OPENROUTER_API_KEY=' sk-or ', KIE_API_KEY='')
        self.assertEqual(b.platforms(), ['openrouter'])
        self.assertEqual(b.platform_keys, {'openrouter': 'sk-or'})
        self.assertEqual(self.make(OPENROUTER_API_KEY='sk-or', LAB_ACCESS_CODE=None).platforms(), [])
        # Platform keys alone (no built-in models) still enable the access code.
        only = Builtins({'OPENROUTER_API_KEY': 'sk-or', 'LAB_ACCESS_CODE': 'x'})
        self.assertTrue(only.enabled)
        self.assertEqual(only.public_list(), [])

    def test_bad_json(self):
        with self.assertLogs('ragent_lab.vision.builtin', level='WARNING'):
            self.assertEqual(Builtins({'VISION_BUILTIN_MODELS': '{oops'}).models, {})


SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}}}
MSGS = [{'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
    {'type': 'text', 'text': '读图'}]}]
MODEL = {'id': 'custom:x', 'name': 'X', 'max_tokens': 4000}


def call(ep, handler):
    compat.transport = httpx.MockTransport(handler)
    delays, compat.RETRY_DELAYS = compat.RETRY_DELAYS, (0.0, 0.0)
    try:
        return asyncio.run(compat.call_endpoint('sk-user', ep, MODEL, MSGS, 'out', SCHEMA))
    finally:
        compat.transport = None
        compat.RETRY_DELAYS = delays


def chat_ok(text='{"a": 1}', **extra):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                  'message': {'content': text}}], **extra})


class CustomEndpointTests(unittest.TestCase):
    def ep(self, **kw):
        base = {'api': 'chat', 'url': 'https://api.deepseek.com/chat/completions',
                'model': 'deepseek-chat', 'json': 'prompt', 'auth': 'bearer',
                'label': 'api.deepseek.com'}
        base.update(kw)
        return base

    def test_chat_json_modes(self):
        for mode, expect in (('schema', 'json_schema'), ('object', 'json_object'), ('prompt', None)):
            with self.subTest(mode):
                seen = []
                r = call(self.ep(json=mode), lambda req: seen.append(req) or chat_ok())
                body = json.loads(seen[0].content)
                self.assertEqual(str(seen[0].url), 'https://api.deepseek.com/chat/completions')
                self.assertEqual(seen[0].headers['Authorization'], 'Bearer sk-user')
                self.assertEqual(body['model'], 'deepseek-chat')
                self.assertEqual((body.get('response_format') or {}).get('type'), expect)
                asks_json = 'JSON' in body['messages'][0]['content'][-1].get('text', '')
                self.assertEqual(asks_json, mode != 'schema')
                self.assertEqual((r.content, r.provider), ({'a': 1}, 'api.deepseek.com'))

    def test_extra_is_merged_into_body(self):
        seen = []
        call(self.ep(extra={'thinking': {'type': 'disabled'}}),
             lambda req: seen.append(req) or chat_ok())
        body = json.loads(seen[0].content)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertTrue(body['stream'])

    def test_anthropic_native_headers(self):
        seen = []
        resp = httpx.Response(200, json={'content': [{'type': 'text', 'text': '{"a": 2}'}],
                                         'stop_reason': 'end_turn'})
        r = call(self.ep(api='anthropic', url='https://api.anthropic.com/v1/messages',
                         model='claude-x', auth='x-api-key'),
                 lambda req: seen.append(req) or resp)
        self.assertEqual(seen[0].headers['x-api-key'], 'sk-user')
        self.assertEqual(seen[0].headers['anthropic-version'], compat.ANTHROPIC_VERSION)
        self.assertNotIn('authorization', seen[0].headers)
        self.assertNotIn('thinkingFlag', json.loads(seen[0].content))
        self.assertEqual(r.content, {'a': 2})

    def test_openrouter_style_cost_is_kept(self):
        r = call(self.ep(), lambda req: chat_ok(usage={'cost': 0.004, 'total_tokens': 9}))
        self.assertEqual((r.cost, r.credits), (0.004, None))

    def test_redirects_are_not_followed(self):
        seen = []
        resp = httpx.Response(302, headers={'location': 'http://169.254.169.254/'})
        with self.assertLogs('ragent_lab.vision.compat', level='WARNING'), \
                self.assertRaises(OpenRouterError):
            call(self.ep(), lambda req: seen.append(req) or resp)
        self.assertEqual(len(seen), 1)

    def test_public_only_rechecked_before_request(self):
        seen = []
        with patch.object(netguard, 'check_public_https',
                          side_effect=netguard.UnsafeURL('接口地址不能指向本机或内网')), \
                self.assertRaises(OpenRouterError) as cm:
            call(self.ep(public_only=True), lambda req: seen.append(req) or chat_ok())
        self.assertIn('内网', cm.exception.message)
        self.assertEqual(seen, [])


if __name__ == '__main__':
    unittest.main()
