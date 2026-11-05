"""API contract tests. External embeddings use a deterministic provider double."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastapi.testclient import TestClient
from ragent_lab.api.app import app
from ragent_lab.embedding.base import EmbeddingResult


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_catalogs_are_serializable_and_have_vision_placeholder(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)
        catalog = self.client.get('/api/experiments').json()
        self.assertEqual(next(x for x in catalog if x['id'] == 'vision')['status'], 'ready')
        self.assertEqual(len(self.client.get('/api/chunking/strategies').json()), 11)
        self.assertEqual(len(self.client.get('/api/embedding/models').json()), 7)

    def test_chunking_returns_content_and_statistics(self):
        response = self.client.post('/api/chunking/run', json={'strategy': '固定长度分块', 'text': 'abcdefghij', 'params': {'chunk_size': 4}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([c['content'] for c in response.json()['chunks']], ['abcd', 'efgh', 'ij'])
        self.assertEqual(response.json()['stats']['count'], 3)

    def test_invalid_inputs_rejected_before_running(self):
        for payload in [
            {'strategy': '固定长度分块', 'text': 'abc', 'params': {'chunk_size': 0}},
            {'strategy': '滑动窗口分块', 'text': 'abc', 'params': {'stride': 0}},
            {'strategy': '递归分块', 'text': 'abc', 'params': {'chunk_size': 4, 'chunk_overlap': 5}},
            {'strategy': '固定长度分块', 'text': '   '},
            {'strategy': 'missing', 'text': 'abc'},
            {'strategy': '固定长度分块', 'text': 'abc', 'params': {'surprise': 1}},
            {'strategy': '固定长度分块', 'text': 'a' * 50001},
            {'strategy': '递归分块', 'text': 'a' * 2000, 'params': {'chunk_size': 1000, 'chunk_overlap': 999}},
            {'strategy': '固定长度分块', 'text': 'a' * 2000, 'params': {'chunk_size': 1}},
        ]:
            with self.subTest(payload=str(payload)[:100]):
                self.assertEqual(self.client.post('/api/chunking/run', json=payload).status_code, 422)

    def test_embedding_query_batched_once_and_ranked(self):
        calls = []
        class Provider:
            def embed(self, texts, **kwargs):
                calls.append(texts)
                return EmbeddingResult([[1., 0.], [0., 1.], [1., 0.]], 'fixture', texts, {})
        with patch('ragent_lab.api.services.get_model', return_value=Provider()):
            response = self.client.post('/api/embedding/run', json={'model': 'openai_text_embedding_3_small', 'texts': ['苹果', '汽车'], 'query': '水果', 'api_key': 'test-secret'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(calls, [['苹果', '汽车', '水果']])
        self.assertEqual(data['rankings'][0]['text'], '苹果')
        self.assertEqual(data['rankings'][0]['score'], 1.0)
        self.assertEqual(data['dimension'], 2)
        self.assertNotIn('test-secret', response.text)

    def test_provider_error_does_not_expose_credentials(self):
        class Provider:
            def embed(self, *args, **kwargs):
                raise RuntimeError('failed with secret-value')
        with patch('ragent_lab.api.services.get_model', return_value=Provider()):
            response = self.client.post('/api/embedding/run', json={'model': 'openai_text_embedding_3_small', 'texts': ['苹果'], 'api_key': 'secret-value'})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('secret-value', response.text)

    def test_server_key_not_used_by_default(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'server-key', 'ALLOW_SERVER_API_KEYS': 'false'}):
            response = self.client.post('/api/embedding/run', json={'model': 'openai_text_embedding_3_small', 'texts': ['苹果']})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('server-key', response.text)

    def test_validation_does_not_echo_api_key(self):
        response = self.client.post('/api/embedding/run', json={'model': 'openai_text_embedding_3_small', 'texts': [], 'api_key': 'secret-value'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('secret-value', response.text)


if __name__ == '__main__':
    unittest.main()
