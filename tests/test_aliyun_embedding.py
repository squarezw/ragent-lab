"""Model service request boundaries without network calls."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ragent_lab.embedding.aliyun_embedding import AliyunEmbeddingModel


class AliyunTests(unittest.TestCase):
    def test_v2_batches_requests_at_provider_limit_and_preserves_order(self):
        payloads = []
        def create(**kwargs):
            if len(kwargs['input']) > 25:
                raise ValueError('provider batch size exceeded')
            if 'dimensions' in kwargs:
                raise ValueError('v2 has fixed dimensions')
            payloads.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[float(t), 1.0]) for t in kwargs['input']], usage=SimpleNamespace(total_tokens=len(kwargs['input'])))
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        model = AliyunEmbeddingModel('text-embedding-v2')
        with patch.object(model, '_get_client', return_value=client):
            result = model.embed([str(i) for i in range(32)], api_key='fixture')
        self.assertEqual(len(payloads), 2)
        self.assertEqual(result.count, 32)
        self.assertEqual(result.embeddings[-1], [31.0, 1.0])
