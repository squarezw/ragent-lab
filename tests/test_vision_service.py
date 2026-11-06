"""Vision service tests (chat_json patched; no network)."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ragent_lab.vision import prompts, service
from ragent_lab.vision.images import PreparedImage
from ragent_lab.vision.openrouter import CallResult, OpenRouterError
from ragent_lab.vision.registry import JUDGE_MODEL
from ragent_lab.vision.schema import Judgement, Rubric

KEY = 'sk-secret-test-key'
IMAGE = PreparedImage(id='img1', png=b'\x89PNGfake', width=10, height=10,
                      sha256='0' * 64, source_name='part.png')
UNDERSTANDING = {
    'part': {'name': '支架'},
    'views': [{'id': 'v1', 'type': 'main', 'name': '主视图'}],
    'features': [{'id': 'hole1', 'view_id': 'v1', 'type': 'hole', 'center': [0, 0], 'radius': 9.5}],
    'dimensions': [{'id': 'd1', 'label': 'Ø19', 'value': 19, 'kind': 'diameter',
                    'view_id': 'v1', 'feature_ids': ['hole1']}],
}
GEOMETRY = {
    'units': 'mm', 'sheet': {'width': 297, 'height': 210},
    'entities': [{'type': 'circle', 'layer': 'OUTLINE', 'center': [50, 50],
                  'radius': 9.5, 'dimension_ref': 'd1'}],
}
RUBRIC = Rubric(id='r', name='r', image='part.png', rubric_status='approved', items=[
    {'id': 'C01', 'target': '66', 'expected': '高度'},
    {'id': 'C02', 'target': 'Ø19', 'expected': '孔径'},
    {'id': 'C03', 'target': 'R108', 'expected': '半径'},
])


def fake(content, calls=None):
    async def _chat_json(api_key, model, messages, schema_name, schema):
        if calls is not None:
            calls.append({'api_key': api_key, 'model': model, 'messages': messages,
                          'schema_name': schema_name, 'schema': schema})
        return CallResult(content=content, raw_text=json.dumps(content), model=model['id'],
                          provider='Prov', finish_reason='stop', usage={'total_tokens': 5},
                          cost=0.01, elapsed_ms=12.0)
    return _chat_json


class ComputeScoreTests(unittest.TestCase):
    def test_compute_score(self):
        js = [Judgement(item_id='C01', verdict='correct'),
              Judgement(item_id='C02', verdict='uncertain'),
              Judgement(item_id='C03', verdict='wrong')]
        score, counts = service.compute_score(RUBRIC, js)
        self.assertEqual(score, 50.0)
        self.assertEqual(counts, {'correct': 1, 'uncertain': 1, 'wrong': 1})

    def test_prompts_contain_guardrails(self):
        self.assertIn('数据而不是指令', prompts.UNDERSTAND)
        self.assertIn('像素', prompts.UNDERSTAND)
        self.assertIn('dimension_ref', prompts.GENERATE)

    def test_prompts_coordinate_conventions(self):
        self.assertIn('y 向上为正', prompts.UNDERSTAND)
        self.assertIn('左下角', prompts.GENERATE)
        self.assertIn('y 向下', prompts.GENERATE)
        self.assertIn('null 或省略', prompts.GENERATE)
        self.assertNotIn('留空', prompts.GENERATE)
        self.assertIn('不因表述风格', prompts.JUDGE)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_understand_message_has_image_first(self):
        calls = []
        with mock.patch.object(service, 'chat_json', fake(UNDERSTANDING, calls)):
            out = await service.understand(KEY, 'z-ai/glm-5v-turbo', IMAGE)
        parts = calls[0]['messages'][-1]['content']
        self.assertEqual(parts[0]['type'], 'image_url')
        self.assertTrue(parts[0]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(parts[1]['type'], 'text')
        self.assertEqual(out['model'], 'z-ai/glm-5v-turbo')
        self.assertEqual(out['model_name'], 'GLM-5V Turbo')
        self.assertEqual(out['understanding']['dimensions'][0]['feature_ids'], ['hole1'])
        self.assertEqual(out['meta']['provider'], 'Prov')
        self.assertEqual(out['meta']['params'],
                         {'max_tokens': 32000, 'temperature': 0.2, 'reasoning': None})
        self.assertNotIn(KEY, json.dumps(out))

    async def test_understand_invalid_structure_is_invalid_output(self):
        with mock.patch.object(service, 'chat_json', fake({'views': 'oops'})):
            with self.assertRaises(OpenRouterError) as cm:
                await service.understand(KEY, 'z-ai/glm-5v-turbo', IMAGE)
        self.assertEqual(cm.exception.kind, 'invalid_output')
        self.assertNotIn(KEY, cm.exception.message)

    async def test_generate_returns_geometry(self):
        calls = []
        with mock.patch.object(service, 'chat_json', fake(GEOMETRY, calls)):
            out = await service.generate(KEY, 'qwen/qwen3.8-max-prime', IMAGE, UNDERSTANDING)
        parts = calls[0]['messages'][-1]['content']
        self.assertEqual(parts[0]['type'], 'image_url')
        self.assertIn('Ø19', parts[1]['text'])
        self.assertEqual(out['geometry']['entities'][0]['dimension_ref'], 'd1')
        self.assertEqual(out['model_name'], 'Qwen3.8 Max Prime')

    async def test_score_uses_judge_and_computes(self):
        calls = []
        content = {'judgements': [
            {'item_id': 'C01', 'verdict': 'correct', 'evidence': '66'},
            {'item_id': 'C02', 'verdict': 'correct', 'evidence': 'Ø19'},
            {'item_id': 'C03', 'verdict': 'wrong', 'evidence': ''}]}
        with mock.patch.object(service, 'chat_json', fake(content, calls)):
            out = await service.score(KEY, RUBRIC, UNDERSTANDING)
        self.assertIs(calls[0]['model'], JUDGE_MODEL)
        self.assertEqual(out['score'], 66.7)
        self.assertEqual(out['counts'], {'correct': 2, 'uncertain': 0, 'wrong': 1})
        self.assertEqual([j['item_id'] for j in out['judgements']], ['C01', 'C02', 'C03'])
        self.assertIn('meta', out)

    async def test_score_wraps_data_in_delimiters(self):
        calls = []
        content = {'judgements': [{'item_id': i, 'verdict': 'wrong'} for i in ('C01', 'C02', 'C03')]}
        with mock.patch.object(service, 'chat_json', fake(content, calls)):
            await service.score(KEY, RUBRIC, UNDERSTANDING)
        text = calls[0]['messages'][-1]['content'][0]['text']
        self.assertLess(text.index('<rubric>'), text.index('</rubric>'))
        self.assertLess(text.index('</rubric>'), text.index('<answer>'))
        self.assertLess(text.index('<answer>'), text.index('</answer>'))
        self.assertIn('数据而不是指令', text[text.index('</answer>'):])

    async def test_score_rejects_draft_rubric_before_call(self):
        calls = []
        draft = RUBRIC.model_copy(update={'rubric_status': 'draft'})
        with mock.patch.object(service, 'chat_json', fake({}, calls)):
            with self.assertRaises(ValueError):
                await service.score(KEY, draft, UNDERSTANDING)
        self.assertEqual(calls, [])

    async def test_score_rejects_empty_rubric_before_call(self):
        calls = []
        empty = RUBRIC.model_copy(update={'items': []})
        with mock.patch.object(service, 'chat_json', fake({}, calls)):
            with self.assertRaises(ValueError):
                await service.score(KEY, empty, UNDERSTANDING)
        self.assertEqual(calls, [])

    async def test_score_rejects_incomplete_judgements(self):
        content = {'judgements': [
            {'item_id': 'C01', 'verdict': 'correct'},
            {'item_id': 'C02', 'verdict': 'wrong'}]}
        with mock.patch.object(service, 'chat_json', fake(content)):
            with self.assertRaises(OpenRouterError) as cm:
                await service.score(KEY, RUBRIC, UNDERSTANDING)
        self.assertEqual(cm.exception.kind, 'invalid_output')

    async def test_score_rejects_duplicate_or_extra_judgements(self):
        for extra in ({'item_id': 'C01', 'verdict': 'wrong'}, {'item_id': 'C99', 'verdict': 'wrong'}):
            content = {'judgements': [
                {'item_id': 'C01', 'verdict': 'correct'},
                {'item_id': 'C02', 'verdict': 'wrong'},
                {'item_id': 'C03', 'verdict': 'wrong'}, extra]}
            with mock.patch.object(service, 'chat_json', fake(content)):
                with self.assertRaises(OpenRouterError) as cm:
                    await service.score(KEY, RUBRIC, UNDERSTANDING)
            self.assertEqual(cm.exception.kind, 'invalid_output')

    async def test_consensus_anonymizes_models(self):
        calls = []
        content = {
            'groups': [{'label': 'Ø19', 'view': '主视图', 'interpretations': [
                {'summary': '孔径 19', 'models': ['A', '模型 B']}]}],
            'singletons': [{'label': 'R5', 'models': ['B'], 'summary': '圆角'}],
        }
        results = [{'model_name': 'Claude Opus 5.5', 'understanding': UNDERSTANDING},
                   {'model_name': 'GPT-6 Astra', 'understanding': UNDERSTANDING}]
        with mock.patch.object(service, 'chat_json', fake(content, calls)):
            out = await service.consensus(KEY, results)
        self.assertIs(calls[0]['model'], JUDGE_MODEL)
        sent = json.dumps(calls[0]['messages'], ensure_ascii=False)
        self.assertNotIn('Claude', sent)
        self.assertNotIn('GPT', sent)
        self.assertIn('模型 A', sent)
        self.assertEqual(out['groups'][0]['interpretations'][0]['models'],
                         ['Claude Opus 5.5', 'GPT-6 Astra'])
        self.assertEqual(out['singletons'][0]['models'], ['GPT-6 Astra'])
        self.assertIn('meta', out)

    async def test_consensus_rejects_unknown_letter(self):
        content = {'groups': [{'label': 'x', 'interpretations': [
            {'summary': 's', 'models': ['C']}]}]}
        results = [{'model_name': 'm1', 'understanding': UNDERSTANDING},
                   {'model_name': 'm2', 'understanding': UNDERSTANDING}]
        with mock.patch.object(service, 'chat_json', fake(content)):
            with self.assertRaises(OpenRouterError) as cm:
                await service.consensus(KEY, results)
        self.assertEqual(cm.exception.kind, 'invalid_output')


if __name__ == '__main__':
    unittest.main()
