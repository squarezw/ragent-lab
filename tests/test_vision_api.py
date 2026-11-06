"""Vision HTTP API tests. Model services are doubles; no network, no real keys."""
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastapi.testclient import TestClient
from PIL import Image
from ragent_lab.api.app import app
from ragent_lab.vision import images as images_mod

KEY = 'sk-fixture-abc123'
MODEL = 'anthropic/claude-opus-5.5'
UNDERSTANDING = {'part': {'name': '测试件'}, 'views': [], 'dimensions': []}
GEOMETRY = {'units': 'mm', 'sheet': {'width': 297, 'height': 210}, 'entities': [
    {'type': 'circle', 'layer': 'OUTLINE', 'center': [150, 100], 'radius': 40}]}


def _png(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, 'white').save(buf, 'PNG')
    return buf.getvalue()


def _write_sample(root: Path, sid: str, status: str) -> None:
    folder = root / sid
    folder.mkdir()
    (folder / 'img.png').write_bytes(_png())
    (folder / 'rubric.json').write_text(json.dumps({
        'id': sid, 'name': f'样例{sid}', 'image': 'img.png', 'rubric_status': status,
        'items': [{'id': 'i1', 'target': 't', 'expected': 'e'}]}), 'utf-8')


async def _fake_understand(api_key, model_id, image):
    return {'model': model_id, 'model_name': 'X', 'understanding': UNDERSTANDING, 'meta': {}}


async def _fake_score(api_key, rubric, understanding, judge):
    return {'score': 100.0, 'counts': {}, 'judgements': [], 'meta': {'judge': judge['id']}}


async def _fake_consensus(api_key, results, judge):
    return {'groups': [], 'singletons': [], 'meta': {'judge': judge['id']}}


async def _fake_generate(api_key, model_id, image, understanding):
    return {'model': model_id, 'model_name': 'X', 'geometry': GEOMETRY, 'meta': {'cost': 0}}


class VisionAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        _write_sample(root, 'ok', 'approved')
        _write_sample(root, 'dr', 'draft')
        env = patch.dict(os.environ, {'VISION_SAMPLES_DIR': str(root)})
        env.start()
        self.addCleanup(env.stop)
        images_mod.reset_samples_cache()
        self.addCleanup(images_mod.reset_samples_cache)
        self.addCleanup(self.tmp.cleanup)
        for name, fake in (('understand', _fake_understand), ('score', _fake_score),
                           ('consensus', _fake_consensus), ('generate', _fake_generate)):
            p = patch(f'ragent_lab.api.vision.service.{name}', fake)
            p.start()
            self.addCleanup(p.stop)
        p = patch('ragent_lab.api.vision.model_prices', _fake_prices)
        p.start()
        self.addCleanup(p.stop)

    def _wait(self, client, job_id):
        texts = []
        for _ in range(100):
            r = client.get(f'/api/vision/jobs/{job_id}')
            texts.append(r.text)
            self.assertEqual(r.status_code, 200)
            if r.json()['status'] in ('done', 'failed'):
                return r.json(), texts
            time.sleep(0.05)
        self.fail('job did not finish')

    def _upload(self, client):
        r = client.post('/api/vision/images', files={'file': ('a.png', _png(), 'image/png')})
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def test_models_and_samples(self):
        with TestClient(app) as client:
            models = client.get('/api/vision/models').json()
            self.assertEqual(len(models), 9)
            self.assertEqual(set(models[0]), {'id', 'name', 'family', 'notes', 'params', 'price'})
            self.assertEqual(set(models[0]['params']), {'max_tokens', 'temperature', 'reasoning'})
            self.assertEqual(client.get('/api/vision/judge').json()['id'], MODEL)
            samples = {s['id']: s for s in client.get('/api/vision/samples').json()}
            self.assertEqual(len(samples), 2)
            self.assertTrue(samples['ok']['scorable'])
            self.assertFalse(samples['dr']['scorable'])
            r = client.get('/api/vision/samples/ok/image')
            self.assertEqual(r.headers['content-type'], 'image/png')
            self.assertEqual(client.get('/api/vision/samples/nope/image').status_code, 404)

    def test_upload_then_understand_job_flow(self):
        with TestClient(app) as client:
            up = self._upload(client)
            body = up.json()
            self.assertEqual((body['width'], body['height']), (64, 48))
            self.assertEqual(body['name'], 'a.png')
            texts = [up.text]
            self.assertEqual(client.get(f"/api/vision/images/{body['image_id']}").headers['content-type'],
                             'image/png')
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'upload', 'id': body['image_id']}, 'model': MODEL, 'api_key': KEY})
            self.assertEqual(r.status_code, 200, r.text)
            texts.append(r.text)
            job, polled = self._wait(client, r.json()['job_id'])
            texts += polled
            self.assertEqual(job['status'], 'done')
            self.assertIn('understanding', job['result'])
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL, 'api_key': KEY})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(all(KEY not in t for t in texts))

    def test_score_consensus_generate_jobs(self):
        with TestClient(app) as client:
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': UNDERSTANDING, 'api_key': KEY})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['score'], 100.0)
            self.assertEqual(job['result']['meta']['judge'], 'anthropic/claude-opus-5.5')
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': UNDERSTANDING, 'api_key': KEY,
                'judge_model': 'qwen/qwen3.8-max-prime'})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['meta']['judge'], 'qwen/qwen3.8-max-prime')
            two = [{'model_name': 'A', 'understanding': UNDERSTANDING},
                   {'model_name': 'B', 'understanding': UNDERSTANDING}]
            r = client.post('/api/vision/consensus', json={'results': two, 'api_key': KEY})
            self.assertEqual(self._wait(client, r.json()['job_id'])[0]['status'], 'done')
            r = client.post('/api/vision/consensus', json={
                'results': two, 'api_key': KEY, 'judge_model': 'google/gemini-4-argon'})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['meta']['judge'], 'google/gemini-4-argon')
            for path, extra in (('score', {'sample_id': 'ok', 'understanding': UNDERSTANDING}),
                                ('consensus', {'results': two})):
                r = client.post(f'/api/vision/{path}', json={
                    **extra, 'api_key': KEY, 'judge_model': 'x/unknown'})
                self.assertEqual(r.status_code, 422)
                self.assertIn('未知评判模型', r.text)
            r = client.post('/api/vision/consensus', json={'results': two[:1], 'api_key': KEY})
            self.assertEqual(r.status_code, 422)
            r = client.post('/api/vision/generate', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL,
                'understanding': UNDERSTANDING, 'api_key': KEY})
            job, polled = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['status'], 'done', job)
            self.assertEqual(set(job['result']), {'geometry', 'svg', 'warnings', 'skipped', 'meta'})
            self.assertIn('<svg', job['result']['svg'])
            self.assertTrue(all(KEY not in t for t in polled))

    def test_missing_key_and_unknown_model_rejected(self):
        with TestClient(app) as client:
            img = {'kind': 'sample', 'id': 'ok'}
            r = client.post('/api/vision/understand', json={'image': img, 'model': MODEL})
            self.assertEqual(r.status_code, 422)
            r = client.post('/api/vision/understand', json={'image': img, 'model': MODEL, 'api_key': ''})
            self.assertEqual(r.status_code, 422)
            self.assertIn('请填写 OpenRouter Key', r.text)
            r = client.post('/api/vision/understand', json={'image': img, 'model': 'x/unknown', 'api_key': KEY})
            self.assertEqual(r.status_code, 422)
            self.assertIn('未知模型', r.text)
            self.assertNotIn(KEY, r.text)
            r = client.post('/api/vision/understand', json={
                'image': img, 'model': MODEL, 'api_key': KEY, 'extra': 1})
            self.assertEqual(r.status_code, 422)
            self.assertNotIn(KEY, r.text)

    def test_score_requires_approved_rubric(self):
        with TestClient(app) as client:
            for sid in ('dr', 'nope'):
                r = client.post('/api/vision/score', json={
                    'sample_id': sid, 'understanding': UNDERSTANDING, 'api_key': KEY})
                self.assertEqual(r.status_code, 422)

    def test_export_pdf_and_dxf(self):
        with TestClient(app) as client:
            r = client.post('/api/vision/export', json={
                'geometry': GEOMETRY, 'format': 'pdf', 'filename': '图纸/a b'})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.headers['content-type'], 'application/pdf')
            self.assertTrue(r.content.startswith(b'%PDF'))
            self.assertIn('attachment', r.headers['content-disposition'])
            r = client.post('/api/vision/export', json={
                'geometry': GEOMETRY, 'format': 'dxf', 'filename': 'a.dxf'})
            self.assertEqual(r.status_code, 200)
            self.assertIn('attachment', r.headers['content-disposition'])
            self.assertIn('.dxf', r.headers['content-disposition'])
            self.assertIn(b'SECTION', r.content)

    def test_export_rejects_over_budget_geometry(self):
        text = {'type': 'text', 'layer': 'TEXT', 'insert': [0, 0], 'height': 5}
        poly = {'type': 'polyline', 'layer': 'OUTLINE', 'points': [[i % 100, 0] for i in range(2000)]}
        cases = {
            'long text': [{**text, 'content': 'x' * 201}],
            'huge text height': [{**text, 'content': 'a', 'height': 1001}],
            'polyline points': [{**poly, 'points': poly['points'] + [[0, 0]]}],
            'total vertices': [poly] * 26,
            'total chars': [{**text, 'content': 'x' * 200}] * 101,
        }
        with TestClient(app) as client:
            for name, ents in cases.items():
                with self.subTest(name):
                    geometry = {**GEOMETRY, 'entities': ents}
                    r = client.post('/api/vision/export', json={
                        'geometry': geometry, 'format': 'dxf', 'filename': 'a'})
                    self.assertEqual(r.status_code, 422)

    def test_expired_job_and_image_404(self):
        with TestClient(app) as client:
            r = client.get('/api/vision/jobs/deadbeef')
            self.assertEqual(r.status_code, 404)
            self.assertIn('任务已失效，请重新运行。', r.text)
            r = client.get('/api/vision/images/deadbeef')
            self.assertEqual(r.status_code, 404)
            self.assertIn('图片已过期，请重新上传。', r.text)
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'upload', 'id': 'gone'}, 'model': MODEL, 'api_key': KEY})
            self.assertEqual(r.status_code, 404)

    def test_upload_too_large_and_bad_file_rejected(self):
        with TestClient(app) as client:
            r = client.post('/api/vision/images', files={
                'file': ('big.png', b'0' * (16 * 1024 * 1024), 'image/png')})
            self.assertEqual(r.status_code, 413)
            r = client.post('/api/vision/images', files={'file': ('x.png', b'not an image', 'image/png')})
            self.assertEqual(r.status_code, 422)


class RenderBoundTests(unittest.TestCase):
    def setUp(self):
        from ragent_lab.api import vision as api_vision
        self.api = api_vision

    def test_concurrency_capped_at_two(self):
        import asyncio
        import threading
        lock, state = threading.Lock(), {'now': 0, 'max': 0}

        def work():
            with lock:
                state['now'] += 1
                state['max'] = max(state['max'], state['now'])
            time.sleep(0.15)
            with lock:
                state['now'] -= 1
            return 1

        async def main():
            return await asyncio.gather(*(self.api.bounded_render('export', work) for _ in range(5)))
        self.assertEqual(asyncio.run(main()), [1] * 5)
        self.assertEqual(state['max'], 2)

    def test_budget_exceeded_raises_busy_and_keeps_slot_until_done(self):
        import asyncio
        with patch.object(self.api, 'RENDER_BUDGET', 0.1):
            async def main():
                with self.assertRaises(self.api.RenderBusy):
                    await self.api.bounded_render('export', time.sleep, 0.4)
                with self.assertRaises(self.api.RenderBusy):
                    await asyncio.gather(*(self.api.bounded_render('export', time.sleep, 0.4) for _ in range(3)))
            asyncio.run(main())

    def test_export_pool_does_not_block_generate_pool(self):
        import asyncio
        with patch.object(self.api, 'RENDER_BUDGET', 0.3):
            async def main():
                hogs = [asyncio.ensure_future(self.api.bounded_render('export', time.sleep, 0.5))
                        for _ in range(2)]
                await asyncio.sleep(0.05)
                got = await self.api.bounded_render('generate', lambda: 'ok')
                with self.assertRaises(self.api.RenderBusy):
                    await self.api.bounded_render('export', lambda: 'x')
                await asyncio.gather(*hogs, return_exceptions=True)
                return got
            self.assertEqual(asyncio.run(main()), 'ok')

    def test_export_returns_503_when_busy(self):
        def slow(*_a):
            time.sleep(0.4)
            return b''
        with patch.object(self.api, 'RENDER_BUDGET', 0.1), \
                patch.object(self.api, '_render_export', slow), TestClient(app) as client:
            r = client.post('/api/vision/export', json={
                'geometry': GEOMETRY, 'format': 'dxf', 'filename': 'a'})
            self.assertEqual(r.status_code, 503)
            self.assertIn('导出繁忙或超时，请稍后重试。', r.text)


async def _fake_prices():
    return {}


if __name__ == '__main__':
    unittest.main()
