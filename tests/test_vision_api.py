"""Vision HTTP API tests. Model services are doubles; no network, no real keys."""
import asyncio
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

KEY = 'sk-fixture-abc123'  # the server's catalog key in these tests
CODE = 'lab-code'
SERVER_ENV = {'AIHUBMIX_API_KEY': KEY, 'LAB_ACCESS_CODE': CODE, 'VISION_BUILTIN_PER_HOUR': '1000'}
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


SEEN_KEYS: list[str] = []


async def _fake_understand(api_key, model_id, image, gateway, purpose='engineering',
                           instructions=None):
    SEEN_KEYS.append(api_key)
    model = model_id if isinstance(model_id, dict) else {'id': model_id, 'name': 'X'}
    return {'model': model['id'], 'model_name': model['name'], 'understanding': UNDERSTANDING,
            'purpose': purpose, 'instructions': instructions,
            'meta': {'gateway': gateway, 'endpoint': (model.get('endpoint') or {}).get('label')}}


async def _fake_score(api_key, rubric, understanding, judge, gateway):
    SEEN_KEYS.append(api_key)
    return {'score': 100.0, 'counts': {}, 'judgements': [],
            'meta': {'judge': judge['id'], 'gateway': gateway}}


async def _fake_consensus(api_key, results, judge, gateway):
    return {'groups': [], 'singletons': [], 'meta': {'judge': judge['id'], 'gateway': gateway}}


async def _fake_generate(api_key, model_id, image, understanding, gateway):
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
        from ragent_lab.vision.builtin import Builtins
        p = patch('ragent_lab.api.vision.builtins', Builtins(SERVER_ENV))
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
            self.assertEqual(len(models), 8)
            self.assertEqual([m['name'] for m in models[:2]], ['Claude Fable 5.2', 'Claude Opus 5.5'])
            self.assertEqual(set(models[0]), {'id', 'name', 'family', 'notes', 'params'})
            self.assertNotIn('aihubmix', json.dumps(models).lower())
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
                'image': {'kind': 'upload', 'id': body['image_id']}, 'model': MODEL, 'access_code': CODE})
            self.assertEqual(r.status_code, 200, r.text)
            texts.append(r.text)
            job, polled = self._wait(client, r.json()['job_id'])
            texts += polled
            self.assertEqual(job['status'], 'done')
            self.assertIn('understanding', job['result'])
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL, 'access_code': CODE})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(all(KEY not in t for t in texts))

    def test_score_consensus_generate_jobs(self):
        with TestClient(app) as client:
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': UNDERSTANDING, 'access_code': CODE})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['score'], 100.0)
            self.assertEqual(job['result']['meta']['judge'], 'anthropic/claude-opus-5.5')
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': UNDERSTANDING, 'access_code': CODE,
                'judge_model': 'openai/gpt-6-astra'})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['meta']['judge'], 'openai/gpt-6-astra')
            two = [{'model_name': 'A', 'understanding': UNDERSTANDING},
                   {'model_name': 'B', 'understanding': UNDERSTANDING}]
            r = client.post('/api/vision/consensus', json={'results': two, 'access_code': CODE})
            self.assertEqual(self._wait(client, r.json()['job_id'])[0]['status'], 'done')
            r = client.post('/api/vision/consensus', json={
                'results': two, 'access_code': CODE, 'judge_model': 'google/gemini-4-argon'})
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['meta']['judge'], 'google/gemini-4-argon')
            for path, extra in (('score', {'sample_id': 'ok', 'understanding': UNDERSTANDING}),
                                ('consensus', {'results': two})):
                r = client.post(f'/api/vision/{path}', json={
                    **extra, 'access_code': CODE, 'judge_model': 'x/unknown'})
                self.assertEqual(r.status_code, 422)
                self.assertIn('未知评判模型', r.text)
            r = client.post('/api/vision/consensus', json={'results': two[:1], 'access_code': CODE})
            self.assertEqual(r.status_code, 422)
            r = client.post('/api/vision/generate', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL,
                'understanding': UNDERSTANDING, 'access_code': CODE})
            job, polled = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['status'], 'done', job)
            self.assertEqual(set(job['result']), {'geometry', 'svg', 'warnings', 'skipped', 'meta'})
            self.assertIn('<svg', job['result']['svg'])
            self.assertTrue(all(KEY not in t for t in polled))

    def test_catalog_needs_the_access_code(self):
        with TestClient(app) as client:
            img = {'kind': 'sample', 'id': 'ok'}
            r = client.post('/api/vision/understand', json={'image': img, 'model': MODEL})
            self.assertEqual(r.status_code, 422)
            self.assertIn('访问码', r.json()['detail'])
            r = client.post('/api/vision/understand', json={
                'image': img, 'model': MODEL, 'access_code': 'nope'})
            self.assertEqual(r.status_code, 403)
            r = client.post('/api/vision/understand', json={
                'image': img, 'model': 'x/unknown', 'access_code': CODE})
            self.assertEqual(r.status_code, 422)
            self.assertIn('未知模型', r.text)
            # Old clients that still send a platform or their own key are refused.
            for extra in ({'gateway': 'openrouter'}, {'api_key': 'sk-mine'}):
                r = client.post('/api/vision/understand', json={
                    'image': img, 'model': MODEL, 'access_code': CODE, **extra})
                self.assertEqual(r.status_code, 422)
            r = client.post('/api/vision/understand', json={
                'image': img, 'model': {'source': 'catalog', 'id': MODEL}, 'access_code': CODE})
            job, polled = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['meta']['gateway'], 'catalog')
            self.assertEqual(SEEN_KEYS[-1], KEY)
            self.assertTrue(all(KEY not in t for t in polled))

    def test_catalog_hidden_without_server_key(self):
        from ragent_lab.vision.builtin import Builtins
        with patch('ragent_lab.api.vision.builtins', Builtins({'LAB_ACCESS_CODE': CODE})), \
                TestClient(app) as client:
            self.assertEqual(client.get('/api/vision/models').json(), [])
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL, 'access_code': CODE})
            self.assertEqual(r.status_code, 422)

    def test_score_requires_approved_rubric(self):
        with TestClient(app) as client:
            for sid in ('dr', 'nope'):
                r = client.post('/api/vision/score', json={
                    'sample_id': sid, 'understanding': UNDERSTANDING, 'access_code': CODE})
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
                'image': {'kind': 'upload', 'id': 'gone'}, 'model': MODEL, 'access_code': CODE})
            self.assertEqual(r.status_code, 404)

    def test_upload_too_large_and_bad_file_rejected(self):
        with TestClient(app) as client:
            r = client.post('/api/vision/images', files={
                'file': ('big.png', b'0' * (16 * 1024 * 1024), 'image/png')})
            self.assertEqual(r.status_code, 413)
            r = client.post('/api/vision/images', files={'file': ('x.png', b'not an image', 'image/png')})
            self.assertEqual(r.status_code, 422)


GENERAL = {'summary': '一张流程图', 'elements': [{'id': 'n1', 'type': '步骤', 'label': '开始'}]}


PROMPT_CODE = 'prompt-pw'


class PurposeAPITests(unittest.TestCase):
    _wait = VisionAPITests._wait

    def setUp(self):
        VisionAPITests.setUp(self)
        self._serve({**SERVER_ENV, 'LAB_PROMPT_CODE': PROMPT_CODE})

    def _serve(self, env):
        from ragent_lab.vision.builtin import Builtins
        p = patch('ragent_lab.api.vision.builtins', Builtins(env))
        p.start()
        self.addCleanup(p.stop)

    def _understand(self, client, **extra):
        return client.post('/api/vision/understand', json={
            'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL, 'access_code': CODE, **extra})

    def test_public_purposes_carry_no_prompt_text(self):
        with TestClient(app) as client:
            out = client.get('/api/vision/purposes').json()
            info = client.get('/api/vision/builtin').json()
        self.assertEqual([p['id'] for p in out], ['engineering', 'chart', 'diagram', 'general'])
        self.assertEqual(out[0]['kind'], 'engineering')
        self.assertTrue(all(set(p) == {'id', 'name', 'kind', 'description'} for p in out))
        self.assertEqual({p['kind'] for p in out[1:]}, {'general'})
        self.assertIs(info['prompts_editable'], True)
        self.assertNotIn(PROMPT_CODE, json.dumps(info))

    def test_unlock_returns_prompts_only_for_the_right_password(self):
        with TestClient(app) as client:
            for body in ({'prompt_code': 'nope'}, {'prompt_code': ''}, {'prompt_code': CODE}):
                r = client.post('/api/vision/purposes/unlock', json=body)
                self.assertEqual((r.status_code, r.json()['detail']), (403, '提示词管理密码不正确'))
            self.assertEqual(client.post('/api/vision/purposes/unlock', json={}).status_code, 422)
            out = client.post('/api/vision/purposes/unlock',
                              json={'prompt_code': f' {PROMPT_CODE} '}).json()
        self.assertEqual([p['id'] for p in out], ['engineering', 'chart', 'diagram', 'general'])
        self.assertTrue(all(p['prompt'] for p in out))

    def test_wrong_prompt_passwords_lock_the_ip_out(self):
        with TestClient(app) as client:
            for _ in range(10):
                client.post('/api/vision/purposes/unlock', json={'prompt_code': 'nope'})
            r = client.post('/api/vision/purposes/unlock', json={'prompt_code': PROMPT_CODE})
            self.assertEqual(r.status_code, 403)

    def test_no_prompt_password_on_the_server_means_nobody_can_manage(self):
        self._serve(SERVER_ENV)
        with TestClient(app) as client:
            self.assertIs(client.get('/api/vision/builtin').json()['prompts_editable'], False)
            for code in ('', PROMPT_CODE):
                r = client.post('/api/vision/purposes/unlock', json={'prompt_code': code})
                self.assertEqual((r.status_code, r.json()['detail']), (403, '服务器未开放提示词管理'))
            self.assertEqual(self._understand(client, prompt='x', prompt_code='').status_code, 403)
            self.assertEqual(self._understand(client).status_code, 200)  # default prompt still runs

    def test_default_prompt_needs_no_password(self):
        with TestClient(app) as client:
            job, _ = self._wait(client, self._understand(client, purpose='diagram').json()['job_id'])
            self.assertEqual((job['result']['purpose'],
                              job['result']['instructions']), ('diagram', None))

    def test_own_prompt_reaches_the_service_only_with_the_password(self):
        with TestClient(app) as client:
            for extra in ({}, {'prompt_code': 'nope'}, {'prompt_code': CODE}):
                r = self._understand(client, purpose='diagram', prompt='只列出节点', **extra)
                self.assertEqual((r.status_code, r.json()['detail']), (403, '提示词管理密码不正确'))
            r = self._understand(client, purpose='diagram', prompt='只列出节点',
                                 prompt_code=PROMPT_CODE)
            job, _ = self._wait(client, r.json()['job_id'])
            self.assertEqual((job['result']['purpose'],
                              job['result']['instructions']), ('diagram', '只列出节点'))

    def test_bad_purpose_or_long_prompt_rejected(self):
        with TestClient(app) as client:
            ok = {'prompt_code': PROMPT_CODE}
            self.assertEqual(self._understand(client, purpose='poetry').status_code, 422)
            self.assertEqual(self._understand(client, prompt='x' * 8001, **ok).status_code, 422)
            self.assertEqual(self._understand(client, prompt='x' * 8000, **ok).status_code, 200)

    def test_score_and_consensus_accept_general_results(self):
        with TestClient(app) as client:
            samples = {s['id']: s for s in client.get('/api/vision/samples').json()}
            self.assertEqual(samples['ok']['kind'], 'engineering')
            # The sample's rubric grades engineering output; a general result cannot be scored.
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': GENERAL, 'access_code': CODE})
            self.assertEqual(r.status_code, 422)
            self.assertIn('用途', r.json()['detail'])
            two = [{'model_name': 'A', 'understanding': GENERAL},
                   {'model_name': 'B', 'understanding': GENERAL}]
            r = client.post('/api/vision/consensus', json={'results': two, 'access_code': CODE})
            self.assertEqual(r.status_code, 200, r.text)
            mixed = [two[0], {'model_name': 'B', 'understanding': UNDERSTANDING}]
            r = client.post('/api/vision/consensus', json={'results': mixed, 'access_code': CODE})
            self.assertEqual(r.status_code, 422)
            # Redraw stays engineering-only.
            r = client.post('/api/vision/generate', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL,
                'understanding': GENERAL, 'access_code': CODE})
            self.assertEqual(r.status_code, 422)


class CancelAPITests(unittest.TestCase):
    _wait = VisionAPITests._wait

    def setUp(self):
        VisionAPITests.setUp(self)

    def test_cancel_job(self):
        started = []

        async def hang(*a, **k):
            started.append(True)
            await asyncio.sleep(30)

        with patch('ragent_lab.api.vision.service.understand', hang), TestClient(app) as client:
            r = client.post('/api/vision/understand', json={
                'image': {'kind': 'sample', 'id': 'ok'}, 'model': MODEL, 'access_code': CODE})
            jid = r.json()['job_id']
            for _ in range(50):
                if started:
                    break
                time.sleep(0.02)
            r = client.post(f'/api/vision/jobs/{jid}/cancel')
            self.assertEqual(r.status_code, 200)
            job, _ = self._wait(client, jid)
            self.assertEqual((job['status'], job['error_kind']), ('failed', 'cancelled'))
            self.assertEqual(client.post('/api/vision/jobs/nope/cancel').status_code, 404)


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


if __name__ == '__main__':
    unittest.main()


BUILTIN_ENV = {
    'VISION_BUILTIN_MODELS': json.dumps([{
        'id': 'deepseek', 'name': 'DeepSeek（内置）', 'url': 'https://api.deepseek.com/chat/completions',
        'model': 'deepseek-chat', 'key_env': 'DS_KEY'}]),
    'DS_KEY': 'sk-server-secret', 'LAB_ACCESS_CODE': 'open-sesame', 'VISION_BUILTIN_PER_HOUR': '2',
}


class ModelRefAPITests(unittest.TestCase):
    """The model / judge references: built-in and custom endpoint."""

    _wait = VisionAPITests._wait

    def setUp(self):
        VisionAPITests.setUp(self)
        from ragent_lab.vision.builtin import Builtins
        p = patch('ragent_lab.api.vision.builtins', Builtins(BUILTIN_ENV))
        p.start()
        self.addCleanup(p.stop)
        SEEN_KEYS.clear()

    def _understand(self, client, model, **extra):
        return client.post('/api/vision/understand', json={
            'image': {'kind': 'sample', 'id': 'ok'}, 'model': model, **extra})

    def test_builtin_needs_access_code_and_is_rate_limited(self):
        with TestClient(app) as client:
            listed = client.get('/api/vision/builtin').json()
            self.assertEqual(listed['models'], [{'id': 'deepseek', 'name': 'DeepSeek（内置）',
                                                 'notes': ''}])
            self.assertEqual(set(listed), {'enabled', 'per_hour', 'models', 'prompts_editable'})
            self.assertNotIn('sk-server-secret', json.dumps(listed))
            self.assertNotIn('api.deepseek.com', json.dumps(listed))
            ref = {'source': 'builtin', 'id': 'deepseek'}
            r = self._understand(client, ref, access_code='wrong')
            self.assertEqual(r.status_code, 403)
            self.assertEqual(client.post('/api/vision/access',
                                         json={'access_code': 'wrong'}).status_code, 403)
            self.assertEqual(client.post('/api/vision/access',
                                         json={'access_code': 'open-sesame'}).json(), {'ok': True})
            r = self._understand(client, ref, access_code='open-sesame')
            job, polled = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['model'], 'builtin:deepseek')
            self.assertEqual(job['result']['meta']['gateway'], 'builtin')
            self.assertEqual(SEEN_KEYS, ['sk-server-secret'])
            self.assertTrue(all('sk-server-secret' not in t for t in polled))
            # Built-in judge counts against the same hourly limit (2 per IP).
            r = client.post('/api/vision/score', json={
                'sample_id': 'ok', 'understanding': UNDERSTANDING, 'judge': ref,
                'access_code': 'open-sesame'})
            self.assertEqual(self._wait(client, r.json()['job_id'])[0]['status'], 'done')
            r = self._understand(client, ref, access_code='open-sesame')
            self.assertEqual(r.status_code, 429)
            self.assertIn('每小时最多调用 2 次', r.text)
            r = self._understand(client, {'source': 'builtin', 'id': 'nope'},
                                 access_code='open-sesame')
            self.assertIn('该模型暂不可用', r.text)

    def test_custom_endpoint(self):
        ref = {'source': 'custom', 'name': 'DeepSeek 官方', 'api': 'chat',
               'url': 'https://api.deepseek.com/chat/completions', 'model': 'deepseek-chat',
               'json_mode': 'object', 'api_key': 'sk-custom-secret'}
        with TestClient(app) as client:
            with patch('ragent_lab.api.vision.netguard.check_public_https',
                       return_value='api.deepseek.com'):
                r = self._understand(client, ref)
                job, polled = self._wait(client, r.json()['job_id'])
            self.assertEqual(job['result']['model_name'], 'DeepSeek 官方')
            self.assertEqual(job['result']['meta'], {'gateway': 'custom',
                                                     'endpoint': 'api.deepseek.com'})
            self.assertEqual(SEEN_KEYS, ['sk-custom-secret'])
            self.assertTrue(all('sk-custom-secret' not in t for t in polled))
            from ragent_lab.vision.netguard import UnsafeURL
            with patch('ragent_lab.api.vision.netguard.check_public_https',
                       side_effect=UnsafeURL('接口地址不能指向本机或内网')):
                r = self._understand(client, {**ref, 'url': 'https://10.0.0.1/v1'})
            self.assertEqual(r.status_code, 422)
            self.assertIn('内网', r.text)
            r = self._understand(client, {**ref, 'api': 'grpc'})
            self.assertEqual(r.status_code, 422)
            self.assertNotIn('sk-custom-secret', r.text)
