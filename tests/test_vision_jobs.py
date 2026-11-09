"""In-memory job runner tests."""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ragent_lab.vision import jobs
from ragent_lab.vision.jobs import JobRunner, public
from ragent_lab.vision.openrouter import OpenRouterError


async def wait_done(runner, job, limit=5.0):
    end = asyncio.get_running_loop().time() + limit
    while job.status in ('queued', 'running'):
        if asyncio.get_running_loop().time() > end:
            raise AssertionError('job did not finish')
        await asyncio.sleep(0.01)


class JobRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_cap(self):
        runner = JobRunner(max_concurrent=3)
        state = {'now': 0, 'max': 0}

        async def work():
            state['now'] += 1
            state['max'] = max(state['max'], state['now'])
            await asyncio.sleep(0.05)
            state['now'] -= 1
            return {'ok': True}

        submitted = [runner.submit('model', f'm{i}', work) for i in range(5)]
        for j in submitted:
            await wait_done(runner, j)
        self.assertEqual(state['max'], 3)
        self.assertTrue(all(j.status == 'done' for j in submitted))
        self.assertEqual(submitted[0].result, {'ok': True})

    async def test_failure_isolated_and_sanitized(self):
        runner = JobRunner()
        secret = 'sk-secret-key-123'

        async def bad():
            _ = secret  # captured in closure
            raise OpenRouterError('auth', 'Key 无效或无权限。')

        async def good():
            return {'v': 1}

        async def boom():
            raise ValueError(f'leak {secret}')

        a = runner.submit('model', 'a', bad)
        b = runner.submit('model', 'b', good)
        c = runner.submit('model', 'c', boom)
        for j in (a, b, c):
            await wait_done(runner, j)
        self.assertEqual((a.status, a.error_kind, a.error), ('failed', 'auth', 'Key 无效或无权限。'))
        self.assertEqual(b.status, 'done')
        self.assertEqual(c.status, 'failed')
        self.assertEqual(c.error, '内部错误，请稍后重试。')
        for j in (a, b, c):
            self.assertNotIn(secret, json.dumps(public(j), ensure_ascii=False))

    async def test_unexpected_exception_logs_type_only(self):
        runner = JobRunner()

        async def boom():
            raise ValueError('user-data-xyz')

        with self.assertLogs('ragent_lab.vision.jobs', level='ERROR') as cm:
            j = runner.submit('model', 'c', boom)
            await wait_done(runner, j)
        text = '\n'.join(cm.output)
        self.assertIn('ValueError', text)
        self.assertNotIn('user-data-xyz', text)

    async def test_timeout(self):
        runner = JobRunner(timeout=0.05)

        async def slow():
            await asyncio.sleep(5)
            return {}

        j = runner.submit('model', 's', slow)
        await wait_done(runner, j)
        self.assertEqual((j.status, j.error_kind), ('failed', 'timeout'))
        self.assertIn('超时', j.error)

    async def test_timeout_excludes_queue_time(self):
        runner = JobRunner(max_concurrent=1, timeout=0.15)

        async def work():
            await asyncio.sleep(0.1)
            return {}

        a = runner.submit('model', 'a', work)
        b = runner.submit('model', 'b', work)
        await wait_done(runner, a)
        await wait_done(runner, b)
        self.assertEqual((a.status, b.status), ('done', 'done'))

    async def test_expiry(self):
        runner = JobRunner(ttl=100)

        async def work():
            return {}

        with mock.patch.object(jobs.time, 'time', return_value=1000.0):
            j = runner.submit('model', 'a', work)
            await wait_done(runner, j)
        with mock.patch.object(jobs.time, 'time', return_value=1050.0):
            self.assertIs(runner.get(j.id), j)
        with mock.patch.object(jobs.time, 'time', return_value=1000.0 + 101):
            self.assertIsNone(runner.get(j.id))

    async def test_max_jobs_evicts_oldest_finished(self):
        runner = JobRunner(max_jobs=2)

        async def work():
            return {}

        first = runner.submit('model', 'a', work)
        await wait_done(runner, first)
        second = runner.submit('model', 'b', work)
        await wait_done(runner, second)
        third = runner.submit('model', 'c', work)
        self.assertIsNone(runner.get(first.id))
        self.assertIsNotNone(runner.get(third.id))

    async def test_cancel_running_and_queued(self):
        runner = JobRunner(max_concurrent=1)
        closed = []

        async def slow():
            try:
                await asyncio.sleep(30)
            finally:
                closed.append(True)  # e.g. the upstream stream being closed
            return {'ok': True}

        running = runner.submit('model', 'a', slow)
        queued = runner.submit('model', 'b', slow)
        await asyncio.sleep(0.02)
        self.assertEqual((running.status, queued.status), ('running', 'queued'))
        self.assertIs(runner.cancel(queued.id), queued)
        await wait_done(runner, queued)
        self.assertEqual((queued.status, queued.error_kind), ('failed', 'cancelled'))
        self.assertIsNone(queued.started)
        runner.cancel(running.id)
        await wait_done(runner, running)
        self.assertEqual((running.error, running.error_kind), (jobs.CANCELLED_MESSAGE, 'cancelled'))
        self.assertEqual(closed, [True])
        self.assertIsNotNone(running.finished)

    async def test_cancel_finished_or_unknown(self):
        runner = JobRunner()

        async def quick():
            return {'ok': True}

        job = runner.submit('model', 'a', quick)
        await wait_done(runner, job)
        self.assertEqual(runner.cancel(job.id).status, 'done')
        self.assertIsNone(runner.cancel('nope'))


class ConstructionTests(unittest.TestCase):
    def test_construct_without_loop(self):
        JobRunner()
        self.assertIsInstance(jobs.runner, JobRunner)


if __name__ == '__main__':
    unittest.main()
