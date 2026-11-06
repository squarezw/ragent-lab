"""In-memory job runner with a concurrency cap. Nothing is persisted."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from .openrouter import OpenRouterError

log = logging.getLogger(__name__)

JobStatus = Literal['queued', 'running', 'done', 'failed']
TIMEOUT_MESSAGE = '模型响应超时（超过 10 分钟）。'
INTERNAL_MESSAGE = '内部错误，请稍后重试。'


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: JobStatus
    created: float
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    error_kind: str | None = None


def public(job: Job) -> dict:
    return {
        'id': job.id, 'kind': job.kind, 'label': job.label, 'status': job.status,
        'created': job.created, 'started': job.started, 'finished': job.finished,
        'result': job.result, 'error': job.error, 'error_kind': job.error_kind,
    }


class JobRunner:
    def __init__(self, max_concurrent: int = 3, ttl: float = 7200, max_jobs: int = 500,
                 timeout: float = 600):
        self.max_concurrent = max_concurrent
        self.ttl = ttl
        self.max_jobs = max_jobs
        self.timeout = timeout
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        # Created lazily on the running loop, so construction needs no loop.
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._sem_loop = loop
        return self._sem

    def _purge(self) -> None:
        now = time.time()
        for jid in [k for k, j in self._jobs.items()
                    if j.status in ('done', 'failed') and now - (j.finished or j.created) > self.ttl
                    or now - j.created > self.ttl * 2]:
            del self._jobs[jid]

    def _evict_over_cap(self) -> None:
        while len(self._jobs) > self.max_jobs:
            finished = [j for j in self._jobs.values() if j.status in ('done', 'failed')]
            if not finished:
                break
            oldest = min(finished, key=lambda j: j.created)
            del self._jobs[oldest.id]

    def submit(self, kind: str, label: str, factory: Callable[[], Awaitable[dict]]) -> Job:
        self._purge()
        job = Job(id=uuid.uuid4().hex, kind=kind, label=label, status='queued', created=time.time())
        self._jobs[job.id] = job
        self._evict_over_cap()
        task = asyncio.get_running_loop().create_task(self._run(job, factory))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job, factory: Callable[[], Awaitable[dict]]) -> None:
        async with self._semaphore():
            job.status = 'running'
            job.started = time.time()
            try:
                job.result = await asyncio.wait_for(factory(), timeout=self.timeout)
                job.status = 'done'
            except asyncio.TimeoutError:
                job.status, job.error, job.error_kind = 'failed', TIMEOUT_MESSAGE, 'timeout'
            except OpenRouterError as err:
                job.status, job.error, job.error_kind = 'failed', err.message, err.kind
            except Exception as err:  # noqa: BLE001 - isolate jobs from each other
                log.error('vision job failed: %s', type(err).__name__)
                job.status, job.error = 'failed', INTERNAL_MESSAGE
            finally:
                job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        self._purge()
        return self._jobs.get(job_id)


runner = JobRunner()
