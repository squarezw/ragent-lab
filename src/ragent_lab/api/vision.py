"""Vision HTTP API: jobs run in memory; the API key lives only inside job closures."""
from __future__ import annotations

import asyncio
import re
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..vision import render, service
from ..vision.images import (MAX_UPLOAD_BYTES, ImageRejected, PreparedImage, get_sample,
                             images, load_samples, prepare_image)
from ..vision.openrouter import OpenRouterError
from ..vision.jobs import public, runner
from ..vision.registry import (JUDGE_MODEL, VISION_MODELS, gateways_of, get_model, judge_model,
                               model_prices)
from ..vision.schema import Geometry, Understanding

router = APIRouter(prefix='/api/vision')

IMAGE_GONE = '图片已过期，请重新上传。'
JOB_GONE = '任务已失效，请重新运行。'


class _Body(BaseModel):
    model_config = ConfigDict(extra='forbid')


class _Keyed(_Body):
    api_key: SecretStr = SecretStr('')
    gateway: Literal['openrouter', 'kie'] = 'openrouter'


class ImageRef(_Body):
    kind: Literal['sample', 'upload']
    id: str


class UnderstandBody(_Keyed):
    image: ImageRef
    model: str


class GenerateBody(UnderstandBody):
    understanding: Understanding


class ScoreBody(_Keyed):
    sample_id: str
    understanding: Understanding
    judge_model: str | None = None


class ConsensusItem(_Body):
    model_name: str
    understanding: Understanding


class ConsensusBody(_Keyed):
    results: list[ConsensusItem] = Field(min_length=2, max_length=9)
    judge_model: str | None = None


class ExportBody(_Body):
    geometry: Geometry
    format: Literal['pdf', 'dxf']
    filename: str = Field(default='drawing', max_length=200)


GATEWAY_NAMES = {'openrouter': 'OpenRouter', 'kie': 'Kie.ai'}


def _key(body: _Keyed) -> str:
    key = body.api_key.get_secret_value().strip()
    if not key:
        raise HTTPException(422, f'请填写 {GATEWAY_NAMES[body.gateway]} Key')
    return key


def _resolve_image(ref: ImageRef) -> PreparedImage:
    if ref.kind == 'sample':
        sample = get_sample(ref.id)
        found = sample.image if sample else None
    else:
        found = images.get(ref.id)
    if found is None:
        raise HTTPException(404, IMAGE_GONE)
    return found


def _check_model(model_id: str, gateway: str = 'openrouter') -> None:
    try:
        model = get_model(model_id)
    except KeyError:
        raise HTTPException(422, '未知模型') from None
    if gateway not in gateways_of(model):
        raise HTTPException(422, f'{model["name"]} 不支持 {GATEWAY_NAMES[gateway]}')


def _judge(model_id: str | None, gateway: str = 'openrouter') -> dict:
    try:
        judge = judge_model(model_id)
    except KeyError:
        raise HTTPException(422, '未知评判模型') from None
    if gateway not in gateways_of(judge):
        raise HTTPException(422, f'评判模型 {judge["name"]} 不支持 {GATEWAY_NAMES[gateway]}')
    return judge


def _png(image: PreparedImage | None) -> Response:
    if image is None:
        raise HTTPException(404, IMAGE_GONE)
    return Response(image.png, media_type='image/png', headers={'Cache-Control': 'private, max-age=3600'})


def _job(kind: str, label: str, factory) -> dict:
    return {'job_id': runner.submit(kind, label, factory).id}


@router.get('/models')
async def models():
    prices = await model_prices()
    return [{'id': m['id'], 'name': m['name'], 'family': m['family'], 'notes': m['notes'],
             'params': {'max_tokens': m['max_tokens'], 'temperature': m['temperature'],
                        'reasoning': m['reasoning']},
             'price': prices.get(m['id']), 'gateways': gateways_of(m)} for m in VISION_MODELS]


@router.get('/judge')
def judge():
    return {'id': JUDGE_MODEL['id'], 'name': JUDGE_MODEL['name']}


@router.get('/samples')
def samples():
    return [{'id': s.id, 'name': s.name,
             'scorable': bool(s.rubric and s.rubric.rubric_status == 'approved')}
            for s in load_samples()]


@router.get('/samples/{sample_id}/image')
def sample_image(sample_id: str):
    sample = get_sample(sample_id)
    return _png(sample.image if sample else None)


@router.get('/images/{image_id}')
def uploaded_image(image_id: str):
    return _png(images.get(image_id))


@router.post('/images')
async def upload_image(file: UploadFile = File(...)):
    # Nginx caps the body at 16m, so reading it whole is bounded.
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, '文件超过 15 MB 上限')
    name = file.filename or 'upload'
    try:
        prepared = await run_in_threadpool(prepare_image, data, name)
    except ImageRejected as exc:
        raise HTTPException(422, str(exc)) from None
    images.put(prepared)
    return {'image_id': prepared.id, 'width': prepared.width, 'height': prepared.height,
            'sha256': prepared.sha256, 'name': name}


@router.post('/understand')
async def understand(body: UnderstandBody):
    key, gateway = _key(body), body.gateway
    _check_model(body.model, body.gateway)
    image = _resolve_image(body.image)
    model_id = body.model

    async def factory():
        return await service.understand(key, model_id, image, gateway)
    return _job('understand', get_model(model_id)['name'], factory)


@router.post('/score')
async def score(body: ScoreBody):
    key, gateway = _key(body), body.gateway
    sample = get_sample(body.sample_id)
    if sample is None or sample.rubric is None or sample.rubric.rubric_status != 'approved' \
            or not sample.rubric.items:
        raise HTTPException(422, '该样例没有可用的评分细则')
    judge = _judge(body.judge_model, body.gateway)
    rubric, understanding = sample.rubric, body.understanding.model_dump()

    async def factory():
        return await service.score(key, rubric, understanding, judge, gateway)
    return _job('score', sample.name, factory)


@router.post('/consensus')
async def consensus(body: ConsensusBody):
    key, gateway = _key(body), body.gateway
    judge = _judge(body.judge_model, body.gateway)
    results = [{'model_name': r.model_name, 'understanding': r.understanding.model_dump()}
               for r in body.results]

    async def factory():
        return await service.consensus(key, results, judge, gateway)
    return _job('consensus', f'{len(results)} 个模型', factory)


@router.post('/generate')
async def generate(body: GenerateBody):
    key, gateway = _key(body), body.gateway
    _check_model(body.model, body.gateway)
    image = _resolve_image(body.image)
    model_id, understanding = body.model, body.understanding

    async def factory():
        out = await service.generate(key, model_id, image, understanding.model_dump(), gateway)
        geometry = Geometry.model_validate(out['geometry'])
        try:
            rendered = await bounded_render('generate', render.render_all, geometry, understanding)
        except RenderBusy:
            raise OpenRouterError('timeout', RENDER_BUSY) from None
        return {'geometry': out['geometry'], **rendered, 'meta': out['meta']}
    return _job('generate', get_model(model_id)['name'], factory)


RENDER_BUSY = '导出繁忙或超时，请稍后重试。'
RENDER_BUDGET = 60.0
RENDER_CONCURRENCY = 2
_render_sems: dict = {}


class RenderBusy(Exception):
    pass


async def bounded_render(pool: str, fn, *args):
    """Run a CPU-bound render in a thread, at most 2 at once per pool ('export' / 'generate'), within a wall-clock budget.

    The slot is held until the thread really finishes, even after a timeout."""
    loop = asyncio.get_running_loop()
    sem = _render_sems.setdefault((loop, pool), asyncio.Semaphore(RENDER_CONCURRENCY))
    try:
        await asyncio.wait_for(sem.acquire(), RENDER_BUDGET)
    except asyncio.TimeoutError:
        raise RenderBusy from None
    task = asyncio.ensure_future(run_in_threadpool(fn, *args))
    task.add_done_callback(lambda t: (sem.release(), t.cancelled() or t.exception()))
    try:
        return await asyncio.wait_for(asyncio.shield(task), RENDER_BUDGET)
    except asyncio.TimeoutError:
        raise RenderBusy from None


def _render_export(geometry: Geometry, fmt: str) -> bytes:
    doc = render.build_doc(geometry)
    return render.to_pdf(doc, geometry.sheet) if fmt == 'pdf' else render.to_dxf(doc)


@router.post('/export')
async def export(body: ExportBody):
    try:
        data = await bounded_render('export', _render_export, body.geometry, body.format)
    except RenderBusy:
        raise HTTPException(503, RENDER_BUSY) from None
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '_', body.filename.strip()).strip('. ')
    stem = re.sub(r'\.(pdf|dxf)$', '', stem, flags=re.I) or 'drawing'
    name = f'{stem}.{body.format}'
    ascii_name = re.sub(r'[^A-Za-z0-9._-]+', '_', name) or f'drawing.{body.format}'
    media = 'application/pdf' if body.format == 'pdf' else 'application/dxf'
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"
    return Response(data, media_type=media, headers={'Content-Disposition': disposition})


@router.get('/jobs/{job_id}')
def job(job_id: str):
    found = runner.get(job_id)
    if found is None:
        raise HTTPException(404, JOB_GONE)
    return public(found)
