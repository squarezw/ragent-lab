"""Vision HTTP API: jobs run in memory; the API key lives only inside job closures."""
from __future__ import annotations

import asyncio
import re
from typing import Annotated, Literal, Union
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..vision import netguard, prompts, render, service
from ..vision.builtin import builtins
from ..vision.images import (MAX_UPLOAD_BYTES, ImageRejected, PreparedImage, get_sample,
                             images, load_samples, prepare_image)
from ..vision.calls import ModelError
from ..vision.jobs import public, runner
from ..vision.registry import JUDGE_MODEL, VISION_MODELS, get_model, judge_model
from ..vision.schema import GeneralUnderstanding, Geometry, Understanding

router = APIRouter(prefix='/api/vision')

IMAGE_GONE = '图片已过期，请重新上传。'
JOB_GONE = '任务已失效，请重新运行。'


class _Body(BaseModel):
    model_config = ConfigDict(extra='forbid')


class CatalogRef(_Body):
    """A catalog model, run with the server's key (needs the access code)."""
    source: Literal['catalog']
    id: str


class BuiltinRef(_Body):
    """A model configured on the server (env); its key never leaves the server."""
    source: Literal['builtin']
    id: str


class CustomRef(_Body):
    """A visitor-defined endpoint: any public https URL speaking one of the wire formats."""
    source: Literal['custom']
    name: str = Field(min_length=1, max_length=60)
    api: Literal['chat', 'responses', 'anthropic'] = 'chat'
    url: str = Field(min_length=9, max_length=500)
    model: str = Field(default='', max_length=200)
    json_mode: Literal['schema', 'object', 'prompt'] = 'prompt'
    auth: Literal['bearer', 'x-api-key'] = 'bearer'
    max_tokens: int = Field(default=16000, ge=1000, le=128000)
    api_key: SecretStr = SecretStr('')


ModelRef = Annotated[Union[CatalogRef, BuiltinRef, CustomRef], Field(discriminator='source')]


class _Keyed(_Body):
    # One code unlocks every server model (catalog and built-in); custom models bring a key.
    access_code: SecretStr = SecretStr('')


class ImageRef(_Body):
    kind: Literal['sample', 'upload']
    id: str


PurposeId = Literal[tuple(prompts.PURPOSES)]
# Either structure; they share no required fields, so validation picks the right one.
AnyUnderstanding = Union[Understanding, GeneralUnderstanding]


class UnderstandBody(_Keyed):
    image: ImageRef
    model: str | ModelRef
    purpose: PurposeId = prompts.DEFAULT_PURPOSE
    # Replaces the purpose's default prompt. Only accepted together with the prompt password.
    prompt: str | None = Field(default=None, max_length=prompts.MAX_PROMPT_CHARS)
    prompt_code: SecretStr = SecretStr('')


class GenerateBody(_Keyed):
    image: ImageRef
    model: str | ModelRef
    understanding: Understanding


class ScoreBody(_Keyed):
    sample_id: str
    understanding: AnyUnderstanding
    judge_model: str | None = None
    judge: ModelRef | None = None


class ConsensusItem(_Body):
    model_name: str
    understanding: AnyUnderstanding


class ConsensusBody(_Keyed):
    results: list[ConsensusItem] = Field(min_length=2, max_length=26)
    judge_model: str | None = None
    judge: ModelRef | None = None


class ExportBody(_Body):
    geometry: Geometry
    format: Literal['pdf', 'dxf']
    filename: str = Field(default='drawing', max_length=200)


def _server_access(body: _Keyed, request: Request) -> None:
    """Every server-paid model needs the access code and counts toward the hourly limit."""
    code = body.access_code.get_secret_value().strip()
    if not code:
        raise HTTPException(422, '请先在「模型设置」填写访问码')
    ip = _client_ip(request)
    if not builtins.check_code(code, ip):
        raise HTTPException(403, '访问码不正确')
    if not builtins.take(ip):
        raise HTTPException(429, f'每小时最多调用 {builtins.per_hour} 次，请稍后再试')


def _resolve_image(ref: ImageRef) -> PreparedImage:
    if ref.kind == 'sample':
        sample = get_sample(ref.id)
        found = sample.image if sample else None
    else:
        found = images.get(ref.id)
    if found is None:
        raise HTTPException(404, IMAGE_GONE)
    return found


def _catalog(model_id: str | None, judge: bool = False) -> dict:
    try:
        return judge_model(model_id) if judge else get_model(model_id or '')
    except KeyError:
        raise HTTPException(422, '未知评判模型' if judge else '未知模型') from None


def _client_ip(request: Request) -> str:
    # The API is only reachable through the web container's Nginx, which sets X-Real-IP.
    return request.headers.get('x-real-ip') or (request.client.host if request.client else '-')


def _ref_key(ref, gateway: str) -> str:  # custom endpoints: always the visitor's key
    key = ref.api_key.get_secret_value().strip()
    if not key:
        raise HTTPException(422, f'请填写 {gateway} Key')
    return key


async def _resolve(ref, body: _Keyed, request: Request, judge: bool = False):
    """Turn a model reference into (model dict, gateway, api key), checking access."""
    if isinstance(ref, str) or ref is None or isinstance(ref, CatalogRef):
        model = _catalog(ref.id if isinstance(ref, CatalogRef) else ref, judge)
        if not builtins.catalog_enabled:
            raise HTTPException(422, '服务器暂未开放这个模型')
        _server_access(body, request)
        return model, 'catalog', builtins.catalog_key
    if isinstance(ref, BuiltinRef):
        model = builtins.models.get(ref.id) if builtins.enabled else None
        if model is None:
            raise HTTPException(422, '该模型暂不可用')
        _server_access(body, request)
        return model, 'builtin', model['api_key']
    # CustomRef
    try:
        host = await run_in_threadpool(netguard.check_public_https, ref.url)
    except netguard.UnsafeURL as exc:
        raise HTTPException(422, str(exc)) from None
    key = _ref_key(ref, ref.name)
    model = {
        'id': f'custom:{ref.name}', 'name': ref.name, 'max_tokens': ref.max_tokens,
        'temperature': None, 'reasoning': None,
        'endpoint': {'api': ref.api, 'url': ref.url.strip(), 'model': ref.model.strip() or None,
                     'json': ref.json_mode, 'auth': ref.auth, 'label': host,
                     'public_only': True},
    }
    return model, 'custom', key


def _png(image: PreparedImage | None) -> Response:
    if image is None:
        raise HTTPException(404, IMAGE_GONE)
    return Response(image.png, media_type='image/png', headers={'Cache-Control': 'private, max-age=3600'})


def _job(kind: str, label: str, factory) -> dict:
    return {'job_id': runner.submit(kind, label, factory).id}


@router.get('/models')
def models():
    """Catalog models (names only). Empty when the server has no catalog key or code."""
    if not builtins.catalog_enabled:
        return []
    return [{'id': m['id'], 'name': m['name'], 'family': m['family'], 'notes': m['notes'],
             'params': {'max_tokens': m['max_tokens'], 'temperature': m['temperature'],
                        'reasoning': m['reasoning']}} for m in VISION_MODELS]


def _purposes(with_prompts: bool) -> list[dict]:
    return [{'id': pid, 'name': p['name'], 'kind': p['kind'], 'description': p['description'],
             **({'prompt': prompts.purpose_prompt(pid)} if with_prompts else {})}
            for pid, p in prompts.PURPOSES.items()]


def _require_prompt_code(code: SecretStr, request: Request) -> None:
    """Prompts are shown and replaced only for whoever knows LAB_PROMPT_CODE."""
    if not builtins.prompts_editable:
        raise HTTPException(403, '服务器未开放提示词管理')
    if not builtins.check_prompt_code(code.get_secret_value().strip(), _client_ip(request)):
        raise HTTPException(403, '提示词管理密码不正确')


@router.get('/purposes')
def purposes():
    """Built-in understanding purposes. The prompt text is not public: see /purposes/unlock."""
    return _purposes(with_prompts=False)


class PromptCodeBody(_Body):
    prompt_code: SecretStr


@router.post('/purposes/unlock')
def unlock_purposes(body: PromptCodeBody, request: Request):
    """The purposes with their default prompts, for whoever has the prompt password."""
    _require_prompt_code(body.prompt_code, request)
    return _purposes(with_prompts=True)


@router.get('/judge')
def judge():
    return {'id': JUDGE_MODEL['id'], 'name': JUDGE_MODEL['name']}


@router.get('/builtin')
def builtin_models():
    """Built-in models (names only) and whether they are offered at all."""
    return {'enabled': builtins.enabled, 'per_hour': builtins.per_hour,
            'models': builtins.public_list(), 'prompts_editable': builtins.prompts_editable}


class AccessBody(_Body):
    access_code: SecretStr


@router.post('/access')
def check_access(body: AccessBody, request: Request):
    if not builtins.enabled:
        raise HTTPException(422, '服务器未开放内置模型')
    if not builtins.check_code(body.access_code.get_secret_value().strip(), _client_ip(request)):
        raise HTTPException(403, '访问码不正确')
    return {'ok': True}


@router.get('/samples')
def samples():
    return [{'id': s.id, 'name': s.name,
             'scorable': bool(s.rubric and s.rubric.rubric_status == 'approved'),
             'kind': s.rubric.kind if s.rubric else 'engineering'}
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
async def understand(body: UnderstandBody, request: Request):
    image = _resolve_image(body.image)
    purpose, instructions = body.purpose, body.prompt
    if instructions is not None:  # checked before the model, so a refused prompt costs no call
        _require_prompt_code(body.prompt_code, request)
    model, gateway, key = await _resolve(body.model, body, request)

    async def factory():
        return await service.understand(key, model, image, gateway, purpose, instructions)
    return _job('understand', model['name'], factory)


@router.post('/score')
async def score(body: ScoreBody, request: Request):
    sample = get_sample(body.sample_id)
    if sample is None or sample.rubric is None or sample.rubric.rubric_status != 'approved' \
            or not sample.rubric.items:
        raise HTTPException(422, '该样例没有可用的评分细则')
    if service.kind_of(body.understanding.model_dump()) != sample.rubric.kind:
        raise HTTPException(422, '这份理解的用途与该样例的评分细则不一致，无法打分')
    judge, gateway, key = await _resolve(body.judge or body.judge_model, body, request, judge=True)
    rubric, understanding = sample.rubric, body.understanding.model_dump()

    async def factory():
        return await service.score(key, rubric, understanding, judge, gateway)
    return _job('score', sample.name, factory)


@router.post('/consensus')
async def consensus(body: ConsensusBody, request: Request):
    if len({type(r.understanding) for r in body.results}) != 1:
        raise HTTPException(422, '只能对照同一用途的理解结果')
    judge, gateway, key = await _resolve(body.judge or body.judge_model, body, request, judge=True)
    results = [{'model_name': r.model_name, 'understanding': r.understanding.model_dump()}
               for r in body.results]

    async def factory():
        return await service.consensus(key, results, judge, gateway)
    return _job('consensus', f'{len(results)} 个模型', factory)


@router.post('/generate')
async def generate(body: GenerateBody, request: Request):
    image = _resolve_image(body.image)
    understanding = body.understanding
    model, gateway, key = await _resolve(body.model, body, request)

    async def factory():
        out = await service.generate(key, model, image, understanding.model_dump(), gateway)
        geometry = Geometry.model_validate(out['geometry'])
        try:
            rendered = await bounded_render('generate', render.render_all, geometry, understanding)
        except RenderBusy:
            raise ModelError('timeout', RENDER_BUSY) from None
        return {'geometry': out['geometry'], **rendered, 'meta': out['meta']}
    return _job('generate', model['name'], factory)


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


@router.post('/jobs/{job_id}/cancel')
async def cancel_job(job_id: str):
    found = runner.cancel(job_id)
    if found is None:
        raise HTTPException(404, JOB_GONE)
    await asyncio.sleep(0)  # let the cancelled task record its state
    return public(found)


@router.get('/jobs/{job_id}')
def job(job_id: str):
    found = runner.get(job_id)
    if found is None:
        raise HTTPException(404, JOB_GONE)
    return public(found)
