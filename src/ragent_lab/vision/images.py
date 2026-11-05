"""Image preparation, in-memory image store and built-in sample loader."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from .schema import Rubric

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_EDGE = 2048
PDF_DPI = 200
STORE_TTL_SECONDS = 7200
STORE_MAX_ITEMS = 100


class ImageRejected(ValueError):
    """Upload cannot be used; the message is user-facing (Chinese)."""


@dataclass
class PreparedImage:
    id: str
    png: bytes
    width: int
    height: int
    sha256: str
    source_name: str


def _render_pdf(data: bytes) -> Image.Image:
    try:
        doc = pdfium.PdfDocument(data)
        try:
            if len(doc) != 1:
                raise ImageRejected('仅支持单页 PDF')
            page = doc[0]
            return page.render(scale=PDF_DPI / 72).to_pil()
        finally:
            doc.close()
    except ImageRejected:
        raise
    except Exception as exc:
        raise ImageRejected('无法解析 PDF 文件') from exc


def _decode_raster(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ('PNG', 'JPEG'):
            raise ImageRejected('仅支持 PNG、JPG 或单页 PDF')
        img.load()
        return img
    except ImageRejected:
        raise
    except Exception as exc:
        raise ImageRejected('无法解码图片，请上传有效的 PNG、JPG 或单页 PDF') from exc


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel('A'))
        return bg
    return img.convert('RGB')


def prepare_image(data: bytes, filename: str) -> PreparedImage:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected('文件超过 15 MB 上限')
    if data.startswith(b'%PDF'):
        img = _render_pdf(data)
    else:
        img = _decode_raster(data)
    img = _to_rgb(img)
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ImageRejected('图片尺寸无效')
    longest = max(w, h)
    if longest > MAX_EDGE:
        ratio = MAX_EDGE / longest
        img = img.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    png = buf.getvalue()
    return PreparedImage(
        id=uuid.uuid4().hex, png=png, width=img.width, height=img.height,
        sha256=hashlib.sha256(png).hexdigest(), source_name=filename,
    )


class ImageStore:
    def __init__(self, ttl: float = STORE_TTL_SECONDS, capacity: int = STORE_MAX_ITEMS):
        self._ttl = ttl
        self._capacity = capacity
        self._items: OrderedDict[str, tuple[float, PreparedImage]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        for key in [k for k, (t, _) in self._items.items() if now - t > self._ttl]:
            del self._items[key]

    def put(self, img: PreparedImage) -> str:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            self._items[img.id] = (now, img)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return img.id

    def get(self, id: str) -> PreparedImage | None:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            entry = self._items.get(id)
            return entry[1] if entry else None


images = ImageStore()


@dataclass
class Sample:
    id: str
    name: str
    image: PreparedImage
    rubric: Rubric | None


def _samples_root(root: Path | None = None) -> Path:
    return Path(root if root is not None else os.environ.get('VISION_SAMPLES_DIR', '/samples'))


def _load_one(folder: Path) -> Sample:
    rubric = Rubric.model_validate_json((folder / 'rubric.json').read_text('utf-8'))
    img_path = (folder / rubric.image).resolve()
    if folder.resolve() not in img_path.parents:
        raise ValueError(f'image path escapes sample folder: {rubric.image}')
    prepared = prepare_image(img_path.read_bytes(), img_path.name)
    return Sample(id=rubric.id, name=rubric.name, image=prepared, rubric=rubric)


def load_samples(root: Path | None = None) -> list[Sample]:
    base = _samples_root(root)
    if not base.is_dir():
        return []
    out: list[Sample] = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        if not (folder / 'rubric.json').is_file():
            continue
        try:
            out.append(_load_one(folder))
        except Exception as exc:
            log.warning('skipping broken sample %s: %s', folder.name, exc)
    return out


_samples_cache: dict[Path, dict[str, Sample]] = {}
_samples_lock = threading.Lock()


def reset_samples_cache() -> None:
    with _samples_lock:
        _samples_cache.clear()


def get_sample(id: str) -> Sample | None:
    base = _samples_root().resolve()
    with _samples_lock:
        cached = _samples_cache.get(base)
        if cached is None:
            cached = {s.id: s for s in load_samples(base)}
            _samples_cache[base] = cached
    return cached.get(id)
