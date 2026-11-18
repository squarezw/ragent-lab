"""PDF -> DXF without a model: copy a vector PDF's drawing exactly (the "no redraw" option).

Mirrors the vector route of the pdf-dwg-to-dxf skill (skills/pdf-dwg-to-dxf), with
pdfminer.six as the reader so the service needs no AGPL library:

1. read the page's paths and text and write them as DXF entities (vector.convert_page);
2. dashed lines with a dash-dot pattern go on CENTER, plain dashes on HIDDEN, and dashed
   lines through a circle's centre or across a shape's middle are promoted to CENTER;
3. the scale comes from the drawing's own dimension labels when at least three of two or more
   values agree, else from a single "1:N" scale note, else the sheet stays at paper size.

It copies what is drawn: a drawing that is not drawn to scale stays so. That is what the
model redraw is for.
"""
from __future__ import annotations

import io
import math
import re

from ezdxf.math import Matrix44

from . import dxfkit, scale as scale_mod
from .vector import NotVector, convert_page, read_page

__all__ = ['convert_pdf', 'NotVector']

_SCALE_NOTE = re.compile(r'\b1\s*[:：]\s*(\d{1,5})\b')
PT_MM = 72 / 25.4


def _apply_scale(doc, factor: float) -> None:
    m = Matrix44.scale(factor, factor, 1)
    for e in doc.modelspace():
        try:
            e.transform(m)
        except Exception:      # an entity type without transform support stays as drawn
            pass
    doc.header['$LTSCALE'] = factor


def convert_pdf(data: bytes, scale: float | None = None) -> tuple[bytes, dict, object]:
    """Convert a one-page vector PDF. scale None = automatic. Returns (dxf bytes, report, doc)."""
    page = read_page(data)
    doc = dxfkit.new_doc()
    rep = convert_page(doc, page, scale=1.0 if scale is None else scale)
    report: dict = {'route': 'vector', 'entities': rep['entities'], 'skipped': rep['skipped'],
                    'profile': rep['profile'], 'warnings': []}
    tol = 0.3 * (1.0 if scale is None else scale)
    report['centre_lines_from_geometry'] = dxfkit.promote_centre_lines(doc, tol)

    frame = rep['frame']
    check = scale_mod.from_labels(doc, frame)
    hints: list[int] = []
    for e in doc.modelspace().query('TEXT'):
        hints += [int(m.group(1)) for m in _SCALE_NOTE.finditer(e.dxf.text)]
    hints = sorted(set(hints), key=hints.index)
    report['label_check'] = {k: v for k, v in check.items() if k not in ('measured',)}
    report['scale_hints'] = hints
    if scale is not None:
        factor, source = scale, 'given'
    elif check['factor']:
        factor, source = check['factor'], 'labels'
        note = float(hints[0]) if len(hints) == 1 else 1.0
        if abs(factor / (note * PT_MM) - 1) < 0.02:
            report['warnings'].append(
                f'这份 PDF 按 1 点 = 1 毫米绘制（标注给出的比例 {factor:.4g} = 72/25.4），'
                '与图上的比例说明不符，已按标注尺寸换算')
        elif hints and abs(factor / note - 1) > 0.03:
            report['warnings'].append(f'图上的比例说明（1:{hints[0]}）与标注尺寸（{factor:.4g}）不一致，已按标注尺寸换算')
    elif len(hints) == 1:
        factor, source = float(hints[0]), f'1:{hints[0]}'
    else:
        factor, source = 1.0, 'paper'
        msg = '没能从标注尺寸确定比例（文字可能是轮廓线，或可读的标注太少），DXF 为图纸原尺寸（毫米）'
        if check.get('suggestion'):
            msg += f'；有 {check["agreeing"]} 个标注提示比例约为 {check["suggestion"]:.4g}'
        report['warnings'].append(msg)
    if scale is None and factor != 1.0:
        _apply_scale(doc, factor)
    report['scale'] = round(factor, 6)
    report['scale_source'] = source
    report['sheet'] = [round(frame[2] * (factor if scale is None else 1.0), 3),
                       round(frame[3] * (factor if scale is None else 1.0), 3)]
    if not rep['profile']['text_chars']:
        report['warnings'].append('PDF 中没有文字对象（文字被转成了轮廓线），文字以线条形式保留')
    report['audit_errors'] = len(doc.audit().errors)
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode('utf-8'), report, doc
