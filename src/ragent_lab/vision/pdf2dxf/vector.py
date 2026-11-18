"""Vector PDF page -> DXF entities, read straight from the page's drawing commands (pdfminer.six).

The same conversion as the pdf-dwg-to-dxf skill's vector route, on an MIT-licensed reader:
every stroke becomes a LINE / LWPOLYLINE / ARC / CIRCLE / SPLINE with its colour, lineweight
and dash pattern, filled shapes become HATCHes, and text becomes TEXT pinned to the extent it
had on the page. Coordinates are millimetres on paper times `scale`.
"""
from __future__ import annotations

import io
import math

import numpy as np
from pdfminer.layout import (LAParams, LTAnno, LTChar, LTCurve, LTFigure, LTImage, LTPage,
                             LTTextContainer, LTTextLine)
from pdfminer.high_level import extract_pages

from . import dxfkit

PT_MM = 25.4 / 72.0


class NotVector(ValueError):
    """The page is a scanned picture, not drawing commands."""


def _rgb(colour) -> tuple[float, float, float] | None:
    """pdfminer colours: gray number, RGB or CMYK tuples (pattern names are ignored)."""
    if colour is None:
        return None
    if isinstance(colour, (int, float)):
        g = float(colour)
        return (g, g, g)
    if isinstance(colour, (list, tuple)):
        vals = [v for v in colour if isinstance(v, (int, float))]
        if len(vals) == 1:
            return (vals[0],) * 3
        if len(vals) == 3:
            return tuple(float(v) for v in vals)
        if len(vals) == 4:
            c, m, y, k = (float(v) for v in vals)
            return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


class _Map:
    """PDF user space (points, y up) -> drawing space (mm x scale), honouring /Rotate."""

    def __init__(self, bbox, rotate: int, scale: float):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        self.rot = rotate % 360
        self.k = PT_MM * scale

    def pt(self, p) -> tuple[float, float]:
        x, y = p[0] - self.x0, p[1] - self.y0
        if self.rot == 90:
            x, y = y, self.w - x
        elif self.rot == 180:
            x, y = self.w - x, self.h - y
        elif self.rot == 270:
            x, y = self.h - y, x
        return (x * self.k, y * self.k)

    @property
    def size(self) -> tuple[float, float]:
        w, h = (self.h, self.w) if self.rot in (90, 270) else (self.w, self.h)
        return (w * self.k, h * self.k)


def _walk(objs):
    for o in objs:
        if isinstance(o, LTFigure):
            yield from _walk(o)
        else:
            yield o


def _segments(path, mp: _Map):
    """pdfminer original_path -> connected runs of ('l', a, b) / ('c', p0, p1, p2, p3), plus dots."""
    runs, cur, dots = [], [], []
    start = last = None

    def push(seg):
        nonlocal cur
        if cur and math.dist(cur[-1][-1], seg[1]) > 1e-9:
            runs.append(cur)
            cur = []
        cur.append(seg)
    for op in path:
        kind, pts = op[0], [mp.pt(p) for p in op[1:]]
        if kind == 'm':
            if cur:
                runs.append(cur)
                cur = []
            start = last = pts[0]
        elif kind == 'l' and last is not None:
            if math.dist(last, pts[0]) > 1e-9:
                push(('l', last, pts[0]))
            else:
                dots.append(last)
            last = pts[0]
        elif kind in ('c', 'v', 'y') and last is not None:
            if kind == 'c':
                c1, c2, end = pts
            elif kind == 'v':                    # first control point = current point
                c1, c2, end = last, pts[0], pts[1]
            else:                                # 'y': second control point = end point
                c1, c2, end = pts[0], pts[1], pts[1]
            push(('c', last, c1, c2, end))
            last = end
        elif kind == 'h' and last is not None and start is not None:
            if math.dist(last, start) > 1e-9:
                push(('l', last, start))
            last = start
    if cur:
        runs.append(cur)
    return runs, dots


def _emit_run(msp, run, attribs, tol, counts):
    lines: list = []
    arc_pts: list = []
    arc_fit = None

    def flush_lines(close=False):
        nonlocal lines
        if len(lines) == 2 and not close:
            msp.add_line(lines[0], lines[1], dxfattribs=attribs)
            counts['LINE'] += 1
        elif len(lines) > 2 or (close and len(lines) > 1):
            msp.add_lwpolyline(lines, close=close, dxfattribs=attribs)
            counts['LWPOLYLINE'] += 1
        lines = []

    def flush_arc():
        nonlocal arc_pts, arc_fit
        if arc_fit is not None:
            counts[dxfkit.add_arc_or_circle(msp, *arc_fit, attribs)] += 1
        arc_pts, arc_fit = [], None

    for seg in run:
        if seg[0] == 'l':
            flush_arc()
            if not lines:
                lines.append(seg[1])
            lines.append(seg[2])
            continue
        pts = dxfkit.bezier_points(*seg[1:5])
        trial = np.vstack(arc_pts + [pts]) if arc_pts else pts
        fit = dxfkit.arc_from_points(trial, tol)
        if fit is not None:
            flush_lines()
            arc_pts.append(pts if not arc_pts else pts[1:])
            arc_fit = fit
            continue
        flush_arc()
        fit = dxfkit.arc_from_points(pts, tol)
        if fit is not None:
            flush_lines()
            arc_pts, arc_fit = [pts], fit
            continue
        flush_lines()
        msp.add_open_spline([(*p, 0) for p in seg[1:5]], degree=3, dxfattribs=attribs)
        counts['SPLINE'] += 1
    flush_arc()
    closed = len(lines) > 2 and math.dist(lines[0], lines[-1]) < 1e-6
    if closed:
        lines = lines[:-1]
    flush_lines(close=closed)


def _rebuild_split_dashes(doc, msp, dots, scale, counts):
    """Plotters often draw each dash as its own stroke: rebuild rows of them as one line."""
    groups: dict[tuple, list] = {}
    for e in msp.query('LINE'):
        if e.dxf.get('linetype', 'BYLAYER') not in ('BYLAYER', 'Continuous'):
            continue
        key = (e.dxf.layer, e.dxf.get('true_color'), e.dxf.get('color'), e.dxf.get('lineweight'))
        groups.setdefault(key, []).append(e)
    for (_layer, _tc, _c, lw), ents in groups.items():
        if len(ents) < 3:
            continue
        segs = [((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)) for e in ents]
        _, dashed = dxfkit.find_dashed(segs, dots, tol=0.05 * scale, max_gap=4.0 * scale,
                                       min_dash=0.5 * scale)
        for d in dashed:
            name, lcolor = dxfkit.LINE_LAYERS[d['kind']]
            attribs = {'layer': dxfkit.ensure_layer(doc, name, color=lcolor),
                       'linetype': dxfkit.ensure_pattern_linetype(doc, d['kind'], d['dash'], d['gap'])}
            if lw is not None:
                attribs['lineweight'] = lw
            msp.add_line(d['start'], d['end'], dxfattribs=attribs)
            for i in d['pieces']:
                msp.delete_entity(ents[i])
            counts['LINE'] += 1 - len(d['pieces'])
            counts[d['kind'].upper() + '_LINES'] = counts.get(d['kind'].upper() + '_LINES', 0) + 1


def _text_runs(line: LTTextLine):
    """Runs of characters in a text line, split at spaces (each number becomes its own TEXT)."""
    run: list[LTChar] = []
    for ch in line:
        if isinstance(ch, LTChar) and ch.get_text().strip():
            run.append(ch)
        elif run:
            yield run
            run = []
    if run:
        yield run


def page_profile(page: LTPage) -> dict:
    """How much of the page is drawing commands, text and pictures."""
    curves = chars = 0
    img_area = 0.0
    area = max((page.x1 - page.x0) * (page.y1 - page.y0), 1.0)
    for o in _walk(page):
        if isinstance(o, LTCurve):
            curves += 1
        elif isinstance(o, LTImage):
            img_area += max(o.x1 - o.x0, 0) * max(o.y1 - o.y0, 0)
        elif isinstance(o, LTTextContainer):
            chars += len(o.get_text().strip())
    return {'paths': curves, 'text_chars': chars, 'image_coverage': round(min(img_area / area, 1.0), 3)}


def read_page(data: bytes) -> LTPage:
    pages = list(extract_pages(io.BytesIO(data), laparams=LAParams(char_margin=1.0, word_margin=0.15),
                               maxpages=2))
    if not pages:
        raise ValueError('PDF 中没有页面')
    if len(pages) > 1:
        raise ValueError('仅支持单页 PDF')
    return pages[0]


def convert_page(doc, page: LTPage, scale: float = 1.0, arc_tol_mm: float = 0.02) -> dict:
    prof = page_profile(page)
    if prof['paths'] == 0 or (prof['image_coverage'] > 0.5 and prof['paths'] < 50):
        raise NotVector('这页 PDF 是扫描图片，没有可直接读取的线条')
    msp = doc.modelspace()
    mp = _Map(page.bbox, getattr(page, 'rotate', 0), scale)
    tol = arc_tol_mm * scale
    counts = {k: 0 for k in ('LINE', 'LWPOLYLINE', 'ARC', 'CIRCLE', 'SPLINE', 'HATCH', 'TEXT')}
    skipped = {'white_fills': 0, 'invisible': 0}
    all_dots = []
    for o in _walk(page):
        if not isinstance(o, LTCurve):
            continue
        path = getattr(o, 'original_path', None) or [('m', o.pts[0])] + [('l', p) for p in o.pts[1:]]
        runs, dots = _segments(path, mp)
        stroke = _rgb(o.stroking_color) if o.stroke else None
        fill = _rgb(o.non_stroking_color) if o.fill else None
        if stroke is None and fill is None:
            skipped['invisible'] += 1
            continue
        style = getattr(o, 'dashing_style', None)
        dash_mm = [float(v) * mp.k for v in (style[0] if style else [])]
        linetype = dxfkit.ensure_linetype(doc, dash_mm) if stroke is not None else 'Continuous'
        kind = dxfkit.classify_dash_array(dash_mm) if linetype != 'Continuous' else None
        if kind:
            name, color = dxfkit.LINE_LAYERS[kind]
            layer = dxfkit.ensure_layer(doc, name, color=color)
        else:
            rgb = stroke or fill or (0, 0, 0)
            hexc = ''.join(f'{max(0, min(255, round(c * 255))):02X}' for c in rgb[:3])
            layer = dxfkit.ensure_layer(doc, f'PDF_{hexc}')
        if stroke is not None:
            all_dots += dots
        if fill is not None:
            if min(fill) >= 0.98:
                skipped['white_fills'] += 1
            else:
                outlines = []
                for run in runs:
                    outline = []
                    for seg in run:
                        outline += [seg[1], seg[2]] if seg[0] == 'l' else \
                            [tuple(p) for p in dxfkit.bezier_points(*seg[1:5])]
                    if len(outline) >= 3:
                        outlines.append(outline)
                if outlines:
                    hatch = msp.add_hatch(dxfattribs={'layer': layer, 'hatch_style': 0,
                                                      **dxfkit.rgb_attribs(fill)})
                    for outline in outlines:
                        hatch.paths.add_polyline_path(outline, is_closed=True)
                    counts['HATCH'] += 1
        if stroke is not None:
            attribs = {'layer': layer, 'lineweight': dxfkit.lineweight(o.linewidth * PT_MM)}
            if linetype != 'Continuous':
                attribs['linetype'] = linetype
            if not kind:                         # centre / hidden lines take the layer colour
                attribs.update(dxfkit.rgb_attribs(stroke))
            for run in runs:
                _emit_run(msp, run, attribs, tol, counts)
    _rebuild_split_dashes(doc, msp, all_dots, scale, counts)

    tlayer = dxfkit.ensure_layer(doc, 'TEXT')
    for o in _walk(page):
        if not isinstance(o, LTTextContainer):
            continue
        lines = [o] if isinstance(o, LTTextLine) else [l for l in o if isinstance(l, LTTextLine)]
        for line in lines:
            for run in _text_runs(line):
                first, last = run[0], run[-1]
                a, b = first.matrix[0], first.matrix[1]
                ang = math.degrees(math.atan2(b, a))
                u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
                start = (first.matrix[4], first.matrix[5])
                end = (last.matrix[4] + u[0] * last.adv, last.matrix[5] + u[1] * last.adv)
                text = ''.join(c.get_text() for c in run)
                p0, p1 = mp.pt(start), mp.pt(end)
                rot = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 360
                ent = msp.add_text(text, height=max(first.size * 0.72 * mp.k, 1e-3), rotation=rot,
                                   dxfattribs={'layer': tlayer,
                                               **dxfkit.rgb_attribs(_rgb(first.graphicstate.ncolor)),
                                               'style': dxfkit.CJK_STYLE if dxfkit.has_cjk(text) else dxfkit.LATIN_STYLE})
                dxfkit.place_text(ent, p0, p1)
                counts['TEXT'] += 1
    width, height = mp.size
    return {'frame': [0.0, 0.0, round(width, 4), round(height, 4)], 'entities': counts,
            'skipped': skipped, 'profile': prof}
