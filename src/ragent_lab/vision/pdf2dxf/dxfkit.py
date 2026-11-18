"""Shared helpers for writing DXF: document setup, layers, colours, linetypes, curve fitting.

Everything here works in "drawing units" (millimetres after scaling). Callers convert from
PDF points or image pixels before calling in.
"""
from __future__ import annotations

import math
import re

import ezdxf
import numpy as np
from ezdxf import units

# DXF lineweights are an enumerated set, in 1/100 mm.
_LINEWEIGHTS = (0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60, 70, 80, 90, 100, 106,
                120, 140, 158, 200, 211)
CJK_STYLE, LATIN_STYLE = 'CJK', 'LATIN'


def new_doc(cjk_font: str = 'simsun.ttc', latin_font: str = 'arial.ttf'):
    """An R2018 document in millimetres with two text styles."""
    doc = ezdxf.new('R2018', setup=['linetypes'], units=units.MM)
    doc.header['$MEASUREMENT'] = 1
    doc.header['$LTSCALE'] = 1.0
    doc.styles.add(LATIN_STYLE, font=latin_font)
    doc.styles.add(CJK_STYLE, font=cjk_font)
    return doc


def has_cjk(text: str) -> bool:
    return any('⺀' <= ch <= '鿿' or '豈' <= ch <= '﫿' or '＀' <= ch <= '￯'
               for ch in text)


def layer_name(raw: str) -> str:
    """A name every CAD program accepts (no <>/\\":;?*|=` and at most 255 chars)."""
    name = re.sub(r'[<>/\\":;?*|=`,\x00-\x1f]', '_', raw).strip() or 'UNNAMED'
    return name[:255]


def ensure_layer(doc, name: str, **attribs) -> str:
    name = layer_name(name)
    if name not in doc.layers:
        doc.layers.add(name, **attribs)
    return name


def lineweight(mm: float) -> int:
    """Nearest valid DXF lineweight (1/100 mm) for a stroke width in millimetres."""
    target = max(0.0, mm) * 100
    return min(_LINEWEIGHTS, key=lambda w: abs(w - target))


def rgb_attribs(rgb) -> dict:
    """DXF colour attributes for an RGB tuple of floats 0..1 (or None).

    Pure black becomes ACI 7, which CAD shows black on white and white on black; any other
    colour is written as true colour so it survives unchanged.
    """
    if rgb is None:
        return {}
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb[:3])
    if max(r, g, b) <= 12:
        return {'color': 7}
    return {'true_color': (r << 16) | (g << 8) | b}


def ensure_linetype(doc, dashes_mm: list[float]) -> str:
    """A linetype for a dash pattern [on, off, on, off, ...] in drawing units; '' = continuous."""
    pat = [round(abs(d), 3) for d in dashes_mm if abs(d) > 1e-6]
    if len(pat) < 2:
        return 'Continuous'
    if len(pat) % 2:
        pat = pat * 2
    name = 'DASH_' + '_'.join(f'{d:g}'.replace('.', 'p') for d in pat)[:60]
    if name not in doc.linetypes:
        elements = [d if i % 2 == 0 else -d for i, d in enumerate(pat)]
        doc.linetypes.add(name, pattern=[sum(pat)] + elements,
                          description='PDF dash ' + ' '.join(f'{d:g}' for d in pat))
    return name


def place_text(entity, start, end) -> None:
    """Pin a TEXT between its baseline start and end ("fit" alignment).

    CAD programs rarely have the PDF's font, and a substitute font has other letter widths.
    Fit alignment keeps the height and stretches the letters to end where the original text
    ended, so labels do not run into the geometry next to them.
    """
    from ezdxf.enums import TextEntityAlignment
    if math.dist(start, end) < 1e-9 or len(entity.dxf.text) < 2:
        entity.dxf.insert = start
        return
    entity.set_placement(start, end, TextEntityAlignment.FIT)


# ---- geometry ---------------------------------------------------------------

def fit_circle(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle through Nx2 points -> (cx, cy, r, max radial error)."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    err = float(np.max(np.abs(np.hypot(x - cx, y - cy) - r))) if r else float('inf')
    return float(cx), float(cy), r, err


def bezier_points(p0, p1, p2, p3, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def arc_from_points(pts: np.ndarray, tol: float, min_sagitta: float = 1.0):
    """If the ordered points lie on one circle within tol, return (cx, cy, r, a0, a1, ccw, sweep).

    Angles are degrees of the first and last point; ccw says which way the points run. Returns
    None for straight runs (the sagitta is below tol) and for fits worse than tol.
    """
    if len(pts) < 3:
        return None
    chord = float(np.hypot(*(pts[-1] - pts[0])))
    length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    if length < 4 * tol:
        return None
    cx, cy, r, err = fit_circle(pts)
    if not math.isfinite(r) or err > tol or r > 20 * length:
        return None
    # A straight run with a little noise also "fits" a very large circle. Only call it an arc
    # when the bulge is clearly more than the tolerance.
    mid = pts[len(pts) // 2]
    if chord > 1e-9:
        d = pts[-1] - pts[0]
        sag = abs(d[0] * (mid[1] - pts[0][1]) - d[1] * (mid[0] - pts[0][0])) / chord
        if sag < min_sagitta * tol and length < 1.2 * chord:
            return None
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    step = (np.diff(ang) + 180) % 360 - 180
    total = float(np.sum(step))
    if abs(total) < 1.0 or np.any(np.sign(step[np.abs(step) > 1e-6]) != np.sign(total)):
        return None
    return cx, cy, r, float(ang[0]) % 360, float(ang[0] + total) % 360, total > 0, abs(total)


def add_arc_or_circle(msp, cx, cy, r, a0, a1, ccw, sweep, attribs) -> str:
    """Write a fitted arc; a sweep of (almost) a full turn becomes a CIRCLE."""
    if sweep >= 359.0:
        msp.add_circle((cx, cy), r, dxfattribs=attribs)
        return 'CIRCLE'
    start, end = (a0, a1) if ccw else (a1, a0)  # DXF arcs always run counter-clockwise
    msp.add_arc((cx, cy), r, start, end, dxfattribs=attribs)
    return 'ARC'


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an open Nx2 polyline."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        n = math.hypot(*d)
        seg = pts[i + 1:j]
        if n < 1e-12:
            dist = np.hypot(*(seg - a).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / n
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


# ---- dashed and centre lines ------------------------------------------------
# Layer colours follow the usual CAD convention (and the ragent lab renderer): centre lines
# red, hidden lines grey. Entities on these layers are BYLAYER unless --colors source.
LINE_LAYERS = {'center': ('CENTER', 1), 'hidden': ('HIDDEN', 8)}


def ensure_pattern_linetype(doc, kind: str, dash: float, gap: float) -> str:
    """A linetype matching a measured pattern (drawing units): CENTER = dash, gap, dot, gap."""
    dash, gap = max(dash, 1e-3), max(gap, 1e-3)
    if kind == 'center':
        elements = [dash, -gap / 2, 0.0, -gap / 2]
        desc = 'Center ____ . ____'
    else:
        elements = [dash, -gap]
        desc = 'Hidden __ __ __'
    name = f'{kind.upper()}_{dash:.3g}_{gap:.3g}'.replace('.', 'p')
    if name not in doc.linetypes:
        doc.linetypes.add(name, pattern=[sum(abs(e) for e in elements)] + elements, description=desc)
    return name


def classify_dash_array(dashes: list[float]) -> str:
    """'center' for a dash-dot pattern, 'hidden' for a plain dash pattern."""
    pat = [abs(d) for d in dashes]
    if len(pat) >= 4 and min(pat[0::2]) <= 0.3 * max(pat[0::2]):
        return 'center'
    return 'hidden'


def find_dashed(segs, dots, tol: float, max_gap: float, min_pieces: int = 3, min_dash: float = 0.0):
    """Find collinear runs of short pieces that form one dashed or dash-dot line.

    segs: [(a, b)] line pieces; dots: [(x, y)] isolated dots (the dots of dash-dot lines).
    Returns (solid_indices, lines) where lines are dicts with start, end, kind ('center' or
    'hidden'), dash and gap (median lengths), and the indices of the pieces they replace.
    """
    from scipy.spatial import cKDTree
    recs = []
    for i, (a, b) in enumerate(segs):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
        if ang > 179.0:
            ang -= 180.0
        recs.append((ang, i, a, b, length))
    recs.sort()
    tree = cKDTree(np.asarray(dots, float)) if len(dots) else None
    used: set[int] = set()
    lines = []
    k = 0
    while k < len(recs):                         # clusters of near-equal angle
        j = k + 1
        while j < len(recs) and recs[j][0] - recs[j - 1][0] <= 0.8:
            j += 1
        group = recs[k:j]
        k = j
        if len(group) < min_pieces:
            continue
        t = math.radians(sum(g[0] * g[4] for g in group) / sum(g[4] for g in group))
        u = np.array([math.cos(t), math.sin(t)])
        n = np.array([-u[1], u[0]])
        rows = sorted(((float(((g[2] + g[3]) / 2) @ n),) + tuple(sorted((float(g[2] @ u), float(g[3] @ u)))) + (g[1], g[4])
                       for g in group), key=lambda r: r[0])
        r0 = 0
        while r0 < len(rows):                    # bands of near-equal offset
            r1 = r0 + 1
            while r1 < len(rows) and rows[r1][0] - rows[r1 - 1][0] <= tol:
                r1 += 1
            band = sorted(rows[r0:r1], key=lambda r: r[1])
            r0 = r1
            chain = [band[0]] if band else []
            for row in band[1:] + [None]:
                if row is not None and -tol <= row[1] - chain[-1][2] <= max_gap:
                    chain.append(row)
                    continue
                if len(chain) >= min_pieces:
                    lengths = [c[4] for c in chain]
                    gaps = [max(b[1] - a[2], 0.0) for a, b in zip(chain, chain[1:])]
                    med_len, med_gap = float(np.median(lengths)), float(np.median(gaps))
                    # Regular pieces: no piece far longer than the rest (a solid line that
                    # merely touches a dashed one is left alone).
                    longest = max(lengths)
                    longs = [x for x in lengths if x >= 0.5 * longest]
                    shorts = [x for x in lengths if x < 0.35 * longest]
                    # Dash-dot: long dashes alternating with short ones (at scan resolution the
                    # "dot" is a short dash). Plain dashes: pieces of similar length.
                    alternating = len(shorts) >= 0.3 * len(lengths) and sum(
                        (a < 0.35 * longest) != (b < 0.35 * longest) for a, b in zip(lengths, lengths[1:])
                    ) >= 0.6 * (len(lengths) - 1)
                    regular = longest <= 4 * med_len + 2 * tol
                    if med_gap > tol and (alternating or regular) and float(np.median(longs)) >= min_dash:
                        rho = sum(c[0] * c[4] for c in chain) / sum(lengths)
                        dotted = 0
                        for a, b in zip(chain, chain[1:]):
                            mid = u * ((a[2] + b[1]) / 2) + n * rho
                            if tree is not None:
                                for q in tree.query_ball_point(mid, (b[1] - a[2]) / 2 + tol):
                                    if abs(float(np.asarray(dots[q]) @ n) - rho) <= tol:
                                        dotted += 1
                                        break
                        if alternating:
                            kind, dash = 'center', float(np.median(longs))
                            gap = 2 * med_gap + float(np.median(shorts))
                        else:
                            kind = 'center' if dotted >= 0.4 * len(gaps) else 'hidden'
                            dash, gap = med_len, med_gap
                        lines.append({'start': tuple(u * chain[0][1] + n * rho),
                                      'end': tuple(u * chain[-1][2] + n * rho), 'kind': kind,
                                      'dash': dash, 'gap': gap,
                                      'pieces': [c[3] for c in chain]})
                        used.update(c[3] for c in chain)
                chain = [row] if row is not None else []
    solid = [i for i in range(len(segs)) if i not in used]
    return solid, lines


def promote_centre_lines(doc, tol: float) -> int:
    """Move dashed lines that are really centre lines from HIDDEN to CENTER.

    Many drawings draw centre lines with the same plain dash as hidden edges, so the pattern
    cannot tell them apart. The geometry can: a centre line runs through the centre of a
    circle or arc and past it, or across the middle of a shape and out past its outline,
    while a hidden edge lies inside the outline.
    """
    msp = doc.modelspace()
    hidden = [e for e in msp.query('LINE') if e.dxf.layer == 'HIDDEN']
    if not hidden:
        return 0
    centres = [(e.dxf.center.x, e.dxf.center.y, e.dxf.radius) for e in msp.query('CIRCLE ARC')
               if e.dxf.layer != 'TEXT']
    boxes = []
    for e in msp.query('LWPOLYLINE'):
        if e.closed and e.dxf.layer not in ('HIDDEN', 'TEXT'):
            pts = np.array([p[:2] for p in e.get_points()], float)
            if len(pts) >= 3:
                boxes.append((pts.min(axis=0), pts.max(axis=0)))
    moved = 0
    for e in hidden:
        a = np.array([e.dxf.start.x, e.dxf.start.y])
        b = np.array([e.dxf.end.x, e.dxf.end.y])
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        u = d / length
        n = np.array([-u[1], u[0]])
        centre = False
        for cx, cy, r in centres:
            p = np.array([cx, cy]) - a
            t = float(p @ u)
            if abs(float(p @ n)) <= max(tol, 0.03 * r) and 0.2 * r <= t <= length - 0.2 * r \
                    and length >= 1.5 * r:
                centre = True
                break
        if not centre:
            axis = abs(u[0]) >= math.cos(math.radians(1.0)) or abs(u[1]) >= math.cos(math.radians(1.0))
            for lo, hi in boxes if axis else []:
                mid = (lo + hi) / 2
                p = mid - a
                t = float(p @ u)
                if abs(float(p @ n)) > tol or not (0 < t < length):
                    continue
                # the line's ends lie outside the box along its own direction
                ext = abs(float((hi - lo) @ u)) / 2
                if t > ext + tol and (length - t) > ext + tol:
                    centre = True
                    break
        if centre:
            e.dxf.layer = ensure_layer(doc, LINE_LAYERS['center'][0], color=LINE_LAYERS['center'][1])
            moved += 1
    return moved
