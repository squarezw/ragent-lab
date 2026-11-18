"""Work out the drawing scale from the drawing's own radius and diameter labels.

A scan, or a PDF drawn at 1:N without a scale note, converts at paper size: the R30 arc on
the sheet comes out as a few millimetres. The labels say what the arcs really measure, so
compare each R / Ø label with the arcs and circles near it and look for one factor that most
labels agree on. Only a drawing that is to scale has such a factor; when the labels disagree
the drawing is not to scale (or the labels were misread) and nothing is changed.
"""
from __future__ import annotations

import math
import re

import numpy as np

# OCR often reads Ø as φ, Φ or p; a lone 'p' before a number is taken as Ø too.
_DIA = r'[ØøΦφ∅pP]|%%[cC]'
_LABEL = re.compile(r'^\s*(?:\d+\s*[-×xX*]\s*)?(?P<kind>R|' + _DIA + r')\s*(?P<val>\d+(?:\.\d+)?)\s*$')


# OCR also reads Ø as the digit 0: "0346", "3-019". A real dimension never starts with a
# zero followed by more digits, so such a label is taken as a diameter.
_MISREAD_DIA = re.compile(r'^\s*(?:\d+\s*[-×xX*]\s*)?0(?P<val>[1-9]\d+(?:\.\d+)?)\s*$')


def parse_label(text: str):
    """('radius' | 'diameter', value) for labels like R30, 4-R30, Ø20, 2-φ19, 0346; else None."""
    t = text.replace('，', '.').strip()
    m = _LABEL.match(t)
    if m:
        return ('radius' if m.group('kind') in ('R', 'r') else 'diameter', float(m.group('val')))
    m = _MISREAD_DIA.match(t)
    if m:
        return ('diameter', float(m.group('val')))
    return None


# Dimensions have at most five digits; longer numbers are dates, part or drawing numbers.
_NUMBER = re.compile(r'^\s*(?P<val>\d{1,5}(?:\.\d+)?)\s*$')
# A drawing scale beyond 1:10000 either way is not plausible.
_LOG_LIMIT = math.log(10000.0)


def _text_axis(e):
    """(centre, unit direction, height, half length) of a TEXT entity."""
    h = float(e.dxf.height)
    t = math.radians(float(e.dxf.get('rotation', 0.0)))
    u = np.array([math.cos(t), math.sin(t)])
    a = np.array([e.dxf.insert.x, e.dxf.insert.y])
    if e.dxf.hasattr('align_point') and e.dxf.get('halign', 0) == 5:   # fit: insert .. align_point
        b = np.array([e.dxf.align_point.x, e.dxf.align_point.y])
    else:
        b = a + u * h * 0.6 * max(len(e.dxf.text), 1)
    return (a + b) / 2 + np.array([-u[1], u[0]]) * h / 2, u, h, float(np.hypot(*(b - a))) / 2


def _linear_votes(texts, lines) -> list[tuple[float, int, str, float]]:
    """Plain numbers written on a dimension line: compare the number with the line's length."""
    votes = []
    if not lines:
        return votes
    P0 = np.array([l[0] for l in lines])
    P1 = np.array([l[1] for l in lines])
    D = P1 - P0
    L = np.hypot(D[:, 0], D[:, 1])
    ok = L > 0
    for idx, (e, value) in texts:
        c, u0, h, half0 = _text_axis(e)
        length = None
        # The dimension line runs along the number, or (a vertical dimension with the number
        # written horizontally) across it, beside the number.
        for u, half, beside in ((u0, half0, False), (np.array([-u0[1], u0[0]]), h / 2, True)):
            length = _dim_length(c, u, half, h, half0, beside, P0, P1, D, L, ok)
            if length is not None:
                break
        if length is not None:
            votes.append((math.log(value / length), idx, e.dxf.text, 0.0))
    return votes


def _dim_length(c, u, half, h, half_text, beside, P0, P1, D, L, ok):
    """Length of the dimension line belonging to a number centred at c, or None."""
    if True:
        n = np.array([-u[1], u[0]])
        cos = np.abs((D @ u)) / np.where(ok, L, 1)
        par = ok & (cos >= math.cos(math.radians(2.0)))
        if not par.any():
            return None
        off0 = (P0 - c) @ n
        off1 = (P1 - c) @ n
        s0 = (P0 - c) @ u
        s1 = (P1 - c) @ u
        lo, hi = np.minimum(s0, s1), np.maximum(s0, s1)
        off = (off0 + off1) / 2
        # the dimension line runs just below (or through) the text and reaches near it; a line
        # beside the number lies off its end, within a couple of character heights
        reach = (half_text + 2 * h) if beside else 2.0 * h
        near = par & (np.abs(off) <= reach) & (lo <= half + 2 * h) & (hi >= -half - 2 * h)
        if beside:
            near &= np.abs(off) >= half_text * 0.5   # beside the number, not through it
        if not near.any():
            return None
        best = min(np.nonzero(near)[0], key=lambda i: abs(off[i]))
        band = par & (np.abs(off - off[best]) <= 0.3 * h)
        segs = sorted(zip(lo[band], hi[band]))
        # grow the span from the text outwards over pieces separated by small gaps
        span_lo, span_hi = -half, half
        gap = max(2 * h, 0.5 * half)
        grew = True
        while grew:
            grew = False
            for a, b in segs:
                if a <= span_hi + gap and b >= span_lo - gap and (a < span_lo or b > span_hi):
                    span_lo, span_hi = min(span_lo, a), max(span_hi, b)
                    grew = True
        # A chain of dimensions (60 | 82 | 60) shares one line: stop at the extension lines
        # crossing it on either side of the number.
        line_off = off[best]
        cross = ok & (cos <= math.sin(math.radians(2.0)))
        if cross.any():
            o0, o1 = off0[cross], off1[cross]
            lo_o, hi_o = np.minimum(o0, o1), np.maximum(o0, o1)
            reaches = (lo_o <= line_off + 0.2 * h) & (hi_o >= line_off - 0.2 * h)
            # an extension line stops just past the dimension line; one that runs on through it
            # (an outline, a centre line, another dimension) is not where this dimension ends
            # It crosses the dimension line and overshoots it a little: a line that stops at
            # the dimension line belongs to another dimension.
            over_hi = (lo_o < line_off - 0.3 * h) & (hi_o - line_off >= 0.3 * h) & (hi_o - line_off <= 1.5 * h)
            over_lo = (hi_o > line_off + 0.3 * h) & (line_off - lo_o >= 0.3 * h) & (line_off - lo_o <= 1.5 * h)
            reaches &= over_hi | over_lo
            at = ((s0 + s1) / 2)[cross][reaches]
            inside = at[(at > span_lo - 0.2 * h) & (at < span_hi + 0.2 * h)]
            left = inside[inside < -half * 0.5]
            right = inside[inside > half * 0.5]
            if left.size and right.size:
                span_lo, span_hi = float(left.max()), float(right.min())
        length = span_hi - span_lo
        if length > 2 * half + 2 * h:          # the line must reach well past the number
            return length
        return None


def _leader_target(c, u, h, half, lines, pts, radii):
    """Index of the curve a leader runs to from the label at c (text centre), or None.

    A leader starts at the label (within about two character heights of the text's box) and
    ends on the curve, or at its centre for a radius drawn from the centre.
    """
    best = None
    for a, b in lines:
        for near_end, far_end in ((np.asarray(a, float), np.asarray(b, float)),
                                  (np.asarray(b, float), np.asarray(a, float))):
            d = near_end - c
            along, across = abs(float(d @ u)), abs(float(d @ np.array([-u[1], u[0]])))
            if along > half + 2 * h or across > 2 * h:
                continue
            gap = np.abs(np.hypot(pts[:, 0] - far_end[0], pts[:, 1] - far_end[1]) - radii)
            k = int(np.argmin(gap))
            if gap[k] <= max(0.05 * radii[k], 0.5 * h) and (best is None or gap[k] < best[0]):
                best = (float(gap[k]), k)
    return None if best is None else best[1]


def from_labels(doc, frame=None, min_agree: int = 3, spread: float = 0.05) -> dict:
    """Find one scale factor (real / drawn) supported by several dimension labels.

    Two kinds of evidence vote: R / Ø labels against the nearest arcs and circles, and plain
    numbers against the dimension line they sit on. The factor most labels agree on wins,
    provided at least min_agree labels of at least two different values back it and it is
    clearly ahead of any other factor (a sheet can hold details drawn at other scales, so
    labels that disagree are expected; a close race means the answer is unclear).
    """
    msp = doc.modelspace()

    def inside(p):
        if frame is None:
            return True
        x, y, w, h = frame
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    curves = []
    for e in msp.query('ARC CIRCLE'):
        c = e.dxf.center
        if inside((c.x, c.y)):
            curves.append((c.x, c.y, e.dxf.radius))
    lines = []
    for e in msp.query('LINE'):
        a, b = e.dxf.start, e.dxf.end
        # dimension and leader lines are solid: centre and hidden lines only mislead here
        if e.dxf.layer not in ('TEXT', 'CENTER', 'HIDDEN') and inside((a.x, a.y)) and inside((b.x, b.y)):
            lines.append(((a.x, a.y), (b.x, b.y)))
    radial, linear = [], []
    for e in msp.query('TEXT'):
        p = e.dxf.insert
        if not inside((p.x, p.y)) or e.dxf.height <= 0:
            continue
        parsed = parse_label(e.dxf.text)
        if parsed:
            radial.append((e, parsed))
            continue
        m = _NUMBER.match(e.dxf.text)
        # a lone digit read by OCR is far more often a list or table number than a dimension
        if m and e.dxf.layer == '_OCR_LABELS' and len(m.group('val')) < 2:
            continue
        if m and float(m.group('val')) > 0:
            linear.append((e, float(m.group('val'))))
    labels = [(e.dxf.text, k, v) for e, (k, v) in radial] + [(e.dxf.text, 'linear', v) for e, v in linear]
    out = {'labels_found': len(labels), 'factor': None, 'agreeing': 0, 'examples': []}
    votes = []                                   # (log factor, label index, text, weight)
    if curves:
        pts = np.array([(c[0], c[1]) for c in curves])
        radii = np.array([c[2] for c in curves])
        for li, (e, (kind, value)) in enumerate(radial):
            target = value if kind == 'radius' else value / 2
            c, u, h, half = _text_axis(e)
            hit = _leader_target(c, u, h, half, lines, pts, radii)
            if hit is not None:                      # the leader says which curve it is
                votes.append((math.log(target / radii[hit]), li, e.dxf.text, 0.0))
                continue
            dist = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - radii)
            for ci in np.argsort(dist)[:2]:         # else only the curves right next to the label
                if radii[ci] > 0 and dist[ci] <= 12 * h:
                    votes.append((math.log(target / radii[ci]), li, e.dxf.text, 0.0))
    votes += [(v[0], v[1] + len(radial), v[2], v[3]) for v in _linear_votes(list(enumerate(linear)), lines)]
    votes = [v for v in votes if abs(v[0]) <= _LOG_LIMIT]
    # Every geometry each label could be measuring, for the dimension check (check_dimensions).
    per: dict[int, list] = {}
    for v in votes:
        per.setdefault(v[1], []).append(math.exp(v[0]))
    out['measured'] = [{'label': labels[i][0], 'value': labels[i][2], 'kind': labels[i][1],
                        'ratios': sorted(r)} for i, r in per.items()]
    if not votes:
        return out
    votes.sort()
    win = math.log(1 + spread)

    def kinds(vs):                               # distinct label values, not repeats of one
        return len({(labels[v[1]][1], labels[v[1]][2]) for v in vs})
    best, lo = None, 0
    for hi in range(len(votes)):                 # sliding window in log space
        while votes[hi][0] - votes[lo][0] > 2 * win:
            lo += 1
        rank = (kinds(votes[lo:hi + 1]), len({v[1] for v in votes[lo:hi + 1]}))
        if best is None or rank > best[0]:
            best = (rank, lo, hi)
    _, lo, hi = best
    n = len({v[1] for v in votes[lo:hi + 1]})
    out['agreeing'] = n
    out['labels_voting'] = len({v[1] for v in votes})
    values = {(labels[v[1]][1], labels[v[1]][2]) for v in votes[lo:hi + 1]}
    # Runner-up: the best window that does not overlap the winner, counting other labels.
    taken = {v[1] for v in votes[lo:hi + 1]}
    edge_lo, edge_hi = votes[lo][0] - 2 * win, votes[hi][0] + 2 * win
    rest = [v for v in votes if (v[0] < edge_lo or v[0] > edge_hi) and v[1] not in taken]
    second, j = 0, 0
    for i in range(len(rest)):
        while rest[i][0] - rest[j][0] > 2 * win:
            j += 1
        second = max(second, kinds(rest[j:i + 1]))
    out['runner_up_values'] = second
    out['agreeing_values'] = len(values)
    # Several labels, of at least two different values (one label repeated, like 4-R30 written
    # three times, or two labels landing on a wrong arc by coincidence must not pass), and
    # clearly ahead of the runner-up.
    # One vote per label (the one nearest the window centre), so a label that also matched a
    # stray arc in the window does not pull the result.
    centre = float(np.median([v[0] for v in votes[lo:hi + 1]]))
    nearest: dict[int, float] = {}
    for v in votes[lo:hi + 1]:
        if v[1] not in nearest or abs(v[0] - centre) < abs(nearest[v[1]] - centre):
            nearest[v[1]] = v[0]
    factor = math.exp(float(np.median(list(nearest.values()))))
    if n < min_agree or len(values) < 2 or len(values) < 1.5 * second:
        if len(values) >= 2 and len(values) >= 1.5 * second:
            # Too little to act on, but worth showing: the user can confirm it.
            out['suggestion'] = factor
            out['suggestion_from'] = sorted({labels[v[1]][0] for v in votes[lo:hi + 1]})
        return out
    out['factor'] = factor
    seen = set()
    for v in votes[lo:hi + 1]:
        if v[1] not in seen and len(out['examples']) < 6:
            seen.add(v[1])
            out['examples'].append(labels[v[1]][0])
    return out


def check_dimensions(measured: list[dict], applied: float, tol: float = 0.02) -> dict:
    """Compare each readable dimension with the converted geometry.

    measured: from_labels(...)['measured'], taken before `applied` (the factor applied after
    the check) was put on the drawing. For each label the candidate geometry closest to the
    labelled size is used; a label that no candidate reproduces within tol is a mismatch,
    which means the drawing is not to scale there (or the label belongs to another view or
    was misread).
    """
    rows, bad = [], []
    for m in measured:
        best = min(m['ratios'], key=lambda r: abs(math.log(applied / r)))
        actual = m['value'] * applied / best
        dev = actual / m['value'] - 1
        row = {'label': m['label'], 'labelled': m['value'], 'measured': round(actual, 3),
               'deviation': round(dev, 4)}
        rows.append(row)
        if abs(dev) > tol:
            bad.append(row)
    bad.sort(key=lambda r: -abs(r['deviation']))
    return {'checked': len(rows), 'matching': len(rows) - len(bad), 'mismatches': bad}
