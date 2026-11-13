"""Render model-produced millimetre geometry to DXF / SVG / PDF, plus dimension cross-checks.

Coordinates: sheet origin lower-left, x right, y up, angles in degrees CCW from +x (same as DXF).
"""
import io
import math
import re
from dataclasses import asdict, dataclass

import ezdxf
from ezdxf import bbox, units
from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
from ezdxf.document import Drawing
from ezdxf.fonts import fonts
from ezdxf.math import BoundingBox2d

from .schema import (ArcE, CircleE, DimensionE, Geometry, LineE, PolylineE, Sheet, TextE,
                     Understanding)

CJK_STYLE = 'CJK'
CJK_FONT = 'NotoSansCJK-Regular.ttc'
# Installed CJK fonts the renderer may substitute for Noto when it is absent (e.g. macOS dev hosts).
_CJK_FALLBACK_FONTS = ('NotoSansCJKsc-Regular.otf', 'NotoSansCJK-Regular.otf',
                       'Hiragino Sans GB.ttc', 'PingFang.ttc', 'STHeiti Light.ttc',
                       'wqy-microhei.ttc', 'msyh.ttc', 'simhei.ttf')

# name -> (ACI color, linetype, lineweight in 1/100 mm)
_LAYERS = {
    'OUTLINE': (7, 'Continuous', 50),
    'CENTER': (1, 'CENTER', 25),
    'HIDDEN': (8, 'HIDDEN', 25),
    'DIM': (3, 'Continuous', 18),
    'TEXT': (5, 'Continuous', 25),
}
# Same scale convention as ezdxf's setup linetypes (CENTER = 5.08 units per cycle).
_HIDDEN_PATTERN = [0.9525, 0.635, -0.3175]
_LTSCALE = 5.0
# Our own dimension style, in mm, used by every dimension and set as the default.
# ezdxf's setup dimstyles assume 1 unit = 1 m and display cm (dimlfac=100); CAD programs
# that ignore per-dimension overrides then show R128 as 12800, so they are not added at all.
DIM_STYLE = 'LAB_MM'
_DIM_STYLE_ATTRS = {
    'dimlfac': 1.0, 'dimscale': 1.0, 'dimtxt': 3.5, 'dimasz': 3.0, 'dimexe': 2.0,
    'dimexo': 1.0, 'dimgap': 1.0, 'dimdec': 2, 'dimzin': 8, 'dimlunit': 2, 'dimtad': 1,
    'dimtxsty': CJK_STYLE,
}
# White paper for the PDF: ezdxf then draws ACI 7 (OUTLINE) black instead of white.
_DRAW_CONFIG = config.Configuration(background_policy=config.BackgroundPolicy.WHITE)
# The on-screen preview is dark, like a CAD model space: ACI 7 is white, and the green
# dimensions and grey hidden lines stand out.
_PREVIEW_CONFIG = config.Configuration(background_policy=config.BackgroundPolicy.BLACK)
_SKIPPED_ATTR = 'vision_skipped'
_SHEET_ATTR = 'vision_sheet'


@dataclass
class DimWarning:
    entity_index: int
    dimension_id: str
    expected: float
    actual: float
    message: str


# ---- DXF document ---------------------------------------------------------

def _xy(p):
    return (float(p[0]), float(p[1]))


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _radial_args(center, mpoint, location, radius) -> dict:
    # ezdxf: `location` (user text position) overrides `mpoint` but then needs `radius`.
    if _dist(center, location) > 0:
        return {'radius': radius, 'location': location}
    return {'mpoint': mpoint}


def _add_dimension(msp, e: DimensionE, attribs: dict) -> None:
    p1, p2, loc = _xy(e.p1), _xy(e.p2), _xy(e.location)
    if _dist(p1, p2) == 0:
        raise ValueError('dimension p1 equals p2')
    text = e.text if e.text else '<>'
    color = _LAYERS[e.layer][0]  # dimension block lines/text follow the layer colour
    override = {'dimclrd': color, 'dimclre': color, 'dimclrt': color}
    common = {'text': text, 'dimstyle': DIM_STYLE, 'override': override, 'dxfattribs': attribs}
    if e.dimtype == 'linear':
        dim = msp.add_linear_dim(base=loc, p1=p1, p2=p2, angle=e.angle, **common)
    elif e.dimtype == 'aligned':
        # signed perpendicular offset of `location` from the p1->p2 line (left side positive)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        distance = (dx * (loc[1] - p1[1]) - dy * (loc[0] - p1[0])) / math.hypot(dx, dy)
        dim = msp.add_aligned_dim(p1=p1, p2=p2, distance=distance, **common)
    elif e.dimtype == 'diameter':  # p1, p2 = diameter end points
        center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        radial = _radial_args(center, p2, loc, _dist(p1, p2) / 2)
        dim = msp.add_diameter_dim(center=center, **radial, **common)
    else:  # radius: p1 = center, p2 = point on the circle
        dim = msp.add_radius_dim(center=p1, **_radial_args(p1, p2, loc, _dist(p1, p2)),
                                 **common)
    dim.render()


def _add_entity(msp, e) -> None:
    attribs = {'layer': e.layer}
    if isinstance(e, LineE):
        msp.add_line(_xy(e.start), _xy(e.end), dxfattribs=attribs)
    elif isinstance(e, CircleE):
        msp.add_circle(_xy(e.center), e.radius, dxfattribs=attribs)
    elif isinstance(e, ArcE):
        msp.add_arc(_xy(e.center), e.radius, e.start_angle, e.end_angle, dxfattribs=attribs)
    elif isinstance(e, PolylineE):
        points = [(p[0], p[1], p[2] if len(p) == 3 else 0.0) for p in e.points]
        msp.add_lwpolyline(points, format='xyb', close=e.closed, dxfattribs=attribs)
    elif isinstance(e, TextE):
        msp.add_text(e.content, height=e.height, rotation=e.rotation,
                     dxfattribs={**attribs, 'style': CJK_STYLE, 'insert': _xy(e.insert)})
    elif isinstance(e, DimensionE):
        _add_dimension(msp, e, attribs)
    else:  # pragma: no cover - schema union is exhaustive
        raise ValueError(f'unsupported entity {type(e).__name__}')


def build_doc(geometry: Geometry) -> Drawing:
    """Build an editable DXF document; entities that fail are skipped and listed on the doc."""
    doc = ezdxf.new('R2018', setup=['linetypes', 'styles'], units=units.MM)
    doc.header['$LTSCALE'] = _LTSCALE
    doc.header['$DIMLFAC'] = 1.0
    if 'HIDDEN' not in doc.linetypes:
        doc.linetypes.add('HIDDEN', pattern=_HIDDEN_PATTERN, description='Hidden __ __ __')
    for name, (color, linetype, lineweight) in _LAYERS.items():
        doc.layers.add(name, color=color, linetype=linetype, lineweight=lineweight)
    doc.styles.add(CJK_STYLE, font=CJK_FONT)
    doc.dimstyles.add(DIM_STYLE, dxfattribs=_DIM_STYLE_ATTRS)
    doc.header['$DIMSTYLE'] = DIM_STYLE
    skipped: list[str] = []
    msp = doc.modelspace()
    for index, entity in enumerate(geometry.entities):
        count = len(msp)
        try:
            _add_entity(msp, entity)
        except Exception as exc:  # one bad entity must not sink the drawing
            for partial in list(msp)[count:]:  # drop half-built entities
                msp.delete_entity(partial)
            skipped.append(f'entity {index} ({entity.type}): {exc}')
    setattr(doc, _SKIPPED_ATTR, skipped)
    setattr(doc, _SHEET_ATTR, (geometry.sheet.width, geometry.sheet.height))
    return doc


def skipped_entities(doc: Drawing) -> list[str]:
    return list(getattr(doc, _SKIPPED_ATTR, []))


def to_dxf(doc: Drawing) -> bytes:
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode('utf-8')


# ---- SVG / PDF ------------------------------------------------------------

def _ensure_cjk_font() -> None:
    """Map the Noto CJK font name to an installed CJK font when Noto itself is missing."""
    fm = fonts.font_manager
    if fm.has_font(CJK_FONT):
        return
    for candidate in _CJK_FALLBACK_FONTS:
        if fm.has_font(candidate):
            fm.add_synonyms({candidate: CJK_FONT}, reverse=False)
            return


def _sheet_of(doc: Drawing) -> tuple[float, float]:
    sheet = getattr(doc, _SHEET_ATTR, None)
    if sheet is not None:
        return sheet
    box = bbox.extents(doc.modelspace(), fast=True)  # doc not built by build_doc
    if box.has_data:
        return (max(box.extmax.x, 1.0), max(box.extmax.y, 1.0))
    return (297.0, 210.0)


def _draw(doc: Drawing, backend, cfg: config.Configuration = _DRAW_CONFIG) -> None:
    frontend = Frontend(RenderContext(doc), backend, config=cfg)
    frontend.draw_layout(doc.modelspace(), finalize=True)


def to_svg(doc: Drawing, sheet: Sheet | None = None, dark: bool = True) -> str:
    """SVG of the modelspace, page sized to the sheet with the sheet origin at lower-left.

    dark: black background with light lines (the on-screen preview); False for white paper.
    """
    _ensure_cjk_font()
    width, height = (sheet.width, sheet.height) if sheet else _sheet_of(doc)
    backend = svg.SVGBackend()
    _draw(doc, backend, _PREVIEW_CONFIG if dark else _DRAW_CONFIG)
    page = layout.Page(width, height, layout.Units.mm)
    settings = layout.Settings(fit_page=False, scale=1.0)
    box = BoundingBox2d([(0, 0), (width, height)])
    return backend.get_string(page, settings=settings, render_box=box, xml_declaration=False)


def to_pdf(doc: Drawing, sheet: Sheet) -> bytes:
    """PDF via matplotlib, page size = sheet size in mm, 1 drawing unit = 1 mm."""
    from matplotlib.figure import Figure
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    _ensure_cjk_font()
    width, height = sheet.width, sheet.height
    # Text is drawn as paths by ezdxf, so no rcParams (global, thread-unsafe) are touched.
    fig = Figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    backend = MatplotlibBackend(ax, adjust_figure=False)
    _draw(doc, backend)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    return buf.getvalue()


# ---- Dimension cross-check ------------------------------------------------

def _measure(entity, kind: str) -> float | None:
    """Measured value of `entity` comparable to an understanding dimension of `kind`."""
    if kind in ('angle', 'count'):
        return None
    if isinstance(entity, (CircleE, ArcE)):
        if kind == 'diameter':
            return 2 * entity.radius
        if kind == 'radius':
            return entity.radius
        return None
    # A line is not measured: a dimension sets a distance, and the lines it positions (an edge
    # at x = 195, a 25 x 25 chamfer's slanted edge) are rarely that long. Lines are covered by
    # the dimension entities that span them.
    if isinstance(entity, DimensionE):
        p1, p2 = entity.p1, entity.p2
        if entity.dimtype == 'linear':
            a = math.radians(entity.angle)
            return abs((p2[0] - p1[0]) * math.cos(a) + (p2[1] - p1[1]) * math.sin(a))
        length = _dist(p1, p2)
        if entity.dimtype == 'diameter' and kind == 'radius':
            return length / 2
        if entity.dimtype == 'radius' and kind == 'diameter':
            return length * 2
        return length
    return None


def check_dimensions(geometry: Geometry, understanding: Understanding) -> list[DimWarning]:
    dims = {d.id: d for d in understanding.dimensions}
    warnings: list[DimWarning] = []
    for index, entity in enumerate(geometry.entities):
        ref = entity.dimension_ref
        if not ref:
            continue
        dim = dims.get(ref)
        if dim is None:
            warnings.append(DimWarning(index, ref, 0.0, 0.0, f'引用了不存在的尺寸 {ref}'))
            continue
        if dim.value is None:
            continue
        actual = _measure(entity, dim.kind)
        if actual is None:
            continue
        expected = float(dim.value)
        tolerance = max(0.5, 0.005 * abs(expected))
        if abs(actual - expected) > tolerance:
            warnings.append(DimWarning(
                index, ref, expected, round(actual, 4),
                f'尺寸 {dim.label or ref} 与图形不符'))
    return warnings


# A dimension's label is what CAD shows, so the length it spans must match the number in it.
# Labels this check reads: 390, 20.5, R108, Ø20, 2-Ø19, 4×Ø8, 60±0.1, Ø20H7. Labels it
# cannot compare are skipped: the measured placeholder <>, angles (45°), chamfers (C2),
# chains written as one label (20+20) and text without a number.
_DIAMETER_MARKS = ('Ø', 'ø', 'Φ', 'φ', '∅', '%%c', '%%C')
_COUNT_PREFIX = re.compile(r'^\s*\d+\s*[-×xX*]\s*(?=[ØøΦφ∅Rr%])')
_NUMBER = re.compile(r'\d+(?:\.\d+)?')


def label_value(text: str | None) -> tuple[float, str] | None:
    """(value, 'linear' | 'diameter' | 'radius') read from a dimension label, or None."""
    if not text:
        return None
    t = text.strip().replace('，', '.').replace('．', '.')
    if '<>' in t or '°' in t or '+' in _COUNT_PREFIX.sub('', t).split('±')[0]:
        return None
    t = _COUNT_PREFIX.sub('', t)
    m = _NUMBER.search(t)
    if m is None:
        return None
    head = t[:m.start()]
    if head.strip().upper().startswith('C'):
        return None
    if any(mark in head for mark in _DIAMETER_MARKS):
        kind = 'diameter'
    elif head.strip().upper().endswith('R'):
        kind = 'radius'
    elif head.strip():
        return None  # some other prefix (M10 thread, S sphere…): not a plain size
    else:
        kind = 'linear'
    return float(m.group()), kind


def check_dimension_labels(geometry: Geometry) -> list[DimWarning]:
    """Every dimension whose label names a size must measure that size (any process)."""
    warnings: list[DimWarning] = []
    for index, entity in enumerate(geometry.entities):
        if not isinstance(entity, DimensionE):
            continue
        parsed = label_value(entity.text)
        if parsed is None:
            continue
        expected, kind = parsed
        actual = _measure(entity, kind)
        if actual is None:
            continue
        tolerance = max(0.5, 0.005 * abs(expected))
        if abs(actual - expected) > tolerance:
            warnings.append(DimWarning(index, entity.text or '', expected, round(actual, 4),
                                       f'标注「{entity.text}」与实际长度不符'))
    return warnings


def render_all(geometry: Geometry, understanding: Understanding | None) -> dict:
    doc = build_doc(geometry)
    # Checked against the understanding (when there is one) and, always, against each label.
    warnings = check_dimensions(geometry, understanding) if understanding is not None else []
    flagged = {w.entity_index for w in warnings}
    warnings += [w for w in check_dimension_labels(geometry) if w.entity_index not in flagged]
    warnings.sort(key=lambda w: w.entity_index)
    return {
        'svg': to_svg(doc, geometry.sheet),
        'warnings': [asdict(w) for w in warnings],
        'skipped': skipped_entities(doc),
    }
