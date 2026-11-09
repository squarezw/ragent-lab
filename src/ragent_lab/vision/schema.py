"""Pydantic schemas for the engineering-drawing understanding and redraw lab."""
import copy
import json
import math
import re
from typing import Annotated, Literal, Union, get_args

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

MAX_ENTITIES = 5000
MAX_COORD = 10000.0

ViewType = Literal['main', 'top', 'bottom', 'left', 'right', 'section',
                   'detail', 'direction', 'other']
FeatureType = Literal['line', 'polyline', 'circle', 'arc', 'hole', 'slot',
                      'chamfer', 'fillet', 'centerline', 'hidden_line',
                      'text', 'other']
Layer = Literal['OUTLINE', 'CENTER', 'HIDDEN', 'DIM', 'TEXT']


# ---- Understanding -------------------------------------------------------
# Models fill these from the image; a reply is only *asked* to follow the schema (some routes
# can't enforce it), so small slips are repaired instead of failing a minutes-long run:
# null -> empty, an unknown category -> "other", an unreadable number -> unknown, a single
# string where a list belongs -> a one-item list. Only the overall shape stays strict
# (views and dimensions for drawings, summary for the general structure).

class _Base(BaseModel):
    model_config = ConfigDict(extra='ignore')


def _text(v):
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return '；'.join(_text(x) for x in v)
    return json.dumps(v, ensure_ascii=False)


def _items(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _objects(v):
    """A list of objects: drop entries that are not objects."""
    return [x for x in _items(v) if isinstance(x, dict)]


_NUMBER = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        m = _NUMBER.search(v.replace('，', ','))
        return float(m.group()) if m else None
    return None


def _confidence(v):
    n = _num(v)
    if n is None:
        return 0.5
    if 1 < n <= 100:  # a percentage
        n /= 100
    return min(max(n, 0.0), 1.0)


def _point(v, size=2):
    if not isinstance(v, (list, tuple)) or len(v) < size:
        return None
    nums = [_num(x) for x in v[:size]]
    return None if None in nums else nums


def _enum(literal):
    allowed = set(get_args(literal))

    def pick(v):
        key = _text(v).strip().lower().replace('-', '_').replace(' ', '_')
        return key if key in allowed else 'other'
    return BeforeValidator(pick)


Text = Annotated[str, BeforeValidator(_text)]
TextList = Annotated[list[Text], BeforeValidator(_items)]
Num = Annotated[float | None, BeforeValidator(_num)]
Confidence = Annotated[float, BeforeValidator(_confidence)]
DimensionKind = Literal['linear', 'diameter', 'radius', 'angle', 'count', 'other']


class Part(_Base):
    name: Text = ''
    drawing_number: Text = ''
    material: Text = ''
    scale: Text = ''
    notes: TextList = []


class View(_Base):
    id: Text = ''
    type: Annotated[ViewType, _enum(ViewType)] = 'other'
    name: Text = ''
    purpose: Text = ''
    bbox: Annotated[list[float] | None, BeforeValidator(lambda v: _point(v, 4))] = None
    relation: Text = ''


class Feature(_Base):
    id: Text = ''
    view_id: Text = ''
    type: Annotated[FeatureType, _enum(FeatureType)] = 'other'
    description: Text = ''
    center: Annotated[list[float] | None, BeforeValidator(_point)] = None
    radius: Num = None
    points: Annotated[list[list[float]], BeforeValidator(
        lambda v: [q for q in (_point(x) for x in _items(v)) if q is not None])] = []
    start_angle: Num = None
    end_angle: Num = None
    pattern: Text = ''
    visible: Annotated[bool, BeforeValidator(
        lambda v: True if v is None else (v.strip().lower() not in ('false', 'no', '0', '否')
                                          if isinstance(v, str) else bool(v)))] = True


class Dimension(_Base):
    id: Text = ''
    label: Text = ''
    value: Num = None
    kind: Annotated[DimensionKind, _enum(DimensionKind)] = 'other'
    view_id: Text = ''
    feature_ids: TextList = []
    from_ref: Text = ''
    to_ref: Text = ''
    confidence: Confidence = 1.0
    uncertainty: Text = ''


class Derived(_Base):
    value: Text = ''
    basis: Text = ''
    assumption: Text = ''


class Understanding(_Base):
    part: Annotated[Part, BeforeValidator(lambda v: v if isinstance(v, dict) else {})] = Part()
    views: Annotated[list[View], BeforeValidator(_objects)]
    features: Annotated[list[Feature], BeforeValidator(_objects)] = []
    dimensions: Annotated[list[Dimension], BeforeValidator(_objects)]
    relations: TextList = []
    derived: Annotated[list[Derived], BeforeValidator(_objects)] = []
    unresolved: TextList = []


# ---- General understanding (charts, diagrams, any image) -----------------

class Element(_Base):
    id: Text = ''
    type: Text = ''
    label: Text = ''
    description: Text = ''
    confidence: Confidence = 1.0
    uncertainty: Text = ''


class DataPoint(_Base):
    name: Text = ''
    value: Text = ''
    unit: Text = ''
    source: Text = ''
    confidence: Confidence = 1.0
    uncertainty: Text = ''


class GeneralUnderstanding(_Base):
    title: Text = ''
    image_type: Text = ''
    summary: Annotated[str, BeforeValidator(lambda v: v if isinstance(v, str) else _text(v))]
    elements: Annotated[list[Element], BeforeValidator(_objects)] = []
    texts: TextList = []
    data: Annotated[list[DataPoint], BeforeValidator(_objects)] = []
    relations: TextList = []
    unresolved: TextList = []


# ---- Geometry ------------------------------------------------------------

MAX_POLY_POINTS = 2000
MAX_BULGE = 10
MAX_TEXT_CHARS = 200
MAX_TEXT_HEIGHT = 1000
MAX_DIM_TEXT_CHARS = 100
MAX_TOTAL_VERTICES = 50_000
MAX_TOTAL_TEXT_CHARS = 20_000


def _check_point(p: list[float]) -> list[float]:
    if len(p) not in (2, 3):
        raise ValueError('point must have 2 (x, y) or 3 (x, y, bulge) values')
    if not all(math.isfinite(v) for v in p):
        raise ValueError('point values must be finite')
    if len(p) == 3 and abs(p[2]) > MAX_BULGE:
        raise ValueError(f'bulge magnitude exceeds {MAX_BULGE}')
    if any(abs(v) > MAX_COORD for v in p[:2]):
        raise ValueError(f'coordinate magnitude exceeds {MAX_COORD}')
    return p


def _check_xy(p: list[float]) -> list[float]:
    if len(p) != 2:
        raise ValueError('point must have exactly 2 values')
    return p


Point = Annotated[list[float], AfterValidator(_check_point), AfterValidator(_check_xy)]
BulgePoint = Annotated[list[float], AfterValidator(_check_point)]
Radius = Annotated[float, Field(gt=0, le=MAX_COORD, allow_inf_nan=False)]
Angle = Annotated[float, Field(allow_inf_nan=False)]


class _Entity(BaseModel):
    layer: Layer
    dimension_ref: str | None = None


class LineE(_Entity):
    type: Literal['line']
    start: Point
    end: Point


class CircleE(_Entity):
    type: Literal['circle']
    center: Point
    radius: Radius


class ArcE(_Entity):
    type: Literal['arc']
    center: Point
    radius: Radius
    start_angle: Angle
    end_angle: Angle

    @model_validator(mode='after')
    def _nonzero_sweep(self):
        if self.start_angle == self.end_angle:
            raise ValueError('arc start_angle must differ from end_angle')
        return self


class PolylineE(_Entity):
    type: Literal['polyline']
    points: Annotated[list[BulgePoint], Field(min_length=2, max_length=MAX_POLY_POINTS)]
    closed: bool = False


class TextE(_Entity):
    type: Literal['text']
    insert: Point
    height: Annotated[float, Field(gt=0, le=MAX_TEXT_HEIGHT, allow_inf_nan=False)]
    content: Annotated[str, Field(max_length=MAX_TEXT_CHARS)]
    rotation: Angle = 0


class DimensionE(_Entity):
    # p1 == p2 is intentionally allowed; the renderer handles it.
    type: Literal['dimension']
    dimtype: Literal['linear', 'aligned', 'diameter', 'radius']
    p1: Point
    p2: Point
    location: Point
    angle: Angle = 0
    text: Annotated[str | None, Field(max_length=MAX_DIM_TEXT_CHARS)] = None


Entity = Annotated[
    Union[LineE, CircleE, ArcE, PolylineE, TextE, DimensionE],
    Field(discriminator='type'),
]


class Sheet(BaseModel):
    width: Annotated[float, Field(gt=0, le=MAX_COORD, allow_inf_nan=False)]
    height: Annotated[float, Field(gt=0, le=MAX_COORD, allow_inf_nan=False)]


class Geometry(BaseModel):
    units: Literal['mm']
    sheet: Sheet
    entities: Annotated[list[Entity], Field(max_length=MAX_ENTITIES)]

    @model_validator(mode='after')
    def _budget(self):
        vertices = chars = 0
        for e in self.entities:
            if isinstance(e, PolylineE):
                vertices += len(e.points)
            elif isinstance(e, (LineE, DimensionE)):
                vertices += 2
            else:
                vertices += 1
            if isinstance(e, TextE):
                chars += len(e.content)
            elif isinstance(e, DimensionE):
                chars += len(e.text or '')
        if vertices > MAX_TOTAL_VERTICES:
            raise ValueError(f'geometry exceeds {MAX_TOTAL_VERTICES} vertices')
        if chars > MAX_TOTAL_TEXT_CHARS:
            raise ValueError(f'geometry exceeds {MAX_TOTAL_TEXT_CHARS} text characters')
        return self


# ---- Rubric / judgement / consensus -------------------------------------

class RubricItem(BaseModel):
    id: str
    target: str
    expected: str
    uncertain_if: str = ''
    wrong_if: str = ''


class Rubric(BaseModel):
    id: str
    name: str
    image: str
    # Which output structure the rubric grades: 'engineering' or 'general'.
    kind: Literal['engineering', 'general'] = 'engineering'
    rubric_status: Literal['approved', 'draft']
    items: list[RubricItem]


class Judgement(BaseModel):
    item_id: str
    verdict: Literal['correct', 'uncertain', 'wrong']
    evidence: str = ''
    reason: str = ''


class JudgeResult(BaseModel):
    judgements: list[Judgement]


class Interpretation(BaseModel):
    summary: str
    models: list[str]


class ConsensusGroup(BaseModel):
    label: str
    view: str = ''
    interpretations: list[Interpretation]


class Singleton(BaseModel):
    label: str
    models: list[str]
    summary: str


class ConsensusResult(BaseModel):
    groups: list[ConsensusGroup]
    singletons: list[Singleton] = []


# ---- JSON schema helper --------------------------------------------------

# Keywords Gemini's response_schema rejects. Dropping them loosens only what the model is
# told; the reply is still validated against the full Pydantic model.
_LEAN_DROP = {'title', 'default', 'exclusiveMinimum', 'exclusiveMaximum', 'minItems',
              'maxItems', 'maxLength', 'minLength'}


def lean_schema(schema: dict) -> dict:
    """A copy Gemini accepts: oneOf becomes anyOf, unsupported limit keywords are dropped."""
    def walk(node, in_properties=False):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if not in_properties and k in _LEAN_DROP:
                    continue
                out['anyOf' if k == 'oneOf' and not in_properties else k] = \
                    walk(v, in_properties=(k == 'properties' and not in_properties))
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def model_json_schema_for(cls) -> dict:
    """JSON schema for response_format with every $ref inlined (some providers reject $ref).

    Every property is listed as required: Anthropic rejects schemas with more than
    24 optional properties. Pydantic still fills defaults if a model omits a field.
    """
    schema = cls.model_json_schema()
    defs = schema.pop('$defs', {})

    def inline(node, seen=()):
        if isinstance(node, dict):
            ref = node.get('$ref')
            if ref is not None:
                name = ref.rsplit('/', 1)[-1]
                if name in seen:
                    raise ValueError(f'recursive schema not supported: {name}')
                merged = inline(copy.deepcopy(defs[name]), seen + (name,))
                extra = {k: inline(v, seen) for k, v in node.items() if k != '$ref'}
                return {**merged, **extra}
            out = {k: inline(v, seen) for k, v in node.items()
                   if not (k == 'discriminator' and isinstance(v, dict) and 'propertyName' in v)}
            if 'const' in out:  # Gemini rejects "const"; a one-value enum means the same
                out.setdefault('enum', [out['const']])
                del out['const']
            if isinstance(out.get('properties'), dict):
                out['required'] = list(out['properties'])
            return out
        if isinstance(node, list):
            return [inline(v, seen) for v in node]
        return node

    return inline(schema)
