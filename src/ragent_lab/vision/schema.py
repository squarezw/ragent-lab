"""Pydantic schemas for the engineering-drawing understanding and redraw lab."""
import copy
import math
from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

MAX_ENTITIES = 5000
MAX_COORD = 10000.0

ViewType = Literal['main', 'top', 'bottom', 'left', 'right', 'section',
                   'detail', 'direction', 'other']
FeatureType = Literal['line', 'polyline', 'circle', 'arc', 'hole', 'slot',
                      'chamfer', 'fillet', 'centerline', 'hidden_line',
                      'text', 'other']
Layer = Literal['OUTLINE', 'CENTER', 'HIDDEN', 'DIM', 'TEXT']


# ---- Understanding -------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(extra='ignore')


class Part(_Base):
    name: str
    drawing_number: str = ''
    material: str = ''
    scale: str = ''
    notes: list[str] = []


class View(_Base):
    id: str
    type: ViewType
    name: str
    purpose: str = ''
    bbox: list[float] | None = None
    relation: str = ''


class Feature(_Base):
    id: str
    view_id: str = ''
    type: FeatureType
    description: str = ''
    center: list[float] | None = None
    radius: float | None = None
    points: list[list[float]] = []
    start_angle: float | None = None
    end_angle: float | None = None
    pattern: str = ''
    visible: bool = True


class Dimension(_Base):
    id: str
    label: str = ''
    value: float | None = None
    kind: Literal['linear', 'diameter', 'radius', 'angle', 'count', 'other'] = 'other'
    view_id: str = ''
    feature_ids: list[str] = []
    from_ref: str = ''
    to_ref: str = ''
    confidence: float = 1.0
    uncertainty: str = ''


class Derived(_Base):
    value: str
    basis: str = ''
    assumption: str = ''


class Understanding(_Base):
    part: Part
    views: list[View]
    features: list[Feature] = []
    dimensions: list[Dimension]
    relations: list[str] = []
    derived: list[Derived] = []
    unresolved: list[str] = []


# ---- Geometry ------------------------------------------------------------

def _check_point(p: list[float]) -> list[float]:
    if len(p) not in (2, 3):
        raise ValueError('point must have 2 (x, y) or 3 (x, y, bulge) values')
    if not all(math.isfinite(v) for v in p):
        raise ValueError('point values must be finite')
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
    points: Annotated[list[BulgePoint], Field(min_length=2)]
    closed: bool = False


class TextE(_Entity):
    type: Literal['text']
    insert: Point
    height: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    content: str
    rotation: Angle = 0


class DimensionE(_Entity):
    # p1 == p2 is intentionally allowed; the renderer handles it.
    type: Literal['dimension']
    dimtype: Literal['linear', 'aligned', 'diameter', 'radius']
    p1: Point
    p2: Point
    location: Point
    angle: Angle = 0
    text: str | None = None


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

def model_json_schema_for(cls) -> dict:
    """JSON schema for response_format with every $ref inlined (some providers reject $ref)."""
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
            return {k: inline(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v, seen) for v in node]
        return node

    return inline(schema)
