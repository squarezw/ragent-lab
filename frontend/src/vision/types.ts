// Mirrors the backend JSON (src/ragent_lab/api/vision.py, vision/schema.py, service.py).

export type VisionModel = {
  id: string;
  name: string;
  family: string;
  notes: string;
  params: {
    max_tokens: number | null;
    temperature: number | null;
    reasoning: Record<string, unknown> | null;
  };
  // Set for options built from the model settings (only custom models get one).
  tag?: string;
};

// Which output structure an understanding uses (see vision/prompts.py PURPOSES).
export type PurposeKind = "engineering" | "general";

export type Sample = {
  id: string;
  name: string;
  scorable: boolean;
  kind?: PurposeKind;
};

export type Purpose = {
  id: string;
  name: string;
  kind: PurposeKind;
  description: string;
  prompt: string;
};

export type ImageRef = { kind: "sample" | "upload"; id: string };

export type UploadedImage = {
  image_id: string;
  width: number;
  height: number;
  sha256: string;
  name: string;
};

export type ViewType =
  | "main" | "top" | "bottom" | "left" | "right"
  | "section" | "detail" | "direction" | "other";
export type FeatureType =
  | "line" | "polyline" | "circle" | "arc" | "hole" | "slot"
  | "chamfer" | "fillet" | "centerline" | "hidden_line" | "text" | "other";
export type Layer = "OUTLINE" | "CENTER" | "HIDDEN" | "DIM" | "TEXT";

export type Understanding = {
  part: {
    name: string;
    drawing_number: string;
    material: string;
    scale: string;
    notes: string[];
  };
  views: {
    id: string;
    type: ViewType;
    name: string;
    purpose: string;
    bbox: number[] | null;
    relation: string;
  }[];
  features: {
    id: string;
    view_id: string;
    type: FeatureType;
    description: string;
    center: number[] | null;
    radius: number | null;
    points: number[][];
    start_angle: number | null;
    end_angle: number | null;
    pattern: string;
    visible: boolean;
  }[];
  dimensions: {
    id: string;
    label: string;
    value: number | null;
    kind: "linear" | "diameter" | "radius" | "angle" | "count" | "other";
    view_id: string;
    feature_ids: string[];
    from_ref: string;
    to_ref: string;
    confidence: number;
    uncertainty: string;
  }[];
  relations: string[];
  derived: { value: string; basis: string; assumption: string }[];
  unresolved: string[];
};

export type GeneralUnderstanding = {
  title: string;
  image_type: string;
  summary: string;
  elements: {
    id: string;
    type: string;
    label: string;
    description: string;
    confidence: number;
    uncertainty: string;
  }[];
  texts: string[];
  data: {
    name: string;
    value: string;
    unit: string;
    source: string;
    confidence: number;
    uncertainty: string;
  }[];
  relations: string[];
  unresolved: string[];
};

export type AnyUnderstanding = Understanding | GeneralUnderstanding;

export const isEngineering = (u: AnyUnderstanding): u is Understanding =>
  "dimensions" in u;

type EntityBase = { layer: Layer; dimension_ref: string | null };
export type Entity =
  | (EntityBase & { type: "line"; start: number[]; end: number[] })
  | (EntityBase & { type: "circle"; center: number[]; radius: number })
  | (EntityBase & {
      type: "arc";
      center: number[];
      radius: number;
      start_angle: number;
      end_angle: number;
    })
  | (EntityBase & { type: "polyline"; points: number[][]; closed: boolean })
  | (EntityBase & {
      type: "text";
      insert: number[];
      height: number;
      content: string;
      rotation: number;
    })
  | (EntityBase & {
      type: "dimension";
      dimtype: "linear" | "aligned" | "diameter" | "radius";
      p1: number[];
      p2: number[];
      location: number[];
      angle: number;
      text: string | null;
    });

export type Geometry = {
  units: "mm";
  sheet: { width: number; height: number };
  entities: Entity[];
};

export type Judgement = {
  item_id: string;
  verdict: "correct" | "uncertain" | "wrong";
  evidence: string;
  reason: string;
};

export type RunMeta = {
  gateway?: string;
  credits?: number | null;
  model: string;
  provider: string | null;
  finish_reason: string | null;
  usage: Record<string, unknown>;
  cost: number | null;
  elapsed_ms: number;
  params: {
    max_tokens: number | null;
    temperature: number | null;
    reasoning: Record<string, unknown> | null;
  };
};

export type UnderstandResult = {
  model: string;
  model_name: string;
  purpose?: string;
  kind?: PurposeKind;
  understanding: AnyUnderstanding;
  meta: RunMeta;
};

export type ScoreResult = {
  score: number;
  counts: { correct: number; uncertain: number; wrong: number };
  judgements: Judgement[];
  meta: RunMeta;
};

export type ConsensusResult = {
  groups: {
    label: string;
    view: string;
    interpretations: { summary: string; models: string[] }[];
  }[];
  singletons: { label: string; models: string[]; summary: string }[];
  meta: RunMeta;
};

// An unknown dimension_ref yields expected = actual = 0 (not a real 0 mm value).
export type DimWarning = {
  entity_index: number;
  dimension_id: string;
  expected: number;
  actual: number;
  message: string;
};

export type GenerateResult = {
  geometry: Geometry;
  svg: string;
  warnings: DimWarning[];
  skipped: string[];
  meta: RunMeta;
};

export type JobStatus = "queued" | "running" | "done" | "failed";

export type Job<T> = {
  id: string;
  kind: string;
  label: string;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  result: T | null;
  error: string | null;
  error_kind: string | null;
};
