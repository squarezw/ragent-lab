import type {
  ConsensusResult, DimWarning, Geometry, ImageRef, RunMeta, ScoreResult, Understanding,
} from "./types";

export type RunStatus = "idle" | "queued" | "running" | "done" | "failed";

export type RunEntry = {
  model: string;
  model_name: string;
  status: RunStatus;
  understanding?: Understanding;
  meta?: RunMeta;
  error?: string;
  score?: ScoreResult;
  scoreStatus?: "idle" | "running" | "done" | "failed";
  scoreError?: string;
};

export type GenerationEntry = {
  id: string;
  sourceModel: string;
  generator: string;
  status: RunStatus;
  geometry?: Geometry;
  svg?: string;
  warnings?: DimWarning[];
  skipped?: string[];
  meta?: RunMeta;
  error?: string;
};

export type ConsensusEntry = {
  status: "idle" | "running" | "done" | "failed";
  result?: ConsensusResult;
  error?: string;
};

export type VisionState = {
  step: 1 | 2 | 3;
  image: ImageRef | null;
  imageName: string;
  scorable: boolean;
  runs: Record<string, RunEntry>;
  consensus?: ConsensusEntry;
  generations: GenerationEntry[];
};

export const initialState: VisionState = {
  step: 1,
  image: null,
  imageName: "",
  scorable: false,
  runs: {},
  generations: [],
};

export type Action =
  | { type: "selectImage"; image: ImageRef; imageName: string; scorable: boolean }
  | { type: "setStep"; step: 1 | 2 | 3 }
  | { type: "runUpdate"; model: string; patch: Partial<RunEntry> }
  | { type: "scoreUpdate"; model: string; patch: Partial<RunEntry> }
  | { type: "consensusUpdate"; consensus: ConsensusEntry | undefined }
  | { type: "generationAdd"; entry: GenerationEntry }
  | { type: "generationUpdate"; id: string; patch: Partial<GenerationEntry> }
  | { type: "importState"; state: VisionState };

export function reducer(state: VisionState, action: Action): VisionState {
  switch (action.type) {
    case "selectImage":
      return {
        ...state,
        step: 1,
        image: action.image,
        imageName: action.imageName,
        scorable: action.scorable,
        runs: {},
        consensus: undefined,
        generations: [],
      };
    case "setStep":
      return { ...state, step: action.step };
    case "runUpdate": {
      const prev = state.runs[action.model] ?? {
        model: action.model,
        model_name: action.patch.model_name ?? action.model,
        status: "idle" as const,
      };
      return { ...state, runs: { ...state.runs, [action.model]: { ...prev, ...action.patch } } };
    }
    case "scoreUpdate": {
      const prev = state.runs[action.model];
      if (!prev) return state;
      return { ...state, runs: { ...state.runs, [action.model]: { ...prev, ...action.patch } } };
    }
    case "consensusUpdate":
      return { ...state, consensus: action.consensus };
    case "generationAdd":
      return { ...state, generations: [...state.generations, action.entry] };
    case "generationUpdate":
      return {
        ...state,
        generations: state.generations.map((g) =>
          g.id === action.id ? { ...g, ...action.patch } : g,
        ),
      };
    case "importState":
      return action.state;
  }
}

export const EXPORT_VERSION = 1;
const INTERRUPTED = "导入时任务未完成，请重新运行。";
const BAD_FILE = "导入文件格式不正确";

export function exportState(s: VisionState): string {
  return JSON.stringify({ version: EXPORT_VERSION, exported_at: new Date().toISOString(), ...s });
}

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export function importState(text: string): VisionState {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(BAD_FILE);
  }
  if (!isObj(data) || data.version !== EXPORT_VERSION) throw new Error(BAD_FILE);
  const image = data.image;
  if (
    !(image === null || (isObj(image) && (image.kind === "sample" || image.kind === "upload") && typeof image.id === "string")) ||
    typeof data.imageName !== "string" ||
    typeof data.scorable !== "boolean" ||
    !isObj(data.runs) ||
    !Array.isArray(data.generations) ||
    !Object.values(data.runs).every((r) => isObj(r) && typeof r.model === "string") ||
    !data.generations.every((g) => isObj(g) && typeof g.id === "string")
  )
    throw new Error(BAD_FILE);
  const pending = (v: unknown) => v === "queued" || v === "running";
  const runs: VisionState["runs"] = {};
  for (const [k, r] of Object.entries(data.runs as Record<string, RunEntry>)) {
    let e = r;
    if (pending(e.status)) e = { ...e, status: "failed", error: INTERRUPTED };
    if (pending(e.scoreStatus)) e = { ...e, scoreStatus: "failed", scoreError: INTERRUPTED };
    runs[k] = e;
  }
  let consensus = isObj(data.consensus) ? (data.consensus as ConsensusEntry) : undefined;
  if (consensus && pending(consensus.status))
    consensus = { ...consensus, status: "failed", error: INTERRUPTED };
  const generations = (data.generations as GenerationEntry[]).map((g) =>
    pending(g.status) ? { ...g, status: "failed" as const, error: INTERRUPTED } : g,
  );
  return {
    step: 2,
    image: image as ImageRef | null,
    imageName: data.imageName,
    scorable: data.scorable,
    runs,
    consensus,
    generations,
  };
}

const KEY_NAME = "vision-openrouter-key";
const JUDGE_NAME = "vision-judge-model";
export const DEFAULT_JUDGE = "anthropic/claude-opus-5.5";

export const judgeStorage = {
  load(): string {
    try {
      return localStorage.getItem(JUDGE_NAME) || DEFAULT_JUDGE;
    } catch {
      return DEFAULT_JUDGE;
    }
  },
  save(model: string): void {
    try {
      if (model === DEFAULT_JUDGE) localStorage.removeItem(JUDGE_NAME);
      else localStorage.setItem(JUDGE_NAME, model);
    } catch {
      /* storage unavailable */
    }
  },
};

export const keyStorage = {
  load(): string {
    try {
      return localStorage.getItem(KEY_NAME) ?? "";
    } catch {
      return "";
    }
  },
  save(key: string, remember: boolean): void {
    try {
      if (remember) localStorage.setItem(KEY_NAME, key);
      else localStorage.removeItem(KEY_NAME);
    } catch {
      /* storage unavailable */
    }
  },
};
