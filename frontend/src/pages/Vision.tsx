import { useEffect, useReducer, useRef, useState } from "react";
import { ErrorMessage, PageHeader } from "../components/UI";
import {
  listModels,
  listSamples,
  pollJob,
  submit,
  type SubmitKind,
} from "../vision/client";
import {
  exportState,
  importState,
  initialState,
  keyStorage,
  reducer,
} from "../vision/state";
import type {
  ConsensusResult,
  GenerateResult,
  ImageRef,
  Sample,
  ScoreResult,
  UnderstandResult,
  Understanding,
  VisionModel,
} from "../vision/types";
import StepSelect from "../vision/StepSelect";
import StepUnderstand from "../vision/StepUnderstand";
import StepRedraw from "../vision/StepRedraw";

const NEED_KEY = "请先在第 ① 步填写 OpenRouter Key。";
const isAbort = (e: unknown) => (e as Error)?.name === "AbortError";

async function runJob<T>(
  kind: SubmitKind,
  body: unknown,
  signal: AbortSignal,
): Promise<T> {
  const id = await submit(kind, body);
  if (signal.aborted) throw new DOMException("Aborted", "AbortError");
  return pollJob<T>(id, { signal });
}

let generationSeq = 0;

export default function Vision() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [models, setModels] = useState<VisionModel[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [apiKey, setApiKey] = useState(() => keyStorage.load());
  const [remember, setRemember] = useState(() => keyStorage.load() !== "");
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  // Every in-flight poll listens to this controller; replaced on re-select / import, aborted on unmount.
  const ctrl = useRef(new AbortController());

  useEffect(() => {
    ctrl.current = new AbortController();
    return () => ctrl.current.abort();
  }, []);
  useEffect(() => {
    listModels()
      .then((list) => {
        setModels(list);
        setSelected(list.map((m) => m.id));
      })
      .catch((e) => setError(e.message));
    listSamples()
      .then(setSamples)
      .catch((e) => setError(e.message));
  }, []);

  function resetJobs() {
    ctrl.current.abort();
    ctrl.current = new AbortController();
    setActionError("");
  }
  function changeKey(key: string, keep = remember) {
    setApiKey(key);
    setRemember(keep);
    keyStorage.save(key, keep);
  }
  function requireKey(): boolean {
    if (apiKey.trim()) return true;
    setActionError(NEED_KEY);
    return false;
  }

  async function score(
    model: string,
    understanding: Understanding,
    image: ImageRef,
  ) {
    const signal = ctrl.current.signal;
    dispatch({
      type: "scoreUpdate",
      model,
      patch: { scoreStatus: "running", scoreError: undefined },
    });
    try {
      const result = await runJob<ScoreResult>(
        "score",
        { sample_id: image.id, understanding, api_key: apiKey },
        signal,
      );
      if (!signal.aborted)
        dispatch({
          type: "scoreUpdate",
          model,
          patch: { scoreStatus: "done", score: result },
        });
    } catch (e) {
      if (!isAbort(e) && !signal.aborted)
        dispatch({
          type: "scoreUpdate",
          model,
          patch: { scoreStatus: "failed", scoreError: (e as Error).message },
        });
    }
  }

  async function understand(model: string) {
    const image = state.image;
    if (!image) return;
    const scorable = state.scorable && image.kind === "sample";
    const signal = ctrl.current.signal;
    const name =
      models.find((m) => m.id === model)?.name ??
      state.runs[model]?.model_name ??
      model;
    dispatch({
      type: "runUpdate",
      model,
      patch: {
        model_name: name,
        status: "running",
        understanding: undefined,
        meta: undefined,
        error: undefined,
        score: undefined,
        scoreStatus: undefined,
        scoreError: undefined,
      },
    });
    try {
      const result = await runJob<UnderstandResult>(
        "understand",
        { image, model, api_key: apiKey },
        signal,
      );
      if (signal.aborted) return;
      dispatch({
        type: "runUpdate",
        model,
        patch: {
          status: "done",
          model_name: result.model_name || name,
          understanding: result.understanding,
          meta: result.meta,
        },
      });
      if (scorable) await score(model, result.understanding, image);
    } catch (e) {
      if (!isAbort(e) && !signal.aborted)
        dispatch({
          type: "runUpdate",
          model,
          patch: { status: "failed", error: (e as Error).message },
        });
    }
  }

  function startAll() {
    setActionError("");
    if (!requireKey()) return;
    if (!selected.length) return setActionError("请至少选择一个模型。");
    dispatch({ type: "consensusUpdate", consensus: undefined });
    selected.forEach((m) => void understand(m));
  }

  function retry(model: string) {
    if (requireKey()) void understand(model);
  }
  function retryScore(model: string) {
    const u = state.runs[model]?.understanding;
    if (u && state.image && requireKey()) void score(model, u, state.image);
  }

  async function consensus() {
    if (!requireKey()) return;
    const signal = ctrl.current.signal;
    const results = Object.values(state.runs)
      .filter((r) => r.status === "done" && r.understanding)
      .map((r) => ({
        model_name: r.model_name,
        understanding: r.understanding,
      }));
    dispatch({ type: "consensusUpdate", consensus: { status: "running" } });
    try {
      const result = await runJob<ConsensusResult>(
        "consensus",
        { results, api_key: apiKey },
        signal,
      );
      if (!signal.aborted)
        dispatch({
          type: "consensusUpdate",
          consensus: { status: "done", result },
        });
    } catch (e) {
      if (!isAbort(e) && !signal.aborted)
        dispatch({
          type: "consensusUpdate",
          consensus: { status: "failed", error: (e as Error).message },
        });
    }
  }

  async function generate(sourceModel: string, generator: string) {
    setActionError("");
    const understanding = state.runs[sourceModel]?.understanding;
    const image = state.image;
    if (!understanding || !image || !requireKey()) return;
    const signal = ctrl.current.signal;
    const id = `gen-${Date.now()}-${++generationSeq}`;
    dispatch({
      type: "generationAdd",
      entry: { id, sourceModel, generator, status: "running" },
    });
    try {
      const r = await runJob<GenerateResult>(
        "generate",
        { image, model: generator, understanding, api_key: apiKey },
        signal,
      );
      if (!signal.aborted)
        dispatch({
          type: "generationUpdate",
          id,
          patch: {
            status: "done",
            geometry: r.geometry,
            svg: r.svg,
            warnings: r.warnings,
            skipped: r.skipped,
            meta: r.meta,
          },
        });
    } catch (e) {
      if (!isAbort(e) && !signal.aborted)
        dispatch({
          type: "generationUpdate",
          id,
          patch: { status: "failed", error: (e as Error).message },
        });
    }
  }

  function exportJson() {
    const blob = new Blob([exportState(state)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vision-results-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  async function importJson(file: File) {
    try {
      const next = importState(await file.text());
      resetJobs();
      dispatch({ type: "importState", state: next });
    } catch (e) {
      setActionError((e as Error).message);
    }
  }

  const hasDone = Object.values(state.runs).some((r) => r.status === "done");
  const steps = [
    { step: 1 as const, label: "① 选图", enabled: true },
    { step: 2 as const, label: "② 理解对比", enabled: !!state.image },
    { step: 3 as const, label: "③ 还原", enabled: hasDone },
  ];
  const go = (step: 1 | 2 | 3) => {
    setActionError("");
    dispatch({ type: "setStep", step });
  };

  return (
    <>
      <PageHeader
        eyebrow="EXPERIMENT 03 / VISION UNDERSTANDING"
        title="从看见，到真正理解。"
        description="同一张工程图纸，交给多个视觉模型理解并打分，再把理解 1:1 还原为可编辑的 DXF。"
        action={
          <span className="header-chip">
            <span className="live-dot" />
            OpenRouter · {models.length || "—"} 个模型
          </span>
        }
      />
      <nav className="vision-steps" aria-label="实验步骤">
        {steps.map((s) => (
          <button
            key={s.step}
            type="button"
            className={state.step === s.step ? "active" : ""}
            aria-current={state.step === s.step ? "step" : undefined}
            disabled={!s.enabled}
            onClick={() => go(s.step)}
          >
            {s.label}
          </button>
        ))}
      </nav>
      <ErrorMessage message={error} />
      {state.step === 1 && (
        <StepSelect
          samples={samples}
          image={state.image}
          imageName={state.imageName}
          scorable={state.scorable}
          apiKey={apiKey}
          remember={remember}
          onSelect={(image, imageName, scorable) => {
            resetJobs();
            dispatch({ type: "selectImage", image, imageName, scorable });
          }}
          onKey={(key) => changeKey(key)}
          onRemember={(keep) => changeKey(apiKey, keep)}
          onNext={() => go(2)}
        />
      )}
      {state.step === 2 && (
        <StepUnderstand
          models={models}
          selected={selected}
          runs={state.runs}
          scorable={state.scorable && state.image?.kind === "sample"}
          consensus={state.consensus}
          error={actionError}
          onToggle={(id) =>
            setSelected((cur) =>
              cur.includes(id) ? cur.filter((m) => m !== id) : [...cur, id],
            )
          }
          onToggleAll={(all) => setSelected(all ? models.map((m) => m.id) : [])}
          onStart={startAll}
          onRetry={retry}
          onRetryScore={retryScore}
          onConsensus={() => void consensus()}
          onExport={exportJson}
          onImport={(f) => void importJson(f)}
          onNext={() => go(3)}
        />
      )}
      {state.step === 3 && (
        <StepRedraw
          models={models}
          runs={state.runs}
          generations={state.generations}
          image={state.image}
          imageName={state.imageName}
          error={actionError}
          onGenerate={(s, g) => void generate(s, g)}
        />
      )}
      <div className="bottom-tip">
        <span>实验手记</span>
        模型理解、打分与还原都直接调用你的 OpenRouter
        Key；任务结果只在服务器内存中保留 2 小时。
      </div>
    </>
  );
}
