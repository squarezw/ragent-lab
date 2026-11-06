import { useEffect, useReducer, useRef, useState } from "react";
import { ErrorMessage, PageHeader } from "../components/UI";
import {
  listBuiltin,
  listModels,
  listSamples,
  pollJob,
  submit,
  uploadImage,
  type SubmitKind,
} from "../vision/client";
import {
  DEFAULT_JUDGE,
  exportState,
  importState,
  initialState,
  judgeStorage,
  reducer,
} from "../vision/state";
import {
  buildOptions,
  useModelSettings,
  type BuiltinInfo,
} from "../vision/modelSettings";
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

const NO_MODELS = "还没有可用的模型：请先到「模型设置」填写 Key、访问码，或添加模型。";
const NOT_CONFIGURED = "这个模型已不在「模型设置」中启用，请先启用或改选其他模型。";
const isAbort = (e: unknown) => (e as Error)?.name === "AbortError";
const isImageGone = (e: unknown) =>
  ((e as Error)?.message ?? "").includes("图片已过期");

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
  // Models the visitor unticked; everything else that is ready starts selected.
  const [deselected, setDeselected] = useState<string[]>([]);
  const [settings] = useModelSettings();
  const [builtin, setBuiltin] = useState<BuiltinInfo | null>(null);
  const [judgePick, setJudgePick] = useState(() => judgeStorage.load());
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  // Every in-flight poll listens to this controller; replaced on re-select / import, aborted on unmount.
  const ctrl = useRef(new AbortController());
  // The uploaded file, kept so it can be re-sent if the server no longer has the image.
  const uploadFile = useRef<File | undefined>(undefined);
  const imageNow = useRef<ImageRef | null>(null);
  imageNow.current = state.image;
  const reupload = useRef<{ from: string; promise: Promise<ImageRef> }>(
    undefined,
  );

  useEffect(() => {
    ctrl.current = new AbortController();
    return () => ctrl.current.abort();
  }, []);
  useEffect(() => {
    listModels()
      .then(setModels)
      .catch((e) => setError(e.message));
    listBuiltin()
      .then(setBuiltin)
      .catch(() => setBuiltin(null));
    listSamples()
      .then(setSamples)
      .catch((e) => setError(e.message));
  }, []);

  function resetJobs() {
    ctrl.current.abort();
    ctrl.current = new AbortController();
    setActionError("");
  }
  // Everything the settings page enabled; only "ready" ones (key / code filled) can run.
  const options = buildOptions(settings, models, builtin);
  const usable = options.filter((o) => o.ready);
  const pending = options.filter((o) => !o.ready);
  // The step components take VisionModel-shaped entries; the option key is the id.
  const available: VisionModel[] = usable.map((o) => ({
    id: o.key,
    name: o.name,
    family: o.source,
    notes: "",
    params: { max_tokens: null, temperature: null, reasoning: null },
    price: null,
    priceLabel: `${o.badge} · ${o.priceLabel}`,
  }));
  const selected = available
    .map((m) => m.id)
    .filter((id) => !deselected.includes(id));
  const refOf = (key: string) => options.find((o) => o.key === key)?.ref;
  const access = { access_code: settings.accessCode };
  // A remembered judge may no longer be usable: fall back to the default, then the first model.
  const listed = (id: string) => available.some((m) => m.id === id);
  const judge = listed(judgePick)
    ? judgePick
    : listed(`catalog:${judgePick}`) // saved before the settings page existed
      ? `catalog:${judgePick}`
      : listed(DEFAULT_JUDGE) || !available.length
        ? DEFAULT_JUDGE
        : available[0].id;
  function changeJudge(model: string) {
    setJudgePick(model);
    judgeStorage.save(model);
  }
  function requireModels(): boolean {
    if (usable.length) return true;
    setActionError(NO_MODELS);
    return false;
  }
  // The request reference for a model key, or an error if it is not configured any more.
  function refFor(key: string) {
    const ref = refOf(key);
    if (!ref || !usable.some((o) => o.key === key)) throw new Error(NOT_CONFIGURED);
    return ref;
  }

  // Run a job that needs the drawing; if the server lost the upload, re-upload once and retry.
  async function withImage<T>(
    image: ImageRef,
    run: (image: ImageRef) => Promise<T>,
  ): Promise<T> {
    try {
      return await run(image);
    } catch (e) {
      const file = uploadFile.current;
      if (image.kind !== "upload" || !file || !isImageGone(e)) throw e;
      // Jobs running side by side share one re-upload of the same lost image.
      const current = imageNow.current;
      if (current && current.id !== image.id) return run(current);
      let pending = reupload.current;
      if (!pending || pending.from !== image.id) {
        const promise = uploadImage(file).then((out) => {
          const next: ImageRef = { kind: "upload", id: out.image_id };
          imageNow.current = next;
          dispatch({ type: "imageReplaced", image: next });
          return next;
        });
        pending = { from: image.id, promise };
        reupload.current = pending;
      }
      return run(await pending.promise);
    }
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
        { sample_id: image.id, understanding, judge: refFor(judge), ...access },
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
      options.find((o) => o.key === model)?.name ??
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
      const result = await withImage(image, (img) =>
        runJob<UnderstandResult>(
          "understand",
          { image: img, model: refFor(model), ...access },
          signal,
        ),
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
    if (!requireModels()) return;
    const runnable = selected;
    if (!runnable.length) return setActionError("请至少选择一个模型。");
    dispatch({ type: "consensusUpdate", consensus: undefined });
    runnable.forEach((m) => void understand(m));
  }

  function retry(model: string) {
    if (requireModels()) void understand(model);
  }
  function retryScore(model: string) {
    const u = state.runs[model]?.understanding;
    if (u && state.image && requireModels()) void score(model, u, state.image);
  }

  async function consensus() {
    if (!requireModels()) return;
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
        { results, judge: refFor(judge), ...access },
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
    if (!understanding || !image || !requireModels()) return;
    const signal = ctrl.current.signal;
    const id = `gen-${Date.now()}-${++generationSeq}`;
    dispatch({
      type: "generationAdd",
      entry: { id, sourceModel, generator, status: "running" },
    });
    try {
      const r = await withImage(image, (img) =>
        runJob<GenerateResult>(
          "generate",
          { image: img, model: refFor(generator), understanding, ...access },
          signal,
        ),
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
          <a className="header-chip" href="#models">
            <span className="live-dot" />
            {available.length || "—"} 个模型可用 · 模型设置
          </a>
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
          models={usable.map((o) => ({ name: o.name, badge: o.badge }))}
          pending={pending.map((o) => `${o.name}：${o.problem ?? "未就绪"}`)}
          onSelect={(image, imageName, scorable, file) => {
            uploadFile.current = image.kind === "upload" ? file : undefined;
            resetJobs();
            dispatch({ type: "selectImage", image, imageName, scorable });
          }}
          onNext={() => go(2)}
        />
      )}
      {state.step === 2 && (
        <StepUnderstand
          models={available}
          selected={selected}
          runs={state.runs}
          scorable={state.scorable && state.image?.kind === "sample"}
          consensus={state.consensus}
          judge={judge}
          onJudge={changeJudge}
          error={actionError}
          onToggle={(id) =>
            setDeselected((cur) =>
              cur.includes(id) ? cur.filter((m) => m !== id) : [...cur, id],
            )
          }
          onToggleAll={(all) =>
            setDeselected(all ? [] : available.map((m) => m.id))
          }
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
          models={available}
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
        模型理解、打分与还原按「模型设置」调用对应平台，Key 只随请求发送、服务器不保存；任务结果只在服务器内存中保留 2 小时，上传的图纸在服务器上保留 7 天后自动删除。
      </div>
    </>
  );
}
