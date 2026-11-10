import { useEffect, useState } from "react";
import { AlertTriangle, ChevronDown, FileDown, PenTool, Square } from "lucide-react";
import { EmptyState, ErrorMessage, Panel, RunButton } from "../components/UI";
import JsonViewer from "../components/JsonViewer";
import { exportFile, imageUrl } from "./client";
import { formatElapsed, StatusIcon } from "./ModelStatusCard";
import type { GenerationEntry, RunEntry } from "./state";
import type { DimWarning, ImageRef, VisionModel } from "./types";

export const svgDataUrl = (svg: string) =>
  `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;

// Unknown dimension_ref → expected = actual = 0: only the message is meaningful.
export function warningDetail(w: DimWarning): string {
  if (w.expected === 0 && w.actual === 0) return "";
  return `预期 ${w.expected} mm · 实际 ${w.actual} mm`;
}

const fileStem = (name: string) =>
  (
    name.replace(/\.[^.]+$/, "").replace(/[^\w一-鿿-]+/g, "-") || "drawing"
  ).slice(0, 80);

function GenerationCard({
  index,
  entry,
  image,
  imageName,
  names,
  onCancel,
}: {
  index: number;
  entry: GenerationEntry;
  image: ImageRef | null;
  imageName: string;
  names: Record<string, string>;
  onCancel: () => void;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<"" | "pdf" | "dxf">("");
  // Warnings stay folded behind a small button until the visitor asks for them.
  const [showNotes, setShowNotes] = useState(false);
  const warnings = entry.warnings ?? [];
  const skipped = entry.skipped?.length ?? 0;
  const notes = warnings.length + (skipped > 0 ? 1 : 0);
  const notesId = `notes-${entry.id}`;
  async function download(format: "pdf" | "dxf") {
    if (!entry.geometry) return;
    setBusy(format);
    setError("");
    try {
      const generator = entry.generator.split("/").pop() ?? "model";
      await exportFile(
        entry.geometry,
        format,
        `${fileStem(imageName)}-${fileStem(generator)}`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  const pending = entry.status === "queued" || entry.status === "running";
  return (
    <article className={`vision-generation is-${entry.status}`}>
      <header>
        <div>
          <span className="vision-card-id">
            REDRAW {String(index + 1).padStart(2, "0")}
          </span>
          <h3>
            {names[entry.sourceModel] ?? entry.sourceModel}
            <i>→</i>
            {names[entry.generator] ?? entry.generator}
          </h3>
        </div>
        <span className={`vision-status-tag is-${entry.status}`}>
          <StatusIcon status={entry.status} />
          {pending
            ? "生成中"
            : entry.status === "done"
              ? "已完成"
              : entry.status === "cancelled"
                ? "已取消"
                : "失败"}
          {!pending && entry.meta && <small>{formatElapsed(entry.meta)}</small>}
        </span>
      </header>
      {entry.status === "failed" && (
        <ErrorMessage message={entry.error ?? "生成失败"} />
      )}
      {pending && (
        <p className="field-help">
          模型正在按理解结果绘制几何，通常需要 1–3 分钟。
        </p>
      )}
      {pending && (
        <button type="button" className="button secondary vision-cancel" onClick={onCancel}>
          <Square size={11} />
          取消
        </button>
      )}
      {entry.status === "done" && entry.svg && (
        <>
          <div className="vision-compare">
            <figure>
              <figcaption>原图</figcaption>
              {image && (
                <img src={imageUrl(image)} alt={`原图：${imageName}`} />
              )}
            </figure>
            <figure aria-label="还原预览">
              <figcaption>还原</figcaption>
              <img src={svgDataUrl(entry.svg)} alt="模型还原的图纸" />
            </figure>
          </div>
          <ErrorMessage message={error} />
          <div className="vision-actions">
            <button
              type="button"
              className="button secondary"
              disabled={!!busy}
              onClick={() => download("pdf")}
            >
              <FileDown size={13} />
              下载 PDF
            </button>
            <button
              type="button"
              className="button secondary"
              disabled={!!busy}
              onClick={() => download("dxf")}
            >
              <FileDown size={13} />
              下载 DXF
            </button>
            {entry.geometry && (
              <JsonViewer
                title="还原几何清单"
                data={entry.geometry}
                filename="geometry"
                label="几何 JSON"
              />
            )}
            {notes > 0 && (
              <button
                type="button"
                className="text-button vision-notes-toggle"
                aria-expanded={showNotes}
                aria-controls={notesId}
                onClick={() => setShowNotes((v) => !v)}
              >
                <AlertTriangle size={12} />
                {notes} 条提示
                <ChevronDown size={12} className={showNotes ? "flip" : ""} />
              </button>
            )}
          </div>
          {notes > 0 && showNotes && (
            <ul className="vision-warnings" id={notesId}>
              {warnings.map((w, i) => (
                <li key={i}>
                  <AlertTriangle size={13} />
                  <span>{w.message}</span>
                  {warningDetail(w) && <small>{warningDetail(w)}</small>}
                </li>
              ))}
              {skipped > 0 && (
                <li>
                  <AlertTriangle size={13} />
                  <span>已跳过 {skipped} 个无法绘制的实体。</span>
                </li>
              )}
            </ul>
          )}
        </>
      )}
    </article>
  );
}

export default function StepRedraw({
  models,
  runs,
  generations,
  image,
  imageName,
  error,
  onGenerate,
  onCancel,
}: {
  models: VisionModel[];
  runs: Record<string, RunEntry>;
  generations: GenerationEntry[];
  image: ImageRef | null;
  imageName: string;
  error: string;
  onGenerate: (sourceModel: string, generator: string) => void;
  onCancel: (id: string) => void;
}) {
  const done = Object.values(runs).filter((r) => r.status === "done");
  // Default to the source model when the current gateway serves it, else the first one it does.
  const pickGenerator = (src: string) =>
    models.some((m) => m.id === src) ? src : (models[0]?.id ?? src);
  const [source, setSource] = useState(done[0]?.model ?? "");
  const [generator, setGenerator] = useState(() =>
    pickGenerator(done[0]?.model ?? ""),
  );
  const doneIds = done.map((r) => r.model).join("\n");
  useEffect(() => {
    const ids = doneIds ? doneIds.split("\n") : [];
    if (ids.length && !ids.includes(source)) {
      setSource(ids[0]);
      setGenerator(pickGenerator(ids[0]));
    }
  }, [doneIds, source]);
  const names: Record<string, string> = Object.fromEntries([
    ...models.map((m) => [m.id, m.name]),
    ...Object.values(runs).map((r) => [r.model, r.model_name]),
  ]);
  const generators =
    models.some((m) => m.id === generator) || !generator
      ? models
      : [
          ...models,
          { id: generator, name: names[generator] ?? generator } as VisionModel,
        ];
  return (
    <>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (source && generator) onGenerate(source, generator);
        }}
      >
        <Panel
          title="还原配置"
          meta={<span className="panel-index">03 / REDRAW</span>}
        >
          <div className="parameter-grid">
            <div>
              <label htmlFor="vision-source">理解来源</label>
              <select
                id="vision-source"
                value={source}
                onChange={(e) => {
                  setSource(e.target.value);
                  setGenerator(pickGenerator(e.target.value));
                }}
              >
                {done.map((r) => (
                  <option key={r.model} value={r.model}>
                    {r.model_name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="vision-generator">生成模型</label>
              <select
                id="vision-generator"
                value={generator}
                onChange={(e) => setGenerator(e.target.value)}
              >
                {generators.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <p className="field-help">
            生成模型读取原图和所选理解，输出以 mm
            为单位的几何清单；尺寸与标注不符时会列出警告。
          </p>
          <ErrorMessage message={error} />
          <RunButton busy={!source} label="生成还原图" />
        </Panel>
      </form>
      <section className="vision-section">
        {generations.length ? (
          <div className="vision-generation-list">
            {[...generations].reverse().map((g) => (
              <GenerationCard
                key={g.id}
                index={generations.indexOf(g)}
                entry={g}
                image={image}
                imageName={imageName}
                names={names}
                onCancel={() => onCancel(g.id)}
              />
            ))}
          </div>
        ) : (
          <Panel title="还原结果" className="result-panel">
            <EmptyState
              icon={<PenTool size={36} strokeWidth={1.3} />}
              title="还原图会出现在这里"
              description="选择一份理解结果，让模型把它画成可编辑的 DXF 图纸。"
            />
          </Panel>
        )}
      </section>
    </>
  );
}
