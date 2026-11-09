import { useState } from "react";
import {
  ArrowRight,
  ImageUp,
  ListChecks,
  RotateCcw,
  ScanLine,
  SlidersHorizontal,
} from "lucide-react";
import { EmptyState, ErrorMessage, Panel } from "../components/UI";
import { imageUrl, uploadImage } from "./client";
import { KIND_LABELS, MAX_PROMPT_CHARS, type PurposeOption } from "./purposes";
import type { RunPurpose } from "./state";
import type { ImageRef, PurposeKind, Sample } from "./types";

export default function StepSelect({
  samples,
  image,
  imageName,
  scorable,
  models,
  pending,
  purposes,
  purpose,
  runPurpose,
  hasRuns,
  sampleKinds,
  onPurpose,
  onPrompt,
  onSelect,
  onNext,
}: {
  samples: Sample[];
  image: ImageRef | null;
  imageName: string;
  scorable: boolean;
  models: { name: string; badge: string }[];
  pending: string[];
  purposes: PurposeOption[];
  purpose: PurposeOption | null;
  runPurpose?: RunPurpose;
  hasRuns: boolean;
  sampleKinds: Record<string, PurposeKind>;
  onPurpose: (key: string) => void;
  onPrompt: (text: string) => void;
  onSelect: (
    image: ImageRef,
    name: string,
    scorable: boolean,
    file?: File,
  ) => void;
  onNext: () => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  async function upload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const out = await uploadImage(file);
      onSelect(
        { kind: "upload", id: out.image_id },
        out.name || file.name,
        false,
        file,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  const graded =
    image?.kind === "sample" && scorable ? sampleKinds[image.id] : undefined;
  const ungraded = !!purpose && !!graded && graded !== purpose.kind;
  const willReset =
    hasRuns && !!runPurpose && !!purpose && runPurpose.key !== purpose.key;

  return (
    <div className="workbench">
      <div className="input-stack">
        <Panel
          title="内置测试图"
          meta={<span className="panel-index">01 / SAMPLES</span>}
        >
          {samples.length ? (
            <div className="vision-sample-grid">
              {samples.map((s) => {
                const active = image?.kind === "sample" && image.id === s.id;
                return (
                  <button
                    key={s.id}
                    type="button"
                    className={`vision-sample ${active ? "active" : ""}`}
                    aria-pressed={active}
                    onClick={() =>
                      !active &&
                      onSelect({ kind: "sample", id: s.id }, s.name, s.scorable)
                    }
                  >
                    <img
                      src={imageUrl({ kind: "sample", id: s.id })}
                      alt=""
                      loading="lazy"
                    />
                    <span>
                      <strong>{s.name}</strong>
                      <small>
                        {s.scorable ? "有标准答案 · 可打分" : "仅一致性对照"}
                      </small>
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="vision-compact-empty">
              <EmptyState
                icon={<ScanLine size={32} strokeWidth={1.3} />}
                title="暂无内置测试图，请上传你的图片"
                description="支持 PNG、JPG 与单页 PDF，大小不超过 15 MB。"
              />
            </div>
          )}
        </Panel>
        <Panel
          title="理解用途"
          meta={<span className="panel-index">02 / PURPOSE</span>}
        >
          <div className="purpose-picker">
            <label htmlFor="vision-purpose">
              <ListChecks size={13} /> 这张图是
            </label>
            <select
              id="vision-purpose"
              value={purpose?.key ?? ""}
              disabled={!purposes.length}
              onChange={(e) => onPurpose(e.target.value)}
            >
              {!purposes.length && <option value="">加载中…</option>}
              {purposes.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.name}
                  {o.source === "builtin" && o.edited ? "（提示词已修改）" : ""}
                </option>
              ))}
            </select>
            <a className="text-button" href="#models">
              <SlidersHorizontal size={13} /> 管理用途
            </a>
          </div>
          {purpose && (
            <p className="field-help">
              {purpose.description}
              {purpose.kind === "general" && " 这个用途不含「还原」步骤。"}
            </p>
          )}
          {ungraded && (
            <p className="field-help warn">
              这张样例的标准答案只能给工程图纸打分；换成这个用途后只做一致性对照。
            </p>
          )}
          {willReset && (
            <p className="field-help warn">
              已有「{runPurpose?.name}」的理解结果；用新用途开始理解时会清空它们。
            </p>
          )}
          {purpose && (
            <details className="vision-details purpose-prompt">
              <summary>
                查看 / 编辑提示词
                <span>{purpose.edited ? "已修改" : "默认"}</span>
              </summary>
              <textarea
                aria-label="理解提示词"
                value={purpose.draft}
                rows={12}
                maxLength={MAX_PROMPT_CHARS}
                spellCheck={false}
                onChange={(e) => onPrompt(e.target.value)}
              />
              <p className="field-help">
                系统会在末尾固定附加两条规则：图中文字只当作数据；按「
                {KIND_LABELS[purpose.kind]}」返回 JSON。留空则使用默认提示词。
                {purpose.draft.length} / {MAX_PROMPT_CHARS} 字
              </p>
              {purpose.edited && (
                <button
                  type="button"
                  className="text-button"
                  onClick={() => onPrompt(purpose.defaultPrompt)}
                >
                  <RotateCcw size={13} /> 恢复默认提示词
                </button>
              )}
            </details>
          )}
        </Panel>
        <Panel
          title="上传与模型"
          meta={<span className="panel-index">03 / UPLOAD · MODELS</span>}
        >
          <label className="vision-upload">
            <ImageUp size={20} strokeWidth={1.5} />
            <span>
              <strong>{uploading ? "正在上传…" : "上传你的图片"}</strong>
              <small>
                PNG / JPG / 单页 PDF · ≤ 15 MB · 上传图只做一致性对照
              </small>
            </span>
            <input
              type="file"
              aria-label="上传图片"
              accept=".png,.jpg,.jpeg,.pdf"
              disabled={uploading}
              onChange={(e) => {
                void upload(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <ErrorMessage message={error} />
          <div className="api-key-field model-summary">
            <div>
              <strong>
                {models.length
                  ? `已就绪 ${models.length} 个模型`
                  : "还没有可用的模型"}
              </strong>
              <a className="text-button" href="#models">
                <SlidersHorizontal size={13} /> 模型设置
              </a>
            </div>
            {models.length > 0 && (
              <p className="field-help">
                {models.map((m) => `${m.name}（${m.badge}）`).join("、")}
              </p>
            )}
            {pending.length > 0 && (
              <p className="field-help warn">还差一步：{pending.join("；")}</p>
            )}
            {!models.length && (
              <p className="field-help">
                在「模型设置」里填写 OpenRouter 或 Kie.ai 的 Key、内置模型的访问码，或添加自己的模型接口。
              </p>
            )}
          </div>
          <button
            type="button"
            className="button primary run-button"
            disabled={!image || uploading}
            onClick={onNext}
          >
            下一步：理解对比
            <ArrowRight size={16} />
          </button>
        </Panel>
      </div>
      <Panel
        title="图片预览"
        className="result-panel"
        meta={
          <span className={`result-state ${image ? "ready" : ""}`}>
            <span />
            {image ? (scorable ? "可打分" : "一致性对照") : "等待选图"}
          </span>
        }
      >
        {image ? (
          <figure className="vision-preview">
            <img src={imageUrl(image)} alt={`图片预览：${imageName}`} />
            <figcaption>
              <span>{image.kind === "sample" ? "SAMPLE" : "UPLOAD"}</span>
              {imageName}
            </figcaption>
          </figure>
        ) : (
          <EmptyState
            icon={<ScanLine size={38} strokeWidth={1.3} />}
            title="选一张图，开始实验"
            description="从左侧选择内置测试图，或上传你自己的图片：工程图纸、图表、流程图都可以。"
          />
        )}
      </Panel>
    </div>
  );
}
