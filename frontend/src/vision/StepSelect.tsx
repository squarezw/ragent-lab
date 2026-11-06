import { useState } from "react";
import { ArrowRight, ImageUp, KeyRound, ScanLine } from "lucide-react";
import {
  EmptyState,
  ErrorMessage,
  ExternalLink,
  Panel,
} from "../components/UI";
import { imageUrl, uploadImage } from "./client";
import { GATEWAY_NAMES, type Gateway, type ImageRef, type Sample } from "./types";

const GATEWAY_INFO: Record<
  Gateway,
  { placeholder: string; keyUrl: string; note: string }
> = {
  openrouter: {
    placeholder: "sk-or-…",
    keyUrl: "https://openrouter.ai/keys",
    note: "9 个模型",
  },
  kie: {
    placeholder: "Kie.ai API Key",
    keyUrl: "https://kie.ai/api-key",
    note: "Claude、GPT、Gemini、Kimi",
  },
};

export default function StepSelect({
  samples,
  image,
  imageName,
  scorable,
  gateway,
  apiKey,
  remember,
  onSelect,
  onGateway,
  onKey,
  onRemember,
  onNext,
}: {
  samples: Sample[];
  image: ImageRef | null;
  imageName: string;
  scorable: boolean;
  gateway: Gateway;
  onGateway: (gateway: Gateway) => void;
  apiKey: string;
  remember: boolean;
  onSelect: (image: ImageRef, name: string, scorable: boolean) => void;
  onKey: (key: string) => void;
  onRemember: (remember: boolean) => void;
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
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

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
                title="暂无内置测试图，请上传你的图纸"
                description="支持 PNG、JPG 与单页 PDF，大小不超过 15 MB。"
              />
            </div>
          )}
        </Panel>
        <Panel
          title="上传与密钥"
          meta={<span className="panel-index">02 / UPLOAD · KEY</span>}
        >
          <label className="vision-upload">
            <ImageUp size={20} strokeWidth={1.5} />
            <span>
              <strong>{uploading ? "正在上传…" : "上传你的图纸"}</strong>
              <small>
                PNG / JPG / 单页 PDF · ≤ 15 MB · 上传图只做一致性对照
              </small>
            </span>
            <input
              type="file"
              aria-label="上传图纸"
              accept=".png,.jpg,.jpeg,.pdf"
              disabled={uploading}
              onChange={(e) => {
                void upload(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <ErrorMessage message={error} />
          <div className="api-key-field">
            <div
              className="vision-gateway"
              role="radiogroup"
              aria-label="模型通道"
            >
              {(Object.keys(GATEWAY_NAMES) as Gateway[]).map((g) => (
                <button
                  key={g}
                  type="button"
                  role="radio"
                  aria-checked={gateway === g}
                  className={gateway === g ? "active" : ""}
                  onClick={() => onGateway(g)}
                >
                  <strong>{GATEWAY_NAMES[g]}</strong>
                  <small>{GATEWAY_INFO[g].note}</small>
                </button>
              ))}
            </div>
            <label htmlFor="vision-key">
              <KeyRound size={13} />
              {GATEWAY_NAMES[gateway]} Key
            </label>
            <input
              id="vision-key"
              aria-label={`${GATEWAY_NAMES[gateway]} Key`}
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder={GATEWAY_INFO[gateway].placeholder}
              value={apiKey}
              onChange={(e) => onKey(e.target.value)}
            />
            <label className="vision-check">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => onRemember(e.target.checked)}
              />
              在本浏览器记住
            </label>
            <p className="field-help">
              Key 只随请求发送给 {GATEWAY_NAMES[gateway]}
              ，不会保存在服务器或导出文件中。
              <ExternalLink href={GATEWAY_INFO[gateway].keyUrl}>
                获取 Key
              </ExternalLink>
            </p>
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
        title="图纸预览"
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
            <img src={imageUrl(image)} alt={`图纸预览：${imageName}`} />
            <figcaption>
              <span>{image.kind === "sample" ? "SAMPLE" : "UPLOAD"}</span>
              {imageName}
            </figcaption>
          </figure>
        ) : (
          <EmptyState
            icon={<ScanLine size={38} strokeWidth={1.3} />}
            title="选一张图纸，开始实验"
            description="从左侧选择内置测试图，或上传你自己的工程图纸。"
          />
        )}
      </Panel>
    </div>
  );
}
