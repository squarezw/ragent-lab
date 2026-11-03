import { useEffect, useState } from "react";
import { Blocks, Copy, FileText, RotateCcw } from "lucide-react";
import { api } from "../api";
import type { ChunkResult, Strategy } from "../types";
import {
  EmptyState,
  ErrorMessage,
  Metrics,
  PageHeader,
  Panel,
  RunButton,
} from "../components/UI";
const EXAMPLE =
  "检索增强生成：让大模型连接你的知识\n\n大语言模型拥有丰富的通用知识，但对于企业内部的业务流程、产品手册和最新信息，往往缺少直接了解。检索增强生成（RAG）通过引入外部知识，让回答更贴近真实业务。\n\n第一步是文本分块。把长文档拆成大小适中的片段，既要保留足够的上下文，也要避免无关内容干扰检索。不同文档结构，需要不同的分块策略。\n\n第二步是向量化。Embedding 模型将每个文本片段映射到向量空间，让语义相近的内容彼此靠近。当用户提出问题时，系统就可以寻找最相关的知识片段。\n\n最后，将检索到的知识和用户的问题一同交给大模型，生成有据可查的回答。分块质量与向量模型的选择，共同影响最终的检索效果。";
export default function Chunking() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selected, setSelected] = useState("递归分块");
  const [text, setText] = useState(EXAMPLE);
  const [params, setParams] = useState<Record<string, string | number>>({});
  const [result, setResult] = useState<ChunkResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    api<Strategy[]>("/chunking/strategies")
      .then(setStrategies)
      .catch((e) => setError(e.message));
  }, []);
  const strategy = strategies.find((s) => s.id === selected);
  function invalidate() {
    setResult(null);
    setCopied(false);
    setError("");
  }
  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await api<ChunkResult>("/chunking/run", {
          text,
          strategy: selected,
          params,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(
        result!.chunks.map((c) => c.content).join("\n\n"),
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("无法访问剪贴板，请手动选择文本复制。");
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="EXPERIMENT 01 / TEXT CHUNKING"
        title="找到文本的自然边界。"
        description="选择一种策略，调整分块参数，观察知识如何被组织。"
      />
      <form onSubmit={run} className="workbench">
        <div className="input-stack">
          <Panel
            title="输入文本"
            meta={<span className="panel-index">01 / INPUT</span>}
          >
            <div className="field-toolbar">
              <span>
                <FileText size={14} />
                待处理文档
              </span>
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => {
                  setText(EXAMPLE);
                  invalidate();
                }}
              >
                <RotateCcw size={13} />
                载入示例
              </button>
            </div>
            <label className="sr-only" htmlFor="chunk-text">
              待分块文本
            </label>
            <textarea
              id="chunk-text"
              className="document-input"
              value={text}
              disabled={busy}
              maxLength={50000}
              onChange={(e) => {
                setText(e.target.value);
                invalidate();
              }}
            />
            <div className="input-footer">
              <span>纯文本 · 中文 / English</span>
              <span>{text.length.toLocaleString()} / 50,000 字符</span>
            </div>
          </Panel>
          <Panel
            title="实验配置"
            meta={<span className="panel-index">02 / CONFIGURE</span>}
          >
            <label htmlFor="strategy">分块策略</label>
            <select
              id="strategy"
              disabled={busy || !strategies.length}
              value={selected}
              onChange={(e) => {
                setSelected(e.target.value);
                setParams({});
                invalidate();
              }}
            >
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <p className="field-help">{strategy?.description}</p>
            <div className="parameter-grid">
              {strategy?.params.map((p) => (
                <div key={p.name}>
                  <label htmlFor={`chunk-${p.name}`}>{p.label}</label>
                  <input
                    id={`chunk-${p.name}`}
                    disabled={busy}
                    type={p.type === "str" ? "text" : "number"}
                    min={p.min}
                    max={p.max}
                    step={p.type === "float" ? 0.01 : 1}
                    value={params[p.name] ?? p.default}
                    onChange={(e) => {
                      setParams({
                        ...params,
                        [p.name]:
                          p.type === "str"
                            ? e.target.value
                            : e.target.value === ""
                              ? ""
                              : Number(e.target.value),
                      });
                      invalidate();
                    }}
                    required
                  />
                </div>
              ))}
            </div>
            {["Agent分块", "多模态分块", "语境丰富分块"].includes(selected) && (
              <p className="field-help callout">
                此策略为规则演示：不调用大模型，不处理图片。
              </p>
            )}
            <ErrorMessage message={error} />
            <RunButton busy={busy || !strategies.length} />
          </Panel>
        </div>
        <Panel
          title="分块结果"
          meta={
            <span className={`result-state ${result ? "ready" : ""}`}>
              <span />
              {busy ? "运行中" : result ? "已完成" : "等待运行"}
            </span>
          }
          className="result-panel"
        >
          {result ? (
            <>
              <Metrics
                items={[
                  { label: "分块数量", value: result.stats.count, unit: "块" },
                  {
                    label: "平均长度",
                    value: result.stats.average_chars,
                    unit: "字符",
                  },
                  {
                    label: "处理耗时",
                    value: result.stats.elapsed_ms,
                    unit: "ms",
                  },
                ]}
              />
              <div className="result-toolbar">
                <span>
                  {result.strategy} <i>·</i>{" "}
                  {result.stats.total_chars.toLocaleString()} 字符
                </span>
                <button type="button" className="text-button" onClick={copy}>
                  <Copy size={14} />
                  {copied ? "已复制" : "复制全部"}
                </button>
              </div>
              <div className="chunk-list">
                {result.chunks.length ? (
                  result.chunks.map((chunk, i) => (
                    <article className="chunk-card" key={i}>
                      <div>
                        <span className="chunk-number">
                          CHUNK {String(i + 1).padStart(2, "0")}
                        </span>
                        <span>
                          {chunk.content.length} 字符
                          {chunk.type ? ` · ${chunk.type}` : ""}
                        </span>
                      </div>
                      {chunk.title && <h3>{chunk.title}</h3>}
                      <p>{chunk.content}</p>
                    </article>
                  ))
                ) : (
                  <div className="inline-empty">
                    没有匹配的分块，试试调整关键词。
                  </div>
                )}
              </div>
            </>
          ) : (
            <EmptyState
              icon={<Blocks size={38} strokeWidth={1.3} />}
              title={
                busy ? "正在寻找合适的文本边界…" : "你的分块结果，会出现在这里"
              }
              description={
                busy
                  ? "语义策略首次使用可能需要下载模型，请稍候。"
                  : "在左侧输入文本并运行实验，比较不同策略对内容结构的影响。"
              }
            />
          )}
        </Panel>
      </form>
      <div className="bottom-tip">
        <span>实验手记</span>
        分块没有唯一的最佳长度。保留语义完整性，再结合实际检索任务验证效果。
      </div>
    </>
  );
}
