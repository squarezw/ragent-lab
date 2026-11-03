import { useEffect, useState } from "react";
import {
  AudioLines,
  GitCompareArrows,
  KeyRound,
  Plus,
  RotateCcw,
  Trash2,
} from "lucide-react";
import { api } from "../api";
import type { EmbeddingResult, Model } from "../types";
import { addComparison } from "../experiment-state";
import {
  EmptyState,
  ErrorMessage,
  ExternalLink,
  Metrics,
  PageHeader,
  Panel,
  RunButton,
} from "../components/UI";
const EXAMPLE =
  "检索增强生成通过引入外部知识，帮助大模型生成更准确的回答。\n文本分块将长文档拆分成适合检索的知识片段。\n向量模型将语义相近的文本映射到相邻的向量空间。\n工程图纸包含零件尺寸、公差和材料等技术要求。\n今天的天气晴朗，适合去公园散步。";
export default function Embedding() {
  const [models, setModels] = useState<Model[]>([]);
  const [selected, setSelected] = useState("aliyun_text_embedding_v2");
  const [corpus, setCorpus] = useState(EXAMPLE);
  const [query, setQuery] = useState("如何让大模型使用企业内部知识？");
  const [key, setKey] = useState("");
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [runs, setRuns] = useState<EmbeddingResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    api<Model[]>("/embedding/models")
      .then(setModels)
      .catch((e) => setError(e.message));
  }, []);
  const model = models.find((m) => m.id === selected);
  const texts = corpus
    .split("\n")
    .map((t) => t.trim())
    .filter(Boolean);
  function invalidate() {
    setResult(null);
    setError("");
  }
  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await api<EmbeddingResult>("/embedding/run", {
          model: selected,
          texts,
          query,
          api_key: key,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function compare() {
    if (!result) return;
    try {
      setRuns(addComparison(runs, result));
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="EXPERIMENT 02 / SEMANTIC EMBEDDINGS"
        title="让语义距离，变得可见。"
        description="用同一组文本和查询，观察不同模型如何理解相关性。"
        action={
          <ExternalLink href="https://huggingface.co/spaces/mteb/leaderboard">
            MTEB 参考评测
          </ExternalLink>
        }
      />
      <form className="workbench" onSubmit={run}>
        <div className="input-stack">
          <Panel
            title="实验数据"
            meta={<span className="panel-index">01 / DATASET</span>}
          >
            <div className="field-toolbar">
              <label htmlFor="embedding-texts">
                候选文本 <span>每行一条</span>
              </label>
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => {
                  setCorpus(EXAMPLE);
                  invalidate();
                }}
              >
                <RotateCcw size={13} />
                示例
              </button>
            </div>
            <textarea
              id="embedding-texts"
              className="embedding-input"
              value={corpus}
              maxLength={32000}
              disabled={busy}
              onChange={(e) => {
                setCorpus(e.target.value);
                invalidate();
              }}
              required
            />
            <div className="input-footer">
              <span>最多 32 条，每条不超过 4,000 字符</span>
              <span>{texts.length} 条文本</span>
            </div>
            <div className="query-field">
              <label htmlFor="query">
                查询文本 <span className="optional">可选</span>
              </label>
              <textarea
                id="query"
                rows={2}
                value={query}
                maxLength={4000}
                placeholder="输入一个问题，观察哪些文本最相关…"
                disabled={busy}
                onChange={(e) => {
                  setQuery(e.target.value);
                  invalidate();
                }}
              />
            </div>
          </Panel>
          <Panel
            title="模型配置"
            meta={<span className="panel-index">02 / MODEL</span>}
          >
            <label htmlFor="model">Embedding 模型</label>
            <select
              id="model"
              value={selected}
              disabled={busy || !models.length}
              onChange={(e) => {
                setSelected(e.target.value);
                setKey("");
                invalidate();
              }}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.enabled}>
                  {m.name.replace("SentenceTransformer ", "")}
                  {!m.enabled ? " · 需额外算力" : ""}
                </option>
              ))}
            </select>
            <p className="field-help">{model?.description}</p>
            <div className="model-spec">
              <span>
                {model?.provider === "local" ? "本地推理" : "云端 API"}
              </span>
              <span>{model?.dimension ?? "—"} 维</span>
              <span>文本嵌入</span>
            </div>
            {model?.provider !== "local" ? (
              <div className="api-key-field">
                <label htmlFor="api-key">
                  <KeyRound size={14} />
                  API Key
                </label>
                <input
                  id="api-key"
                  type="password"
                  autoComplete="off"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="输入你的模型服务 API Key"
                  disabled={busy}
                  required
                />
                <p className="field-help">
                  仅用于本次请求，不写入浏览器存储或实验记录。
                </p>
              </div>
            ) : (
              <div className="callout field-help">
                首次使用会下载模型，可能需要几分钟。大模型按服务器算力逐步开放。
              </div>
            )}
            <ErrorMessage message={error} />
            <RunButton busy={busy || !models.length} label="生成向量并分析" />
          </Panel>
        </div>
        <Panel
          title="相似度分析"
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
                  { label: "向量维度", value: result.dimension, unit: "维" },
                  { label: "候选文本", value: result.count, unit: "条" },
                  {
                    label: "生成耗时",
                    value: (result.elapsed_ms / 1000).toFixed(2),
                    unit: "s",
                  },
                ]}
              />
              <div className="result-query">
                <span>QUERY</span>
                <p>
                  {result.query || "未设置查询。填写查询后可查看相似度排名。"}
                </p>
              </div>
              <div className="ranking-list">
                {result.rankings.map((item, i) => (
                  <article
                    className={`ranking-item ${i === 0 ? "top-match" : ""}`}
                    key={item.index}
                  >
                    <div className="ranking-top">
                      <span>
                        <b>{String(i + 1).padStart(2, "0")}</b>
                        {i === 0 ? "最相关" : "候选匹配"}
                      </span>
                      <strong>{item.score.toFixed(4)}</strong>
                    </div>
                    <p>{item.text}</p>
                    <div className="score-track">
                      <span
                        style={{ width: `${Math.max(0, item.score) * 100}%` }}
                      />
                    </div>
                  </article>
                ))}
              </div>
              <div className="result-bottom">
                <span>余弦相似度 · 越接近 1 越相关</span>
                <button
                  type="button"
                  className="button secondary"
                  onClick={compare}
                >
                  <Plus size={15} />
                  加入模型对比
                </button>
              </div>
            </>
          ) : (
            <EmptyState
              icon={<AudioLines size={40} strokeWidth={1.3} />}
              title={busy ? "模型正在理解这些文本…" : "看看模型眼中的语义关系"}
              description={
                busy
                  ? "向量与查询将一次生成，完成后即可查看排名。"
                  : "输入候选文本与查询，选择模型，发现语义最接近的内容。"
              }
            />
          )}
        </Panel>
      </form>
      <section className="comparison-section">
        <div className="section-title">
          <div>
            <span className="eyebrow">COMPARE & DISCOVER</span>
            <h2>
              <GitCompareArrows size={21} />
              模型对比 <span>{String(runs.length).padStart(2, "0")}</span>
            </h2>
          </div>
          {runs.length > 0 && (
            <button className="text-button" onClick={() => setRuns([])}>
              <Trash2 size={14} />
              清空对比
            </button>
          )}
        </div>
        {runs.length ? (
          <div className="comparison-grid">
            {runs.map((run) => (
              <article className="comparison-card" key={run.model}>
                <div>
                  <span>{run.dimension} DIMENSIONS</span>
                  <button
                    className="icon-button"
                    aria-label={`移除 ${run.model_name}`}
                    onClick={() =>
                      setRuns(runs.filter((r) => r.model !== run.model))
                    }
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
                <h3>{run.model_name.replace("SentenceTransformer ", "")}</h3>
                <p className="comparison-query">
                  查询：{run.query || "未设置"}
                </p>
                {run.rankings.slice(0, 3).map((r, i) => (
                  <div className="comparison-match" key={r.index}>
                    <span>#{i + 1}</span>
                    <p>{r.text}</p>
                    <strong>{r.score.toFixed(3)}</strong>
                  </div>
                ))}
                <div className="comparison-foot">
                  {run.count} 条相同候选文本
                  <span>{(run.elapsed_ms / 1000).toFixed(2)}s</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="comparison-empty">
            <GitCompareArrows size={23} />
            <div>
              <strong>不同的模型，同一把尺子。</strong>
              <p>
                完成实验后点击“加入模型对比”，所有对比将使用相同的候选文本和查询。
              </p>
            </div>
          </div>
        )}
        <p className="comparison-disclaimer">
          相似度分数用于观察模型行为，不代表跨模型的绝对质量。请结合标注数据和实际检索效果评估。
        </p>
      </section>
    </>
  );
}
