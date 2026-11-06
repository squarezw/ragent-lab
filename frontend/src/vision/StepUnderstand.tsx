import { useRef } from "react";
import { Download, GitCompareArrows, Scale, Upload } from "lucide-react";
import { ErrorMessage, Panel, RunButton } from "../components/UI";
import ModelStatusCard from "./ModelStatusCard";
import type { ConsensusEntry, RunEntry } from "./state";
import type { ConsensusResult, Gateway, VisionModel } from "./types";

export function formatPrice(
  price: VisionModel["price"],
  gateway: Gateway = "openrouter",
): string {
  if (gateway === "kie") return "按 Kie.ai 积分计费";
  if (!price || price.prompt == null || price.completion == null)
    return "价格未知";
  const perM = (v: string) => {
    const n = Number(v) * 1e6;
    return Number.isFinite(n) ? `$${n.toFixed(2)}` : "?";
  };
  return `输入 ${perM(price.prompt)} · 输出 ${perM(price.completion)} / 百万 token`;
}

type ConsensusRow = {
  key: string;
  label: string;
  view: string;
  summary: string;
  models: string[];
  minority: boolean;
};

// An interpretation backed by fewer models than the group's best one is the minority;
// a singleton (only some models mention the item at all) is always a minority.
export function consensusRows(result: ConsensusResult): ConsensusRow[] {
  const rows: ConsensusRow[] = [];
  result.groups.forEach((g, gi) => {
    const top = Math.max(0, ...g.interpretations.map((i) => i.models.length));
    g.interpretations.forEach((i, ii) =>
      rows.push({
        key: `g${gi}-${ii}`,
        label: ii === 0 ? g.label : "",
        view: ii === 0 ? g.view : "",
        summary: i.summary,
        models: i.models,
        minority: g.interpretations.length > 1 && i.models.length < top,
      }),
    );
  });
  result.singletons.forEach((s, si) =>
    rows.push({
      key: `s${si}`,
      label: s.label,
      view: "仅少数模型提到",
      summary: s.summary,
      models: s.models,
      minority: true,
    }),
  );
  return rows;
}

export default function StepUnderstand({
  models,
  selected,
  runs,
  scorable,
  consensus,
  judge,
  onJudge,
  error,
  onToggle,
  onToggleAll,
  onStart,
  onRetry,
  onRetryScore,
  onConsensus,
  onExport,
  onImport,
  onNext,
}: {
  models: VisionModel[];
  selected: string[];
  runs: Record<string, RunEntry>;
  scorable: boolean;
  consensus?: ConsensusEntry;
  judge: string;
  onJudge: (model: string) => void;
  error: string;
  onToggle: (id: string) => void;
  onToggleAll: (all: boolean) => void;
  onStart: () => void;
  onRetry: (model: string) => void;
  onRetryScore: (model: string) => void;
  onConsensus: () => void;
  onExport: () => void;
  onImport: (file: File) => void;
  onNext: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const order = models.map((m) => m.id);
  const cards = Object.values(runs).sort(
    (a, b) =>
      (order.indexOf(a.model) + 1 || 99) - (order.indexOf(b.model) + 1 || 99),
  );
  const running = cards.some(
    (r) => r.status === "queued" || r.status === "running",
  );
  const done = cards.filter((r) => r.status === "done");
  const consensusBusy = consensus?.status === "running";
  return (
    <>
      <div className="vision-toolbar">
        <div className="vision-judge-note">
          <Scale size={14} />
          <label htmlFor="vision-judge">评判模型</label>
          <select
            id="vision-judge"
            value={judge}
            onChange={(e) => onJudge(e.target.value)}
          >
            {(models.length ? models : [{ id: judge, name: judge }]).map(
              (m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ),
            )}
          </select>
          <span>打分和一致性对照都用它，费用计入你的 Key</span>
        </div>
        <div>
          <button
            type="button"
            className="button secondary"
            onClick={onExport}
            disabled={!cards.length}
          >
            <Download size={13} />
            导出结果 JSON
          </button>
          <button
            type="button"
            className="button secondary"
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={13} />
            导入结果 JSON
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            hidden
            aria-label="选择结果 JSON 文件"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) onImport(file);
            }}
          />
        </div>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onStart();
        }}
      >
        <Panel
          title="选择模型"
          meta={
            <span className="vision-panel-meta">
              <button
                type="button"
                className="text-button"
                onClick={() => onToggleAll(selected.length !== models.length)}
              >
                {selected.length === models.length ? "全部取消" : "全选"}
              </button>
              <span className="panel-index">
                {String(selected.length).padStart(2, "0")} /{" "}
                {String(models.length).padStart(2, "0")} MODELS
              </span>
            </span>
          }
        >
          <div className="vision-model-list">
            {models.map((m) => (
              <label key={m.id} className="vision-model-option">
                <input
                  type="checkbox"
                  checked={selected.includes(m.id)}
                  onChange={() => onToggle(m.id)}
                />
                <span>
                  <strong>{m.name}</strong>
                  <small>{m.priceLabel ?? formatPrice(m.price)}</small>
                </span>
              </label>
            ))}
          </div>
          <ErrorMessage message={error} />
          <RunButton busy={running} label="开始理解" />
        </Panel>
      </form>
      {cards.length > 0 && (
        <section className="vision-section">
          <div className="section-title">
            <div>
              <span className="eyebrow">MODEL RUNS</span>
              <h2>
                理解结果 <span>{String(cards.length).padStart(2, "0")}</span>
              </h2>
            </div>
            {done.length > 0 && (
              <button type="button" className="text-button" onClick={onNext}>
                下一步：还原 →
              </button>
            )}
          </div>
          <div className="vision-model-grid">
            {cards.map((run) => (
              <ModelStatusCard
                key={run.model}
                run={run}
                scorable={scorable}
                onRetry={() => onRetry(run.model)}
                onRetryScore={() => onRetryScore(run.model)}
              />
            ))}
          </div>
        </section>
      )}
      {!scorable && cards.length > 0 && (
        <Panel
          title="一致性对照"
          className="vision-section"
          meta={<span className="panel-index">CONSENSUS</span>}
        >
          <p className="field-help">
            这张图没有标准答案：让判定模型对比各模型的理解，标出意见分歧和只有少数模型提到的内容。
          </p>
          <button
            type="button"
            className="button primary"
            disabled={done.length < 2 || consensusBusy || running}
            onClick={onConsensus}
          >
            <GitCompareArrows size={15} />
            {consensusBusy ? "对照生成中…" : "生成一致性对照"}
          </button>
          {done.length < 2 && (
            <p className="field-help">至少需要 2 个模型完成理解。</p>
          )}
          {consensus?.status === "failed" && (
            <ErrorMessage message={consensus.error ?? "生成失败"} />
          )}
          {consensus?.status === "done" && consensus.result && (
            <div className="vision-table-wrap">
              <table className="vision-table vision-consensus">
                <thead>
                  <tr>
                    <th>条目</th>
                    <th>视图</th>
                    <th>解释</th>
                    <th>模型</th>
                  </tr>
                </thead>
                <tbody>
                  {consensusRows(consensus.result).map((row) => (
                    <tr
                      key={row.key}
                      className={row.minority ? "minority" : ""}
                    >
                      <td>{row.label}</td>
                      <td>{row.view}</td>
                      <td>{row.summary}</td>
                      <td>
                        <span className="vision-chips">
                          {row.models.map((m) => (
                            <span key={m}>{m}</span>
                          ))}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      )}
    </>
  );
}
