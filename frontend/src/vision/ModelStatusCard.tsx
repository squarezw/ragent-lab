import {
  CheckCircle2,
  CircleDashed,
  LoaderCircle,
  RotateCcw,
  Square,
  XCircle,
} from "lucide-react";
import JsonViewer from "../components/JsonViewer";
import { ErrorMessage } from "../components/UI";
import type { RunEntry, RunStatus } from "./state";
import { isEngineering, type RunMeta } from "./types";

const STATUS: Record<RunStatus, string> = {
  idle: "未开始",
  queued: "排队中",
  running: "运行中",
  done: "已完成",
  failed: "失败",
  cancelled: "已取消",
};
const VERDICT = { correct: "正确", uncertain: "存疑", wrong: "错误" } as const;

export const formatElapsed = (meta?: RunMeta) =>
  meta ? `${(meta.elapsed_ms / 1000).toFixed(1)} s` : "—";

export function StatusIcon({ status }: { status: RunStatus }) {
  if (status === "done") return <CheckCircle2 size={14} />;
  if (status === "failed") return <XCircle size={14} />;
  if (status === "idle" || status === "cancelled") return <CircleDashed size={14} />;
  return <LoaderCircle className="spin" size={14} />;
}

export default function ModelStatusCard({
  run,
  scorable,
  onRetry,
  onRetryScore,
  onCancel,
}: {
  run: RunEntry;
  scorable: boolean;
  onRetry: () => void;
  onRetryScore: () => void;
  onCancel: () => void;
}) {
  const pending = run.status === "queued" || run.status === "running";
  const busy = pending || run.scoreStatus === "running";
  return (
    <article className={`vision-model-card is-${run.status}`}>
      <div className="vision-card-top">
        <span className={`vision-status-tag is-${run.status}`}>
          <StatusIcon status={run.status} />
          {STATUS[run.status]}
        </span>
        <span className="vision-card-id">{run.model}</span>
      </div>
      <h3>{run.model_name}</h3>
      <dl className="vision-card-stats">
        <div>
          <dt>耗时</dt>
          <dd>{pending ? "…" : formatElapsed(run.meta)}</dd>
        </div>
        {scorable && run.status === "done" && (
          <div className="vision-score">
            <dt>得分</dt>
            <dd>
              {run.scoreStatus === "done" && run.score ? (
                <strong>{run.score.score.toFixed(1)}</strong>
              ) : run.scoreStatus !== "running" ? (
                "未评分"
              ) : (
                <>
                  <LoaderCircle className="spin" size={13} /> 评分中
                </>
              )}
            </dd>
          </div>
        )}
      </dl>
      {busy && (
        <button type="button" className="button secondary" onClick={onCancel}>
          <Square size={11} />
          {pending ? "取消" : "取消评分"}
        </button>
      )}
      {(run.status === "failed" || run.status === "cancelled") && (
        <>
          {run.status === "failed" && (
            <ErrorMessage message={run.error ?? "运行失败"} />
          )}
          <button type="button" className="button secondary" onClick={onRetry}>
            <RotateCcw size={13} />
            重试
          </button>
        </>
      )}
      {scorable &&
        run.status === "done" &&
        (run.scoreStatus ?? "idle") !== "running" &&
        run.scoreStatus !== "done" && (
          <>
            {run.scoreError && <ErrorMessage message={run.scoreError} />}
            <button
              type="button"
              className="button secondary"
              onClick={onRetryScore}
            >
              <RotateCcw size={13} />
              重试评分
            </button>
          </>
        )}
      {run.score && run.scoreStatus === "done" && (
        <details className="vision-details">
          <summary>
            逐项评分
            <span>
              正确 {run.score.counts.correct} · 存疑{" "}
              {run.score.counts.uncertain} · 错误 {run.score.counts.wrong}
            </span>
          </summary>
          <div className="vision-table-wrap">
            <table className="vision-table">
              <thead>
                <tr>
                  <th>条目</th>
                  <th>判定</th>
                  <th>依据</th>
                </tr>
              </thead>
              <tbody>
                {run.score.judgements.map((j) => (
                  <tr key={j.item_id} title={j.reason}>
                    <td className="mono">{j.item_id}</td>
                    <td>
                      <span className={`vision-verdict ${j.verdict}`}>
                        {VERDICT[j.verdict]}
                      </span>
                    </td>
                    <td>{j.evidence || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {run.understanding && !isEngineering(run.understanding) && (
        <p className="vision-summary">{run.understanding.summary}</p>
      )}
      {run.understanding && (
        <JsonViewer
          title={`${run.model_name} · 理解结果`}
          data={run.understanding}
          filename={`understanding-${run.model_name}`}
        />
      )}
    </article>
  );
}
