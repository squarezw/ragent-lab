import type { EmbeddingResult } from "./types";
export function experimentKey(texts: string[], query: string): string {
  return JSON.stringify([texts, query.trim()]);
}
export function addComparison(
  runs: EmbeddingResult[],
  run: EmbeddingResult,
): EmbeddingResult[] {
  if (
    runs.length &&
    experimentKey(runs[0].texts, runs[0].query) !==
      experimentKey(run.texts, run.query)
  )
    throw new Error(
      "对比需要使用相同的候选文本和查询。请先清空已有对比，或恢复原始输入。",
    );
  return [...runs.filter((r) => r.model !== run.model), run];
}
