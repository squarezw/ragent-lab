import { describe, expect, it } from "vitest";
import { experimentKey, addComparison } from "./experiment-state";
import type { EmbeddingResult } from "./types";
const first: EmbeddingResult = {
  model: "a",
  model_name: "A",
  texts: ["苹果", "汽车"],
  query: "水果",
  dimension: 2,
  count: 2,
  elapsed_ms: 20,
  rankings: [{ index: 0, text: "苹果", score: 1 }],
};
describe("immutable comparison runs", () => {
  it("refuses a comparison with a different query or corpus", () => {
    expect(() =>
      addComparison([first], { ...first, model: "b", query: "交通" }),
    ).toThrow();
    expect(() =>
      addComparison([first], {
        ...first,
        model: "b",
        texts: ["苹果", "自行车"],
      }),
    ).toThrow();
  });
  it("keeps independent model runs for identical inputs", () => {
    expect(
      addComparison([first], { ...first, model: "b" }).map((r) => r.model),
    ).toEqual(["a", "b"]);
    expect(addComparison([first], { ...first, elapsed_ms: 30 })).toHaveLength(
      1,
    );
  });
  it("detects dirty inputs and ignores surrounding query whitespace", () => {
    expect(experimentKey(["苹果", "汽车"], "水果")).not.toBe(
      experimentKey(["苹果"], "水果"),
    );
    expect(experimentKey(["苹果"], " 水果 ")).toBe(
      experimentKey(["苹果"], "水果"),
    );
  });
});
