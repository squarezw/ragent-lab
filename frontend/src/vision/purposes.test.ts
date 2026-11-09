import { describe, expect, it } from "vitest";
import {
  DEFAULT_PURPOSES,
  parsePurposes,
  purposeOptions,
  purposeRequest,
  selectedPurpose,
  withPrompt,
  type PurposeSettings,
} from "./purposes";
import { reducer, initialState, importState, exportState } from "./state";
import type { Purpose } from "./types";

const BUILTINS: Purpose[] = [
  { id: "engineering", name: "工程图纸", kind: "engineering", description: "d", prompt: "ENG" },
  { id: "chart", name: "数据图表", kind: "general", description: "d", prompt: "CHART" },
];
const settings = (patch: Partial<PurposeSettings>): PurposeSettings => ({
  ...structuredClone(DEFAULT_PURPOSES),
  ...patch,
});

describe("purposes", () => {
  it("defaults to the engineering purpose and sends no prompt for it", () => {
    const o = selectedPurpose(settings({}), BUILTINS)!;
    expect(o.key).toBe("builtin:engineering");
    expect(purposeRequest(o)).toEqual({ purpose: "engineering" });
  });

  it("an edited built-in prompt is sent; a blank one falls back to the default", () => {
    let s = settings({ selected: "builtin:chart" });
    const chart = selectedPurpose(s, BUILTINS)!;
    s = withPrompt(s, chart, "只读图例");
    const edited = selectedPurpose(s, BUILTINS)!;
    expect(edited.edited).toBe(true);
    expect(purposeRequest(edited)).toEqual({ purpose: "chart", prompt: "只读图例" });
    s = withPrompt(s, edited, "  ");
    const blank = selectedPurpose(s, BUILTINS)!;
    expect(blank.draft).toBe("  ");
    expect(purposeRequest(blank)).toEqual({ purpose: "chart" });
    s = withPrompt(s, blank, "CHART");
    expect(s.overrides).toEqual({});
  });

  it("custom purposes use their base structure and always send their prompt", () => {
    const s = settings({
      selected: "custom:u1",
      custom: [
        { uid: "u1", name: "电路图", base: "chart", prompt: "读电路" },
        { uid: "u2", name: "孤儿", base: "gone", prompt: "x" },
      ],
    });
    expect(purposeOptions(s, BUILTINS).map((o) => o.key)).toEqual([
      "builtin:engineering",
      "builtin:chart",
      "custom:u1",
    ]);
    const o = selectedPurpose(s, BUILTINS)!;
    expect(o.kind).toBe("general");
    expect(purposeRequest(o)).toEqual({ purpose: "chart", prompt: "读电路" });
    // A deleted selection falls back to the default.
    expect(selectedPurpose({ ...s, selected: "custom:nope" }, BUILTINS)!.key).toBe(
      "builtin:engineering",
    );
  });

  it("parses stored settings defensively", () => {
    expect(parsePurposes("{oops")).toEqual(DEFAULT_PURPOSES);
    expect(
      parsePurposes(
        JSON.stringify({ selected: 3, overrides: { chart: "x", bad: 1 }, custom: [{ uid: 1 }] }),
      ),
    ).toEqual({ ...DEFAULT_PURPOSES, overrides: { chart: "x" } });
  });
});

describe("run purpose in state", () => {
  const eng = { key: "builtin:engineering", name: "工程图纸", kind: "engineering" as const,
    request: { purpose: "engineering" } };
  const chart = { key: "builtin:chart", name: "数据图表", kind: "general" as const,
    request: { purpose: "chart" } };

  it("a different purpose or prompt clears results; the same one keeps them", () => {
    let s = reducer(initialState, { type: "setPurpose", purpose: eng });
    s = reducer(s, { type: "runUpdate", model: "m", patch: { status: "done" } });
    expect(reducer(s, { type: "setPurpose", purpose: { ...eng } }).runs).toHaveProperty("m");
    expect(reducer(s, { type: "setPurpose", purpose: chart }).runs).toEqual({});
    const edited = { ...eng, request: { purpose: "engineering", prompt: "新" } };
    expect(reducer(s, { type: "setPurpose", purpose: edited }).runs).toEqual({});
  });

  it("survives export and import; old files have none", () => {
    const s = { ...reducer(initialState, { type: "setPurpose", purpose: chart }) };
    expect(importState(exportState(s)).purpose).toEqual(chart);
    const old = JSON.parse(exportState(initialState));
    delete old.purpose;
    expect(importState(JSON.stringify(old)).purpose).toBeUndefined();
  });
});
