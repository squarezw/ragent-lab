import { describe, expect, it } from "vitest";
import {
  exportState, importState, initialState, reducer, type VisionState,
} from "./state";

const populated: VisionState = {
  step: 3,
  image: { kind: "upload", id: "abc" },
  imageName: "a.png",
  scorable: true,
  runs: { "m/x": { model: "m/x", model_name: "X", status: "failed", error: "超时" } },
  consensus: { status: "idle" },
  generations: [{ id: "g1", sourceModel: "m/x", generator: "m/y", status: "done" }],
};

describe("vision state", () => {
  it("selectImage clears runs and generations", () => {
    const next = reducer(populated, {
      type: "selectImage", image: { kind: "sample", id: "s1" }, imageName: "s", scorable: false,
    });
    expect(next.runs).toEqual({});
    expect(next.generations).toEqual([]);
    expect(next.consensus).toBeUndefined();
    expect(next.image).toEqual({ kind: "sample", id: "s1" });
  });

  it("export→import roundtrip preserves runs and drops nothing", () => {
    const text = exportState(populated);
    expect(text).not.toContain("api_key");
    expect(JSON.parse(text).version).toBe(1);
    expect(JSON.parse(text).exported_at).toBeTruthy();
    const back = importState(text);
    expect(back).toEqual({ ...populated, step: 2 });
  });

  it("importState rejects wrong version", () => {
    const bad = JSON.stringify({ ...JSON.parse(exportState(populated)), version: 2 });
    expect(() => importState(bad)).toThrow("导入文件格式不正确");
    expect(() => importState("not json")).toThrow("导入文件格式不正确");
    expect(() => importState(JSON.stringify({ version: 1 }))).toThrow("导入文件格式不正确");
  });

  it("importState turns unfinished work into failed", () => {
    const busy: VisionState = {
      ...populated,
      runs: { m: { model: "m", model_name: "M", status: "running", scoreStatus: "running" },
              q: { model: "q", model_name: "Q", status: "queued" } },
      consensus: { status: "running" },
      generations: [{ id: "g", sourceModel: "m", generator: "m", status: "queued" }],
    };
    const back = importState(exportState(busy));
    const msg = "导入时任务未完成，请重新运行。";
    expect(back.runs.m).toMatchObject({ status: "failed", error: msg, scoreStatus: "failed", scoreError: msg });
    expect(back.runs.q).toMatchObject({ status: "failed", error: msg });
    expect(back.consensus).toMatchObject({ status: "failed", error: msg });
    expect(back.generations[0]).toMatchObject({ status: "failed", error: msg });
  });

  it("runUpdate creates then merges entries", () => {
    let s = reducer(initialState, { type: "runUpdate", model: "m", patch: { model_name: "M", status: "running" } });
    s = reducer(s, { type: "runUpdate", model: "m", patch: { status: "done" } });
    expect(s.runs.m).toMatchObject({ model_name: "M", status: "done" });
  });
});
