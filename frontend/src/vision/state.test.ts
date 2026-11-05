import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  exportState, importState, initialState, keyStorage, reducer, type VisionState,
} from "./state";

const populated: VisionState = {
  step: 3,
  image: { kind: "upload", id: "abc" },
  imageName: "a.png",
  scorable: true,
  runs: { "m/x": { model: "m/x", model_name: "X", status: "failed", error: "超时" } },
  consensus: { status: "idle" },
  generations: [{ id: "g1", sourceModel: "m/x", generator: "m/y", status: "queued" }],
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

  it("runUpdate creates then merges entries", () => {
    let s = reducer(initialState, { type: "runUpdate", model: "m", patch: { model_name: "M", status: "running" } });
    s = reducer(s, { type: "runUpdate", model: "m", patch: { status: "done" } });
    expect(s.runs.m).toMatchObject({ model_name: "M", status: "done" });
  });
});

describe("keyStorage", () => {
  const store = new Map<string, string>();
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  beforeEach(() => {
    store.clear();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => void store.set(k, v),
        removeItem: (k: string) => void store.delete(k),
      },
    });
  });
  afterEach(() => {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else delete (globalThis as { localStorage?: unknown }).localStorage;
  });

  it("does not persist when remember is false", () => {
    keyStorage.save("sk-test", false);
    expect(store.has("vision-openrouter-key")).toBe(false);
    expect(keyStorage.load()).toBe("");
  });

  it("persists when remember is true and removes on later opt-out", () => {
    keyStorage.save("sk-test", true);
    expect(keyStorage.load()).toBe("sk-test");
    keyStorage.save("sk-test", false);
    expect(keyStorage.load()).toBe("");
  });
});
