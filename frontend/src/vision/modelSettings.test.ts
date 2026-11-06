import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_SETTINGS,
  PRESETS,
  buildOptions,
  newCustom,
  parseSettings,
  storedForm,
  type ModelSettings,
} from "./modelSettings";
import type { VisionModel } from "./types";

const model = (id: string, gateways: ("openrouter" | "kie")[]): VisionModel => ({
  id,
  name: id.toUpperCase(),
  family: "x",
  notes: "",
  params: { max_tokens: 1, temperature: null, reasoning: null },
  price: { prompt: "0.000001", completion: "0.000002" },
  gateways,
});
const CATALOG = [model("a/claude", ["openrouter", "kie"]), model("b/qwen", ["openrouter"])];
const settings = (patch: Partial<ModelSettings>): ModelSettings => ({
  ...structuredClone(DEFAULT_SETTINGS),
  ...patch,
});

describe("buildOptions", () => {
  it("catalog models default to enabled via OpenRouter and need that key", () => {
    const none = buildOptions(settings({}), CATALOG, null);
    expect(none.map((o) => [o.key, o.ready, o.problem])).toEqual([
      ["catalog:a/claude", false, "未填写 OpenRouter Key"],
      ["catalog:b/qwen", false, "未填写 OpenRouter Key"],
    ]);
    const ok = buildOptions(
      settings({ keys: { openrouter: "sk-or", kie: "" } }),
      CATALOG,
      null,
    );
    expect(ok[0].ref).toEqual({
      source: "catalog",
      id: "a/claude",
      gateway: "openrouter",
      api_key: "sk-or",
    });
    expect(ok[0].priceLabel).toContain("$1.00");
  });

  it("uses the chosen platform and ignores one the model is not on", () => {
    const s = settings({
      keys: { openrouter: "", kie: "kie-key" },
      catalog: {
        "a/claude": { enabled: true, via: "kie" },
        "b/qwen": { enabled: true, via: "kie" }, // qwen is not on Kie.ai
      },
    });
    const [claude, qwen] = buildOptions(s, CATALOG, null);
    expect(claude).toMatchObject({ ready: true, badge: "Kie.ai", priceLabel: "按 Kie.ai 积分计费" });
    expect(qwen).toMatchObject({ ready: false, badge: "OpenRouter" });
  });

  it("disabled models are hidden unless asked for", () => {
    const s = settings({ catalog: { "b/qwen": { enabled: false, via: "openrouter" } } });
    expect(buildOptions(s, CATALOG, null).map((o) => o.key)).toEqual(["catalog:a/claude"]);
    expect(buildOptions(s, CATALOG, null, { includeDisabled: true })).toHaveLength(2);
  });

  it("built-in models need the server to offer them and an access code", () => {
    const info = { enabled: true, per_hour: 60, models: [{ id: "ds", name: "DS", notes: "" }] };
    expect(buildOptions(settings({}), [], { ...info, enabled: false })).toEqual([]);
    const [locked] = buildOptions(settings({}), [], info);
    expect(locked).toMatchObject({ ready: false, problem: "未填写访问码" });
    const [open] = buildOptions(settings({ accessCode: "code" }), [], info);
    expect(open).toMatchObject({ ready: true, ref: { source: "builtin", id: "ds" } });
  });

  it("custom models send their own endpoint and key", () => {
    const c = {
      ...newCustom(PRESETS.find((p) => p.id === "deepseek")!),
      name: "DeepSeek 官方",
      model: "deepseek-x",
      api_key: "sk-ds",
    };
    const [opt] = buildOptions(settings({ custom: [c] }), [], null);
    expect(opt).toMatchObject({ ready: true, badge: "自定义", priceLabel: "api.deepseek.com" });
    expect(opt.ref).toEqual({
      source: "custom",
      name: "DeepSeek 官方",
      api: "chat",
      url: "https://api.deepseek.com/chat/completions",
      model: "deepseek-x",
      json_mode: "object",
      auth: "bearer",
      max_tokens: 16000,
      api_key: "sk-ds",
    });
    const [noKey] = buildOptions(settings({ custom: [{ ...c, api_key: "" }] }), [], null);
    expect(noKey).toMatchObject({ ready: false, problem: "未填写 Key" });
  });
});

describe("storage form", () => {
  it("drops every key when the visitor opts out", () => {
    const c = { ...newCustom(PRESETS[0]), name: "x", api_key: "sk-c" };
    const s = settings({
      keys: { openrouter: "sk-or", kie: "k" },
      accessCode: "code",
      custom: [c],
      rememberKeys: false,
    });
    const text = JSON.stringify(storedForm(s));
    for (const secret of ["sk-or", '"k"', "code", "sk-c"]) expect(text).not.toContain(secret);
    expect(storedForm({ ...s, rememberKeys: true })).toEqual({ ...s, rememberKeys: true });
  });

  it("parses saved settings and tolerates junk", () => {
    const s = settings({ keys: { openrouter: "a", kie: "b" }, accessCode: "c" });
    expect(parseSettings(JSON.stringify(s))).toEqual(s);
    expect(parseSettings("{oops")).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings('{"custom":[1,{"uid":"u"}]}').custom).toHaveLength(1);
  });
});

describe("first run", () => {
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

  it("carries over keys saved by the old key form", () => {
    store.set("vision-openrouter-key", "old-or");
    store.set("vision-kie-key", "old-kie");
    expect(parseSettings(null).keys).toEqual({ openrouter: "old-or", kie: "old-kie" });
  });
});
