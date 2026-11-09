import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_SETTINGS,
  PRESETS,
  buildOptions,
  catalogChoice,
  newCustom,
  parseSettings,
  storedForm,
  type BuiltinInfo,
  type ModelSettings,
} from "./modelSettings";
import type { VisionModel } from "./types";

const model = (id: string, gateways: ("openrouter" | "kie" | "aihubmix")[]): VisionModel => ({
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
      settings({ keys: { openrouter: "sk-or", kie: "", aihubmix: "" } }),
      CATALOG,
      null,
    );
    expect(ok[0].ref).toEqual({
      source: "catalog",
      id: "a/claude",
      gateway: "openrouter",
      api_key: "sk-or",
    });
    expect(ok[0]).not.toHaveProperty("priceLabel");
  });

  it("the server's platform key works with the access code; an own key wins", () => {
    const info = { enabled: true, per_hour: 60, models: [], platforms: ["openrouter" as const] };
    const noCode = buildOptions(settings({}), CATALOG, info);
    expect(noCode[0]).toMatchObject({ ready: false, problem: "未填写 OpenRouter Key 或访问码" });
    const [server] = buildOptions(settings({ accessCode: "code" }), CATALOG, info);
    expect(server).toMatchObject({ ready: true, badge: "OpenRouter · 服务器 Key" });
    expect(server.ref).toMatchObject({ gateway: "openrouter", api_key: "" });
    const [own] = buildOptions(
      settings({ accessCode: "code", keys: { openrouter: "sk-mine", kie: "", aihubmix: "" } }),
      CATALOG,
      info,
    );
    expect(own).toMatchObject({ ready: true, badge: "OpenRouter" });
    expect(own.ref).toMatchObject({ api_key: "sk-mine" });
    // Kie.ai is not offered by the server: still needs the visitor's key.
    const kie = buildOptions(
      settings({ accessCode: "code", catalog: { "a/claude": { enabled: true, via: "kie" } } }),
      CATALOG,
      info,
    );
    expect(kie[0]).toMatchObject({ ready: false, problem: "未填写 Kie.ai Key" });
  });

  it("an unset model defaults to the first usable platform, preferring AIHubMix", () => {
    const m = model("a/claude", ["openrouter", "kie", "aihubmix"]);
    const pick = (patch: Partial<ModelSettings>, info: BuiltinInfo | null = null) =>
      catalogChoice(settings(patch), m, info).via;
    expect(pick({})).toBe("openrouter"); // nothing usable: the first platform
    expect(pick({ keys: { openrouter: "sk-or", kie: "", aihubmix: "" } })).toBe("openrouter");
    expect(pick({ keys: { openrouter: "sk-or", kie: "", aihubmix: "sk-ahm" } })).toBe("aihubmix");
    const info = { enabled: true, per_hour: 60, models: [], platforms: ["aihubmix" as const] };
    expect(pick({ accessCode: "c" }, info)).toBe("aihubmix");
    // An explicit choice always wins.
    expect(
      pick({ accessCode: "c", catalog: { "a/claude": { enabled: true, via: "kie" } } }, info),
    ).toBe("kie");
  });

  it("uses the chosen platform and ignores one the model is not on", () => {
    const s = settings({
      keys: { openrouter: "", kie: "kie-key", aihubmix: "" },
      catalog: {
        "a/claude": { enabled: true, via: "kie" },
        "b/qwen": { enabled: true, via: "kie" }, // qwen is not on Kie.ai
      },
    });
    const [claude, qwen] = buildOptions(s, CATALOG, null);
    expect(claude).toMatchObject({ ready: true, badge: "Kie.ai" });
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
    expect(opt).toMatchObject({ ready: true, badge: "自定义" });
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
      keys: { openrouter: "sk-or", kie: "k", aihubmix: "" },
      accessCode: "code",
      custom: [c],
      rememberKeys: false,
    });
    const text = JSON.stringify(storedForm(s));
    for (const secret of ["sk-or", '"k"', "code", "sk-c"]) expect(text).not.toContain(secret);
    expect(storedForm({ ...s, rememberKeys: true })).toEqual({ ...s, rememberKeys: true });
  });

  it("parses saved settings and tolerates junk", () => {
    const s = settings({ keys: { openrouter: "a", kie: "b", aihubmix: "" }, accessCode: "c" });
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
    expect(parseSettings(null).keys).toEqual({ openrouter: "old-or", kie: "old-kie", aihubmix: "" });
  });
});
