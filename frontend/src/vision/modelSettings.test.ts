import { describe, expect, it } from "vitest";
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

const model = (id: string): VisionModel => ({
  id,
  name: id.toUpperCase(),
  family: "x",
  notes: "",
  params: { max_tokens: 1, temperature: null, reasoning: null },
});
const CATALOG = [model("a/claude"), model("b/gpt")];
const settings = (patch: Partial<ModelSettings>): ModelSettings => ({
  ...structuredClone(DEFAULT_SETTINGS),
  ...patch,
});
const INFO = { enabled: true, per_hour: 60, models: [{ id: "ds", name: "DS", notes: "快速" }] };

describe("buildOptions", () => {
  it("catalog and built-in models need only the access code, with no platform shown", () => {
    const locked = buildOptions(settings({}), CATALOG, INFO);
    expect(locked.map((o) => [o.key, o.ready, o.problem])).toEqual([
      ["catalog:a/claude", false, "未填写访问码"],
      ["catalog:b/gpt", false, "未填写访问码"],
      ["builtin:ds", false, "未填写访问码"],
    ]);
    const open = buildOptions(settings({ accessCode: "code" }), CATALOG, INFO);
    expect(open.every((o) => o.ready && o.badge === "")).toBe(true);
    expect(open[0].ref).toEqual({ source: "catalog", id: "a/claude" });
    expect(open[2].ref).toEqual({ source: "builtin", id: "ds" });
    expect(JSON.stringify(open)).not.toMatch(/openrouter|kie|aihubmix|api_key/i);
  });

  it("disabled models are hidden unless asked for", () => {
    const s = settings({ catalog: { "b/gpt": { enabled: false } }, builtin: { ds: { enabled: false } } });
    expect(buildOptions(s, CATALOG, INFO).map((o) => o.key)).toEqual(["catalog:a/claude"]);
    expect(buildOptions(s, CATALOG, INFO, { includeDisabled: true })).toHaveLength(3);
  });

  it("built-in models appear only when the server offers them", () => {
    expect(buildOptions(settings({}), [], { ...INFO, enabled: false })).toEqual([]);
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

  it("presets are vendors' own APIs only", () => {
    expect(PRESETS.map((p) => p.url).join(" ")).not.toMatch(/openrouter|kie\.ai|aihubmix/);
  });
});

describe("storage form", () => {
  it("drops the access code and custom keys when the visitor opts out", () => {
    const c = { ...newCustom(PRESETS[0]), name: "x", api_key: "sk-c" };
    const s = settings({ accessCode: "code", custom: [c], rememberKeys: false });
    const text = JSON.stringify(storedForm(s));
    for (const secret of ["code", "sk-c"]) expect(text).not.toContain(secret);
    expect(storedForm({ ...s, rememberKeys: true })).toEqual({ ...s, rememberKeys: true });
  });

  it("parses saved settings, old ones included, and tolerates junk", () => {
    const s = settings({ accessCode: "c", catalog: { "a/claude": { enabled: false } } });
    expect(parseSettings(JSON.stringify(s))).toEqual(s);
    // Settings saved before: platform keys and per-model platforms are dropped.
    const old = parseSettings(JSON.stringify({
      keys: { openrouter: "sk-or" }, accessCode: "c",
      catalog: { "a/claude": { enabled: true, via: "kie" }, "b/gpt": { enabled: false, via: "openrouter" } },
    }));
    expect(old.catalog).toEqual({ "a/claude": { enabled: true }, "b/gpt": { enabled: false } });
    expect(JSON.stringify(old)).not.toContain("sk-or");
    expect(parseSettings("{oops")).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings('{"custom":[1,{"uid":"u"}]}').custom).toHaveLength(1);
  });
});
