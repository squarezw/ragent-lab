// Visitor model settings: platform keys, built-in access code, which catalog models run on
// which platform, and custom endpoints. Kept in this browser only (localStorage); the server
// never stores them. Shared by the settings page and the vision workbench.
import { useEffect, useState } from "react";
import { GATEWAY_NAMES, type Gateway, type VisionModel } from "./types";

export type Platform = Gateway;
export type WireApi = "chat" | "responses" | "anthropic";
export type JsonMode = "schema" | "object" | "prompt";
export type AuthMode = "bearer" | "x-api-key";

export type CustomModel = {
  uid: string;
  name: string;
  preset: string;
  api: WireApi;
  url: string;
  model: string;
  json_mode: JsonMode;
  auth: AuthMode;
  max_tokens: number;
  api_key: string;
  enabled: boolean;
};

export type ModelSettings = {
  version: 1;
  keys: Record<Platform, string>;
  rememberKeys: boolean;
  accessCode: string;
  // Catalog models not listed here are enabled and use OpenRouter.
  catalog: Record<string, { enabled: boolean; via: Platform }>;
  // Built-in models not listed here are enabled.
  builtin: Record<string, { enabled: boolean }>;
  custom: CustomModel[];
};

export type BuiltinInfo = {
  enabled: boolean;
  per_hour: number;
  models: { id: string; name: string; notes: string }[];
};

// What the server needs to call a model. Keys travel only in request bodies.
export type ModelRef =
  | { source: "catalog"; id: string; gateway: Platform; api_key: string }
  | { source: "builtin"; id: string }
  | ({ source: "custom" } & Omit<CustomModel, "uid" | "preset" | "enabled">);

export type ModelOption = {
  key: string;
  name: string;
  source: "catalog" | "builtin" | "custom";
  badge: string;
  priceLabel: string;
  ready: boolean;
  problem?: string;
  ref: ModelRef;
};

export type Preset = {
  id: string;
  label: string;
  api: WireApi;
  url: string;
  json_mode: JsonMode;
  auth: AuthMode;
  modelHint: string;
  keyUrl?: string;
  note?: string;
};

// Presets fill in endpoint and format; the visitor still types the model name and key.
export const PRESETS: Preset[] = [
  {
    id: "openrouter",
    label: "OpenRouter（任意模型 ID）",
    api: "chat",
    url: "https://openrouter.ai/api/v1/chat/completions",
    json_mode: "schema",
    auth: "bearer",
    modelHint: "例如 qwen/qwen3.8-max-prime",
    keyUrl: "https://openrouter.ai/keys",
  },
  {
    id: "kie-claude",
    label: "Kie.ai · Claude",
    api: "anthropic",
    url: "https://api.kie.ai/claude/v1/messages",
    json_mode: "prompt",
    auth: "bearer",
    modelHint: "例如 claude-sonnet-5-5",
    keyUrl: "https://kie.ai/api-key",
  },
  {
    id: "kie-gpt",
    label: "Kie.ai · GPT（Codex 接口）",
    api: "responses",
    url: "https://api.kie.ai/codex/v1/responses",
    json_mode: "schema",
    auth: "bearer",
    modelHint: "例如 gpt-6-luna",
    keyUrl: "https://kie.ai/api-key",
  },
  {
    id: "kie-responses",
    label: "Kie.ai · Kimi / DeepSeek / Grok",
    api: "responses",
    url: "https://api.kie.ai/openai/v1/responses",
    json_mode: "schema",
    auth: "bearer",
    modelHint: "例如 deepseek-v4-1-flash",
    keyUrl: "https://kie.ai/api-key",
  },
  {
    id: "kie-gemini",
    label: "Kie.ai · Gemini",
    api: "chat",
    url: "https://api.kie.ai/gemini-3-8-flash-openai/v1/chat/completions",
    json_mode: "prompt",
    auth: "bearer",
    modelHint: "留空（模型由地址决定）",
    keyUrl: "https://kie.ai/api-key",
    note: "Kie.ai 的 Gemini 每个模型一个地址，把地址中的 gemini-3-8-flash 换成要用的模型。",
  },
  {
    id: "deepseek",
    label: "DeepSeek 官方",
    api: "chat",
    url: "https://api.deepseek.com/chat/completions",
    json_mode: "object",
    auth: "bearer",
    modelHint: "填 DeepSeek 控制台里支持图片的模型名",
    keyUrl: "https://platform.deepseek.com/api_keys",
  },
  {
    id: "moonshot",
    label: "Moonshot（Kimi）官方",
    api: "chat",
    url: "https://api.moonshot.cn/v1/chat/completions",
    json_mode: "object",
    auth: "bearer",
    modelHint: "填 Moonshot 控制台里的视觉模型名",
    keyUrl: "https://platform.moonshot.cn/console/api-keys",
  },
  {
    id: "dashscope",
    label: "阿里云百炼（Qwen）",
    api: "chat",
    url: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    json_mode: "object",
    auth: "bearer",
    modelHint: "例如 qwen-vl-max",
    keyUrl: "https://bailian.console.aliyun.com/",
  },
  {
    id: "zhipu",
    label: "智谱开放平台（GLM）",
    api: "chat",
    url: "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    json_mode: "prompt",
    auth: "bearer",
    modelHint: "填智谱控制台里的视觉模型名（带 v）",
    keyUrl: "https://open.bigmodel.cn/usercenter/apikeys",
  },
  {
    id: "openai",
    label: "OpenAI 官方",
    api: "chat",
    url: "https://api.openai.com/v1/chat/completions",
    json_mode: "schema",
    auth: "bearer",
    modelHint: "填 OpenAI 的模型名",
    keyUrl: "https://platform.openai.com/api-keys",
  },
  {
    id: "anthropic",
    label: "Anthropic 官方",
    api: "anthropic",
    url: "https://api.anthropic.com/v1/messages",
    json_mode: "prompt",
    auth: "x-api-key",
    modelHint: "填 Anthropic 的模型名",
    keyUrl: "https://console.anthropic.com/settings/keys",
  },
  {
    id: "other",
    label: "其他（兼容 OpenAI 的接口）",
    api: "chat",
    url: "https://",
    json_mode: "prompt",
    auth: "bearer",
    modelHint: "模型名",
  },
];

export const DEFAULT_SETTINGS: ModelSettings = {
  version: 1,
  keys: { openrouter: "", kie: "" },
  rememberKeys: true,
  accessCode: "",
  catalog: {},
  builtin: {},
  custom: [],
};

const STORAGE = "lab-model-settings";
const LEGACY_KEYS: Record<Platform, string> = {
  openrouter: "vision-openrouter-key",
  kie: "vision-kie-key",
};

function read(name: string): string | null {
  try {
    return localStorage.getItem(name);
  } catch {
    return null;
  }
}

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function parseSettings(raw: string | null): ModelSettings {
  const base: ModelSettings = structuredClone(DEFAULT_SETTINGS);
  if (!raw) {
    // First visit after the settings page shipped: carry over keys saved by the old form.
    for (const p of ["openrouter", "kie"] as Platform[])
      base.keys[p] = read(LEGACY_KEYS[p]) ?? "";
    return base;
  }
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return base;
  }
  if (!isObj(data)) return base;
  const keys = isObj(data.keys) ? data.keys : {};
  return {
    version: 1,
    keys: {
      openrouter: typeof keys.openrouter === "string" ? keys.openrouter : "",
      kie: typeof keys.kie === "string" ? keys.kie : "",
    },
    rememberKeys: data.rememberKeys !== false,
    accessCode: typeof data.accessCode === "string" ? data.accessCode : "",
    catalog: isObj(data.catalog)
      ? (data.catalog as ModelSettings["catalog"])
      : {},
    builtin: isObj(data.builtin)
      ? (data.builtin as ModelSettings["builtin"])
      : {},
    custom: Array.isArray(data.custom)
      ? (data.custom as CustomModel[]).filter(
          (c) => isObj(c) && typeof c.uid === "string",
        )
      : [],
  };
}

// What is written to localStorage: without any keys when the visitor opted out.
export function storedForm(s: ModelSettings): ModelSettings {
  if (s.rememberKeys) return s;
  return {
    ...s,
    keys: { openrouter: "", kie: "" },
    accessCode: "",
    custom: s.custom.map((c) => ({ ...c, api_key: "" })),
  };
}

// One in-memory copy for the whole app, so keys survive navigation even when not saved.
let current: ModelSettings = parseSettings(read(STORAGE));
const listeners = new Set<(s: ModelSettings) => void>();

export function getSettings(): ModelSettings {
  return current;
}

export function saveSettings(next: ModelSettings): void {
  current = next;
  try {
    localStorage.setItem(STORAGE, JSON.stringify(storedForm(next)));
    for (const p of ["openrouter", "kie"] as Platform[])
      localStorage.removeItem(LEGACY_KEYS[p]);
  } catch {
    /* storage unavailable: settings last for this page view */
  }
  listeners.forEach((fn) => fn(next));
}

export function useModelSettings(): [
  ModelSettings,
  (update: (s: ModelSettings) => ModelSettings) => void,
] {
  const [settings, setSettings] = useState(current);
  useEffect(() => {
    listeners.add(setSettings);
    setSettings(current);
    return () => {
      listeners.delete(setSettings);
    };
  }, []);
  return [settings, (update) => saveSettings(update(current))];
}

export function catalogChoice(s: ModelSettings, m: VisionModel) {
  const saved = s.catalog[m.id];
  const served = m.gateways ?? ["openrouter"];
  const via = saved && served.includes(saved.via) ? saved.via : served[0];
  return { enabled: saved?.enabled ?? true, via };
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

function perMillion(price: VisionModel["price"]): string {
  if (!price || price.prompt == null || price.completion == null)
    return "价格未知";
  const fmt = (v: string) => {
    const n = Number(v) * 1e6;
    return Number.isFinite(n) ? `$${n.toFixed(2)}` : "?";
  };
  return `输入 ${fmt(price.prompt)} · 输出 ${fmt(price.completion)} / 百万 token`;
}

// Every model the visitor can pick, in display order: catalog, built-in, custom.
export function buildOptions(
  s: ModelSettings,
  catalog: VisionModel[],
  builtin: BuiltinInfo | null,
  { includeDisabled = false } = {},
): ModelOption[] {
  const out: ModelOption[] = [];
  for (const m of catalog) {
    const { enabled, via } = catalogChoice(s, m);
    if (!enabled && !includeDisabled) continue;
    const key = s.keys[via];
    out.push({
      key: `catalog:${m.id}`,
      name: m.name,
      source: "catalog",
      badge: GATEWAY_NAMES[via],
      priceLabel: via === "kie" ? "按 Kie.ai 积分计费" : perMillion(m.price),
      ready: enabled && key.trim() !== "",
      problem: key.trim() ? undefined : `未填写 ${GATEWAY_NAMES[via]} Key`,
      ref: { source: "catalog", id: m.id, gateway: via, api_key: key },
    });
  }
  if (builtin?.enabled)
    for (const m of builtin.models) {
      const enabled = s.builtin[m.id]?.enabled ?? true;
      if (!enabled && !includeDisabled) continue;
      const code = s.accessCode.trim() !== "";
      out.push({
        key: `builtin:${m.id}`,
        name: m.name,
        source: "builtin",
        badge: "内置",
        priceLabel: m.notes || "服务器提供",
        ready: enabled && code,
        problem: code ? undefined : "未填写访问码",
        ref: { source: "builtin", id: m.id },
      });
    }
  for (const c of s.custom) {
    if (!c.enabled && !includeDisabled) continue;
    const missing = !c.url.startsWith("https://")
      ? "接口地址必须以 https:// 开头"
      : !c.api_key.trim()
        ? "未填写 Key"
        : undefined;
    out.push({
      key: `custom:${c.uid}`,
      name: c.name,
      source: "custom",
      badge: "自定义",
      priceLabel: hostOf(c.url),
      ready: c.enabled && !missing,
      problem: missing,
      ref: {
        source: "custom",
        name: c.name,
        api: c.api,
        url: c.url,
        model: c.model,
        json_mode: c.json_mode,
        auth: c.auth,
        max_tokens: c.max_tokens,
        api_key: c.api_key,
      },
    });
  }
  return out;
}

export function newCustom(preset: Preset): CustomModel {
  return {
    uid:
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).slice(2, 10),
    name: "",
    preset: preset.id,
    api: preset.api,
    url: preset.url,
    model: "",
    json_mode: preset.json_mode,
    auth: preset.auth,
    max_tokens: 16000,
    api_key: "",
    enabled: true,
  };
}
