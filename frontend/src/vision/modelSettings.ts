// Visitor model settings: the access code (unlocks every server model), which models are
// enabled, and custom endpoints. Kept in this browser only (localStorage); the server never
// stores them. Shared by the settings page and the vision workbench.
import { useEffect, useState } from "react";
import type { VisionModel } from "./types";

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
  // Remember the access code and custom-model keys in this browser.
  rememberKeys: boolean;
  accessCode: string;
  // Catalog models not listed here are enabled.
  catalog: Record<string, { enabled: boolean }>;
  // Built-in models not listed here are enabled.
  builtin: Record<string, { enabled: boolean }>;
  custom: CustomModel[];
};

export type BuiltinInfo = {
  enabled: boolean;
  per_hour: number;
  models: { id: string; name: string; notes: string }[];
  // The server has a prompt password, so prompt management can be unlocked.
  prompts_editable?: boolean;
};

// What the server needs to call a model. Keys travel only in request bodies.
export type ModelRef =
  | { source: "catalog"; id: string }
  | { source: "builtin"; id: string }
  | ({ source: "custom" } & Omit<CustomModel, "uid" | "preset" | "enabled">);

export type ModelOption = {
  key: string;
  name: string;
  source: "catalog" | "builtin" | "custom";
  badge: string;
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
  rememberKeys: true,
  accessCode: "",
  catalog: {},
  builtin: {},
  custom: [],
};

const STORAGE = "lab-model-settings";

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
  if (!raw) return base;
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return base;
  }
  if (!isObj(data)) return base;
  return {
    version: 1,
    rememberKeys: data.rememberKeys !== false,
    accessCode: typeof data.accessCode === "string" ? data.accessCode : "",
    catalog: isObj(data.catalog)
      ? Object.fromEntries(
          Object.entries(data.catalog).map(([id, v]) => [
            id,
            { enabled: !(isObj(v) && v.enabled === false) },
          ]),
        )
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

export function catalogEnabled(s: ModelSettings, m: VisionModel): boolean {
  return s.catalog[m.id]?.enabled ?? true;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

// Every model the visitor can pick, in display order: catalog, built-in, custom.
// Catalog and built-in models both run on the server and need only the access code.
export function buildOptions(
  s: ModelSettings,
  catalog: VisionModel[],
  builtin: BuiltinInfo | null,
  { includeDisabled = false } = {},
): ModelOption[] {
  const out: ModelOption[] = [];
  const code = s.accessCode.trim() !== "";
  for (const m of catalog) {
    const enabled = catalogEnabled(s, m);
    if (!enabled && !includeDisabled) continue;
    out.push({
      key: `catalog:${m.id}`,
      name: m.name,
      source: "catalog",
      badge: "",
      ready: enabled && code,
      problem: code ? undefined : "未填写访问码",
      ref: { source: "catalog", id: m.id },
    });
  }
  if (builtin?.enabled)
    for (const m of builtin.models) {
      const enabled = s.builtin[m.id]?.enabled ?? true;
      if (!enabled && !includeDisabled) continue;
      out.push({
        key: `builtin:${m.id}`,
        name: m.name,
        source: "builtin",
        badge: "",
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
