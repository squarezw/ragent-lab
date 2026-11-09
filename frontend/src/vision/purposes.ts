// Understanding purposes: what the image is (engineering drawing, chart, diagram…) and the
// prompt used to read it. Built-in purposes come from the server; the visitor can edit their
// prompts or save their own purposes. Kept in this browser only (localStorage).
import { useEffect, useState } from "react";
import type { Purpose, PurposeKind } from "./types";

export type CustomPurpose = {
  uid: string;
  name: string;
  // The built-in purpose whose output structure (and steps) this one uses.
  base: string;
  prompt: string;
};

export type PurposeSettings = {
  version: 1;
  selected: string;
  // Edited prompts of built-in purposes, by purpose id.
  overrides: Record<string, string>;
  custom: CustomPurpose[];
};

export type PurposeOption = {
  key: string;
  name: string;
  source: "builtin" | "custom";
  base: string;
  kind: PurposeKind;
  description: string;
  // The text the visitor typed (may be blank while editing).
  draft: string;
  // What is sent: the draft, or the default prompt when the draft is blank.
  prompt: string;
  defaultPrompt: string;
  edited: boolean;
};

// What the understand request carries; the server adds its fixed rules and the schema.
export type PurposeRequest = { purpose: string; prompt?: string };

export const MAX_PROMPT_CHARS = 8000;
export const DEFAULT_PURPOSE_KEY = "builtin:engineering";
export const KIND_LABELS: Record<PurposeKind, string> = {
  engineering: "工程图纸结构（可还原为 DXF）",
  general: "通用结构（概述、元素、文字、数据）",
};

export const DEFAULT_PURPOSES: PurposeSettings = {
  version: 1,
  selected: DEFAULT_PURPOSE_KEY,
  overrides: {},
  custom: [],
};

const STORAGE = "lab-purposes";

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export function parsePurposes(raw: string | null): PurposeSettings {
  let data: unknown = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = null;
  }
  if (!isObj(data)) return structuredClone(DEFAULT_PURPOSES);
  const overrides: Record<string, string> = {};
  if (isObj(data.overrides))
    for (const [k, v] of Object.entries(data.overrides))
      if (typeof v === "string") overrides[k] = v;
  return {
    version: 1,
    selected:
      typeof data.selected === "string" ? data.selected : DEFAULT_PURPOSE_KEY,
    overrides,
    custom: Array.isArray(data.custom)
      ? (data.custom as CustomPurpose[]).filter(
          (c) =>
            isObj(c) &&
            typeof c.uid === "string" &&
            typeof c.name === "string" &&
            typeof c.base === "string" &&
            typeof c.prompt === "string",
        )
      : [],
  };
}

function read(): string | null {
  try {
    return localStorage.getItem(STORAGE);
  } catch {
    return null;
  }
}

let current: PurposeSettings = parsePurposes(read());
const listeners = new Set<(s: PurposeSettings) => void>();

export function savePurposes(next: PurposeSettings): void {
  current = next;
  try {
    localStorage.setItem(STORAGE, JSON.stringify(next));
  } catch {
    /* storage unavailable: settings last for this page view */
  }
  listeners.forEach((fn) => fn(next));
}

export function usePurposeSettings(): [
  PurposeSettings,
  (update: (s: PurposeSettings) => PurposeSettings) => void,
] {
  const [settings, setSettings] = useState(current);
  useEffect(() => {
    listeners.add(setSettings);
    setSettings(current);
    return () => {
      listeners.delete(setSettings);
    };
  }, []);
  return [settings, (update) => savePurposes(update(current))];
}

// Built-in purposes first, then the visitor's own (those whose base is unknown are dropped).
export function purposeOptions(
  s: PurposeSettings,
  builtins: Purpose[],
): PurposeOption[] {
  const out: PurposeOption[] = builtins.map((p) => {
    const draft = s.overrides[p.id] ?? p.prompt;
    const prompt = draft.trim() ? draft : p.prompt;
    return {
      key: `builtin:${p.id}`,
      name: p.name,
      source: "builtin",
      base: p.id,
      kind: p.kind,
      description: p.description,
      draft,
      prompt,
      defaultPrompt: p.prompt,
      edited: prompt !== p.prompt,
    };
  });
  for (const c of s.custom) {
    const base = builtins.find((p) => p.id === c.base);
    if (!base) continue;
    out.push({
      key: `custom:${c.uid}`,
      name: c.name,
      source: "custom",
      base: base.id,
      kind: base.kind,
      description: `自定义 · 基于「${base.name}」`,
      draft: c.prompt,
      prompt: c.prompt.trim() ? c.prompt : base.prompt,
      defaultPrompt: base.prompt,
      edited: c.prompt.trim() !== "" && c.prompt !== base.prompt,
    });
  }
  return out;
}

// The selected purpose, falling back to the default and then the first one.
export function selectedPurpose(
  s: PurposeSettings,
  builtins: Purpose[],
): PurposeOption | null {
  const options = purposeOptions(s, builtins);
  return (
    options.find((o) => o.key === s.selected) ??
    options.find((o) => o.key === DEFAULT_PURPOSE_KEY) ??
    options[0] ??
    null
  );
}

// Unchanged built-in prompts are not sent, so the server's current default is used.
export function purposeRequest(o: PurposeOption): PurposeRequest {
  return o.edited || o.source === "custom"
    ? { purpose: o.base, prompt: o.prompt.slice(0, MAX_PROMPT_CHARS) }
    : { purpose: o.base };
}

// Change the prompt of a purpose: an override for a built-in one, the text of a custom one.
export function withPrompt(
  s: PurposeSettings,
  o: PurposeOption,
  prompt: string,
): PurposeSettings {
  if (o.source === "custom")
    return {
      ...s,
      custom: s.custom.map((c) =>
        `custom:${c.uid}` === o.key ? { ...c, prompt } : c,
      ),
    };
  const overrides = { ...s.overrides };
  if (prompt === o.defaultPrompt) delete overrides[o.base];
  else overrides[o.base] = prompt;
  return { ...s, overrides };
}

export function newPurpose(base: Purpose): CustomPurpose {
  return {
    uid:
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).slice(2, 10),
    name: "",
    base: base.id,
    prompt: base.prompt,
  };
}
