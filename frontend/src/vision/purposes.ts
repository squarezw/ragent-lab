// Understanding purposes: what the image is (engineering drawing, chart, diagram…) and the
// prompt used to read it. Built-in purposes come from the server. Their prompts are hidden;
// whoever unlocks prompt management with the server's prompt password can read them, edit them
// and save their own purposes. Edits are kept in this browser only (localStorage) and the
// server refuses an edited prompt that comes without the password.
import { useEffect, useState } from "react";
import { listPurposes, unlockPurposes } from "./client";
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
// This is also saved with the results (and exported), so it never holds the prompt password:
// withPromptCode adds that when the request is sent.
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
// While prompt management is locked, saved edits and own purposes are ignored: every purpose
// runs the server's default prompt, whose text this browser does not have.
export function purposeOptions(
  s: PurposeSettings,
  builtins: Purpose[],
  unlocked: boolean,
): PurposeOption[] {
  const out: PurposeOption[] = builtins.map((p) => {
    const base = unlocked ? (p.prompt ?? "") : "";
    const draft = unlocked ? (s.overrides[p.id] ?? base) : base;
    const prompt = draft.trim() ? draft : base;
    return {
      key: `builtin:${p.id}`,
      name: p.name,
      source: "builtin",
      base: p.id,
      kind: p.kind,
      description: p.description,
      draft,
      prompt,
      defaultPrompt: base,
      edited: prompt !== base,
    };
  });
  for (const c of unlocked ? s.custom : []) {
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
      prompt: c.prompt.trim() ? c.prompt : (base.prompt ?? ""),
      defaultPrompt: base.prompt ?? "",
      edited: c.prompt.trim() !== "" && c.prompt !== base.prompt,
    });
  }
  return out;
}

// The selected purpose, falling back to the default and then the first one.
export function selectedPurpose(
  s: PurposeSettings,
  builtins: Purpose[],
  unlocked: boolean,
): PurposeOption | null {
  const options = purposeOptions(s, builtins, unlocked);
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

// The body fields to send: an own prompt goes with the prompt password (the server refuses
// it otherwise); the default prompt needs none.
export function withPromptCode(
  r: PurposeRequest,
  promptCode: string,
): PurposeRequest & { prompt_code?: string } {
  return r.prompt === undefined ? r : { ...r, prompt_code: promptCode };
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

// ---- Prompt management lock ------------------------------------------------
// The password is checked by the server and kept for this tab only (sessionStorage), so a
// shared computer does not stay unlocked. One copy for the whole app.

export type PromptAccess = {
  // The built-in purposes; with their prompt text only when unlocked.
  purposes: Purpose[];
  unlocked: boolean;
  loaded: boolean;
  error: string;
};

const CODE_STORAGE = "lab-prompt-code";

function readCode(): string {
  try {
    return sessionStorage.getItem(CODE_STORAGE) ?? "";
  } catch {
    return "";
  }
}

function writeCode(code: string): void {
  try {
    if (code) sessionStorage.setItem(CODE_STORAGE, code);
    else sessionStorage.removeItem(CODE_STORAGE);
  } catch {
    /* storage unavailable: unlocked for this page view only */
  }
}

let promptCode = readCode();
let access: PromptAccess = { purposes: [], unlocked: false, loaded: false, error: "" };
let loading: Promise<void> | null = null;
const accessListeners = new Set<(a: PromptAccess) => void>();

function setAccess(next: PromptAccess): void {
  access = next;
  accessListeners.forEach((fn) => fn(next));
}

async function loadAccess(): Promise<void> {
  if (promptCode) {
    try {
      const purposes = await unlockPurposes(promptCode);
      return setAccess({ purposes, unlocked: true, loaded: true, error: "" });
    } catch {
      promptCode = "";
      writeCode("");
    }
  }
  try {
    setAccess({ purposes: await listPurposes(), unlocked: false, loaded: true, error: "" });
  } catch (e) {
    setAccess({ ...access, loaded: true, error: (e as Error).message });
  }
}

// The password to send with an own prompt; empty while locked.
export const currentPromptCode = (): string => (access.unlocked ? promptCode : "");

// Rejects with the server's message when the password is wrong.
export async function unlockPrompts(code: string): Promise<void> {
  const purposes = await unlockPurposes(code.trim());
  promptCode = code.trim();
  writeCode(promptCode);
  setAccess({ purposes, unlocked: true, loaded: true, error: "" });
}

export function lockPrompts(): void {
  promptCode = "";
  writeCode("");
  setAccess({
    ...access,
    unlocked: false,
    purposes: access.purposes.map(({ prompt: _hidden, ...rest }) => rest),
  });
}

export function usePromptAccess(): PromptAccess {
  const [state, setState] = useState(access);
  useEffect(() => {
    accessListeners.add(setState);
    setState(access);
    loading ??= loadAccess();
    return () => {
      accessListeners.delete(setState);
    };
  }, []);
  return state;
}

export function newPurpose(base: Purpose): CustomPurpose {
  return {
    uid:
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).slice(2, 10),
    name: "",
    base: base.id,
    prompt: base.prompt ?? "",
  };
}
