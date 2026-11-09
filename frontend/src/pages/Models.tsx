import { useEffect, useState } from "react";
import { ArrowRight, KeyRound, ListChecks, Plus, Server, Trash2 } from "lucide-react";
import { ErrorMessage, ExternalLink, PageHeader, Panel } from "../components/UI";
import { checkAccess, listBuiltin, listModels, listPurposes } from "../vision/client";
import {
  PRESETS,
  buildOptions,
  catalogChoice,
  hostOf,
  newCustom,
  useModelSettings,
  type BuiltinInfo,
  type CustomModel,
  type Platform,
} from "../vision/modelSettings";
import {
  KIND_LABELS,
  MAX_PROMPT_CHARS,
  newPurpose,
  purposeOptions,
  usePurposeSettings,
  withPrompt,
  type CustomPurpose,
  type PurposeOption,
} from "../vision/purposes";
import { GATEWAY_NAMES, type Purpose, type VisionModel } from "../vision/types";

const API_LABELS = {
  chat: "OpenAI Chat Completions",
  responses: "OpenAI Responses",
  anthropic: "Anthropic Messages",
} as const;
const JSON_LABELS = {
  schema: "发送 JSON Schema（结构化输出）",
  object: "JSON 模式（json_object）",
  prompt: "只在提示词里要求 JSON",
} as const;
const PLATFORM_KEY_URL: Record<Platform, string> = {
  openrouter: "https://openrouter.ai/keys",
  kie: "https://kie.ai/api-key",
};

function CustomForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: CustomModel;
  onSave: (m: CustomModel) => void;
  onCancel: () => void;
}) {
  const [m, setM] = useState(initial);
  const [error, setError] = useState("");
  const preset = PRESETS.find((p) => p.id === m.preset) ?? PRESETS[PRESETS.length - 1];
  const set = <K extends keyof CustomModel>(k: K, v: CustomModel[K]) =>
    setM((cur) => ({ ...cur, [k]: v }));
  function pickPreset(id: string) {
    const p = PRESETS.find((x) => x.id === id) ?? preset;
    setM((cur) => ({
      ...cur,
      preset: p.id,
      api: p.api,
      url: p.url,
      json_mode: p.json_mode,
      auth: p.auth,
    }));
  }
  function save() {
    if (!m.name.trim()) return setError("请填写显示名称。");
    if (!m.url.startsWith("https://") || hostOf(m.url) === m.url)
      return setError("接口地址必须是完整的 https:// 地址。");
    if (!m.api_key.trim()) return setError("请填写这个模型的 API Key。");
    onSave({ ...m, name: m.name.trim(), url: m.url.trim(), model: m.model.trim() });
  }
  return (
    <div className="model-form">
      <div className="parameter-grid">
        <div>
          <label htmlFor="cm-preset">从平台预设开始</label>
          <select
            id="cm-preset"
            value={m.preset}
            onChange={(e) => pickPreset(e.target.value)}
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="cm-name">显示名称</label>
          <input
            id="cm-name"
            value={m.name}
            placeholder="例如 DeepSeek 官方"
            maxLength={60}
            onChange={(e) => set("name", e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="cm-model">模型名</label>
          <input
            id="cm-model"
            value={m.model}
            placeholder={preset.modelHint}
            maxLength={200}
            spellCheck={false}
            onChange={(e) => set("model", e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="cm-key">API Key</label>
          <input
            id="cm-key"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={m.api_key}
            onChange={(e) => set("api_key", e.target.value)}
          />
        </div>
      </div>
      <label htmlFor="cm-url">接口地址</label>
      <input
        id="cm-url"
        value={m.url}
        spellCheck={false}
        maxLength={500}
        onChange={(e) => set("url", e.target.value)}
      />
      {preset.note && <p className="field-help">{preset.note}</p>}
      <div className="parameter-grid">
        <div>
          <label htmlFor="cm-api">接口格式</label>
          <select
            id="cm-api"
            value={m.api}
            onChange={(e) => set("api", e.target.value as CustomModel["api"])}
          >
            {Object.entries(API_LABELS).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="cm-json">JSON 输出方式</label>
          <select
            id="cm-json"
            value={m.json_mode}
            onChange={(e) =>
              set("json_mode", e.target.value as CustomModel["json_mode"])
            }
          >
            {Object.entries(JSON_LABELS).map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="cm-auth">认证方式</label>
          <select
            id="cm-auth"
            value={m.auth}
            onChange={(e) => set("auth", e.target.value as CustomModel["auth"])}
          >
            <option value="bearer">Authorization: Bearer</option>
            <option value="x-api-key">x-api-key（Anthropic 官方）</option>
          </select>
        </div>
        <div>
          <label htmlFor="cm-max">最大输出 token</label>
          <input
            id="cm-max"
            type="number"
            min={1000}
            max={128000}
            step={1000}
            value={m.max_tokens}
            onChange={(e) =>
              set("max_tokens", Math.min(128000, Math.max(1000, Number(e.target.value) || 16000)))
            }
          />
        </div>
      </div>
      <p className="field-help">
        服务器只会调用公网 https 地址（不能指向本机或内网）。
        {preset.keyUrl && (
          <ExternalLink href={preset.keyUrl}>获取 Key</ExternalLink>
        )}
      </p>
      <ErrorMessage message={error} />
      <div className="model-form-actions">
        <button type="button" className="button secondary" onClick={onCancel}>
          取消
        </button>
        <button type="button" className="button primary" onClick={save}>
          保存模型
        </button>
      </div>
    </div>
  );
}

// Edit a custom purpose (name, structure, prompt) or only the prompt of a built-in one.
function PurposeForm({
  builtins,
  initial,
  fixedName,
  onSave,
  onCancel,
}: {
  builtins: Purpose[];
  initial: CustomPurpose;
  fixedName?: string;
  onSave: (p: CustomPurpose) => void;
  onCancel: () => void;
}) {
  const [p, setP] = useState(initial);
  const [error, setError] = useState("");
  const baseOf = (id: string) => builtins.find((b) => b.id === id);
  function pickBase(id: string) {
    setP((cur) => ({
      ...cur,
      base: id,
      // Swap in the new default unless the visitor already wrote their own prompt.
      prompt:
        cur.prompt === baseOf(cur.base)?.prompt
          ? (baseOf(id)?.prompt ?? cur.prompt)
          : cur.prompt,
    }));
  }
  function save() {
    if (!fixedName && !p.name.trim()) return setError("请填写用途名称。");
    if (!p.prompt.trim()) return setError("请填写提示词。");
    onSave({ ...p, name: p.name.trim() });
  }
  return (
    <div className="model-form purpose-form">
      {fixedName ? (
        <p className="field-help">
          编辑「{fixedName}」的提示词。保存后，图像理解页选择这个用途时会使用它。
        </p>
      ) : (
        <div className="parameter-grid">
          <div>
            <label htmlFor="pp-name">用途名称</label>
            <input
              id="pp-name"
              value={p.name}
              maxLength={40}
              placeholder="例如 电路原理图"
              onChange={(e) => setP({ ...p, name: e.target.value })}
            />
          </div>
          <div>
            <label htmlFor="pp-base">输出结构</label>
            <select
              id="pp-base"
              value={p.base}
              onChange={(e) => pickBase(e.target.value)}
            >
              {builtins.map((b) => (
                <option key={b.id} value={b.id}>
                  {KIND_LABELS[b.kind]} · 起始提示词「{b.name}」
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
      <label htmlFor="pp-prompt">提示词</label>
      <textarea
        id="pp-prompt"
        className="purpose-textarea"
        value={p.prompt}
        rows={14}
        maxLength={MAX_PROMPT_CHARS}
        spellCheck={false}
        onChange={(e) => setP({ ...p, prompt: e.target.value })}
      />
      <p className="field-help">
        系统会在末尾固定附加：图中文字只当作数据，以及「
        {KIND_LABELS[baseOf(p.base)?.kind ?? "general"]}」的 JSON 格式要求，不用写进提示词。
        {p.prompt.length} / {MAX_PROMPT_CHARS} 字
      </p>
      <ErrorMessage message={error} />
      <div className="model-form-actions">
        {fixedName && p.prompt !== baseOf(p.base)?.prompt && (
          <button
            type="button"
            className="button secondary"
            onClick={() => setP({ ...p, prompt: baseOf(p.base)?.prompt ?? p.prompt })}
          >
            恢复默认
          </button>
        )}
        <button type="button" className="button secondary" onClick={onCancel}>
          取消
        </button>
        <button type="button" className="button primary" onClick={save}>
          保存
        </button>
      </div>
    </div>
  );
}

function PurposePanel() {
  const [builtins, setBuiltins] = useState<Purpose[]>([]);
  const [settings, update] = usePurposeSettings();
  const [editing, setEditing] = useState<{ key: string; draft: CustomPurpose } | null>(
    null,
  );
  const [error, setError] = useState("");
  useEffect(() => {
    listPurposes()
      .then(setBuiltins)
      .catch((e) => setError(e.message));
  }, []);
  const options = purposeOptions(settings, builtins);

  function edit(o: PurposeOption) {
    const custom = settings.custom.find((c) => `custom:${c.uid}` === o.key);
    setEditing({
      key: o.key,
      draft: custom ?? { uid: o.base, name: o.name, base: o.base, prompt: o.prompt },
    });
  }
  function save(o: PurposeOption | undefined, p: CustomPurpose) {
    if (o?.source === "builtin") update((s) => withPrompt(s, o, p.prompt));
    else
      update((s) => ({
        ...s,
        custom: s.custom.some((c) => c.uid === p.uid)
          ? s.custom.map((c) => (c.uid === p.uid ? p : c))
          : [...s.custom, p],
        selected: o ? s.selected : `custom:${p.uid}`,
      }));
    setEditing(null);
  }
  const form = (o?: PurposeOption) =>
    editing && (
      <PurposeForm
        key={editing.key}
        builtins={builtins}
        initial={editing.draft}
        fixedName={o?.source === "builtin" ? o.name : undefined}
        onSave={(p) => save(o, p)}
        onCancel={() => setEditing(null)}
      />
    );

  return (
    <Panel
      title="理解用途与提示词"
      meta={
        <button
          type="button"
          className="text-button"
          disabled={!builtins.length}
          onClick={() => {
            const draft = newPurpose(builtins.find((b) => b.kind === "general") ?? builtins[0]);
            setEditing({ key: `new:${draft.uid}`, draft });
          }}
        >
          <Plus size={13} /> 添加用途
        </button>
      }
    >
      <p className="field-help">
        <ListChecks size={12} /> 图像理解页按这里选的用途读图。内置用途的提示词可以修改；也可以新建自己的用途，
        比如电路图、地图、医学影像。工程图纸结构的用途才有「还原」步骤，其他用途输出通用结构，
        同样可以做一致性对照。设置只保存在本浏览器。
      </p>
      <ErrorMessage message={error} />
      <div className="purpose-list">
        {options.map((o) =>
          editing?.key === o.key ? (
            <div key={o.key}>{form(o)}</div>
          ) : (
            <div key={o.key} className="custom-model purpose-row">
              <div className="purpose-name">
                <strong>{o.name}</strong>
                <small>
                  {" "}
                  · {o.source === "builtin" ? (o.edited ? "提示词已修改" : "默认提示词") : "自定义"}
                </small>
              </div>
              <small>{o.description}</small>
              <span>
                <button
                  type="button"
                  className="text-button"
                  aria-label={`编辑 ${o.name}`}
                  onClick={() => edit(o)}
                >
                  编辑
                </button>
                {o.source === "builtin" && o.edited && (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => update((s) => withPrompt(s, o, o.defaultPrompt))}
                  >
                    恢复默认
                  </button>
                )}
                {o.source === "custom" && (
                  <button
                    type="button"
                    className="text-button"
                    aria-label={`删除 ${o.name}`}
                    onClick={() =>
                      update((s) => ({
                        ...s,
                        custom: s.custom.filter((c) => `custom:${c.uid}` !== o.key),
                      }))
                    }
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </span>
            </div>
          ),
        )}
      </div>
      {editing?.key.startsWith("new:") && form()}
    </Panel>
  );
}

export default function Models() {
  const [settings, update] = useModelSettings();
  const [catalog, setCatalog] = useState<VisionModel[]>([]);
  const [builtin, setBuiltin] = useState<BuiltinInfo | null>(null);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState<CustomModel | null>(null);
  const [access, setAccess] = useState<"" | "checking" | "ok" | string>("");

  useEffect(() => {
    listModels()
      .then(setCatalog)
      .catch((e) => setError(e.message));
    listBuiltin()
      .then(setBuiltin)
      .catch(() => setBuiltin(null));
  }, []);

  const options = buildOptions(settings, catalog, builtin);
  const ready = options.filter((o) => o.ready).length;

  async function verify() {
    setAccess("checking");
    try {
      await checkAccess(settings.accessCode.trim());
      setAccess("ok");
    } catch (e) {
      setAccess((e as Error).message);
    }
  }
  function saveCustom(m: CustomModel) {
    update((s) => ({
      ...s,
      custom: s.custom.some((c) => c.uid === m.uid)
        ? s.custom.map((c) => (c.uid === m.uid ? m : c))
        : [...s.custom, m],
    }));
    setEditing(null);
  }

  return (
    <>
      <PageHeader
        eyebrow="SETTINGS / MODELS"
        title="模型设置"
        description="选择通过哪个平台调用模型，解锁服务器内置模型，接入你自己的模型接口，并设置图像理解的用途和提示词。设置只保存在本浏览器。"
        action={
          <a className="header-chip" href="#vision">
            <span className="live-dot" />
            {ready} 个模型可用 · 去图像理解
            <ArrowRight size={13} />
          </a>
        }
      />
      <ErrorMessage message={error} />
      <div className="models-page">
        <Panel title="代理平台 Key" meta={<span className="panel-index">01</span>}>
          <div className="parameter-grid">
            {(Object.keys(GATEWAY_NAMES) as Platform[]).map((p) => (
              <div key={p}>
                <label htmlFor={`key-${p}`}>
                  <KeyRound size={13} /> {GATEWAY_NAMES[p]} Key
                </label>
                <input
                  id={`key-${p}`}
                  aria-label={`${GATEWAY_NAMES[p]} Key`}
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  value={settings.keys[p]}
                  onChange={(e) =>
                    update((s) => ({ ...s, keys: { ...s.keys, [p]: e.target.value } }))
                  }
                />
                <p className="field-help">
                  <ExternalLink href={PLATFORM_KEY_URL[p]}>获取 Key</ExternalLink>
                </p>
              </div>
            ))}
          </div>
          <label className="vision-check">
            <input
              type="checkbox"
              checked={settings.rememberKeys}
              onChange={(e) =>
                update((s) => ({ ...s, rememberKeys: e.target.checked }))
              }
            />
            在本浏览器保存 Key（含访问码和自定义模型的 Key）
          </label>
          <p className="field-help">
            Key 只保存在本浏览器的 localStorage，调用时随请求发给对应平台，服务器不保存、不记录。
            不勾选时，关闭页面后需要重新填写。
          </p>
        </Panel>

        <Panel
          title="内置模型"
          meta={
            <span className="panel-index">
              <Server size={12} /> 02
            </span>
          }
        >
          {builtin?.enabled ? (
            <>
              <div className="access-row">
                <div>
                  <label htmlFor="access-code">访问码</label>
                  <input
                    id="access-code"
                    type="password"
                    autoComplete="off"
                    value={settings.accessCode}
                    onChange={(e) => {
                      setAccess("");
                      update((s) => ({ ...s, accessCode: e.target.value }));
                    }}
                  />
                </div>
                <button
                  type="button"
                  className="button secondary"
                  disabled={!settings.accessCode.trim() || access === "checking"}
                  onClick={() => void verify()}
                >
                  {access === "checking" ? "验证中…" : "验证"}
                </button>
              </div>
              {access === "ok" && <p className="field-help ok">访问码正确，可以使用内置模型。</p>}
              {access && access !== "ok" && access !== "checking" && (
                <ErrorMessage message={access} />
              )}
              <p className="field-help">
                内置模型由服务器付费，每个访问者每小时最多调用 {builtin.per_hour} 次。
              </p>
              <div className="model-toggle-list">
                {builtin.models.map((m) => (
                  <label key={m.id} className="vision-check">
                    <input
                      type="checkbox"
                      checked={settings.builtin[m.id]?.enabled ?? true}
                      onChange={(e) =>
                        update((s) => ({
                          ...s,
                          builtin: { ...s.builtin, [m.id]: { enabled: e.target.checked } },
                        }))
                      }
                    />
                    <strong>{m.name}</strong>
                    {m.notes && <small> · {m.notes}</small>}
                  </label>
                ))}
              </div>
            </>
          ) : (
            <p className="field-help">
              这台服务器没有开放内置模型。管理员在 API 环境变量中设置
              <code>VISION_BUILTIN_MODELS</code> 和 <code>LAB_ACCESS_CODE</code>
              后，这里会出现可用的模型。
            </p>
          )}
        </Panel>

        <Panel title="模型目录" meta={<span className="panel-index">03</span>}>
          <p className="field-help">
            实验预设的模型。勾选后出现在图像理解页和评判模型中；同一个模型可以选择经 OpenRouter
            或 Kie.ai 调用（Kie.ai 只提供其中一部分）。
          </p>
          <div className="vision-table-wrap">
            <table className="vision-table">
              <thead>
                <tr>
                  <th>启用</th>
                  <th>模型</th>
                  <th>调用平台</th>
                </tr>
              </thead>
              <tbody>
                {catalog.map((m) => {
                  const choice = catalogChoice(settings, m);
                  const served = m.gateways ?? ["openrouter"];
                  const set = (patch: Partial<typeof choice>) =>
                    update((s) => ({
                      ...s,
                      catalog: { ...s.catalog, [m.id]: { ...choice, ...patch } },
                    }));
                  return (
                    <tr key={m.id}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`启用 ${m.name}`}
                          checked={choice.enabled}
                          onChange={(e) => set({ enabled: e.target.checked })}
                        />
                      </td>
                      <td>
                        <strong>{m.name}</strong>
                        <small className="mono"> {m.id}</small>
                      </td>
                      <td>
                        <select
                          aria-label={`${m.name} 调用平台`}
                          value={choice.via}
                          disabled={served.length < 2}
                          onChange={(e) => set({ via: e.target.value as Platform })}
                        >
                          {served.map((g) => (
                            <option key={g} value={g}>
                              {GATEWAY_NAMES[g]}
                            </option>
                          ))}
                        </select>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title="自定义模型"
          meta={
            <button
              type="button"
              className="text-button"
              onClick={() => setEditing(newCustom(PRESETS[0]))}
            >
              <Plus size={13} /> 添加模型
            </button>
          }
        >
          {settings.custom.length === 0 && !editing && (
            <p className="field-help">
              接入任意兼容的模型接口：DeepSeek、Kimi、Qwen、GLM 的官方 API，或 OpenRouter /
              Kie.ai 上不在目录里的模型。选择平台预设会自动填好地址和格式。
            </p>
          )}
          <div className="custom-model-list">
            {settings.custom.map((c) =>
              editing?.uid === c.uid ? null : (
                <div key={c.uid} className="custom-model">
                  <label className="vision-check">
                    <input
                      type="checkbox"
                      aria-label={`启用 ${c.name}`}
                      checked={c.enabled}
                      onChange={(e) =>
                        saveCustom({ ...c, enabled: e.target.checked })
                      }
                    />
                    <strong>{c.name}</strong>
                  </label>
                  <small className="mono">
                    {hostOf(c.url)}
                    {c.model ? ` · ${c.model}` : ""} · {API_LABELS[c.api]}
                    {c.api_key ? "" : " · 未填 Key"}
                  </small>
                  <span>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => setEditing(c)}
                    >
                      编辑
                    </button>
                    <button
                      type="button"
                      className="text-button"
                      aria-label={`删除 ${c.name}`}
                      onClick={() =>
                        update((s) => ({
                          ...s,
                          custom: s.custom.filter((x) => x.uid !== c.uid),
                        }))
                      }
                    >
                      <Trash2 size={13} />
                    </button>
                  </span>
                </div>
              ),
            )}
          </div>
          {editing && (
            <CustomForm
              key={editing.uid}
              initial={editing}
              onSave={saveCustom}
              onCancel={() => setEditing(null)}
            />
          )}
        </Panel>

        <PurposePanel />
      </div>
    </>
  );
}
