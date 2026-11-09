import { useEffect, useRef, useState, type ReactNode } from "react";
import { Check, Copy, Download, Maximize2, X } from "lucide-react";

// Splits pretty-printed JSON into colored tokens: keys, strings, numbers, literals.
const TOKEN =
  /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

export function highlightJson(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let n = 0;
  for (const m of text.matchAll(TOKEN)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    const cls = m[1]
      ? m[2]
        ? "json-key"
        : "json-string"
      : m[3]
        ? "json-literal"
        : "json-number";
    out.push(
      <span key={n++} className={cls}>
        {m[1] ?? m[0]}
      </span>,
    );
    if (m[2]) out.push(m[2]);
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function JsonViewer({
  title,
  data,
  filename = "result",
  label = "查看完整 JSON",
}: {
  title: string;
  data: unknown;
  filename?: string;
  label?: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const text = open ? JSON.stringify(data, null, 2) : "";

  useEffect(() => {
    const d = dialog.current;
    if (open && d && !d.open) d.showModal?.();
  }, [open]);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      /* clipboard blocked: the text can still be selected by hand */
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([text], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filename.replace(/[\\/:*?"<>|\s]+/g, "_")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return (
    <>
      <button
        type="button"
        className="button secondary json-open"
        onClick={() => setOpen(true)}
      >
        <Maximize2 size={13} />
        {label}
      </button>
      {open && (
        <dialog
          ref={dialog}
          className="json-dialog"
          aria-label={title}
          onClose={() => setOpen(false)}
          onClick={(e) => {
            // A click on the backdrop (outside the panel) closes the dialog.
            if (e.target === e.currentTarget) e.currentTarget.close();
          }}
        >
          <div className="json-dialog-panel">
            <header>
              <strong>{title}</strong>
              <span>{text.split("\n").length} 行</span>
              <div>
                <button type="button" className="button secondary" onClick={() => void copy()}>
                  {copied ? <Check size={13} /> : <Copy size={13} />}
                  {copied ? "已复制" : "复制"}
                </button>
                <button type="button" className="button secondary" onClick={download}>
                  <Download size={13} />
                  下载
                </button>
                <button
                  type="button"
                  className="button secondary"
                  aria-label="关闭"
                  onClick={() => dialog.current?.close()}
                >
                  <X size={14} />
                </button>
              </div>
            </header>
            <pre className="json-dialog-body">{highlightJson(text)}</pre>
          </div>
        </dialog>
      )}
    </>
  );
}
