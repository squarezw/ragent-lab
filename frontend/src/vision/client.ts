import { api } from "../api";
import type { BuiltinInfo } from "./modelSettings";
import type {
  Geometry, ImageRef, Job, Purpose, Sample, UploadedImage, VisionModel,
} from "./types";

export const JOB_GONE = "任务已失效，请重新运行。";
const NETWORK_ERROR = "连接中断或请求超时，请检查网络后重试。";
const TOO_LARGE = "文件超过 15 MB，请压缩后重试。";
const UNAVAILABLE = "服务暂时不可用，请稍后重试。";

async function detailOf(response: Response): Promise<string> {
  const data = await response.json().catch(() => null);
  return typeof data?.detail === "string" ? data.detail : UNAVAILABLE;
}

export const listModels = () => api<VisionModel[]>("/vision/models");
export const listSamples = () => api<Sample[]>("/vision/samples");
export const listPurposes = () => api<Purpose[]>("/vision/purposes");
export const listBuiltin = () => api<BuiltinInfo>("/vision/builtin");
export const checkAccess = (access_code: string) =>
  api<{ ok: boolean }>("/vision/access", { access_code });

export async function uploadImage(file: File): Promise<UploadedImage> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch("/api/vision/images", {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(180000),
    });
  } catch {
    throw new Error(NETWORK_ERROR);
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    if (typeof data?.detail === "string") throw new Error(data.detail);
    throw new Error(response.status === 413 ? TOO_LARGE : UNAVAILABLE);
  }
  return (await response.json()) as UploadedImage;
}

export type SubmitKind = "understand" | "score" | "consensus" | "generate";

export async function submit(kind: SubmitKind, body: unknown): Promise<string> {
  const out = await api<{ job_id: string }>(`/vision/${kind}`, body);
  return out.job_id;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

const MAX_POLL_RETRIES = 3;
const POLL_TIMEOUT_MS = 30000;
const abortError = () => new DOMException("Aborted", "AbortError");

export async function pollJob<T>(
  id: string,
  { signal, intervalMs = 2000 }: { signal?: AbortSignal; intervalMs?: number } = {},
): Promise<T> {
  let failures = 0;
  for (;;) {
    if (signal?.aborted) throw abortError();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), POLL_TIMEOUT_MS);
    const forward = () => ctrl.abort();
    signal?.addEventListener("abort", forward, { once: true });
    let status = 0;
    let data: unknown = null;
    let transient = false;
    try {
      const response = await fetch(`/api/vision/jobs/${encodeURIComponent(id)}`, {
        signal: ctrl.signal,
      });
      status = response.status;
      data = await response.json().catch((err) => {
        if (ctrl.signal.aborted) throw err;
        return null;
      });
    } catch {
      transient = true;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", forward);
    }
    if (signal?.aborted) throw abortError();
    if (transient) {
      failures += 1;
      if (failures > MAX_POLL_RETRIES) throw new Error(NETWORK_ERROR);
      await sleep(intervalMs, signal);
      continue;
    }
    failures = 0;
    if (status === 404) throw new Error(JOB_GONE);
    const body = data as (Job<T> & { detail?: unknown }) | null;
    if (status < 200 || status >= 300)
      throw new Error(typeof body?.detail === "string" ? body.detail : UNAVAILABLE);
    if (!body) throw new Error(UNAVAILABLE);
    if (body.status === "done") return body.result as T;
    if (body.status === "failed") throw new Error(body.error ?? UNAVAILABLE);
    await sleep(intervalMs, signal);
  }
}

export async function exportFile(
  geometry: Geometry,
  format: "pdf" | "dxf",
  filename: string,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/vision/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ geometry, format, filename }),
      signal: AbortSignal.timeout(180000),
    });
  } catch {
    throw new Error(NETWORK_ERROR);
  }
  if (!response.ok) throw new Error(await detailOf(response));
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.replace(/\.(pdf|dxf)$/i, "") + `.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function imageUrl(ref: ImageRef): string {
  const id = encodeURIComponent(ref.id);
  return ref.kind === "sample"
    ? `/api/vision/samples/${id}/image`
    : `/api/vision/images/${id}`;
}
