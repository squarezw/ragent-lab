import { api } from "../api";
import type {
  Geometry, ImageRef, Job, Sample, UploadedImage, VisionModel,
} from "./types";

export const JOB_GONE = "任务已失效，请重新运行。";
const NETWORK_ERROR = "连接中断或请求超时，请检查网络后重试。";
const UNAVAILABLE = "服务暂时不可用，请稍后重试。";

async function detailOf(response: Response): Promise<string> {
  const data = await response.json().catch(() => null);
  return typeof data?.detail === "string" ? data.detail : UNAVAILABLE;
}

export const listModels = () => api<VisionModel[]>("/vision/models");
export const listSamples = () => api<Sample[]>("/vision/samples");

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
  if (!response.ok) throw new Error(await detailOf(response));
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

export async function pollJob<T>(
  id: string,
  { signal, intervalMs = 2000 }: { signal?: AbortSignal; intervalMs?: number } = {},
): Promise<T> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    let response: Response;
    try {
      response = await fetch(`/api/vision/jobs/${encodeURIComponent(id)}`, {
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30000)]) : AbortSignal.timeout(30000),
      });
    } catch (err) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      throw new Error(NETWORK_ERROR);
    }
    if (response.status === 404) throw new Error(JOB_GONE);
    if (!response.ok) throw new Error(await detailOf(response));
    const job = (await response.json()) as Job<T>;
    if (job.status === "done") return job.result as T;
    if (job.status === "failed") throw new Error(job.error ?? UNAVAILABLE);
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
