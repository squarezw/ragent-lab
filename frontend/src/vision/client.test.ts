import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { pollJob, uploadImage } from "./client";

const json = (status: number, body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status }));
const job = (status: string, extra: object = {}) => ({ id: "j", status, result: null, error: null, ...extra });

// A fetch that never resolves but rejects when its signal aborts.
const hang = (_u: unknown, init?: RequestInit) =>
  new Promise<Response>((_res, rej) =>
    init?.signal?.addEventListener("abort", () => rej(new DOMException("Aborted", "AbortError"))));

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("pollJob", () => {
  it("resolves the result when done, after polling a running job", async () => {
    const f = vi.fn()
      .mockImplementationOnce(() => json(200, job("running")))
      .mockImplementationOnce(() => json(200, job("done", { result: { x: 1 } })));
    vi.stubGlobal("fetch", f);
    const p = pollJob<{ x: number }>("j", { intervalMs: 10 });
    await vi.advanceTimersByTimeAsync(10);
    await expect(p).resolves.toEqual({ x: 1 });
    expect(f).toHaveBeenCalledTimes(2);
  });

  it("rejects with job.error when failed", async () => {
    vi.stubGlobal("fetch", () => json(200, job("failed", { error: "模型响应超时" })));
    await expect(pollJob("j")).rejects.toThrow("模型响应超时");
  });

  it("maps 404 to the expired message without retrying", async () => {
    const f = vi.fn(() => json(404, { detail: "x" }));
    vi.stubGlobal("fetch", f);
    await expect(pollJob("j")).rejects.toThrow("任务已失效，请重新运行。");
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("aborts during sleep and leaves no timers", async () => {
    vi.stubGlobal("fetch", () => json(200, job("running")));
    const ac = new AbortController();
    const p = pollJob("j", { signal: ac.signal, intervalMs: 1000 });
    const assertion = expect(p).rejects.toMatchObject({ name: "AbortError" });
    await vi.advanceTimersByTimeAsync(10);
    ac.abort();
    await assertion;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("aborts during fetch and leaves no timers", async () => {
    vi.stubGlobal("fetch", hang);
    const ac = new AbortController();
    const p = pollJob("j", { signal: ac.signal });
    const assertion = expect(p).rejects.toMatchObject({ name: "AbortError" });
    await vi.advanceTimersByTimeAsync(10);
    ac.abort();
    await assertion;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("times out a hung poll after 30 s and retries", async () => {
    const f = vi.fn()
      .mockImplementationOnce(hang)
      .mockImplementationOnce(() => json(200, job("done", { result: 7 })));
    vi.stubGlobal("fetch", f);
    const p = pollJob<number>("j", { intervalMs: 10 });
    await vi.advanceTimersByTimeAsync(30000 + 10);
    await expect(p).resolves.toBe(7);
  });

  it("retries transient errors up to 3 times, resetting on success", async () => {
    const boom = () => Promise.reject(new TypeError("net"));
    const f = vi.fn()
      .mockImplementationOnce(boom).mockImplementationOnce(boom).mockImplementationOnce(boom)
      .mockImplementationOnce(() => json(200, job("running")))
      .mockImplementationOnce(boom).mockImplementationOnce(boom).mockImplementationOnce(boom)
      .mockImplementationOnce(() => json(200, job("done", { result: "ok" })));
    vi.stubGlobal("fetch", f);
    const p = pollJob<string>("j", { intervalMs: 10 });
    await vi.advanceTimersByTimeAsync(200);
    await expect(p).resolves.toBe("ok");
    expect(f).toHaveBeenCalledTimes(8);
  });

  it("gives up after 4 consecutive transient failures", async () => {
    const f = vi.fn(() => Promise.reject(new TypeError("net")));
    vi.stubGlobal("fetch", f);
    const p = pollJob("j", { intervalMs: 10 });
    const assertion = expect(p).rejects.toThrow("连接中断或请求超时");
    await vi.advanceTimersByTimeAsync(100);
    await assertion;
    expect(f).toHaveBeenCalledTimes(4);
  });
});

describe("uploadImage", () => {
  const file = new File(["x"], "a.png");
  it("explains a bare 413", async () => {
    vi.stubGlobal("fetch", () => Promise.resolve(new Response("<html>", { status: 413 })));
    await expect(uploadImage(file)).rejects.toThrow("文件超过 15 MB，请压缩后重试。");
  });
  it("prefers the backend detail", async () => {
    vi.stubGlobal("fetch", () => json(422, { detail: "不支持的格式" }));
    await expect(uploadImage(file)).rejects.toThrow("不支持的格式");
  });
});
