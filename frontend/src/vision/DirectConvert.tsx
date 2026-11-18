// Direct PDF -> DXF: the server copies the PDF's own lines exactly (no model, no redraw) and
// scales them from the drawing's dimension labels. Shown in step 1 for uploaded PDFs.
import { useEffect, useRef, useState } from "react";
import { FileDown, Play, Square } from "lucide-react";
import { ErrorMessage } from "../components/UI";
import { cancelJob, convertedUrl, imageUrl, pollJob, submit } from "./client";
import { svgDataUrl } from "./StepRedraw";
import type { ConvertResult, ImageRef } from "./types";

const SOURCES: Record<string, string> = {
  labels: "按图上的标注尺寸换算",
  given: "按填写的比例",
  paper: "未能确定比例，按图纸原尺寸（毫米）",
};

export default function DirectConvert({
  image,
  imageName,
}: {
  image: ImageRef;
  imageName: string;
}) {
  const [status, setStatus] = useState<"" | "running" | "done">("");
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState("");
  const [scale, setScale] = useState("");
  const job = useRef<{ id: string; ctrl: AbortController } | null>(null);

  useEffect(() => {
    // A new file starts over; leaving the page stops a running conversion.
    setStatus("");
    setResult(null);
    setError("");
    return () => stop();
  }, [image.id]);

  function stop() {
    if (job.current) {
      job.current.ctrl.abort();
      cancelJob(job.current.id);
      job.current = null;
    }
  }

  async function run() {
    const value = scale.trim() ? Number(scale) : null;
    if (value !== null && !(value > 0)) return setError("比例请填写大于 0 的数字，例如 100 表示 1:100");
    stop();
    setStatus("running");
    setError("");
    setResult(null);
    const ctrl = new AbortController();
    try {
      const id = await submit("convert", { image, scale: value });
      job.current = { id, ctrl };
      const out = await pollJob<ConvertResult>(id, { signal: ctrl.signal });
      setResult(out);
      setStatus("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError((e as Error).message);
      setStatus("");
    } finally {
      if (job.current?.ctrl === ctrl) job.current = null;
    }
  }

  const report = result?.report;
  const counts = report
    ? Object.entries(report.entities)
        .filter(([, n]) => n > 0)
        .map(([k, n]) => `${k} ${n}`)
        .join(" · ")
    : "";
  return (
    <div className="direct-convert">
      <div className="direct-convert-row">
        <label htmlFor="direct-scale">比例（可选）</label>
        <input
          id="direct-scale"
          inputMode="decimal"
          placeholder="自动"
          value={scale}
          disabled={status === "running"}
          onChange={(e) => setScale(e.target.value)}
        />
        {status === "running" ? (
          <button type="button" className="button secondary" onClick={() => { stop(); setStatus(""); }}>
            <Square size={11} />
            取消
          </button>
        ) : (
          <button type="button" className="button primary" onClick={() => void run()}>
            <Play size={13} />
            {status === "done" ? "重新转换" : "直接转换"}
          </button>
        )}
      </div>
      <p className="field-help">
        留空则自动：按图上的标注尺寸（如 200、R30）或比例说明（如 1:50）换算到实际尺寸。填 100 表示图纸是
        1:100。
      </p>
      {status === "running" && <p className="field-help">正在转换，通常几秒钟。</p>}
      <ErrorMessage message={error} />
      {report && result && (
        <>
          <div className="vision-compare">
            <figure>
              <figcaption>原图</figcaption>
              <img src={imageUrl(image)} alt={`原图：${imageName}`} />
            </figure>
            <figure aria-label="直接转换预览" className="is-dark">
              <figcaption>直接转换</figcaption>
              <img src={svgDataUrl(result.svg)} alt="直接转换的图纸" />
            </figure>
          </div>
          <ul className="direct-convert-facts">
            <li>
              比例 <strong>{report.scale}</strong>：{SOURCES[report.scale_source] ?? `按比例说明 ${report.scale_source}`}
            </li>
            <li>
              图面范围 {report.sheet[0]} × {report.sheet[1]}（按实际尺寸，单位 mm） · {counts}
            </li>
            {report.centre_lines_from_geometry > 0 && (
              <li>{report.centre_lines_from_geometry} 条虚线按位置判定为中心线（CENTER 层，红色）</li>
            )}
          </ul>
          {report.warnings.map((w) => (
            <p key={w} className="field-help warn">
              {w}
            </p>
          ))}
          <p className="field-help">
            直接转换按原图逐条复制线条，不重绘：原图没有按比例画的地方，转换结果也一样。需要每个尺寸都等于标注时，请选「理解后重绘」。
          </p>
          <div className="vision-actions">
            <a className="button secondary" href={convertedUrl(result.download_id)} download>
              <FileDown size={13} />
              下载 DXF
            </a>
          </div>
        </>
      )}
    </div>
  );
}
