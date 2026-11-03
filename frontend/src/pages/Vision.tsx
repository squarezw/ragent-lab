import {
  ArrowUpRight,
  Box,
  FileSearch,
  Focus,
  ScanLine,
  Shapes,
} from "lucide-react";
import { ExternalLink, PageHeader } from "../components/UI";
const capabilities = [
  {
    icon: ScanLine,
    title: "图纸信息提取",
    text: "识别标题栏、零件名称、材料与图纸编号。",
  },
  {
    icon: Focus,
    title: "尺寸与公差理解",
    text: "探索尺寸标注、形位公差及技术要求的理解能力。",
  },
  {
    icon: Shapes,
    title: "结构与特征识别",
    text: "理解视图关系、几何特征和零件结构。",
  },
  {
    icon: FileSearch,
    title: "多模型对照评测",
    text: "以相同图纸与问题，比较模型的准确性和解释能力。",
  },
];
export default function Vision() {
  return (
    <>
      <PageHeader
        eyebrow="EXPERIMENT 03 / VISION UNDERSTANDING"
        title="从看见，到真正理解。"
        description="面向工程图纸的多模态实验空间，正在规划中。"
        action={<span className="planned-badge">PLANNED / 规划中</span>}
      />
      <section className="vision-hero">
        <div>
          <span className="eyebrow">THE NEXT FRONTIER</span>
          <h2>
            一张图纸，
            <br />
            还有多少信息值得发现？
          </h2>
          <p>
            让大模型读取工程语言。我们将从真实图纸出发，探索视觉理解、结构化提取与专业推理的边界。
          </p>
          <span className="vision-status">
            <span />
            实验筹备中 · 暂未开放上传和推理
          </span>
        </div>
        <div className="blueprint" aria-hidden="true">
          <span className="blueprint-id">FIG. 03 — VISION LAB</span>
          <svg viewBox="0 0 400 300">
            <defs>
              <pattern
                id="grid"
                width="20"
                height="20"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M20 0H0V20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth=".3"
                />
              </pattern>
            </defs>
            <rect width="400" height="300" fill="url(#grid)" />
            <g fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="m110 130 95-55 95 55v100l-95 55-95-55zM110 130l95 55 95-55M205 185v100" />
              <path
                d="m110 155 95 55 95-55M110 180l95 55 95-55"
                strokeDasharray="4 5"
                opacity=".5"
              />
              <ellipse cx="205" cy="130" rx="35" ry="20" />
              <path
                d="M170 130v30c0 27 70 27 70 0v-30M88 130V75m222 55V75M90 87h220M100 82l-8 10m208-10 8 10M310 140h35m-35 90h35m-12-90v90m-5-80 10-10m-10 80 10 10"
                strokeWidth="1"
              />
              <path
                d="M65 205h65M97 173v65"
                strokeDasharray="4 3"
                strokeWidth=".6"
              />
            </g>
            <g fill="currentColor" fontSize="10" fontFamily="monospace">
              <text x="185" y="76">
                Ø 120
              </text>
              <text x="340" y="190">
                80
              </text>
              <text x="172" y="135">
                M24
              </text>
            </g>
          </svg>
          <div className="blueprint-footer">
            <Box size={15} />
            <span>ENGINEERING DRAWING / 理解每一个细节</span>
          </div>
        </div>
      </section>
      <div className="section-title">
        <div>
          <span className="eyebrow">WHAT WE'RE EXPLORING</span>
          <h2>计划探索的能力</h2>
        </div>
        <span className="muted">从真实业务问题出发</span>
      </div>
      <div className="capability-grid">
        {capabilities.map(({ icon: Icon, title, text }, i) => (
          <article key={title}>
            <span>
              0{i + 1}
              <Icon size={22} />
            </span>
            <h3>{title}</h3>
            <p>{text}</p>
          </article>
        ))}
      </div>
      <div className="lab-note">
        <div className="note-symbol">
          <ScanLine size={20} />
        </div>
        <div>
          <strong>为更多实验，留出空间。</strong>
          <p>
            后续实验将沿用独立页面与 API
            模块，逐步加入新的数据、模型和评估方法。
          </p>
        </div>
        <ExternalLink href="https://github.com/squarezw/ragent-lab/issues">
          提出实验想法
          <ArrowUpRight size={14} />
        </ExternalLink>
      </div>
    </>
  );
}
