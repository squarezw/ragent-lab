import {
  ArrowRight,
  ArrowUpRight,
  AudioLines,
  Blocks,
  CircleDot,
  Command,
  Layers3,
  ScanLine,
  Sparkles,
} from "lucide-react";
import type { Experiment, Page } from "../types";
import { PageHeader } from "../components/UI";
const icons = { chunking: Blocks, embedding: AudioLines, vision: ScanLine };
export default function Overview({
  experiments,
  navigate,
}: {
  experiments: Experiment[];
  navigate: (page: Page) => void;
}) {
  return (
    <>
      <PageHeader
        eyebrow="YOUR IDEAS. REAL EXPERIMENTS."
        title="把想法，放进实验室。"
        description="从文本到视觉，拆解 AI 能力，验证每一种可能。"
        action={
          <div className="header-chip">
            <span className="live-dot" />
            开放探索 · 持续进化
          </div>
        }
      />
      <section className="hero">
        <div className="hero-content">
          <div className="hero-kicker">
            <Sparkles size={15} />
            RAG & BEYOND
          </div>
          <h2>
            更好的 AI 应用，
            <br />
            从理解每个环节开始。
          </h2>
          <p>
            调整一次分块，比较一组向量。
            <br />
            让抽象的模型能力，变成看得见的实验结果。
          </p>
          <button
            className="button hero-button"
            onClick={() => navigate("chunking")}
          >
            开始第一个实验 <ArrowRight size={16} />
          </button>
          <div className="hero-caption">
            <span>01 / 观察</span>
            <span>02 / 比较</span>
            <span>03 / 发现</span>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orbital orbital-one" />
          <div className="orbital orbital-two" />
          <div className="art-grid" />
          <div className="diagram-node node-input">
            <span className="node-icon">
              <Command size={19} />
            </span>
            <div>
              <small>INPUT</small>
              <strong>一个想法</strong>
            </div>
            <span className="node-port" />
          </div>
          <div className="diagram-node node-process">
            <div className="node-top">
              <Layers3 size={21} />
              <span>EXPERIMENT</span>
              <i />
            </div>
            <div className="mini-bars">
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
            <div className="node-bottom">
              探索 · 调整 · 验证 <CircleDot size={13} />
            </div>
          </div>
          <div className="diagram-node node-result">
            <span className="result-glyph">✳</span>
            <div>
              <small>INSIGHT</small>
              <strong>新的发现</strong>
            </div>
            <span className="result-check">↗</span>
          </div>
          <div className="art-connector connector-one" />
          <div className="art-connector connector-two" />
          <span className="art-label label-one">make it measurable.</span>
          <span className="art-label label-two">∞ possibilities</span>
        </div>
      </section>
      <div className="section-title">
        <div>
          <span className="eyebrow">EXPLORE THE LAB</span>
          <h2>
            选择你的实验方向{" "}
            <span>{experiments.length.toString().padStart(2, "0")}</span>
          </h2>
        </div>
        <span className="muted">小步探索，让结果说话。</span>
      </div>
      <div className="experiment-grid">
        {experiments.map((exp, index) => {
          const Icon = icons[exp.id as keyof typeof icons] ?? Blocks;
          return (
            <button
              key={exp.id}
              className={`experiment-card card-${exp.id}`}
              onClick={() => navigate(exp.id as Page)}
            >
              <div className="card-top">
                <span className="experiment-icon">
                  <Icon size={25} />
                </span>
                <span
                  className={`status-tag ${exp.status === "planned" ? "planned" : ""}`}
                >
                  {exp.status === "ready" ? (
                    <>
                      <span />
                      可用
                    </>
                  ) : (
                    "规划中"
                  )}
                </span>
              </div>
              <div className="card-category">
                EXPERIMENT / {String(index + 1).padStart(2, "0")}{" "}
                <span>{exp.category}</span>
              </div>
              <h3>
                {exp.name}
                <ArrowUpRight size={20} />
              </h3>
              <p>{exp.description}</p>
              <div className="card-bottom">
                <span>{exp.tag}</span>
                <span>
                  {exp.status === "ready" ? "进入实验" : "查看规划"}
                  <ArrowRight size={15} />
                </span>
              </div>
            </button>
          );
        })}
      </div>
      <div className="lab-note">
        <div className="note-symbol">
          <Sparkles size={20} />
        </div>
        <div>
          <strong>这只是探索的开始。</strong>
          <p>实验目录将持续扩展，连接更多模型、模态与真实业务场景。</p>
        </div>
        <span className="note-end">
          ALWAYS IN PROGRESS <span>↗</span>
        </span>
      </div>
    </>
  );
}
