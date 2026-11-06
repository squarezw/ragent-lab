import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  AudioLines,
  Blocks,
  ChevronRight,
  Github,
  LayoutDashboard,
  Menu,
  ScanLine,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { api } from "./api";
import type { Experiment, Page } from "./types";
import { Brand, ErrorMessage } from "./components/UI";
import Overview from "./pages/Overview";
import Chunking from "./pages/Chunking";
import Embedding from "./pages/Embedding";
import Vision from "./pages/Vision";
import Models from "./pages/Models";
const icons = { chunking: Blocks, embedding: AudioLines, vision: ScanLine };
function currentPage(): Page {
  const key = location.hash.slice(1);
  return ["chunking", "embedding", "vision", "models"].includes(key)
    ? (key as Page)
    : "overview";
}
export default function App() {
  const [page, setPage] = useState<Page>(currentPage);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [error, setError] = useState("");
  const [menu, setMenu] = useState(false);
  const [visionOpened, setVisionOpened] = useState(page === "vision");
  useEffect(() => {
    if (page === "vision") setVisionOpened(true);
  }, [page]);
  useEffect(() => {
    api<Experiment[]>("/experiments")
      .then(setExperiments)
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    const change = () => {
      setPage(currentPage());
      setMenu(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  const navigate = (next: Page) => {
    location.hash = next;
    setMenu(false);
  };
  const title =
    page === "overview"
      ? "实验概览"
      : (experiments.find((e) => e.id === page)?.name ??
        {
          chunking: "文本分块",
          embedding: "Embedding 对比",
          vision: "图像理解",
          models: "模型设置",
        }[page]);
  return (
    <div className="app-shell">
      {menu && (
        <button
          className="sidebar-backdrop"
          aria-label="关闭导航"
          onClick={() => setMenu(false)}
        />
      )}
      <aside className={`sidebar ${menu ? "open" : ""}`}>
        <Brand />
        <button
          className="mobile-close"
          aria-label="关闭导航"
          onClick={() => setMenu(false)}
        >
          <X size={20} />
        </button>
        <div className="workspace-label">
          WORKSPACE <span>个人实验空间</span>
        </div>
        <nav aria-label="实验导航">
          <a
            href="#overview"
            className={`nav-item ${page === "overview" ? "active" : ""}`}
            onClick={() => setMenu(false)}
          >
            <LayoutDashboard size={18} />
            实验概览{page === "overview" && <span className="active-dot" />}
          </a>
          <div className="nav-caption">
            实验项目{" "}
            <span>{experiments.length.toString().padStart(2, "0")}</span>
          </div>
          {experiments.map((exp) => {
            const Icon = icons[exp.id as keyof typeof icons] ?? Blocks;
            return (
              <a
                key={exp.id}
                href={`#${exp.id}`}
                className={`nav-item ${page === exp.id ? "active" : ""}`}
                onClick={() => setMenu(false)}
              >
                <Icon size={18} />
                <span>{exp.name}</span>
                {exp.status === "planned" ? (
                  <small className="nav-soon">SOON</small>
                ) : page === exp.id ? (
                  <span className="active-dot" />
                ) : null}
              </a>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <a
            href="#models"
            className={`nav-item sidebar-settings ${page === "models" ? "active" : ""}`}
            aria-current={page === "models" ? "page" : undefined}
            onClick={() => setMenu(false)}
          >
            <SlidersHorizontal size={17} />
            <span>模型设置</span>
            {page === "models" && <span className="active-dot" />}
          </a>
          <div className="sidebar-note">
            <span className="tiny-spark">✳</span>
            <strong>让想法，经过验证。</strong>
            <p>
              在这里探索模型能力，
              <br />
              找到适合业务的 AI 方案。
            </p>
            <a
              href="https://github.com/squarezw/ragent-lab"
              target="_blank"
              rel="noreferrer"
            >
              参与构建 <ArrowUpRight size={14} />
            </a>
          </div>
          <a
            className="github-link"
            href="https://github.com/squarezw/ragent-lab"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={17} />
            GitHub Repository
            <ArrowUpRight size={14} />
          </a>
          <div className="sidebar-version">
            <span className="live-dot" />
            LAB v2.0 <span>持续探索中</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <div className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-menu"
              aria-label="打开导航"
              onClick={() => setMenu(true)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{title}</strong>
          </div>
          <div className="topbar-right">
            <span className="environment">
              <span className="live-dot" />
              实验环境
            </span>
            <span className="avatar">R</span>
          </div>
        </div>
        <main>
          <ErrorMessage message={error} />
          <div hidden={page !== "overview"}>
            {page === "overview" && (
              <Overview experiments={experiments} navigate={navigate} />
            )}
          </div>
          <div hidden={page !== "chunking"}>
            <Chunking />
          </div>
          <div hidden={page !== "embedding"}>
            <Embedding />
          </div>
          {/* Kept mounted once opened, so visiting 模型设置 keeps the drawing and results. */}
          <div hidden={page !== "vision"}>{visionOpened && <Vision />}</div>
          <div hidden={page !== "models"}>
            {page === "models" && <Models />}
          </div>
        </main>
        <footer className="main-footer">
          <span>
            RAGENT LAB <i>/</i> 从实验，到更好的 AI 应用。
          </span>
          <span>Built for curiosity.</span>
        </footer>
      </div>
    </div>
  );
}
