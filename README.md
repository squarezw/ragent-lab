# Ragent Lab

中文 AI 实验工作台：把 RAG 和多模态里的每个环节拆开来调参、对比、验证。

**在线体验：<https://lab.ragents.net>**

![Ragent Lab 首页](docs/images/home.png)

## 实验

| 实验 | 状态 | 内容 |
| --- | --- | --- |
| 文本分块 | 可用 | 11 种分块策略，参数可调，提供分块预览和长度统计 |
| Embedding 对比 | 可用 | 7 个向量模型；按查询相似度排名，用同一组候选文本对比各模型 |
| 图像理解 | 规划中 | 面向工程图纸的视觉理解与信息提取；目前只有规划页，还不能上传图片或推理 |

## 技术栈

- **前端**：React 19 + TypeScript + Vite，Vitest 单元测试，Playwright 端到端测试
- **后端**：FastAPI（Python 3.12），分块与 Embedding 各有策略注册表，本地模型按需加载
- **部署**：前后端各一个 Docker 镜像，由 GitHub Actions 构建并推送到 GHCR，再通过 SSH 按提交版本发布

架构和 API 说明见 [docs/architecture.md](docs/architecture.md)。

## 目录结构

```text
frontend/              React 前端（pages/ 下每个实验一个页面）
src/ragent_lab/api/    FastAPI 路由、schema、实验目录
src/ragent_lab/chunking/   分块策略与注册表
src/ragent_lab/embedding/  Embedding 模型与注册表
docker/                前后端镜像与 compose 配置
deploy/                服务器发布与回滚脚本
tests/                 后端单元测试
```

## 本地开发

需要 Python 3.12、Node.js 24。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
python -m nltk.downloader punkt punkt_tab
python -m uvicorn ragent_lab.api.app:app --app-dir src --reload --port 8000
```

另开终端：

```bash
cd frontend
npm ci
npm run dev
```

打开 http://127.0.0.1:5173。Vite 将 `/api` 代理到后端。API 文档位于 http://127.0.0.1:8000/docs。

基础 API 无需安装 PyTorch。需要本地语义分块或本地 Embedding 时，再安装 CPU/GPU 对应的 PyTorch 和 `requirements-models.txt`：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-models.txt
```

生产 API 镜像已包含 CPU 推理依赖。模型首次调用时下载并缓存；E5 Mistral 7B 和 GME 默认禁用，需要额外算力并设置 `ENABLE_HEAVY_MODELS=true`。当前 GME 接口仅用于文本嵌入。

## 测试

```bash
python -m unittest discover -s tests -v
cd frontend
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

端到端测试会启动 React 和 API；分块测试使用真实后端，Embedding UI 使用固定的服务响应，不调用付费接口。

## Docker

```bash
docker compose -f docker/docker-compose.yml up -d --build --wait
```

前端监听 `127.0.0.1:8501`，通过 `/api` 转发至独立 API 容器。向量模型缓存在 Docker volume 中。默认仅允许浏览器提交用户自己的 API Key，不向前端提供服务器密钥。

可选 `.env` 设置见 [env.example](env.example)。`ALLOW_SERVER_API_KEYS=true` 会允许客户端使用服务器额度，只应在可信访问环境开启；默认关闭。应用不提供账号系统。

## GitHub Actions CI/CD

配置文件：[.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml)。

- PR、`dev` 和 `feat/**`：API 单元测试、前端单元测试、TypeScript 检查、构建、浏览器测试。
- `master`：检查通过后构建两个 GHCR 镜像，以提交 SHA 标记，SSH 发布到服务器。
- 部署目录：`/opt/rag-lab/releases/<sha>`；成功后更新 `current` 链接。
- 部署会先拉取镜像，再更新容器；校验前端代理和 API 提交版本。失败恢复上一版本；首次迁移失败会重启旧 `rag-lab` 容器。
- 服务器现有 Nginx 继续将 `lab.ragents.net` 转发到 `127.0.0.1:8501`。无需修改其他应用的容器或网络。

仓库 Secrets：

| 名称 | 内容 |
| --- | --- |
| `DEPLOY_SSH_KEY` | 此仓库专用的服务器 SSH 私钥 |
| `DEPLOY_KNOWN_HOSTS` | 已核对的 `209.209.40.180` SSH host key |

Actions 使用内置 `GITHUB_TOKEN` 发布和拉取 GHCR 镜像。服务器需要 Docker Compose v2.30+、curl、flock、tar；首次迁移保留 `/opt/rag-lab/.env` 和旧容器，部署有短暂重启窗口。

手工回滚：

```bash
cd /opt/rag-lab/releases/<previous-sha>
LAB_SKIP_PULL=true bash deploy/deploy.sh <previous-sha>
```

## 增加实验

1. 在 `src/ragent_lab/api/catalog.py` 注册实验 ID、名称、状态和说明。
2. 在 API 下增加独立路由/服务，并在 `api/app.py` 注册。
3. 在 `frontend/src/pages` 添加页面，在 `App.tsx` 中绑定对应 ID。
4. 实验未实现前使用 `planned` 状态，展示规划而非模拟执行结果。

模型使用 `embedding/registry.py` 注册；分块策略使用 `chunking/registry.py` 注册。

旧 Streamlit 文件保留用于迁移参考，不参与生产构建或运行。其依赖仍见 `requirements.txt`。
