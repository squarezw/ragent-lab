# Ragent Lab

React + FastAPI 的中文 AI 实验工作台，部署于 https://lab.ragents.net。

- **文本分块**：11 种策略、参数调整、分块预览和长度统计。
- **Embedding 对比**：7 个模型注册项、查询相似度排名、固定候选文本与查询的模型对比。
- **图像理解**：工程图纸方向的独立规划页面，预留实验注册入口；尚未实现图片上传与推理。

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
