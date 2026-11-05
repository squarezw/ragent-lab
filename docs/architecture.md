# Ragent Lab 架构

浏览器 → 主机 Nginx（HTTPS）→ React 静态资源 Nginx（127.0.0.1:8501）→ `/api` → FastAPI → Python 算法 / 模型服务。

## 模块

- `frontend/`：React + TypeScript + Vite。实验导航、概览、分块工作台、向量对比、图像理解工作台。
- `src/ragent_lab/api/`：请求 schema、实验目录、算法适配及 HTTP 路由。
- `src/ragent_lab/chunking/`：11 种分块策略和注册表。语义模型延迟加载、缓存。
- `src/ragent_lab/embedding/`：Embedding 模型接口和注册表。云端 API 或本地模型。
- `src/ragent_lab/vision/`：图像理解与重绘。模型目录（`registry.py`）、OpenRouter 调用（`openrouter.py`）、提示词（`prompts.py`）、图片预处理与仓库外样例加载（`images.py`）、内存任务队列（`jobs.py`）、理解/打分/一致性/生成服务（`service.py`）、几何 schema（`schema.py`）与 PDF/DXF 渲染（`render.py`）。
- `docker/`：独立前后端镜像，内部网络和持久模型缓存。
- `deploy/`：按提交版本部署、健康检查和失败恢复。

## API

| 路由 | 用途 |
| --- | --- |
| `GET /api/health` | 健康状态、版本与提交 |
| `GET /api/experiments` | 实验目录和可用状态 |
| `GET /api/chunking/strategies` | 分块策略和参数定义 |
| `POST /api/chunking/run` | 执行分块、返回内容和统计 |
| `GET /api/embedding/models` | 模型目录与算力可用状态 |
| `POST /api/embedding/run` | 一次生成候选与查询向量，返回相似度排名 |
| `GET /api/vision/models` | 视觉模型目录、默认参数和价格 |
| `GET /api/vision/judge` | 判定模型 |
| `GET /api/vision/samples` | 内置样例列表及是否可打分 |
| `GET /api/vision/samples/{sample_id}/image` | 样例预处理后的 PNG |
| `POST /api/vision/images` | 上传图纸（PNG、JPG、单页 PDF，≤ 15 MB），返回图片 ID |
| `GET /api/vision/images/{image_id}` | 已上传图片的 PNG |
| `POST /api/vision/understand` | 提交理解任务，返回 `job_id` |
| `POST /api/vision/score` | 按已审定细则对理解结果打分，返回 `job_id` |
| `POST /api/vision/consensus` | 比较多个模型的理解一致性，返回 `job_id` |
| `POST /api/vision/generate` | 按理解生成几何清单并渲染预览，返回 `job_id` |
| `POST /api/vision/export` | 将几何清单导出为 PDF 或 DXF |
| `GET /api/vision/jobs/{job_id}` | 查询任务状态与结果 |

请求示例：

```json
{"strategy":"递归分块","text":"需要分块的文档","params":{"chunk_size":150,"chunk_overlap":20}}
```

```json
{"model":"openai_text_embedding_3_small","texts":["候选文本"],"query":"查询问题","api_key":"用户自己的密钥"}
```

## 状态与资源

前端实验记录仅存在当前页面内存，不持久保存 API Key。修改文本、查询或模型后清空当前结果；横向对比只允许相同文本和查询。展示结果及加入对比不再调用向量服务。

API 使用单进程锁限制实验并发，忙碌时返回 429。输入有数量和长度上限；大型本地模型默认禁用。短期使用同步请求，未来长时间图纸分析可引入任务队列与持久实验记录。

API 不导入 Streamlit，不返回模型实例、函数或服务端密钥。验证错误不回显用户输入；上游错误统一转换为安全提示。生产同源代理，不配置通配 CORS。

## 扩展

实验目录是前后端共享的发现入口。每个新实验增加独立页面和 API 模块。图像理解（`vision`）通过 OpenRouter 调用视觉模型，Key 由访客在网页输入；长耗时的理解、打分、一致性和生成走内存任务队列（2 小时过期，不落盘），前端轮询 `/api/vision/jobs/{job_id}`。内置样例和细则放在仓库外的 `VISION_SAMPLES_DIR`，格式见 README“图像理解样例”。
