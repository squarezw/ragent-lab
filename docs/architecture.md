# Ragent Lab 架构

浏览器 → 主机 Nginx（HTTPS）→ React 静态资源 Nginx（127.0.0.1:8501）→ `/api` → FastAPI → Python 算法 / 模型服务。

## 模块

- `frontend/`：React + TypeScript + Vite。实验导航、概览、分块工作台、向量对比、图像理解规划页。
- `src/ragent_lab/api/`：请求 schema、实验目录、算法适配及 HTTP 路由。
- `src/ragent_lab/chunking/`：11 种分块策略和注册表。语义模型延迟加载、缓存。
- `src/ragent_lab/embedding/`：Embedding 模型接口和注册表。云端 API 或本地模型。
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

实验目录是前后端共享的发现入口。每个新实验增加独立页面和 API 模块。图像理解已注册为 `vision/planned`，后续可扩展文件上传、题目集、模型调用和质量评估。
