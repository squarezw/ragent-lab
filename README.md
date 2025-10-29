# RAG Lab

一个用于RAG（检索增强生成）应用的文本分块和嵌入模型综合工具包。

## 项目结构

```
ragent-lab/
├── src/ragent_lab/          # 主要源代码
│   ├── chunking/            # 文本分块策略
│   │   ├── base.py          # 基础工具和类
│   │   ├── strategies.py    # 分块策略实现
│   │   └── registry.py      # 策略注册和管理
│   ├── embedding/           # 嵌入模型
│   │   ├── base.py          # 基础嵌入类和工具
│   │   ├── strategies.py    # 嵌入模型实现
│   │   └── registry.py      # 模型注册和管理
│   ├── config/              # 配置模块
│   │   └── settings.py      # 应用配置
│   ├── utils/               # 工具函数
│   │   └── text_utils.py    # 文本处理工具
│   ├── web/                 # Web界面
│   │   ├── app.py          # 主应用
│   │   ├── components.py   # 分块UI组件
│   │   ├── embedding_app.py # 嵌入应用
│   │   └── embedding_components.py # 嵌入UI组件
│   └── __init__.py
├── main.py                 # 主入口文件
├── streamlit_app.py        # Streamlit应用入口
└── requirements.txt        # 依赖文件
```

## 功能特性

### 文本分块策略

- **固定长度分块**: 按固定字符长度分块
- **基于句子的分块**: 按句子边界分块，支持中英文自适应
- **基于段落的分块**: 按段落分块
- **滑动窗口分块**: 解决分块边界信息丢失问题
- **语义分块**: 基于句子嵌入相似度的语义分块
- **递归分块**: 使用LangChain实现的递归分块
- **语境丰富分块**: 为每个分块添加上下文标题
- **多模态分块**: 针对混合内容（代码+文本）的分块
- **Agent分块**: 模拟LLM决策的自适应分块
- **子文档分块**: 提取包含特定关键词的子文档
- **混合分块**: 组合多种分块策略

## 安装和使用

### 环境要求

- Python 3.8+

### 本地开发安装

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（可选）
cp env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key

# 初始化 NLTK 数据（首次运行需要）
python init_nltk_data.py
```

### 本地运行

```bash
# 方式1：使用统一入口（推荐）
python main.py web      # 启动Web界面
python main.py cli      # 运行命令行演示
```

### 编程使用

#### 文本分块
```python
from ragent_lab.chunking import fixed_length_chunking, semantic_chunking
from ragent_lab.config.settings import DEFAULT_TEXT

# 使用固定长度分块
chunks = fixed_length_chunking(DEFAULT_TEXT, chunk_size=100)

# 使用语义分块
semantic_chunks = semantic_chunking(DEFAULT_TEXT, threshold=0.85)
```

#### 嵌入模型
```python
from ragent_lab.embedding import (
    sentence_transformer_embedding,
    openai_embedding,
    get_all_models
)

# 使用 SentenceTransformer (默认使用BGE-M3)
texts = ["Hello world", "你好世界"]
result = sentence_transformer_embedding(texts)
print(f"向量维度: {result.dimension}")

# 使用指定的模型
result = sentence_transformer_embedding(texts, "intfloat/multilingual-e5-large")

# 使用 OpenAI (需要 API Key)
result = openai_embedding(texts, api_key="your-api-key")

# 获取所有可用模型
models = get_all_models()
for name, info in models.items():
    print(f"{name}: {info['desc']}")
```

## Docker 部署

### 使用 Docker Compose（推荐）

```bash
# 在 docker 目录下运行
cd docker
docker-compose up -d
```

### 手动 Docker 构建

```bash
# 构建镜像
docker build -t rag-lab -f docker/Dockerfile .

# 运行容器（后台，端口8501）
docker run -d -p 8501:8501 --name rag-lab rag-lab
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License