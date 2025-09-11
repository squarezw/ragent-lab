# RAG 分段策略测试 Lab

## 初始化 NLTK 数据（本地开发首次需运行）
```bash
python init_nltk_data.py
```

## 本地开发运行
```bash
# 激活 conda 环境
conda env create -f environment.yml  # 仅首次
conda activate rag-lab
streamlit run lab_streamlit.py
```

## Docker 构建与部署
```bash
# 构建镜像（从 docker 目录）
docker build -t rag-lab -f docker/Dockerfile .
# 运行容器（后台，端口8501）
docker run -d -p 8501:8501 --name rag-lab rag-lab
```

### 使用 Docker Compose（推荐）
```bash
# 在 docker 目录下运行
cd docker
docker-compose up -d
```

## 访问服务
浏览器访问：http://服务器IP:8501