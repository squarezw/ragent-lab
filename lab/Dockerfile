FROM python:3.12-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 下载 NLTK 数据（如有 init_nltk_data.py）
RUN python init_nltk_data.py

# 设置 Streamlit 端口和允许外部访问
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 启动 Streamlit
CMD ["streamlit", "run", "lab_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"] 