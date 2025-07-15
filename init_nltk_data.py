import os
import nltk

# 目标目录为项目内的 nltk_data
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# 下载 punkt 和 punkt_tab 到本地
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)

print(f"nltk_data 已下载到: {NLTK_DATA_PATH}") 