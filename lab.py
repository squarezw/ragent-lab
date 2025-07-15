# pip install nltk numpy scikit-learn sentence-transformers langchain

import os
import nltk

# 优先使用项目内的nltk_data
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 示例文本
text = """在大规模语言模型应用中，文档分块(chunking)是处理长文本的关键技术。不同的分块策略适用于不同场景：
1. 固定长度分块简单高效，但可能破坏语义连贯性。
2. 基于句子的分块保留完整语义单元。
3. 基于段落的分块适合结构清晰的文档。
4. 滑动窗口分块解决边界信息丢失问题。
5. 语义分块通过嵌入相似度聚类相关句子。
6. 递归分块处理多层文档结构。
7. 语境丰富分块添加上下文标题。
8. 特定模态分块处理多格式内容。
9. 主体性分块使用LLM动态决策。
10. 子文档分块提取关键片段。
11. 混合分块组合多种策略。"""

# 1. 固定长度分块 (Fixed-Length)
def fixed_length_chunking(text, chunk_size=100):
    """
    将文本按固定字符长度分块
    :param text: 输入文本
    :param chunk_size: 每块字符数
    :return: 分块列表
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 2. 基于句子的分块 (Sentence-Based)
def sentence_based_chunking(text):
    """
    按句子边界分块（中英文自适应）
    :param text: 输入文本
    :return: 句子列表
    """
    import re
    # 判断是否为中文（出现中文字符的比例大于一定阈值）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    ratio = len(chinese_chars) / max(len(text), 1)
    if ratio > 0.2:  # 超过20%为中文，认为是中文文本
        # 按中文句号、问号、感叹号、换行、冒号分割
        return [s for s in re.split(r'[。！？:]', text) if s.strip()]
    else:
        return nltk.sent_tokenize(text)

# 3. 基于段落的分块 (Paragraph-Based)
def paragraph_based_chunking(text):
    """
    按段落分块（假设段落由换行符分隔）
    :param text: 输入文本
    :return: 段落列表
    """
    return [p.strip() for p in text.split('\n') if p.strip()]

# 4. 滑动窗口分块 (Sliding Window)
def sliding_window_chunking(text, window_size=3, stride=2):
    """
    滑动窗口分块（按句子）
    :param text: 输入文本
    :param window_size: 窗口包含的句子数
    :param stride: 滑动步长
    :return: 重叠分块列表
    """
    sentences = sentence_based_chunking(text)
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = ' '.join(sentences[i:i+window_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# 5. 语义分块 (Semantic)
def semantic_chunking(text, threshold=0.65):
    """
    基于嵌入相似度的语义分块
    :param text: 输入文本
    :param threshold: 相似度阈值
    :return: 语义分块列表
    """
    # 加载嵌入模型（首次使用需下载）
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    sentences = sentence_based_chunking(text)
    embeddings = model.encode(sentences)

    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        # 计算相邻句子相似度
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    chunks.append(" ".join(current_chunk))
    return chunks

# 6. 递归分块 (Recursive)
def recursive_chunking(text):
    """
    递归分块（使用LangChain实现）
    结构优先，长度优先，不做语义分析，不考虑上下文，不考虑语境
    :param text: 输入文本
    :return: 分块列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len
    )
    return splitter.split_text(text)

# 7. 语境丰富分块 (Context-Enriched)
def context_enriched_chunking(text):
    """
    添加标题的语境丰富分块
    :param text: 输入文本
    :return: 带标题的分块字典列表
    """
    chunks = paragraph_based_chunking(text)
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        # 生成标题（简化版，实际应用可用LLM生成）
        title = f"区块_{i+1}: {chunk[:15]}..." if len(chunk) > 15 else chunk
        enriched_chunks.append({
            "title": title,
            "content": chunk
        })
    return enriched_chunks

# 8. 特定模态分块 (Modality-Specific)
def modality_specific_chunking(text):
    """
    处理多模态内容的分块（示例：代码+文本混合）
    :param text: 输入文本
    :return: 分块列表（带类型标记）
    """
    # 实际应用中需根据内容类型使用不同解析器
    chunks = []
    current_type = None
    buffer = ""

    # 简化逻辑：根据行首字符判断类型
    for line in text.split('\n'):
        if line.startswith('    ') or line.startswith('def ') or line.startswith('import'):
            if current_type != 'code':
                if buffer: chunks.append({"type": current_type, "content": buffer})
                current_type = 'code'
                buffer = line + '\n'
            else:
                buffer += line + '\n'
        else:
            if current_type != 'text':
                if buffer: chunks.append({"type": current_type, "content": buffer})
                current_type = 'text'
                buffer = line + '\n'
            else:
                buffer += line + '\n'

    if buffer: chunks.append({"type": current_type, "content": buffer})
    return chunks

# 9. 主体性分块 (Agentic)
def agentic_chunking(text):
    """
    使用LLM决策的分块策略（模拟）
    :param text: 输入文本
    :return: 分块列表
    """
    # 实际应用需集成LLM API
    print("模拟LLM决策：分析文档结构并动态选择分块策略")
    if len(text) > 1000:
        return sliding_window_chunking(text)
    elif '\n' in text:
        return paragraph_based_chunking(text)
    else:
        return fixed_length_chunking(text, chunk_size=150)

# 10. 子文档分块 (Subdocument)
def subdocument_chunking(text, keywords=['分块', '语义']):
    """
    提取包含关键词的子文档
    :param text: 输入文本
    :param keywords: 关键词列表
    :return: 子文档列表
    """
    paragraphs = paragraph_based_chunking(text)
    subdocs = []
    for para in paragraphs:
        if any(kw in para for kw in keywords):
            subdocs.append(para)
    return subdocs

# 11. 混合分块 (Hybrid)
def hybrid_chunking(text):
    """
    组合多种分块策略
    :param text: 输入文本
    :return: 分块列表
    """
    # 步骤1: 先按段落分块
    chunks = paragraph_based_chunking(text)

    # 步骤2: 对长段落进行语义分块
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 200:
            final_chunks.extend(semantic_chunking(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks

# 测试所有分块策略
strategies = {
    "固定长度分块": fixed_length_chunking,
    "基于句子的分块": sentence_based_chunking,
    "基于段落的分块": paragraph_based_chunking,
    "滑动窗口分块": sliding_window_chunking,
    "语义分块": semantic_chunking,
    "递归分块": recursive_chunking,
    "语境丰富分块": context_enriched_chunking,
    "特定模态分块": modality_specific_chunking,
    "主体性分块": agentic_chunking,
    "子文档分块": subdocument_chunking,
    "混合分块": hybrid_chunking
}

if __name__ == "__main__":
    # 执行并显示结果
    for name, func in strategies.items():
        print(f"\n=== {name} ===\n")
        chunks = func(text)

        # 特殊处理字典类型输出
        if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
            for i, chunk in enumerate(chunks):
                print(f"[区块 {i+1}]")
                for k, v in chunk.items():
                    print(f"{k}: {v}")
                print()
        else:
            for i, chunk in enumerate(chunks):
                print(f"区块 {i+1}: {chunk[:70]}{'...' if len(chunk)>70 else ''}")
