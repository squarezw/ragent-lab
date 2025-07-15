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
text = """关于完善中国特色现代企业制度的意见
三、完善公司治理结构
（三）健全企业产权结构。尊重企业独立法人财产权，形成归属清晰、结构合理、流转顺畅的企业产权制度。国有企业要根据功能定位逐步调整优化股权结构，形成股权结构多元、股东行为规范、内部约束有效、运行高效灵活的经营机制。鼓励民营企业构建简明、清晰、可穿透的股权结构。
（四）完善国有企业公司治理。加快建立健全权责法定、权责透明、协调运转、有效制衡的公司治理机制，强化章程在公司治理中的基础作用。党委（党组）发挥把方向、管大局、保落实的领导作用。股东会是公司的权力机构，股东按照出资比例和章程行使表决权，不得超出章程规定干涉企业日常经营。董事会发挥定战略、作决策、防风险的作用，推动集团总部授权放权与分批分类落实子企业董事会职权有机衔接，规范落实董事会向经理层授权制度。完善外部董事评价和激励约束机制，落实外部董事知情权、表决权、监督权、建议权。经理层发挥谋经营、抓落实、强管理的作用，全面推进任期制和契约化管理。鼓励国有企业参照经理层成员任期制和契约化管理方式，更大范围、分层分类落实管理人员经营管理责任。
（五）支持民营企业优化法人治理结构。鼓励民营企业根据实际情况采取合伙制、公司制等多种组织形式，完善内部治理规则，制定规范的章程，保持章程与出资协议的一致性，规范控股股东、实际控制人行为。支持引导民营企业完善治理结构和管理制度，鼓励有条件的民营企业规范组建股东会、董事会、经理层。鼓励家族企业创新管理模式、组织结构、企业文化，逐步建立现代企业制度。
（六）发挥资本市场对完善公司治理的推动作用。强化控股股东对公司的诚信义务，支持上市公司引入持股比例5%以上的机构投资者作为积极股东。严格落实上市公司独立董事制度，设置独立董事占多数的审计委员会和独立董事专门会议机制。完善上市公司治理领域信息披露制度，促进提升决策管理的科学性。
```
import numpy as np
```
八、保障措施
各地区各部门要结合实际抓好本意见贯彻落实。企业要深刻认识完善中国特色现代企业制度的重要意义，落实主体责任，以企业制度创新推动高质量发展。完善相关法律法规，推动修订企业国有资产法等，推动企业依法经营、依法治企。规范会计、咨询、法律、信用评级等专业机构执业行为，加强对专业机构的从业监管，发挥其执业监督和专业服务作用，维护公平竞争、诚信规范的良好市场环境。加强对现代企业制度实践探索和成功经验的宣传，总结一批企业党建典型经验，推广一批公司治理典型实践案例。
"""

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

# 9. AI Agent 分块 (Agentic)
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
    "固定长度分块": {
        "func": fixed_length_chunking,
        "desc": "将文本按固定字符长度分块，简单高效，适合无结构文本。",
        "rating": 1
    },
    "基于句子的分块": {
        "func": sentence_based_chunking,
        "desc": "按句子边界分块，适合语义连贯的文本。中英文自适应。",
        "rating": 2
    },
    "基于段落的分块": {
        "func": paragraph_based_chunking,
        "desc": "按段落分块，适合结构清晰、段落分明的文档。",
        "rating": 2
    },
    "滑动窗口分块": {
        "func": sliding_window_chunking,
        "desc": "滑动窗口分块，解决分块边界信息丢失问题，适合上下文相关任务。",
        "rating": 3
    },
    "语义分块": {
        "func": semantic_chunking,
        "desc": "基于句子嵌入相似度的语义分块，自动聚合相关句子。",
        "rating": 3
    },
    "递归分块": {
        "func": recursive_chunking,
        "desc": "递归分块，优先按自然边界（段落/句子）分割，保证分块长度和结构。",
        "rating": 4
    },
    "语境丰富分块": {
        "func": context_enriched_chunking,
        "desc": "为每个分块添加上下文标题，便于理解和检索。",
        "rating": 3
    },
    "特定模态分块": {
        "func": modality_specific_chunking,
        "desc": "针对多模态内容（如代码+文本）分块，适合混合内容场景。",
        "rating": 4
    },
    "Agent分块": {
        "func": agentic_chunking,
        "desc": "模拟 LLM 决策的分块策略，根据文本结构动态选择分块方式。",
        "rating": 5
    },
    "子文档分块": {
        "func": subdocument_chunking,
        "desc": "提取包含关键词的子文档，适合聚焦特定主题内容。",
        "rating": 3
    },
    "混合分块": {
        "func": hybrid_chunking,
        "desc": "组合多种分块策略，先按段落再对长段落做语义分块。",
        "rating": 4
    }
}

if __name__ == "__main__":
    # 执行并显示结果
    for name, strategy_info in strategies.items():
        print(f"\n=== {name} ===\n")
        func = strategy_info["func"]
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
