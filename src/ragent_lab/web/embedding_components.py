"""
Reusable Streamlit components for embedding interface.
"""

from typing import Dict, Any, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ..config.settings import AppConfig
from ..embedding.registry import get_all_models
from ..embedding.base import EmbeddingResult


def truncate_text(text: str, max_length: int = 50) -> str:
    """截断文本并添加省略号."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_registry_model_name(display_name: str) -> str:
    """将显示名称转换为注册表中的模型名称."""
    # 映射显示名称到注册名称
    name_mapping = {
        "OpenAI text-embedding-3-small": "openai_text_embedding_3_small",
        "OpenAI text-embedding-3-large": "openai_text_embedding_3_large", 
        "SentenceTransformer BAAI/bge-m3": "bge_m3",
        "SentenceTransformer intfloat/multilingual-e5-large": "multilingual_e5_large",
        "SentenceTransformer intfloat/e5-mistral-7b-instruct": "e5_mistral_7b_instruct",
        "GME Alibaba-NLP/gme-Qwen2-VL-2B-Instruct": "gme_qwen2_vl_2b_instruct"
    }
    return name_mapping.get(display_name, display_name)


def render_embedding_header():
    """Render the embedding page header."""
    st.markdown(
        "<h1 style='margin-top: 0.5rem; margin-bottom: 1.2rem;'>RAG Embedding 模型测试</h1>",
        unsafe_allow_html=True
    )


def render_text_inputs() -> Tuple[List[str], str]:
    """Render text input areas for embedding."""
    
    # Create the main columns for text input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("待嵌入文本")            
    
    with col1:
        # Get example text based on selection
        example_texts = {
            "short": "人工智能正在改变世界\n\n机器学习是AI的核心技术\n\n深度学习推动了AI的发展\n\n自然语言处理让机器理解人类语言\n\n计算机视觉让机器看懂世界",
            "long": """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

机器学习（Machine Learning，简称ML）是人工智能的核心技术之一，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测或决策。机器学习可以分为监督学习、无监督学习和强化学习三大类。

深度学习（Deep Learning）是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。卷积神经网络（CNN）和循环神经网络（RNN）是深度学习中最重要的两种网络结构。

自然语言处理（Natural Language Processing，简称NLP）是人工智能和语言学领域的交叉学科，旨在让计算机能够理解、解释和生成人类语言。NLP技术包括文本分析、情感分析、机器翻译、问答系统等。

计算机视觉（Computer Vision）是人工智能的另一个重要分支，它使计算机能够从图像或视频中获取信息并理解其内容。计算机视觉技术广泛应用于自动驾驶、医疗影像分析、安防监控等领域。"""
        }
        
        # Set default example
        if 'text_example' not in st.session_state:
            st.session_state['text_example'] = "short"
        
        # Handle legacy 'medium' selection by resetting to 'short'
        if st.session_state['text_example'] == "medium":
            st.session_state['text_example'] = "short"
        
        current_example = example_texts[st.session_state['text_example']]
        
        st.markdown("**输入多个段落（每段用空行分隔）：**")
        texts_input = st.text_area(
            "文本输入", 
            value=current_example,
            height=300,
            help="每段输入一个文本，系统会为每个段落生成嵌入向量。段落之间用空行分隔。"
        )
        
        # Parse texts (split by empty lines for paragraphs, or by single lines if no empty lines)
        paragraphs = [p.strip() for p in texts_input.split('\n\n') if p.strip()]
        
        # If no paragraphs found (no double newlines), try splitting by single newlines
        if not paragraphs:
            paragraphs = [p.strip() for p in texts_input.split('\n') if p.strip()]
        
        texts = []
        for paragraph in paragraphs:
            # Split paragraph into sentences if it's too long
            if len(paragraph) > 500:  # If paragraph is too long, split by sentences
                sentences = [s.strip() for s in paragraph.split('。') if s.strip()]
                texts.extend([s + '。' if not s.endswith('。') else s for s in sentences])
            else:
                texts.append(paragraph)
        
    
    with col2:
        st.subheader("查询文本")
        st.markdown("**输入查询句子：**")
        query_text = st.text_input(
            "查询文本",
            value="AI技术发展迅速",
            help="输入一个查询句子，系统会计算它与待嵌入文本的相似度"
        )
        
        # Display example text switching
        if texts:
            st.markdown("**示例文本切换：**")
            
            # Create buttons for switching between examples (vertical layout)
            if st.button("短文本示例", key="switch_short_text"):
                st.session_state['text_example'] = "short"
                st.rerun()
            if st.button("长文本示例", key="switch_long_text"):
                st.session_state['text_example'] = "long"
                st.rerun()
    
    return texts, query_text


def render_model_selection() -> Tuple[str, bool]:
    """Render embedding model selection interface."""
    st.subheader("Embedding 模型")
    
    models = get_all_models()
    model_names = list(models.keys())
    
    # Simple radio selection without complex styling
    selected_model = st.radio(
        "选择模型",
        model_names,
        index=0,
        help="选择要使用的embedding模型"
    )
    
    # Display model information
    model_info = models[selected_model]
    st.markdown("**模型信息：**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(model_info["desc"])
    
    with col2:
        # Display rating stars
        rating = model_info.get("rating", 0)
        stars = "★" * rating + "☆" * (5 - rating)
        st.markdown(f"<div style='text-align: center; color: gold; font-size: 20px;'>{stars}</div>", 
                   unsafe_allow_html=True)
    
    # Generate button
    generate = st.button("生成 Embedding", type="primary")
    
    return selected_model, generate


def render_model_parameters(selected_model: str) -> Dict[str, Any]:
    """Render parameter inputs for the selected model."""
    import os
    models = get_all_models()
    param_values = {}
    params = models[selected_model].get("params", [])
    
    if params:
        st.markdown("---")
        st.write("**模型参数：**")
        for param in params:
            key = f"param_{selected_model}_{param['name']}"
            if param["type"] == "str":
                # For API keys, try to get from environment first
                default_value = param["default"]
                if "api_key" in param["name"].lower():
                    env_key = os.getenv("OPENAI_API_KEY", "")
                    if env_key:
                        default_value = env_key
                        st.info("✅ 已从环境变量加载 OpenAI API Key")
                    else:
                        st.warning("⚠️ 未找到环境变量 OPENAI_API_KEY，请在 .env 文件中配置")
                
                param_values[param["name"]] = st.text_input(
                    param["label"], 
                    value=default_value, 
                    key=key,
                    type="password" if "key" in param["name"].lower() else "default"
                )
            elif param["type"] == "int":
                param_values[param["name"]] = st.number_input(
                    param["label"], 
                    value=param["default"], 
                    step=1, 
                    key=key
                )
            elif param["type"] == "float":
                param_values[param["name"]] = st.number_input(
                    param["label"], 
                    value=param["default"], 
                    step=0.01, 
                    format="%.4f", 
                    key=key
                )
    
    return param_values


def render_embedding_results(result: EmbeddingResult, query_text: str = ""):
    """Render embedding results with visualizations."""
    if result is None:
        st.info("请点击'生成 Embedding'按钮")
        return
    
    # Basic information
    col1, col2 = st.columns(2)
    with col1:
        st.text("向量维度")
        st.write(result.dimension)
    with col2:
        st.text("模型")
        st.write(result.model_name)
    
    
    # Similarity analysis if query text provided
    if query_text and result.count > 0:
        st.markdown("---")
        st.subheader("相似度分析")
        
        # Generate embedding for query text using the same model
        try:
            from ..embedding.registry import get_all_models
            models = get_all_models()
            registry_name = get_registry_model_name(result.model_name)
            model_info = models.get(registry_name)
            
            if model_info:
                model = model_info["model"]
                query_result = model.embed([query_text])
                query_embedding = query_result.embeddings[0]
                
                # Calculate similarities between query and all texts
                similarities = []
                for i, text in enumerate(result.texts):
                    sim = np.dot(query_embedding, result.embeddings[i]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(result.embeddings[i])
                    )
                    similarities.append({
                        "文本": truncate_text(text, 80),  # 截断文本用于表格显示
                        "相似度": float(sim),
                        "排名": 0  # Will be updated after sorting
                    })
            else:
                st.warning("无法找到模型信息，跳过相似度分析")
                return
        except Exception as e:
            st.error(f"生成查询文本embedding时出错: {e}")
            return
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["相似度"], reverse=True)
        for i, item in enumerate(similarities):
            item["排名"] = i + 1
        
        # Display similarity table
        sim_df = pd.DataFrame(similarities)
        st.dataframe(sim_df, use_container_width=True)
        
        # Similarity visualization
        # 为图表创建截断的文本标签
        sim_df_chart = sim_df.copy()
        sim_df_chart["文本_图表"] = sim_df_chart["文本"].apply(lambda x: truncate_text(x, 40))
        
        fig = px.bar(
            sim_df_chart, 
            x="相似度", 
            y="文本_图表",
            orientation="h",
            title="文本相似度对比",
            color="相似度",
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            height=max(400, len(similarities) * 30),
            yaxis_title="文本"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Model metadata
    if result.metadata:
        st.markdown("---")
        st.subheader("模型元数据")
        st.json(result.metadata)


def render_embedding_comparison(results: Dict[str, EmbeddingResult], query_text: str = ""):
    """Render comparison between different embedding models."""
    if not results:
        return
    
    st.subheader("模型对比")
    
    # Get model registry for additional info
    models_registry = get_all_models()
    
    # Basic comparison metrics
    comparison_data = []
    for model_name, result in results.items():
        # Get model info from registry
        model_info = models_registry.get(model_name, {})
        model_desc = model_info.get('desc', '')
        
        # Get model size from result
        model_size = result.model_size or "未知"
        
        comparison_data.append({
            "模型": model_name,
            "参数大小": model_size,
            "维度": result.dimension,
            "文本数": result.count,
            "描述": model_desc[:50] + "..." if len(model_desc) > 50 else model_desc
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Top 3 similarity matches (query vs texts)
    if query_text and len(results) > 0:  # Only show if query text is provided and results exist
        st.markdown("---")
        st.markdown("#### Top 3 相似度匹配")
        st.markdown("**说明**: 此表显示查询文本与待嵌入文本的相似度对比，Top1是与查询文本最相似的待嵌入文本。")
        top_matches_data = []
        
        for model_name, result in results.items():
            # Generate embedding for query text using the same model
            try:
                models = get_all_models()
                registry_name = get_registry_model_name(result.model_name)
                model_info = models.get(registry_name)
                
                if model_info:
                    model = model_info["model"]
                    query_result = model.embed([query_text])
                    query_embedding = query_result.embeddings[0]
                    
                    # Calculate similarities between query and all texts
                    similarities = []
                    for i, text_embedding in enumerate(result.embeddings):
                        sim = np.dot(query_embedding, text_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
                        )
                        similarities.append({
                            'similarity': sim,
                            'text': result.texts[i],
                            'index': i
                        })
                    
                    # Sort by similarity and get top 3
                    similarities.sort(key=lambda x: x['similarity'], reverse=True)
                    top3_matches = similarities[:3]
                    
                    for idx, match in enumerate(top3_matches):
                        top_matches_data.append({
                            "模型": model_name if idx == 0 else "",  # Only show model name for first row
                            "排名": f"Top {idx + 1}",
                            "相似度": f"{match['similarity']:.4f}",
                            "查询文本": truncate_text(query_text, 50),
                            "匹配文本": truncate_text(match['text'], 50)
                        })
                else:
                    st.warning(f"无法找到模型 {model_name} 的信息")
            except Exception as e:
                st.warning(f"模型 {model_name} 处理查询文本时出错: {e}")
                continue
        
        if top_matches_data:
            top_matches_df = pd.DataFrame(top_matches_data)
            st.dataframe(top_matches_df, use_container_width=True)
