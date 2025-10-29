"""
Main Streamlit application for embedding testing.
"""

from typing import Any, Dict, List
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..config.settings import AppConfig
from ..embedding.registry import get_all_models, get_model
from ..stats import StatsManager
from .embedding_components import (
    render_embedding_header,
    render_text_inputs,
    render_model_selection,
    render_model_parameters,
    render_embedding_results,
    render_embedding_comparison
)


def initialize_embedding_session_state():
    """Initialize session state variables for embedding."""
    if 'embedding_result' not in st.session_state:
        st.session_state['embedding_result'] = None
    if 'embedding_model' not in st.session_state:
        st.session_state['embedding_model'] = None
    if 'embedding_comparison' not in st.session_state:
        st.session_state['embedding_comparison'] = {}
    if 'embedding_generating' not in st.session_state:
        st.session_state['embedding_generating'] = False


def handle_embedding_user_tracking():
    """Handle user access tracking for embedding page."""
    stats_manager = StatsManager(AppConfig.STATS_FILE)
    
    # Get user information (simplified version without streamlit_javascript)
    try:
        user_agent = "Unknown"
        ip = "Unknown"
    except:
        user_agent = None
        ip = None
    
    # Track access if not already logged in this session
    if user_agent and ip and not st.session_state.get("embedding_stats_logged"):
        stats_manager.track_user_access(ip, user_agent)
        st.session_state["embedding_stats_logged"] = True


def process_embedding_request(texts: List[str], selected_model: str, 
                             param_values: Dict[str, Any]) -> Any:
    """Process embedding request and return results."""
    try:
        model = get_model(selected_model)
        
        # Handle special parameter processing
        if param_values:
            result = model.embed(texts, **param_values)
        else:
            result = model.embed(texts)
        
        return result
    except Exception as e:
        return f"Embedding 生成出错: {e}"


def create_embedding_app():
    """Create and run the embedding Streamlit application."""
    # Note: page configuration is handled by the main app
    
    # Hide Streamlit header/footer/menu and reduce top padding
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 1rem !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Render header
    render_embedding_header()
    
    # Initialize session state
    initialize_embedding_session_state()
    
    # Handle user tracking
    handle_embedding_user_tracking()
    
    # Create main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Text inputs section
        texts, query_text = render_text_inputs()
        
        # Model selection section
        selected_model, generate = render_model_selection()
        
        # Parameter inputs
        param_values = render_model_parameters(selected_model)
        
        # Process embedding request
        if generate and texts:
            st.session_state['embedding_generating'] = True
            with st.spinner("正在生成 embedding..."):
                result = process_embedding_request(texts, selected_model, param_values)
                if isinstance(result, str):  # Error message
                    st.error(result)
                    st.session_state['embedding_generating'] = False
                else:
                    st.session_state['embedding_result'] = result
                    st.session_state['embedding_model'] = selected_model
                    st.session_state['embedding_query_text'] = query_text  # Store query text
                    st.session_state['embedding_params'] = param_values  # Store parameters
                    
                    # Generate and cache query embedding for comparison
                    if query_text:
                        try:
                            model = get_model(selected_model)
                            if param_values:
                                query_result = model.embed([query_text], **param_values)
                            else:
                                query_result = model.embed([query_text])
                            st.session_state['embedding_query_embedding'] = query_result.embeddings[0]
                        except Exception as e:
                            st.warning(f"生成查询文本embedding时出错: {e}")
                            st.session_state['embedding_query_embedding'] = None
                    else:
                        st.session_state['embedding_query_embedding'] = None
                    
                    st.session_state['embedding_generating'] = False
                    st.success("Embedding 生成成功！")
    
    with col2:
        # Results section
        render_embedding_results(
            st.session_state['embedding_result'], 
            query_text,
            st.session_state.get('embedding_params', {})
        )
    
    # Comparison section (full width)
    if st.session_state['embedding_result'] and not st.session_state['embedding_generating']:
        st.markdown("---")
        
        # Add to comparison - only enabled when embedding is successfully generated and not currently generating
        if st.button("添加到对比", disabled=False):
            current_result = st.session_state['embedding_result']
            current_model = st.session_state['embedding_model']
            current_params = st.session_state.get('embedding_params', {})
            current_query_embedding = st.session_state.get('embedding_query_embedding', None)
            
            st.session_state['embedding_comparison'][current_model] = current_result
            # 保存模型参数
            if 'embedding_comparison_params' not in st.session_state:
                st.session_state['embedding_comparison_params'] = {}
            st.session_state['embedding_comparison_params'][current_model] = current_params
            
            # 保存查询文本的embedding
            if 'embedding_comparison_query_embeddings' not in st.session_state:
                st.session_state['embedding_comparison_query_embeddings'] = {}
            st.session_state['embedding_comparison_query_embeddings'][current_model] = current_query_embedding
            
            st.success(f"已将 {current_model} 添加到对比")
        
        # Clear comparison
        if st.button("清空对比"):
            st.session_state['embedding_comparison'] = {}
            st.session_state['embedding_comparison_params'] = {}
            st.session_state['embedding_comparison_query_embeddings'] = {}
            st.success("已清空对比")
        
        # Show comparison
        if st.session_state['embedding_comparison']:
            # Get query text from session state or use default
            query_text = st.session_state.get('embedding_query_text', "")
            render_embedding_comparison(
                st.session_state['embedding_comparison'], 
                query_text,
                st.session_state.get('embedding_comparison_params', {}),
                st.session_state.get('embedding_comparison_query_embeddings', {})
            )
    
    # Reference evaluation site section
    st.markdown("---")
    st.markdown("### 📊 参考评测站点")
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
            <p style="margin: 0; color: #666;">
                <strong>MTEB (Massive Text Embedding Benchmark)</strong> 是评估文本嵌入模型性能的标准基准测试平台。
                您可以访问以下链接查看各种embedding模型的详细评测结果：
            </p>
            <div style="margin-top: 0.5rem;">
                <a href="https://huggingface.co/spaces/mteb/leaderboard" target="_blank" 
                   style="color: #1f77b4; text-decoration: none; font-weight: bold;">
                    🔗 MTEB Leaderboard - Hugging Face
                </a>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; color: #888;">
                该平台提供了多种embedding模型在不同任务上的性能对比，包括检索、分类、聚类等任务。
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    create_embedding_app()
