"""
Reusable Streamlit components for the web interface.
"""

from typing import Dict, Any, List
from ..config.settings import AppConfig
from ..utils.text_utils import format_text_stats, format_chunk_stats
from ..chunking.registry import get_all_strategies


def render_header():
    """Render the page header with GitHub link."""
    import streamlit as st
    st.markdown(
        f"""
        <style>
        .github-float {{
            position: fixed;
            top: 18px;
            right: 32px;
            z-index: 9999;
            background: #fff;
            color: #333;
            border-radius: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 6px 18px;
            font-size: 15px;
            border: 1px solid #eee;
            opacity: 0.92;
            transition: opacity 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .github-float:hover {{ opacity: 1; }}
        .github-float svg {{ vertical-align: middle; margin-right: 2px; }}
        </style>
        <div class="github-float">
            <a href="{AppConfig.GITHUB_URL}" target="_blank" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24"><path fill="#333" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z"/></svg>
                GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


def setup_page_config():
    """Setup Streamlit page configuration and hide default elements."""
    import streamlit as st
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE, 
        layout=AppConfig.PAGE_LAYOUT
    )
    
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
    
    # Compact main title
    st.markdown(
        "<h1 style='margin-top: 0.5rem; margin-bottom: 1.2rem;'>RAG 分段策略测试</h1>",
        unsafe_allow_html=True
    )


def render_text_input(default_text: str) -> str:
    """Render text input area and return user input."""
    import streamlit as st
    st.subheader("原始文本")
    user_text = st.text_area("请输入文本", value=default_text, height=AppConfig.DEFAULT_TEXT_AREA_HEIGHT)
    
    # Display text statistics
    st.markdown(format_text_stats(user_text))
    return user_text


def render_strategy_selection() -> tuple[str, bool]:
    """Render strategy selection interface and return selected strategy and generate button state."""
    import streamlit as st
    st.subheader("分段策略")
    strategies = get_all_strategies()
    strategy_names = list(strategies.keys())
    selected_strategy = st.radio("分段策略", strategy_names, index=0)
    
    # Generate button (placed before strategy description)
    generate = st.button("生成")
    
    # Display strategy description and rating
    st.write("当前策略说明：")
    st.info(strategies[selected_strategy]["desc"])
    
    # Display rating stars
    rating = strategies[selected_strategy].get("rating", 0)
    stars = "★" * rating + "☆" * (5 - rating)
    st.markdown(f"推荐指数：<span style='color:gold;font-size:22px'>{stars}</span>", unsafe_allow_html=True)
    
    return selected_strategy, generate


def render_parameters(selected_strategy: str) -> Dict[str, Any]:
    """Render parameter inputs for the selected strategy."""
    import streamlit as st
    strategies = get_all_strategies()
    param_values = {}
    params = strategies[selected_strategy].get("params", [])
    
    if params:
        st.markdown("---")
        st.write("**参数设置：**")
        for param in params:
            key = f"param_{selected_strategy}_{param['name']}"
            if param["type"] == "int":
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
            elif param["type"] == "str":
                param_values[param["name"]] = st.text_input(
                    param["label"], 
                    value=param["default"], 
                    key=key
                )
    
    return param_values


def render_results(result: Any, strategy_name: str):
    """Render chunking results."""
    import streamlit as st
    st.subheader("结果")
    
    if result is None:
        st.info("请点击左侧'生成'按钮")
        return
    
    # Handle error messages
    if isinstance(result, str):
        st.error(result)
        return
    
    # Handle dictionary results (enriched chunks)
    if isinstance(result, list) and result and isinstance(result[0], dict):
        for i, chunk in enumerate(result):
            with st.container():
                st.markdown(f"---\n**区块 {i+1}**")
                st.json(chunk)
                st.markdown(format_chunk_stats(chunk), unsafe_allow_html=True)
    else:
        # Handle regular text chunks
        for i, chunk in enumerate(result):
            with st.container():
                st.markdown(f"---\n**区块 {i+1}**")
                st.code(chunk, language=None)
                st.markdown(format_chunk_stats(chunk), unsafe_allow_html=True)
