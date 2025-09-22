"""
Main Streamlit application for the RAG lab (chunking and embedding).
"""

from typing import Any, Dict

from ..config.settings import DEFAULT_TEXT, AppConfig
from ..chunking.registry import get_all_strategies
from ..stats import StatsManager
from .components import (
    setup_page_config, 
    render_text_input, 
    render_strategy_selection, 
    render_parameters, 
    render_results
)
from .embedding_app import create_embedding_app


def initialize_session_state():
    """Initialize session state variables."""
    import streamlit as st
    if 'result' not in st.session_state:
        st.session_state['result'] = None
    if 'result_strategy' not in st.session_state:
        st.session_state['result_strategy'] = None


def handle_user_tracking():
    """Handle user access tracking."""
    import streamlit as st
    stats_manager = StatsManager(AppConfig.STATS_FILE)
    
    # Get user information (simplified version without streamlit_javascript)
    try:
        user_agent = "Unknown"
        ip = "Unknown"
    except:
        user_agent = None
        ip = None
    
    # Track access if not already logged in this session
    if user_agent and ip and not st.session_state.get("stats_logged"):
        stats_manager.track_user_access(ip, user_agent)
        st.session_state["stats_logged"] = True


def process_chunking_request(user_text: str, selected_strategy: str, param_values: Dict[str, Any]) -> Any:
    """Process chunking request and return results."""
    strategies = get_all_strategies()
    func = strategies[selected_strategy]["func"]
    
    try:
        # Handle special parameter processing
        if param_values:
            # Special handling for keywords parameter
            if "keywords" in param_values:
                param_values["keywords"] = [
                    k.strip() for k in param_values["keywords"].split(",") if k.strip()
                ]
            result = func(user_text, **param_values)
        else:
            result = func(user_text)
        
        return result
    except Exception as e:
        return f"分段出错: {e}"


def create_app():
    """Create and run the main Streamlit application with navigation."""
    import streamlit as st
    
    # Setup page configuration
    setup_page_config()
    
    # Add GitHub link (fixed position, always visible)
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
    
    # Set wide layout for better embedding visualization
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Navigation sidebar
    st.sidebar.title("RAG Lab")
    
    # Simple radio selection without complex styling
    page = st.sidebar.radio(
        "选择功能",
        ["文本分段", "Embedding 测试"],
        help="选择要使用的RAG功能"
    )
    
    if page == "文本分段":
        # Chunking page - show chunking title
        st.markdown(
            "<h1 style='margin-top: 0.5rem; margin-bottom: 1.2rem;'>RAG 分段策略测试</h1>",
            unsafe_allow_html=True
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Handle user tracking
        handle_user_tracking()
        
        # Create three-column layout
        col1, col2, col3 = st.columns(AppConfig.get_column_ratios())
        
        with col1:
            # Text input section
            user_text = render_text_input(DEFAULT_TEXT)
        
        with col2:
            # Strategy selection section (now includes generate button)
            selected_strategy, generate = render_strategy_selection()
            
            # Parameter inputs
            param_values = render_parameters(selected_strategy)
            
            # Process chunking request
            if generate and user_text.strip():
                result = process_chunking_request(user_text, selected_strategy, param_values)
                st.session_state['result'] = result
                st.session_state['result_strategy'] = selected_strategy
        
        with col3:
            # Results section
            render_results(st.session_state['result'], st.session_state.get('result_strategy'))
    
    elif page == "Embedding 测试":
        # Embedding page
        create_embedding_app()


if __name__ == "__main__":
    create_app()
