"""
Main Streamlit application for the RAG chunking lab.
"""

from typing import Any, Dict

from ..config.settings import DEFAULT_TEXT, AppConfig
from ..chunking.registry import get_all_strategies
from ..stats import StatsManager
from .components import (
    render_header, 
    setup_page_config, 
    render_text_input, 
    render_strategy_selection, 
    render_parameters, 
    render_results
)


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
    
    # Get user information (only in Streamlit context)
    try:
        from streamlit_javascript import st_javascript
        user_agent = st_javascript("window.navigator.userAgent")
        ip = st_javascript("fetch('https://api.ipify.org?format=text').then(r => r.text())")
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
    """Create and run the main Streamlit application."""
    import streamlit as st
    
    # Setup page configuration
    setup_page_config()
    render_header()
    
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


if __name__ == "__main__":
    create_app()
