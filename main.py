"""
Main entry point for the RAG Chunking Lab application.

This script can be used to run the Streamlit web interface or execute
chunking strategies from the command line.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ragent_lab.web.app import create_app
from ragent_lab.chunking.registry import get_all_strategies
from ragent_lab.config.settings import DEFAULT_TEXT


def run_web_app():
    """Run the Streamlit web application."""
    import subprocess
    import sys
    import os
    
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(current_dir, "streamlit_app.py")
    
    # 启动Streamlit应用
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_app_path])


def run_cli_demo():
    """Run a command-line demo of chunking strategies."""
    print("RAG Chunking Lab - Command Line Demo")
    print("=" * 50)
    
    strategies = get_all_strategies()
    
    for name, strategy_info in strategies.items():
        print(f"\n=== {name} ===")
        print(strategy_info["desc"])
        
        func = strategy_info["func"]
        
        # Handle parameters
        params = {}
        if "params" in strategy_info:
            for param_info in strategy_info["params"]:
                param_name = param_info["name"]
                default = param_info["default"]
                label = param_info["label"]
                
                if param_info["type"] == "str" and param_name == "keywords":
                    # Special handling for keywords
                    params[param_name] = default.split(",")
                else:
                    params[param_name] = default
        
        # Execute strategy
        try:
            chunks = func(DEFAULT_TEXT, **params)
            
            # Display results
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                for i, chunk in enumerate(chunks):
                    print(f"\n[区块 {i+1}]")
                    for k, v in chunk.items():
                        print(f"{k}: {v}")
            else:
                for i, chunk in enumerate(chunks):
                    preview = chunk[:70] + "..." if len(chunk) > 70 else chunk
                    print(f"区块 {i+1}: {preview}")
        except Exception as e:
            print(f"执行出错: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Chunking Lab")
    parser.add_argument(
        "mode", 
        choices=["web", "cli"], 
        help="Run mode: 'web' for Streamlit app, 'cli' for command line demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        run_web_app()
    elif args.mode == "cli":
        run_cli_demo()


if __name__ == "__main__":
    main()
