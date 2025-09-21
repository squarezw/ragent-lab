"""
Streamlit entry point for the RAG Chunking Lab web application.

Run this file with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ragent_lab.web.app import create_app

if __name__ == "__main__":
    create_app()
