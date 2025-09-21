"""
RAG Chunking Lab - A comprehensive toolkit for text chunking strategies in RAG applications.

This package provides various text chunking strategies, statistics tracking,
and a web interface for testing and comparing different chunking approaches.
"""

__version__ = "1.0.0"
__author__ = "RAG Lab Team"
__description__ = "A comprehensive toolkit for text chunking strategies in RAG applications"

# Import main modules for easy access
from . import chunking
from . import stats
from . import config
from . import utils
from . import web

__all__ = ['chunking', 'stats', 'config', 'utils', 'web']
