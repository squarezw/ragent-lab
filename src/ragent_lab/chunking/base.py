"""
Base classes and utilities for text chunking strategies.
"""

import re
import nltk
import os
from typing import List, Union, Dict, Any


def setup_nltk_data():
    """Setup NLTK data path to use local data."""
    NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'nltk_data')
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_PATH)


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Chinese or English based on character ratio.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'chinese' or 'english'
    """
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    ratio = len(chinese_chars) / max(len(text), 1)
    return 'chinese' if ratio > 0.2 else 'english'


def split_chinese_sentences(text: str) -> List[str]:
    """Split Chinese text by sentence boundaries."""
    return [s for s in re.split(r'[。！？:]', text) if s.strip()]


def split_english_sentences(text: str) -> List[str]:
    """Split English text by sentence boundaries using NLTK."""
    setup_nltk_data()
    return nltk.sent_tokenize(text)


def get_sentences(text: str) -> List[str]:
    """
    Get sentences from text, automatically detecting language.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    language = detect_language(text)
    if language == 'chinese':
        return split_chinese_sentences(text)
    else:
        return split_english_sentences(text)


class ChunkingResult:
    """Container for chunking results with metadata."""
    
    def __init__(self, chunks: List[Any], strategy_name: str, metadata: Dict[str, Any] = None):
        self.chunks = chunks
        self.strategy_name = strategy_name
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, index: int) -> Any:
        return self.chunks[index]
    
    def __iter__(self):
        return iter(self.chunks)
