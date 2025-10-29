"""
Text utility functions for character counting and statistics.
"""

import re
from typing import Union, Dict, Any


def count_chars(text: str) -> int:
    """Count total characters in text."""
    return len(text)


def count_chinese_chars(text: str) -> int:
    """Count Chinese characters in text."""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars)


def format_text_stats(text: str) -> str:
    """
    Format text statistics for display.
    
    Args:
        text: Input text
        
    Returns:
        Formatted statistics string
    """
    char_count = count_chars(text)
    chinese_count = count_chinese_chars(text)
    return f"**字符数：** {char_count}  |  **中文字数：** {chinese_count}"


def get_chunk_stats(chunk: Union[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    Get statistics for a text chunk.
    
    Args:
        chunk: Text chunk (string or dict with 'content' key)
        
    Returns:
        Dictionary with character and Chinese character counts
    """
    if isinstance(chunk, dict):
        content = chunk.get('content', '')
    else:
        content = chunk
    
    return {
        'char_count': count_chars(content),
        'chinese_char_count': count_chinese_chars(content)
    }


def format_chunk_stats(chunk: Union[str, Dict[str, Any]]) -> str:
    """
    Format chunk statistics for display.
    
    Args:
        chunk: Text chunk
        
    Returns:
        Formatted statistics string
    """
    stats = get_chunk_stats(chunk)
    return f"<span style='color:gray'>字符数：{stats['char_count']} | 中文字数：{stats['chinese_char_count']}</span>"
