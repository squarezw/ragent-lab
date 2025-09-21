"""
Tests for chunking strategies.
"""

import sys
from pathlib import Path
import unittest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ragent_lab.chunking import (
    fixed_length_chunking,
    sentence_based_chunking,
    paragraph_based_chunking
)


class TestChunkingStrategies(unittest.TestCase):
    """Test cases for chunking strategies."""
    
    def setUp(self):
        """Set up test data."""
        self.test_text = "这是一个测试文本。它包含多个句子。每个句子都有不同的内容。"
    
    def test_fixed_length_chunking(self):
        """Test fixed-length chunking."""
        chunks = fixed_length_chunking(self.test_text, chunk_size=10)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 10)
    
    def test_sentence_based_chunking(self):
        """Test sentence-based chunking."""
        chunks = sentence_based_chunking(self.test_text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
    
    def test_paragraph_based_chunking(self):
        """Test paragraph-based chunking."""
        text_with_paragraphs = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = paragraph_based_chunking(text_with_paragraphs)
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 3)


if __name__ == "__main__":
    unittest.main()
