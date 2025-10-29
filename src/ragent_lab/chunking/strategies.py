"""
Various text chunking strategies for RAG applications.
"""

import re
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from .base import get_sentences, ChunkingResult


def fixed_length_chunking(text: str, chunk_size: int = 100) -> List[str]:
    """
    Split text into fixed-length chunks.
    
    Args:
        text: Input text
        chunk_size: Character count per chunk
        
    Returns:
        List of text chunks
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def sentence_based_chunking(text: str) -> List[str]:
    """
    Split text by sentence boundaries with language auto-detection.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    return get_sentences(text)


def paragraph_based_chunking(text: str) -> List[str]:
    """
    Split text by paragraphs (newline boundaries).
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    return [p.strip() for p in text.split('\n') if p.strip()]


def sliding_window_chunking(text: str, window_size: int = 3, stride: int = 2) -> List[str]:
    """
    Sliding window chunking based on sentences.
    
    Args:
        text: Input text
        window_size: Number of sentences per window
        stride: Step size for sliding window
        
    Returns:
        List of overlapping chunks
    """
    sentences = get_sentences(text)
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk = ' '.join(sentences[i:i+window_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def semantic_chunking(text: str, threshold: float = 0.65) -> List[str]:
    """
    Semantic chunking based on sentence embedding similarity.
    
    Args:
        text: Input text
        threshold: Similarity threshold for grouping sentences
        
    Returns:
        List of semantic chunks
    """
    # Load embedding model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    sentences = get_sentences(text)
    if len(sentences) <= 1:
        return sentences
    
    embeddings = model.encode(sentences)
    
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        # Calculate similarity between adjacent sentences
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(" ".join(current_chunk))
    return chunks


def recursive_chunking(text: str, chunk_size: int = 150, chunk_overlap: int = 20) -> List[str]:
    """
    Recursive chunking using LangChain implementation.
    
    Args:
        text: Input text
        chunk_size: Maximum chunk length in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


def context_enriched_chunking(text: str) -> List[Dict[str, str]]:
    """
    Create chunks with enriched context (titles).
    
    Args:
        text: Input text
        
    Returns:
        List of dictionaries with 'title' and 'content' keys
    """
    chunks = paragraph_based_chunking(text)
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        # Generate title (simplified - in practice, use LLM)
        title = f"区块_{i+1}: {chunk[:15]}..." if len(chunk) > 15 else chunk
        enriched_chunks.append({
            "title": title,
            "content": chunk
        })
    return enriched_chunks


def modality_specific_chunking(text: str) -> List[Dict[str, str]]:
    """
    Chunking for multi-modal content (e.g., code + text).
    
    Args:
        text: Input text
        
    Returns:
        List of dictionaries with 'type' and 'content' keys
    """
    chunks = []
    current_type = None
    buffer = ""
    
    # Simplified logic: determine type based on line prefixes
    for line in text.split('\n'):
        if line.startswith('    ') or line.startswith('def ') or line.startswith('import'):
            if current_type != 'code':
                if buffer: 
                    chunks.append({"type": current_type, "content": buffer})
                current_type = 'code'
                buffer = line + '\n'
            else:
                buffer += line + '\n'
        else:
            if current_type != 'text':
                if buffer: 
                    chunks.append({"type": current_type, "content": buffer})
                current_type = 'text'
                buffer = line + '\n'
            else:
                buffer += line + '\n'
    
    if buffer: 
        chunks.append({"type": current_type, "content": buffer})
    return chunks


def agentic_chunking(text: str, text_length: int = 1000) -> List[str]:
    """
    LLM-based adaptive chunking strategy (simulated).
    
    Args:
        text: Input text
        text_length: Length threshold for strategy selection
        
    Returns:
        List of text chunks
    """
    print("模拟LLM决策：分析文档结构并动态选择分块策略")
    if len(text) > text_length:
        return sliding_window_chunking(text)
    elif '\n' in text:
        return paragraph_based_chunking(text)
    else:
        return fixed_length_chunking(text, chunk_size=150)


def subdocument_chunking(text: str, keywords: List[str] = None) -> List[str]:
    """
    Extract subdocuments containing specific keywords.
    
    Args:
        text: Input text
        keywords: List of keywords to filter by
        
    Returns:
        List of matching paragraphs
    """
    if keywords is None:
        keywords = ['电话']
    
    paragraphs = paragraph_based_chunking(text)
    subdocs = []
    for para in paragraphs:
        if any(kw in para for kw in keywords):
            subdocs.append(para)
    return subdocs


def hybrid_chunking(text: str, chunk_len: int = 200) -> List[str]:
    """
    Hybrid chunking combining multiple strategies.
    
    Args:
        text: Input text
        chunk_len: Length threshold for semantic chunking
        
    Returns:
        List of text chunks
    """
    # Step 1: Paragraph-based chunking
    chunks = paragraph_based_chunking(text)
    
    # Step 2: Semantic chunking for long paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_len:
            final_chunks.extend(semantic_chunking(chunk))
        else:
            final_chunks.append(chunk)
    
    return final_chunks
