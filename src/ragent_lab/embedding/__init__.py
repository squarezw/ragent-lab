"""
Embedding module for RAG applications.
"""

from .base import EmbeddingResult, EmbeddingModel
from .registry import get_all_models, register_model
from .strategies import (
    openai_embedding,
    sentence_transformer_embedding
)

__all__ = [
    'EmbeddingResult',
    'EmbeddingModel', 
    'get_all_models',
    'register_model',
    'openai_embedding',
    'sentence_transformer_embedding'
]
