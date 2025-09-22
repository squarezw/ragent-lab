"""
Base classes and utilities for embedding models.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: List[List[float]]
    model_name: str
    texts: List[str]
    metadata: Dict[str, Any]
    model_size: Optional[str] = None
    
    def __post_init__(self):
        """Validate embeddings after initialization."""
        if not self.embeddings:
            raise ValueError("Embeddings cannot be empty")
        
        # Ensure all embeddings have the same dimension
        dims = [len(emb) for emb in self.embeddings]
        if len(set(dims)) > 1:
            raise ValueError("All embeddings must have the same dimension")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embeddings[0]) if self.embeddings else 0
    
    @property
    def count(self) -> int:
        """Get number of embeddings."""
        return len(self.embeddings)
    


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, name: str, description: str, dimension: int, 
                 max_tokens: Optional[int] = None, 
                 supports_batch: bool = True):
        self.name = name
        self.description = description
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.supports_batch = supports_batch
    
    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings for given texts."""
        pass
    
    
    def __str__(self) -> str:
        return f"{self.name} ({self.dimension}D)"
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(name='{self.name}', dimension={self.dimension})"
