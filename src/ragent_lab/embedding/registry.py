"""
Registry for embedding models.
"""

from typing import Dict, Any, List
from .base import EmbeddingModel
from .strategies import (
    openai_embedding_model,
    sentence_transformer_embedding_model,
    gme_embedding_model
)
from .aliyun_embedding import aliyun_embedding_model

# Global registry for embedding models
_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, model: EmbeddingModel, 
                  description: str = "", rating: int = 0,
                  params: List[Dict[str, Any]] = None):
    """Register an embedding model."""
    _EMBEDDING_MODELS[name] = {
        "model": model,
        "desc": description or model.description,
        "rating": rating,
        "params": params or []
    }


def get_all_models() -> Dict[str, Dict[str, Any]]:
    """Get all registered embedding models."""
    return _EMBEDDING_MODELS.copy()


def get_model(name: str) -> EmbeddingModel:
    """Get a specific embedding model by name."""
    if name not in _EMBEDDING_MODELS:
        raise ValueError(f"Model '{name}' not found in registry")
    return _EMBEDDING_MODELS[name]["model"]


def initialize_default_models():
    """Initialize default embedding models."""
    # OpenAI Embeddings
    register_model(
        "openai_text_embedding_3_small",
        openai_embedding_model("text-embedding-3-small"),
        "OpenAI 最新的小型嵌入模型，性能优秀，成本低",
        rating=5,
        params=[
            {
                "name": "api_key",
                "label": "API Key",
                "type": "str",
                "default": ""
            }
        ]
    )
    
    register_model(
        "openai_text_embedding_3_large", 
        openai_embedding_model("text-embedding-3-large"),
        "OpenAI 大型嵌入模型，最高质量",
        rating=5,
        params=[
            {
                "name": "api_key",
                "label": "API Key", 
                "type": "str",
                "default": ""
            }
        ]
    )
    
    # 指定的4个评测候选模型
    register_model(
        "bge_m3",
        sentence_transformer_embedding_model("BAAI/bge-m3"),
        "BAAI/bge-m3 - 1.2B多语言嵌入模型",
        rating=5,
        params=[]
    )
    
    register_model(
        "multilingual_e5_large",
        sentence_transformer_embedding_model("intfloat/multilingual-e5-large"),
        "intfloat/multilingual-e5-large - 1.2B E5多语言大模型",
        rating=5,
        params=[]
    )
    
    register_model(
        "e5_mistral_7b_instruct",
        sentence_transformer_embedding_model("intfloat/e5-mistral-7b-instruct"),
        "intfloat/e5-mistral-7b-instruct - 7B基于Mistral的E5指令模型",
        rating=4,
        params=[]
    )
    
    register_model(
        "gme_qwen2_vl_2b_instruct",
        gme_embedding_model("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"),
        "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct - 2B通用多模态嵌入模型（支持文本+图像）",
        rating=5,
        params=[]
    )
    
    # Aliyun DashScope Embeddings
    register_model(
        "aliyun_text_embedding_v2",
        aliyun_embedding_model("text-embedding-v2"),
        "阿里云 text-embedding-v2 - 多语言统一文本向量模型，支持100+语种",
        rating=5,
        params=[
            {
                "name": "api_key",
                "label": "Aliyun_API_KEY",
                "type": "str",
                "default": ""
            },
            {
                "name": "dimensions",
                "label": "向量维度",
                "type": "int",
                "default": 1024,
                "options": [2048, 1536, 1024, 768, 512, 256, 128, 64]
            }
        ]
    )

# Initialize default models when module is imported
initialize_default_models()
