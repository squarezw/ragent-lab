"""
Registry for managing chunking strategies and their configurations.
"""

from typing import Dict, Any, List, Callable
from .strategies import (
    fixed_length_chunking,
    sentence_based_chunking,
    paragraph_based_chunking,
    sliding_window_chunking,
    semantic_chunking,
    recursive_chunking,
    context_enriched_chunking,
    modality_specific_chunking,
    agentic_chunking,
    subdocument_chunking,
    hybrid_chunking
)


class ChunkingStrategyRegistry:
    """Registry for managing chunking strategies and their metadata."""
    
    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all default chunking strategies."""
        strategies_config = {
            "固定长度分块": {
                "func": fixed_length_chunking,
                "desc": "将文本按固定字符长度分块，简单高效，适合无结构文本。",
                "rating": 1,
                "params": [
                    {"name": "chunk_size", "type": "int", "default": 100, "label": "分块长度（字符数）"}
                ]
            },
            "基于句子的分块": {
                "func": sentence_based_chunking,
                "desc": "按句子边界分块，适合语义连贯的文本。中英文自适应。",
                "rating": 2
            },
            "基于段落的分块": {
                "func": paragraph_based_chunking,
                "desc": "按段落分块，适合结构清晰、段落分明的文档。",
                "rating": 2
            },
            "滑动窗口分块": {
                "func": sliding_window_chunking,
                "desc": "滑动窗口分块，解决分块边界信息丢失问题，适合上下文相关任务。",
                "rating": 3,
                "params": [
                    {"name": "window_size", "type": "int", "default": 3, "label": "窗口大小（句子数）"},
                    {"name": "stride", "type": "int", "default": 2, "label": "滑动步长"}
                ]
            },
            "语义分块": {
                "func": semantic_chunking,
                "desc": "基于句子嵌入相似度的语义分块，自动聚合相关句子。",
                "rating": 3,
                "params": [
                    {"name": "threshold", "type": "float", "default": 0.85, "label": "相似度阈值（0~1）"}
                ]
            },
            "递归分块": {
                "func": recursive_chunking,
                "desc": "递归分块，优先按自然边界（段落/句子）分割，保证分块长度和结构。",
                "rating": 4,
                "params": [
                    {"name": "chunk_size", "type": "int", "default": 150, "label": "分块长度（字符数）"},
                    {"name": "chunk_overlap", "type": "int", "default": 20, "label": "分块重叠（字符数）"}
                ]
            },
            "语境丰富分块": {
                "func": context_enriched_chunking,
                "desc": "为每个分块添加上下文标题，便于理解和检索。",
                "rating": 3
            },
            "多模态分块": {
                "func": modality_specific_chunking,
                "desc": "针对多模态内容（如代码+文本）分块，适合混合内容场景。",
                "rating": 4
            },
            "Agent分块": {
                "func": agentic_chunking,
                "desc": "模拟 LLM 决策的分块策略，根据文本结构动态选择分块方式。",
                "rating": 5,
                "params": [
                    {"name": "text_length", "type": "int", "default": 1000, "label": "文本长度阈值"}
                ]
            },
            "子文档分块": {
                "func": subdocument_chunking,
                "desc": "提取包含关键词的子文档，适合聚焦特定主题内容。",
                "rating": 3,
                "params": [
                    {"name": "keywords", "type": "str", "default": "完善,电话,邮箱,###", "label": "关键词（逗号分隔）"}
                ]
            },
            "混合分块": {
                "func": hybrid_chunking,
                "desc": "组合多种分块策略，先按段落再对长段落做语义分块。",
                "rating": 3,
                "params": [
                    {"name": "chunk_len", "type": "int", "default": 500, "label": "长段落分块阈值（字符数）"}
                ]
            }
        }
        
        for name, config in strategies_config.items():
            self.register_strategy(name, **config)
    
    def register_strategy(self, name: str, func: Callable, desc: str, rating: int = 1, 
                         params: List[Dict[str, Any]] = None):
        """Register a new chunking strategy."""
        self._strategies[name] = {
            "func": func,
            "desc": desc,
            "rating": rating,
            "params": params or []
        }
    
    def get_strategy(self, name: str) -> Dict[str, Any]:
        """Get a strategy by name."""
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered strategies."""
        return self._strategies.copy()


# Global registry instance
_registry = ChunkingStrategyRegistry()


def get_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Get all registered chunking strategies."""
    return _registry.get_all_strategies()


def register_strategy(name: str, func: Callable, desc: str, rating: int = 1, 
                     params: List[Dict[str, Any]] = None):
    """Register a new chunking strategy."""
    _registry.register_strategy(name, func, desc, rating, params)
