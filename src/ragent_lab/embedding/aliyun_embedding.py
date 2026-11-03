"""
独立的阿里云嵌入模型实现，避免与其他模块的 torch 导入冲突
"""

import os
import warnings
from typing import List
from .base import EmbeddingModel, EmbeddingResult


class AliyunEmbeddingModel(EmbeddingModel):
    """阿里云 DashScope 嵌入模型包装器 - 独立实现避免 torch 冲突"""
    
    def __init__(self, model_name: str, dimensions: int = 1536):
        super().__init__(
            name=f"Aliyun {model_name}",
            description=f"Aliyun {model_name} embedding model",
            dimension=dimensions,
            max_tokens=2048 if model_name == "text-embedding-v2" else 8192
        )
        self.model_name = model_name
        self.dimensions = dimensions
        self._client = None
    
    def _get_client(self, api_key: str = None):
        """获取阿里云 DashScope 客户端"""
        try:
            # 抑制所有可能的警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                import openai
                
                client = openai.OpenAI(
                    api_key=api_key or os.getenv('Aliyun_API_KEY'),
                    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                    timeout=90.0, max_retries=1,
                )
                return client
                
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        except Exception as e:
            raise RuntimeError(f"创建阿里云客户端失败: {e}")
    
    def embed(self, texts: List[str], api_key: str = None, **kwargs) -> EmbeddingResult:
        """使用阿里云 DashScope API 生成嵌入向量"""
        # 抑制所有可能的警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            client = self._get_client(api_key)
            
            # 获取维度参数，优先使用传入的参数，否则使用默认值
            dimensions = kwargs.get('dimensions', self.dimensions)
            
            try:
                embeddings = []
                usage = {}
                batch_size = 25 if self.model_name == "text-embedding-v2" else 10
                for offset in range(0, len(texts), batch_size):
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=texts[offset:offset + batch_size],
                        **({"dimensions": dimensions} if self.model_name != "text-embedding-v2" else {}),
                        encoding_format="float"
                    )
                    embeddings.extend(data.embedding for data in response.data)
                    if getattr(response, 'usage', None):
                        for name, value in response.usage.__dict__.items():
                            if isinstance(value, (int, float)):
                                usage[name] = usage.get(name, 0) + value

                return EmbeddingResult(
                    embeddings=embeddings,
                    model_name=self.name,
                    texts=texts,
                    metadata={
                        "model": self.model_name,
                        "dimensions": dimensions,
                        "usage": usage,
                        "api_key_provided": bool(api_key),
                        "provider": "aliyun"
                    },
                    model_size="API调用"
                )
            except Exception as e:
                raise RuntimeError(f"阿里云嵌入失败: {e}")


def aliyun_embedding_model(model_name: str, dimensions: int = 1536) -> AliyunEmbeddingModel:
    """创建阿里云嵌入模型"""
    return AliyunEmbeddingModel(model_name, dimensions)


def aliyun_embedding(texts: List[str], model_name: str = "text-embedding-v2", 
                     api_key: str = None, dimensions: int = 1536) -> EmbeddingResult:
    """直接调用阿里云嵌入的便捷函数"""
    model = aliyun_embedding_model(model_name, dimensions)
    return model.embed(texts, api_key=api_key)
