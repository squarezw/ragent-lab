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
            max_tokens=8192
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
                
                # 创建客户端配置
                client_kwargs = {
                    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
                }
                
                if api_key:
                    client_kwargs['api_key'] = api_key
                
                # 处理代理设置
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']
                proxies = {}
                
                for var in proxy_vars:
                    if var in os.environ:
                        proxy_url = os.environ[var]
                        if var.upper() in ['HTTP_PROXY', 'http_proxy']:
                            proxies['http'] = proxy_url
                        elif var.upper() in ['HTTPS_PROXY', 'https_proxy']:
                            proxies['https'] = proxy_url
                        elif var == 'all_proxy':
                            proxies['http'] = proxy_url
                            proxies['https'] = proxy_url
                
                # 添加代理配置
                if proxies:
                    try:
                        import httpx
                        http_client = httpx.Client(proxies=proxies)
                        client_kwargs['http_client'] = http_client
                    except Exception:
                        # 静默忽略代理配置错误
                        pass
                
                client = openai.OpenAI(**client_kwargs)
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
                response = client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    dimensions=dimensions,
                    encoding_format="float"
                )
                
                embeddings = [data.embedding for data in response.data]
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model_name=self.name,
                    texts=texts,
                    metadata={
                        "model": self.model_name,
                        "dimensions": dimensions,
                        "usage": response.usage.__dict__ if hasattr(response, 'usage') else {},
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
