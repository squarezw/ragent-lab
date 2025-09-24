"""
Various embedding strategies and models for RAG applications.
"""

from typing import List
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel, EmbeddingResult


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model wrapper."""
    
    def __init__(self, model_name: str):
        super().__init__(
            name=f"OpenAI {model_name}",
            description=f"OpenAI {model_name} embedding model",
            dimension=self._get_dimension(model_name),
            max_tokens=8191
        )
        self.model_name = model_name
        self._client = None
    
    def _get_dimension(self, model_name: str) -> int:
        """Get embedding dimension for OpenAI model."""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimension_map.get(model_name, 1536)
    
    def _get_client(self, api_key: str = None):
        """Get OpenAI client."""
        try:
            import openai
            import os
            
            # Create client with explicit proxy handling for latest version
            client_kwargs = {}
            if api_key:
                client_kwargs['api_key'] = api_key
            
            # Handle proxy settings for latest OpenAI version
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
            
            # Only add proxy configuration if proxies are found
            if proxies:
                try:
                    # For latest OpenAI version, use http_client parameter
                    import httpx
                    # Create httpx client with proper proxy configuration
                    http_client = httpx.Client(proxies=proxies)
                    client_kwargs['http_client'] = http_client
                except Exception as proxy_error:
                    # Silently continue without proxy configuration
                    pass
            
            client = openai.OpenAI(**client_kwargs)
            return client
            
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAI client: {e}")
    
    def embed(self, texts: List[str], api_key: str = None, **kwargs) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        client = self._get_client(api_key)
        
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.name,
                texts=texts,
                metadata={
                    "model": self.model_name,
                    "usage": response.usage.__dict__ if hasattr(response, 'usage') else {},
                    "api_key_provided": bool(api_key)
                },
                model_size="API调用"
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Sentence Transformer embedding model wrapper."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        
        # Get dimension without loading the full model
        dimension = self._get_dimension(model_name)
        
        super().__init__(
            name=f"SentenceTransformer {model_name}",
            description=f"SentenceTransformer {model_name} model",
            dimension=dimension,
            max_tokens=None  # Depends on tokenizer
        )
    
    def _get_dimension(self, model_name: str) -> int:
        """Get embedding dimension for SentenceTransformer model."""
        dimension_map = {
            "BAAI/bge-m3": 1024,
            "intfloat/multilingual-e5-large": 1024,
            "intfloat/e5-mistral-7b-instruct": 4096,
            "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct": 1536
        }
        return dimension_map.get(model_name, 1024)  # Default to 1024
    
    def _get_model_size(self, model_name: str) -> str:
        """Get model size for SentenceTransformer model."""
        size_map = {
            "BAAI/bge-m3": "1.2B",
            "intfloat/multilingual-e5-large": "1.2B", 
            "intfloat/e5-mistral-7b-instruct": "7B",
            "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct": "2B"
        }
        return size_map.get(model_name, "未知")
    
    
    def _get_model(self):
        """Get or load SentenceTransformer model."""
        if self._model is None:
            # Special handling for GME model
            if "gme-Qwen2-VL" in self.model_name:
                # Use transformers for GME model
                from transformers import AutoModel
                import torch
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map='auto',
                    trust_remote_code=True
                )
            else:
                # Use sentence_transformers for other models
                self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings using SentenceTransformer or transformers."""
        model = self._get_model()
        
        try:
            # Special handling for GME model
            if "gme-Qwen2-VL" in self.model_name:
                # Use GME model's specific methods
                embeddings = model.get_text_embeddings(texts=texts)
                embeddings = embeddings.tolist()
            else:
                # Use sentence_transformers for other models
                embeddings = model.encode(texts, convert_to_tensor=False)
                embeddings = embeddings.tolist()
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.name,
                texts=texts,
                metadata={
                    "model": self.model_name,
                    "device": str(getattr(model, 'device', 'unknown')),
                    "batch_size": len(texts)
                },
                model_size=self._get_model_size(self.model_name)
            )
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")


class GMEEmbeddingModel(EmbeddingModel):
    """GME (General Multimodal Embedding) model wrapper."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        
        super().__init__(
            name=f"GME {model_name}",
            description=f"GME {model_name} multimodal embedding model",
            dimension=1536,  # GME models have 1536 dimensions
            max_tokens=32768  # GME models support up to 32768 tokens
        )
    
    def _get_model(self):
        """Get or load GME model."""
        if self._model is None:
            from transformers import AutoModel
            import torch
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
        return self._model
    
    def embed(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings using GME model."""
        model = self._get_model()
        
        try:
            # Use GME model's get_text_embeddings method
            embeddings = model.get_text_embeddings(texts=texts)
            embeddings = embeddings.tolist()
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.name,
                texts=texts,
                metadata={
                    "model": self.model_name,
                    "device": str(getattr(model, 'device', 'unknown')),
                    "batch_size": len(texts),
                    "model_type": "GME"
                },
                model_size="2B"
            )
        except Exception as e:
            raise RuntimeError(f"GME embedding failed: {e}")




# Factory functions for creating models
def openai_embedding_model(model_name: str) -> OpenAIEmbeddingModel:
    """Create OpenAI embedding model."""
    return OpenAIEmbeddingModel(model_name)




def sentence_transformer_embedding_model(model_name: str) -> SentenceTransformerEmbeddingModel:
    """Create SentenceTransformer embedding model."""
    return SentenceTransformerEmbeddingModel(model_name)


def gme_embedding_model(model_name: str) -> GMEEmbeddingModel:
    """Create GME embedding model."""
    return GMEEmbeddingModel(model_name)


# Convenience functions for direct embedding
def openai_embedding(texts: List[str], model_name: str = "text-embedding-3-small", 
                     api_key: str = None) -> EmbeddingResult:
    """Direct OpenAI embedding function."""
    model = openai_embedding_model(model_name)
    return model.embed(texts, api_key=api_key)


def sentence_transformer_embedding(texts: List[str], 
                                  model_name: str = "BAAI/bge-m3") -> EmbeddingResult:
    """Direct SentenceTransformer embedding function."""
    model = sentence_transformer_embedding_model(model_name)
    return model.embed(texts)
