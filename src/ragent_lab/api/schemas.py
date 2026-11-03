"""Bounded public request schemas. Never include secrets in validation responses."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ChunkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    strategy: str = Field(max_length=100)
    text: str = Field(min_length=1, max_length=50000)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('请输入文本')
        return value


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(max_length=100)
    texts: list[str] = Field(min_length=1, max_length=32)
    query: str = Field(default='', max_length=4000)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(''))

    @field_validator('texts')
    @classmethod
    def bounded_texts(cls, values):
        if any(not value.strip() or len(value) > 4000 for value in values):
            raise ValueError('每条文本需为 1–4000 个字符')
        if sum(map(len, values)) > 32000:
            raise ValueError('候选文本总长度不能超过 32000 个字符')
        return values
