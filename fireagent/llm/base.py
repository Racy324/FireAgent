"""LLM 客户端抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMClientError(RuntimeError):
    """LLM 调用失败时抛出的领域异常。"""


class LLMMessage(BaseModel):
    """OpenAI-compatible message 结构。"""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """LLM 响应结构。"""

    model_config = ConfigDict(extra="ignore")

    content: str
    model: str = ""
    provider: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """FireAgent LLM 客户端抽象基类。"""

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """根据 messages 生成回答。"""

    def generate_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """流式生成回答，逐 token 返回。默认回退到非流式 generate。"""
        response = self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        yield response.content

