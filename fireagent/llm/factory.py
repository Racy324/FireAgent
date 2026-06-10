"""LLM 客户端工厂。"""

from __future__ import annotations

from fireagent.llm.base import BaseLLMClient, LLMClientError
from fireagent.llm.openai_compatible import OpenAICompatibleLLMClient
from fireagent.utils.config import FireAgentConfig, get_config


def create_llm_client(config: FireAgentConfig | None = None) -> BaseLLMClient:
    """根据配置创建 LLM 客户端。"""
    cfg = config or get_config()
    provider = cfg.llm.provider.lower()
    if provider in {"openai_compatible", "openai-compatible", "dashscope"}:
        return OpenAICompatibleLLMClient(config=cfg)
    raise LLMClientError(f"不支持的 LLM provider：{cfg.llm.provider}")

