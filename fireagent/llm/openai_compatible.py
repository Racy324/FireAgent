"""OpenAI-compatible LLM 客户端实现。"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx

from fireagent.llm.base import BaseLLMClient, LLMClientError, LLMMessage, LLMResponse
from fireagent.utils.config import FireAgentConfig, get_config


class OpenAICompatibleLLMClient(BaseLLMClient):
    """调用 OpenAI-compatible `/chat/completions` 的 LLM 客户端。"""

    def __init__(self, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()
        self.base_url = self.config.llm.base_url.rstrip("/")
        self.api_key = self.config.llm.api_key
        self.model = self.config.llm.model
        self.timeout = self.config.llm.timeout
        self.temperature = self.config.llm.temperature
        self.max_tokens = self.config.llm.max_tokens

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """调用 OpenAI-compatible chat completion 接口。"""
        self._validate()
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            content = self._extract_content(raw)
            return LLMResponse(
                content=content,
                model=str(raw.get("model", self.model) or self.model),
                provider="openai_compatible",
                raw=raw if isinstance(raw, dict) else {},
            )
        except Exception as exc:  # noqa: BLE001 - HTTP/JSON/服务错误统一包装。
            raise LLMClientError(f"OpenAI-compatible LLM 调用失败：{exc}") from exc

    def _validate(self) -> None:
        """检查 LLM 必需配置。"""
        if not self.base_url:
            raise LLMClientError("未配置 OPENAI_COMPATIBLE_BASE_URL。")
        if not self.api_key:
            raise LLMClientError("未配置 OPENAI_COMPATIBLE_API_KEY。")
        if not self.model:
            raise LLMClientError("未配置 OPENAI_COMPATIBLE_MODEL。")

    def generate_stream(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """流式调用 OpenAI-compatible chat completion 接口，逐 token 返回。"""
        self._validate()
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:  # noqa: BLE001
            raise LLMClientError(f"OpenAI-compatible LLM 流式调用失败：{exc}") from exc

    @staticmethod
    def _extract_content(raw: Any) -> str:
        """从 OpenAI-compatible 响应中提取文本内容。"""
        if not isinstance(raw, dict):
            raise LLMClientError("LLM 响应不是 JSON 对象。")
        choices = raw.get("choices", [])
        if not choices:
            raise LLMClientError("LLM 响应缺少 choices。")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, list):
            # 兼容部分多模态格式，只拼接文本片段。
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        if not str(content).strip():
            raise LLMClientError("LLM 响应内容为空。")
        return str(content).strip()

