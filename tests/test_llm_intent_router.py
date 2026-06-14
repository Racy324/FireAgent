"""LLM 意图路由器测试。"""

from __future__ import annotations

import pytest

from fireagent.graph.llm_router import LLMIntentRouter
from fireagent.graph.router_schema import IntentRouteResult
from fireagent.llm import BaseLLMClient, LLMMessage, LLMResponse
from fireagent.utils.config import FireAgentConfig


class FakeRouterLLM(BaseLLMClient):
    """返回预设内容的 fake LLM。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content, model="fake-router", provider="fake")


def _config() -> FireAgentConfig:
    return FireAgentConfig(
        router={
            "enabled": True,
            "mode": "hybrid",
            "base_url": "http://localhost:11434/v1",
            "api_key": "test-key",
            "model": "qwen3-4b",
            "temperature": 0.0,
            "max_tokens": 200,
            "confidence_threshold": 0.75,
            "low_confidence_threshold": 0.55,
            "fallback_to_rules": True,
        }
    )


def test_llm_router_parses_valid_json() -> None:
    """合法 JSON 应被解析为路由结果。"""
    client = FakeRouterLLM(
        '{"intent":"rag","sub_intent":"knowledge_qa","confidence":0.86,'
        '"need_rag":true,"need_memory":false,"need_safety_notice":false,'
        '"reason":"火灾知识问答"}'
    )
    router = LLMIntentRouter(config=_config(), llm_client=client)

    result = router.route("隧道火灾烟气对人员疏散有什么影响")

    assert result.intent == "rag"
    assert result.sub_intent == "knowledge_qa"
    assert result.confidence == pytest.approx(0.86)
    assert result.source == "llm"
    assert client.calls


def test_llm_router_falls_back_on_invalid_json() -> None:
    """非法 JSON 应回退到 legacy rules。"""
    router = LLMIntentRouter(config=_config(), llm_client=FakeRouterLLM("不是 JSON"))

    result = router.route("隧道火灾烟气对人员疏散有什么影响")

    assert result.intent == "rag"
    assert result.source == "fallback_rule"
    assert "回退" in result.reason


def test_llm_router_falls_back_on_invalid_intent() -> None:
    """非法 intent 应回退到 legacy rules。"""
    router = LLMIntentRouter(
        config=_config(),
        llm_client=FakeRouterLLM('{"intent":"unknown","confidence":0.99,"reason":"bad"}'),
    )

    result = router.route("总结火灾检测论文的主要方法")

    assert result.intent == "paper"
    assert result.source == "fallback_rule"


def test_llm_router_low_confidence_uses_conservative_fallback() -> None:
    """低置信度结果不应直接采纳。"""
    router = LLMIntentRouter(
        config=_config(),
        llm_client=FakeRouterLLM(
            '{"intent":"reject","sub_intent":"out_of_domain","confidence":0.44,'
            '"reason":"不确定"}'
        ),
    )

    result = router.route("烟雾里怎么判断方向")

    assert result.intent != "reject"
    assert result.source == "fallback_rule"
    assert "低置信度" in result.reason


def test_hard_guardrail_wins_before_llm() -> None:
    """明确偏好写入应由硬规则护栏处理，不调用 LLM。"""
    client = FakeRouterLLM('{"intent":"rag","confidence":0.99,"reason":"wrong"}')
    router = LLMIntentRouter(config=_config(), llm_client=client)

    result = router.route("以后所有火灾笔记都用中文，并在标题前加【火灾笔记】")

    assert result.intent == "chat"
    assert result.sub_intent == "preference_write"
    assert result.source == "hard_rule"
    assert client.calls == []


def test_router_disabled_uses_fallback_rules() -> None:
    """关闭 router 时应使用 legacy rules。"""
    config = FireAgentConfig(router={"enabled": False})
    router = LLMIntentRouter(config=config, llm_client=FakeRouterLLM("{}"))

    result = router.route("隧道火灾烟气对人员疏散有什么影响")

    assert result.intent == "rag"
    assert result.source == "fallback_rule"


def test_router_can_disable_legacy_fallback() -> None:
    """配置关闭规则回退时，LLM 失败应返回澄清类 chat。"""
    config = FireAgentConfig(
        router={
            "enabled": True,
            "fallback_to_rules": False,
            "base_url": "http://localhost:11434/v1",
            "api_key": "test-key",
            "model": "qwen3-4b",
        }
    )
    router = LLMIntentRouter(config=config, llm_client=FakeRouterLLM("bad json"))

    result = router.route("隧道火灾烟气对人员疏散有什么影响")

    assert result.intent == "chat"
    assert result.sub_intent == "route_uncertain"
    assert result.confidence == 0.0


def test_route_result_schema_rejects_bad_confidence() -> None:
    """路由结果 schema 应限制 confidence 范围。"""
    with pytest.raises(ValueError):
        IntentRouteResult(intent="rag", confidence=2.0)
