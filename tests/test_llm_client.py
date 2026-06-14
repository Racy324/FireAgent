"""LLM 客户端与回答生成节点测试。"""

from __future__ import annotations

from fireagent.graph.nodes import FireAgentGraphNodes
from fireagent.llm.openai_compatible import OpenAICompatibleLLMClient
from fireagent.llm import BaseLLMClient, LLMMessage, LLMResponse
from fireagent.utils.config import FireAgentConfig


class FakeLLMClient(BaseLLMClient):
    """测试用 LLM 客户端。"""

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """返回固定回答，并确认 prompt 已包含证据上下文。"""
        assert messages
        assert "本地论文证据1" in messages[-1].content
        return LLMResponse(content="这是 LLM 基于证据生成的回答。", model="fake", provider="fake")


def test_answer_generate_node_uses_llm_when_context_exists() -> None:
    """有证据上下文且 LLM 启用时，回答生成节点应调用 LLM 客户端。"""
    config = FireAgentConfig(
        llm={
            "enabled": True,
            "provider": "openai_compatible",
            "base_url": "https://example.com/v1",
            "api_key": "test-key",
            "model": "test-model",
        }
    )
    nodes = FireAgentGraphNodes(config=config, llm_client=FakeLLMClient())

    update = nodes.answer_generate_node(
        {
            "user_query": "隧道火灾烟气有什么影响",
            "intent": "rag",
            "final_context": "[本地论文证据1]\n正文：烟气会降低能见度。",
            "citations": ["本地论文证据1: 示例论文，张三，2024，第1页"],
            "safety_notice": "",
            "errors": [],
        }
    )

    assert update["final_answer"] == "这是 LLM 基于证据生成的回答。"


def test_ollama_embedding_and_local_flagembedding_reranker_config_is_supported() -> None:
    """配置对象应支持 Ollama embedding 与本地 FlagEmbedding reranker。"""
    config = FireAgentConfig(
        embedding={"provider": "ollama", "model_name": "bge-m3", "base_url": "http://localhost:11434"},
        reranker={
            "provider": "flagembedding",
            "model_name": "models/bge-reranker-v2-m3",
            "base_url": "http://localhost:11434",
        },
    )

    assert config.embedding.provider == "ollama"
    assert config.embedding.model_name == "bge-m3"
    assert config.reranker.provider == "flagembedding"
    assert config.reranker.model_name == "models/bge-reranker-v2-m3"


def test_openai_compatible_extracts_json_from_reasoning_when_content_empty() -> None:
    """Ollama/Qwen3 可能把 JSON 放在 reasoning 中，content 为空时应兼容提取。"""
    raw = {
        "model": "qwen3:8b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": '思考过程...\n{"intent":"rag","confidence":0.9}',
                }
            }
        ],
    }

    content = OpenAICompatibleLLMClient._extract_content(raw)

    assert '{"intent":"rag","confidence":0.9}' in content


def test_openai_compatible_rejects_reasoning_without_json_when_content_empty() -> None:
    """content 为空且 reasoning 不含 JSON 时，应判定为无有效内容。"""
    raw = {
        "model": "qwen3:8b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "只有思考过程，没有最终 JSON。",
                }
            }
        ],
    }

    try:
        OpenAICompatibleLLMClient._extract_content(raw)
    except Exception as exc:  # noqa: BLE001
        assert "内容为空" in str(exc) or "reasoning" in str(exc)
    else:
        raise AssertionError("reasoning 不含 JSON 时不应返回内容")
