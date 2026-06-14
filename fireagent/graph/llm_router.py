"""混合式意图路由器：确定性护栏 + 小模型 JSON 分类 + legacy fallback。"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from fireagent.graph.router_schema import IntentRouteResult
from fireagent.llm import BaseLLMClient, LLMClientError, LLMMessage, OpenAICompatibleLLMClient
from fireagent.utils.config import FireAgentConfig, get_config


CHAT_PATTERNS = ("你好", "您好", "hello", "hi", "你是谁", "谢谢", "感谢")
PREFERENCE_INSTRUCTION_PATTERNS = (
    "默认",
    "我希望",
    "我不想",
    "固定采用",
    "一直用",
    "始终",
    "一律",
    "统一",
    "全部用",
    "都用",
    "偏好是",
    "偏好设置",
)
PREFERENCE_INSTRUCTION_REGEX = re.compile(
    r"(以后|默认|始终|一律|统一|全部|都).{0,30}(笔记|回答|输出|语言|中文|英文|格式)"
)
MEMORY_QUERY_PATTERNS = (
    "我的偏好",
    "偏好有什么",
    "偏好是什么",
    "长期记忆",
    "记住了什么",
    "你记得什么",
    "我设置过",
    "我之前设置",
    "应该用什么格式",
    "用什么格式",
    "用什么语言",
    "标题前缀",
    "标题前加",
)
MEMORY_QUERY_REGEX = re.compile(
    r"(笔记|回答|输出|标题|格式|语言).{0,20}(应该|需要|用什么|怎么|如何|是什么)"
)
HISTORY_QUERY_PATTERNS = (
    "之前问过",
    "刚才问过",
    "前面问过",
    "上次问过",
    "上一条问题",
    "历史问题",
    "问过什么",
    "刚才的问题",
)
PAPER_PATTERNS = ("总结", "综述", "概括", "对比", "比较", "论文", "文献", "作者", "摘要")
EMERGENCY_PATTERNS = ("怎么办", "如何逃生", "应急", "报警", "疏散路线", "自救", "逃生", "灭火器", "被困")
FIRE_DOMAIN_PATTERNS = (
    "火灾",
    "消防",
    "烟气",
    "烟雾",
    "浓烟",
    "火焰",
    "燃烧",
    "疏散",
    "排烟",
    "森林火",
    "隧道",
    "防火",
    "灭火",
    "火警",
)
HARMFUL_PATTERNS = ("纵火", "制造火灾", "规避消防", "绕过报警", "破坏灭火", "如何放火", "怎么纵火")


def is_history_query(query: str) -> bool:
    """判断是否在询问当前会话历史。"""
    return any(pattern in query for pattern in HISTORY_QUERY_PATTERNS)


def is_memory_query(query: str) -> bool:
    """判断是否在询问已保存的长期记忆或用户偏好。"""
    if any(pattern in query for pattern in MEMORY_QUERY_PATTERNS):
        return True
    return bool(MEMORY_QUERY_REGEX.search(query))


def is_preference_instruction(query: str) -> bool:
    """判断是否为用户偏好或长期记忆写入指令。"""
    return (
        any(pattern in query for pattern in PREFERENCE_INSTRUCTION_PATTERNS)
        or bool(PREFERENCE_INSTRUCTION_REGEX.search(query))
    )


def route_by_hard_guardrails(query: str) -> IntentRouteResult | None:
    """只处理高确定性护栏，不使用宽泛火灾关键词直接路由。"""
    normalized = query.strip().lower()
    if not normalized:
        return IntentRouteResult(
            intent="chat",
            sub_intent="empty_input",
            reason="空问题，按闲聊/澄清处理。",
            source="hard_rule",
        )
    if is_history_query(query):
        return IntentRouteResult(
            intent="chat",
            sub_intent="history_query",
            need_memory=False,
            reason="命中会话历史查询护栏。",
            source="hard_rule",
        )
    if is_preference_instruction(query):
        return IntentRouteResult(
            intent="chat",
            sub_intent="preference_write",
            need_memory=True,
            reason="命中用户偏好/长期记忆写入护栏。",
            source="hard_rule",
        )
    if is_memory_query(query):
        return IntentRouteResult(
            intent="chat",
            sub_intent="memory_query",
            need_memory=True,
            reason="命中长期记忆/用户偏好查询护栏。",
            source="hard_rule",
        )
    if any(pattern in query for pattern in HARMFUL_PATTERNS):
        return IntentRouteResult(
            intent="reject",
            sub_intent="harmful_fire_request",
            reason="命中危险/滥用请求护栏。",
            source="hard_rule",
        )
    if any(pattern in normalized for pattern in CHAT_PATTERNS) and len(normalized) <= 30:
        return IntentRouteResult(
            intent="chat",
            sub_intent="greeting",
            reason="命中明确闲聊问候护栏。",
            source="hard_rule",
        )
    if any(pattern in query for pattern in EMERGENCY_PATTERNS) and any(
        pattern in query for pattern in FIRE_DOMAIN_PATTERNS
    ):
        return IntentRouteResult(
            intent="emergency",
            sub_intent="emergency_escape",
            need_rag=True,
            need_safety_notice=True,
            reason="命中明确火灾应急安全护栏。",
            source="hard_rule",
        )
    return None


def route_by_legacy_rules(query: str, reason_prefix: str = "") -> IntentRouteResult:
    """模型失败或关闭时使用旧规则兜底。"""
    normalized = query.strip().lower()
    prefix = f"{reason_prefix}；" if reason_prefix else ""
    if not normalized:
        return IntentRouteResult(intent="chat", sub_intent="empty_input", reason=f"{prefix}空问题，按闲聊处理。", source="fallback_rule")
    guardrail = route_by_hard_guardrails(query)
    if guardrail is not None:
        return guardrail.model_copy(update={"source": "fallback_rule", "reason": f"{prefix}{guardrail.reason}"})
    if any(pattern in query for pattern in PAPER_PATTERNS) and any(pattern in query for pattern in FIRE_DOMAIN_PATTERNS):
        return IntentRouteResult(intent="paper", sub_intent="paper_summary", need_rag=True, reason=f"{prefix}legacy 命中论文总结/对比类问题。", source="fallback_rule")
    if any(pattern in query for pattern in FIRE_DOMAIN_PATTERNS):
        return IntentRouteResult(intent="rag", sub_intent="knowledge_qa", need_rag=True, reason=f"{prefix}legacy 命中火灾领域知识问答。", source="fallback_rule")
    return IntentRouteResult(intent="reject", sub_intent="out_of_domain", reason=f"{prefix}legacy 未命中火灾领域关键词。", source="fallback_rule")


class LLMIntentRouter:
    """FireAgent 混合式意图路由器。"""

    def __init__(
        self,
        config: FireAgentConfig | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.llm_client = llm_client

    def route(
        self,
        query: str,
        conversation_context: str = "",
        long_term_memories: str = "",
    ) -> IntentRouteResult:
        """返回结构化意图路由结果。"""
        guardrail = route_by_hard_guardrails(query)
        if guardrail is not None:
            return guardrail

        router_cfg = self.config.router
        if not router_cfg.enabled or router_cfg.mode == "rules":
            return route_by_legacy_rules(query, "router disabled")

        if router_cfg.mode == "shadow":
            legacy = route_by_legacy_rules(query, "shadow mode")
            shadow = self._try_llm_route(query, conversation_context, long_term_memories)
            if shadow is not None:
                return legacy.model_copy(
                    update={
                        "reason": (
                            f"{legacy.reason}；shadow_llm={shadow.intent}/"
                            f"{shadow.sub_intent}/{shadow.confidence:.2f}：{shadow.reason}"
                        )
                    }
                )
            return legacy

        result = self._try_llm_route(query, conversation_context, long_term_memories)
        if result is None:
            return self._fallback(query, "LLM 路由失败")
        if result.confidence < router_cfg.confidence_threshold:
            return self._fallback(query, f"LLM 路由低置信度 {result.confidence:.2f}")
        return result

    def _fallback(self, query: str, reason: str) -> IntentRouteResult:
        """根据配置决定是否回退 legacy rules。"""
        if self.config.router.fallback_to_rules:
            return route_by_legacy_rules(query, f"{reason}，回退")
        return IntentRouteResult(
            intent="chat",
            sub_intent="route_uncertain",
            confidence=0.0,
            reason=f"{reason}，且未启用规则回退。请补充更明确的问题意图。",
            source="fallback_rule",
        )

    def _try_llm_route(
        self,
        query: str,
        conversation_context: str,
        long_term_memories: str,
    ) -> IntentRouteResult | None:
        try:
            client = self.llm_client or self._create_router_client()
            response = client.generate(
                [
                    LLMMessage(role="system", content="你是 FireAgent 的意图路由器，只输出 JSON。"),
                    LLMMessage(
                        role="user",
                        content=self._build_prompt(query, conversation_context, long_term_memories),
                    ),
                ],
                temperature=self.config.router.temperature,
                max_tokens=self.config.router.max_tokens,
            )
            payload = _extract_json_object(response.content)
            result = IntentRouteResult.model_validate(payload)
            return result.model_copy(update={"source": "llm"})
        except (json.JSONDecodeError, ValidationError, LLMClientError, Exception):
            return None

    def _create_router_client(self) -> BaseLLMClient:
        router_cfg = self.config.router
        if router_cfg.provider.lower() not in {"openai_compatible", "openai-compatible", "dashscope"}:
            raise LLMClientError(f"不支持的 router provider：{router_cfg.provider}")
        router_config = FireAgentConfig(
            llm={
                "provider": router_cfg.provider,
                "base_url": router_cfg.base_url or self.config.llm.base_url,
                "api_key": router_cfg.api_key or self.config.llm.api_key,
                "model": router_cfg.model or self.config.llm.model,
                "temperature": router_cfg.temperature,
                "max_tokens": router_cfg.max_tokens,
                "timeout": router_cfg.timeout,
                "enabled": True,
            }
        )
        return OpenAICompatibleLLMClient(config=router_config)

    def _build_prompt(self, query: str, conversation_context: str, long_term_memories: str) -> str:
        return (
            "你是 FireAgent 的意图路由器。只输出 JSON，不要输出解释性正文。\n\n"
            "可选 intent：\n"
            "- chat：闲聊、会话历史查询、长期记忆查询、偏好写入确认，不需要论文证据。\n"
            "- reject：明显不属于火灾/消防安全领域，且不是用户偏好/记忆/会话问题。\n"
            "- emergency：火灾现场、逃生、报警、被困、浓烟、应急处置等安全优先问题。\n"
            "- paper：论文总结、文献综述、方法对比、作者/年份/文献整理。\n"
            "- rag：火灾领域普通知识问答，需要基于论文证据回答。\n\n"
            "输出字段：intent、sub_intent、confidence、need_rag、need_memory、"
            "need_safety_notice、reason。\n\n"
            f"用户问题：\n{query}\n\n"
            f"短期上下文：\n{conversation_context[:1200]}\n\n"
            f"长期记忆摘要：\n{long_term_memories[:800]}\n\n"
            "输出 JSON："
        )


def _extract_json_object(content: str) -> dict[str, Any]:
    """从模型输出中提取 JSON object。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise json.JSONDecodeError("No JSON object found", content, 0)
        text = text[start : end + 1]
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise json.JSONDecodeError("Router output is not object", content, 0)
    return raw
