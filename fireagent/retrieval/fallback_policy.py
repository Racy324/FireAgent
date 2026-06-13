"""Web Fallback 决策策略。

将"本地证据是否充分"和"下一步应该做什么"分离。
SufficiencyChecker 只判断证据够不够，FallbackPolicy 决定具体动作。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from fireagent.retrieval.schema import RerankedRetrievalResult, SufficiencyResult
from fireagent.utils.config import FireAgentConfig, get_config


class FallbackAction(str, Enum):
    """Fallback 决策动作。"""

    ANSWER_LOCAL = "answer_local"           # 本地证据足够，直接回答
    USE_WEB = "use_web"                     # 需要联网补充
    REFUSE = "refuse"                       # 危险请求，拒答
    ASK_CLARIFY = "ask_clarify"             # 问题模糊，要求澄清
    ANSWER_INSUFFICIENT = "answer_insufficient"  # 本地证据不足，说明不足


class FallbackDecision(BaseModel):
    """Fallback 决策结果。"""

    model_config = ConfigDict(extra="ignore")

    action: FallbackAction
    reason: str
    confidence: float = 1.0
    required_source_type: Optional[str] = None
    priority_domains: list[str] = Field(default_factory=list)
    query_flags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class FallbackPolicy:
    """基于规则的 Fallback 决策器。

    决策优先级：
    1. 危险请求 → refuse
    2. 明确要求本地知识库 → answer_insufficient（不联网）
    3. 时间敏感/官方来源需求 → use_web
    4. 本地证据足够 → answer_local
    5. 本地证据不足 → use_web（如果允许）或 answer_insufficient
    """

    def __init__(self, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()

    def decide(
        self,
        query: str,
        intent: str,
        sufficiency: SufficiencyResult,
        candidates: list[RerankedRetrievalResult] | None = None,
    ) -> FallbackDecision:
        """根据问题、意图和充分性结果做出 fallback 决策。"""
        query_flags: list[str] = []
        metadata = dict(sufficiency.metadata) if sufficiency.metadata else {}

        # 检测关键词
        temporal_kws = [kw for kw in self.config.web.temporal_keywords if kw in query]
        official_kws = [kw for kw in self.config.web.official_keywords if kw in query]
        local_scope_kws = [kw for kw in self.config.web.local_scope_keywords if kw in query]
        harmful_kws = [kw for kw in self.config.web.harmful_keywords if kw in query]
        ambiguous_kws = [kw for kw in self.config.web.ambiguous_keywords if kw in query]

        if temporal_kws:
            query_flags.append("temporal")
        if official_kws:
            query_flags.append("official")
        if local_scope_kws:
            query_flags.append("local_scope")
        if harmful_kws:
            query_flags.append("harmful")
        if ambiguous_kws:
            query_flags.append("ambiguous")

        # ── 规则 1：危险请求优先拒答 ──
        if harmful_kws and self.config.web.disable_for_harmful:
            return FallbackDecision(
                action=FallbackAction.REFUSE,
                reason="harmful_request_detected",
                confidence=1.0,
                query_flags=query_flags,
                metadata={**metadata, "harmful_keywords": harmful_kws},
            )

        # ── 规则 2：明确要求本地知识库时不联网 ──
        if (
            local_scope_kws
            and not sufficiency.sufficient
            and self.config.web.disable_for_local_scope
        ):
            return FallbackDecision(
                action=FallbackAction.ANSWER_INSUFFICIENT,
                reason="local_scope_requested_but_evidence_insufficient",
                confidence=0.95,
                query_flags=query_flags,
                metadata={**metadata, "local_scope_keywords": local_scope_kws},
            )

        # ── 规则 3：时间敏感问题强制联网 ──
        if (
            temporal_kws
            and self.config.web.force_web_temporal
            and self.config.rag.enable_web_fallback
        ):
            return FallbackDecision(
                action=FallbackAction.USE_WEB,
                reason="temporal_information_required",
                confidence=0.9,
                required_source_type="current_web",
                priority_domains=["gov.cn", "mem.gov.cn", "119.gov.cn"],
                query_flags=query_flags,
                metadata={**metadata, "temporal_keywords": temporal_kws},
            )

        # ── 规则 4：标准/法规/政策类问题优先联网 ──
        if (
            official_kws
            and self.config.web.force_web_official
            and self.config.rag.enable_web_fallback
        ):
            return FallbackDecision(
                action=FallbackAction.USE_WEB,
                reason="official_information_required",
                confidence=0.9,
                required_source_type="official_web",
                priority_domains=["gov.cn", "mem.gov.cn", "119.gov.cn"],
                query_flags=query_flags,
                metadata={**metadata, "official_keywords": official_kws},
            )

        # ── 规则 5：本地证据足够 → 直接回答 ──
        if sufficiency.sufficient:
            return FallbackDecision(
                action=FallbackAction.ANSWER_LOCAL,
                reason="local_evidence_sufficient",
                confidence=0.85,
                query_flags=query_flags,
                metadata=metadata,
            )

        # ── 规则 6：模糊指代问题先澄清 ──
        if ambiguous_kws and self.config.web.ask_clarify_for_ambiguous:
            return FallbackDecision(
                action=FallbackAction.ASK_CLARIFY,
                reason="ambiguous_reference_detected",
                confidence=0.75,
                query_flags=query_flags,
                metadata={**metadata, "ambiguous_keywords": ambiguous_kws},
            )

        # ── 规则 7：本地证据不足 → 联网或说明不足 ──
        if self.config.rag.enable_web_fallback:
            return FallbackDecision(
                action=FallbackAction.USE_WEB,
                reason="local_evidence_insufficient_and_web_allowed",
                confidence=0.65,
                query_flags=query_flags,
                metadata=metadata,
            )

        return FallbackDecision(
            action=FallbackAction.ANSWER_INSUFFICIENT,
            reason="local_evidence_insufficient_and_web_disabled",
            confidence=0.8,
            query_flags=query_flags,
            metadata=metadata,
        )


def decide_fallback(
    query: str,
    intent: str,
    sufficiency: SufficiencyResult,
    candidates: list[RerankedRetrievalResult] | None = None,
    config: FireAgentConfig | None = None,
) -> FallbackDecision:
    """便捷函数。"""
    return FallbackPolicy(config=config).decide(
        query=query,
        intent=intent,
        sufficiency=sufficiency,
        candidates=candidates,
    )
