"""LangGraph 条件路由函数。"""

from __future__ import annotations

from typing import Literal

from fireagent.graph.state import FireAgentState
from fireagent.retrieval.fallback_policy import FallbackAction


def route_after_intent(state: FireAgentState) -> Literal["answer_generate", "query_rewrite"]:
    """根据意图路由到直接回答或 RAG 流程。"""
    intent = str(state.get("intent", "") or "").lower()
    if intent in {"chat", "reject"}:
        return "answer_generate"
    return "query_rewrite"


def route_after_sufficiency(state: FireAgentState) -> Literal["context_build", "web_search"]:
    """根据 fallback decision 决定是否联网兜底。"""
    fallback_decision = state.get("fallback_decision")
    if fallback_decision is not None:
        action = getattr(fallback_decision, "action", None)
        if action == FallbackAction.USE_WEB:
            return "web_search"
        # answer_local, refuse, ask_clarify, answer_insufficient 都走本地回答
        return "context_build"
    # 兼容旧逻辑
    if bool(state.get("evidence_sufficient", False)):
        return "context_build"
    return "web_search"

