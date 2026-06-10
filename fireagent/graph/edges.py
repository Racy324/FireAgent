"""LangGraph 条件路由函数。"""

from __future__ import annotations

from typing import Literal

from fireagent.graph.state import FireAgentState


def route_after_intent(state: FireAgentState) -> Literal["answer_generate", "query_rewrite"]:
    """根据意图路由到直接回答或 RAG 流程。"""
    intent = str(state.get("intent", "") or "").lower()
    if intent in {"chat", "reject"}:
        return "answer_generate"
    return "query_rewrite"


def route_after_sufficiency(state: FireAgentState) -> Literal["context_build", "web_search"]:
    """根据本地证据充分性决定是否联网兜底。"""
    if bool(state.get("evidence_sufficient", False)):
        return "context_build"
    return "web_search"

