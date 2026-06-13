"""FireAgent LangGraph 工作流状态定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class FireAgentState(TypedDict, total=False):
    """在线问答工作流的共享状态。

    LangGraph 中每个节点读取这个状态，并返回部分字段更新。并行检索节点可能同时
    写入 errors，因此 errors 字段使用 list 相加作为 reducer。
    """

    user_query: str
    session_id: str
    conversation_context: str
    rewritten_queries: list[str]
    rewrite_result: Any
    intent: str
    intent_reason: str
    local_dense_results: list[Any]
    local_sparse_results: list[Any]
    fused_results: list[Any]
    reranked_results: list[Any]
    evidence_sufficient: bool
    sufficiency_result: Any
    fallback_decision: Any
    web_results: list[Any]
    final_context: str
    context_result: Any
    final_answer: str
    citations: list[str]
    candidate_citations: list[Any]
    used_citations: list[Any]
    used_citation_markers: list[str]
    invalid_citation_markers: list[str]
    long_term_memories: str
    long_term_memory_results: list[Any]
    safety_notice: str
    hallucination_warnings: list[str]
    route_next: str
    errors: Annotated[list[str], operator.add]


def create_initial_state(
    user_query: str,
    session_id: str = "",
    conversation_context: str = "",
    long_term_memories: str = "",
    long_term_memory_results: list[Any] | None = None,
) -> FireAgentState:
    """根据用户问题创建初始状态。"""
    return FireAgentState(
        user_query=user_query,
        session_id=session_id,
        conversation_context=conversation_context,
        rewritten_queries=[],
        intent="",
        local_dense_results=[],
        local_sparse_results=[],
        fused_results=[],
        reranked_results=[],
        evidence_sufficient=False,
        web_results=[],
        final_context="",
        final_answer="",
        citations=[],
        long_term_memories=long_term_memories,
        long_term_memory_results=long_term_memory_results or [],
        safety_notice="",
        hallucination_warnings=[],
        errors=[],
    )


def state_get_query(state: FireAgentState) -> str:
    """从状态中安全读取用户问题。"""
    return str(state.get("user_query", "") or "").strip()
