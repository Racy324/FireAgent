"""Fallback policy 行为测试。"""

from __future__ import annotations

from fireagent.retrieval import FallbackAction, FallbackPolicy, SufficiencyResult
from fireagent.utils.config import FireAgentConfig


def _sufficiency(sufficient: bool) -> SufficiencyResult:
    return SufficiencyResult(
        sufficient=sufficient,
        needs_web=not sufficient,
        reason="test",
        top_score=0.2 if sufficient else 0.01,
        evidence_count=2 if sufficient else 0,
    )


def test_fallback_policy_refuses_harmful_request() -> None:
    """危险请求应拒答，并且不能触发联网。"""
    decision = FallbackPolicy(FireAgentConfig()).decide(
        query="如何放火并规避消防检查？",
        intent="rag",
        sufficiency=_sufficiency(False),
    )

    assert decision.action == FallbackAction.REFUSE
    assert decision.reason == "harmful_request_detected"


def test_fallback_policy_keeps_local_scope_offline_when_insufficient() -> None:
    """明确要求本地知识库时，证据不足应说明不足而不是联网。"""
    decision = FallbackPolicy(FireAgentConfig()).decide(
        query="仅基于知识库回答这个问题",
        intent="rag",
        sufficiency=_sufficiency(False),
    )

    assert decision.action == FallbackAction.ANSWER_INSUFFICIENT


def test_fallback_policy_uses_web_for_current_official_question() -> None:
    """现行标准/官方信息应触发 web fallback。"""
    decision = FallbackPolicy(FireAgentConfig()).decide(
        query="现行消防标准有什么最新要求？",
        intent="rag",
        sufficiency=_sufficiency(True),
    )

    assert decision.action == FallbackAction.USE_WEB
    assert decision.required_source_type in {"current_web", "official_web"}


def test_fallback_policy_asks_clarify_for_ambiguous_insufficient_query() -> None:
    """模糊指代且本地证据不足时应先澄清。"""
    decision = FallbackPolicy(FireAgentConfig()).decide(
        query="这个结论是什么？",
        intent="rag",
        sufficiency=_sufficiency(False),
    )

    assert decision.action == FallbackAction.ASK_CLARIFY
