"""评估样本分组规则。

根据 case 的 intent、tags、metadata 字段，返回样本所属的分组列表。
一条样本可以同时属于多个分组（多标签）。
"""

from __future__ import annotations

from typing import Any

from fireagent.evaluation.schema import EvaluationCase


# ── 分组常量 ──

GROUP_ALL = "all"
GROUP_LOCAL_ANSWERABLE = "local_answerable"
GROUP_LOCAL_SINGLE_DOC = "local_single_doc"
GROUP_LOCAL_MULTI_DOC = "local_multi_doc"
GROUP_ROUTER_MANUAL_DOC_HIT = "router_manual_doc_hit"
GROUP_FALLBACK_SAFETY_EXCLUDED = "fallback_safety_excluded"
GROUP_EMERGENCY = "emergency"
GROUP_EMERGENCY_SAFETY = "emergency_safety"
GROUP_EMERGENCY_LOCAL_EVIDENCE = "emergency_local_evidence"

ALL_GROUPS = [
    GROUP_ALL,
    GROUP_LOCAL_ANSWERABLE,
    GROUP_LOCAL_SINGLE_DOC,
    GROUP_LOCAL_MULTI_DOC,
    GROUP_ROUTER_MANUAL_DOC_HIT,
    GROUP_FALLBACK_SAFETY_EXCLUDED,
    GROUP_EMERGENCY,
    GROUP_EMERGENCY_SAFETY,
    GROUP_EMERGENCY_LOCAL_EVIDENCE,
]


# ── 集合常量 ──

_LOCAL_ANSWERABLE_EXPECTED_BEHAVIORS = {
    "answer_with_local_evidence",
    "answer_with_required_citation",
    "answer_with_local_evidence_and_safety_disclaimer",
}

_LOCAL_ANSWERABLE_EVIDENCE_SCOPES = {
    "single_doc",
    "multi_doc",
    "single_doc_plus_safety_policy",
    "multi_doc_plus_safety_policy",
    "safety_rule_plus_local_doc",
}

_SINGLE_DOC_EVIDENCE_SCOPES = {
    "single_doc",
    "single_doc_plus_safety_policy",
    "safety_rule_plus_local_doc",
}

_MULTI_DOC_EVIDENCE_SCOPES = {
    "multi_doc",
    "multi_doc_plus_safety_policy",
}

_FALLBACK_SAFETY_EVIDENCE_SCOPES = {
    "requires_current_web",
    "no_local_evidence",
    "harmful_request",
    "out_of_domain",
    "unsupported_absolute_claim",
}

_FALLBACK_SAFETY_EXPECTED_BEHAVIORS = {
    "use_web_or_state_limitation",
    "refuse_or_clarify",
    "refuse_or_state_insufficient_evidence",
    "safety_refusal_with_safe_alternative",
    "refuse_or_state_out_of_scope",
    "refuse_or_qualify_claim",
}

_EMERGENCY_EVIDENCE_SCOPES = {
    "single_doc_plus_safety_policy",
    "multi_doc_plus_safety_policy",
    "safety_rule_plus_local_doc",
}


# ── 核心函数 ──


def get_evaluation_groups_from_case(case: EvaluationCase) -> list[str]:
    """从 EvaluationCase 返回样本所属分组。"""
    return get_evaluation_groups(
        intent=case.intent,
        tags=list(case.tags),
        metadata=dict(case.metadata),
    )


def get_evaluation_groups(
    *,
    intent: str,
    tags: list[str],
    metadata: dict[str, Any],
) -> list[str]:
    """根据 intent、tags、metadata 返回样本所属分组列表。"""
    groups: list[str] = [GROUP_ALL]

    expected_behavior = str(metadata.get("expected_behavior", ""))
    expected_action = str(metadata.get("expected_action", ""))
    evidence_scope = str(metadata.get("evidence_scope", ""))
    question_type = str(metadata.get("question_type", ""))

    is_local_answerable = (
        expected_behavior in _LOCAL_ANSWERABLE_EXPECTED_BEHAVIORS
        and evidence_scope in _LOCAL_ANSWERABLE_EVIDENCE_SCOPES
    )

    is_single_doc = (
        is_local_answerable
        and evidence_scope in _SINGLE_DOC_EVIDENCE_SCOPES
    )

    is_multi_doc = (
        evidence_scope in _MULTI_DOC_EVIDENCE_SCOPES
        or question_type == "cross_doc_compare"
    )

    is_router_manual = evidence_scope == "router_manual_added"

    is_fallback_safety = (
        evidence_scope in _FALLBACK_SAFETY_EVIDENCE_SCOPES
        or expected_action == "refuse"
        or expected_behavior in _FALLBACK_SAFETY_EXPECTED_BEHAVIORS
    )

    is_emergency = (
        intent == "emergency"
        or question_type == "emergency"
        or evidence_scope in _EMERGENCY_EVIDENCE_SCOPES
    )

    # local_answerable
    if is_local_answerable:
        groups.append(GROUP_LOCAL_ANSWERABLE)

    # local_single_doc
    if is_single_doc:
        groups.append(GROUP_LOCAL_SINGLE_DOC)

    # local_multi_doc
    if is_multi_doc:
        groups.append(GROUP_LOCAL_MULTI_DOC)

    # router_manual_doc_hit
    if is_router_manual:
        groups.append(GROUP_ROUTER_MANUAL_DOC_HIT)

    # fallback_safety_excluded
    if is_fallback_safety:
        groups.append(GROUP_FALLBACK_SAFETY_EXCLUDED)

    # emergency 及其子分组
    if is_emergency:
        groups.append(GROUP_EMERGENCY)
        groups.append(GROUP_EMERGENCY_SAFETY)
        if is_local_answerable or evidence_scope in _LOCAL_ANSWERABLE_EVIDENCE_SCOPES:
            groups.append(GROUP_EMERGENCY_LOCAL_EVIDENCE)

    return groups


# ── 辅助函数 ──


def extract_case_metadata_for_detail(metadata: dict[str, Any]) -> dict[str, Any]:
    """从 case.metadata 中提取需要写入 detail 的关键字段。"""
    keys = [
        "expected_behavior",
        "expected_action",
        "evidence_scope",
        "question_type",
        "source_files",
        "source_documents",
        "router_expected_intent",
        "router_expected_sub_intent",
    ]
    return {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
