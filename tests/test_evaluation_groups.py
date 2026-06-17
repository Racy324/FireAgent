"""评估样本分组规则测试。"""

from __future__ import annotations

from fireagent.evaluation.evaluation_groups import (
    GROUP_ALL,
    GROUP_EMERGENCY,
    GROUP_EMERGENCY_LOCAL_EVIDENCE,
    GROUP_EMERGENCY_SAFETY,
    GROUP_FALLBACK_SAFETY_EXCLUDED,
    GROUP_LOCAL_ANSWERABLE,
    GROUP_LOCAL_MULTI_DOC,
    GROUP_LOCAL_SINGLE_DOC,
    GROUP_ROUTER_MANUAL_DOC_HIT,
    extract_case_metadata_for_detail,
    get_evaluation_groups,
    get_evaluation_groups_from_case,
)
from fireagent.evaluation.schema import EvaluationCase


def test_single_doc_answer_with_local_evidence() -> None:
    """single_doc + answer_with_local_evidence 应属于 all/local_answerable/local_single_doc。"""
    groups = get_evaluation_groups(
        intent="rag",
        tags=[],
        metadata={
            "expected_behavior": "answer_with_local_evidence",
            "evidence_scope": "single_doc",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_LOCAL_ANSWERABLE in groups
    assert GROUP_LOCAL_SINGLE_DOC in groups
    assert GROUP_LOCAL_MULTI_DOC not in groups
    assert GROUP_FALLBACK_SAFETY_EXCLUDED not in groups


def test_multi_doc_cross_doc_compare() -> None:
    """multi_doc + cross_doc_compare 应属于 all/local_answerable/local_multi_doc。"""
    groups = get_evaluation_groups(
        intent="rag",
        tags=[],
        metadata={
            "expected_behavior": "answer_with_local_evidence",
            "evidence_scope": "multi_doc",
            "question_type": "cross_doc_compare",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_LOCAL_ANSWERABLE in groups
    assert GROUP_LOCAL_MULTI_DOC in groups


def test_multi_doc_via_question_type() -> None:
    """question_type=cross_doc_compare 即使 evidence_scope 不在集合中也应归入 local_multi_doc。"""
    groups = get_evaluation_groups(
        intent="rag",
        tags=[],
        metadata={
            "expected_behavior": "answer_with_local_evidence",
            "evidence_scope": "single_doc",
            "question_type": "cross_doc_compare",
        },
    )
    assert GROUP_LOCAL_MULTI_DOC in groups


def test_router_manual_added() -> None:
    """router_manual_added 应属于 all/router_manual_doc_hit，不属于 local_answerable。"""
    groups = get_evaluation_groups(
        intent="chat",
        tags=[],
        metadata={
            "evidence_scope": "router_manual_added",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_ROUTER_MANUAL_DOC_HIT in groups
    assert GROUP_LOCAL_ANSWERABLE not in groups


def test_requires_current_web_fallback_excluded() -> None:
    """requires_current_web 应属于 all/fallback_safety_excluded。"""
    groups = get_evaluation_groups(
        intent="rag",
        tags=[],
        metadata={
            "evidence_scope": "requires_current_web",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_FALLBACK_SAFETY_EXCLUDED in groups
    assert GROUP_LOCAL_ANSWERABLE not in groups


def test_harmful_request_refuse() -> None:
    """harmful_request + expected_action=refuse 应属于 all/fallback_safety_excluded。"""
    groups = get_evaluation_groups(
        intent="reject",
        tags=[],
        metadata={
            "evidence_scope": "harmful_request",
            "expected_action": "refuse",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_FALLBACK_SAFETY_EXCLUDED in groups


def test_refuse_behavior_fallback_excluded() -> None:
    """refuse_or_clarify 行为应归入 fallback_safety_excluded。"""
    groups = get_evaluation_groups(
        intent="reject",
        tags=[],
        metadata={
            "evidence_scope": "out_of_domain",
            "expected_behavior": "refuse_or_clarify",
        },
    )
    assert GROUP_FALLBACK_SAFETY_EXCLUDED in groups


def test_intent_emergency() -> None:
    """intent=emergency 应属于 all/emergency/emergency_safety。"""
    groups = get_evaluation_groups(
        intent="emergency",
        tags=[],
        metadata={
            "evidence_scope": "single_doc",
            "expected_behavior": "answer_with_local_evidence",
        },
    )
    assert GROUP_ALL in groups
    assert GROUP_EMERGENCY in groups
    assert GROUP_EMERGENCY_SAFETY in groups


def test_emergency_with_local_evidence() -> None:
    """emergency + single_doc_plus_safety_policy 应同时属于 emergency_local_evidence。"""
    groups = get_evaluation_groups(
        intent="emergency",
        tags=[],
        metadata={
            "evidence_scope": "single_doc_plus_safety_policy",
            "expected_behavior": "answer_with_local_evidence_and_safety_disclaimer",
        },
    )
    assert GROUP_EMERGENCY in groups
    assert GROUP_EMERGENCY_SAFETY in groups
    assert GROUP_EMERGENCY_LOCAL_EVIDENCE in groups
    assert GROUP_LOCAL_ANSWERABLE in groups


def test_question_type_emergency() -> None:
    """question_type=emergency 也应归入 emergency 分组。"""
    groups = get_evaluation_groups(
        intent="rag",
        tags=[],
        metadata={
            "evidence_scope": "single_doc",
            "question_type": "emergency",
        },
    )
    assert GROUP_EMERGENCY in groups


def test_get_evaluation_groups_from_case() -> None:
    """从 EvaluationCase 对象提取分组。"""
    case = EvaluationCase(
        case_id="test-001",
        question="隧道火灾排烟方法",
        intent="rag",
        tags=[],
        metadata={
            "expected_behavior": "answer_with_local_evidence",
            "evidence_scope": "single_doc",
        },
    )
    groups = get_evaluation_groups_from_case(case)
    assert GROUP_ALL in groups
    assert GROUP_LOCAL_ANSWERABLE in groups
    assert GROUP_LOCAL_SINGLE_DOC in groups


def test_extract_case_metadata_for_detail() -> None:
    """提取 detail 需要的 metadata 字段。"""
    metadata = {
        "expected_behavior": "answer_with_local_evidence",
        "evidence_scope": "single_doc",
        "source_files": ["data/raw_pdfs/test.pdf"],
        "irrelevant_field": "should_not_appear",
    }
    extracted = extract_case_metadata_for_detail(metadata)
    assert extracted["expected_behavior"] == "answer_with_local_evidence"
    assert extracted["evidence_scope"] == "single_doc"
    assert extracted["source_files"] == ["data/raw_pdfs/test.pdf"]
    assert "irrelevant_field" not in extracted


def test_empty_metadata_returns_all_only() -> None:
    """空 metadata 只返回 all。"""
    groups = get_evaluation_groups(intent="rag", tags=[], metadata={})
    assert groups == [GROUP_ALL]


def test_multi_tag_sample() -> None:
    """一条样本可以同时属于 local_answerable 和 emergency。"""
    groups = get_evaluation_groups(
        intent="emergency",
        tags=[],
        metadata={
            "expected_behavior": "answer_with_local_evidence_and_safety_disclaimer",
            "evidence_scope": "single_doc_plus_safety_policy",
        },
    )
    assert GROUP_LOCAL_ANSWERABLE in groups
    assert GROUP_LOCAL_SINGLE_DOC in groups
    assert GROUP_EMERGENCY in groups
    assert GROUP_EMERGENCY_LOCAL_EVIDENCE in groups
