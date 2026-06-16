"""意图路由评估脚本测试。"""

from __future__ import annotations

from scripts.evaluate_intent_router import compute_metrics, convert_rag_eval_to_router_format


def test_compute_router_metrics() -> None:
    """评估脚本应输出准确率、混淆矩阵和 fallback 数量。"""
    rows = [
        {"expected_intent": "chat", "predicted_intent": "chat", "source": "hard_rule"},
        {"expected_intent": "rag", "predicted_intent": "rag", "source": "llm"},
        {"expected_intent": "paper", "predicted_intent": "rag", "source": "fallback_rule"},
    ]

    metrics = compute_metrics(rows)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["fallback_count"] == 1
    assert metrics["confusion_matrix"]["paper"]["rag"] == 1


def test_compute_router_metrics_includes_latency_sub_intent_and_source_groups() -> None:
    """评估脚本应统计路由延迟、sub_intent 准确率和来源分组。"""
    rows = [
        {
            "expected_intent": "rag",
            "expected_sub_intent": "knowledge_qa",
            "predicted_intent": "rag",
            "predicted_sub_intent": "knowledge_qa",
            "source": "seed60_converted",
            "source_intent": "web_fallback",
            "route_latency_seconds": 0.1,
        },
        {
            "expected_intent": "chat",
            "expected_sub_intent": "greeting",
            "predicted_intent": "rag",
            "predicted_sub_intent": "knowledge_qa",
            "source": "manual_added",
            "route_latency_seconds": 0.3,
        },
    ]

    metrics = compute_metrics(rows)

    assert metrics["latency_seconds"]["mean"] == 0.2
    assert metrics["latency_seconds"]["p50"] == 0.1
    assert metrics["latency_seconds"]["p95"] == 0.3
    assert metrics["sub_intent_accuracy"] == 0.5
    assert metrics["source_group_metrics"]["seed60_converted"]["accuracy"] == 1.0
    assert metrics["source_intent_metrics"]["web_fallback"]["accuracy"] == 1.0


def test_compute_router_metrics_counts_boolean_mismatches() -> None:
    """如果样本带期望布尔字段，应统计路由布尔决策差异。"""
    rows = [
        {
            "expected_intent": "rag",
            "predicted_intent": "rag",
            "expected_need_rag": True,
            "need_rag": False,
            "expected_need_safety_notice": False,
            "need_safety_notice": False,
        }
    ]

    metrics = compute_metrics(rows)

    assert metrics["boolean_field_mismatches"]["need_rag"] == {
        "total": 1,
        "mismatches": 1,
    }
    assert metrics["boolean_field_mismatches"]["need_safety_notice"] == {
        "total": 1,
        "mismatches": 0,
    }


def test_convert_rag_eval_rag_intent() -> None:
    """rag intent 应映射为 rag。"""
    rows = [{"case_id": "c1", "question": "什么是火灾?", "intent": "rag", "tags": []}]
    converted = convert_rag_eval_to_router_format(rows)
    assert converted[0]["query"] == "什么是火灾?"
    assert converted[0]["expected_intent"] == "rag"
    assert converted[0]["source_case_id"] == "c1"


def test_convert_rag_eval_emergency_intent() -> None:
    """emergency intent 应映射为 emergency。"""
    rows = [{"case_id": "c2", "question": "火灾怎么办?", "intent": "emergency", "tags": []}]
    converted = convert_rag_eval_to_router_format(rows)
    assert converted[0]["expected_intent"] == "emergency"


def test_convert_rag_eval_web_fallback_maps_to_rag() -> None:
    """web_fallback intent 应映射为 rag。"""
    rows = [{"case_id": "c3", "question": "最新政策?", "intent": "web_fallback", "tags": []}]
    converted = convert_rag_eval_to_router_format(rows)
    assert converted[0]["expected_intent"] == "rag"


def test_convert_rag_eval_paper_intent() -> None:
    """paper intent 应映射为 paper。"""
    rows = [{"case_id": "c4", "question": "总结这篇论文", "intent": "paper", "tags": ["fire"]}]
    converted = convert_rag_eval_to_router_format(rows)
    assert converted[0]["expected_intent"] == "paper"
    assert converted[0]["source_format"] == "rag_eval"
