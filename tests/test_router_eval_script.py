"""意图路由评估脚本测试。"""

from __future__ import annotations

from scripts.evaluate_intent_router import compute_metrics


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
