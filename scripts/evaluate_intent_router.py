"""评估 FireAgent 意图路由器。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fireagent.graph.llm_router import LLMIntentRouter
from fireagent.utils.config import get_config
from scripts.convert_rag_eval_to_router_eval import convert_rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON") from exc
            rows.append(row)
    return rows


KNOWN_INTENTS = {"chat", "reject", "emergency", "paper", "rag"}


def convert_rag_eval_to_router_format(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 RAG 评估集（question/intent）转换为路由评估格式（query/expected_intent）。"""
    converted = convert_rows(rows)
    for row in converted:
        row["source_format"] = "rag_eval"
    return converted


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    """计算延迟均值和分位数。"""
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def percentile(ratio: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * ratio))
        return round(ordered[index], 4)

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
    }


def _accuracy_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算一组样本的基础命中率。"""
    total = len(rows)
    correct = sum(
        1
        for row in rows
        if str(row.get("expected_intent", "")) == str(row.get("predicted_intent", ""))
    )
    return {"total": total, "accuracy": correct / total if total else 0.0}


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算路由评估指标。"""
    total = len(rows)
    correct = 0
    fallback_count = 0
    low_confidence_count = 0
    sub_intent_total = 0
    sub_intent_correct = 0
    latencies: list[float] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels = set()
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_source_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boolean_field_mismatches: dict[str, dict[str, int]] = {}

    for row in rows:
        expected = str(row["expected_intent"])
        predicted = str(row["predicted_intent"])
        route_source = str(row.get("route_source", row.get("source", "")))
        confidence = float(row.get("confidence", 1.0))
        labels.add(expected)
        labels.add(predicted)
        confusion[expected][predicted] += 1
        if expected == predicted:
            correct += 1
        if route_source == "fallback_rule":
            fallback_count += 1
        if confidence < 0.75:
            low_confidence_count += 1
        if "route_latency_seconds" in row:
            latencies.append(float(row["route_latency_seconds"]))
        expected_sub_intent = str(row.get("expected_sub_intent", "") or "")
        if expected_sub_intent:
            sub_intent_total += 1
            if expected_sub_intent == str(row.get("predicted_sub_intent", "") or ""):
                sub_intent_correct += 1
        dataset_source = str(row.get("source", "") or "")
        if dataset_source:
            rows_by_source[dataset_source].append(row)
        source_intent = str(row.get("source_intent", "") or "")
        if source_intent:
            rows_by_source_intent[source_intent].append(row)
        for field in ("need_rag", "need_web_search", "need_safety_notice"):
            expected_key = f"expected_{field}"
            if expected_key not in row:
                continue
            stats = boolean_field_mismatches.setdefault(field, {"total": 0, "mismatches": 0})
            stats["total"] += 1
            if bool(row.get(expected_key)) != bool(row.get(field)):
                stats["mismatches"] += 1

    per_intent: dict[str, dict[str, float]] = {}
    for label in sorted(labels):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(count for pred, count in confusion[label].items() if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = (
        sum(item["f1"] for item in per_intent.values()) / len(per_intent)
        if per_intent
        else 0.0
    )
    metrics: dict[str, Any] = {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "fallback_count": fallback_count,
        "low_confidence_count": low_confidence_count,
        "latency_seconds": _latency_summary(latencies),
        "source_group_metrics": {
            key: _accuracy_for_rows(value) for key, value in sorted(rows_by_source.items())
        },
        "source_intent_metrics": {
            key: _accuracy_for_rows(value) for key, value in sorted(rows_by_source_intent.items())
        },
        "boolean_field_mismatches": boolean_field_mismatches,
    }
    if sub_intent_total:
        metrics["sub_intent_accuracy"] = sub_intent_correct / sub_intent_total
    return metrics


def validate_expected_intents(rows: list[dict[str, Any]]) -> None:
    """校验 expected_intent 是否属于当前 router 支持的类别。"""
    unknown = sorted(
        {
            str(row.get("expected_intent", ""))
            for row in rows
            if str(row.get("expected_intent", "")) not in KNOWN_INTENTS
        }
    )
    if unknown:
        raise ValueError(f"未知 expected_intent：{', '.join(unknown)}")


def evaluate(
    dataset: Path,
    dataset_format: str = "router",
    fail_on_unknown_intent: bool = False,
) -> dict[str, Any]:
    """运行路由器并计算指标。"""
    config = get_config()
    router = LLMIntentRouter(config=config)
    raw_rows = load_jsonl(dataset)
    if dataset_format == "rag_eval":
        raw_rows = convert_rag_eval_to_router_format(raw_rows)
    if fail_on_unknown_intent:
        validate_expected_intents(raw_rows)
    evaluated: list[dict[str, Any]] = []
    for row in raw_rows:
        query = str(row["query"])
        start = time.perf_counter()
        result = router.route(query)
        route_latency = round(time.perf_counter() - start, 4)
        evaluated.append(
            {
                **row,
                "predicted_intent": result.intent,
                "predicted_sub_intent": result.sub_intent,
                "confidence": result.confidence,
                "route_source": result.source,
                "reason": result.reason,
                "need_rag": result.need_rag,
                "need_memory": result.need_memory,
                "need_safety_notice": result.need_safety_notice,
                "route_latency_seconds": route_latency,
            }
        )
    metrics = compute_metrics(evaluated)
    metrics["items"] = evaluated
    return metrics


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="Evaluate FireAgent intent router.")
    parser.add_argument(
        "--dataset",
        default="data/eval/router/fireagent_router_eval_v1.jsonl",
        help="Router eval JSONL dataset path.",
    )
    parser.add_argument("--output", default="", help="Optional metrics JSON output path.")
    parser.add_argument(
        "--dataset-format",
        choices=["router", "rag_eval"],
        default="router",
        help="Dataset format: 'router' uses query/expected_intent; 'rag_eval' uses question/intent from seed60.",
    )
    parser.add_argument(
        "--fail-on-unknown-intent",
        action="store_true",
        help="Fail if expected_intent contains a label outside router supported intents.",
    )
    args = parser.parse_args()

    metrics = evaluate(
        Path(args.dataset),
        dataset_format=args.dataset_format,
        fail_on_unknown_intent=args.fail_on_unknown_intent,
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
