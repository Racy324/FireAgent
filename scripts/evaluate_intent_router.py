"""评估 FireAgent 意图路由器。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fireagent.graph.llm_router import LLMIntentRouter
from fireagent.utils.config import get_config


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


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """计算路由评估指标。"""
    total = len(rows)
    correct = 0
    fallback_count = 0
    low_confidence_count = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels = set()

    for row in rows:
        expected = str(row["expected_intent"])
        predicted = str(row["predicted_intent"])
        source = str(row.get("source", ""))
        confidence = float(row.get("confidence", 1.0))
        labels.add(expected)
        labels.add(predicted)
        confusion[expected][predicted] += 1
        if expected == predicted:
            correct += 1
        if source == "fallback_rule":
            fallback_count += 1
        if confidence < 0.75:
            low_confidence_count += 1

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
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "fallback_count": fallback_count,
        "low_confidence_count": low_confidence_count,
    }


def evaluate(dataset: Path) -> dict[str, Any]:
    """运行路由器并计算指标。"""
    config = get_config()
    router = LLMIntentRouter(config=config)
    evaluated: list[dict[str, Any]] = []
    for row in load_jsonl(dataset):
        query = str(row["query"])
        result = router.route(query)
        evaluated.append(
            {
                **row,
                "predicted_intent": result.intent,
                "predicted_sub_intent": result.sub_intent,
                "confidence": result.confidence,
                "source": result.source,
                "reason": result.reason,
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
    args = parser.parse_args()

    metrics = evaluate(Path(args.dataset))
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
