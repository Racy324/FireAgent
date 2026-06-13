"""评测集与预测结果的 JSONL 读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from fireagent.evaluation.schema import EvaluationCase, EvaluationPrediction, ManualScore


T = TypeVar("T", bound=BaseModel)


def load_evaluation_cases(path: str | Path, limit: int | None = None) -> list[EvaluationCase]:
    """从 JSONL 文件读取评测样本。"""
    return _load_jsonl(path, EvaluationCase, limit=limit)


def load_predictions(path: str | Path, limit: int | None = None) -> list[EvaluationPrediction]:
    """从 JSONL 文件读取预测结果。"""
    return _load_jsonl(path, EvaluationPrediction, limit=limit)


def write_predictions(path: str | Path, predictions: Iterable[EvaluationPrediction]) -> None:
    """把预测结果写为 JSONL。"""
    _write_jsonl(path, predictions)


def write_manual_scores(path: str | Path, scores: Iterable[ManualScore]) -> None:
    """把人工/规则评分明细写为 JSONL。"""
    _write_jsonl(path, scores)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    """写入格式化 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manual_review_csv(
    path: str | Path,
    cases: list[EvaluationCase],
    predictions: list[EvaluationPrediction],
    scores: list[ManualScore],
) -> None:
    """生成便于人工复核的 CSV 文件。"""
    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_by_id = {item.case_id: item for item in predictions}
    score_by_id = {item.case_id: item for item in scores}

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "question",
                "answer",
                "reference_answer",
                "contexts",
                "citations",
                "overall_score",
                "groundedness_proxy",
                "expected_action",
                "actual_action",
                "fallback_correct",
                "sufficiency_correct",
                "web_triggered",
                "false_web",
                "missed_web",
                "fallback_reason",
                "missing_keywords",
                "人工评分",
                "人工备注",
            ],
        )
        writer.writeheader()
        for case in cases:
            prediction = prediction_by_id.get(case.case_id)
            score = score_by_id.get(case.case_id)
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "answer": prediction.answer if prediction else "",
                    "reference_answer": case.reference_answer,
                    "contexts": "\n---\n".join(prediction.contexts if prediction else []),
                    "citations": "\n".join(prediction.citations if prediction else []),
                    "overall_score": score.overall_score if score else "",
                    "groundedness_proxy": score.groundedness_proxy if score else "",
                    "expected_action": score.expected_action if score else "",
                    "actual_action": score.actual_action if score else "",
                    "fallback_correct": score.fallback_correct if score else "",
                    "sufficiency_correct": score.sufficiency_correct if score else "",
                    "web_triggered": score.web_triggered if score else "",
                    "false_web": score.false_web if score else "",
                    "missed_web": score.missed_web if score else "",
                    "fallback_reason": prediction.fallback_reason if prediction else "",
                    "missing_keywords": "；".join(score.missing_keywords) if score else "",
                    "人工评分": "",
                    "人工备注": "",
                }
            )


def _load_jsonl(path: str | Path, model_type: type[T], limit: int | None = None) -> list[T]:
    """通用 JSONL 加载函数。"""
    input_path = Path(path)
    items: list[T] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                items.append(model_type.model_validate_json(stripped))
            except Exception as exc:  # noqa: BLE001 - 带上行号便于修数据集。
                raise ValueError(f"评测 JSONL 第 {line_number} 行格式错误：{input_path}") from exc
            if limit is not None and len(items) >= limit:
                break
    return items


def _write_jsonl(path: str | Path, items: Iterable[BaseModel]) -> None:
    """通用 JSONL 写入函数。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(item.model_dump_json(exclude_none=True) + "\n")
