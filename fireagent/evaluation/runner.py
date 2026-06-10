"""FireAgent RAG 评测运行器。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fireagent.evaluation.dataset import (
    load_evaluation_cases,
    load_predictions,
    write_json,
    write_manual_review_csv,
    write_manual_scores,
    write_predictions,
)
from fireagent.evaluation.manual import ManualRAGEvaluator
from fireagent.evaluation.ragas_adapter import RagasEvaluationError, evaluate_with_ragas
from fireagent.evaluation.schema import EvaluationCase, EvaluationPrediction, EvaluationSummary
from fireagent.graph.workflow import build_fireagent_workflow, run_fireagent_workflow
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore import FireAgentQdrantClient


AnswerFn = Callable[[EvaluationCase], EvaluationPrediction]


class RAGEvaluationRunner:
    """负责生成预测、执行评测并落盘报告。"""

    def __init__(
        self,
        config: FireAgentConfig | None = None,
        vectorstore: FireAgentQdrantClient | None = None,
        answer_fn: AnswerFn | None = None,
    ) -> None:
        self.config = config or get_config()
        self.vectorstore = vectorstore
        self.answer_fn = answer_fn
        self.manual_evaluator = ManualRAGEvaluator()

    def run(
        self,
        dataset_path: str | Path,
        output_dir: str | Path = "data/eval/runs",
        mode: str = "manual",
        predictions_path: str | Path | None = None,
        limit: int | None = None,
        fail_under: float | None = None,
        ragas_metrics: list[str] | None = None,
    ) -> EvaluationSummary:
        """运行一次完整评测。"""
        cases = load_evaluation_cases(dataset_path, limit=limit)
        run_dir = self._make_run_dir(output_dir)

        if predictions_path:
            predictions = load_predictions(predictions_path, limit=limit)
        else:
            predictions = self.generate_predictions(cases)

        prediction_output = run_dir / "predictions.jsonl"
        write_predictions(prediction_output, predictions)

        scores = self.manual_evaluator.score_all(cases, predictions)
        manual_scores_path = run_dir / "manual_scores.jsonl"
        manual_review_path = run_dir / "manual_review.csv"
        write_manual_scores(manual_scores_path, scores)
        write_manual_review_csv(manual_review_path, cases, predictions, scores)

        average_scores = self.manual_evaluator.average_scores(scores)
        errors: list[str] = []
        ragas_report_path = ""

        selected_mode = mode.lower()
        if selected_mode in {"ragas", "both"}:
            try:
                ragas_payload = evaluate_with_ragas(cases, predictions, metric_names=ragas_metrics)
                ragas_path = run_dir / "ragas_report.json"
                write_json(ragas_path, ragas_payload)
                ragas_report_path = str(ragas_path)
            except RagasEvaluationError as exc:
                if selected_mode == "ragas":
                    raise
                errors.append(str(exc))

        overall = average_scores.get("overall_score", 0.0)
        passed = fail_under is None or overall >= fail_under
        summary = EvaluationSummary(
            total_cases=len(cases),
            average_scores=average_scores,
            passed=passed,
            fail_under=fail_under,
            run_dir=str(run_dir),
            prediction_path=str(prediction_output),
            manual_report_path=str(manual_review_path),
            ragas_report_path=ragas_report_path,
            errors=errors,
        )
        write_json(run_dir / "summary.json", summary.model_dump())
        return summary

    def generate_predictions(self, cases: list[EvaluationCase]) -> list[EvaluationPrediction]:
        """对评测集逐条调用 FireAgent，生成预测结果。"""
        predictions: list[EvaluationPrediction] = []
        # 复用同一个 workflow 实例，避免每条用例重复加载模型（reranker/embedding）
        workflow = None
        if self.answer_fn is None:
            workflow = build_fireagent_workflow(config=self.config, vectorstore=self.vectorstore)
        for case in cases:
            if self.answer_fn is not None:
                predictions.append(self.answer_fn(case))
                continue
            start = time.perf_counter()
            try:
                from fireagent.graph.state import create_initial_state
                result = workflow.invoke(create_initial_state(case.question))
                state = dict(result)
                latency = time.perf_counter() - start
                context = str(state.get("final_context", "") or "")
                predictions.append(
                    EvaluationPrediction(
                        case_id=case.case_id,
                        question=case.question,
                        answer=str(state.get("final_answer", "") or ""),
                        contexts=[context] if context else [],
                        citations=list(state.get("citations", []) or []),
                        intent=str(state.get("intent", "") or ""),
                        evidence_sufficient=bool(state.get("evidence_sufficient", False)),
                        errors=list(state.get("errors", []) or []),
                        latency_seconds=round(latency, 4),
                        metadata={
                            "hallucination_warnings": list(
                                state.get("hallucination_warnings", []) or []
                            )
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 单条评测失败也要保留记录。
                predictions.append(
                    EvaluationPrediction(
                        case_id=case.case_id,
                        question=case.question,
                        answer="",
                        errors=[str(exc)],
                        latency_seconds=round(time.perf_counter() - start, 4),
                    )
                )
        return predictions

    @staticmethod
    def _make_run_dir(output_dir: str | Path) -> Path:
        """生成本次评测的输出目录。"""
        base_dir = Path(output_dir)
        run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
