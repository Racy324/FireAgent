"""FireAgent RAG 评测运行器。"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
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

try:  # tqdm 是可选依赖；缺失时退化为简洁文本进度。
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - 覆盖所有可选依赖导入异常。
    _tqdm = None


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
        show_progress: bool = True,
    ) -> EvaluationSummary:
        """运行一次完整评测。"""
        cases = load_evaluation_cases(dataset_path, limit=limit)
        run_dir = self._make_run_dir(output_dir)

        if predictions_path:
            predictions = load_predictions(predictions_path, limit=limit)
        else:
            predictions = self.generate_predictions(cases, show_progress=show_progress)

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

    def generate_predictions(
        self,
        cases: list[EvaluationCase],
        show_progress: bool = True,
    ) -> list[EvaluationPrediction]:
        """对评测集逐条调用 FireAgent，生成预测结果。"""
        predictions: list[EvaluationPrediction] = []
        # 复用同一个 workflow 实例，避免每条用例重复加载模型（reranker/embedding）
        workflow = None
        if self.answer_fn is None:
            workflow = build_fireagent_workflow(config=self.config, vectorstore=self.vectorstore)
        for case in _iter_cases_with_progress(cases, enabled=show_progress):
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
                sufficiency_result = state.get("sufficiency_result")
                fallback_decision = state.get("fallback_decision")
                fallback_action = ""
                fallback_reason = ""
                fallback_payload: dict[str, object] = {}
                if fallback_decision is not None:
                    action = getattr(fallback_decision, "action", "")
                    fallback_action = action.value if hasattr(action, "value") else str(action)
                    fallback_reason = str(getattr(fallback_decision, "reason", "") or "")
                    if hasattr(fallback_decision, "model_dump"):
                        fallback_payload = fallback_decision.model_dump(mode="json")
                sufficiency_payload = (
                    sufficiency_result.model_dump(mode="json")
                    if hasattr(sufficiency_result, "model_dump")
                    else {}
                )
                candidate_citations = list(state.get("candidate_citations", []) or [])
                used_citation_markers = list(state.get("used_citation_markers", []) or [])
                invalid_citation_markers = list(state.get("invalid_citation_markers", []) or [])
                route_decision = state.get("route_decision", {})
                route_payload = route_decision if isinstance(route_decision, dict) else {}
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
                        route_intent=str(route_payload.get("intent", "") or ""),
                        route_sub_intent=str(route_payload.get("sub_intent", "") or ""),
                        route_confidence=(
                            float(route_payload["confidence"])
                            if route_payload.get("confidence") is not None
                            else None
                        ),
                        route_source=str(route_payload.get("source", "") or ""),
                        route_reason=str(route_payload.get("reason", "") or ""),
                        metadata={
                            "hallucination_warnings": list(
                                state.get("hallucination_warnings", []) or []
                            ),
                            "sufficiency": sufficiency_payload,
                            "fallback": fallback_payload,
                            "route_decision": route_payload,
                            "web_triggered": fallback_action == "use_web",
                        },
                        fallback_action=fallback_action,
                        fallback_reason=fallback_reason,
                        used_citation_markers=used_citation_markers,
                        invalid_citation_markers=invalid_citation_markers,
                        candidate_citation_count=len(candidate_citations),
                        used_citation_count=len(used_citation_markers),
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


def _iter_cases_with_progress(
    cases: list[EvaluationCase],
    enabled: bool = True,
) -> Iterator[EvaluationCase]:
    """按用例迭代，并在终端展示预测生成进度。"""
    total = len(cases)
    if not enabled or total == 0:
        yield from cases
        return

    if _tqdm is not None:
        yield from _tqdm(
            cases,
            total=total,
            desc="生成预测",
            unit="题",
            dynamic_ncols=True,
        )
        return

    for index, case in enumerate(cases, start=1):
        print(f"生成预测 {index}/{total}: {case.case_id}", file=sys.stderr, flush=True)
        yield case
