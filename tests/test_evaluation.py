"""RAG 评测流程测试。"""

from __future__ import annotations

import json

import fireagent.evaluation.runner as runner_module
from fireagent.evaluation import (
    EvaluationCase,
    EvaluationPrediction,
    ManualRAGEvaluator,
    RAGEvaluationRunner,
    load_evaluation_cases,
)
from fireagent.retrieval import FallbackAction, FallbackDecision, SufficiencyResult


def test_manual_evaluator_scores_keyword_and_citation() -> None:
    """人工辅助评测应统计关键词、引用和 groundedness。"""
    case = EvaluationCase(
        case_id="case-1",
        question="隧道火灾烟气影响？",
        reference_answer="烟气会降低能见度并影响人员疏散。",
        expected_keywords=["烟气", "能见度", "疏散"],
        required_citation_substrings=["隧道"],
    )
    prediction = EvaluationPrediction(
        case_id="case-1",
        question=case.question,
        answer="隧道火灾烟气会降低能见度，并影响人员疏散。",
        contexts=["隧道火灾烟气会降低能见度，影响人员疏散安全。"],
        citations=["本地论文证据：隧道火灾研究"],
    )

    score = ManualRAGEvaluator().score_case(case, prediction)

    assert score.answer_keyword_recall == 1.0
    assert score.context_keyword_recall == 1.0
    assert score.citation_score == 1.0
    assert score.overall_score > 0.8


def test_evaluation_runner_writes_reports(tmp_path) -> None:
    """评测运行器应生成 predictions、summary 和人工复核 CSV。"""
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "question": "火灾烟气有什么影响？",
                "reference_answer": "烟气会降低能见度并影响疏散。",
                "expected_keywords": ["烟气", "能见度", "疏散"],
                "required_citation_substrings": ["烟气"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_answer(case: EvaluationCase) -> EvaluationPrediction:
        """构造测试用预测结果。"""
        return EvaluationPrediction(
            case_id=case.case_id,
            question=case.question,
            answer="火灾烟气会降低能见度，并影响疏散。",
            contexts=["烟气降低能见度，影响疏散。"],
            citations=["本地论文证据：烟气研究"],
        )

    runner = RAGEvaluationRunner(answer_fn=fake_answer)
    summary = runner.run(dataset_path=dataset_path, output_dir=tmp_path / "runs", fail_under=0.5)

    assert summary.passed is True
    assert summary.total_cases == 1
    assert summary.average_scores["overall_score"] > 0.5
    assert (tmp_path / "runs").exists()
    assert summary.prediction_path.endswith("predictions.jsonl")
    assert summary.manual_report_path.endswith("manual_review.csv")


def test_evaluation_runner_uses_tqdm_progress_when_available(monkeypatch) -> None:
    """生成预测时应优先使用 tqdm 展示进度条。"""
    calls: list[dict[str, object]] = []

    def fake_tqdm(items, **kwargs):
        calls.append(kwargs)
        yield from items

    monkeypatch.setattr(runner_module, "_tqdm", fake_tqdm)
    cases = [
        EvaluationCase(case_id="case-1", question="问题 1"),
        EvaluationCase(case_id="case-2", question="问题 2"),
    ]

    def fake_answer(case: EvaluationCase) -> EvaluationPrediction:
        return EvaluationPrediction(
            case_id=case.case_id,
            question=case.question,
            answer="测试答案",
        )

    predictions = RAGEvaluationRunner(answer_fn=fake_answer).generate_predictions(cases)

    assert len(predictions) == 2
    assert calls == [
        {
            "total": 2,
            "desc": "生成预测",
            "unit": "题",
            "dynamic_ncols": True,
        }
    ]


def test_evaluation_runner_prints_plain_progress_without_tqdm(monkeypatch, capsys) -> None:
    """缺少 tqdm 时应退化为 stderr 文本进度，避免评估静默卡住。"""
    monkeypatch.setattr(runner_module, "_tqdm", None)
    cases = [
        EvaluationCase(case_id="case-1", question="问题 1"),
        EvaluationCase(case_id="case-2", question="问题 2"),
    ]

    def fake_answer(case: EvaluationCase) -> EvaluationPrediction:
        return EvaluationPrediction(
            case_id=case.case_id,
            question=case.question,
            answer="测试答案",
        )

    predictions = RAGEvaluationRunner(answer_fn=fake_answer).generate_predictions(cases)

    assert len(predictions) == 2
    captured = capsys.readouterr()
    assert "生成预测 1/2: case-1" in captured.err
    assert "生成预测 2/2: case-2" in captured.err


def test_evaluation_runner_records_sufficiency_and_fallback_metadata(monkeypatch) -> None:
    """生成预测时应把 sufficiency 与 fallback 决策写入 prediction。"""

    class FakeWorkflow:
        def invoke(self, _state):
            return {
                "final_answer": "测试答案",
                "final_context": "测试上下文",
                "citations": ["证据 1"],
                "intent": "rag",
                "evidence_sufficient": False,
                "errors": [],
                "hallucination_warnings": [],
                "sufficiency_result": SufficiencyResult(
                    sufficient=False,
                    needs_web=True,
                    reason="top_score_below_threshold",
                    top_score=0.1,
                    evidence_count=1,
                    metadata={"term_coverage": 0.2},
                ),
                "fallback_decision": FallbackDecision(
                    action=FallbackAction.USE_WEB,
                    reason="local_evidence_insufficient_and_web_allowed",
                ),
            }

    monkeypatch.setattr(runner_module, "build_fireagent_workflow", lambda **_kwargs: FakeWorkflow())

    case = EvaluationCase(case_id="case-1", question="最新消防标准是什么？")
    prediction = RAGEvaluationRunner().generate_predictions([case], show_progress=False)[0]

    assert prediction.fallback_action == "use_web"
    assert prediction.fallback_reason == "local_evidence_insufficient_and_web_allowed"
    assert prediction.metadata["web_triggered"] is True
    assert prediction.metadata["sufficiency"]["reason"] == "top_score_below_threshold"


def test_manual_evaluator_scores_fallback_decision() -> None:
    """人工评估应标记 fallback 动作是否符合期望。"""
    case = EvaluationCase(
        case_id="case-web",
        question="最新消防政策是什么？",
        metadata={"question_type": "web_fallback"},
    )
    prediction = EvaluationPrediction(
        case_id=case.case_id,
        question=case.question,
        answer="测试答案",
        fallback_action="answer_local",
        evidence_sufficient=True,
    )

    score = ManualRAGEvaluator().score_case(case, prediction)

    assert score.expected_action == "use_web"
    assert score.actual_action == "answer_local"
    assert score.fallback_correct is False
    assert score.missed_web is True


def test_load_sample_eval_dataset() -> None:
    """示例评测集应保持 JSONL 可解析。"""
    cases = load_evaluation_cases("data/eval/fireagent_eval_sample.jsonl")

    assert len(cases) >= 3
    assert all(case.case_id and case.question for case in cases)
