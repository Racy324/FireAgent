"""RAG 评测流程测试。"""

from __future__ import annotations

import json

from fireagent.evaluation import (
    EvaluationCase,
    EvaluationPrediction,
    ManualRAGEvaluator,
    RAGEvaluationRunner,
    load_evaluation_cases,
)


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


def test_load_sample_eval_dataset() -> None:
    """示例评测集应保持 JSONL 可解析。"""
    cases = load_evaluation_cases("data/eval/fireagent_eval_sample.jsonl")

    assert len(cases) >= 3
    assert all(case.case_id and case.question for case in cases)
