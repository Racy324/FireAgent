"""Retrieval-only evaluation metric tests."""

from __future__ import annotations

from pathlib import Path

from fireagent.evaluation.retrieval_metrics import (
    RetrievedChunk,
    RetrievalGold,
    score_retrieved_chunks,
)
from scripts.evaluate_retrieval import build_failure_report, build_summary, parse_ks, write_failure_csv


def test_score_retrieved_chunks_uses_citation_and_reference_context_gold() -> None:
    """Retrieval metrics should score ranked chunks against citation and context gold signals."""
    gold = RetrievalGold(
        required_citation_substrings=["隧道火灾", "排烟路径"],
        reference_contexts=[
            "高海拔特长公路隧道研究关注平行导洞排烟路径和人员疏散协同。",
            "分岔隧道研究关注 CO 输运。",
        ],
    )
    chunks = [
        RetrievedChunk(
            rank=1,
            chunk_id="c1",
            text="高海拔特长公路隧道研究关注平行导洞排烟路径和人员疏散协同。",
            score=0.9,
        ),
        RetrievedChunk(
            rank=2,
            chunk_id="c2",
            text="这个片段只讨论普通建筑防火设计。",
            score=0.8,
        ),
        RetrievedChunk(
            rank=3,
            chunk_id="c3",
            text="分岔隧道火灾研究关注 CO 输运和有毒烟气扩散。",
            score=0.7,
        ),
    ]

    score = score_retrieved_chunks(gold=gold, chunks=chunks, ks=(1, 2, 3))

    assert score.gold_signal_count == 4
    assert score.hit_at_k["hit@1"] == 1.0
    assert score.hit_at_k["hit@2"] == 1.0
    assert score.recall_at_k["recall@1"] == 0.5
    assert score.recall_at_k["recall@2"] == 0.5
    assert score.recall_at_k["recall@3"] == 1.0
    assert score.precision_at_k["precision@1"] == 1.0
    assert score.precision_at_k["precision@2"] == 0.5
    assert score.mrr_at_k["mrr@3"] == 1.0
    assert score.required_citation_hit_rate == 1.0
    assert score.reference_context_coverage == 1.0
    assert chunks[0].matched_required_citation_substrings == ["排烟路径"]
    assert chunks[0].matched_reference_context_indices == [0]
    assert chunks[2].matched_required_citation_substrings == ["隧道火灾"]
    assert chunks[2].matched_reference_context_indices == [1]


def test_score_retrieved_chunks_returns_empty_metrics_without_gold() -> None:
    """Cases without gold signals should not pretend retrieval quality is known."""
    score = score_retrieved_chunks(
        gold=RetrievalGold(required_citation_substrings=[], reference_contexts=[]),
        chunks=[RetrievedChunk(rank=1, chunk_id="c1", text="任意文本", score=0.1)],
        ks=(5, 10),
    )

    assert score.gold_signal_count == 0
    assert score.hit_at_k["hit@5"] is None
    assert score.recall_at_k["recall@10"] is None
    assert score.precision_at_k["precision@5"] is None
    assert score.mrr_at_k["mrr@10"] is None
    assert score.required_citation_hit_rate is None
    assert score.reference_context_coverage is None


def test_parse_ks_sorts_and_deduplicates_values() -> None:
    """CLI top-k parsing should be stable and deterministic."""
    assert parse_ks("10,5,10") == (5, 10)


def _make_detail(
    case_id: str,
    evaluation_groups: list[str],
    recall_10: float | None = 0.5,
    gold_signal_count: int = 2,
) -> dict:
    """构造测试用 detail record。"""
    return {
        "case_id": case_id,
        "question": "test question",
        "intent": "rag",
        "tags": [],
        "metadata": {},
        "evaluation_groups": evaluation_groups,
        "gold": {
            "required_citation_substrings": [],
            "reference_contexts": [],
            "gold_signal_count": gold_signal_count,
        },
        "stage_scores": {
            "dense": {"hit_at_k": {}, "recall_at_k": {}, "precision_at_k": {}, "mrr_at_k": {}},
            "sparse": {"hit_at_k": {}, "recall_at_k": {}, "precision_at_k": {}, "mrr_at_k": {}},
            "fused": {"hit_at_k": {}, "recall_at_k": {}, "precision_at_k": {}, "mrr_at_k": {}},
            "reranked": {
                "gold_signal_count": gold_signal_count,
                "hit_at_k": {"hit@10": 1.0 if recall_10 and recall_10 > 0 else 0.0},
                "recall_at_k": {"recall@10": recall_10},
                "precision_at_k": {"precision@10": 0.5},
                "mrr_at_k": {"mrr@10": 0.5},
                "required_citation_hit_rate": None,
                "reference_context_coverage": None,
            },
        },
        "retrieved": {"reranked": []},
        "latency_seconds": 0.1,
        "errors": [],
    }


def test_build_summary_has_group_summaries() -> None:
    """summary 应包含 group_summaries。"""
    from unittest.mock import MagicMock

    details = [
        _make_detail("c1", ["all", "local_answerable", "local_single_doc"], recall_10=0.8),
        _make_detail("c2", ["all", "local_answerable", "local_multi_doc"], recall_10=0.3),
        _make_detail("c3", ["all", "fallback_safety_excluded"], recall_10=None, gold_signal_count=0),
    ]
    cfg = MagicMock()
    cfg.qdrant.collection = "test"
    cfg.retrieval.max_rewrite_queries = 4
    cfg.retrieval.parallel_rewrite_queries = False
    cfg.retrieval.rewrite_query_max_workers = 4
    cfg.retrieval.dense_top_k = 30
    cfg.retrieval.sparse_top_k = 30
    cfg.retrieval.fusion_top_k = 50
    cfg.retrieval.rerank_top_k = 10
    cfg.retrieval.dense_weight = 0.65
    cfg.retrieval.sparse_weight = 0.35
    cfg.retrieval.rrf_k = 60
    cfg.retrieval.chunk_type_filter_mode = "off"
    cfg.retrieval.filter_references_for_regular_rag = False
    cfg.retrieval.min_figure_caption_chars_for_regular_rag = 50
    cfg.reranker.provider = "flagembedding"
    cfg.reranker.model_name = "test"

    summary = build_summary(
        details=details,
        cfg=cfg,
        dataset_path="test.jsonl",
        run_dir=Path("/tmp/test"),
        details_path=Path("/tmp/test/details.jsonl"),
        ks=(5, 10),
    )

    # 全量 summary 保留
    assert summary["total_cases"] == 3
    assert "average_scores" in summary

    # group_summaries 存在
    assert "group_summaries" in summary
    gs = summary["group_summaries"]

    assert "all" in gs
    assert gs["all"]["total_cases"] == 3

    assert "local_answerable" in gs
    assert gs["local_answerable"]["total_cases"] == 2

    assert "local_single_doc" in gs
    assert gs["local_single_doc"]["total_cases"] == 1

    assert "local_multi_doc" in gs
    assert gs["local_multi_doc"]["total_cases"] == 1

    assert "fallback_safety_excluded" in gs
    assert gs["fallback_safety_excluded"]["total_cases"] == 1


def test_build_summary_group_summaries_average_correct() -> None:
    """分组平均值应只包含该组的样本。"""
    from unittest.mock import MagicMock

    details = [
        _make_detail("c1", ["all", "local_answerable"], recall_10=0.8),
        _make_detail("c2", ["all", "local_answerable"], recall_10=0.6),
        _make_detail("c3", ["all", "fallback_safety_excluded"], recall_10=None, gold_signal_count=0),
    ]
    cfg = MagicMock()
    cfg.qdrant.collection = "test"
    cfg.retrieval.max_rewrite_queries = 4
    cfg.retrieval.parallel_rewrite_queries = False
    cfg.retrieval.rewrite_query_max_workers = 4
    cfg.retrieval.dense_top_k = 30
    cfg.retrieval.sparse_top_k = 30
    cfg.retrieval.fusion_top_k = 50
    cfg.retrieval.rerank_top_k = 10
    cfg.retrieval.dense_weight = 0.65
    cfg.retrieval.sparse_weight = 0.35
    cfg.retrieval.rrf_k = 60
    cfg.retrieval.chunk_type_filter_mode = "off"
    cfg.retrieval.filter_references_for_regular_rag = False
    cfg.retrieval.min_figure_caption_chars_for_regular_rag = 50
    cfg.reranker.provider = "flagembedding"
    cfg.reranker.model_name = "test"

    summary = build_summary(
        details=details,
        cfg=cfg,
        dataset_path="test.jsonl",
        run_dir=Path("/tmp/test"),
        details_path=Path("/tmp/test/details.jsonl"),
        ks=(5, 10),
    )

    gs = summary["group_summaries"]
    # local_answerable: (0.8 + 0.6) / 2 = 0.7
    assert gs["local_answerable"]["average_scores"]["reranked.recall@10"] == 0.7
    # all 包含 3 条，但 c3 的 recall@10 为 None 不参与平均
    assert gs["all"]["average_scores"]["reranked.recall@10"] == 0.7


def test_build_failure_report_groups_by_evaluation_groups() -> None:
    """failure report 应按分组输出低分样本。"""
    details = [
        _make_detail("c1", ["all", "local_answerable"], recall_10=0.9),
        _make_detail("c2", ["all", "local_answerable"], recall_10=0.1),
        _make_detail("c3", ["all", "fallback_safety_excluded"], recall_10=None, gold_signal_count=0),
    ]
    report = build_failure_report(details, max_per_group=10)

    assert "all" in report
    assert "local_answerable" in report
    assert "fallback_safety_excluded" in report

    # local_answerable 应按 recall@10 升序排列，c2 (0.1) 排第一
    la = report["local_answerable"]
    assert len(la) == 2
    assert la[0]["case_id"] == "c2"
    assert la[1]["case_id"] == "c1"


def test_build_failure_report_max_per_group() -> None:
    """每组最多输出 max_per_group 条。"""
    details = [_make_detail(f"c{i}", ["all"], recall_10=i * 0.1) for i in range(30)]
    report = build_failure_report(details, max_per_group=5)
    assert len(report["all"]) == 5


def test_build_failure_report_contains_fields() -> None:
    """failure record 应包含文档要求的关键字段。"""
    details = [_make_detail("c1", ["all"], recall_10=0.3)]
    report = build_failure_report(details)
    record = report["all"][0]
    expected_keys = [
        "case_id", "question", "intent", "groups",
        "gold_signal_count", "reranked_recall@10",
        "top_retrieved_titles", "errors",
    ]
    for key in expected_keys:
        assert key in record, f"missing key: {key}"


def test_write_failure_csv(tmp_path: Path) -> None:
    """write_failure_csv 应输出可读 CSV。"""
    import csv

    details = [
        _make_detail("c1", ["all", "local_answerable"], recall_10=0.5),
        _make_detail("c2", ["all"], recall_10=0.1),
    ]
    report = build_failure_report(details, max_per_group=10)
    csv_path = tmp_path / "failures.csv"
    write_failure_csv(csv_path, report)

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3  # c1 in all+local_answerable, c2 in all
    assert "group" in rows[0]
    assert "case_id" in rows[0]
