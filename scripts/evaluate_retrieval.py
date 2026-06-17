"""Evaluate FireAgent retrieval stages without generating answers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.evaluation.dataset import load_evaluation_cases, write_json
from fireagent.evaluation.doc_hit_metrics import (
    DocHitGold,
    extract_doc_hit_gold_from_metadata,
    score_doc_hit,
)
from fireagent.evaluation.evaluation_groups import (
    GROUP_ALL,
    extract_case_metadata_for_detail,
    get_evaluation_groups_from_case,
)
from fireagent.evaluation.retrieval_metrics import (
    RetrievedChunk,
    RetrievalGold,
    score_retrieved_chunks,
)
from fireagent.retrieval.candidate_filter import RegularRAGCandidateFilter
from fireagent.retrieval.dense_retriever import DenseRetriever
from fireagent.retrieval.hybrid_fusion import WeightedRRFFusion
from fireagent.retrieval.query_rewriter import QueryRewriter
from fireagent.retrieval.reranker import LexicalReranker, create_reranker
from fireagent.retrieval.schema import FusedRetrievalResult, RerankedRetrievalResult
from fireagent.retrieval.sparse_retriever import SparseRetriever
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore import FireAgentQdrantClient
from fireagent.vectorstore.schema import VectorSearchResult


def parse_args() -> argparse.Namespace:
    """Parse retrieval evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="运行 FireAgent 检索阶段专项评估。")
    parser.add_argument("--dataset", required=True, help="RAG 评估 JSONL 数据集路径。")
    parser.add_argument("--output-dir", required=True, help="评估输出目录。")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条样本。")
    parser.add_argument(
        "--ks",
        default="5,10",
        help="Top-k 指标列表，逗号分隔，例如 5,10,20。",
    )
    parser.add_argument(
        "--reference-overlap-threshold",
        type=float,
        default=0.75,
        help="reference_contexts 词项覆盖阈值，默认 0.75。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭逐题进度输出。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 打印 summary。")
    parser.add_argument(
        "--include-full-text",
        action="store_true",
        help="在 detail 中写入完整 chunk text（默认只写 text_preview）。",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = get_config()
    ks = parse_ks(args.ks)
    cases = load_evaluation_cases(args.dataset, limit=args.limit)
    run_dir = make_run_dir(args.output_dir)

    evaluator = RetrievalEvaluationRunner(config=cfg, ks=ks)
    details = evaluator.run_cases(
        cases,
        show_progress=not args.no_progress,
        reference_overlap_threshold=args.reference_overlap_threshold,
        include_full_text=args.include_full_text,
    )

    details_path = run_dir / "retrieval_details.jsonl"
    write_jsonl(details_path, details)
    summary = build_summary(
        details=details,
        cfg=cfg,
        dataset_path=args.dataset,
        run_dir=run_dir,
        details_path=details_path,
        ks=ks,
    )
    write_json(run_dir / "summary.json", summary)

    failure_report = build_failure_report(details, max_per_group=20)
    write_json(run_dir / "retrieval_failures.json", failure_report)
    write_failure_csv(run_dir / "retrieval_failures.csv", failure_report)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"评估样本数：{summary['total_cases']}")
        print(f"可评分样本数：{summary['scored_cases']}")
        print(f"输出目录：{run_dir}")
        print("平均指标：")
        for name, value in summary["average_scores"].items():
            print(f"- {name}: {value:.4f}")
        if summary.get("group_summaries"):
            print("分组指标：")
            for group_name, group_data in summary["group_summaries"].items():
                print(f"  [{group_name}] n={group_data['total_cases']}")


class RetrievalEvaluationRunner:
    """Run query rewrite, dense/sparse retrieval, fusion, and rerank for evaluation cases."""

    def __init__(self, config: FireAgentConfig | None = None, ks: tuple[int, ...] = (5, 10)) -> None:
        self.config = config or get_config()
        self.ks = ks
        self.vectorstore = FireAgentQdrantClient(config=self.config)
        self.query_rewriter = QueryRewriter(config=self.config)
        self.dense_retriever = DenseRetriever(self.vectorstore, config=self.config)
        self.sparse_retriever = SparseRetriever(self.vectorstore, config=self.config)
        self.candidate_filter = RegularRAGCandidateFilter(config=self.config)
        self.fusion = WeightedRRFFusion(config=self.config)
        self.reranker = create_reranker(config=self.config)

    def run_cases(
        self,
        cases,
        show_progress: bool = True,
        reference_overlap_threshold: float = 0.75,
        include_full_text: bool = False,
    ) -> list[dict[str, Any]]:
        """Evaluate all cases and return serializable detail records."""
        records: list[dict[str, Any]] = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            if show_progress:
                print(f"检索评估 {index}/{total}: {case.case_id}", file=sys.stderr)
            records.append(
                self.run_case(
                    case,
                    reference_overlap_threshold=reference_overlap_threshold,
                    include_full_text=include_full_text,
                )
            )
        return records

    def run_case(self, case, reference_overlap_threshold: float = 0.75, include_full_text: bool = False) -> dict[str, Any]:
        """Evaluate one case across retrieval stages."""
        start = time.perf_counter()
        gold = RetrievalGold(
            required_citation_substrings=list(case.required_citation_substrings or []),
            reference_contexts=list(case.reference_contexts or []),
        )
        errors: list[str] = []
        stage_chunks: dict[str, list[RetrievedChunk]] = {
            "dense": [],
            "sparse": [],
            "fused": [],
            "reranked": [],
        }
        rewrite_payload: dict[str, Any] = {}
        filter_payload: dict[str, Any] = {}

        try:
            rewrite_result = self.query_rewriter.rewrite(case.question)
            rewrite_payload = {
                "original_query": rewrite_result.original_query,
                "main_query": rewrite_result.main_query,
                "expanded_queries": rewrite_result.expanded_queries,
                "all_queries": rewrite_result.all_queries,
                "metadata": rewrite_result.metadata,
            }

            dense_results = self.dense_retriever.retrieve_many(rewrite_result)
            sparse_results = self.sparse_retriever.retrieve_many(rewrite_result)
            dense_filtered, dense_filter_stats = self.candidate_filter.filter_many(
                rewrite_result.main_query,
                dense_results,
            )
            sparse_filtered, sparse_filter_stats = self.candidate_filter.filter_many(
                rewrite_result.main_query,
                sparse_results,
            )
            filter_payload = {
                "dense": dense_filter_stats.__dict__,
                "sparse": sparse_filter_stats.__dict__,
            }
            fused_results = self.fusion.fuse(dense_filtered, sparse_filtered)
            try:
                reranked_results = self.reranker.rerank(
                    rewrite_result.main_query,
                    fused_results,
                    top_k=self.config.retrieval.rerank_top_k,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"rerank fallback: {exc}")
                reranked_results = LexicalReranker().rerank(
                    rewrite_result.main_query,
                    fused_results,
                    top_k=self.config.retrieval.rerank_top_k,
                )

            stage_chunks = {
                "dense": vector_results_to_chunks(dense_filtered, stage="dense"),
                "sparse": vector_results_to_chunks(sparse_filtered, stage="sparse"),
                "fused": fused_results_to_chunks(fused_results),
                "reranked": reranked_results_to_chunks(reranked_results),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        stage_scores = {
            stage: score_retrieved_chunks(
                gold=gold,
                chunks=list(chunks),
                ks=self.ks,
                reference_overlap_threshold=reference_overlap_threshold,
            ).to_dict()
            for stage, chunks in stage_chunks.items()
        }

        # Phase 3: doc-hit 指标
        doc_hit_gold = extract_doc_hit_gold_from_metadata(case.metadata)
        doc_hit_scores: dict[str, dict[str, object]] = {}
        if doc_hit_gold.gold_doc_names:
            for stage, chunks in stage_chunks.items():
                doc_hit_scores[stage] = score_doc_hit(
                    gold=doc_hit_gold,
                    chunks=list(chunks),
                    ks=self.ks,
                ).to_dict()

        # Step 2: 写入 metadata 和 evaluation_groups
        case_metadata = extract_case_metadata_for_detail(case.metadata)
        evaluation_groups = get_evaluation_groups_from_case(case)

        # Step 5: 控制 text 字段写入
        retrieved_payload: dict[str, Any] = {}
        for stage, chunks in stage_chunks.items():
            chunk_dicts = []
            for chunk in chunks:
                d = chunk.to_dict()
                if not include_full_text:
                    d.pop("text", None)
                # 清理 text_preview 中的控制字符
                if d.get("text_preview"):
                    d["text_preview"] = _sanitize_text(d["text_preview"])
                chunk_dicts.append(d)
            retrieved_payload[stage] = chunk_dicts

        return {
            "case_id": case.case_id,
            "question": case.question,
            "intent": case.intent,
            "tags": case.tags,
            "metadata": case_metadata,
            "evaluation_groups": evaluation_groups,
            "gold": {
                "required_citation_substrings": gold.required_citation_substrings,
                "reference_contexts": gold.reference_contexts,
                "gold_signal_count": len(gold.required_citation_substrings) + len(gold.reference_contexts),
            },
            "rewrite": rewrite_payload,
            "filter": filter_payload,
            "stage_scores": stage_scores,
            "doc_hit_scores": doc_hit_scores,
            "retrieved": retrieved_payload,
            "latency_seconds": round(time.perf_counter() - start, 4),
            "errors": errors,
        }


def vector_results_to_chunks(
    results: list[VectorSearchResult],
    stage: str,
) -> list[RetrievedChunk]:
    """Convert vector search results to ranked chunk records."""
    chunks: list[RetrievedChunk] = []
    for index, result in enumerate(results, start=1):
        payload = result.payload or {}
        chunks.append(
            RetrievedChunk(
                rank=int(result.rank or index),
                chunk_id=result.chunk_id,
                text=result.text or str(payload.get("text", "") or ""),
                score=float(result.score),
                stage=stage,
                doc_id=str(payload.get("doc_id", "") or ""),
                parent_id=str(payload.get("parent_id", "") or ""),
                paper_title=str(payload.get("paper_title", "") or ""),
                section_title=str(payload.get("section_title", "") or ""),
                chunk_type=str(payload.get("chunk_type", "") or ""),
                page_start=_optional_int(payload.get("page_start")),
                page_end=_optional_int(payload.get("page_end")),
                sources=[stage],
            )
        )
    return chunks


def fused_results_to_chunks(results: list[FusedRetrievalResult]) -> list[RetrievedChunk]:
    """Convert fused retrieval results to ranked chunk records."""
    chunks: list[RetrievedChunk] = []
    for index, result in enumerate(results, start=1):
        payload = result.payload or {}
        chunks.append(
            RetrievedChunk(
                rank=int(result.rank or index),
                chunk_id=result.chunk_id,
                text=result.text or str(payload.get("text", "") or ""),
                score=float(result.fusion_score),
                stage="fused",
                doc_id=str(payload.get("doc_id", "") or ""),
                parent_id=str(payload.get("parent_id", "") or ""),
                paper_title=str(payload.get("paper_title", "") or ""),
                section_title=str(payload.get("section_title", "") or ""),
                chunk_type=str(payload.get("chunk_type", "") or ""),
                page_start=_optional_int(payload.get("page_start")),
                page_end=_optional_int(payload.get("page_end")),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
                dense_score=result.dense_score,
                sparse_score=result.sparse_score,
                fusion_score=result.fusion_score,
                sources=list(result.sources),
            )
        )
    return chunks


def reranked_results_to_chunks(results: list[RerankedRetrievalResult]) -> list[RetrievedChunk]:
    """Convert reranked retrieval results to ranked chunk records."""
    chunks: list[RetrievedChunk] = []
    for result in results:
        payload = result.payload or {}
        chunks.append(
            RetrievedChunk(
                rank=int(result.rank),
                chunk_id=result.chunk_id,
                text=result.text or str(payload.get("text", "") or ""),
                score=float(result.final_score),
                stage="reranked",
                doc_id=str(payload.get("doc_id", "") or ""),
                parent_id=str(payload.get("parent_id", "") or ""),
                paper_title=str(payload.get("paper_title", "") or ""),
                section_title=str(payload.get("section_title", "") or ""),
                chunk_type=str(payload.get("chunk_type", "") or ""),
                page_start=_optional_int(payload.get("page_start")),
                page_end=_optional_int(payload.get("page_end")),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
                dense_score=result.dense_score,
                sparse_score=result.sparse_score,
                fusion_score=result.fusion_score,
                rerank_score=result.rerank_score,
                sources=list(result.sources),
            )
        )
    return chunks


def _compute_average_scores(
    details: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, float]:
    """从 detail 列表计算平均检索指标。"""
    average_scores: dict[str, float] = {}
    stages = ("dense", "sparse", "fused", "reranked")
    for stage in stages:
        for metric_group in ("hit_at_k", "recall_at_k", "precision_at_k", "mrr_at_k"):
            keys = [f"{metric_group.split('_at_k')[0]}@{k}" for k in ks]
            for key in keys:
                values = [
                    record["stage_scores"][stage][metric_group].get(key)
                    for record in details
                    if record["stage_scores"][stage][metric_group].get(key) is not None
                ]
                if values:
                    average_scores[f"{stage}.{key}"] = round(sum(values) / len(values), 4)
        for scalar in ("required_citation_hit_rate", "reference_context_coverage"):
            values = [
                record["stage_scores"][stage].get(scalar)
                for record in details
                if record["stage_scores"][stage].get(scalar) is not None
            ]
            if values:
                average_scores[f"{stage}.{scalar}"] = round(sum(values) / len(values), 4)
    # doc-hit 指标（仅对有 doc_hit_scores 的样本计算）
    doc_hit_stages = ("reranked",)
    doc_hit_metric_groups = ("hit_at_k", "recall_at_k", "unique_doc_count_at_k")
    for stage in doc_hit_stages:
        for metric_group in doc_hit_metric_groups:
            prefix = "doc_" if "unique" not in metric_group else ""
            keys = [f"{prefix}{metric_group.split('_at_k')[0]}@{k}" for k in ks]
            for key in keys:
                values = [
                    record["doc_hit_scores"][stage][metric_group].get(key)
                    for record in details
                    if record.get("doc_hit_scores", {}).get(stage, {}).get(metric_group, {}).get(key) is not None
                ]
                if values:
                    average_scores[f"doc_hit.{stage}.{key}"] = round(sum(values) / len(values), 4)

    return average_scores


def build_summary(
    details: list[dict[str, Any]],
    cfg: FireAgentConfig,
    dataset_path: str,
    run_dir: Path,
    details_path: Path,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Build aggregate retrieval evaluation summary."""
    average_scores = _compute_average_scores(details, ks)

    scored_cases = sum(
        1 for record in details if record["stage_scores"]["reranked"]["gold_signal_count"] > 0
    )
    error_cases = sum(1 for record in details if record.get("errors"))

    # Step 3: 按分组计算 group_summaries
    group_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in details:
        for group in record.get("evaluation_groups", [GROUP_ALL]):
            group_details[group].append(record)

    group_summaries: dict[str, dict[str, Any]] = {}
    for group_name in sorted(group_details.keys()):
        group_records = group_details[group_name]
        group_scored = sum(
            1 for r in group_records if r["stage_scores"]["reranked"]["gold_signal_count"] > 0
        )
        group_avg = _compute_average_scores(group_records, ks)
        group_summaries[group_name] = {
            "total_cases": len(group_records),
            "scored_cases": group_scored,
            "no_gold_cases": len(group_records) - group_scored,
            "error_cases": sum(1 for r in group_records if r.get("errors")),
            "average_scores": group_avg,
        }

    return {
        "total_cases": len(details),
        "scored_cases": scored_cases,
        "no_gold_cases": len(details) - scored_cases,
        "error_cases": error_cases,
        "dataset_path": dataset_path,
        "run_dir": str(run_dir),
        "details_path": str(details_path),
        "average_scores": average_scores,
        "group_summaries": group_summaries,
        "config": {
            "qdrant_collection": cfg.qdrant.collection,
            "max_rewrite_queries": cfg.retrieval.max_rewrite_queries,
            "parallel_rewrite_queries": cfg.retrieval.parallel_rewrite_queries,
            "rewrite_query_max_workers": cfg.retrieval.rewrite_query_max_workers,
            "dense_top_k": cfg.retrieval.dense_top_k,
            "sparse_top_k": cfg.retrieval.sparse_top_k,
            "fusion_top_k": cfg.retrieval.fusion_top_k,
            "rerank_top_k": cfg.retrieval.rerank_top_k,
            "dense_weight": cfg.retrieval.dense_weight,
            "sparse_weight": cfg.retrieval.sparse_weight,
            "rrf_k": cfg.retrieval.rrf_k,
            "chunk_type_filter_mode": cfg.retrieval.chunk_type_filter_mode,
            "filter_references_for_regular_rag": cfg.retrieval.filter_references_for_regular_rag,
            "min_figure_caption_chars_for_regular_rag": cfg.retrieval.min_figure_caption_chars_for_regular_rag,
            "reranker_provider": cfg.reranker.provider,
            "reranker_model_name": cfg.reranker.model_name,
        },
    }


def parse_ks(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated top-k list."""
    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError("--ks 中的 k 必须为正整数。")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--ks 至少需要一个正整数。")
    return tuple(sorted(values))


def make_run_dir(output_dir: str | Path) -> Path:
    """Create a timestamped run directory."""
    base_dir = Path(output_dir)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write detail records as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _sanitize_text(text: str) -> str:
    """清理文本中的控制字符和非法 surrogate，保证 JSONL 稳定性。"""
    # 移除 C0 控制字符（保留 \t \n \r）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # 移除 C1 控制字符
    text = re.sub(r"[\x80-\x9f]", "", text)
    # 移除 lone surrogates
    text = re.sub(r"[\ud800-\udfff]", "", text)
    return text


def build_failure_report(
    details: list[dict[str, Any]],
    max_per_group: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """为每个分组输出 recall@10 最低的样本。"""
    group_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in details:
        for group in record.get("evaluation_groups", [GROUP_ALL]):
            group_details[group].append(record)

    report: dict[str, list[dict[str, Any]]] = {}
    for group_name in sorted(group_details.keys()):
        records = group_details[group_name]

        def _sort_key(r: dict[str, Any]) -> tuple[float, float, float]:
            scores = r.get("stage_scores", {}).get("reranked", {})
            recall = scores.get("recall_at_k", {}).get("recall@10")
            hit = scores.get("hit_at_k", {}).get("hit@10")
            mrr = scores.get("mrr_at_k", {}).get("mrr@10")
            return (
                recall if recall is not None else 1.0,
                hit if hit is not None else 1.0,
                mrr if mrr is not None else 1.0,
            )

        sorted_records = sorted(records, key=_sort_key)
        failures: list[dict[str, Any]] = []
        for r in sorted_records[:max_per_group]:
            reranked_scores = r.get("stage_scores", {}).get("reranked", {})
            reranked_chunks = r.get("retrieved", {}).get("reranked", [])
            top_titles = [c.get("paper_title", "") for c in reranked_chunks[:10] if c.get("paper_title")]
            top_sections = [c.get("section_title", "") for c in reranked_chunks[:10] if c.get("section_title")]
            top_chunk_ids = [c.get("chunk_id", "") for c in reranked_chunks[:10]]
            failures.append({
                "case_id": r.get("case_id"),
                "question": r.get("question"),
                "intent": r.get("intent"),
                "question_type": r.get("metadata", {}).get("question_type", ""),
                "evidence_scope": r.get("metadata", {}).get("evidence_scope", ""),
                "expected_behavior": r.get("metadata", {}).get("expected_behavior", ""),
                "expected_action": r.get("metadata", {}).get("expected_action", ""),
                "groups": r.get("evaluation_groups", []),
                "gold_signal_count": r.get("gold", {}).get("gold_signal_count", 0),
                "reranked_hit@10": reranked_scores.get("hit_at_k", {}).get("hit@10"),
                "reranked_recall@10": reranked_scores.get("recall_at_k", {}).get("recall@10"),
                "reranked_precision@10": reranked_scores.get("precision_at_k", {}).get("precision@10"),
                "reranked_mrr@10": reranked_scores.get("mrr_at_k", {}).get("mrr@10"),
                "required_citation_hit_rate": reranked_scores.get("required_citation_hit_rate"),
                "reference_context_coverage": reranked_scores.get("reference_context_coverage"),
                "gold_required_citation_substrings": r.get("gold", {}).get("required_citation_substrings", []),
                "gold_reference_context_count": len(r.get("gold", {}).get("reference_contexts", [])),
                "top_retrieved_titles": top_titles,
                "top_retrieved_sections": top_sections,
                "top_retrieved_chunk_ids": top_chunk_ids,
                "matched_required_citation_substrings": [
                    c.get("matched_required_citation_substrings", [])
                    for c in reranked_chunks[:10]
                    if c.get("matched_required_citation_substrings")
                ],
                "matched_reference_context_indices": [
                    c.get("matched_reference_context_indices", [])
                    for c in reranked_chunks[:10]
                    if c.get("matched_reference_context_indices")
                ],
                "errors": r.get("errors", []),
            })
        report[group_name] = failures

    return report


def write_failure_csv(
    path: str | Path,
    report: dict[str, list[dict[str, Any]]],
) -> None:
    """将 failure report 写为 CSV。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "case_id",
        "question",
        "intent",
        "evidence_scope",
        "expected_behavior",
        "gold_signal_count",
        "reranked_hit@10",
        "reranked_recall@10",
        "reranked_precision@10",
        "reranked_mrr@10",
        "required_citation_hit_rate",
        "reference_context_coverage",
        "top_retrieved_titles",
        "errors",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for group_name, failures in sorted(report.items()):
            for failure in failures:
                row = {**failure, "group": group_name}
                # 列表字段转为分号分隔字符串
                for key in ("top_retrieved_titles", "errors"):
                    if isinstance(row.get(key), list):
                        row[key] = "; ".join(str(v) for v in row[key])
                writer.writerow(row)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
