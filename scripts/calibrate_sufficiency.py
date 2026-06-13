"""Sufficiency 阈值校准脚本。

用评估集批量运行本地检索和 sufficiency checker，导出特征数据，
做阈值网格搜索，输出推荐配置。

用法：
    python scripts/calibrate_sufficiency.py \
        --dataset data/eval/fireagent_benchmark_v1_seed60.jsonl \
        --output-dir data/eval/runs/sufficiency_calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.evaluation.dataset import load_evaluation_cases
from fireagent.graph.nodes import route_intent
from fireagent.retrieval import (
    DenseRetriever,
    FallbackAction,
    FallbackPolicy,
    LexicalReranker,
    LocalEvidenceSufficiencyChecker,
    QueryRewriter,
    SparseRetriever,
    WeightedRRFFusion,
    create_reranker,
    decide_fallback,
)
from fireagent.utils.config import get_config
from fireagent.vectorstore import FireAgentQdrantClient


def infer_expected_action(question_type: str, metadata: dict) -> str:
    """根据评估集的 question_type 和 metadata 推断期望动作。"""
    if metadata.get("expected_action"):
        return str(metadata["expected_action"])
    if metadata.get("requires_current_web"):
        return "use_web"
    if metadata.get("harmful"):
        return "refuse"

    qt = question_type.lower()
    if qt in {"fact", "method", "mechanism", "single_doc_summary", "citation"}:
        return "answer_local"
    if qt in {"cross_doc_compare"}:
        return "answer_local"
    if qt in {"web_fallback"}:
        return "use_web"
    if qt in {"emergency"}:
        return "answer_local"
    if qt in {"no_answer"}:
        if metadata.get("expected_web"):
            return "use_web"
        if metadata.get("expected_refuse"):
            return "refuse"
        return "answer_insufficient"
    return "answer_local"


def run_calibration(
    dataset_path: str,
    output_dir: str,
    limit: int | None = None,
) -> dict:
    """运行校准流程。"""
    cfg = get_config()
    cases = load_evaluation_cases(dataset_path, limit=limit)

    vs = FireAgentQdrantClient(config=cfg)
    rewriter = QueryRewriter(config=cfg)
    fusion = WeightedRRFFusion(config=cfg)
    try:
        reranker = create_reranker(cfg)
    except Exception:
        reranker = LexicalReranker()
    dense_retriever = DenseRetriever(vs, config=cfg)
    sparse_retriever = SparseRetriever(vs, config=cfg)

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    features_list: list[dict] = []

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case.question[:60]}...")
        start = time.time()

        # 查询改写
        rewrite_result = rewriter.rewrite(case.question)

        # 检索
        dense_results = dense_retriever.retrieve_many(rewrite_result)
        sparse_results = sparse_retriever.retrieve_many(rewrite_result)
        fused = fusion.fuse(dense_results, sparse_results)
        reranked = reranker.rerank(rewrite_result.main_query, fused, top_k=cfg.retrieval.rerank_top_k)

        # Sufficiency check
        sufficiency_checker = LocalEvidenceSufficiencyChecker(config=cfg)
        sufficiency_result = sufficiency_checker.check(rewrite_result.main_query, reranked)

        # Fallback decision
        intent, _ = route_intent(case.question)
        fallback_decision = decide_fallback(
            query=rewrite_result.main_query,
            intent=intent,
            sufficiency=sufficiency_result,
            candidates=reranked,
            config=cfg,
        )

        elapsed = time.time() - start

        # 期望动作
        expected_action = infer_expected_action(
            case.metadata.get("question_type", "") if case.metadata else "",
            case.metadata or {},
        )

        # 记录特征
        feature = {
            "case_id": case.case_id,
            "question": case.question,
            "question_type": case.metadata.get("question_type", "") if case.metadata else "",
            "expected_action": expected_action,
            "intent": intent,
            "top_score": round(sufficiency_result.top_score, 4),
            "evidence_count": sufficiency_result.evidence_count,
            "term_coverage": round(sufficiency_result.metadata.get("term_coverage", 0), 4),
            "doc_count": sufficiency_result.metadata.get("doc_count", 0),
            "score_gap": sufficiency_result.metadata.get("score_gap", 0),
            "matched_terms": sufficiency_result.matched_terms,
            "missing_terms": sufficiency_result.missing_terms,
            "temporal_keywords": sufficiency_result.metadata.get("temporal_keywords", []),
            "official_keywords": sufficiency_result.metadata.get("official_keywords", []),
            "local_scope_keywords": sufficiency_result.metadata.get("local_scope_keywords", []),
            "harmful_keywords": sufficiency_result.metadata.get("harmful_keywords", []),
            "sufficient": sufficiency_result.sufficient,
            "needs_web": sufficiency_result.needs_web,
            "sufficiency_reason": sufficiency_result.reason,
            "fallback_action": fallback_decision.action.value,
            "fallback_reason": fallback_decision.reason,
            "actual_action": fallback_decision.action.value,
            "correct": fallback_decision.action.value == expected_action,
            "elapsed": round(elapsed, 2),
        }
        features_list.append(feature)

    # 写出 CSV
    csv_path = run_dir / "sufficiency_features.csv"
    if features_list:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=features_list[0].keys())
            writer.writeheader()
            writer.writerows(features_list)

    # 写出 JSONL
    jsonl_path = run_dir / "sufficiency_features.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for feat in features_list:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")

    # 统计指标
    metrics = compute_metrics(features_list)

    # 网格搜索
    grid_results = grid_search(features_list, cfg)

    # 写出 summary
    summary = {
        "dataset": dataset_path,
        "total_cases": len(features_list),
        "metrics": metrics,
        "best_grid_config": grid_results.get("best_config", {}),
        "best_grid_score": grid_results.get("best_score", 0),
        "grid_results_top5": grid_results.get("top5", []),
    }
    summary_path = run_dir / "sufficiency_calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 写出失败案例分析
    failures = [f for f in features_list if not f["correct"]]
    failure_path = run_dir / "fallback_failure_analysis.csv"
    if failures:
        with open(failure_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=failures[0].keys())
            writer.writeheader()
            writer.writerows(failures)

    print(f"\n{'='*50}")
    print(f"校准完成")
    print(f"{'='*50}")
    print(f"总样本数: {len(features_list)}")
    print(f"正确数: {sum(1 for f in features_list if f['correct'])}")
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"Web recall: {metrics['web_recall']:.2%}")
    print(f"Local recall: {metrics['local_recall']:.2%}")
    print(f"False web rate: {metrics['false_web_rate']:.2%}")
    print(f"Refusal accuracy: {metrics['refusal_accuracy']:.2%}")
    print(f"\n输出目录: {run_dir}")
    print(f"  - sufficiency_features.csv")
    print(f"  - sufficiency_features.jsonl")
    print(f"  - sufficiency_calibration_summary.json")
    print(f"  - fallback_failure_analysis.csv")

    return summary


def compute_metrics(features: list[dict]) -> dict:
    """计算各项指标。"""
    total = len(features)
    if total == 0:
        return {}

    correct = sum(1 for f in features if f["correct"])

    # 按 expected_action 分组
    expected_local = [f for f in features if f["expected_action"] == "answer_local"]
    expected_web = [f for f in features if f["expected_action"] == "use_web"]
    expected_refuse = [f for f in features if f["expected_action"] == "refuse"]
    expected_insufficient = [f for f in features if f["expected_action"] == "answer_insufficient"]

    # local_recall: 应该本地回答的，实际也是本地回答
    local_correct = sum(1 for f in expected_local if f["actual_action"] == "answer_local")
    local_recall = local_correct / max(len(expected_local), 1)

    # web_recall: 应该联网的，实际也联网了
    web_correct = sum(1 for f in expected_web if f["actual_action"] == "use_web")
    web_recall = web_correct / max(len(expected_web), 1)

    # web_precision: 联网的里面，确实应该联网的
    actual_web = [f for f in features if f["actual_action"] == "use_web"]
    web_true = sum(1 for f in actual_web if f["expected_action"] == "use_web")
    web_precision = web_true / max(len(actual_web), 1)

    # false_web_rate: 本地可答却触发 web
    false_web = sum(1 for f in expected_local if f["actual_action"] == "use_web")
    false_web_rate = false_web / max(len(expected_local), 1)

    # missed_web_rate: 应该联网却没联网
    missed_web = sum(1 for f in expected_web if f["actual_action"] != "use_web")
    missed_web_rate = missed_web / max(len(expected_web), 1)

    # refusal_accuracy: 应拒答的是否正确拒答
    refusal_correct = sum(1 for f in expected_refuse if f["actual_action"] == "refuse")
    refusal_accuracy = refusal_correct / max(len(expected_refuse), 1)

    return {
        "accuracy": correct / total,
        "local_recall": local_recall,
        "web_recall": web_recall,
        "web_precision": web_precision,
        "false_web_rate": false_web_rate,
        "missed_web_rate": missed_web_rate,
        "refusal_accuracy": refusal_accuracy,
        "total": total,
        "expected_local": len(expected_local),
        "expected_web": len(expected_web),
        "expected_refuse": len(expected_refuse),
        "expected_insufficient": len(expected_insufficient),
    }


def grid_search(features: list[dict], cfg) -> dict:
    """网格搜索阈值组合。"""
    min_score_candidates = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    min_evidence_candidates = [1, 2, 3]
    min_term_coverage_candidates = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    results = []
    total_combos = len(min_score_candidates) * len(min_evidence_candidates) * len(min_term_coverage_candidates)
    print(f"\n网格搜索: {total_combos} 种阈值组合...")

    for min_score in min_score_candidates:
        for min_evidence in min_evidence_candidates:
            for min_coverage in min_term_coverage_candidates:
                # 用新阈值重新判断每条样本
                correct = 0
                web_recall_num = 0
                web_recall_den = 0
                local_recall_num = 0
                local_recall_den = 0
                false_web_num = 0
                refusal_correct_num = 0
                refusal_den = 0

                for feat in features:
                    # 重新判断 sufficiency
                    sufficient = (
                        feat["top_score"] >= min_score
                        and feat["evidence_count"] >= min_evidence
                        and feat["term_coverage"] >= min_coverage
                        and not feat["temporal_keywords"]
                    )

                    # 重新判断 fallback
                    if feat["harmful_keywords"]:
                        actual = "refuse"
                    elif feat["local_scope_keywords"] and not sufficient:
                        actual = "answer_insufficient"
                    elif feat["temporal_keywords"]:
                        actual = "use_web"
                    elif feat["official_keywords"]:
                        actual = "use_web"
                    elif sufficient:
                        actual = "answer_local"
                    else:
                        actual = "use_web"

                    expected = feat["expected_action"]
                    if actual == expected:
                        correct += 1

                    if expected == "use_web":
                        web_recall_den += 1
                        if actual == "use_web":
                            web_recall_num += 1
                    if expected == "answer_local":
                        local_recall_den += 1
                        if actual == "answer_local":
                            local_recall_num += 1
                        if actual == "use_web":
                            false_web_num += 1
                    if expected == "refuse":
                        refusal_den += 1
                        if actual == "refuse":
                            refusal_correct_num += 1

                web_recall = web_recall_num / max(web_recall_den, 1)
                local_recall = local_recall_num / max(local_recall_den, 1)
                false_web_rate = false_web_num / max(local_recall_den, 1)
                refusal_acc = refusal_correct_num / max(refusal_den, 1)

                # 加权目标
                score = (
                    0.35 * web_recall
                    + 0.25 * local_recall
                    + 0.15 * refusal_acc
                    - 0.20 * false_web_rate
                )

                results.append({
                    "min_score": min_score,
                    "min_evidence": min_evidence,
                    "min_term_coverage": min_coverage,
                    "accuracy": correct / len(features),
                    "web_recall": web_recall,
                    "local_recall": local_recall,
                    "false_web_rate": false_web_rate,
                    "refusal_accuracy": refusal_acc,
                    "weighted_score": score,
                })

    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    best = results[0] if results else {}

    return {
        "best_config": {
            "sufficiency_min_score": best.get("min_score"),
            "sufficiency_min_evidence": best.get("min_evidence"),
            "sufficiency_min_term_coverage": best.get("min_term_coverage"),
        },
        "best_score": best.get("weighted_score", 0),
        "top5": results[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Sufficiency 阈值校准")
    parser.add_argument("--dataset", required=True, help="评估集路径")
    parser.add_argument("--output-dir", default="data/eval/runs/sufficiency_calibration", help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="限制样本数")
    args = parser.parse_args()

    run_calibration(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
