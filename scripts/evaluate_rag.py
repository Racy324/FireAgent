"""FireAgent RAG 评测命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.evaluation import RAGEvaluationRunner
from fireagent.utils.config import get_config
from fireagent.vectorstore import FireAgentQdrantClient


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 FireAgent RAG 评测。")
    parser.add_argument(
        "--dataset",
        default=None,
        help="评测集 JSONL 路径；默认读取 EVAL_DATASET_PATH。",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="已有预测 JSONL；提供后不会重新调用 FireAgent。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="评测报告输出目录；默认读取 EVAL_OUTPUT_DIR。",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "ragas", "both"],
        default=None,
        help="评测模式：manual 为内置规则指标，ragas 为 RAGAS，both 同时运行。",
    )
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条样本。")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="overall_score 低于该阈值时以非 0 状态退出。",
    )
    parser.add_argument(
        "--hash-embedding",
        action="store_true",
        help="查询时使用哈希 embedding；适合开发索引。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭逐题生成预测的进度显示。",
    )
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness,answer_relevancy,context_precision",
        help="RAGAS 指标名，逗号分隔。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 打印 summary。")
    return parser.parse_args()


def main() -> None:
    """CLI 主入口。"""
    args = parse_args()
    cfg = get_config()
    vectorstore = (
        FireAgentQdrantClient(config=cfg, allow_hash_dense_fallback=True)
        if args.hash_embedding and not args.predictions
        else None
    )
    runner = RAGEvaluationRunner(config=cfg, vectorstore=vectorstore)
    summary = runner.run(
        dataset_path=args.dataset or cfg.evaluation.dataset_path,
        output_dir=args.output_dir or cfg.evaluation.output_dir,
        mode=args.mode or cfg.evaluation.default_mode,
        predictions_path=args.predictions,
        limit=args.limit,
        fail_under=args.fail_under if args.fail_under is not None else cfg.evaluation.fail_under,
        ragas_metrics=[metric.strip() for metric in args.ragas_metrics.split(",") if metric.strip()],
        show_progress=not args.no_progress,
    )

    if args.json:
        print(summary.model_dump_json(indent=2))
    else:
        print(f"评测样本数：{summary.total_cases}")
        print(f"输出目录：{summary.run_dir}")
        print("平均分：")
        for name, score in summary.average_scores.items():
            print(f"- {name}: {score:.4f}")
        if summary.errors:
            print("提示：")
            for error in summary.errors:
                print(f"- {error}")
        print(f"是否通过：{'是' if summary.passed else '否'}")

    if not summary.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
