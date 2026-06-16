"""将 RAG 评估集转换为 intent router 评估集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SEED60_INTENT_MAP: dict[str, str] = {
    "rag": "rag",
    "emergency": "emergency",
    "paper": "paper",
    "web_fallback": "rag",
    "chat": "chat",
    "reject": "reject",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON") from exc
    return rows


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 question/intent 格式转换为 query/expected_intent 格式。"""
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("question", "") or "").strip()
        if not question:
            case_id = row.get("case_id", f"row-{index}")
            raise ValueError(f"{case_id} missing question")

        source_intent = str(row.get("intent", "rag") or "rag")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        expected_intent = str(
            metadata.get("router_expected_intent")
            or SEED60_INTENT_MAP.get(source_intent, "rag")
        )
        converted_row = {
            "query": question,
            "expected_intent": expected_intent,
            "source_intent": source_intent,
            "source_case_id": row.get("case_id", ""),
            "source": metadata.get("router_eval_source", "seed60_converted"),
            "question_type": metadata.get("question_type", ""),
            "tags": row.get("tags", []),
            "expected_behavior": row.get("expected_behavior", {}),
            "notes": "converted from fireagent_benchmark_v1_seed60",
        }
        router_field_map = {
            "router_expected_sub_intent": "expected_sub_intent",
            "router_expected_need_rag": "expected_need_rag",
            "router_expected_need_web_search": "expected_need_web_search",
            "router_expected_need_memory": "expected_need_memory",
            "router_expected_need_safety_notice": "expected_need_safety_notice",
        }
        for metadata_key, output_key in router_field_map.items():
            if metadata_key in metadata:
                converted_row[output_key] = metadata[metadata_key]
        converted.append(converted_row)
    return converted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Convert RAG eval JSONL to router eval JSONL.")
    parser.add_argument("--input", required=True, help="Input RAG eval JSONL path.")
    parser.add_argument("--output", required=True, help="Output router eval JSONL path.")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 入口。"""
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_jsonl(input_path)
    converted = convert_rows(rows)
    write_jsonl(output_path, converted)
    summary = {
        "input_rows": len(rows),
        "output_rows": len(converted),
        "output": str(output_path),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(f"输入样本数：{summary['input_rows']}")
        print(f"输出样本数：{summary['output_rows']}")
        print(f"输出文件：{summary['output']}")


if __name__ == "__main__":
    main(sys.argv[1:])
