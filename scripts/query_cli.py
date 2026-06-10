"""FireAgent 命令行问答入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.graph.workflow import run_fireagent_workflow
from fireagent.utils.config import get_config
from fireagent.vectorstore import FireAgentQdrantClient


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="FireAgent 命令行问答。")
    parser.add_argument("query", nargs="*", help="用户问题；不传则进入交互模式。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整状态摘要。")
    parser.add_argument("--show-context", action="store_true", help="显示最终上下文。")
    parser.add_argument(
        "--hash-embedding",
        action="store_true",
        help="使用哈希 embedding 查询开发索引；需与入库时保持一致。",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    args = parse_args()
    query = " ".join(args.query).strip()
    if query:
        state = ask(query, hash_embedding=args.hash_embedding)
        print_answer(state, as_json=args.json, show_context=args.show_context)
        return

    interactive_loop(hash_embedding=args.hash_embedding, as_json=args.json, show_context=args.show_context)


def ask(query: str, hash_embedding: bool = False) -> dict[str, object]:
    """执行一次命令行问答。"""
    cfg = get_config()
    vectorstore = (
        FireAgentQdrantClient(config=cfg, allow_hash_dense_fallback=True) if hash_embedding else None
    )
    state = run_fireagent_workflow(query, config=cfg, vectorstore=vectorstore)
    return dict(state)


def interactive_loop(hash_embedding: bool = False, as_json: bool = False, show_context: bool = False) -> None:
    """交互式问答循环。"""
    print("FireAgent CLI，输入 exit 或 quit 退出。")
    while True:
        try:
            query = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"exit", "quit", "q"}:
            return
        if not query:
            continue
        try:
            state = ask(query, hash_embedding=hash_embedding)
            print_answer(state, as_json=as_json, show_context=show_context)
        except Exception as exc:  # noqa: BLE001 - CLI 层打印清晰错误即可。
            print(f"问答失败：{exc}", file=sys.stderr)


def print_answer(state: dict[str, object], as_json: bool = False, show_context: bool = False) -> None:
    """打印问答结果。"""
    if as_json:
        output = {
            "answer": state.get("final_answer", ""),
            "intent": state.get("intent", ""),
            "evidence_sufficient": state.get("evidence_sufficient", False),
            "citations": state.get("citations", []),
            "errors": state.get("errors", []),
        }
        if show_context:
            output["context"] = state.get("final_context", "")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print("\n回答：")
    print(state.get("final_answer", ""))
    citations = state.get("citations", []) or []
    if citations:
        print("\n资料来源：")
        for citation in citations:
            print(f"- {citation}")
    errors = state.get("errors", []) or []
    if errors:
        print("\n运行提示：")
        for error in errors:
            print(f"- {error}")
    if show_context:
        print("\n最终上下文：")
        print(state.get("final_context", ""))


if __name__ == "__main__":
    main()
