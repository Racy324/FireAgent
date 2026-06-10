"""重建 Qdrant collection 并重新入库 PDF。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from scripts.ingest_pdfs import ingest_pdfs, print_summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="重建 FireAgent Qdrant 索引。")
    parser.add_argument("--input-dir", default="data/raw_pdfs", help="PDF 输入目录。")
    parser.add_argument("--pdf", action="append", default=[], help="指定单个 PDF，可重复传入。")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理多少个 PDF。")
    parser.add_argument("--max-pages", type=int, default=None, help="每个 PDF 最多解析多少页。")
    parser.add_argument("--batch-size", type=int, default=64, help="Qdrant upsert 批大小。")
    parser.add_argument("--hash-embedding", action="store_true", help="使用哈希 embedding 做开发验证。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出统计结果。")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = ingest_pdfs(
        input_dir=args.input_dir,
        pdf_paths=args.pdf,
        max_files=args.max_files,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
        recreate=True,
        dry_run=False,
        hash_embedding=args.hash_embedding,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)
    if result["failed_files"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
