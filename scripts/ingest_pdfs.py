"""批量解析 PDF 并写入 Qdrant。

示例：
    python scripts/ingest_pdfs.py --input-dir data/raw_pdfs --max-files 2 --hash-embedding
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.ingestion import PDFIndexBuilder
from fireagent.utils.config import PROJECT_ROOT, get_config
from fireagent.vectorstore import FireAgentQdrantClient


logger = logging.getLogger("fireagent.ingest")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="解析本地火灾论文 PDF 并写入 Qdrant。")
    parser.add_argument("--input-dir", default="data/raw_pdfs", help="PDF 输入目录。")
    parser.add_argument("--pdf", action="append", default=[], help="指定单个 PDF，可重复传入。")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理多少个 PDF。")
    parser.add_argument("--max-pages", type=int, default=None, help="每个 PDF 最多解析多少页。")
    parser.add_argument(
        "--parser",
        choices=["pdfplumber", "mineru"],
        default=None,
        help="PDF 解析器；默认读取配置 PDF_PARSER。",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Qdrant upsert 批大小。")
    parser.add_argument("--recreate", action="store_true", help="先删除并重建 collection。")
    parser.add_argument("--dry-run", action="store_true", help="只解析和切块，不写入 Qdrant。")
    parser.add_argument(
        "--hash-embedding",
        action="store_true",
        help="使用哈希 embedding，适合无模型环境下开发验证。",
    )
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
        parser_name=args.parser,
        batch_size=args.batch_size,
        recreate=args.recreate,
        dry_run=args.dry_run,
        hash_embedding=args.hash_embedding,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)
    if result["failed_files"] > 0:
        sys.exit(1)


def ingest_pdfs(
    input_dir: str = "data/raw_pdfs",
    pdf_paths: Optional[list[str]] = None,
    max_files: Optional[int] = None,
    max_pages: Optional[int] = None,
    parser_name: Optional[str] = None,
    batch_size: int = 64,
    recreate: bool = False,
    dry_run: bool = False,
    hash_embedding: bool = False,
) -> dict[str, object]:
    """批量解析 PDF，并可选写入 Qdrant。"""
    cfg = get_config()
    paths = collect_pdf_paths(input_dir=input_dir, pdf_paths=pdf_paths or [], max_files=max_files)
    builder = PDFIndexBuilder.from_config(cfg, parser_name=parser_name, max_pages=max_pages)

    vectorstore = None
    if not dry_run:
        vectorstore = FireAgentQdrantClient(
            config=cfg,
            allow_hash_dense_fallback=hash_embedding,
        )
        vectorstore.create_collection(recreate=recreate)

    files: list[dict[str, object]] = []
    total_chunks = 0
    for path in paths:
        logger.info("解析 PDF：%s", path)
        file_result: dict[str, object] = {"path": str(path), "status": "running", "chunks": 0}
        try:
            chunks = builder.build_from_pdf(path)
            file_result["chunks"] = len(chunks)
            total_chunks += len(chunks)
            if vectorstore is not None:
                vectorstore.upsert_chunks(chunks, batch_size=batch_size)
            file_result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - 单文件失败继续处理其他文件。
            logger.exception("PDF 入库失败：%s", path)
            file_result["status"] = "failed"
            file_result["error"] = str(exc)
        files.append(file_result)

    failed = sum(1 for item in files if item["status"] == "failed")
    return {
        "status": "ok" if failed == 0 else "partial_failed",
        "collection": cfg.qdrant.collection,
        "dry_run": dry_run,
        "parsed_files": sum(1 for item in files if item["status"] == "ok"),
        "failed_files": failed,
        "total_chunks": total_chunks,
        "files": files,
    }


def collect_pdf_paths(
    input_dir: str = "data/raw_pdfs",
    pdf_paths: Optional[list[str]] = None,
    max_files: Optional[int] = None,
) -> list[Path]:
    """收集待处理 PDF 路径。"""
    paths: list[Path]
    if pdf_paths:
        paths = [resolve_project_path(path) for path in pdf_paths]
    else:
        directory = resolve_project_path(input_dir)
        paths = sorted(directory.glob("*.pdf"))

    if max_files is not None:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(f"未找到 PDF 文件：{input_dir}")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PDF 文件不存在：{missing[:3]}")
    return paths


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析到项目根目录。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def print_summary(result: dict[str, object]) -> None:
    """打印入库摘要。"""
    print(f"状态：{result['status']}")
    print(f"Collection：{result['collection']}")
    print(f"成功文件数：{result['parsed_files']}")
    print(f"失败文件数：{result['failed_files']}")
    print(f"总 chunk 数：{result['total_chunks']}")
    for item in result["files"]:  # type: ignore[index]
        status = item["status"]  # type: ignore[index]
        chunks = item.get("chunks", 0)  # type: ignore[union-attr]
        path = item["path"]  # type: ignore[index]
        print(f"- [{status}] {chunks} chunks：{path}")


if __name__ == "__main__":
    main()
