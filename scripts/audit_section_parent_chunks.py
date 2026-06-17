"""入库前 section/parent/child chunk 审计脚本。

该脚本只运行 PDF 解析、清洗、章节切分和递归切块统计，不写入 Qdrant，
也不调用 embedding，用于在正式重建索引前判断 chunk 参数和数据质量。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from fireagent.ingestion.cleaner import PDFTextCleaner
from fireagent.ingestion.index_builder import create_pdf_parser_from_config
from fireagent.ingestion.metadata_extractor import PaperMetadataExtractor
from fireagent.ingestion.schema import ParsedDocument, Section
from fireagent.ingestion.section_splitter import SectionSplitter
from fireagent.ingestion.semantic_chunker import RecursiveSemanticChunker
from fireagent.utils.config import FireAgentConfig, get_config


SUSPICIOUS_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff\ufffd]")
TEXT_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="审计入库前 section/parent/child chunk 分布。")
    parser.add_argument("--input-dir", default="data/raw_pdfs", help="PDF 输入目录。")
    parser.add_argument("--pdf", action="append", default=[], help="指定单个 PDF，可重复传入。")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理多少个 PDF。")
    parser.add_argument("--max-pages", type=int, default=None, help="每个 PDF 最多解析多少页。")
    parser.add_argument("--parser", default=None, help="覆盖配置中的 PDF parser。")
    parser.add_argument("--output-dir", default="data/eval/chunk_audit", help="审计输出目录。")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 CHUNK_SIZE。")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="覆盖 CHUNK_OVERLAP。")
    parser.add_argument("--parent-chunk-size", type=int, default=None, help="覆盖 PARENT_CHUNK_SIZE。")
    parser.add_argument("--short-threshold", type=int, default=100, help="极短 child chunk 阈值。")
    parser.add_argument("--sample-limit", type=int, default=30, help="每类样本最多保留多少条。")
    parser.add_argument("--json", action="store_true", help="在终端输出 JSON summary。")
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    args = parse_args()
    cfg = get_config()
    chunker = RecursiveSemanticChunker(
        chunk_size=args.chunk_size or cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap,
        parent_chunk_size=args.parent_chunk_size or cfg.rag.parent_chunk_size,
    )
    pdf_paths = collect_pdf_paths(args.input_dir, args.pdf, args.max_files)
    result = audit_pdf_paths(
        pdf_paths=pdf_paths,
        config=cfg,
        chunker=chunker,
        parser_name=args.parser,
        max_pages=args.max_pages,
        short_chunk_threshold=args.short_threshold,
        sample_limit=args.sample_limit,
    )
    output_dir = Path(args.output_dir)
    write_outputs(result, output_dir)
    if args.json:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        print_summary(result["summary"], output_dir)

    if result["summary"]["failed_files"] > 0:
        sys.exit(1)


def collect_pdf_paths(input_dir: str | Path, pdf_paths: list[str], max_files: int | None) -> list[Path]:
    """收集待审计 PDF 路径。"""
    if pdf_paths:
        paths = [Path(path) for path in pdf_paths]
    else:
        paths = sorted(Path(input_dir).glob("*.pdf"))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def audit_pdf_paths(
    pdf_paths: list[Path],
    config: FireAgentConfig,
    chunker: RecursiveSemanticChunker,
    parser_name: str | None = None,
    max_pages: int | None = None,
    short_chunk_threshold: int = 100,
    sample_limit: int = 30,
) -> dict[str, Any]:
    """对一组 PDF 执行入库前审计。"""
    parser = create_pdf_parser_from_config(config, parser_name=parser_name, max_pages=max_pages)
    cleaner = PDFTextCleaner()
    metadata_extractor = PaperMetadataExtractor()
    section_splitter = SectionSplitter()

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        try:
            document = parser.parse(pdf_path)
            cleaned = cleaner.clean_document(document)
            metadata = metadata_extractor.extract(cleaned)
            sections = section_splitter.split(cleaned, metadata=metadata)
            audit = audit_sections(
                file_name=Path(cleaned.file_name).name,
                sections=sections,
                chunker=chunker,
                parsed_document=cleaned,
                short_chunk_threshold=short_chunk_threshold,
                sample_limit=sample_limit,
            )
            documents.append(audit)
        except Exception as exc:  # noqa: BLE001 - 审计应继续处理后续文件。
            errors.append({"file_name": str(pdf_path), "error": str(exc)})

    summary = summarize_document_audits(
        documents,
        errors,
        chunker=chunker,
        short_chunk_threshold=short_chunk_threshold,
    )
    return {"summary": summary, "documents": documents, "errors": errors}


def audit_sections(
    file_name: str,
    sections: list[Section],
    chunker: RecursiveSemanticChunker,
    parsed_document: ParsedDocument | None = None,
    short_chunk_threshold: int = 100,
    sample_limit: int = 30,
) -> dict[str, Any]:
    """审计一个文档的 section、parent 和 child chunk 分布。"""
    section_lengths: list[int] = []
    parent_lengths: list[int] = []
    child_lengths: list[int] = []
    parents_per_section: list[int] = []
    children_per_parent: list[int] = []
    chunk_type_counts: Counter[str] = Counter()
    chunk_type_lengths: dict[str, list[int]] = {}
    short_child_samples: list[dict[str, Any]] = []
    suspicious_heading_samples: list[dict[str, Any]] = []
    long_section_samples: list[dict[str, Any]] = []
    suspicious_heading_count = 0

    for section in sections:
        text = chunker._normalize_text(section.text)
        length = len(text)
        section_lengths.append(length)
        chunk_type_counts[section.chunk_type] += 1
        chunk_type_lengths.setdefault(section.chunk_type, []).append(length)

        if is_suspicious_heading(section.title):
            suspicious_heading_count += 1
            add_sample(
                suspicious_heading_samples,
                {
                    "title": section.title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "chunk_type": section.chunk_type,
                    "text_preview": preview_text(text),
                },
                sample_limit,
            )

        if length > chunker.parent_chunk_size:
            add_sample(
                long_section_samples,
                {
                    "title": section.title,
                    "length": length,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "chunk_type": section.chunk_type,
                },
                sample_limit,
            )

        parent_texts = chunker._split_without_overlap(text, chunker.parent_chunk_size)
        parents_per_section.append(len(parent_texts))
        for parent_index, parent_text in enumerate(parent_texts):
            parent_lengths.append(len(parent_text))
            child_texts = chunker._split_with_overlap(parent_text)
            children_per_parent.append(len(child_texts))
            for child_index, child_text in enumerate(child_texts):
                child_len = len(child_text.strip())
                child_lengths.append(child_len)
                if 0 < child_len <= short_chunk_threshold:
                    add_sample(
                        short_child_samples,
                        {
                            "section_title": section.title,
                            "chunk_type": section.chunk_type,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                            "parent_index": parent_index,
                            "child_index": child_index,
                            "length": child_len,
                            "text_preview": preview_text(child_text),
                        },
                        sample_limit,
                    )

    aux_stats = audit_auxiliary_chunks(parsed_document) if parsed_document is not None else {}
    section_count = len(sections)
    within_parent = sum(1 for length in section_lengths if length <= chunker.parent_chunk_size)

    return {
        "file_name": file_name,
        "section_count": section_count,
        "parent_count": len(parent_lengths),
        "child_count": len(child_lengths),
        "section_length": describe_numbers(section_lengths),
        "parent_length": describe_numbers(parent_lengths),
        "child_length": describe_numbers(child_lengths),
        "parents_per_section": describe_numbers(parents_per_section),
        "children_per_parent": describe_numbers(children_per_parent),
        "section_within_parent_size_ratio": round(within_parent / section_count, 4) if section_count else 0.0,
        "long_section_count": sum(1 for length in section_lengths if length > chunker.parent_chunk_size),
        "short_child_count": sum(1 for length in child_lengths if 0 < length <= short_chunk_threshold),
        "chunk_type_counts": dict(chunk_type_counts),
        "chunk_type_length": {
            chunk_type: describe_numbers(lengths)
            for chunk_type, lengths in sorted(chunk_type_lengths.items())
        },
        "_section_lengths": section_lengths,
        "_parent_lengths": parent_lengths,
        "_child_lengths": child_lengths,
        "_parents_per_section": parents_per_section,
        "_children_per_parent": children_per_parent,
        "auxiliary_chunks": aux_stats,
        "short_child_samples": short_child_samples,
        "suspicious_heading_count": suspicious_heading_count,
        "suspicious_heading_samples": suspicious_heading_samples,
        "long_section_samples": long_section_samples,
    }


def audit_auxiliary_chunks(document: ParsedDocument) -> dict[str, Any]:
    """统计表格和图注等非 section 辅助 chunk。"""
    table_lengths = [len(table.text.strip()) for table in document.tables if table.text.strip()]
    caption_lengths = [len(caption.text.strip()) for caption in document.figure_captions if caption.text.strip()]
    return {
        "table_count": len(table_lengths),
        "table_length": describe_numbers(table_lengths),
        "figure_caption_count": len(caption_lengths),
        "figure_caption_length": describe_numbers(caption_lengths),
    }


def summarize_document_audits(
    documents: list[dict[str, Any]],
    errors: list[dict[str, str]],
    chunker: RecursiveSemanticChunker,
    short_chunk_threshold: int,
) -> dict[str, Any]:
    """汇总所有文档审计结果。"""
    section_lengths: list[int] = []
    parent_lengths: list[int] = []
    child_lengths: list[int] = []
    parents_per_section: list[int] = []
    children_per_parent: list[int] = []
    chunk_type_counts: Counter[str] = Counter()
    suspicious_samples: list[dict[str, Any]] = []
    short_samples: list[dict[str, Any]] = []
    long_section_samples: list[dict[str, Any]] = []

    for document in documents:
        section_lengths.extend(document.get("_section_lengths", []))
        parent_lengths.extend(document.get("_parent_lengths", []))
        child_lengths.extend(document.get("_child_lengths", []))
        parents_per_section.extend(document.get("_parents_per_section", []))
        children_per_parent.extend(document.get("_children_per_parent", []))
        chunk_type_counts.update(document["chunk_type_counts"])
        suspicious_samples.extend(add_file_name(document["file_name"], document["suspicious_heading_samples"]))
        short_samples.extend(add_file_name(document["file_name"], document["short_child_samples"]))
        long_section_samples.extend(add_file_name(document["file_name"], document["long_section_samples"]))

    total_sections = sum(document["section_count"] for document in documents)
    total_within_parent = sum(
        round(document["section_within_parent_size_ratio"] * document["section_count"])
        for document in documents
    )
    total_short = sum(document["short_child_count"] for document in documents)
    total_child = sum(document["child_count"] for document in documents)

    return {
        "processed_files": len(documents),
        "failed_files": len(errors),
        "chunk_size": chunker.chunk_size,
        "chunk_overlap": chunker.chunk_overlap,
        "parent_chunk_size": chunker.parent_chunk_size,
        "short_chunk_threshold": short_chunk_threshold,
        "section_count": total_sections,
        "parent_count": sum(document["parent_count"] for document in documents),
        "child_count": total_child,
        "section_length": describe_numbers(section_lengths),
        "parent_length": describe_numbers(parent_lengths),
        "child_length": describe_numbers(child_lengths),
        "parents_per_section": describe_numbers(parents_per_section),
        "children_per_parent": describe_numbers(children_per_parent),
        "section_within_parent_size_ratio": (
            round(total_within_parent / total_sections, 4) if total_sections else 0.0
        ),
        "short_child_count": total_short,
        "short_child_ratio": round(total_short / total_child, 4) if total_child else 0.0,
        "chunk_type_counts": dict(chunk_type_counts),
        "suspicious_heading_count": sum(document["suspicious_heading_count"] for document in documents),
        "suspicious_heading_samples": suspicious_samples[:50],
        "short_child_samples": short_samples[:50],
        "long_section_samples": sorted(long_section_samples, key=lambda item: item.get("length", 0), reverse=True)[:50],
    }


def describe_numbers(values: list[int]) -> dict[str, Any]:
    """返回整数列表的描述性统计。"""
    if not values:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "p10": 0,
            "p25": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "min": min(values),
        "max": max(values),
        "p10": percentile(sorted_values, 10),
        "p25": percentile(sorted_values, 25),
        "p75": percentile(sorted_values, 75),
        "p90": percentile(sorted_values, 90),
        "p95": percentile(sorted_values, 95),
    }


def percentile(sorted_values: list[int], percent: int) -> int:
    """按 nearest-rank 风格返回百分位数。"""
    if not sorted_values:
        return 0
    index = round((percent / 100) * (len(sorted_values) - 1))
    return int(sorted_values[index])


def is_suspicious_heading(title: str) -> bool:
    """判断标题是否疑似 OCR/页眉页脚/公式误识别。"""
    stripped = title.strip()
    if not stripped:
        return True
    if SUSPICIOUS_PRIVATE_USE_RE.search(stripped):
        return True
    text_chars = len(TEXT_CHAR_RE.findall(stripped))
    visible_chars = len(re.sub(r"\s+", "", stripped))
    if visible_chars == 0:
        return True
    non_text_ratio = 1.0 - text_chars / visible_chars
    if visible_chars >= 6 and non_text_ratio >= 0.45:
        return True
    digit_symbol_ratio = len(re.findall(r"[\d０-９._＿\-—－·•、〇○．]", stripped)) / visible_chars
    return visible_chars >= 8 and digit_symbol_ratio >= 0.65


def preview_text(text: str, limit: int = 120) -> str:
    """压缩文本预览。"""
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= limit:
        return preview
    return preview[:limit].rstrip() + "..."


def add_sample(samples: list[dict[str, Any]], sample: dict[str, Any], limit: int) -> None:
    """在限制内追加样本。"""
    if len(samples) < limit:
        samples.append(sample)


def add_file_name(file_name: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给样本补充文件名。"""
    return [{**sample, "file_name": file_name} for sample in samples]


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    """写出 JSON、CSV 和 Markdown 审计结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "section_parent_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_document_csv(output_dir / "section_parent_audit_by_file.csv", result["documents"])
    write_samples_csv(output_dir / "suspicious_headings.csv", result["summary"]["suspicious_heading_samples"])
    write_samples_csv(output_dir / "short_child_samples.csv", result["summary"]["short_child_samples"])
    write_markdown_report(output_dir / "section_parent_audit.md", result)


def write_document_csv(path: Path, documents: list[dict[str, Any]]) -> None:
    """写出按文档聚合的 CSV。"""
    fields = [
        "file_name",
        "section_count",
        "parent_count",
        "child_count",
        "section_median",
        "section_p90",
        "section_max",
        "section_within_parent_size_ratio",
        "parents_per_section_p90",
        "short_child_count",
        "suspicious_heading_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for document in documents:
            writer.writerow(
                {
                    "file_name": document["file_name"],
                    "section_count": document["section_count"],
                    "parent_count": document["parent_count"],
                    "child_count": document["child_count"],
                    "section_median": document["section_length"]["median"],
                    "section_p90": document["section_length"]["p90"],
                    "section_max": document["section_length"]["max"],
                    "section_within_parent_size_ratio": document["section_within_parent_size_ratio"],
                    "parents_per_section_p90": document["parents_per_section"]["p90"],
                    "short_child_count": document["short_child_count"],
                    "suspicious_heading_count": document["suspicious_heading_count"],
                }
            )


def write_samples_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    """写出样本 CSV。"""
    fieldnames = sorted({key for sample in samples for key in sample.keys()})
    if not fieldnames:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    """写出 Markdown 汇总报告。"""
    summary = result["summary"]
    lines = [
        "# Section/Parent Chunk Audit",
        "",
        "## Summary",
        "",
        f"- processed_files: {summary['processed_files']}",
        f"- failed_files: {summary['failed_files']}",
        f"- chunk_size: {summary['chunk_size']}",
        f"- chunk_overlap: {summary['chunk_overlap']}",
        f"- parent_chunk_size: {summary['parent_chunk_size']}",
        f"- section_count: {summary['section_count']}",
        f"- parent_count: {summary['parent_count']}",
        f"- child_count: {summary['child_count']}",
        f"- section_within_parent_size_ratio: {summary['section_within_parent_size_ratio']}",
        f"- short_child_ratio: {summary['short_child_ratio']}",
        f"- suspicious_heading_count: {summary['suspicious_heading_count']}",
        "",
        "## Length Stats",
        "",
        "| target | count | mean | median | p75 | p90 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("section_length", "parent_length", "child_length"):
        stats = summary[name]
        lines.append(
            f"| {name} | {stats['count']} | {stats['mean']} | {stats['median']} | "
            f"{stats['p75']} | {stats['p90']} | {stats['p95']} | {stats['max']} |"
        )
    lines.extend(
        [
            "",
            "## Chunk Type Counts",
            "",
            "| chunk_type | count |",
            "| --- | ---: |",
        ]
    )
    for chunk_type, count in sorted(summary["chunk_type_counts"].items()):
        lines.append(f"| {chunk_type} | {count} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 该报告统计的是入库前 section/parent/child 分布，不写入 Qdrant。",
            "- `section_within_parent_size_ratio` 越高，说明当前 parent_chunk_size 越能覆盖完整 section。",
            "- `short_child_ratio` 和 suspicious heading 用于判断短图注、乱码标题和页眉页脚污染。",
            "",
        ]
    )
    if result["errors"]:
        lines.extend(["## Errors", ""])
        for error in result["errors"]:
            lines.append(f"- {error['file_name']}: {error['error']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """打印简要 summary。"""
    print("=" * 50)
    print("入库前 section/parent/child 审计")
    print("=" * 50)
    print(f"处理文件: {summary['processed_files']}，失败: {summary['failed_files']}")
    print(f"section: {summary['section_count']}，parent: {summary['parent_count']}，child: {summary['child_count']}")
    print(f"section <= parent_chunk_size 比例: {summary['section_within_parent_size_ratio']:.2%}")
    print(f"短 child 比例: {summary['short_child_ratio']:.2%}")
    print(f"可疑 heading: {summary['suspicious_heading_count']}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
