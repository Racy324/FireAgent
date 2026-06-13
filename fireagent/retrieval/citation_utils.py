"""引用标记解析与 used citation 过滤工具。"""

from __future__ import annotations

import re
from typing import Optional

from fireagent.retrieval.schema import StructuredCitation

CITATION_MARKER_RE = re.compile(r"\[(L|W)(\d+)\]")


def extract_used_citation_markers(answer: str) -> list[str]:
    """从模型回答中提取实际使用的引用标记，按出现顺序去重。

    示例:
        >>> extract_used_citation_markers("结论一 [L1]。结论二 [W2]。再次 [L1]。")
        ['[L1]', '[W2]']
    """
    seen: set[str] = set()
    markers: list[str] = []
    for match in CITATION_MARKER_RE.finditer(answer or ""):
        marker = f"[{match.group(1)}{match.group(2)}]"
        if marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return markers


def filter_used_citations(
    candidate_citations: list[StructuredCitation],
    markers: list[str],
) -> list[StructuredCitation]:
    """根据实际出现的 marker 过滤候选引用，只保留回答中使用过的。"""
    marker_set = set(markers)
    return [item for item in candidate_citations if item.marker in marker_set]


def citation_to_string(citation: StructuredCitation) -> str:
    """将结构化引用转为兼容字符串格式。"""
    if citation.source_type.lower().startswith("web"):
        suffix = f" ({citation.url})" if citation.url else ""
        return f"{citation.marker}: {citation.title}{suffix}"

    author_text = "、".join(citation.authors) if citation.authors else "作者未知"
    year_text = str(citation.year) if citation.year else "年份未知"
    page_text = _page_text(citation.page_start, citation.page_end)
    return (
        f"{citation.marker}: {citation.title or '题名未知'}，{author_text}，"
        f"{year_text}，{citation.section_title or '章节未知'}，{page_text}"
    )


def build_used_citation_result(
    answer: str,
    candidate_citations: list[StructuredCitation],
) -> tuple[list[str], list[StructuredCitation], list[str], list[str]]:
    """一站式解析：提取 marker → 过滤 used → 转字符串 → 识别无效 marker。

    Returns:
        (markers, used_citations, used_strings, invalid_markers)
    """
    markers = extract_used_citation_markers(answer)
    used = filter_used_citations(candidate_citations, markers)
    used_strings = [citation_to_string(item) for item in used]
    valid_marker_set = {item.marker for item in candidate_citations}
    invalid_markers = [m for m in markers if m not in valid_marker_set]
    return markers, used, used_strings, invalid_markers


def _page_text(page_start: Optional[int], page_end: Optional[int]) -> str:
    """格式化页码范围。"""
    if page_start and page_end and page_start != page_end:
        return f"第{page_start}-{page_end}页"
    if page_start:
        return f"第{page_start}页"
    return "页码未知"
