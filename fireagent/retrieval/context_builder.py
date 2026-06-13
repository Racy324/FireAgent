"""最终回答上下文构建模块。"""

from __future__ import annotations

import re
import hashlib
from collections import defaultdict
from typing import Optional

from fireagent.retrieval.schema import (
    ContextBuildResult,
    EvidenceItem,
    RerankedRetrievalResult,
    StructuredCitation,
)
from fireagent.utils.config import FireAgentConfig, get_config


class ContextBuilder:
    """对重排结果进行去重、parent 回填、相邻 chunk 合并和上下文压缩。"""

    def __init__(
        self,
        max_context_chars: Optional[int] = None,
        max_evidence_chars: Optional[int] = None,
        config: FireAgentConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.max_context_chars = max_context_chars or self.config.retrieval.context_max_chars
        self.max_evidence_chars = max_evidence_chars or self.config.retrieval.context_max_evidence_chars

    def build(
        self,
        candidates: list[RerankedRetrievalResult],
        query: str = "",
    ) -> ContextBuildResult:
        """构建最终上下文字符串和引用列表。"""
        evidence = [self._candidate_to_evidence(candidate) for candidate in candidates]
        evidence = self._dedupe(evidence)
        evidence = self._merge_adjacent(evidence)
        evidence = sorted(evidence, key=lambda item: item.score, reverse=True)

        selected: list[EvidenceItem] = []
        context_parts: list[str] = []
        total_chars = 0
        local_index = 0
        web_index = 0

        for item in evidence:
            text = self._compress_text(item.text, self.max_evidence_chars)
            if self._is_web_source(item.source_type):
                web_index += 1
                citation_id = f"W{web_index}"
            else:
                local_index += 1
                citation_id = f"L{local_index}"
            item.evidence_id = citation_id
            item.text = text
            item.citation = self._build_citation(item)

            block = self._format_evidence_block(item)
            if total_chars + len(block) > self.max_context_chars and selected:
                break
            context_parts.append(block)
            total_chars += len(block)
            selected.append(item)

        candidate_citations = [self._build_structured_citation(item) for item in selected]
        candidate_citation_strings = [self._citation_to_string(sc) for sc in candidate_citations]

        return ContextBuildResult(
            final_context="\n\n".join(context_parts),
            evidence_items=selected,
            citations=candidate_citation_strings,
            candidate_citations=candidate_citations,
            total_chars=total_chars,
        )

    def _candidate_to_evidence(self, candidate: RerankedRetrievalResult) -> EvidenceItem:
        """将重排候选转换为证据项。"""
        payload = candidate.payload or {}
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        parent_text = payload.get("parent_text") or metadata.get("parent_text")
        text = str(parent_text or candidate.text or payload.get("text", ""))
        return EvidenceItem(
            evidence_id="",
            chunk_id=candidate.chunk_id,
            parent_id=str(payload.get("parent_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            paper_title=str(payload.get("paper_title", "")),
            authors=[str(author) for author in payload.get("authors", []) or []],
            year=payload.get("year"),
            section_title=str(payload.get("section_title", "")),
            section_path=[str(part) for part in payload.get("section_path", []) or []],
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            text=text,
            source_type=str(payload.get("source_type", "local_pdf")),
            score=candidate.final_score,
            metadata={**metadata, "fusion_score": candidate.fusion_score, "reranker": candidate.reranker_name},
        )

    def _dedupe(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        """按 chunk_id 和文本指纹去重，保留分数更高的证据。"""
        best_by_key: dict[str, EvidenceItem] = {}
        for item in evidence:
            normalized_text = self._normalize_text(item.text)
            key = item.chunk_id or normalized_text[:160]
            text_key = f"text:{hashlib.sha1(normalized_text.encode('utf-8')).hexdigest()}"
            for dedupe_key in (key, text_key):
                current = best_by_key.get(dedupe_key)
                if current is None or item.score > current.score:
                    best_by_key[dedupe_key] = item

        unique: dict[str, EvidenceItem] = {}
        for item in best_by_key.values():
            unique[item.chunk_id or self._normalize_text(item.text)[:160]] = item
        return list(unique.values())

    def _merge_adjacent(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        """合并同一 parent 下页码相邻的证据，降低上下文碎片化。"""
        grouped: dict[tuple[str, str, str], list[EvidenceItem]] = defaultdict(list)
        for item in evidence:
            key = (item.source_type, item.doc_id, item.parent_id or item.section_title)
            grouped[key].append(item)

        merged: list[EvidenceItem] = []
        for items in grouped.values():
            items.sort(key=lambda item: (item.page_start or 0, item.page_end or 0, -item.score))
            current: EvidenceItem | None = None
            for item in items:
                if current is None:
                    current = item
                    continue
                if self._can_merge(current, item):
                    current.text = self._join_text(current.text, item.text)
                    current.page_start = min(current.page_start or 0, item.page_start or current.page_start or 0)
                    current.page_end = max(current.page_end or 0, item.page_end or current.page_end or 0)
                    current.score = max(current.score, item.score)
                    current.metadata.setdefault("merged_chunk_ids", []).append(item.chunk_id)
                else:
                    merged.append(current)
                    current = item
            if current is not None:
                merged.append(current)
        return merged

    def _can_merge(self, left: EvidenceItem, right: EvidenceItem) -> bool:
        """判断两个证据是否属于同一上下文窗口。"""
        if left.source_type != right.source_type or left.doc_id != right.doc_id:
            return False
        if (left.parent_id or left.section_title) != (right.parent_id or right.section_title):
            return False
        if len(left.text) + len(right.text) > self.max_evidence_chars:
            return False
        left_end = left.page_end or left.page_start or 0
        right_start = right.page_start or right.page_end or 0
        return right_start <= left_end + 1

    @staticmethod
    def _join_text(left: str, right: str) -> str:
        """拼接两个证据文本，避免简单重复。"""
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right or right in left:
            return left
        return f"{left}\n{right}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        """用于去重的文本规范化。"""
        return re.sub(r"\s+", "", text).strip()

    @staticmethod
    def _compress_text(text: str, max_chars: int) -> str:
        """压缩单条证据文本，保留开头和结尾。"""
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        if len(text) <= max_chars:
            return text
        head_len = int(max_chars * 0.72)
        tail_len = max_chars - head_len - 8
        return f"{text[:head_len].rstrip()}\n...\n{text[-tail_len:].lstrip()}"

    def _format_evidence_block(self, item: EvidenceItem) -> str:
        """格式化上下文中的单条证据。"""
        page_text = self._page_text(item.page_start, item.page_end)
        source = self._source_label(item)
        return (
            f"[{item.evidence_id}]\n"
            f"来源：{source}\n"
            f"章节：{item.section_title or '未知'}；页码：{page_text}；分数：{item.score:.4f}\n"
            f"正文：{item.text}"
        )

    def _source_label(self, item: EvidenceItem) -> str:
        """生成来源描述文本。"""
        if self._is_web_source(item.source_type):
            title = item.paper_title or item.metadata.get("title") or item.metadata.get("url") or "联网资料"
            url = item.metadata.get("url", "")
            return f"{title}{f' ({url})' if url else ''}"

        author_text = "、".join(item.authors) if item.authors else "作者未知"
        year_text = str(item.year) if item.year else "年份未知"
        page_text = self._page_text(item.page_start, item.page_end)
        return (
            f"{item.paper_title or '题名未知'}，{author_text}，"
            f"{year_text}，{item.section_title or '章节未知'}，{page_text}"
        )

    def _build_citation(self, item: EvidenceItem) -> str:
        """生成引用描述，回答阶段可直接展示。"""
        if self._is_web_source(item.source_type):
            title = item.paper_title or item.metadata.get("title") or item.metadata.get("url") or "联网资料"
            url = item.metadata.get("url", "")
            return f"{item.evidence_id}: {title}{f' ({url})' if url else ''}"

        author_text = "、".join(item.authors) if item.authors else "作者未知"
        year_text = str(item.year) if item.year else "年份未知"
        page_text = self._page_text(item.page_start, item.page_end)
        return (
            f"{item.evidence_id}: {item.paper_title or '题名未知'}，{author_text}，"
            f"{year_text}，{item.section_title or '章节未知'}，{page_text}"
        )

    def _build_structured_citation(self, item: EvidenceItem) -> StructuredCitation:
        """从 EvidenceItem 构建结构化引用。"""
        is_web = self._is_web_source(item.source_type)
        title = (
            item.paper_title
            or item.metadata.get("title")
            or item.metadata.get("url")
            or ("联网资料" if is_web else "题名未知")
        )
        return StructuredCitation(
            citation_id=item.evidence_id,
            marker=f"[{item.evidence_id}]",
            source_type=item.source_type,
            title=title,
            authors=item.authors,
            year=item.year,
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
            parent_id=item.parent_id,
            section_title=item.section_title,
            section_path=item.section_path,
            page_start=item.page_start,
            page_end=item.page_end,
            url=str(item.metadata.get("url", "")),
            score=item.score,
            text_preview=item.text[:300],
            metadata=item.metadata,
        )

    @staticmethod
    def _citation_to_string(citation: StructuredCitation) -> str:
        """将结构化引用转为兼容字符串。"""
        if citation.source_type.lower().startswith("web"):
            suffix = f" ({citation.url})" if citation.url else ""
            return f"{citation.marker}: {citation.title}{suffix}"

        author_text = "、".join(citation.authors) if citation.authors else "作者未知"
        year_text = str(citation.year) if citation.year else "年份未知"
        page_text = ContextBuilder._page_text(citation.page_start, citation.page_end)
        return (
            f"{citation.marker}: {citation.title or '题名未知'}，{author_text}，"
            f"{year_text}，{citation.section_title or '章节未知'}，{page_text}"
        )

    @staticmethod
    def _page_text(page_start: Optional[int], page_end: Optional[int]) -> str:
        """格式化页码范围。"""
        if page_start and page_end and page_start != page_end:
            return f"第{page_start}-{page_end}页"
        if page_start:
            return f"第{page_start}页"
        return "页码未知"

    @staticmethod
    def _is_web_source(source_type: str) -> bool:
        """判断证据是否来自联网资料。"""
        return source_type.lower().startswith("web") or source_type.lower() in {"tavily", "online"}


def build_context(candidates: list[RerankedRetrievalResult], query: str = "") -> ContextBuildResult:
    """便捷函数：构建默认上下文。"""
    return ContextBuilder().build(candidates, query=query)
