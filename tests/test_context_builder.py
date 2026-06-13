"""上下文构建测试。"""

from __future__ import annotations

from fireagent.retrieval.context_builder import ContextBuilder
from fireagent.retrieval.schema import RerankedRetrievalResult


def local_result(
    chunk_id: str,
    parent_id: str,
    page: int,
    text: str,
    score: float,
) -> RerankedRetrievalResult:
    """构造本地论文重排结果。"""
    return RerankedRetrievalResult(
        chunk_id=chunk_id,
        text=text,
        payload={
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "doc_id": "doc-1",
            "paper_title": "隧道火灾烟气控制研究",
            "authors": ["张三"],
            "year": 2024,
            "section_title": "烟气控制",
            "section_path": ["烟气控制"],
            "page_start": page,
            "page_end": page,
            "text": text,
            "source_type": "local_pdf",
        },
        fusion_score=0.02,
        rerank_score=score,
        final_score=score,
        rank=1,
        reranker_name="test",
    )


def web_result(chunk_id: str, text: str, score: float) -> RerankedRetrievalResult:
    """构造联网资料重排结果。"""
    return RerankedRetrievalResult(
        chunk_id=chunk_id,
        text=text,
        payload={
            "chunk_id": chunk_id,
            "parent_id": "web-doc",
            "doc_id": "web-doc",
            "paper_title": "现行消防技术标准",
            "section_title": "联网资料",
            "section_path": ["联网资料"],
            "text": text,
            "source_type": "web_tavily",
            "metadata": {"url": "https://example.com/std", "title": "现行消防技术标准"},
        },
        fusion_score=0.01,
        rerank_score=score,
        final_score=score,
        rank=1,
        reranker_name="test",
    )


def test_context_builder_dedupes_and_merges_adjacent_parent_chunks() -> None:
    """同一 parent 的相邻本地 chunk 应合并，重复 chunk 应去重。"""
    candidates = [
        local_result("c1", "p1", 1, "隧道火灾烟气会降低能见度。", 0.90),
        local_result("c2", "p1", 2, "烟气中的一氧化碳会影响人员疏散安全。", 0.80),
        local_result("c1", "p1", 1, "隧道火灾烟气会降低能见度。", 0.70),
    ]

    result = ContextBuilder(max_context_chars=2000, max_evidence_chars=300).build(candidates)

    assert len(result.evidence_items) == 1
    evidence = result.evidence_items[0]
    assert evidence.evidence_id == "L1"
    assert evidence.page_start == 1
    assert evidence.page_end == 2
    assert "降低能见度" in evidence.text
    assert "一氧化碳" in evidence.text
    assert "[L1]" in result.final_context
    assert result.candidate_citations[0].marker == "[L1]"
    assert result.citations[0].startswith("[L1]")


def test_context_builder_separates_local_and_web_evidence() -> None:
    """上下文应区分本地论文证据和联网资料证据。"""
    candidates = [
        web_result("w1", "联网资料说明现行规范关注疏散距离和安全出口。", 0.95),
        local_result("c1", "p1", 5, "论文讨论了火灾疏散时间和风险评估。", 0.80),
    ]

    result = ContextBuilder(max_context_chars=2000, max_evidence_chars=300).build(candidates)

    evidence_ids = [item.evidence_id for item in result.evidence_items]
    assert "W1" in evidence_ids
    assert "L1" in evidence_ids
    assert any("https://example.com/std" in citation for citation in result.citations)
    assert "[W1]" in result.final_context
    assert "[L1]" in result.final_context
    assert result.candidate_citations[0].marker == "[W1]"
    assert result.candidate_citations[1].marker == "[L1]"


def test_context_builder_uses_parent_text_when_available() -> None:
    """如果 payload 中提供 parent_text，应优先作为证据上下文。"""
    candidate = local_result("c1", "p1", 1, "子 chunk 文本。", 0.90)
    candidate.payload["metadata"] = {"parent_text": "父级上下文包含更完整的火灾烟气控制内容。"}

    result = ContextBuilder(max_context_chars=1200, max_evidence_chars=500).build([candidate])

    assert "父级上下文" in result.evidence_items[0].text
    assert "子 chunk 文本" not in result.evidence_items[0].text

