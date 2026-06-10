"""语义切块模块测试。"""

from __future__ import annotations

import pytest

from fireagent.ingestion.schema import Section
from fireagent.ingestion.semantic_chunker import RecursiveSemanticChunker


def make_section(text: str) -> Section:
    """构造测试用 section。"""
    return Section(
        section_id="section-1",
        doc_id="doc-1",
        title="1 测试章节",
        path=["1 测试章节"],
        page_start=1,
        page_end=2,
        text=text,
    )


def test_chunker_does_not_drop_text_without_overlap() -> None:
    """无 overlap 时，切块拼接后不应丢失正文字符。"""
    text = "火灾烟气控制" * 80
    chunker = RecursiveSemanticChunker(
        chunk_size=50,
        chunk_overlap=0,
        parent_chunk_size=120,
        separators=[""],
    )

    chunks = chunker.split_section(make_section(text))

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(0 < len(chunk.text) <= 50 for chunk in chunks)
    assert {chunk.doc_id for chunk in chunks} == {"doc-1"}
    assert {chunk.section_title for chunk in chunks} == {"1 测试章节"}


def test_chunker_respects_size_and_parent_metadata_with_overlap() -> None:
    """有 overlap 时，chunk 长度仍应受控，并保留 parent/child 元数据。"""
    text = "隧道火灾烟气蔓延影响人员疏散安全" * 40
    chunker = RecursiveSemanticChunker(
        chunk_size=64,
        chunk_overlap=12,
        parent_chunk_size=140,
        separators=[""],
    )

    chunks = chunker.split_section(make_section(text))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 64 for chunk in chunks)
    assert all(chunk.parent_id for chunk in chunks)
    assert all("parent_index" in chunk.metadata for chunk in chunks)
    assert all("child_index" in chunk.metadata for chunk in chunks)
    assert len({chunk.parent_id for chunk in chunks}) > 1


def test_chunker_rejects_invalid_sizes() -> None:
    """非法 chunk 参数应快速失败。"""
    with pytest.raises(ValueError):
        RecursiveSemanticChunker(chunk_size=100, chunk_overlap=100, parent_chunk_size=200)

    with pytest.raises(ValueError):
        RecursiveSemanticChunker(chunk_size=100, chunk_overlap=10, parent_chunk_size=50)

