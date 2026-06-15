"""入库前 section/parent 审计脚本测试。"""

from __future__ import annotations

from fireagent.ingestion.schema import Section
from fireagent.ingestion.semantic_chunker import RecursiveSemanticChunker
from scripts.audit_section_parent_chunks import audit_sections


def make_section(
    section_id: str,
    title: str,
    text: str,
    chunk_type: str = "section",
) -> Section:
    """构造测试用 section。"""
    return Section(
        section_id=section_id,
        doc_id="doc-1",
        title=title,
        path=[title],
        page_start=1,
        page_end=2,
        text=text,
        chunk_type=chunk_type,
    )


def test_audit_sections_reports_parent_and_child_distribution() -> None:
    """审计应统计 section、parent、child 的层级分布。"""
    chunker = RecursiveSemanticChunker(
        chunk_size=50,
        chunk_overlap=0,
        parent_chunk_size=120,
        separators=[""],
    )
    sections = [
        make_section("s1", "1 绪论", "火灾烟气控制" * 10),
        make_section("s2", "2 方法", "隧道火灾疏散安全" * 35),
    ]

    audit = audit_sections("paper.pdf", sections, chunker)

    assert audit["file_name"] == "paper.pdf"
    assert audit["section_count"] == 2
    assert audit["parent_count"] > 2
    assert audit["child_count"] >= audit["parent_count"]
    assert audit["section_length"]["max"] > audit["section_length"]["min"]
    assert 0 < audit["section_within_parent_size_ratio"] < 1
    assert audit["parents_per_section"]["max"] > 1
    assert audit["children_per_parent"]["max"] >= 1


def test_audit_sections_flags_short_children_and_suspicious_headings() -> None:
    """审计应抽样极短 chunk 和可疑标题。"""
    chunker = RecursiveSemanticChunker(
        chunk_size=80,
        chunk_overlap=0,
        parent_chunk_size=160,
        separators=[""],
    )
    sections = [
        make_section("s1", "０娜《、 〇獅 ． 娜", "火灾"),
        make_section("s2", "参考文献", "火灾检测论文标题。" * 20, chunk_type="references"),
    ]

    audit = audit_sections("paper.pdf", sections, chunker, short_chunk_threshold=10)

    assert audit["chunk_type_counts"]["section"] == 1
    assert audit["chunk_type_counts"]["references"] == 1
    assert audit["short_child_count"] >= 1
    assert audit["short_child_samples"]
    assert audit["suspicious_heading_count"] == 1
    assert "０娜" in audit["suspicious_heading_samples"][0]["title"]
