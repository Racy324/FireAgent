"""引用标记解析工具测试。"""

from __future__ import annotations

from fireagent.retrieval.citation_utils import (
    build_used_citation_result,
    citation_to_string,
    extract_used_citation_markers,
    filter_used_citations,
)
from fireagent.retrieval.schema import StructuredCitation


# ── extract_used_citation_markers ──


def test_extract_markers_basic() -> None:
    answer = "烟气会降低能见度 [L1]。一氧化碳有毒 [W2]。"
    assert extract_used_citation_markers(answer) == ["[L1]", "[W2]"]


def test_extract_markers_dedupes_in_order() -> None:
    answer = "结论一 [L1]。结论二 [W2]。再次说明 [L1]。"
    assert extract_used_citation_markers(answer) == ["[L1]", "[W2]"]


def test_extract_markers_ignores_non_evidence_brackets() -> None:
    answer = "参考 [图1] 和 [表2]，实际证据 [L3]。"
    assert extract_used_citation_markers(answer) == ["[L3]"]


def test_extract_markers_empty_answer() -> None:
    assert extract_used_citation_markers("") == []
    assert extract_used_citation_markers(None) == []  # type: ignore[arg-type]


def test_extract_markers_multiple_local_and_web() -> None:
    answer = "[L1] [L2] [W1] [L3] [W2]"
    assert extract_used_citation_markers(answer) == ["[L1]", "[L2]", "[W1]", "[L3]", "[W2]"]


# ── filter_used_citations ──


def test_filter_used_citations_only_keeps_answer_markers() -> None:
    candidates = [
        StructuredCitation(citation_id="L1", marker="[L1]", source_type="local_pdf", title="A"),
        StructuredCitation(citation_id="L2", marker="[L2]", source_type="local_pdf", title="B"),
        StructuredCitation(citation_id="W1", marker="[W1]", source_type="web_tavily", title="C"),
    ]
    used = filter_used_citations(candidates, ["[L2]"])
    assert [item.marker for item in used] == ["[L2]"]


def test_filter_used_citations_empty_markers() -> None:
    candidates = [
        StructuredCitation(citation_id="L1", marker="[L1]", source_type="local_pdf", title="A"),
    ]
    assert filter_used_citations(candidates, []) == []


def test_filter_used_citations_empty_candidates() -> None:
    assert filter_used_citations([], ["[L1]"]) == []


# ── citation_to_string ──


def test_citation_to_string_local() -> None:
    citation = StructuredCitation(
        citation_id="L1",
        marker="[L1]",
        source_type="local_pdf",
        title="隧道火灾烟气控制研究",
        authors=["张三"],
        year=2024,
        section_title="烟气控制",
        page_start=1,
        page_end=2,
    )
    result = citation_to_string(citation)
    assert "[L1]" in result
    assert "隧道火灾烟气控制研究" in result
    assert "张三" in result
    assert "2024" in result
    assert "第1-2页" in result


def test_citation_to_string_web() -> None:
    citation = StructuredCitation(
        citation_id="W1",
        marker="[W1]",
        source_type="web_tavily",
        title="现行消防技术标准",
        url="https://example.com/std",
    )
    result = citation_to_string(citation)
    assert "[W1]" in result
    assert "现行消防技术标准" in result
    assert "https://example.com/std" in result


# ── build_used_citation_result ──


def test_build_used_citation_result_integration() -> None:
    candidates = [
        StructuredCitation(
            citation_id="L1", marker="[L1]", source_type="local_pdf",
            title="论文A", authors=["作者A"], year=2024, section_title="章节1", page_start=1,
        ),
        StructuredCitation(
            citation_id="L2", marker="[L2]", source_type="local_pdf",
            title="论文B",
        ),
        StructuredCitation(
            citation_id="W1", marker="[W1]", source_type="web_tavily",
            title="网页C", url="https://example.com",
        ),
    ]
    answer = "烟气有毒 [L1]。规范要求 [W1]。"

    markers, used, used_strings, invalid = build_used_citation_result(answer, candidates)

    assert markers == ["[L1]", "[W1]"]
    assert len(used) == 2
    assert used[0].marker == "[L1]"
    assert used[1].marker == "[W1]"
    assert len(used_strings) == 2
    assert invalid == []


def test_build_used_citation_result_detects_invalid_markers() -> None:
    candidates = [
        StructuredCitation(citation_id="L1", marker="[L1]", source_type="local_pdf", title="A"),
    ]
    answer = "结论 [L1] [L99] [W1]。"

    markers, used, used_strings, invalid = build_used_citation_result(answer, candidates)

    assert "[L1]" in markers
    assert "[L99]" in markers
    assert "[W1]" in markers
    assert len(used) == 1
    assert "[L99]" in invalid
    assert "[W1]" in invalid
