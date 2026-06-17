"""RegularRAGCandidateFilter 单元测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fireagent.retrieval.candidate_filter import (
    CandidateFilterStats,
    RegularRAGCandidateFilter,
)
from fireagent.utils.config import load_config, RetrievalConfig


@dataclass
class FakeCandidate:
    """测试用假候选结果。"""

    chunk_id: str = ""
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


def _make_config(**overrides) -> "FireAgentConfig":
    """创建测试用配置。"""
    from fireagent.utils.config import FireAgentConfig

    cfg = load_config()
    retrieval_overrides = {}
    for k, v in overrides.items():
        retrieval_overrides[k] = v
    if retrieval_overrides:
        cfg.retrieval = RetrievalConfig(**{**cfg.retrieval.model_dump(), **retrieval_overrides})
    return cfg


def _candidate(chunk_id: str, chunk_type: str, text: str = "") -> FakeCandidate:
    return FakeCandidate(
        chunk_id=chunk_id,
        text=text,
        payload={"chunk_type": chunk_type, "text": text},
    )


# ── mode=off 测试 ──


def test_mode_off_keeps_all_candidates():
    cfg = _make_config(chunk_type_filter_mode="off")
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "section", "正文证据"),
        _candidate("b", "references", "参考文献标题"),
        _candidate("c", "figure_caption", "图 3-1"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气如何控制？", candidates)

    assert len(kept) == 3
    assert stats.mode == "off"
    assert stats.filtered_references == 0
    assert stats.filtered_short_figure_captions == 0
    assert stats.bypassed is False


# ── 过滤 references 测试 ──


def test_filter_removes_references_when_enabled():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
        min_figure_caption_chars_for_regular_rag=0,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "section", "正文证据"),
        _candidate("b", "references", "参考文献标题"),
        _candidate("c", "section", "另一段正文"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气如何控制？", candidates)

    assert [c.chunk_id for c in kept] == ["a", "c"]
    assert stats.filtered_references == 1
    assert stats.input_count == 3
    assert stats.output_count == 2


def test_filter_keeps_references_when_disabled():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=False,
        min_figure_caption_chars_for_regular_rag=0,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "section", "正文证据"),
        _candidate("b", "references", "参考文献标题"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气如何控制？", candidates)

    assert len(kept) == 2
    assert stats.filtered_references == 0


# ── 过滤极短 figure_caption 测试 ──


def test_filter_removes_short_figure_caption():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=False,
        min_figure_caption_chars_for_regular_rag=50,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "figure_caption", "图 3-1"),
        _candidate("b", "figure_caption", "图 3-2 隧道火灾烟气在纵向通风条件下的温度分布变化规律，包含多个不同通风风速工况下的对比曲线，横轴为时间，纵轴为温度。"),
        _candidate("c", "section", "正文证据"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气温度分布", candidates)

    assert [c.chunk_id for c in kept] == ["b", "c"]
    assert stats.filtered_short_figure_captions == 1


def test_filter_keeps_long_figure_caption():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        min_figure_caption_chars_for_regular_rag=50,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "figure_caption", "图 3-2 隧道火灾烟气在纵向通风条件下的温度分布变化规律，包含多个不同通风风速工况下的对比曲线，横轴为时间，纵轴为温度。"),
    ]

    kept, stats = f.filter_many("温度分布", candidates)

    assert len(kept) == 1
    assert stats.filtered_short_figure_captions == 0


# ── 组合过滤测试 ──


def test_filter_removes_both_references_and_short_caption():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
        min_figure_caption_chars_for_regular_rag=50,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "section", "正文证据"),
        _candidate("b", "references", "参考文献标题"),
        _candidate("c", "figure_caption", "图 3-1"),
        _candidate("d", "figure_caption", "图 3-2 详细描述了隧道火灾场景下温度分布的完整变化趋势，包含多组实验数据的对比分析和详细的讨论说明。"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气如何控制？", candidates)

    assert [c.chunk_id for c in kept] == ["a", "d"]
    assert stats.filtered_references == 1
    assert stats.filtered_short_figure_captions == 1
    assert stats.output_count == 2


# ── 绕过过滤测试 ──


def test_bypass_for_reference_query():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
        min_figure_caption_chars_for_regular_rag=50,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "references", "参考文献标题"),
        _candidate("b", "figure_caption", "图 3-1"),
    ]

    kept, stats = f.filter_many("这篇论文参考文献有哪些？", candidates)

    assert len(kept) == 2
    assert stats.bypassed is True
    assert stats.filtered_references == 0


def test_bypass_for_figure_query():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
        min_figure_caption_chars_for_regular_rag=50,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "figure_caption", "图 3-1"),
    ]

    kept, stats = f.filter_many("图 3-1 说明了什么？", candidates)

    assert len(kept) == 1
    assert stats.bypassed is True


def test_bypass_for_table_query():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "references", "参考文献标题"),
    ]

    kept, stats = f.filter_many("表格数据是什么？", candidates)

    assert len(kept) == 1
    assert stats.bypassed is True


def test_no_bypass_for_normal_query():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        _candidate("a", "references", "参考文献标题"),
    ]

    kept, stats = f.filter_many("隧道火灾烟气如何控制？", candidates)

    assert len(kept) == 0
    assert stats.bypassed is False


# ── should_bypass 单独测试 ──


def test_should_bypass_case_insensitive():
    cfg = _make_config(chunk_type_filter_mode="regular_rag")
    f = RegularRAGCandidateFilter(config=cfg)
    assert f.should_bypass("References for this paper") is True
    assert f.should_bypass("FIGURE 3 shows") is True
    assert f.should_bypass("隧道火灾烟气") is False


# ── 空候选测试 ──


def test_filter_empty_candidates():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
    )
    f = RegularRAGCandidateFilter(config=cfg)

    kept, stats = f.filter_many("测试问题", [])

    assert kept == []
    assert stats.input_count == 0
    assert stats.output_count == 0


# ── chunk_type 缺失测试 ──


def test_filter_handles_missing_chunk_type():
    cfg = _make_config(
        chunk_type_filter_mode="regular_rag",
        filter_references_for_regular_rag=True,
    )
    f = RegularRAGCandidateFilter(config=cfg)
    candidates = [
        FakeCandidate(chunk_id="a", text="正文", payload={}),
        FakeCandidate(chunk_id="b", text="正文", payload={"chunk_type": ""}),
    ]

    kept, stats = f.filter_many("测试问题", candidates)

    assert len(kept) == 2
    assert stats.filtered_references == 0
