"""文档级命中指标。

用于 router_manual_added 和 multi_doc 样本，判断目标论文是否出现在检索结果中。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

from fireagent.evaluation.retrieval_metrics import RetrievedChunk


@dataclass
class DocHitGold:
    """文档级命中 gold 信号。"""

    source_files: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)

    @property
    def gold_doc_names(self) -> list[str]:
        """从 source_files / source_documents 提取去重后的文档名。"""
        names: list[str] = []
        seen: set[str] = set()
        for raw in self.source_files + self.source_documents:
            name = _normalize_doc_name(raw)
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names


@dataclass
class DocHitScore:
    """文档级命中评分。"""

    gold_doc_count: int
    hit_at_k: dict[str, float | None] = field(default_factory=dict)
    recall_at_k: dict[str, float | None] = field(default_factory=dict)
    unique_doc_count_at_k: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_doc_hit(
    gold: DocHitGold,
    chunks: list[RetrievedChunk],
    ks: tuple[int, ...] = (5, 10),
) -> DocHitScore:
    """计算文档级命中指标。"""
    gold_names = gold.gold_doc_names
    if not gold_names:
        return DocHitScore(
            gold_doc_count=0,
            hit_at_k={f"doc_hit@{k}": None for k in ks},
            recall_at_k={f"doc_recall@{k}": None for k in ks},
            unique_doc_count_at_k={f"unique_doc_count@{k}": None for k in ks},
        )

    hit_at_k: dict[str, float | None] = {}
    recall_at_k: dict[str, float | None] = {}
    unique_doc_count_at_k: dict[str, int | None] = {}

    for k in ks:
        top_chunks = [c for c in chunks if c.rank <= k]
        matched_docs = _matched_gold_docs(top_chunks, gold_names)
        unique_docs = _unique_doc_ids(top_chunks)

        hit_at_k[f"doc_hit@{k}"] = 1.0 if matched_docs else 0.0
        recall_at_k[f"doc_recall@{k}"] = round(len(matched_docs) / len(gold_names), 4)
        unique_doc_count_at_k[f"unique_doc_count@{k}"] = unique_docs

    return DocHitScore(
        gold_doc_count=len(gold_names),
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        unique_doc_count_at_k=unique_doc_count_at_k,
    )


def _normalize_doc_name(raw: str) -> str:
    """从路径或标题提取规范化文档名。"""
    if not raw:
        return ""
    # 取 basename
    name = PurePosixPath(raw.replace("\\", "/")).stem
    # 去掉常见后缀
    for suffix in (".pdf", ".PDF"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip().lower()


def _matched_gold_docs(
    chunks: list[RetrievedChunk],
    gold_names: list[str],
) -> set[str]:
    """返回 top-k 中命中的 gold 文档名集合。"""
    matched: set[str] = set()
    for chunk in chunks:
        haystack = " ".join(
            part.lower()
            for part in [chunk.doc_id, chunk.paper_title, chunk.text[:500]]
            if part
        )
        for name in gold_names:
            if name in haystack:
                matched.add(name)
    return matched


def _unique_doc_ids(chunks: list[RetrievedChunk]) -> int:
    """返回 top-k 中不同 doc_id 的数量。"""
    return len({c.doc_id for c in chunks if c.doc_id})


def extract_doc_hit_gold_from_metadata(metadata: dict[str, object]) -> DocHitGold:
    """从 case.metadata 中提取 doc-hit gold。"""
    return DocHitGold(
        source_files=list(metadata.get("source_files") or []),
        source_documents=list(metadata.get("source_documents") or []),
    )
