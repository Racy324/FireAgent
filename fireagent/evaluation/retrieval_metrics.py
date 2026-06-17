"""Deterministic retrieval-only metrics for FireAgent evaluation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fireagent.vectorstore.sparse_embedding import tokenize_text


@dataclass
class RetrievalGold:
    """Gold evidence signals available in one evaluation case."""

    required_citation_substrings: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """Ranked chunk used by retrieval-only evaluation output."""

    rank: int
    chunk_id: str
    text: str
    score: float = 0.0
    stage: str = "reranked"
    doc_id: str = ""
    parent_id: str = ""
    paper_title: str = ""
    section_title: str = ""
    chunk_type: str = ""
    page_start: int | None = None
    page_end: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    sources: list[str] = field(default_factory=list)
    matched_required_citation_substrings: list[str] = field(default_factory=list)
    matched_reference_context_indices: list[int] = field(default_factory=list)
    text_preview: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable chunk record."""
        payload = asdict(self)
        if not payload["text_preview"]:
            payload["text_preview"] = self.text[:300]
        return payload


@dataclass
class RetrievalScore:
    """Retrieval-only score for one ranked result list."""

    gold_signal_count: int
    retrieved_count: int
    relevant_retrieved_count: int
    hit_at_k: dict[str, float | None]
    recall_at_k: dict[str, float | None]
    precision_at_k: dict[str, float | None]
    mrr_at_k: dict[str, float | None]
    required_citation_hit_rate: float | None
    reference_context_coverage: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable score record."""
        return asdict(self)


def score_retrieved_chunks(
    gold: RetrievalGold,
    chunks: list[RetrievedChunk],
    ks: tuple[int, ...] = (5, 10),
    reference_overlap_threshold: float = 0.75,
) -> RetrievalScore:
    """Score ranked chunks using citation substrings and reference contexts as gold signals."""
    required = [item.strip() for item in gold.required_citation_substrings if item.strip()]
    reference_contexts = [item.strip() for item in gold.reference_contexts if item.strip()]
    gold_signal_count = len(required) + len(reference_contexts)

    for chunk in chunks:
        _annotate_chunk_matches(
            chunk,
            required=required,
            reference_contexts=reference_contexts,
            reference_overlap_threshold=reference_overlap_threshold,
        )

    if gold_signal_count == 0:
        return RetrievalScore(
            gold_signal_count=0,
            retrieved_count=len(chunks),
            relevant_retrieved_count=0,
            hit_at_k={f"hit@{k}": None for k in ks},
            recall_at_k={f"recall@{k}": None for k in ks},
            precision_at_k={f"precision@{k}": None for k in ks},
            mrr_at_k={f"mrr@{k}": None for k in ks},
            required_citation_hit_rate=None,
            reference_context_coverage=None,
        )

    relevant_ranks = [
        chunk.rank
        for chunk in chunks
        if chunk.matched_required_citation_substrings or chunk.matched_reference_context_indices
    ]
    relevant_retrieved_count = len(relevant_ranks)
    hit_at_k: dict[str, float | None] = {}
    recall_at_k: dict[str, float | None] = {}
    precision_at_k: dict[str, float | None] = {}
    mrr_at_k: dict[str, float | None] = {}

    for k in ks:
        top_chunks = [chunk for chunk in chunks if chunk.rank <= k]
        covered_required, covered_contexts = _covered_gold(top_chunks)
        covered_count = len(covered_required) + len(covered_contexts)
        relevant_top_count = sum(
            1
            for chunk in top_chunks
            if chunk.matched_required_citation_substrings or chunk.matched_reference_context_indices
        )
        first_relevant_rank = min((rank for rank in relevant_ranks if rank <= k), default=None)

        hit_at_k[f"hit@{k}"] = 1.0 if covered_count > 0 else 0.0
        recall_at_k[f"recall@{k}"] = round(covered_count / gold_signal_count, 4)
        precision_at_k[f"precision@{k}"] = (
            round(relevant_top_count / len(top_chunks), 4) if top_chunks else 0.0
        )
        mrr_at_k[f"mrr@{k}"] = round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0

    covered_required_all, covered_contexts_all = _covered_gold(chunks)
    required_hit_rate = (
        round(len(covered_required_all) / len(required), 4) if required else None
    )
    context_coverage = (
        round(len(covered_contexts_all) / len(reference_contexts), 4) if reference_contexts else None
    )

    return RetrievalScore(
        gold_signal_count=gold_signal_count,
        retrieved_count=len(chunks),
        relevant_retrieved_count=relevant_retrieved_count,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mrr_at_k=mrr_at_k,
        required_citation_hit_rate=required_hit_rate,
        reference_context_coverage=context_coverage,
    )


def _annotate_chunk_matches(
    chunk: RetrievedChunk,
    required: list[str],
    reference_contexts: list[str],
    reference_overlap_threshold: float,
) -> None:
    text = chunk.text or ""
    haystack = " ".join(
        part
        for part in [
            text,
            chunk.paper_title,
            chunk.section_title,
            chunk.doc_id,
            chunk.chunk_type,
        ]
        if part
    ).lower()

    chunk.matched_required_citation_substrings = [
        item for item in required if item.lower() in haystack
    ]
    chunk.matched_reference_context_indices = [
        index
        for index, reference_context in enumerate(reference_contexts)
        if _context_matches(text, reference_context, threshold=reference_overlap_threshold)
    ]


def _context_matches(text: str, reference_context: str, threshold: float) -> bool:
    if not text or not reference_context:
        return False
    if reference_context in text or text in reference_context:
        return True
    text_terms = set(tokenize_text(text))
    reference_terms = set(tokenize_text(reference_context))
    if not text_terms or not reference_terms:
        return False
    overlap = text_terms & reference_terms
    return len(overlap) / max(len(reference_terms), 1) >= threshold


def _covered_gold(chunks: list[RetrievedChunk]) -> tuple[set[str], set[int]]:
    covered_required: set[str] = set()
    covered_contexts: set[int] = set()
    for chunk in chunks:
        covered_required.update(chunk.matched_required_citation_substrings)
        covered_contexts.update(chunk.matched_reference_context_indices)
    return covered_required, covered_contexts
