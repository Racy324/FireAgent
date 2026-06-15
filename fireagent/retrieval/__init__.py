"""FireAgent 混合检索、融合、重排和上下文构建模块。"""

from fireagent.retrieval.candidate_filter import CandidateFilterStats, RegularRAGCandidateFilter
from fireagent.retrieval.query_parallel import ParallelQueryStats, merge_query_results, retrieve_queries_parallel
from fireagent.retrieval.context_builder import ContextBuilder, build_context
from fireagent.retrieval.dense_retriever import DenseRetriever
from fireagent.retrieval.fallback_policy import FallbackAction, FallbackDecision, FallbackPolicy, decide_fallback
from fireagent.retrieval.hybrid_fusion import WeightedRRFFusion, weighted_rrf
from fireagent.retrieval.query_rewriter import QueryRewriter, rewrite_query
from fireagent.retrieval.reranker import (
    BaseReranker,
    CrossEncoderReranker,
    FallbackReranker,
    FlagEmbeddingReranker,
    LexicalReranker,
    OllamaEmbeddingReranker,
    create_reranker,
)
from fireagent.retrieval.schema import (
    ContextBuildResult,
    EvidenceItem,
    FusedRetrievalResult,
    QueryRewriteResult,
    RerankedRetrievalResult,
    StructuredCitation,
    SufficiencyResult,
)
from fireagent.retrieval.sparse_retriever import SparseRetriever
from fireagent.retrieval.sufficiency_checker import (
    LocalEvidenceSufficiencyChecker,
    check_local_evidence_sufficiency,
)

__all__ = [
    "BaseReranker",
    "CandidateFilterStats",
    "ContextBuildResult",
    "ContextBuilder",
    "CrossEncoderReranker",
    "DenseRetriever",
    "EvidenceItem",
    "FallbackAction",
    "FallbackDecision",
    "FallbackPolicy",
    "FallbackReranker",
    "FlagEmbeddingReranker",
    "FusedRetrievalResult",
    "LexicalReranker",
    "LocalEvidenceSufficiencyChecker",
    "OllamaEmbeddingReranker",
    "ParallelQueryStats",
    "QueryRewriteResult",
    "QueryRewriter",
    "RerankedRetrievalResult",
    "RegularRAGCandidateFilter",
    "SparseRetriever",
    "StructuredCitation",
    "SufficiencyResult",
    "WeightedRRFFusion",
    "build_context",
    "check_local_evidence_sufficiency",
    "create_reranker",
    "decide_fallback",
    "merge_query_results",
    "retrieve_queries_parallel",
    "rewrite_query",
    "weighted_rrf",
]
