"""FireAgent 混合检索、融合、重排和上下文构建模块。"""

from fireagent.retrieval.context_builder import ContextBuilder, build_context
from fireagent.retrieval.dense_retriever import DenseRetriever
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
    SufficiencyResult,
)
from fireagent.retrieval.sparse_retriever import SparseRetriever
from fireagent.retrieval.sufficiency_checker import (
    LocalEvidenceSufficiencyChecker,
    check_local_evidence_sufficiency,
)

__all__ = [
    "BaseReranker",
    "ContextBuildResult",
    "ContextBuilder",
    "CrossEncoderReranker",
    "DenseRetriever",
    "EvidenceItem",
    "FallbackReranker",
    "FlagEmbeddingReranker",
    "FusedRetrievalResult",
    "LexicalReranker",
    "LocalEvidenceSufficiencyChecker",
    "OllamaEmbeddingReranker",
    "QueryRewriteResult",
    "QueryRewriter",
    "RerankedRetrievalResult",
    "SparseRetriever",
    "SufficiencyResult",
    "WeightedRRFFusion",
    "build_context",
    "check_local_evidence_sufficiency",
    "create_reranker",
    "rewrite_query",
    "weighted_rrf",
]
