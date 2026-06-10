"""FireAgent 向量库集成。"""

from fireagent.vectorstore.dense_embedding import (
    BaseDenseEmbedder,
    HashingDenseEmbedder,
    OllamaDenseEmbedder,
    SentenceTransformerDenseEmbedder,
    create_dense_embedder,
)
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient
from fireagent.vectorstore.schema import (
    HybridSearchResultSet,
    QdrantCollectionSchema,
    SparseVectorData,
    VectorSearchResult,
    VectorStoreError,
)
from fireagent.vectorstore.sparse_embedding import (
    BM25SparseEncoder,
    BaseSparseEncoder,
    LocalBM25SparseIndex,
)

__all__ = [
    "BM25SparseEncoder",
    "BaseDenseEmbedder",
    "BaseSparseEncoder",
    "FireAgentQdrantClient",
    "HashingDenseEmbedder",
    "HybridSearchResultSet",
    "LocalBM25SparseIndex",
    "OllamaDenseEmbedder",
    "QdrantCollectionSchema",
    "SentenceTransformerDenseEmbedder",
    "SparseVectorData",
    "VectorSearchResult",
    "VectorStoreError",
    "create_dense_embedder",
]
