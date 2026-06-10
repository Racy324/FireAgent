"""BM25 sparse embedding 与本地稀疏检索 fallback。"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

from fireagent.ingestion.schema import DocumentChunk
from fireagent.vectorstore.schema import SparseVectorData, VectorSearchResult


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]")


class BaseSparseEncoder(ABC):
    """BM25 sparse encoder 抽象接口。"""

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """基于语料统计 IDF 与平均长度。"""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[SparseVectorData]:
        """编码文档稀疏向量。"""

    @abstractmethod
    def encode_query(self, query: str) -> SparseVectorData:
        """编码查询稀疏向量。"""


class BM25SparseEncoder(BaseSparseEncoder):
    """可写入 Qdrant sparse vector 的 BM25 风格稀疏编码器。"""

    def __init__(
        self,
        num_features: int = 2_000_003,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.num_features = num_features
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 1.0
        self.idf_by_index: dict[int, float] = {}
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """统计语料 IDF。

        MVP 版本按当前批次统计 IDF。后续全量重建索引时，应对整个 collection 的
        chunk 语料重新 fit，以获得更稳定的 BM25 权重。
        """
        tokenized = [tokenize_text(text) for text in texts]
        if not tokenized:
            self.avg_doc_len = 1.0
            self.idf_by_index = {}
            self._fitted = True
            return

        doc_count = len(tokenized)
        doc_freq: Counter[int] = Counter()
        total_len = 0
        for tokens in tokenized:
            total_len += len(tokens)
            doc_freq.update(set(self._token_to_index(token) for token in tokens))

        self.avg_doc_len = max(total_len / max(doc_count, 1), 1.0)
        self.idf_by_index = {
            index: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for index, freq in doc_freq.items()
        }
        self._fitted = True

    def encode_documents(self, texts: list[str]) -> list[SparseVectorData]:
        """编码文档 sparse vector。"""
        self._ensure_fitted(texts)
        return [self._encode_document(text) for text in texts]

    def encode_query(self, query: str) -> SparseVectorData:
        """编码查询 sparse vector。"""
        tokens = tokenize_text(query)
        if not tokens:
            return SparseVectorData()

        counter = Counter(self._token_to_index(token) for token in tokens)
        indices: list[int] = []
        values: list[float] = []
        for index, tf in sorted(counter.items()):
            idf = self.idf_by_index.get(index, 1.0)
            indices.append(index)
            values.append(float(idf * (1.0 + math.log1p(tf))))
        return SparseVectorData(indices=indices, values=values)

    def _encode_document(self, text: str) -> SparseVectorData:
        """编码单个文档文本。"""
        tokens = tokenize_text(text)
        if not tokens:
            return SparseVectorData()

        doc_len = max(len(tokens), 1)
        counter = Counter(self._token_to_index(token) for token in tokens)
        indices: list[int] = []
        values: list[float] = []
        for index, tf in sorted(counter.items()):
            idf = self.idf_by_index.get(index, 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
            score = idf * (tf * (self.k1 + 1.0)) / max(denominator, 1e-9)
            indices.append(index)
            values.append(float(score))
        return SparseVectorData(indices=indices, values=values)

    def _ensure_fitted(self, texts: list[str]) -> None:
        """若尚未 fit，则用当前文本批次初始化统计信息。"""
        if not self._fitted:
            self.fit(texts)

    def _token_to_index(self, token: str) -> int:
        """将 token 稳定映射到 Qdrant sparse vector index。"""
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.num_features


class LocalBM25SparseIndex:
    """本地 BM25 fallback 检索器，用于 Qdrant sparse vector 不可用的场景。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.tokenized_docs: list[list[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 1.0

    def fit(self, chunks: list[DocumentChunk]) -> None:
        """用 chunk 语料初始化本地 BM25 索引。"""
        self.chunks = chunks
        self.tokenized_docs = [tokenize_text(chunk.text) for chunk in chunks]
        self.doc_freq = Counter()
        total_len = 0
        for tokens in self.tokenized_docs:
            total_len += len(tokens)
            self.doc_freq.update(set(tokens))
        self.avg_doc_len = max(total_len / max(len(self.tokenized_docs), 1), 1.0)

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        """执行本地 BM25 检索。"""
        if not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, DocumentChunk]] = []
        doc_count = len(self.tokenized_docs)
        query_counter = Counter(query_tokens)

        for chunk, doc_tokens in zip(self.chunks, self.tokenized_docs):
            if not doc_tokens:
                continue
            doc_counter = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for token, query_tf in query_counter.items():
                tf = doc_counter.get(token, 0)
                if tf <= 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
                score += query_tf * idf * (tf * (self.k1 + 1.0)) / max(denominator, 1e-9)
            if score > 0:
                scored.append((float(score), chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[VectorSearchResult] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append(
                VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                    payload=chunk.model_dump(),
                    source="local_bm25",
                    vector_name="bm25",
                    rank=rank,
                )
            )
        return results


def tokenize_text(text: str) -> list[str]:
    """面向中文论文与英文术语的简单 BM25 token 化。"""
    raw_tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    tokens: list[str] = []
    chinese_buffer: list[str] = []

    def flush_chinese() -> None:
        if not chinese_buffer:
            return
        tokens.extend(chinese_buffer)
        tokens.extend("".join(chinese_buffer[index : index + 2]) for index in range(len(chinese_buffer) - 1))
        chinese_buffer.clear()

    for token in raw_tokens:
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            chinese_buffer.append(token)
        else:
            flush_chinese()
            tokens.append(token)
    flush_chinese()
    return tokens

