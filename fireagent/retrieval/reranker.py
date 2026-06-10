"""候选文档重排模块。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from fireagent.retrieval.schema import FusedRetrievalResult, RerankedRetrievalResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.dense_embedding import OllamaDenseEmbedder
from fireagent.vectorstore.schema import VectorStoreError
from fireagent.vectorstore.sparse_embedding import tokenize_text


class BaseReranker(ABC):
    """重排器抽象接口。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """对融合候选进行重排。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回重排器名称。"""


class FlagEmbeddingReranker(BaseReranker):
    """基于 FlagEmbedding FlagReranker 的 cross-encoder 重排器。"""

    def __init__(self, model_name: str, use_fp16: bool = False) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def name(self) -> str:
        """返回重排器名称。"""
        return f"flagembedding:{self.model_name}"

    @property
    def model(self) -> object:
        """懒加载 FlagReranker，避免程序启动时下载大模型。"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise VectorStoreError("缺少 FlagEmbedding，无法加载默认 reranker。") from exc
            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """使用 FlagEmbedding 计算 query-document 相关性分数。"""
        pairs = [[query, candidate.text or str(candidate.payload.get("text", ""))] for candidate in candidates]
        if not pairs:
            return []
        scores = self.model.compute_score(pairs)
        return build_reranked_results(
            candidates=candidates,
            scores=ensure_score_list(scores),
            top_k=top_k,
            reranker_name=self.name,
        )


class CrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers CrossEncoder 的重排器。"""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = None if device == "auto" else device
        self._model = None

    @property
    def name(self) -> str:
        """返回重排器名称。"""
        return f"cross-encoder:{self.model_name}"

    @property
    def model(self) -> object:
        """懒加载 CrossEncoder 模型。"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise VectorStoreError("缺少 sentence-transformers，无法加载 CrossEncoder reranker。") from exc
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """使用 CrossEncoder 计算 query-document 相关性分数。"""
        pairs = [(query, candidate.text or str(candidate.payload.get("text", ""))) for candidate in candidates]
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return build_reranked_results(
            candidates=candidates,
            scores=ensure_score_list(scores),
            top_k=top_k,
            reranker_name=self.name,
        )


class LexicalReranker(BaseReranker):
    """轻量词项覆盖重排器，用于无模型或模型不可用时的 fallback。"""

    @property
    def name(self) -> str:
        """返回重排器名称。"""
        return "lexical-fallback"

    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """基于查询词覆盖率、文档词密度和融合分数进行重排。"""
        query_terms = set(tokenize_text(query))
        scores: list[float] = []
        for candidate in candidates:
            text = candidate.text or str(candidate.payload.get("text", ""))
            text_terms = tokenize_text(text)
            scores.append(self._score(query_terms, text_terms, candidate.fusion_score))
        return build_reranked_results(
            candidates=candidates,
            scores=scores,
            top_k=top_k,
            reranker_name=self.name,
        )

    @staticmethod
    def _score(query_terms: set[str], text_terms: list[str], fusion_score: float) -> float:
        """计算 0 到 1 附近的可解释相关性分数。"""
        if not query_terms or not text_terms:
            return min(fusion_score * 10.0, 0.05)
        text_set = set(text_terms)
        overlap = query_terms & text_set
        coverage = len(overlap) / max(len(query_terms), 1)
        density = sum(1 for token in text_terms if token in query_terms) / max(len(text_terms), 1)
        fusion_bonus = min(fusion_score * 10.0, 0.15)
        return float(0.75 * coverage + 0.10 * math.sqrt(density) + fusion_bonus)


class OllamaEmbeddingReranker(BaseReranker):
    """使用 Ollama embedding 相似度进行候选重排。

    Ollama 官方提供 embedding API，但没有标准 rerank API。该实现用于兼容
    Ollama 部署的 bge 类模型；如果需要真正 cross-encoder reranker，建议使用
    FlagEmbeddingReranker。
    """

    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.embedder = OllamaDenseEmbedder(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

    @property
    def name(self) -> str:
        """返回重排器名称。"""
        return f"ollama-embedding-reranker:{self.model_name}"

    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """使用 query/document embedding 余弦相似度重排候选。"""
        if not candidates:
            return []
        texts = [candidate.text or str(candidate.payload.get("text", "")) for candidate in candidates]
        query_vector = self.embedder.embed_query(query)
        document_vectors = self.embedder.embed_documents(texts)
        scores = [cosine_similarity(query_vector, vector) for vector in document_vectors]
        return build_reranked_results(
            candidates=candidates,
            scores=scores,
            top_k=top_k,
            reranker_name=self.name,
        )


class FallbackReranker(BaseReranker):
    """带模型失败回退能力的重排器。"""

    def __init__(self, primary: BaseReranker, fallback: BaseReranker | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or LexicalReranker()
        self.last_fallback_reason: Optional[str] = None

    @property
    def name(self) -> str:
        """返回当前重排器名称。"""
        return self.primary.name

    def rerank(
        self,
        query: str,
        candidates: list[FusedRetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RerankedRetrievalResult]:
        """优先使用主模型，失败时退回轻量重排。"""
        try:
            self.last_fallback_reason = None
            return self.primary.rerank(query, candidates, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - 模型下载、导入、设备错误都应触发 fallback。
            self.last_fallback_reason = str(exc)
            results = self.fallback.rerank(query, candidates, top_k=top_k)
            for result in results:
                result.fallback_reason = self.last_fallback_reason
                result.reranker_name = self.fallback.name
            return results


def build_reranked_results(
    candidates: list[FusedRetrievalResult],
    scores: list[float],
    top_k: Optional[int],
    reranker_name: str,
) -> list[RerankedRetrievalResult]:
    """将重排分数和融合候选合并为标准结果。"""
    paired: list[tuple[float, FusedRetrievalResult]] = []
    for candidate, score in zip(candidates, scores):
        paired.append((float(score), candidate))

    paired.sort(key=lambda item: (item[0], item[1].fusion_score), reverse=True)
    limit = top_k or len(paired)
    results: list[RerankedRetrievalResult] = []
    for rank, (score, candidate) in enumerate(paired[:limit], start=1):
        results.append(
            RerankedRetrievalResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                payload=candidate.payload,
                fusion_score=candidate.fusion_score,
                rerank_score=score,
                final_score=score,
                dense_score=candidate.dense_score,
                sparse_score=candidate.sparse_score,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                source_scores=candidate.source_scores,
                source_ranks=candidate.source_ranks,
                sources=candidate.sources,
                rank=rank,
                reranker_name=reranker_name,
            )
        )
    return results


def ensure_score_list(scores: object) -> list[float]:
    """将模型返回的标量、numpy 数组或列表统一转成 float list。"""
    if isinstance(scores, (int, float)):
        return [float(scores)]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, Iterable):
        return [float(score) for score in scores]
    return [float(scores)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def create_reranker(
    config: FireAgentConfig | None = None,
    fallback_to_lexical: bool = True,
) -> BaseReranker:
    """根据配置创建默认 reranker。"""
    cfg = config or get_config()
    provider = cfg.reranker.provider.lower()
    if provider == "lexical":
        return LexicalReranker()
    if provider == "ollama":
        primary = OllamaEmbeddingReranker(
            model_name=cfg.reranker.model_name,
            base_url=cfg.reranker.base_url,
            timeout=cfg.reranker.timeout,
            batch_size=cfg.reranker.batch_size,
        )
    elif provider == "flagembedding" or "bge-reranker" in cfg.reranker.model_name.lower():
        primary: BaseReranker = FlagEmbeddingReranker(cfg.reranker.model_name)
    else:
        primary = CrossEncoderReranker(cfg.reranker.model_name, device=cfg.reranker.device)
    if fallback_to_lexical and cfg.reranker.fallback_to_lexical:
        return FallbackReranker(primary=primary, fallback=LexicalReranker())
    return primary
