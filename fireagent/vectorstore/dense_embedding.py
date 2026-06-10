"""Dense embedding 封装。"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Iterable

import httpx

from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.schema import VectorStoreError


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]")


class BaseDenseEmbedder(ABC):
    """Dense embedding 抽象接口。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本。"""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """编码用户查询。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""


class SentenceTransformerDenseEmbedder(BaseDenseEmbedder):
    """基于 sentence-transformers 的 dense embedding 实现。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = None if device == "auto" else device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self) -> object:
        """懒加载 sentence-transformers 模型，避免导入阶段加载大模型。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise VectorStoreError(
                    "缺少 sentence-transformers，无法加载 dense embedding 模型；"
                    "可安装依赖或在测试中使用 HashingDenseEmbedder。"
                ) from exc

            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本。"""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [self._to_float_list(vector) for vector in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """编码用户查询。"""
        return self.embed_documents([query])[0]

    @property
    def dimension(self) -> int:
        """返回模型向量维度。"""
        if self._dimension is not None:
            return self._dimension
        get_dim = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            dimension = get_dim()
            if dimension:
                self._dimension = int(dimension)
                return self._dimension
        self._dimension = len(self.embed_query("FireAgent dimension probe"))
        return self._dimension

    @staticmethod
    def _to_float_list(vector: Iterable[float]) -> list[float]:
        """将 numpy/torch 向量转换为 Python float list。"""
        return [float(value) for value in vector]


class HashingDenseEmbedder(BaseDenseEmbedder):
    """轻量哈希向量 fallback，主要用于无模型环境下的本地验证。"""

    def __init__(self, dimension: int = 1024, normalize: bool = True) -> None:
        self._dimension = dimension
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成确定性的哈希 dense 向量。"""
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """生成查询的哈希 dense 向量。"""
        return self._embed(query)

    @property
    def dimension(self) -> int:
        """返回哈希向量维度。"""
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        """将 token 哈希到固定维度向量。"""
        vector = [0.0] * self._dimension
        for token in tokenize_for_hashing(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        if self.normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
        return vector


class OllamaDenseEmbedder(BaseDenseEmbedder):
    """基于 Ollama `/api/embed` 的 dense embedding 实现。"""

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量调用 Ollama embedding 接口。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """编码用户查询。"""
        return self.embed_documents([query])[0]

    @property
    def dimension(self) -> int:
        """返回 Ollama embedding 向量维度。"""
        if self._dimension is None:
            self._dimension = len(self.embed_query("FireAgent dimension probe"))
        return self._dimension

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用新版 `/api/embed`，失败时兼容旧版 `/api/embeddings`。"""
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            embeddings = raw.get("embeddings")
            if isinstance(embeddings, list):
                return [self._normalize(self._to_float_list(vector)) for vector in embeddings]
            raise VectorStoreError("Ollama /api/embed 响应缺少 embeddings。")
        except Exception as embed_exc:  # noqa: BLE001 - 尝试旧版接口后再统一报错。
            if len(texts) != 1:
                raise VectorStoreError(f"Ollama embedding 调用失败：{embed_exc}") from embed_exc
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": texts[0]},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()
                embedding = raw.get("embedding")
                if isinstance(embedding, list):
                    return [self._normalize(self._to_float_list(embedding))]
                raise VectorStoreError("Ollama /api/embeddings 响应缺少 embedding。")
            except Exception as legacy_exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"Ollama embedding 调用失败，请确认 Ollama 已启动且模型已拉取：{self.model_name}"
                ) from legacy_exc

    def _normalize(self, vector: list[float]) -> list[float]:
        """按需对向量做 L2 归一化。"""
        if not self.normalize_embeddings:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _to_float_list(vector: Iterable[float]) -> list[float]:
        """将 Ollama 返回向量转换为 Python float list。"""
        return [float(value) for value in vector]


def tokenize_for_hashing(text: str) -> list[str]:
    """面向中英文混合文本的简单 token 化。"""
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


def create_dense_embedder(
    config: FireAgentConfig | None = None,
    allow_hash_fallback: bool = False,
) -> BaseDenseEmbedder:
    """根据配置创建 dense embedding 实例。"""
    cfg = config or get_config()
    if allow_hash_fallback:
        return HashingDenseEmbedder(dimension=cfg.qdrant.dense_vector_size)
    if cfg.embedding.provider.lower() == "ollama":
        return OllamaDenseEmbedder(
            model_name=cfg.embedding.model_name,
            base_url=cfg.embedding.base_url,
            timeout=cfg.embedding.timeout,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=cfg.embedding.normalize_embeddings,
        )
    return SentenceTransformerDenseEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        normalize_embeddings=cfg.embedding.normalize_embeddings,
        batch_size=cfg.embedding.batch_size,
    )
