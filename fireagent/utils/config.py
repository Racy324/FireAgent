"""Typed configuration loading for FireAgent.

The loader merges repository YAML defaults with values from ``.env`` and the
process environment. Environment variables always take precedence.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class FireAgentConfigError(RuntimeError):
    """Raised when FireAgent configuration cannot be loaded or validated."""


class ConfigSection(BaseModel):
    """Base class for typed config sections."""

    model_config = ConfigDict(extra="ignore")


class AppConfig(ConfigSection):
    """Application runtime settings."""

    name: str = "FireAgent"
    environment: str = "local"
    log_level: str = "INFO"


class RAGConfig(ConfigSection):
    """RAG chunking and fallback settings."""

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    parent_chunk_size: int = Field(default=1800, gt=0)
    enable_web_fallback: bool = True

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "RAGConfig":
        """Ensure chunk sizes are internally consistent."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.parent_chunk_size < self.chunk_size:
            raise ValueError("parent_chunk_size must be greater than or equal to chunk_size")
        return self


class IngestionConfig(ConfigSection):
    """PDF 解析与入库阶段配置。"""

    pdf_parser: Literal["pdfplumber", "mineru", "mineru_api"] = "pdfplumber"
    mineru_output_dir: str = "data/parsed/mineru"
    mineru_backend: str = "pipeline"
    mineru_model_source: str = ""
    mineru_timeout: float = Field(default=1800.0, gt=0)
    mineru_keep_output: bool = True
    mineru_cli_path: str = "mineru"
    # MinerU 云端 API 配置
    mineru_api_key: str = ""
    mineru_api_base_url: str = "https://mineru.net"
    mineru_api_enable_ocr: bool = True
    mineru_api_enable_formula: bool = True
    mineru_api_enable_table: bool = True
    mineru_api_language: str = "ch"


class RetrievalConfig(ConfigSection):
    """Hybrid retrieval, fusion, and reranking settings."""

    max_rewrite_queries: int = Field(default=4, gt=0)
    dense_top_k: int = Field(default=30, gt=0)
    sparse_top_k: int = Field(default=30, gt=0)
    fusion_top_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=10, gt=0)
    dense_weight: float = Field(default=0.65, ge=0.0)
    sparse_weight: float = Field(default=0.35, ge=0.0)
    rrf_k: int = Field(default=60, gt=0)
    context_max_chars: int = Field(default=6000, gt=0)
    context_max_evidence_chars: int = Field(default=1200, gt=0)
    sufficiency_min_score: float = Field(default=0.18, ge=0.0)
    sufficiency_min_evidence: int = Field(default=2, gt=0)
    sufficiency_min_term_coverage: float = Field(default=0.30, ge=0.0, le=1.0)
    cross_doc_min_docs: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "RetrievalConfig":
        """Ensure retrieval limits and fusion weights are usable."""
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one of dense_weight or sparse_weight must be positive")
        if self.rerank_top_k > self.fusion_top_k:
            raise ValueError("rerank_top_k must be less than or equal to fusion_top_k")
        return self


class EmbeddingConfig(ConfigSection):
    """Dense embedding model settings."""

    provider: str = "sentence_transformers"
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    normalize_embeddings: bool = True
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=60.0, gt=0)
    batch_size: int = Field(default=32, gt=0)


class RerankerConfig(ConfigSection):
    """Cross-encoder reranker model settings."""

    provider: str = "flagembedding"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=60.0, gt=0)
    batch_size: int = Field(default=16, gt=0)
    fallback_to_lexical: bool = True


class LLMConfig(ConfigSection):
    """OpenAI-compatible LLM settings."""

    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, gt=0)
    timeout: float = Field(default=60.0, gt=0)
    enabled: bool = True


class HNSWConfig(ConfigSection):
    """Qdrant HNSW index settings."""

    m: int = Field(default=16, gt=0)
    ef_construct: int = Field(default=100, gt=0)


class QdrantConfig(ConfigSection):
    """Qdrant connection and collection settings."""

    url: str = "http://localhost:6333"
    api_key: str = ""
    collection: str = "fireagent_papers"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    dense_vector_size: int = Field(default=1024, gt=0)
    distance: Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)


class TavilyConfig(ConfigSection):
    """Tavily web-search fallback settings."""

    api_key: str = ""
    base_url: str = "https://api.tavily.com"
    max_results: int = Field(default=5, gt=0)
    search_depth: Literal["basic", "advanced"] = "basic"
    topic: str = "general"
    include_answer: bool = True
    include_raw_content: bool = False
    include_images: bool = False
    timeout: float = Field(default=30.0, gt=0)
    chunks_per_source: int = Field(default=3, gt=0)


class EvaluationConfig(ConfigSection):
    """RAG 评测流程默认配置。"""

    dataset_path: str = "data/eval/fireagent_eval_sample.jsonl"
    output_dir: str = "data/eval/runs"
    default_mode: Literal["manual", "ragas", "both"] = "manual"
    fail_under: Optional[float] = None


class ShortTermMemoryConfig(ConfigSection):
    """短期上下文管理配置。"""

    enabled: bool = True
    recent_message_limit: int = Field(default=8, gt=0)
    max_chars: int = Field(default=3500, gt=0)
    rolling_summary_max_chars: int = Field(default=1500, gt=0)
    summarize_after_messages: int = Field(default=20, gt=0)
    intermediate_result_limit: int = Field(default=8, gt=0)
    pinned_message_limit: int = Field(default=8, gt=0)


class LongTermMemoryConfig(ConfigSection):
    """长期记忆配置。"""

    enabled: bool = True
    collection: str = "fireagent_memories"
    top_k: int = Field(default=5, gt=0)
    max_prompt_chars: int = Field(default=1200, gt=0)
    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)
    write_enabled: bool = True
    importance_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_candidates_per_turn: int = Field(default=2, gt=0)
    default_half_life_days: float = Field(default=180.0, gt=0)
    semantic_half_life_days: float = Field(default=365.0, gt=0)
    episodic_half_life_days: float = Field(default=90.0, gt=0)
    procedural_half_life_days: float = Field(default=365.0, gt=0)
    dedup_similarity_threshold: float = Field(default=0.88, ge=0.0, le=1.0)


class MemoryConfig(ConfigSection):
    """会话历史和记忆系统配置。"""

    enabled: bool = True
    database_path: str = "data/app/fireagent.db"
    recent_message_limit: int = Field(default=8, gt=0)
    short_term: ShortTermMemoryConfig = Field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)


class WebFallbackConfig(ConfigSection):
    """Web fallback trigger settings."""

    temporal_keywords: list[str] = Field(
        default_factory=lambda: [
            "最新",
            "政策",
            "标准",
            "规范",
            "法规",
            "事故",
            "2026",
            "今年",
            "最近",
            "现行",
            "当前",
            "近期",
            "刚发布",
            "现在",
        ]
    )
    official_keywords: list[str] = Field(
        default_factory=lambda: [
            "标准",
            "规范",
            "法规",
            "政策",
            "官方",
            "通报",
            "统计",
            "应急管理部",
            "消防救援局",
            "GB",
            "现行版本",
        ]
    )
    local_scope_keywords: list[str] = Field(
        default_factory=lambda: [
            "根据本地知识库",
            "仅基于知识库",
            "只看上传文档",
            "根据这些论文",
            "这批论文",
            "根据论文",
            "仅根据本地",
        ]
    )
    harmful_keywords: list[str] = Field(
        default_factory=lambda: [
            "纵火",
            "制造火灾",
            "规避消防",
            "绕过报警",
            "破坏灭火",
            "提高火灾破坏",
            "如何放火",
            "怎么纵火",
        ]
    )
    ambiguous_keywords: list[str] = Field(
        default_factory=lambda: ["那个", "这个", "这篇", "它", "上述", "前面提到"]
    )
    force_web_temporal: bool = True
    force_web_official: bool = True
    disable_for_local_scope: bool = True
    disable_for_harmful: bool = True
    ask_clarify_for_ambiguous: bool = True


class PromptConfig(ConfigSection):
    """Prompt file registry."""

    intent_router: str = "fireagent/prompts/intent_router.md"
    query_rewrite: str = "fireagent/prompts/query_rewrite.md"
    sufficiency_check: str = "fireagent/prompts/sufficiency_check.md"
    answer_generation: str = "fireagent/prompts/answer_generation.md"
    hallucination_check: str = "fireagent/prompts/hallucination_check.md"


class FireAgentConfig(ConfigSection):
    """Top-level FireAgent configuration."""

    app: AppConfig = Field(default_factory=AppConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    web: WebFallbackConfig = Field(default_factory=WebFallbackConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)


ENV_TO_CONFIG_PATH: dict[str, tuple[str, ...]] = {
    "QDRANT_URL": ("qdrant", "url"),
    "QDRANT_API_KEY": ("qdrant", "api_key"),
    "QDRANT_COLLECTION": ("qdrant", "collection"),
    "QDRANT_DENSE_VECTOR_NAME": ("qdrant", "dense_vector_name"),
    "QDRANT_SPARSE_VECTOR_NAME": ("qdrant", "sparse_vector_name"),
    "QDRANT_DENSE_VECTOR_SIZE": ("qdrant", "dense_vector_size"),
    "TAVILY_API_KEY": ("tavily", "api_key"),
    "TAVILY_BASE_URL": ("tavily", "base_url"),
    "TAVILY_MAX_RESULTS": ("tavily", "max_results"),
    "TAVILY_SEARCH_DEPTH": ("tavily", "search_depth"),
    "TAVILY_TOPIC": ("tavily", "topic"),
    "TAVILY_INCLUDE_ANSWER": ("tavily", "include_answer"),
    "TAVILY_INCLUDE_RAW_CONTENT": ("tavily", "include_raw_content"),
    "TAVILY_INCLUDE_IMAGES": ("tavily", "include_images"),
    "TAVILY_TIMEOUT": ("tavily", "timeout"),
    "TAVILY_CHUNKS_PER_SOURCE": ("tavily", "chunks_per_source"),
    "OPENAI_COMPATIBLE_BASE_URL": ("llm", "base_url"),
    "OPENAI_COMPATIBLE_API_KEY": ("llm", "api_key"),
    "OPENAI_COMPATIBLE_MODEL": ("llm", "model"),
    "OPENAI_COMPATIBLE_TEMPERATURE": ("llm", "temperature"),
    "OPENAI_COMPATIBLE_MAX_TOKENS": ("llm", "max_tokens"),
    "OPENAI_COMPATIBLE_TIMEOUT": ("llm", "timeout"),
    "ENABLE_LLM_ANSWER": ("llm", "enabled"),
    "LLM_PROVIDER": ("llm", "provider"),
    "EMBEDDING_PROVIDER": ("embedding", "provider"),
    "EMBEDDING_MODEL_NAME": ("embedding", "model_name"),
    "EMBEDDING_BASE_URL": ("embedding", "base_url"),
    "EMBEDDING_TIMEOUT": ("embedding", "timeout"),
    "EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
    "OLLAMA_BASE_URL": ("embedding", "base_url"),
    "RERANKER_PROVIDER": ("reranker", "provider"),
    "RERANKER_MODEL_NAME": ("reranker", "model_name"),
    "RERANKER_BASE_URL": ("reranker", "base_url"),
    "RERANKER_TIMEOUT": ("reranker", "timeout"),
    "RERANKER_BATCH_SIZE": ("reranker", "batch_size"),
    "RERANKER_FALLBACK_TO_LEXICAL": ("reranker", "fallback_to_lexical"),
    "CHUNK_SIZE": ("rag", "chunk_size"),
    "CHUNK_OVERLAP": ("rag", "chunk_overlap"),
    "PARENT_CHUNK_SIZE": ("rag", "parent_chunk_size"),
    "MAX_REWRITE_QUERIES": ("retrieval", "max_rewrite_queries"),
    "DENSE_TOP_K": ("retrieval", "dense_top_k"),
    "SPARSE_TOP_K": ("retrieval", "sparse_top_k"),
    "FUSION_TOP_K": ("retrieval", "fusion_top_k"),
    "RERANK_TOP_K": ("retrieval", "rerank_top_k"),
    "DENSE_WEIGHT": ("retrieval", "dense_weight"),
    "SPARSE_WEIGHT": ("retrieval", "sparse_weight"),
    "RRF_K": ("retrieval", "rrf_k"),
    "CONTEXT_MAX_CHARS": ("retrieval", "context_max_chars"),
    "CONTEXT_MAX_EVIDENCE_CHARS": ("retrieval", "context_max_evidence_chars"),
    "SUFFICIENCY_MIN_SCORE": ("retrieval", "sufficiency_min_score"),
    "SUFFICIENCY_MIN_EVIDENCE": ("retrieval", "sufficiency_min_evidence"),
    "SUFFICIENCY_MIN_TERM_COVERAGE": ("retrieval", "sufficiency_min_term_coverage"),
    "ENABLE_WEB_FALLBACK": ("rag", "enable_web_fallback"),
    "PDF_PARSER": ("ingestion", "pdf_parser"),
    "MINERU_OUTPUT_DIR": ("ingestion", "mineru_output_dir"),
    "MINERU_BACKEND": ("ingestion", "mineru_backend"),
    "MINERU_MODEL_SOURCE": ("ingestion", "mineru_model_source"),
    "MINERU_TIMEOUT": ("ingestion", "mineru_timeout"),
    "MINERU_KEEP_OUTPUT": ("ingestion", "mineru_keep_output"),
    "MINERU_CLI_PATH": ("ingestion", "mineru_cli_path"),
    "MINERU_API_KEY": ("ingestion", "mineru_api_key"),
    "MINERU_API_BASE_URL": ("ingestion", "mineru_api_base_url"),
    "MINERU_API_ENABLE_OCR": ("ingestion", "mineru_api_enable_ocr"),
    "MINERU_API_ENABLE_FORMULA": ("ingestion", "mineru_api_enable_formula"),
    "MINERU_API_ENABLE_TABLE": ("ingestion", "mineru_api_enable_table"),
    "MINERU_API_LANGUAGE": ("ingestion", "mineru_api_language"),
    "EVAL_DATASET_PATH": ("evaluation", "dataset_path"),
    "EVAL_OUTPUT_DIR": ("evaluation", "output_dir"),
    "EVAL_DEFAULT_MODE": ("evaluation", "default_mode"),
    "EVAL_FAIL_UNDER": ("evaluation", "fail_under"),
    "MEMORY_ENABLED": ("memory", "enabled"),
    "MEMORY_DATABASE_PATH": ("memory", "database_path"),
    "MEMORY_RECENT_MESSAGE_LIMIT": ("memory", "recent_message_limit"),
    "APP_ENVIRONMENT": ("app", "environment"),
    "LOG_LEVEL": ("app", "log_level"),
}


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    if not path.exists():
        raise FireAgentConfigError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise FireAgentConfigError(f"Configuration file must contain a YAML mapping: {path}")
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries and return a new dictionary."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested dictionary value, creating intermediate dictionaries as needed."""
    current = data
    for key in path[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise FireAgentConfigError(f"Cannot set nested config path {'.'.join(path)}")
        current = child
    current[path[-1]] = value


def load_environment(env_file: Path | None = None) -> dict[str, str]:
    """Load ``.env`` values and overlay process environment variables."""
    values: dict[str, str] = {}
    if env_file and env_file.exists():
        values.update({key: value for key, value in dotenv_values(env_file).items() if value is not None})
    values.update(os.environ)
    return values


def load_raw_config(
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge YAML defaults with environment overrides."""
    config_path = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    env_path = Path(env_file) if env_file else DEFAULT_ENV_FILE

    raw: dict[str, Any] = {}
    for filename in ("app.yaml", "models.yaml", "qdrant.yaml", "prompts.yaml"):
        raw = deep_merge(raw, load_yaml_file(config_path / filename))

    env_values = load_environment(env_path)
    for env_name, config_path_tuple in ENV_TO_CONFIG_PATH.items():
        value = env_values.get(env_name)
        if value is not None and value != "":
            set_nested_value(raw, config_path_tuple, value)

    ollama_base_url = env_values.get("OLLAMA_BASE_URL")
    if ollama_base_url:
        if not env_values.get("EMBEDDING_BASE_URL"):
            set_nested_value(raw, ("embedding", "base_url"), ollama_base_url)
        if not env_values.get("RERANKER_BASE_URL"):
            set_nested_value(raw, ("reranker", "base_url"), ollama_base_url)

    return raw


def load_config(
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> FireAgentConfig:
    """Return a validated FireAgent configuration object."""
    raw = load_raw_config(config_dir=config_dir, env_file=env_file)
    try:
        return FireAgentConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - surface pydantic details in a domain error.
        raise FireAgentConfigError(f"Invalid FireAgent configuration: {exc}") from exc


@lru_cache(maxsize=1)
def _cached_config(config_dir: str | None, env_file: str | None) -> FireAgentConfig:
    return load_config(config_dir=config_dir, env_file=env_file)


def get_config(
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> FireAgentConfig:
    """Return cached configuration for application runtime use."""
    config_dir_key = str(Path(config_dir).resolve()) if config_dir else None
    env_file_key = str(Path(env_file).resolve()) if env_file else None
    return _cached_config(config_dir_key, env_file_key)


def clear_config_cache() -> None:
    """Clear cached configuration, mainly for tests and scripts."""
    _cached_config.cache_clear()
