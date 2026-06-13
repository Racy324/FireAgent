"""FireAgent API 请求与响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """API 模型基类。"""

    model_config = ConfigDict(extra="ignore")


class ChatRequest(APIModel):
    """问答请求。"""

    query: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(default=None, description="会话 ID，MVP 阶段仅透传")
    include_context: bool = Field(default=False, description="是否返回最终上下文")
    include_debug: bool = Field(default=False, description="是否返回调试字段")


class ChatResponse(APIModel):
    """问答响应。"""

    answer: str
    session_id: str = ""
    message_id: str = ""
    intent: str = ""
    evidence_sufficient: bool = False
    citations: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class SessionCreateRequest(APIModel):
    """创建会话请求。"""

    title: str = ""


class SessionUpdateRequest(APIModel):
    """更新会话请求。"""

    title: str


class SessionItem(APIModel):
    """会话列表项。"""

    session_id: str
    title: str = ""
    created_at: str
    updated_at: str


class SessionsListResponse(APIModel):
    """会话列表响应。"""

    sessions: list[SessionItem] = Field(default_factory=list)


class MessageItem(APIModel):
    """会话消息项。"""

    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    intent: str = ""
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMessagesResponse(APIModel):
    """会话消息响应。"""

    session_id: str
    messages: list[MessageItem] = Field(default_factory=list)


class IngestRequest(APIModel):
    """PDF 入库请求。"""

    pdf_paths: list[str] = Field(default_factory=list, description="指定 PDF 文件路径")
    raw_pdf_dir: str = Field(default="data/raw_pdfs", description="未指定 pdf_paths 时扫描的目录")
    recreate_collection: bool = Field(default=False, description="是否重建 Qdrant collection")
    max_files: Optional[int] = Field(default=None, gt=0, description="最多处理多少个 PDF")
    max_pages: Optional[int] = Field(default=None, gt=0, description="每个 PDF 最多解析多少页")
    batch_size: int = Field(default=64, gt=0, description="Qdrant upsert 批大小")
    hash_embedding: bool = Field(default=False, description="是否使用哈希 embedding 进行开发验证")
    dry_run: bool = Field(default=False, description="只解析和切块，不写入 Qdrant")


class IngestFileResult(APIModel):
    """单个 PDF 入库结果。"""

    path: str
    chunks: int = 0
    status: str = "pending"
    error: Optional[str] = None


class IngestResponse(APIModel):
    """PDF 入库响应。"""

    status: str
    collection: str
    dry_run: bool = False
    parsed_files: int = 0
    total_chunks: int = 0
    failed_files: int = 0
    files: list[IngestFileResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(APIModel):
    """服务健康检查响应。"""

    status: str
    app: str
    version: str
    qdrant_url: str
    qdrant_collection: str
    enable_web_fallback: bool
    message: str = ""


class PaperItem(APIModel):
    """论文列表项。"""

    doc_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class PaperDetail(APIModel):
    """论文详情。"""

    doc_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class PapersStats(APIModel):
    """知识库统计。"""

    total_papers: int = 0
    total_chunks: int = 0
    year_distribution: dict[int, int] = Field(default_factory=dict)
    top_keywords: dict[str, int] = Field(default_factory=dict)
