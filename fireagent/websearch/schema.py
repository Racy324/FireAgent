"""联网搜索模块的数据结构。"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class WebSearchError(RuntimeError):
    """联网搜索失败时抛出的领域异常。"""


class WebSearchModel(BaseModel):
    """联网搜索模块的 Pydantic 基类。"""

    model_config = ConfigDict(extra="ignore")


class TavilySearchRequest(WebSearchModel):
    """Tavily 搜索请求参数。"""

    query: str
    max_results: int = Field(default=5, gt=0)
    search_depth: str = "basic"
    topic: str = "general"
    include_answer: bool = True
    include_raw_content: bool = False
    include_images: bool = False
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    time_range: Optional[str] = None


class TavilySearchResult(WebSearchModel):
    """标准化后的 Tavily 单条搜索结果。"""

    title: str = ""
    url: str = ""
    content: str = ""
    raw_content: str = ""
    score: float = 0.0
    published_date: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TavilySearchResponse(WebSearchModel):
    """标准化后的 Tavily 搜索响应。"""

    query: str
    answer: str = ""
    results: list[TavilySearchResult] = Field(default_factory=list)
    images: list[Any] = Field(default_factory=list)
    response_time: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebEvidenceChunk(WebSearchModel):
    """由联网资料解析得到的临时 evidence chunk。

    这些 chunk 只参与当前问答流程，不写入主 Qdrant collection。
    """

    chunk_id: str
    parent_id: str
    doc_id: str
    title: str = ""
    url: str = ""
    text: str
    snippet: str = ""
    source_type: str = "web_tavily"
    score: float = 0.0
    published_date: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_web_chunk_id(url: str, text: str, index: int = 0) -> str:
    """根据 URL、文本和序号生成稳定的 web chunk id。"""
    raw = f"web|{url}|{index}|{text[:160]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def make_web_doc_id(url: str) -> str:
    """根据 URL 生成稳定的 web doc id。"""
    raw = f"web-doc|{url}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

