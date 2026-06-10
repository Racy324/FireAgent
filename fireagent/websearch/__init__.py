"""FireAgent 联网搜索兜底模块。"""

from fireagent.websearch.schema import (
    TavilySearchRequest,
    TavilySearchResponse,
    TavilySearchResult,
    WebEvidenceChunk,
    WebSearchError,
)
from fireagent.websearch.tavily_client import TavilySearchClient
from fireagent.websearch.web_evidence_builder import WebEvidenceBuilder, should_trigger_web_search
from fireagent.websearch.web_parser import WebSearchResultParser

__all__ = [
    "TavilySearchClient",
    "TavilySearchRequest",
    "TavilySearchResponse",
    "TavilySearchResult",
    "WebEvidenceBuilder",
    "WebEvidenceChunk",
    "WebSearchError",
    "WebSearchResultParser",
    "should_trigger_web_search",
]

