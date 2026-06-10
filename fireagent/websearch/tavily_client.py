"""Tavily 联网搜索客户端封装。"""

from __future__ import annotations

from typing import Any, Optional

from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.websearch.schema import (
    TavilySearchRequest,
    TavilySearchResponse,
    TavilySearchResult,
    WebSearchError,
)


class TavilySearchClient:
    """Tavily 搜索客户端。

    优先使用 tavily-python SDK；若当前环境未安装 SDK，则退回 HTTP REST API。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[FireAgentConfig] = None,
        timeout: Optional[float] = None,
        prefer_sdk: bool = True,
    ) -> None:
        self.config = config or get_config()
        self.api_key = api_key if api_key is not None else self.config.tavily.api_key
        self.base_url = self.config.tavily.base_url.rstrip("/")
        self.timeout = timeout or self.config.tavily.timeout
        self.prefer_sdk = prefer_sdk
        self._sdk_client: Any = None

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        topic: Optional[str] = None,
        include_answer: Optional[bool] = None,
        include_raw_content: Optional[bool] = None,
        include_images: Optional[bool] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        time_range: Optional[str] = None,
    ) -> TavilySearchResponse:
        """执行 Tavily 搜索并返回标准化响应。"""
        self._ensure_api_key()
        request = TavilySearchRequest(
            query=query,
            max_results=max_results or self.config.tavily.max_results,
            search_depth=search_depth or self.config.tavily.search_depth,
            topic=topic or self.config.tavily.topic,
            include_answer=self.config.tavily.include_answer if include_answer is None else include_answer,
            include_raw_content=(
                self.config.tavily.include_raw_content
                if include_raw_content is None
                else include_raw_content
            ),
            include_images=self.config.tavily.include_images if include_images is None else include_images,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
            time_range=time_range,
        )

        if self.prefer_sdk:
            try:
                raw = self._search_with_sdk(request)
                return self._normalize_response(raw, request.query, source="sdk")
            except ImportError:
                pass

        raw = self._search_with_http(request)
        return self._normalize_response(raw, request.query, source="http")

    def _ensure_api_key(self) -> None:
        """确认 Tavily API key 已配置。"""
        if not self.api_key:
            raise WebSearchError(
                "未配置 TAVILY_API_KEY，无法执行联网搜索。请在 .env 或环境变量中设置。"
            )

    @property
    def sdk_client(self) -> Any:
        """懒加载 tavily-python SDK 客户端。"""
        if self._sdk_client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise ImportError("当前环境未安装 tavily-python SDK。") from exc
            self._sdk_client = TavilyClient(api_key=self.api_key)
        return self._sdk_client

    def _search_with_sdk(self, request: TavilySearchRequest) -> dict[str, Any]:
        """通过 tavily-python SDK 调用搜索。"""
        payload = self._request_payload(request, include_api_key=False)
        try:
            return dict(self.sdk_client.search(**payload))
        except TypeError:
            query = payload.pop("query")
            return dict(self.sdk_client.search(query, **payload))
        except Exception as exc:  # noqa: BLE001 - SDK 异常类型不稳定，统一包装。
            raise WebSearchError(f"Tavily SDK 搜索失败：{exc}") from exc

    def _search_with_http(self, request: TavilySearchRequest) -> dict[str, Any]:
        """通过 Tavily REST API 调用搜索。"""
        try:
            import httpx
        except ImportError as exc:
            raise WebSearchError("缺少 httpx，无法调用 Tavily REST API。") from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/search",
                headers=headers,
                json=self._request_payload(request, include_api_key=False),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return dict(response.json())
        except Exception as exc:  # noqa: BLE001 - HTTP 与 JSON 错误统一包装。
            raise WebSearchError(f"Tavily HTTP 搜索失败：{exc}") from exc

    @staticmethod
    def _request_payload(request: TavilySearchRequest, include_api_key: bool = False) -> dict[str, Any]:
        """将请求模型转换为 Tavily payload。"""
        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": request.max_results,
            "search_depth": request.search_depth,
            "topic": request.topic,
            "include_answer": request.include_answer,
            "include_raw_content": request.include_raw_content,
            "include_images": request.include_images,
        }
        if request.include_domains:
            payload["include_domains"] = request.include_domains
        if request.exclude_domains:
            payload["exclude_domains"] = request.exclude_domains
        if request.time_range:
            payload["time_range"] = request.time_range
        return payload

    @staticmethod
    def _normalize_response(raw: dict[str, Any], query: str, source: str) -> TavilySearchResponse:
        """将 Tavily 原始响应转换为项目内标准结构。"""
        results: list[TavilySearchResult] = []
        for item in raw.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            results.append(
                TavilySearchResult(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    content=str(item.get("content", "") or ""),
                    raw_content=str(item.get("raw_content", "") or ""),
                    score=float(item.get("score", 0.0) or 0.0),
                    published_date=item.get("published_date"),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "title",
                            "url",
                            "content",
                            "raw_content",
                            "score",
                            "published_date",
                        }
                    },
                )
            )

        return TavilySearchResponse(
            query=str(raw.get("query", query) or query),
            answer=str(raw.get("answer", "") or ""),
            results=results,
            images=list(raw.get("images", []) or []),
            response_time=raw.get("response_time"),
            metadata={
                "source": source,
                "follow_up_questions": raw.get("follow_up_questions"),
            },
        )

