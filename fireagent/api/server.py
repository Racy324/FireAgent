"""FireAgent FastAPI 服务入口。"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from fireagent import __version__
from fireagent.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestFileResult,
    IngestRequest,
    IngestResponse,
    PaperDetail,
    PaperItem,
    PapersStats,
)
from fireagent.api.streaming import run_streaming_workflow
from fireagent.graph.workflow import run_fireagent_workflow
from fireagent.ingestion import PDFIndexBuilder
from fireagent.ingestion.pdfplumber_parser import PdfPlumberPDFParser
from fireagent.utils.config import FireAgentConfig, PROJECT_ROOT, get_config
from fireagent.vectorstore import FireAgentQdrantClient


logger = logging.getLogger(__name__)


def create_app(config: Optional[FireAgentConfig] = None) -> object:
    """创建 FastAPI 应用。

    返回类型用 object 是为了让未安装 FastAPI 的环境也能静态编译本文件；真正调用
    create_app 时会懒导入 FastAPI。
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("缺少 FastAPI，请先安装依赖：pip install fastapi uvicorn") from exc

    cfg = config or get_config()
    app = FastAPI(
        title="FireAgent API",
        description="火灾领域本地论文 RAG + Tavily 联网兜底问答服务",
        version=__version__,
    )

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = cfg
    app.state.vectorstore = None

    def get_vectorstore(hash_embedding: bool = False) -> FireAgentQdrantClient:
        """懒加载 Qdrant 向量库客户端。"""
        if app.state.vectorstore is None or hash_embedding:
            app.state.vectorstore = FireAgentQdrantClient(
                config=cfg,
                allow_hash_dense_fallback=hash_embedding,
            )
        return app.state.vectorstore

    # ── 健康检查 ──

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """健康检查接口。"""
        return HealthResponse(
            status="ok",
            app=cfg.app.name,
            version=__version__,
            qdrant_url=cfg.qdrant.url,
            qdrant_collection=cfg.qdrant.collection,
            enable_web_fallback=cfg.rag.enable_web_fallback,
            message="FireAgent API is running.",
        )

    # ── 问答（同步） ──

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """在线问答接口。"""
        try:
            state = run_fireagent_workflow(request.query, config=cfg, vectorstore=None)
        except Exception as exc:  # noqa: BLE001 - API 层统一转 HTTP 错误。
            logger.exception("FireAgent /chat failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        debug = {}
        if request.include_debug:
            debug = {
                "rewritten_queries": state.get("rewritten_queries", []),
                "dense_count": len(state.get("local_dense_results", []) or []),
                "sparse_count": len(state.get("local_sparse_results", []) or []),
                "fused_count": len(state.get("fused_results", []) or []),
                "reranked_count": len(state.get("reranked_results", []) or []),
                "web_count": len(state.get("web_results", []) or []),
                "hallucination_warnings": state.get("hallucination_warnings", []),
            }

        return ChatResponse(
            answer=str(state.get("final_answer", "") or ""),
            intent=str(state.get("intent", "") or ""),
            evidence_sufficient=bool(state.get("evidence_sufficient", False)),
            citations=list(state.get("citations", []) or []),
            context=str(state.get("final_context", "") or "") if request.include_context else None,
            errors=list(state.get("errors", []) or []),
            debug=debug,
        )

    # ── 问答（SSE 流式） ──

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        """SSE 流式问答接口。"""
        def event_generator():
            yield from run_streaming_workflow(
                request.query,
                config=cfg,
                vectorstore=None,
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── PDF 入库 ──

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        """批量解析 PDF 并写入 Qdrant。"""
        try:
            pdf_paths = resolve_pdf_paths(request.pdf_paths, request.raw_pdf_dir, request.max_files)
            vectorstore = None
            if not request.dry_run:
                vectorstore = get_vectorstore(hash_embedding=request.hash_embedding)
                vectorstore.create_collection(recreate=request.recreate_collection)
            builder = PDFIndexBuilder.from_config(cfg)
            builder.parser = PdfPlumberPDFParser(max_pages=request.max_pages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("FireAgent /ingest setup failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        files: list[IngestFileResult] = []
        total_chunks = 0
        for pdf_path in pdf_paths:
            result = IngestFileResult(path=str(pdf_path), status="running")
            try:
                chunks = builder.build_from_pdf(pdf_path)
                if vectorstore is not None:
                    vectorstore.upsert_chunks(chunks, batch_size=request.batch_size)
                result.status = "ok"
                result.chunks = len(chunks)
                total_chunks += len(chunks)
            except Exception as exc:  # noqa: BLE001 - 单文件失败不影响其他文件。
                logger.exception("Failed to ingest PDF: %s", pdf_path)
                result.status = "failed"
                result.error = str(exc)
            files.append(result)

        failed = sum(1 for item in files if item.status == "failed")
        return IngestResponse(
            status="ok" if failed == 0 else "partial_failed",
            collection=cfg.qdrant.collection,
            dry_run=request.dry_run,
            parsed_files=sum(1 for item in files if item.status == "ok"),
            total_chunks=total_chunks,
            failed_files=failed,
            files=files,
            errors=[item.error for item in files if item.error],
        )

    # ── 论文列表 ──

    @app.get("/papers")
    def list_papers(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        keyword: str = Query("", description="搜索关键词"),
        year: Optional[int] = Query(None, description="按年份筛选"),
        sort: str = Query("year_desc", description="排序：year_desc, year_asc, title"),
    ):
        """论文列表，支持搜索、筛选、分页。"""
        try:
            vs = get_vectorstore()
            papers_map = _query_papers_from_qdrant(vs)

            # 筛选
            papers = list(papers_map.values())
            if keyword:
                kw = keyword.lower()
                papers = [
                    p for p in papers
                    if kw in p["title"].lower()
                    or any(kw in a.lower() for a in p["authors"])
                    or any(kw in k.lower() for k in p["keywords"])
                    or kw in p.get("abstract", "").lower()
                ]
            if year is not None:
                papers = [p for p in papers if p.get("year") == year]

            # 排序
            if sort == "year_desc":
                papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
            elif sort == "year_asc":
                papers.sort(key=lambda p: p.get("year") or 0)
            elif sort == "title":
                papers.sort(key=lambda p: p.get("title", ""))

            total = len(papers)
            start = (page - 1) * page_size
            end = start + page_size
            page_papers = papers[start:end]

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "papers": page_papers,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("FireAgent /papers failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── 知识库统计（必须在 /papers/{doc_id} 之前注册，否则 "stats" 会被当作 doc_id）──

    @app.get("/papers/stats")
    def papers_stats():
        """知识库统计信息。"""
        try:
            vs = get_vectorstore()
            papers_map = _query_papers_from_qdrant(vs)
            papers = list(papers_map.values())

            total_papers = len(papers)
            total_chunks = sum(p.get("chunk_count", 0) for p in papers)

            # 年份分布
            year_counter: Counter = Counter()
            for p in papers:
                y = p.get("year")
                if y:
                    year_counter[y] += 1
            year_dist = dict(sorted(year_counter.items()))

            # 关键词频率
            keyword_counter: Counter = Counter()
            for p in papers:
                for kw in p.get("keywords", []):
                    if kw:
                        keyword_counter[kw] += 1
            top_keywords = dict(keyword_counter.most_common(30))

            return {
                "total_papers": total_papers,
                "total_chunks": total_chunks,
                "year_distribution": year_dist,
                "top_keywords": top_keywords,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("FireAgent /papers/stats failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── 论文详情 ──

    @app.get("/papers/{doc_id}")
    def get_paper(doc_id: str):
        """论文详情：元数据 + chunk 列表。"""
        try:
            vs = get_vectorstore()
            chunks = _query_chunks_by_doc_id(vs, doc_id)
            if not chunks:
                raise HTTPException(status_code=404, detail=f"论文 {doc_id} 未找到")

            first = chunks[0]
            payload = first.get("payload", {})
            return {
                "doc_id": doc_id,
                "title": payload.get("paper_title", ""),
                "authors": payload.get("authors", []),
                "year": payload.get("year"),
                "abstract": payload.get("abstract", ""),
                "keywords": payload.get("keywords", []),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "text": c.get("payload", {}).get("text", "")[:500],
                        "section_title": c.get("payload", {}).get("section_title", ""),
                        "page_start": c.get("payload", {}).get("page_start"),
                        "page_end": c.get("payload", {}).get("page_end"),
                        "chunk_type": c.get("payload", {}).get("chunk_type", ""),
                    }
                    for c in chunks
                ],
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("FireAgent /papers/{doc_id} failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ── 辅助函数 ──


def _query_papers_from_qdrant(vs: FireAgentQdrantClient) -> dict:
    """从 Qdrant 中查询所有论文，按 doc_id 聚合去重。"""
    from qdrant_client.models import FieldCondition, Filter, MatchExcept, MatchValue

    client = vs._create_client()
    collection = vs.config.qdrant.collection

    # scroll 取所有 points 的 payload
    papers: dict = {}
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            if not doc_id:
                continue
            if doc_id not in papers:
                papers[doc_id] = {
                    "doc_id": doc_id,
                    "title": payload.get("paper_title", ""),
                    "authors": payload.get("authors", []),
                    "year": payload.get("year"),
                    "abstract": payload.get("abstract", ""),
                    "keywords": payload.get("keywords", []),
                    "chunk_count": 0,
                }
            papers[doc_id]["chunk_count"] += 1
            # 补充可能缺失的字段
            if not papers[doc_id]["abstract"] and payload.get("abstract"):
                papers[doc_id]["abstract"] = payload["abstract"]
            if not papers[doc_id]["keywords"] and payload.get("keywords"):
                papers[doc_id]["keywords"] = payload["keywords"]

        if next_offset is None:
            break
        offset = next_offset

    return papers


def _query_chunks_by_doc_id(vs: FireAgentQdrantClient, doc_id: str) -> list:
    """从 Qdrant 中查询指定 doc_id 的所有 chunks。"""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = vs._create_client()
    collection = vs.config.qdrant.collection

    chunks = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        )
        points, next_offset = result
        for point in points:
            payload = point.payload or {}
            chunks.append({
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "payload": payload,
            })
        if next_offset is None:
            break
        offset = next_offset

    return chunks


def resolve_pdf_paths(
    pdf_paths: list[str],
    raw_pdf_dir: str = "data/raw_pdfs",
    max_files: Optional[int] = None,
) -> list[Path]:
    """解析 PDF 路径列表。"""
    paths: list[Path]
    if pdf_paths:
        paths = [resolve_project_path(path) for path in pdf_paths]
    else:
        directory = resolve_project_path(raw_pdf_dir)
        paths = sorted(directory.glob("*.pdf"))

    if max_files is not None:
        paths = paths[:max_files]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PDF 文件不存在：{missing[:3]}")
    return paths


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为项目根目录下的绝对路径。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


app = create_app()


def main() -> None:
    """本地启动 API 服务。"""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("缺少 uvicorn，请先安装依赖：pip install uvicorn") from exc
    uvicorn.run("fireagent.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
