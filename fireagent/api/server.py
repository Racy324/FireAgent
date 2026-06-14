"""FireAgent FastAPI 服务入口。"""

from __future__ import annotations

import logging
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from fireagent import __version__
from fireagent.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationItem,
    HealthResponse,
    IngestFileResult,
    IngestRequest,
    IngestResponse,
    PaperDetail,
    PaperItem,
    PapersStats,
    MessageItem,
    SessionCreateRequest,
    SessionItem,
    SessionMessagesResponse,
    SessionsListResponse,
    SessionUpdateRequest,
)
from fireagent.api.streaming import run_streaming_workflow
from fireagent.graph.workflow import run_fireagent_workflow
from fireagent.ingestion import PDFIndexBuilder
from fireagent.ingestion.pdfplumber_parser import PdfPlumberPDFParser
from fireagent.memory import MemoryService
from fireagent.memory.long_term_service import LongTermMemoryService
from fireagent.observability import TraceRecorder, trace_context
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
    app.state.memory_service = None
    app.state.long_term_memory_service = None

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

    def get_memory_service() -> MemoryService | None:
        """读取会话历史服务。"""
        if not cfg.memory.enabled:
            return None
        if app.state.memory_service is None:
            app.state.memory_service = MemoryService(config=cfg)
        return app.state.memory_service

    def get_long_term_memory_service() -> LongTermMemoryService | None:
        """读取长期记忆服务。"""
        if not cfg.memory.long_term.enabled:
            return None
        if app.state.long_term_memory_service is None:
            app.state.long_term_memory_service = LongTermMemoryService(config=cfg)
        return app.state.long_term_memory_service

    @app.post("/sessions", response_model=SessionItem)
    def create_session(request: SessionCreateRequest) -> SessionItem:
        """创建新会话。"""
        memory = get_memory_service()
        if memory is None:
            raise HTTPException(status_code=503, detail="会话历史功能未启用。")
        session = memory.store.create_session(title=request.title)
        return _session_item(session)

    @app.get("/sessions", response_model=SessionsListResponse)
    def list_sessions() -> SessionsListResponse:
        """列出历史会话。"""
        memory = get_memory_service()
        if memory is None:
            return SessionsListResponse(sessions=[])
        return SessionsListResponse(sessions=[_session_item(item) for item in memory.store.list_sessions()])

    @app.patch("/sessions/{session_id}", response_model=SessionItem)
    def update_session(session_id: str, request: SessionUpdateRequest) -> SessionItem:
        """更新会话标题。"""
        memory = get_memory_service()
        if memory is None:
            raise HTTPException(status_code=503, detail="会话历史功能未启用。")
        session = memory.store.update_session_title(session_id, request.title)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        return _session_item(session)

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        """删除会话。"""
        memory = get_memory_service()
        if memory is not None:
            memory.store.delete_session(session_id)
        return None

    @app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
    def list_session_messages(session_id: str) -> SessionMessagesResponse:
        """读取会话消息。"""
        memory = get_memory_service()
        if memory is None:
            return SessionMessagesResponse(session_id=session_id, messages=[])
        session = memory.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        messages = memory.store.list_messages(session_id)
        return SessionMessagesResponse(
            session_id=session_id,
            messages=[_message_item(item) for item in messages],
        )

    def _make_recorder() -> TraceRecorder:
        """根据配置创建 TraceRecorder。"""
        obs_cfg = cfg.observability
        trace_dir = Path(obs_cfg.trace_dir)
        if not trace_dir.is_absolute():
            trace_dir = PROJECT_ROOT / trace_dir
        return TraceRecorder(
            trace_dir=trace_dir,
            enabled=obs_cfg.enabled,
            save_trace=obs_cfg.save_trace,
            save_index=obs_cfg.save_index,
            max_summary_chars=obs_cfg.max_summary_chars,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """在线问答接口。"""
        memory = get_memory_service()
        long_term_svc = get_long_term_memory_service()
        session_id = request.session_id or ""
        conversation_context = ""
        long_term_memories = ""
        short_term_result = None
        user_message_id = ""
        lt_memory_results = []

        recorder = _make_recorder()

        try:
            # ── 会话与上下文准备 ──
            if memory is not None:
                session = memory.get_or_create_session(request.session_id, query=request.query)
                session_id = session.session_id

            recorder.start_trace(
                endpoint="/chat",
                user_goal=request.query,
                session_id=session_id,
            )

            with recorder.step("api_receive", current_goal="接收用户问答请求", tool_name="FastAPI /chat") as step:
                step.tool_args_summary = {
                    "include_debug": request.include_debug,
                    "include_context": request.include_context,
                    "query_chars": len(request.query),
                }
                step.state_delta_summary = {"session_id_provided": bool(request.session_id)}

            if memory is not None:
                with recorder.step("short_term_context", current_goal="构建短期对话上下文") as step:
                    short_term_result = memory.build_short_term_context(session_id, query=request.query)
                    conversation_context = short_term_result.text
                    step.tool_result_summary = {
                        "total_chars": short_term_result.total_chars if short_term_result else 0,
                        "truncated": short_term_result.truncated if short_term_result else False,
                    }

                # 检索长期记忆
                if long_term_svc is not None:
                    with recorder.step(
                        "long_term_memory_retrieve",
                        current_goal="检索与当前问题相关的长期记忆",
                        tool_name="LongTermMemoryService.retrieve_for_query",
                    ) as step:
                        try:
                            lt_memory_results = long_term_svc.retrieve_for_query(request.query)
                            long_term_memories = long_term_svc.format_for_prompt(lt_memory_results)
                            step.tool_result_summary = {
                                "memory_count": len(lt_memory_results),
                                "top_score": lt_memory_results[0].final_score if lt_memory_results else None,
                            }
                            step.state_delta_summary = {"long_term_memories_chars": len(long_term_memories)}
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("长期记忆检索失败，跳过: %s", exc)
                            step.status = "skipped"
                            step.error = str(exc)[:500]

                with recorder.step("record_user_message", current_goal="保存用户消息到会话历史") as step:
                    user_message = memory.record_user_message(session_id, request.query)
                    user_message_id = user_message.message_id
                    step.tool_result_summary = {"user_message_id": user_message_id}

            # ── 执行工作流 ──
            with trace_context(recorder):
                state = run_fireagent_workflow(
                    request.query,
                    config=cfg,
                    vectorstore=None,
                    session_id=session_id,
                    conversation_context=conversation_context,
                    long_term_memories=long_term_memories,
                    long_term_memory_results=[r.model_dump() for r in lt_memory_results],
                )

            debug = {}
            if request.include_debug:
                debug = {
                    "rewritten_queries": state.get("rewritten_queries", []),
                    "dense_count": len(state.get("local_dense_results", []) or []),
                    "sparse_count": len(state.get("local_sparse_results", []) or []),
                    "fused_count": len(state.get("fused_results", []) or []),
                    "reranked_count": len(state.get("reranked_results", []) or []),
                    "web_count": len(state.get("web_results", []) or []),
                    "candidate_citation_count": len(state.get("candidate_citations", []) or []),
                    "used_citation_count": len(state.get("used_citations", []) or []),
                    "used_citation_markers": list(state.get("used_citation_markers", []) or []),
                    "invalid_citation_markers": list(state.get("invalid_citation_markers", []) or []),
                    "hallucination_warnings": state.get("hallucination_warnings", []),
                    "route_decision": state.get("route_decision", {}),
                    "conversation_context": conversation_context,
                    "long_term_memories": long_term_memories,
                    "long_term_memory_count": len(lt_memory_results),
                    "short_term_truncated": short_term_result.truncated if short_term_result else False,
                    "short_term_chars": short_term_result.total_chars if short_term_result else 0,
                }

            answer = str(state.get("final_answer", "") or "")
            used_citations_raw = list(state.get("used_citations", []) or [])
            used_citations = [_citation_item(item) for item in used_citations_raw]
            used_citation_markers = list(state.get("used_citation_markers", []) or [])
            invalid_citation_markers = list(state.get("invalid_citation_markers", []) or [])
            citations_strings = list(state.get("citations", []) or [])
            evidence_sufficient = _final_evidence_sufficient(
                answer,
                bool(state.get("evidence_sufficient", False)),
                used_citations_raw,
            )

            assistant_message_id = ""
            if memory is not None:
                with recorder.step("record_assistant_message", current_goal="保存助手回答到会话历史") as step:
                    assistant = memory.record_assistant_message(
                        session_id=session_id,
                        answer=answer,
                        intent=str(state.get("intent", "") or ""),
                        citations=citations_strings,
                        debug=debug,
                        metadata={
                            "user_message_id": user_message_id,
                            "evidence_sufficient": evidence_sufficient,
                            "used_citations": [item.model_dump() for item in used_citations_raw],
                            "used_citation_markers": used_citation_markers,
                            "invalid_citation_markers": invalid_citation_markers,
                            "errors": list(state.get("errors", []) or []),
                        },
                    )
                    assistant_message_id = assistant.message_id
                    step.tool_result_summary = {"assistant_message_id": assistant_message_id, "answer_chars": len(answer)}

                # 更新短期状态
                memory.update_short_term_state_after_turn(
                    session_id=session_id,
                    user_query=request.query,
                    assistant_answer=answer,
                    metadata={
                        "short_term_truncated": short_term_result.truncated if short_term_result else False,
                        "short_term_chars": short_term_result.total_chars if short_term_result else 0,
                    },
                )

                # 尝试写入长期记忆
                if long_term_svc is not None:
                    with recorder.step(
                        "long_term_memory_write",
                        current_goal="尝试写入长期记忆",
                        tool_name="LongTermMemoryService.maybe_write_after_turn",
                    ) as step:
                        try:
                            written = long_term_svc.maybe_write_after_turn(
                                session_id=session_id,
                                user_message_id=user_message_id,
                                assistant_message_id=assistant_message_id,
                                user_query=request.query,
                                assistant_answer=answer,
                            )
                            step.tool_result_summary = {"written_count": len(written)}
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("长期记忆写入失败，跳过: %s", exc)
                            step.status = "skipped"
                            step.error = str(exc)[:500]

            # ── 更新 trace 元数据并落盘 ──
            recorder.update_trace(
                intent=str(state.get("intent", "") or ""),
                sub_intent=str(state.get("route_decision", {}).get("sub_intent", "") or ""),
                evidence_sufficient=evidence_sufficient,
                used_citation_count=len(used_citations),
                normalized_goal=(
                    str(state.get("rewritten_queries", [""])[0])
                    if state.get("rewritten_queries")
                    else request.query
                ),
                model_version={
                    "llm": cfg.llm.model,
                    "router": cfg.router.model,
                    "embedding": cfg.embedding.model_name,
                    "reranker": cfg.reranker.model_name,
                },
                policy_version={
                    "router_mode": cfg.router.mode,
                    "web_fallback": cfg.rag.enable_web_fallback,
                    "memory_enabled": cfg.memory.enabled,
                    "long_term_memory_enabled": cfg.memory.long_term.enabled,
                },
            )

            with recorder.step("api_response", current_goal="返回 API 响应") as step:
                step.tool_result_summary = {"final_status": "success"}

            final_status = "degraded" if state.get("errors") else "success"
            recorder.finalize(
                final_status=final_status,
                message_id=assistant_message_id,
                final_eval={
                    "answer_chars": len(answer),
                    "used_citation_markers": used_citation_markers,
                    "invalid_citation_markers": invalid_citation_markers,
                    "hallucination_warning_count": len(state.get("hallucination_warnings", []) or []),
                },
            )
            recorder.save()

            return ChatResponse(
                answer=answer,
                session_id=session_id,
                message_id=assistant_message_id,
                trace_id=recorder.trace.trace_id if recorder.trace else "",
                intent=str(state.get("intent", "") or ""),
                evidence_sufficient=evidence_sufficient,
                citations=citations_strings,
                used_citations=used_citations,
                used_citation_markers=used_citation_markers,
                invalid_citation_markers=invalid_citation_markers,
                context=str(state.get("final_context", "") or "") if request.include_context else None,
                errors=list(state.get("errors", []) or []),
                debug=debug,
            )

        except Exception as exc:  # noqa: BLE001 - API 层统一转 HTTP 错误。
            logger.exception("FireAgent /chat failed")
            if recorder.trace is not None:
                recorder.finalize(final_status="failed", error=str(exc))
                recorder.save()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── 问答（SSE 流式） ──

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        """SSE 流式问答接口。"""
        def event_generator():
            memory = get_memory_service()
            long_term_svc = get_long_term_memory_service()
            session_id = request.session_id or ""
            conversation_context = ""
            long_term_memories = ""
            short_term_result = None
            user_message_id = ""
            lt_memory_results = []

            recorder = _make_recorder()

            try:
                # ── 会话与上下文准备 ──
                if memory is not None:
                    session = memory.get_or_create_session(request.session_id, query=request.query)
                    session_id = session.session_id

                recorder.start_trace(
                    endpoint="/chat/stream",
                    user_goal=request.query,
                    session_id=session_id,
                )

                with recorder.step("api_receive", current_goal="接收用户流式问答请求", tool_name="FastAPI /chat/stream") as step:
                    step.tool_args_summary = {
                        "query_chars": len(request.query),
                    }
                    step.state_delta_summary = {"session_id_provided": bool(request.session_id)}

                if memory is not None:
                    with recorder.step("short_term_context", current_goal="构建短期对话上下文") as step:
                        short_term_result = memory.build_short_term_context(session_id, query=request.query)
                        conversation_context = short_term_result.text
                        step.tool_result_summary = {
                            "total_chars": short_term_result.total_chars if short_term_result else 0,
                            "truncated": short_term_result.truncated if short_term_result else False,
                        }

                    if long_term_svc is not None:
                        with recorder.step(
                            "long_term_memory_retrieve",
                            current_goal="检索与当前问题相关的长期记忆",
                            tool_name="LongTermMemoryService.retrieve_for_query",
                        ) as step:
                            try:
                                lt_memory_results = long_term_svc.retrieve_for_query(request.query)
                                long_term_memories = long_term_svc.format_for_prompt(lt_memory_results)
                                step.tool_result_summary = {
                                    "memory_count": len(lt_memory_results),
                                    "top_score": lt_memory_results[0].final_score if lt_memory_results else None,
                                }
                                step.state_delta_summary = {"long_term_memories_chars": len(long_term_memories)}
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("长期记忆检索失败，跳过: %s", exc)
                                step.status = "skipped"
                                step.error = str(exc)[:500]

                    with recorder.step("record_user_message", current_goal="保存用户消息到会话历史") as step:
                        user_msg = memory.record_user_message(session_id, request.query)
                        user_message_id = user_msg.message_id
                        step.tool_result_summary = {"user_message_id": user_message_id}

                # ── 流式工作流（通过 trace_context 设置 ContextVar） ──
                answer_parts: list[str] = []
                last_done: dict = {}

                with trace_context(recorder):
                    for event_text in run_streaming_workflow(
                        request.query,
                        config=cfg,
                        vectorstore=None,
                        session_id=session_id,
                        conversation_context=conversation_context,
                        long_term_memories=long_term_memories,
                        recorder=recorder,
                    ):
                        event_name, payload = _parse_sse_event(event_text)
                        if event_name == "trace":
                            yield event_text
                            continue
                        if event_name == "token":
                            answer_parts.append(str(payload.get("token", "")))
                        if event_name == "done":
                            last_done = payload
                            assistant_message_id = ""
                            if memory is not None:
                                with recorder.step("record_assistant_message", current_goal="保存助手回答到会话历史") as step:
                                    assistant = memory.record_assistant_message(
                                        session_id=session_id,
                                        answer="".join(answer_parts),
                                        intent=str(payload.get("intent", "") or ""),
                                        citations=list(payload.get("citations", []) or []),
                                        metadata={
                                            "evidence_sufficient": bool(payload.get("evidence_sufficient", False)),
                                            "used_citations": list(payload.get("used_citations", []) or []),
                                            "used_citation_markers": list(payload.get("used_citation_markers", []) or []),
                                            "invalid_citation_markers": list(payload.get("invalid_citation_markers", []) or []),
                                            "fallback": payload.get("fallback", {}),
                                        },
                                    )
                                    assistant_message_id = assistant.message_id
                                    step.tool_result_summary = {"assistant_message_id": assistant_message_id, "answer_chars": len("".join(answer_parts))}

                                memory.update_short_term_state_after_turn(
                                    session_id=session_id,
                                    user_query=request.query,
                                    assistant_answer="".join(answer_parts),
                                    metadata={
                                        "short_term_truncated": short_term_result.truncated if short_term_result else False,
                                        "short_term_chars": short_term_result.total_chars if short_term_result else 0,
                                    },
                                )

                                if long_term_svc is not None:
                                    with recorder.step(
                                        "long_term_memory_write",
                                        current_goal="尝试写入长期记忆",
                                        tool_name="LongTermMemoryService.maybe_write_after_turn",
                                    ) as step:
                                        try:
                                            written = long_term_svc.maybe_write_after_turn(
                                                session_id=session_id,
                                                user_message_id=user_message_id,
                                                assistant_message_id=assistant_message_id,
                                                user_query=request.query,
                                                assistant_answer="".join(answer_parts),
                                            )
                                            step.tool_result_summary = {"written_count": len(written)}
                                        except Exception as exc:  # noqa: BLE001
                                            logger.warning("长期记忆写入失败，跳过: %s", exc)
                                            step.status = "skipped"
                                            step.error = str(exc)[:500]

                            last_done["session_id"] = session_id
                            last_done["message_id"] = assistant_message_id
                            last_done["trace_id"] = recorder.trace.trace_id if recorder.trace else ""

                            # 统一落盘 trace（唯一落盘点）
                            recorder.update_trace(
                                intent=str(payload.get("intent", "") or ""),
                                evidence_sufficient=bool(payload.get("evidence_sufficient", False)),
                            )
                            with recorder.step("api_response", current_goal="返回流式 API 响应") as step:
                                step.tool_result_summary = {"final_status": "success"}
                            recorder.finalize(final_status="success", message_id=assistant_message_id)
                            recorder.save()

                            yield _format_sse_event("done", last_done)
                        else:
                            yield event_text

            except Exception as exc:  # noqa: BLE001
                if recorder.trace is not None:
                    recorder.finalize(final_status="failed", error=str(exc))
                    recorder.save()
                logger.exception("FireAgent /chat/stream failed")

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


def _citation_item(citation: Any) -> CitationItem:
    """把结构化引用转成 API 响应。"""
    if hasattr(citation, "model_dump"):
        return CitationItem(**citation.model_dump())
    if isinstance(citation, dict):
        return CitationItem(**citation)
    raise TypeError(f"Unsupported citation type: {type(citation)}")


def _session_item(session) -> SessionItem:
    """把存储层 session 转成 API 响应。"""
    return SessionItem(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_item(message) -> MessageItem:
    """把存储层 message 转成 API 响应。"""
    return MessageItem(
        message_id=message.message_id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        intent=message.intent,
        citations=message.citations,
        metadata=message.metadata,
    )


def _parse_sse_event(event_text: str) -> tuple[str, dict]:
    """解析当前内部 SSE 文本。"""
    event_name = ""
    payload: dict = {}
    for line in event_text.splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ").strip()
        elif line.startswith("data: "):
            raw = line.removeprefix("data: ")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
    return event_name, payload


def _format_sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _final_evidence_sufficient(answer: str, raw_sufficient: bool, used_citations: list) -> bool:
    """把检索充分性收敛为最终回答可展示的证据状态。"""
    if not raw_sufficient or not used_citations:
        return False
    insufficient_markers = ("证据不足", "无法回答", "不能回答")
    return not any(marker in answer for marker in insufficient_markers)


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
