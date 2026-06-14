"""SSE 流式问答实现。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Optional

from fireagent.graph.llm_router import LLMIntentRouter
from fireagent.graph.nodes import generate_answer_from_state
from fireagent.graph.state import FireAgentState, create_initial_state
from fireagent.llm import BaseLLMClient, LLMMessage
from fireagent.observability import TraceRecorder
from fireagent.prompts import PromptTemplateLoader
from fireagent.retrieval.citation_utils import build_used_citation_result
from fireagent.retrieval import (
    ContextBuilder,
    DenseRetriever,
    FallbackAction,
    LexicalReranker,
    LocalEvidenceSufficiencyChecker,
    QueryRewriter,
    SparseRetriever,
    WeightedRRFFusion,
    create_reranker,
    decide_fallback,
)
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient
from fireagent.websearch import TavilySearchClient, WebEvidenceBuilder, WebSearchResultParser


logger = logging.getLogger(__name__)


def _sse_event(event: str, data: dict) -> str:
    """构造 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _final_evidence_sufficient(answer: str, raw_sufficient: bool, used_citations: list) -> bool:
    """把检索充分性收敛为最终回答可展示的证据状态。"""
    if not raw_sufficient or not used_citations:
        return False
    insufficient_markers = ("证据不足", "无法回答", "不能回答")
    return not any(marker in answer for marker in insufficient_markers)


def run_streaming_workflow(
    user_query: str,
    config: Optional[FireAgentConfig] = None,
    vectorstore: Optional[FireAgentQdrantClient] = None,
    llm_client: Optional[BaseLLMClient] = None,
    session_id: str = "",
    conversation_context: str = "",
    long_term_memories: str = "",
    recorder: Optional[TraceRecorder] = None,
) -> Generator[str, None, None]:
    """流式运行 FireAgent 工作流，逐阶段返回 SSE 事件。

    不通过 LangGraph 编排，手动按序执行各节点，以便在每个阶段产出 SSE 事件。
    """
    cfg = config or get_config()
    state = create_initial_state(
        user_query,
        session_id=session_id,
        conversation_context=conversation_context,
        long_term_memories=long_term_memories,
    )
    start_time = time.time()

    # 发送 trace 事件
    if recorder and recorder.trace:
        yield _sse_event("trace", {"trace_id": recorder.trace.trace_id})

    # ── 1. 意图路由 ──
    if recorder:
        step_ctx = recorder.step("intent_router", current_goal="识别用户请求意图", tool_name="LLMIntentRouter.route")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()

    with step_ctx as step:
        route_result = LLMIntentRouter(config=cfg, llm_client=llm_client).route(
            query=user_query,
            conversation_context=conversation_context,
            long_term_memories=long_term_memories,
        )
        intent = route_result.intent
        state["intent"] = intent
        state["intent_reason"] = route_result.reason
        state["route_decision"] = route_result.model_dump()
        if intent == "emergency" or route_result.need_safety_notice:
            state["safety_notice"] = (
                "⚠️ 如果您正在经历火灾或紧急情况，请立即拨打 119 报警并撤离现场。"
            )
        if step:
            step.tool_result_summary = {
                "intent": route_result.intent,
                "sub_intent": route_result.sub_intent,
                "confidence": route_result.confidence,
                "source": route_result.source,
                "need_rag": route_result.need_rag,
                "need_memory": route_result.need_memory,
            }
            step.state_delta_summary = {"intent": intent, "sub_intent": route_result.sub_intent}

    yield _sse_event(
        "stage",
        {"stage": "intent", "value": intent, "route_decision": state["route_decision"]},
    )

    # ── chat/reject 直接回答 ──
    if intent in ("chat", "reject"):
        fallback_answer = generate_answer_from_state(state)
        for char in fallback_answer:
            yield _sse_event("token", {"token": char})
        done_data = {
            "citations": [],
            "used_citations": [],
            "used_citation_markers": [],
            "invalid_citation_markers": [],
            "intent": intent,
            "route_decision": state.get("route_decision", {}),
            "evidence_sufficient": False,
            "elapsed": round(time.time() - start_time, 2),
        }
        # 不在此处 finalize/save —— 由 server.py chat_stream() 统一处理
        yield _sse_event("done", done_data)
        return

    # ── 2. 查询改写 ──
    if recorder:
        step_ctx = recorder.step("query_rewrite", current_goal="对用户问题进行查询改写")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        rewriter = QueryRewriter(config=cfg)
        rewrite_result = rewriter.rewrite(user_query)
        state["rewrite_result"] = rewrite_result
        state["rewritten_queries"] = rewrite_result.all_queries
        if step:
            step.tool_result_summary = {"query_count": len(rewrite_result.all_queries), "main_query": rewrite_result.main_query}
    yield _sse_event("stage", {
        "stage": "rewrite",
        "queries": rewrite_result.all_queries[:5],
    })

    # ── 3. Dense + Sparse 并行检索 ──
    vs = vectorstore or FireAgentQdrantClient(config=cfg)
    dense_retriever = DenseRetriever(vs, config=cfg)
    sparse_retriever = SparseRetriever(vs, config=cfg)

    if recorder:
        step_ctx = recorder.step("dense_retrieve", current_goal="执行 dense 向量检索")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        dense_results = dense_retriever.retrieve_many(rewrite_result)
        if step:
            step.tool_result_summary = {"count": len(dense_results)}
    state["local_dense_results"] = dense_results

    if recorder:
        step_ctx = recorder.step("sparse_retrieve", current_goal="执行 sparse/BM25 检索")
    else:
        step_ctx = nullcontext()
    with step_ctx as step:
        sparse_results = sparse_retriever.retrieve_many(rewrite_result)
        if step:
            step.tool_result_summary = {"count": len(sparse_results)}
    state["local_sparse_results"] = sparse_results

    yield _sse_event("stage", {
        "stage": "retrieve",
        "dense_count": len(dense_results),
        "sparse_count": len(sparse_results),
    })

    # ── 4. RRF 融合 ──
    if recorder:
        step_ctx = recorder.step("fusion", current_goal="对 dense 与 sparse 结果做 RRF 融合")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        fusion = WeightedRRFFusion(config=cfg)
        fused = fusion.fuse(dense_results, sparse_results)
        state["fused_results"] = fused
        if step:
            step.tool_result_summary = {"fused_count": len(fused), "dense_weight": cfg.retrieval.dense_weight, "sparse_weight": cfg.retrieval.sparse_weight}
    yield _sse_event("stage", {"stage": "fusion", "count": len(fused)})

    # ── 5. 重排 ──
    if recorder:
        step_ctx = recorder.step("rerank", current_goal="对融合候选执行 cross-encoder 重排")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        rerank_fallback = False
        try:
            reranker = create_reranker(cfg)
        except Exception:  # noqa: BLE001
            reranker = LexicalReranker()
            rerank_fallback = True
        reranked = reranker.rerank(
            rewrite_result.main_query,
            fused,
            top_k=cfg.retrieval.rerank_top_k,
        )
        state["reranked_results"] = reranked
        if step:
            top_score = getattr(reranked[0], "final_score", None) if reranked else None
            step.tool_result_summary = {"reranked_count": len(reranked), "top_score": top_score, "fallback_to_lexical": rerank_fallback}
    yield _sse_event("stage", {"stage": "rerank", "count": len(reranked)})

    # ── 6. 充分性检查 + Fallback 决策 ──
    if recorder:
        step_ctx = recorder.step("sufficiency_check", current_goal="判断本地证据充分性")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        sufficiency_checker = LocalEvidenceSufficiencyChecker(config=cfg)
        sufficiency_result = sufficiency_checker.check(rewrite_result.main_query, reranked)
        fallback_decision = decide_fallback(
            query=rewrite_result.main_query,
            intent=intent,
            sufficiency=sufficiency_result,
            candidates=reranked,
            config=cfg,
        )
        state["evidence_sufficient"] = sufficiency_result.sufficient
        state["sufficiency_result"] = sufficiency_result
        state["fallback_decision"] = fallback_decision
        if step:
            step.tool_result_summary = {
                "sufficient": sufficiency_result.sufficient,
                "evidence_count": len(reranked),
                "fallback_action": fallback_decision.action.value,
            }
    yield _sse_event("stage", {
        "stage": "sufficiency",
        "sufficient": sufficiency_result.sufficient,
        "evidence_count": len(reranked),
        "fallback_action": fallback_decision.action.value,
        "fallback_reason": fallback_decision.reason,
    })

    # ── 6.5 处理特殊 fallback 动作 ──
    if fallback_decision.action == FallbackAction.REFUSE:
        refuse_answer = "抱歉，我无法回答涉及纵火、规避消防检查等危险行为的问题。如遇火灾紧急情况，请立即拨打 119。"
        for char in refuse_answer:
            yield _sse_event("token", {"token": char})
        yield _sse_event("done", {
            "citations": [],
            "used_citations": [],
            "used_citation_markers": [],
            "invalid_citation_markers": [],
            "intent": intent,
            "route_decision": state.get("route_decision", {}),
            "evidence_sufficient": False,
            "safety_notice": str(state.get("safety_notice", "") or ""),
            "fallback": {"action": fallback_decision.action.value, "reason": fallback_decision.reason},
            "elapsed": round(time.time() - start_time, 2),
        })
        return

    if fallback_decision.action == FallbackAction.ANSWER_INSUFFICIENT:
        insuff_answer = f"当前知识库中没有找到足够的本地论文证据来回答该问题。"
        for char in insuff_answer:
            yield _sse_event("token", {"token": char})
        yield _sse_event("done", {
            "citations": [],
            "used_citations": [],
            "used_citation_markers": [],
            "invalid_citation_markers": [],
            "intent": intent,
            "route_decision": state.get("route_decision", {}),
            "evidence_sufficient": False,
            "fallback": {"action": fallback_decision.action.value, "reason": fallback_decision.reason},
            "elapsed": round(time.time() - start_time, 2),
        })
        return

    if fallback_decision.action == FallbackAction.ASK_CLARIFY:
        clarify_answer = "这个问题里的指代还不够明确。请补充具体论文、事故、标准名称，或说明你希望我基于哪一批本地资料回答。"
        for char in clarify_answer:
            yield _sse_event("token", {"token": char})
        yield _sse_event("done", {
            "citations": [],
            "used_citations": [],
            "used_citation_markers": [],
            "invalid_citation_markers": [],
            "intent": intent,
            "route_decision": state.get("route_decision", {}),
            "evidence_sufficient": False,
            "fallback": {"action": fallback_decision.action.value, "reason": fallback_decision.reason},
            "elapsed": round(time.time() - start_time, 2),
        })
        return

    # ── 7. 联网搜索兜底 ──
    if fallback_decision.action == FallbackAction.USE_WEB:
        if recorder:
            step_ctx = recorder.step("web_search", current_goal="本地证据不足时联网搜索")
        else:
            from contextlib import nullcontext
            step_ctx = nullcontext()
        with step_ctx as step:
            try:
                response = TavilySearchClient(config=cfg).search(rewrite_result.main_query)
                web_chunks = WebSearchResultParser(config=cfg).parse_response(response)
                combined = WebEvidenceBuilder(
                    reranker=LexicalReranker(),
                    config=cfg,
                ).combine_and_rerank(
                    query=rewrite_result.main_query,
                    local_results=reranked,
                    web_chunks=web_chunks,
                    top_k=cfg.retrieval.rerank_top_k,
                )
                state["web_results"] = web_chunks
                state["reranked_results"] = combined
                reranked = combined
                if step:
                    step.tool_result_summary = {"web_count": len(web_chunks)}
                yield _sse_event("stage", {
                    "stage": "web_search",
                    "web_count": len(web_chunks),
                })
            except Exception as exc:  # noqa: BLE001
                state["errors"] = [f"联网搜索失败：{exc}"]
                if step:
                    step.status = "skipped"
                    step.error = str(exc)[:500]
                yield _sse_event("stage", {"stage": "web_search", "error": str(exc)})

    # ── 8. 上下文构建 ──
    if recorder:
        step_ctx = recorder.step("context_build", current_goal="构建最终回答上下文")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        context_builder = ContextBuilder(config=cfg)
        context_result = context_builder.build(reranked, query=rewrite_result.main_query)
        state["context_result"] = context_result
        state["final_context"] = context_result.final_context
        state["candidate_citations"] = context_result.candidate_citations
        candidate_citations = list(context_result.candidate_citations or [])
        if step:
            step.tool_result_summary = {
                "final_context_chars": len(context_result.final_context or ""),
                "candidate_citation_count": len(candidate_citations),
            }
    yield _sse_event("stage", {"stage": "context", "evidence_count": len(context_result.evidence_items)})

    # ── 9. 流式 LLM 回答 ──
    final_context = context_result.final_context or ""
    answer_parts: list[str] = []

    if recorder:
        step_ctx = recorder.step("answer_generate", current_goal="生成最终回答")
    else:
        from contextlib import nullcontext
        step_ctx = nullcontext()
    with step_ctx as step:
        if not final_context or not cfg.llm.enabled:
            fallback_answer = generate_answer_from_state(state)
            for char in fallback_answer:
                yield _sse_event("token", {"token": char})
            state["final_answer"] = fallback_answer
            answer_parts.append(fallback_answer)
            if step:
                step.tool_result_summary = {"answer_chars": len(fallback_answer), "llm_used": False}
        else:
            prompt_loader = PromptTemplateLoader(config=cfg)
            prompt = prompt_loader.render(
                "answer_generation",
                user_query=user_query,
                intent=intent,
                conversation_context=conversation_context,
                long_term_memories=long_term_memories,
                final_context=final_context,
                safety_notice=str(state.get("safety_notice", "") or ""),
            )

            llm = llm_client
            if llm is None:
                from fireagent.llm import create_llm_client
                llm = create_llm_client(cfg)

            try:
                for token in llm.generate_stream([
                    LLMMessage(
                        role="system",
                        content="你是 FireAgent，必须严格基于给定证据回答火灾领域问题。",
                    ),
                    LLMMessage(role="user", content=prompt),
                ]):
                    answer_parts.append(token)
                    yield _sse_event("token", {"token": token})
                if step:
                    step.tool_result_summary = {"answer_chars": len("".join(answer_parts)), "llm_used": True}
            except Exception as exc:  # noqa: BLE001
                fallback_answer = generate_answer_from_state(state)
                for char in fallback_answer:
                    yield _sse_event("token", {"token": char})
                state["final_answer"] = fallback_answer
                answer_parts.append(fallback_answer)
                state["errors"] = [f"LLM 流式调用失败，已回退模板回答：{exc}"]
                if step:
                    step.status = "failed"
                    step.error = str(exc)[:500]
                    step.tool_result_summary = {"answer_chars": len(fallback_answer), "llm_used": False}

    # ── 10. 解析 used citations ──
    full_answer = "".join(answer_parts)
    markers, used, used_strings, invalid = build_used_citation_result(
        full_answer,
        candidate_citations,
    )
    evidence_sufficient = _final_evidence_sufficient(
        full_answer,
        bool(state.get("evidence_sufficient", False)),
        used,
    )

    # ── 11. 完成 ──
    fallback_info = {}
    fd = state.get("fallback_decision")
    if fd is not None:
        action = getattr(fd, "action", "")
        fallback_info = {
            "action": action.value if hasattr(action, "value") else str(action),
            "reason": getattr(fd, "reason", ""),
        }
    yield _sse_event("done", {
        "citations": used_strings,
        "used_citations": [item.model_dump() for item in used],
        "used_citation_markers": markers,
        "invalid_citation_markers": invalid,
        "intent": intent,
        "route_decision": state.get("route_decision", {}),
        "evidence_sufficient": evidence_sufficient,
        "safety_notice": str(state.get("safety_notice", "") or ""),
        "fallback": fallback_info,
        "elapsed": round(time.time() - start_time, 2),
    })
