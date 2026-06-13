"""SSE 流式问答实现。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Optional

from fireagent.graph.nodes import FireAgentGraphNodes, route_intent, generate_answer_from_state
from fireagent.graph.state import FireAgentState, create_initial_state
from fireagent.llm import BaseLLMClient, LLMMessage
from fireagent.prompts import PromptTemplateLoader
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


def run_streaming_workflow(
    user_query: str,
    config: Optional[FireAgentConfig] = None,
    vectorstore: Optional[FireAgentQdrantClient] = None,
    llm_client: Optional[BaseLLMClient] = None,
    session_id: str = "",
    conversation_context: str = "",
) -> Generator[str, None, None]:
    """流式运行 FireAgent 工作流，逐阶段返回 SSE 事件。

    不通过 LangGraph 编排，手动按序执行各节点，以便在每个阶段产出 SSE 事件。
    """
    cfg = config or get_config()
    state = create_initial_state(
        user_query,
        session_id=session_id,
        conversation_context=conversation_context,
    )
    start_time = time.time()

    # ── 1. 意图路由 ──
    intent, intent_reason = route_intent(user_query)
    state["intent"] = intent
    state["intent_reason"] = intent_reason
    if intent == "emergency":
        state["safety_notice"] = (
            "⚠️ 如果您正在经历火灾或紧急情况，请立即拨打 119 报警并撤离现场。"
        )
    yield _sse_event("stage", {"stage": "intent", "value": intent})

    # ── chat/reject 直接回答 ──
    if intent in ("chat", "reject"):
        fallback_answer = generate_answer_from_state(state)
        for char in fallback_answer:
            yield _sse_event("token", {"token": char})
        yield _sse_event("done", {
            "citations": [],
            "intent": intent,
            "evidence_sufficient": False,
            "elapsed": round(time.time() - start_time, 2),
        })
        return

    # ── 2. 查询改写 ──
    rewriter = QueryRewriter(config=cfg)
    rewrite_result = rewriter.rewrite(user_query)
    state["rewrite_result"] = rewrite_result
    state["rewritten_queries"] = rewrite_result.all_queries
    yield _sse_event("stage", {
        "stage": "rewrite",
        "queries": rewrite_result.all_queries[:5],
    })

    # ── 3. Dense + Sparse 并行检索 ──
    vs = vectorstore or FireAgentQdrantClient(config=cfg)
    dense_retriever = DenseRetriever(vs, config=cfg)
    sparse_retriever = SparseRetriever(vs, config=cfg)

    dense_results = dense_retriever.retrieve_many(rewrite_result)
    sparse_results = sparse_retriever.retrieve_many(rewrite_result)
    state["local_dense_results"] = dense_results
    state["local_sparse_results"] = sparse_results
    yield _sse_event("stage", {
        "stage": "retrieve",
        "dense_count": len(dense_results),
        "sparse_count": len(sparse_results),
    })

    # ── 4. RRF 融合 ──
    fusion = WeightedRRFFusion(config=cfg)
    fused = fusion.fuse(dense_results, sparse_results)
    state["fused_results"] = fused
    yield _sse_event("stage", {"stage": "fusion", "count": len(fused)})

    # ── 5. 重排 ──
    try:
        reranker = create_reranker(cfg)
    except Exception:  # noqa: BLE001
        reranker = LexicalReranker()
    reranked = reranker.rerank(
        rewrite_result.main_query,
        fused,
        top_k=cfg.retrieval.rerank_top_k,
    )
    state["reranked_results"] = reranked
    yield _sse_event("stage", {"stage": "rerank", "count": len(reranked)})

    # ── 6. 充分性检查 + Fallback 决策 ──
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
            "intent": intent,
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
            "intent": intent,
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
            "intent": intent,
            "evidence_sufficient": False,
            "fallback": {"action": fallback_decision.action.value, "reason": fallback_decision.reason},
            "elapsed": round(time.time() - start_time, 2),
        })
        return

    # ── 7. 联网搜索兜底 ──
    if fallback_decision.action == FallbackAction.USE_WEB:
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
            yield _sse_event("stage", {
                "stage": "web_search",
                "web_count": len(web_chunks),
            })
        except Exception as exc:  # noqa: BLE001
            state["errors"] = [f"联网搜索失败：{exc}"]
            yield _sse_event("stage", {"stage": "web_search", "error": str(exc)})

    # ── 8. 上下文构建 ──
    context_builder = ContextBuilder(config=cfg)
    context_result = context_builder.build(reranked, query=rewrite_result.main_query)
    state["context_result"] = context_result
    state["final_context"] = context_result.final_context
    state["citations"] = context_result.citations
    yield _sse_event("stage", {"stage": "context", "evidence_count": len(context_result.evidence_items)})

    # ── 9. 流式 LLM 回答 ──
    final_context = context_result.final_context or ""
    citations = list(context_result.citations or [])

    if not final_context or not cfg.llm.enabled:
        fallback_answer = generate_answer_from_state(state)
        for char in fallback_answer:
            yield _sse_event("token", {"token": char})
        state["final_answer"] = fallback_answer
    else:
        prompt_loader = PromptTemplateLoader(config=cfg)
        prompt = prompt_loader.render(
            "answer_generation",
            user_query=user_query,
            intent=intent,
            conversation_context=conversation_context,
            final_context=final_context,
            citations="\n".join(str(c) for c in citations),
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
                yield _sse_event("token", {"token": token})
        except Exception as exc:  # noqa: BLE001
            fallback_answer = generate_answer_from_state(state)
            for char in fallback_answer:
                yield _sse_event("token", {"token": char})
            state["final_answer"] = fallback_answer
            state["errors"] = [f"LLM 流式调用失败，已回退模板回答：{exc}"]

    # ── 10. 完成 ──
    fallback_info = {}
    fd = state.get("fallback_decision")
    if fd is not None:
        action = getattr(fd, "action", "")
        fallback_info = {
            "action": action.value if hasattr(action, "value") else str(action),
            "reason": getattr(fd, "reason", ""),
        }
    yield _sse_event("done", {
        "citations": citations,
        "intent": intent,
        "evidence_sufficient": bool(state.get("evidence_sufficient", False)),
        "safety_notice": str(state.get("safety_notice", "") or ""),
        "fallback": fallback_info,
        "elapsed": round(time.time() - start_time, 2),
    })
