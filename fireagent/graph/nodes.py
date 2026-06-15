"""FireAgent LangGraph 节点实现。"""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

from fireagent.graph.llm_router import (
    LLMIntentRouter,
    is_history_query,
    is_memory_query,
    is_preference_instruction,
)
from fireagent.graph.state import FireAgentState, state_get_query
from fireagent.llm import BaseLLMClient, LLMMessage, create_llm_client
from fireagent.prompts import PromptTemplateLoader
from fireagent.retrieval.citation_utils import build_used_citation_result
from fireagent.retrieval import (
    ContextBuilder,
    DenseRetriever,
    FallbackAction,
    FallbackDecision,
    FallbackPolicy,
    LexicalReranker,
    LocalEvidenceSufficiencyChecker,
    QueryRewriter,
    RegularRAGCandidateFilter,
    SparseRetriever,
    WeightedRRFFusion,
    create_reranker,
    decide_fallback,
)
from fireagent.retrieval.reranker import BaseReranker
from fireagent.retrieval.schema import RerankedRetrievalResult
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient
from fireagent.websearch import TavilySearchClient, WebEvidenceBuilder, WebSearchResultParser


CHAT_PATTERNS = ("你好", "您好", "hello", "hi", "你是谁", "谢谢", "感谢")
PREFERENCE_INSTRUCTION_PATTERNS = (
    "默认",
    "我希望",
    "我不想",
    "固定采用",
    "一直用",
    "始终",
    "一律",
    "统一",
    "全部用",
    "都用",
    "偏好是",
    "偏好设置",
)
PREFERENCE_INSTRUCTION_REGEX = re.compile(
    r"(以后|默认|始终|一律|统一|全部|都).{0,30}(笔记|回答|输出|语言|中文|英文|格式)"
)
MEMORY_QUERY_PATTERNS = (
    "我的偏好",
    "偏好有什么",
    "偏好是什么",
    "长期记忆",
    "记住了什么",
    "你记得什么",
    "我设置过",
    "我之前设置",
    "应该用什么格式",
    "用什么格式",
    "用什么语言",
    "标题前缀",
    "标题前加",
)
MEMORY_QUERY_REGEX = re.compile(
    r"(笔记|回答|输出|标题|格式|语言).{0,20}(应该|需要|用什么|怎么|如何|是什么)"
)
HISTORY_QUERY_PATTERNS = (
    "之前问过",
    "刚才问过",
    "前面问过",
    "上次问过",
    "上一条问题",
    "历史问题",
    "问过什么",
    "刚才的问题",
)
PAPER_PATTERNS = ("总结", "综述", "概括", "对比", "比较", "论文", "文献", "作者", "摘要")
EMERGENCY_PATTERNS = ("怎么办", "如何逃生", "应急", "报警", "疏散路线", "自救", "逃生", "灭火器")
FIRE_DOMAIN_PATTERNS = (
    "火灾",
    "消防",
    "烟气",
    "烟雾",
    "火焰",
    "燃烧",
    "疏散",
    "排烟",
    "森林火",
    "隧道",
    "防火",
    "灭火",
    "火警",
)


@dataclass
class FireAgentNodeContext:
    """节点依赖容器，用于注入向量库、重排器和配置。"""

    config: FireAgentConfig
    vectorstore: Optional[FireAgentQdrantClient] = None
    reranker: Optional[BaseReranker] = None
    llm_client: Optional[BaseLLMClient] = None


class FireAgentGraphNodes:
    """FireAgent 在线问答工作流节点集合。"""

    def __init__(
        self,
        config: Optional[FireAgentConfig] = None,
        vectorstore: Optional[FireAgentQdrantClient] = None,
        reranker: Optional[BaseReranker] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.context = FireAgentNodeContext(
            config=config or get_config(),
            vectorstore=vectorstore,
            reranker=reranker,
            llm_client=llm_client,
        )
        self.query_rewriter = QueryRewriter(config=self.context.config)
        self.fusion = WeightedRRFFusion(config=self.context.config)
        self.sufficiency_checker = LocalEvidenceSufficiencyChecker(config=self.context.config)
        self.fallback_policy = FallbackPolicy(config=self.context.config)
        self.context_builder = ContextBuilder(config=self.context.config)
        self.web_parser = WebSearchResultParser(config=self.context.config)
        self.candidate_filter = RegularRAGCandidateFilter(config=self.context.config)
        self.prompt_loader = PromptTemplateLoader(config=self.context.config)
        self.intent_router = LLMIntentRouter(config=self.context.config)

    @property
    def vectorstore(self) -> FireAgentQdrantClient:
        """懒加载 Qdrant 向量库客户端。"""
        if self.context.vectorstore is None:
            self.context.vectorstore = FireAgentQdrantClient(config=self.context.config)
        return self.context.vectorstore

    @property
    def reranker(self) -> BaseReranker:
        """懒加载 reranker；模型不可用时 create_reranker 会回退到词项重排。"""
        if self.context.reranker is None:
            self.context.reranker = create_reranker(config=self.context.config)
        return self.context.reranker

    @property
    def llm_client(self) -> BaseLLMClient:
        """懒加载 LLM 客户端。"""
        if self.context.llm_client is None:
            self.context.llm_client = create_llm_client(config=self.context.config)
        return self.context.llm_client

    def intent_router_node(self, state: FireAgentState) -> FireAgentState:
        """识别用户问题意图，并写入 intent。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("intent_router", current_goal="识别用户请求意图", tool_name="LLMIntentRouter.route")
            if recorder else nullcontext()
        )

        query = state_get_query(state)
        with step_ctx as step:
            route_result = self.intent_router.route(
                query=query,
                conversation_context=str(state.get("conversation_context", "") or ""),
                long_term_memories=str(state.get("long_term_memories", "") or ""),
            )
            intent = route_result.intent
            reason = route_result.reason
            safety_notice = ""
            if intent == "emergency" or route_result.need_safety_notice:
                safety_notice = "安全提醒：如现场存在明火、浓烟、爆炸或人员受困，请立即拨打 119，并优先撤离到安全区域。"
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
        return FireAgentState(
            intent=intent,
            intent_reason=reason,
            route_decision=route_result.model_dump(),
            safety_notice=safety_notice,
        )

    def query_rewrite_node(self, state: FireAgentState) -> FireAgentState:
        """对用户问题进行查询改写。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("query_rewrite", current_goal="对用户问题进行查询改写")
            if recorder else nullcontext()
        )

        query = state_get_query(state)
        with step_ctx as step:
            rewrite_result = self.query_rewriter.rewrite(query)
            if step:
                step.tool_result_summary = {"query_count": len(rewrite_result.all_queries), "main_query": rewrite_result.main_query}
        return FireAgentState(
            rewrite_result=rewrite_result,
            rewritten_queries=rewrite_result.all_queries,
        )

    def dense_retrieve_node(self, state: FireAgentState) -> FireAgentState:
        """执行 dense 检索。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("dense_retrieve", current_goal="执行 dense 向量检索")
            if recorder else nullcontext()
        )

        try:
            with step_ctx as step:
                rewrite_result = state.get("rewrite_result")
                if rewrite_result is None:
                    rewrite_result = self.query_rewriter.rewrite(state_get_query(state))
                results = DenseRetriever(self.vectorstore, config=self.context.config).retrieve_many(rewrite_result)
                if step:
                    top_score = getattr(results[0], "final_score", None) if results else None
                    step.tool_result_summary = {"count": len(results), "top_score": top_score}
            return FireAgentState(local_dense_results=results)
        except Exception as exc:  # noqa: BLE001 - 外部服务与模型错误统一写入状态。
            return FireAgentState(local_dense_results=[], errors=[f"dense_retrieve_node 失败：{exc}"])

    def sparse_retrieve_node(self, state: FireAgentState) -> FireAgentState:
        """执行 sparse/BM25 检索。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("sparse_retrieve", current_goal="执行 sparse/BM25 检索")
            if recorder else nullcontext()
        )

        try:
            with step_ctx as step:
                rewrite_result = state.get("rewrite_result")
                if rewrite_result is None:
                    rewrite_result = self.query_rewriter.rewrite(state_get_query(state))
                results = SparseRetriever(self.vectorstore, config=self.context.config).retrieve_many(rewrite_result)
                if step:
                    step.tool_result_summary = {"count": len(results)}
            return FireAgentState(local_sparse_results=results)
        except Exception as exc:  # noqa: BLE001
            return FireAgentState(local_sparse_results=[], errors=[f"sparse_retrieve_node 失败：{exc}"])

    def fusion_node(self, state: FireAgentState) -> FireAgentState:
        """对 dense 与 sparse 结果做 Weighted RRF 融合。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("fusion", current_goal="对 dense 与 sparse 结果做 RRF 融合")
            if recorder else nullcontext()
        )

        with step_ctx as step:
            query = get_main_query(state)
            dense_results = list(state.get("local_dense_results", []) or [])
            sparse_results = list(state.get("local_sparse_results", []) or [])

            dense_results, dense_filter_stats = self.candidate_filter.filter_many(query, dense_results)
            sparse_results, sparse_filter_stats = self.candidate_filter.filter_many(query, sparse_results)

            fused = self.fusion.fuse(dense_results, sparse_results)
            if step:
                step.tool_result_summary = {
                    "chunk_type_filter_mode": dense_filter_stats.mode,
                    "filter_bypassed": dense_filter_stats.bypassed or sparse_filter_stats.bypassed,
                    "dense_input_count": dense_filter_stats.input_count,
                    "dense_output_count": dense_filter_stats.output_count,
                    "dense_filtered_references": dense_filter_stats.filtered_references,
                    "dense_filtered_short_figure_captions": dense_filter_stats.filtered_short_figure_captions,
                    "sparse_input_count": sparse_filter_stats.input_count,
                    "sparse_output_count": sparse_filter_stats.output_count,
                    "sparse_filtered_references": sparse_filter_stats.filtered_references,
                    "sparse_filtered_short_figure_captions": sparse_filter_stats.filtered_short_figure_captions,
                    "fused_count": len(fused),
                    "dense_weight": self.context.config.retrieval.dense_weight,
                    "sparse_weight": self.context.config.retrieval.sparse_weight,
                }
        return FireAgentState(fused_results=fused)

    def rerank_node(self, state: FireAgentState) -> FireAgentState:
        """对融合候选执行 cross-encoder 重排。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("rerank", current_goal="对融合候选执行 cross-encoder 重排")
            if recorder else nullcontext()
        )

        query = get_main_query(state)
        fused = list(state.get("fused_results", []) or [])
        if not fused:
            return FireAgentState(reranked_results=[])
        with step_ctx as step:
            rerank_fallback = False
            try:
                reranked = self.reranker.rerank(
                    query,
                    fused,
                    top_k=self.context.config.retrieval.rerank_top_k,
                )
            except Exception as exc:  # noqa: BLE001
                rerank_fallback = True
                reranked = LexicalReranker().rerank(
                    query,
                    fused,
                    top_k=self.context.config.retrieval.rerank_top_k,
                )
                if step:
                    step.error = str(exc)[:500]
            if step:
                top_score = getattr(reranked[0], "final_score", None) if reranked else None
                step.tool_result_summary = {"reranked_count": len(reranked), "top_score": top_score, "fallback_to_lexical": rerank_fallback}
        if rerank_fallback:
            return FireAgentState(
                reranked_results=reranked,
                errors=[f"rerank_node 使用 fallback"],
            )
        return FireAgentState(reranked_results=reranked)

    def sufficiency_check_node(self, state: FireAgentState) -> FireAgentState:
        """判断本地证据是否足够，并运行 fallback policy 决策。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("sufficiency_check", current_goal="判断本地证据充分性")
            if recorder else nullcontext()
        )

        with step_ctx as step:
            query = get_main_query(state)
            intent = str(state.get("intent", "") or "")
            reranked = list(state.get("reranked_results", []) or [])
            sufficiency_result = self.sufficiency_checker.check(query, reranked)
            fallback_decision = self.fallback_policy.decide(
                query=query,
                intent=intent,
                sufficiency=sufficiency_result,
                candidates=reranked,
            )
            if step:
                step.tool_result_summary = {
                    "sufficient": sufficiency_result.sufficient,
                    "evidence_count": len(reranked),
                    "fallback_action": fallback_decision.action.value,
                }
        return FireAgentState(
            evidence_sufficient=sufficiency_result.sufficient,
            sufficiency_result=sufficiency_result,
            fallback_decision=fallback_decision,
        )

    def web_search_node(self, state: FireAgentState) -> FireAgentState:
        """本地证据不足时执行 Tavily 联网搜索，并与本地证据统一重排。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("web_search", current_goal="本地证据不足时联网搜索")
            if recorder else nullcontext()
        )

        query = get_main_query(state)
        local_results = list(state.get("reranked_results", []) or [])
        if not self.context.config.rag.enable_web_fallback:
            return FireAgentState(
                web_results=[],
                reranked_results=local_results,
                errors=["联网兜底已被配置关闭。"],
            )

        with step_ctx as step:
            try:
                response = TavilySearchClient(config=self.context.config).search(query)
                web_chunks = self.web_parser.parse_response(response)
                combined = WebEvidenceBuilder(
                    reranker=LexicalReranker(),
                    config=self.context.config,
                ).combine_and_rerank(
                    query=query,
                    local_results=local_results,
                    web_chunks=web_chunks,
                    top_k=self.context.config.retrieval.rerank_top_k,
                )
                if step:
                    step.tool_result_summary = {"web_count": len(web_chunks)}
                return FireAgentState(web_results=web_chunks, reranked_results=combined)
            except Exception as exc:  # noqa: BLE001
                if step:
                    step.status = "skipped"
                    step.error = str(exc)[:500]
                return FireAgentState(
                    web_results=[],
                    reranked_results=local_results,
                    errors=[f"web_search_node 失败：{exc}"],
                )

    def context_build_node(self, state: FireAgentState) -> FireAgentState:
        """构建最终回答上下文和引用列表。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("context_build", current_goal="构建最终回答上下文")
            if recorder else nullcontext()
        )

        with step_ctx as step:
            query = get_main_query(state)
            reranked = list(state.get("reranked_results", []) or [])
            context_result = self.context_builder.build(reranked, query=query)
            if step:
                step.tool_result_summary = {
                    "final_context_chars": len(context_result.final_context or ""),
                    "candidate_citation_count": len(context_result.candidate_citations or []),
                }
        return FireAgentState(
            context_result=context_result,
            final_context=context_result.final_context,
            citations=context_result.citations,
            candidate_citations=context_result.candidate_citations,
        )

    def answer_generate_node(self, state: FireAgentState) -> FireAgentState:
        """生成最终回答。

        优先使用正式 LLM 客户端；如果 LLM 未启用、配置缺失或调用失败，则回退到
        证据模板生成，保证系统仍可返回有边界的答案。
        """
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("answer_generate", current_goal="生成最终回答")
            if recorder else nullcontext()
        )

        with step_ctx as step:
            fallback_answer = generate_answer_from_state(state)
            intent = str(state.get("intent", "") or "")
            final_context = str(state.get("final_context", "") or "").strip()

            # 检查 fallback decision，处理特殊动作
            fallback_decision = state.get("fallback_decision")
            if fallback_decision is not None:
                action = getattr(fallback_decision, "action", None)
                reason = getattr(fallback_decision, "reason", "")
                if action == FallbackAction.REFUSE:
                    if step:
                        step.tool_result_summary = {"answer_chars": 53, "llm_used": False, "fallback_action": "REFUSE"}
                    return FireAgentState(
                        final_answer="抱歉，我无法回答涉及纵火、规避消防检查等危险行为的问题。如遇火灾紧急情况，请立即拨打 119。",
                        citations=[],
                        used_citations=[],
                        used_citation_markers=[],
                        invalid_citation_markers=[],
                    )
                if action == FallbackAction.ANSWER_INSUFFICIENT:
                    if step:
                        step.tool_result_summary = {"llm_used": False, "fallback_action": "ANSWER_INSUFFICIENT"}
                    return FireAgentState(
                        final_answer=f"当前知识库中没有找到足够的本地论文证据来回答该问题。{f'（原因：{reason}）' if reason else ''}",
                        citations=[],
                        used_citations=[],
                        used_citation_markers=[],
                        invalid_citation_markers=[],
                    )
                if action == FallbackAction.ASK_CLARIFY:
                    if step:
                        step.tool_result_summary = {"llm_used": False, "fallback_action": "ASK_CLARIFY"}
                    return FireAgentState(
                        final_answer="这个问题里的指代还不够明确。请补充具体论文、事故、标准名称，或说明你希望我基于哪一批本地资料回答。",
                        citations=[],
                        used_citations=[],
                        used_citation_markers=[],
                        invalid_citation_markers=[],
                    )

            if intent in {"chat", "reject"} or not final_context or not self.context.config.llm.enabled:
                if step:
                    step.tool_result_summary = {"answer_chars": len(fallback_answer), "llm_used": False}
                return FireAgentState(
                    final_answer=fallback_answer,
                    citations=[],
                    used_citations=[],
                    used_citation_markers=[],
                    invalid_citation_markers=[],
                )

            try:
                prompt = self.prompt_loader.render(
                    "answer_generation",
                    user_query=state_get_query(state),
                    intent=intent,
                    conversation_context=str(state.get("conversation_context", "") or ""),
                    long_term_memories=str(state.get("long_term_memories", "") or ""),
                    final_context=final_context,
                    safety_notice=str(state.get("safety_notice", "") or ""),
                )
                response = self.llm_client.generate(
                    [
                        LLMMessage(
                            role="system",
                            content="你是 FireAgent，必须严格基于给定证据回答火灾领域问题。",
                        ),
                        LLMMessage(role="user", content=prompt),
                    ]
                )
                answer = response.content.strip()
                markers, used, used_strings, invalid = build_used_citation_result(
                    answer,
                    list(state.get("candidate_citations", []) or []),
                )
                if step:
                    step.tool_result_summary = {"answer_chars": len(answer), "llm_used": True, "used_marker_count": len(markers)}
                return FireAgentState(
                    final_answer=answer,
                    citations=used_strings,
                    used_citation_markers=markers,
                    used_citations=used,
                    invalid_citation_markers=invalid,
                )
            except Exception as exc:  # noqa: BLE001 - LLM 失败时回退模板回答。
                if step:
                    step.status = "failed"
                    step.error = str(exc)[:500]
                    step.tool_result_summary = {"answer_chars": len(fallback_answer), "llm_used": False}
                return FireAgentState(
                    final_answer=fallback_answer,
                    citations=[],
                    used_citations=[],
                    used_citation_markers=[],
                    invalid_citation_markers=[],
                    errors=[f"answer_generate_node LLM 回退：{exc}"],
                )

    def hallucination_check_node(self, state: FireAgentState) -> FireAgentState:
        """执行轻量幻觉检查，确保无证据时明确说明不足。"""
        from fireagent.observability import get_current_trace_recorder

        recorder = get_current_trace_recorder()
        step_ctx = (
            recorder.step("hallucination_check", current_goal="轻量幻觉检测")
            if recorder else nullcontext()
        )

        with step_ctx as step:
            warnings: list[str] = []
            intent = str(state.get("intent", "") or "")
            answer = str(state.get("final_answer", "") or "")
            final_context = str(state.get("final_context", "") or "")
            if intent not in {"chat", "reject"} and not final_context and "证据不足" not in answer:
                warnings.append("回答缺少证据不足提示，已在节点中标记。")
                answer = f"{answer}\n\n不确定性：当前证据不足，不能给出确定结论。"
            if step:
                step.tool_result_summary = {"warning_count": len(warnings)}
        return FireAgentState(final_answer=answer, hallucination_warnings=warnings)


def route_intent(query: str) -> tuple[str, str]:
    """基于规则识别用户意图。"""
    result = LLMIntentRouter().route(query=query)
    return result.intent, result.reason


def get_main_query(state: FireAgentState) -> str:
    """优先返回改写后的主查询，否则返回原始问题。"""
    rewrite_result = state.get("rewrite_result")
    if rewrite_result is not None and getattr(rewrite_result, "main_query", ""):
        return str(rewrite_result.main_query)
    return state_get_query(state)


def generate_answer_from_state(state: FireAgentState) -> str:
    """根据状态生成模板化回答。"""
    query = state_get_query(state)
    intent = str(state.get("intent", "") or "")
    if intent == "chat":
        if _is_history_query(query):
            return _answer_history_query(
                query,
                str(state.get("conversation_context", "") or ""),
            )
        if _is_preference_instruction(query):
            return _answer_preference_instruction(query)
        if _is_memory_query(query):
            return _answer_memory_query(
                query,
                str(state.get("long_term_memories", "") or ""),
            )
        return "你好，我是 FireAgent，可以围绕火灾论文知识库回答火灾机理、烟气控制、疏散、检测预警和消防安全等问题。"
    if intent == "reject":
        return "抱歉，我主要回答火灾与消防安全领域问题。你可以把问题改写为与火灾机理、消防管理、疏散、检测预警或应急安全相关的方向。"

    final_context = str(state.get("final_context", "") or "").strip()
    citations = list(state.get("citations", []) or [])
    safety_notice = str(state.get("safety_notice", "") or "").strip()
    sufficiency = state.get("sufficiency_result")
    sufficient = bool(state.get("evidence_sufficient", False))
    errors = list(state.get("errors", []) or [])

    if not final_context:
        parts = [
            "1. 直接回答\n证据不足，当前本地论文证据和联网资料都不足以可靠回答该问题。",
            "2. 依据说明\n未检索到可用于支撑结论的有效 evidence chunk。",
        ]
        if safety_notice:
            parts.append(f"3. 安全提醒\n{safety_notice}")
        if errors:
            parts.append(f"4. 不确定性\n检索过程中出现错误：{'；'.join(errors[:3])}")
        return "\n\n".join(parts)

    evidence_summary = summarize_context(final_context)
    source_text = "\n".join(f"- {citation}" for citation in citations) if citations else "- 暂无结构化引用"
    suff_reason = getattr(sufficiency, "reason", "") if sufficiency is not None else ""
    uncertainty = "本地证据已通过充分性检查。" if sufficient else "本地证据不足，回答中已结合联网资料或保留不确定性。"

    parts = [
        f"1. 直接回答\n针对“{query}”，可依据检索到的证据做如下概括：{evidence_summary}",
        "2. 依据说明\n以上内容来自最终上下文中的本地论文证据和/或联网资料证据，未使用未给出的外部结论。",
        "3. 补充分析\n如果问题涉及最新政策、标准、法规或近期事故，应优先核对联网资料和官方发布源；本地论文更适合支撑机理、模型和实验结论。",
        f"4. 资料来源\n{source_text}",
        f"5. 不确定性\n{uncertainty}{f' {suff_reason}' if suff_reason else ''}",
    ]
    if safety_notice:
        parts.insert(3, f"4. 安全提醒\n{safety_notice}")
        parts[-2] = parts[-2].replace("4. 资料来源", "5. 资料来源")
        parts[-1] = parts[-1].replace("5. 不确定性", "6. 不确定性")
    return "\n\n".join(parts)


def summarize_context(context: str, max_chars: int = 450) -> str:
    """从最终上下文中抽取简短摘要，不额外编造信息。"""
    lines = []
    for line in context.splitlines():
        stripped = line.strip()
        if stripped.startswith("正文："):
            stripped = stripped.removeprefix("正文：").strip()
            if stripped:
                lines.append(stripped)
    text = "；".join(lines) if lines else re.sub(r"\s+", " ", context).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _is_history_query(query: str) -> bool:
    """判断是否在询问当前会话历史。"""
    return is_history_query(query)


def _is_memory_query(query: str) -> bool:
    """判断是否在询问已保存的长期记忆或用户偏好。"""
    return is_memory_query(query)


def _is_preference_instruction(query: str) -> bool:
    """判断是否为用户偏好或长期记忆写入指令。"""
    return is_preference_instruction(query)


def _answer_preference_instruction(query: str) -> str:
    """确认用户偏好指令。"""
    cleaned = " ".join(query.strip().split()).rstrip("。！？!?. ")
    return (
        "直接回答\n"
        f"已记录：{cleaned}。\n\n"
        "依据说明\n"
        "这是用户偏好/长期记忆指令，不是火灾论文知识问题，因此不会触发 RAG 检索，也不需要论文引用。"
    )


def _answer_memory_query(query: str, long_term_memories: str) -> str:
    """从已检索的长期记忆中回答偏好/记忆查询。"""
    memories = _extract_memory_lines(long_term_memories)
    if "火灾" in query or "笔记" in query:
        focused = [
            memory for memory in memories
            if any(keyword in memory for keyword in ("火灾", "笔记", "中文", "英文", "标题", "格式"))
        ]
        if focused:
            memories = focused

    if not memories:
        return (
            "直接回答\n"
            "我没有检索到与你这个问题相关的长期记忆。\n\n"
            "依据说明\n"
            "这个问题是在询问用户偏好/长期记忆，不是火灾论文知识问题，因此不会触发 RAG 检索。"
            "如果你刚刚设置过偏好，请确认长期记忆服务和 Qdrant 已启动，并且当前会话已经完成一次回答写入。"
        )

    lines = "\n".join(f"- {memory}" for memory in memories[:5])
    return (
        "直接回答\n"
        f"根据长期记忆，你的相关偏好是：\n{lines}\n\n"
        "依据说明\n"
        "这个回答来自已检索到的长期记忆，不是论文 RAG 证据，因此不需要论文引用。"
    )


def _extract_memory_lines(long_term_memories: str) -> list[str]:
    """清理 prompt 中的长期记忆条目，提取可展示文本。"""
    memories: list[str] = []
    for line in long_term_memories.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^-\s*\[[^\]]+\]\s*", "", stripped)
        stripped = stripped.lstrip("- ").strip()
        if stripped:
            memories.append(stripped)
    return memories


def _answer_history_query(query: str, conversation_context: str) -> str:
    """从短期上下文中回答历史问题查询。"""
    user_questions = _extract_user_questions(conversation_context)
    if "火灾" in query or "消防" in query:
        user_questions = [
            question for question in user_questions
            if any(pattern in question for pattern in FIRE_DOMAIN_PATTERNS)
        ]

    if not user_questions:
        return (
            "直接回答\n"
            "我当前没有读到可用于回答这个问题的历史用户提问。请确认你是在同一个会话中继续提问，"
            "并且前端请求带上了当前 session_id。"
        )

    lines = "\n".join(f"- {question}" for question in user_questions[-8:])
    return (
        "直接回答\n"
        f"你之前问过这些相关问题：\n{lines}\n\n"
        "依据说明\n"
        "这个回答来自当前会话的短期上下文，不使用 RAG 论文证据，因此不需要论文引用。"
    )


def _extract_user_questions(conversation_context: str) -> list[str]:
    """从短期上下文文本中提取历史用户问题。"""
    questions: list[str] = []
    for line in conversation_context.splitlines():
        stripped = line.strip()
        if not stripped.startswith("用户："):
            continue
        question = stripped.removeprefix("用户：").strip()
        if question:
            questions.append(question)
    return questions


_DEFAULT_NODES: FireAgentGraphNodes | None = None


def default_nodes() -> FireAgentGraphNodes:
    """返回模块级默认节点集合。"""
    global _DEFAULT_NODES
    if _DEFAULT_NODES is None:
        _DEFAULT_NODES = FireAgentGraphNodes()
    return _DEFAULT_NODES


def intent_router_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：意图路由。"""
    return default_nodes().intent_router_node(state)


def query_rewrite_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：查询改写。"""
    return default_nodes().query_rewrite_node(state)


def dense_retrieve_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：dense 检索。"""
    return default_nodes().dense_retrieve_node(state)


def sparse_retrieve_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：sparse 检索。"""
    return default_nodes().sparse_retrieve_node(state)


def fusion_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：结果融合。"""
    return default_nodes().fusion_node(state)


def rerank_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：候选重排。"""
    return default_nodes().rerank_node(state)


def sufficiency_check_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：证据充分性判断。"""
    return default_nodes().sufficiency_check_node(state)


def web_search_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：联网搜索。"""
    return default_nodes().web_search_node(state)


def context_build_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：上下文构建。"""
    return default_nodes().context_build_node(state)


def answer_generate_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：回答生成。"""
    return default_nodes().answer_generate_node(state)


def hallucination_check_node(state: FireAgentState) -> FireAgentState:
    """模块级节点函数：幻觉检查。"""
    return default_nodes().hallucination_check_node(state)
