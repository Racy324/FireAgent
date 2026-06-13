"""FireAgent LangGraph 节点实现。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

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
        self.prompt_loader = PromptTemplateLoader(config=self.context.config)

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
        query = state_get_query(state)
        intent, reason = route_intent(query)
        safety_notice = ""
        if intent == "emergency":
            safety_notice = "安全提醒：如现场存在明火、浓烟、爆炸或人员受困，请立即拨打 119，并优先撤离到安全区域。"
        return FireAgentState(intent=intent, intent_reason=reason, safety_notice=safety_notice)

    def query_rewrite_node(self, state: FireAgentState) -> FireAgentState:
        """对用户问题进行查询改写。"""
        query = state_get_query(state)
        rewrite_result = self.query_rewriter.rewrite(query)
        return FireAgentState(
            rewrite_result=rewrite_result,
            rewritten_queries=rewrite_result.all_queries,
        )

    def dense_retrieve_node(self, state: FireAgentState) -> FireAgentState:
        """执行 dense 检索。"""
        try:
            rewrite_result = state.get("rewrite_result")
            if rewrite_result is None:
                rewrite_result = self.query_rewriter.rewrite(state_get_query(state))
            results = DenseRetriever(self.vectorstore, config=self.context.config).retrieve_many(rewrite_result)
            return FireAgentState(local_dense_results=results)
        except Exception as exc:  # noqa: BLE001 - 外部服务与模型错误统一写入状态。
            return FireAgentState(local_dense_results=[], errors=[f"dense_retrieve_node 失败：{exc}"])

    def sparse_retrieve_node(self, state: FireAgentState) -> FireAgentState:
        """执行 sparse/BM25 检索。"""
        try:
            rewrite_result = state.get("rewrite_result")
            if rewrite_result is None:
                rewrite_result = self.query_rewriter.rewrite(state_get_query(state))
            results = SparseRetriever(self.vectorstore, config=self.context.config).retrieve_many(rewrite_result)
            return FireAgentState(local_sparse_results=results)
        except Exception as exc:  # noqa: BLE001
            return FireAgentState(local_sparse_results=[], errors=[f"sparse_retrieve_node 失败：{exc}"])

    def fusion_node(self, state: FireAgentState) -> FireAgentState:
        """对 dense 与 sparse 结果做 Weighted RRF 融合。"""
        dense_results = list(state.get("local_dense_results", []) or [])
        sparse_results = list(state.get("local_sparse_results", []) or [])
        fused = self.fusion.fuse(dense_results, sparse_results)
        return FireAgentState(fused_results=fused)

    def rerank_node(self, state: FireAgentState) -> FireAgentState:
        """对融合候选执行 cross-encoder 重排。"""
        query = get_main_query(state)
        fused = list(state.get("fused_results", []) or [])
        if not fused:
            return FireAgentState(reranked_results=[])
        try:
            reranked = self.reranker.rerank(
                query,
                fused,
                top_k=self.context.config.retrieval.rerank_top_k,
            )
            return FireAgentState(reranked_results=reranked)
        except Exception as exc:  # noqa: BLE001
            fallback = LexicalReranker().rerank(
                query,
                fused,
                top_k=self.context.config.retrieval.rerank_top_k,
            )
            return FireAgentState(
                reranked_results=fallback,
                errors=[f"rerank_node 使用 fallback：{exc}"],
            )

    def sufficiency_check_node(self, state: FireAgentState) -> FireAgentState:
        """判断本地证据是否足够，并运行 fallback policy 决策。"""
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
        return FireAgentState(
            evidence_sufficient=sufficiency_result.sufficient,
            sufficiency_result=sufficiency_result,
            fallback_decision=fallback_decision,
        )

    def web_search_node(self, state: FireAgentState) -> FireAgentState:
        """本地证据不足时执行 Tavily 联网搜索，并与本地证据统一重排。"""
        query = get_main_query(state)
        local_results = list(state.get("reranked_results", []) or [])
        if not self.context.config.rag.enable_web_fallback:
            return FireAgentState(
                web_results=[],
                reranked_results=local_results,
                errors=["联网兜底已被配置关闭。"],
            )

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
            return FireAgentState(web_results=web_chunks, reranked_results=combined)
        except Exception as exc:  # noqa: BLE001
            return FireAgentState(
                web_results=[],
                reranked_results=local_results,
                errors=[f"web_search_node 失败：{exc}"],
            )

    def context_build_node(self, state: FireAgentState) -> FireAgentState:
        """构建最终回答上下文和引用列表。"""
        query = get_main_query(state)
        reranked = list(state.get("reranked_results", []) or [])
        context_result = self.context_builder.build(reranked, query=query)
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
        fallback_answer = generate_answer_from_state(state)
        intent = str(state.get("intent", "") or "")
        final_context = str(state.get("final_context", "") or "").strip()

        # 检查 fallback decision，处理特殊动作
        fallback_decision = state.get("fallback_decision")
        if fallback_decision is not None:
            action = getattr(fallback_decision, "action", None)
            reason = getattr(fallback_decision, "reason", "")
            if action == FallbackAction.REFUSE:
                return FireAgentState(
                    final_answer="抱歉，我无法回答涉及纵火、规避消防检查等危险行为的问题。如遇火灾紧急情况，请立即拨打 119。",
                    citations=[],
                    used_citations=[],
                    used_citation_markers=[],
                    invalid_citation_markers=[],
                )
            if action == FallbackAction.ANSWER_INSUFFICIENT:
                return FireAgentState(
                    final_answer=f"当前知识库中没有找到足够的本地论文证据来回答该问题。{f'（原因：{reason}）' if reason else ''}",
                    citations=[],
                    used_citations=[],
                    used_citation_markers=[],
                    invalid_citation_markers=[],
                )
            if action == FallbackAction.ASK_CLARIFY:
                return FireAgentState(
                    final_answer="这个问题里的指代还不够明确。请补充具体论文、事故、标准名称，或说明你希望我基于哪一批本地资料回答。",
                    citations=[],
                    used_citations=[],
                    used_citation_markers=[],
                    invalid_citation_markers=[],
                )

        if intent in {"chat", "reject"} or not final_context or not self.context.config.llm.enabled:
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
            return FireAgentState(
                final_answer=answer,
                citations=used_strings,
                used_citation_markers=markers,
                used_citations=used,
                invalid_citation_markers=invalid,
            )
        except Exception as exc:  # noqa: BLE001 - LLM 失败时回退模板回答。
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
        warnings: list[str] = []
        intent = str(state.get("intent", "") or "")
        answer = str(state.get("final_answer", "") or "")
        final_context = str(state.get("final_context", "") or "")
        if intent not in {"chat", "reject"} and not final_context and "证据不足" not in answer:
            warnings.append("回答缺少证据不足提示，已在节点中标记。")
            answer = f"{answer}\n\n不确定性：当前证据不足，不能给出确定结论。"
        return FireAgentState(final_answer=answer, hallucination_warnings=warnings)


def route_intent(query: str) -> tuple[str, str]:
    """基于规则识别用户意图。"""
    normalized = query.strip().lower()
    if not normalized:
        return "chat", "空问题，按闲聊处理。"
    if any(pattern in normalized for pattern in CHAT_PATTERNS) and len(normalized) <= 30:
        return "chat", "命中闲聊问候模式。"
    if any(pattern in query for pattern in EMERGENCY_PATTERNS) and any(
        pattern in query for pattern in FIRE_DOMAIN_PATTERNS
    ):
        return "emergency", "命中火灾应急安全类问题。"
    if any(pattern in query for pattern in PAPER_PATTERNS) and any(
        pattern in query for pattern in FIRE_DOMAIN_PATTERNS
    ):
        return "paper", "命中论文总结/对比类问题。"
    if any(pattern in query for pattern in FIRE_DOMAIN_PATTERNS):
        return "rag", "命中火灾领域知识问答。"
    return "reject", "未命中火灾领域关键词。"


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
