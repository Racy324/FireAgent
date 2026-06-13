"""LangGraph 工作流编译与运行入口。"""

from __future__ import annotations

from typing import Any, Optional

from fireagent.graph.edges import route_after_intent, route_after_sufficiency
from fireagent.graph.nodes import FireAgentGraphNodes
from fireagent.graph.state import FireAgentState, create_initial_state
from fireagent.llm import BaseLLMClient
from fireagent.utils.config import FireAgentConfig, get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient


class FireAgentWorkflowError(RuntimeError):
    """工作流构建或运行失败时抛出的异常。"""


def build_fireagent_workflow(
    config: Optional[FireAgentConfig] = None,
    vectorstore: Optional[FireAgentQdrantClient] = None,
    llm_client: Optional[BaseLLMClient] = None,
    nodes: Optional[FireAgentGraphNodes] = None,
) -> Any:
    """编译 FireAgent LangGraph 工作流。

    官方当前 API 使用 ``StateGraph``、``START``、``END``，节点函数接收 State 并
    返回 Partial State；条件边使用 ``add_conditional_edges``。
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise FireAgentWorkflowError(
            "缺少 langgraph，无法编译工作流。请先安装依赖：pip install langgraph"
        ) from exc

    cfg = config or get_config()
    graph_nodes = nodes or FireAgentGraphNodes(config=cfg, vectorstore=vectorstore, llm_client=llm_client)
    builder = StateGraph(FireAgentState)

    builder.add_node("intent_router", graph_nodes.intent_router_node)
    builder.add_node("query_rewrite", graph_nodes.query_rewrite_node)
    builder.add_node("dense_retrieve", graph_nodes.dense_retrieve_node)
    builder.add_node("sparse_retrieve", graph_nodes.sparse_retrieve_node)
    builder.add_node("fusion", graph_nodes.fusion_node)
    builder.add_node("rerank", graph_nodes.rerank_node)
    builder.add_node("sufficiency_check", graph_nodes.sufficiency_check_node)
    builder.add_node("web_search", graph_nodes.web_search_node)
    builder.add_node("context_build", graph_nodes.context_build_node)
    builder.add_node("answer_generate", graph_nodes.answer_generate_node)
    builder.add_node("hallucination_check", graph_nodes.hallucination_check_node)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "answer_generate": "answer_generate",
            "query_rewrite": "query_rewrite",
        },
    )
    builder.add_edge("query_rewrite", "dense_retrieve")
    builder.add_edge("query_rewrite", "sparse_retrieve")
    builder.add_edge(["dense_retrieve", "sparse_retrieve"], "fusion")
    builder.add_edge("fusion", "rerank")
    builder.add_edge("rerank", "sufficiency_check")
    builder.add_conditional_edges(
        "sufficiency_check",
        route_after_sufficiency,
        {
            "context_build": "context_build",
            "web_search": "web_search",
        },
    )
    builder.add_edge("web_search", "context_build")
    builder.add_edge("context_build", "answer_generate")
    builder.add_edge("answer_generate", "hallucination_check")
    builder.add_edge("hallucination_check", END)

    return builder.compile()


def run_fireagent_workflow(
    user_query: str,
    config: Optional[FireAgentConfig] = None,
    vectorstore: Optional[FireAgentQdrantClient] = None,
    llm_client: Optional[BaseLLMClient] = None,
    session_id: str = "",
    conversation_context: str = "",
    long_term_memories: str = "",
    long_term_memory_results: list[Any] | None = None,
) -> FireAgentState:
    """便捷函数：编译并运行 FireAgent 工作流。"""
    workflow = build_fireagent_workflow(config=config, vectorstore=vectorstore, llm_client=llm_client)
    result = workflow.invoke(
        create_initial_state(
            user_query,
            session_id=session_id,
            conversation_context=conversation_context,
            long_term_memories=long_term_memories,
            long_term_memory_results=long_term_memory_results,
        )
    )
    return FireAgentState(**dict(result))
