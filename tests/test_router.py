"""意图路由测试。"""

from __future__ import annotations

from fireagent.graph.edges import route_after_intent, route_after_sufficiency
from fireagent.graph.nodes import FireAgentGraphNodes, route_intent
from fireagent.graph.state import create_initial_state


def test_route_intent_chat() -> None:
    """问候类问题应进入 chat。"""
    intent, reason = route_intent("你好")

    assert intent == "chat"
    assert reason


def test_route_intent_fire_rag() -> None:
    """火灾领域知识问答应进入 RAG。"""
    intent, _ = route_intent("隧道火灾烟气如何控制")

    assert intent == "rag"


def test_route_intent_paper_summary() -> None:
    """论文总结/对比类问题应进入 paper 流程。"""
    intent, _ = route_intent("请总结火灾检测论文的主要方法")

    assert intent == "paper"


def test_route_intent_emergency() -> None:
    """火灾应急问题应进入 emergency 流程。"""
    intent, _ = route_intent("发生火灾怎么办，如何逃生")

    assert intent == "emergency"


def test_route_intent_reject() -> None:
    """非火灾领域问题应被拒答或引导。"""
    intent, _ = route_intent("今天午饭吃什么")

    assert intent == "reject"


def test_graph_edge_routing() -> None:
    """条件边路由应匹配意图和充分性状态。"""
    assert route_after_intent({"intent": "chat"}) == "answer_generate"
    assert route_after_intent({"intent": "reject"}) == "answer_generate"
    assert route_after_intent({"intent": "rag"}) == "query_rewrite"

    assert route_after_sufficiency({"evidence_sufficient": True}) == "context_build"
    assert route_after_sufficiency({"evidence_sufficient": False}) == "web_search"


def test_intent_router_node_adds_safety_notice_for_emergency() -> None:
    """应急类节点输出必须包含安全提醒。"""
    nodes = FireAgentGraphNodes()
    state = create_initial_state("发生火灾怎么办")

    update = nodes.intent_router_node(state)

    assert update["intent"] == "emergency"
    assert "119" in update["safety_notice"]

