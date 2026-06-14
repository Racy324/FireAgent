"""意图路由测试。"""

from __future__ import annotations

from fireagent.graph.edges import route_after_intent, route_after_sufficiency
from fireagent.graph.nodes import FireAgentGraphNodes, generate_answer_from_state, route_intent
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


def test_route_intent_history_question_does_not_enter_rag() -> None:
    """询问历史问题时即使命中火灾词，也不应进入 RAG。"""
    intent, reason = route_intent("我之前问过什么火灾问题")

    assert intent == "chat"
    assert "历史" in reason


def test_route_intent_preference_instruction_does_not_enter_rag() -> None:
    """用户偏好/记忆指令即使命中火灾词，也不应进入 RAG。"""
    intent, reason = route_intent("以后有关火灾的笔记全部用中文")

    assert intent == "chat"
    assert "偏好" in reason


def test_route_intent_memory_query_does_not_enter_rag() -> None:
    """询问火灾笔记偏好时，应查询长期记忆而不是进入论文 RAG。"""
    intent, reason = route_intent("火灾笔记应该用什么格式和语言")

    assert intent == "chat"
    assert "长期记忆" in reason


def test_route_intent_preference_question_is_not_write_instruction() -> None:
    """询问已有偏好不应被误判为新的偏好写入指令。"""
    intent, reason = route_intent("我的偏好有什么")

    assert intent == "chat"
    assert "长期记忆" in reason


def test_generate_answer_for_preference_instruction_acknowledges_memory() -> None:
    """偏好指令应直接确认记录，不应要求 RAG 证据。"""
    state = create_initial_state("以后有关火灾的笔记全部用中文")
    state["intent"] = "chat"

    answer = generate_answer_from_state(state)

    assert "已记录" in answer
    assert "中文" in answer
    assert "证据不足" not in answer
    assert "[L1]" not in answer


def test_generate_answer_for_memory_query_uses_long_term_memories() -> None:
    """长期记忆查询应直接复述相关偏好，不应要求论文证据。"""
    state = create_initial_state(
        "火灾笔记应该用什么格式和语言",
        long_term_memories=(
            "- [semantic | 0.92] 用户偏好：以后所有火灾笔记都用中文，"
            "并在标题前加【火灾笔记】。"
        ),
    )
    state["intent"] = "chat"

    answer = generate_answer_from_state(state)

    assert "中文" in answer
    assert "【火灾笔记】" in answer
    assert "长期记忆" in answer
    assert "证据不足" not in answer
    assert "[L1]" not in answer


def test_generate_answer_for_history_question_uses_conversation_context() -> None:
    """历史问题应基于短期上下文回答，而不是论文证据。"""
    state = create_initial_state(
        "我之前问过什么火灾问题",
        conversation_context=(
            "# 最近对话\n"
            "用户：隧道火灾烟气对人员疏散有什么影响？\n"
            "助手：烟气会降低能见度。\n"
            "用户：今天午饭吃什么？\n"
        ),
    )
    state["intent"] = "chat"

    answer = generate_answer_from_state(state)

    assert "隧道火灾烟气对人员疏散有什么影响" in answer
    assert "今天午饭吃什么" not in answer
    assert "[L1]" not in answer


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


def test_intent_router_node_exposes_route_decision() -> None:
    """意图路由节点应输出可调试的 route_decision。"""
    nodes = FireAgentGraphNodes()
    state = create_initial_state("火灾笔记应该用什么格式和语言")

    update = nodes.intent_router_node(state)

    assert update["intent"] == "chat"
    assert update["route_decision"]["sub_intent"] == "memory_query"
    assert update["route_decision"]["source"] == "hard_rule"
