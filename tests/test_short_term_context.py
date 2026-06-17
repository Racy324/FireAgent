"""短期上下文管理器测试。"""

from __future__ import annotations

from fireagent.memory.context_manager import ContextManager
from fireagent.memory.store import SQLiteMemoryStore
from fireagent.utils.config import ShortTermMemoryConfig


def _make_config(**overrides) -> ShortTermMemoryConfig:
    return ShortTermMemoryConfig(**overrides)


def test_context_manager_includes_structured_state_and_recent_messages(tmp_path) -> None:
    """输出应包含结构化状态和最近消息。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="测试")

    store.update_session_state(
        session.session_id,
        current_goal="实现 MVP-2",
        current_step="编写测试",
        constraints=["不引入新数据库"],
        pending_items=["ContextManager", "API 接入"],
    )
    store.save_message(session.session_id, "user", "隧道火灾烟气有什么影响？")
    store.save_message(session.session_id, "assistant", "烟气会降低能见度。")
    store.save_message(session.session_id, "user", "继续下一步")

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id)

    assert "实现 MVP-2" in ctx.text
    assert "编写测试" in ctx.text
    assert "不引入新数据库" in ctx.text
    assert "隧道火灾烟气有什么影响？" in ctx.text
    assert "继续下一步" in ctx.text
    assert "当前任务目标" in ctx.text
    assert "当前步骤" in ctx.text
    assert "最近对话" in ctx.text
    assert ctx.truncated is False
    assert len(ctx.recent_message_ids) == 3


def test_context_manager_omits_empty_sections(tmp_path) -> None:
    """空区域不应出现在输出中。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="空状态")
    store.save_message(session.session_id, "user", "你好")

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id)

    # current_goal 为空，不应出现
    assert "当前任务目标" not in ctx.text
    # 有消息，最近对话应出现
    assert "最近对话" in ctx.text
    assert "你好" in ctx.text


def test_context_manager_respects_max_chars(tmp_path) -> None:
    """输出不应超过 max_chars 太多。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="长度测试")

    store.update_session_state(
        session.session_id,
        current_goal="这是一个很长的目标" * 20,
        rolling_summary="这是一个很长的摘要" * 20,
    )
    for i in range(10):
        store.save_message(session.session_id, "user", f"问题 {i}：" + "内容" * 50)
        store.save_message(session.session_id, "assistant", f"回答 {i}：" + "内容" * 50)

    config = _make_config(max_chars=800, recent_message_limit=10)
    ctx = ContextManager(store, config).build(session.session_id)

    assert ctx.total_chars <= 1200  # 允许少量超出（最后一条消息被截断）


def test_context_manager_marks_truncated_when_budget_exceeded(tmp_path) -> None:
    """超预算时 truncated=True。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="截断测试")

    store.update_session_state(
        session.session_id,
        current_goal="目标" * 100,
    )

    config = _make_config(max_chars=200)
    ctx = ContextManager(store, config).build(session.session_id)

    assert ctx.truncated is True


def test_context_manager_does_not_include_current_query(tmp_path) -> None:
    """构建时传入的 query 不应出现在上下文中。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="query 测试")
    store.save_message(session.session_id, "user", "历史问题")

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id, query="当前新问题")

    # 当前新问题不应出现在上下文中
    assert "当前新问题" not in ctx.text
    # 历史问题应出现
    assert "历史问题" in ctx.text


def test_context_manager_formats_intermediate_results(tmp_path) -> None:
    """中间结果应正确格式化。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="中间结果测试")

    store.append_intermediate_result(session.session_id, {
        "step": "测试通过",
        "status": "ok",
        "summary": "47 个测试全部通过",
    })
    store.append_intermediate_result(session.session_id, {
        "step": "TypeScript 检查",
        "status": "ok",
    })

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id)

    assert "中间结果" in ctx.text
    assert "47 个测试全部通过" in ctx.text


def test_context_manager_includes_pinned_messages(tmp_path) -> None:
    """pinned messages 应作为关键消息区域注入，排在最近对话之前。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="关键消息测试")

    # 保存若干消息
    m1 = store.save_message(session.session_id, "user", "初始目标：分析隧道火灾数据")
    store.save_message(session.session_id, "assistant", "好的，开始分析。")
    m3 = store.save_message(session.session_id, "user", "结论是什么？")
    store.save_message(session.session_id, "assistant", "烟气是主要致死因素。")
    store.save_message(session.session_id, "user", "最新问题")

    # 把 m1 和 m3 设为 pinned
    store.update_session_state(
        session.session_id,
        pinned_message_ids=[m1.message_id, m3.message_id],
    )

    config = _make_config(max_chars=5000, recent_message_limit=4)
    ctx = ContextManager(store, config).build(session.session_id)

    assert "关键消息" in ctx.text
    assert "初始目标：分析隧道火灾数据" in ctx.text
    assert "结论是什么？" in ctx.text
    # 关键消息区域应在最近对话之前
    pinned_pos = ctx.text.index("关键消息")
    recent_pos = ctx.text.index("最近对话")
    assert pinned_pos < recent_pos
    assert ctx.pinned_message_ids == [m1.message_id, m3.message_id]


def test_context_manager_omits_pinned_when_empty(tmp_path) -> None:
    """无 pinned messages 时不生成关键消息区域。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="无关键消息")
    store.save_message(session.session_id, "user", "你好")

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id)

    assert "关键消息" not in ctx.text


def test_maybe_refresh_rolling_summary_skips_when_below_threshold(tmp_path) -> None:
    """消息数未超过阈值时不刷新摘要。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="摘要跳过")
    store.save_message(session.session_id, "user", "问题 1")
    store.save_message(session.session_id, "assistant", "回答 1")

    config = _make_config(summarize_after_messages=20)
    ctx_mgr = ContextManager(store, config)
    summary = ctx_mgr.maybe_refresh_rolling_summary(session.session_id)

    assert summary == ""


def test_maybe_refresh_rolling_summary_generates_extractive_summary(tmp_path) -> None:
    """消息数超过阈值后应生成 extractive 摘要。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="摘要生成")

    # 先保存一条初始目标
    store.save_message(session.session_id, "user", "分析隧道火灾烟气控制的最新研究进展")
    store.save_message(session.session_id, "assistant", "好的，开始分析。")

    # 再保存足够多消息超过阈值
    for i in range(10):
        store.save_message(session.session_id, "user", f"问题 {i}：关于火灾疏散的细节")
        store.save_message(session.session_id, "assistant", f"回答 {i}：已完成分析。结论是烟气控制很关键。")

    config = _make_config(
        summarize_after_messages=10,
        recent_message_limit=4,
        rolling_summary_max_chars=500,
    )
    ctx_mgr = ContextManager(store, config)
    summary = ctx_mgr.maybe_refresh_rolling_summary(session.session_id)

    assert summary != ""
    assert "最初目标" in summary
    assert "隧道火灾" in summary

    # 验证持久化
    state = store.get_session_state(session.session_id)
    assert state is not None
    assert state.rolling_summary == summary


def test_context_manager_injects_rolling_summary(tmp_path) -> None:
    """已有 rolling_summary 时应注入到上下文中。"""
    store = SQLiteMemoryStore(tmp_path / "fireagent.db")
    store.init_db()
    session = store.create_session(title="摘要注入")

    store.update_session_state(
        session.session_id,
        rolling_summary="最初目标：分析烟气控制\n历史问题：疏散时间；毒性分析",
    )
    store.save_message(session.session_id, "user", "继续")

    config = _make_config(max_chars=5000)
    ctx = ContextManager(store, config).build(session.session_id)

    assert "滚动摘要" in ctx.text
    assert "分析烟气控制" in ctx.text
