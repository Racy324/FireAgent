"""长期记忆重要性评分测试。"""

from __future__ import annotations

from fireagent.memory.importance import extract_candidates, score_candidate


def test_extracts_explicit_user_preference() -> None:
    """明确的用户偏好应得高分。"""
    importance, confidence, reason, mem_type, tags = score_candidate(
        "以后所有项目笔记默认使用中文，不要写英文。"
    )
    assert importance >= 0.45
    assert "preference" in tags
    assert "偏好" in reason or "明确" in reason


def test_extracts_project_fact() -> None:
    """项目事实应被识别。"""
    importance, _, reason, _, tags = score_candidate(
        "这个项目使用 Qdrant 作为向量库。"
    )
    assert importance >= 0.25
    assert "project" in tags


def test_extracts_negative_constraint() -> None:
    """明确约束应被识别。"""
    importance, _, _, _, tags = score_candidate(
        "不要做记忆管理页面，以后再考虑。"
    )
    assert importance >= 0.20
    assert "constraint" in tags


def test_rejects_transient_message() -> None:
    """临时错误信息应得低分。"""
    importance, _, reason, _, _ = score_candidate("这次测试报错了，error code 500")
    assert importance <= 0.0
    assert "临时" in reason


def test_rejects_secret_like_content() -> None:
    """疑似密钥内容应得低分。"""
    importance, _, reason, _, _ = score_candidate("api_key=sk-abc123xyz 这个 token 不要泄露")
    assert importance <= 0.0
    assert "敏感" in reason


def test_rejects_short_text() -> None:
    """过短文本不应产生候选。"""
    importance, _, _, _, _ = score_candidate("你好")
    assert importance == 0.0


def test_limits_candidates_per_turn() -> None:
    """候选数量应受 max_candidates 限制。"""
    query = "以后默认中文。这个项目使用 Qdrant。不要做管理页面。固定采用 pytest。"
    candidates = extract_candidates(query, max_candidates=2)
    assert len(candidates) <= 2
    # 按重要性降序
    if len(candidates) == 2:
        assert candidates[0].importance >= candidates[1].importance


def test_extract_candidates_from_realistic_query() -> None:
    """真实用户问题应能抽取候选。"""
    query = "以后 FireAgent 项目相关笔记默认写中文，不要写英文。"
    candidates = extract_candidates(query, max_candidates=2)
    assert len(candidates) >= 1
    assert candidates[0].importance >= 0.45


def test_extract_candidates_normalizes_content() -> None:
    """候选内容应规范化为结构化格式。"""
    query = "以后所有笔记默认使用中文。"
    candidates = extract_candidates(query, max_candidates=1)
    assert len(candidates) == 1
    assert candidates[0].content.startswith("用户偏好：")

    query2 = "这个项目使用 Qdrant 作为向量库。"
    candidates2 = extract_candidates(query2, max_candidates=1)
    assert len(candidates2) == 1
    assert candidates2[0].content.startswith("项目事实：")

    query3 = "不要做记忆管理页面。"
    candidates3 = extract_candidates(query3, max_candidates=1)
    assert len(candidates3) == 1
    assert candidates3[0].content.startswith("约束：")
