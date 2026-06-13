"""长期记忆时间衰退测试。"""

from __future__ import annotations

import pytest

from fireagent.memory.decay import (
    age_days_from_now,
    decay_factor,
    importance_boost,
    memory_final_score,
)


def test_decay_factor_half_life() -> None:
    """半衰期时衰退因子应为 0.5。"""
    assert decay_factor(age_days=90, half_life_days=90) == pytest.approx(0.5)
    assert decay_factor(age_days=365, half_life_days=365) == pytest.approx(0.5)


def test_decay_factor_new_memory() -> None:
    """新建记忆衰退因子应接近 1.0。"""
    assert decay_factor(age_days=0, half_life_days=180) == pytest.approx(1.0)
    assert decay_factor(age_days=1, half_life_days=365) > 0.99


def test_decay_factor_old_memory() -> None:
    """很旧的记忆衰退因子应很小。"""
    assert decay_factor(age_days=730, half_life_days=365) == pytest.approx(0.25)


def test_final_score_combines_similarity_decay_and_importance() -> None:
    """最终分数应融合语义分数、衰退和重要性。"""
    score = memory_final_score(
        semantic_score=0.8,
        importance=0.9,
        age_days=0,
        half_life_days=365,
    )
    # decay ~1.0, boost = 0.75 + 0.5*0.9 = 1.2
    assert score == pytest.approx(0.8 * 1.0 * 1.2, rel=0.01)


def test_final_score_decays_with_age() -> None:
    """旧记忆的最终分数应低于新记忆。"""
    new_score = memory_final_score(0.8, 0.8, age_days=0, half_life_days=180)
    old_score = memory_final_score(0.8, 0.8, age_days=180, half_life_days=180)
    assert new_score > old_score
    assert old_score == pytest.approx(new_score * 0.5, rel=0.01)


def test_age_days_from_now_handles_empty() -> None:
    """空字符串应返回 0。"""
    assert age_days_from_now("") == 0.0


def test_age_days_from_now_handles_iso_format() -> None:
    """ISO 格式时间应正确计算天数。"""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    recent = now.isoformat()
    assert age_days_from_now(recent) < 0.01  # 几乎为 0
