"""Fallback 相关配置加载测试。"""

from __future__ import annotations

from fireagent.utils.config import load_config


def test_fallback_policy_config_fields_load_from_yaml() -> None:
    """app.yaml 中的 fallback policy 字段应被配置模型保留。"""
    cfg = load_config()

    assert cfg.retrieval.cross_doc_min_docs == 2
    assert cfg.web.force_web_temporal is True
    assert cfg.web.force_web_official is True
    assert "官方" in cfg.web.official_keywords
    assert "仅基于知识库" in cfg.web.local_scope_keywords
    assert "纵火" in cfg.web.harmful_keywords
    assert "这个" in cfg.web.ambiguous_keywords
