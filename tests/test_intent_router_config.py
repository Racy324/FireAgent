"""意图路由配置测试。"""

from __future__ import annotations

from fireagent.utils.config import FireAgentConfig, load_raw_config


def test_router_config_defaults() -> None:
    """默认配置应包含独立 router 配置。"""
    config = FireAgentConfig()

    assert config.router.enabled is True
    assert config.router.mode == "hybrid"
    assert config.router.temperature == 0.0
    assert config.router.max_tokens == 200
    assert config.router.confidence_threshold == 0.75
    assert config.router.low_confidence_threshold == 0.55
    assert config.router.fallback_to_rules is True


def test_router_env_overrides(tmp_path) -> None:
    """环境变量应能覆盖 router 配置。"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name in ("app.yaml", "qdrant.yaml", "prompts.yaml"):
        (config_dir / name).write_text("{}\n", encoding="utf-8")
    (config_dir / "models.yaml").write_text(
        "router:\n"
        "  enabled: true\n"
        "  mode: hybrid\n"
        "  model: default-router\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ROUTER_ENABLED=false\n"
        "ROUTER_BASE_URL=http://localhost:11434/v1\n"
        "ROUTER_API_KEY=test-key\n"
        "ROUTER_MODEL=qwen3-4b\n"
        "ROUTER_CONFIDENCE_THRESHOLD=0.8\n",
        encoding="utf-8",
    )

    raw = load_raw_config(config_dir=config_dir, env_file=env_file)
    config = FireAgentConfig.model_validate(raw)

    assert config.router.enabled is False
    assert config.router.base_url == "http://localhost:11434/v1"
    assert config.router.api_key == "test-key"
    assert config.router.model == "qwen3-4b"
    assert config.router.confidence_threshold == 0.8
