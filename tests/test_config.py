"""动态窗口与压缩阈值推导测试。"""

from config import (
    DEFAULT_CONTEXT_WINDOW,
    Config,
    context_window_for_model,
)

# from_env 会合并 .env 文件；测试置空，只走 monkeypatch 的环境变量。
_ENV_KEYS = (
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "CODING_AGENT_CONTEXT_WINDOW",
    "CODING_AGENT_COMPACT_THRESHOLD",
)


def _clean_env(monkeypatch):
    monkeypatch.setattr("config._dotenv_values", lambda p: {})
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_window_for_model_longest_match_wins():
    """「gpt-4o」必须命中 128k，而不是被更短的「gpt-4」（8k）抢走。"""
    assert context_window_for_model("gpt-4o-mini") == 128_000
    assert context_window_for_model("GPT-4O") == 128_000  # 大小写不敏感
    assert context_window_for_model("gpt-4") == 8_192
    assert context_window_for_model("gpt-4.1-mini") == 1_000_000
    assert context_window_for_model("deepseek-chat") == 64_000
    assert context_window_for_model("qwen3.8-max") == 131_072  # 落到「qwen」条目


def test_window_for_model_unknown_falls_back_to_default():
    assert context_window_for_model("some-future-model") == DEFAULT_CONTEXT_WINDOW
    assert context_window_for_model("") == DEFAULT_CONTEXT_WINDOW


def test_from_env_derives_window_and_threshold_from_model(monkeypatch):
    """未显式配置时：窗口按模型名查表，阈值取窗口 80%，两者联动。"""
    _clean_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")

    cfg = Config.from_env()

    assert cfg.context_window == 64_000
    assert cfg.compact_threshold == int(64_000 * 0.8)  # 51_200


def test_from_env_env_overrides_beat_derivation(monkeypatch):
    """显式配置优先：窗口覆盖查表值，阈值不再从窗口推导。"""
    _clean_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("CODING_AGENT_CONTEXT_WINDOW", "32000")
    monkeypatch.setenv("CODING_AGENT_COMPACT_THRESHOLD", "90000")

    cfg = Config.from_env()

    assert cfg.context_window == 32_000
    assert cfg.compact_threshold == 90_000  # 原样生效，由兜底线防爆窗
