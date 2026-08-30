"""命令层测试：验证 /delete、/back 各分支与 token 显示辅助函数，不依赖真实 LLM。

用最小 _FakeAgent stub 暴露 _handle_command 用到的接口：真实 SessionStore(tmp_path)、
真实 ContextManager（验证 /back 真正截断并落盘）与禁用的 CheckpointStore
（/back 走「仅回退对话」分支），配合 monkeypatch 的 input 覆盖交互分支。
另覆盖 Claude Code 式用量显示的纯函数：_fmt_tokens / _context_bar。
"""

import uuid

from llm.client import ModelResponse
from config import Config
from main import _handle_command, _fmt_tokens, _context_bar
from agent.session import SessionStore
from agent.context import ContextManager
from agent.checkpoints import CheckpointStore


class _FakeAgent:
    """最小 agent stub：暴露 _handle_command 用到的接口。

    真实 CodingAgent 的 new_session 会重建 ContextManager，这里只模拟
    「生成新 id 并切换 session_id」的对外可观察行为；context 与 checkpoints
    用真实对象（禁用快照），足以验证命令逻辑与落盘效果。
    """

    def __init__(self, store, sid):
        self.store = store
        self.session_id = sid
        self.cleared = False
        self.context = ContextManager("sys", store=store, session_id=sid)
        self.checkpoints = CheckpointStore(
            store.root / "ckpt", store.workspace, sid, enabled=False
        )

    def new_session(self, name=None):
        new = name or uuid.uuid4().hex[:12]
        self.session_id = new
        return new

    def clear_context(self):
        self.cleared = True
        self.context.reset()


def _make(store, sid="current1"):
    return _FakeAgent(store, sid), Config(workspace="/ws")


def test_delete_no_arg_prints_usage(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    assert _handle_command(agent, "/delete", config) is True
    assert "用法" in capsys.readouterr().out


def test_delete_nonexistent_id(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    store.append("real1", {"role": "user", "content": "x"})
    agent, config = _make(store)
    _handle_command(agent, "/delete nope", config)
    out = capsys.readouterr().out
    assert "不存在" in out
    # 不能误删别的会话
    assert store.path("real1").exists()


def test_delete_other_session_removes_file_and_keeps_current(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    store.append("victim", {"role": "user", "content": "x"})
    agent, config = _make(store, "current1")
    _handle_command(agent, "/delete victim", config)
    out = capsys.readouterr().out
    assert "已删除会话" in out
    assert not store.path("victim").exists()
    # 当前会话不变
    assert agent.session_id == "current1"


def test_delete_current_session_opens_new(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    store.append("current1", {"role": "user", "content": "x"})
    agent, config = _make(store, "current1")
    _handle_command(agent, "/delete current1", config)
    out = capsys.readouterr().out
    assert "已删除当前会话" in out
    assert not store.path("current1").exists()
    # 删当前会话后必须切到新会话，避免内存 messages 与文件不一致
    assert agent.session_id != "current1"


def test_delete_all_confirmed_clears_and_opens_new(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    for sid in ["a", "b", "c"]:
        store.append(sid, {"role": "user", "content": "x"})
    agent, config = _make(store, "current1")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    _handle_command(agent, "/delete all", config)
    out = capsys.readouterr().out
    assert "已删除 3 个会话" in out
    assert store.list_sessions() == []
    # 当前会话必被删，开新空会话
    assert agent.session_id != "current1"


def test_delete_all_cancelled_keeps_everything(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    store.append("a", {"role": "user", "content": "x"})
    agent, config = _make(store, "current1")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    _handle_command(agent, "/delete all", config)
    out = capsys.readouterr().out
    assert "已取消" in out
    assert store.list_sessions() != []  # 未删
    assert agent.session_id == "current1"  # 未切会话


def test_delete_all_when_no_sessions(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")  # 空，无任何会话
    agent, config = _make(store, "current1")
    _handle_command(agent, "/delete all", config)
    out = capsys.readouterr().out
    assert "无会话可删" in out


def test_clear_resets_context_in_place(capsys, tmp_path):
    """/clear 原地清空当前会话上下文：id 不变，不开新会话。"""
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store, "current1")
    assert _handle_command(agent, "/clear", config) is True
    out = capsys.readouterr().out
    assert "已清空当前会话上下文" in out
    assert agent.cleared is True
    assert agent.session_id == "current1"  # id 不变（区别于 /new）


# ── /back：对话回退 ───────────────────────────────────────────────────────


def _seed_turns(agent, n=2):
    for i in range(1, n + 1):
        agent.context.add_user(f"任务{i}")
        agent.context.add_assistant(ModelResponse(text=f"回答{i}"))


def test_back_without_turns_prints_hint(capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    assert _handle_command(agent, "/back", config) is True
    assert "没有可回退" in capsys.readouterr().out


def test_back_lists_turns_and_rewinds_on_choice(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "1")

    _handle_command(agent, "/back", config)

    out = capsys.readouterr().out
    assert "回退到哪条消息之前" in out
    assert "任务1" in out and "任务2" in out
    # 回退到第 1 条之前：只剩 system，文件同步清空
    assert len(agent.context.messages) == 1
    assert store.load(agent.session_id) == []
    assert "已回退" in out


def test_back_rewind_to_second_turn_keeps_first(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "2")

    _handle_command(agent, "/back", config)

    # 回退到第 2 条之前：保留第一轮
    assert [m.get("content") for m in agent.context.messages[1:]] == ["任务1", "回答1"]
    assert [m.get("content") for m in store.load(agent.session_id)] == ["任务1", "回答1"]


def test_back_cancel_on_empty_input(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")

    _handle_command(agent, "/back", config)

    assert "已取消" in capsys.readouterr().out
    assert len(agent.context.messages) == 5  # 未变


def test_back_cancel_on_invalid_number(monkeypatch, capsys, tmp_path):
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "9")

    _handle_command(agent, "/back", config)

    assert "已取消" in capsys.readouterr().out
    assert len(agent.context.messages) == 5  # 未变


def test_back_direct_number_arg_skips_prompt(monkeypatch, capsys, tmp_path):
    """/back <n> 直接回退，不询问编号（input 被调用即失败）。"""
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)

    def _must_not_ask(*a, **k):
        raise AssertionError("/back <n> 不应询问编号")

    monkeypatch.setattr("builtins.input", _must_not_ask)
    _handle_command(agent, "/back 2", config)

    # 回退到第 2 条之前：保留第一轮
    assert [m.get("content") for m in agent.context.messages[1:]] == ["任务1", "回答1"]
    assert "已回退" in capsys.readouterr().out


def test_back_skips_code_restore_when_ledger_misaligned(capsys, tmp_path):
    """账本与轮次不对齐（如旧会话无快照）时仅回退对话，并给出提示。"""
    store = SessionStore(tmp_path, "/ws")
    agent, config = _make(store)
    _seed_turns(agent)
    # enabled=True 但账本为空 -> 与 2 个轮次不对齐
    agent.checkpoints.enabled = True

    _handle_command(agent, "/back 1", config)

    out = capsys.readouterr().out
    assert "不对齐" in out
    assert len(agent.context.messages) == 1  # 对话仍被回退


# ── Claude Code 式用量显示辅助函数 ────────────────────────────────────────


def test_fmt_tokens_abbr():
    assert _fmt_tokens(930) == "930"
    assert _fmt_tokens(9_999) == "9,999"
    assert _fmt_tokens(10_000) == "10k"
    assert _fmt_tokens(12_800) == "12.8k"
    assert _fmt_tokens(128_000) == "128k"


def test_context_bar_fill_and_colors():
    bar = _context_bar(32_000, 128_000, width=20)  # 25% -> 5 格
    assert bar.count("█") == 5
    assert bar.count("░") == 15
    assert "\033[32m" in bar  # 绿色 <50%

    assert _context_bar(70_000, 128_000, width=20).count("█") == 11  # 54.7% -> 黄
    assert "\033[33m" in _context_bar(70_000, 128_000, width=20)
    assert "\033[31m" in _context_bar(120_000, 128_000, width=20)  # >=80% 红
    # 超出窗口时封顶不越界
    assert _context_bar(999_999, 128_000, width=20).count("█") == 20
