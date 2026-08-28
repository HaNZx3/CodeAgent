"""命令层测试：验证 /delete 各分支，不依赖真实 LLM 与交互。

用最小 _FakeAgent stub 暴露 _handle_command 用到的 store/session_id/new_session，
配合真实 SessionStore(tmp_path) 与 monkeypatch 的 input 覆盖：
  无参数 / 不存在 / 删其它会话 / 删当前会话 / all 确认 / all 取消 / all 无会话
"""

import uuid

from config import Config
from main import _handle_command
from agent.session import SessionStore


class _FakeAgent:
    """最小 agent stub：暴露 _handle_command 在 /delete 分支用到的接口。

    真实 CodingAgent 的 new_session 会重建 ContextManager，这里只模拟
    「生成新 id 并切换 session_id」的对外可观察行为，足以验证命令逻辑。
    """

    def __init__(self, store, sid):
        self.store = store
        self.session_id = sid

    def new_session(self, name=None):
        new = name or uuid.uuid4().hex[:12]
        self.session_id = new
        return new


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
