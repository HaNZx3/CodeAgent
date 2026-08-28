"""SessionStore 单元测试：jsonl 持久化、原子重写、workspace 隔离。"""

import json
import re
import time

from agent.session import SessionStore


def test_load_empty_session_returns_empty_list(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    assert store.load("nonexistent") == []


def test_append_then_load_roundtrip(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    msgs = [
        {"role": "user", "content": "task1"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "task2"},
    ]
    for m in msgs:
        store.append("s1", m)

    loaded = store.load("s1")
    assert loaded == msgs
    # 顺序必须保留
    assert [m["content"] for m in loaded] == ["task1", "answer1", "task2"]


def test_rewrite_replaces_all(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    for i in range(5):
        store.append("s1", {"role": "user", "content": f"old{i}"})
    assert len(store.load("s1")) == 5

    # rewrite 应整文件覆盖，不是追加
    store.rewrite("s1", [{"role": "user", "content": "new1"},
                         {"role": "assistant", "content": "new2"}])
    loaded = store.load("s1")
    assert len(loaded) == 2
    assert loaded[0]["content"] == "new1"
    assert loaded[1]["content"] == "new2"


def test_clear_removes_file(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    store.append("s1", {"role": "user", "content": "x"})
    assert store.path("s1").exists()

    store.clear("s1")
    assert not store.path("s1").exists()
    # 清除不存在的会话不抛错
    store.clear("nonexistent")


def test_list_sessions_returns_mtime_desc(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    store.append("old", {"role": "user", "content": "old session"})
    # 确保 mtime 有差异
    time.sleep(0.05)
    store.append("new", {"role": "user", "content": "new session"})

    sessions = store.list_sessions()
    assert len(sessions) == 2
    # mtime 倒序：new 在前
    assert sessions[0][0] == "new"
    assert sessions[1][0] == "old"
    assert sessions[0][2] >= sessions[1][2]


def test_list_sessions_preview_is_first_user(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    store.append("s1", {"role": "assistant", "content": "先说话"})  # 非 user，不作为预览
    store.append("s1", {"role": "user", "content": "这是第一条任务"})
    store.append("s1", {"role": "assistant", "content": "回复"})

    sessions = store.list_sessions()
    assert len(sessions) == 1
    sid, preview, _ = sessions[0]
    assert sid == "s1"
    assert preview == "这是第一条任务"


def test_workspace_dir_has_slug_and_hash(tmp_path):
    store = SessionStore(tmp_path, "D:/my/project")
    dir_name = store._dir.name
    # 形如 {slug}-{12位hex}
    assert re.match(r"^.+-[0-9a-f]{12}$", dir_name), f"目录名格式不符: {dir_name}"


def test_different_workspaces_isolated(tmp_path):
    store_a = SessionStore(tmp_path, "/workspaceA")
    store_b = SessionStore(tmp_path, "/workspaceB")
    store_a.append("s1", {"role": "user", "content": "from A"})
    store_b.append("s1", {"role": "user", "content": "from B"})

    # 同名 session_id，但不同 workspace，文件不混
    assert store_a.load("s1") == [{"role": "user", "content": "from A"}]
    assert store_b.load("s1") == [{"role": "user", "content": "from B"}]
    # 目录路径不同
    assert store_a._dir != store_b._dir


def test_rewrite_is_atomic(tmp_path, monkeypatch):
    """rewrite 用「先写 .tmp 再 os.replace」模式：成功后 .tmp 不残留，内容正确。

    原子性由 os.replace 保证（POSIX 原子），本测试验证实现确实用了该模式，
    而非直接覆盖原文件。直接 monkeypatch builtins.open 拦不到 Path.open
    （CPython C 实现），故改用语义验证。
    """
    import os
    store = SessionStore(tmp_path, "/workspace")
    store.append("s1", {"role": "user", "content": "原始数据"})

    # 记录 os.replace 是否被调用（证明走了 tmp+replace 模式）
    replace_calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replace_calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    store.rewrite("s1", [{"role": "user", "content": "新数据"}])

    # 1) os.replace 被调用过，且源是 .tmp 文件
    assert len(replace_calls) == 1
    src, dst = replace_calls[0]
    assert ".jsonl.tmp" in src
    assert dst == str(store.path("s1"))

    # 2) rewrite 后 .tmp 不残留
    from pathlib import Path
    tmp_file = Path(src)
    assert not tmp_file.exists()

    # 3) 内容正确
    loaded = store.load("s1")
    assert loaded == [{"role": "user", "content": "新数据"}]

    # 4) 原文件被替换（不是追加）
    assert len(loaded) == 1


def test_clear_all_removes_every_session(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    for sid in ["s1", "s2", "s3"]:
        store.append(sid, {"role": "user", "content": "x"})
    assert len(store.list_sessions()) == 3

    n = store.clear_all()
    assert n == 3
    assert store.list_sessions() == []
    # 目录本身保留（list_sessions 容错处理目录不存在）
    assert store._dir.exists()


def test_clear_all_on_missing_dir(tmp_path):
    store = SessionStore(tmp_path, "/workspace")
    # 目录尚不存在（从未 append 过）
    assert store.clear_all() == 0


def test_clear_all_only_touches_current_workspace(tmp_path):
    store_a = SessionStore(tmp_path, "/workspaceA")
    store_b = SessionStore(tmp_path, "/workspaceB")
    store_a.append("s1", {"role": "user", "content": "A"})
    store_b.append("s1", {"role": "user", "content": "B"})

    n = store_a.clear_all()
    assert n == 1
    # workspace B 不受影响
    assert store_b.load("s1")[0]["content"] == "B"
    assert store_b.list_sessions() != []


def test_clear_all_cleans_tmp_residue(tmp_path):
    """rewrite 崩溃残留的 .jsonl.tmp 一并清理。"""
    store = SessionStore(tmp_path, "/workspace")
    store.append("s1", {"role": "user", "content": "x"})
    # 模拟崩溃残留的临时文件
    store.path("s1").with_suffix(".jsonl.tmp").write_text("partial", encoding="utf-8")

    n = store.clear_all()
    assert n == 1  # .tmp 不计入 .jsonl 计数
    # .tmp 残留也被清理
    assert not list(store._dir.glob("*.jsonl.tmp"))
