"""Tool 单元测试：read / write / edit / search / list / shell。"""

import pytest

from tools.workspace import Workspace
from tools.file_tools import ListFilesTool, ReadFileTool, WriteFileTool, EditFileTool
from tools.search_tool import SearchTool
from tools.shell_tool import ShellTool


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)


# ---------- read_file ----------

def test_read_file_roundtrip(ws):
    (ws.root / "a.txt").write_text("hello", encoding="utf-8")
    r = ReadFileTool(ws).execute({"path": "a.txt"})
    assert r.success and r.output == "hello"


def test_read_file_missing(ws):
    r = ReadFileTool(ws).execute({"path": "nope.txt"})
    assert not r.success and "不存在" in r.error


def test_read_file_path_traversal(ws):
    outside = ws.root.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    r = ReadFileTool(ws).execute({"path": "../secret.txt"})
    assert not r.success and "越界" in r.error


# ---------- write_file ----------

def test_write_file_creates_parents(ws):
    r = WriteFileTool(ws).execute({"path": "sub/b.txt", "content": "hi"})
    assert r.success
    assert (ws.root / "sub" / "b.txt").read_text(encoding="utf-8") == "hi"


# ---------- edit_file ----------

def test_edit_file_success(ws):
    (ws.root / "c.py").write_text("x = 1\n", encoding="utf-8")
    r = EditFileTool(ws).execute({"path": "c.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert r.success
    assert (ws.root / "c.py").read_text(encoding="utf-8") == "x = 2\n"


def test_edit_file_old_not_found(ws):
    (ws.root / "c.py").write_text("x = 1\n", encoding="utf-8")
    r = EditFileTool(ws).execute({"path": "c.py", "old_text": "zzz", "new_text": "yyy"})
    assert not r.success and "未找到" in r.error


def test_edit_file_replaces_first_only(ws):
    (ws.root / "c.py").write_text("a a a\n", encoding="utf-8")
    r = EditFileTool(ws).execute({"path": "c.py", "old_text": "a", "new_text": "b"})
    assert r.success
    assert (ws.root / "c.py").read_text(encoding="utf-8") == "b a a\n"


# ---------- search_code ----------

def test_search_code(ws):
    (ws.root / "m.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (ws.root / "n.py").write_text("foo bar\n", encoding="utf-8")
    r = SearchTool(ws).execute({"query": "foo"})
    assert r.success
    assert "m.py:1" in r.output
    assert "n.py:1" in r.output


# ---------- list_files ----------

def test_list_files_ignores_junk(ws):
    (ws.root / ".git").mkdir()
    (ws.root / "node_modules").mkdir()
    (ws.root / "real.py").write_text("", encoding="utf-8")
    r = ListFilesTool(ws).execute({"path": "."})
    assert "real.py" in r.output
    assert ".git" not in r.output
    assert "node_modules" not in r.output


# ---------- shell ----------

def test_shell_echo(ws):
    r = ShellTool(ws, timeout=5).execute({"command": "echo hello"})
    assert r.success and "hello" in r.output


def test_shell_dangerous_command(ws):
    r = ShellTool(ws, timeout=5).execute({"command": "rm -rf /"})
    assert not r.success and "拦截" in r.error


def test_shell_timeout(ws):
    r = ShellTool(ws, timeout=1).execute(
        {"command": "python -c \"import time; time.sleep(5)\""}
    )
    assert not r.success and "超时" in r.error


def test_shell_nonzero_exit(ws):
    r = ShellTool(ws, timeout=5).execute({"command": "python -c \"import sys; sys.exit(3)\""})
    assert not r.success
    assert "3" in r.error
