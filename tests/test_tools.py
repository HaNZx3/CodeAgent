"""Tool 单元测试：read / write / edit / search / list / shell。"""

import pytest

from tools.base import RiskInfo
from tools.registry import ToolRegistry
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


# ---------- 高危检测（risk）----------

def test_shell_risk_rm_single_file(ws):
    """rm 单文件属破坏性：risk 命中，交 registry 确认。"""
    risk = ShellTool(ws).risk({"command": "rm a.txt"})
    assert risk is not None and "删除" in risk.action


def test_shell_risk_rm_rf_root_is_hard_block_not_confirm(ws):
    """rm -rf / 是灾难级：risk 返回 None（交回 execute 硬拒，不重复询问）。"""
    risk = ShellTool(ws).risk({"command": "rm -rf /"})
    assert risk is None
    r = ShellTool(ws, timeout=5).execute({"command": "rm -rf /"})
    assert not r.success and "拦截" in r.error


def test_shell_risk_rm_rf_subdir_is_confirmable(ws):
    """rm -rf <workspace 子目录> 非灾难级：走确认层，不硬拦。

    与 rm -rf /（灾难级、risk 返回 None + execute 硬拦）对照：此处 risk 命中。
    （不测真实删除：rm 在 Windows 不可用，跨平台只验分类逻辑。）
    """
    risk = ShellTool(ws).risk({"command": "rm -rf subdir"})
    assert risk is not None and "删除" in risk.action
    # execute 不硬拦（确认由 registry 门控；此处不验命令是否真跑得通）
    r = ShellTool(ws, timeout=5).execute({"command": "rm -rf subdir"})
    assert "拦截" not in (r.error or "")


def test_shell_risk_del_and_remove_item(ws):
    """Windows del / Remove-Item 也走确认。"""
    assert ShellTool(ws).risk({"command": "del a.txt"}) is not None
    assert ShellTool(ws).risk({"command": "Remove-Item a.txt"}) is not None


def test_shell_risk_redirect_overwrite_existing(ws):
    """> 覆写已存在文件才确认；写新文件不打扰。"""
    (ws.root / "old.txt").write_text("v1", encoding="utf-8")
    risk = ShellTool(ws).risk({"command": "echo v2 > old.txt"})
    assert risk is not None and "old.txt" in risk.files
    # 新文件：不确认
    assert ShellTool(ws).risk({"command": "echo v2 > new.txt"}) is None
    # >> 追加：不确认
    assert ShellTool(ws).risk({"command": "echo v2 >> old.txt"}) is None


def test_shell_risk_safe_command(ws):
    """普通命令无 risk。"""
    assert ShellTool(ws).risk({"command": "echo hello"}) is None
    assert ShellTool(ws).risk({"command": "pytest -q"}) is None


def test_write_file_risk_overwrite(ws):
    """覆盖已有非空文件确认；新建与覆盖空文件不确认。"""
    (ws.root / "a.txt").write_text("内容", encoding="utf-8")
    (ws.root / "empty.txt").write_text("", encoding="utf-8")
    assert WriteFileTool(ws).risk({"path": "a.txt", "content": "x"}) is not None
    assert WriteFileTool(ws).risk({"path": "empty.txt", "content": "x"}) is None
    assert WriteFileTool(ws).risk({"path": "new.txt", "content": "x"}) is None


def test_edit_file_risk_empty_new_text(ws):
    """new_text 为空 = 删除片段，确认；非空替换不确认。"""
    args_del = {"path": "c.py", "old_text": "x", "new_text": ""}
    args_mod = {"path": "c.py", "old_text": "x", "new_text": "y"}
    assert EditFileTool(ws).risk(args_del) is not None
    assert EditFileTool(ws).risk(args_mod) is None


# ---------- registry 确认门控 ----------

def test_registry_confirm_blocks_without_side_effect(ws):
    """confirm_callback 返回 False：工具不执行，无副作用。"""
    (ws.root / "a.txt").write_text("orig", encoding="utf-8")
    reg = ToolRegistry(confirm_callback=lambda risk: False)
    reg.register(WriteFileTool(ws))
    r = reg.execute("write_file", {"path": "a.txt", "content": "clobbered"})
    assert not r.success and "已取消" in r.error
    # 文件未被改动
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "orig"


def test_registry_confirm_allows_when_approved(ws):
    """confirm_callback 返回 True：照常执行。"""
    (ws.root / "a.txt").write_text("orig", encoding="utf-8")
    seen: list[RiskInfo] = []
    reg = ToolRegistry(confirm_callback=lambda risk: (seen.append(risk) or True))
    reg.register(WriteFileTool(ws))
    r = reg.execute("write_file", {"path": "a.txt", "content": "new"})
    assert r.success
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "new"
    assert seen and seen[0].action == "覆盖已有文件"


def test_registry_no_confirm_callback_means_no_gate(ws):
    """未设置 confirm_callback（默认）时不确认：直接执行，测试/非交互不受影响。"""
    (ws.root / "a.txt").write_text("orig", encoding="utf-8")
    reg = ToolRegistry()
    reg.register(WriteFileTool(ws))
    r = reg.execute("write_file", {"path": "a.txt", "content": "new"})
    assert r.success
    assert (ws.root / "a.txt").read_text(encoding="utf-8") == "new"
