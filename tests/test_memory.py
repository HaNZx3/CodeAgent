"""load_project_memory 单元测试：AGENT.md / USER.md 注入。"""

from pathlib import Path

from agent.memory import load_project_memory


def test_no_memory_files_returns_empty(tmp_path):
    # workspace 无 AGENT.md，且 USER.md 默认不存在
    assert load_project_memory(tmp_path) == ""


def test_project_memory_loaded(tmp_path):
    (tmp_path / "AGENT.md").write_text("- 测试：pytest\n- 包管理：uv", encoding="utf-8")
    result = load_project_memory(tmp_path)
    assert "# Project Memory" in result
    assert "测试：pytest" in result
    assert "包管理：uv" in result


def test_user_memory_loaded(tmp_path, monkeypatch):
    # 把 Path.home() 重定向到 tmp_path，写 USER.md
    fake_home = tmp_path / "fakehome"
    (fake_home / ".coding-agent").mkdir(parents=True)
    (fake_home / ".coding-agent" / "USER.md").write_text(
        "偏好：使用类型注解", encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    result = load_project_memory(tmp_path)
    assert "# User Preferences" in result
    assert "使用类型注解" in result


def test_both_memories_joined(tmp_path, monkeypatch):
    (tmp_path / "AGENT.md").write_text("项目内容", encoding="utf-8")
    fake_home = tmp_path / "fakehome"
    (fake_home / ".coding-agent").mkdir(parents=True)
    (fake_home / ".coding-agent" / "USER.md").write_text("用户内容", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    result = load_project_memory(tmp_path)
    # 两段用 \n\n 连接
    assert "# Project Memory\n项目内容" in result
    assert "# User Preferences\n用户内容" in result
    assert result.index("# Project Memory") < result.index("# User Preferences")


def test_project_memory_missing_file_safe(tmp_path):
    # 不存在 AGENT.md 不应抛异常
    result = load_project_memory(tmp_path / "nonexistent_subdir")
    assert result == ""


def test_relative_workspace_resolved(tmp_path, monkeypatch):
    # 用相对路径传 workspace，resolve 后仍能读到 AGENT.md
    abs_ws = tmp_path / "proj"
    abs_ws.mkdir()
    (abs_ws / "AGENT.md").write_text("相对路径测试", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = load_project_memory("proj")
    assert "相对路径测试" in result
