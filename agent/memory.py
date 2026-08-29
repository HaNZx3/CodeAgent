"""项目级长期记忆：从 AGENT.md / USER.md 读取并注入 system prompt。

为什么需要它？
    仿 Claude Code 的 CLAUDE.md 机制：项目特定的构建命令、测试方式、
    代码约定、已知坑点等"项目记忆"放在 workspace/AGENT.md，每次 Agent
    启动自动注入 system prompt，无需用户重复交代。
    跨项目的个人偏好（编辑器、语言习惯）放 ~/.coding-agent/USER.md。
"""

from __future__ import annotations

from pathlib import Path


def load_project_memory(workspace: str | Path) -> str:
    """读取 workspace/AGENT.md 和 ~/.coding-agent/USER.md，拼成 system prompt 段落。

    两者都缺失时返回空串。OSError 容错跳过（文件读不到不应阻塞启动）。
    """
    parts: list[str] = []

    proj = Path(workspace).resolve() / "AGENT.md"
    try:
        content = proj.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"# Project Memory\n{content}")
    except OSError:
        pass

    user = Path.home() / ".coding-agent" / "USER.md"
    try:
        content = user.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"# User Preferences\n{content}")
    except OSError:
        pass

    if not parts:
        return ""
    return "\n\n".join(parts)
