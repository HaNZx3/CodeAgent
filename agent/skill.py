"""轻量 Skill 机制（可选特色）。

Skill 不是可执行脚本，而是一段 markdown 描述的行为指引。
Agent 启动时读取 skills/ 下每个 skill.md，追加到 System Prompt 中，
从而扩展 Agent 的行为模式。Skill 本身不获得任何宿主机代码执行权限。
"""

from __future__ import annotations

from pathlib import Path


def load_skills(skills_dir: str | Path) -> str:
    root = Path(skills_dir)
    if not root.is_dir():
        return ""

    parts: list[str] = []
    for skill_md in sorted(root.rglob("skill.md")):
        try:
            content = skill_md.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            parts.append(f"<skill>\n{content}\n</skill>")

    if not parts:
        return ""
    return "\n\n# 可用 Skills\n" + "\n\n".join(parts)
