"""搜索工具：在项目内定位代码。

意义：
    Agent 先通过 search_code 定位相关代码，再 read_file 读取具体文件，
    而不是盲目读取整个项目。优先使用标准库 os.walk + re，不引入复杂框架。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import Tool, ToolResult
from .file_tools import IGNORED_DIRS
from .workspace import Workspace, WorkspaceError

TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".js", ".ts", ".html", ".css", ".sh", ".bat",
}
MAX_RESULTS = 100


class SearchTool(Tool):
    name = "search_code"
    description = "在指定目录内按关键词（子串或正则）搜索代码，返回 文件:行号 列表。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要搜索的关键词或正则表达式"},
            "path": {"type": "string", "description": "搜索起点目录，默认 '.'"},
            "regex": {"type": "boolean", "description": "是否按正则匹配，默认 false"},
        },
        "required": ["query"],
    }

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def execute(self, arguments: dict) -> ToolResult:
        query = arguments.get("query", "")
        if not query:
            return ToolResult.fail("query 不能为空")

        try:
            root = self.workspace.resolve(arguments.get("path", "."))
        except WorkspaceError as e:
            return ToolResult.fail(str(e))
        if not root.is_dir():
            return ToolResult.fail(f"不是目录: {arguments.get('path')}")

        use_regex = bool(arguments.get("regex", False))
        pattern = None
        if use_regex:
            try:
                pattern = re.compile(query)
            except re.error as e:
                return ToolResult.fail(f"非法正则: {e}")

        matches: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")
            ]
            for f in sorted(filenames):
                if f.startswith("."):
                    continue
                file_path = Path(dirpath) / f
                if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    hit = (pattern.search(line) is not None) if pattern else (query in line)
                    if hit:
                        rel = file_path.relative_to(self.workspace.root)
                        matches.append(f"{rel}:{lineno}: {line.strip()}")
                        if len(matches) >= MAX_RESULTS:
                            return ToolResult.ok(self._format(matches))
        return ToolResult.ok(self._format(matches))

    @staticmethod
    def _format(matches: list[str]) -> str:
        if not matches:
            return "未找到匹配结果"
        return f"共 {len(matches)} 条匹配:\n" + "\n".join(matches)
