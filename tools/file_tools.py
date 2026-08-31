"""文件相关工具：list_files / read_file / write_file / edit_file。

所有路径都先经过 Workspace 边界校验，越界即拒绝。
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import RiskInfo, Tool, ToolResult
from .workspace import Workspace, WorkspaceError, prune_ignored_dirs

MAX_READ_BYTES = 200 * 1024  # read_file 单次最多读取 200KB


class ListFilesTool(Tool):
    name = "list_files"
    description = "列出 workspace 内某个目录的结构（有限深度），用于先了解项目布局。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要列出的目录，默认 '.'"},
            "depth": {"type": "integer", "description": "递归深度，默认 3"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def execute(self, arguments: dict) -> ToolResult:
        """列出 workspace 内某目录的树形结构（限深度），展示项目布局。"""
        try:
            root = self.workspace.resolve(arguments.get("path", "."))
        except WorkspaceError as e:
            return ToolResult.fail(str(e))
        if not root.is_dir():
            return ToolResult.fail(f"不是目录: {arguments.get('path')}")

        depth = int(arguments.get("depth", 3))
        lines = self._walk(root, max_depth=depth)
        return ToolResult.ok("\n".join(lines) if lines else "(空目录)")

    def _walk(self, root: Path, max_depth: int) -> list[str]:
        """os.walk 遍历并格式化为缩进树，忽略无关目录、限制深度。"""
        lines: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # 剔除无关目录，避免 .git/node_modules 等撑爆输出
            prune_ignored_dirs(dirnames)
            rel = Path(dirpath).relative_to(root)
            current_depth = len(rel.parts)
            if current_depth > max_depth:
                dirnames[:] = []
                continue

            indent = "  " * current_depth
            # 根目录显示为 "./"，与工具"相对 workspace"的路径约定对齐，
            # 避免 LLM 把 workspace 名（如 demo/）当成路径前缀去 read_file。
            name = "." if current_depth == 0 else Path(dirpath).name
            lines.append(f"{indent}{name}/")
            for f in sorted(filenames):
                if f.startswith(".") or f.endswith((".pyc", ".pyo")):
                    continue
                lines.append(f"{indent}  {f}")
        return lines


class ReadFileTool(Tool):
    name = "read_file"
    description = "以 UTF-8 读取文本文件内容，返回完整文本。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的文件路径"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def execute(self, arguments: dict) -> ToolResult:
        """以 UTF-8 读取文本文件全文；越界 / 过大 / 非文本均返回失败。"""
        raw = arguments.get("path", "")
        try:
            path = self.workspace.resolve(raw)
        except WorkspaceError as e:
            return ToolResult.fail(str(e))

        if not path.exists():
            return ToolResult.fail(f"文件不存在: {raw}")
        if path.is_dir():
            return ToolResult.fail(f"目标是目录，请用 list_files: {raw}")

        try:
            data = path.read_bytes()
        except OSError as e:
            return ToolResult.fail(str(e))

        if len(data) > MAX_READ_BYTES:
            return ToolResult.fail(
                f"文件过大 ({len(data)} bytes)，超过单次读取上限 {MAX_READ_BYTES} bytes"
            )
        try:
            return ToolResult.ok(data.decode("utf-8"))
        except UnicodeDecodeError:
            return ToolResult.fail(f"不是 UTF-8 文本文件，无法读取: {raw}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "创建新文件或完整覆盖已有文件。修改已有代码请优先用 edit_file。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的文件路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def risk(self, arguments: dict) -> RiskInfo | None:
        """覆盖已有非空文件属破坏性操作：清空前确认。

        目标不存在或为空文件时不确认（创建新文件 / 改空文件无损失）。
        路径越界时交回 execute 报错，不在此处理。
        """
        raw = arguments.get("path", "")
        try:
            p = self.workspace.resolve(raw)
        except WorkspaceError:
            return None
        if p.is_file() and p.stat().st_size > 0:
            return RiskInfo(action="覆盖已有文件", detail=raw, files=[raw])
        return None

    def execute(self, arguments: dict) -> ToolResult:
        """创建新文件或完整覆盖已有文件（父目录自动创建）。"""
        raw = arguments.get("path", "")
        try:
            path = self.workspace.resolve(raw)
        except WorkspaceError as e:
            return ToolResult.fail(str(e))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(arguments.get("content", ""), encoding="utf-8")
            size = path.stat().st_size
        except OSError as e:
            return ToolResult.fail(str(e))

        return ToolResult.ok(f"已写入 {path.relative_to(self.workspace.root)} ({size} bytes)")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "对文件做一次精确替换：把第一次出现的 old_text 替换为 new_text。"
        "old_text 必须与文件当前内容完全一致，否则失败。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的文件路径"},
            "old_text": {"type": "string", "description": "要被替换的原文"},
            "new_text": {"type": "string", "description": "替换后的新文本"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def risk(self, arguments: dict) -> RiskInfo | None:
        """new_text 为空 = 删除一段代码，属破坏性操作：执行前确认。

        其余替换是常规编辑，不确认（old_text 不匹配时 execute 本就会失败）。
        """
        if arguments.get("new_text", "") == "":
            raw = arguments.get("path", "")
            return RiskInfo(action="删除代码片段（替换为空）", detail=raw, files=[raw])
        return None

    def execute(self, arguments: dict) -> ToolResult:
        """精确替换文件中首次出现的 old_text；old_text 不匹配则失败（避免误改）。"""
        raw = arguments.get("path", "")
        try:
            path = self.workspace.resolve(raw)
        except WorkspaceError as e:
            return ToolResult.fail(str(e))

        if not path.exists():
            return ToolResult.fail(f"文件不存在: {raw}")

        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult.fail(str(e))

        # 相比直接覆盖整个文件，精确替换更安全，也更符合真实代码修改场景。
        if old_text not in content:
            return ToolResult.fail(
                "未找到 old_text，无法替换。请先用 read_file 确认文件当前内容。"
            )

        new_content = content.replace(old_text, new_text, 1)
        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult.fail(str(e))

        return ToolResult.ok(f"已替换 1 处: {path.relative_to(self.workspace.root)}")
