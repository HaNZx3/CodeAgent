"""Shell 工具：在 workspace 内执行命令。

这是最需要限制的工具。第一版不追求操作系统级沙箱，但必须画出明确的安全边界：
    1. cwd 固定为 workspace
    2. 命令黑名单拦截明显危险命令
    3. 设置执行超时
    4. 限制 stdout/stderr 输出大小
"""

from __future__ import annotations

import re
import subprocess

from .base import Tool, ToolResult
from .workspace import Workspace

MAX_OUTPUT = 8000  # 每个流最多保留的字符数

# 危险命令黑名单（正则，忽略大小写）。这是「明确的安全边界」，而非完整沙箱。
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",                 # rm -rf ...
    r"\brm\s+-r\s+-f\b",             # rm -r -f ...
    r"\bdel\s+/[fsq]\b",             # Windows del /s /f /q
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\s+[a-z]:",            # format C:
    r"\bmkfs\b",
    r"\bdd\s+if=",                   # dd 覆写磁盘
    r">\s*/dev/sd",                  # 重定向覆盖块设备
    r"\bchmod\s+-R\s+777\s+/",
    r":\(\)\s*\{.*\}\s*;",           # fork bomb（简化检测）
]


class ShellTool(Tool):
    name = "run_command"
    description = "在 workspace 内执行一条 shell 命令，返回 stdout/stderr。用于运行测试、构建等。"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令"},
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Workspace, timeout: float = 30.0):
        self.workspace = workspace
        self.timeout = timeout

    def execute(self, arguments: dict) -> ToolResult:
        command = arguments.get("command", "")
        if not command.strip():
            return ToolResult.fail("command 不能为空")

        blocked = self._blocked_reason(command)
        if blocked:
            return ToolResult.fail(f"危险命令被拦截（匹配规则: {blocked}）")

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace.root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"命令超时（>{self.timeout}s）: {command}")
        except OSError as e:
            return ToolResult.fail(str(e))

        stdout = self._trim(proc.stdout or "")
        stderr = self._trim(proc.stderr or "")

        parts: list[str] = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if proc.returncode != 0:
            parts.append(f"[exit code] {proc.returncode}")
        output = "\n".join(parts) if parts else "(无输出)"

        # 非零退出码：把输出也交还给 LLM 以便其分析失败原因，但标记为失败。
        if proc.returncode != 0:
            return ToolResult(
                success=False, output=output, error=f"命令退出码 {proc.returncode}"
            )
        return ToolResult.ok(output)

    def _blocked_reason(self, command: str) -> str | None:
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return pattern
        return None

    @staticmethod
    def _trim(text: str) -> str:
        if len(text) <= MAX_OUTPUT:
            return text
        head = int(MAX_OUTPUT * 0.8)
        tail = int(MAX_OUTPUT * 0.2)
        return text[:head] + "\n...[output truncated]...\n" + text[-tail:]
