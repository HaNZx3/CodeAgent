"""Shell 工具：在 workspace 内执行命令。

这是最需要限制的工具。第一版不追求操作系统级沙箱，但必须画出明确的安全边界：
    1. cwd 固定为 workspace
    2. 灾难级命令（rm -rf /、format、mkfs、dd、关机重启…）硬拒绝
    3. 破坏性但可经 /back 找回的命令（rm / del / rmdir / Remove-Item、
       > 覆写已存在文件）执行前须经用户确认
    4. 设置执行超时
    5. 限制 stdout/stderr 输出大小
"""

from __future__ import annotations

import re
import subprocess

from .base import RiskInfo, Tool, ToolResult
from .workspace import Workspace, WorkspaceError

MAX_OUTPUT = 8000  # 每个流最多保留的字符数

# 灾难级命令（不可逆 / 系统级）：直接拒绝，不询问。
# rm -rf 仅在目标为根/家目录/通配时才算灾难；workspace 内的 rm -rf subdir
# 属破坏性但可回退，走确认。
_HARD_BLOCK_PATTERNS = [
    r"\brm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*|-r\s+-f|-f\s+-r)\s+(/(\s|$|\*)|~(\s|$)|\*(\s|$)|\$HOME)",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\s+[a-z]:",
    r"\bmkfs\b",
    r"\bdd\s+if=",                   # dd 覆写磁盘
    r">\s*/dev/sd",                  # 重定向覆盖块设备
    r"\bchmod\s+-R\s+777\s+/",
    r":\(\)\s*\{.*\}\s*;",           # fork bomb（简化检测）
]

# 破坏性但可经 /back 找回：执行前确认（列出命令本身）。
_CONFIRM_PATTERNS = [
    r"\brm\b",                       # rm（含 rm -rf subdir，已排除灾难级）
    r"\bdel\b",                      # Windows del
    r"\brmdir\b",
    r"\bRemove-Item\b",
    r"\brd\s+/s\b",
]

_REDIRECT_RE = re.compile(r">+\s*([^\s|;&]+)")


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

    def risk(self, arguments: dict) -> RiskInfo | None:
        """删除/覆写类命令：先排除灾难级（交回 execute 硬拒），再确认。

        灾难级命中时返回 None——不重复询问，让 execute 给出「被拦截」。
        其余破坏性命令返回 RiskInfo（detail=完整命令），由 registry 弹窗。
        > 重定向仅在目标为 workspace 内已存在文件时才确认（创建新文件不算）。
        """
        command = arguments.get("command", "")
        if not command:
            return None
        if self._hard_block_reason(command):
            return None  # 灾难级，交回 execute 硬拒
        for pattern in _CONFIRM_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return RiskInfo(
                    action="执行删除类命令", detail=command, files=[],
                )
        # > 重定向覆写已存在文件：精确到文件级确认，避免 echo > 新文件也被打扰
        existing = self._redirect_existing_targets(command)
        if existing:
            return RiskInfo(
                action="命令重定向覆写已有文件", detail=command, files=existing,
            )
        return None

    def execute(self, arguments: dict) -> ToolResult:
        command = arguments.get("command", "")
        if not command.strip():
            return ToolResult.fail("command 不能为空")

        blocked = self._hard_block_reason(command)
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

    def _hard_block_reason(self, command: str) -> str | None:
        for pattern in _HARD_BLOCK_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return pattern
        return None

    def _redirect_existing_targets(self, command: str) -> list[str]:
        """提取 > / 2> / &> 重定向目标中、workspace 内已存在的文件路径。

        >>（追加）不算覆写，跳过。只确认「会清空已有文件」的重定向。
        """
        targets: list[str] = []
        for m in _REDIRECT_RE.finditer(command):
            if m.group(0).startswith(">>"):  # 追加，非破坏性
                continue
            tok = m.group(1).strip('"\'')
            if not tok:
                continue
            try:
                p = self.workspace.resolve(tok)
            except WorkspaceError:
                continue
            if p.is_file():
                targets.append(tok)
        return targets

    @staticmethod
    def _trim(text: str) -> str:
        if len(text) <= MAX_OUTPUT:
            return text
        head = int(MAX_OUTPUT * 0.8)
        tail = int(MAX_OUTPUT * 0.2)
        return text[:head] + "\n...[output truncated]...\n" + text[-tail:]
