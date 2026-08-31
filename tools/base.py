"""Tool 抽象基类与统一的 ToolResult。

为什么需要统一接口？
    如果每个 Tool 的返回值格式都不一样，Agent Loop 就要为每个 Tool
    写一份错误处理逻辑。统一成 ToolResult（success / output / error）
    之后，Loop 只需要检查这三个字段，新增 Tool 无需改动 Loop。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolResult:
    """所有工具统一返回的结构。"""

    success: bool
    output: str
    error: str | None = None

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, output="", error=error)


@dataclass
class RiskInfo:
    """高危操作描述：供 ToolRegistry 在 execute 前弹窗确认。

    action  动作描述，如「删除文件」「覆盖已有文件」「执行删除命令」
    detail  具体内容，如文件路径或完整 shell 命令
    files   受影响的 workspace 相对路径（尽力而为，可为空）
    """

    action: str
    detail: str
    files: list[str]


class Tool:
    """工具基类。

    每个具体工具需要提供三个类属性 + 一个方法：
        name        工具名
        description 工具用途（会进入 system 的 tool schema）
        parameters  JSON Schema（描述参数）
        execute()   本地执行函数，返回 ToolResult

    可选重写 risk()：若本次调用属高危操作，返回 RiskInfo，
    ToolRegistry 会在 execute 前据此询问用户；返回 None 表示无需确认。
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def execute(self, arguments: dict) -> ToolResult:
        raise NotImplementedError

    def risk(self, arguments: dict) -> RiskInfo | None:
        """默认无高危检测；具体工具按需重写。"""
        return None

    def schema(self) -> dict:
        """转换成 OpenAI tool-calling 需要的 function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
