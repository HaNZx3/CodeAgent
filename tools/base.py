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


class Tool:
    """工具基类。

    每个具体工具需要提供三个类属性 + 一个方法：
        name        工具名
        description 工具用途（会进入 system 的 tool schema）
        parameters  JSON Schema（描述参数）
        execute()   本地执行函数，返回 ToolResult
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def execute(self, arguments: dict) -> ToolResult:
        raise NotImplementedError

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
