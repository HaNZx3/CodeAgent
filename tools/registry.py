"""Tool Registry：管理所有可用工具，并向 Agent Loop 暴露统一入口。

为什么需要它？
    Agent Loop 不直接依赖具体 Tool 类，只通过 registry.get(name) /
    registry.execute(name, args) 来调度。新增工具只需 register 一次，
    Loop 完全不用改。这满足「Agent Core 不依赖具体 Tool」的原则。
"""

from __future__ import annotations

from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """把已注册工具转换成发给模型的 tool schema 列表。"""
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """调度工具执行，统一兜底异常。

        任何未知工具或工具抛出的异常，都转成失败结果交还给 LLM，
        而不是让 Agent Loop 直接崩溃。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"未知工具: {name}")
        try:
            return tool.execute(arguments)
        except Exception as exc:  # 兜底：工具异常 -> 失败结果交给 LLM 恢复
            return ToolResult.fail(f"{type(exc).__name__}: {exc}")
