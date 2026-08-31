"""Tool Registry：管理所有可用工具，并向 Agent Loop 暴露统一入口。

为什么需要它？
    Agent Loop 不直接依赖具体 Tool 类，只通过 registry.get(name) /
    registry.execute(name, args) 来调度。新增工具只需 register 一次，
    Loop 完全不用改。这满足「Agent Core 不依赖具体 Tool」的原则。
"""

from __future__ import annotations

from collections.abc import Callable

from .base import RiskInfo, Tool, ToolResult


class ToolRegistry:
    def __init__(self, confirm_callback: Callable[[RiskInfo], bool] | None = None):
        self._tools: dict[str, Tool] = {}
        # 高危确认回调：工具 risk() 返回 RiskInfo 时，registry 在 execute 前调用它。
        # 返回 False 则跳过执行（无副作用），返回 True 才放行。
        # 默认 None → 不确认（测试 / 非交互场景不受影响）。
        self.confirm_callback = confirm_callback

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

        高危确认：若工具自报 risk()，且设置了 confirm_callback，
        先询问用户；未获许可则直接返回失败，不触碰任何副作用。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"未知工具: {name}")
        if self.confirm_callback is not None:
            risk = tool.risk(arguments)
            if risk is not None and not self.confirm_callback(risk):
                return ToolResult.fail(f"已取消：{risk.action}（{risk.detail}）")
        try:
            return tool.execute(arguments)
        except Exception as exc:  # 兜底：工具异常 -> 失败结果交给 LLM 恢复
            return ToolResult.fail(f"{type(exc).__name__}: {exc}")
