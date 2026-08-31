"""工具层抽象与调度：Tool 基类 / ToolResult / RiskInfo + ToolRegistry。

为什么统一接口与注册表？
    所有工具返回统一的 ToolResult（success / output / error），Agent Loop 只
    检查这三个字段，新增工具无需改动 Loop。Loop 也不直接依赖具体 Tool 类，
    只通过 registry.schemas() / registry.execute() 调度——新增工具只需 register
    一次。高危操作（Tool.risk() 返回 RiskInfo）在 registry.execute 前经
    confirm_callback 询问用户，未获许可则跳过执行（无副作用）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

_TRUNCATE_MARKER = "\n...[output truncated]...\n"


def truncate_middle(text: str, limit: int, head_ratio: float = 0.75) -> str:
    """超长文本按「前缀 + 标记 + 后缀」截断。

    工具输出（命令 stdout、pytest 日志）与上下文里的 tool 结果都可能很长，
    统一用同一截断策略：保留 head_ratio 比例的开头 + 末尾尾巴，中间用固定
    标记隔开，让模型既看到开头关键信息又知道末尾被省略。
    """
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = int(limit * (1 - head_ratio))
    return text[:head] + _TRUNCATE_MARKER + text[-tail:]


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


class ToolRegistry:
    """管理所有可用工具，向 Agent Loop 暴露统一调度入口。

    高危确认：工具 risk() 返回 RiskInfo 且设置了 confirm_callback 时，execute
    前先询问；未获许可则直接返回失败，不触碰任何副作用。confirm_callback 默认
    None → 不确认（测试 / 非交互场景不受影响）。
    """

    def __init__(self, confirm_callback: Callable[[RiskInfo], bool] | None = None):
        self._tools: dict[str, Tool] = {}
        self.confirm_callback = confirm_callback

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

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
        if self.confirm_callback is not None:
            risk = tool.risk(arguments)
            if risk is not None and not self.confirm_callback(risk):
                return ToolResult.fail(f"已取消：{risk.action}（{risk.detail}）")
        try:
            return tool.execute(arguments)
        except Exception as exc:  # 兜底：工具异常 -> 失败结果交给 LLM 恢复
            return ToolResult.fail(f"{type(exc).__name__}: {exc}")
