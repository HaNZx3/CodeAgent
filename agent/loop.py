"""Agent Loop：整个系统的核心闭环。

    LLM -> Tool -> Environment -> Observation -> LLM

本模块只负责「循环」本身：调模型、解析、执行工具、把结果写回上下文、
检查停止条件。不包含任何具体工具实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm.client import LLMClient
from tools.registry import ToolRegistry
from .context import ContextManager
from .stop import StopController


@dataclass
class StepRecord:
    """单次工具执行的观测记录（用于日志 / 视频展示）。"""

    step: int
    tool_name: str | None
    arguments: dict | None
    success: bool | None
    detail: str
    duration_ms: float


@dataclass
class RunResult:
    """一次 run() 的完整结果。"""

    final_text: str | None
    stop_reason: str | None
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.final_text is not None


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        context: ContextManager,
        stop: StopController,
    ):
        self.llm = llm
        self.registry = registry
        self.context = context
        self.stop = stop

    def run(self, task: str) -> RunResult:
        """执行一次完整任务，返回最终回答或停止原因。"""
        self.context.add_user(task)
        self.stop.start()

        consecutive_errors = 0
        steps: list[StepRecord] = []

        for step in range(1, self.stop.max_steps + 1):
            # 停止条件 3：最大运行时间
            if self.stop.runtime_exceeded():
                return RunResult(
                    None,
                    f"maximum runtime ({self.stop.max_runtime}s) exceeded",
                    steps,
                )

            # 调用 LLM。LLM 出错时不直接崩溃，而是作为停止原因返回。
            try:
                response = self.llm.chat(
                    self.context.get_messages(), self.registry.schemas()
                )
            except Exception as exc:
                return RunResult(
                    None,
                    f"LLM 调用失败: {type(exc).__name__}: {exc}",
                    steps,
                )

            if response.has_tool_calls:
                # 有工具调用：先把 assistant(tool_calls) 写入上下文，再逐个执行。
                self.context.add_assistant(response)
                for call in response.tool_calls:
                    t0 = time.monotonic()
                    result = self.registry.execute(call.name, call.arguments)
                    duration_ms = (time.monotonic() - t0) * 1000

                    self.context.add_tool_result(call, result)

                    if result.success:
                        consecutive_errors = 0
                        detail = result.output
                    else:
                        consecutive_errors += 1
                        detail = result.error or result.output or "unknown error"

                    steps.append(
                        StepRecord(
                            step=step,
                            tool_name=call.name,
                            arguments=call.arguments,
                            success=result.success,
                            detail=detail,
                            duration_ms=duration_ms,
                        )
                    )

                    # 停止条件 4：连续工具失败
                    if self.stop.errors_exceeded(consecutive_errors):
                        return RunResult(
                            None,
                            "maximum consecutive tool errors "
                            f"({self.stop.max_consecutive_errors}) reached",
                            steps,
                        )
            else:
                # 停止条件 1：模型给出最终回答
                return RunResult(response.text or "", None, steps)

        # 停止条件 2：最大循环次数
        return RunResult(
            None,
            f"maximum number of steps ({self.stop.max_steps}) reached",
            steps,
        )
