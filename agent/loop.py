"""Agent Loop：整个系统的核心闭环。

    LLM -> Tool -> Environment -> Observation -> LLM

本模块只负责「循环」本身：调模型、解析、执行工具、把结果写回上下文、
检查停止条件。不包含任何具体工具实现。
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
    # 本次 run 期间 LLM 返回的真实用量累计（prompt/completion/total_tokens 与
    # 调用次数 calls）。全部来自 API usage 字段，不估算；calls=0 表示没有成功调用。
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})

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

    def run(
        self,
        task: str,
        on_step: Callable[[StepRecord], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_step_start: Callable[[str, dict], None] | None = None,
    ) -> RunResult:
        """执行一次完整任务，返回最终回答或停止原因。

        on_step 可选：每步工具执行完毕、以及模型在调用工具前给出的过程说明
        时立即回调，供 CLI 实时渲染，而不是等整个 run 结束后一次性打印。
        tool_name 为 None 的 StepRecord 表示模型的过程说明。

        on_text 可选：开启流式。模型输出的文本增量实时转发（打字机效果）；
        此时过程说明已通过流式渠道展示，不再走 on_step，避免重复打印。

        on_step_start 可选：工具开始执行前回调（带工具名与参数），
        供 CLI 在工具执行期间显示 spinner 等加载状态。
        """
        self.context.add_user(task)
        # 压缩在 stop.start() 之前：summarizer 的 LLM 调用不计入 max_runtime；
        # 新 task 已在末尾，必然落在 recent 段不会被压缩掉。
        self.context.maybe_compact()
        self.stop.start()

        consecutive_errors = 0
        steps: list[StepRecord] = []
        # 本次 run 的真实用量累计；每个返回点统一经 _result 附加到 RunResult。
        run_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

        def _result(final_text: str | None, stop_reason: str | None) -> RunResult:
            return RunResult(final_text, stop_reason, steps, dict(run_usage))

        for step in range(1, self.stop.max_steps + 1):
            # 停止条件 3：最大运行时间
            if self.stop.runtime_exceeded():
                return _result(
                    None,
                    f"maximum runtime ({self.stop.max_runtime}s) exceeded",
                )

            # 调用 LLM。LLM 出错时不直接崩溃，而是作为停止原因返回。
            # FakeLLM 等自定义 chat 不认识 on_text 参数，仅在开启流式时才传。
            try:
                kwargs = {"on_text": on_text} if on_text else {}
                response = self.llm.chat(
                    self.context.get_messages(), self.registry.schemas(), **kwargs
                )
                # 同步真实 prompt_tokens 给 context，下次 maybe_compact 据此判断阈值。
                # 不估算——一切用 API 返回的真实 usage。用 getattr 兼容测试 mock
                # 等未实现 last_usage 的 chat 实现（它们没有真实 usage 可同步）。
                usage = getattr(self.llm, "last_usage", None)
                if usage and "prompt_tokens" in usage:
                    self.context.last_prompt_tokens = usage["prompt_tokens"]
                    # 累计本次 run 的真实用量（RunResult.usage 供收尾行显示）。
                    p = usage.get("prompt_tokens", 0)
                    c = usage.get("completion_tokens", 0)
                    run_usage["prompt_tokens"] += p
                    run_usage["completion_tokens"] += c
                    run_usage["total_tokens"] += usage.get("total_tokens", p + c)
                    run_usage["calls"] += 1
            except Exception as exc:
                return _result(
                    None,
                    f"LLM 调用失败: {type(exc).__name__}: {exc}",
                )

            if response.has_tool_calls:
                # 有工具调用：先把 assistant(tool_calls) 写入上下文，再逐个执行。
                self.context.add_assistant(response)
                # 模型常会在调工具的同时说一句过程说明（如「我先看看文件」），
                # 把它也实时发出去，增强交互过程中的对话感。
                # 流式模式下该文本已逐字展示过（on_text），跳过以免重复打印。
                if response.text and on_step and on_text is None:
                    on_step(
                        StepRecord(
                            step=step, tool_name=None, arguments=None,
                            success=None, detail=response.text, duration_ms=0.0,
                        )
                    )
                for call in response.tool_calls:
                    if on_step_start:
                        on_step_start(call.name, call.arguments)
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

                    rec = StepRecord(
                        step=step,
                        tool_name=call.name,
                        arguments=call.arguments,
                        success=result.success,
                        detail=detail,
                        duration_ms=duration_ms,
                    )
                    steps.append(rec)
                    if on_step:
                        on_step(rec)

                    # 停止条件 4：连续工具失败
                    if self.stop.errors_exceeded(consecutive_errors):
                        return _result(
                            None,
                            "maximum consecutive tool errors "
                            f"({self.stop.max_consecutive_errors}) reached",
                        )
            else:
                # 停止条件 1：模型给出最终回答。
                # 必须把这条 assistant 消息写回上下文，否则多轮对话时
                # 模型不记得自己上一轮给用户的总结说过什么。
                self.context.add_assistant(response)
                return _result(response.text or "", None)

        # 停止条件 2：最大循环次数
        return _result(
            None,
            f"maximum number of steps ({self.stop.max_steps}) reached",
        )
