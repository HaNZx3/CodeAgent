"""停止条件控制：防止 Agent 无限循环。

为什么需要它？
    模型可能反复调用工具或反复出错，如果不设上限，会陷入死循环、
    烧 token 甚至反复执行命令。显式的停止条件是「Agent 可解释性」的关键一环，
    也是题目明确要求自行设计的核心逻辑。
"""

from __future__ import annotations

import time


class StopController:
    """管理 4 类停止条件中的 3 类：

    1. 模型产生最终回答（由 Agent Loop 判断，无需这里处理）
    2. 最大循环次数（由 Agent Loop 的 for range 保证）
    3. 最大运行时间
    4. 连续工具失败次数
    """

    def __init__(
        self,
        max_steps: int = 20,
        max_runtime: float = 300.0,
        max_consecutive_errors: int = 3,
    ):
        self.max_steps = max_steps
        self.max_runtime = max_runtime
        self.max_consecutive_errors = max_consecutive_errors
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()

    def runtime_exceeded(self) -> bool:
        assert self._start is not None, "请先调用 start()"
        return (time.monotonic() - self._start) > self.max_runtime

    def errors_exceeded(self, consecutive_errors: int) -> bool:
        return consecutive_errors >= self.max_consecutive_errors
