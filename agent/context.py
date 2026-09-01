"""上下文管理：维护发给模型的完整 messages 列表。

为什么需要它？
    Agent 是多轮交互：user 任务 -> assistant(tool_call) -> tool 结果 -> ...
    这些消息必须按正确的角色和顺序拼成 API 需要的 messages。
    同时，工具输出可能非常大（一次 pytest 日志），必须截断，
    否则会撑爆上下文窗口。

能力：
    1. 基础：追加 system/user/assistant/tool 消息，工具输出截断。
    2. 持久化：注入 store + session_id 后，每条消息同步落盘，进程重启后可 /resume <id> 恢复历史。
    3. 自动压缩：maybe_compact 在调 LLM 前调用，用上次 API 返回的真实 prompt_tokens
       判断是否触发——超阈值（默认按模型窗口的 80% 推导）或逼近窗口上限（兜底）
       时，把旧轮次摘要成一段 system 消息，保留最近 keep_recent 轮原文。
       不估算，全部用真实数据。
"""

from __future__ import annotations

import json
from collections.abc import Callable

from llm.client import ModelResponse, ToolCall
from tools.core import ToolResult, truncate_middle

# 历史摘要消息在 messages[1] 的内容前缀；用于幂等守卫，避免对摘要再做摘要。
_SUMMARY_PREFIX = "[Previous conversation summary]"

# 爆窗兜底：距窗口上限多少 tokens 内无条件强制压缩（即使未达 compact_threshold）。
# 阈值可能被配得比窗口还大（如换小窗口模型而沿用旧配置），兜底线保证不会
# 一直不压缩、直到请求超出窗口被 API 拒绝。余量给下一轮 user 消息 + 回复留空间。
_HARD_COMPACT_MARGIN = 8_000


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        max_tool_output: int = 8192,
        *,
        compact_threshold: int = 80_000,
        keep_recent: int = 6,
        context_window: int = 128_000,
        summarizer: Callable[[list[dict]], str] | None = None,
        store=None,
        session_id: str | None = None,
        on_user_turns_pruned: Callable[[int], None] | None = None,
    ):
        """维护 messages 列表。

        - system_prompt:   注入 messages[0]，全程不变。
        - max_tool_output: 单条工具输出上限，超出按「前 6KB + 标记 + 后 2KB」截断。
        - compact_threshold: 上次调用真实 prompt_tokens 超此值时触发 maybe_compact。
        - keep_recent:     压缩时保留最近 N 轮（一轮=user+后续 assistant/tool）原文。
        - context_window:  模型上下文窗口，用于逼近上限时的兜底强制压缩。
        - summarizer:      把旧轮次摘要成字符串的回调；为 None 时优雅退化（不压缩）。
        - store + session_id: 注入后每条消息同步落盘，可跨进程恢复。
        - on_user_turns_pruned: 压缩丢弃旧用户轮次后回调（参数=丢弃数），
          供快照账本同步修剪，维持「轮次 <-> 快照」1:1。
        """
        self.max_tool_output = max_tool_output
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self.context_window = context_window
        self._summarizer = summarizer
        self.store = store
        self.session_id = session_id
        self._on_user_turns_pruned = on_user_turns_pruned
        # 上次调 LLM 的真实 prompt_tokens（由 loop 从 API usage 同步）。
        # None 表示尚未调用过，maybe_compact 跳过——不估算，只用真实数据。
        self.last_prompt_tokens: int | None = None

        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # resume 时拉回历史（只含 user/assistant/tool，不含 system）。
        if store is not None and session_id is not None:
            self.messages += store.load(session_id)

    def reset(self) -> None:
        """原地清空对话历史：仅保留 system 消息，会话 id 不变。

        会话文件同步删除——否则内存已清空而文件还在，下次 add_user
        会把新消息 append 到旧历史后面，/resume 会得到不一致的状态。
        last_prompt_tokens 一并归零：清空前的真实用量不再代表当前上下文。
        """
        self.messages = [self.messages[0]]
        self.last_prompt_tokens = None
        if self.store is not None and self.session_id is not None:
            self.store.clear(self.session_id)

    def add_user(self, text: str) -> None:
        msg = {"role": "user", "content": text}
        self.messages.append(msg)
        self._persist(msg)

    def add_assistant(self, response: ModelResponse) -> None:
        """把模型的 assistant 消息（含可能存在的 tool_calls）加入上下文。"""
        message: dict = {"role": "assistant"}
        if response.text:
            message["content"] = response.text
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
        self.messages.append(message)
        self._persist(message)

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """把一次工具执行结果以 role=tool 消息加入上下文。

        失败时把错误信息写清楚，让模型能够据此自行恢复，
        而不是让 Agent 直接终止（这是 Agent 与普通脚本的核心区别）。
        """
        if result.success:
            content = result.output
        else:
            content = f"Tool execution failed.\n\nError:\n{result.error or 'unknown'}"
            if result.output:
                content += f"\n\nOutput:\n{result.output}"

        message = {
            "role": "tool",
            "tool_call_id": call.id,
            "content": self._truncate(content),
        }
        self.messages.append(message)
        self._persist(message)

    def get_messages(self) -> list[dict]:
        return self.messages

    def user_turns(self) -> list[int]:
        """可回退的用户消息索引（/back 的候选列表）。

        role 过滤天然排除 system 与历史摘要（messages[1]，role=system）：
        摘要替换掉的轮次已物理删除，无从回退；摘要之后的轮次照常可回退。
        """
        return [
            i for i, m in enumerate(self.messages)
            if i > 0 and m.get("role") == "user"
        ]

    def rewind_to(self, index: int) -> int:
        """回退到 messages[index]（一条用户消息）之前：删除它及其后所有消息。

        返回删除的消息数。落盘文件同步重写；last_prompt_tokens 置空——
        旧的真实用量属于已回退的历史，不能再用于压缩判断，下次调用后重新取。
        index 必须来自 user_turns()，否则抛 ValueError（保证回退点是
        user 轮次边界，不会留下无 tool 结果的孤儿 tool_call）。
        """
        if index not in self.user_turns():
            raise ValueError(f"非法回退点：messages[{index}] 不是可回退的用户消息")
        removed = len(self.messages) - index
        self.messages = self.messages[:index]
        self.last_prompt_tokens = None
        if self.store is not None and self.session_id is not None:
            self.store.rewrite(self.session_id, self.messages[1:])
        return removed

    def maybe_compact(self, force: bool = False) -> str:
        """超阈值或逼近窗口上限时，把旧轮次压缩成一段 summary system 消息。

        触发线取两者较小值（双触发，Claude Code 同款思路）：
          1. 常规阈值 compact_threshold——上下文长了就整理，保持低占用；
          2. 爆窗兜底线（窗口 − 8K，极小窗口下不低于 60%）——阈值被配得
             比窗口还大时，逼近上限也强制压缩，避免请求被 API 直接拒绝。

        切分按「对话轮次」而非消息数，保证 recent 段以 user 开头、
        内部 assistant(tool_calls)+tool 配对完整——否则 OpenAI 兼容 API
        会因孤立 tool 消息直接 400。

        summarizer 失败时静默跳过，不让一次压缩失败拖垮整个 run。

        返回结果码，供 /compact 区分展示（自动路径忽略返回值）：
          "ok"              压缩成功；
          "already_summary" 历史已是摘要（幂等守卫）；
          "few_turns"       user 轮次不足，没有可压缩的旧轮次；
          "failed"          summarizer 异常，已保留原 messages；
          "noop"            其他无需压缩情形（未达阈值/无 summarizer 等）。
        """
        # 只看真实 token：上次 API 返回的 prompt_tokens。None 表示尚未调用过
        # LLM，此时无法判断——不压缩，等首次调用拿到 usage 后再说。不估算。
        tokens = self.last_prompt_tokens
        if not force:
            if tokens is None:
                return "noop"
            # 兜底线在极小窗口下钳到不低于 60%，避免每轮都强制压缩。
            hard_limit = max(
                self.context_window - _HARD_COMPACT_MARGIN,
                self.context_window * 3 // 5,
            )
            if tokens < min(self.compact_threshold, hard_limit):
                return "noop"
        if not self._summarizer:
            return "noop"  # 优雅退化：未注入摘要器时不压缩
        # 幂等守卫：messages[1] 已是摘要则不重复压缩。
        if (
            len(self.messages) > 1
            and isinstance(self.messages[1].get("content"), str)
            and self.messages[1]["content"].startswith(_SUMMARY_PREFIX)
        ):
            return "already_summary"
        cut = self._find_compact_split()
        if cut <= 0:
            return "few_turns"  # 不够轮次，无法压缩
        old = self.messages[1:cut]
        if not old:
            return "noop"  # 极端情况：无旧消息可压缩
        try:
            summary = self._summarizer(old)
        except Exception:
            return "failed"  # 压缩失败：保留原 messages，主流程继续
        self.messages = (
            self.messages[:1]
            + [{"role": "system", "content": f"{_SUMMARY_PREFIX}\n{summary}"}]
            + self.messages[cut:]
        )
        # 压缩改写了历史，整文件重写以保持落盘一致。
        if self.store is not None and self.session_id is not None:
            self.store.rewrite(self.session_id, self.messages[1:])
        # 通知快照账本同步修剪（被摘要的轮次不可再回退）。回调失败只影响
        # 代码快照对齐，不影响压缩结果本身。
        dropped = sum(1 for m in old if m.get("role") == "user")
        if dropped and self._on_user_turns_pruned is not None:
            try:
                self._on_user_turns_pruned(dropped)
            except Exception:
                pass
        return "ok"

    def _find_compact_split(self) -> int:
        """返回 messages 上一个安全切分点索引。

        recent 段 = messages[cut:]，必须以 user 开头且包含最后 keep_recent 个 user。
        old 段 = messages[1:cut]，自包含（内部 tool_call/tool 配对完整）。

        不够 keep_recent 轮时返回 -1（不压缩）。
        """
        user_indices = self.user_turns()
        if len(user_indices) <= self.keep_recent:
            return -1
        return user_indices[-self.keep_recent]

    def _persist(self, message: dict) -> None:
        """若注入了 store + session_id，把消息同步追加到会话文件。"""
        if self.store is not None and self.session_id is not None:
            self.store.append(self.session_id, message)

    def _truncate(self, text: str) -> str:
        """超过上限时保留「前 6KB + 标记 + 后 2KB」（默认 8KB）。"""
        return truncate_middle(text, self.max_tool_output, head_ratio=0.75)
