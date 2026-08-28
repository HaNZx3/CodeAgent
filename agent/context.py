"""上下文管理：维护发给模型的完整 messages 列表。

为什么需要它？
    Agent 是多轮交互：user 任务 -> assistant(tool_call) -> tool 结果 -> ...
    这些消息必须按正确的角色和顺序拼成 API 需要的 messages。
    同时，工具输出可能非常大（一次 pytest 日志），必须截断，
    否则会撑爆上下文窗口。

三阶段能力：
    1. 基础：追加 system/user/assistant/tool 消息，工具输出截断。
    2. 持久化（Phase 2）：注入 store + session_id 后，每条消息同步落盘，
       进程重启后可 /resume <id> 恢复历史。
    3. 自动压缩（Phase 1）：maybe_compact 在调 LLM 前调用，用上次 API 返回的
       真实 prompt_tokens 判断超阈值时，把旧轮次摘要成一段 system 消息，
       保留最近 keep_recent 轮原文。不估算，全部用真实数据。
"""

from __future__ import annotations

import json
from collections.abc import Callable

from llm.client import ModelResponse, ToolCall
from tools.base import ToolResult

# 历史摘要消息在 messages[1] 的内容前缀；用于幂等守卫，避免对摘要再做摘要。
_SUMMARY_PREFIX = "[Previous conversation summary]"


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        max_tool_output: int = 8192,
        *,
        compact_threshold: int = 80_000,
        keep_recent: int = 6,
        summarizer: Callable[[list[dict]], str] | None = None,
        store=None,
        session_id: str | None = None,
    ):
        """维护 messages 列表。

        - system_prompt:   注入 messages[0]，全程不变。
        - max_tool_output: 单条工具输出上限，超出按「前 6KB + 标记 + 后 2KB」截断。
        - compact_threshold: 上次调用真实 prompt_tokens 超此值时触发 maybe_compact。
        - keep_recent:     压缩时保留最近 N 轮（一轮=user+后续 assistant/tool）原文。
        - summarizer:      把旧轮次摘要成字符串的回调；为 None 时优雅退化（不压缩）。
        - store + session_id: 注入后每条消息同步落盘，可跨进程恢复。
        """
        self.max_tool_output = max_tool_output
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self._summarizer = summarizer
        self.store = store
        self.session_id = session_id
        # 上次调 LLM 的真实 prompt_tokens（由 loop 从 API usage 同步）。
        # None 表示尚未调用过，maybe_compact 跳过——不估算，只用真实数据。
        self.last_prompt_tokens: int | None = None

        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # resume 时拉回历史（只含 user/assistant/tool，不含 system）。
        if store is not None and session_id is not None:
            self.messages += store.load(session_id)

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

    def clear(self, system_prompt: str | None = None) -> None:
        """清空对话历史（内存）。保留原 system prompt，或替换为新的。

        注意：本方法只清内存，不删 session 文件。如需「开新会话保留旧文件」
        的语义，应在 Agent 层调用 new_session()，而不是直接 clear()。
        """
        prompt = system_prompt if system_prompt is not None else self.messages[0]["content"]
        self.messages = [{"role": "system", "content": prompt}]

    def maybe_compact(self, force: bool = False) -> None:
        """超阈值时把旧轮次压缩成一段 summary system 消息。

        切分按「对话轮次」而非消息数，保证 recent 段以 user 开头、
        内部 assistant(tool_calls)+tool 配对完整——否则 OpenAI 兼容 API
        会因孤立 tool 消息直接 400。

        summarizer 失败时静默跳过，不让一次压缩失败拖垮整个 run。
        """
        # 只看真实 token：上次 API 返回的 prompt_tokens。None 表示尚未调用过 LLM，
        # 此时无法判断阈值——不压缩，等首次调用拿到 usage 后再说。不估算。
        if not force and (
            self.last_prompt_tokens is None
            or self.last_prompt_tokens < self.compact_threshold
        ):
            return
        if not self._summarizer:
            return  # 优雅退化：未注入摘要器时不压缩
        # 幂等守卫：messages[1] 已是摘要则不重复压缩。
        if (
            len(self.messages) > 1
            and isinstance(self.messages[1].get("content"), str)
            and self.messages[1]["content"].startswith(_SUMMARY_PREFIX)
        ):
            return
        cut = self._find_compact_split()
        if cut <= 0:
            return  # 不够轮次，无法压缩
        old = self.messages[1:cut]
        if not old:
            return  # 极端情况：无旧消息可压缩
        try:
            summary = self._summarizer(old)
        except Exception:
            return  # 压缩失败：保留原 messages，主流程继续
        self.messages = (
            self.messages[:1]
            + [{"role": "system", "content": f"{_SUMMARY_PREFIX}\n{summary}"}]
            + self.messages[cut:]
        )
        # 压缩改写了历史，整文件重写以保持落盘一致。
        if self.store is not None and self.session_id is not None:
            self.store.rewrite(self.session_id, self.messages[1:])

    def _find_compact_split(self) -> int:
        """返回 messages 上一个安全切分点索引。

        recent 段 = messages[cut:]，必须以 user 开头且包含最后 keep_recent 个 user。
        old 段 = messages[1:cut]，自包含（内部 tool_call/tool 配对完整）。

        不够 keep_recent 轮时返回 -1（不压缩）。
        """
        user_indices = [
            i for i, m in enumerate(self.messages)
            if i > 0 and m.get("role") == "user"
        ]
        if len(user_indices) <= self.keep_recent:
            return -1
        return user_indices[-self.keep_recent]

    def _persist(self, message: dict) -> None:
        """若注入了 store + session_id，把消息同步追加到会话文件。"""
        if self.store is not None and self.session_id is not None:
            self.store.append(self.session_id, message)

    def _truncate(self, text: str) -> str:
        """超过上限时保留「前 6KB + 标记 + 后 2KB」（默认 8KB）。"""
        limit = self.max_tool_output
        if len(text) <= limit:
            return text
        head = int(limit * 0.75)
        tail = int(limit * 0.25)
        return text[:head] + "\n...[output truncated]...\n" + text[-tail:]
