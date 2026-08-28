"""上下文管理：维护发给模型的完整 messages 列表。

为什么需要它？
    Agent 是多轮交互：user 任务 -> assistant(tool_call) -> tool 结果 -> ...
    这些消息必须按正确的角色和顺序拼成 API 需要的 messages。
    同时，工具输出可能非常大（一次 pytest 日志），必须截断，
    否则会撑爆上下文窗口。
"""

from __future__ import annotations

import json

from llm.client import ModelResponse, ToolCall
from tools.base import ToolResult


class ContextManager:
    def __init__(self, system_prompt: str, max_tool_output: int = 8192):
        self.max_tool_output = max_tool_output
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

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

        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": self._truncate(content),
            }
        )

    def get_messages(self) -> list[dict]:
        return self.messages

    def clear(self, system_prompt: str | None = None) -> None:
        """清空对话历史。保留原 system prompt，或替换为新的。

        用于交互式 REPL 中的 /clear 命令：让用户在不重启进程的情况下
        开启全新对话，丢弃之前所有轮次（含工具输出）。
        """
        prompt = system_prompt if system_prompt is not None else self.messages[0]["content"]
        self.messages = [{"role": "system", "content": prompt}]

    def _truncate(self, text: str) -> str:
        """超过上限时保留「前 6KB + 标记 + 后 2KB」（默认 8KB）。"""
        limit = self.max_tool_output
        if len(text) <= limit:
            return text
        head = int(limit * 0.75)
        tail = int(limit * 0.25)
        return text[:head] + "\n...[output truncated]...\n" + text[-tail:]
