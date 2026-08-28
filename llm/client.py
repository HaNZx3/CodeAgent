"""LLM 客户端：唯一与模型 API 通信的模块。

职责边界（重要）：
    - 只负责把 messages + tools 发给模型，并把厂商的原始响应
      标准化成统一的 ModelResponse / ToolCall 结构。
    - 不执行文件操作、不执行 shell、不决定何时结束、不修改 Agent 状态。

为什么需要统一结构？
    不同厂商（OpenAI / DeepSeek / Qwen ...）的响应字段略有差异，
    在客户端一次性适配成内部结构后，Agent Loop 就完全不用关心厂商细节，
    切换模型只需改 base_url / model 配置。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ToolCall:
    """一次标准化的工具调用。"""

    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """一次模型响应的标准化结构。

    - text:       模型直接给出的文本（最终回答或伴随文本），可能为 None。
    - tool_calls: 模型请求的工具调用列表，空列表表示这是一次最终回答。
    """

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """OpenAI 兼容接口的封装。

    通过 base_url 可切换到 DeepSeek / Qwen 等任何 OpenAI-compatible 服务。
    """

    def __init__(self, api_key: str, base_url: str | None = None, model: str = "gpt-4o-mini"):
        self.model = model
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        # 上次调用的真实 token 用量（来自 API 返回的 usage 字段）。
        # None 表示尚未调用过 LLM。/status 据此显示真实数据。
        self.last_usage: dict | None = None

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """调用模型并返回标准化响应。

        on_text 提供时走流式接口：每个文本增量实时回调（打字机效果），
        结束后仍返回完整统一的 ModelResponse；工具调用参数按分片拼装，
        对上层完全透明——无论是否流式，Agent Loop 拿到的结构一模一样。
        """
        params: dict = {"model": self.model, "messages": messages}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if on_text is None:
            completion = self._create_with_retry(params)
            self.last_usage = self._usage_to_dict(getattr(completion, "usage", None))
            message = completion.choices[0].message
            return self._normalize(message)

        # 流式：边收边转发文本增量，最后组装成与非流式相同的结构。
        params["stream"] = True
        # 请求最后一个 chunk 携带真实 usage（OpenAI 兼容服务多数支持；
        # 不支持时 chunk.usage 为 None，last_usage 保持上次值，不报错）。
        params["stream_options"] = {"include_usage": True}
        text_parts: list[str] = []
        # 工具调用的 id/name 首包出现，arguments 可能拆成多个分片，按 index 归槽拼接。
        acc: dict[int, dict] = {}
        stream_usage = None
        for chunk in self._create_with_retry(params):
            # usage 可能出现在任意 chunk（通常最后一个），且该 chunk 可能无 choices。
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                stream_usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                on_text(content)
            for frag in getattr(delta, "tool_calls", None) or []:
                idx = getattr(frag, "index", 0)
                slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if frag.id:
                    slot["id"] = frag.id
                fn = getattr(frag, "function", None)
                if fn and fn.name:
                    slot["name"] = fn.name
                if fn and fn.arguments:
                    slot["args"] += fn.arguments

        self.last_usage = self._usage_to_dict(stream_usage)

        tool_calls: list[ToolCall] = []
        for _, slot in sorted(acc.items()):
            try:
                arguments = json.loads(slot["args"]) if slot["args"] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=arguments))
        return ModelResponse(text="".join(text_parts) or None, tool_calls=tool_calls)

    def _create_with_retry(self, params: dict, attempts: int = 3):
        """带限流退避的 create 调用。

        免费模型（如 GLM Flash 系列）高峰期常返回 429「访问量过大」，
        对这类瞬时限流按 2s/4s/8s 指数退避重试；其他错误原样抛出。
        """
        delay = 2.0
        for attempt in range(attempts):
            try:
                return self._client.chat.completions.create(**params)
            except Exception as exc:
                rate_limited = (
                    type(exc).__name__ == "RateLimitError"
                    or getattr(exc, "status_code", None) == 429
                )
                if not rate_limited or attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def _usage_to_dict(usage) -> dict | None:
        """把 API 返回的 usage 对象转成 dict，便于 /status 展示真实数据。"""
        if usage is None:
            return None
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    @staticmethod
    def _normalize(message) -> ModelResponse:
        """把厂商原始 message 对象标准化为 ModelResponse。"""
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            arguments: dict = {}
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )
        return ModelResponse(text=message.content, tool_calls=tool_calls)
