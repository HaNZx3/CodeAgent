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

    def chat(self, messages: list, tools: list | None = None) -> ModelResponse:
        """调用模型并返回标准化响应。"""
        params: dict = {"model": self.model, "messages": messages}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        completion = self._client.chat.completions.create(**params)
        message = completion.choices[0].message
        return self._normalize(message)

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
