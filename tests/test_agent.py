"""Agent Loop 测试：用 FakeLLM 模拟「tool call -> result -> ... -> final」的循环。"""

from llm.client import ModelResponse, ToolCall
from tools.base import Tool, ToolResult
from tools.registry import ToolRegistry
from agent.context import ContextManager
from agent.stop import StopController
from agent.loop import AgentLoop


class FakeLLM:
    """按脚本顺序返回预设响应，返回完后再给最终回答。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(text="done")


class AlwaysToolCallLLM:
    """永远返回一次工具调用。"""

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        return ModelResponse(tool_calls=[ToolCall(id=str(self.n), name="record", arguments={})])


class RecordingTool(Tool):
    name = "record"
    description = "record"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return ToolResult.ok("ok")


class FailTool(Tool):
    name = "record"
    description = "record"
    parameters = {"type": "object", "properties": {}}

    def execute(self, arguments):
        return ToolResult.fail("boom")


def make_loop(llm, registry, max_steps=5, max_runtime=10, max_errors=3):
    ctx = ContextManager("sys")
    stop = StopController(max_steps, max_runtime, max_errors)
    return AgentLoop(llm, registry, ctx, stop)


def test_loop_tool_then_final():
    tool = RecordingTool()
    registry = ToolRegistry()
    registry.register(tool)

    llm = FakeLLM([
        ModelResponse(tool_calls=[ToolCall(id="1", name="record", arguments={"a": 1})]),
        ModelResponse(text="完成"),
    ])
    result = make_loop(llm, registry).run("task")

    assert result.final_text == "完成"
    assert result.stop_reason is None
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "record"
    assert result.steps[0].success is True
    assert tool.calls == [{"a": 1}]


def test_loop_stops_on_max_steps():
    registry = ToolRegistry()
    registry.register(RecordingTool())
    result = make_loop(AlwaysToolCallLLM(), registry, max_steps=2, max_errors=100).run("task")

    assert result.final_text is None
    assert "maximum number of steps" in result.stop_reason
    assert len(result.steps) == 2


def test_loop_stops_on_consecutive_errors():
    registry = ToolRegistry()
    registry.register(FailTool())
    result = make_loop(AlwaysToolCallLLM(), registry, max_steps=20, max_errors=2).run("task")

    assert result.final_text is None
    assert "consecutive" in result.stop_reason
    assert len(result.steps) == 2


def test_loop_handles_llm_error():
    class BoomLLM:
        def chat(self, messages, tools=None):
            raise RuntimeError("network down")

    result = make_loop(BoomLLM(), ToolRegistry()).run("task")
    assert result.final_text is None
    assert "LLM 调用失败" in result.stop_reason


def test_loop_unknown_tool_is_failure():
    # 模型请求一个未注册的工具，registry 应返回失败结果并交给模型恢复。
    class UnknownToolCallLLM:
        def chat(self, messages, tools=None):
            return ModelResponse(tool_calls=[ToolCall(id="1", name="ghost", arguments={})])

    result = make_loop(UnknownToolCallLLM(), ToolRegistry(), max_errors=1).run("task")
    assert result.final_text is None
    assert "consecutive" in result.stop_reason
    assert "未知工具" in result.steps[0].detail
