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


def test_final_answer_written_to_context():
    # 多轮对话的前提：最终回答必须写回上下文，否则下一轮模型不记得自己说过什么。
    registry = ToolRegistry()
    registry.register(RecordingTool())
    llm = FakeLLM([ModelResponse(text="这是我的总结")])
    ctx = ContextManager("sys")
    loop = AgentLoop(llm, registry, ctx, StopController(5, 10, 3))

    loop.run("task")

    messages = ctx.get_messages()
    last = messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "这是我的总结"


def test_on_step_callback_reports_each_tool_execution():
    tool = RecordingTool()
    registry = ToolRegistry()
    registry.register(tool)

    llm = FakeLLM([
        ModelResponse(text="我先记录一下", tool_calls=[
            ToolCall(id="1", name="record", arguments={"a": 1}),
        ]),
        ModelResponse(text="完成"),
    ])
    seen: list = []
    result = make_loop(llm, registry).run("task", on_step=seen.append)

    # 过程说明 + 工具执行都应实时回调；最终回答不触发 on_step（由 render 收尾）。
    # 注意 steps 只含工具记录，过程说明只走回调。
    assert len(seen) == 2
    assert seen[0].tool_name is None and seen[0].detail == "我先记录一下"
    assert seen[1].tool_name == "record" and seen[1].success is True
    assert [r for r in seen if r.tool_name is not None] == result.steps


def test_run_without_callback_keeps_api_compatible():
    registry = ToolRegistry()
    registry.register(RecordingTool())
    result = make_loop(FakeLLM([]), registry).run("task")  # 不传 on_step
    assert result.final_text == "done"
    assert len(result.steps) == 0


def test_on_text_streaming_receives_deltas_without_note_duplicates():
    # 流式：两轮对话（过程说明+工具调用 -> 最终回答），增量实时回调，
    # 且同一文本不再以过程说明事件重复发出。
    calls = {"n": 0}

    class StreamingTwoRoundLLM:
        def chat(self, messages, tools=None, on_text=None):
            calls["n"] += 1
            if calls["n"] == 1:
                if on_text:
                    on_text("我先看看")
                    on_text("文件")
                return ModelResponse(text="我先看看文件", tool_calls=[
                    ToolCall(id="1", name="record", arguments={}),
                ])
            if on_text:
                for piece in ("结", "论"):
                    on_text(piece)
            return ModelResponse(text="结论")

    registry = ToolRegistry()
    registry.register(RecordingTool())
    deltas: list = []
    events: list = []
    result = make_loop(StreamingTwoRoundLLM(), registry).run(
        "task", on_step=events.append, on_text=deltas.append
    )

    # 跨轮次的全部文本增量都实时到达：过程说明 + 最终回答。
    assert deltas == ["我先看看", "文件", "结", "论"]
    # 流式模式下过程说明不再以事件重复发出，但工具执行记录照常保留。
    assert [rec for rec in events if rec.tool_name is None] == []
    assert [rec for rec in events if rec.tool_name is not None] == result.steps
    assert result.final_text == "结论"
    assert len(result.steps) == 1


def test_streamed_final_answer_forwarded_and_written_back():
    class StreamyFinalLLM:
        def chat(self, messages, tools=None, on_text=None):
            if on_text:
                for piece in ("结", "论"):
                    on_text(piece)
            return ModelResponse(text="结论")

    ctx = ContextManager("sys")
    deltas: list = []
    result = AgentLoop(StreamyFinalLLM(), ToolRegistry(), ctx,
                       StopController(5, 10, 3)).run("task", on_text=deltas.append)

    assert deltas == ["结", "论"]
    assert result.final_text == "结论"
    assert ctx.get_messages()[-1]["role"] == "assistant"
    assert ctx.get_messages()[-1]["content"] == "结论"


def test_on_step_start_fires_with_tool_name_and_args():
    # on_step_start 在工具开始执行前回调，带工具名与参数，
    # 供 CLI 在工具执行期间显示 spinner 等加载状态。
    tool = RecordingTool()
    registry = ToolRegistry()
    registry.register(tool)

    llm = FakeLLM([
        ModelResponse(tool_calls=[ToolCall(id="1", name="record", arguments={"a": 1})]),
        ModelResponse(text="完成"),
    ])
    starts: list = []
    result = make_loop(llm, registry).run(
        "task", on_step_start=lambda n, a: starts.append((n, a))
    )

    assert len(starts) == 1
    assert starts[0] == ("record", {"a": 1})
    assert tool.calls == [{"a": 1}]  # 工具确实被执行
    assert len(result.steps) == 1
