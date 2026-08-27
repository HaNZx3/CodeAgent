"""ContextManager 单元测试。"""

from llm.client import ModelResponse, ToolCall
from tools.base import ToolResult
from agent.context import ContextManager


def test_system_prompt_first():
    cm = ContextManager("sys")
    assert cm.get_messages()[0] == {"role": "system", "content": "sys"}


def test_add_user():
    cm = ContextManager("sys")
    cm.add_user("hi")
    assert cm.get_messages()[-1] == {"role": "user", "content": "hi"}


def test_assistant_tool_call_roundtrip():
    cm = ContextManager("sys")
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
    cm.add_assistant(ModelResponse(tool_calls=[call]))
    cm.add_tool_result(call, ToolResult.ok("data"))

    msgs = cm.get_messages()
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "read_file"

    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "c1"
    assert msgs[2]["content"] == "data"


def test_tool_result_error_wrapped():
    cm = ContextManager("sys")
    call = ToolCall(id="c1", name="read_file", arguments={})
    cm.add_tool_result(call, ToolResult.fail("File not found"))
    content = cm.get_messages()[-1]["content"]
    assert "Tool execution failed" in content
    assert "File not found" in content


def test_tool_output_truncated():
    cm = ContextManager("sys", max_tool_output=100)
    call = ToolCall(id="c1", name="x", arguments={})
    cm.add_tool_result(call, ToolResult.ok("a" * 1000))
    content = cm.get_messages()[-1]["content"]
    assert len(content) < 1000
    assert "truncated" in content
