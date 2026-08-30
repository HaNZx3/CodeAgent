"""ContextManager 单元测试。"""

import pytest

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


def test_clear_resets_to_system_prompt_only():
    # /clear 命令依赖此方法：清空历史但保留 system prompt，开启新对话。
    cm = ContextManager("sys")
    cm.add_user("task1")
    cm.add_assistant(ModelResponse(text="answer1"))
    assert len(cm.get_messages()) == 3  # system + user + assistant

    cm.clear()

    msgs = cm.get_messages()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": "sys"}


def test_clear_can_replace_system_prompt():
    cm = ContextManager("old")
    cm.add_user("x")
    cm.clear("new system prompt")
    msgs = cm.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "new system prompt"


# ── Phase 1：自动压缩 ────────────────────────────────────────────────────


def test_maybe_compact_noop_below_threshold():
    cm = ContextManager("sys", compact_threshold=1_000_000, summarizer=lambda m: "S")
    cm.add_user("hi")
    cm.last_prompt_tokens = 500  # 真实值 < 1_000_000，未超阈值
    before = list(cm.get_messages())
    cm.maybe_compact()
    assert cm.get_messages() == before  # 未超阈值，messages 不变


def test_maybe_compact_no_summarizer_returns_early():
    cm = ContextManager("sys", compact_threshold=1, summarizer=None)
    cm.add_user("hi")
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值，但因无 summarizer 早返回
    before = list(cm.get_messages())
    cm.maybe_compact()
    assert cm.get_messages() == before  # 未注入 summarizer，优雅退化


def test_maybe_compact_invokes_summarizer():
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "FAKE_SUMMARY")
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值
    cm.maybe_compact()
    msgs = cm.get_messages()
    assert msgs[1]["role"] == "system"
    assert "FAKE_SUMMARY" in msgs[1]["content"]


def test_maybe_compact_preserves_recent_turns():
    cm = ContextManager("sys", compact_threshold=1, keep_recent=2,
                        summarizer=lambda m: "S")
    for i in range(1, 5):
        cm.add_user(f"t{i}")
        cm.add_assistant(ModelResponse(text=f"a{i}"))
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值
    cm.maybe_compact()
    msgs = cm.get_messages()
    recent = msgs[2:]  # 跳过 system + summary
    # recent 段以 user 开头
    assert recent[0]["role"] == "user"
    # recent 含最后 keep_recent=2 个 user
    user_contents = [m["content"] for m in recent if m["role"] == "user"]
    assert user_contents == ["t3", "t4"]


def test_maybe_compact_does_not_split_tool_call_pair():
    """关键回归：压缩切分不能切断 assistant(tool_calls)+tool 配对。

    OpenAI 兼容 API 校验 tool 消息必须有紧邻前驱 assistant(tool_calls)，
    否则直接 400。本测试构造含 tool 调用的多轮对话，验证压缩后
    recent 段以 user 开头且内部配对完整。
    """
    cm = ContextManager("sys", compact_threshold=1, keep_recent=2,
                        summarizer=lambda m: "历史摘要")
    # 4 轮对话，recent 2 轮含 tool 调用
    cm.add_user("task1")
    cm.add_assistant(ModelResponse(text="answer1"))
    cm.add_user("task2")
    call1 = ToolCall(id="c1", name="read", arguments={"path": "a"})
    cm.add_assistant(ModelResponse(text="reading", tool_calls=[call1]))
    cm.add_tool_result(call1, ToolResult.ok("data1"))
    cm.add_assistant(ModelResponse(text="done1"))
    cm.add_user("task3")
    cm.add_assistant(ModelResponse(text="answer3"))
    cm.add_user("task4")
    call2 = ToolCall(id="c2", name="read", arguments={"path": "b"})
    cm.add_assistant(ModelResponse(text="reading", tool_calls=[call2]))
    cm.add_tool_result(call2, ToolResult.ok("data2"))
    cm.add_assistant(ModelResponse(text="done4"))
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值

    cm.maybe_compact()

    msgs = cm.get_messages()
    assert msgs[1]["content"].startswith("[Previous conversation summary]")
    recent = msgs[2:]
    # recent 段必须以 user 开头（task3）
    assert recent[0]["role"] == "user"
    assert recent[0]["content"] == "task3"
    # recent 段中每个 tool 消息的前一条必须是带 tool_calls 的 assistant
    for i, m in enumerate(recent):
        if m["role"] == "tool":
            assert i > 0, "recent 段不能以 tool 消息开头"
            prev = recent[i - 1]
            assert prev["role"] == "assistant", "tool 消息前必须是 assistant"
            assert prev.get("tool_calls"), "tool 消息前的 assistant 必须有 tool_calls"
            ids = [tc["id"] for tc in prev["tool_calls"]]
            assert m["tool_call_id"] in ids, "tool_call_id 必须匹配前驱 assistant"


def test_maybe_compact_force_param():
    cm = ContextManager("sys", compact_threshold=1_000_000, keep_recent=1,
                       summarizer=lambda m: "FORCED")
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    cm.maybe_compact(force=True)  # 强制压缩，即使未超阈值
    msgs = cm.get_messages()
    assert "FORCED" in msgs[1]["content"]


def test_maybe_compact_summarizer_failure_skipped():
    def bad_summarizer(m):
        raise RuntimeError("LLM 挂了")
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=bad_summarizer)
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值，进入压缩
    before = list(cm.get_messages())
    cm.maybe_compact()  # 不应抛错
    assert cm.get_messages() == before  # 失败时保留原 messages


def test_maybe_compact_idempotent_on_summary():
    calls = []

    def summarizer(m):
        calls.append(m)
        return "总结"

    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=summarizer)
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    cm.last_prompt_tokens = 100_000  # 真实值 >= 1，超阈值
    cm.maybe_compact()
    assert len(calls) == 1
    # 再次压缩：messages[1] 已是 summary，应跳过
    cm.maybe_compact()
    assert len(calls) == 1  # 没有再次调用


def test_last_prompt_tokens_defaults_none_and_blocks_compact():
    """未调用过 LLM 时 last_prompt_tokens 为 None，maybe_compact 不压缩——不估算。

    压缩判断只用 API 返回的真实 prompt_tokens，None 表示尚无真实数据，
    此时跳过压缩，等首次调用拿到 usage 后再说。
    """
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "SHOULD_NOT_APPEAR")
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    before = list(cm.get_messages())
    cm.maybe_compact()  # last_prompt_tokens 为 None，不压缩
    assert cm.get_messages() == before


def test_last_prompt_tokens_drives_threshold():
    """真实 prompt_tokens 超阈值时触发压缩，未超时不压缩。"""
    # 未超阈值：不压缩
    cm = ContextManager("sys", compact_threshold=1_000, keep_recent=1,
                        summarizer=lambda m: "S")
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))
    cm.last_prompt_tokens = 500  # 真实值 < 1000
    before = list(cm.get_messages())
    cm.maybe_compact()
    assert cm.get_messages() == before  # 未超阈值，不压缩

    # 超阈值：压缩
    cm.last_prompt_tokens = 2_000  # 真实值 > 1000
    cm.maybe_compact()
    msgs = cm.get_messages()
    assert "S" in msgs[1]["content"]


# ── Phase 2：会话持久化 ──────────────────────────────────────────────────


def test_persisted_messages_on_add_user(tmp_path):
    from agent.session import SessionStore
    store = SessionStore(tmp_path, "/ws")
    cm = ContextManager("sys", store=store, session_id="s1")
    cm.add_user("hello")
    loaded = store.load("s1")
    assert len(loaded) == 1
    assert loaded[0] == {"role": "user", "content": "hello"}


def test_persisted_messages_on_add_tool_result(tmp_path):
    from agent.session import SessionStore
    store = SessionStore(tmp_path, "/ws")
    cm = ContextManager("sys", store=store, session_id="s1")
    call = ToolCall(id="c1", name="read", arguments={"path": "a"})
    cm.add_assistant(ModelResponse(tool_calls=[call]))
    cm.add_tool_result(call, ToolResult.ok("data"))
    loaded = store.load("s1")
    # system 不落盘，只有 assistant + tool 两条
    assert len(loaded) == 2
    assert loaded[0]["role"] == "assistant"
    assert loaded[0]["tool_calls"][0]["id"] == "c1"
    assert loaded[1]["role"] == "tool"
    assert loaded[1]["tool_call_id"] == "c1"
    assert loaded[1]["content"] == "data"


# ── Phase 4：对话回退（/back） ────────────────────────────────────────────


def _seed_two_turns(cm):
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    cm.add_user("t2")
    cm.add_assistant(ModelResponse(text="a2"))


def test_user_turns_lists_user_message_indices():
    cm = ContextManager("sys")
    _seed_two_turns(cm)
    assert cm.user_turns() == [1, 3]


def test_user_turns_excludes_summary_message():
    """历史摘要（role=system）不是回退点：其替换的轮次已物理删除。"""
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "S")
    _seed_two_turns(cm)
    cm.last_prompt_tokens = 100
    cm.maybe_compact()
    # messages: [system, summary, t2, a2] -> 只剩 t2 可回退
    assert cm.user_turns() == [2]


def test_rewind_to_later_turn_keeps_earlier(tmp_path):
    from agent.session import SessionStore
    store = SessionStore(tmp_path, "/ws")
    cm = ContextManager("sys", store=store, session_id="s1")
    _seed_two_turns(cm)

    removed = cm.rewind_to(3)  # 回退到 t2 之前

    assert removed == 2
    assert [m["content"] for m in cm.messages[1:]] == ["t1", "a1"]
    assert [m["content"] for m in store.load("s1")] == ["t1", "a1"]
    # 回退后继续对话：正常追加，文件一致
    cm.add_user("t3")
    assert store.load("s1")[-1]["content"] == "t3"


def test_rewind_to_first_turn_clears_all_and_persists(tmp_path):
    from agent.session import SessionStore
    store = SessionStore(tmp_path, "/ws")
    cm = ContextManager("sys", store=store, session_id="s1")
    _seed_two_turns(cm)

    assert cm.rewind_to(1) == 4
    assert cm.messages == [{"role": "system", "content": "sys"}]
    assert store.load("s1") == []


def test_rewind_invalid_index_raises():
    """回退点必须是 user_turns() 中的索引：防止留下孤儿 tool_call。"""
    cm = ContextManager("sys")
    cm.add_user("t1")
    cm.add_assistant(ModelResponse(text="a1"))
    for bad in (0, 2, 99, -1):
        with pytest.raises(ValueError):
            cm.rewind_to(bad)


def test_rewind_resets_last_prompt_tokens():
    """回退后旧的真实用量不再代表当前上下文，必须置空防误触发压缩。"""
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "SHOULD_NOT_APPEAR")
    _seed_two_turns(cm)
    cm.last_prompt_tokens = 100_000
    cm.rewind_to(3)
    assert cm.last_prompt_tokens is None
    cm.maybe_compact()  # None -> 不压缩
    assert len(cm.messages) == 3  # [sys, t1, a1]，未被压缩


def test_maybe_compact_notifies_pruned_user_turns():
    """压缩丢弃旧轮次时回调通知（供快照账本同步修剪）。"""
    pruned = []
    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "S", on_user_turns_pruned=pruned.append)
    _seed_two_turns(cm)
    cm.last_prompt_tokens = 100
    cm.maybe_compact()
    assert pruned == [1]  # t1 所在的 1 个用户轮次被摘要


def test_prune_callback_failure_does_not_break_compact():
    def boom(n):
        raise RuntimeError("账本挂了")

    cm = ContextManager("sys", compact_threshold=1, keep_recent=1,
                        summarizer=lambda m: "S", on_user_turns_pruned=boom)
    _seed_two_turns(cm)
    cm.last_prompt_tokens = 100
    cm.maybe_compact()  # 不应抛错
    assert "S" in cm.messages[1]["content"]  # 压缩本身成功
