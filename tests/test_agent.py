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


# ── Phase 1+2：上下文压缩与会话隔离集成 ──────────────────────────────────


def test_maybe_compact_called_in_loop_before_llm():
    """maybe_compact 应在第一次 chat 之前执行：第一次 chat 时 messages 已被压缩。"""
    snapshots = []

    class SnapshotLLM:
        def chat(self, messages, tools=None):
            snapshots.append([m.get("role") for m in messages])
            return ModelResponse(text="final")

    ctx = ContextManager("sys", compact_threshold=1, keep_recent=1,
                         summarizer=lambda m: "SUMMARY")
    ctx.add_user("t1")
    ctx.add_assistant(ModelResponse(text="a1"))
    ctx.add_user("t2")
    ctx.add_assistant(ModelResponse(text="a2"))
    ctx.last_prompt_tokens = 100  # 模拟上次调用真实 prompt_tokens 超阈值
    loop = AgentLoop(SnapshotLLM(), ToolRegistry(), ctx, StopController(5, 10, 3))
    loop.run("task3")

    # 第一次 chat 时 messages 应已被压缩为：system + summary_system + user(task3)
    assert len(snapshots) == 1
    roles = snapshots[0]
    assert roles == ["system", "system", "user"]
    assert len(roles) == 3  # 压缩后只剩 3 条，证明 compact 在 chat 之前发生


def test_loop_compact_does_not_count_toward_runtime():
    """compact 在 stop.start() 之前，summarizer 耗时不计入 max_runtime。"""
    import time

    def slow_summarizer(m):
        time.sleep(0.5)  # 模拟 summarizer 调 LLM 耗时
        return "总结"

    ctx = ContextManager("sys", compact_threshold=1, keep_recent=1,
                         summarizer=slow_summarizer)
    ctx.add_user("t1")
    ctx.add_assistant(ModelResponse(text="a1"))
    ctx.add_user("t2")
    ctx.add_assistant(ModelResponse(text="a2"))
    ctx.last_prompt_tokens = 100  # 模拟上次调用真实 prompt_tokens 超阈值，触发 compact
    # max_runtime=0.2 < summarizer 耗时 0.5；若 compact 在 start 之后会立即超时
    loop = AgentLoop(FakeLLM([ModelResponse(text="done")]), ToolRegistry(), ctx,
                     StopController(5, 0.2, 3))
    result = loop.run("task3")
    # 应成功完成，而非因 runtime 超时
    assert result.final_text == "done"
    assert result.stop_reason is None


def test_persisted_session_survives_new_context(tmp_path):
    """跑一次 loop 后，用新 ContextManager 加载同一 session_id，历史应恢复。"""
    from agent.session import SessionStore

    store = SessionStore(tmp_path, "/ws")
    ctx1 = ContextManager("sys", store=store, session_id="s1",
                          compact_threshold=1_000_000, summarizer=lambda m: "S")
    loop = AgentLoop(FakeLLM([ModelResponse(text="done")]), ToolRegistry(), ctx1,
                     StopController(5, 10, 3))
    loop.run("task1")

    # session 文件应有内容
    assert len(store.load("s1")) > 0

    # 新 context 加载同一 session
    ctx2 = ContextManager("sys", store=store, session_id="s1")
    msgs = ctx2.get_messages()
    # system + user(task1) + assistant(done) = 3
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "task1"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "done"


def test_agent_new_session_creates_empty_context(tmp_path):
    """CodingAgent.new_session 后 context 只剩 system 消息。"""
    from config import Config
    from agent.agent import CodingAgent

    config = Config(api_key="fake-key", workspace=str(tmp_path),
                   session_root=str(tmp_path / "sessions"),
                   checkpoint_root=str(tmp_path / "ckpt"))
    agent = CodingAgent(config)
    old_sid = agent.session_id
    new_sid = agent.new_session()
    assert new_sid != old_sid
    msgs = agent.context.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


def test_agent_switch_session_loads_history(tmp_path):
    """CodingAgent.switch_session 到已有历史的 session_id，应加载历史消息。"""
    from config import Config
    from agent.agent import CodingAgent

    config = Config(api_key="fake-key", workspace=str(tmp_path),
                   session_root=str(tmp_path / "sessions"),
                   checkpoint_root=str(tmp_path / "ckpt"))
    agent = CodingAgent(config)
    # 预填一个 session 的历史
    agent.store.append("predefined", {"role": "user", "content": "old task"})
    agent.store.append("predefined", {"role": "assistant", "content": "old answer"})

    agent.switch_session("predefined")
    assert agent.session_id == "predefined"
    msgs = agent.context.get_messages()
    # system + user + assistant
    assert len(msgs) == 3
    assert msgs[1]["content"] == "old task"
    assert msgs[2]["content"] == "old answer"
    # loop 的 context 引用也必须同步更新
    assert agent.loop.context is agent.context


def test_fake_llm_as_summarizer():
    """FakeLLM 返回固定文本作 summarizer，验证 summary 出现在 messages[1]。"""
    summarizer_llm = FakeLLM([ModelResponse(text="这是历史总结")])

    def summarizer(old_messages):
        resp = summarizer_llm.chat([])
        return resp.text or ""

    ctx = ContextManager("sys", compact_threshold=1, keep_recent=1,
                         summarizer=summarizer)
    ctx.add_user("t1")
    ctx.add_assistant(ModelResponse(text="a1"))
    ctx.add_user("t2")
    ctx.add_assistant(ModelResponse(text="a2"))
    ctx.last_prompt_tokens = 100  # 模拟上次调用真实 prompt_tokens 超阈值
    ctx.maybe_compact()
    msgs = ctx.get_messages()
    assert "这是历史总结" in msgs[1]["content"]
    assert summarizer_llm.calls == 1  # summarizer LLM 被调用一次


# ── 真实用量统计（Claude Code 式 token 显示）──────────────────────────────


class UsageLLM:
    """按脚本顺序返回 (响应, (prompt, completion))；每次 chat 先设置 last_usage
    再返回响应，模拟真实 API 在响应中携带 usage 字段的行为。
    脚本耗尽后返回最终文本且 last_usage 置 None（模拟无 usage 的调用）。"""

    def __init__(self, script):
        self.script = list(script)
        self.last_usage = None

    def chat(self, messages, tools=None):
        if not self.script:
            self.last_usage = None
            return ModelResponse(text="done")
        response, (p, c) = self.script.pop(0)
        self.last_usage = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
        return response


def _tool_call(i: int) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(id=str(i), name="record", arguments={})])


def test_runresult_usage_accumulates_real_tokens():
    """一次 run 内多次 LLM 调用的 usage 应累计，且均为真实值。"""
    registry = ToolRegistry()
    registry.register(RecordingTool())
    llm = UsageLLM([
        (_tool_call(1), (100, 10)),
        (_tool_call(2), (150, 20)),
        (ModelResponse(text="done"), (200, 30)),
    ])
    loop = make_loop(llm, registry)

    result = loop.run("task")

    assert result.succeeded
    assert result.usage["calls"] == 3
    assert result.usage["prompt_tokens"] == 450
    assert result.usage["completion_tokens"] == 60
    assert result.usage["total_tokens"] == 510


def test_context_reset_keeps_system_and_clears_file(tmp_path):
    """ContextManager.reset：内存只留 system、last_prompt_tokens 归零、会话文件清空。"""
    from agent.session import SessionStore

    store = SessionStore(tmp_path, "/ws")
    ctx = ContextManager("sys", store=store, session_id="s1",
                         compact_threshold=1_000_000, summarizer=lambda m: "S")
    ctx.add_user("t1")
    ctx.add_assistant(ModelResponse(text="a1"))
    ctx.last_prompt_tokens = 930
    assert len(store.load("s1")) == 2  # 文件里有历史

    ctx.reset()

    assert len(ctx.messages) == 1
    assert ctx.messages[0]["role"] == "system"
    assert ctx.last_prompt_tokens is None
    assert store.load("s1") == []  # 文件同步清空
    # 清空后继续对话：消息正常追加、文件重建
    ctx.add_user("t2")
    assert [m["content"] for m in store.load("s1")] == ["t2"]


def test_agent_clear_context_keeps_session_id(tmp_path):
    """CodingAgent.clear_context：原地清空，会话 id 不变，历史全部移除。"""
    from config import Config
    from agent.agent import CodingAgent

    config = Config(api_key="fake-key", workspace=str(tmp_path),
                   session_root=str(tmp_path / "sessions"),
                   checkpoint_root=str(tmp_path / "ckpt"))
    agent = CodingAgent(config)
    agent.store.append(agent.session_id, {"role": "user", "content": "old"})
    agent.context.add_assistant(ModelResponse(text="a"))
    agent.context.last_prompt_tokens = 500

    sid = agent.session_id
    agent.clear_context()

    assert agent.session_id == sid  # id 不变（与 /new 的区别）
    assert len(agent.context.get_messages()) == 1
    assert agent.context.get_messages()[0]["role"] == "system"
    assert agent.context.last_prompt_tokens is None
    assert agent.store.load(sid) == []


def test_usage_absent_llm_yields_zero_usage():
    """无 last_usage 的 chat 实现（如测试 mock）不报错，usage 保持全零。"""
    llm = FakeLLM([ModelResponse(text="done")])
    result = make_loop(llm, ToolRegistry()).run("task")
    assert result.succeeded
    assert result.usage["calls"] == 0
    assert result.usage["prompt_tokens"] == 0


# ── Phase 4：代码快照与对话回退的 Agent 装配 ──────────────────────────────


def test_agent_checkpoint_wiring(tmp_path):
    """CodingAgent 装配 CheckpointStore：/clear 清账本、切换会话切账本。"""
    from config import Config
    from agent.agent import CodingAgent

    config = Config(api_key="fake-key", workspace=str(tmp_path),
                   session_root=str(tmp_path / "sessions"),
                   checkpoint_root=str(tmp_path / "ckpt"))
    agent = CodingAgent(config)
    assert agent.checkpoints.enabled
    assert agent.checkpoints.session_id == agent.session_id

    # /clear：对话清空，账本同步清空（workspace 文件不动）
    agent.context.add_user("t1")
    agent.checkpoints._ledger.append(
        {"turn": 0, "commit": "abc", "ts": 1.0, "preview": "t1"})
    agent.clear_context()
    assert agent.checkpoints.entries() == []

    # /new、/resume 走 switch_session：账本切到新会话
    agent.switch_session("other")
    assert agent.checkpoints.session_id == "other"
    assert agent.checkpoints.entries() == []
