"""CodingAgent 组装层：把 llm / tools / context / stop / loop 装配成一个可用的 Agent。"""

from __future__ import annotations

import uuid
from pathlib import Path

from config import Config
from llm.client import LLMClient
from tools.registry import ToolRegistry
from tools.workspace import Workspace
from tools.file_tools import ListFilesTool, ReadFileTool, WriteFileTool, EditFileTool
from tools.search_tool import SearchTool
from tools.shell_tool import ShellTool

from .context import ContextManager
from .stop import StopController
from .loop import AgentLoop, RunResult
from .skill import load_skills
from .session import SessionStore
from .memory import load_project_memory

# 系统提示词不宜过长，重点是建立稳定的工作流程。
SYSTEM_PROMPT = """你是一个 Coding Agent，可以在当前 workspace 中通过工具完成编程任务。

工作原则：
1. 修改代码之前，先了解相关代码（list_files / search_code / read_file）。
2. 不要读取与任务无关的大量文件。
3. 修改代码后，尽量运行相关测试（run_command）。
4. 根据工具执行结果继续分析，不要假设工具执行成功。
5. 工具失败时，先分析错误原因再尝试恢复，不要直接放弃。
6. 任务完成后，给出简洁的总结。
7. 不得尝试访问 workspace 之外的文件。
"""


class CodingAgent:
    def __init__(self, config: Config):
        config.ensure_api_key()

        self.config = config
        self.llm = LLMClient(config.api_key, config.base_url, config.model)
        self.workspace = Workspace(config.workspace)
        self.registry = self._build_registry()

        # 会话持久化存储：每个 workspace 一个子目录，进程退出后可 /resume。
        session_root = (
            Path(config.session_root)
            if config.session_root
            else (Path.home() / ".coding-agent" / "sessions")
        )
        self.store = SessionStore(session_root, config.workspace)

        # 初始会话：每次启动开一个新 session_id。
        # resume 走 switch_session，由 REPL 命令触发。
        self._session_id = self._new_session_id()
        self.context = self._make_context(self._session_id)
        self.stop = StopController(
            config.max_steps, config.max_runtime, config.max_consecutive_errors
        )
        self.loop = AgentLoop(self.llm, self.registry, self.context, self.stop)

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ListFilesTool(self.workspace))
        registry.register(ReadFileTool(self.workspace))
        registry.register(WriteFileTool(self.workspace))
        registry.register(EditFileTool(self.workspace))
        registry.register(SearchTool(self.workspace))
        registry.register(ShellTool(self.workspace, self.config.command_timeout))
        return registry

    @staticmethod
    def _new_session_id() -> str:
        """生成 12 位十六进制会话 ID。"""
        return uuid.uuid4().hex[:12]

    @property
    def session_id(self) -> str:
        return self._session_id

    def _build_system_prompt(self) -> str:
        """System Prompt 顺序：基础约束 + workspace 路径 + 项目记忆 + Skill 指引。

        注入真实 workspace 路径，避免模型臆造为 /workspace。
        项目级约束（AGENT.md）应在通用 skill 之前被模型读到。
        """
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        ws = self.config.workspace or str(Path.cwd())
        return (
            SYSTEM_PROMPT
            + f"\n当前 workspace 路径：{ws}\n"
            + load_project_memory(self.config.workspace)
            + load_skills(skills_dir)
        )

    def _make_context(self, session_id: str) -> ContextManager:
        """工厂：用当前 system prompt + store 构造 ContextManager。

        summarizer 复用同一 LLMClient（tools=None 走纯文本），
        捕获局部 llm 而非 self，规避对象引用环。
        """
        llm = self.llm

        def _summarizer(old_messages: list[dict]) -> str:
            resp = llm.chat(
                [
                    {
                        "role": "system",
                        "content": "简明总结之前的对话，保留关键决策、文件路径、已做修改。",
                    },
                    *old_messages,
                ],
                tools=None,
            )
            return resp.text or ""

        return ContextManager(
            self._build_system_prompt(),
            self.config.max_tool_output,
            compact_threshold=self.config.compact_threshold,
            keep_recent=self.config.keep_recent,
            summarizer=_summarizer,
            store=self.store,
            session_id=session_id,
        )

    def switch_session(self, session_id: str) -> None:
        """切换到已有会话：用当前 system prompt 重建 ContextManager，
        从 store 加载该 session 的历史消息。"""
        self._session_id = session_id
        self.context = self._make_context(session_id)
        # loop 内部持有 context 引用，切换后必须同步更新。
        self.loop.context = self.context

    def new_session(self, name: str | None = None) -> str:
        """开启新会话：生成新 session_id（或用 name 作为 id），切过去。

        旧 session 文件保留，可后续 /resume <id> 回看。
        """
        sid = name if name else self._new_session_id()
        self.switch_session(sid)
        return sid

    def run(self, task: str, on_step=None, on_text=None, on_step_start=None) -> RunResult:
        return self.loop.run(
            task, on_step=on_step, on_text=on_text, on_step_start=on_step_start
        )
