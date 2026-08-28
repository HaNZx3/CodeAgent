"""CodingAgent 组装层：把 llm / tools / context / stop / loop 装配成一个可用的 Agent。"""

from __future__ import annotations

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

        # System Prompt + Skill 指引（Skill 是可选扩展）
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        system_prompt = SYSTEM_PROMPT + load_skills(skills_dir)

        self.context = ContextManager(system_prompt, config.max_tool_output)
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

    def run(self, task: str, on_step=None, on_text=None) -> RunResult:
        return self.loop.run(task, on_step=on_step, on_text=on_text)
