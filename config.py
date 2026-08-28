"""全局配置。

为什么需要它？
    把「从哪读取 API Key / 模型名 / 工作目录 / 各项阈值」集中到一处，
    main.py、agent、tools 都只依赖一个 Config 对象，而不是在代码各处
    散落 os.environ 和魔法数字。
    如果没有它，改一个默认值或阈值就要满仓库搜索，且很容易写错。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # 未安装 python-dotenv 时，回退到只使用真实环境变量
    _load_dotenv = None


@dataclass
class Config:
    """Agent 运行时配置。

    API Key 只能通过环境变量提供，绝不能硬编码进代码或仓库。
    也支持从项目根目录的 .env 文件加载（.env 已被 .gitignore 忽略，
    不会进入仓库），但 shell 中已导出的真实环境变量始终优先。
    """

    api_key: str = ""
    base_url: str | None = None
    model: str = "gpt-4o-mini"

    workspace: str = ""  # 空=用当前工作目录（Claude Code 式行为）

    # Agent Loop 停止条件（见 agent/stop.py）
    max_steps: int = 20
    max_runtime: float = 300.0
    max_consecutive_errors: int = 3

    # 工具相关限制
    max_tool_output: int = 8192
    command_timeout: float = 30.0

    # 上下文自动压缩（Phase 1）
    compact_threshold: int = 80_000   # 上次 API 返回的真实 prompt_tokens 超此值时自动压缩历史
    keep_recent: int = 6             # 压缩时保留最近 N 轮（一轮=user+后续 assistant/tool）

    # 会话持久化（Phase 2）
    session_root: str = ""           # 空=~/.coding-agent/sessions

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量（或 .env 文件）构建配置。

        .env 加载顺序（后者覆盖前者，shell 环境变量始终最高优先级）：
          1) 用户主目录 ~/.coding-agent/.env  —— 全局凭据，任意目录启动都能读到
          2) 当前工作目录 ./                .env —— 项目本地可覆盖
          3) 项目根目录（本文件所在目录）.env —— 兼容旧用法
        这样 Agent 可像 Claude Code 一样在任意文件夹启动，凭据仍来自全局位置。
        """
        if _load_dotenv is not None:
            home_env = Path.home() / ".coding-agent" / ".env"
            if home_env.exists():
                _load_dotenv(home_env)
            _load_dotenv(Path.cwd() / ".env", override=True)
            _load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

        # 优先 OPENAI_API_KEY，兼容 DEEPSEEK_API_KEY 等常见命名。
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            workspace=os.environ.get("CODING_AGENT_WORKSPACE", ""),
            max_steps=int(os.environ.get("CODING_AGENT_MAX_STEPS", "20")),
            max_runtime=float(os.environ.get("CODING_AGENT_MAX_RUNTIME", "300")),
            max_consecutive_errors=int(os.environ.get("CODING_AGENT_MAX_ERRORS", "3")),
            max_tool_output=int(os.environ.get("CODING_AGENT_MAX_TOOL_OUTPUT", "8192")),
            command_timeout=float(os.environ.get("CODING_AGENT_COMMAND_TIMEOUT", "30")),
            compact_threshold=int(os.environ.get("CODING_AGENT_COMPACT_THRESHOLD", "80000")),
            keep_recent=int(os.environ.get("CODING_AGENT_KEEP_RECENT", "6")),
            session_root=os.environ.get("CODING_AGENT_SESSION_ROOT", ""),
        )

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "未找到 API Key。请复制 .env.example 为 .env 并填入 key，"
                "或设置环境变量，例如：\n"
                "  export OPENAI_API_KEY=...   # 或 DEEPSEEK_API_KEY=...\n"
                "凭据只走 .env / 环境变量，不进入仓库。"
            )
