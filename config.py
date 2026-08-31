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
    from dotenv import dotenv_values as _dotenv_values
except ImportError:  # 未安装 python-dotenv 时，回退到只使用真实环境变量
    _dotenv_values = None

def coding_agent_home() -> Path:
    """Agent 用户级数据根目录：~/.coding-agent。

    会话、代码快照、USER.md、全局 .env 都落在这里；任意目录启动都能读到。
    每次调用都重新解析 Path.home()，便于测试用 monkeypatch 替换主目录。
    """
    return Path.home() / ".coding-agent"


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

    # 上下文自动压缩
    compact_threshold: int = 80_000   # 上次 API 返回的真实 prompt_tokens 超此值时自动压缩历史
    keep_recent: int = 6             # 压缩时保留最近 N 轮（一轮=user+后续 assistant/tool）
    # 模型上下文窗口大小（仅用于 /status 与回复后指示条的占用百分比显示，
    # 不参与压缩判断——压缩只看 compact_threshold 与真实 prompt_tokens）
    context_window: int = 128_000

    # 会话持久化
    session_root: str = ""           # 空=~/.coding-agent/sessions（from_env 自动填充）

    # 代码快照与对话回退
    checkpoints: bool = True         # 影子 git 快照，供 /back 回退代码；git 缺失时自动禁用
    checkpoint_root: str = ""        # 空=~/.coding-agent/checkpoints（from_env 自动填充）

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量（或 .env 文件）构建配置。

        凭据加载顺序（shell 环境变量始终最高优先级；.env 中项目根覆盖 cwd
        覆盖全局 ~/.coding-agent/.env）：
          1) 用户主目录 ~/.coding-agent/.env  —— 全局凭据，任意目录启动都能读到
          2) 当前工作目录 ./                .env —— 项目本地可覆盖
          3) 项目根目录（本文件所在目录）.env —— 兼容从项目根启动
        shell 中已导出的真实环境变量永不被 .env 覆盖。
        """
        if _dotenv_values is not None:
            home_env = coding_agent_home() / ".env"
            cwd_env = Path.cwd() / ".env"
            project_env = Path(__file__).resolve().parent / ".env"
            # 合并 .env：按 低→高 顺序读入，后者覆盖前者；再 setdefault 写回
            # os.environ——shell 变量已在 os.environ 中，永不被覆盖。
            merged: dict[str, str | None] = {}
            for p in (home_env, cwd_env, project_env):
                if p.exists():
                    merged.update(_dotenv_values(p))
            for k, v in merged.items():
                if v is not None:
                    os.environ.setdefault(k, v)

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
            context_window=int(os.environ.get("CODING_AGENT_CONTEXT_WINDOW", "128000")),
            session_root=os.environ.get("CODING_AGENT_SESSION_ROOT")
            or str(coding_agent_home() / "sessions"),
            checkpoint_root=os.environ.get("CODING_AGENT_CHECKPOINT_ROOT")
            or str(coding_agent_home() / "checkpoints"),
            checkpoints=os.environ.get("CODING_AGENT_CHECKPOINTS", "1")
            .strip().lower() not in ("0", "false", "no", "off"),
        )

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "未找到 API Key。请复制 .env.example 为 .env 并填入 key，"
                "或设置环境变量，例如：\n"
                "  export OPENAI_API_KEY=...   # 或 DEEPSEEK_API_KEY=...\n"
                "凭据只走 .env / 环境变量，不进入仓库。"
            )
