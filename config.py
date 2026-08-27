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

    workspace: str = "./demo"

    # Agent Loop 停止条件（见 agent/stop.py）
    max_steps: int = 20
    max_runtime: float = 300.0
    max_consecutive_errors: int = 3

    # 工具相关限制
    max_tool_output: int = 8192
    command_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量（或 .env 文件）构建配置。"""
        if _load_dotenv is not None:
            # 项目根目录下的 .env；默认不覆盖 shell 中已导出的变量。
            _load_dotenv(Path(__file__).resolve().parent / ".env")

        # 优先 OPENAI_API_KEY，兼容 DEEPSEEK_API_KEY 等常见命名。
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            workspace=os.environ.get("CODING_AGENT_WORKSPACE", "./demo"),
            max_steps=int(os.environ.get("CODING_AGENT_MAX_STEPS", "20")),
            max_runtime=float(os.environ.get("CODING_AGENT_MAX_RUNTIME", "300")),
            max_consecutive_errors=int(os.environ.get("CODING_AGENT_MAX_ERRORS", "3")),
            max_tool_output=int(os.environ.get("CODING_AGENT_MAX_TOOL_OUTPUT", "8192")),
            command_timeout=float(os.environ.get("CODING_AGENT_COMMAND_TIMEOUT", "30")),
        )

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "未找到 API Key。请复制 .env.example 为 .env 并填入 key，"
                "或设置环境变量，例如：\n"
                "  export OPENAI_API_KEY=...   # 或 DEEPSEEK_API_KEY=...\n"
                "凭据只走 .env / 环境变量，不进入仓库。"
            )
