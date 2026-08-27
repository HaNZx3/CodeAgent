"""Workspace 安全边界。

为什么需要它？
    我们不希望模型通过 read_file / write_file 读写任意系统文件。
    所有文件操作必须先经过本模块：把相对路径解析成绝对路径，再校验
    结果是否仍在 workspace 目录内，越界直接拒绝。
    如果没有它，模型一句 read_file("../../secret.txt") 就能读走用户文件。

Shell 工具也会复用 Workspace：把命令的 cwd 固定为 workspace。
"""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """路径越界或非法时抛出的异常。"""


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | Path) -> Path:
        """把用户/模型给出的路径解析为 workspace 内的绝对路径。

        越界或非法时抛出 WorkspaceError。
        """
        # 绝对路径会覆盖 self.root，因此这里能正确处理模型给出绝对路径的情况，
        # 随后再用 relative_to 校验是否仍落在 workspace 内。
        candidate = (self.root / Path(path)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(
                f"路径越界，禁止访问 workspace 之外的位置: {path}"
            )
        return candidate
