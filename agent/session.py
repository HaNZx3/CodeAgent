"""会话持久化存储：每个会话一个 jsonl 文件，按 workspace 哈希分目录。

为什么需要它？
    仿 Claude Code 的 ~/.claude/projects/<path-hash>/<session_id>.jsonl：
    - 同一 workspace 的所有会话落在同一子目录，便于 /sessions 列出
    - 不同 workspace 的会话物理隔离，互不干扰
    - jsonl 追加写入对崩溃友好；rewrite 用 os.replace 原子覆盖
    - 进程退出后可 /resume <id> 恢复历史上下文
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def workspace_slug(workspace: str) -> str:
    """workspace -> 「可读片段-sha1前12位」目录名。

    SessionStore（会话文件）与 CheckpointStore（代码快照）共用同一规则，
    保证同一 workspace 的两类数据落在同一子目录、可按目录互相发现。
    """
    h = hashlib.sha1(workspace.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", workspace).strip("-")[:30] or "default"
    return f"{slug}-{h}"


class SessionStore:
    """以 jsonl 文件持久化会话消息。

    构造时绑定 workspace（agent.workspace 固定），后续所有操作都基于
    该 workspace 的子目录，不需要每次传 workspace。
    """

    def __init__(self, root: str | Path, workspace: str):
        self.root = Path(root)
        self.workspace = workspace
        self._dir = self._compute_dir()

    def _compute_dir(self) -> Path:
        return self.root / workspace_slug(self.workspace)

    def path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.jsonl"

    def load(self, session_id: str) -> list[dict]:
        """读取整个会话历史。文件不存在返回空列表（即新会话）。"""
        p = self.path(session_id)
        if not p.exists():
            return []
        out: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # 跳过损坏行，不让整个会话失效
                    continue
        return out

    def append(self, session_id: str, msg: dict) -> None:
        """追加单条消息。每行一个 JSON 对象，崩溃时只丢最后一条。"""
        p = self.path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def rewrite(self, session_id: str, messages: list[dict]) -> None:
        """整文件重写（压缩后用）。

        原子写：先写 .tmp 临时文件，再 os.replace 覆盖。POSIX 上 os.replace
        是原子的，Windows 上也保证目标文件不会被半写破坏。
        """
        p = self.path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        os.replace(tmp, p)

    def clear(self, session_id: str) -> None:
        """删除会话文件。不存在则静默。"""
        p = self.path(session_id)
        if p.exists():
            p.unlink()

    def clear_all(self) -> int:
        """删除本 workspace 的所有会话文件，返回删除数量。

        目录本身保留（list_sessions 会自动处理目录不存在的情况）。
        rewrite 崩溃残留的 .jsonl.tmp 临时文件一并清理。
        """
        if not self._dir.exists():
            return 0
        n = 0
        for f in self._dir.glob("*.jsonl"):
            f.unlink()
            n += 1
        for f in self._dir.glob("*.jsonl.tmp"):
            f.unlink(missing_ok=True)
        return n

    def list_sessions(self) -> list[tuple[str, str, float]]:
        """列出本 workspace 的所有会话。

        返回 [(session_id, 首条 user 消息预览前 60 字符, mtime), ...]，
        按 mtime 倒序（最近在前）。
        """
        if not self._dir.exists():
            return []
        out: list[tuple[str, str, float]] = []
        for f in self._dir.glob("*.jsonl"):
            sid = f.stem
            preview = ""
            for m in self.load(sid):
                if m.get("role") == "user":
                    preview = (m.get("content", "") or "")[:60]
                    break
            out.append((sid, preview, f.stat().st_mtime))
        out.sort(key=lambda x: x[2], reverse=True)
        return out
