"""代码快照（checkpoint）：用影子 git 仓库记录每轮用户消息发出前的 workspace 状态。

为什么需要它？
    对话回退（/back）若只截断消息，agent 已改的代码不会还原，模型下一轮
    会基于错误的现实继续工作。把代码状态与对话轮次一一对应起来：
    第 k 个快照 = 第 k 条用户消息发出前的 workspace 全量状态。

设计要点（对齐 Claude Code 的 checkpoint 思路）：
    - 影子仓库：git 目录建在 {root}/{workspace-slug}/ 下，通过
      --git-dir / --work-tree 指向 workspace。绝不动用户自己的 .git，
      workspace 是否 git init 都不影响；凭据（.env*）与缓存目录被
      内置 exclude 永久排除，不进任何快照。
    - 快照时机：每轮用户消息提交前由 CodingAgent.run 调用 snapshot()。
    - 账本（ledger）：{root}/{slug}/{session_id}.ckpt.json，记录
      [{turn, commit, ts, preview}]。不变量：第 k 项 = 第 k 条用户消息
      发出前的快照。压缩/清空/回退必须同步修剪账本以维持 1:1，
      否则「轮次 <-> 快照」错位，代码还原会打到错误的状态。
    - 还原（全量）= git reset --hard <hash>：能正确删除快照之后新增的
      文件（checkout <hash> -- . 做不到）。还原前先打一个「安全快照」
      commit，保证任何还原本身可撤销——还原前的状态永远在影子仓库里找得回。
    - 精确回退（/back 默认）：全量还原会把其它会话在目标快照之后的改动
      一并抹掉。restore_precise 采用反向 apply 一系列 (起点快照 → 链上
      下一个快照) 的区间 diff：快照在每轮开始时打，这段 diff 恰是以该
      快照为起点的那轮的私有改动；只选本会话条目对应的区间，其它会话的
      工作天然保留。restore 前创建的安全快照恰好充当「最后一轮之后缺失
      的下一个快照」，让最后一轮的改动也有据可撤。同文件交叉 apply 失败
      时回滚到安全快照，由调用方决定是否降级全量回退。
    - 跨会话冲突：同一 workspace 的其它会话账本里若存在晚于目标快照的
      条目，plan_restore 列出其改动文件，供 /back 确认时提示——精确回退
      会保留这些改动（同文件交叉除外），全量回退则一并撤销。
    - 失败降级：git 不可用或命令失败时 enabled=False 并停止快照，
      对话回退不受影响。快照只是增强，绝不阻塞主循环。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .session import workspace_slug

# 影子仓库统一排除：凭据绝不进快照（呼应「凭据不入库」的硬约束）；
# 缓存/依赖目录体量大且无还原价值。workspace 内 .gitignore 也会被 git 自动尊重。
_EXCLUDES = """\
# coding-agent checkpoints 内置排除
.env*
__pycache__/
*.pyc
.venv/
venv/
node_modules/
.git/
"""


@dataclass
class RestorePlan:
    """还原预览：只读计算，供确认界面展示。"""

    target: str  # 目标快照 commit hash
    diffstat: str  # 当前 -> 目标 将被还原的改动概览（--stat）
    conflicts: list[tuple[str, list[str]]]  # [(其它会话 id, 其在快照后改过的文件)]


@dataclass
class RestoreResult:
    """还原结果。safety = 还原前自动安全快照的 hash（误还原时找得回）。"""

    target: str
    safety: str


@dataclass
class PreciseResult:
    """精确回退结果（第 3 层）。

    ok=True：本会话快照后的改动已反向 apply，files 为被撤销的文件；
    ok=False：与其它会话同文件交叉，apply 失败——工作区已回滚到还原前
    状态，files 为冲突文件，调用方可选择降级为全量回退（restore）。
    """

    ok: bool
    safety: str
    files: list[str]


class CheckpointStore:
    """影子 git 仓库 + 每会话账本，管理「对话轮次 <-> 代码状态」的映射。"""

    def __init__(
        self,
        root: str | Path,
        workspace: str,
        session_id: str | None = None,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.workspace = workspace
        # git 缺失时立即降级；之后任何 git 失败也会置 False（一次性降级）。
        self.enabled = enabled and shutil.which("git") is not None
        self._repo_dir = self.root / workspace_slug(workspace)
        self.session_id: str | None = None
        self._ledger: list[dict] = []
        if session_id:
            self.set_session(session_id)

    # ── 账本：[{turn, commit, ts, preview}]，与会话用户轮次 1:1 ─────────────

    @property
    def _ledger_path(self) -> Path:
        return self._repo_dir / f"{self.session_id}.ckpt.json"

    def set_session(self, session_id: str) -> None:
        """切换会话（/new、/resume 时）：载入该会话的账本，无则视为空。"""
        self.session_id = session_id
        self._ledger = self._load_ledger()

    def _load_ledger(self) -> list[dict]:
        p = self._ledger_path
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []  # 账本损坏视作无快照：对话回退仍可用，代码还原跳过

    def _save_ledger(self) -> None:
        """账本原子写：tmp + os.replace（与会话文件同一约定，防半写损坏）。"""
        p = self._ledger_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._ledger, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(tmp, p)

    def entries(self) -> list[dict]:
        return list(self._ledger)

    def prune(self, dropped_user_turns: int) -> None:
        """压缩摘要掉前 N 个用户轮次后同步修剪账本，维持与轮次的 1:1 对齐。"""
        if dropped_user_turns <= 0 or not self._ledger:
            return
        self._ledger = self._ledger[dropped_user_turns:]
        self._save_ledger()

    def truncate(self, keep: int) -> None:
        """回退到第 keep+1 轮之前时：账本只保留前 keep 项。"""
        if keep >= len(self._ledger):
            return
        self._ledger = self._ledger[:keep]
        self._save_ledger()

    def clear_ledger(self) -> None:
        """/clear 清空对话时同步清账本（workspace 文件不动）。

        未启用且从未有过账本时不写文件；文件已存在（此前启用过）则
        必须写空——否则账本与已清空的对话错位。
        """
        self._ledger = []
        if self.enabled or self._ledger_path.exists():
            self._save_ledger()

    def drop_session(self, session_id: str) -> None:
        """删除某会话的账本（/delete 时）。影子 commit 保留，无引用即垃圾。"""
        try:
            (self._repo_dir / f"{session_id}.ckpt.json").unlink(missing_ok=True)
        except OSError:
            pass

    def drop_all(self) -> None:
        """删除本 workspace 全部账本（/delete all）。影子 commit 同上保留。"""
        if self._repo_dir.exists():
            for f in self._repo_dir.glob("*.ckpt.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
        self._ledger = []

    # ── 影子仓库 git 操作 ──────────────────────────────────────────────────

    def _run(self, *args: str, cwd: str | None = None,
             input: str | None = None) -> subprocess.CompletedProcess:
        """在影子仓库上执行 git 命令。

        --git-dir/--work-tree 显式指向影子仓库与 workspace，不依赖用户
        的 git 环境（全局配置/分支/stash 都不受影响）。quotepath 关闭
        以免中文文件名被转义成八进制。cwd/input 供 git apply 使用：
        patch 内路径相对工作树根解析，patch 内容走 stdin。
        """
        cmd = [
            "git",
            "-c", "core.quotepath=false",
            f"--git-dir={self._repo_dir / '.git'}",
            f"--work-tree={self.workspace}",
            *args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=120, cwd=cwd, input=input)

    def _ensure_repo(self) -> bool:
        """首次使用时初始化影子仓库（幂等）。失败置 enabled=False 并返回 False。"""
        if (self._repo_dir / ".git").exists():
            return True
        try:
            self._repo_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "init", "-q", str(self._repo_dir)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                self.enabled = False
                return False
            cfg = lambda *a: subprocess.run(  # noqa: E731
                ["git", f"--git-dir={self._repo_dir / '.git'}", *a],
                capture_output=True, text=True, timeout=30)
            # 本地配置与用户全局配置隔离：固定身份、关签名、关换行转换。
            cfg("config", "user.name", "coding-agent")
            cfg("config", "user.email", "coding-agent@local")
            cfg("config", "commit.gpgsign", "false")
            cfg("config", "core.autocrlf", "false")
            info = self._repo_dir / ".git" / "info"
            info.mkdir(parents=True, exist_ok=True)
            (info / "exclude").write_text(_EXCLUDES, encoding="utf-8")
            return True
        except (OSError, subprocess.SubprocessError):
            self.enabled = False
            return False

    def _head(self) -> str | None:
        r = self._run("rev-parse", "HEAD")
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None

    def _commit_all(self, message: str) -> str | None:
        """add -A + commit，返回 commit hash。

        无变更且已有历史时复用当前 HEAD（空树快照无意义，省 commit）；
        无变更且无历史（首快照遇空 workspace）时 --allow-empty 建根提交，
        保证账本条目永远有 hash 可指。
        任何失败：置 enabled=False 并返回 None（快照降级，不抛错）。
        """
        if not self._ensure_repo():
            return None
        try:
            if self._run("add", "-A").returncode != 0:
                self.enabled = False
                return None
            head = self._head()
            if head is not None and self._run("diff", "--cached", "--quiet").returncode == 0:
                return head  # 无变更，复用当前 HEAD
            if self._run("commit", "-q", "--allow-empty", "-m", message).returncode != 0:
                self.enabled = False
                return None
            new = self._head()
            if new is None:
                self.enabled = False
            return new
        except (OSError, subprocess.SubprocessError):
            self.enabled = False
            return None

    def snapshot(self, preview: str) -> str | None:
        """为当前会话记录一个轮次快照：commit workspace 状态并追加账本。

        必须在用户消息 add_user 之前调用（快照 = 消息发出前的代码状态）。
        返回 commit hash；禁用或失败返回 None（仅快照失效，不影响任务）。
        """
        if not self.enabled or not self.session_id:
            return None
        commit = self._commit_all(f"[snapshot] {self.session_id}")
        if commit is None:
            return None
        self._ledger.append({
            "turn": len(self._ledger),
            "commit": commit,
            "ts": time.time(),
            "preview": (preview or "")[:60],
        })
        self._save_ledger()
        return commit

    # ── 还原：预览（只读）与执行（第 1 层防线 = 先打安全快照） ───────────────

    def plan_restore(self, entry: dict) -> RestorePlan | None:
        """只读计算还原预览：将被还原的改动 + 跨会话冲突（第 2 层防线）。

        扫描同 workspace 其它会话的账本，找时间晚于目标快照的条目，
        用其最新 commit 与目标做 --name-only diff：差集即「回退会一并
        撤销的其它会话改动」。diffstat 以影子仓库 HEAD 近似当前状态
        （用户在最后快照后的手改会在还原时的安全快照中被捕获）。
        """
        if not self.enabled:
            return None
        target = entry.get("commit") or ""
        ts = entry.get("ts") or 0.0
        if not target:
            return None
        conflicts: list[tuple[str, list[str]]] = []
        for p in self._repo_dir.glob("*.ckpt.json"):
            # 注意 stem 只剥最后一个后缀，须完整去掉 .ckpt.json 才是会话 id
            sid = p.name[: -len(".ckpt.json")]
            if sid == self.session_id:
                continue
            try:
                other = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            later = [e for e in other
                     if isinstance(e, dict) and e.get("commit") and e.get("ts", 0) > ts]
            if not later:
                continue
            files = self._diff_names(target, later[-1]["commit"])
            if files:
                conflicts.append((sid, files))
        head = self._head() or target
        return RestorePlan(target=target, diffstat=self._diff_stat(head, target),
                           conflicts=conflicts)

    def restore(self, entry: dict) -> RestoreResult | None:
        """还原 workspace 到目标快照。

        顺序不可颠倒：先对当前状态打安全快照（只进影子仓库不进账本——
        账本与用户轮次 1:1，不可被打破），再 reset --hard。任何一步失败
        返回 None，调用方应放弃整个回退（对话与代码保持一致）。
        """
        if not self.enabled:
            return None
        target = entry.get("commit") or ""
        if not target:
            return None
        safety = self._commit_all("[safety] 回退前自动快照")
        if safety is None:
            return None
        try:
            if self._run("reset", "--hard", target).returncode != 0:
                return None
        except (OSError, subprocess.SubprocessError):
            return None
        return RestoreResult(target=target, safety=safety)

    # ── 精确回退（第 3 层）：只撤销本会话的改动 ────────────────────────────

    def _entry_index(self, entry: dict) -> int:
        commit = entry.get("commit")
        for i, e in enumerate(self._ledger):
            if e.get("commit") == commit:
                return i
        return -1

    def _chain(self) -> list[str]:
        """影子仓库全局 commit 链（时间序，线性历史，无合并）。"""
        r = self._run("log", "--reverse", "--format=%H")
        if r.returncode != 0:
            return []
        return [line.strip() for line in r.stdout.splitlines() if line.strip()]

    def _own_starts(self, entry: dict) -> list[str]:
        """目标条目及其后本会话各轮的起点快照（去重，链上时间序）。

        无变更的轮次复用 HEAD（账本相邻同 hash），去重后自然跳过——
        该轮无工作可撤销。
        """
        idx = self._entry_index(entry)
        if idx < 0:
            return []
        starts: list[str] = []
        for e in self._ledger[idx:]:
            c = e.get("commit") or ""
            if c and (not starts or c != starts[-1]):
                starts.append(c)
        return starts

    def _start_shared(self, commit: str) -> bool:
        """该快照是否同时是其它会话某轮的起点（出现在其它会话账本中）。

        快照在轮开始时打：若其它会话的某轮也以同一 commit 为起点，说明
        从该快照到下一个快照之间工作区没有任何变化（否则后来那轮的起点
        会是一个新 commit）——即本会话以它为起点的那轮必然没做改动，
        这段 diff 全部属于其它会话，必须跳过而不是撤销。
        """
        for p in self._repo_dir.glob("*.ckpt.json"):
            sid = p.name[: -len(".ckpt.json")]
            if sid == self.session_id:
                continue
            try:
                other = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if any(isinstance(e, dict) and e.get("commit") == commit for e in other):
                return True
        return False

    def _revert_pairs(self, entry: dict) -> list[tuple[str, str]]:
        """需反向 apply 的 (起点快照 P, 链上下一快照 C) 对，按链序返回。

        快照在每轮开始时打，因此 P 与其直接子提交 C 之间的全部改动 =
        以 P 为起点的那轮的私有工作（其它会话的快照同样是轮开始时打，
        它们自己的工作必然发生各自的快照之后，落在别的区间里）。
        本会话要撤销的 = 目标条目及其后每个条目对应的区间 diff；
        restore 时先创建的安全快照恰好充当「最后一轮之后缺失的下一个
        快照」，使最后一轮的改动也有据可撤。
        """
        chain = self._chain()
        pos = {c: i for i, c in enumerate(chain)}
        pairs: list[tuple[str, str]] = []
        for P in self._own_starts(entry):
            if self._start_shared(P):
                continue
            i = pos.get(P)
            if i is not None and i + 1 < len(chain):
                pairs.append((P, chain[i + 1]))
        return pairs

    def _pair_status(self, pair: tuple[str, str]) -> list[tuple[str, str]]:
        """区间内各文件的变更类型 [(A/D/M/T, path)]。"""
        P, C = pair
        r = self._run("diff", "--name-status", "--no-renames", "--no-color", P, C)
        out: list[tuple[str, str]] = []
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0] and parts[1].strip():
                    out.append((parts[0], parts[1].strip()))
        return out

    def _blob_size(self, rev_path: str) -> int:
        """对象字节数（如 HEAD:path）；取不到返回 -1（走 patch 兜底路径）。"""
        r = self._run("cat-file", "-s", rev_path)
        if r.returncode != 0:
            return -1
        try:
            return int(r.stdout.strip())
        except ValueError:
            return -1

    def _single_patch(self, pair: tuple[str, str], f: str) -> str | None:
        P, C = pair
        r = self._run("diff", "--binary", "--no-renames", "--no-color",
                      P, C, "--", f)
        return r.stdout if r.returncode == 0 else None

    def _apply_pair_reverse(self, pair: tuple[str, str]) -> tuple[bool, list[str]]:
        """把一个区间 diff 反向应用到工作区。返回 (是否全部成功, 冲突文件)。

        逐文件处理：空文件的增删在 diff 里没有 hunk（git 不输出 ---/+++
        行，git apply 无法解析这种 header），因此 A/D 且目标 blob 为空的
        文件直接做文件系统操作并校验现状（现状不符 = 冲突）；其余文件走
        git apply -R 单文件 patch，上下文不匹配即该文件冲突。
        """
        P, C = pair
        conflicts: list[str] = []
        for status, f in self._pair_status(pair):
            handled = False
            if status == "A" and self._blob_size(f"{C}:{f}") == 0:
                # 反向 = 删除该空文件：现状须为空文件（删）或已不存在（免）
                p = Path(self.workspace) / f
                try:
                    if p.exists():
                        if p.stat().st_size == 0:
                            p.unlink()
                        else:
                            conflicts.append(f)
                    handled = True
                except OSError:
                    conflicts.append(f)
                    handled = True
            elif status == "D" and self._blob_size(f"{P}:{f}") == 0:
                # 反向 = 创建空文件：已存在且非空 = 冲突，已存在空文件 = 免
                p = Path(self.workspace) / f
                try:
                    if p.exists():
                        if p.stat().st_size != 0:
                            conflicts.append(f)
                    else:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"")
                    handled = True
                except OSError:
                    conflicts.append(f)
                    handled = True
            if handled:
                continue
            patch = self._single_patch(pair, f)
            if not patch or not patch.strip():
                continue
            try:
                r = self._run("apply", "-R", "--whitespace=nowarn",
                              cwd=self.workspace, input=patch)
            except (OSError, subprocess.SubprocessError):
                r = None
            if r is None or r.returncode != 0:
                conflicts.append(f)
        return (not conflicts, conflicts)

    def _worktree_diff_files(self, commit: str) -> list[str]:
        """工作区相对某快照的改动文件（含未跟踪新增），预览专用。"""
        files: list[str] = []
        r = self._run("diff", "--name-only", "--no-renames", "--no-color", commit)
        if r.returncode == 0:
            files += [line for line in r.stdout.splitlines() if line.strip()]
        r = self._run("ls-files", "--others", "--exclude-standard")
        if r.returncode == 0:
            files += [line for line in r.stdout.splitlines() if line.strip()]
        return files

    def precise_files(self, entry: dict) -> list[str]:
        """只读：精确回退将撤销的文件（/back 确认前预览用）。

        此刻安全快照尚未创建，最后一轮的改动还没有「下一个快照」可作
        diff 终点，用工作区相对该快照的 diff（含未跟踪文件）补全。
        """
        if not self.enabled:
            return []
        chain = self._chain()
        pos = {c: i for i, c in enumerate(chain)}
        files: list[str] = []
        for P in self._own_starts(entry):
            if self._start_shared(P):
                continue
            i = pos.get(P)
            if i is not None and i + 1 < len(chain):
                cand = self._diff_names(P, chain[i + 1], no_renames=True)
            else:
                cand = self._worktree_diff_files(P)
            for f in cand:
                if f not in files:
                    files.append(f)
        return files

    def restore_precise(self, entry: dict) -> PreciseResult | None:
        """精确回退：反向 apply 本会话各轮的私有 diff，其它会话的交叉修改
        原样保留。

        与 restore（reset --hard 全量回退）的区别：全量回退会把 workspace
        整个拖回旧快照，其它会话之后的改动一并被抹掉；本方法只反向 apply
        本会话各轮对应的 (快照 → 下一快照) 区间 diff，改动范围精确到本
        会话碰过的文件。同文件交叉时 apply 失败：reset --hard + clean -fd
        回滚到安全快照（还原前状态），返回 ok=False 及冲突文件，调用方可
        降级全量回退。返回 None 表示 git 层面失败（entry 非法 / 安全快照
        失败），调用方应放弃整个回退。
        """
        if not self.enabled:
            return None
        if not entry.get("commit") or self._entry_index(entry) < 0:
            return None
        safety = self._commit_all("[safety] 回退前自动快照")
        if safety is None:
            return None
        reverted: list[str] = []
        for pair in reversed(self._revert_pairs(entry)):
            # 逐文件反向 apply：空文件的增删 diff 没有 hunk（git apply
            # 无法解析这种 header），直接做文件系统操作并校验现状；
            # 其余文件走单文件 patch。任一文件失败 = 与其它会话同文件交叉。
            ok, conflicts = self._apply_pair_reverse(pair)
            if not ok:
                # apply 不动 index（仍停在安全快照），reset --hard 恢复
                # 已应用的部分；apply 可能凭空重建过「本会话曾删除的
                # 文件」，clean -fd 只清这类非忽略的未跟踪残留（安全
                # 快照已 add -A，其余未跟踪文件不存在）。
                self._run("reset", "--hard", safety)
                self._run("clean", "-fd")
                return PreciseResult(ok=False, safety=safety, files=conflicts)
            for f in self._diff_names(*pair, no_renames=True):
                if f not in reverted:
                    reverted.append(f)
        return PreciseResult(ok=True, safety=safety, files=reverted)

    def _diff_names(self, a: str, b: str, *, no_renames: bool = False) -> list[str]:
        """git diff --name-only [a..b] 的文件列表。

        no_renames=True 时加 --no-renames --no-color（区间私有改动需禁用
        重命名检测，把「改名」统一成「删除+新建」，反向 apply 才能逐文件处理）。
        """
        args = ["diff", "--name-only"]
        if no_renames:
            args += ["--no-renames", "--no-color"]
        args += [a, b]
        r = self._run(*args)
        if r.returncode != 0:
            return []
        return [line for line in r.stdout.splitlines() if line.strip()]

    def _diff_stat(self, a: str, b: str) -> str:
        r = self._run("diff", "--stat", a, b)
        return r.stdout.strip() if r.returncode == 0 else ""
