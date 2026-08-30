"""CLI 入口。

用法：
    codeagent                # 进入交互式 REPL
    codeagent "任务描述"      # 一次性执行一个任务
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from config import Config
from agent.agent import CodingAgent
from agent.loop import RunResult, StepRecord


# ── ANSI 颜色 ──────────────────────────────────────────────────────────────
# 现代终端（Windows Terminal / *nix）均原生支持；旧版 Windows 控制台需显式启用 VT。


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _enable_ansi() -> None:
    """启用 Windows 终端的 ANSI 转义序列支持（VT100）。"""
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(h, mode.value | 0x0004)


# ── Spinner：后台线程旋转的加载符号 ───────────────────────────────────────


class _Spinner:
    """Braille 旋转符号，在 LLM 思考 / 工具执行期间显示加载状态。"""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._msg = ""

    def start(self, msg: str = "") -> None:
        self.stop()
        # 非交互式终端（管道/重定向）不显示 spinner，避免帧堆积成乱码。
        if not sys.stdout.isatty():
            return
        self._msg = msg
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            line = f"{C.CYAN}{frame}{C.RESET} {C.DIM}{self._msg}{C.RESET}"
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)
        # 清行：用空格覆盖残留的旋转帧
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None


# ── 视图：把一次 run() 的状态变化渲染成终端输出 ─────────────────────────────


class _TurnView:
    """一次 run() 的终端视图：

    - run() 开始时启动「思考中」spinner；
    - 模型文字流式到达时停 spinner，打字机式输出；
    - 工具开始执行时切到「Running xxx」spinner；
    - 工具结束后显示颜色化的结果行，再重启「思考中」等待下一轮。
    """

    def __init__(self, context_window: int = 128_000):
        self._dirty = False  # 已有未收尾的流式输出
        self._spinner = _Spinner()
        self._context_window = context_window  # 占用百分比显示用（真实 usage / 配置窗口）

    def begin(self) -> None:
        """run() 开始时调用，启动思考 spinner。"""
        self._spinner.start("Thinking…")

    def text(self, delta: str) -> None:
        if self._spinner.active:
            self._spinner.stop()
        sys.stdout.write(delta)
        sys.stdout.flush()
        self._dirty = True

    def step_start(self, tool_name: str, arguments: dict) -> None:
        self._newline()
        self._spinner.start(f"Running {tool_name}…")

    def step(self, rec: StepRecord) -> None:
        self._spinner.stop()
        self._newline()
        if rec.tool_name is None:
            note = _shorten(rec.detail.strip(), 300)
            if note:
                print(f"{C.YELLOW}✻ {note}{C.RESET}")
            self._spinner.start("Thinking…")
            return
        mark = f"{C.GREEN}✓{C.RESET}" if rec.success else f"{C.RED}✗{C.RESET}"
        args = _shorten(str(rec.arguments))
        print(
            f"{C.CYAN}● {C.BOLD}{rec.tool_name}{C.RESET}"
            f"{C.GRAY} {args}{C.RESET} {mark}"
            f" {C.GRAY}{rec.duration_ms:.0f}ms{C.RESET}"
        )
        detail = rec.detail.strip()
        if detail:
            preview = "\n".join(detail.splitlines()[:6])
            print(f"{C.GRAY}{_shorten(preview, 300)}{C.RESET}")
        self._spinner.start("Thinking…")

    def _newline(self) -> None:
        if self._dirty:
            print()
            self._dirty = False

    def finish(self, result: RunResult) -> None:
        """收尾：最终回答已流式展示，这里只补一行收尾标记（含真实用量）。"""
        self._spinner.stop()
        self._newline()
        print()
        # 真实用量并入状态行：prompt 取本次最后一次调用的值——最终回答后
        # messages 不再变化，它就是当前上下文的真实规模。
        u = result.usage
        usage_part = ""
        if u.get("calls"):
            used = u["prompt_tokens"]
            pct = used / self._context_window * 100 if self._context_window > 0 else 100.0
            usage_part = (
                f" · 上下文 {_fmt_tokens(used)}/{_fmt_tokens(self._context_window)}"
                f" ({pct:.1f}%) · 本轮 {u['calls']} 次调用 {_fmt_tokens(u['total_tokens'])} tokens"
            )
        if result.final_text is not None:
            print(f"{C.GREEN}{'─' * 40}{C.RESET}")
            print(f"{C.GREEN}✓ 任务完成{C.RESET}{C.GRAY}{usage_part}{C.RESET}")
        else:
            print(f"{C.RED}✗ 任务停止：{result.stop_reason}{C.RESET}{C.GRAY}{usage_part}{C.RESET}")


def _shorten(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _fmt_tokens(n: int) -> str:
    """Claude Code 式 token 缩写：930 -> '930'，12_800 -> '12.8k'，128_000 -> '128k'。"""
    if n < 10_000:
        return f"{n:,}"
    k = n / 1000
    s = f"{k:.1f}".rstrip("0").rstrip(".")
    return f"{s}k"


def _context_bar(used: int, window: int, width: int = 20) -> str:
    """上下文占用进度条：按 used/window 比例填充，绿色<50% / 黄色<80% / 红色≥80%。"""
    if window <= 0:
        ratio = 1.0
    else:
        ratio = min(used / window, 1.0)
    filled = round(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C.GREEN if ratio < 0.5 else (C.YELLOW if ratio < 0.8 else C.RED)
    return f"{color}{bar}{C.RESET}"


# ── 内置斜杠命令 ───────────────────────────────────────────────────────────


COMMANDS: dict[str, str] = {
    "/help": "显示可用命令",
    "/exit": "退出 Agent",
    "/clear": "清空当前会话上下文（仅保留系统提示词，id 不变）",
    "/new": "开新会话（可选名称 /new <name>）",
    "/resume": "恢复历史会话 /resume <id>",
    "/sessions": "列出当前 workspace 的所有会话",
    "/delete": "删除会话：/delete <id> 或 /delete all",
    "/back": "回退对话（及代码）到某条用户消息之前 /back [n]",
    "/compact": "手动压缩当前对话历史",
    "/model": "显示当前模型",
    "/workspace": "显示当前工作目录",
    "/status": "显示会话状态",
}


class _QuitRepl(Exception):
    """REPL 中收到退出命令时抛出，用于跨函数跳出循环。"""


def _common_prefix(strings: list[str]) -> str:
    """多个字符串的最长公共前缀，用于多匹配时补全到公共部分。"""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _readline(prompt: str) -> str:
    """读取一行输入；Windows TTY 下支持实时补全预览（ghost text）。

    交互规则（Claude Code 式）：
      - 输入过程中实时显示补全预览：单匹配时把补全部分以灰色显示在
        光标后（如输入 /h 显示 /h + 灰色 elp），按 Tab 或回车即确认；
      - 按 Tab：确认补全预览；若无预览且多匹配，在下方列出候选列表；
      - 无匹配：不补全。
    非 TTY（管道/重定向）或非 Windows 回退到普通 input()，保证可移植。
    """
    if not (sys.platform == "win32" and sys.stdin.isatty()):
        return input(prompt)

    import msvcrt

    buf = ""

    def _ghost() -> str:
        """当前 buf 的补全预览（单匹配时补全部分）。"""
        if not buf:
            return ""
        matches = [c for c in COMMANDS if c.startswith(buf)]
        if len(matches) == 1:
            return matches[0][len(buf):]
        return ""

    def _refresh() -> None:
        """重写输入行：prompt + buf + 灰色补全预览，光标停在 buf 末尾。"""
        ghost = _ghost()
        # \r 回行首；\033[K 清到行尾；重写整行
        sys.stdout.write(f"\r\033[K{prompt}{buf}{C.GRAY}{ghost}{C.RESET}")
        if ghost:
            # 光标左移 ghost 长度，回到 buf 末尾（补全部分右侧）
            sys.stdout.write(f"\033[{len(ghost)}D")
        sys.stdout.flush()

    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            # 回车：若有补全预览则确认，再执行 buf
            ghost = _ghost()
            if ghost:
                buf += ghost
            sys.stdout.write("\n")
            return buf
        if ch == "\t":
            ghost = _ghost()
            if ghost:
                # 有预览：Tab 确认补全
                buf += ghost
                _refresh()
                continue
            # 无预览：尝试多匹配列候选
            matches = [c for c in COMMANDS if c.startswith(buf)]
            if not matches:
                continue
            if len(matches) == 1:
                buf = matches[0]
                _refresh()
            else:
                # 多匹配：补全到公共前缀，列出候选
                buf = _common_prefix(matches)
                print()
                for m in matches:
                    print(f"  {C.CYAN}{m:<14}{C.RESET} {C.GRAY}{COMMANDS[m]}{C.RESET}")
                sys.stdout.write(f"{prompt}{buf}")
                sys.stdout.flush()
        elif ch == "\x08":  # Backspace
            if buf:
                buf = buf[:-1]
                _refresh()
        elif ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        elif ch == "\x1a":  # Ctrl+Z
            raise EOFError
        elif ch == "\xe0":  # 方向键/功能键前缀，丢弃下一个字节
            msvcrt.getwch()
        elif ch.isprintable():
            buf += ch
            _refresh()


def _handle_back(agent: CodingAgent, arg: str) -> None:
    """/back：回退对话（及代码）到某条用户消息之前。

    代码还原先于对话截断：还原可能因 git 异常失败，失败时整个回退取消，
    保证「对话与代码要么都回退，要么都不动」。
    还原默认走精确回退（只反向 apply 本会话的改动，保留其它会话的修改）；
    与其它会话同文件交叉时降级询问是否全量回退。
    快照不可用（未启用 / 账本与轮次不对齐）时退化为仅回退对话。
    """
    turns = agent.context.user_turns()
    if not turns:
        print(f"{C.GRAY}（没有可回退的用户消息）{C.RESET}")
        return
    msgs = agent.context.get_messages()
    ck = getattr(agent, "checkpoints", None)
    # 账本第 k 项 = 第 k 条用户消息发出前的快照；条数对齐才可还原代码。
    use_ckpt = (
        ck is not None
        and ck.enabled
        and len(ck.entries()) == len(turns)
    )
    if ck is not None and ck.enabled and not use_ckpt:
        print(f"{C.GRAY}（快照账本与对话轮次不对齐，仅回退对话）{C.RESET}")

    # ── 选择回退点 ──
    n: int | None = None
    if arg.isdigit() and 1 <= int(arg) <= len(turns):
        n = int(arg)  # /back <n> 直接回退，不再询问
    else:
        print(f"{C.CYAN}回退到哪条消息之前？{C.RESET}")
        for i, idx in enumerate(turns, 1):
            preview = _shorten((msgs[idx].get("content") or "").splitlines()[0], 46)
            mark = ""
            if use_ckpt:
                mark = f"  {C.GRAY}[快照 {ck.entries()[i - 1]['commit'][:7]}]{C.RESET}"
            print(f"  {C.BOLD}{i}{C.RESET}  {preview}{mark}")
        try:
            choice = input(f"{C.GRAY}输入编号（回车取消）{C.RESET} ").strip()
        except EOFError:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(turns):
            n = int(choice)
        else:
            print(f"{C.GRAY}已取消{C.RESET}")
            return
    idx = turns[n - 1]

    # ── 代码还原（先于对话截断）──
    if use_ckpt:
        entry = ck.entries()[n - 1]
        plan = ck.plan_restore(entry)
        if plan is not None:
            own_files = ck.precise_files(entry)
            if plan.conflicts:
                print(f"{C.YELLOW}⚠ 以下会话在该快照之后也修改过 workspace：{C.RESET}")
                for sid, files in plan.conflicts:
                    print(f"    会话 {sid}：{_shorten(', '.join(files), 80)}")
                print(f"{C.GRAY}  精确回退只撤销本会话的改动，其它会话的修改会保留。{C.RESET}")
            if own_files:
                print(f"{C.GRAY}将撤销本会话对以下文件的改动：{C.RESET}")
                print(f"    {_shorten(', '.join(own_files), 80)}")
            else:
                print(f"{C.GRAY}本会话在该快照之后没有代码改动{C.RESET}")
            try:
                ok = input(f"{C.YELLOW}确认回退对话+代码？[y/N]{C.RESET} ").strip().lower()
            except EOFError:
                ok = ""
            if ok != "y":
                print(f"{C.GRAY}已取消{C.RESET}")
                return
            pr = ck.restore_precise(entry)
            if pr is not None and pr.ok:
                if pr.files:
                    print(f"{C.GREEN}✓ 已精确回退本会话对 {len(pr.files)} 个文件的改动{C.RESET}")
                else:
                    print(f"{C.GREEN}✓ 代码无需改动{C.RESET}")
                print(f"{C.GRAY}还原前状态已存为安全快照 {pr.safety[:7]}，"
                      f"如需找回见 README「快照与回退」{C.RESET}")
            else:
                if pr is None:
                    print(f"{C.RED}✗ 代码还原失败，回退已取消（对话与代码保持原样）{C.RESET}")
                    return
                # 同文件交叉，反向 apply 失败：工作区已回滚，询问是否全量回退
                print(f"{C.YELLOW}✗ 其它会话与你在相同文件上有交叉改动，无法精确回退：{C.RESET}")
                print(f"    {_shorten(', '.join(pr.files), 80)}")
                print(f"{C.GRAY}工作区已回滚到还原前状态（安全快照 {pr.safety[:7]}）{C.RESET}")
                try:
                    hard = input(f"{C.YELLOW}改用全量回退？将一并撤销其它会话的改动 [y/N]{C.RESET} ").strip().lower()
                except EOFError:
                    hard = ""
                if hard != "y":
                    print(f"{C.GRAY}已取消{C.RESET}")
                    return
                result = ck.restore(entry)
                if result is None:
                    print(f"{C.RED}✗ 全量回退失败，对话与代码保持原样{C.RESET}")
                    return
                print(f"{C.GREEN}✓ 代码已全量还原至快照 {result.target[:7]}{C.RESET}")
                print(f"{C.GRAY}还原前状态已存为安全快照 {result.safety[:7]}，"
                      f"如需找回见 README「快照与回退」{C.RESET}")
        # 无论是否走了代码还原，账本都必须与回退后的轮次保持 1:1。
        ck.truncate(n - 1)

    # ── 对话截断 ──
    removed = agent.context.rewind_to(idx)
    print(f"{C.GREEN}✓ 已回退：删除其后 {removed} 条消息{C.RESET}")


def _handle_command(agent: CodingAgent, cmd: str, config: Config) -> bool:
    """处理斜杠内置命令。返回 True 表示已处理（不应发给 Agent）。

    非斜杠开头返回 False，交给 Agent 作为任务处理；未知命令也算已处理
    （给出提示，避免把 /xxx 当任务发给模型）。
    """
    cmd = cmd.strip().lower()
    if not cmd.startswith("/"):
        return False
    # 拆出子命令名和参数（参数可能含空格，故 maxsplit=1）。
    parts = cmd.split(maxsplit=1)
    name = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if name == "/help":
        print(f"{C.CYAN}可用命令：{C.RESET}")
        for n, d in COMMANDS.items():
            print(f"  {C.BOLD}{n:<12}{C.RESET}{C.GRAY}{d}{C.RESET}")
        return True
    if name == "/exit":
        raise _QuitRepl()
    if name == "/clear":
        # 原地清空当前会话：id 不变，只抹掉对话历史（与 /new 开新会话区分）。
        sid = agent.session_id
        agent.clear_context()
        print(f"{C.GREEN}✓ 已清空当前会话上下文：{sid}{C.RESET}")
        print(f"{C.GRAY}仅保留系统提示词，历史已从内存与会话文件中移除{C.RESET}")
        return True
    if name == "/new":
        sid = agent.new_session(arg or None)
        print(f"{C.GREEN}✓ 新会话：{sid}{C.RESET}")
        return True
    if name == "/resume":
        if not arg:
            print(f"{C.RED}✗ 用法：/resume <session_id>{C.RESET}")
            return True
        try:
            agent.switch_session(arg)
        except Exception as e:
            print(f"{C.RED}✗ 恢复失败：{e}{C.RESET}")
            return True
        history = agent.context.get_messages()
        preview = ""
        for m in history:
            if m.get("role") == "user":
                preview = (m.get("content", "") or "")[:60]
                break
        print(f"{C.GREEN}✓ 已恢复会话：{agent.session_id}{C.RESET}")
        print(f"  {C.GRAY}历史消息{C.RESET}    {len(history)} 条")
        if preview:
            print(f"  {C.GRAY}首条任务{C.RESET}    {preview}")
        return True
    if name == "/sessions":
        sessions = agent.store.list_sessions()
        if not sessions:
            print(f"{C.GRAY}（暂无会话记录）{C.RESET}")
            return True
        print(f"{C.CYAN}最近会话{C.RESET} {C.GRAY}（workspace: {config.workspace}）{C.RESET}")
        for sid, preview, mtime in sessions:
            mark = f"{C.GREEN}*{C.RESET}" if sid == agent.session_id else " "
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            print(f"  {mark} {C.BOLD}{sid}{C.RESET}  {C.GRAY}{preview:<40}{ts}{C.RESET}")
        print(f"{C.GRAY}用 /resume <id> 恢复某个会话{C.RESET}")
        return True
    if name == "/delete":
        if not arg:
            print(f"{C.RED}✗ 用法：/delete <id> 删除指定会话，或 /delete all 删除全部{C.RESET}")
            return True
        if arg == "all":
            sessions = agent.store.list_sessions()
            if not sessions:
                print(f"{C.GRAY}（当前 workspace 无会话可删）{C.RESET}")
                return True
            # 批量删除不可逆，要求显式确认。
            try:
                confirm = input(
                    f"{C.YELLOW}确认删除全部 {len(sessions)} 个会话？此操作不可逆 [y/N]{C.RESET} "
                ).strip().lower()
            except EOFError:
                confirm = ""
            if confirm != "y":
                print(f"{C.GRAY}已取消{C.RESET}")
                return True
            n = agent.store.clear_all()
            ck = getattr(agent, "checkpoints", None)
            if ck is not None:
                ck.drop_all()  # 快照账本随会话一并清理
            # 当前会话必被删，开新空会话避免内存 messages 与文件不一致。
            new_id = agent.new_session()
            print(f"{C.GREEN}✓ 已删除 {n} 个会话{C.RESET}")
            print(f"{C.GRAY}当前会话也被删，已开新会话：{new_id}{C.RESET}")
            return True
        target = arg
        if target == agent.session_id:
            agent.store.clear(target)
            ck = getattr(agent, "checkpoints", None)
            if ck is not None:
                ck.drop_session(target)
            new_id = agent.new_session()
            print(f"{C.GREEN}✓ 已删除当前会话，已开新会话：{new_id}{C.RESET}")
            return True
        if not agent.store.path(target).exists():
            print(f"{C.RED}✗ 会话 {target} 不存在{C.RESET}")
            print(f"{C.GRAY}用 /sessions 查看可用会话{C.RESET}")
            return True
        agent.store.clear(target)
        ck = getattr(agent, "checkpoints", None)
        if ck is not None:
            ck.drop_session(target)
        print(f"{C.GREEN}✓ 已删除会话：{target}{C.RESET}")
        return True
    if name == "/back":
        _handle_back(agent, arg)
        return True
    if name == "/compact":
        before = len(agent.context.get_messages())
        agent.context.maybe_compact(force=True)
        after = len(agent.context.get_messages())
        print(f"{C.GREEN}✓ 压缩完成{C.RESET}  {C.GRAY}消息数 {before} -> {after}{C.RESET}")
        return True
    if name == "/model":
        print(f"  {C.GRAY}model{C.RESET}      {config.model}")
        return True
    if name == "/workspace":
        print(f"  {C.GRAY}workspace{C.RESET}  {config.workspace}")
        return True
    if name == "/status":
        n = len(agent.context.messages)
        window = config.context_window
        print(f"  {C.GRAY}历史消息{C.RESET}    {n} 条")
        # 上下文占用：最后一次 API 返回的真实 prompt_tokens（= 当前 messages 规模）。
        # 标签按终端等宽对齐：全角字占 2 列，「距自动压缩/本会话累计」5 个全角字后补 2 空格。
        last = agent.context.last_prompt_tokens
        if last is not None:
            pct = last / window * 100 if window > 0 else 100.0
            print(f"  {C.GRAY}上下文{C.RESET}      "
                  f"{last:,} / {window:,} tokens ({pct:.1f}%)  {_context_bar(last, window)}")
            remain = agent.context.compact_threshold - last
            if remain > 0:
                print(f"  {C.GRAY}距自动压缩{C.RESET}  "
                      f"还剩 {remain:,} tokens（阈值 {agent.context.compact_threshold:,}）")
            else:
                print(f"  {C.GRAY}距自动压缩{C.RESET}  "
                      f"已达阈值 {agent.context.compact_threshold:,}，下次任务开始时压缩")
        else:
            print(f"  {C.GRAY}上下文{C.RESET}      尚未调用 LLM（窗口 {window:,} tokens）")
        print(f"  {C.GRAY}当前会话{C.RESET}    {agent.session_id}")
        ck = getattr(agent, "checkpoints", None)
        if ck is not None:
            if ck.enabled:
                print(f"  {C.GRAY}代码快照{C.RESET}    {len(ck.entries())} 个（/back 可回退代码）")
            else:
                print(f"  {C.GRAY}代码快照{C.RESET}    不可用（git 缺失或已关闭）")
        print(f"  {C.GRAY}workspace{C.RESET}   {config.workspace}")
        return True
    print(f"{C.RED}✗ 未知命令：{name}{C.RESET}  输入 {C.BOLD}/help{C.RESET} 查看可用命令")
    return True


# ── 启动横幅 ───────────────────────────────────────────────────────────────


def _print_banner(agent: CodingAgent, config: Config) -> None:
    print(f"{C.CYAN}{C.BOLD}✦ CodeAgent{C.RESET}{C.GRAY} v0.2.0{C.RESET}")
    print(f"  {C.GRAY}workspace{C.RESET}  {config.workspace}")
    print(f"  {C.GRAY}model{C.RESET}      {config.model}")
    print(f"  {C.GRAY}session{C.RESET}    {agent.session_id}")
    print()


# ── 入口 ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    _enable_ansi()
    parser = argparse.ArgumentParser(description="自研 Coding Agent")
    parser.add_argument("task", nargs="?", default=None, help="要执行的任务（不传则进入交互模式）")
    parser.add_argument("--workspace", default=None, help="覆盖工作目录")
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--base-url", default=None, help="覆盖 API base_url")
    args = parser.parse_args(argv)

    config = Config.from_env()
    if args.workspace:
        config.workspace = args.workspace
    elif not config.workspace:
        # 不传 --workspace 且环境变量也没设时，绑定当前工作目录，
        # 让 Agent 能像 Claude Code 一样在任意文件夹启动。
        config.workspace = str(Path.cwd())
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url

    try:
        agent = CodingAgent(config)
    except RuntimeError as e:
        print(f"{C.RED}✗ {e}{C.RESET}", file=sys.stderr)
        return 1

    _print_banner(agent, config)

    if args.task:
        _run_once(agent, args.task)
    else:
        _run_repl(agent, config)
    return 0


def _run_once(agent: CodingAgent, task: str) -> None:
    print(f"{C.CYAN}❯ {C.RESET}{task}\n")
    view = _TurnView(agent.config.context_window)
    view.begin()
    view.finish(
        agent.run(
            task,
            on_step=view.step,
            on_text=view.text,
            on_step_start=view.step_start,
        )
    )


def _run_repl(agent: CodingAgent, config: Config) -> None:
    print(f"{C.GRAY}输入任务开始，/help 查看命令，/exit 退出{C.RESET}\n")
    while True:
        try:
            task = _readline(f"{C.CYAN}❯ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}再见{C.RESET}")
            break
        if not task:
            continue
        try:
            if _handle_command(agent, task, config):
                continue
        except _QuitRepl:
            print(f"{C.GRAY}再见{C.RESET}")
            break
        print()
        view = _TurnView(config.context_window)
        view.begin()
        view.finish(
            agent.run(
                task,
                on_step=view.step,
                on_text=view.text,
                on_step_start=view.step_start,
            )
        )
        print()


if __name__ == "__main__":
    raise SystemExit(main())
