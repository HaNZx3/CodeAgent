"""CLI 入口。

用法：
    codeagent                # 进入交互式 REPL
    codeagent "任务描述"      # 一次性执行一个任务
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path

from config import Config
from agent.agent import CodingAgent
from agent.loop import RunResult, StepRecord
from tools.base import RiskInfo


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
        """收尾：最终回答已流式展示，这里只补一行收尾标记（含本轮用量）。

        上下文规模不再在此重复——它已由输入栏右下角常驻显示，收尾行只报
        本轮消耗，避免与状态栏信息冗余。
        """
        self._spinner.stop()
        self._newline()
        print()
        u = result.usage
        usage_part = ""
        if u.get("calls"):
            usage_part = (
                f" · 本轮 {u['calls']} 次调用 {_fmt_tokens(u['total_tokens'])} tokens"
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


def _confirm_destructive(risk: RiskInfo, view: _TurnView | None) -> bool:
    """高危操作执行前确认（对齐 Claude Code）。

    暂停当前 spinner，打印动作 / 命令 / 受影响文件，y/N 确认。
    提示代码快照可用 /back 找回。非交互场景（view 为 None）直接放行。
    """
    if view is not None:
        view._spinner.stop()
        view._newline()
    print(f"{C.YELLOW}⚠ 即将{risk.action}：{C.RESET}{C.BOLD}{risk.detail}{C.RESET}")
    if risk.files:
        print(f"{C.GRAY}  受影响文件：{_shorten(', '.join(risk.files), 80)}{C.RESET}")
    print(f"{C.GRAY}  已开启代码快照，误操作可用 /back 找回{C.RESET}")
    try:
        ok = input(f"{C.YELLOW}确认执行？[y/N]{C.RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ok = ""
    return ok == "y"


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


_ANSI_RE = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def _visible(s: str) -> str:
    """去掉 ANSI 转义序列，用于计算显示宽度。"""
    return _ANSI_RE.sub("", s)


def _disp_width(s: str) -> int:
    """字符串的终端显示宽度：CJK 全角字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "FW" else 1 for ch in s)


def _wtruncate(text: str, budget: int) -> str:
    """按显示宽度截断（CJK 感知），超预算以 … 结尾。"""
    if _disp_width(text) <= budget:
        return text
    out, w = "", 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in "FW" else 1
        if w + cw > budget - 1:
            return out + "…"
        out += ch
        w += cw
    return out


def _cursor_row() -> int | None:
    """DSR（\033[6n）查询光标所在行；短时间无响应返回 None。

    用于菜单打开前的滚屏预留。查询走 stdin 回读，用 kbhit + 截止时间
    防止在不响应 DSR 的终端上永久阻塞。
    """
    if not (sys.platform == "win32" and sys.stdin.isatty()):
        return None
    import msvcrt

    sys.stdout.write("\033[6n")
    sys.stdout.flush()
    deadline = time.time() + 0.2
    resp = ""
    while time.time() < deadline:
        if msvcrt.kbhit():
            resp += msvcrt.getwch()
            if resp.endswith("R"):
                m = re.search(r"(\d+);\d+R", resp)
                return int(m.group(1)) if m else None
        else:
            time.sleep(0.01)
    return None


def _readline(prompt: str, status=None) -> str:
    """读取一行输入（Claude Code 式交互）。

    Windows TTY 下：
      - 输入以 / 开头时，输入栏下方实时弹出命令菜单：按前缀过滤、
        精确匹配置顶，↑/↓ 选择，Tab 补全，Enter 补全并执行；单匹配
        时另有灰色 ghost 预览；Esc 关闭菜单，继续输入自动重新打开；
      - 输入栏右下角常驻状态文本（status 回调提供，来自真实 API usage）；
      - 打开输入行前用 DSR 查询光标行，过低则先滚屏预留出最大菜单的
        空间，保证菜单绘制不会越过屏幕底边覆盖输入行。
    非 TTY / 非 Windows 回退普通 input()，保证可移植。
    """
    if not (sys.platform == "win32" and sys.stdin.isatty()):
        return input(prompt)

    import msvcrt

    prompt_w = _disp_width(_visible(prompt))
    term_w = shutil.get_terminal_size().columns
    term_h = shutil.get_terminal_size().lines
    buf = ""
    sel = 0            # 菜单当前选中项
    esc_closed = False  # Esc 后暂时关闭菜单，直到输入变化

    def _matches() -> list[str]:
        if not buf.startswith("/") or " " in buf:
            return []
        exact = [c for c in COMMANDS if c == buf]
        return exact + [c for c in COMMANDS if c != buf and c.startswith(buf)]

    def _ghost() -> str:
        ms = _matches()
        if esc_closed or len(ms) != 1:
            return ""
        return ms[0][len(buf):]

    def _corner() -> str:
        """右下角常驻状态；输入行右侧放不下时省略。"""
        if status is None:
            return ""
        text = status()
        if not text:
            return ""
        used = prompt_w + _disp_width(buf) + _disp_width(_ghost())
        pad = term_w - used - _disp_width(_visible(text)) - 2
        if pad < 2:
            return ""
        return " " * pad + C.GRAY + text + C.RESET

    def _draw() -> None:
        """重绘输入行 + 菜单；光标回到 buf 末尾。"""
        ms = [] if esc_closed else _matches()
        ghost = _ghost()
        col = 1 + prompt_w + _disp_width(buf)
        out = ["\r\033[K", prompt, buf]
        if ghost:
            out += [C.GRAY, ghost, C.RESET]
        out.append(_corner())
        sys.stdout.write("".join(out))
        if ms:
            for i, cmd in enumerate(ms):
                desc = _wtruncate(COMMANDS[cmd], term_w - 16)
                if i == sel:
                    row = (f"{C.BOLD}{C.CYAN}❯ {cmd}{C.RESET}"
                           f" {C.GRAY}{desc}{C.RESET}")
                else:
                    row = f"{C.GRAY}  {cmd}  {desc}{C.RESET}"
                sys.stdout.write(f"\033[B\r\033[K{row}")
        # 清掉比上一帧多出来的菜单行（\033[J 从光标清到屏幕尾）
        sys.stdout.write("\033[B\r\033[J")
        sys.stdout.write(f"\r\033[{len(ms) + 1}A\033[{col}G")
        sys.stdout.flush()

    sys.stdout.write(prompt)
    sys.stdout.flush()
    # 菜单预留：光标行过低时先滚屏，保证最大菜单不会越过屏幕底边。
    # 固定发 N 个换行：光标先走到屏幕底，之后的每个 \n 精确滚 1 行，
    # 恰好滚到「输入行落在倒数第 N 行」，菜单空间即腾出。
    row = _cursor_row()
    if row is not None and row > term_h - (len(COMMANDS) + 1):
        sys.stdout.write("\n" * (len(COMMANDS) + 1))
        sys.stdout.write(f"\r\033[{len(COMMANDS) + 1}A")
        sys.stdout.flush()
    _draw()

    while True:
        ms = [] if esc_closed else _matches()
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            # 回车：菜单打开时补全选中命令并执行
            if ms:
                buf = ms[min(sel, len(ms) - 1)]
            sys.stdout.write("\r\033[K" + prompt + buf + "\n\033[J")
            sys.stdout.flush()
            return buf
        if ch == "\t":
            if ms:
                buf = ms[min(sel, len(ms) - 1)]
                sel = 0
                _draw()
        elif ch == "\x08" or ch == "\x7f":  # Backspace
            if buf:
                buf = buf[:-1]
                esc_closed = False
                sel = 0
                _draw()
        elif ch == "\x03":  # Ctrl+C
            sys.stdout.write("\033[B\r\033[J")
            raise KeyboardInterrupt
        elif ch == "\x1a":  # Ctrl+Z
            sys.stdout.write("\033[B\r\033[J")
            raise EOFError
        elif ch == "\x1b":  # Esc：关闭菜单
            if ms:
                esc_closed = True
                _draw()
        elif ch in ("\xe0", "\x00"):  # 方向键/功能键前缀
            nxt = msvcrt.getwch()
            if nxt == "H" and ms:  # ↑
                sel = max(sel - 1, 0)
                _draw()
            elif nxt == "P" and ms:  # ↓
                sel = min(sel + 1, len(ms) - 1)
                _draw()
        elif ch.isprintable():
            buf += ch
            esc_closed = False
            sel = 0
            _draw()


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
    # 一次性模式同样接入高危确认（同 REPL）。
    current: list[_TurnView | None] = [view]
    agent.registry.confirm_callback = lambda risk: _confirm_destructive(risk, current[0])
    view.begin()
    try:
        result = agent.run(
            task,
            on_step=view.step,
            on_text=view.text,
            on_step_start=view.step_start,
        )
    except KeyboardInterrupt:
        # Ctrl+C：打断当前思考 / 工具执行，返回（一次性模式即结束）。
        view._spinner.stop()
        view._newline()
        print(f"\n{C.YELLOW}⏹ 已中断（Ctrl+C）{C.RESET}")
        return
    view.finish(result)


def _run_repl(agent: CodingAgent, config: Config) -> None:
    print(f"{C.GRAY}输入任务开始，/help 查看命令，/exit 退出；运行中按 Ctrl+C 中断{C.RESET}\n")
    # 输入栏右下角常驻状态的数据：全部取自 API 返回的真实 usage。
    # ctx = 最近一次调用的 prompt_tokens（即当前上下文规模），
    # total = 本会话累计消耗 tokens。
    usage = {"ctx": 0, "total": 0}

    def _corner() -> str:
        if not usage["ctx"]:
            return ""
        win = config.context_window
        pct = usage["ctx"] / win * 100 if win > 0 else 0.0
        return (f"⏵ {_fmt_tokens(usage['ctx'])}/{_fmt_tokens(win)}"
                f" · {pct:.1f}% · 累计 {_fmt_tokens(usage['total'])} tokens")

    # 高危确认：registry 在 execute 前调用。需要暂停当前轮的 spinner 再 input，
    # 因此用 holder 持有「当前 _TurnView」，每轮 run 前更新。
    current: list[_TurnView | None] = [None]
    agent.registry.confirm_callback = lambda risk: _confirm_destructive(risk, current[0])

    while True:
        try:
            task = _readline(f"{C.CYAN}❯ {C.RESET}", status=_corner).strip()
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
        current[0] = view
        view.begin()
        try:
            result = agent.run(
                task,
                on_step=view.step,
                on_text=view.text,
                on_step_start=view.step_start,
            )
        except KeyboardInterrupt:
            # Ctrl+C：打断当前思考 / 工具执行，停 spinner 后回到输入栏。
            view._spinner.stop()
            view._newline()
            print(f"\n{C.YELLOW}⏹ 已中断（Ctrl+C）{C.RESET}")
            current[0] = None
            print()
            continue
        current[0] = None
        view.finish(result)
        u = result.usage
        if u.get("calls"):
            usage["ctx"] = u["prompt_tokens"]
            usage["total"] += u["total_tokens"]
        print()


if __name__ == "__main__":
    raise SystemExit(main())
