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

    def __init__(self):
        self._dirty = False  # 已有未收尾的流式输出
        self._spinner = _Spinner()

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
        """收尾：最终回答已流式展示，这里只补分隔与状态标记。"""
        self._spinner.stop()
        self._newline()
        print()
        if result.final_text is not None:
            print(f"{C.GREEN}{'─' * 40}{C.RESET}")
            print(f"{C.GREEN}✓ 任务完成{C.RESET}")
        else:
            print(f"{C.RED}✗ 任务停止：{result.stop_reason}{C.RESET}")


def _shorten(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# ── 内置斜杠命令 ───────────────────────────────────────────────────────────


COMMANDS: dict[str, str] = {
    "/help": "显示可用命令",
    "/exit": "退出 Agent",
    "/quit": "退出 Agent",
    "/clear": "清空对话历史，重新开始",
    "/model": "显示当前模型",
    "/workspace": "显示当前工作目录",
    "/status": "显示对话历史长度",
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
    """读取一行输入；Windows TTY 下支持 Tab 补全斜杠命令。

    Tab 补全规则（Claude Code 式）：
      - 单匹配：直接补全为完整命令；
      - 多匹配：补全到公共前缀，并在下方列出全部候选；
      - 无匹配：不补全。
    非 TTY（管道/重定向）或非 Windows 回退到普通 input()，保证可移植。
    """
    if not (sys.platform == "win32" and sys.stdin.isatty()):
        return input(prompt)

    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf = ""
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            return buf
        if ch == "\t":
            matches = [c for c in COMMANDS if c.startswith(buf)]
            if not matches:
                continue
            if len(matches) == 1:
                # 单匹配：补全为完整命令，追加缺失字符
                completion = matches[0][len(buf):]
                buf = matches[0]
                sys.stdout.write(completion)
                sys.stdout.flush()
            else:
                # 多匹配：补全到公共前缀，并列出候选列表
                prefix = _common_prefix(matches)
                completion = prefix[len(buf):]
                if completion:
                    buf = prefix
                    sys.stdout.write(completion)
                    sys.stdout.flush()
                # 用 print 确保每行正确换行（write+\n 在部分终端下不可靠）
                print()
                for m in matches:
                    print(f"  {C.CYAN}{m:<14}{C.RESET} {C.GRAY}{COMMANDS[m]}{C.RESET}")
                # 重新显示提示符 + 当前 buf，让用户继续输入缩小范围
                print(f"{prompt}{buf}", end="", flush=True)
        elif ch == "\x08":  # Backspace
            if buf:
                buf = buf[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
        elif ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        elif ch == "\x1a":  # Ctrl+Z
            raise EOFError
        elif ch == "\xe0":  # 方向键/功能键前缀，丢弃下一个字节
            msvcrt.getwch()
        elif ch.isprintable():
            buf += ch
            sys.stdout.write(ch)
            sys.stdout.flush()


def _handle_command(agent: CodingAgent, cmd: str, config: Config) -> bool:
    """处理斜杠内置命令。返回 True 表示已处理（不应发给 Agent）。

    非斜杠开头返回 False，交给 Agent 作为任务处理；未知命令也算已处理
    （给出提示，避免把 /xxx 当任务发给模型）。
    """
    cmd = cmd.strip().lower()
    if not cmd.startswith("/"):
        return False
    if cmd == "/help":
        print(f"{C.CYAN}可用命令：{C.RESET}")
        for name, desc in COMMANDS.items():
            print(f"  {C.BOLD}{name:<12}{C.RESET}{C.GRAY}{desc}{C.RESET}")
        return True
    if cmd in ("/exit", "/quit"):
        raise _QuitRepl()
    if cmd == "/clear":
        agent.context.clear()
        print(f"{C.GREEN}✓ 对话历史已清空，开启新对话{C.RESET}")
        return True
    if cmd == "/model":
        print(f"  {C.GRAY}model{C.RESET}      {config.model}")
        return True
    if cmd == "/workspace":
        print(f"  {C.GRAY}workspace{C.RESET}  {config.workspace}")
        return True
    if cmd == "/status":
        n = len(agent.context.messages)
        print(f"  {C.GRAY}历史消息{C.RESET}    {n} 条")
        return True
    print(f"{C.RED}✗ 未知命令：{cmd}{C.RESET}  输入 {C.BOLD}/help{C.RESET} 查看可用命令")
    return True


# ── 启动横幅 ───────────────────────────────────────────────────────────────


def _print_banner(config: Config) -> None:
    print(f"{C.CYAN}{C.BOLD}✦ CodeAgent{C.RESET}{C.GRAY} v0.2.0{C.RESET}")
    print(f"  {C.GRAY}workspace{C.RESET}  {config.workspace}")
    print(f"  {C.GRAY}model{C.RESET}      {config.model}")
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

    _print_banner(config)

    if args.task:
        _run_once(agent, args.task)
    else:
        _run_repl(agent, config)
    return 0


def _run_once(agent: CodingAgent, task: str) -> None:
    print(f"{C.CYAN}❯ {C.RESET}{task}\n")
    view = _TurnView()
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
        view = _TurnView()
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
