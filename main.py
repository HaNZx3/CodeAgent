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
        _run_repl(agent)
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


def _run_repl(agent: CodingAgent) -> None:
    while True:
        try:
            task = input(f"{C.CYAN}❯ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}再见{C.RESET}")
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
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
