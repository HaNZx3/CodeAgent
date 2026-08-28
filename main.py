"""CLI 入口。

用法：
    python main.py                 # 进入交互式 REPL
    python main.py "任务描述"       # 一次性执行一个任务
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import Config
from agent.agent import CodingAgent
from agent.loop import RunResult, StepRecord


class _TurnView:
    """一次 run() 的终端视图：模型文字逐字流出（打字机效果），
    工具行与换行时机由本类统一调度，避免重复打印与粘连。"""

    def __init__(self):
        self._dirty = False  # 已有未收尾的流式输出

    def text(self, delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()
        self._dirty = True

    def _newline(self) -> None:
        if self._dirty:
            print()
            self._dirty = False

    def step(self, rec: StepRecord) -> None:
        self._newline()
        if rec.tool_name is None:
            note = _shorten(rec.detail.strip(), 300)
            if note:
                print(f"[Agent] {note}")
            return
        status = "成功" if rec.success else "失败"
        args = _shorten(str(rec.arguments))
        print(f"[Tool] {rec.tool_name} {args}  ->  {status}  ({rec.duration_ms:.0f}ms)")
        detail = rec.detail.strip()
        if detail:
            preview = "\n".join(detail.splitlines()[:6])
            print(f"       {_shorten(preview, 300)}")

    def finish(self, result: RunResult) -> None:
        """收尾：最终回答已流式展示，这里只补换行和状态标记。"""
        self._newline()
        print()
        if result.final_text is not None:
            print("-" * 40)
            print("[Agent] 任务完成")
        else:
            print(f"[Agent] 任务停止：{result.stop_reason}")


def _shorten(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def main(argv: list[str] | None = None) -> int:
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
        print(f"[Error] {e}", file=sys.stderr)
        return 1

    print("Coding Agent")
    print(f"Workspace: {config.workspace}")
    print(f"Model:     {config.model}")
    print()

    if args.task:
        _run_once(agent, args.task)
    else:
        _run_repl(agent)
    return 0


def _run_once(agent: CodingAgent, task: str) -> None:
    print(f"> {task}\n")
    view = _TurnView()
    view.finish(agent.run(task, on_step=view.step, on_text=view.text))


def _run_repl(agent: CodingAgent) -> None:
    while True:
        try:
            task = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not task:
            continue
        print()
        view = _TurnView()
        view.finish(agent.run(task, on_step=view.step, on_text=view.text))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
