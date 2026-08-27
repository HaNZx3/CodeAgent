"""CLI 入口。

用法：
    python main.py                 # 进入交互式 REPL
    python main.py "任务描述"       # 一次性执行一个任务
"""

from __future__ import annotations

import argparse
import sys

from config import Config
from agent.agent import CodingAgent
from agent.loop import RunResult


def render(result: RunResult) -> None:
    """把 RunResult 渲染成便于观察 Agent 决策过程的日志。"""
    for rec in result.steps:
        if rec.tool_name is None:
            continue
        status = "成功" if rec.success else "失败"
        args = _shorten(str(rec.arguments))
        print(f"[Tool] {rec.tool_name} {args}  ->  {status}  ({rec.duration_ms:.0f}ms)")
        detail = rec.detail.strip()
        if detail:
            preview = "\n".join(detail.splitlines()[:6])
            print(f"       {_shorten(preview, 300)}")
    print()
    if result.final_text is not None:
        print("[Agent] 任务完成")
        print("-" * 40)
        print(result.final_text)
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
    render(agent.run(task))


def _run_repl(agent: CodingAgent) -> None:
    while True:
        try:
            task = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            break
        print()
        render(agent.run(task))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
