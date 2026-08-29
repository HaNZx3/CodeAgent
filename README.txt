# 自研 Coding Agent

一个从零实现的轻量级编码智能体：调用大模型，自主读写本地项目文件、
搜索代码、执行测试与命令，并根据结果迭代直到完成任务。

核心的 Agent 运行逻辑（对话历史与上下文管理、工具定义与注册、工具调用解析、
工具本地执行、Agent Loop、停止条件、错误恢复、Workspace 安全边界）全部自行实现，
不依赖 LangChain / AutoGen / CrewAI 等任何 Agent 框架。

## 快速开始
1. pip install -r requirements.txt
2. 配置凭据（凭据只走 .env / 环境变量，不进入仓库）：
   方式 A（推荐）：复制 .env.example 为 .env，填入 OPENAI_API_KEY
   方式 B：export OPENAI_API_KEY=...   # 或 DEEPSEEK_API_KEY
   可选：OPENAI_BASE_URL / OPENAI_MODEL 切换 DeepSeek、Qwen 等兼容服务
3. python main.py "修复 demo 项目中的 bug 并让所有测试通过"
   或运行 python main.py 进入交互模式

## 目录结构
agent/   Agent 状态与循环（loop / context / stop / session / memory）
llm/     模型 API 通信
tools/   工具定义与本地执行（file / search / shell）
demo/    演示用小型 Python 项目（含故意植入的 bug）
tests/   自动化测试（tool / context / agent loop / session / memory）

## 运行测试
pytest tests/

## 演示
demo/calculator.py 的 multiply 函数有一个 bug（+ 应为 *）。
让 Agent 执行「修复测试错误」，它会自动：list_files -> search_code ->
read_file -> edit_file -> run_command(pytest) -> 失败后修复 -> 通过 -> 总结。

## 项目记忆（可选）
在 workspace 根目录放 AGENT.md，写入项目特定指引（构建命令、测试方式、
代码约定、已知坑点），Agent 启动时自动注入 system prompt。
在 ~/.coding-agent/USER.md 写入跨项目的个人偏好（编辑器、语言习惯）。

示例 AGENT.md：
  # 项目记忆
  - 测试：pytest tests/
  - 包管理：uv
  - 约定：所有函数加类型注解

## 会话管理
/new [名称]      开新会话（不传名称则自动生成 id）
/resume <id>     恢复历史会话
/sessions        列出当前 workspace 的所有会话
/delete <id>     删除指定会话；/delete all 删除全部（删当前会话后自动开新）
/compact         手动压缩当前对话历史
/clear           清空当前会话上下文（仅保留系统提示词，会话 id 不变）
/status          显示上下文占用（真实 tokens / 窗口 + 进度条）、距压缩余量、当前会话 id

每次任务完成的收尾行附带真实用量（Claude Code 式）：
  ✓ 任务完成 · 上下文 930/128k (0.7%) · 本轮 2 次调用 1.1k tokens
所有数字均来自 API 返回的 usage 字段，不使用任何估算。
模型上下文窗口可用 CODING_AGENT_CONTEXT_WINDOW 配置（默认 128000）。

会话文件存于 ~/.coding-agent/sessions/{slug}-{hash}/{session_id}.jsonl
进程退出后可 /resume <id> 恢复历史上下文。

## Git 仓库
https://github.com/HaNZx3/CodeAgent
