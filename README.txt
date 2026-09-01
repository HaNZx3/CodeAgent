# 自研 Coding Agent

从零实现的轻量编程智能体（简化版 Claude Code）：调用大语言模型，自主读写
文件、搜索代码、执行命令与测试，迭代直到完成任务。核心逻辑全部自行实现——
对话历史与上下文管理、工具定义与本地执行、流式输出解析、循环终止条件、
错误恢复、Workspace 安全边界；不用任何 Agent 框架，仅用 OpenAI 兼容 API
与模型原生 tool calling。

## 功能特性
- Agent Loop 闭环：LLM → 工具执行 → 观测回传 → 继续推理，直到最终回答
- 六个本地工具：list_files / read_file / write_file / edit_file /
  search_code / run_command（workspace 沙箱边界，拒绝越界）
- 动态上下文压缩：按模型名自动推断窗口，阈值取窗口 80%；以 API 返回的
  真实 prompt_tokens 判断（不估算），逼近上限时兜底强制压缩
- 会话持久化：消息原子落盘 JSONL，按 workspace 分目录，可 /resume 恢复
- 高危操作确认：删除/覆写前弹 y/N；rm -rf /、format 等灾难级硬拒
- 对话回退 + 代码快照（/back）：影子 git 按轮次记录 workspace，回退对话
  时精确撤销本会话改动（保留其它会话交叉修改），还原前自动安全快照
- 项目记忆：workspace/AGENT.md 与 ~/.coding-agent/USER.md 注入 system prompt
- Claude Code 式体验：/ 弹命令菜单（前缀过滤、↑↓ 选择、Tab 补全），输入栏
  右下角常驻真实用量，流式输出 + spinner；行尾 \ 或 Ctrl+J 多行输入；
  Ctrl+C 即时打断 LLM 调用，返回输入栏重新提问

## 快速开始
1. 安装：pip install -e ".[dev]"（仅运行用 pip install -e .）
2. 配置凭据（只走 .env / 环境变量，不入仓库）：复制 .env.example 为 .env，
   填入 OPENAI_API_KEY（或 DEEPSEEK_API_KEY）；可选 OPENAI_BASE_URL /
   OPENAI_MODEL 切换 DeepSeek、Qwen、GLM 等兼容服务；全局凭据放
   ~/.coding-agent/.env
3. 运行：
   codeagent                                # 交互模式
   codeagent "修复 demo 项目中的 bug 并让测试通过"
   codeagent --workspace ./demo "任务描述"
   （未安装时也可用 python main.py，参数一致）

## 交互命令
/help 命令  /new [名] 新会话  /resume <id> 恢复  /sessions 列会话
/delete <id>|all 删除  /clear 清空  /back [n] 回退对话+代码（y/N 确认）
/compact 手动压缩  /status 占用与余量  /model /workspace /exit
任务完成收尾行附带真实用量（调用次数、本轮 tokens、上下文占比），
均来自 API usage 字段。

## 配置项
见 .env.example：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL /
CODING_AGENT_CONTEXT_WINDOW（按模型名推断：gpt-4o 128k、deepseek 64k、
gpt-4.1 1M 等）/ CODING_AGENT_COMPACT_THRESHOLD / CODING_AGENT_MAX_STEPS /
CODING_AGENT_MAX_RUNTIME。

## 目录结构
main.py CLI 入口（REPL / 菜单 / spinner / 用量显示）   config.py 全局配置
agent/ 循环与状态（loop / context / stop / session / memory / checkpoints）
llm/ 兼容客户端与流式解析   tools/ 工具定义与本地执行（core / workspace
边界 / file / search / shell）   demo/ 演示项目（故意植入 bug）
tests/ 自动化测试（133 用例，pytest tests/）

## 演示
demo/calculator.py 的 multiply 有 bug（+ 应为 *）。执行「修复 demo 项目
中的 bug 并让所有测试通过」，Agent 会自动 list_files → search_code →
read_file → edit_file → run_command(pytest) → 依失败继续修复 → 通过总结。

## Git 仓库
https://github.com/HaNZx3/CodeAgent
