# 自研 Coding Agent

一个从零实现的轻量级编程智能体（简化版 Claude Code）：调用大语言模型，
自主读写本地文件、搜索代码、执行命令与测试，并根据执行结果迭代直到完成任务。

核心 Agent 逻辑全部自行实现——对话历史与上下文管理、工具的定义与本地执行、
模型输出（含流式分片）的解析、循环终止条件、错误恢复、Workspace 安全边界，
不依赖 LangChain / LlamaIndex / OpenAI Agents SDK / AutoGen / CrewAI 等
任何 Agent 框架或 SDK，仅使用 OpenAI 兼容 API 客户端与模型原生 tool calling。

## 功能特性
- Agent Loop 闭环：LLM -> 工具执行 -> 观测回传 -> 继续推理，直到给出最终回答
- 六个本地工具：list_files / read_file / write_file / edit_file / search_code / run_command
  （workspace 沙箱边界，拒绝越界访问）
- 上下文自动压缩：真实 prompt_tokens 超阈值时把旧轮次摘要成一条 system 消息，
  保留最近 N 轮原文；按 user 轮次切分，保证 tool_call/tool 消息配对完整
- 会话持久化与隔离：每条消息同步落盘 JSONL，按 workspace 哈希分目录，
  进程退出后可 /resume 恢复；原子重写（.tmp + os.replace）防崩溃损坏
- 对话回退 + 代码快照（/back）：影子 git 仓库按轮次记录 workspace 状态，
  回退对话时精确撤销本会话的代码改动（其它会话的交叉修改原样保留）；
  还原前自动安全快照，还原本身永远可撤销
- 项目记忆：workspace/AGENT.md 与 ~/.coding-agent/USER.md 自动注入 system prompt
- Claude Code 式用量显示：全部取自 API 返回的 usage 字段，无任何估算
- Claude Code 式输入体验：输入 / 弹出命令菜单（前缀过滤、精确匹配置顶、
  ↑/↓ 选择、Tab/Enter 补全），输入栏右下角常驻真实用量；流式打字机输出、
  工具执行 spinner

## 快速开始
1. 安装（可编辑安装会同时装好依赖，并注册 codeagent 全局命令）：
   pip install -e ".[dev]"     # 含 pytest；只要运行依赖则 pip install -e .
2. 配置凭据（凭据只走 .env / 环境变量，不进入仓库）：
   方式 A（推荐）：复制 .env.example 为 .env，填入 OPENAI_API_KEY
   方式 B：export OPENAI_API_KEY=...   # 或 DEEPSEEK_API_KEY
   可选：OPENAI_BASE_URL / OPENAI_MODEL 切换 DeepSeek、Qwen、GLM 等兼容服务
   全局凭据建议放 ~/.coding-agent/.env（Windows: %USERPROFILE%\.coding-agent\.env），
   任意目录启动都能读到；项目根 .env 可按项目覆盖
3. 运行：
   codeagent                                        # 任意目录直接启动交互模式
   codeagent "修复 demo 项目中的 bug 并让所有测试通过"
   codeagent --workspace ./demo "任务描述"          # 指定工作目录
   （未安装时也可在项目目录内用 python main.py，参数完全一致）

## 设计要点（核心逻辑说明，各文件 docstring 有完整动机）
1. 对话历史与上下文管理（agent/context.py）
   维护发给模型的 messages 列表；工具输出按「前 6KB + 标记 + 后 2KB」截断；
   压缩判断用上次 API 返回的真实 prompt_tokens（不估算），摘要消息带幂等前缀防再压缩。
2. 工具定义与本地执行（tools/）
   Tool 基类用 JSON Schema 声明参数并自动生成 tool 定义；registry 统一分发；
   工具失败不终止 Agent，而是把错误写回上下文让模型自行恢复。
3. 模型输出解析（llm/client.py）
   把各厂商响应标准化为 ModelResponse / ToolCall；流式模式下 tool_call 参数
   分片按 index 拼槽，上层无感；usage 从最后一个 chunk 提取真实值。
4. 循环终止条件（agent/stop.py）
   四重停止：模型最终回答 / 最大步数 / 最大运行时间 / 连续工具失败；
   压缩发生在计时开始前，summarizer 耗时不占 max_runtime。
5. 错误恢复
   LLM 异常作为停止原因返回而非崩溃；JSONL 损坏行跳过不影响整会话。

## 交互命令
/help            显示可用命令
/new [名称]      开新会话（不传名称则自动生成 id）
/resume <id>     恢复历史会话
/sessions        列出当前 workspace 的所有会话
/delete <id>     删除指定会话；/delete all 删除全部（需确认，删当前会话后自动开新）
/clear           清空当前会话上下文（仅保留系统提示词，会话 id 不变）
/back [n]        回退到第 n 条用户消息之前（无参数时列候选菜单）；
                 精确回退本会话的代码改动（同文件交叉时询问是否全量回退），
                 需 y/N 确认
/compact         手动压缩当前对话历史
/status          上下文占用（真实 tokens / 窗口 + 进度条）、距压缩余量、当前会话
/model           显示当前模型
/workspace       显示当前工作目录
/exit            退出

每次任务完成的收尾行附带真实用量：
  ✓ 任务完成 · 上下文 930/128k (0.7%) · 本轮 2 次调用 1.1k tokens
输入栏右下角同时常驻显示当前上下文占用与本会话累计 tokens（无任务前不显示）。
所有数字均来自 API 返回的 usage 字段。窗口大小用
CODING_AGENT_CONTEXT_WINDOW 配置（默认 128000，仅用于显示，不参与压缩判断）。

## 配置项（环境变量 / .env）
OPENAI_API_KEY                  API Key（或 DEEPSEEK_API_KEY）
OPENAI_BASE_URL                 OpenAI 兼容网关地址
OPENAI_MODEL                    模型名（默认 gpt-4o-mini）
CODING_AGENT_WORKSPACE          工作目录（默认绑定启动时的当前目录）
CODING_AGENT_MAX_STEPS          单任务最大步数（默认 20）
CODING_AGENT_MAX_RUNTIME        单任务最大运行秒数（默认 300）
CODING_AGENT_MAX_ERRORS         最大连续工具失败次数（默认 3）
CODING_AGENT_MAX_TOOL_OUTPUT    单条工具输出上限（默认 8192）
CODING_AGENT_COMMAND_TIMEOUT    命令超时秒数（默认 30）
CODING_AGENT_COMPACT_THRESHOLD  自动压缩阈值（默认 80000，按真实 prompt_tokens 判断）
CODING_AGENT_KEEP_RECENT        压缩时保留最近轮数（默认 6）
CODING_AGENT_CONTEXT_WINDOW     上下文窗口大小，仅用于占用显示（默认 128000）
CODING_AGENT_SESSION_ROOT       会话文件根目录（默认 ~/.coding-agent/sessions）
CODING_AGENT_CHECKPOINTS        代码快照开关（默认 1 开启；git 缺失时自动降级）
CODING_AGENT_CHECKPOINT_ROOT    快照根目录（默认 ~/.coding-agent/checkpoints）

## 目录结构
agent/   Agent 状态与循环（loop / context / stop / session / memory / checkpoints）
llm/     模型 API 通信与响应标准化
tools/   工具定义与本地执行（file / search / shell / workspace 边界）
demo/    演示用小型 Python 项目（含故意植入的 bug）
tests/   自动化测试（tool / context / agent loop / session / memory / checkpoint / 命令层）

## 运行测试
pytest tests/

## 演示
demo/calculator.py 的 multiply 函数有一个 bug（+ 应为 *）。
让 Agent 执行「修复 demo 项目中的 bug 并让所有测试通过」，它会自动：
list_files -> search_code -> read_file -> edit_file -> run_command(pytest)
-> 根据失败继续修复 -> 通过 -> 总结。

## 快照与回退（/back）
每轮任务开始前，Agent 把 workspace 全量状态 commit 到一个影子 git 仓库
（~/.coding-agent/checkpoints/<workspace-slug>/，通过 --git-dir/--work-tree
指向你的目录，绝不触碰你自己的 .git）。快照与对话轮次一一对应：
第 k 个快照 = 第 k 条用户消息发出前的代码状态（账本持久化，跨进程可回退）。

/back 列出历史消息，选中某条后：对话截断到该消息之前，代码精确回退本会话
在该快照之后的改动——影子仓库中每个快照 commit 相对其父提交的 diff 恰好
就是该会话当时做的改动，把这些 commit 逆序反向 apply 即只撤销自己的修改，
其它会话的交叉修改原样保留；仅当与其它会话改了同一文件时才询问是否降级
为全量回退（git reset --hard，会正确删除快照后新增的文件）。

安全设计：
- .env* 等凭据文件被内置 exclude 永久排除，不进任何快照、还原也不触碰
- 还原前自动打「安全快照」——任何还原本身都可撤销，误回退零丢失；
  安全快照 hash 会打印出来，找回方式（手动执行）：
  git --git-dir ~/.coding-agent/checkpoints/<slug>/.git \
      --work-tree <workspace> reset --hard <safety_hash>
- 同一 workspace 多会话交叉修改时，回退前列出其它会话在目标快照之后
  改过的文件：精确回退保留这些改动（同文件交叉除外），全量回退才覆盖
- git 缺失或命令失败：快照自动降级，/back 仅回退对话，不影响正常使用
- 被压缩摘要掉的轮次无法回退（Claude Code 同样如此）

## 项目记忆（可选）
在 workspace 根目录放 AGENT.md，写入项目特定指引（构建命令、测试方式、
代码约定、已知坑点），Agent 启动时自动注入 system prompt。
在 ~/.coding-agent/USER.md 写入跨项目的个人偏好（编辑器、语言习惯）。

示例 AGENT.md：
  # 项目记忆
  - 测试：pytest tests/
  - 包管理：uv
  - 约定：所有函数加类型注解

## Git 仓库
https://github.com/HaNZx3/CodeAgent
