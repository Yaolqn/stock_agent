# LangChain AI 聊天助手

> 基于 [LangChain](https://www.langchain.com/) 的终端多轮 AI 聊天助手。无前端、无窗口，直接在终端对话。
> 支持 **OpenAI 官方 / DeepSeek / Moonshot 等任意 OpenAI 兼容服务**，也支持 **Ollama 本地模型**，并内置无需 API Key 的**离线演示模式**。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [关键执行命令](#关键执行命令)
- [配置说明](#配置说明)
- [接入不同模型提供商](#接入不同模型提供商)
- [使用说明](#使用说明)
- [工作原理](#工作原理)
- [运行测试](#运行测试)
- [安装为系统命令](#安装为系统命令)
- [扩展指南](#扩展指南)
- [常见问题 FAQ](#常见问题-faq)
- [License](#license)

---

## 功能特性

- ✅ **开箱即用**：默认 `LLM_PROVIDER=fake` 离线演示模式，零配置、零 Key 即可运行体验
- ✅ **多轮记忆**：基于 `RunnableWithMessageHistory` 自动维护上下文，支持滑动窗口裁剪
- ✅ **多会话隔离**：不同 `session_id` 的记忆互不干扰，可扩展为多用户/多会话
- ✅ **流式输出**：模型回复逐字打印，体验流畅
- ✅ **多种模型后端**：OpenAI 官方、DeepSeek、Moonshot、Ollama 本地模型，以及任意 OpenAI 兼容网关
- ✅ **终端命令**：`/help` `/clear` `/history` `/model` `/exit` 等常用命令
- ✅ **标准工程化**：包结构、配置中心、依赖锁定、单元测试、打包脚本一应俱全
- ✅ **中文友好**：代码注释、文档、默认系统提示词均为中文
- ✅ **LangGraph 状态机版**：`main_langgraph.py` 用图编排「大脑 ⇄ 工具」循环，另建 `app/agent/` 独立板块，原 LangChain 链路（`main.py`）不受影响
- ✅ **工具调用**：内置查天气工具（`app/tools/`），模型自主决定何时调用；新增工具只需加一个文件并登记一次
- ✅ **状态机可视化**：`python -m app.agent.visualize` 一键输出状态机 ASCII 图 / Mermaid 图

## 技术栈

| 组件 | 技术 |
| --- | --- |
| 语言 | Python ≥ 3.10 |
| 框架 | LangChain 0.3.x（langchain-core / langchain-openai） |
| 状态机 | LangGraph 0.6.x（app/agent/，独立板块） |
| 配置 | pydantic-settings + python-dotenv |
| 测试 | pytest |
| 打包 | hatchling（pyproject.toml） |

## 目录结构

```
langchain-ai-chatbot/
├── main.py                  # 程序入口①：LangChain 手写 Agent 循环版（python main.py）
├── main_langgraph.py        # 程序入口②：LangGraph 状态机版（python main_langgraph.py，推荐）
├── requirements.txt         # 运行时依赖
├── requirements-dev.txt     # 开发依赖（测试 / lint）
├── pyproject.toml           # 项目元数据与打包配置（pip install -e .）
├── .env.example             # 环境变量模板（复制为 .env 后按需修改）
├── .env                     # 实际环境配置（已被 .gitignore 排除，含密钥）
├── .gitignore
├── LICENSE                  # MIT 许可证
├── README.md                # 本文档
├── app/                     # 应用源码包（两版入口共用配置 / 记忆 / 提示词）
│   ├── __init__.py
│   ├── config.py            # 配置中心：从环境变量 / .env 加载全部配置
│   ├── models.py            # 模型工厂：按配置创建 ChatOpenAI / 演示模型
│   ├── prompts.py           # Prompt 模板：系统提示词 + 历史占位符
│   ├── memory.py            # 会话记忆：多会话隔离 + 滑动窗口裁剪
│   ├── chain.py             # ①LangChain 版核心：手写 Agent 循环 + RunnableWithMessageHistory
│   ├── cli.py               # 终端交互：REPL 循环、命令解析、流式输出（①使用）
│   ├── agent/               # ②LangGraph 版 Agent 包（独立板块）
│   │   ├── __init__.py
│   │   ├── state.py         #   状态定义：AgentState（TypedDict + add_messages）
│   │   ├── agent.py         #   图定义、节点实现、对外接口（LangGraphAgent / build_lg_agent）
│   │   └── visualize.py     #   状态机可视化：ASCII 图 / Mermaid 代码
│   ├── tools/               # 工具包：新增工具在此加文件并登记 __init__.py
│   │   ├── __init__.py      #   工具登记与聚合（_TOOL_MODULES / get_available_tools）
│   │   └── weather.py       #   查天气工具（uapis.cn API）
│   └── py.typed             # PEP 561 类型标记
└── tests/
    ├── test_chain.py        # ①LangChain 版离线单元测试
    └── test_lg_agent.py     # ②LangGraph 版离线单元测试
```

## 快速开始

### 1. 克隆 / 进入项目

```bash
cd langchain-ai-chatbot
```

### 2. 创建并激活虚拟环境

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 4. 运行（默认离线演示模式，无需任何 Key）

两个入口任选其一（共用同一份 `.env` 配置）：

**① LangGraph 状态机版（推荐，支持工具调用）**

```bash
python main_langgraph.py
```

**② 原 LangChain 手写 Agent 循环版**

```bash
python main.py
```

启动后直接在终端输入消息即可对话：

```
你 > 你好
AI > （演示模式）你好！我是离线模拟回复。设置 LLM_PROVIDER=openai 并填入 API Key 后即可接入真实模型。
```

### 5.（可选）接入真实模型

复制配置模板并填写 Key：

```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

编辑 `.env`，将 `LLM_PROVIDER` 改为 `openai`，填入 `OPENAI_API_KEY`，重启程序即可。

> 项目已自带一份 `.env`（默认 `LLM_PROVIDER=fake`），如果你不需要演示模式，直接编辑它即可，无需复制。

## 关键执行命令

| 命令 | 作用 |
| --- | --- |
| `python main.py` | 启动 **① LangChain 手写 Agent 循环版**（原版入口） |
| `python main_langgraph.py` | 启动 **② LangGraph 状态机版**（推荐；真实模型下自动启用查天气等工具） |
| `python -m app.agent.visualize` | **可视化状态机**：终端输出 ASCII 图 + Mermaid 代码（Mermaid 可粘贴到 `mermaid.live` 渲染成图） |
| `python -m pytest -q` | 运行全部离线单元测试（11 个用例，覆盖 ①② 两版） |
| `python -m pip install -e .` | 安装为系统命令（安装后任意目录执行 `ai-chat` 即可启动 ①） |

> 说明：① 与 ② 共用 `app/config.py`、`app/models.py`、`app/memory.py`、`app/prompts.py`，切换入口无需改动任何配置。

## 配置说明

所有配置项集中在 `.env` 文件（或系统环境变量）中，优先级：**系统环境变量 > .env 文件 > 代码默认值**。

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `LLM_PROVIDER` | 模型提供商：`openai` / `ollama` / `fake` | `fake` |
| `OPENAI_API_KEY` | OpenAI 兼容服务的 API Key | 空 |
| `OPENAI_BASE_URL` | 接口地址（DeepSeek 填 `https://api.deepseek.com/v1`） | OpenAI 官方 |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | Ollama 的 OpenAI 兼容端点 | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Ollama 本地模型名 | `qwen2.5:7b` |
| `TEMPERATURE` | 采样温度（0~2，越高越发散） | `0.7` |
| `MAX_TOKENS` | 单次回复最大 token 数 | `2048` |
| `MAX_HISTORY_MESSAGES` | 单会话记忆窗口（超出裁剪最旧消息） | `20` |
| `SESSION_ID` | 会话 ID（用于隔离记忆） | `default` |
| `SYSTEM_PROMPT` | 系统提示词（设定助手角色） | 内置中文助手 |

## 接入不同模型提供商

### OpenAI 官方

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### DeepSeek（国内直连，无需代理）

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-你的DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

### Moonshot / Kimi

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-你的MoonshotKey
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_MODEL=moonshot-v1-8k
```

### Ollama 本地模型（免费、离线）

先安装并启动 [Ollama](https://ollama.com/)，拉取模型：

```bash
ollama serve
ollama pull qwen2.5:7b
```

再配置：

```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b
```

> Ollama 通过其 OpenAI 兼容端点暴露 API，因此实现上复用了 `ChatOpenAI`，无需额外依赖。

### 离线演示模式（无需 Key）

```ini
LLM_PROVIDER=fake
```

使用 `GenericFakeChatModel` 循环返回固定回复，用于无网络、无 Key 时验证完整链路。**默认启用**。

## 使用说明

程序支持以下命令：

| 命令 | 作用 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 清空当前会话记忆（重新开始上下文） |
| `/history` | 查看当前会话的历史消息 |
| `/model` | 查看当前模型配置 |
| `/exit` 或 `/quit` | 退出程序（或按 `Ctrl+C` / `Ctrl+D`） |

示例会话（演示模式）：

```
你 > /model
provider=fake（离线演示模式，不调用真实模型）

你 > 今天天气怎么样？
AI > （演示模式）这是第二条模拟回复。你可以试试 /clear 清空记忆、/history 查看历史。

你 > /history
[用户] 今天天气怎么样？
[AI] （演示模式）这是第二条模拟回复。你可以试试 /clear 清空记忆、/history 查看历史。

你 > /exit
再见！
```

## 工作原理

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────┐
│ app/cli.py   终端交互循环（命令解析 / 流式打印） │
└───────────────┬──────────────────────────────┘
                │  {"input": "..."}
                ▼
┌──────────────────────────────────────────────┐
│ app/chain.py  RunnableWithMessageHistory      │
│               ① 按 session_id 取出历史消息     │
│               ② 注入 prompt 的 history 占位符  │
│               ③ 调用后把本轮对话写回记忆        │
└───────────────┬──────────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌───────────────┐  ┌──────────────┐
│ app/prompts.py│  │ app/memory.py │
│ ChatPrompt   │  │ 滑动窗口记忆   │
└───────┬───────┘  └──────┬───────┘
        ▼                 │
┌───────────────┐         │
│ app/models.py │         │
│ 模型工厂       │         │
└───────┬───────┘         │
        │                 │
        ▼                 ▼
  LLM 服务         历史消息读写
 (OpenAI/DeepSeek  (进程内存储，
  /Ollama/fake)    重启即清空)
```

核心链路（LCEL）：

```python
runnable = prompt | llm                      # 提示模板 → 大模型
chain = RunnableWithMessageHistory(          # 包一层多轮记忆
    runnable,
    memory.get_session_history,              # 记忆回调
    input_messages_key="input",              # 用户输入字段
    history_messages_key="history",          # 历史注入占位符
)
```

### LangGraph 版工作原理（状态机）

入口 `main_langgraph.py`：把「当前全部消息」交给图，由 `StateGraph` 编排「大脑 ⇄ 手脚」循环，直至模型直接作答：

```
 START
   │
   ▼
─────────────────────────────────
 call_model（大脑：模型作答 / 提名工具）
─────────────────────────────────
   │
   ▼ 路由判断
 ┌────────┴────────┐
有 tool_calls？是   否
   │                │
   ▼                ▼
───────────     ─────────
 action      →   __end__
（执行工具）        │
   │           最终回复写回记忆
   └──► 回到 call_model（循环，有递归上限）
```

状态（`AgentState`）与节点（`call_model` / `action`）定义在 `app/agent/`：
`state.py` 声明状态画板，`agent.py` 声明图结构、路由函数与对外接口 `ask()`。

运行时每一步的拓扑可用以下命令可视化（输出 ASCII 图 + Mermaid 代码，Mermaid 代码可粘贴到 [mermaid.live](https://mermaid.live) 或 VS Code「Markdown Preview Mermaid Support」插件中渲染成图）：

```bash
python -m app.agent.visualize
```

## 运行测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试全部离线运行（使用假模型），共 11 个用例，同时覆盖两版：
- **① LangChain 版**（`test_chain.py`）：Prompt 结构、多轮记忆写入、窗口裁剪、会话隔离、记忆清空
- **② LangGraph 版**（`test_lg_agent.py`）：状态机两轮对话写历史、会话隔离、执行天气工具、fake 模型回退

## 安装为系统命令

```bash
python -m pip install -e .
ai-chat   # 任意目录下直接启动
```

## 扩展指南

- **接入更多模型**：在 `app/models.py` 的 `create_llm` 中新增分支即可，上层代码零改动。
- **新增工具（Function Calling）**：在 `app/tools/` 下新建工具文件（参考 `weather.py`，用 `@tool` 装饰器定义、模块内实现 `get_tools()`），再到 `app/tools/__init__.py` 的 `_TOOL_MODULES` 登记即可；LangGraph 版（`build_lg_agent`）会自动绑定给模型。
- **检索增强生成（RAG）**：引入 `langchain-community` + 向量库，把检索结果拼入 Prompt 或使用 `create_retrieval_chain`。
- **记忆持久化**：将 `MemoryManager` 内的存储实现替换为 LangChain 官方的 Redis / SQLite / Postgres 历史类（只需实现 `BaseChatMessageHistory` 接口）。
- **Web 界面**：可基于 `chain` 直接包装 FastAPI / Streamlit / LangServe，一行暴露 REST 或 WebSocket 接口。
- **代码规范**：`python -m ruff check .` 可进行 Lint 检查。

## 常见问题 FAQ

**Q1：报错 `AuthenticationError` / 401？**
A：API Key 错误或 `OPENAI_BASE_URL` 与服务商不匹配。用 `/model` 查看当前配置，并核对 Key 与官方文档的接口地址。

**Q2：网络超时 / 无法连接？**
A：国内访问 OpenAI 官方接口通常需要代理；建议改用 DeepSeek / Moonshot 等国内服务，或在系统层面配置代理。

**Q3：Ollama 模式报连接错误？**
A：确认 `ollama serve` 已启动、模型已 `ollama pull`，且 `OLLAMA_BASE_URL` 端口为 11434。

**Q4：终端中文乱码？**
A：新版 Windows Terminal / PowerShell 7 / VS Code 终端一般无此问题；旧版 cmd 可先执行 `chcp 65001`。程序内部已做 UTF-8 输出适配。

**Q5：为什么模型不记得之前的对话？**
A：先执行 `/history` 查看记忆是否存在。若存在，可能是 `MAX_HISTORY_MESSAGES` 太小导致上下文被裁剪；若为空，请确认两次对话使用同一个 `SESSION_ID`。

**Q6：能否升级到 LangChain 1.x？**
A：可以。代码已兼容 1.x 的公开 API（`ChatOpenAI`、`RunnableWithMessageHistory`、`GenericFakeChatModel`），去掉 requirements 与 pyproject 中的版本上限后安装最新版即可，如有兼容问题欢迎提 issue。

## License

[MIT](LICENSE) © 2025 langchain-ai-chatbot
