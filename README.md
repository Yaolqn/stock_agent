# LangChain AI 聊天助手

> 基于 [LangChain](https://www.langchain.com/) 1.x + [LangGraph](https://www.langchain.com/langgraph) 的终端多轮 AI 聊天助手。无前端、无窗口，直接在终端对话。
> 支持 **OpenAI 官方 / DeepSeek / Moonshot 等任意 OpenAI 兼容服务**，也支持 **Ollama 本地模型**，并内置无需 API Key 的**离线演示模式**。
> 项目包含三大子系统：**app**（对话 Agent）、**stock_agent**（多 Agent 股票投研）、**rag**（文档检索增强问答）。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
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
- ✅ **多轮记忆**：基于 LangGraph 状态机 + `MemoryManager` 维护上下文，支持滑动窗口裁剪、LLM 记忆压缩与 SQLite 持久化（重启不丢）
- ✅ **多会话隔离**：不同 `session_id` 的记忆互不干扰，可扩展为多用户/多会话
- ✅ **流式输出**：模型回复逐字打印，体验流畅
- ✅ **多种模型后端**：OpenAI 官方、DeepSeek、Moonshot、Ollama 本地模型，以及任意 OpenAI 兼容网关
- ✅ **终端命令**：`/help` `/clear` `/history` `/model` `/exit` 等常用命令
- ✅ **标准工程化**：包结构、配置中心、依赖锁定、单元测试、打包脚本一应俱全
- ✅ **中文友好**：代码注释、文档、默认系统提示词均为中文
- ✅ **LangGraph 状态机 Agent**：全项目对话统一由 `app/agent`（大脑 call_model ⇄ 手脚 action）驱动，天然支持并行工具调用（Send API）、递归上限与工具失败自愈
- ✅ **工具调用**：内置查天气（`app/tools/weather.py`）、A 股行情/扫描/深度分析（`app/tools/stock.py`），模型自主决定何时调用；新增工具只需加一个文件并登记一次
- ✅ **多 Agent 投研**：`stock_agent` 总控协调官调度五位专家（选股师/分析师/情报官/决策官/风险官），另有 `/report` 完整投研流水线
- ✅ **RAG 问答**：`rag` 子系统支持文档上传（PDF/DOCX/TXT/MD）→ 向量化入库（Milvus）→ 检索增强问答（重排 + 可选 BM25 混合）
- ✅ **状态机可视化**：`python -m app.agent.visualize` 一键输出状态机 ASCII 图 / Mermaid 图

## 技术栈

| 组件 | 技术 |
| --- | --- |
| 语言 | Python ≥ 3.10 |
| 框架 | LangChain 1.x（langchain-core / langchain-openai / langchain-classic） |
| 状态机 | LangGraph 1.x（app/agent/，全项目对话统一驱动） |
| 配置 | pydantic-settings + python-dotenv |
| 记忆 | SQLite（可选持久化）+ 滑动窗口裁剪 + LLM 压缩 |
| 向量库 | Milvus（langchain-milvus + pymilvus） |
| 测试 | pytest |
| 打包 | hatchling（pyproject.toml） |

## 目录结构

```
langchain-ai-chatbot/
├── main.py                  # 程序入口（python main.py，LangGraph 状态机 Agent）
├── requirements.txt         # 运行时依赖
├── requirements-dev.txt     # 开发依赖（测试 / lint）
├── pyproject.toml           # 项目元数据与打包配置（pip install -e .）
├── .env.example             # 环境变量模板（复制为 .env 后按需修改）
├── .env                     # 实际环境配置（已被 .gitignore 排除，含密钥）
├── .gitignore
├── LICENSE                  # MIT 许可证
├── README.md                # 本文档
├── app/                     # 应用源码包（对话 Agent）
│   ├── __init__.py
│   ├── config.py            # 配置中心：从环境变量 / .env 加载全部配置
│   ├── models.py            # 模型工厂：按配置创建 ChatOpenAI / 演示模型
│   ├── prompts.py           # Prompt 模板：记忆压缩提示词
│   ├── memory.py            # 会话记忆：多会话隔离 + 滑动窗口裁剪 + 可选 SQLite 持久化
│   ├── cli.py               # 终端交互：REPL 循环、命令解析、流式输出
│   ├── agent/               # LangGraph 状态机 Agent 包（全项目对话统一核心）
│   │   ├── __init__.py
│   │   ├── state.py         #   状态定义：AgentState（TypedDict + add_messages + retries）
│   │   ├── agent.py         #   图定义、节点实现、对外接口（LangGraphAgent / build_lg_agent）
│   │   └── visualize.py     #   状态机可视化：ASCII 图 / Mermaid 代码
│   └── tools/               # 工具包：新增工具在此加文件并登记 __init__.py
│       ├── __init__.py      #   工具登记与聚合（_TOOL_MODULES / get_available_tools）
│       ├── weather.py       #   查天气工具（uapis.cn API）
│       └── stock.py         #   股票工具（akshare：行情 / K线 / 扫描 / 深度分析 / 资金流 / 新闻）
├── stock_agent/             # 多 Agent 股票投研子系统（python -m stock_agent.main）
│   ├── supervisor.py        #   总控协调官（只调度决策，绝不直接取数）+ wrap_expert_as_tool
│   ├── experts.py           #   五位专家工厂（选股师 / 分析师 / 情报官 / 决策官 / 风险官）
│   ├── pipeline.py          #   投研流水线（StateGraph：筛选→分析→决策→风控）
│   ├── main.py              #   终端入口（灵活问答 + /report 双模式）
│   └── DESIGN.md            #   设计文档
├── rag/                     # RAG 子系统（python -m rag.main）
│   ├── config.py            #   RAG 配置（嵌入 / Redis / Milvus / 分块 / 检索 / 重试）
│   ├── embeddings.py        #   嵌入模型工厂 + Redis 缓存（langchain_classic.CacheBackedEmbeddings）
│   ├── document.py          #   文档提取与语义分块
│   ├── vectorstore.py       #   Milvus 向量库封装
│   ├── retrieval.py         #   向量检索 + 关键词/语义重排 + BM25 混合检索
│   ├── prompts.py           #   RAG 提示词模板
│   ├── service.py           #   RAG 问答服务（上传 / 问答 / 管理）
│   ├── retry.py             #   指数退避重试
│   └── main.py              #   终端 CLI 入口
└── tests/
    ├── test_lg_agent.py     # LangGraph Agent 离线测试
    ├── test_memory_persistence.py  # SQLite 记忆持久化测试
    ├── test_stock_agent.py  # 多 Agent / 流水线离线测试
    └── test_rag.py          # RAG 分块 / 检索 / 重试离线测试
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

```bash
python main.py
```

启动后直接在终端输入消息即可对话：

```
你 > 你好
AI > （演示模式）你好！我是离线模拟回复。设置 LLM_PROVIDER=openai 并填入 API Key 后即可接入真实模型。
```

其它入口：
- **多 Agent 股票投研**：`python -m stock_agent.main`（灵活问答 + `/report` 完整投研流水线）
- **RAG 文档问答**：`python -m rag.main`（需先启动 Milvus / 配置嵌入模型）

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
| `python main.py` | 启动对话 Agent（LangGraph 状态机；真实模型下自动启用天气 / 股票等工具） |
| `python -m stock_agent.main` | 启动多 Agent 股票投研（总控协调官 + `/report` 流水线） |
| `python -m rag.main` | 启动 RAG 文档问答（上传文档 → 向量化入库 → 检索增强问答） |
| `python -m app.agent.visualize` | **可视化状态机**：终端输出 ASCII 图 + Mermaid 代码（Mermaid 可粘贴到 `mermaid.live` 渲染成图） |
| `python -m pytest -q` | 运行全部离线单元测试（覆盖 app / stock_agent / rag） |
| `python -m pip install -e .` | 安装为系统命令（安装后任意目录执行 `ai-chat` 即可启动） |

> 三个子系统共用 `app/config.py`、`app/models.py`、`app/memory.py`、`app/prompts.py`，无需重复配置。

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
| `MEMORY_DB_PATH` | 记忆持久化 SQLite 文件路径（留空 = 纯内存，重启即失） | 空 |
| `SESSION_ID` | 会话 ID（用于隔离记忆） | `default` |
| `SYSTEM_PROMPT` | 系统提示词（设定助手角色） | 内置中文助手 |

> 更多 RAG 配置（嵌入模型 / Redis / Milvus / 分块 / 检索 / 重试）见 `.env.example` 中 `RAG 知识库` 一节。

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

全项目对话统一由 **LangGraph 状态机 Agent**（`app/agent`）驱动：

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────┐
│ app/cli.py   终端交互循环（命令解析 / 流式打印） │
└───────────────┬──────────────────────────────┘
                │  ask(user_input, session_id)
                ▼
┌──────────────────────────────────────────────┐
│ app/agent/agent.py   LangGraphAgent（状态图） │
│   START → call_model ⇄ action → END          │
└───────┬──────────────────────┬───────────────┘
        │                      │
        ▼                      ▼
┌───────────────┐      ┌──────────────┐
│ app/memory.py │      │ app/models.py│
│ 会话记忆       │      │ 模型工厂      │
│ (窗口/压缩/    │      └──────┬───────┘
│  SQLite持久化) │             ▼
└───────────────┘        LLM 服务
                     (OpenAI/DeepSeek
                      /Ollama/fake)
```

状态（`AgentState`）与节点（`call_model` / `action`）定义在 `app/agent/`：
`state.py` 声明状态画板，`agent.py` 声明图结构、路由函数与对外接口 `ask()` / `ask_stream()`。

每次对话流程：
1. 从记忆取出会话历史，连同系统提示词、本轮输入拼成初始消息；
2. 图进入 `call_model`（大脑）：模型作答，若返回 `tool_calls` 则用 **Send API** 并行扇出 `action` 分支（每个工具调用一条分支）；
3. `action` 分支各自执行一个工具，结果追加为 `ToolMessage` 并自动合并回 `call_model`；
4. 直至模型直接回答（无工具调用）→ END；最终回复写回记忆。工具失败有自愈重试上限（`_RETRY_LIMIT`），超限自动收场，避免死循环。

运行时每一步的拓扑可用以下命令可视化（输出 ASCII 图 + Mermaid 代码，Mermaid 代码可粘贴到 [mermaid.live](https://mermaid.live) 或 VS Code「Markdown Preview Mermaid Support」插件中渲染成图）：

```bash
python -m app.agent.visualize
```

### 多 Agent 投研（stock_agent）

总控协调官（Supervisor）只绑定专家工具、负责调度与决策，绝不直接取数；所有数据获取必须由专家（选股师 / 分析师 / 情报官 / 决策官 / 风险官）中转完成。`/report` 走 LangGraph 状态图流水线：`screener → (analyst ∥ intel) → decision → risk`，各阶段产物写入共享状态字段，层层递进。

### RAG 问答（rag）

`python -m rag.main`：上传文档 → 提取文本 → 语义分块 → 嵌入（可走 Redis 缓存）→ 写入 Milvus → 检索相关块（可选关键词/语义重排、BM25 混合）→ 拼接上下文 → 聊天模型作答（带来源标注）。

## 运行测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试全部离线运行（使用假模型 / monkeypatch，不访问网络），共 19 个用例：
- **`test_lg_agent.py`**：状态机两轮对话写历史、会话隔离、执行天气工具、fake 模型回退
- **`test_memory_persistence.py`**：SQLite 持久化（重启恢复、clear 删除、纯内存回退、窗口裁剪、会话隔离）
- **`test_stock_agent.py`**：专家工具封装、五位专家组装、总控回退、流水线完整跑通、显式代码跳过选股
- **`test_rag.py`**：文档分块、块元数据、关键词重排、BM25 检索、重试判定

## 安装为系统命令

```bash
python -m pip install -e .
ai-chat   # 任意目录下直接启动
```

## 扩展指南

- **接入更多模型**：在 `app/models.py` 的 `create_llm` 中新增分支即可，上层代码零改动。
- **新增工具（Function Calling）**：在 `app/tools/` 下新建工具文件（参考 `weather.py` / `stock.py`，用 `@tool` 装饰器定义、模块内实现 `get_tools()`），再到 `app/tools/__init__.py` 的 `_TOOL_MODULES` 登记即可；`build_lg_agent` 会自动绑定给模型。
- **新增专家 Agent**：在 `stock_agent/experts.py` 增加 `build_xxx` 工厂并加入 `build_expert_tools()`；Supervisor 与流水线会自动获得该能力。
- **记忆持久化**：配置 `MEMORY_DB_PATH=chat_history.db` 即可启用 SQLite 持久化（重启不丢）；实现沿用 `BaseChatMessageHistory` 接口，替换存储实现不影响上层。
- **Web 界面**：可基于 `LangGraphAgent` 直接包装 FastAPI / Streamlit，一行暴露 REST 或 WebSocket 接口。
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
A：先执行 `/history` 查看记忆是否存在。若存在，可能是 `MAX_HISTORY_MESSAGES` 太小导致上下文被裁剪；若为空，请确认两次对话使用同一个 `SESSION_ID`。另外注意：配置了 `MEMORY_DB_PATH` 后重启不丢；未配置则为纯内存，重启即清空。

**Q6：项目使用什么版本？**
A：LangChain 1.x（langchain / langchain-core / langchain-openai，已弃用的 API 由 langchain-classic 提供）+ LangGraph 1.x。对话统一由 LangGraph 状态机 Agent 驱动；1.x 中弃用的 `RunnableWithMessageHistory` 与手写 LCEL 工具循环（`app/chain.py`）已随升级移除。

## License

[MIT](LICENSE) © 2025 langchain-ai-chatbot
