# LangChain 1.x 升级 —— 更新前后功能对比清单

> 升级目标：LangChain 0.3.x → 1.x，重构 app / stock_agent / rag 三大子系统，
> 移除不再需要的内容，保证运行正常且运行效果与升级前一致。
>
> 当前版本：langchain 1.4.0 · langchain-core 1.6.3 · langgraph 1.2.11 ·
> langchain-milvus 0.4.0 · pymilvus 3.0.1 · langchain-openai 1.6.2 ·
> langchain-community 0.4.2 · langchain-classic 1.0.8

---

## 1. 依赖与框架版本对比

| 组件 | 更新前 | 更新后 | 说明 |
| --- | --- | --- | --- |
| langchain | 0.3.x | 1.4.0（`>=1.4.0,<2.0.0`） | 主框架大版本升级 |
| langchain-core | 0.3.x | 1.6.3 | 核心抽象层随大版本升级 |
| langchain-classic | 无 | 1.0.8 | **新增**：1.x 中被移除的旧 API（如 `CacheBackedEmbeddings`）由此包提供 |
| langchain-openai | 0.3.x | 1.6.2 | OpenAI 兼容接口 |
| langchain-community | 0.3.x | 0.4.2 | 社区集成（Redis 存储等）随 1.x 升级 |
| langgraph | 0.2.x | 1.2.11 | 状态机框架大版本升级 |
| langchain-milvus | 0.1.x | 0.4.0 | 向量库集成适配 langchain-core 1.x |
| pymilvus | <2.6 | 3.0.1（`>=3.0.0,<4.0.0`） | 底层客户端升级为 MilvusClient API |
| pydantic-settings | 2.x | 2.15.0 | 配置加载，无破坏性变更 |

---

## 2. 对话系统（app）功能对比

| 功能项 | 更新前 | 更新后 | 效果变化 |
| --- | --- | --- | --- |
| 对话实现 | **两套并行**：LCEL 手写工具循环（`app/chain.py` + `main.py`）与 LangGraph 状态机（`app/agent` + `main_langgraph.py`） | **统一为 LangGraph 状态机 Agent**（`app/agent`），全项目（含 stock_agent）共用一份实现 | 同一入口，架构收敛 |
| 程序入口 | `main.py` 与 `main_langgraph.py` 双入口 | 仅 `main.py`（`python main.py`） | 入口唯一 |
| 记忆注入 | `RunnableWithMessageHistory`（1.x 已弃用） | Agent 内部直接从 `MemoryManager` 读写会话历史 | 功能不变 |
| 多轮记忆 | 滑动窗口裁剪 | 滑动窗口 + LLM 记忆压缩（<300 字摘要） | 长对话上下文更完整 |
| 记忆持久化 | 纯内存，重启即失 | SQLite 持久化（`MEMORY_DB_PATH=chat_history.db`，重启不丢） | **增强** |
| 多会话隔离 | 按 session_id 隔离 | 不变（且磁盘上同样隔离） | 不变 |
| 工具调用 | 仅天气 | 天气 + A 股股票工具（行情/K线/全市场扫描/深度分析/资金流/新闻） | **增强** |
| 并行工具调用 | 顺序执行 | LangGraph Send API 扇出并行执行 | **增强** |
| 工具失败处理 | 无 | 失败自动重试（最多 2 次）后收场 | **增强** |
| 流式输出 | 有 | 有（`ask_stream`） | 不变 |
| 状态机可视化 | `python -m app.agent.visualize` | 不变（适配 LangGraph 1.x 的 `get_graph()`） | 不变 |
| 终端命令 | `/help /clear /history /model /exit` | 不变 | 不变 |
| 模型后端 | OpenAI 兼容 / Ollama / fake 演示 | 不变 | 不变 |

---

## 3. 多 Agent 投研（stock_agent）功能对比

| 功能项 | 更新前 | 更新后 | 效果变化 |
| --- | --- | --- | --- |
| 框架依赖 | LangGraph 0.2.x + LangChain 0.3.x | LangGraph 1.2.x + LangChain 1.x | 升级，逻辑不变 |
| 总控协调官（Supervisor） | 只绑定专家工具，只做调度决策，绝不直接取数 | 不变 | 不变 |
| 专家数量 | 5 位：选股师 / 分析师 / 情报官 / 决策官 / 风险官 | 不变 | 不变 |
| 数据获取方式 | 经独立 `data_engineer.py` 工具层取数 | 移除该层，数据能力内聚到 `app/tools/stock.py`，专家直接绑定数据工具 | 架构简化，能力不变 |
| 专家封装 | `wrap_expert_as_tool()` + 执行日志（开始/完成/跳过） | 不变 | 不变 |
| 投研流水线 | LangGraph StateGraph：`screener → (analyst ∥ intel) → decision → risk` | 不变 | 不变 |
| 显式代码跳过选股 | 诉求带具体代码时跳过全市场扫描 | 不变 | 不变 |
| 入口 | `python -m stock_agent.main`（灵活问答 + `/report` 双模式） | 不变 | 不变 |

---

## 4. RAG（rag）功能对比

| 功能项 | 更新前 | 更新后 | 效果变化 |
| --- | --- | --- | --- |
| 向量库底层 API | langchain-milvus 0.1.x + pymilvus <2.6，使用 ORM `Collection` 对象（`query()/delete()/flush()/drop()`） | langchain-milvus 0.4.x + pymilvus 3.x **MilvusClient** API（`client.query()/delete()/flush()/drop_collection()`） | 修复：旧 API 已移除，导致无法查询库中资料 |
| 嵌入缓存 | `CacheBackedEmbeddings`（`langchain.embeddings`） | `CacheBackedEmbeddings`（`langchain_classic.embeddings`） | 修复：1.x 中旧导入路径失效 |
| 文档解析/分块 | PDF / DOCX / TXT / MD，语义分块 + 重叠 | 不变 | 不变 |
| 向量检索 | Milvus 相似度检索 | 不变 | 不变 |
| 重排 | 关键词 / 语义重排 | 不变 | 不变 |
| 混合检索 | 可选 BM25（rank_bm25） | 不变 | 不变 |
| 失败重试 | 指数退避（区分可重试/不可重试错误） | 不变 | 不变 |
| 文档管理命令 | `/upload /documents /status /delete /clear /select` | 不变 | 不变 |

---

## 5. 移除的内容

| 文件/代码 | 类型 | 移除原因 |
| --- | --- | --- |
| `app/chain.py` | 源码 | LCEL 手写工具循环，依赖已弃用的 `RunnableWithMessageHistory`；对话已统一由 LangGraph Agent 驱动 |
| `main_langgraph.py` | 源码 | 重复入口，功能并入 `main.py` |
| `stock_agent/data_engineer.py` | 源码 | 冗余数据工具层，数据能力内聚到 `app/tools/stock.py` |
| `tests/test_chain.py` | 测试 | 针对已移除的 chain.py |
| `app/prompts.py::build_prompt` | 死代码 | 仅被已移除的 chain.py 调用，无其他引用方 |
| 各文件对已删模块的过时注释 | 注释 | 保持文档与实现一致 |

---

## 6. 新增的内容

| 内容 | 说明 |
| --- | --- |
| `langchain-classic` 依赖 | 提供 1.x 中迁移出的旧 API（`CacheBackedEmbeddings`） |
| `tests/test_stock_agent.py` | 5 个离线用例：专家工具封装、五位专家组装、总控回退、流水线跑通、显式代码跳过选股 |
| `tests/test_rag.py` | 5 个离线用例：文档分块、块元数据、关键词重排、BM25 检索、重试判定 |
| `MEMORY_DB_PATH` 配置 | SQLite 记忆持久化开关（`chat_history.db`，已加入 `.gitignore`） |
| 各方法/类缺失的 docstring | 为 8 处方法补齐用途、参数、返回说明 |

---

## 7. 测试对比

| 项 | 更新前 | 更新后 |
| --- | --- | --- |
| 测试文件 | 2 个（test_chain.py + test_lg_agent.py） | 4 个（test_lg_agent / test_memory_persistence / test_stock_agent / test_rag） |
| 用例数量 | 16 个（含记忆持久化） | **19 个** |
| 覆盖范围 | app 对话 + 记忆持久化 | app 对话 + 记忆持久化 + 多 Agent 投研 + RAG 纯逻辑 |
| 运行方式 | 全离线（fake 模型 / monkeypatch） | 不变 |
| 结果 | 通过 | **全部通过** |

---

## 8. 运行效果一致性核对

| 场景 | 升级前（LangGraph 版） | 升级后 | 是否一致 |
| --- | --- | --- | --- |
| 对话多轮记忆 | 窗口裁剪 + 写回 | 相同（另有压缩与持久化增强） | 一致（增强） |
| 工具调用（天气/股票） | 模型点名 → 执行 → 回填 | 相同 | 一致 |
| 流式输出 / 终端命令 | 有 | 有 | 一致 |
| 股票专家问答与 /report 流水线 | 五专家 + 状态图 DAG | 相同逻辑 | 一致 |
| RAG 上传/检索/问答 | 分块→嵌入→Milvus→检索→问答 | 相同，底层 API 适配 | 一致 |
| 离线演示模式（fake） | 免 Key 可跑 | 相同 | 一致 |

**结论**：三大子系统功能行为与升级前一致；差异仅在架构收敛（统一 LangGraph）、
API 适配（pymilvus 3.x / langchain-classic）与能力增强（股票工具、记忆持久化、并行工具）。

---

## 9. 注意事项

1. **旧数据可读**：Milvus 中已入库的集合/数据与升级前完全兼容（字段 schema 未变），无需重建。
2. **嵌入缓存**：Redis 缓存键格式未变，缓存可继续命中。
3. **`langchain_classic` 仅用于 `CacheBackedEmbeddings`**，若后续 1.x 提供替代方案可再迁移。
4. 对话记忆库（`chat_history.db`）为运行时产物，已在 `.gitignore` 中排除。
