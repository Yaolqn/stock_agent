# 多 Agent 股票投研 — 设计稿

一个面向 A 股投研的协作式多 Agent，由若干专家角色协同完成「选股 → 分析 → 决策」的完整链路。

## 角色划分（分三类）

| 类型 | 角色 | 核心职责 |
|---|---|---|
| Supervisor | 总控协调官 | 接收指令、拆解任务、调度其他 Agent、汇总输出 |
| 专家 Agent | 智能选股师 | 用量化指标在全市场筛选强势股 |
| 专家 Agent | 多维分析师 | 对个股做技术、资金、基本面等多维分析 |
| 专家 Agent | 市场情报官 | 监控市场环境、新闻热点、舆情（提供宏观背景） |
| 专家 Agent | 投资决策官 | 综合各方报告，做出最终评价 |
| 专家 Agent | 风险管理官 | 评估推荐风险，拥有一票否决权 |
| 基础设施 | 数据工程师 | 为所有 Agent 提供数据支持 |

## 关键设计决策（已确认）

- **数据工程师退化为工具层**：不作为单独图节点，而是封装 akshare 供所有 Agent 直接调用。复用现有 `app/tools/` 思路。
- **风控否决 → 打回重做**：风控否决时打回投资决策官重做一次，再不行直接收场。
- **候选股并行分析**：选股师给出候选清单后，用 Send API 并行拆分支，每只股票一条分支（复用现有 Send 技术）。

## 主流程

```
用户指令 → 总控协调官
          ├─ 调度智能选股师 → 候选强势股
          └─ 调度市场情报官 → 宏观背景/新闻舆情
                          ↓
          多维分析师（对每只候选股并行分析）
                          ↓
          投资决策官（综合各方，给出最终评价）
                          ↓
          风险管理官（评估风险，一票否决）
              ├─ 通过 → 输出推荐
              └─ 否决 → 打回投资决策官重做，再不行收场
```

## 技术要点

- 数据源：新浪系列接口（东财接口经验证会被限流，新浪源稳定）。
- 并行：LangGraph `Send` 扇出，每股一条分支，reducer 合并结果。

## 待确认

- 7 个角色是否各自独立 LLM，还是共用模型 + 不同 system_prompt。
- 内存与成本结构设计。

## 里程碑记录

### ✅ 第一步：总控协调官 + 数据工程师（最基础闭环）
- `stock_agent/data_engineer.py`：数据工程师 → 工具层，聚合 app/tools/stock 的两个数据工具
- `stock_agent/supervisor.py`：总控协调官 → 复用单 agent(LangGraphAgent)，绑定数据工具 + 股票领域提示词
- `stock_agent/main.py`：终端入口（`python -m stock_agent.main`）
- 数据工具来源：`app/tools/stock.py`（新浪源），已在 app/tools/__init__.py 登记
- 验证：fake 模式闭环通过，11/11 测试通过
- 下一步：实现专家 agent（选股师/分析师/情报官...），再把总控协调官升级为真正的 supervisor 节点

### ✅ 第二步：智能选股师 + 多维分析师（两个核心功能）
- 数据工程师新增两个核心数据工具（app/tools/stock.py）：
  - `scan_market` —— 全市场扫描，按涨跌幅/成交额返回强势股榜
  - `analyze_stock_deep` —— 单只深度分析（均线、区间涨跌幅、年化波动率、量能）
- `stock_agent/experts.py`：两专家复用 LangGraphAgent，各自绑定数据工具子集 + 角色提示词
  - 智能选股师（build_screener）→ scan_market + get_stock_quote，实现「扫描全市场」
  - 多维分析师（build_analyst）→ analyze_stock_deep + get_stock_quote，实现「深度分析单只股票」
- `stock_agent/main.py` 升级为三角色切换入口（/role 命令）
- 验证：两工具真实数据返回正常；fake 模式三角色闭环通过；11/11 测试通过

### ✅ 第三步：总控协调官升级为 Supervisor（只调度、不碰数据）+ 专家独立记忆
- 收紧架构：Supervisor 只绑定「专家工具」，**绝不直接绑定数据工具**（supervisor.py 删除 get_data_tools 直连）
- `wrap_expert_as_tool()`：把专家 agent 封装为 supervisor 可调度工具；`build_supervisor(..., expert_tools=...)`
- 入口收敛为单一总控（main.py 去掉 /role 三角色切换）
- 记忆隔离：`build_expert_tools(settings)` 内为每位专家新建独立 MemoryManager，与总控互相隔离
- 验证：fake 组装正常；真实调用「北方稀土」→总控委派专家取数全链路通过；11/11 测试通过

### ✅ 第四步：投研流水线（筛选 → 分析 → 决策 → 风控闭环）
- 数据工程师新增两个工具（app/tools/stock.py）：
  - `get_stock_fund_flow` —— 个股资金流向（东方财富源）
  - `get_stock_news` —— 个股新闻标题（东方财富源）
  - 已登记进 TOOLBOX，真实返回验证通过
- `stock_agent/experts.py` 新增三位专家（复用 build_* 模式，独立记忆）：
  - 市场情报官 build_intel —— 绑 get_stock_news，判消息面
  - 投资决策官 build_decision —— 绑 get_stock_fund_flow + quote，出最终评价
  - 风险管理官 build_risk —— 绑 fund_flow + analyze_deep，一票否决
- `stock_agent/pipeline.py` 新增 `StockResearchPipeline`：
  - 五阶段**顺序依赖**：选股师→分析师→情报官→决策官→风险官
  - 各阶段独立 MemoryManager，产物经任务描述层层传递
  - 风控为最终把关（风险评级 + 放行/一票否决），不做打回重做
  - 【v2】改为 **LangGraph StateGraph 共享状态画板 `ResearchState`**：
    各阶段是图节点，产物写入 `screener/analyst/intel/decision/risk` 字段，
    分析师与情报官并行、汇入决策官，再由风控收尾 —— 阶段间经共享状态交接，
    而非文本拼接；`run()` 返回五段报告
- 验证：fake 五阶段组装通过；真实模型跑通「茅台」完整闭环
  （候选→各报告→决策→风险「中/放行」），11/11 测试通过
- 下一步：决策官/风险官的「打回重做」有向环(可选)、get_historical_impact 事件量化