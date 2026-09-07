"""stock_agent —— 多 Agent 股票投研子系统（独立开发空间）。

当前为「第一步：最基础闭环」：
  数据工程师（data_engineer.py）→ 工具层，为所有 agent 提供数据
  总控协调官（supervisor.py）→ 复用单 agent（LangGraphAgent），注入数据工具

后续将按 DESIGN.md 逐步加入选股师 / 分析师 / 情报官 / 决策官 / 风控官。
"""