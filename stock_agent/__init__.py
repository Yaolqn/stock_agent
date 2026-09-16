"""stock_agent —— 多 Agent 股票投研子系统（独立开发空间）。

结构（与 DESIGN.md 保持一致）：
  supervisor.py  总控协调官（Supervisor）→ 只做调度与决策，绝不直接取数；
                所有数据获取必须由专家中转，专家内部才绑定数据工具。
  experts.py     专家 Agent 工厂：智能选股师 / 多维分析师 / 市场情报官 /
                投资决策官 / 风险管理官，并封装为 supervisor 可调度的工具。
  pipeline.py    投研流水线：把多位专家串成「筛选 → 分析 → 决策 → 风控」闭环。
  main.py        终端入口（总控协调官 + /report 流水线双模式）。

说明：原先的数据工程师（data_engineer.py）工具层已被移除，数据访问能力
直接内聚在 app/tools/stock.py 中，专家通过绑定这些数据工具取数。
"""
