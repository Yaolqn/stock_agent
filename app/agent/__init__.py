"""Agent 包：基于 LangGraph 的对话 Agent。

结构说明（按职责拆分，便于后续扩展）：
  state.py     —— 状态画板定义（AgentState，日后新增字段在此处声明）；
  agent.py     —— 图定义、节点实现、对外接口（LangGraphAgent / build_lg_agent）；
  visualize.py —— 状态机可视化（ASCII 图 / Mermaid 代码，`python -m app.agent.visualize`）。

入口统一从本包导入：
    from app.agent import LangGraphAgent, build_lg_agent
"""

from app.agent.agent import LangGraphAgent, build_lg_agent

__all__ = ["LangGraphAgent", "build_lg_agent"]