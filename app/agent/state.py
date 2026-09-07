"""Agent 状态定义：整轮对话共享的「记忆画板」。

新增状态字段（例如：剩余步数、预算、中间推理草稿）时，只需在此处
扩展 AgentState，并在对应节点里读写即可。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 定义 Agent 状态：整轮对话的完整消息轨迹。
class AgentState(TypedDict):
    """Agent 的「记忆画板」：整轮对话的完整消息轨迹。"""

    # add_messages 是 LangGraph 提供的合并器：新节点返回的消息
    # 会自动追加到现有 messages 列表尾部，而不是覆盖整张表。
    # add_messages 合并器：多个节点返回的消息会自动合并加到列表尾部，而不是覆盖。<追加>
    messages: Annotated[list[BaseMessage], add_messages]

    # 工具失败累计次数（用于限制「软重试/自愈」轮数，防止模型反复撞同一个错）。
    # operator.add 合并器：多个 action 分支（Send 并行）返回的失败计数
    # 会累加而非互相覆盖，公共节点读取到的是合并后的完整计数。
    # 整轮对话从 0 起算，初始缺失时用 state.get("retries", 0) 读取。
    # operator.add 合并器：多个节点返回的失败计数会累加而非互相覆盖。<累加>
    retries: Annotated[int, operator.add]