"""投研流水线：把多位专家串成「筛选 → 分析 → 决策 → 风控」的闭环。

与总控协调官（Supervisor）并行调度不同，流水线强调**顺序依赖**。
这里用 LangGraph 状态图（StateGraph）实现：定义一块「共享状态画板」
ResearchState，各阶段是图的一个节点，把产物写入自己负责的字段；后续
阶段节点从共享状态读取上一阶段的产物，从而形成层层递进的依赖闭环，
而不必靠"把上一段文本粘进下一段提示词"来传递。

流程（DAG）：
  选股师(screener) ──┬──→ 分析师(analyst) ──┐
                     └──→ 情报官(intel)   ──┴──→ 决策官(decision) ──→ 风险官(risk)

共享状态字段（ResearchState）：
  query     用户诉求
  screener  选股师报告
  candidates  选股师提取出的候选股票代码
  analyst   分析师报告（对候选逐股深挖）
  intel     情报官报告（消息面/舆情）
  decision  决策官最终评价
  risk      风险官裁决（一票否决）
"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from app.agent import LangGraphAgent
from app.config import Settings
from app.memory import MemoryManager
from stock_agent.experts import (
    build_analyst,
    build_decision,
    build_intel,
    build_risk,
    build_screener,
)

# ---- 共享状态画板 ----
class ResearchState(TypedDict):
    """投研流水线的共享状态画板（各阶段产物在此交接）。"""

    query: str  # 用户诉求
    screener: str  # ① 选股师报告
    candidates: list[str]  # ① 选股师提取出的候选股票代码
    analyst: str  # ② 分析师报告
    intel: str  # ③ 情报官报告
    decision: str  # ④ 决策官最终评价
    risk: str  # ⑤ 风险官裁决


# 从文本里尽量提取 6 位股票代码（sh600519 / 000001 / 600519.SH 等）
_CODE_RE = re.compile(r"(?<![0-9])([0369]\d{5})(?![0-9])")

# ---- 代码提取函数 ----
def _extract_codes(text: str) -> list[str]:
    """从决策/选股文本中提取候选股票代码，去重保序。"""
    # 记录"已经见过的代码"，用于去重
    seen: set[str] = set()
    # 提取所有匹配的股票代码
    codes: list[str] = []
    for m in _CODE_RE.findall(text):
        # 如果是第一次遇到，记录下并添加到结果列表
        if m not in seen:
            seen.add(m)
            codes.append(m)
    return codes

# ---- 投研流水线 ----
class StockResearchPipeline:
    """基于 LangGraph 状态图的投研流水线。

    五个专家节点挂在共享状态上，各节点读取所需字段、写入自己的产物，
    由图的边决定执行顺序（并行 / 串行）。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # 各专家持有彼此独立的记忆空间，避免上下文污染
        self._memories: dict[str, MemoryManager] = {
            "screener": MemoryManager(settings),
            "analyst": MemoryManager(settings),
            "intel": MemoryManager(settings),
            "decision": MemoryManager(settings),
            "risk": MemoryManager(settings),
        }
        # 初始化专家节点：每个专家持有独立的记忆空间，避免上下文污染
        self._experts: dict[str, LangGraphAgent] = {
            "screener": build_screener(settings, self._memories["screener"]),
            "analyst": build_analyst(settings, self._memories["analyst"]),
            "intel": build_intel(settings, self._memories["intel"]),
            "decision": build_decision(settings, self._memories["decision"]),
            "risk": build_risk(settings, self._memories["risk"]),
        }
        self._graph = self._build_graph()

    # ---- 图定义：节点挂到共享状态，边决定依赖 ----
    def _build_graph(self) -> StateGraph:
        """构建并编译投研流水线状态图（DAG）。

        节点：screener → (analyst ∥ intel) → decision → risk → END，
        其中 analyst 与 intel 都依赖 screener，二者可并行。

        Returns:
            编译后的 StateGraph（真正驱动流水线的本体）。
        """
        graph = StateGraph(ResearchState)
        # 添加节点：选股师、分析师、情报官、决策官、风控官
        # graph.add_node(节点名， 节点函数)
        graph.add_node("screener", self._node_screener)
        graph.add_node("analyst", self._node_analyst)
        graph.add_node("intel", self._node_intel)
        graph.add_node("decision", self._node_decision)
        graph.add_node("risk", self._node_risk)
        # 添加边：选股师 -> 分析师、情报官 -> 决策官 -> 风险官
        # graph.add_edge(源节点， 目标节点)
        # 依赖：分析师与情报官都依赖选股师，可并行；决策依赖二者；风控收尾
        graph.add_edge(START, "screener")
        graph.add_edge("screener", "analyst")
        graph.add_edge("screener", "intel")
        graph.add_edge("analyst", "decision")
        graph.add_edge("intel", "decision")
        graph.add_edge("decision", "risk")
        graph.add_edge("risk", END)
        return graph.compile()

    # ---- 单阶段：让某个专家基于给定任务作答 ----
    def _run(self, name: str, task: str) -> str:
        """让指定专家基于一段任务描述作答，返回其回复文本。

        Args:
            name: 专家节点名（screener / analyst / intel / decision / risk）。
            task: 交给该专家的任务描述。

        Returns:
            专家回复的文本内容（BaseMessage 取 content，否则转字符串）。
        """
        # 从专家节点字典中获取对应专家
        expert = self._experts[name]
        print(f"\n  ▶ [专家·{name}] 收到任务，开始执行…（独立会话 pipeline_{name}）")
        # 调用专家节点的 ask 方法，执行任务
        reply = expert.ask(task, f"pipeline_{name}")
        print(f"  ✔ [专家·{name}] 执行完成。")
        # 返回专家回复的内容（或直接转换为字符串）
        return reply.content if isinstance(reply, BaseMessage) else str(reply)

    # ---- 各阶段节点：读写共享状态字段 ----
    # 选股师节点
    def _node_screener(self, state: ResearchState) -> dict:
        """选股师：全市场筛选，产物写入 screener / candidates。

        若用户诉求里已明确给出股票代码（例如 600519），说明用户只想针对
        该标研究，无需再扫描全市场——选股师直接「放行」此标的即可，
        把节约的成本留给后续分析。
        """
        query = state.get("query", "")
        explicit = _extract_codes(query)  # 诉求里带的具体代码
        if explicit:
            # 若用户诉求里已明确给出股票代码，直接放行
            candidates = explicit[:3]
            print(f"  ⏭ [专家·screener] 用户已指定标的 {candidates}，跳过全市场筛选。")
            return {
                "screener": (
                    "（用户诉求已指定标的，跳过全市场筛选，直接进入个股分析。）\n"
                    f"候选标的代码：{', '.join(candidates)}"
                ),
                "candidates": candidates,
            }
        report = self._run(
            "screener",
            f"请扫描全市场，筛选强势股候选。用户诉求：{query}。"
            "请务必在回复中列明候选股票的 6 位代码，例如 600519。",
        )
        # 从选股师回复中提取候选股票代码，去重保序
        candidates = _extract_codes(report)
        return {"screener": report, "candidates": candidates}

    # 分析师节点
    def _node_analyst(self, state: ResearchState) -> dict:
        """分析师：对候选逐股深挖，产物写入 analyst。"""
        # 从状态状态中获取用户诉求和候选股票代码
        query = state.get("query", "")  # 用户诉求
        candidates = state.get("candidates", [])  # 候选股票代码
        # 对每个候选股票进行深度分析
        parts: list[str] = []
        for code in candidates[:3]:  # 最多深挖 3 只，控制成本
            parts.append(
                self._run(
                    "analyst",
                    f"请对候选股票 {code} 做多维度深度分析。"
                    f"用户诉求：{query}。选股师已将其列入候选，请给出技术面结论。",
                )
            )
        return {
            "analyst": "；\n".join(parts)
            or "（分析师：候选股票清单为空，未展开）"
        }

    # 情报官节点
    def _node_intel(self, state: ResearchState) -> dict:
        """情报官：补消息面/舆情，产物写入 intel。"""
        # 从状态状态中获取用户诉求和候选股票代码
        query = state.get("query", "")  # 用户诉求
        candidates = state.get("candidates", [])  # 候选股票代码
        # 优先调研重点标的，若无重点标的则调研所有候选股票
        focus = candidates[0] if candidates else ""
        task = (
            f"请就重点标的 {focus} 调研市场消息面。用户诉求：{query}。"
            if focus
            else f"请就本次投研涉及标的调研市场消息面。用户诉求：{query}。"
        )
        return {"intel": self._run("intel", task)}

    # 决策官节点
    def _node_decision(self, state: ResearchState) -> dict:
        """决策官：综合各方报告，产物写入 decision。"""
        query = state.get("query", "")
        task = (
            f"请基于以下材料给出最终投资评价。用户诉求：{query}。\n"
            f"【选股师】{state.get('screener', '')}\n"
            f"【分析师】{state.get('analyst', '')}\n"
            f"【情报官】{state.get('intel', '')}\n"
        )
        return {"decision": self._run("decision", task)}

    # 风险官节点
    def _node_risk(self, state: ResearchState) -> dict:
        """风险官：一票否决，最终把关，产物写入 risk。"""
        query = state.get("query", "")
        task = (
            f"请审查以下投资推荐的风险，并做出裁决。用户诉求：{query}。\n"
            f"【决策官推荐】{state.get('decision', '')}\n"
            f"【分析师】{state.get('analyst', '')}\n"
        )
        return {"risk": self._run("risk", task)}

    # ---- 对外接口 ----
    def run(self, query: str) -> dict[str, str]:
        """跑完整流水线，返回各阶段报告（与 ResearchState 对应字段）。"""
        final = self._graph.invoke({"query": query})
        return {
            "screener": final.get("screener", ""),
            "analyst": final.get("analyst", ""),
            "intel": final.get("intel", ""),
            "decision": final.get("decision", ""),
            "risk": final.get("risk", ""),
        }


def build_pipeline(settings: Settings) -> StockResearchPipeline:
    """组装一条完整的投研流水线。"""
    return StockResearchPipeline(settings)