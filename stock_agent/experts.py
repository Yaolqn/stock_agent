"""专家 Agent：智能选股师 / 多维分析师 / 市场情报官 / 投资决策官 / 风险管理官。

结构与总控协调官一致：每个专家都是复用单 agent（LangGraphAgent），
只是绑定的数据工具子集 + 角色提示词不同。当前五位专家：
  智能选股师（build_screener）—— 扫描全市场，筛选强势股
  多维分析师（build_analyst） —— 深度分析单只股票
  市场情报官（build_intel）   —— 调研消息面/新闻热点/舆情
  投资决策官（build_decision）—— 综合各方给出最终投资评价
  风险管理官（build_risk）    —— 风险审查 + 一票否决

数据获取方式：专家直接绑定 app/tools/stock.py 中的数据工具
（scan_market / analyze_stock_deep / get_stock_quote / get_stock_news /
get_stock_fund_flow）。原先独立的 data_engineer 工具层已移除，
数据能力内聚在 app.tools.stock，避免重复实现与层次冗余。

此外提供 build_expert_tools()：把五位专家组装为 supervisor 可调度的工具，
供总控协调官（Supervisor）并行调度使用。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agent import LangGraphAgent
from app.config import Settings
from app.memory import MemoryManager
from app.models import create_llm
from app.tools.stock import (
    analyze_stock_deep,
    get_stock_fund_flow,
    get_stock_news,
    get_stock_quote,
    scan_market,
)
from stock_agent.supervisor import wrap_expert_as_tool

# ---------------- 角色提示词 ----------------
# 智能选股师角色提示词
SCREENER_PROMPT = (
    "你是股票投研系统的智能选股师。你的职责是扫描全市场，用量化指标筛选强势股。"
    "请使用 scan_market 工具获取全市场榜单（按成交额或涨跌幅排序），"
    "然后从中挑选出值得关注的强势股，并给出简要的筛选理由。"
    "若用户指定了关注方向（如行业、风格），结合榜单信息给出建议。"
    "回答用简洁、准确、友好的中文。"
)

# 多维分析师角色提示词
ANALYST_PROMPT = (
    "你是股票投研系统的多维分析师。你的职责是对单只股票做多维度深度分析。"
    "请使用 analyze_stock_deep 工具获取该股票的技术面数据（均线、区间涨跌幅、"
    "波动率、量能），必要时可结合 get_stock_quote 了解实时行情。"
    "然后综合这些维度，给出结构化的分析结论（趋势、强弱、风险点）。"
    "回答用简洁、准确、友好的中文。"
)

# 情报官角色提示词
INTEL_PROMPT = (
    "你是股票投研系统的市场情报官。你的职责是监控研究标的的市场环境、"
    "新闻热点与舆情，为投研提供宏观/消息面背景。"
    "请使用 get_stock_news 工具获取相关股票的近期新闻标题，"
    "并结合你的判断，指出可能影响股价的消息面因素（利好/利空/中性）。"
    "回答用简洁、准确、友好的中文。"
)

# 决策官角色提示词
DECISION_PROMPT = (
    "你是股票投研系统的投资决策官。你的职责是综合选股师、分析师、情报官"
    "各方报告，汇成对目标股票的最终评价。"
    "请使用 get_stock_fund_flow 工具参考资金流向佐证人气，必要时用 "
    "get_stock_quote 确认最新行情。"
    "然后给出结构化的最终评价：评级（推荐/观望/回避）、核心投资逻辑、"
    "以及明确的持有/买入/卖出倾向。回答用简洁、准确、友好的中文。"
)

# 风险官角色提示词
RISK_PROMPT = (
    "你是股票投研系统的风险管理官。你的职责是对决策官给出的推荐做风险审查，"
    "并拥有一票否决权。"
    "请使用 get_stock_fund_flow 检查资金流向风险、用 analyze_stock_deep 复核"
    "技术面风险（波动率、趋势背离等）。"
    "输出格式：先是「风险评级：低/中/高」与理由，再给出你的裁决"
    "「放行」或「一票否决」，以及需补充说明的提示。回答用简洁、准确、中文。"
)


# ---------------- 专家 Assembly ----------------

def _bind_if_supported(llm, tools):
    """绑定工具；模型不支持工具调用时回退为纯对话 llm。"""
    try:
        return llm.bind_tools(tools)
    except NotImplementedError:
        return llm


def build_screener(settings: Settings, memory: MemoryManager) -> LangGraphAgent:
    """组装智能选股师：绑定 scan_market（扫描）+ quote（行情核对）。"""
    llm = create_llm(settings)
    tools = [scan_market, get_stock_quote] if settings.llm_provider in ("openai", "ollama") else []
    model = _bind_if_supported(llm, tools) if tools else llm
    return LangGraphAgent(model, tools, memory, SCREENER_PROMPT)


def build_analyst(settings: Settings, memory: MemoryManager) -> LangGraphAgent:
    """组装多维分析师：绑定 analyze_stock_deep（深度）+ quote（实时行情）。"""
    llm = create_llm(settings)
    tools = [analyze_stock_deep, get_stock_quote] if settings.llm_provider in ("openai", "ollama") else []
    model = _bind_if_supported(llm, tools) if tools else llm
    return LangGraphAgent(model, tools, memory, ANALYST_PROMPT)


def build_intel(settings: Settings, memory: MemoryManager) -> LangGraphAgent:
    """组装市场情报官：绑定 get_stock_news（消息面）+ quote（行情核对）。"""
    llm = create_llm(settings)
    tools = [get_stock_news, get_stock_quote] if settings.llm_provider in ("openai", "ollama") else []
    model = _bind_if_supported(llm, tools) if tools else llm
    return LangGraphAgent(model, tools, memory, INTEL_PROMPT)


def build_decision(settings: Settings, memory: MemoryManager) -> LangGraphAgent:
    """组装投资决策官：绑定 get_stock_fund_flow（资金）+ quote（行情）。"""
    llm = create_llm(settings)
    tools = [get_stock_fund_flow, get_stock_quote] if settings.llm_provider in ("openai", "ollama") else []
    model = _bind_if_supported(llm, tools) if tools else llm
    return LangGraphAgent(model, tools, memory, DECISION_PROMPT)


def build_risk(settings: Settings, memory: MemoryManager) -> LangGraphAgent:
    """组装风险管理官：绑定 fund_flow（资金）+ analyze_stock_deep（技术）。"""
    llm = create_llm(settings)
    tools = [get_stock_fund_flow, analyze_stock_deep] if settings.llm_provider in ("openai", "ollama") else []
    model = _bind_if_supported(llm, tools) if tools else llm
    return LangGraphAgent(model, tools, memory, RISK_PROMPT)


def build_expert_tools(settings: Settings) -> list[BaseTool]:
    """组装专家为 supervisor 可调度的工具列表（供 Supervisor 并行调度）。

    每个专家拥有**完全独立的记忆空间**：各自新建一个 MemoryManager，
    互不共享、也绝不复用总控的记忆。这样既避免专家之间上下文互相污染，
    也杜绝专家读到总控（或其内部调用序列）的历史。

    各专家仍会独立构建一份（含各自绑定的数据工具），封装成工具返回，
    交由 build_supervisor(..., expert_tools=...) 使用。
    """
    # 各专家各自独立的记忆空间（与总控 memory 完全隔离）
    screener_memory = MemoryManager(settings)
    analyst_memory = MemoryManager(settings)
    intel_memory = MemoryManager(settings)
    decision_memory = MemoryManager(settings)
    risk_memory = MemoryManager(settings)
    return [
        wrap_expert_as_tool(  # 智能选股师 → 工具名 scan
            "scan",
            "扫描全市场，按量化指标筛选强势股。需要选股/找强势股时调用。",
            build_screener(settings, screener_memory),
        ),
        wrap_expert_as_tool(  # 多维分析师 → 工具名 analyze
            "analyze",
            "对单只股票做多维度深度分析（均线/涨跌幅/波动率/量能）。需要深度分析单只股票时调用。",
            build_analyst(settings, analyst_memory),
        ),
        wrap_expert_as_tool(  # 市场情报官 → 工具名 intel
            "intel",
            "调研指定股票的市场消息面/新闻热点/舆情。需要宏观背景、消息面解读时调用。",
            build_intel(settings, intel_memory),
        ),
        wrap_expert_as_tool(  # 投资决策官 → 工具名 decision
            "decision",
            "综合选股师/分析师/情报官的结论，给出对目标股票的投资决策（评级/建议）。需要最终投资评价时调用。",
            build_decision(settings, decision_memory),
        ),
        wrap_expert_as_tool(  # 风险管理官 → 工具名 risk
            "risk",
            "对投资推荐做风险审查，拥有一票否决权，输出风险评级与裁决。需要对推荐把关、确认风险可控时调用。",
            build_risk(settings, risk_memory),
        ),
    ]