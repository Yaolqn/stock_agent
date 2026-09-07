"""数据工程师：为所有 Agent 提供数据支持的「工具层」。

按设计稿（DESIGN.md）的决策，数据工程师不作为单独图节点，而是退化为
工具层——本质上就是复用 app.tools 里已有的股票工具。数据工程师的职责
是「把这些数据工具显式暴露/聚合给股票子系统使用」，方便其它 Agent 理解
数据的来源与能力边界。

目前接入的数据工具（来自 app/tools/stock.py，新浪数据源）：
  get_stock_quote      —— 单只股票实时行情
  get_stock_history    —— 单只股票历史日 K 线
  scan_market          —— 全市场扫描，按涨幅/成交额返回强势股榜
  analyze_stock_deep   —— 单只股票多维度深度分析（均线/波动率/量能）

后续数据工程师可在本文件扩展：财务、板块、资金流、新闻等工具。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.stock import (
    analyze_stock_deep,
    get_stock_fund_flow,
    get_stock_history,
    get_stock_news,
    get_stock_quote,
    scan_market,
)

# 数据工程师当前可提供的数据能力清单（供各 Agent 绑定 & 聚合）
TOOLBOX: tuple[BaseTool, ...] = (
    get_stock_quote,
    get_stock_history,
    scan_market,
    analyze_stock_deep,
    get_stock_fund_flow,
    get_stock_news,
)


def get_data_tools() -> list[BaseTool]:
    """返回数据工程师提供的全部数据工具，供 Agent 绑定。"""
    return list(TOOLBOX)