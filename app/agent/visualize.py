"""LangGraph 状态机可视化工具。

输出当前 Agent 的状态机拓扑（图结构），两种形式：
  1. ASCII 图    —— 终端直接查看（依赖 grandalf，未安装时自动提示）；
  2. Mermaid 代码 —— 可粘贴到 https://mermaid.live、VS Code Markdown 预览等
                     任意支持 Mermaid 的渲染器中变成图形。

命令行用法：
    python -m app.agent.visualize

代码中复用：
    from app.agent.visualize import print_mermaid
    graph = ...          # 任意已编译的状态图（CompiledStateGraph）
    print_mermaid(graph)
"""

from __future__ import annotations

import sys

from app.agent.agent import build_lg_agent
from app.config import get_settings
from app.memory import MemoryManager
from app.tools import get_available_tools


def _ensure_utf8_stdio() -> None:
    """确保 Windows 终端能正确显示中文。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _build_graph():
    """按与终端入口（app/cli.py）相同的条件组装 agent，返回其编译后的状态图。

    图的结构与是否启用工具无关（call_model ⇄ action 的骨架始终存在），
    这里保持与入口一致的装配方式，便于后续更新时可视化结果与真实一致。
    """
    settings = get_settings()
    # 与入口保持一致：只有真实模型（openai / ollama）才启用工具
    tools = get_available_tools() if settings.llm_provider in ("openai", "ollama") else None
    agent = build_lg_agent(settings, MemoryManager(settings), tools=tools)
    return agent.graph.get_graph()


def print_ascii(graph) -> None:
    """终端打印 ASCII 图；缺 grandalf 依赖时给出一条安装提示。"""
    try:
        print(graph.draw_ascii())
    except ImportError as exc:
        print(f"[提示] 生成 ASCII 图依赖 grandalf：{exc}")
        print("       安装：pip install grandalf")


def print_mermaid(graph) -> None:
    """打印 Mermaid 代码，供任意 Mermaid 渲染器（mermaid.live / VS Code 等）使用。"""
    print(graph.draw_mermaid())


def main() -> None:
    """入口：组装 agent → 依次打印 ASCII 图与 Mermaid 代码。"""
    _ensure_utf8_stdio()
    graph = _build_graph()

    print("===== 状态机 ASCII 图 =====")
    print_ascii(graph)

    print("\n===== 状态机 Mermaid 代码 =====")
    print("渲染方式：1) 粘贴到 https://mermaid.live   2) VS Code 装「Markdown Preview Mermaid Support」后在 .md 文件里预览")
    print("```mermaid")
    print_mermaid(graph)
    print("```")


if __name__ == "__main__":
    main()