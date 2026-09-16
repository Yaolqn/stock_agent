"""stock_agent 多 Agent 模块的离线测试（不调用真实模型、不访问网络）。

验证核心承诺：
  1. wrap_expert_as_tool 能把专家 agent 封装成 supervisor 可调用的工具；
  2. build_expert_tools 在 fake 模型下仍能组装出 5 个专家工具；
  3. 投研流水线（StateGraph）在 fake 模型下能完整跑完并产出 5 个阶段字段；
  4. 用户诉求自带股票代码时，流水线自动跳过全市场筛选（候选直接放行）。
"""

from __future__ import annotations

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agent import LangGraphAgent
from app.config import Settings
from app.memory import MemoryManager
from stock_agent.experts import build_expert_tools
from stock_agent.pipeline import StockResearchPipeline
from stock_agent.supervisor import build_supervisor, wrap_expert_as_tool


def _settings() -> Settings:
    """构造纯内存、离线可用的 fake 配置。"""
    return Settings(llm_provider="fake", memory_db_path="")


def _fake_llm():
    """构造固定回复的假模型（离线可用，不发起任何网络请求）。"""
    return GenericFakeChatModel(
        messages=cycle(AIMessage(content="（专家回复）") for _ in range(10))
    )


def test_wrap_expert_as_tool_invokes_expert():
    """专家 agent 封装为工具后，invoke 会执行专家并回传其结论文本。"""
    settings = _settings()
    llm = _fake_llm()
    memory = MemoryManager(settings)
    # 直接构造一个极简专家 agent（无工具），验证封装层本身可用
    expert = LangGraphAgent(llm, [], memory, "你是股票分析师。")

    tool = wrap_expert_as_tool("analyze", "对单只股票做深度分析", expert)

    # 工具应能用 LangChain 1.x 的 StructuredTool 协议被调用
    result = tool.invoke({"task": "分析一下600519"})
    assert result == "（专家回复）"
    # 工具注册名与描述应正确暴露给 supervisor 大脑
    assert tool.name == "analyze"
    assert "深度分析" in tool.description


def test_build_expert_tools_returns_five_tools():
    """fake 模型下 build_expert_tools 仍应返回 5 个专家工具（scan/analyze/intel/decision/risk）。"""
    settings = _settings()
    tools = build_expert_tools(settings)

    names = [tool.name for tool in tools]
    assert sorted(names) == ["analyze", "decision", "intel", "risk", "scan"]
    # 每个工具都应能离线执行（fake 专家直接返回固定文本）
    for tool in tools:
        reply = tool.invoke({"task": "请完成你的职责"})
        assert isinstance(reply, str) and reply


def test_build_supervisor_with_fake_model_falls_back_to_chat():
    """fake 模型不支持工具绑定时，总控应退化为纯对话且组装不报错。"""
    settings = _settings()
    memory = MemoryManager(settings)
    tools = build_expert_tools(settings)

    agent = build_supervisor(settings, memory, expert_tools=tools)

    # build_supervisor 内部用 create_llm(settings) 创建 fake 模型，
    # 回复为 app.models 的固定演示文案（离线可验证）
    reply = agent.ask("贵州茅台今天怎么样？", "s1")
    assert isinstance(reply.content, str) and reply.content
    # 记忆应正常写回（1 用户 + 1 AI）
    messages = memory.get_session_history("s1").messages
    assert len(messages) == 2


def test_pipeline_runs_full_graph_with_fake_llm():
    """投研流水线（screener→analyst∥intel→decision→risk）应完整跑完，返回 5 个字段。"""
    settings = _settings()
    pipeline = StockResearchPipeline(settings)

    reports = pipeline.run("帮我从全市场挑选强势股")

    # 5 个阶段产物字段都应存在（fake 模型下为固定文本 / 占位提示）
    for key in ("screener", "analyst", "intel", "decision", "risk"):
        assert key in reports and isinstance(reports[key], str)


def test_pipeline_skips_scan_when_explicit_code():
    """诉求自带 600519 时，选股师直接放行该标的，candidates 应为该代码。"""
    settings = _settings()
    pipeline = StockResearchPipeline(settings)

    # 直接调用选股师节点，验证"跳过全市场筛选"分支
    state = pipeline._node_screener({"query": "分析一下贵州茅台600519"})
    assert state["candidates"] == ["600519"]
    assert "跳过全市场筛选" in state["screener"]
