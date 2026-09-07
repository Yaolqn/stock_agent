"""LangGraph 版 Agent 的离线测试（不调用任何真实模型、不访问网络）。

与 tests/test_chain.py 对应，验证同一套能力（工具执行、记忆写回、
会话隔离）在 LangGraph 状态机实现下依然成立。
"""

from __future__ import annotations

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.agent import LangGraphAgent, build_lg_agent
from app.memory import MemoryManager
from app.tools import get_available_tools


def _make_agent(max_history: int = 6) -> LangGraphAgent:
    """构造一个使用假模型的 LangGraph Agent（离线可用）。"""
    settings = Settings(llm_provider="fake", max_history_messages=max_history)
    fake_llm = GenericFakeChatModel(
        messages=cycle(AIMessage(content=f"回复{i}") for i in range(100))
    )
    memory = MemoryManager(settings)
    agent = LangGraphAgent(fake_llm, [], memory, settings.system_prompt)
    return agent, memory


def test_two_round_conversation_writes_history():
    """连续两轮对话后，记忆中应保存 4 条消息（2 用户 + 2 AI）。"""
    agent, memory = _make_agent()
    assert agent.ask("你好", "s1").content == "回复0"
    assert agent.ask("介绍一下你自己", "s1").content == "回复1"

    messages = memory.get_session_history("s1").messages
    assert len(messages) == 4
    assert [m.content for m in messages] == [
        "你好",
        "回复0",
        "介绍一下你自己",
        "回复1",
    ]


def test_sessions_are_isolated():
    """不同 session_id 的记忆互不影响。"""
    agent, memory = _make_agent()
    agent.ask("只属于会话A", "A")
    agent.ask("只属于会话B", "B")

    assert [m.content for m in memory.get_session_history("A").messages] == [
        "只属于会话A",
        "回复0",
    ]
    assert [m.content for m in memory.get_session_history("B").messages] == [
        "只属于会话B",
        "回复1",
    ]


def test_lg_agent_executes_weather_tool(monkeypatch):
    """模型请求调用 get_weather 时，工具被执行，最终回复来自工具结果。"""
    # 用固定数据替换真实天气接口，保证测试离线、快速、可重复
    monkeypatch.setattr(
        "app.tools.weather._fetch_weather",
        lambda city: {"city": "北京", "weather": "晴", "temperature": 28},
    )
    settings = Settings(llm_provider="fake", max_history_messages=6)
    # 第一轮返回「要求调用工具」，第二轮（拿到工具结果后）返回最终回复
    fake_llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"city": "北京"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="北京今天：晴，最高 28°C。"),
            ]
        )
    )
    memory = MemoryManager(settings)
    agent = LangGraphAgent(fake_llm, get_available_tools(), memory, settings.system_prompt)

    reply = agent.ask("北京天气怎么样？", "s1")
    assert reply.content == "北京今天：晴，最高 28°C。"
    # 最终回复应被写入记忆；带 tool_calls 的中间消息被净化后不落记忆
    messages = memory.get_session_history("s1").messages
    assert len(messages) == 2
    assert messages[-1].content == "北京今天：晴，最高 28°C。"


def test_build_lg_agent_falls_back_for_fake_model():
    """fake 模型不支持工具绑定时应回退为普通对话，行为不变。"""
    settings = Settings(llm_provider="fake", max_history_messages=6)
    memory = MemoryManager(settings)
    agent = build_lg_agent(settings, memory, tools=get_available_tools())

    reply = agent.ask("你好", "s1")
    assert reply.content == "（演示模式）你好！我是离线模拟回复。设置 LLM_PROVIDER=openai 并填入 API Key 后即可接入真实模型。"
    assert len(memory.get_session_history("s1").messages) == 2