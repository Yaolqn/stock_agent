"""链与记忆的离线测试（不调用任何真实模型、不访问网络）。"""

from __future__ import annotations

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.chain import build_chain
from app.config import Settings
from app.memory import MemoryManager
from app.prompts import build_prompt
from app.tools import get_available_tools


def _make_chain(max_history: int = 6):
    """构造一条使用假模型的完整链（离线可用）。"""
    settings = Settings(llm_provider="fake", max_history_messages=max_history)
    fake_llm = GenericFakeChatModel(
        messages=cycle(AIMessage(content=f"回复{i}") for i in range(100))
    )
    memory = MemoryManager(settings)
    chain = build_chain(fake_llm, build_prompt(settings), memory)
    return chain, memory


def _ask(chain, text: str, session_id: str = "s1") -> str:
    """向链发送一条消息，返回拼接后的完整回复。"""
    config = {"configurable": {"session_id": session_id}}
    parts = []
    for chunk in chain.stream({"input": text}, config=config):
        if isinstance(chunk.content, str):
            parts.append(chunk.content)
    return "".join(parts)


def test_prompt_contains_required_variables():
    """Prompt 模板应包含 input 与 history 两个变量。"""
    settings = Settings(llm_provider="fake")
    prompt = build_prompt(settings)
    assert "input" in prompt.input_variables
    assert "history" in prompt.input_variables


def test_two_round_conversation_writes_history():
    """连续两轮对话后，记忆中应保存 4 条消息（2 用户 + 2 AI）。"""
    chain, memory = _make_chain()
    assert _ask(chain, "你好") == "回复0"
    assert _ask(chain, "介绍一下你自己") == "回复1"

    messages = memory.get_session_history("s1").messages
    assert len(messages) == 4
    assert messages[0].content == "你好"
    assert messages[1].content == "回复0"
    assert messages[2].content == "介绍一下你自己"
    assert messages[3].content == "回复1"


def test_window_trims_oldest_messages():
    """超过窗口上限时，最旧的消息应被裁剪。"""
    chain, memory = _make_chain(max_history=4)
    for i in range(4):
        _ask(chain, f"问题{i}")

    messages = memory.get_session_history("s1").messages
    assert len(messages) == 4
    assert messages[0].content == "问题2"  # 问题0、回复0、问题1、回复1 已被裁剪
    assert messages[-1].content == "回复3"


def test_sessions_are_isolated():
    """不同 session_id 的记忆互不影响。"""
    chain, memory = _make_chain()
    _ask(chain, "只属于会话A", session_id="A")
    _ask(chain, "只属于会话B", session_id="B")

    history_a = memory.get_session_history("A").messages
    history_b = memory.get_session_history("B").messages
    assert len(history_a) == 2
    assert len(history_b) == 2
    assert history_a[0].content == "只属于会话A"
    assert history_b[0].content == "只属于会话B"


def test_clear_resets_memory():
    """清空后记忆应归零。"""
    chain, memory = _make_chain()
    _ask(chain, "你好")
    memory.clear("s1")
    assert len(memory.get_session_history("s1").messages) == 0


def test_chain_with_tools_still_works_for_fake_model():
    """传入工具但模型不支持工具调用时，应回退为普通对话，行为不变。"""
    settings = Settings(llm_provider="fake", max_history_messages=6)
    fake_llm = GenericFakeChatModel(
        messages=cycle(AIMessage(content=f"回复{i}") for i in range(100))
    )
    memory = MemoryManager(settings)
    chain = build_chain(
        fake_llm, build_prompt(settings), memory, tools=get_available_tools()
    )
    assert _ask(chain, "你好") == "回复0"
    assert len(memory.get_session_history("s1").messages) == 2


def test_tool_agent_executes_weather_tool(monkeypatch):
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
    chain = build_chain(
        fake_llm, build_prompt(settings), memory, tools=get_available_tools()
    )

    assert _ask(chain, "北京天气怎么样？") == "北京今天：晴，最高 28°C。"
    # 最终回复应被写入记忆（工具调用过程不落记忆，属预期设计）
    messages = memory.get_session_history("s1").messages
    assert len(messages) == 2
    assert messages[-1].content == "北京今天：晴，最高 28°C。"
