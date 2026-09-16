"""记忆持久化（SQLite）的离线测试。

验证核心承诺：写入 → 重建实例（模拟进程重启）→ 历史完整恢复；
clear 同步删除库中数据；未配置 db_path 时保持纯内存（不产生文件）；
持久化不影响窗口裁剪行为。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.memory import MemoryManager


def _settings(tmp_path, **kwargs) -> Settings:
    """构造启用持久化的测试配置（SQLite 文件放进 pytest 临时目录）。"""
    return Settings(
        llm_provider="fake",
        memory_db_path=str(tmp_path / "memory.db"),
        **kwargs,
    )


def test_roundtrip_after_restart(tmp_path):
    """写入 → 用全新实例（模拟进程重启）→ 消息完整恢复。"""
    settings = _settings(tmp_path, max_history_messages=20)
    m1 = MemoryManager(settings)
    m1.get_session_history("s1").add_messages(
        [HumanMessage(content="你好"), AIMessage(content="你好呀")]
    )

    # 全新 MemoryManager + 全新历史对象 = 模拟进程重启后的首次访问
    m2 = MemoryManager(settings)
    messages = m2.get_session_history("s1").messages
    assert [m.content for m in messages] == ["你好", "你好呀"]
    # 消息类型也应正确恢复（Human / AI），而非退化为同一种类型
    assert messages[0].type == "human"
    assert messages[1].type == "ai"


def test_clear_deletes_persisted_row(tmp_path):
    """clear 后重建实例，历史应为空（库中的行已被删除）。"""
    settings = _settings(tmp_path)
    m1 = MemoryManager(settings)
    m1.get_session_history("s1").add_messages([HumanMessage(content="临时消息")])
    m1.clear("s1")

    m2 = MemoryManager(settings)
    assert m2.get_session_history("s1").messages == []


def test_no_db_path_stays_in_memory(tmp_path):
    """未配置 db_path 时不产生任何文件（纯内存，老行为不变）。"""
    settings = Settings(llm_provider="fake", memory_db_path="")
    manager = MemoryManager(settings)
    manager.get_session_history("s1").add_messages([HumanMessage(content="只在内存")])

    # 临时目录里不应出现任何文件（没建库、没写盘）
    assert list(tmp_path.iterdir()) == []


def test_window_still_applies_with_db(tmp_path):
    """持久化不影响窗口裁剪：超限仍按 max_messages 裁剪后落库。"""
    settings = _settings(tmp_path, max_history_messages=4)
    m1 = MemoryManager(settings)
    history = m1.get_session_history("s1")
    history.add_messages([HumanMessage(content=str(i)) for i in range(6)])

    # 未注入压缩 llm → 硬裁剪为最近 4 条
    assert len(history.messages) == 4

    # 重启后恢复的应是被裁剪后的形态（2/3/4/5），而不是全部 6 条
    m2 = MemoryManager(settings)
    restored = m2.get_session_history("s1").messages
    assert [m.content for m in restored] == ["2", "3", "4", "5"]


def test_sessions_are_isolated_on_disk(tmp_path):
    """持久化后不同会话仍然隔离：各自恢复各自的历史。"""
    settings = _settings(tmp_path)
    m1 = MemoryManager(settings)
    m1.get_session_history("A").add_messages([HumanMessage(content="会话A的消息")])
    m1.get_session_history("B").add_messages([HumanMessage(content="会话B的消息")])

    m2 = MemoryManager(settings)
    assert [m.content for m in m2.get_session_history("A").messages] == ["会话A的消息"]
    assert [m.content for m in m2.get_session_history("B").messages] == ["会话B的消息"]
