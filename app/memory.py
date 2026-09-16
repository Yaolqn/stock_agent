"""会话记忆模块：按会话 ID 隔离记忆，带滑动窗口裁剪、记忆压缩与 SQLite 持久化。

持久化：配置 memory_db_path（.env 的 MEMORY_DB_PATH）时，会话历史写入
SQLite 文件，程序重启后自动恢复；未配置（默认空串）时使用进程内存储
（InMemory），程序重启后历史清空。两种模式下对外接口完全一致。

记忆压缩：消息超限时不再「硬丢」最旧消息，而是交给 llm 把「旧摘要 +
即将过期的消息」提炼成一段新摘要继续保留，历史 = 摘要 + 最近完整消息。
未注入 llm 时退化为纯窗口裁剪（老行为），保证演示/测试不受影响。
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from app.config import Settings
from app.prompts import build_compress_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# ---- SQLite 持久化辅助 ----
# 表结构：一行 = 一个会话（session_id 为主键）。summary 存累计摘要，
# messages 存序列化（JSON）后的完整消息列表。窗口裁剪/压缩始终在内存
# 中完成后，才把最终形态整体落库，而不是每追加一条就写一行。
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS chat_history ("
    "  session_id TEXT PRIMARY KEY,"
    "  summary    TEXT NOT NULL DEFAULT '',"
    "  messages   TEXT NOT NULL DEFAULT '[]'"
    ")"
)


def _execute(db_path: str, sql: str, params: tuple = ()) -> None:
    """执行一条写语句并提交（连接即开即关，避免句柄残留）。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _ensure_table(db_path: str) -> None:
    """建库建表（幂等）：SQLite 文件与表不存在时自动创建。"""
    _execute(db_path, _SCHEMA)


def _delete_session(db_path: str, session_id: str) -> None:
    """从 SQLite 中删除指定会话的持久化行（配合 clear 使用）。"""
    _execute(db_path, "DELETE FROM chat_history WHERE session_id = ?", (session_id,))


class WindowedChatMessageHistory(BaseChatMessageHistory):
    """带滑动窗口 + 可选记忆压缩 + 可选 SQLite 持久化的会话历史。

    当消息数量超过 max_messages 时：
      · 有 llm —— 最旧一批消息先压缩成摘要再移除（信息只浓缩、不丢弃）；
      · 无 llm —— 退回纯窗口裁剪，直接丢弃最旧超额消息。

    持久化：db_path 与 session_id 同时配置时启用——初始化即从库中恢复
    该会话历史，之后每次变更都把最新形态整体写回，重启不丢。
    """

    # max_messages 为会话历史记录的最大消息数，默认 100 条
    # keep_recent 为每次压缩后保留的完整消息条数（太新、太细的不压）
    # _messages 为实际存储的消息列表，按时间顺序排列
    def __init__(
        self,
        max_messages: int,
        llm: BaseChatModel | None = None,
        keep_recent: int = 20,
        db_path: str | None = None,     # SQLite 文件路径；None = 纯内存
        session_id: str | None = None,  # 会话 ID（持久化的行键）
    ) -> None:
        self._max_messages = max_messages
        self._llm = llm          # 压缩模型；None 表示关闭压缩
        self._keep_recent = keep_recent
        self._summary = ""       # 累计摘要（旧信息的浓缩）
        self._messages: list[BaseMessage] = []
        # 持久化开关：db_path + session_id 齐备才启用；
        # 初始化即建表并从库中恢复该会话已有历史（重启后记忆不丢）
        self._db_path = db_path
        self._session_id = session_id
        if db_path and session_id:
            _ensure_table(db_path)
            self._load_from_db()

    @property
    # 把方法 messages 转换为属性 messages，使外部代码更方便访问历史消息
    # 这样可以避免每次调用 messages 都返回一个新的列表，提高效率
    def messages(self) -> list[BaseMessage]:
        """当前会话给模型看的全部消息 = 摘要（如有）+ 最近完整消息。"""
        if not self._summary:
            return self._messages
        # 摘要以 SystemMessage 垫在最前，模型把它当背景知识
        return [
            SystemMessage(content=f"对话历史摘要：{self._summary}")
        ] + self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """追加消息；超限时优先压缩最旧消息，否则硬裁剪。"""
        self._messages.extend(messages)
        if len(self._messages) > self._max_messages:
            if self._llm is None:
                # 没有压缩模型 → 老逻辑：直接丢弃最旧超额消息
                self._messages = self._messages[-self._max_messages :]
            else:
                # 有压缩模型 → 把最旧的一批提炼成摘要，而不是硬丢
                # 取出「除了最后 keep_recent 条以外的全部消息」，作为压缩候选
                overflow = self._messages[: -self._keep_recent]
                if overflow:
                    # 有压缩候选 → 调用压缩模型提炼新摘要
                    self._summary = self._compress(self._summary, overflow)
                    # 压缩后，移除压缩候选，只保留最近 keep_recent 条完整消息
                    self._messages = self._messages[-self._keep_recent :]
        # 持久化：无论是否触发裁剪/压缩，都把处理后的最新形态整体落库
        self._save()

    # 压缩模型调用：把「旧摘要 + 即将过期的消息」压成一段新摘要
    def _compress(
        self, summary: str, transcript: list[BaseMessage], limit: int = 300
    ) -> str:
        """用模型把「旧摘要 + 即将过期的消息」压成一段新摘要。"""
        # 只取消息文本用于压缩；工具调用细节、内部字段不参与提炼
        text = "\n".join(str(m.content) for m in transcript)
        # 构建压缩提示模板，包含旧摘要和即将过期的消息
        prompt = build_compress_prompt()
        # 调用压缩模型，返回新摘要（截断至 limit 字符）
        # 注意：这里假设压缩模型返回的是文本，而不是结构化数据（如 JSON）
        reply = self._llm.invoke(prompt.format(summary=summary, transcript=text))  # type: ignore[misc]
        # 返回压缩摘要（截断至 limit 字符）
        # 注意：这里假设压缩模型返回的是文本，而不是结构化数据（如 JSON）
        return str(reply.content)[:limit]

    # ---- SQLite 持久化（db_path + session_id 齐备时生效） ----
    def _load_from_db(self) -> None:
        """从 SQLite 恢复该会话的累计摘要与消息列表。"""
        conn = sqlite3.connect(self._db_path or "")
        try:
            row = conn.execute(
                "SELECT summary, messages FROM chat_history WHERE session_id = ?",
                (self._session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return  # 新会话：库里还没有该行，从空历史开始
        # 反序列化：JSON → BaseMessage 列表（Human/AI/System 均可往返）
        self._summary = row[0]
        self._messages = messages_from_dict(json.loads(row[1]))

    def _save(self) -> None:
        """把「摘要 + 消息列表」整体写回 SQLite；未启用持久化时跳过。"""
        if not (self._db_path and self._session_id):
            return
        # ensure_ascii=False：中文按原文存储，库文件可直接阅读
        payload = json.dumps(messages_to_dict(self._messages), ensure_ascii=False)
        _execute(
            self._db_path,
            "INSERT OR REPLACE INTO chat_history (session_id, summary, messages)"
            " VALUES (?, ?, ?)",
            (self._session_id, self._summary, payload),
        )

    def clear(self) -> None:
        """清空全部消息（含累计摘要）。"""
        self._messages.clear()
        self._summary = ""
        # 有持久化 → 同步删除库中的行，避免重启后旧历史“复活”
        if self._db_path and self._session_id:
            _delete_session(self._db_path, self._session_id)


class MemoryManager:
    """多会话记忆管理器。

    以 session_id 为键维护多个相互隔离的会话记忆。
    get_session_history 的签名与 LangChain 官方的会话历史工厂约定一致
    （LangChain 1.x 弃用 RunnableWithMessageHistory 后，本项目的
    LangGraph Agent 直接调用该方法读写指定会话的历史）。
    """

    def __init__(
        self, settings: Settings, llm: BaseChatModel | None = None
    ) -> None:
        self._settings = settings
        # llm 用于记忆压缩；None 时各会话退化为纯窗口裁剪
        self._llm = llm
        # 持久化路径：settings 未配置（空串）时退化为纯进程内存
        self._db_path = settings.memory_db_path or None
        # 会话历史仓库：任意新会话 ID 首次访问时自动创建空历史
        # （配置了 SQLite 时，新建历史会同时从库中恢复旧消息）
        self._histories: dict[str, WindowedChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取（必要时创建）指定会话的历史记录。"""
        if session_id not in self._histories:
            self._histories[session_id] = WindowedChatMessageHistory(
                self._settings.max_history_messages,
                llm=self._llm,
                db_path=self._db_path,
                session_id=session_id,  # 传给历史对象，作为持久化的行键
            )
        return self._histories[session_id]

    def clear(self, session_id: str) -> None:
        """清空指定会话的记忆。"""
        self._histories.pop(session_id, None)
        # 有持久化 → 同步删除库中的行，避免重启后旧历史复活
        if self._db_path:
            _delete_session(self._db_path, session_id)

    @property
    def session_ids(self) -> list[str]:
        """当前存在的全部会话 ID。"""
        return sorted(self._histories)