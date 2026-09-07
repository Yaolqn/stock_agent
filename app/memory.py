"""会话记忆模块：按会话 ID 隔离记忆，带滑动窗口裁剪与记忆压缩。

默认使用进程内存储（InMemory），程序重启后历史清空。
如需持久化，可参考 LangChain 官方提供的 Redis / SQLite / Postgres
等实现，替换 MemoryManager 内部工厂即可（接口不变）。

记忆压缩：消息超限时不再「硬丢」最旧消息，而是交给 llm 把「旧摘要 +
即将过期的消息」提炼成一段新摘要继续保留，历史 = 摘要 + 最近完整消息。
未注入 llm 时退化为纯窗口裁剪（老行为），保证演示/测试不受影响。
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage

from app.config import Settings
from app.prompts import build_compress_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class WindowedChatMessageHistory(BaseChatMessageHistory):
    """带滑动窗口 + 可选记忆压缩的会话历史。

    当消息数量超过 max_messages 时：
      · 有 llm —— 最旧一批消息先压缩成摘要再移除（信息只浓缩、不丢弃）；
      · 无 llm —— 退回纯窗口裁剪，直接丢弃最旧超额消息。
    """

    # max_messages 为会话历史记录的最大消息数，默认 100 条
    # keep_recent 为每次压缩后保留的完整消息条数（太新、太细的不压）
    # _messages 为实际存储的消息列表，按时间顺序排列
    def __init__(
        self,
        max_messages: int,
        llm: BaseChatModel | None = None,
        keep_recent: int = 20,
    ) -> None:
        self._max_messages = max_messages
        self._llm = llm          # 压缩模型；None 表示关闭压缩
        self._keep_recent = keep_recent
        self._summary = ""       # 累计摘要（旧信息的浓缩）
        self._messages: list[BaseMessage] = []

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
        if len(self._messages) <= self._max_messages:
            return
        if self._llm is None:
            # 没有压缩模型 → 老逻辑：直接丢弃最旧超额消息
            self._messages = self._messages[-self._max_messages :]
            return
        # 有压缩模型 → 把最旧的一批提炼成摘要，而不是硬丢
        # 取出「除了最后 keep_recent 条以外的全部消息」，作为压缩候选
        overflow = self._messages[: -self._keep_recent]
        if overflow:
            # 有压缩候选 → 调用压缩模型提炼新摘要
            self._summary = self._compress(self._summary, overflow)
            # 压缩后，移除压缩候选，只保留最近 keep_recent 条完整消息
            self._messages = self._messages[-self._keep_recent :]

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

    def clear(self) -> None:
        """清空全部消息（含累计摘要）。"""
        self._messages.clear()
        self._summary = ""


class MemoryManager:
    """多会话记忆管理器。

    以 session_id 为键维护多个相互隔离的会话记忆。
    get_session_history 的签名与 LangChain 的
    RunnableWithMessageHistory 回调要求完全一致。
    """

    def __init__(
        self, settings: Settings, llm: BaseChatModel | None = None
    ) -> None:
        self._settings = settings
        # llm 用于记忆压缩；None 时各会话退化为纯窗口裁剪
        self._llm = llm
        # defaultdict 保证任意新会话 ID 首次访问时自动创建空历史
        self._histories: dict[str, WindowedChatMessageHistory] = defaultdict(
            lambda: WindowedChatMessageHistory(
                settings.max_history_messages, llm=llm
            )
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取（必要时创建）指定会话的历史记录。"""
        return self._histories[session_id]

    def clear(self, session_id: str) -> None:
        """清空指定会话的记忆。"""
        self._histories.pop(session_id, None)

    @property
    def session_ids(self) -> list[str]:
        """当前存在的全部会话 ID。"""
        return sorted(self._histories)