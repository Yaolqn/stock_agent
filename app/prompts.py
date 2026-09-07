"""Prompt 模板模块：定义系统提示词与对话消息模板。"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import Settings


def build_prompt(settings: Settings) -> ChatPromptTemplate:
    """构建聊天提示模板。

    模板包含三部分：
      1. system  —— 系统提示词，设定助手的角色与行为准则；
      2. history —— 历史消息占位符，由 RunnableWithMessageHistory 自动注入；
      3. human   —— 用户本轮输入。

    Args:
        settings: 应用配置（用于读取 system_prompt）。

    Returns:
        组装好的 ChatPromptTemplate。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", settings.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

# 压缩模型提示模板：把「旧摘要 + 即将过期的消息」压成一段新摘要
def build_compress_prompt() -> ChatPromptTemplate:
    """构建记忆压缩提示模板。

    当会话历史超限时，用模型把「旧摘要 + 即将过期的消息」浓缩成一段
    更精炼的新摘要，替代直接丢弃最旧消息（详见 app/memory.py）。
    """
    return ChatPromptTemplate.from_template(
        "你是对话记忆管理员。把「旧摘要」和「新对话」合并，提炼成一段"
        "更精炼的新摘要。要求：保留用户的身份信息、长期偏好、关键结论；"
        "丢弃客套话、临时性细节和工具执行噪音；用中文，不超过 300 字。\n\n"
        "旧摘要：{summary}\n\n"
        "新对话（即将过期）：\n{transcript}\n\n"
        "新摘要："
    )
