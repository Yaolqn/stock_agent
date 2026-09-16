"""Prompt 模板模块：定义记忆压缩等提示模板。

说明：LangChain 1.x 已弃用 RunnableWithMessageHistory，原先基于 LCEL
对话链（app/chain.py）的 build_prompt 随链一起移除。当前对话统一由
LangGraph 状态机 Agent（app/agent）驱动，消息拼装直接在 Agent 内部完成，
因此本模块只保留被 app/memory.py 使用的压缩提示模板。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_compress_prompt() -> ChatPromptTemplate:
    """构建记忆压缩提示模板。

    当会话历史超限时，用模型把「旧摘要 + 即将过期的消息」浓缩成一段
    更精炼的新摘要，替代直接丢弃最旧消息（详见 app/memory.py）。

    Returns:
        组装好的 ChatPromptTemplate（含 {summary} 与 {transcript} 两个占位符）。
    """
    return ChatPromptTemplate.from_template(
        "你是对话记忆管理员。把「旧摘要」和「新对话」合并，提炼成一段"
        "更精炼的新摘要。要求：保留用户的身份信息、长期偏好、关键结论；"
        "丢弃客套话、临时性细节和工具执行噪音；用中文，不超过 300 字。\n\n"
        "旧摘要：{summary}\n\n"
        "新对话（即将过期）：\n{transcript}\n\n"
        "新摘要："
    )
