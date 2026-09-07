"""模型工厂模块：根据配置创建对应的聊天模型实例。

所有提供商统一返回 langchain_core 的 BaseChatModel 抽象类型，
上层（prompt / chain）只依赖该抽象，因此切换提供商无需改动任何业务代码。
"""

from __future__ import annotations

from itertools import cycle

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import Settings

# 离线演示模式的固定回复（循环使用），用于无网络、无 Key 时验证完整流程
_FAKE_REPLIES = [
    "（演示模式）你好！我是离线模拟回复。设置 LLM_PROVIDER=openai 并填入 API Key 后即可接入真实模型。",
    "（演示模式）这是第二条模拟回复。你可以试试 /clear 清空记忆、/history 查看历史。",
    "（演示模式）当前输出由 GenericFakeChatModel 提供，用于在无网络、无 Key 时验证完整链路。",
]

# create_llm 函数根据配置创建对应的聊天模型实例 (-> 返回类型是 BaseChatModel)
# 所有提供商统一返回 langchain_core 的 BaseChatModel 抽象类型，
# 上层（prompt / chain）只依赖该抽象，因此切换提供商无需改动任何业务代码。
def create_llm(settings: Settings) -> BaseChatModel:
    """按配置创建聊天模型实例。

    Args:
        settings: 应用配置。

    Returns:
        一个可用的 BaseChatModel 实例。

    Raises:
        ValueError: 配置了未知的 llm_provider。
    """
    if settings.llm_provider == "fake":
        # 离线演示：循环返回固定回复，不发起任何网络请求
        return GenericFakeChatModel(
            messages=cycle(AIMessage(content=text) for text in _FAKE_REPLIES)
        )

    if settings.llm_provider == "ollama":
        # Ollama 通过 OpenAI 兼容端点暴露 API：用 ChatOpenAI 指向本地端口即可
        return ChatOpenAI(
            model=settings.ollama_model,
            api_key="ollama",  # Ollama 不校验 Key，占位即可
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if settings.llm_provider == "openai":
        # OpenAI 官方或任意兼容服务（DeepSeek / Moonshot / vLLM / OneAPI 等）
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    # 理论上不可达（pydantic Literal 已约束），但保留兜底便于排查
    raise ValueError(f"未知的 llm_provider: {settings.llm_provider}")


def describe_llm(settings: Settings) -> str:
    """生成当前模型配置的文字描述（供 /model 命令展示）。"""
    if settings.llm_provider == "fake":
        return "provider=fake（离线演示模式，不调用真实模型）"
    if settings.llm_provider == "ollama":
        return (
            f"provider=ollama | model={settings.ollama_model}"
            f" | base_url={settings.ollama_base_url}"
        )
    base = settings.openai_base_url or "https://api.openai.com/v1"
    key = settings.openai_api_key[:8] + "..." if settings.openai_api_key else "(未配置)"
    return (
        f"provider=openai | model={settings.openai_model}"
        f" | base_url={base} | api_key={key}"
    )
