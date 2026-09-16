"""应用配置模块：集中管理所有可配置项。

使用 pydantic-settings 从「环境变量」与「.env 文件」加载配置，
加载优先级：真实环境变量 > .env 文件 > 代码中的默认值。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 支持的模型提供商：
#   openai —— 任意 OpenAI 兼容服务（OpenAI / DeepSeek / Moonshot / vLLM / Ollama 等）
#   ollama —— 本地 Ollama 服务（通过其 OpenAI 兼容端点访问）
#   fake   —— 离线演示模式，不调用任何真实模型（无需 API Key）
ProviderName = Literal["openai", "ollama", "fake"]


class Settings(BaseSettings):
    """应用配置模型（字段名与 .env 中的变量名一一对应）。"""

    model_config = SettingsConfigDict(
        env_file=".env",             # 自动读取项目根目录下的 .env 文件
        env_file_encoding="utf-8",
        extra="ignore",              # 忽略未声明但存在于环境中的其它变量
    )

    # ---------------- 模型提供商 ----------------
    llm_provider: ProviderName = "fake"

    # ---------------- OpenAI 兼容接口（provider=openai 时生效） ----------------
    openai_api_key: str = ""             # API Key，例如 sk-xxxxxxxx
    openai_base_url: str | None = None   # 默认 OpenAI 官方；DeepSeek 填 https://api.deepseek.com/v1
    openai_model: str = "gpt-4o-mini"    # 模型名，DeepSeek 可填 deepseek-chat

    # ---------------- Ollama 本地模型（provider=ollama 时生效） ----------------
    ollama_base_url: str = "http://localhost:11434/v1"  # Ollama 的 OpenAI 兼容端点
    ollama_model: str = "qwen2.5:7b"                     # 需先用 `ollama pull` 拉取

    # ---------------- 生成参数 ----------------
    temperature: float = 0.7         # 采样温度：越高越发散，越低越确定
    max_tokens: int | None = 2048    # 单次回复的最大 token 数

    # ---------------- 记忆 ----------------
    max_history_messages: int = 20   # 单会话最多保留的消息条数（超出裁剪最旧消息）
    memory_db_path: str = ""         # 记忆持久化 SQLite 文件路径；留空 = 纯内存（重启即失）

    # ---------------- 会话 ----------------
    session_id: str = "default"      # 当前会话 ID，用于隔离不同会话的记忆

    # ---------------- 系统提示词 ----------------
    system_prompt: str = (
        "你是一个乐于助人、知识渊博的 AI 助手。"
        "请用简洁、准确、友好的中文回答用户的问题。"
    )


@lru_cache 
# @lru_cache  同一参数的调用结果会被缓存，避免重复解析
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例（带缓存，避免重复解析 .env 文件）。"""
    # 触发获取配置，确保缓存被填充
    return Settings()
