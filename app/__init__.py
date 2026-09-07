"""LangChain AI 聊天助手 —— 应用包。

包结构：
    config.py  配置加载（pydantic-settings + .env）
    models.py  模型工厂（OpenAI 兼容 / Ollama / 离线演示）
    prompts.py Prompt 模板
    memory.py  会话记忆（滑动窗口裁剪）
    chain.py   对话链组装（RunnableWithMessageHistory）
    cli.py     终端交互循环
"""

__version__ = "1.0.0"
