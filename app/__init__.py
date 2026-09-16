"""LangChain AI 聊天助手 —— 应用包。

包结构：
    config.py  配置加载（pydantic-settings + .env）
    models.py  模型工厂（OpenAI 兼容 / Ollama / 离线演示）
    prompts.py 提示模板（记忆压缩）
    memory.py  会话记忆（滑动窗口裁剪 + 可选压缩 + SQLite 持久化）
    cli.py     终端交互循环（LangGraph Agent 驱动）
    agent/     LangGraph 状态机 Agent（对话核心，替代已移除的 chain.py）
    tools/     工具包（天气 / 股票等，供 Agent 自主调用）
"""

__version__ = "1.0.0"
