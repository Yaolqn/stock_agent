"""终端交互模块：REPL 循环、命令解析、流式输出。

不依赖任何前端或窗口 UI，直接在终端完成多轮对话。
"""

from __future__ import annotations

import sys
import warnings

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.chain import build_chain, stream_reply
from app.config import Settings, get_settings
from app.memory import MemoryManager
from app.models import create_llm, describe_llm
from app.prompts import build_prompt
from app.tools import get_available_tools

PROJECT_NAME = "LangChain AI 聊天助手"
VERSION = "1.0.0"

BANNER = f"""
{'=' * 58}
  {PROJECT_NAME}  v{VERSION}
  基于 LangChain 的终端多轮对话机器人
  输入 /help 查看命令；输入 /exit 或按 Ctrl+C 退出
{'=' * 58}
"""

HELP_TEXT = """可用命令：
  /help      显示本帮助
  /clear     清空当前会话的记忆
  /history   查看当前会话的历史消息
  /model     查看当前模型配置
  /exit      退出程序（或按 Ctrl+C / Ctrl+D）
"""


def _ensure_utf8_stdio() -> None:
    """确保 Windows 终端能正确显示中文（兼容旧版 cmd）。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # 非流式终端（如某些 IDE 内嵌控制台）无 reconfigure，忽略即可


def _print_history(memory: MemoryManager, session_id: str) -> None:
    """打印指定会话的历史消息。"""
    messages = memory.get_session_history(session_id).messages
    if not messages:
        print("（暂无历史消息）")
        return
    for msg in messages:
        # 流式调用写入记忆的是 *Chunk 子类（如 AIMessageChunk），
        # 其 type 与基类不同，因此用 isinstance 判断而非字符串比较
        if isinstance(msg, HumanMessage):
            role = "用户"
        elif isinstance(msg, AIMessage):
            role = "AI"
        elif isinstance(msg, SystemMessage):
            role = "系统"
        else:
            role = msg.type
        print(f"[{role}] {msg.content}")


def _handle_command(
    command: str,
    memory: MemoryManager,
    session_id: str,
    settings: Settings,
) -> bool:
    """处理斜杠命令；返回 False 表示应当退出程序。"""
    if command in ("/exit", "/quit"):
        print("再见！")
        return False
    if command == "/help":
        print(HELP_TEXT)
    elif command == "/clear":
        memory.clear(session_id)
        print("已清空当前会话记忆。")
    elif command == "/history":
        _print_history(memory, session_id)
    elif command == "/model":
        print(describe_llm(settings))
    else:
        print(f"未知命令：{command}，输入 /help 查看帮助。")
    return True


def _stream_answer(
    chain: RunnableWithMessageHistory,
    user_input: str,
    session_id: str,
) -> None:
    """流式调用模型并逐字打印回复；异常时给出友好提示。"""
    print("AI > ", end="", flush=True)
    try:
        for piece in stream_reply(chain, user_input, session_id):
            print(piece, end="", flush=True)
        print()
    except Exception as exc:  # noqa: BLE001 —— 交互程序需要兜底所有异常
        print(f"\n[错误] {exc}")
        print("[提示] 请检查 API Key / Base URL / 网络（代理）配置；输入 /model 查看当前配置。")


def run_repl(
    chain: RunnableWithMessageHistory,
    memory: MemoryManager,
    settings: Settings,
) -> None:
    """主交互循环：读取用户输入 → 执行命令或调用模型 → 输出回复。"""
    # 确保终端能正确显示中文（兼容旧版 cmd）
    _ensure_utf8_stdio()
    # 从配置中获取会话 ID
    session_id = settings.session_id

    # 打印欢迎 banner
    print(BANNER)
    # 打印帮助文本
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C：正常退出
            print("\n再见！")
            break

        if not user_input:  # 空输入直接跳过
            continue
        if user_input.startswith("/"):  # 斜杠命令
            if not _handle_command(user_input, memory, session_id, settings):
                break
            continue
        _stream_answer(chain, user_input, session_id)


def main() -> None:
    """程序入口：加载配置 → 创建模型 → 组装链 → 进入交互循环。"""
    settings = get_settings()

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("[提示] 未检测到 OPENAI_API_KEY，真实对话将报错；")
        print("       请先在 .env 中配置 Key，或用 LLM_PROVIDER=fake 体验演示模式。")

    try:
        # 返回一个类型为 BaseChatModel 的实例
        llm = create_llm(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 模型初始化失败：{exc}")
        print("[提示] 请检查 .env 中的 LLM_PROVIDER 与相关 Key 配置。")
        raise SystemExit(1) from exc

    # 构建提示模板
    prompt = build_prompt(settings)
    # 初始化记忆管理器；真实模型注入 llm 开启记忆压缩，fake 演示不注入
    memory = MemoryManager(
        settings,
        llm=llm if settings.llm_provider in ("openai", "ollama") else None,
    )
    # 只有真实模型（openai / ollama）支持工具调用；fake 演示模型不支持
    tools = get_available_tools() if settings.llm_provider in ("openai", "ollama") else None
    # 组装链
    chain = build_chain(llm, prompt, memory, tools=tools)
    if tools:
        print("[提示] 已启用工具：get_weather（可直接问天气，例如「北京今天天气怎么样？」）")

    run_repl(chain, memory, settings)


if __name__ == "__main__":
    main()
