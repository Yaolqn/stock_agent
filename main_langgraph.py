"""LangGraph 版 Agent —— 终端入口（与 main.py 平行的新板块）。

用法：
    python main_langgraph.py           # 直接运行（配置见 .env）

与 main.py 的区别：对话链改为 LangGraph 状态机实现（app/agent/），
模型「提案 → 执行工具 → 再提案」的循环由图来编排，原 langchain 手写
循环（app/chain.py）保持不动。
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings, get_settings
from app.agent import build_lg_agent
from app.memory import MemoryManager
from app.models import create_llm, describe_llm
from app.tools import get_available_tools

PROJECT_NAME = "LangChain AI 聊天助手（LangGraph 版）"
VERSION = "1.0.0"

BANNER = f"""
{'=' * 58}
  {PROJECT_NAME}  v{VERSION}
  状态机编排：大脑(call_model) ⇄ 手脚(action)，由 LangGraph 驱动
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
    """确保 Windows 终端能正确显示中文。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _print_history(memory: MemoryManager, session_id: str) -> None:
    """打印指定会话的历史消息。"""
    messages = memory.get_session_history(session_id).messages
    if not messages:
        print("（暂无历史消息）")
        return
    for msg in messages:
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


def main() -> None:
    """入口：加载配置 → 创建模型 → 组装 LangGraph 图 → 进入交互循环。"""
    settings = get_settings()
    _ensure_utf8_stdio()

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("[提示] 未检测到 OPENAI_API_KEY，真实对话将报错；")
        print("       请先在 .env 中配置 Key，或用 LLM_PROVIDER=fake 体验演示模式。")

    # 只有真实模型（openai / ollama）支持工具调用；fake 演示模型不支持
    tools = get_available_tools() if settings.llm_provider in ("openai", "ollama") else None
    # 注入压缩专用模型：真实模型开启「记忆压缩」，fake 演示保持纯窗口裁剪
    memory = MemoryManager(
        settings,
        llm=create_llm(settings) if settings.llm_provider in ("openai", "ollama") else None,
    )
    agent = build_lg_agent(settings, memory, tools=tools)

    print(BANNER)
    print(HELP_TEXT)
    if tools:
        print("[提示] 已启用工具：get_weather（可直接问天气，例如「北京今天天气怎么样？」）")

    session_id = settings.session_id
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.startswith("/"):
            if not _handle_command(user_input, memory, session_id, settings):
                break
            continue

        print("AI > ", end="", flush=True)
        try:
            reply = agent.ask(user_input, session_id)
            print(reply.content)
        except Exception as exc:  # noqa: BLE001 —— 交互程序需要兜底所有异常
            print(f"\n[错误] {exc}")
            print("[提示] 请检查 API Key / Base URL / 网络（代理）配置；输入 /model 查看当前配置。")


if __name__ == "__main__":
    main()