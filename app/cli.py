"""终端交互模块：REPL 循环、命令解析、流式输出。

对话统一由 LangGraph 状态机 Agent（app/agent.LangGraphAgent）驱动。
LangChain 1.x 已弃用 RunnableWithMessageHistory，原先基于 LCEL 手写
工具循环的 app/chain.py 已移除，本模块与 stock_agent 复用同一套 Agent
内核，保证整个项目只有一份「大脑 ⇄ 手脚」的对话实现。

命令说明：
    /help      显示帮助
    /clear     清空当前会话的记忆
    /history   查看当前会话的历史消息
    /model     查看当前模型配置
    /exit      退出程序（或按 Ctrl+C / Ctrl+D）
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent import LangGraphAgent, build_lg_agent
from app.config import Settings, get_settings
from app.memory import MemoryManager
from app.models import create_llm, describe_llm
from app.tools import get_available_tools

PROJECT_NAME = "LangChain AI 聊天助手"
VERSION = "1.0.0"

BANNER = f"""
{'=' * 58}
  {PROJECT_NAME}  v{VERSION}
  基于 LangGraph 状态机的终端多轮对话机器人
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
    """打印指定会话的历史消息。

    Args:
        memory: 会话记忆管理器。
        session_id: 会话 ID，决定读取哪一份记忆。
    """
    messages = memory.get_session_history(session_id).messages
    if not messages:
        print("（暂无历史消息）")
        return
    for msg in messages:
        # 用 isinstance 判断消息类型（流式写回的 *Chunk 子类 type 与基类不同）
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
    """处理斜杠命令；返回 False 表示应当退出程序。

    Args:
        command: 用户输入的完整命令字符串。
        memory: 会话记忆管理器。
        session_id: 当前会话 ID。
        settings: 应用配置（供 /model 展示模型信息）。

    Returns:
        True 表示继续交互循环，False 表示退出。
    """
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


def _stream_answer(agent: LangGraphAgent, user_input: str, session_id: str) -> None:
    """流式调用 Agent 并逐字打印回复；异常时给出友好提示。

    Args:
        agent: 已组装的 LangGraph Agent。
        user_input: 用户本轮输入。
        session_id: 当前会话 ID。
    """
    print("AI > ", end="", flush=True)
    collected: list[str] = []
    try:
        # ask_stream 逐 token 产出最终回复；工具调用阶段不产生文字，自动跳过
        for piece in agent.ask_stream(user_input, session_id):
            print(piece, end="", flush=True)
            collected.append(piece)
        print()
        if not collected:
            print("（模型未返回可显示内容）")
    except Exception as exc:  # noqa: BLE001 —— 交互程序需要兜底所有异常
        print(f"\n[错误] {exc}")
        print("[提示] 请检查 API Key / Base URL / 网络（代理）配置；输入 /model 查看当前配置。")


def run_repl(
    agent: LangGraphAgent,
    memory: MemoryManager,
    settings: Settings,
) -> None:
    """主交互循环：读取用户输入 → 执行命令或调用模型 → 输出回复。

    Args:
        agent: 已组装的 LangGraph Agent。
        memory: 会话记忆管理器。
        settings: 应用配置（会话 ID、模型信息等）。
    """
    # 确保终端能正确显示中文（兼容旧版 cmd）
    _ensure_utf8_stdio()
    # 从配置中获取会话 ID
    session_id = settings.session_id

    # 打印欢迎 banner 与帮助文本
    print(BANNER)
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
        # 普通输入：交给 Agent 流式作答
        _stream_answer(agent, user_input, session_id)


def main() -> None:
    """程序入口：加载配置 → 创建模型 → 组装 LangGraph Agent → 进入交互循环。"""
    settings = get_settings()

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("[提示] 未检测到 OPENAI_API_KEY，真实对话将报错；")
        print("       请先在 .env 中配置 Key，或用 LLM_PROVIDER=fake 体验演示模式。")

    try:
        # 创建聊天模型实例（类型为 BaseChatModel）
        llm = create_llm(settings)
    except Exception as exc:  # noqa: BLE001 —— 初始化失败给出可操作提示
        print(f"[错误] 模型初始化失败：{exc}")
        print("[提示] 请检查 .env 中的 LLM_PROVIDER 与相关 Key 配置。")
        raise SystemExit(1) from exc

    # 记忆管理器；真实模型注入 llm 开启记忆压缩，fake 演示保持纯窗口裁剪
    memory = MemoryManager(
        settings,
        llm=llm if settings.llm_provider in ("openai", "ollama") else None,
    )
    # 只有真实模型（openai / ollama）支持工具调用；fake 演示模型不支持
    tools = get_available_tools() if settings.llm_provider in ("openai", "ollama") else None
    # 组装 LangGraph 状态机 Agent（大脑 call_model ⇄ 手脚 action）
    agent = build_lg_agent(settings, memory, tools=tools)
    if tools:
        print("[提示] 已启用工具：get_weather（可直接问天气，例如「北京今天天气怎么样？」）")
    # 进入交互循环
    run_repl(agent, memory, settings)


if __name__ == "__main__":
    main()
