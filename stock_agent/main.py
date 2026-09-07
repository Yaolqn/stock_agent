"""stock_agent —— 终端入口（总控协调官 Supervisor）。

用法：
    python -m stock_agent.main
    .\.venv\Scripts\python.exe -m stock_agent.main

入口固定为「总控协调官」（Supervisor），它只负责调度与决策，不直接碰数据。
所有行情/选股/分析请求，总控会委派给其调度的专家 Agent（智能选股师 /
多维分析师）代为完成；专家内部才绑定数据工具。

用 `/help` 查看命令；用 `/exit` 或按 Ctrl+C 退出。
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.memory import MemoryManager
from app.models import describe_llm
from stock_agent.experts import build_expert_tools
from stock_agent.pipeline import StockResearchPipeline
from stock_agent.supervisor import build_supervisor


BANNER = f"""
{'=' * 58}
  股票投研 Agent（stock_agent）
  · 总控协调官：灵活问答，会把股票问题委派给专家
  · 投研流水线：/report 诉求 → 完整研报（筛选→分析→决策→风控）
  输入 /help 查看命令；输入 /exit 或按 Ctrl+C 退出
{'=' * 58}
"""

HELP_TEXT = """可用命令：
  /report 诉求   对某只股票（或选股诉求）跑一条完整投研流水线
                   例：/report 分析一下贵州茅台600519，值得买吗
  /tools        查看总控可调度的专家工具
  /clear        清空当前会话的记忆
  /history      查看当前会话的历史消息
  /model        查看当前模型配置
  /help         显示本帮助
  /exit         退出程序（或按 Ctrl+C / Ctrl+D）
"""

# 提问引导：教用户怎么用词，让流水线/总控能正确触发专家取数
GUIDE_TEXT = """【提问小贴士】
  查单只股票（不需要选股）→ 用总控协调官直接提问，最省：
    问：贵州茅台今天涨了吗?
    问：分析一下北方稀土600111的走势
    
  → 注意：如果你已经指定单只股票，就用上面的直接提问，不要用 /report
    （/report 适合"从全市场选股 + 多只逐一深挖"这类需要筛选的诉求）。
    若仍用 /report 且带了明确代码，流水线会自动跳过选股师这一步。

  真正的选股诉求（需要全市场筛选）→ 才用 /report：
    /report 帮我筛选今天成交额最大的强势股
    /report 帮我从全市场挑出近期最强的几只股票并逐一分析

  提示：带上 6 位股票代码（如 600519），专家能更精准地取数。

{'-' * 58}"""


def _ensure_utf8_stdio() -> None:
    """确保 Windows 终端能正确显示中文。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _check_akshare() -> None:
    """检查 akshare 是否可用；缺失时给出明确指引而非让专家"假性失败"。

    注意：本项目依赖装在 .venv，需用 `.venv\\Scripts\\python.exe` 运行，
    否则全局 Python 可能缺少 akshare 导致股票工具全部失效。
    """
    try:
        import akshare  # noqa: F401

        return
    except ImportError:
        print("\n[环境错误] 未检测到股票数据依赖 akshare，行情/扫描/深度分析将无法工作。")
        print("  请使用项目虚拟环境运行：")
        print("    .\\venv\\Scripts\\python.exe -m stock_agent.main")
        print("  或安装依赖：")
        print("    .\\venv\\Scripts\\python.exe -m pip install akshare\n")


def main() -> None:
    settings = get_settings()
    _ensure_utf8_stdio()

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("[提示] 未检测到 OPENAI_API_KEY，真实对话将报错；")
        print("       请先在 .env 中配置 Key，或用 LLM_PROVIDER=fake 体验演示模式。")

    # 校验股票数据依赖是否可用，缺失时给出明确指引（避免专家"假性失败"）
    _check_akshare()

    # 初始化记忆管理器：负责存储会话历史消息
    memory = MemoryManager(settings)
    # 专家工具：把智能选股师 / 多维分析师封装为总控可调度的工具。
    # 每个专家在 build_expert_tools 内部拥有完全独立的记忆空间，
    # 与总控 memory 隔离，杜绝多 agent 上下文相互污染。
    expert_tools = build_expert_tools(settings)
    # 总控协调官：只绑专家工具，不绑任何数据工具
    agent = build_supervisor(settings, memory, expert_tools=expert_tools)
    tools = agent._tools if hasattr(agent, "_tools") else {}
    # 投研流水线：跑完整研报（筛选→分析→决策→风控）
    pipeline = StockResearchPipeline(settings)

    print(BANNER)
    print(HELP_TEXT)
    print(GUIDE_TEXT)
    print(f"[总控] 可调度的专家工具：{', '.join(tools.keys()) if tools else '无（fake 演示模型）'}")

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
            low = user_input.lower()
            if low in ("/exit", "/quit"):
                print("再见！")
                break
            elif low == "/help":
                print(HELP_TEXT)
                print(GUIDE_TEXT)
            elif low == "/clear":
                memory.clear(session_id)
                print("已清空当前会话记忆。")
            elif low == "/history":
                messages = memory.get_session_history(session_id).messages
                if not messages:
                    print("（暂无历史消息）")
                else:
                    for msg in messages:
                        print(f"[{msg.type}] {msg.content}")
            elif low == "/model":
                print(describe_llm(settings))
            elif low == "/tools":
                if tools:
                    print("总控可调度的专家工具：")
                    for name in tools:
                        print(f"  - {name}")
                else:
                    print("未启用专家工具（fake 演示模型不支持工具调用）。")
            elif low.startswith("/report"):
                query = user_input[len("/report"):].strip()
                if not query:
                    print("[引导] /report 后面需要带上你的诉求，例如：\n  /report 分析一下贵州茅台600519，值得买吗")
                    continue
                print(f"\n[流水线] 收到诉求：{query}")
                print("[流水线] 开始投研：筛选 → 分析 → 决策 → 风控…\n")
                try:
                    # 执行投研流水线
                    reports = pipeline.run(query)
                except Exception as exc:  # noqa: BLE001 —— 流水线内部异常兜底
                    print(f"\n[错误] 流水线执行失败：{exc}")
                    print("[提示] 请检查网络 / API Key 配置即可，输入 /model 查看；流水线中间状态可能已写入专家记忆。")
                    continue
                # 带阶段的引导渲染
                print(f"\n{'=' * 58}")
                print(f"  投研究报 · 诉求：{query}")
                print(f"{'=' * 58}")
                print("—— ① 选股师（全市场筛选，产出候选）——")
                print(reports.get("screener", "（空）")[:1500])
                print("\n—— ② 分析师（对候选逐股深挖）——")
                print(reports.get("analyst", "（空）")[:1500])
                print("\n—— ③ 情报官（消息面 / 舆情）——")
                print(reports.get("intel", "（空）")[:1200])
                print("\n—— ④ 决策官（最终投资评价）——")
                print(reports.get("decision", "（空）")[:1500])
                print("\n—— ⑤ 风险官（一票否决 · 最终把关）——")
                print(reports.get("risk", "（空）")[:1200])
                print(f"\n{'=' * 58}")
                print("[提示] 结论仅供参考，不构成投资建议。输入 /report 新诉求 或直接提问继续。")
            else:
                print(f"未知命令：{user_input}，输入 /help 查看帮助。")
            continue

        # 普通输入：交给总控协调官灵活问答
        print("总控协调官 > ", end="", flush=True)
        try:
            # 流式输出：模型开口回答时逐字打印（工具阶段仅显示"正在调用"由 agent 内部跳过）
            collected: list[str] = []
            for token in agent.ask_stream(user_input, session_id):
                print(token, end="", flush=True)
                collected.append(token)
            print()
            if not collected:
                print("（总控未返回可显示内容）")
        except Exception as exc:  # noqa: BLE001 —— 交互程序需要兜底所有异常
            print(f"\n[错误] {exc}")
            print("[提示] 请检查 API Key / Base URL / 网络（代理）配置；输入 /model 查看当前配置。")


if __name__ == "__main__":
    main()