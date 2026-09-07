"""总控协调官：多 Agent 股票投研的「Supervisor」节点。

Supervisor 是唯一的指挥者：它本身仍是一个 LangGraphAgent（大脑 call_model
⇄ 手脚 action 的状态机），但它的"手脚"只有一类：
  专家工具（把各专家 agent 封装成可调用的工具）——把任务委派给专家

关键约束：Supervisor 只负责调度与决策，绝不经手数据。所有数据获取必须
由专家中转（专家内部才绑定数据工具）。因此 supervisor 不绑定任何数据工具。

调度机制完全复用单 agent 已实现的 Send 并行：supervisor 大脑可同时发起
多个专家 tool_call（比如既让选股师扫描、又让分析师深挖某只股），LangGraph
自动为每个调用开一条 action 分支并行执行，结果合并回消息流。

本文件同时提供把专家 agent 封装为工具的工具函数 wrap_expert_as_tool。
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.agent import LangGraphAgent
from app.config import Settings
from app.memory import MemoryManager
from app.models import create_llm

# 总控协调官的领域提示词：只调度决策，绝不经手数据（数据一律由专家中转）
STOCK_SUPERVISOR_PROMPT = (
    "你是股票投研系统的总控协调官（Supervisor）。你负责拆解用户需求、调度专家完成投研任务，"
    "并汇总专家结论给用户。\n"
    "你手上唯一的能力是专家工具：\n"
    "  - scan（智能选股师）：扫描全市场、筛选强势股；\n"
    "  - analyze（多维分析师）：深度分析单只股票；\n"
    "  - intel（市场情报官）：调研消息面/新闻热点/舆情；\n"
    "  - decision（投资决策官）：综合各方给出最终投资评价；\n"
    "  - risk（风险管理官）：风险审查 + 一票否决。\n"
    "铁律：你绝不直接查询行情、历史K线等数据，也没有数据工具可用。"
    "所有数据获取必须通过专家完成——用户任何涉及「股价、行情、走势、选股、分析某只股」的诉求，"
    "都要交给相应专家去完成。\n"
    "调度原则（按诉求类型分级，避免过度调度）：\n"
    "  1. 只需「查看/分析走势、行情、消息」等描述性诉求——只派对应专家，用完即答，不要再排后面的决策/风控：\n"
    "     - 查实时行情/量价（如「某股现价多少」）交给 analyze，必要时辅以 scan；\n"
    "     - 只分析走势/K线（如「分析贵州茅台走势」）交给 analyze；\n"
    "     - 只看消息面（如「茅台最近有什么消息」）交给 intel。\n"
    "  2. 需要「选股、从全市场找强势股」——交给 scan。\n"
    "  3. 需要「最终投资建议/评级/能不能买」这类决策诉求——才收集各方结论后依次交给 decision 定夺，外加 risk 风险把关。\n"
    "  4. 用户一次性提了多个关注点时，才可同时委派多个专家并行工作，再综合各自结论回答。\n"
    "核心：你的调度要贴合问题本身，别把「描述性咨询」升级成「完整投研」。用户只问走势就只答走势，不必自动跑全套决策。\n"
    "回答用简洁、准确、友好的中文。"
)


class _ExpertTask(BaseModel):
    """专家工具的入参：一段交给专家的任务描述。"""

    task: str = Field(description="交给该专家的具体任务描述")


def wrap_expert_as_tool(name: str, description: str, expert: LangGraphAgent) -> BaseTool:
    """把一个专家 agent 封装成 supervisor 可调用的工具。

    专家内部是完整的 LangGraphAgent（有自己的大脑/手脚/数据访问），
    封装后 supervisor 只需把用户任务串给它，它独立完成并回传文本结论。

    Args:
        name: 工具名（supervisor 大脑据此识别该调用谁）。
        description: 工具用途说明（帮助 supervisor 决策何时调用）。
        expert: 已组装的专家 agent。
    """
    expert_id = f"expert_{name}"

    def _run(task: str) -> str:
        print(f"\n  ▶ [专家·{name}] 收到任务，开始执行…（独立会话 {expert_id}）")
        reply = expert.ask(task, expert_id)  # 独立会话，避免与 supervisor 主会话串史
        print(f"  ✔ [专家·{name}] 执行完成。")
        return reply.content if isinstance(reply, BaseMessage) else str(reply)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=_ExpertTask,
    )


def build_supervisor(
    settings: Settings,
    memory: MemoryManager,
    expert_tools: list[BaseTool] | None = None,
) -> LangGraphAgent:
    """组装总控协调官（Supervisor）。

    只绑定专家工具（绝不绑数据工具），确保使用者不会绕过专家直接拿数据。

    Args:
        settings: 应用配置。
        memory: 会话记忆。
        expert_tools: 已封装的专家工具列表。缺省时无任何工具（退化为纯对话）。
    """
    llm = create_llm(settings)
    tools: list[BaseTool] = []
    if settings.llm_provider in ("openai", "ollama") and expert_tools:
        tools = list(expert_tools)
    if tools:
        # 绑定专家工具；模型不支持时回退纯对话
        try:
            model = llm.bind_tools(tools)
        except NotImplementedError:
            model = llm
    else:
        model = llm
    return LangGraphAgent(model, tools, memory, STOCK_SUPERVISOR_PROMPT)