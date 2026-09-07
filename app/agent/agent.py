"""LangGraph 版 Agent：图定义、节点实现与对外接口（Agent 包核心）。

与 app/chain.py 的手写 Agent 循环相比，本模块用 LangGraph 的状态机
方式声明「大脑 → 手脚 → 回到大脑」的编排，图结构一目了然，且天然
支持：递归上限、中途断点、按节点流式、checkpointer 持久化等能力。

四部件对应关系：
  ① 大脑(call_model 节点)  —— 调用模型，产出 AIMessage（可能含 tool_calls）
  ② 手脚(action 节点)      —— 执行一个工具，产出 ToolMessage；用 LangGraph 的
                              Send API 并行扇出：一个工具调用开一条独立分支
  ③ 记忆(MemoryManager)    —— 复用 app.memory，会话内消息读写
  ④ 循环(图边缘)          —— call_model →(有工具调用)→ Send 扇出 action 分支
                                    └──(无工具调用 / 失败超限)──→ END
                            action 分支执行完自动合并回 call_model
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.agent.state import AgentState
from app.config import Settings
from app.memory import MemoryManager
from app.models import create_llm

# 单轮对话中「模型↔工具」往返的最大次数（防止死循环）
_MAX_ROUNDS = 5
# 工具失败后的自愈轮数上限：允许模型「看到错误 → 修正」几次，超限直接收场
_RETRY_LIMIT = 2
# 对应到 LangGraph 的递归深度上限。Send 并行扇出时每个 action 分支都单独
# 计数，因此按「每轮 1 次大脑 + 最多 8 个工具分支」的余量估算
_RECURSION_LIMIT = _MAX_ROUNDS * 8 + 10


def _to_context_message(message: BaseMessage) -> BaseMessage:
    """把最终回复整理成可写进历史的形态。

    带 tool_calls 的 AIMessage 若直接写回记忆，下一轮喂给模型提供方时
    会因为缺少对应的 ToolMessage 而被拒绝（孤儿工具调用）。因此历史
    中只保留文本内容，丢弃工具调用信息。
    """
    if isinstance(message, AIMessage) and message.tool_calls:
        return AIMessage(content=message.content)
    if isinstance(message, ToolMessage):
        # 工具连续失败超限直接收场时，最终消息可能是 ToolMessage；
        # 转成普通文本 AIMessage，避免历史里出现找不到来源的孤儿 ToolMessage
        return AIMessage(content=message.content)
    return message


class LangGraphAgent:
    """基于 LangGraph 状态机的对话 Agent。

    每次 ask() 的工作流程（由编译后的图驱动）：
      1. 取出会话历史，连同系统提示词、本轮输入拼成初始消息列表；
      2. 图进入 call_model：模型作答，若返回 tool_calls 则 Send 并行扇出 action；
      3. action 分支各自执行一个工具，结果追加为 ToolMessage 并自动合并回 call_model；
      4. 直至模型直接回答（无 tool_calls）→ END；最终回复写回记忆。
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool],
        memory: MemoryManager,
        system_prompt: str,
    ) -> None:
        self._model = model
        self._tools = {tool.name: tool for tool in tools}
        self._memory = memory
        self._system_prompt = system_prompt
        # 编译好的图就是"完整 Agent"本体，之后一切对话都走它
        self._graph = self._build_graph()

    # ---------------- 图定义 ----------------

    def _build_graph(self) -> StateGraph:
        # 定义状态图，状态为 AgentState 类型
        graph = StateGraph(AgentState)
        # 添加节点：call_model, action
        graph.add_node("call_model", self._call_model)  # 大脑节点
        graph.add_node("action", self._take_action)  # 手脚节点
        # 添加边：START → call_model <入口→大脑>
        graph.add_edge(START, "call_model")
        # 条件边：无工具调用 → 结束；失败超限 → 结束；否则 → Send 并行扇出到 action
        graph.add_conditional_edges("call_model", self._route_tools)
        # 添加边：action 分支执行完自动合并回大脑，进入下一轮决策
        graph.add_edge("action", "call_model")
        # 返回编译后的图，供 ask() 调用
        return graph.compile()

    @property
    def graph(self) -> CompiledStateGraph:
        """编译好的状态图，供可视化 / 调试使用。"""
        return self._graph

    # ---------------- 节点（图的每个方框） ----------------

    # 大脑节点：让模型基于当前全部消息作答/提名工具。
    def _call_model(self, state: AgentState) -> dict:
        """大脑：让模型基于当前全部消息作答/提名工具。"""
        response = self._model.invoke(state["messages"])
        return {"messages": [response]}

    # 路由函数：读最新模型输出与失败计数，决定下一步走向。
    def _route_tools(self, state: AgentState) -> str | list[Send]:
        """路由：读最新模型输出与失败计数，决定下一步走向。

        无工具调用 → END；失败已超自愈上限(_RETRY_LIMIT) → END（不再扇出，
        避免模型反复撞同一个错）；否则每个工具调用发一条 Send 分支并行执行，
        分支子状态只携带 {"tool_call": call}，互不干扰。
        """
        # 读取最新模型输出（AIMessage）
        last = state["messages"][-1]
        # 没有工具调用，直接结束
        if not last.tool_calls:
            return END
        # 失败已超自愈上限(_RETRY_LIMIT) → END（不再扇出，
        # 避免模型反复撞同一个错）
        if state.get("retries", 0) > _RETRY_LIMIT:
            return END
        # 每个工具调用发一条 Send 分支并行执行，分支子状态只携带 {"tool_call": call}<载荷payload>，互不干扰(副本)
        # 每个 Send 分支执行一个工具，结果作为 ToolMessage 回传
        return [Send("action", {"tool_call": call}) for call in last.tool_calls]

    # 手脚节点：执行模型点名的工具，结果作为 ToolMessage 回传。
    def _take_action(self, state: AgentState) -> dict:
        """手脚：执行一个工具，结果作为 ToolMessage 回传。

        一条 Send 分支只处理一个工具调用（state 中只有 tool_call 字段）。
        失败的工具不会吞掉异常：错误文本同样写回 ToolMessage，让模型
        看到原因后「自己修正」（软重试）；失败计数通过 retries 返回 1，
        由 operator.add 合并器累加，供 _route_tools 判断是否该收场。
        """
        # 读取最新模型输出（AIMessage）
        # tool_call 是分支子状态携带的工具调用信息（state 中只有 tool_call 字段）,不是AgentState固定字段
        call = state["tool_call"]
        try:
            result = self._tools[call["name"]].invoke(call["args"])
            failed = False
        except Exception as exc:  # noqa: BLE001 —— 工具失败也把原因交还模型
            result = f"工具「{call['name']}」执行失败：{exc}"
            failed = True
        return {
            # 手脚节点返回：ToolMessage（包含工具调用结果result）
            # 失败计数：1（失败时累加），0（成功时不累加）
            "messages": [ToolMessage(content=str(result), tool_call_id=call["id"])],
            "retries": 1 if failed else 0,
        }

    # ---------------- 对外接口 ----------------

    def ask(self, user_input: str, session_id: str) -> AIMessage:
        """执行一轮完整对话：注入历史 → 图推理 → 写回记忆 → 返回回复。"""
        # 取出会话历史
        history = self._memory.get_session_history(session_id)
        # 初始消息 = 系统提示词 + 会话历史 + 本轮输入
        messages = [SystemMessage(content=self._system_prompt)]
        # 合并会话历史
        messages.extend(history.messages)
        # 追加本轮输入
        messages.append(HumanMessage(content=user_input))

        # 图推理：根据当前全部消息，让模型作答/提名工具
        result = self._graph.invoke(
            # 图推理输入：当前全部消息
            {"messages": messages},
            # 图推理配置：递归调用时递归深度
            config={"recursion_limit": _RECURSION_LIMIT},
        )
        final = result["messages"][-1]  # 最终回复（无 tool_calls 的 AIMessage）
        # 记忆写回：本轮「输入 + 回复」，回复中的工具调用信息被净化掉
        history.add_messages(
            [HumanMessage(content=user_input), _to_context_message(final)]
        )
        return final

    def ask_stream(self, user_input: str, session_id: str):
        """流式版 ask：逐 token 产出最终回复（生成器），供终端逐字显示。

        与 ask() 唯一区别：用 graph.stream(stream_mode="messages") 打开
        按 token 粒度的流，把模型真正开口回答的文字一段段 yield 出来。
        工具调用阶段（专家取数）不产生可显示文字，自然被跳过；等模型
        归纳出最终回答时才逐字出现。记忆写回与 ask() 完全一致。

        若底层模型不支持按消息流式（如 fake 演示模型），自动回退为
        一次性 invoke，保证不崩。
        """
        # 取出会话历史
        history = self._memory.get_session_history(session_id)
        # 初始消息 = 系统提示词 + 会话历史 + 本轮输入
        messages = [SystemMessage(content=self._system_prompt)]
        # 合并会话历史
        messages.extend(history.messages)
        # 追加本轮输入
        messages.append(HumanMessage(content=user_input))

        try:
            # 图推理 + 逐 token 流：stream_mode="messages" 只吐 LLM 文字块
            final_full = ""
            # 流式迭代：每个 chunk 是一段模型输出文字
            for chunk, _meta in self._graph.stream(
                {"messages": messages},
                # 流式配置：递归深度上限
                config={"recursion_limit": _RECURSION_LIMIT},
                # 流式模式：按消息粒度的 token 流
                stream_mode="messages",
            ):
                # 只收集「模型 AI 回答」的文本块；专家工具返回的结果块
                # （ToolMessageChunk，content 非空）会原样吐出、污染终端，必须跳过。
                # 用 isinstance 判断而非 type=='ai'：fake 模型的 chunk 其 type
                # 属性是类名而非 'ai'，isinstance 对真实模型与 fake 都成立。
                if (
                    isinstance(chunk, AIMessageChunk)
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    final_full += chunk.content
                    yield chunk.content
            # 记忆写回：本轮「输入 + 完整回复」；流式文本不含工具调用，直接存
            history.add_messages(
                [
                    # 记录用户输入
                    HumanMessage(content=user_input),
                    # 记录模型回复
                    AIMessage(content=final_full or "（无回复）"),
                ]
            )
        except Exception:  # noqa: BLE001 —— 不支持按消息流式时回退为一次性
            # 回退：一次性 invoke，把完整回复一次性 yield
            reply = self.ask(user_input, session_id)
            yield reply.content


def build_lg_agent(
    settings: Settings,
    memory: MemoryManager,
    tools: list[BaseTool] | None = None,
) -> LangGraphAgent:
    """组装 LangGraph 版 Agent。

    Args:
        settings: 应用配置（模型类型、系统提示词等）。
        memory:   会话记忆管理器。
        tools:    可选工具列表。传入时模型可自主调用（Agent 能力）；
                  否则退化为纯对话。

    Returns:
        编译好的 LangGraphAgent。
    """
    # 创建模型实例（同 chain.py 中的 create_llm）
    llm = create_llm(settings)
    if tools:
        # 给模型绑定工具清单；不支持工具调用的模型（如 fake）自动回退
        try:
            # 绑定工具清单，让模型知道有哪些工具可调用
            model = llm.bind_tools(list(tools))
        except NotImplementedError:
            model = llm
    else:
        model = llm
    return LangGraphAgent(
        model,
        list(tools) if tools else [],
        memory,
        settings.system_prompt,
    )
