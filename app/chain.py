"""聊天链模块：将 Prompt、LLM、记忆（与可选工具）组装成一条可调用的链。

支持两种模式：
  - 无工具：prompt | llm 的直接 LCEL 管道（普通多轮对话）；
  - 有工具：在对话基础上增加「模型自主调用工具」的能力（Agent 循环）：
    模型决定要调用工具 → 执行工具 → 把结果回传给模型 → 得到最终回复。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool

from app.memory import MemoryManager

# 模型在一轮对话中最多连续调用工具的轮数（防止陷入工具调用死循环）
_MAX_TOOL_ROUNDS = 5

# 带工具调用的对话链（轻量 Agent 循环）
class ToolAgentRunnable(Runnable[dict, BaseMessage]):
    """带工具调用的对话链（轻量 Agent 循环）。

    对每一次用户提问，工作流程如下：
      1. 按「系统提示词 + 历史消息 + 用户输入」拼好初始消息列表；
      2. 让模型作答；若模型返回 tool_calls（想调用工具），则依次执行
         对应工具，并把工具结果作为 ToolMessage 追加进消息列表，然后
         再让模型继续作答；
      3. 循环直至模型给出最终回复（不再调用工具）或达到轮数上限。
    """
    # 初始化时绑定提示模板、模型、工具
    def __init__(
        self,
        prompt: ChatPromptTemplate,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
    ) -> None:
        self._prompt = prompt
        self._model = model
        self._tools = {tool.name: tool for tool in tools}

    # 运行时根据用户输入与历史消息，生成模型回复
    def _run(self, user_input: str, history: list[BaseMessage]) -> BaseMessage:
        # 历史消息由 RunnableWithMessageHistory 注入（不含本轮新输入）
        # messages 是系统提示词 + 历史消息 + 用户本轮输入拼好后的消息列表
        messages = self._prompt.invoke(
            {"history": history, "input": user_input}
        ).to_messages()

        for _ in range(_MAX_TOOL_ROUNDS):
            # 让模型作答 invoke（包含 tool_calls）
            # AIMessage对象 ：content(模型回复), tool_calls(模型要求调用的工具调用记录), usage_metadata(计费)
            ai_message = self._model.invoke(messages)
            # 把模型回复追加进消息列表
            messages.append(ai_message)
            if not ai_message.tool_calls:  # 模型决定直接回答 → 结束循环
                return ai_message

            # 模型要求调用工具 → 执行并回传结果
            # call 是模型要求调用的工具调用记录，包含工具名称、参数、ID 等信息
            # args 是工具参数，需要被模型解析为 Python 字典
            for call in ai_message.tool_calls:  # 模型要求调用工具 → 执行并回传结果
                try:
                    # 执行工具，获取结果
                    # name由模型指定，对应 self._tools 中的工具实例
                    result = self._tools[call["name"]].invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 —— 工具失败也要把原因交还给模型
                    result = f"工具「{call['name']}」执行失败：{exc}"
                # 把工具执行结果作为 ToolMessage 追加进消息列表
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )
        return messages[-1]  # 达到轮数上限，返回最后一轮模型输出


    def invoke(self, input: dict, config: RunnableConfig | None = None, **kwargs) -> BaseMessage:  # noqa: A002
        # 必须经由 _call_with_config 走 LangChain 的 run 生命周期：
        # RunnableWithMessageHistory 的 _exit_history（回写记忆）依赖
        # with_listeners 注册的回调，直接执行会跳过它导致记忆不被写回。
        return self._call_with_config(self._invoke_impl, input, config, **kwargs)

    def _invoke_impl(self, input: dict, config: RunnableConfig | None = None) -> BaseMessage:  # noqa: A002
        return self._run(input["input"], list(input.get("history", [])))


def build_chain(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    memory: MemoryManager,
    tools: Sequence[BaseTool] | None = None,
) -> RunnableWithMessageHistory:
    """组装多轮对话链。

    每次调用的工作流程：
      1. RunnableWithMessageHistory 根据 session_id 从 memory 取出历史消息；
      2. 历史消息自动注入 prompt 的 history 占位符，与用户输入一起送入 LLM；
      3. 模型输出返回的同时，本轮「用户输入 + 模型回复」被写回 memory。

    Args:
        llm:    聊天模型实例。
        prompt: 聊天提示模板。
        memory: 会话记忆管理器。
        tools:  可选工具列表。传入了工具时，模型可自主调用这些工具
                （Agent 循环）；不传则退化为普通对话。

    Returns:
        支持流式输出与多轮记忆的可运行链。
    """
    if tools:
        # 把工具绑定给模型；若模型不支持工具调用则回退为普通对话
        try:
            # 把工具绑定给模型，使模型能直接调用工具
            # bind_tools 方法返回一个新的模型实例，包含工具调用能力，告诉模型哪些工具可调用
            model = llm.bind_tools(list(tools))
        except NotImplementedError:
            # 模型不支持工具调用，回退为普通对话
            model = llm
        # 组装带工具调用的 Runnable 《重点》
        runnable = ToolAgentRunnable(prompt, model, list(tools))
    else:
        runnable = prompt | llm  # LCEL：提示模板 → 大模型
    return RunnableWithMessageHistory(
        runnable,
        memory.get_session_history,
        input_messages_key="input",      # 用户输入取自输入字典的该字段
        history_messages_key="history",  # 历史消息注入 prompt 的该占位符
    )


def stream_reply(
    chain: RunnableWithMessageHistory,
    user_input: str,
    session_id: str,
) -> Iterator[str]:
    """流式生成回复，逐块产出文本片段。

    Args:
        chain:      组装好的对话链。
        user_input: 用户本轮输入。
        session_id: 会话 ID，决定使用哪一份记忆。

    Yields:
        模型输出的文本片段（调用方负责打印与拼接）。
    """
    config = {"configurable": {"session_id": session_id}}
    for chunk in chain.stream({"input": user_input}, config=config):
        # 文本模型的 content 是 str；部分多模态模型可能是列表，这里只处理文本
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content
