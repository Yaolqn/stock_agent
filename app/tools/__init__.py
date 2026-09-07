"""工具包：集中注册所有可供模型自主调用的工具。

新增一个工具的标准步骤：
  1. 在 app/tools/ 下新建模块（如 xxx.py），用 @tool 装饰器定义工具；
  2. 在模块末尾实现 get_tools()，返回本模块的工具列表；
  3. 在下方 _TOOL_MODULES 中登记该模块（导入一次即可）。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools import stock  # noqa: F401 —— 登记股票工具模块
from app.tools import weather  # noqa: F401 —— 登记天气工具模块

# 所有工具模块在此登记；新增模块时在 get_available_tools 里同步聚合
_TOOL_MODULES = (weather, stock)


def get_available_tools() -> list[BaseTool]:
    """返回当前全部可用工具（供链组装时绑定给模型）。"""
    tools: list[BaseTool] = []
    for module in _TOOL_MODULES:
        tools.extend(module.get_tools())
    return tools