"""指数退避重试工具（对齐 Java 参考项目的 RetryUtil）。

- 可重试：超时 / 连接 / 网络 / 限流 / 502 / 503 / 504 等瞬时故障
- 不重试：invalid / unauthorized / forbidden / not found / authentication 等
  参数或鉴权类错误（重试无意义）
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from rag.config import RagSettings

T = TypeVar("T")

# 命中即【不重试】的关键词
_NO_RETRY_KEYWORDS = (
    "invalid",
    "unauthorized",
    "forbidden",
    "not found",
    "authentication",
)

# 命中即【重试】的关键词
_RETRY_KEYWORDS = (
    "timeout",
    "timed out",
    "connection",
    "network",
    "rate limit",
    "too many requests",
    "503",
    "502",
    "504",
)


def should_retry(exc: Exception) -> bool:
    """判断异常是否值得重试。"""
    message = str(exc).lower()
    if not message:
        # 无消息的异常默认重试（通常是网络层的瞬时错误）
        return True
    if any(keyword in message for keyword in _NO_RETRY_KEYWORDS):
        return False
    if any(keyword in message for keyword in _RETRY_KEYWORDS):
        return True
    # 默认重试
    return True


def get_friendly_error_message(exc: Exception) -> str:
    """把底层异常翻译成面向用户的中文提示。"""
    message = str(exc).lower()
    if not message:
        return "服务暂时不可用，请稍后重试"
    if "timeout" in message or "timed out" in message:
        return "服务响应超时，请稍后重试"
    if "connection" in message or "connect" in message:
        return "网络连接失败，请检查网络后重试"
    if "rate limit" in message or "too many requests" in message:
        return "服务繁忙，请稍后重试"
    if "unauthorized" in message or "authentication" in message:
        return "认证失败，请检查 API 密钥配置"
    if "invalid" in message:
        return "请求参数错误，请检查输入"
    return "服务暂时不可用，请稍后重试"


def execute_with_retry(
    operation: Callable[[], T],
    operation_name: str,
    settings: RagSettings,
) -> T:
    """执行带指数退避重试的操作。

    Args:
        operation: 无参可调用对象，返回值即为结果。
        operation_name: 操作名称，用于日志。
        settings: RAG 配置（提供重试开关与退避参数）。

    Returns:
        操作结果。

    Raises:
        Exception: 重试耗尽后抛出最后一次异常；不可重试异常立即抛出。
    """
    if not settings.retry_enabled:
        return operation()

    last_exc: Exception | None = None
    delay = settings.retry_initial_delay

    for attempt in range(1, settings.retry_max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 —— 重试工具需要捕获所有异常
            last_exc = exc
            if not should_retry(exc):
                raise
            if attempt < settings.retry_max_attempts:
                print(f"  [重试] {operation_name} 失败（第 {attempt} 次尝试），"
                      f"{delay}ms 后重试… 原因：{exc}")
                time.sleep(delay / 1000)
                delay = min(int(delay * settings.retry_multiplier),
                            settings.retry_max_delay)

    print(f"  [重试] {operation_name} 重试 {settings.retry_max_attempts} 次后仍失败")
    assert last_exc is not None
    raise last_exc
