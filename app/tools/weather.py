"""天气工具：查询指定城市的实时天气与未来预报。

数据来源为 uapis.cn 的免费天气 API
（文档：https://uapis.cn/docs/api-reference/get-misc-weather，无需申请 key）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

# uapis.cn 天气接口：GET 请求，city 传城市名即可，无需认证
_WEATHER_API_URL = "https://uapis.cn/api/v1/misc/weather"
_REQUEST_TIMEOUT = 10  # 秒


class _CityNotFoundError(RuntimeError):
    """指定的城市在天气数据源中不存在。"""


def _fetch_weather(city: str) -> dict[str, Any] | None:
    """调用天气接口，返回原始 JSON 字段。

    Returns:
        解析后的 JSON 字典；网络/服务故障时返回 None。
        城市不存在时抛出 _CityNotFoundError。

    Raises:
        _CityNotFoundError: 城市在数据源中不存在（接口返回 404）。
    """
    query = urllib.parse.urlencode(
        {"city": city, "extended": "true", "forecast": "true"}
    )
    try:
        with urllib.request.urlopen(
            f"{_WEATHER_API_URL}?{query}", timeout=_REQUEST_TIMEOUT
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # 城市查不到 → 接口返回 404
            raise _CityNotFoundError(f"城市不存在：{city}") from exc
        return None  # 其它 HTTP 错误，视为服务故障
    except Exception:  # noqa: BLE001 —— 网络抖动等临时故障
        return None


def _parse_target_date(date_desc: str) -> str | None:
    """把「今天/明天/后天/YYYY-MM-DD」解析为具体日期，无法解析返回 None。"""
    today = date.today()
    offset = {"今天": 0, "明天": 1, "后天": 2}.get(date_desc.strip())
    if offset is not None:
        return str(today + timedelta(days=offset))
    try:
        return str(date.fromisoformat(date_desc.strip()))
    except ValueError:
        return None


def _format_weather(city: str, date_desc: str, data: dict[str, Any]) -> str:
    """把接口返回的 JSON 转成一段自然语言描述。"""
    city_name = data.get("city") or city
    # 若用户指定了具体日期（如明天），优先从多天预报里取对应日期的天气
    target = _parse_target_date(date_desc)
    if target is not None:
        for item in data.get("forecast", []):
            if str(item.get("date")) == target:
                return (
                    f"{city_name} {date_desc}：{item.get('weather_day', '')}，"
                    f"气温 {item.get('temp_min')}~{item.get('temp_max')}°C，"
                    f"{item.get('wind_dir_day', '')} {item.get('wind_scale_day', '')}。"
                )
        if target != str(date.today()):
            return f"抱歉，暂无「{city_name}」在 {date_desc} 的天气预报（最多可查未来 7 天）。"
    # 实时天气（默认情况）
    parts = [f"{city_name}当前：{data.get('weather', '未知')}"]
    temperature = data.get("temperature")
    if temperature is not None:
        parts.append(f"气温 {temperature}°C")
    if data.get("temp_min") is not None and data.get("temp_max") is not None:
        parts.append(f"今日 {data.get('temp_min')}~{data.get('temp_max')}°C")
    wind = f"{data.get('wind_direction', '')} {data.get('wind_power', '')}".strip()
    if wind:
        parts.append(f"{wind}风")
    if data.get("humidity") is not None:
        parts.append(f"湿度 {data.get('humidity')}%")
    if data.get("aqi_category"):
        parts.append(f"空气质量 {data.get('aqi_category')}")
    return "；".join(parts) + "。"


@tool
def get_weather(city: str, date: str = "今天") -> str:
    """查询指定城市在指定日期的天气情况（实时天气与未来预报）。

    Args:
        city: 城市名称，例如「北京」「上海」「广州」「深圳」。
        date: 日期描述，支持「今天」「明天」「后天」或形如 2026-09-04 的具体日期。
              默认今天（返回实时天气）。
    """
    try:
        data = _fetch_weather(city)
    except _CityNotFoundError:
        return (
            f"抱歉，没有查询到「{city}」的天气数据，"
            "请确认城市名称后重试（例如「北京」「上海」「广州」）。"
        )
    if data is None:
        return f"抱歉，天气服务暂时不可用，暂时无法查询「{city}」的天气，请稍后再试。"
    return _format_weather(city, date, data)


def get_tools() -> list[BaseTool]:
    """返回本模块提供的工具列表（供工具包统一聚合）。"""
    return [get_weather]