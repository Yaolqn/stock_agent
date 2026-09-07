"""股票工具：查询 A 股单只股票的实时行情与历史日 K 线。

数据来源为免费开源库 akshare（新浪财经数据源，无需申请 key）。
安装：pip install akshare

选型说明：起初使用东方财富（em）系列接口，但被验证在此网络环境下
间歇性断开（限流）；新浪系列的 stock_zh_a_spot / stock_zh_a_daily
更为稳定，故本模块统一走新浪源。
"""

from __future__ import annotations

from datetime import datetime, time

from functools import lru_cache

from langchain_core.tools import BaseTool, tool


try:
    import akshare as ak
except ImportError:  # pragma: no cover —— 依赖未安装时给出提示而非崩溃
    ak = None


def _normalize_symbol(symbol: str) -> str:
    """把用户给的股票代码规整为 6 位纯数字（去掉 sh./sz./市场前缀）。"""
    s = symbol.strip().lower()
    for prefix in ("sh.", "sz.", "bj.", "sh", "sz", "bj"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in (".sh", ".sz", ".bj"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def _with_market(code: str) -> str:
    """为新浪接口补市场前缀：沪市(6/9开头)->sh，深市(0/3开)->sz。"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


@lru_cache(maxsize=1)
def _spot_snapshot() -> object:
    """拉取全市场实时快照(Sina)。返回 DataFrame，按行情字段命中过滤单只。"""
    return ak.stock_zh_a_spot()


def _fresh_spot() -> object:
    """刷新全市场快照。"""
    _spot_snapshot.cache_clear()
    return _spot_snapshot()


# 沪深 A 股连续竞价交易时段：09:30–11:30、13:00–15:00
_AM_START, _AM_END = time(9, 30), time(11, 30)
_PM_START, _PM_END = time(13, 0), time(15, 0)


@lru_cache(maxsize=1)
def _trade_dates() -> set[str]:
    """新浪交易日历：返回所有 A 股交易日（YYYY-MM-DD）的字符串集合。"""
    try:
        df = ak.tool_trade_date_hist_sina()
        return set(df["trade_date"].astype(str))
    except Exception:  # noqa: BLE001 —— 日历接口临时故障时退化为"不提示"
        return set()


def _market_context() -> str:
    """判断现在是否交易时间，返回一段提示文本（数据时效性说明）。

    规则：
      - 非交易日（周末/节假日）→ 提示休市，数据为最近交易日
      - 交易日但非交易时段 → 提示盘中未开盘/已收盘，数据为最近交易日
      - 交易时段内 → 提示实时
    """
    if ak is None:
        return ""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    is_trade_day = today in _trade_dates()
    if not is_trade_day:
        return "（注意：今日非交易日（周末/节假日休市），以下为最近交易日收盘数据。）"
    t = now.time()
    in_session = (_AM_START <= t <= _AM_END) or (_PM_START <= t <= _PM_END)
    if in_session:
        return "（盘中实时数据）"
    return "（注意：当前非交易时段，以下为最近交易日收盘数据。）"


@tool
def get_stock_quote(symbol: str) -> str:
    """查询 A 股单只股票的实时行情（最新价、涨跌幅、开高低、昨收等）。

    Args:
        symbol: 6 位股票代码，例如「600519」（贵州茅台）、「000001」（平安银行）。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    code = _normalize_symbol(symbol)
    if not (code.isdigit() and len(code) == 6):
        return f"股票代码「{symbol}」格式不正确，请提供 6 位数字代码，例如 600519。"
    try:
        df = _spot_snapshot()
    except Exception as exc:  # noqa: BLE001 —— 网络抖动等临时故障
        return f"抱歉，暂时无法获取实时行情（{exc}），请稍后再试。"
    # 新浪快照的「代码」列带市场前缀（sh600519 / sz000001），按带前缀形式匹配；
    # 找不到则再拉一次新鲜快照（可能缓存过期）
    target = _with_market(code)
    row = df[df["代码"].astype(str) == target]
    if row.empty:
        try:
            spot = _fresh_spot()
            row = spot[spot["代码"].astype(str) == target]
        except Exception:  # noqa: BLE001
            pass
    if row.empty:
        return f"抱歉，没有查询到股票「{code}」，请确认代码是否正确（如 600519）。"

    r = row.iloc[0]
    return (
        f"{code} {r['名称']} 最新价 {r['最新价']} 元，"
        f"涨跌幅 {r['涨跌幅']}%，今开 {r['今开']} 元，"
        f"最高 {r['最高']} 元，最低 {r['最低']} 元，昨收 {r['昨收']} 元。"
        f"{_market_context()}"
    )


@tool
def get_stock_history(symbol: str, start_date: str, end_date: str) -> str:
    """查询 A 股单只股票在指定日期区间内的历史日 K 线（收盘价与涨跌幅）。

    Args:
        symbol: 6 位股票代码，例如「600519」（贵州茅台）。
        start_date: 起始日期，格式 YYYY-MM-DD，例如 2024-01-01。
        end_date: 结束日期，格式 YYYY-MM-DD，例如 2024-12-31。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    code = _normalize_symbol(symbol)
    if not (code.isdigit() and len(code) == 6):
        return f"股票代码「{symbol}」格式不正确，请提供 6 位数字代码，例如 600519。"
    try:
        df = ak.stock_zh_a_daily(
            symbol=_with_market(code),
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # 前复权，便于观察真实涨跌
        )
    except Exception as exc:  # noqa: BLE001
        return f"抱歉，暂时无法获取「{code}」的历史行情（{exc}），请稍后再试。"
    if df is None or df.empty:
        return f"在 {start_date} 至 {end_date} 之间没有「{code}」的K线数据。"
    # 新浪 daily 不含涨跌幅列，按相邻交易日收盘价计算
    df = df.copy()
    df["pct"] = df["close"].pct_change() * 100
    lines = [
        f"{code} 最近 {len(df)} 个交易日（区间 {start_date}~{end_date}）概览："
    ]
    for _, r in df.tail(8).iterrows():
        pct = "" if r["pct"] != r["pct"] else f"{r['pct']:.2f}%"  # 首行无涨跌幅
        lines.append(f"{r['date']}：收盘 {r['close']} 元，涨跌幅 {pct}")
    return "；".join(lines) + "。"


@tool
def scan_market(top_n: int = 10, sort_by: str = "成交额") -> str:
    """扫描全市场 A 股，按量化指标返回当日强势股榜单。

    Args:
        top_n: 返回的股票数量，默认 10。
        sort_by: 排序依据，可选「成交额」（资金关注度）或「涨跌幅」（涨幅强弱），
                 默认按成交额排序。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    try:
        df = _spot_snapshot()
    except Exception as exc:  # noqa: BLE001
        return f"抱歉，暂时无法扫描全市场（{exc}），请稍后再试。"
    if df is None or df.empty:
        return "暂时没有获取到全市场行情数据。"

    # 过滤：排除无成交（停牌/未开盘）、异常代码（如指数/非A股）
    df = df[
        ~(df["代码"].astype(str).str.startswith(("bj",)))
        & (df["成交额"].astype(float) > 0)
    ].copy()

    if sort_by == "涨跌幅":
        df = df.sort_values("涨跌幅", ascending=False)
        label = "涨幅榜"
    else:
        df = df.sort_values("成交额", ascending=False)
        label = "成交额榜"
    df = df.head(top_n)

    lines = [f"全市场扫描 · {label} Top {top_n}（新浪数据）："]
    for _, r in df.iterrows():
        lines.append(
            f"{r['代码']} {r['名称']}：最新价 {r['最新价']} 元，"
            f"涨跌幅 {r['涨跌幅']}%，成交额 {r['成交额']/100000000:.1f} 亿"
        )
    return "；".join(lines) + f"。{_market_context()}"


@tool
def analyze_stock_deep(symbol: str) -> str:
    """对单只 A 股做多维度深度分析（实时行情 + 技术面指标）。

    基于近 ~120 个交易日的历史K线计算技术指标，
    与实时行情一起给出综合评价所需的数据素材。

    Args:
        symbol: 6 位股票代码，例如「600519」（贵州茅台）。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    code = _normalize_symbol(symbol)
    if not (code.isdigit() and len(code) == 6):
        return f"股票代码「{symbol}」格式不正确，请提供 6 位数字代码，例如 600519。"
    try:
        df = ak.stock_zh_a_daily(
            symbol=_with_market(code),
            start_date=None,              # 新浪接口缺省即返回尽量长区间
            end_date=None,
            adjust="qfq",
        )
    except Exception as exc:  # noqa: BLE001
        return f"抱歉，暂时无法分析「{code}」（{exc}），请稍后再试。"
    if df is None or df.empty:
        return f"没有「{code}」的历史数据，无法分析。"
    df = df.tail(120).copy()
    # 技术指标：均线、区间涨跌幅、20日波动率(年化)、量能对比
    close = df["close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    daily_ret = close.pct_change().dropna()
    vol_annual = daily_ret.std() * (252**0.5) * 100 if len(daily_ret) > 1 else float("nan")
    vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(5).mean() if len(df) >= 5 else float("nan")
    latest = close.iloc[-1]

    def fmt(v: float, suffix: str = "") -> str:
        return f"{v:.2f}{suffix}" if v == v else "—"

    return (
        f"{code} 深度分析（数据分析结果）：\n"
        f"- 最新收盘 {fmt(latest)} 元；近{len(df)}日区间涨跌幅 {fmt(period_return, '%')}\n"
        f"- 均线：MA5={fmt(ma5)}，MA20={fmt(ma20)}，MA60={fmt(ma60)}"
        f"（{'价上均线' if latest >= ma20 else '价破MA20'}）\n"
        f"- 20日年化波动率 {fmt(vol_annual, '%')}；当前量能/5日均量 {fmt(vol_ratio)}"
    )


@tool
def get_stock_fund_flow(symbol: str) -> str:
    """查询单只 A 股的近期资金流向（主力/大单净流入等）。

    数据来源 akshare(东方财富)的个股资金流，用于判断资金关注度与多空。

    Args:
        symbol: 6 位股票代码，例如「600519」（贵州茅台）。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    code = _normalize_symbol(symbol)
    if not (code.isdigit() and len(code) == 6):
        return f"股票代码「{symbol}」格式不正确，请提供 6 位数字代码，例如 600519。"
    try:
        df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith(("6", "9")) else "sz")
    except Exception as exc:  # noqa: BLE001 —— 东财接口限流时降级
        try:
            df = ak.stock_zh_a_spot_em()  # 备选：东财全市场快照（含主力净流入等）
        except Exception:  # noqa: BLE001
            df = None
    if df is None or not hasattr(df, "empty") or df.empty:
        return f"抱歉，暂时没有获取到「{code}」的资金流向数据，请稍后再试。"
    # 取最近一列含「净流入/主力」字段的行
    try:
        last = df.iloc[-1]
        keys = [c for c in df.columns if "净" in str(c) or "主力" in str(c)]
        flow = {k: last[k] for k in keys[:5]}
        flow_str = "，".join(f"{k}={v}" for k, v in flow.items())
    except Exception:  # noqa: BLE001
        flow_str = "（字段解析失败）"
    return f"{code} 最近资金流向（数据仅供参考）：{flow_str} (最新日期 {df.columns[0] if len(df.columns) else ''})。"


@tool
def get_stock_news(symbol: str) -> str:
    """查询单只 A 股最近的相关新闻标题。

    数据来源 akshare(东方财富)个股新闻；返回最近若干条标题供研判消息面。

    Args:
        symbol: 6 位股票代码，例如「600519」（贵州茅台）。
    """
    if ak is None:
        return "股票工具依赖未安装：请先执行 pip install akshare，然后再试。"
    code = _normalize_symbol(symbol)
    if not (code.isdigit() and len(code) == 6):
        return f"股票代码「{symbol}」格式不正确，请提供 6 位数字代码，例如 600519。"
    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as exc:  # noqa: BLE001
        return f"抱歉，暂时无法获取「{code}」的新闻（{exc}），请稍后再试。"
    if df is None or df.empty:
        return f"没有查询到「{code}」的相关新闻。"
    titles = [str(r.get("新闻标题")).strip() for _, r in df.head(8).iterrows()]
    return f"{code} 最近相关新闻：\n" + "\n".join(f"- {t}" for t in titles if t)


def get_tools() -> list[BaseTool]:
    """返回本模块提供的工具列表（供工具包统一聚合）。"""
    return [
        get_stock_quote,
        get_stock_history,
        scan_market,
        analyze_stock_deep,
        get_stock_fund_flow,
        get_stock_news,
    ]