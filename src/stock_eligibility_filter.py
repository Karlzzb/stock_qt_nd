import time
from pathlib import Path

import pandas as pd
from datetime import datetime
from typing import Optional

from tinyshare_auth import get_pro_api


def _get_pro():
    """延迟初始化 pro API，避免模块导入时因环境变量缺失而崩溃。"""
    return get_pro_api()


# 交易日之间 API 调用节流（秒）：tinyshare 有每分钟频次上限（429），
# 0.3s ≈ 200 次/分钟，留有余量；触发 429 时另有指数退避重试兜底
_API_THROTTLE_S = 0.3
_429_MAX_RETRIES = 6


def _st_cache_file(trade_date: str) -> Path:
    """ST 列表磁盘缓存路径（按交易日一个文件，空集也落盘以区分"未拉取"）。"""
    from config.settings import ST_FILTER_DATA_DIR
    d = Path(ST_FILTER_DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"st_{trade_date}.csv"


def _fetch_st_stocks(trade_date: str) -> set[str]:
    """拉取某日 ST 集合：磁盘缓存优先，API 调用带节流 + 429 指数退避重试。

    只有 API 成功返回才写缓存（空结果也写空文件），
    避免把限流失败误存成"当日无 ST"。
    """
    cache_file = _st_cache_file(trade_date)
    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype=str)
        return set(df["ts_code"]) if "ts_code" in df.columns else set()

    last_err: Exception | None = None
    for attempt in range(_429_MAX_RETRIES):
        try:
            time.sleep(_API_THROTTLE_S)
            df = _get_pro().stock_st(trade_date=trade_date)
            codes = set(df["ts_code"]) if df is not None and len(df) > 0 else set()
            pd.DataFrame({"ts_code": sorted(codes)}).to_csv(cache_file, index=False)
            return codes
        except Exception as e:  # noqa: BLE001 - 429 重试，其他异常直接抛
            last_err = e
            if "429" in str(e) and attempt < _429_MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise last_err  # pragma: no cover



class StockEligibilityFilter:
    """
    统一股票过滤器，共用于回测和实盘。

    过滤规则（可独立开关）：
    1. 主板过滤：只允许 ^[60] 开头
    2. ST 过滤：通过 Tushare stock_st API 获取指定日期的 ST 列表
    3. 次新股过滤：上市 < 100 天（基于 list_date 计算）
    """

    def __init__(self,
                 filter_main_board: bool = False,
                 filter_st: bool = False,
                 filter_new_stock: bool = False,
                 st_preloaded: dict[str, set[str]] | None = None):
        """
        Args:
            st_preloaded: 预加载的 ST 缓存，格式 {trade_date: set of ts_codes}，
                          用于 Phase2 批量回测，避免每个实例重复调 API。
        """
        self.filter_main_board = filter_main_board
        self.filter_st = filter_st
        self.filter_new_stock = filter_new_stock

        # 一次性缓存：{symbol: list_date}
        self._stock_basic: dict[str, str] = {}
        # 按日缓存：{trade_date: set of symbols}，可传入预加载数据
        self._st_cache: dict[str, set[str]] = dict(st_preloaded) if st_preloaded else {}

        self._init_stock_basic()

    def _init_stock_basic(self):
        """初始化时调用一次，加载所有股票基础信息"""
        pro = _get_pro()
        df = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,list_date'
        )
        for _, row in df.iterrows():
            # ts_code 格式如 "000001.SZ"，与 DataFrame index 格式一致
            self._stock_basic[row['ts_code']] = row['list_date']

    def _get_st_stocks(self, trade_date: str) -> set[str]:
        """
        获取某日 ST 股票集合（日缓存 + 磁盘缓存 + 429 重试）。
        trade_date: YYYYMMDD 格式
        """
        if trade_date not in self._st_cache:
            self._st_cache[trade_date] = _fetch_st_stocks(trade_date)
        return self._st_cache[trade_date]

    def _is_new_stock(self, symbol: str, trade_date: str) -> bool:
        """
        判断是否次新股（上市 < 100 天）。
        trade_date: YYYYMMDD 格式
        """
        list_date_str = self._stock_basic.get(symbol)
        if not list_date_str:
            return True  # 找不到信息，默认当次新处理
        try:
            list_date = pd.to_datetime(list_date_str, format='%Y%m%d')
            trade_dt = pd.to_datetime(trade_date, format='%Y%m%d')
            days = (trade_dt - list_date).days
            return days < 100
        except Exception:
            return True

    def filter(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        对候选股票 df 进行过滤，返回过滤后的 df。

        Args:
            df: 候选股票 DataFrame，symbol 在 index 中
            trade_date: 交易日期 YYYYMMDD 格式

        Returns:
            过滤后的 DataFrame
        """
        if len(df) == 0:
            return df.copy()

        result = df.copy()
        symbols = result.index  # symbol 在 index 中

        if self.filter_main_board:
            result = result[result.index.astype(str).str.match(r'^[60]')]
        else:
            result = result[result.index.astype(str).str.match(r'^[630]')]

        if self.filter_st:
            st_set = self._get_st_stocks(trade_date)
            result = result[~result.index.isin(st_set)]
            symbols = result.index

        if self.filter_new_stock:
            result = result[
                ~result.index.to_series().apply(lambda s: self._is_new_stock(s, trade_date))
            ]

        return result