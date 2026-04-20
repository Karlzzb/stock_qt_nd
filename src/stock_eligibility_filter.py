import tinyshare as ts
import pandas as pd
from datetime import datetime
from typing import Optional


# Tushare Token 配置（与 st_stock_filter_v2.py 保持一致）
token = "3Q4RY56w8deQac5uQkcba5wzoaUf8XBdiLvBti22gv5jTstJ4d0ywZKU247ade48"
ts.set_token(token)
pro = ts.pro_api()


class StockEligibilityFilter:
    """
    统一股票过滤器，共用于回测和实盘。

    过滤规则（可独立开关）：
    1. 主板过滤：只允许 ^[60] 开头
    2. ST 过滤：通过 Tushare stock_st API 获取指定日期的 ST 列表
    3. 次新股过滤：上市 < 100 天（基于 list_date 计算）
    """

    def __init__(self,
                 filter_main_board: bool = True,
                 filter_st: bool = True,
                 filter_new_stock: bool = True):
        self.filter_main_board = filter_main_board
        self.filter_st = filter_st
        self.filter_new_stock = filter_new_stock

        # 一次性缓存：{symbol: list_date}
        self._stock_basic: dict[str, str] = {}
        # 按日缓存：{trade_date: set of symbols}
        self._st_cache: dict[str, set[str]] = {}

        self._init_stock_basic()

    def _init_stock_basic(self):
        """初始化时调用一次，加载所有股票基础信息"""
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
        获取某日 ST 股票集合（日缓存）。
        trade_date: YYYYMMDD 格式
        """
        if trade_date not in self._st_cache:
            df = pro.stock_st(trade_date=trade_date)
            if df is not None and len(df) > 0:
                # stock_st 返回 ts_code 列（如 "000001.SZ"）
                self._st_cache[trade_date] = set(df['ts_code'])
            else:
                self._st_cache[trade_date] = set()
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
            result = result[result.index.str.match(r'^[60]')]
            symbols = result.index

        if self.filter_st:
            st_set = self._get_st_stocks(trade_date)
            result = result[~result.index.isin(st_set)]
            symbols = result.index

        if self.filter_new_stock:
            result = result[
                ~result.index.to_series().apply(lambda s: self._is_new_stock(s, trade_date))
            ]

        return result