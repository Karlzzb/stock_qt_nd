"""
时点股票池（Point-in-Time Universe）。

回答「任意历史日期 T 当天在市的 A 股集合」，消除幸存者偏差。

用法::

    from src.universe import PointInTimeUniverse

    pit = PointInTimeUniverse.from_tinyshare(pro)     # 从 API 构建
    # 或
    pit = PointInTimeUniverse.load(path)               # 从 Parquet 加载

    stocks = pit.get_universe("2020-01-02")            # 返回当日在市 set[str]
    df     = pit.get_universe_df("2020-01-02")         # 返回 DataFrame

数据契约
--------
内部维护一份 DataFrame（``_df``），列如下：

- ``ts_code``     : str  — tinyshare 格式，如 ``000001.SZ``
- ``name``        : str  — 股票简称
- ``list_date``   : date — 上市日期
- ``delist_date`` : date | NaT — 退市日期；在市股票为 NaT / None

「在市」定义：list_date <= T，且 delist_date IS NULL 或 delist_date > T。
即退市当天不再出现于股票池。
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Set, Union

import pandas as pd


_DateLike = Union[str, date, datetime, pd.Timestamp]


def _to_date(d: _DateLike) -> date:
    """将多种日期类型统一转成 date 对象。"""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    # str：支持 "YYYYMMDD" 和 "YYYY-MM-DD"
    s = str(d).strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d").date()
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


class PointInTimeUniverse:
    """时点股票池。

    Parameters
    ----------
    df:
        包含 ts_code / name / list_date / delist_date 列的 DataFrame。
        list_date 和 delist_date 可以是字符串（"YYYYMMDD"）、datetime、
        pandas Timestamp 或 None/NaN（delist_date 为空表示仍在市）。
    """

    _REQUIRED_COLS = {"ts_code", "list_date"}

    def __init__(self, df: pd.DataFrame) -> None:
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"PointInTimeUniverse: 缺少必要列 {missing}")

        df = df.copy()
        df["list_date"]   = pd.to_datetime(df["list_date"],   errors="coerce").dt.date
        df["delist_date"] = pd.to_datetime(
            df.get("delist_date"), errors="coerce"
        ).dt.date if "delist_date" in df.columns else None

        # 删除 list_date 为空的行（数据污损）
        df = df[df["list_date"].notna()].reset_index(drop=True)

        self._df = df

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_tinyshare(cls, pro) -> "PointInTimeUniverse":
        """从 tinyshare pro API 构建股票池（拉在市 + 退市两份列表合并）。

        Parameters
        ----------
        pro:
            ``tinyshare_auth.get_pro_api()`` 返回的 pro 对象。
        """
        fields = "ts_code,name,list_date,delist_date"

        df_live = pro.stock_basic(
            exchange="", list_status="L", fields=fields
        )
        df_dead = pro.stock_basic(
            exchange="", list_status="D", fields=fields
        )

        df = pd.concat([df_live, df_dead], ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"])
        return cls(df)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PointInTimeUniverse":
        """从 Parquet 文件加载。"""
        df = pd.read_parquet(path)
        return cls(df)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """序列化到 Parquet。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 日期列转字符串以保持 Parquet 类型简洁
        out = self._df.copy()
        for col in ("list_date", "delist_date"):
            if col in out.columns:
                out[col] = out[col].astype(str).replace("None", "").replace("NaT", "")
        out.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

    # ------------------------------------------------------------------
    # 核心查询
    # ------------------------------------------------------------------

    def get_universe(self, query_date: _DateLike) -> Set[str]:
        """返回 query_date 当天在市的股票代码集合（set[str]）。

        上市当天计入，退市当天**不**计入（delist_date > query_date）。
        """
        d = _to_date(query_date)
        df = self._df

        listed  = df["list_date"] <= d

        if "delist_date" in df.columns and df["delist_date"].notna().any():
            not_yet_delisted = df["delist_date"].isna() | (df["delist_date"] > d)
        else:
            not_yet_delisted = pd.Series(True, index=df.index)

        mask = listed & not_yet_delisted
        return set(df.loc[mask, "ts_code"].tolist())

    def get_universe_df(self, query_date: _DateLike) -> pd.DataFrame:
        """返回 query_date 当天在市的完整 DataFrame 行（含 name / list_date 等列）。"""
        codes = self.get_universe(query_date)
        return self._df[self._df["ts_code"].isin(codes)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------

    @property
    def total_stocks(self) -> int:
        return len(self._df)

    @property
    def live_stocks(self) -> int:
        if "delist_date" not in self._df.columns:
            return self.total_stocks
        return int(self._df["delist_date"].isna().sum())

    @property
    def delisted_stocks(self) -> int:
        return self.total_stocks - self.live_stocks

    def __repr__(self) -> str:
        return (
            f"PointInTimeUniverse("
            f"total={self.total_stocks}, "
            f"live={self.live_stocks}, "
            f"delisted={self.delisted_stocks})"
        )
