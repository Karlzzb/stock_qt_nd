#!/usr/bin/env python3
"""V4 日频特征的事件日快照重建（特征统筹来源 3，issue #22）。

语义登记（相对已删除的旧缓存 feature_cache_v4_clean.parquet；特征函数本体复用
src/feature_pipeline_v2.py 既有管线，逐股调用）：

1. 原管线把个股截尾 100 个交易日（FEATURE_NEED_MAX_DAYS）后算 talib 递推指标，
   指标值依赖窗口起点（种子效应），是"训练窗口 vs 实盘增量"口径漂移源。
   本模块改为在个股全历史上做纯因果计算（rolling/shift/ewm/talib 均只看过去），
   事件日取值与"仅用不晚于该日数据重算"逐位一致（前缀稳定），由断言保证。
2. 不计算标签族：future_return_* / stop_loss_return_* / *_sell_date_*（28 列）。
3. 不计算 V1 检测器事件列（prev_time/formation_period/divergence_strength 等约 18 列）：
   它们是旧信号的几何属性，在 v5 池事件上无定义。
4. pv_corr_10 原实现未按 symbol 分组（跨股串联污染）；逐股计算下污染天然消失。
5. rank_return / rank_volume 原为全市场当日横截面百分位，本模块单独做全市场口径
    pass 补齐（逐股链内单股计算的恒 1.0 值被丢弃）。
   （注：该两列在主表阶段因与 s2 工厂同式列 |ρ|=1.0 被去重删除，见主表报告。）
6. 横截面 cs_*_rankpct / cs_*_z 排名群体为当日全市场面板（与原 enrich 语义一致：
   原 target_df 即全市场当日行，非仅信号股）；由驱动脚本两阶段装配实现。
7. 个股总历史不足 100 行时原管线函数拒绝计算（len<100 守卫），本模块同样放弃，
   特征记 NaN 并计数上报（原面板语义下该守卫从不触发，偏离仅影响次新股）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.feature_pipeline_v2 as fp2  # noqa: E402

# 逐股链调用顺序与 enrich() 完全一致（feature_pipeline_v2.py:452-456）
PIPELINE_STEPS = (
    "_calculate_basic_technical_features",
    "_calculate_advance_technical_features",
    "_generate_alpha_features",
    "generate_structure_features",
    "generate_lag_features",
)

# 逐股链产出的列中，以下列由全市场口径 pass 覆盖（见模块 docstring 第 5 条）
MARKET_RANK_COLS = ("rank_return", "rank_volume")

RAW_COLS = ("open", "high", "low", "close", "volume")
KEY_COLS = ("timestamp", "symbol")

# 日历特征（原 data_process.prepare_real_daily_features:148-163，后加工，不参与横截面排名）
CALENDAR_COLS = ("day_of_year", "week_of_year", "quarter", "day_of_month",
                 "is_month_end", "is_month_start", "is_quarter_end", "is_quarter_start")

# 泄漏/标签族列名模式（防御：本模块本就不算这些列，断言用）
FORBIDDEN_PATTERNS = (r"^future_return_", r"^future_sell_date_", r"^stop_loss_return_",
                      r"^stop_loss_sell_date_", r"^label_")


def load_stock_v2(path, symbol):
    """读 stock_data/daily parquet，转成 V2 管线格式（timestamp/symbol/open..volume）。"""
    df = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close", "vol"])
    df = df.rename(columns={"trade_date": "timestamp", "vol": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["close"]).drop_duplicates("timestamp").sort_values("timestamp")
    df["symbol"] = symbol
    return df.reset_index(drop=True)


def load_index_v2(path):
    """指数 parquet -> datetime 索引的 OHLCV（V2 大盘特征口径）。"""
    df = pd.read_parquet(path)
    date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def compute_stock_features(df, pipe=None):
    """逐股全历史因果特征链。df 为 V2 格式（timestamp/symbol/OHLCV），按 timestamp 升序。

    返回特征 df（含 timestamp/symbol + 原始行情透传 + 全部特征列）；
    个股历史不足 100 行时返回 None（守卫语义见 docstring 第 7 条）。
    """
    if pipe is None:
        pipe = fp2.FeaturePipeline(None, None)
    out = df
    for step in PIPELINE_STEPS:
        out = getattr(pipe, step)(out)
        if out is None:
            return None
    return out


def add_calendar_features(df):
    """8 个日历特征，口径与 data_process.prepare_real_daily_features 一致。"""
    ts = df["timestamp"]
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["quarter"] = ts.dt.quarter
    df["day_of_month"] = ts.dt.day
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    df["is_quarter_end"] = ts.dt.is_quarter_end.astype(int)
    df["is_quarter_start"] = ts.dt.is_quarter_start.astype(int)
    return df


def snapshot_rows(feat_df, event_dates):
    """从全历史特征 df 中取事件日行。event_dates 为 datetime64 数组/列表。"""
    ed = pd.to_datetime(pd.Series(list(event_dates)))
    snap = feat_df[feat_df["timestamp"].isin(set(ed))].copy()
    return snap


def prefix_recompute_at(path, symbol, T, pipe=None):
    """前缀重算: 只用不晚于 T 的数据重跑逐股链, 取 T 日行。时点一致性的对照实现。"""
    df = load_stock_v2(path, symbol)
    df = df[df["timestamp"] <= pd.Timestamp(T)].reset_index(drop=True)
    feat = compute_stock_features(df, pipe=pipe)
    if feat is None or len(feat) == 0:
        return None
    row = feat[feat["timestamp"] == pd.Timestamp(T)]
    return row if len(row) else None
