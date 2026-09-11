#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 1+2:全市场逐日基础特征面板 + 横截面聚合。

预登记军令状 = 同目录 README.md(commit 49b9c5b,冻结);公式唯一来源 = 同目录
anatomy_report.md(解剖报告);v2 原始代码(/tmp/v2_excavation/)只读参考,禁止 import。

施工内容(README §一.2、§四):
  Pass A  逐股全历史因果计算 README §四 第 1~7 族中全部按股列(市场族为日级另算),
          v2 的 Python 循环(rolling_slope/rolling_cv/macd_percentile/obv_trend)全部
          向量化重写、公式逐字保真;修正性偏离 P1(按股分组)/P3/P4(剔除 wavelet/FFD)/
          P7(按股截尾百分位)/P15(全历史)逐条落实;
          事件表 event_close/event_vol/event_amount 与日线 close/vol/amount 逐位对账;
          产出 per-stock parts(仅事件日行)+ vol_long 侧车(P6 分母用)+ 事件 aux。
  断言    README §九.1 因果性:抽样 12 股 × 3 事件日截断重算(仅用 ≤ 当日数据),
          与全历史值逐位一致(rtol=1e-9,equal_nan)。
  Pass B  横截面聚合(README §一.2 原口径 = 同日全市场股票当日行):
          按 event_date 聚合全市场当日行,P5(按日截面 lambda boxcox_atr)、
          P6(滚动 100 个交易日内全市场均值为分母的 vol_divergence)、
          rank_return/rank_volume(当日截面 rank pct)、市场大盘族(指数截断 ≤ 当日)、
          再按 §四.9 排除集对 195 列做 rankpct/z + cs_n,切出 96,577 事件行。

口径要点(全部先登记于 README §三/§四,本文件零新增偏离):
  - 横截面 universe = stock_data/daily/*.parquet 全文件减去两个指数文件
    (000001.SH/399001.SZ),与 M2 v4daily 先例一致(README §十 的「约 5,188 股」为估计值,
    实测股数落台账与修订记录)。
  - vol_divergence 分母(P6):逐市场交易日 t,先算当日全市场 vol_long=close.rolling(60).std()
    的(skipna)均值 M_t,再取 t 及之前 100 个市场交易日 M 的均值(min_periods=1);
    市场交易日历 = 全市场个股日线交易日并集(2026-09-11 二轮裁定:指数日线文件仅
    近 8,000 行,SH 起于 1993-10-08/SZ 起于 1993-09-30,早期日历物理缺失,
    见 README 修订记录)。
  - macd_percentile(P7):按股截尾 100 行(不含当日)percentileofscore(kind='rank')
    向量化;不足 100 行 → NaN;v2 原 quirk(值恰为 50 → NaN,anatomy L898-901)逐字保留。
  - rsi_robust/macd_robust:逐股全历史 expanding 中位数/IQR(README §四.5 注登记的
    P5~P7 同类修正),输入为 alpha 收尾 fillna(0) 后的序列(与 v2 操作序一致)。

用法:
  python3 build_panel_v2on6.py --smoke        # 50 股 × 最近 200 事件日冒烟并计时
  python3 build_panel_v2on6.py --full         # 全量(约 5,889 股 × 5,696 事件日)
  python3 build_panel_v2on6.py --full --rerun-tag run2   # 第二进程重跑(双跑 md5 对账)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import talib
from scipy import stats

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
DAILY_DIR = REPO / "stock_data" / "daily"
INDEX_CODES = ("000001.SH", "399001.SZ")
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

EPS = 1e-9                       # v2 comm_fun.py:L699 逐字
EXPECTED_PROFIT = 1.15           # comm_fun.py:L188
RETURN_PERIODS = [3, 5, 10, 15, 20, 25, 30]      # comm_fun.py:L200
WINDOWS_VOLATILITY = [3, 5, 10, 15, 20, 25, 30]  # comm_fun.py:L201
LAG_PERIODS = [3, 5, 10, 15, 20, 25, 30]         # comm_fun.py:L202

DATE_CHUNK = 250                 # 聚合日期块大小(M2 v4daily 同值)
N_ASSERT_STOCKS = 12             # README §九.1 因果性抽检规模
ASSERT_DAYS_PER_STOCK = 3
ASSERT_SEED = 42

# v2 FULL_FEATURE_COLS 610 列逐字(comm_fun.py:L232-361;列顺序 = v2 落盘顺序,
# 仅用于推导本线列规格,不作为选择清单)
FULL_FEATURE_COLS_V2 = [
    'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi_6', 'rsi_14', 'rsi_24', 'ma_5', 'ma_20', 'ma_60', 'bb_upper',
    'bb_middle', 'bb_lower', 'volume_ma_20', 'obv', 'atr', 'slowk', 'slowd', 'price_vs_ma5', 'price_vs_ma20',
    'price_vs_ma60', 'ma_arrangement', 'bb_position', 'bb_squeeze', 'rsi_oversold_6', 'rsi_oversold_14',
    'rsi_oversold_24', 'rsi_overbought_6', 'rsi_overbought_14', 'rsi_overbought_24', 'macd_signal_distance',
    'macd_golden_cross', 'volume_ma20', 'volume_ratio', 'volume_spike', 'volume_dryup', 'atr_ratio',
    'hammer_pattern', 'downtrend', 'hammer_signal', 'doji_pattern', 'distance_to_support', 'distance_to_resistance',
    'stoch_oversold', 'stoch_overbought', 'macd_percentile', 'obv_trend', 'volatility_3d', 'volatility_5d',
    'volatility_10d', 'volatility_15d', 'volatility_20d', 'volatility_25d', 'volatility_30d', 'engulfing_pattern',
    'signed_volume_strength', 'close_vs_high', 'volume_ma_ratio', 'rsi_momentum', 'rsi_turning_simple',
    'rsi_turning', 'volume_trend_5', 'volume_trend_10', 'price_trend_5', 'price_volume_divergence',
    'volume_consistency', 'macd_hist_trend_5', 'macd_death_cross', 'macd_signal_cross', 'macd_zero_cross_up',
    'macd_zero_cross_down', 'macd_zero_cross', 'macd_hist_amplitude', 'macd_hist_direction',
    'macd_hist_acceleration', 'macd_signal_convergence', 'macd_signal_convergence_trend', 'pct_change', 'clv',
    'upper_shadow_ratio', 'body_strength', 'rank_return', 'rank_volume', 'signed_vol_strength', 'pv_corr_10',
    'dist_to_high_60', 'vol_divergence', 'vol_gk', 'vol_gk_ratio', 'illiq', 'efficiency_ratio', 'intraday_pos',
    'ret_overnight', 'ret_intraday', 'smart_money_diff', 'high_mean_20', 'low_mean_20', 'support_resistance_ratio',
    'log_volume', 'boxcox_atr', 'rsi_robust', 'macd_robust', 'close_wavelet', 'close_d0.4', 'close_lag_3',
    'open_lag_3', 'high_lag_3', 'low_lag_3', 'close_lag_5', 'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_10',
    'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_15', 'open_lag_15', 'high_lag_15', 'low_lag_15',
    'close_lag_20', 'open_lag_20', 'high_lag_20', 'low_lag_20', 'close_lag_25', 'open_lag_25', 'high_lag_25',
    'low_lag_25', 'close_lag_30', 'open_lag_30', 'high_lag_30', 'low_lag_30', 'volume_lag_3', 'volume_lag_5',
    'volume_lag_10', 'volume_lag_15', 'volume_lag_20', 'volume_lag_25', 'volume_lag_30', 'daily_return',
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20', 'amplitude',
    'amplitude_lag_1', 'amplitude_lag_3', 'amplitude_lag_5', 'vol_gk_lag_1', 'vol_gk_ratio_lag_1', 'vol_gk_lag_3',
    'vol_gk_ratio_lag_3', 'vol_gk_lag_5', 'vol_gk_ratio_lag_5', 'vol_gk_lag_10', 'vol_gk_ratio_lag_10',
    'illiq_lag_1', 'illiq_lag_3', 'illiq_lag_5', 'efficiency_ratio_lag_1', 'efficiency_ratio_lag_3',
    'efficiency_ratio_lag_5', 'intraday_pos_lag_1', 'intraday_pos_lag_2', 'intraday_pos_lag_3',
    'smart_money_diff_lag_1', 'ret_overnight_lag_1', 'ret_intraday_lag_1', 'smart_money_diff_lag_3',
    'ret_overnight_lag_3', 'ret_intraday_lag_3', 'smart_money_diff_lag_5', 'ret_overnight_lag_5',
    'ret_intraday_lag_5', 'support_resistance_ratio_lag_1', 'support_resistance_ratio_lag_3',
    'support_resistance_ratio_lag_5', 'sh_price_change', 'sh_amplitude', 'sh_volume_ratio', 'sh_price_change_abs',
    'sh_price_wave_abs', 'sh_sentiment', 'sh_volume_signal', 'sz_price_change', 'sz_amplitude', 'sz_volume_ratio',
    'sz_price_change_abs', 'sz_price_wave_abs', 'sz_sentiment', 'sz_volume_signal', 'sh_sz_sync_direction',
    'sh_sz_sync_strength', 'market_sentiment', 'market_avg_change', 'market_avg_amplitude', 'market_sync_score',
    'cs_n', 'volume_rankpct', 'volume_z', 'macd_rankpct', 'macd_z', 'macd_signal_rankpct', 'macd_signal_z',
    'macd_hist_rankpct', 'macd_hist_z', 'rsi_6_rankpct', 'rsi_6_z', 'rsi_14_rankpct', 'rsi_14_z', 'rsi_24_rankpct',
    'rsi_24_z', 'ma_5_rankpct', 'ma_5_z', 'ma_20_rankpct', 'ma_20_z', 'ma_60_rankpct', 'ma_60_z',
    'bb_upper_rankpct', 'bb_upper_z', 'bb_middle_rankpct', 'bb_middle_z', 'bb_lower_rankpct', 'bb_lower_z',
    'volume_ma_20_rankpct', 'volume_ma_20_z', 'obv_rankpct', 'obv_z', 'atr_rankpct', 'atr_z', 'slowk_rankpct',
    'slowk_z', 'slowd_rankpct', 'slowd_z', 'price_vs_ma5_rankpct', 'price_vs_ma5_z', 'price_vs_ma20_rankpct',
    'price_vs_ma20_z', 'price_vs_ma60_rankpct', 'price_vs_ma60_z', 'ma_arrangement_rankpct', 'ma_arrangement_z',
    'bb_position_rankpct', 'bb_position_z', 'bb_squeeze_rankpct', 'bb_squeeze_z', 'rsi_oversold_6_rankpct',
    'rsi_oversold_6_z', 'rsi_oversold_14_rankpct', 'rsi_oversold_14_z', 'rsi_oversold_24_rankpct',
    'rsi_oversold_24_z', 'rsi_overbought_6_rankpct', 'rsi_overbought_6_z', 'rsi_overbought_14_rankpct',
    'rsi_overbought_14_z', 'rsi_overbought_24_rankpct', 'rsi_overbought_24_z', 'macd_signal_distance_rankpct',
    'macd_signal_distance_z', 'macd_golden_cross_rankpct', 'macd_golden_cross_z', 'volume_ma20_rankpct',
    'volume_ma20_z', 'volume_ratio_rankpct', 'volume_ratio_z', 'volume_spike_rankpct', 'volume_spike_z',
    'volume_dryup_rankpct', 'volume_dryup_z', 'atr_ratio_rankpct', 'atr_ratio_z', 'hammer_pattern_rankpct',
    'hammer_pattern_z', 'doji_pattern_rankpct', 'doji_pattern_z', 'distance_to_support_rankpct',
    'distance_to_support_z', 'distance_to_resistance_rankpct', 'distance_to_resistance_z', 'stoch_oversold_rankpct',
    'stoch_oversold_z', 'stoch_overbought_rankpct', 'stoch_overbought_z', 'macd_percentile_rankpct',
    'macd_percentile_z', 'obv_trend_rankpct', 'obv_trend_z', 'volatility_3d_rankpct', 'volatility_3d_z',
    'volatility_5d_rankpct', 'volatility_5d_z', 'volatility_10d_rankpct', 'volatility_10d_z',
    'volatility_15d_rankpct', 'volatility_15d_z', 'volatility_20d_rankpct', 'volatility_20d_z',
    'volatility_25d_rankpct', 'volatility_25d_z', 'volatility_30d_rankpct', 'volatility_30d_z',
    'engulfing_pattern_rankpct', 'engulfing_pattern_z', 'signed_volume_strength_rankpct',
    'signed_volume_strength_z', 'close_vs_high_rankpct', 'close_vs_high_z', 'volume_ma_ratio_rankpct',
    'volume_ma_ratio_z', 'rsi_momentum_rankpct', 'rsi_momentum_z', 'rsi_turning_simple_rankpct',
    'rsi_turning_simple_z', 'rsi_turning_rankpct', 'rsi_turning_z', 'volume_trend_5_rankpct', 'volume_trend_5_z',
    'volume_trend_10_rankpct', 'volume_trend_10_z', 'price_trend_5_rankpct', 'price_trend_5_z',
    'price_volume_divergence_rankpct', 'price_volume_divergence_z', 'volume_consistency_rankpct',
    'volume_consistency_z', 'macd_hist_trend_5_rankpct', 'macd_hist_trend_5_z', 'macd_death_cross_rankpct',
    'macd_death_cross_z', 'macd_signal_cross_rankpct', 'macd_signal_cross_z', 'macd_zero_cross_up_rankpct',
    'macd_zero_cross_up_z', 'macd_zero_cross_down_rankpct', 'macd_zero_cross_down_z', 'macd_zero_cross_rankpct',
    'macd_zero_cross_z', 'macd_hist_amplitude_rankpct', 'macd_hist_amplitude_z', 'macd_hist_direction_rankpct',
    'macd_hist_direction_z', 'macd_hist_acceleration_rankpct', 'macd_hist_acceleration_z',
    'macd_signal_convergence_rankpct', 'macd_signal_convergence_z', 'macd_signal_convergence_trend_rankpct',
    'macd_signal_convergence_trend_z', 'pct_change_rankpct', 'pct_change_z', 'clv_rankpct', 'clv_z',
    'upper_shadow_ratio_rankpct', 'upper_shadow_ratio_z', 'body_strength_rankpct', 'body_strength_z',
    'rank_return_rankpct', 'rank_return_z', 'rank_volume_rankpct', 'rank_volume_z', 'signed_vol_strength_rankpct',
    'signed_vol_strength_z', 'pv_corr_10_rankpct', 'pv_corr_10_z', 'dist_to_high_60_rankpct', 'dist_to_high_60_z',
    'vol_divergence_rankpct', 'vol_divergence_z', 'vol_gk_rankpct', 'vol_gk_z', 'vol_gk_ratio_rankpct',
    'vol_gk_ratio_z', 'illiq_rankpct', 'illiq_z', 'efficiency_ratio_rankpct', 'efficiency_ratio_z',
    'intraday_pos_rankpct', 'intraday_pos_z', 'ret_overnight_rankpct', 'ret_overnight_z', 'ret_intraday_rankpct',
    'ret_intraday_z', 'smart_money_diff_rankpct', 'smart_money_diff_z', 'high_mean_20_rankpct', 'high_mean_20_z',
    'low_mean_20_rankpct', 'low_mean_20_z', 'support_resistance_ratio_rankpct', 'support_resistance_ratio_z',
    'log_volume_rankpct', 'log_volume_z', 'boxcox_atr_rankpct', 'boxcox_atr_z', 'rsi_robust_rankpct',
    'rsi_robust_z', 'macd_robust_rankpct', 'macd_robust_z', 'close_wavelet_rankpct', 'close_wavelet_z',
    'close_d0.4_rankpct', 'close_d0.4_z', 'close_lag_3_rankpct', 'close_lag_3_z', 'open_lag_3_rankpct',
    'open_lag_3_z', 'high_lag_3_rankpct', 'high_lag_3_z', 'low_lag_3_rankpct', 'low_lag_3_z', 'close_lag_5_rankpct',
    'close_lag_5_z', 'open_lag_5_rankpct', 'open_lag_5_z', 'high_lag_5_rankpct', 'high_lag_5_z',
    'low_lag_5_rankpct', 'low_lag_5_z', 'close_lag_10_rankpct', 'close_lag_10_z', 'open_lag_10_rankpct',
    'open_lag_10_z', 'high_lag_10_rankpct', 'high_lag_10_z', 'low_lag_10_rankpct', 'low_lag_10_z',
    'close_lag_15_rankpct', 'close_lag_15_z', 'open_lag_15_rankpct', 'open_lag_15_z', 'high_lag_15_rankpct',
    'high_lag_15_z', 'low_lag_15_rankpct', 'low_lag_15_z', 'close_lag_20_rankpct', 'close_lag_20_z',
    'open_lag_20_rankpct', 'open_lag_20_z', 'high_lag_20_rankpct', 'high_lag_20_z', 'low_lag_20_rankpct',
    'low_lag_20_z', 'close_lag_25_rankpct', 'close_lag_25_z', 'open_lag_25_rankpct', 'open_lag_25_z',
    'high_lag_25_rankpct', 'high_lag_25_z', 'low_lag_25_rankpct', 'low_lag_25_z', 'close_lag_30_rankpct',
    'close_lag_30_z', 'open_lag_30_rankpct', 'open_lag_30_z', 'high_lag_30_rankpct', 'high_lag_30_z',
    'low_lag_30_rankpct', 'low_lag_30_z', 'volume_lag_3_rankpct', 'volume_lag_3_z', 'volume_lag_5_rankpct',
    'volume_lag_5_z', 'volume_lag_10_rankpct', 'volume_lag_10_z', 'volume_lag_15_rankpct', 'volume_lag_15_z',
    'volume_lag_20_rankpct', 'volume_lag_20_z', 'volume_lag_25_rankpct', 'volume_lag_25_z', 'volume_lag_30_rankpct',
    'volume_lag_30_z', 'daily_return_rankpct', 'daily_return_z', 'return_lag_1_rankpct', 'return_lag_1_z',
    'return_lag_2_rankpct', 'return_lag_2_z', 'return_lag_3_rankpct', 'return_lag_3_z', 'return_lag_5_rankpct',
    'return_lag_5_z', 'return_lag_10_rankpct', 'return_lag_10_z', 'return_lag_20_rankpct', 'return_lag_20_z',
    'amplitude_rankpct', 'amplitude_z', 'amplitude_lag_1_rankpct', 'amplitude_lag_1_z', 'amplitude_lag_3_rankpct',
    'amplitude_lag_3_z', 'amplitude_lag_5_rankpct', 'amplitude_lag_5_z', 'vol_gk_lag_1_rankpct', 'vol_gk_lag_1_z',
    'vol_gk_ratio_lag_1_rankpct', 'vol_gk_ratio_lag_1_z', 'vol_gk_lag_3_rankpct', 'vol_gk_lag_3_z',
    'vol_gk_ratio_lag_3_rankpct', 'vol_gk_ratio_lag_3_z', 'vol_gk_lag_5_rankpct', 'vol_gk_lag_5_z',
    'vol_gk_ratio_lag_5_rankpct', 'vol_gk_ratio_lag_5_z', 'vol_gk_lag_10_rankpct', 'vol_gk_lag_10_z',
    'vol_gk_ratio_lag_10_rankpct', 'vol_gk_ratio_lag_10_z', 'illiq_lag_1_rankpct', 'illiq_lag_1_z',
    'illiq_lag_3_rankpct', 'illiq_lag_3_z', 'illiq_lag_5_rankpct', 'illiq_lag_5_z',
    'efficiency_ratio_lag_1_rankpct', 'efficiency_ratio_lag_1_z', 'efficiency_ratio_lag_3_rankpct',
    'efficiency_ratio_lag_3_z', 'efficiency_ratio_lag_5_rankpct', 'efficiency_ratio_lag_5_z',
    'intraday_pos_lag_1_rankpct', 'intraday_pos_lag_1_z', 'intraday_pos_lag_2_rankpct', 'intraday_pos_lag_2_z',
    'intraday_pos_lag_3_rankpct', 'intraday_pos_lag_3_z', 'smart_money_diff_lag_1_rankpct',
    'smart_money_diff_lag_1_z', 'ret_overnight_lag_1_rankpct', 'ret_overnight_lag_1_z',
    'ret_intraday_lag_1_rankpct', 'ret_intraday_lag_1_z', 'smart_money_diff_lag_3_rankpct',
    'smart_money_diff_lag_3_z', 'ret_overnight_lag_3_rankpct', 'ret_overnight_lag_3_z',
    'ret_intraday_lag_3_rankpct', 'ret_intraday_lag_3_z', 'smart_money_diff_lag_5_rankpct',
    'smart_money_diff_lag_5_z', 'ret_overnight_lag_5_rankpct', 'ret_overnight_lag_5_z',
    'ret_intraday_lag_5_rankpct', 'ret_intraday_lag_5_z', 'support_resistance_ratio_lag_1_rankpct',
    'support_resistance_ratio_lag_1_z', 'support_resistance_ratio_lag_3_rankpct',
    'support_resistance_ratio_lag_3_z', 'support_resistance_ratio_lag_5_rankpct',
    'support_resistance_ratio_lag_5_z', 'sh_price_change_rankpct', 'sh_price_change_z', 'sh_amplitude_rankpct',
    'sh_amplitude_z', 'sh_volume_ratio_rankpct', 'sh_volume_ratio_z', 'sh_price_change_abs_rankpct',
    'sh_price_change_abs_z', 'sh_price_wave_abs_rankpct', 'sh_price_wave_abs_z', 'sh_sentiment_rankpct',
    'sh_sentiment_z', 'sh_volume_signal_rankpct', 'sh_volume_signal_z', 'sz_price_change_rankpct',
    'sz_price_change_z', 'sz_amplitude_rankpct', 'sz_amplitude_z', 'sz_volume_ratio_rankpct', 'sz_volume_ratio_z',
    'sz_price_change_abs_rankpct', 'sz_price_change_abs_z', 'sz_price_wave_abs_rankpct', 'sz_price_wave_abs_z',
    'sz_sentiment_rankpct', 'sz_sentiment_z', 'sz_volume_signal_rankpct', 'sz_volume_signal_z',
    'sh_sz_sync_direction_rankpct', 'sh_sz_sync_direction_z', 'sh_sz_sync_strength_rankpct',
    'sh_sz_sync_strength_z', 'market_sentiment_rankpct', 'market_sentiment_z', 'market_avg_change_rankpct',
    'market_avg_change_z', 'market_avg_amplitude_rankpct', 'market_avg_amplitude_z', 'market_sync_score_rankpct',
    'market_sync_score_z', 'is_quick_divergence_x', 'compare_rank', 'close_current', 'close_previous',
    'macd_current', 'macd_previous', 'price_decline_pct', 'macd_increase_pct', 'formation_period',
    'is_quick_divergence_y', 'divergence_strength', 'volume_signal', 'price_macd_ratio', 'divergence_magnitude',
    'confirmation_score', 'divergence_amount',
]

# ---------------------------------------------------------------- 列规格推导(机检) ----
DROP_COLS_P3P4 = {"close_wavelet", "close_d0.4"}          # P3/P4:剔除及其 rankpct/z
DROP_COLS_P8 = {"is_quick_divergence_x"}                  # P8:恒 0 废列剔除
ADD_COLS_P10 = ["v2j2_macd_increase_pct"]                 # P10:增补列(命名隔离)

_pos_cs = FULL_FEATURE_COLS_V2.index("cs_n")
BASE_FAM1_7_V2 = FULL_FEATURE_COLS_V2[:_pos_cs]           # 第 1~7 族(volume..market_sync_score)
assert len(BASE_FAM1_7_V2) == 199, len(BASE_FAM1_7_V2)
DIVERGENCE_COLS_V2 = FULL_FEATURE_COLS_V2[-16:]           # 背离结构族 16 列(v2 尾部)
RPZ_BASE_V2 = [c[:-len("_rankpct")] for c in FULL_FEATURE_COLS_V2
               if c.endswith("_rankpct")]
assert len(RPZ_BASE_V2) == 197, len(RPZ_BASE_V2)
assert all(f + "_z" in FULL_FEATURE_COLS_V2 for f in RPZ_BASE_V2)
# v2 截面规则下无 rp/z 变体的 19 列 = 背离 16 + cs_n + downtrend + hammer_signal
assert set(FULL_FEATURE_COLS_V2[:_pos_cs + 1] + DIVERGENCE_COLS_V2) - set(RPZ_BASE_V2) == {
    "cs_n", "downtrend", "hammer_signal"} | set(DIVERGENCE_COLS_V2)

# 本线列规格(P3/P4/P8/P10 落实后)
BASE_FAM1_7 = [c for c in BASE_FAM1_7_V2 if c not in DROP_COLS_P3P4]      # 197 列(不含 cs_n)
RPZ_BASE = [c for c in RPZ_BASE_V2 if c not in DROP_COLS_P3P4]            # 195 列
DIVERGENCE_COLS = [c for c in DIVERGENCE_COLS_V2 if c not in DROP_COLS_P8] + ADD_COLS_P10  # 16 列
assert len(BASE_FAM1_7) == 197 and len(RPZ_BASE) == 195 and len(DIVERGENCE_COLS) == 16
MASTER_FEATURE_COLS = (BASE_FAM1_7 + ["cs_n"]
                       + [f + "_rankpct" for f in RPZ_BASE] + [f + "_z" for f in RPZ_BASE]
                       + DIVERGENCE_COLS)
assert len(MASTER_FEATURE_COLS) == 604, len(MASTER_FEATURE_COLS)  # README §四.9「604 左右」

MARKET_COLS = [c for c in BASE_FAM1_7 if c.startswith(("sh_", "sz_", "market_"))]
assert len(MARKET_COLS) == 20, MARKET_COLS
# 聚合期现算列(P5/P6/当日截面 rank)
DEFERRED_COLS = ["rank_return", "rank_volume", "vol_divergence", "boxcox_atr"]
# Pass A 逐股产出列(197 − 市场 20 − 聚合期 4 = 173)+ 中间列 _vol5_std
PER_STOCK_COLS = [c for c in BASE_FAM1_7 if c not in MARKET_COLS + DEFERRED_COLS]
assert len(PER_STOCK_COLS) == 173, len(PER_STOCK_COLS)
INTERMEDIATE_COLS = ["_vol5_std"]  # vol_divergence 分子(聚合后丢弃,不进主表)

# v2 大盘缺失默认行(anatomy L1092-1101 逐字)
INDEX_DEFAULTS = dict(price_change=0, amplitude=0.02, volume_ratio=1,
                      price_change_abs=0, price_wave_abs=0.02, sentiment=1, volume_signal=0)

# ---------------------------------------------------------------- 全局(fork 共享,只读)
_G: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [build_panel] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ================================================================ 向量化原语 ==========
def _slope_weights(n: int) -> np.ndarray:
    """最小二乘斜率的固定权重: slope = Σ w_i·y_i, x = 0..n-1。

    与 linregress(x, y).slope 及 obv_trend 公式法 (nΣxy−ΣxΣy)/(nΣx²−(Σx)²) 逐字等价。
    """
    x = np.arange(n, dtype=np.float64)
    denom = n * (x ** 2).sum() - x.sum() ** 2
    return (n * x - x.sum()) / denom


def _rolling_slope_fixed(y: np.ndarray, window: int, warmup_zero: bool = True) -> np.ndarray:
    """定窗最小二乘斜率(v2 rolling_slope/obv_trend 向量化;窗内含当日)。

    i < window-1 → 0(v2 初始化 zeros 原样);窗内 NaN → NaN(np.sum 语义原样)。
    """
    out = np.zeros(len(y), dtype=np.float64)
    if len(y) >= window:
        w = _slope_weights(window)
        sw = np.lib.stride_tricks.sliding_window_view(y, window)
        out[window - 1:] = sw @ w
    return out


def _rolling_slope_minp2(y: np.ndarray, window: int) -> np.ndarray:
    """rolling(window, min_periods=2) 的 linear_slope(v2 macd_features_rolling 原样)。

    短窗(行 1..window-2)以 x=arange(当前窗长) 计算;行 0 → 0(v2 fillna(0));
    窗内 NaN → NaN → 0(v2 rolling.apply 后 fillna(0))。
    """
    out = np.zeros(len(y), dtype=np.float64)
    n = len(y)
    for i in range(1, min(window - 1, n)):  # 短窗:每 stock 至多 window-2 行
        w = _slope_weights(i + 1)
        out[i] = y[:i + 1] @ w
    if n >= window:
        w = _slope_weights(window)
        sw = np.lib.stride_tricks.sliding_window_view(y, window)
        out[window - 1:] = sw @ w
    return np.where(np.isnan(out), 0.0, out)


def _macd_percentile_p7(macd: np.ndarray, window: int = 100) -> np.ndarray:
    """P7 口径 macd_percentile:按股截尾 100 行(不含当日)percentileofscore(kind='rank')。

    scipy kind='rank' 语义 = (count(< x) + count(<= x)) / 2 / n × 100;不足 100 行 → NaN(P7)。
    暖机窗 NaN 口径(2026-09-11 监工复核裁定,README 修订记录三轮):scipy 1.17.1 实测
    percentileofscore 对含 NaN 输入整窗返回 NaN,故窗内含任一 NaN → NaN(scipy 字面),
    随后经 v2 L897 面板 fillna(0) 与原管线汇合。
    """
    n = len(macd)
    out = np.full(n, np.nan, dtype=np.float64)
    if n > window:
        sw = np.lib.stride_tricks.sliding_window_view(macd, window)  # [s] = macd[s:s+window]
        cur = macd[window:]
        hist = sw[:n - window]
        with np.errstate(invalid="ignore"):
            less = (hist < cur[:, None]).sum(axis=1)
            leq = (hist <= cur[:, None]).sum(axis=1)
        vals = (less + leq) / 2.0 / window * 100.0
        vals[np.isnan(hist).any(axis=1)] = np.nan  # scipy 字面:含 NaN 窗 → NaN
        out[window:] = vals
    return out


def _expanding_robust_z(x: pd.Series) -> np.ndarray:
    """README §四.5 注登记口径:逐股全历史 expanding 中位数/IQR 稳健 z。

    robust_zscore 公式逐字 = (x − median)/(IQR/1.34896),IQR=0 退化为 x − median;
    统计量只用 ≤ 当日数据(expanding)。
    """
    med = x.expanding().median()
    q75 = x.expanding().quantile(0.75)
    q25 = x.expanding().quantile(0.25)
    iqr = q75 - q25
    out = np.where(iqr.to_numpy() == 0, (x - med).to_numpy(),
                   ((x - med) / (iqr / 1.34896)).to_numpy())
    return out


# ================================================================ 逐股特征链 ==========
def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """单股全历史特征链:README §四 第 1~6 族按股列(P1 按股分组、P15 全历史)。

    操作序逐字复刻 v2 enrich 链(基础 TA → 进阶 TA → alpha(收尾 fillna(0) +
    macd_percentile==50→NaN)→ 结构化 → lag),输入 df 已按 trade_date 升序、
    含 open/high/low/close/volume 五列(调用方保证无 NaN)。
    """
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(index=df.index)

    # ---------- 族 1:基础 TA(talib,anatomy §2.1 逐字)----------
    out["macd"], out["macd_signal"], out["macd_hist"] = talib.MACD(c)
    out["rsi_6"] = talib.RSI(c, timeperiod=6)
    out["rsi_14"] = talib.RSI(c, timeperiod=14)
    out["rsi_24"] = talib.RSI(c, timeperiod=24)
    out["ma_5"] = talib.MA(c, timeperiod=5)
    out["ma_20"] = talib.MA(c, timeperiod=20)
    out["ma_60"] = talib.MA(c, timeperiod=60)
    out["bb_upper"], out["bb_middle"], out["bb_lower"] = talib.BBANDS(c)
    out["volume_ma_20"] = talib.MA(v, timeperiod=20)
    out["obv"] = talib.OBV(c, v)
    out["atr"] = talib.ATR(h, l, c)
    out["slowk"], out["slowd"] = talib.STOCH(h, l, c)

    # ---------- 族 2:进阶 TA(anatomy §2.2 逐字)----------
    out["price_vs_ma5"] = c / out["ma_5"] - 1
    out["price_vs_ma20"] = c / out["ma_20"] - 1
    out["price_vs_ma60"] = c / out["ma_60"] - 1
    out["ma_arrangement"] = np.where(
        (out["ma_5"] > out["ma_20"]) & (out["ma_20"] > out["ma_60"]), 1,
        np.where((out["ma_5"] < out["ma_20"]) & (out["ma_20"] < out["ma_60"]), -1, 0))
    bb_width = out["bb_upper"] - out["bb_lower"]
    out["bb_position"] = np.where(bb_width > 0, (c - out["bb_lower"]) / bb_width, 0.5)
    out["bb_squeeze"] = np.where(bb_width / c < 0.05, 1, 0)
    for n in (6, 14, 24):
        out[f"rsi_oversold_{n}"] = np.where(out[f"rsi_{n}"] < 30, 1, 0)
        out[f"rsi_overbought_{n}"] = np.where(out[f"rsi_{n}"] > 70, 1, 0)
    out["macd_signal_distance"] = out["macd"] - out["macd_signal"]
    # macd_golden_cross 先在族 2 赋 distance>0,族 3 末被 macd_features_rolling 覆盖(v2 原样)
    out["macd_golden_cross"] = np.where(out["macd_signal_distance"] > 0, 1, 0)
    out["volume_ma20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    out["volume_ratio"] = np.where(out["volume_ma20"] > 0, v / out["volume_ma20"], 1)
    out["volume_spike"] = np.where(out["volume_ratio"] > 2.0, 1, 0)
    out["volume_dryup"] = np.where(out["volume_ratio"] < 0.5, 1, 0)
    out["atr_ratio"] = np.where(c > 0, out["atr"] / c, 0)
    body = np.abs(c - o)
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    total_range = h - l
    out["hammer_pattern"] = np.where(
        total_range == 0, 0,
        ((lower_shadow >= 2 * body) & (upper_shadow <= body * 0.5)
         & (lower_shadow >= total_range * 0.6)).astype(np.int64))
    out["downtrend"] = (df["close"].rolling(5).mean() < df["close"].rolling(10).mean())
    out["hammer_signal"] = (out["hammer_pattern"] == 1) & out["downtrend"]
    out["doji_pattern"] = np.where(total_range == 0, 0,
                                   (body < total_range * 0.1).astype(np.int64))
    support = df["low"].rolling(window=20, min_periods=1).min()
    resistance = df["high"].rolling(window=20, min_periods=1).max()
    d_sup = np.where(c > 0, (c - support) / c, 0.0)
    d_res = np.where(c > 0, (resistance - c) / c, 0.0)
    d_sup[:20] = 0.0   # v2 vectorized_support_resistance:前 window 行强制 0
    d_res[:20] = 0.0
    out["distance_to_support"] = d_sup
    out["distance_to_resistance"] = d_res
    out["stoch_oversold"] = np.where(out["slowk"] < 20, 1, 0)
    out["stoch_overbought"] = np.where(out["slowd"] > 80, 1, 0)
    out["macd_percentile"] = _macd_percentile_p7(out["macd"].to_numpy(dtype=np.float64))
    out["obv_trend"] = _rolling_slope_fixed(out["obv"].to_numpy(dtype=np.float64), 6)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for n in WINDOWS_VOLATILITY:
        out[f"volatility_{n}d"] = (log_ret.rolling(window=n, min_periods=n).std()
                                   * np.sqrt(252)).fillna(0)
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    prev_bearish, prev_bullish = prev_c < prev_o, prev_c > prev_o
    curr_bullish, curr_bearish = df["close"] > df["open"], df["close"] < df["open"]
    engulfing = ((prev_bearish & curr_bullish & (df["open"] <= prev_c)
                  & (df["close"] >= prev_o))
                 | (prev_bullish & curr_bearish & (df["open"] >= prev_c)
                    & (df["close"] <= prev_o))).astype(np.int64)
    engulfing.iloc[0] = 0
    out["engulfing_pattern"] = engulfing
    out["signed_volume_strength"] = out["volume_ratio"] * ((c - o) / o)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["close_vs_high"] = (h - c) / (h - l)
    vol_ma_short = df["volume"].rolling(window=5, min_periods=5).mean().shift(1)
    vol_ma_long = df["volume"].rolling(window=10, min_periods=10).mean().shift(1)
    ratio = vol_ma_short / vol_ma_long
    ratio = np.where(vol_ma_long > 0, ratio, 1.0)
    out["volume_ma_ratio"] = np.where(pd.isna(ratio), 1.0, ratio)
    rsi_diff1 = out["rsi_14"].diff(1)
    out["rsi_momentum"] = out["rsi_14"].diff().fillna(0)
    out["rsi_turning_simple"] = ((rsi_diff1.shift(1) * rsi_diff1) < 0).astype(np.int64)
    top_turn = ((out["rsi_14"].shift(1) > out["rsi_14"].shift(2))
                & (out["rsi_14"] < out["rsi_14"].shift(1)))
    bot_turn = ((out["rsi_14"].shift(1) < out["rsi_14"].shift(2))
                & (out["rsi_14"] > out["rsi_14"].shift(1)))
    out["rsi_turning"] = (top_turn | bot_turn).astype(np.int64)
    out["volume_trend_5"] = _rolling_slope_fixed(v, 6)
    out["volume_trend_10"] = _rolling_slope_fixed(v, 11)
    out["price_trend_5"] = _rolling_slope_fixed(c, 6)
    out["price_volume_divergence"] = (
        (out["price_trend_5"] * out["volume_trend_5"]) < 0).astype(np.int64)
    vol_mean6 = df["volume"].rolling(6, min_periods=6).mean()
    vol_std6 = df["volume"].rolling(6, min_periods=6).std()
    out["volume_consistency"] = np.where(vol_mean6 > 0, vol_std6 / vol_mean6, 0.0)

    # ---------- 族 3:MACD 深度族(anatomy §2.2 macd_features_rolling 逐字)----------
    hist_clean = out["macd_hist"].replace([np.inf, -np.inf], np.nan).fillna(0)
    macd_clean = out["macd"].replace([np.inf, -np.inf], np.nan).fillna(0)
    signal_clean = out["macd_signal"].replace([np.inf, -np.inf], np.nan).fillna(0)
    out["macd_hist_trend_5"] = _rolling_slope_minp2(hist_clean.to_numpy(), 6)
    out["macd_golden_cross"] = ((macd_clean.shift(1) < signal_clean.shift(1))
                                & (macd_clean > signal_clean)).astype(np.int64)
    out["macd_death_cross"] = ((macd_clean.shift(1) > signal_clean.shift(1))
                               & (macd_clean < signal_clean)).astype(np.int64)
    out["macd_signal_cross"] = out["macd_golden_cross"]
    out["macd_zero_cross_up"] = ((macd_clean.shift(1) <= 0) & (macd_clean > 0)).astype(np.int64)
    out["macd_zero_cross_down"] = ((macd_clean.shift(1) >= 0) & (macd_clean < 0)).astype(np.int64)
    out["macd_zero_cross"] = ((out["macd_zero_cross_up"] == 1)
                              | (out["macd_zero_cross_down"] == 1)).astype(np.int64)
    out["macd_hist_amplitude"] = hist_clean.rolling(window=20, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True).fillna(0)
    out["macd_hist_direction"] = np.sign(hist_clean).astype(np.int64)
    out["macd_hist_acceleration"] = hist_clean.diff().diff()
    out["macd_signal_convergence"] = macd_clean - signal_clean
    out["macd_signal_convergence_trend"] = _rolling_slope_minp2(
        out["macd_signal_convergence"].to_numpy(), 5)

    # ---------- 族 4:alpha 族按股列(anatomy §2.3 逐字;rank_return/rank_volume/
    # vol_divergence 为聚合期列,P6 分母口径)----------
    out["pct_change"] = df["close"].pct_change()
    hl_range = h - l
    out["clv"] = (c - l) / (hl_range + EPS)
    out["upper_shadow_ratio"] = (h - np.maximum(o, c)) / (hl_range + EPS)
    out["body_strength"] = (c - o) / (hl_range + EPS)
    out["signed_vol_strength"] = v * np.sign(c - o)
    out["pv_corr_10"] = df["close"].rolling(10).corr(df["volume"])
    out["dist_to_high_60"] = c / (df["close"].rolling(60).max() + EPS)
    vol_long = df["close"].rolling(60).std()          # P6 侧车用(fillna 前,NaN 跳过语义)
    out["_vol5_std"] = df["close"].rolling(5).std()   # vol_divergence 分子(中间列)
    # v2 alpha 收尾 L897:全帧 fillna(0);L898-901:macd_percentile==50 → NaN
    for col in out.columns:
        if out[col].dtype != bool:
            out[col] = out[col].fillna(0)
    out.loc[out["macd_percentile"] == 50, "macd_percentile"] = np.nan

    # ---------- 族 5:结构化族(anatomy §2.4 逐字;boxcox_atr 为聚合期列 P5)----------
    log_hl = np.log(h / (l + EPS))
    log_co = np.log(c / (o + EPS))
    with np.errstate(invalid="ignore"):
        out["vol_gk"] = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)
    out["vol_gk_ratio"] = out["vol_gk"] / out["vol_gk"].rolling(20).mean()
    out["illiq"] = np.log((df["close"].pct_change().abs() / (c * v + EPS)) + 1)
    change10 = (df["close"] - df["close"].shift(10)).abs()
    vol10 = df["close"].diff().abs().rolling(10).sum()
    out["efficiency_ratio"] = change10 / (vol10 + EPS)
    out["intraday_pos"] = (c - l) / (h - l + EPS)
    out["ret_overnight"] = (df["open"] / df["close"].shift(1)) - 1
    out["ret_intraday"] = (df["close"] / df["open"]) - 1
    out["smart_money_diff"] = out["ret_intraday"] - out["ret_overnight"]
    out["high_mean_20"] = df["high"].rolling(20).mean()
    out["low_mean_20"] = df["low"].rolling(20).mean()
    out["support_resistance_ratio"] = out["high_mean_20"] / out["low_mean_20"]
    out["log_volume"] = np.log1p(v)
    out["rsi_robust"] = _expanding_robust_z(out["rsi_14"])
    out["macd_robust"] = _expanding_robust_z(out["macd"])

    # ---------- 族 6:lag 族(anatomy §2.5 逐字,纯 shift;NaN 保留)----------
    for lag in LAG_PERIODS:
        out[f"close_lag_{lag}"] = df["close"].shift(lag)
        out[f"open_lag_{lag}"] = df["open"].shift(lag)
        out[f"high_lag_{lag}"] = df["high"].shift(lag)
        out[f"low_lag_{lag}"] = df["low"].shift(lag)
        out[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    out["daily_return"] = df["close"].pct_change()
    for lag in (1, 2, 3, 5, 10, 20):
        out[f"return_lag_{lag}"] = out["daily_return"].shift(lag)
    out["amplitude"] = (h - l) / l
    for lag in (1, 3, 5):
        out[f"amplitude_lag_{lag}"] = out["amplitude"].shift(lag)
    for lag in (1, 3, 5, 10):
        out[f"vol_gk_lag_{lag}"] = out["vol_gk"].shift(lag)
        out[f"vol_gk_ratio_lag_{lag}"] = out["vol_gk_ratio"].shift(lag)
    for lag in (1, 3, 5):
        out[f"illiq_lag_{lag}"] = out["illiq"].shift(lag)
        out[f"efficiency_ratio_lag_{lag}"] = out["efficiency_ratio"].shift(lag)
    for lag in (1, 2, 3):
        out[f"intraday_pos_lag_{lag}"] = out["intraday_pos"].shift(lag)
    for lag in (1, 3, 5):
        out[f"smart_money_diff_lag_{lag}"] = out["smart_money_diff"].shift(lag)
        out[f"ret_overnight_lag_{lag}"] = out["ret_overnight"].shift(lag)
        out[f"ret_intraday_lag_{lag}"] = out["ret_intraday"].shift(lag)
        out[f"support_resistance_ratio_lag_{lag}"] = out["support_resistance_ratio"].shift(lag)

    # ---------- 原始 volume 列(FULL 首列,参与截面)----------
    out["volume"] = df["volume"].to_numpy(dtype=np.float64)
    out.attrs["vol_long"] = vol_long.to_numpy(dtype=np.float64)  # 侧车(fillna 前口径)
    return out


# ================================================================ Pass A worker ======
def _load_stock(code: str) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_DIR / f"{code}.parquet",
                         columns=["trade_date", "open", "high", "low", "close", "vol", "amount"])
    df = df.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    assert not df["trade_date"].duplicated().any(), f"{code} trade_date 重复"
    df = df.rename(columns={"vol": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        assert not df[col].isna().any(), f"{code} {col} 含 NaN,停工待处置"
    return df


def _worker(code: str) -> tuple:
    """单股:全历史特征 → 事件日行 part + vol_long 侧车 + 事件 aux(对账断言)。"""
    try:
        df = _load_stock(code)
        n = len(df)
        feat = compute_stock_features(df)
        vol_long = feat.attrs["vol_long"]
        dts = df["trade_date"].to_numpy()
        close = df["close"].to_numpy(dtype=np.float64)
        vol = df["volume"].to_numpy(dtype=np.float64)
        amount = df["amount"].to_numpy(dtype=np.float64)
        pos = {pd.Timestamp(d): i for i, d in enumerate(dts)}

        # ---- 事件对账(README §九.4:event_close 对日线 close 逐位对账)+ aux ----
        aux_rows = []
        for ev in _G["events_by_code"].get(code, []):
            j = pos.get(ev["event_date"])
            assert j is not None, f"{code} 事件日 {ev['event_date']} 不在日线"
            assert j == int(ev["event_row"]), \
                f"{code} {ev['event_date']} 行号 {j} != 事件表 event_row {ev['event_row']}"
            assert close[j] == float(ev["event_close"]), \
                f"{code} {ev['event_date']} close {close[j]!r} != event_close {ev['event_close']!r}"
            assert vol[j] == float(ev["event_vol"]), \
                f"{code} {ev['event_date']} vol {vol[j]!r} != event_vol {ev['event_vol']!r}"
            assert amount[j] == float(ev["event_amount"]), \
                f"{code} {ev['event_date']} amount {amount[j]!r} != event_amount {ev['event_amount']!r}"
            jp = pos.get(ev["cross_prev_date"])
            assert jp is not None, \
                f"{code} 前一金叉日 {ev['cross_prev_date']} 不在日线(P12 数据缺失),停工待处置"
            aux_rows.append((ev["event_date"], vol[jp]))

        # ---- 事件日行 part ----
        ev_mask = df["trade_date"].isin(_G["event_dates_set"])
        part = feat.loc[ev_mask.to_numpy(), PER_STOCK_COLS + INTERMEDIATE_COLS].copy()
        part.insert(0, "trade_date", df.loc[ev_mask.to_numpy(), "trade_date"].to_numpy())
        part.insert(1, "ts_code", code)
        part = part.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        part.attrs = {}  # 清除自 feat 传播的 attrs(vol_long ndarray 不可序列化)
        part.to_parquet(_G["parts_dir"] / f"{code}.parquet", index=False)

        # ---- vol_long 侧车(全历史,P6 分母)----
        side = pd.DataFrame({"trade_date": dts, "vol_long": vol_long})
        side = side.dropna(subset=["vol_long"])
        side.to_parquet(_G["vollong_dir"] / f"{code}.parquet", index=False)

        # ---- 事件 aux(volume_signal 的前金叉日量,P12)----
        if aux_rows:
            aux = pd.DataFrame(aux_rows, columns=["event_date", "prev_cross_vol"])
            aux.to_parquet(_G["aux_dir"] / f"{code}.parquet", index=False)
        return code, n, int(ev_mask.sum()), None
    except Exception as e:  # noqa: BLE001
        return code, 0, 0, repr(e)


# ================================================================ 因果性抽检 =========
def causality_assert(codes: list[str], events_by_code: dict) -> dict:
    """README §九.1:抽样 12 股 × 3 事件日,截断重算(仅用 ≤ 当日数据)与全历史逐位一致。"""
    rng = np.random.default_rng(ASSERT_SEED)
    cand = sorted(c for c in codes if len(events_by_code.get(c, [])) >= ASSERT_DAYS_PER_STOCK)
    picks = rng.choice(len(cand), size=min(N_ASSERT_STOCKS, len(cand)), replace=False)
    n_checked, mismatches = 0, []
    cols = PER_STOCK_COLS + INTERMEDIATE_COLS
    for i in picks:
        code = cand[i]
        df = _load_stock(code)
        feat_full = compute_stock_features(df)
        days = sorted(ev["event_date"] for ev in events_by_code[code])
        day_picks = rng.choice(len(days), size=min(ASSERT_DAYS_PER_STOCK, len(days)),
                               replace=False)
        for jj in day_picks:
            T = days[jj]
            df_trunc = df[df["trade_date"] <= T].reset_index(drop=True)
            feat_trunc = compute_stock_features(df_trunc)
            row_full = feat_full.loc[df["trade_date"] == T, cols].iloc[0]
            row_trunc = feat_trunc[cols].iloc[-1]
            n_checked += 1
            for coln in cols:
                a, b = row_full[coln], row_trunc[coln]
                if isinstance(a, (bool, np.bool_)):
                    if bool(a) != bool(b):
                        mismatches.append(f"{code} {T.date()} {coln}: {a} != {b}")
                    continue
                if not np.isclose(float(a), float(b), rtol=1e-9, atol=0.0, equal_nan=True):
                    mismatches.append(f"{code} {T.date()} {coln}: {a!r} != {b!r}")
        log(f"  [causality] {code} 抽检 {len(day_picks)} 日完成(累计 {n_checked} 单元)")
    assert n_checked > 0, "因果性抽检为空"
    return dict(n_checked=n_checked, mismatches=mismatches)


# ================================================================ 市场族(日级) ======
def _load_index(code: str) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_DIR / f"{code}.parquet")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    assert not df["trade_date"].duplicated().any(), f"{code} 指数日期重复"
    return df


def _single_index_features(df_index: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """anatomy §2.6 单指数 7 列(指数序列截断 ≤ 当日,逐日向量化等价)。"""
    c = df_index["close"].to_numpy(dtype=np.float64)
    o = df_index["open"].to_numpy(dtype=np.float64)
    h = df_index["high"].to_numpy(dtype=np.float64)
    l = df_index["low"].to_numpy(dtype=np.float64)
    v = df_index["volume"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(index=df_index["trade_date"])
    out[f"{prefix}_price_change"] = (c - o) / o
    out[f"{prefix}_amplitude"] = (h - l) / l
    vol_ma20 = df_index["volume"].rolling(20).mean().to_numpy()
    # v2:len(index_hist)>=20 用 rolling(20).mean().iloc[-1](含当日),否则 1;avg==0 → 1
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.where(np.isfinite(vol_ma20) & (vol_ma20 != 0), v / vol_ma20, 1.0)
    out[f"{prefix}_volume_ratio"] = vr
    out[f"{prefix}_price_change_abs"] = np.abs(out[f"{prefix}_price_change"])
    out[f"{prefix}_price_wave_abs"] = out[f"{prefix}_amplitude"]
    conds = [(out[f"{prefix}_price_change"] > 0.01) & (out[f"{prefix}_amplitude"] < 0.02),
             (out[f"{prefix}_price_change"] < -0.01) & (out[f"{prefix}_amplitude"] > 0.03)]
    out[f"{prefix}_sentiment"] = np.select(conds, [2, 0], default=1).astype(np.int64)
    out[f"{prefix}_volume_signal"] = (out[f"{prefix}_volume_ratio"] > 1.2).astype(np.int64)
    return out.reset_index()


def build_market_frame(event_dates: list) -> tuple[pd.DataFrame, dict]:
    """市场大盘族 20 列 × 全部事件日(anatomy §2.6;缺失日 = v2 默认行,计数披露)。"""
    sh = _single_index_features(_load_index("000001.SH"), "sh")
    sz = _single_index_features(_load_index("399001.SZ"), "sz")
    sh = sh.set_index("trade_date")
    sz = sz.set_index("trade_date")
    ed = pd.DatetimeIndex(sorted(event_dates))
    n_sh_missing = int((~ed.isin(sh.index)).sum())
    n_sz_missing = int((~ed.isin(sz.index)).sum())
    sh_def = {f"sh_{k}": v for k, v in INDEX_DEFAULTS.items()}
    sz_def = {f"sz_{k}": v for k, v in INDEX_DEFAULTS.items()}
    sh_al = sh.reindex(ed).fillna(sh_def)
    sz_al = sz.reindex(ed).fillna(sz_def)
    mkt = pd.DataFrame(index=ed)
    for c_ in sh_al.columns:
        mkt[c_] = sh_al[c_].to_numpy()
    for c_ in sz_al.columns:
        mkt[c_] = sz_al[c_].to_numpy()
    pc_sh, pc_sz = mkt["sh_price_change"], mkt["sz_price_change"]
    mkt["sh_sz_sync_direction"] = (pc_sh * pc_sz > 0).astype(np.int64)
    mkt["sh_sz_sync_strength"] = (pc_sh - pc_sz).abs()
    avg_change = (pc_sh + pc_sz) / 2.0
    avg_amp = (mkt["sh_amplitude"] + mkt["sz_amplitude"]) / 2.0
    mkt["market_avg_change"] = avg_change
    mkt["market_avg_amplitude"] = avg_amp
    conds = [(avg_change > 0.01) & (avg_amp < 0.02), (avg_change < -0.01) & (avg_amp > 0.03)]
    mkt["market_sentiment"] = np.select(conds, [2, 0], default=1).astype(np.int64)
    max_diff = np.maximum(pc_sh, pc_sz) - np.minimum(pc_sh, pc_sz)
    mkt["market_sync_score"] = 1 - np.minimum(1.0, max_diff / 0.02)
    for c_ in ("sh_sentiment", "sz_sentiment", "market_sentiment"):
        mkt[c_] = mkt[c_].astype(np.int64)
    for c_ in ("sh_volume_signal", "sz_volume_signal", "sh_sz_sync_direction"):
        mkt[c_] = mkt[c_].astype(np.int64)
    mkt = mkt[MARKET_COLS]  # 列序 = 规格序
    info = dict(n_sh_missing=n_sh_missing, n_sz_missing=n_sz_missing)
    return mkt.reset_index(names="trade_date"), info


def build_vol_long_mean_100(codes: list[str], vollong_dir: Path) -> pd.DataFrame:
    """P6 分母:滚动 100 个市场交易日内的全市场 vol_long 均值(只用 ≤ 当日)。

    逐市场交易日 t:M_t = 当日全市场 vol_long(skipna)均值;分母 = M 在 t 及之前
    100 个市场交易日上的均值(rolling(100, min_periods=1))。
    市场交易日历 = 全市场个股日线交易日并集(2026-09-11 裁定,README 修订记录:
    指数日线文件仅近 8,000 行,SH 起于 1993-10-08 / SZ 起于 1993-09-30,
    早期交易日指数日历物理缺失;并集日历下每个事件日必有 ≥1 股 vol_long 有限)。
    """
    sums: dict = {}
    n_rows = 0
    for code in codes:
        side = pd.read_parquet(vollong_dir / f"{code}.parquet")
        for d, x in zip(side["trade_date"].to_numpy(), side["vol_long"].to_numpy()):
            key = pd.Timestamp(d)
            acc = sums.get(key)
            if acc is None:
                sums[key] = [x, 1.0]
            else:
                acc[0] += x
                acc[1] += 1.0
        n_rows += len(side)
    cal = pd.DatetimeIndex(sorted(sums))
    m_t = np.array([sums[d][0] / sums[d][1] for d in cal], dtype=np.float64)
    m_ser = pd.Series(m_t, index=cal)
    denom = m_ser.rolling(100, min_periods=1).mean()
    out = pd.DataFrame({"trade_date": m_ser.index, "vol_long_mean_100": denom.to_numpy()})
    sh_start = pd.Timestamp("1993-10-08")  # 指数文件首日(实测)
    info = dict(n_calendar_days=int(len(cal)),
                n_sidecar_rows=int(n_rows),
                n_days_pre_index_start=int((cal < sh_start).sum()),
                n_days_zero_coverage=0)  # 并集日历每日必有覆盖(构造保证)
    return out, info


# ================================================================ Pass B 聚合 ========
def _day_cross_section(frame: pd.DataFrame, mkt_row: pd.Series,
                       vol_long_mean_100: float) -> pd.DataFrame:
    """单日全市场横截面:P5/P6/rank 列 + 市场族 + 195 列 rankpct/z + cs_n。"""
    # ---- 市场族(当日广播)----
    for c_ in MARKET_COLS:
        frame[c_] = mkt_row[c_]
    # ---- 当日截面 rank(alpha 族,fillna(0) = v2 L897 收尾口径)----
    frame["rank_return"] = frame["pct_change"].rank(pct=True).fillna(0.0)
    frame["rank_volume"] = frame["volume"].rank(pct=True).fillna(0.0)
    # ---- P5:boxcox_atr 按日截面 lambda(scipy MLE;atr≤0 或 NaN → eps,v2 原样)----
    atr_day = frame["atr"].to_numpy(dtype=np.float64).copy()
    atr_day[(atr_day <= 0) | np.isnan(atr_day)] = EPS
    transformed, lmbda = stats.boxcox(atr_day)
    assert np.isfinite(lmbda), f"boxcox lambda 非有限({lmbda}),停工待处置"
    frame["boxcox_atr"] = transformed
    frame.attrs["boxcox_lambda"] = float(lmbda)
    # ---- P6:vol_divergence = 5 日波动 / (滚动 100 日全市场均值 + eps) ----
    frame["vol_divergence"] = frame["_vol5_std"] / (vol_long_mean_100 + EPS)
    frame = frame.drop(columns=INTERMEDIATE_COLS)
    # ---- 一次性守护:v2 截面 dtype 规则(anatomy L1457-1460)在本帧上推导的集合
    # 必须与规格 RPZ_BASE 逐名一致(bool 列 downtrend/hammer_signal 被规则排除)----
    if frame.attrs.get("_dtype_rule_checked") is None and not _G.get("dtype_rule_checked"):
        exclude = {"timestamp", "symbol", "label", "open", "close", "high", "low",
                   "trade_date", "ts_code"}
        label_cols = {f"{k}_{p}d" for p in RETURN_PERIODS
                      for k in ("future_return", "future_sell_date",
                                "stop_loss_return", "stop_loss_sell_date")}
        exclude |= label_cols
        rule_set = {c_ for c_ in frame.columns
                    if c_ not in exclude
                    and frame[c_].dtype in (np.float64, np.int64)}
        assert rule_set == set(RPZ_BASE), \
            f"v2 dtype 规则推导集与规格不符: 差集 {sorted(rule_set ^ set(RPZ_BASE))[:10]}"
        _G["dtype_rule_checked"] = True
    # ---- 横截面族(README §四.9:195 列 rankpct/z + cs_n)----
    present_rpz = [c_ for c_ in RPZ_BASE if c_ in frame.columns]
    assert present_rpz == RPZ_BASE, \
        f"截面列缺失: {sorted(set(RPZ_BASE) - set(present_rpz))}"
    frame["cs_n"] = np.int64(len(frame))
    for f in RPZ_BASE:
        x = frame[f]
        frame[f + "_rankpct"] = x.rank(pct=True)
        frame[f + "_z"] = (x - x.median()) / (x.std(ddof=0) + 1e-9)
    return frame


def aggregate_chunks(event_dates_sorted: list, events_by_date: dict,
                     mkt: pd.DataFrame, denom: pd.DataFrame,
                     chunks_dir: Path, eventrows_dir: Path, tag: str) -> dict:
    """按 250 日块做全市场横截面聚合,切出 96,577 事件行(README §一.2/§四.9)。

    P6 分母处置(2026-09-11 二轮裁定,README 修订记录):分母日历 = 全市场个股日线
    交易日并集(指数文件仅近 8,000 行,早期日历物理缺失),每个事件日必有分母;
    断言分母有限,停工不绕过。
    """
    mkt_ix = mkt.set_index("trade_date")
    denom_ix = denom.set_index("trade_date")["vol_long_mean_100"]
    ev_dates_set = set(pd.DatetimeIndex(event_dates_sorted))
    total_events = sum(len(v) for d, v in events_by_date.items() if d in ev_dates_set)
    found = 0
    boxcox_lambdas: list = []
    t0 = time.time()
    n_chunks = (len(event_dates_sorted) + DATE_CHUNK - 1) // DATE_CHUNK
    for ci in range(n_chunks):
        cdates = event_dates_sorted[ci * DATE_CHUNK:(ci + 1) * DATE_CHUNK]
        chunk = pd.read_parquet(chunks_dir / f"chunk_{ci:03d}.parquet")
        chunk = chunk.sort_values(["trade_date", "ts_code"], kind="mergesort") \
                     .reset_index(drop=True)
        day_codes = chunk["trade_date"].to_numpy()
        # 逐日切片(mergesort 后日期连续)
        _, starts = np.unique(day_codes, return_index=True)
        starts = list(starts) + [len(chunk)]
        ev_parts = []
        for si in range(len(starts) - 1):
            day_frame = chunk.iloc[starts[si]:starts[si + 1]].copy()
            d = day_frame["trade_date"].iloc[0]
            mkt_row = mkt_ix.loc[d]
            d_ts = pd.Timestamp(d)
            assert d_ts in denom_ix.index, f"{d} 不在 P6 分母日历,停工待处置"
            vl100 = float(denom_ix.loc[d_ts])
            assert np.isfinite(vl100), f"{d} vol_long_mean_100 非有限,停工待处置"
            day_frame = _day_cross_section(day_frame, mkt_row, vl100)
            boxcox_lambdas.append((pd.Timestamp(d), float(day_frame.attrs["boxcox_lambda"]),
                                   len(day_frame)))
            evs = events_by_date.get(pd.Timestamp(d), [])
            if evs:
                ev_codes = {e["ts_code"] for e in evs}
                sel = day_frame[day_frame["ts_code"].isin(ev_codes)]
                assert len(sel) == len(evs), \
                    f"{d} 事件行切出 {len(sel)} != 事件数 {len(evs)},停工待处置"
                ev_parts.append(sel)
                found += len(sel)
        if ev_parts:
            out = pd.concat(ev_parts, ignore_index=True)
            out.attrs = {}
            out.to_parquet(eventrows_dir / f"chunk_{ci:03d}.parquet", index=False)
        log(f"  [aggregate:{tag}] chunk {ci + 1}/{n_chunks} 完成,累计事件行 {found}"
            f"({time.time() - t0:.0f}s)")
    assert found == total_events, f"事件行 {found} != 事件表 {total_events}"
    lam = pd.DataFrame(boxcox_lambdas,
                       columns=["event_date", "boxcox_lambda", "n_stocks_day"])
    lam.to_parquet(CACHE_DIR / f"boxcox_lambda_{tag}.parquet", index=False)
    log(f"  [aggregate:{tag}] boxcox lambda 台账 {len(lam)} 日 → "
        f"boxcox_lambda_{tag}.parquet(范围 "
        f"[{lam['boxcox_lambda'].min():.4f}, {lam['boxcox_lambda'].max():.4f}])")
    return dict(n_event_rows=int(found), n_boxcox_days=int(len(lam)))


def repack_parts_to_chunks(codes: list[str], parts_dir: Path, chunks_dir: Path,
                           event_dates_sorted: list) -> dict:
    """单次遍历 per-stock parts,按 250 事件日一块重打包(确定性:按代码序读、序写)。"""
    bounds = pd.DatetimeIndex(event_dates_sorted)
    n_chunks = (len(bounds) + DATE_CHUNK - 1) // DATE_CHUNK
    writers: dict[int, pq.ParquetWriter] = {}
    schema = None
    n_rows = 0
    t0 = time.time()
    try:
        for i, code in enumerate(codes):
            part = pd.read_parquet(parts_dir / f"{code}.parquet")
            if not len(part):
                continue
            n_rows += len(part)
            cids = np.searchsorted(bounds.to_numpy(),
                                   part["trade_date"].to_numpy()) // DATE_CHUNK
            part = part.assign(_cid=cids)
            for cid, sub in part.groupby("_cid"):
                tbl = pa.Table.from_pandas(sub.drop(columns=["_cid"]),
                                           preserve_index=False)
                if schema is None:
                    schema = tbl.schema
                tbl = tbl.cast(schema)
                if cid not in writers:
                    writers[cid] = pq.ParquetWriter(chunks_dir / f"chunk_{cid:03d}.parquet",
                                                    schema)
                writers[cid].write_table(tbl)
            if (i + 1) % 500 == 0:
                log(f"  [repack] {i + 1}/{len(codes)} ({time.time() - t0:.0f}s)")
    finally:
        for w in writers.values():
            w.close()
    assert len(list(chunks_dir.glob("chunk_*.parquet"))) == n_chunks, "chunk 数不符"
    return dict(n_chunk_rows=int(n_rows), n_chunks=int(n_chunks))


# ================================================================ 主流程 =============
def main() -> None:
    ap = argparse.ArgumentParser(description="v2on6 全市场逐日基础特征面板 + 横截面聚合")
    ap.add_argument("--smoke", action="store_true", help="50 股 × 最近 200 事件日冒烟")
    ap.add_argument("--full", action="store_true", help="全量")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--rerun-tag", default="run1", help="双跑标签(确定性对账用)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="复用已落盘 parts/vollong/aux/chunks,仅重跑市场族+P6分母+聚合")
    args = ap.parse_args()
    assert args.smoke != args.full, "--smoke 与 --full 二选一"

    t_all = time.time()
    tag = "smoke" if args.smoke else args.rerun_tag
    parts_dir = CACHE_DIR / f"panel_parts_{tag}"
    vollong_dir = CACHE_DIR / f"vollong_parts_{tag}"
    aux_dir = CACHE_DIR / f"event_aux_{tag}"
    chunks_dir = CACHE_DIR / f"chunks_{tag}"
    eventrows_dir = CACHE_DIR / f"eventrows_{tag}"
    clear_dirs = (eventrows_dir,) if args.aggregate_only \
        else (parts_dir, vollong_dir, aux_dir, chunks_dir, eventrows_dir)
    for d in clear_dirs:
        if d.exists():
            for f in d.glob("*.parquet"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # ---- 输入装载 ----
    ev = pd.read_parquet(EVENTS_PATH)
    assert len(ev) == 96577 and not ev.duplicated(["ts_code", "event_date"]).any()
    event_dates_sorted = sorted(ev["event_date"].unique())
    if args.smoke:
        event_dates_sorted = event_dates_sorted[-200:]
    events_by_code: dict[str, list] = {}
    events_by_date: dict = {}
    for r in ev.itertuples(index=False):
        events_by_code.setdefault(r.ts_code, []).append(
            dict(event_date=r.event_date, event_row=r.event_row,
                 event_close=r.event_close, event_vol=r.event_vol,
                 event_amount=r.event_amount, cross_prev_date=r.cross_prev_date))
        events_by_date.setdefault(r.event_date, []).append(
            dict(ts_code=r.ts_code, event_date=r.event_date))
    all_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet")
                       if p.stem not in INDEX_CODES)
    if args.smoke:
        all_codes = all_codes[:50]
        code_set = set(all_codes)
        events_by_date = {d: [e for e in v if e["ts_code"] in code_set]
                          for d, v in events_by_date.items()}
        events_by_date = {d: v for d, v in events_by_date.items() if v}
    log(f"PANEL BUILD START | tag={tag} | 股票 {len(all_codes)} | 事件日 "
        f"{len(event_dates_sorted)} | workers={args.workers} | CPU {os.cpu_count()}")

    _G["events_by_code"] = events_by_code
    _G["event_dates_set"] = set(pd.DatetimeIndex(event_dates_sorted))
    _G["parts_dir"] = parts_dir
    _G["vollong_dir"] = vollong_dir
    _G["aux_dir"] = aux_dir

    ledger: dict = dict(tag=tag, n_codes=len(all_codes),
                        n_event_dates=len(event_dates_sorted), workers=args.workers)

    # ---- Pass A:逐股全历史特征 ----
    if args.aggregate_only:
        assert len(list(parts_dir.glob("*.parquet"))) == len(all_codes), \
            "aggregate-only:parts 不全"
        assert len(list(chunks_dir.glob("chunk_*.parquet"))) > 0, \
            "aggregate-only:chunks 缺失"
        ledger["passA"] = "reused(aggregate-only)"
        ledger["causality"] = "reused(aggregate-only,run1 已 36 单元全过)"
        log("aggregate-only:Pass A/因果性/重打包复用落盘产物")
    else:
        t0 = time.time()
        n_rows, n_ev_rows, errors = 0, 0, {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (code, n, nev, err) in enumerate(
                    ex.map(_worker, all_codes, chunksize=8)):
                if err:
                    errors[code] = err
                    log(f"  [passA-error] {code}: {err}")
                else:
                    n_rows += n
                    n_ev_rows += nev
                if (i + 1) % 500 == 0:
                    log(f"  [passA] {i + 1}/{len(all_codes)} ({time.time() - t0:.0f}s)")
        assert not errors, f"Pass A 失败 {len(errors)} 股: {dict(list(errors.items())[:3])}"
        ledger["passA"] = dict(sec=round(time.time() - t0, 1), n_hist_rows=n_rows,
                               n_event_rows=n_ev_rows)
        log(f"Pass A 完成:全历史行 {n_rows},事件日行 {n_ev_rows},"
            f"{time.time() - t0:.0f}s")

        # ---- 断言 §九.1:因果性抽检(仅 run1/smoke;重跑同种子同结论,仍执行)----
        t0 = time.time()
        causal = causality_assert(all_codes, events_by_code)
        ledger["causality"] = dict(sec=round(time.time() - t0, 1),
                                   n_checked=causal["n_checked"],
                                   n_mismatch=len(causal["mismatches"]),
                                   mismatch_head=causal["mismatches"][:5])
        assert not causal["mismatches"], \
            f"因果性抽检不一致: {causal['mismatches'][:5]}"
        log(f"因果性抽检 {causal['n_checked']} 单元全过({time.time() - t0:.0f}s)")

    # ---- 市场族 + P6 分母 ----
    t0 = time.time()
    mkt, mkt_info = build_market_frame(event_dates_sorted)
    denom, denom_info = build_vol_long_mean_100(all_codes, vollong_dir)
    ledger["market"] = dict(sec=round(time.time() - t0, 1), **mkt_info, **denom_info)
    log(f"市场族 {mkt.shape},P6 分母序列 {denom.shape}(sh 缺日 {mkt_info['n_sh_missing']},"
        f"sz 缺日 {mkt_info['n_sz_missing']},分母日历 {denom_info['n_calendar_days']} 天,"
        f"指数起点前 {denom_info['n_days_pre_index_start']} 天)({time.time() - t0:.0f}s)")

    # ---- Pass B:重打包 + 横截面聚合 ----
    t0 = time.time()
    if args.aggregate_only:
        repack = "reused(aggregate-only)"
    else:
        repack = repack_parts_to_chunks(all_codes, parts_dir, chunks_dir,
                                        event_dates_sorted)
        log(f"重打包完成:{repack}({time.time() - t0:.0f}s)")
    t1 = time.time()
    agg = aggregate_chunks(event_dates_sorted, events_by_date, mkt, denom,
                           chunks_dir, eventrows_dir, tag)
    passb = dict(sec=round(time.time() - t1, 1), **agg)
    if isinstance(repack, dict):
        passb.update(repack)
    else:
        passb["repack"] = repack
    ledger["passB"] = passb
    log(f"横截面聚合完成:事件行 {agg['n_event_rows']}({time.time() - t1:.0f}s)")

    # ---- 事件行总表 md5(双跑对账用)----
    ev_files = sorted(eventrows_dir.glob("chunk_*.parquet"))
    md5s = {}
    for f in ev_files:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk_b in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk_b)
        md5s[f.name] = h.hexdigest()
    ledger["eventrows_md5"] = md5s
    ledger["total_sec"] = round(time.time() - t_all, 1)
    with open(CACHE_DIR / f"panel_ledger_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2, default=str)
    log(f"PANEL BUILD DONE | tag={tag} | 总耗时 {time.time() - t_all:.0f}s | "
        f"事件行 {agg['n_event_rows']} | 台账 panel_ledger_{tag}.json")


if __name__ == "__main__":
    main()
