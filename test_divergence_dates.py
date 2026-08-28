#!/usr/bin/env python3
"""分析失败日期需要多长的窗口"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from datetime import datetime
from config.settings import DAILY_PARQUET_DIR
from src.divergence_detector_v2 import DivergenceDetectorV2
import talib

# 测试多个失败的日期
failed_dates = [
    "2010-01-04",
    "2010-02-23",
    "2010-10-13",
    "2010-11-15"
]

sample_file = list(DAILY_PARQUET_DIR.glob("000001.SZ.parquet"))[0]
df = pd.read_parquet(sample_file, columns=["trade_date", "open", "high", "low", "close", "vol"])
df = df.rename(columns={"vol": "volume"})
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()

detector = DivergenceDetectorV2()

print("分析失败日期的背离间隔:\n")

for date_str in failed_dates:
    target_date = pd.Timestamp(date_str)
    hist_data = df[df.index <= target_date].copy()

    hist_data['macd'], hist_data['macd_signal'], hist_data['macd_hist'] = talib.MACD(
        hist_data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
    )

    all_divergence = detector._detect_divergence_by_close_historical(hist_data)

    if len(all_divergence) > 0:
        all_divergence['timestamp'] = pd.to_datetime(all_divergence['timestamp'])
        last_divergence = all_divergence['timestamp'].max()
        gap_days = (target_date - last_divergence).days

        print(f"{date_str}:")
        print(f"  最后背离: {last_divergence.date()}")
        print(f"  间隔天数: {gap_days} 天")
        print()

print("\n建议：将DIVERGENCE_LOOKBACK_DAYS增加到120天（约4个月）")
