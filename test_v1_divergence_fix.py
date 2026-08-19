#!/usr/bin/env python3
"""
验证回退到 V1 后能否检测到当天的背离
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from datetime import datetime
from config.settings import DAILY_PARQUET_DIR
from src.divergence_detector import DivergenceDetector
import talib

# 加载测试股票
sample_file = list(DAILY_PARQUET_DIR.glob("000001.SZ.parquet"))[0]
df = pd.read_parquet(sample_file, columns=["trade_date", "open", "high", "low", "close", "vol"])
df = df.rename(columns={"vol": "volume"})
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date').sort_index()

# 添加 MACD
macd, signal, hist = talib.MACD(df['close'])
df['macd'] = macd
df['macd_signal'] = signal
df['macd_hist'] = hist

# 测试几个日期
test_dates = [
    datetime(2010, 6, 1).date(),
    datetime(2010, 6, 10).date(),
    datetime(2010, 6, 30).date(),
]

detector = DivergenceDetector()

print("="*60)
print("V1 背离检测器测试")
print("="*60)

for target_date in test_dates:
    # 截取到目标日期的历史数据
    hist_data = df[df.index <= pd.Timestamp(target_date)].tail(350)

    print(f"\n测试日期: {target_date}")
    print(f"历史数据: {len(hist_data)} 天 (最后一天: {hist_data.index[-1].date()})")

    # 调用 V1 的 detect_daily_divergence
    result = detector.detect_daily_divergence(hist_data, "000001.SZ", target_date)

    if result.empty:
        print(f"  ❌ 未检测到背离")
    else:
        print(f"  ✅ 检测到 {len(result)} 个背离点")
        for _, row in result.iterrows():
            print(f"     - {row['timestamp'].date()}: close={row['close_current']:.2f}, "
                  f"macd_increase={row['macd_increase_pct']:.4f}")

print("\n" + "="*60)
print("测试完成")
print("="*60)
