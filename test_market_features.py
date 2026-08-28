#!/usr/bin/env python3
"""
测试 _calculate_market_features 返回什么
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from datetime import datetime
from config.settings import DAILY_PARQUET_DIR

# 加载指数数据
df_sh = pd.read_parquet(DAILY_PARQUET_DIR / "000001.SH.parquet")
df_sh = df_sh.rename(columns={'trade_date': 'timestamp', 'vol': 'volume'})
df_sh['timestamp'] = pd.to_datetime(df_sh['timestamp'])
df_sh = df_sh.set_index('timestamp').sort_index()

df_sz = pd.read_parquet(DAILY_PARQUET_DIR / "399001.SZ.parquet")
df_sz = df_sz.rename(columns={'trade_date': 'timestamp', 'vol': 'volume'})
df_sz['timestamp'] = pd.to_datetime(df_sz['timestamp'])
df_sz = df_sz.set_index('timestamp').sort_index()

# 测试 _calculate_market_features
from src.divergence_detector import DivergenceDetector
from src.feature_pipeline_v2 import FeaturePipeline

detector = DivergenceDetector()
pipeline = FeaturePipeline(detector, {})

test_date = pd.Timestamp('2022-07-01')

print(f"测试日期: {test_date}")
print(f"df_sh: {len(df_sh)} 行")
print(f"df_sz: {len(df_sz)} 行")

market_df = pipeline._calculate_market_features(test_date, df_sh=df_sh, df_sz=df_sz)

print(f"\n返回的 DataFrame 形状: {market_df.shape}")
print(f"列数: {len(market_df.columns)}")
print(f"\n所有列:")
for col in sorted(market_df.columns):
    print(f"  {col}: {market_df[col].iloc[0]}")
