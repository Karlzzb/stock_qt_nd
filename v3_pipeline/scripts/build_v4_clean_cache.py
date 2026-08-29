#!/usr/bin/env python3
"""
V4 标签重建：从原始日线重算 future_return，彻底清除止盈/止损/截断逻辑。

背景：
- 原 cache 的 future_return_*d 由 feature_pipeline_v2._calculate_original_return 生成，
  含 +15% 止盈 cap（EXPECTED_PROFIT=1.15）——标签被人为截断
- stop_loss_return_*d 是_future_模拟交易结果，混入特征造成泄漏（见 issue #27 更正）
- v3_fix_remove_capped.py 的"删除截断样本"方案系统性删除了赢家，universe 均值恶化 9 倍

本脚本：
1. 从 stock_data/daily/*.parquet 重算纯 close-to-close future return
   （沿用原约定：信号日 t 收盘买入，t+1+h 收盘卖出，h=horizon）
2. 覆盖 cache 中的 future_return_*d 列（不删任何行）
3. 物理删除 stop_loss_return_* / *_sell_date_* 列（杜绝再次泄漏）
4. 重算 rank labels，保存 feature_cache_v4_clean.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v3_pipeline.src.ranking_labels import compute_all_ranking_labels

HORIZONS = [3, 5, 10, 15, 20, 25, 30]
DAILY_DIR = REPO_ROOT / "stock_data/daily"
CACHE_IN = REPO_ROOT / "feature_cache_all.parquet"
CACHE_OUT = REPO_ROOT / "v3_pipeline/feature_cache_v4_clean.parquet"

print("=" * 80)
print("V4 LABEL REBUILD: pure close-to-close future returns")
print("=" * 80)

# === 1. 加载全部日线（只要 code/date/close） ===
print("\n1. Loading daily price files...")
files = sorted(DAILY_DIR.glob("*.parquet"))
frames = []
for f in files:
    d = pd.read_parquet(f)
    if "ts_code" not in d.columns:
        d["ts_code"] = f.stem  # 指数文件等缺 ts_code，用文件名
    frames.append(d[["ts_code", "trade_date", "close"]])
prices = pd.concat(frames, ignore_index=True)
prices["trade_date"] = pd.to_datetime(prices["trade_date"])
prices["date"] = prices["trade_date"].dt.strftime("%Y-%m-%d")
prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
print(f"   {len(files)} symbols, {len(prices):,} rows")

# === 2. 每股计算 future returns：close[t+1+h]/close[t]-1（沿用原 pipeline 约定） ===
print("\n2. Computing pure close-to-close future returns...")
g = prices.groupby("ts_code")["close"]
for h in HORIZONS:
    prices[f"future_return_{h}d"] = g.shift(-(h + 1)) / prices["close"] - 1

ret_cols = [f"future_return_{h}d" for h in HORIZONS]
label_map = prices[["ts_code", "date"] + ret_cols].copy()

# === 3. 合并进原 cache（覆盖旧 label 列） ===
print("\n3. Loading original cache and replacing labels...")
cache = pd.read_parquet(CACHE_IN)
print(f"   cache rows: {len(cache):,}")
cache["date"] = cache["timestamp"].str[:10]

old_means = {c: cache[c].mean() for c in ret_cols}
cache = cache.drop(columns=ret_cols)
cache = cache.merge(
    label_map, left_on=["symbol", "date"], right_on=["ts_code", "date"], how="left"
)
cache = cache.drop(columns=["ts_code", "date"])

matched = cache["future_return_3d"].notna().mean()
print(f"   label match rate (future_return_3d non-NaN): {matched:.2%}")

print("\n   label 均值对比（旧=止盈截断版 vs 新=纯收盘）:")
for c in ret_cols:
    print(f"   {c}: old={old_means[c]:+.4%}  new={cache[c].mean():+.4%}  "
          f"new_max={cache[c].max():+.2%}  new_min={cache[c].min():+.2%}")

# === 4. 物理删除所有未来派生列 ===
print("\n4. Dropping all future-derived feature columns...")
drop_cols = [c for c in cache.columns
             if c.startswith("stop_loss_return_")
             or c.startswith("stop_loss_sell_date_")
             or c.startswith("future_sell_date_")]
print(f"   dropping {len(drop_cols)} columns: {sorted(drop_cols)[:5]}...")
cache = cache.drop(columns=drop_cols)

# 同时删除旧的 rank 列（基于截断 label 算的），稍后重算
old_rank_cols = [c for c in cache.columns if c.startswith("rank_future_return_")]
if old_rank_cols:
    cache = cache.drop(columns=old_rank_cols)
    print(f"   dropping {len(old_rank_cols)} stale rank columns")

# === 5. 重算 rank labels ===
print("\n5. Recomputing cross-sectional rank labels...")
horizon_suffixes = [f"{h}d" for h in HORIZONS]
cache = compute_all_ranking_labels(cache, horizons=horizon_suffixes, validate=True)

# === 6. 保存 ===
print(f"\n6. Saving to {CACHE_OUT}...")
cache.to_parquet(CACHE_OUT, index=False)
size_mb = CACHE_OUT.stat().st_size / 1024 / 1024
print(f"   saved {len(cache):,} rows × {len(cache.columns)} cols, {size_mb:.0f} MB")

print("\n✓ V4 clean cache built.")
print("  Next: retrain with exclude_patterns including ^stop_loss_return_ (防御性保留，")
print("  虽然列已物理删除）")
