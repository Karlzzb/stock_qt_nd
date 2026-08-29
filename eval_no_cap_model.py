#!/usr/bin/env python3
"""
测试无截断模型的策略表现。

对比：
- 有截断模型：年化614万%，Top1截断率87.2%
- 无截断模型：？

如果无截断模型仍然高收益 → 模型有真实预测能力
如果无截断模型崩溃 → 证明之前的高收益来自标签泄漏
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent
HORIZON_DAYS = 3
PERIODS_PER_YEAR = 252 / HORIZON_DAYS  # 84 periods

print("=" * 80)
print("NO-CAP MODEL STRATEGY EVALUATION")
print("=" * 80)

# Load predictions from no-cap model
pred_path = REPO_ROOT / "v3_pipeline/models/v3_0_3_no_cap/pred_3d.parquet"
print(f"\nLoading predictions from {pred_path}...")
df = pd.read_parquet(pred_path)

print(f"Loaded {len(df):,} predictions")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique periods: {df['timestamp'].nunique()}")

# Remove duplicates
df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
df = df.sort_values(['timestamp', 'prediction'], ascending=[True, False])

# Test strategies
strategies = [
    ('Long-Top1', 1, 0),
    ('Long-Top5', 5, 0),
    ('Long-Top10', 10, 0),
    ('L/S-Top1vsBot1', 1, 1),
    ('L/S-Top5vsBot5', 5, 5),
    ('L/S-Top10vsBot10', 10, 10),
]

print(f"\nTesting {len(strategies)} strategies on validation set (2022-2025)...")

def evaluate_strategy(df, top_n, bottom_n):
    """Evaluate one strategy."""
    period_returns = []

    for timestamp, group in df.groupby('timestamp'):
        group = group.sort_values('prediction', ascending=False)

        long_stocks = group.head(top_n)

        if bottom_n == 0:
            # Long-only
            if len(long_stocks) > 0:
                period_return = long_stocks['actual_return'].mean()
                period_returns.append(period_return)
        else:
            # Long-short
            short_stocks = group.tail(bottom_n)
            if len(long_stocks) > 0 and len(short_stocks) > 0:
                long_return = long_stocks['actual_return'].mean() * 0.5
                short_return = -short_stocks['actual_return'].mean() * 0.5
                period_return = long_return + short_return
                period_returns.append(period_return)

    returns = np.array(period_returns)
    n_periods = len(returns)

    # Cumulative return
    cum_return = np.prod(1 + returns) - 1

    # Annualized return
    annual_return = (1 + cum_return) ** (PERIODS_PER_YEAR / n_periods) - 1

    # Volatility
    period_vol = returns.std()
    daily_vol = period_vol / np.sqrt(HORIZON_DAYS)
    annual_vol = daily_vol * np.sqrt(252)

    # Sharpe
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = drawdowns.min()

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        'n_periods': n_periods,
        'cum_return': cum_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_period_return': returns.mean()
    }

results = []
for name, top_n, bottom_n in strategies:
    result = evaluate_strategy(df, top_n, bottom_n)
    results.append((name, result))

    print(f"\n{name:25s}:")
    print(f"  Annual return: {result['annual_return']:>10.2%}")
    print(f"  Sharpe ratio:  {result['sharpe']:>10.2f}")
    print(f"  Max drawdown:  {result['max_dd']:>10.2%}")
    print(f"  Win rate:      {result['win_rate']:>10.1%}")
    print(f"  Avg period:    {result['avg_period_return']:>10.2%}")

# Check if Top1 stocks are at the max value
print("\n" + "=" * 80)
print("CAPPING ANALYSIS (No-Cap Model)")
print("=" * 80)

top1_returns = []
at_max = 0
total = 0

for timestamp, group in df.groupby('timestamp'):
    group = group.sort_values('prediction', ascending=False)
    top1_return = group.iloc[0]['actual_return']
    top1_returns.append(top1_return)

    # Check if at maximum (allowing small tolerance)
    group_max = group['actual_return'].max()
    if abs(top1_return - group_max) < 0.0001:
        at_max += 1
    total += 1

top1_returns = np.array(top1_returns)

print(f"\nTop1 stock analysis:")
print(f"  Mean return: {top1_returns.mean():.4f} ({top1_returns.mean()*100:.2f}%)")
print(f"  Median: {np.median(top1_returns):.4f}")
print(f"  Max: {top1_returns.max():.4f}")
print(f"  Min: {top1_returns.min():.4f}")
print(f"  Std: {top1_returns.std():.4f}")
print(f"\n  Periods where Top1 is the best: {at_max}/{total} ({at_max/total:.1%})")

# Compare with capped model
print("\n" + "=" * 80)
print("COMPARISON: Capped vs No-Cap Model")
print("=" * 80)

print("\nCapped model (v3_0_1):")
print("  Long-Top1: Annual 6,140,914%, Sharpe 237,562")
print("  Top1 截断率: 87.2%")
print("  Top1 平均收益: 14.06%")

print("\nNo-Cap model (v3_0_3):")
best_long = results[0][1]  # Long-Top1
print(f"  Long-Top1: Annual {best_long['annual_return']:.2%}, Sharpe {best_long['sharpe']:.2f}")
print(f"  Top1 is best: {at_max/total:.1%}")
print(f"  Top1 平均收益: {top1_returns.mean()*100:.2f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if best_long['annual_return'] > 1.0:  # > 100%
    print("\n✓ No-Cap模型仍然有高收益 → 模型有真实预测能力")
    print("  标签泄漏假设被推翻")
elif best_long['annual_return'] < 0.2:  # < 20%
    print("\n✓ No-Cap模型收益崩溃 → 证明标签泄漏假设正确")
    print("  之前的高收益来自识别截断股票")
else:
    print("\n? No-Cap模型表现中等 → 需要进一步分析")

print(f"\n性能下降：{(1 - best_long['annual_return'] / 61409.14) * 100:.1f}%")
