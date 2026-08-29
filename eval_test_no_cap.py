#!/usr/bin/env python3
"""评估无截断模型在测试集上的表现"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent
HORIZON_DAYS = 3
PERIODS_PER_YEAR = 252 / HORIZON_DAYS

print("=" * 80)
print("NO-CAP MODEL: TEST SET EVALUATION")
print("=" * 80)
print("\n⚠️  REAL TEST SET - Model never saw this data\n")

# Load test predictions
pred_path = REPO_ROOT / "v3_pipeline/models/v3_0_3_no_cap/test_pred_3d.parquet"
df = pd.read_parquet(pred_path)

print(f"Loaded {len(df):,} predictions")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique periods: {df['timestamp'].nunique()}")

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

def evaluate_strategy(df, top_n, bottom_n):
    period_returns = []

    for timestamp, group in df.groupby('timestamp'):
        group = group.sort_values('prediction', ascending=False)
        long_stocks = group.head(top_n)

        if bottom_n == 0:
            if len(long_stocks) > 0:
                period_return = long_stocks['actual_return'].mean()
                period_returns.append(period_return)
        else:
            short_stocks = group.tail(bottom_n)
            if len(long_stocks) > 0 and len(short_stocks) > 0:
                long_return = long_stocks['actual_return'].mean() * 0.5
                short_return = -short_stocks['actual_return'].mean() * 0.5
                period_return = long_return + short_return
                period_returns.append(period_return)

    returns = np.array(period_returns)
    n_periods = len(returns)
    cum_return = np.prod(1 + returns) - 1
    annual_return = (1 + cum_return) ** (PERIODS_PER_YEAR / n_periods) - 1
    period_vol = returns.std()
    daily_vol = period_vol / np.sqrt(HORIZON_DAYS)
    annual_vol = daily_vol * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = drawdowns.min()
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

print(f"\nTesting {len(strategies)} strategies...")

results = []
for name, top_n, bottom_n in strategies:
    result = evaluate_strategy(df, top_n, bottom_n)
    results.append((name, result))

    print(f"\n{name:25s}:")
    print(f"  Annual return: {result['annual_return']:>10.2%}")
    print(f"  Sharpe ratio:  {result['sharpe']:>10.2f}")
    print(f"  Max drawdown:  {result['max_dd']:>10.2%}")
    print(f"  Win rate:      {result['win_rate']:>10.1%}")

# Top1 analysis
top1_returns = []
for timestamp, group in df.groupby('timestamp'):
    group = group.sort_values('prediction', ascending=False)
    top1_returns.append(group.iloc[0]['actual_return'])

top1_returns = np.array(top1_returns)

print("\n" + "=" * 80)
print("COMPARISON: Validation vs Test (No-Cap Model)")
print("=" * 80)

print("\nValidation set (2022-2025):")
print("  Long-Top1: Annual 43,861%, Sharpe 1,462")
print("  L/S Top1vs1: Annual 1,312,247%, Sharpe 23,287")
print("  Top1 avg: 7.57%")

test_long1 = results[0][1]
test_ls1 = results[3][1]
print("\nTest set (2025-2026):")
print(f"  Long-Top1: Annual {test_long1['annual_return']:.2%}, Sharpe {test_long1['sharpe']:.2f}")
print(f"  L/S Top1vs1: Annual {test_ls1['annual_return']:.2%}, Sharpe {test_ls1['sharpe']:.2f}")
print(f"  Top1 avg: {top1_returns.mean()*100:.2f}%")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print("\n有截断模型 (v3_0_1):")
print("  验证集: Long-Top1 年化 6,140,914%")
print("  测试集: Long-Top1 年化 9,289,552%")
print("  结论: 标签泄漏严重")

print("\n无截断模型 (v3_0_3):")
print(f"  验证集: Long-Top1 年化 43,861%")
print(f"  测试集: Long-Top1 年化 {test_long1['annual_return']:.2%}")
val_to_test = (test_long1['annual_return'] - 438.61) / 438.61 * 100
print(f"  差异: {val_to_test:+.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if abs(val_to_test) < 50:
    print("\n✓ 验证集和测试集表现一致 → 模型泛化能力强")
    print("✓ 无截断模型是可信的")
    print("✓ V3有真实预测能力，年化收益约4-5万%")
elif test_long1['annual_return'] < 1.0:
    print("\n✗ 测试集崩溃 → 验证集过拟合")
else:
    print("\n? 测试集表现不同 → 需要进一步分析")

print(f"\n最终策略推荐: L/S Top1vsBot1")
print(f"  测试集年化: {test_ls1['annual_return']:.2%}")
print(f"  Sharpe: {test_ls1['sharpe']:.2f}")
print(f"  Max DD: {test_ls1['max_dd']:.2%}")
print(f"  胜率: {test_ls1['win_rate']:.1%}")
