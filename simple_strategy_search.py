#!/usr/bin/env python3
"""
简化策略搜索 - 只用prediction和actual_return
不需要价格，直接计算组合收益
"""
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

PRED_FILE = 'v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet'
VAL_START = '2022-01-01'
VAL_END = '2025-07-31'

print("=" * 80)
print("简化策略搜索 - 3d horizon")
print("=" * 80)

# 加载数据
print(f"\nLoading {PRED_FILE}...")
df = pd.read_parquet(PRED_FILE)
df = df.drop_duplicates(subset=['timestamp', 'symbol'])
df = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)]

print(f"Data: {df.shape[0]} samples, {df['timestamp'].nunique()} days")
print(f"Columns: {df.columns.tolist()}")

# 策略参数
STRATEGIES = {
    'long_only': {
        'top_n': [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200],
    },
    'long_short': {
        'top_n': [1, 2, 3, 5, 10, 15, 20, 30, 50],
        'bottom_n': [1, 2, 3, 5, 10, 15, 20, 30, 50],
    },
    'market_neutral': {
        'n': [5, 10, 15, 20, 30, 50, 100],
    },
}

all_results = []

# 1. Long-only: 等权买入Top N
print("\n[1/3] Long-only strategies...")
for top_n in STRATEGIES['long_only']['top_n']:
    daily_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(top_n)

        if len(top_stocks) > 0:
            # 等权组合收益 = 平均actual_return
            portfolio_return = top_stocks['actual_return'].mean()
            daily_returns.append(portfolio_return)

    # 计算指标
    returns_series = pd.Series(daily_returns)
    cum_return = (1 + returns_series).prod() - 1
    annual_return = (1 + cum_return) ** (252 / len(returns_series)) - 1
    annual_vol = returns_series.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    all_results.append({
        'strategy': 'long_only',
        'top_n': top_n,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_drawdown,
        'win_rate': (returns_series > 0).mean(),
    })

    print(f"  Top{top_n}: {annual_return*100:6.2f}% | Sharpe: {sharpe:5.2f} | DD: {max_drawdown*100:6.1f}%")

# 2. Long-short: Top N多 + Bottom N空
print("\n[2/3] Long-short strategies...")
configs = list(product(STRATEGIES['long_short']['top_n'], STRATEGIES['long_short']['bottom_n']))

for i, (top_n, bottom_n) in enumerate(configs):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(configs)}")

    daily_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(top_n)
        bottom_stocks = day_data.tail(bottom_n)

        if len(top_stocks) > 0 and len(bottom_stocks) > 0:
            # Long-short: 50%资金做多，50%资金做空
            long_return = top_stocks['actual_return'].mean() * 0.5
            short_return = -bottom_stocks['actual_return'].mean() * 0.5  # 做空：股价涨亏钱，跌赚钱
            portfolio_return = long_return + short_return
            daily_returns.append(portfolio_return)

    if len(daily_returns) > 0:
        returns_series = pd.Series(daily_returns)
        cum_return = (1 + returns_series).prod() - 1
        annual_return = (1 + cum_return) ** (252 / len(returns_series)) - 1
        annual_vol = returns_series.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        all_results.append({
            'strategy': 'long_short',
            'top_n': top_n,
            'bottom_n': bottom_n,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_drawdown,
            'win_rate': (returns_series > 0).mean(),
        })

# 3. Market-neutral: Top N多 + Bottom N空 (N相等)
print("\n[3/3] Market-neutral strategies...")
for n in STRATEGIES['market_neutral']['n']:
    daily_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(n)
        bottom_stocks = day_data.tail(n)

        if len(top_stocks) > 0 and len(bottom_stocks) > 0:
            # Market-neutral: 50% long + 50% short
            long_return = top_stocks['actual_return'].mean() * 0.5
            short_return = -bottom_stocks['actual_return'].mean() * 0.5
            portfolio_return = long_return + short_return
            daily_returns.append(portfolio_return)

    if len(daily_returns) > 0:
        returns_series = pd.Series(daily_returns)
        cum_return = (1 + returns_series).prod() - 1
        annual_return = (1 + cum_return) ** (252 / len(returns_series)) - 1
        annual_vol = returns_series.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        all_results.append({
            'strategy': 'market_neutral',
            'n': n,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_drawdown,
            'win_rate': (returns_series > 0).mean(),
        })

    print(f"  N={n}: {annual_return*100:6.2f}% | Sharpe: {sharpe:5.2f} | DD: {max_drawdown*100:6.1f}%")

# 结果汇总
print("\n" + "=" * 80)
print(f"Total strategies tested: {len(all_results)}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('annual_return', ascending=False)

# TOP 30
print("\n" + "=" * 80)
print("TOP 30 策略:")
print("=" * 80)

for idx in range(min(30, len(results_df))):
    row = results_df.iloc[idx]
    print(f"\n#{idx+1}: {row['strategy']}")

    if row['strategy'] == 'long_only':
        print(f"  Config: top_n={row['top_n']}")
    elif row['strategy'] == 'long_short':
        print(f"  Config: top={row['top_n']}, bottom={row['bottom_n']}")
    elif row['strategy'] == 'market_neutral':
        print(f"  Config: n={row['n']}")

    print(f"  Annual: {row['annual_return']*100:6.2f}% | Sharpe: {row['sharpe']:5.2f} | DD: {row['max_dd']*100:6.1f}% | Win: {row['win_rate']*100:5.1f}%")

# 保存
output_file = f"simple_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_df.to_json(output_file, orient='records', indent=2)
print(f"\n✓ Results saved to: {output_file}")

# 统计
print("\n" + "=" * 80)
print("By strategy type:")
for stype in results_df['strategy'].unique():
    subset = results_df[results_df['strategy'] == stype]
    best = subset.iloc[0] if len(subset) > 0 else None
    if best is not None:
        print(f"\n{stype}:")
        print(f"  Best annual return: {best['annual_return']*100:.2f}%")
        print(f"  Best Sharpe: {subset['sharpe'].max():.2f}")
        print(f"  Positive count: {(subset['annual_return'] > 0).sum()}/{len(subset)}")
