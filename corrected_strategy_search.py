#!/usr/bin/env python3
"""
修正版策略搜索 - 正确处理3d收益的复利计算
"""
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

PRED_FILE = 'v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet'
VAL_START = '2022-01-01'
VAL_END = '2025-07-31'
HORIZON_DAYS = 3  # 3天为一个周期

print("=" * 80)
print("修正版策略搜索 - 3d horizon")
print("=" * 80)

# 加载数据
print(f"\nLoading {PRED_FILE}...")
df = pd.read_parquet(PRED_FILE)
df = df.drop_duplicates(subset=['timestamp', 'symbol'])
df = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)]

print(f"Data: {df.shape[0]} samples, {df['timestamp'].nunique()} periods")
print(f"每个period = {HORIZON_DAYS}天")

# 策略参数
STRATEGIES = {
    'long_only': [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200],
    'long_short': list(product([1, 2, 3, 5, 10, 15, 20, 30, 50],
                               [1, 2, 3, 5, 10, 15, 20, 30, 50])),
    'market_neutral': [5, 10, 15, 20, 30, 50, 100],
}

all_results = []

# 1. Long-only
print("\n[1/3] Long-only strategies...")
for top_n in STRATEGIES['long_only']:
    period_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)
        top_stocks = day_data.head(top_n)

        if len(top_stocks) > 0:
            # 等权组合的3天收益
            portfolio_return = top_stocks['actual_return'].mean()
            period_returns.append(portfolio_return)

    # 计算指标（每个period是3天）
    returns_series = pd.Series(period_returns)
    n_periods = len(returns_series)

    # 累计收益
    cum_return = (1 + returns_series).prod() - 1

    # 年化收益：(1+cum)^(252/n_days) - 1，其中n_days = n_periods * 3
    n_days = n_periods * HORIZON_DAYS
    annual_return = (1 + cum_return) ** (252 / n_days) - 1

    # 年化波动率：需要转换为日波动率再年化
    # period波动率 -> 日波动率 (除以sqrt(3)) -> 年化 (乘以sqrt(252))
    period_vol = returns_series.std()
    daily_vol = period_vol / np.sqrt(HORIZON_DAYS)
    annual_vol = daily_vol * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (returns_series > 0).mean()

    all_results.append({
        'strategy': 'long_only',
        'top_n': top_n,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_drawdown,
        'win_rate': win_rate,
        'n_periods': n_periods,
    })

    print(f"  Top{top_n:3d}: {annual_return*100:7.2f}% | Sharpe: {sharpe:5.2f} | DD: {max_drawdown*100:6.1f}% | Win: {win_rate*100:5.1f}%")

# 2. Long-short
print("\n[2/3] Long-short strategies...")
for i, (top_n, bottom_n) in enumerate(STRATEGIES['long_short']):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(STRATEGIES['long_short'])}")

    period_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(top_n)
        bottom_stocks = day_data.tail(bottom_n)

        if len(top_stocks) > 0 and len(bottom_stocks) > 0:
            # Long-short: 50%做多，50%做空
            long_return = top_stocks['actual_return'].mean() * 0.5
            short_return = -bottom_stocks['actual_return'].mean() * 0.5
            portfolio_return = long_return + short_return
            period_returns.append(portfolio_return)

    if len(period_returns) > 0:
        returns_series = pd.Series(period_returns)
        n_periods = len(returns_series)

        cum_return = (1 + returns_series).prod() - 1
        n_days = n_periods * HORIZON_DAYS
        annual_return = (1 + cum_return) ** (252 / n_days) - 1

        period_vol = returns_series.std()
        daily_vol = period_vol / np.sqrt(HORIZON_DAYS)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = (returns_series > 0).mean()

        all_results.append({
            'strategy': 'long_short',
            'top_n': top_n,
            'bottom_n': bottom_n,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_drawdown,
            'win_rate': win_rate,
            'n_periods': n_periods,
        })

# 3. Market-neutral
print("\n[3/3] Market-neutral strategies...")
for n in STRATEGIES['market_neutral']:
    period_returns = []

    for date in df['timestamp'].unique():
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(n)
        bottom_stocks = day_data.tail(n)

        if len(top_stocks) > 0 and len(bottom_stocks) > 0:
            long_return = top_stocks['actual_return'].mean() * 0.5
            short_return = -bottom_stocks['actual_return'].mean() * 0.5
            portfolio_return = long_return + short_return
            period_returns.append(portfolio_return)

    if len(period_returns) > 0:
        returns_series = pd.Series(period_returns)
        n_periods = len(returns_series)

        cum_return = (1 + returns_series).prod() - 1
        n_days = n_periods * HORIZON_DAYS
        annual_return = (1 + cum_return) ** (252 / n_days) - 1

        period_vol = returns_series.std()
        daily_vol = period_vol / np.sqrt(HORIZON_DAYS)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cum_returns = (1 + returns_series).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = (returns_series > 0).mean()

        all_results.append({
            'strategy': 'market_neutral',
            'n': n,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_drawdown,
            'win_rate': win_rate,
            'n_periods': n_periods,
        })

    print(f"  N={n:3d}: {annual_return*100:7.2f}% | Sharpe: {sharpe:5.2f} | DD: {max_drawdown*100:6.1f}% | Win: {win_rate*100:5.1f}%")

# 结果汇总
print("\n" + "=" * 80)
print(f"Total strategies tested: {len(all_results)}")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('annual_return', ascending=False)

# TOP 30
print("\n" + "=" * 80)
print("TOP 30 策略 (按年化收益排序):")
print("=" * 80)

for idx in range(min(30, len(results_df))):
    row = results_df.iloc[idx]
    print(f"\n#{idx+1}: {row['strategy']}")

    if row['strategy'] == 'long_only':
        print(f"  Config: top_n={int(row['top_n'])}")
    elif row['strategy'] == 'long_short':
        print(f"  Config: top={int(row['top_n'])}, bottom={int(row['bottom_n'])}")
    elif row['strategy'] == 'market_neutral':
        print(f"  Config: n={int(row['n'])}")

    print(f"  Annual: {row['annual_return']*100:7.2f}% | Sharpe: {row['sharpe']:5.2f} | DD: {row['max_dd']*100:6.1f}% | Win: {row['win_rate']*100:5.1f}%")

# 保存
output_file = f"corrected_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_df.to_json(output_file, orient='records', indent=2)
print(f"\n✓ Results saved to: {output_file}")

# 统计
print("\n" + "=" * 80)
print("By strategy type:")
for stype in results_df['strategy'].unique():
    subset = results_df[results_df['strategy'] == stype]
    print(f"\n{stype}:")
    print(f"  Tested: {len(subset)} configs")
    print(f"  Best annual: {subset['annual_return'].max()*100:.2f}%")
    print(f"  Best Sharpe: {subset['sharpe'].max():.2f}")
    print(f"  Positive: {(subset['annual_return'] > 0).sum()}/{len(subset)}")
    print(f"  Median annual: {subset['annual_return'].median()*100:.2f}%")
