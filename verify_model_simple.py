#!/usr/bin/env python3
"""
最简单的手工验证：模型预测 vs 实际收益
不用任何回测框架，只看原始数据
"""

import pandas as pd
import numpy as np

# 加载预测
pred = pd.read_parquet('v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet')
pred = pred.rename(columns={'timestamp': 'date', 'symbol': 'stock_code', 'prediction': 'score', 'actual_return': 'return'})
pred['date'] = pd.to_datetime(pred['date'])

# 验证集
val = pred[(pred['date'] >= '2022-01-01') & (pred['date'] <= '2025-07-31')].copy()

print("="*80)
print("手工验证：模型预测能力")
print("="*80)

# 按日期分组
dates = sorted(val['date'].unique())

results = []

for date in dates[:50]:  # 只看前50天
    day_data = val[val['date'] == date].copy()

    if len(day_data) < 20:
        continue

    # 按 score 排序
    day_data = day_data.sort_values('score', ascending=False)

    # Top 10 vs Bottom 10
    top10 = day_data.head(10)
    bottom10 = day_data.tail(10)

    top10_return = top10['return'].mean()
    bottom10_return = bottom10['return'].mean()
    spread = top10_return - bottom10_return

    results.append({
        'date': date,
        'top10_return': top10_return,
        'bottom10_return': bottom10_return,
        'spread': spread,
        'top10_positive': (top10['return'] > 0).sum(),
        'bottom10_positive': (bottom10['return'] > 0).sum()
    })

    print(f"{date.date()}: Top10={top10_return*100:>6.2f}%, Bottom10={bottom10_return*100:>6.2f}%, Spread={spread*100:>6.2f}%")

df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("汇总统计")
print("="*80)
print(f"平均 Top10 收益:    {df_results['top10_return'].mean()*100:>6.2f}%")
print(f"平均 Bottom10 收益: {df_results['bottom10_return'].mean()*100:>6.2f}%")
print(f"平均 Spread:        {df_results['spread'].mean()*100:>6.2f}%")
print(f"Spread > 0 的天数:  {(df_results['spread'] > 0).sum()} / {len(df_results)} ({(df_results['spread'] > 0).sum()/len(df_results)*100:.1f}%)")

print("\n" + "="*80)
print("结论")
print("="*80)

avg_spread = df_results['spread'].mean()
if avg_spread > 0:
    print(f"✓ 模型有预测能力：Top10 平均比 Bottom10 高 {avg_spread*100:.2f}%")

    # 简单策略：Long Top10, Short Bottom10, 每3天换仓
    long_return = df_results['top10_return'].sum()
    short_return = -df_results['bottom10_return'].sum()
    total_return = 0.5 * long_return + 0.5 * short_return

    print(f"\n简化 Long-Short 策略（50% Long Top10 + 50% Short Bottom10）:")
    print(f"  累计收益: {total_return*100:.2f}%")
    print(f"  期数: {len(df_results)}")
    print(f"  平均每期: {total_return/len(df_results)*100:.2f}%")
else:
    print(f"✗ 模型没有预测能力：Top10 平均比 Bottom10 低 {abs(avg_spread)*100:.2f}%")
