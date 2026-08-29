#!/usr/bin/env python3
"""
对 Stage 1 训练的 7 个 horizon 模型分别跑回测，用 feature_cache 的真实收益。
目的：验证数据截断是否是 V3 失败的主因。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from v3_pipeline.backtest.ranking_strategy import TopNEqualWeightStrategy, TransactionCostModel
import json

def load_predictions_and_returns(horizon: str) -> pd.DataFrame:
    """加载预测并从 feature_cache 合并真实收益"""

    # 1. 加载预测文件（只有 prediction score）
    pred_file = project_root / "v3_pipeline" / "models" / "v3_0_1_label_selection" / f"pred_{horizon}.parquet"
    pred_df = pd.read_parquet(pred_file)
    print(f"  加载预测: {len(pred_df):,} 行")

    # 2. 加载 feature_cache（有真实收益）
    cache_file = project_root / "v3_pipeline" / "feature_cache_v3.parquet"
    cache_df = pd.read_parquet(cache_file, columns=['timestamp', 'symbol', f'future_return_{horizon}'])
    print(f"  加载特征缓存: {len(cache_df):,} 行")

    # 3. 合并：用 (timestamp, symbol) 作为 key
    merged = pred_df.merge(
        cache_df,
        left_on=['timestamp', 'symbol'],
        right_on=['timestamp', 'symbol'],
        how='inner'
    )

    print(f"  合并后: {len(merged):,} 行")

    # 4. 去重（关键！）- 同一天同一只股票可能有重复
    merged = merged.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
    print(f"  去重后: {len(merged):,} 行")

    # 4. 重命名列以匹配回测接口
    merged = merged.rename(columns={
        'timestamp': 'date',
        'symbol': 'stock_code',
        'prediction': 'score',
        f'future_return_{horizon}': f'future_return_{horizon}'  # 保持原名
    })

    return merged

def run_backtest_for_horizon(horizon: str, data: pd.DataFrame) -> dict:
    """对单个 horizon 运行回测"""

    data['date'] = pd.to_datetime(data['date'])

    # 验证集：2022-01-01 to 2025-07-31
    val_start = pd.Timestamp('2022-01-01')
    val_end = pd.Timestamp('2025-07-31')

    val_data = data[
        (data['date'] >= val_start) &
        (data['date'] <= val_end)
    ].copy()

    print(f"  验证集: {len(val_data):,} 行, {val_data['date'].nunique()} 天")

    # 提取 holding period
    holding_period = int(horizon.replace('d', ''))

    # 创建策略
    cost_model = TransactionCostModel(commission_rate=0.0003, slippage_rate=0.001)
    strategy = TopNEqualWeightStrategy(cost_model=cost_model)

    # 运行回测
    result = strategy.backtest(
        predictions_df=val_data,
        n_positions=10,
        rebalance_threshold=0.7,
        holding_period=holding_period
    )

    return result

def main():
    horizons = ['3d', '5d', '10d', '15d', '20d', '25d', '30d']

    print("="*80)
    print("V3 全 Horizon 回测验证 v2")
    print("="*80)
    print("使用 feature_cache 的真实收益（未截断）")
    print("策略：Top10 equal weight, rebalance_threshold=0.7")
    print("验证集：2022-01-01 to 2025-07-31")
    print("="*80)

    results = {}

    for i, horizon in enumerate(horizons, 1):
        print(f"\n[{i}/{len(horizons)}] 回测 {horizon} horizon...")

        try:
            # 加载预测 + 真实收益
            data = load_predictions_and_returns(horizon)

            # 运行回测
            result = run_backtest_for_horizon(horizon, data)

            # 提取指标
            results[horizon] = {
                'annual_return': result.get('annual_return', np.nan),
                'sharpe_ratio': result.get('sharpe_ratio', np.nan),
                'max_drawdown': result.get('max_drawdown', np.nan),
                'win_rate': result.get('win_rate', np.nan),
                'total_return': result.get('total_return', np.nan),
                'trading_days': result.get('trading_days', 0)
            }

            print(f"  ✓ 年化收益: {result.get('annual_return', 0)*100:.2f}%")
            print(f"  ✓ Sharpe: {result.get('sharpe_ratio', 0):.3f}")
            print(f"  ✓ 最大回撤: {result.get('max_drawdown', 0)*100:.2f}%")
            print(f"  ✓ 胜率: {result.get('win_rate', 0)*100:.2f}%")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results[horizon] = {
                'annual_return': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'error': str(e)
            }

    # 汇总
    print("\n" + "="*80)
    print("回测结果汇总")
    print("="*80)
    print(f"{'Horizon':<10} {'年化收益':<12} {'Sharpe':<10} {'最大回撤':<12} {'胜率':<10}")
    print("-"*80)

    for horizon in horizons:
        r = results[horizon]
        ann_ret = r['annual_return'] * 100 if not np.isnan(r['annual_return']) else np.nan
        sharpe = r['sharpe_ratio'] if not np.isnan(r['sharpe_ratio']) else np.nan
        mdd = r['max_drawdown'] * 100 if not np.isnan(r['max_drawdown']) else np.nan
        wr = r['win_rate'] * 100 if not np.isnan(r['win_rate']) else np.nan

        print(f"{horizon:<10} {ann_ret:>10.2f}%  {sharpe:>8.3f}  {mdd:>10.2f}%  {wr:>8.2f}%")

    # 保存
    output_file = project_root / "v3_pipeline" / "results" / "all_horizons_backtest_v2.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n结果已保存: {output_file}")

    # 结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    positive_horizons = [h for h, r in results.items()
                         if not np.isnan(r['annual_return']) and r['annual_return'] > 0]

    if len(positive_horizons) > 0:
        print(f"✓ 发现 {len(positive_horizons)} 个正收益 horizon: {', '.join(positive_horizons)}")
        print("  → 数据截断是主要问题，但模型在验证集上可能还有其他问题")
        print("  → 需要进一步分析为什么正收益 horizon 在测试集失败")
    else:
        print("✗ 所有 horizon 均为负收益")
        print("  → 问题不只是数据截断")
        print("  → 特征集在 2022-2025 验证期没有profitable edge")
        print("  → 建议：")
        print("     1. 放弃当前特征集")
        print("     2. 回到 V2 修复")
        print("     3. 或彻底重新设计 V3")

if __name__ == '__main__':
    main()
