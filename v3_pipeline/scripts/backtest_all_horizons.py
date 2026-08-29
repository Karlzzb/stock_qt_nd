#!/usr/bin/env python3
"""
对 Stage 1 训练的 7 个 horizon 模型分别跑回测，用真实收益评估。
目的：验证数据截断是否是 V3 失败的主因。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from v3_pipeline.backtest.ranking_strategy import TopNEqualWeightStrategy
import json

def load_predictions(horizon: str) -> pd.DataFrame:
    """加载某个 horizon 的预测结果"""
    pred_file = project_root / "v3_pipeline" / "models" / "v3_0_1_label_selection" / f"pred_{horizon}.parquet"
    if not pred_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")

    df = pd.read_parquet(pred_file)
    print(f"  加载 {horizon} 预测: {len(df):,} 行")
    return df

def load_price_data() -> dict:
    """加载所有股票的价格数据用于回测 - 不需要，backtest 方法不用价格"""
    # Stage 3 的 backtest 直接用 predictions_df 里的 future_return 列
    # 不需要单独加载价格数据
    return {}

def run_backtest_for_horizon(horizon: str, predictions: pd.DataFrame) -> dict:
    """对单个 horizon 运行回测"""

    # 重命名列以匹配 backtest 方法期望的格式
    rename_map = {}
    if 'timestamp' in predictions.columns:
        rename_map['timestamp'] = 'date'
    if 'symbol' in predictions.columns:
        rename_map['symbol'] = 'stock_code'
    if 'prediction' in predictions.columns:
        rename_map['prediction'] = 'score'
    if 'actual_return' in predictions.columns:
        rename_map['actual_return'] = f'future_return_{horizon}'

    predictions = predictions.rename(columns=rename_map)
    predictions['date'] = pd.to_datetime(predictions['date'])

    # 验证集：2022-01-01 to 2025-07-31
    val_start = pd.Timestamp('2022-01-01')
    val_end = pd.Timestamp('2025-07-31')

    val_pred = predictions[
        (predictions['date'] >= val_start) &
        (predictions['date'] <= val_end)
    ].copy()

    print(f"  验证集数据: {len(val_pred):,} 行, {val_pred['date'].nunique()} 天")

    # 提取 holding period（从 horizon 字符串，如 "3d" -> 3）
    holding_period = int(horizon.replace('d', ''))

    # 创建交易成本模型
    from v3_pipeline.backtest.ranking_strategy import TransactionCostModel
    cost_model = TransactionCostModel(
        commission_rate=0.0003,
        slippage_rate=0.001
    )

    # 创建策略（只传 cost_model）
    strategy = TopNEqualWeightStrategy(cost_model=cost_model)

    # 运行回测（参数传给 backtest 方法）
    result = strategy.backtest(
        predictions_df=val_pred,
        n_positions=10,
        rebalance_threshold=0.7,
        holding_period=holding_period
    )

    return result

def main():
    horizons = ['3d', '5d', '10d', '15d', '20d', '25d', '30d']

    print("="*80)
    print("V3 全 Horizon 回测验证")
    print("="*80)
    print("目的：用真实收益回测验证数据截断影响")
    print("策略：Top10 equal weight, rebalance_threshold=0.7")
    print("验证集：2022-01-01 to 2025-07-31")
    print("="*80)

    # 对每个 horizon 跑回测
    results = {}

    for i, horizon in enumerate(horizons, 1):
        print(f"\n[{i+1}/{len(horizons)}] 回测 {horizon} horizon...")

        try:
            # 加载预测
            predictions = load_predictions(horizon)

            # 运行回测
            result = run_backtest_for_horizon(horizon, predictions)

            # 提取关键指标
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

        except Exception as e:
            print(f"  ✗ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            results[horizon] = {
                'annual_return': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'error': str(e)
            }

    # 汇总结果
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

    # 保存结果
    output_file = project_root / "v3_pipeline" / "results" / "all_horizons_backtest.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_file}")

    # 结论
    print("\n" + "="*80)
    print("结论分析")
    print("="*80)

    positive_horizons = [h for h, r in results.items() if not np.isnan(r['annual_return']) and r['annual_return'] > 0]

    if len(positive_horizons) > 0:
        print(f"✓ 发现 {len(positive_horizons)} 个正收益 horizon: {', '.join(positive_horizons)}")
        print("  → 说明数据截断是主要问题，生成无截断数据可能修复 V3")
    else:
        print("✗ 所有 horizon 均为负收益")
        print("  → 说明问题不只是数据截断，特征集本身可能没有边缘")
        print("  → 建议：")
        print("     1. 检查特征工程质量")
        print("     2. 考虑放弃当前特征集，回到 V2 修复")
        print("     3. 或重新设计 V3 特征体系")

if __name__ == '__main__':
    main()
