#!/usr/bin/env python3
"""
使用 Stage 3 现有框架测试更多策略类型
重点：Long-Short 和 Market Neutral 策略
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from v3_pipeline.backtest.ranking_strategy import (
    TopNEqualWeightStrategy,
    ScoreWeightedStrategy,
    TransactionCostModel
)
import json

def load_predictions(horizon='3d'):
    """加载预测"""
    pred_file = project_root / "v3_pipeline" / "models" / "v3_0_1_label_selection" / f"pred_{horizon}.parquet"
    pred_df = pd.read_parquet(pred_file)

    pred_df = pred_df.rename(columns={
        'timestamp': 'date',
        'symbol': 'stock_code',
        'prediction': 'score',
        'actual_return': f'future_return_{horizon}'
    })

    pred_df['date'] = pd.to_datetime(pred_df['date'])

    # 验证集
    val_df = pred_df[(pred_df['date'] >= '2022-01-01') & (pred_df['date'] <= '2025-07-31')].copy()

    return val_df

class LongShortStrategy(TopNEqualWeightStrategy):
    """Long-Short 策略：做多 top N，做空 bottom N"""

    def backtest(self, predictions_df, n_long, n_short, rebalance_threshold=0.5,
                 initial_capital=1000000, holding_period=3):
        """
        Long-short backtest

        Args:
            n_long: 做多股票数
            n_short: 做空股票数
        """
        self.net_value_history = []
        self.position_history = []
        self.trade_history = []
        current_capital = initial_capital
        self.positions = {}

        dates = sorted(predictions_df['date'].unique())

        print(f"\nLong-Short Backtest: Long {n_long}, Short {n_short}, Holding {holding_period}d")

        i = 0
        while i < len(dates):
            date = dates[i]
            day_data = predictions_df[predictions_df['date'] == date].copy()

            if len(day_data) < n_long + n_short:
                i += holding_period
                continue

            # 选择 long 和 short
            long_stocks = day_data.nlargest(n_long, 'score')['stock_code'].tolist()
            short_stocks = day_data.nsmallest(n_short, 'score')['stock_code'].tolist()

            # 计算收益（只用 3d，因为数据里只有这列）
            return_col = 'future_return_3d'

            # Long 部分：等权重
            long_return = 0
            for stock in long_stocks:
                stock_return = day_data[day_data['stock_code'] == stock][return_col].iloc[0]
                if pd.notna(stock_return):
                    long_return += stock_return / n_long

            # Short 部分：等权重，反向收益
            short_return = 0
            for stock in short_stocks:
                stock_return = day_data[day_data['stock_code'] == stock][return_col].iloc[0]
                if pd.notna(stock_return):
                    short_return += (-stock_return) / n_short

            # 总收益（50% long + 50% short）
            portfolio_return = 0.5 * long_return + 0.5 * short_return

            # 交易成本（简化：假设每次全部换仓）
            turnover = 2.0  # Long 100% + Short 100%
            cost = turnover * (0.0003 + 0.001)

            # 更新资金
            current_capital = current_capital * (1 + portfolio_return - cost)
            self.net_value_history.append(current_capital)

            i += holding_period

        # 计算指标
        if len(self.net_value_history) == 0:
            return None

        returns = pd.Series(self.net_value_history).pct_change().dropna()

        total_return = (self.net_value_history[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(self.net_value_history)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252 / holding_period) if returns.std() > 0 else 0

        peak = self.net_value_history[0]
        max_dd = 0
        for value in self.net_value_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        win_rate = (returns > 0).sum() / len(returns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'final_capital': self.net_value_history[-1]
        }

def test_long_short_strategies(data):
    """测试各种 Long-Short 配置"""

    results = []

    configs = [
        # (n_long, n_short, holding_period) - 只用 3d，因为数据限制
        (10, 10, 3),
        (20, 20, 3),
        (30, 30, 3),
        (50, 50, 3),
        (100, 100, 3),
        # 不对称 long-short
        (20, 10, 3),
        (30, 10, 3),
        (10, 20, 3),
        (10, 30, 3),
        (50, 30, 3),
        (30, 50, 3),
        (100, 50, 3),
        (50, 100, 3),
    ]

    print("="*80)
    print("测试 Long-Short 策略")
    print("="*80)

    for n_long, n_short, holding in configs:
        strategy = LongShortStrategy()
        result = strategy.backtest(
            data,
            n_long=n_long,
            n_short=n_short,
            holding_period=holding
        )

        if result:
            result['config'] = {'n_long': n_long, 'n_short': n_short, 'holding_period': holding}
            results.append(result)

            print(f"  Long{n_long}/Short{n_short}/{holding}d: "
                  f"年化={result['annual_return']*100:>7.2f}%, "
                  f"Sharpe={result['sharpe_ratio']:>6.3f}, "
                  f"MaxDD={result['max_drawdown']*100:>6.2f}%")

    return results

def test_more_long_only_configs(data):
    """测试更多 long-only 配置"""

    results = []

    # 更激进的参数（只用 3d）
    configs = [
        # Top N, holding_period (都用 3)
        (3, 3),
        (5, 3),
        (10, 3),
        (20, 3),
        (50, 3),
        (100, 3),
        (200, 3),
    ]

    print("\n" + "="*80)
    print("测试更多 Long-Only 配置")
    print("="*80)

    for n, holding in configs:
        cost_model = TransactionCostModel(commission_rate=0.0003, slippage_rate=0.001)
        strategy = TopNEqualWeightStrategy(cost_model=cost_model)

        try:
            result = strategy.backtest(
                data,
                n_positions=n,
                rebalance_threshold=0.7,
                holding_period=holding
            )

            if result:
                result['config'] = {'n_positions': n, 'holding_period': holding, 'type': 'long_only'}
                results.append(result)

                print(f"  Top{n:3d}/{holding}d: "
                      f"年化={result['annual_return']*100:>7.2f}%, "
                      f"Sharpe={result['sharpe_ratio']:>6.3f}, "
                      f"MaxDD={result['max_drawdown']*100:>6.2f}%")
        except Exception as e:
            print(f"  Top{n}/{holding}d: 失败 - {e}")

    return results

def main():
    print("="*80)
    print("V3 策略扩展搜索")
    print("="*80)
    print("使用 Stage 3 现有回测框架，测试 Long-Short 和更多配置")
    print("="*80)

    # 加载数据
    print("\n加载 3d horizon 数据...")
    data = load_predictions('3d')
    print(f"验证集: {len(data):,} 行, {data['date'].nunique()} 天")

    # 测试 Long-Short
    longshort_results = test_long_short_strategies(data)

    # 测试更多 Long-Only
    longonly_results = test_more_long_only_configs(data)

    # 合并结果
    all_results = longshort_results + longonly_results

    # 保存
    output_file = project_root / "v3_pipeline" / "results" / "extended_strategy_search.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存: {output_file}")

    # 找最佳策略
    print("\n" + "="*80)
    print("最佳策略（按 Sharpe）")
    print("="*80)

    positive_results = [r for r in all_results if r['annual_return'] > 0]

    if len(positive_results) > 0:
        positive_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

        print(f"\n找到 {len(positive_results)} 个正收益策略\n")

        for i, result in enumerate(positive_results[:10]):
            config = result['config']
            print(f"[{i+1}] {config}")
            print(f"    年化收益: {result['annual_return']*100:>8.2f}%")
            print(f"    Sharpe:   {result['sharpe_ratio']:>8.3f}")
            print(f"    最大回撤: {result['max_drawdown']*100:>8.2f}%")
            print(f"    胜率:     {result['win_rate']*100:>8.2f}%")
            print()
    else:
        print("\n⚠️  没有找到正收益策略")
        print("\n最小亏损的策略：")
        all_results.sort(key=lambda x: x['annual_return'], reverse=True)
        for i, result in enumerate(all_results[:5]):
            config = result['config']
            print(f"[{i+1}] {config}")
            print(f"    年化收益: {result['annual_return']*100:>8.2f}%")
            print(f"    Sharpe:   {result['sharpe_ratio']:>8.3f}")
            print()

if __name__ == '__main__':
    main()
