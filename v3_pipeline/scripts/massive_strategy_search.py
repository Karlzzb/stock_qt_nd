#!/usr/bin/env python3
"""
大规模策略搜索：测试各种可能的策略组合
目标：找到能将 V3 模型的强预测能力（IC=0.98）转化为盈利的策略
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from itertools import product
import json
from tqdm import tqdm

def load_model_predictions(horizon='3d'):
    """加载模型预测和真实收益"""
    pred_file = project_root / "v3_pipeline" / "models" / "v3_0_1_label_selection" / f"pred_{horizon}.parquet"
    pred_df = pd.read_parquet(pred_file)

    # 重命名列
    pred_df = pred_df.rename(columns={
        'timestamp': 'date',
        'symbol': 'stock_code',
        'prediction': 'score',
        'actual_return': 'return'
    })

    pred_df['date'] = pd.to_datetime(pred_df['date'])

    # 验证集
    val_df = pred_df[(pred_df['date'] >= '2022-01-01') & (pred_df['date'] <= '2025-07-31')].copy()

    return val_df

class StrategyBacktest:
    """通用策略回测引擎"""

    def __init__(self, predictions_df, initial_capital=1000000):
        self.data = predictions_df.sort_values('date')
        self.initial_capital = initial_capital
        self.dates = sorted(self.data['date'].unique())

    def backtest_strategy(self, strategy_config):
        """
        回测一个策略配置

        strategy_config = {
            'type': 'long_only' | 'long_short' | 'market_neutral',
            'selection_method': 'top_n' | 'threshold' | 'percentile' | 'dynamic',
            'weighting': 'equal' | 'score' | 'inverse_rank' | 'exponential',
            'rebalance_freq': 1-30 (天数),
            'position_sizing': dict with params,
            'risk_control': dict with params,
        }
        """

        capital = self.initial_capital
        positions = {}
        equity_curve = []

        holding_period = strategy_config.get('rebalance_freq', 3)

        for i in range(0, len(self.dates), holding_period):
            if i >= len(self.dates):
                break

            date = self.dates[i]
            day_data = self.data[self.data['date'] == date].copy()

            if len(day_data) == 0:
                continue

            # 选择股票
            selected_stocks = self._select_stocks(day_data, strategy_config)

            if len(selected_stocks) == 0:
                continue

            # 计算权重
            weights = self._calculate_weights(selected_stocks, strategy_config)

            # 计算该周期收益
            portfolio_return = (selected_stocks['return'] * weights).sum()

            # 交易成本
            turnover = self._calculate_turnover(positions, dict(zip(selected_stocks['stock_code'], weights)))
            cost = turnover * (0.0003 + 0.001)  # 佣金 + 滑点

            # 更新资金
            capital = capital * (1 + portfolio_return - cost)
            equity_curve.append(capital)

            # 更新持仓
            positions = dict(zip(selected_stocks['stock_code'], weights))

        if len(equity_curve) == 0:
            return None

        # 计算指标
        returns = pd.Series(equity_curve).pct_change().dropna()

        if len(returns) == 0:
            return None

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252 / holding_period) if returns.std() > 0 else 0
        max_dd = self._calculate_max_drawdown(equity_curve)
        win_rate = (returns > 0).sum() / len(returns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'n_periods': len(equity_curve),
            'final_capital': equity_curve[-1]
        }

    def _select_stocks(self, day_data, config):
        """选择股票"""
        method = config['selection_method']
        stype = config['type']

        if method == 'top_n':
            n = config.get('n_long', 10)
            if stype == 'long_only':
                return day_data.nlargest(n, 'score')
            elif stype == 'long_short':
                n_short = config.get('n_short', 10)
                long_stocks = day_data.nlargest(n, 'score').copy()
                short_stocks = day_data.nsmallest(n_short, 'score').copy()
                long_stocks['position_type'] = 'long'
                short_stocks['position_type'] = 'short'
                short_stocks['return'] = -short_stocks['return']  # 做空反向
                return pd.concat([long_stocks, short_stocks])
            elif stype == 'market_neutral':
                long_stocks = day_data.nlargest(n, 'score').copy()
                short_stocks = day_data.nsmallest(n, 'score').copy()
                long_stocks['position_type'] = 'long'
                short_stocks['position_type'] = 'short'
                short_stocks['return'] = -short_stocks['return']
                return pd.concat([long_stocks, short_stocks])

        elif method == 'percentile':
            pct_long = config.get('pct_long', 0.05)  # top 5%
            threshold = day_data['score'].quantile(1 - pct_long)

            if stype == 'long_only':
                return day_data[day_data['score'] >= threshold]
            elif stype in ['long_short', 'market_neutral']:
                pct_short = config.get('pct_short', 0.05)
                threshold_short = day_data['score'].quantile(pct_short)
                long_stocks = day_data[day_data['score'] >= threshold].copy()
                short_stocks = day_data[day_data['score'] <= threshold_short].copy()
                long_stocks['position_type'] = 'long'
                short_stocks['position_type'] = 'short'
                short_stocks['return'] = -short_stocks['return']
                return pd.concat([long_stocks, short_stocks])

        elif method == 'threshold':
            score_threshold = config.get('score_threshold', 0)
            selected = day_data[day_data['score'] > score_threshold].copy()
            if stype == 'long_only':
                return selected

        elif method == 'dynamic':
            # 根据 IC 强度动态调整仓位数
            mean_score = day_data['score'].mean()
            std_score = day_data['score'].std()
            z_scores = (day_data['score'] - mean_score) / std_score

            n_base = config.get('n_base', 20)
            z_threshold = config.get('z_threshold', 1.5)

            selected = day_data[np.abs(z_scores) > z_threshold].copy()
            if len(selected) > n_base:
                selected = selected.nlargest(n_base, 'score')
            return selected

        return day_data.iloc[:0]  # 空 DataFrame

    def _calculate_weights(self, stocks, config):
        """计算权重"""
        weighting = config['weighting']

        if weighting == 'equal':
            return np.ones(len(stocks)) / len(stocks)

        elif weighting == 'score':
            scores = stocks['score'].values
            scores = scores - scores.min() + 1e-8  # 避免负数
            weights = scores / scores.sum()
            return weights

        elif weighting == 'inverse_rank':
            ranks = stocks['score'].rank(ascending=False)
            inv_ranks = 1.0 / ranks
            weights = inv_ranks / inv_ranks.sum()
            return weights

        elif weighting == 'exponential':
            alpha = config.get('weight_alpha', 0.1)
            scores = stocks['score'].values
            scores_norm = (scores - scores.mean()) / (scores.std() + 1e-8)
            weights = np.exp(alpha * scores_norm)
            weights = weights / weights.sum()
            return weights

        elif weighting == 'volatility_adjusted':
            # 简化：用历史标准差倒数加权（这里用 score 作为proxy）
            scores = np.abs(stocks['score'].values) + 0.1
            inv_vol = 1.0 / scores
            weights = inv_vol / inv_vol.sum()
            return weights

        return np.ones(len(stocks)) / len(stocks)

    def _calculate_turnover(self, old_positions, new_positions):
        """计算换手率"""
        old_set = set(old_positions.keys())
        new_set = set(new_positions.keys())

        turnover = 0.0

        # 卖出的
        for stock in old_set - new_set:
            turnover += old_positions[stock]

        # 买入的
        for stock in new_set - old_set:
            turnover += new_positions[stock]

        # 调整的
        for stock in old_set & new_set:
            turnover += abs(new_positions[stock] - old_positions[stock])

        return turnover

    def _calculate_max_drawdown(self, equity_curve):
        """计算最大回撤"""
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

def generate_strategy_space():
    """生成策略搜索空间"""

    strategies = []

    # 1. Long-only 策略
    for selection in ['top_n', 'percentile']:
        for weighting in ['equal', 'score', 'inverse_rank', 'exponential']:
            for rebalance in [1, 3, 5, 10, 15, 20]:
                if selection == 'top_n':
                    for n in [5, 10, 15, 20, 30, 50, 100]:
                        strategies.append({
                            'type': 'long_only',
                            'selection_method': selection,
                            'weighting': weighting,
                            'rebalance_freq': rebalance,
                            'n_long': n
                        })
                elif selection == 'percentile':
                    for pct in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
                        strategies.append({
                            'type': 'long_only',
                            'selection_method': selection,
                            'weighting': weighting,
                            'rebalance_freq': rebalance,
                            'pct_long': pct
                        })

    # 2. Long-short 策略
    for selection in ['top_n', 'percentile']:
        for weighting in ['equal', 'score']:
            for rebalance in [3, 5, 10, 15, 20]:
                if selection == 'top_n':
                    for n_long in [10, 20, 30]:
                        for n_short in [10, 20, 30]:
                            strategies.append({
                                'type': 'long_short',
                                'selection_method': selection,
                                'weighting': weighting,
                                'rebalance_freq': rebalance,
                                'n_long': n_long,
                                'n_short': n_short
                            })
                elif selection == 'percentile':
                    for pct in [0.05, 0.10, 0.15]:
                        strategies.append({
                            'type': 'long_short',
                            'selection_method': selection,
                            'weighting': weighting,
                            'rebalance_freq': rebalance,
                            'pct_long': pct,
                            'pct_short': pct
                        })

    # 3. Market neutral 策略
    for selection in ['top_n', 'percentile']:
        for weighting in ['equal', 'score']:
            for rebalance in [5, 10, 15]:
                if selection == 'top_n':
                    for n in [10, 20, 30, 50]:
                        strategies.append({
                            'type': 'market_neutral',
                            'selection_method': selection,
                            'weighting': weighting,
                            'rebalance_freq': rebalance,
                            'n_long': n,
                            'n_short': n
                        })

    print(f"生成了 {len(strategies)} 个策略配置")
    return strategies

def main():
    print("="*80)
    print("V3 大规模策略搜索")
    print("="*80)
    print("目标：找到能将模型预测（IC=0.98）转化为盈利的策略")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    data = load_model_predictions('3d')
    print(f"验证集: {len(data):,} 行, {data['date'].nunique()} 天")

    # 生成策略空间
    print("\n生成策略空间...")
    strategies = generate_strategy_space()

    # 创建回测引擎
    backtester = StrategyBacktest(data)

    # 批量回测
    print(f"\n开始回测 {len(strategies)} 个策略...")
    print("这可能需要一些时间...\n")

    results = []
    positive_strategies = []

    for i, strategy in enumerate(tqdm(strategies)):
        result = backtester.backtest_strategy(strategy)

        if result is not None:
            result['strategy'] = strategy
            results.append(result)

            # 记录正收益策略
            if result['annual_return'] > 0:
                positive_strategies.append(result)

        # 每1000个输出一次进度
        if (i + 1) % 1000 == 0:
            n_positive = len([r for r in results if r['annual_return'] > 0])
            print(f"\n已测试 {i+1}/{len(strategies)}, 找到 {n_positive} 个正收益策略")

    # 保存所有结果
    output_file = project_root / "v3_pipeline" / "results" / "massive_strategy_search.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n完整结果已保存: {output_file}")

    # 分析结果
    print("\n" + "="*80)
    print("搜索结果汇总")
    print("="*80)

    print(f"\n总测试策略数: {len(results)}")
    print(f"正收益策略数: {len(positive_strategies)}")
    print(f"正收益率: {len(positive_strategies)/len(results)*100:.2f}%")

    if len(positive_strategies) > 0:
        # 按 Sharpe 排序
        positive_strategies.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

        print("\n" + "="*80)
        print("Top 10 策略（按 Sharpe 排序）")
        print("="*80)

        for i, result in enumerate(positive_strategies[:10]):
            strategy = result['strategy']
            print(f"\n[{i+1}] {strategy['type']} - {strategy['selection_method']} - {strategy['weighting']}")
            print(f"    参数: {json.dumps({k: v for k, v in strategy.items() if k not in ['type', 'selection_method', 'weighting']}, indent=8)}")
            print(f"    年化收益: {result['annual_return']*100:>8.2f}%")
            print(f"    Sharpe:   {result['sharpe_ratio']:>8.3f}")
            print(f"    最大回撤: {result['max_drawdown']*100:>8.2f}%")
            print(f"    胜率:     {result['win_rate']*100:>8.2f}%")

        # 保存最佳策略
        best_file = project_root / "v3_pipeline" / "results" / "best_strategies.json"
        with open(best_file, 'w') as f:
            json.dump(positive_strategies[:50], f, indent=2, default=str)

        print(f"\nTop 50 策略已保存: {best_file}")

    else:
        print("\n⚠️  没有找到正收益策略")
        print("建议：")
        print("  1. 测试更长的 holding period")
        print("  2. 添加更复杂的风控机制")
        print("  3. 考虑市场环境过滤")
        print("  4. 测试其他 horizon (5d, 10d, etc.)")

if __name__ == '__main__':
    main()
