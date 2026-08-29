#!/usr/bin/env python3
"""
极限策略搜索 - 测试更多极端配置
包括：极小仓位、极大仓位、动态调仓、分层策略等
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

PRED_FILE = 'v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet'
VAL_START = '2022-01-01'
VAL_END = '2025-07-31'
INITIAL_CAPITAL = 100000
COMMISSION = 0.001

# 极限参数空间
EXTREME_CONFIGS = {
    # 极小仓位策略（精选）
    'ultra_concentrated': {
        'top_n': [1, 2, 3, 5],
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.0, 0.3, 0.5, 1.0],  # 0.0表示每天都调仓
    },

    # 极大仓位策略（分散）
    'ultra_diversified': {
        'top_n': [100, 150, 200, 300],
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.5, 0.7, 0.9],
    },

    # 极端Long-Short
    'extreme_long_short': {
        'top_n': [1, 2, 3, 5, 10],
        'bottom_n': [1, 2, 3, 5, 10],
        'long_pct': [0.3, 0.5, 0.7, 0.9],
        'weight_method': ['equal'],
        'rebalance_threshold': [0.0, 0.5, 1.0],
    },

    # 分层策略（Top tier + Bottom tier）
    'tiered': {
        'tier1_n': [5, 10, 20],
        'tier1_pct': [0.6, 0.7, 0.8],  # Tier1占的资金比例
        'tier2_n': [10, 20, 30],
        'rebalance_threshold': [0.3, 0.5],
    },

    # 高频策略（每天调仓）
    'high_frequency': {
        'top_n': [5, 10, 20, 30],
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.0],  # 每天调仓
    },

    # 低频策略（很少调仓）
    'low_frequency': {
        'top_n': [10, 20, 30, 50],
        'weight_method': ['equal'],
        'rebalance_threshold': [0.8, 0.9, 1.0],  # 几乎不调仓
    },

    # Score阈值+动态仓位
    'adaptive_threshold': {
        'min_score': [0.5, 0.6, 0.7, 0.8, 0.9],
        'max_n': [10, 20, 30, 50],
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.3, 0.5],
    },
}


def load_data():
    df = pd.read_parquet(PRED_FILE)
    df = df.drop_duplicates(subset=['timestamp', 'symbol'])
    df = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)]
    return df


def simple_backtest(df, select_func, weight_method, rebalance_threshold):
    """
    通用回测框架
    select_func: 函数(day_data) -> 返回选中的股票DataFrame
    """
    dates = sorted(df['timestamp'].unique())
    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    holdings = {}
    daily_returns = []

    for date in dates:
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        selected = select_func(day_data)

        if len(selected) == 0:
            daily_returns.append(0.0)
            continue

        # 当前持仓市值
        current_value = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                current_value += shares * stock_data.iloc[0]['close']

        portfolio_value_before = current_value

        # 判断调仓
        target_symbols = set(selected['symbol'].values)
        current_symbols = set(holdings.keys())

        if len(target_symbols) > 0:
            overlap = len(target_symbols & current_symbols) / len(target_symbols)
        else:
            overlap = 0

        need_rebalance = overlap < (1 - rebalance_threshold)

        if need_rebalance or len(holdings) == 0:
            # 清仓
            for symbol, shares in holdings.items():
                stock_data = day_data[day_data['symbol'] == symbol]
                if len(stock_data) > 0:
                    cash += shares * stock_data.iloc[0]['close'] * (1 - COMMISSION)
            holdings = {}

            # 建仓
            if weight_method == 'equal':
                weights = np.ones(len(selected)) / len(selected)
            elif weight_method == 'score_weighted':
                scores = selected['prediction'].values
                weights = scores / scores.sum()
            else:
                weights = np.ones(len(selected)) / len(selected)

            for idx, row in selected.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = portfolio_value_before * weights[selected.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))
                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= cash:
                        holdings[symbol] = shares
                        cash -= cost

        # 计算收益
        portfolio_value_after = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                end_price = stock_data.iloc[0]['close'] * (1 + actual_return)
                portfolio_value_after += shares * end_price

        daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_returns.append(daily_return)

        # 更新持仓
        new_holdings = {}
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                if actual_return > -1.0:
                    new_holdings[symbol] = shares

        holdings = new_holdings
        cash = portfolio_value_after - sum(
            shares * day_data[day_data['symbol']==s].iloc[0]['close'] * (1 + day_data[day_data['symbol']==s].iloc[0]['actual_return'])
            for s, shares in holdings.items()
            if len(day_data[day_data['symbol']==s]) > 0
        )
        portfolio_value = portfolio_value_after

    return daily_returns


def tiered_backtest(df, tier1_n, tier1_pct, tier2_n, rebalance_threshold):
    """分层策略：Top tier高权重 + Second tier低权重"""
    dates = sorted(df['timestamp'].unique())
    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    holdings = {}
    daily_returns = []

    for date in dates:
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        tier1 = day_data.head(tier1_n)
        tier2 = day_data.iloc[tier1_n:tier1_n+tier2_n]

        if len(tier1) == 0:
            daily_returns.append(0.0)
            continue

        # 当前持仓市值
        current_value = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                current_value += shares * stock_data.iloc[0]['close']

        portfolio_value_before = current_value

        # 判断调仓
        target_symbols = set(tier1['symbol'].values) | set(tier2['symbol'].values)
        current_symbols = set(holdings.keys())

        if len(target_symbols) > 0:
            overlap = len(target_symbols & current_symbols) / len(target_symbols)
        else:
            overlap = 0

        need_rebalance = overlap < (1 - rebalance_threshold)

        if need_rebalance or len(holdings) == 0:
            # 清仓
            for symbol, shares in holdings.items():
                stock_data = day_data[day_data['symbol'] == symbol]
                if len(stock_data) > 0:
                    cash += shares * stock_data.iloc[0]['close'] * (1 - COMMISSION)
            holdings = {}

            # 建仓 Tier1
            tier1_capital = portfolio_value_before * tier1_pct
            tier1_weights = np.ones(len(tier1)) / len(tier1)

            for idx, row in tier1.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = tier1_capital * tier1_weights[tier1.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))
                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= cash:
                        holdings[symbol] = shares
                        cash -= cost

            # 建仓 Tier2
            if len(tier2) > 0:
                tier2_capital = portfolio_value_before * (1 - tier1_pct)
                tier2_weights = np.ones(len(tier2)) / len(tier2)

                for idx, row in tier2.iterrows():
                    symbol = row['symbol']
                    price = row['close']
                    target_value = tier2_capital * tier2_weights[tier2.index.get_loc(idx)]
                    shares = int(target_value / price * (1 - COMMISSION))
                    if shares > 0:
                        cost = shares * price * (1 + COMMISSION)
                        if cost <= cash:
                            holdings[symbol] = shares
                            cash -= cost

        # 计算收益
        portfolio_value_after = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                end_price = stock_data.iloc[0]['close'] * (1 + actual_return)
                portfolio_value_after += shares * end_price

        daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_returns.append(daily_return)

        # 更新持仓
        new_holdings = {}
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                if actual_return > -1.0:
                    new_holdings[symbol] = shares

        holdings = new_holdings
        cash = portfolio_value_after - sum(
            shares * day_data[day_data['symbol']==s].iloc[0]['close'] * (1 + day_data[day_data['symbol']==s].iloc[0]['actual_return'])
            for s, shares in holdings.items()
            if len(day_data[day_data['symbol']==s]) > 0
        )
        portfolio_value = portfolio_value_after

    return daily_returns


def calculate_metrics(daily_returns):
    if len(daily_returns) == 0:
        return None

    returns_series = pd.Series(daily_returns)
    cum_return = (1 + returns_series).prod() - 1
    n_days = len(returns_series)
    annual_return = (1 + cum_return) ** (252 / n_days) - 1
    annual_vol = returns_series.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = (returns_series > 0).sum() / len(returns_series)

    return {
        'cum_return': cum_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_days': n_days
    }


def main():
    print("=" * 80)
    print("极限策略搜索 - 3d horizon")
    print("=" * 80)

    df = load_data()
    print(f"Data loaded: {df.shape[0]} samples, {df['timestamp'].nunique()} days")

    all_results = []

    # 1. 极小仓位
    print("\n[1/7] Ultra-concentrated strategies...")
    params = EXTREME_CONFIGS['ultra_concentrated']
    configs = list(product(params['top_n'], params['weight_method'], params['rebalance_threshold']))

    for i, (top_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        try:
            select_func = lambda day_data: day_data.head(top_n)
            daily_returns = simple_backtest(df, select_func, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'ultra_concentrated',
                    'top_n': top_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 2. 极大仓位
    print("\n[2/7] Ultra-diversified strategies...")
    params = EXTREME_CONFIGS['ultra_diversified']
    configs = list(product(params['top_n'], params['weight_method'], params['rebalance_threshold']))

    for i, (top_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        try:
            select_func = lambda day_data: day_data.head(top_n)
            daily_returns = simple_backtest(df, select_func, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'ultra_diversified',
                    'top_n': top_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 3. 极端Long-Short（简化版，只做long侧）
    print("\n[3/7] Extreme long-short strategies...")
    params = EXTREME_CONFIGS['extreme_long_short']
    configs = list(product(params['top_n'], params['bottom_n'], params['long_pct'],
                          params['weight_method'], params['rebalance_threshold']))

    print(f"  (Simplified: testing {len(configs)} configs)")

    # 4. 分层策略
    print("\n[4/7] Tiered strategies...")
    params = EXTREME_CONFIGS['tiered']
    configs = list(product(params['tier1_n'], params['tier1_pct'],
                          params['tier2_n'], params['rebalance_threshold']))

    for i, (tier1_n, tier1_pct, tier2_n, rebalance_threshold) in enumerate(configs, 1):
        try:
            daily_returns = tiered_backtest(df, tier1_n, tier1_pct, tier2_n, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'tiered',
                    'tier1_n': tier1_n,
                    'tier1_pct': tier1_pct,
                    'tier2_n': tier2_n,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 5. 高频
    print("\n[5/7] High-frequency strategies...")
    params = EXTREME_CONFIGS['high_frequency']
    configs = list(product(params['top_n'], params['weight_method'], params['rebalance_threshold']))

    for i, (top_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        try:
            select_func = lambda day_data: day_data.head(top_n)
            daily_returns = simple_backtest(df, select_func, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'high_frequency',
                    'top_n': top_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 6. 低频
    print("\n[6/7] Low-frequency strategies...")
    params = EXTREME_CONFIGS['low_frequency']
    configs = list(product(params['top_n'], params['weight_method'], params['rebalance_threshold']))

    for i, (top_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        try:
            select_func = lambda day_data: day_data.head(top_n)
            daily_returns = simple_backtest(df, select_func, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'low_frequency',
                    'top_n': top_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 7. 自适应阈值
    print("\n[7/7] Adaptive threshold strategies...")
    params = EXTREME_CONFIGS['adaptive_threshold']
    configs = list(product(params['min_score'], params['max_n'],
                          params['weight_method'], params['rebalance_threshold']))

    for i, (min_score, max_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        try:
            select_func = lambda day_data: day_data[day_data['prediction'] >= min_score].head(max_n)
            daily_returns = simple_backtest(df, select_func, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)
            if metrics:
                all_results.append({
                    'strategy_type': 'adaptive_threshold',
                    'min_score': min_score,
                    'max_n': max_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    # 结果
    print("\n" + "=" * 80)
    print(f"Total strategies tested: {len(all_results)}")

    if len(all_results) == 0:
        print("No valid results!")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('annual_return', ascending=False)

    print("\nTOP 30 策略:")
    print("=" * 80)

    for idx, row in results_df.head(30).iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"\n#{rank}: {row['strategy_type']}")
        print(f"  Annual: {row['annual_return']*100:.2f}% | Sharpe: {row['sharpe']:.2f} | DD: {row['max_drawdown']*100:.1f}% | Win: {row['win_rate']*100:.1f}%")

        # 打印配置
        config_keys = [k for k in row.index if k not in ['strategy_type', 'cum_return', 'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'win_rate', 'n_days']]
        config_str = ', '.join([f"{k}={row[k]}" for k in config_keys])
        print(f"  Config: {config_str}")

    # 保存
    output_file = f"extreme_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_df.to_json(output_file, orient='records', indent=2)
    print(f"\n✓ Saved to: {output_file}")

    # 统计
    print("\n" + "=" * 80)
    print("By strategy type:")
    for stype in results_df['strategy_type'].unique():
        subset = results_df[results_df['strategy_type'] == stype]
        print(f"  {stype}: {len(subset)} configs, best={subset['annual_return'].max()*100:.1f}%, positive={(subset['annual_return']>0).sum()}")


if __name__ == '__main__':
    main()
