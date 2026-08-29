#!/usr/bin/env python3
"""
大规模策略搜索 - 修复版
测试数千种策略配置，找到能利用V3模型预测能力的策略
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============= 配置 =============
PRED_FILE = 'v3_pipeline/models/v3_0_1_label_selection/pred_3d.parquet'
VAL_START = '2022-01-01'
VAL_END = '2025-07-31'
INITIAL_CAPITAL = 100000
COMMISSION = 0.001  # 0.1%

# ============= 策略参数空间 =============
STRATEGY_SPACE = {
    # Long-only策略
    'long_only': {
        'top_n': [5, 10, 15, 20, 30, 50, 100],
        'weight_method': ['equal', 'score_weighted', 'inverse_rank'],
        'rebalance_threshold': [0.2, 0.3, 0.5, 0.7, 1.0],
    },

    # Long-short策略
    'long_short': {
        'top_n': [5, 10, 15, 20, 30, 50],
        'bottom_n': [5, 10, 15, 20, 30, 50],
        'long_pct': [0.5, 0.6, 0.7, 0.8],  # 资金中多头占比
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.3, 0.5, 0.7],
    },

    # Market-neutral策略
    'market_neutral': {
        'top_n': [10, 20, 30, 50],
        'bottom_n': [10, 20, 30, 50],
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.3, 0.5, 0.7],
    },

    # 动态阈值策略
    'dynamic_threshold': {
        'top_pct': [0.05, 0.10, 0.15, 0.20],  # 选前X%的股票
        'score_threshold': [0.6, 0.7, 0.8, 0.9],  # 预测分数阈值
        'weight_method': ['equal', 'score_weighted'],
        'rebalance_threshold': [0.3, 0.5],
    },
}


def load_and_prepare_data():
    """加载并准备数据"""
    print(f"Loading {PRED_FILE}...")
    df = pd.read_parquet(PRED_FILE)

    # 去重
    df = df.drop_duplicates(subset=['timestamp', 'symbol'])

    # 过滤验证集
    df = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)]

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique dates: {df['timestamp'].nunique()}")

    return df


def calculate_position_weights(stocks_df, weight_method):
    """计算仓位权重"""
    if weight_method == 'equal':
        weights = np.ones(len(stocks_df)) / len(stocks_df)

    elif weight_method == 'score_weighted':
        scores = stocks_df['prediction'].values
        weights = scores / scores.sum()

    elif weight_method == 'inverse_rank':
        # 排名越靠前，权重越大
        ranks = stocks_df['prediction'].rank(ascending=False)
        inv_ranks = 1.0 / ranks
        weights = inv_ranks / inv_ranks.sum()

    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    return weights


def backtest_long_only(df, top_n, weight_method, rebalance_threshold):
    """Long-only策略回测"""
    dates = sorted(df['timestamp'].unique())

    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    holdings = {}  # {symbol: shares}

    daily_returns = []

    for date in dates:
        day_data = df[df['timestamp'] == date].copy()

        # 按预测分数排序，选top_n
        day_data = day_data.sort_values('prediction', ascending=False)
        top_stocks = day_data.head(top_n)

        if len(top_stocks) == 0:
            daily_returns.append(0.0)
            continue

        # 计算当前持仓市值
        current_holdings_value = 0
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                price = stock_data.iloc[0]['close']
                current_holdings_value += shares * price

        portfolio_value_before = cash + current_holdings_value

        # 判断是否需要调仓
        target_symbols = set(top_stocks['symbol'].values)
        current_symbols = set(holdings.keys())

        overlap = len(target_symbols & current_symbols) / top_n
        need_rebalance = overlap < (1 - rebalance_threshold)

        if need_rebalance or len(holdings) == 0:
            # 清仓
            for symbol, shares in holdings.items():
                stock_data = day_data[day_data['symbol'] == symbol]
                if len(stock_data) > 0:
                    sell_price = stock_data.iloc[0]['close']
                    cash += shares * sell_price * (1 - COMMISSION)
            holdings = {}

            # 建仓
            weights = calculate_position_weights(top_stocks, weight_method)

            for idx, row in top_stocks.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = portfolio_value_before * weights[top_stocks.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))

                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= cash:
                        holdings[symbol] = shares
                        cash -= cost

        # 计算今日收益
        portfolio_value_after = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                # 使用actual_return计算收益
                actual_return = stock_data.iloc[0]['actual_return']
                end_price = stock_data.iloc[0]['close'] * (1 + actual_return)
                portfolio_value_after += shares * end_price

        daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_returns.append(daily_return)

        # 更新现金和持仓（用actual_return）
        new_cash = cash
        new_holdings = {}
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                if actual_return > -1.0:  # 没有退市
                    new_holdings[symbol] = shares

        cash = new_cash
        holdings = new_holdings
        portfolio_value = portfolio_value_after

    return daily_returns


def backtest_long_short(df, top_n, bottom_n, long_pct, weight_method, rebalance_threshold):
    """Long-short策略回测"""
    dates = sorted(df['timestamp'].unique())

    portfolio_value = INITIAL_CAPITAL
    long_capital = INITIAL_CAPITAL * long_pct
    short_capital = INITIAL_CAPITAL * (1 - long_pct)

    long_cash = long_capital
    short_cash = short_capital
    long_holdings = {}
    short_holdings = {}

    daily_returns = []

    for date in dates:
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        top_stocks = day_data.head(top_n)
        bottom_stocks = day_data.tail(bottom_n)

        if len(top_stocks) == 0 or len(bottom_stocks) == 0:
            daily_returns.append(0.0)
            continue

        # 计算当前持仓市值
        long_value = long_cash
        for symbol, shares in long_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                long_value += shares * stock_data.iloc[0]['close']

        short_value = short_cash
        for symbol, shares in short_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                short_value += shares * stock_data.iloc[0]['close']

        portfolio_value_before = long_value + short_value

        # 判断是否需要调仓
        target_long = set(top_stocks['symbol'].values)
        target_short = set(bottom_stocks['symbol'].values)
        current_long = set(long_holdings.keys())
        current_short = set(short_holdings.keys())

        long_overlap = len(target_long & current_long) / top_n if top_n > 0 else 0
        short_overlap = len(target_short & current_short) / bottom_n if bottom_n > 0 else 0

        need_rebalance = (long_overlap < (1 - rebalance_threshold)) or (short_overlap < (1 - rebalance_threshold))

        if need_rebalance or len(long_holdings) == 0:
            # 平多头
            for symbol, shares in long_holdings.items():
                stock_data = day_data[day_data['symbol'] == symbol]
                if len(stock_data) > 0:
                    long_cash += shares * stock_data.iloc[0]['close'] * (1 - COMMISSION)
            long_holdings = {}

            # 平空头
            for symbol, shares in short_holdings.items():
                stock_data = day_data[day_data['symbol'] == symbol]
                if len(stock_data) > 0:
                    short_cash += shares * stock_data.iloc[0]['close'] * (1 - COMMISSION)
            short_holdings = {}

            # 开多头
            long_weights = calculate_position_weights(top_stocks, weight_method)
            for idx, row in top_stocks.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = long_value * long_weights[top_stocks.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))
                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= long_cash:
                        long_holdings[symbol] = shares
                        long_cash -= cost

            # 开空头
            short_weights = calculate_position_weights(bottom_stocks, weight_method)
            for idx, row in bottom_stocks.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = short_value * short_weights[bottom_stocks.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))
                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= short_cash:
                        short_holdings[symbol] = shares
                        short_cash -= cost

        # 计算今日收益
        long_value_after = long_cash
        short_value_after = short_cash

        for symbol, shares in long_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                end_price = stock_data.iloc[0]['close'] * (1 + actual_return)
                long_value_after += shares * end_price

        for symbol, shares in short_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                # 空头：股价涨，亏钱；股价跌，赚钱
                short_value_after += shares * stock_data.iloc[0]['close'] * (1 - actual_return)

        portfolio_value_after = long_value_after + short_value_after
        daily_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        daily_returns.append(daily_return)

        # 更新持仓
        new_long_holdings = {}
        for symbol, shares in long_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                if actual_return > -1.0:
                    new_long_holdings[symbol] = shares

        new_short_holdings = {}
        for symbol, shares in short_holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                actual_return = stock_data.iloc[0]['actual_return']
                if actual_return > -1.0:
                    new_short_holdings[symbol] = shares

        long_holdings = new_long_holdings
        short_holdings = new_short_holdings
        long_cash = long_value_after - sum(
            shares * day_data[day_data['symbol']==s].iloc[0]['close'] * (1 + day_data[day_data['symbol']==s].iloc[0]['actual_return'])
            for s, shares in long_holdings.items()
            if len(day_data[day_data['symbol']==s]) > 0
        )
        short_cash = short_value_after - sum(
            shares * day_data[day_data['symbol']==s].iloc[0]['close'] * (1 - day_data[day_data['symbol']==s].iloc[0]['actual_return'])
            for s, shares in short_holdings.items()
            if len(day_data[day_data['symbol']==s]) > 0
        )
        portfolio_value = portfolio_value_after

    return daily_returns


def backtest_market_neutral(df, top_n, bottom_n, weight_method, rebalance_threshold):
    """Market-neutral策略（50%多 50%空）"""
    return backtest_long_short(df, top_n, bottom_n, 0.5, weight_method, rebalance_threshold)


def backtest_dynamic_threshold(df, top_pct, score_threshold, weight_method, rebalance_threshold):
    """动态阈值策略"""
    dates = sorted(df['timestamp'].unique())

    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    holdings = {}

    daily_returns = []

    for date in dates:
        day_data = df[df['timestamp'] == date].copy()
        day_data = day_data.sort_values('prediction', ascending=False)

        # 选择：前top_pct% 且 分数>=score_threshold
        n_stocks = int(len(day_data) * top_pct)
        top_stocks = day_data.head(n_stocks)
        top_stocks = top_stocks[top_stocks['prediction'] >= score_threshold]

        if len(top_stocks) == 0:
            daily_returns.append(0.0)
            continue

        # 计算当前持仓市值
        current_value = cash
        for symbol, shares in holdings.items():
            stock_data = day_data[day_data['symbol'] == symbol]
            if len(stock_data) > 0:
                current_value += shares * stock_data.iloc[0]['close']

        portfolio_value_before = current_value

        # 判断是否需要调仓
        target_symbols = set(top_stocks['symbol'].values)
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
            weights = calculate_position_weights(top_stocks, weight_method)
            for idx, row in top_stocks.iterrows():
                symbol = row['symbol']
                price = row['close']
                target_value = portfolio_value_before * weights[top_stocks.index.get_loc(idx)]
                shares = int(target_value / price * (1 - COMMISSION))
                if shares > 0:
                    cost = shares * price * (1 + COMMISSION)
                    if cost <= cash:
                        holdings[symbol] = shares
                        cash -= cost

        # 计算今日收益
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
    """计算策略指标"""
    if len(daily_returns) == 0:
        return None

    returns_series = pd.Series(daily_returns)

    # 累计收益
    cum_return = (1 + returns_series).prod() - 1

    # 年化收益（假设252个交易日）
    n_days = len(returns_series)
    annual_return = (1 + cum_return) ** (252 / n_days) - 1

    # 年化波动率
    annual_vol = returns_series.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
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
    print("大规模策略搜索 V2 - 3d horizon")
    print("=" * 80)

    # 加载数据
    df = load_and_prepare_data()

    all_results = []

    # 1. Long-only策略
    print("\n[1/4] Testing Long-only strategies...")
    params = STRATEGY_SPACE['long_only']
    configs = list(product(
        params['top_n'],
        params['weight_method'],
        params['rebalance_threshold']
    ))

    for i, (top_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        try:
            daily_returns = backtest_long_only(df, top_n, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)

            if metrics:
                all_results.append({
                    'strategy_type': 'long_only',
                    'top_n': top_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error in config {i}: {e}")

    # 2. Long-short策略
    print("\n[2/4] Testing Long-short strategies...")
    params = STRATEGY_SPACE['long_short']
    configs = list(product(
        params['top_n'],
        params['bottom_n'],
        params['long_pct'],
        params['weight_method'],
        params['rebalance_threshold']
    ))

    for i, (top_n, bottom_n, long_pct, weight_method, rebalance_threshold) in enumerate(configs, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        try:
            daily_returns = backtest_long_short(df, top_n, bottom_n, long_pct, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)

            if metrics:
                all_results.append({
                    'strategy_type': 'long_short',
                    'top_n': top_n,
                    'bottom_n': bottom_n,
                    'long_pct': long_pct,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error in config {i}: {e}")

    # 3. Market-neutral策略
    print("\n[3/4] Testing Market-neutral strategies...")
    params = STRATEGY_SPACE['market_neutral']
    configs = list(product(
        params['top_n'],
        params['bottom_n'],
        params['weight_method'],
        params['rebalance_threshold']
    ))

    for i, (top_n, bottom_n, weight_method, rebalance_threshold) in enumerate(configs, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        try:
            daily_returns = backtest_market_neutral(df, top_n, bottom_n, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)

            if metrics:
                all_results.append({
                    'strategy_type': 'market_neutral',
                    'top_n': top_n,
                    'bottom_n': bottom_n,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error in config {i}: {e}")

    # 4. Dynamic threshold策略
    print("\n[4/4] Testing Dynamic threshold strategies...")
    params = STRATEGY_SPACE['dynamic_threshold']
    configs = list(product(
        params['top_pct'],
        params['score_threshold'],
        params['weight_method'],
        params['rebalance_threshold']
    ))

    for i, (top_pct, score_threshold, weight_method, rebalance_threshold) in enumerate(configs, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(configs)}")

        try:
            daily_returns = backtest_dynamic_threshold(df, top_pct, score_threshold, weight_method, rebalance_threshold)
            metrics = calculate_metrics(daily_returns)

            if metrics:
                all_results.append({
                    'strategy_type': 'dynamic_threshold',
                    'top_pct': top_pct,
                    'score_threshold': score_threshold,
                    'weight_method': weight_method,
                    'rebalance_threshold': rebalance_threshold,
                    **metrics
                })
        except Exception as e:
            print(f"  Error in config {i}: {e}")

    # 汇总结果
    print("\n" + "=" * 80)
    print(f"Total strategies tested: {len(all_results)}")

    if len(all_results) == 0:
        print("No valid results!")
        return

    results_df = pd.DataFrame(all_results)

    # 按年化收益排序
    results_df = results_df.sort_values('annual_return', ascending=False)

    # Top 20
    print("\n" + "=" * 80)
    print("TOP 20 策略 (by annual return):")
    print("=" * 80)

    top20 = results_df.head(20)
    for idx, row in top20.iterrows():
        print(f"\nRank {top20.index.get_loc(idx) + 1}:")
        print(f"  Type: {row['strategy_type']}")

        if row['strategy_type'] == 'long_only':
            print(f"  Config: top_n={row['top_n']}, weight={row['weight_method']}, rebal={row['rebalance_threshold']}")
        elif row['strategy_type'] == 'long_short':
            print(f"  Config: top={row['top_n']}, bottom={row['bottom_n']}, long_pct={row['long_pct']}, weight={row['weight_method']}, rebal={row['rebalance_threshold']}")
        elif row['strategy_type'] == 'market_neutral':
            print(f"  Config: top={row['top_n']}, bottom={row['bottom_n']}, weight={row['weight_method']}, rebal={row['rebalance_threshold']}")
        elif row['strategy_type'] == 'dynamic_threshold':
            print(f"  Config: top_pct={row['top_pct']}, score_thresh={row['score_threshold']}, weight={row['weight_method']}, rebal={row['rebalance_threshold']}")

        print(f"  Annual Return: {row['annual_return']*100:.2f}%")
        print(f"  Sharpe: {row['sharpe']:.2f}")
        print(f"  Max DD: {row['max_drawdown']*100:.2f}%")
        print(f"  Win Rate: {row['win_rate']*100:.1f}%")

    # 保存结果
    output_file = f"massive_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_df.to_json(output_file, orient='records', indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    # 统计
    print("\n" + "=" * 80)
    print("Statistics by strategy type:")
    print("=" * 80)

    for stype in results_df['strategy_type'].unique():
        subset = results_df[results_df['strategy_type'] == stype]
        print(f"\n{stype}:")
        print(f"  Count: {len(subset)}")
        print(f"  Best annual return: {subset['annual_return'].max()*100:.2f}%")
        print(f"  Median annual return: {subset['annual_return'].median()*100:.2f}%")
        print(f"  Positive count: {(subset['annual_return'] > 0).sum()}")


if __name__ == '__main__':
    main()
