#!/usr/bin/env python3 -u
"""
Stage 3: Ranking Strategy Development
Converts ranking scores into tradeable strategies with complete backtesting framework.
"""

import sys
import json
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import pickle
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TransactionCostModel:
    """Model for transaction costs including commission, slippage, and impact."""

    def __init__(self, commission_rate=0.0003, slippage_rate=0.001, impact_rate=0.0):
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Commission rate (default 0.03%)
            slippage_rate: Slippage rate (default 0.1%)
            impact_rate: Market impact rate (default 0%, can be dynamic)
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.impact_rate = impact_rate

    def calculate_cost(self, turnover_value, position_count=None):
        """
        Calculate total transaction cost for a given turnover.

        Args:
            turnover_value: Total value of trades (buy + sell)
            position_count: Number of positions (for impact cost calculation)

        Returns:
            Total transaction cost as fraction of turnover
        """
        # Base cost: commission + slippage
        base_cost = self.commission_rate + self.slippage_rate

        # Impact cost increases with position count (optional)
        impact_cost = self.impact_rate
        if position_count is not None and position_count > 0:
            # Simple linear impact model
            impact_cost = self.impact_rate * (position_count / 50.0)

        return (base_cost + impact_cost) * turnover_value


class RankingStrategy:
    """Base class for ranking-based trading strategies."""

    def __init__(self, cost_model=None):
        """
        Initialize ranking strategy.

        Args:
            cost_model: TransactionCostModel instance
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.positions = {}  # Current holdings {stock_code: weight}
        self.net_value_history = []
        self.position_history = []
        self.trade_history = []

    def filter_tradeable_stocks(self, df, date):
        """
        Filter out stocks that cannot be traded.

        Args:
            df: DataFrame with stock data for the date
            date: Current date

        Returns:
            Filtered DataFrame with only tradeable stocks
        """
        # Start with all stocks
        tradeable = df.copy()

        # Filter out ST stocks (special treatment)
        if 'is_st' in tradeable.columns:
            tradeable = tradeable[tradeable['is_st'] == False]

        # Filter out suspended stocks
        if 'is_suspended' in tradeable.columns:
            tradeable = tradeable[tradeable['is_suspended'] == False]

        # Filter out limit-up stocks (can't buy)
        if 'is_limit_up' in tradeable.columns:
            tradeable = tradeable[tradeable['is_limit_up'] == False]

        # Filter out stocks with missing prices
        if 'close' in tradeable.columns:
            tradeable = tradeable[tradeable['close'].notna()]

        return tradeable

    def select_positions(self, scores, n_positions):
        """
        Select stocks to hold based on ranking scores.
        Must be implemented by subclasses.

        Args:
            scores: Series of ranking scores indexed by stock_code
            n_positions: Number of positions to hold

        Returns:
            Dict of {stock_code: weight}
        """
        raise NotImplementedError

    def rebalance(self, current_positions, target_positions, threshold=0.3):
        """
        Determine which positions need to be rebalanced.

        Args:
            current_positions: Dict of current holdings {stock_code: weight}
            target_positions: Dict of target holdings {stock_code: weight}
            threshold: Minimum weight difference to trigger rebalance

        Returns:
            Tuple of (positions_to_buy, positions_to_sell, turnover_fraction)
        """
        # Calculate position changes
        all_stocks = set(current_positions.keys()) | set(target_positions.keys())

        to_buy = {}
        to_sell = {}
        turnover = 0.0

        for stock in all_stocks:
            current_weight = current_positions.get(stock, 0.0)
            target_weight = target_positions.get(stock, 0.0)
            diff = target_weight - current_weight

            if abs(diff) > threshold / len(target_positions):
                if diff > 0:
                    to_buy[stock] = diff
                    turnover += diff
                else:
                    to_sell[stock] = -diff
                    turnover += -diff

        return to_buy, to_sell, turnover

    def backtest(self, predictions_df, n_positions, rebalance_threshold=0.5,
                 initial_capital=1000000, holding_period=3):
        """
        Run backtest on validation set with non-overlapping returns.

        Args:
            predictions_df: DataFrame with columns [date, stock_code, score, future_return_Xd]
            n_positions: Number of positions to hold
            rebalance_threshold: Minimum weight change to trigger rebalance
            initial_capital: Starting capital
            holding_period: Number of trading days to hold positions (default 3 for 3d returns)

        Returns:
            Dict with backtest results
        """
        # Initialize
        self.net_value_history = []
        self.position_history = []
        self.trade_history = []
        current_capital = initial_capital
        self.positions = {}

        # Get unique dates sorted
        dates = sorted(predictions_df['date'].unique())

        print(f"\nRunning backtest: {len(dates)} trading days, {n_positions} positions, "
              f"rebalance_threshold={rebalance_threshold}, holding_period={holding_period}d")

        # Process every holding_period days to avoid overlapping returns
        i = 0
        while i < len(dates):
            date = dates[i]

            # Get data for this date
            day_data = predictions_df[predictions_df['date'] == date].copy()

            # Filter tradeable stocks
            day_data = self.filter_tradeable_stocks(day_data, date)

            if len(day_data) == 0:
                # No tradeable stocks, skip this period
                i += holding_period
                continue

            # Select target positions based on scores
            scores = day_data.set_index('stock_code')['score']
            target_positions = self.select_positions(scores, n_positions)

            # Rebalance
            to_buy, to_sell, turnover = self.rebalance(
                self.positions, target_positions, rebalance_threshold
            )

            # Calculate transaction costs
            transaction_cost = self.cost_model.calculate_cost(
                turnover * current_capital, len(target_positions)
            )
            current_capital -= transaction_cost

            # Record trade
            if len(to_buy) > 0 or len(to_sell) > 0:
                self.trade_history.append({
                    'date': date,
                    'bought': list(to_buy.keys()),
                    'sold': list(to_sell.keys()),
                    'turnover': turnover,
                    'cost': transaction_cost
                })

            # Update positions
            self.positions = target_positions.copy()

            # Get the return column name
            return_col = [c for c in day_data.columns if c.startswith('future_return_')][0]

            # Calculate portfolio return for this holding period
            portfolio_return = 0.0
            for stock, weight in self.positions.items():
                if stock in day_data['stock_code'].values:
                    stock_return = day_data[day_data['stock_code'] == stock][return_col].iloc[0]
                    if pd.notna(stock_return):
                        portfolio_return += weight * stock_return

            # Update capital
            current_capital *= (1 + portfolio_return)

            # Record net value
            self.net_value_history.append({
                'date': date,
                'net_value': current_capital,
                'position_count': len(self.positions),
                'portfolio_return': portfolio_return
            })

            if (i + 1) % (100 * holding_period) == 0:
                print(f"  Progress: {i+1}/{len(dates)} days, "
                      f"net_value={current_capital:.2f}, "
                      f"return={(current_capital/initial_capital - 1)*100:.2f}%")

            # Move to next period
            i += holding_period

        # Calculate performance metrics
        results = self._calculate_metrics(initial_capital)
        results['n_positions'] = n_positions
        results['rebalance_threshold'] = rebalance_threshold

        return results

    def _calculate_metrics(self, initial_capital):
        """Calculate performance metrics from backtest history."""
        df = pd.DataFrame(self.net_value_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calculate returns
        df['daily_return'] = df['net_value'].pct_change()

        # Overall metrics
        total_return = (df['net_value'].iloc[-1] / initial_capital) - 1
        trading_days = len(df)
        years = trading_days / 252

        # Annualized return
        annual_return = (1 + total_return) ** (1 / years) - 1

        # Sharpe ratio (assume 252 trading days per year, 0% risk-free rate)
        daily_returns = df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

        # Maximum drawdown
        df['cummax'] = df['net_value'].cummax()
        df['drawdown'] = (df['net_value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0

        # Monthly returns
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_returns = df.groupby('year_month')['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        )

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'trading_days': int(trading_days),
            'final_net_value': float(df['net_value'].iloc[-1]),
            'monthly_returns': monthly_returns.to_dict() if len(monthly_returns) > 0 else {},
            'net_value_curve': df[['date', 'net_value', 'position_count']].to_dict('records')
        }


class TopNEqualWeightStrategy(RankingStrategy):
    """Top-N equal weight strategy: select top N stocks and weight equally."""

    def select_positions(self, scores, n_positions):
        """Select top N stocks with equal weights."""
        # Sort by score descending and take top N
        top_stocks = scores.nlargest(n_positions)

        # Equal weight
        weight = 1.0 / len(top_stocks)
        positions = {stock: weight for stock in top_stocks.index}

        return positions


class ScoreWeightedStrategy(RankingStrategy):
    """Score-weighted strategy: weight stocks proportional to their scores."""

    def select_positions(self, scores, n_positions):
        """Select top N stocks and weight by normalized scores."""
        # Sort by score descending and take top N
        top_scores = scores.nlargest(n_positions)

        # Normalize scores to sum to 1
        # Use softmax-like transformation to avoid negative weights
        min_score = top_scores.min()
        if min_score < 0:
            top_scores = top_scores - min_score + 1e-8

        total_score = top_scores.sum()
        positions = {stock: score / total_score for stock, score in top_scores.items()}

        return positions


def load_model_and_data(model_dir, cache_path, val_start, val_end, label_horizon):
    """Load optimized model and validation data."""
    model_dir = Path(model_dir)

    # Load model
    print(f"Loading model from {model_dir}...")
    model = lgb.Booster(model_file=str(model_dir / f"model_{label_horizon}.txt"))

    # Load imputer
    with open(model_dir / f"imputer_{label_horizon}.pkl", 'rb') as f:
        imputer = pickle.load(f)

    # Load feature names (selected features for model)
    with open(model_dir / "features.txt", 'r') as f:
        selected_features = [line.strip() for line in f]

    # Load metadata to get all features that imputer was trained on
    with open(model_dir / "training_summary.json", 'r') as f:
        metadata = json.load(f)

    # The imputer was trained on all remaining features after collinearity removal
    # We need to impute those first, then select the model features
    print(f"Model uses {len(selected_features)} selected features")
    print(f"Imputer expects {imputer.n_features_in_} features")

    # Load validation data
    print(f"Loading validation data from {cache_path}...")
    df = pd.read_parquet(cache_path)

    # Convert and rename columns
    df['date'] = pd.to_datetime(df['timestamp'])
    if 'symbol' in df.columns:
        df['stock_code'] = df['symbol']

    # Filter validation period
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    print(f"Validation set: {len(val_df)} rows, {val_df['date'].min()} to {val_df['date'].max()}")

    return model, imputer, selected_features, val_df


def generate_predictions(model, imputer, feature_names, val_df, label_horizon):
    """Generate ranking scores for validation set."""
    print("\nGenerating predictions...")

    # Prepare features - only the selected features
    X_val = val_df[feature_names].copy()

    # Handle inf values
    for col in feature_names:
        X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)

    # Check if imputer expects different number of features
    if imputer.n_features_in_ != len(feature_names):
        print(f"  WARNING: Imputer expects {imputer.n_features_in_} features but model uses {len(feature_names)}")
        print(f"  Creating new imputer for selected features...")
        # Create and fit a new imputer on the selected features
        from sklearn.impute import SimpleImputer
        new_imputer = SimpleImputer(strategy='median')
        X_val_imputed = new_imputer.fit_transform(X_val)
    else:
        # Use the provided imputer
        X_val_imputed = imputer.transform(X_val)

    # Generate predictions
    predictions = model.predict(X_val_imputed)

    # Create predictions dataframe
    return_col = f'future_return_{label_horizon}'
    predictions_df = pd.DataFrame({
        'date': val_df['date'],
        'stock_code': val_df['stock_code'],
        'score': predictions,
        return_col: val_df[return_col]
    })

    print(f"Generated {len(predictions_df)} predictions")
    print(f"Score range: [{predictions_df['score'].min():.4f}, {predictions_df['score'].max():.4f}]")

    return predictions_df


def grid_search_strategy(predictions_df, strategy_class, position_counts,
                        rebalance_thresholds, max_drawdown_constraint=0.30):
    """
    Grid search over strategy parameters.

    Args:
        predictions_df: DataFrame with predictions
        strategy_class: Strategy class to instantiate
        position_counts: List of position counts to try
        rebalance_thresholds: List of rebalance thresholds to try
        max_drawdown_constraint: Maximum allowed drawdown

    Returns:
        List of results dicts, sorted by Sharpe ratio
    """
    results = []

    print(f"\n=== Grid Search: {strategy_class.__name__} ===")
    print(f"Position counts: {position_counts}")
    print(f"Rebalance thresholds: {rebalance_thresholds}")
    print(f"Max drawdown constraint: {max_drawdown_constraint}")

    total_runs = len(position_counts) * len(rebalance_thresholds)
    run_num = 0

    for n_pos in position_counts:
        for rebal_thresh in rebalance_thresholds:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}] Testing: n_positions={n_pos}, "
                  f"rebalance_threshold={rebal_thresh}")

            # Create strategy instance
            strategy = strategy_class()

            # Run backtest
            result = strategy.backtest(
                predictions_df.copy(),
                n_positions=n_pos,
                rebalance_threshold=rebal_thresh
            )

            # Check constraint
            result['meets_constraint'] = result['max_drawdown'] >= -max_drawdown_constraint

            print(f"  Results: Sharpe={result['sharpe_ratio']:.3f}, "
                  f"Return={result['annual_return']*100:.2f}%, "
                  f"MaxDD={result['max_drawdown']*100:.2f}%, "
                  f"Constraint={'✓' if result['meets_constraint'] else '✗'}")

            results.append(result)

    # Sort by Sharpe ratio (descending), but prioritize those meeting constraint
    results_sorted = sorted(results,
                           key=lambda x: (x['meets_constraint'], x['sharpe_ratio']),
                           reverse=True)

    return results_sorted


def save_results(results, output_dir, label_horizon):
    """Save strategy search results to JSON."""
    output_file = output_dir / f"strategy_search_{label_horizon}.json"

    # Convert to serializable format (remove net_value_curve for cleaner output)
    results_clean = []
    for r in results:
        r_clean = r.copy()
        # Keep only summary metrics, save full curve separately
        r_clean.pop('net_value_curve', None)
        # Convert monthly_returns Period keys to strings for JSON serialization
        if 'monthly_returns' in r_clean:
            r_clean['monthly_returns'] = {str(k): v for k, v in r_clean['monthly_returns'].items()}
        results_clean.append(r_clean)

    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return output_file


def generate_report(results, predictions_df, output_dir, label_horizon):
    """Generate markdown report for validation backtest."""
    best_result = results[0]

    report_path = output_dir / f"backtest_validation_{label_horizon}.md"

    with open(report_path, 'w') as f:
        f.write(f"# Backtest Report: Top-N Equal Weight Strategy\n\n")
        f.write(f"**Label Horizon:** {label_horizon}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Best Strategy Configuration\n\n")
        f.write(f"- **Position Count:** {best_result['n_positions']}\n")
        f.write(f"- **Rebalance Threshold:** {best_result['rebalance_threshold']}\n")
        f.write(f"- **Meets Constraint:** {'Yes' if best_result['meets_constraint'] else 'No'}\n\n")

        f.write("## Performance Metrics\n\n")
        f.write(f"- **Total Return:** {best_result['total_return']*100:.2f}%\n")
        f.write(f"- **Annual Return:** {best_result['annual_return']*100:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {best_result['sharpe_ratio']:.3f}\n")
        f.write(f"- **Calmar Ratio:** {best_result['calmar_ratio']:.3f}\n")
        f.write(f"- **Max Drawdown:** {best_result['max_drawdown']*100:.2f}%\n")
        f.write(f"- **Win Rate:** {best_result['win_rate']*100:.2f}%\n")
        f.write(f"- **Trading Days:** {best_result['trading_days']}\n")
        f.write(f"- **Final Net Value:** {best_result['final_net_value']:.2f}\n\n")

        f.write("## Grid Search Results\n\n")
        f.write("| Rank | Positions | Rebalance | Sharpe | Annual Return | Max DD | Constraint |\n")
        f.write("|------|-----------|-----------|--------|---------------|--------|------------|\n")

        for i, r in enumerate(results[:10], 1):  # Top 10
            f.write(f"| {i} | {r['n_positions']} | {r['rebalance_threshold']} | "
                   f"{r['sharpe_ratio']:.3f} | {r['annual_return']*100:.2f}% | "
                   f"{r['max_drawdown']*100:.2f}% | "
                   f"{'✓' if r['meets_constraint'] else '✗'} |\n")

        f.write("\n## Net Value Curve\n\n")
        f.write("```\n")
        f.write("Date, Net Value, Positions\n")
        for record in best_result['net_value_curve'][::10]:  # Sample every 10th point
            f.write(f"{record['date']}, {record['net_value']:.2f}, {record['position_count']}\n")
        f.write("```\n\n")

        f.write("## Monthly Return Distribution\n\n")
        f.write("| Year-Month | Return |\n")
        f.write("|------------|--------|\n")

        # Get monthly returns from best result
        monthly_returns = best_result.get('monthly_returns', {})
        if monthly_returns:
            for period, ret in sorted(monthly_returns.items()):
                f.write(f"| {period} | {ret*100:.2f}% |\n")
        else:
            f.write("| - | No data |\n")
        f.write("\n")

        f.write("## Data Summary\n\n")
        f.write(f"- **Validation Period:** {predictions_df['date'].min()} to {predictions_df['date'].max()}\n")
        f.write(f"- **Total Predictions:** {len(predictions_df)}\n")
        f.write(f"- **Unique Dates:** {predictions_df['date'].nunique()}\n")
        f.write(f"- **Unique Stocks:** {predictions_df['stock_code'].nunique()}\n")

    print(f"Report saved to {report_path}")
    return report_path


def main():
    """Main execution."""
    # Configuration
    config_path = "v3_pipeline/configs/v3_0_2_feature_screening.yaml"

    print("=== Stage 3: Ranking Strategy Development ===")
    print(f"Loading configuration from {config_path}...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    label_horizon = config['label_horizon']
    model_dir = config['output_model_dir']
    cache_path = "v3_pipeline/feature_cache_v3.parquet"
    val_start = config['data']['val_start']
    val_end = config['data']['val_end']

    # Create output directories
    results_dir = Path("v3_pipeline/results")
    reports_dir = Path("v3_pipeline/reports")
    results_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Load model and generate predictions
    model, imputer, feature_names, val_df = load_model_and_data(
        model_dir, cache_path, val_start, val_end, label_horizon
    )

    predictions_df = generate_predictions(
        model, imputer, feature_names, val_df, label_horizon
    )

    # Grid search parameters
    position_counts = [10, 20, 30, 50]
    rebalance_thresholds = [0.3, 0.5, 0.7]
    max_drawdown_constraint = 0.30

    # Test Top-N Equal Weight strategy as specified in issue
    print(f"\n{'='*70}")
    print(f"Testing Top-N Equal Weight Strategy")
    print('='*70)

    # Grid search
    results = grid_search_strategy(
        predictions_df,
        TopNEqualWeightStrategy,
        position_counts,
        rebalance_thresholds,
        max_drawdown_constraint
    )

    # Save results
    save_results(results, results_dir, label_horizon)

    # Generate report
    generate_report(results, predictions_df, reports_dir, label_horizon)

    print("\n=== Stage 3 Complete ===")
    print("Strategy backtest finished successfully!")


if __name__ == "__main__":
    main()
