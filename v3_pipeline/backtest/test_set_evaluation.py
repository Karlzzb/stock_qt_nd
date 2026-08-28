#!/usr/bin/env python3 -u
"""
Stage 4: Test Set Final Validation and V2 Comparison

Evaluates the final strategy configuration on the sealed test set (2025-08-01+)
and compares V3 performance against V2 baseline.
"""

import sys
import json
import yaml
import hashlib
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

from v3_pipeline.backtest.ranking_strategy import (
    TransactionCostModel,
    TopNEqualWeightStrategy,
    generate_predictions
)


def load_test_model_and_data(model_dir, cache_path, test_start, test_end, label_horizon):
    """Load model and test data efficiently with date filtering."""
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

    print(f"Model uses {len(selected_features)} selected features")
    print(f"Imputer expects {imputer.n_features_in_} features")

    # Load test data with filtering
    print(f"Loading test data from {cache_path} (filtering {test_start} to {test_end})...")
    df = pd.read_parquet(cache_path, filters=[('timestamp', '>=', test_start)])

    # Convert and rename columns
    df['date'] = pd.to_datetime(df['timestamp'])
    if 'symbol' in df.columns:
        df['stock_code'] = df['symbol']

    # Filter test period
    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    print(f"Test set: {len(test_df)} rows, {test_df['date'].min()} to {test_df['date'].max()}")

    return model, imputer, selected_features, test_df



def compute_config_hash(config_dict):
    """
    Compute configuration hash for experiment tracking.

    Args:
        config_dict: Configuration parameters

    Returns:
        8-character hex hash
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def run_test_set_evaluation(predictions_df, best_config, label_horizon):
    """
    Run final strategy evaluation on test set.

    Args:
        predictions_df: Predictions with columns [date, stock_code, score, label]
        best_config: Best configuration from validation {n_positions, rebalance_threshold}
        label_horizon: Label horizon in days

    Returns:
        Test set results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Running Test Set Evaluation")
    print(f"Configuration: {best_config}")
    print('='*70)

    # Initialize strategy with best config
    strategy = TopNEqualWeightStrategy(
        cost_model=TransactionCostModel(
            commission_rate=0.0003,
            slippage_rate=0.001,
            impact_rate=0.0
        )
    )

    # Run backtest with 3-day holding period
    result = strategy.backtest(
        predictions_df=predictions_df,
        n_positions=best_config['n_positions'],
        rebalance_threshold=best_config['rebalance_threshold'],
        initial_capital=1000000,
        holding_period=label_horizon
    )

    # Add configuration info
    result['n_positions'] = best_config['n_positions']
    result['rebalance_threshold'] = best_config['rebalance_threshold']
    result['label_horizon'] = label_horizon

    print(f"\n{'='*70}")
    print(f"Test Set Results")
    print('='*70)
    print(f"Total Return: {result['total_return']*100:.2f}%")
    print(f"Annual Return: {result['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"Calmar Ratio: {result['calmar_ratio']:.3f}")
    print(f"Win Rate: {result['win_rate']*100:.2f}%")
    print(f"Trading Days: {result['trading_days']}")

    return result


def compute_validation_test_gap(val_result, test_result):
    """
    Compute validation-test performance gap.

    Args:
        val_result: Validation set results
        test_result: Test set results

    Returns:
        Dictionary with gap metrics and warning flags
    """
    gaps = {
        'annual_return_gap': abs(test_result['annual_return'] - val_result['annual_return']),
        'sharpe_gap': abs(test_result['sharpe_ratio'] - val_result['sharpe_ratio']),
        'max_dd_gap': abs(test_result['max_drawdown'] - val_result['max_drawdown']),
        'win_rate_gap': abs(test_result['win_rate'] - val_result['win_rate'])
    }

    # Flag if annual return gap > 20%
    gaps['flag_return_gap'] = gaps['annual_return_gap'] > 0.20

    # Flag if Sharpe gap > 0.5
    gaps['flag_sharpe_gap'] = gaps['sharpe_gap'] > 0.5

    # Overall warning
    gaps['warning'] = gaps['flag_return_gap'] or gaps['flag_sharpe_gap']

    return gaps


def generate_v2_comparison(v3_val_result, v3_test_result=None):
    """
    Generate comprehensive V2 vs V3 comparison.

    Args:
        v3_val_result: V3 validation results
        v3_test_result: V3 test results (optional if not available yet)

    Returns:
        Comparison dictionary
    """
    # V2 baseline from docs/evaluation-protocol-v2.md
    v2_metrics = {
        'production': {
            'annual_return': 0.12,  # +12% actual production
            'period': '2026-01-26 to 2026-05-07',
            'description': 'V2 real money (v8 param2)'
        },
        'backtest': {
            'annual_return': 0.41,  # +41% backtest promise
            'description': 'V2 backtest (same v8 param2)'
        },
        'gap': {
            'value': 0.41 - 0.12,  # 29% overfitting gap
            'description': 'Selection effect from 6400 param combinations'
        }
    }

    comparison = {
        'v2': v2_metrics,
        'v3_validation': {
            'annual_return': v3_val_result['annual_return'],
            'sharpe_ratio': v3_val_result['sharpe_ratio'],
            'max_drawdown': v3_val_result['max_drawdown'],
            'win_rate': v3_val_result['win_rate'],
            'period': f"{v3_val_result.get('start_date', 'N/A')} to {v3_val_result.get('end_date', 'N/A')}"
        }
    }

    if v3_test_result:
        comparison['v3_test'] = {
            'annual_return': v3_test_result['annual_return'],
            'sharpe_ratio': v3_test_result['sharpe_ratio'],
            'max_drawdown': v3_test_result['max_drawdown'],
            'win_rate': v3_test_result['win_rate'],
            'period': f"{v3_test_result.get('start_date', 'N/A')} to {v3_test_result.get('end_date', 'N/A')}"
        }

        # Compute validation-test gap
        comparison['validation_test_gap'] = compute_validation_test_gap(
            v3_val_result, v3_test_result
        )

    return comparison


def make_go_nogo_decision(comparison, gap_threshold=0.15):
    """
    Make final Go/No-Go decision for production consideration.

    Args:
        comparison: V2 vs V3 comparison dictionary
        gap_threshold: Validation-test gap threshold (default 15%)

    Returns:
        Decision dictionary with recommendation and reasoning
    """
    decision = {
        'timestamp': datetime.now().isoformat(),
        'gap_threshold': gap_threshold
    }

    if 'validation_test_gap' not in comparison:
        decision['status'] = 'PENDING'
        decision['reason'] = 'Test set data not yet available'
        decision['recommendation'] = 'Wait for 2025-08-01+ data to complete evaluation'
        return decision

    gap = comparison['validation_test_gap']
    v3_val = comparison['v3_validation']
    v3_test = comparison['v3_test']

    # Check validation-test gap
    if gap['annual_return_gap'] < gap_threshold:
        decision['status'] = 'GO'
        decision['reason'] = f"Validation-test gap ({gap['annual_return_gap']*100:.1f}%) < threshold ({gap_threshold*100:.0f}%)"

        # Additional checks
        concerns = []
        if v3_test['annual_return'] < 0:
            concerns.append(f"Negative test return ({v3_test['annual_return']*100:.1f}%)")
        if v3_test['max_drawdown'] < -0.30:
            concerns.append(f"High drawdown ({v3_test['max_drawdown']*100:.1f}%)")
        if v3_test['sharpe_ratio'] < 0:
            concerns.append(f"Negative Sharpe ({v3_test['sharpe_ratio']:.2f})")

        if concerns:
            decision['recommendation'] = f"Approve with caution: {'; '.join(concerns)}"
        else:
            decision['recommendation'] = 'Approve for production consideration'
    else:
        decision['status'] = 'NO-GO'
        decision['reason'] = f"Validation-test gap ({gap['annual_return_gap']*100:.1f}%) >= threshold ({gap_threshold*100:.0f}%)"
        decision['recommendation'] = 'Document overfitting sources and iterate'

        # Identify overfitting sources
        decision['overfitting_sources'] = []
        if gap['flag_return_gap']:
            decision['overfitting_sources'].append('Annual return degradation')
        if gap['flag_sharpe_gap']:
            decision['overfitting_sources'].append('Sharpe ratio degradation')
        if abs(v3_test['max_drawdown']) > abs(v3_val['max_drawdown']) * 1.5:
            decision['overfitting_sources'].append('Drawdown magnification')

    return decision


def save_test_results(test_result, comparison, decision, output_dir, label_horizon):
    """
    Save test set results to JSON with config hash.

    Args:
        test_result: Test set evaluation results
        comparison: V2 vs V3 comparison
        decision: Go/No-Go decision
        output_dir: Output directory path
        label_horizon: Label horizon

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Compute config hash
    config = {
        'n_positions': test_result['n_positions'],
        'rebalance_threshold': test_result['rebalance_threshold'],
        'label_horizon': test_result['label_horizon'],
        'commission_rate': 0.0003,
        'slippage_rate': 0.001
    }
    config_hash = compute_config_hash(config)

    # Prepare output
    output = {
        'config_hash': config_hash,
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'test_result': test_result,
        'comparison': comparison,
        'decision': decision
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output = convert_types(output)

    # Save
    output_file = output_dir / f"test_set_evaluation_{label_horizon}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nTest results saved to {output_file}")
    return output_file


def generate_test_report(test_result, comparison, decision, predictions_df, output_dir, label_horizon):
    """
    Generate markdown report for test set evaluation.

    Args:
        test_result: Test set results (or None if not available)
        comparison: V2 vs V3 comparison
        decision: Go/No-Go decision
        predictions_df: Predictions dataframe
        output_dir: Output directory
        label_horizon: Label horizon

    Returns:
        Path to report file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    report_path = output_dir / f"test_set_final_{label_horizon}.md"

    with open(report_path, 'w') as f:
        f.write(f"# Stage 4: Test Set Final Validation - {label_horizon}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        if test_result:
            f.write("## Configuration\n\n")
            f.write(f"- **Label Horizon:** {test_result['label_horizon']} days\n")
            f.write(f"- **Positions:** {test_result['n_positions']}\n")
            f.write(f"- **Rebalance Threshold:** {test_result['rebalance_threshold']}\n")
            f.write(f"- **Commission Rate:** 0.03%\n")
            f.write(f"- **Slippage Rate:** 0.10%\n\n")

        # Test Set Results
        f.write("## Test Set Results\n\n")

        if test_result:
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Return | {test_result['total_return']*100:.2f}% |\n")
            f.write(f"| Annual Return | {test_result['annual_return']*100:.2f}% |\n")
            f.write(f"| Sharpe Ratio | {test_result['sharpe_ratio']:.3f} |\n")
            f.write(f"| Max Drawdown | {test_result['max_drawdown']*100:.2f}% |\n")
            f.write(f"| Calmar Ratio | {test_result['calmar_ratio']:.3f} |\n")
            f.write(f"| Win Rate | {test_result['win_rate']*100:.2f}% |\n")
            f.write(f"| Trading Days | {test_result['trading_days']} |\n")
            f.write(f"| Final Net Value | ¥{test_result['final_net_value']:,.2f} |\n\n")
        else:
            f.write("**Status:** Test set data (2025-08-01+) not yet available.\n\n")
            f.write("The model has been trained on data up to 2021-12-31 and validated on 2022-01-01 to 2025-07-31.\n")
            f.write("Test set evaluation will be performed once data from 2025-08-01 onwards becomes available.\n\n")

        # V2 vs V3 Comparison
        f.write("## V2 vs V3 Performance Comparison\n\n")

        v2 = comparison['v2']
        v3_val = comparison['v3_validation']

        f.write("| Version | Type | Annual Return | Sharpe | Max DD | Period/Notes |\n")
        f.write("|---------|------|---------------|--------|--------|---------------|\n")
        f.write(f"| V2 | Production | {v2['production']['annual_return']*100:.2f}% | N/A | N/A | {v2['production']['period']} (real money) |\n")
        f.write(f"| V2 | Backtest | {v2['backtest']['annual_return']*100:.2f}% | N/A | N/A | Overfitted (6400 params) |\n")
        f.write(f"| V2 | Gap | {v2['gap']['value']*100:.2f}% | - | - | Selection effect |\n")
        f.write(f"| V3 | Validation | {v3_val['annual_return']*100:.2f}% | {v3_val['sharpe_ratio']:.3f} | {v3_val['max_drawdown']*100:.2f}% | {v3_val['period']} |\n")

        if 'v3_test' in comparison:
            v3_test = comparison['v3_test']
            f.write(f"| V3 | Test | {v3_test['annual_return']*100:.2f}% | {v3_test['sharpe_ratio']:.3f} | {v3_test['max_drawdown']*100:.2f}% | {v3_test['period']} |\n")
        else:
            f.write("| V3 | Test | *Pending* | *Pending* | *Pending* | Awaiting 2025-08+ data |\n")

        f.write("\n")

        # Validation-Test Gap
        if 'validation_test_gap' in comparison:
            gap = comparison['validation_test_gap']
            f.write("## Validation-Test Performance Gap\n\n")
            f.write("| Metric | Gap | Warning |\n")
            f.write("|--------|-----|----------|\n")
            f.write(f"| Annual Return | {gap['annual_return_gap']*100:.2f}% | {'⚠️' if gap['flag_return_gap'] else '✓'} |\n")
            f.write(f"| Sharpe Ratio | {gap['sharpe_gap']:.3f} | {'⚠️' if gap['flag_sharpe_gap'] else '✓'} |\n")
            f.write(f"| Max Drawdown | {gap['max_dd_gap']*100:.2f}% | - |\n")
            f.write(f"| Win Rate | {gap['win_rate_gap']*100:.2f}% | - |\n\n")

            if gap['warning']:
                f.write("**⚠️ Warning:** Significant validation-test gap detected. Review for overfitting.\n\n")

        # Go/No-Go Decision
        f.write("## Go/No-Go Decision\n\n")
        f.write(f"**Status:** {decision['status']}\n\n")
        f.write(f"**Reason:** {decision['reason']}\n\n")
        f.write(f"**Recommendation:** {decision['recommendation']}\n\n")

        if 'overfitting_sources' in decision and decision['overfitting_sources']:
            f.write("**Identified Overfitting Sources:**\n\n")
            for source in decision['overfitting_sources']:
                f.write(f"- {source}\n")
            f.write("\n")

        # Data Summary
        f.write("## Data Summary\n\n")
        if not predictions_df.empty:
            f.write(f"- **Evaluation Period:** {predictions_df['date'].min()} to {predictions_df['date'].max()}\n")
            f.write(f"- **Total Predictions:** {len(predictions_df)}\n")
            f.write(f"- **Unique Dates:** {predictions_df['date'].nunique()}\n")
            f.write(f"- **Unique Stocks:** {predictions_df['stock_code'].nunique()}\n")
        else:
            f.write("- **Status:** No test set data available yet\n")
        f.write("\n")

        # Next Steps
        f.write("## Next Steps\n\n")
        if decision['status'] == 'PENDING':
            f.write("1. Wait for market data from 2025-08-01 onwards\n")
            f.write("2. Regenerate features for test period\n")
            f.write("3. Run model predictions on test set\n")
            f.write("4. Execute this evaluation script on test predictions\n")
            f.write("5. Review validation-test gap and make final decision\n")
        elif decision['status'] == 'GO':
            f.write("1. Prepare production deployment plan\n")
            f.write("2. Set up real-time feature generation pipeline\n")
            f.write("3. Implement position monitoring and risk controls\n")
            f.write("4. Start with paper trading for final validation\n")
        else:  # NO-GO
            f.write("1. Analyze identified overfitting sources\n")
            f.write("2. Review feature engineering and selection process\n")
            f.write("3. Consider simpler model architectures\n")
            f.write("4. Evaluate alternative label definitions\n")
            f.write("5. Re-run validation with adjusted approach\n")

    print(f"Report saved to {report_path}")
    return report_path


def update_experiment_ledger(config_hash, test_result, decision, label_horizon):
    """
    Update docs/evaluation-protocol-v2.md experiment ledger.

    Args:
        config_hash: Configuration hash
        test_result: Test results (or None)
        decision: Go/No-Go decision
        label_horizon: Label horizon
    """
    ledger_path = Path("docs/evaluation-protocol-v2.md")

    if not ledger_path.exists():
        print(f"Warning: Experiment ledger not found at {ledger_path}")
        return

    # Read existing ledger
    with open(ledger_path, 'r') as f:
        content = f.read()

    # Prepare ledger entry
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if test_result:
        entry = f"\n### Test Set Evaluation - {label_horizon} ({timestamp})\n\n"
        entry += f"- **Config Hash:** `{config_hash}`\n"
        entry += f"- **Status:** {decision['status']}\n"
        entry += f"- **Annual Return:** {test_result['annual_return']*100:.2f}%\n"
        entry += f"- **Sharpe Ratio:** {test_result['sharpe_ratio']:.3f}\n"
        entry += f"- **Max Drawdown:** {test_result['max_drawdown']*100:.2f}%\n"
        entry += f"- **Decision:** {decision['recommendation']}\n"
    else:
        entry = f"\n### Test Set Evaluation - {label_horizon} ({timestamp})\n\n"
        entry += f"- **Config Hash:** `{config_hash}`\n"
        entry += f"- **Status:** PENDING - Test data not available\n"
        entry += f"- **Test Period:** 2025-08-01+ (awaiting data)\n"
        entry += f"- **Validation Complete:** Best config selected, ready for test evaluation\n"

    # Append to ledger
    with open(ledger_path, 'a') as f:
        f.write(entry)

    print(f"Experiment ledger updated: {ledger_path}")


def main():
    """Main execution for Stage 4."""
    print("=== Stage 4: Test Set Final Validation and V2 Comparison ===\n")

    # Load configuration
    config_path = "v3_pipeline/configs/v3_0_2_feature_screening.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    label_horizon = config['label_horizon']

    # Load validation results to get best configuration
    val_results_path = Path("v3_pipeline/results/strategy_search_3d.json")

    if not val_results_path.exists():
        print(f"Error: Validation results not found at {val_results_path}")
        print("Please run Stage 3 first to generate validation results.")
        sys.exit(1)

    with open(val_results_path, 'r') as f:
        val_results = json.load(f)

    # Best configuration is the first result (highest Sharpe)
    best_val_result = val_results[0]
    best_config = {
        'n_positions': best_val_result['n_positions'],
        'rebalance_threshold': best_val_result['rebalance_threshold']
    }

    print(f"Best validation configuration: {best_config}")
    print(f"Validation Sharpe: {best_val_result['sharpe_ratio']:.3f}")
    print(f"Validation Annual Return: {best_val_result['annual_return']*100:.2f}%\n")

    # Check if test set data is available
    cache_path = "v3_pipeline/feature_cache_v3.parquet"
    test_start = "2025-08-01"

    # Check if cache file exists and has test data
    test_data_available = False
    test_result = None
    predictions_df = pd.DataFrame()

    print(f"Checking for test set data from {test_start}...")

    # Quick check: see if feature cache has data >= test_start
    if Path(cache_path).exists():
        try:
            df_check = pd.read_parquet(cache_path, columns=['timestamp'])
            df_check['date'] = pd.to_datetime(df_check['timestamp'])
            print(f"Cache date range: {df_check['date'].min()} to {df_check['date'].max()}")
            test_count = len(df_check[df_check['date'] >= test_start])

            if test_count > 0:
                print(f"Found {test_count} test set samples in cache")
                test_data_available = True

                # Load model and data
                model_dir = config['output_model_dir']
                model, imputer, feature_names, test_df = load_test_model_and_data(
                    model_dir, cache_path, test_start, "2099-12-31", label_horizon
                )

                # Generate predictions
                predictions_df = generate_predictions(
                    model, imputer, feature_names, test_df, label_horizon
                )

                # Run test set evaluation
                test_result = run_test_set_evaluation(
                    predictions_df, best_config, label_horizon
                )
            else:
                print(f"No test set data found. Cache ends at {df_check['date'].max()}")
        except Exception as e:
            print(f"Error checking cache: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Feature cache not found at {cache_path}")

    # Generate V2 comparison
    comparison = generate_v2_comparison(best_val_result, test_result)

    # Make Go/No-Go decision
    decision = make_go_nogo_decision(comparison, gap_threshold=0.15)

    # Save results
    results_dir = Path("v3_pipeline/results")
    reports_dir = Path("v3_pipeline/reports")

    if test_result:
        save_test_results(test_result, comparison, decision, results_dir, label_horizon)

    # Generate report (even if test data not available)
    generate_test_report(
        test_result, comparison, decision, predictions_df, reports_dir, label_horizon
    )

    # Update experiment ledger
    if test_result:
        config = {
            'n_positions': test_result['n_positions'],
            'rebalance_threshold': test_result['rebalance_threshold'],
            'label_horizon': test_result['label_horizon'],
            'commission_rate': 0.0003,
            'slippage_rate': 0.001
        }
        config_hash = compute_config_hash(config)
    else:
        config = {
            'n_positions': best_config['n_positions'],
            'rebalance_threshold': best_config['rebalance_threshold'],
            'label_horizon': label_horizon,
            'commission_rate': 0.0003,
            'slippage_rate': 0.001
        }
        config_hash = compute_config_hash(config)

    update_experiment_ledger(config_hash, test_result, decision, label_horizon)

    print(f"\n{'='*70}")
    print("Stage 4 Complete")
    print('='*70)
    print(f"Status: {decision['status']}")
    print(f"Recommendation: {decision['recommendation']}")


if __name__ == "__main__":
    main()
