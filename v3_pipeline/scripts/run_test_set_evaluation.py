#!/usr/bin/env python3 -u
"""
Script to run Stage 4: Test Set Final Validation and V2 Comparison
Uses the simplified version that actually works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from the working simplified version, then run full backtest
import json
import yaml
import pandas as pd
import lightgbm as lgb
import pickle
import numpy as np
from datetime import datetime

from v3_pipeline.backtest.ranking_strategy import (
    TransactionCostModel,
    TopNEqualWeightStrategy,
    generate_predictions
)

def load_test_model_and_data_filtered(model_dir, cache_path, test_start, label_horizon):
    """Load model and test data with efficient filtering."""
    from pathlib import Path

    model_dir = Path(model_dir)

    # Load model
    print(f"Loading model from {model_dir}...")
    model = lgb.Booster(model_file=str(model_dir / f"model_{label_horizon}.txt"))

    # Load imputer
    with open(model_dir / f"imputer_{label_horizon}.pkl", 'rb') as f:
        imputer = pickle.load(f)

    # Load feature names
    with open(model_dir / "features.txt", 'r') as f:
        selected_features = [line.strip() for line in f]

    print(f"Model uses {len(selected_features)} features")

    # Load test data with date filtering
    print(f"Loading test data from {test_start}...")
    df = pd.read_parquet(cache_path, filters=[('timestamp', '>=', test_start)])

    df['date'] = pd.to_datetime(df['timestamp'])
    if 'symbol' in df.columns:
        df['stock_code'] = df['symbol']

    print(f"Test set: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

    return model, imputer, selected_features, df

def run_test_backtest(predictions_df, best_config, label_horizon):
    """Run backtest on test set."""
    strategy = TopNEqualWeightStrategy(
        cost_model=TransactionCostModel(
            commission_rate=0.0003,
            slippage_rate=0.001,
            impact_rate=0.0
        )
    )

    result = strategy.backtest(
        predictions_df=predictions_df,
        n_positions=best_config['n_positions'],
        rebalance_threshold=best_config['rebalance_threshold'],
        initial_capital=1000000,
        holding_period=label_horizon
    )

    result['n_positions'] = best_config['n_positions']
    result['rebalance_threshold'] = best_config['rebalance_threshold']

    return result

if __name__ == "__main__":
    print("=== Stage 4: Test Set Evaluation (Full Execution) ===\n")

    # Load config
    config_path = "v3_pipeline/configs/v3_0_2_feature_screening.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    label_horizon = config['label_horizon']
    label_horizon_days = int(label_horizon.replace('d', ''))

    # Load validation results
    val_results_path = Path("v3_pipeline/results/strategy_search_3d.json")
    with open(val_results_path, 'r') as f:
        val_results = json.load(f)

    best_config = {
        'n_positions': val_results[0]['n_positions'],
        'rebalance_threshold': val_results[0]['rebalance_threshold']
    }

    print(f"Best config from validation: {best_config}\n")

    # Load test data and run
    model_dir = config['output_model_dir']
    cache_path = "v3_pipeline/feature_cache_v3.parquet"
    test_start = "2025-08-01"

    model, imputer, feature_names, test_df = load_test_model_and_data_filtered(
        model_dir, cache_path, test_start, label_horizon
    )

    print("\nGenerating predictions on test set...")
    predictions_df = generate_predictions(
        model, imputer, feature_names, test_df, label_horizon
    )

    print("\nRunning backtest on test set...")
    test_result = run_test_backtest(predictions_df, best_config, label_horizon_days)

    print(f"\n{'='*70}")
    print("Test Set Results")
    print('='*70)
    print(f"Annual Return: {test_result['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio: {test_result['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {test_result['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {test_result['win_rate']*100:.2f}%")

    # Save result and regenerate report using simplified script
    exec(open("v3_pipeline/scripts/run_test_set_evaluation_simple.py").read())
