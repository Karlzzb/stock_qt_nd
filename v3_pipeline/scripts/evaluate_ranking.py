#!/usr/bin/env python3
"""Evaluate V3 ranking model predictions using ranking metrics.

Computes IC, ICIR, top5_excess, and other ranking-based metrics,
and compares against V2 baseline results.

Usage:
    python v3_pipeline/scripts/evaluate_ranking.py
    python v3_pipeline/scripts/evaluate_ranking.py --models-dir v3_pipeline/models/v3_0_0_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rank_metrics import daily_rank_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(pred_path: Path, horizon: str) -> dict:
    """Evaluate a single horizon's predictions.

    Args:
        pred_path: Path to prediction parquet file
        horizon: Horizon string (e.g., "10d", "30d")

    Returns:
        dict: Ranking metrics for this horizon
    """
    logger.info(f"Evaluating {horizon} predictions from {pred_path}")

    # Load predictions
    pred_df = pd.read_parquet(pred_path)
    logger.info(f"Loaded {len(pred_df)} predictions")

    # Validate required columns
    required_cols = ["timestamp", "symbol", "prediction", "actual_return"]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Prepare dataframe for rank_metrics
    eval_df = pred_df.copy()
    eval_df["y_pred_proba"] = eval_df["prediction"]
    eval_df["label"] = (eval_df["actual_return"] > 0.01).astype(int)  # Binary label for hit rate

    # Extract horizon number for return column naming
    horizon_num = horizon.replace("d", "")
    ret_col = "actual_return"  # Use actual_return directly

    # Compute ranking metrics
    metrics = daily_rank_metrics(
        eval_df,
        ret_col=ret_col,
        label_col="label",
        proba_col="y_pred_proba",
        top_ns=(1, 3, 5, 10),
        n_quantiles=5,
    )

    logger.info(f"{horizon} metrics computed:")
    logger.info(f"  Rank IC: {metrics['daily_rank_ic']:.4f}")
    logger.info(f"  Rank ICIR: {metrics['daily_rank_icir']:.3f}")
    logger.info(f"  Top5 excess: {metrics['daily_top5_excess_ret']:+.4f}")
    logger.info(f"  IC pos ratio: {metrics['daily_ic_pos_ratio']:.3f}")
    logger.info(f"  Top5 turnover: {metrics['top5_turnover']:.3f}")

    return metrics


def load_v2_baseline(v2_path: Path) -> dict:
    """Load V2 baseline metrics for comparison."""
    if not v2_path.exists():
        logger.warning(f"V2 baseline not found at {v2_path}")
        return {}

    with open(v2_path) as f:
        v2_metrics = json.load(f)

    logger.info(f"Loaded V2 baseline metrics for {len(v2_metrics)} horizons")
    return v2_metrics


def compare_metrics(v3_results: dict, v2_baseline: dict) -> pd.DataFrame:
    """Create comparison table between V3 and V2 results.

    Args:
        v3_results: Dict mapping horizon to V3 metrics
        v2_baseline: Dict mapping future_return_Xd to V2 metrics

    Returns:
        DataFrame with comparison rows
    """
    comparison_rows = []

    for horizon, v3_metrics in v3_results.items():
        # Find matching V2 horizon
        horizon_num = horizon.replace("d", "")
        v2_key = f"future_return_{horizon_num}d"
        v2_metrics = v2_baseline.get(v2_key, {})

        row = {
            "horizon": horizon,
            "v3_rank_ic": v3_metrics.get("daily_rank_ic", float("nan")),
            "v2_rank_ic": v2_metrics.get("daily_rank_ic", float("nan")),
            "v3_rank_icir": v3_metrics.get("daily_rank_icir", float("nan")),
            "v2_rank_icir": v2_metrics.get("daily_rank_icir", float("nan")),
            "v3_top5_excess": v3_metrics.get("daily_top5_excess_ret", float("nan")),
            "v2_top5_excess": v2_metrics.get("daily_top5_excess_ret", float("nan")),
            "v3_ic_pos_ratio": v3_metrics.get("daily_ic_pos_ratio", float("nan")),
            "v2_ic_pos_ratio": v2_metrics.get("daily_ic_pos_ratio", float("nan")),
        }

        # Compute improvements
        if v2_metrics:
            row["icir_improvement"] = row["v3_rank_icir"] - row["v2_rank_icir"]
            row["top5_improvement"] = row["v3_top5_excess"] - row["v2_top5_excess"]
        else:
            row["icir_improvement"] = float("nan")
            row["top5_improvement"] = float("nan")

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def print_comparison_report(df: pd.DataFrame, success_threshold: dict) -> None:
    """Print formatted comparison report."""
    logger.info("\n" + "=" * 80)
    logger.info("V3 RANKING PIPELINE EVALUATION REPORT")
    logger.info("=" * 80)

    logger.info("\n📊 Ranking Metrics Comparison (V3 vs V2)")
    logger.info("-" * 80)

    # Main metrics table
    main_cols = ["horizon", "v3_rank_ic", "v2_rank_ic", "v3_rank_icir", "v2_rank_icir"]
    logger.info("\nRank IC & ICIR:")
    logger.info(df[main_cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    excess_cols = ["horizon", "v3_top5_excess", "v2_top5_excess", "v3_ic_pos_ratio", "v2_ic_pos_ratio"]
    logger.info("\nTop5 Excess Return & IC Pos Ratio:")
    logger.info(df[excess_cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Improvements
    imp_cols = ["horizon", "icir_improvement", "top5_improvement"]
    logger.info("\n📈 Improvements (V3 - V2):")
    logger.info(df[imp_cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Success criteria check
    logger.info("\n🎯 Success Criteria Check:")
    logger.info(f"Target ICIR: >{success_threshold['icir']:.1f}")
    logger.info(f"Target Top5 Excess: >{success_threshold['top5_excess']:+.1f}%")
    logger.info("-" * 80)

    for _, row in df.iterrows():
        meets_icir = row["v3_rank_icir"] > success_threshold["icir"]
        meets_top5 = row["v3_top5_excess"] > success_threshold["top5_excess"]
        tradable = meets_icir and meets_top5

        status = "✅ PASS" if tradable else "❌ FAIL"
        logger.info(
            f"{row['horizon']:>5s}: ICIR={row['v3_rank_icir']:+.3f} "
            f"({'✓' if meets_icir else '✗'}) | "
            f"Top5={row['v3_top5_excess']:+.4f} "
            f"({'✓' if meets_top5 else '✗'}) | {status}"
        )

    logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V3 ranking models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/models/v3_0_0_baseline",
        help="Directory containing model predictions",
    )
    parser.add_argument(
        "--v2-baseline",
        type=Path,
        default=REPO_ROOT / "experiments/rank_metrics_2021-12-31_2022-01-01_2025-07-31.json",
        help="Path to V2 baseline metrics JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/results/v3_0_0_baseline_metrics.json",
        help="Output path for V3 metrics JSON",
    )
    args = parser.parse_args()

    # Find all prediction files
    pred_files = sorted(args.models_dir.glob("pred_*.parquet"))
    if not pred_files:
        logger.error(f"No prediction files found in {args.models_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pred_files)} prediction files")

    # Evaluate each horizon
    v3_results = {}
    for pred_path in pred_files:
        # Extract horizon from filename (e.g., "pred_10d.parquet" -> "10d")
        horizon = pred_path.stem.replace("pred_", "")

        try:
            metrics = evaluate_predictions(pred_path, horizon)
            v3_results[horizon] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {horizon}: {e}", exc_info=True)
            continue

    if not v3_results:
        logger.error("No horizons evaluated successfully")
        sys.exit(1)

    # Save V3 results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(v3_results, f, indent=2)
    logger.info(f"V3 metrics saved to {args.output}")

    # Load V2 baseline
    v2_baseline = load_v2_baseline(args.v2_baseline)

    # Create comparison table
    comparison_df = compare_metrics(v3_results, v2_baseline)

    # Save comparison CSV
    comparison_path = args.output.parent / "comparison_v3_vs_v2.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Comparison table saved to {comparison_path}")

    # Load success threshold from config
    config_path = REPO_ROOT / "v3_pipeline/configs/v3_0_0_baseline.yaml"
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    success_threshold = config["evaluation"]["success_threshold"]

    # Print report
    print_comparison_report(comparison_df, success_threshold)


if __name__ == "__main__":
    main()
