#!/usr/bin/env python3
"""Generate label comparison report for Stage 1 label selection.

Analyzes all 7 horizon predictions, computes ranking metrics with year-by-year
breakdown, and generates a markdown report with Go/No-Go recommendations.

Usage:
    python v3_pipeline/scripts/compare_labels.py
    python v3_pipeline/scripts/compare_labels.py --models-dir v3_pipeline/models/v3_0_1_label_selection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rank_metrics import daily_rank_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_horizon_by_year(pred_df: pd.DataFrame, horizon: str) -> dict:
    """Evaluate a horizon with year-by-year breakdown.

    Args:
        pred_df: DataFrame with columns [timestamp, symbol, prediction, actual_return]
        horizon: Horizon string (e.g., "3d", "10d")

    Returns:
        dict: Metrics overall and by year
    """
    logger.info(f"Evaluating {horizon}...")

    # Prepare for rank_metrics
    eval_df = pred_df.copy()
    eval_df["y_pred_proba"] = eval_df["prediction"]
    eval_df["label"] = (eval_df["actual_return"] > 0.01).astype(int)

    # Overall metrics
    overall_metrics = daily_rank_metrics(
        eval_df,
        ret_col="actual_return",
        label_col="label",
        proba_col="y_pred_proba",
        top_ns=(1, 3, 5, 10),
        n_quantiles=5,
    )

    # Year-by-year breakdown
    eval_df["year"] = pd.to_datetime(eval_df["timestamp"]).dt.year
    yearly_metrics = {}

    for year in sorted(eval_df["year"].unique()):
        year_df = eval_df[eval_df["year"] == year].copy()
        if len(year_df) < 100:  # Skip years with too few samples
            continue

        year_metrics = daily_rank_metrics(
            year_df,
            ret_col="actual_return",
            label_col="label",
            proba_col="y_pred_proba",
            top_ns=(5,),
            n_quantiles=5,
        )

        yearly_metrics[int(year)] = {
            "rank_ic": year_metrics["daily_rank_ic"],
            "rank_icir": year_metrics["daily_rank_icir"],
            "top5_excess": year_metrics["daily_top5_excess_ret"],
            "ic_pos_ratio": year_metrics["daily_ic_pos_ratio"],
            "n_days": len(year_df["timestamp"].unique()),
        }

    # Compute ICIR stability (std of yearly ICIRs)
    yearly_icirs = [m["rank_icir"] for m in yearly_metrics.values()]
    icir_std = np.std(yearly_icirs) if len(yearly_icirs) > 1 else np.nan

    result = {
        "horizon": horizon,
        "overall": {
            "rank_ic": overall_metrics["daily_rank_ic"],
            "rank_icir": overall_metrics["daily_rank_icir"],
            "top5_excess": overall_metrics["daily_top5_excess_ret"],
            "ic_pos_ratio": overall_metrics["daily_ic_pos_ratio"],
            "top5_turnover": overall_metrics["top5_turnover"],
            "icir_std": icir_std,
        },
        "by_year": yearly_metrics,
    }

    logger.info(f"  {horizon}: IC={result['overall']['rank_ic']:.4f}, "
                f"ICIR={result['overall']['rank_icir']:.3f}, "
                f"Top5={result['overall']['top5_excess']:+.4f}")

    return result


def check_success_criteria(metrics: dict, thresholds: dict) -> dict:
    """Check if metrics meet success criteria.

    Args:
        metrics: Overall metrics dict
        thresholds: Success threshold dict with icir, ic_pos_ratio, top5_excess

    Returns:
        dict: Pass/fail status for each criterion
    """
    overall = metrics["overall"]

    # Convert top5_excess to annual return (not percentage)
    # daily_top5_excess_ret is already in decimal form (0.15 = 15%)
    # Annualize: multiply by 252 trading days
    top5_annual = overall["top5_excess"] * 252  # Daily excess to annual excess

    return {
        "meets_icir": overall["rank_icir"] > thresholds["icir"],
        "meets_ic_pos": overall["ic_pos_ratio"] > thresholds["ic_pos_ratio"],
        "meets_top5": top5_annual > thresholds["top5_excess"],
        "icir": overall["rank_icir"],
        "ic_pos_ratio": overall["ic_pos_ratio"],
        "top5_annual": top5_annual,
        "all_pass": (
            overall["rank_icir"] > thresholds["icir"]
            and overall["ic_pos_ratio"] > thresholds["ic_pos_ratio"]
            and top5_annual > thresholds["top5_excess"]
        ),
    }


def generate_markdown_report(
    all_results: dict,
    thresholds: dict,
    output_path: Path,
) -> None:
    """Generate comprehensive markdown comparison report.

    Args:
        all_results: Dict mapping horizon to evaluation results
        thresholds: Success threshold dict
        output_path: Path to save markdown report
    """
    lines = []

    # Header
    lines.append("# V3 Label Selection Experiment Report")
    lines.append("")
    lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Experiment:** Stage 1 - Label Selection")
    lines.append(f"**Validation Period:** 2022-01-01 to 2025-07-31")
    lines.append("")
    lines.append("⚠️ **Data Quality Note:** Future returns are capped at ±15% in the feature cache. ")
    lines.append("This causes artificially inflated Top5 excess return metrics (3000%+). ")
    lines.append("The relative comparison between horizons remains valid for ranking purposes. ")
    lines.append("ICIR and Rank IC are the primary evaluation metrics.")
    lines.append("")

    # Success criteria
    lines.append("## Success Criteria")
    lines.append("")
    lines.append(f"- **ICIR:** > {thresholds['icir']}")
    lines.append(f"- **IC Positive Ratio:** > {thresholds['ic_pos_ratio']:.0%}")
    lines.append(f"- **Top5 Excess Return (Annual):** > {thresholds['top5_excess']:.0%}")
    lines.append("")

    # Overall comparison table
    lines.append("## Overall Metrics Comparison")
    lines.append("")
    lines.append("| Horizon | Rank IC | ICIR | ICIR Std | Top5 Excess (Annual) | IC Pos % | Top5 Turnover | Pass |")
    lines.append("|---------|---------|------|----------|----------------------|----------|---------------|------|")

    summary_rows = []
    for horizon in sorted(all_results.keys(), key=lambda h: int(h.replace("d", ""))):
        result = all_results[horizon]
        overall = result["overall"]
        status = check_success_criteria(result, thresholds)

        pass_mark = "✅" if status["all_pass"] else "❌"
        top5_annual = overall["top5_excess"] * 252  # Annualize daily excess

        lines.append(
            f"| {horizon:>7s} "
            f"| {overall['rank_ic']:+.4f} "
            f"| {overall['rank_icir']:+.3f} "
            f"| {overall['icir_std']:+.3f} "
            f"| {top5_annual:+.1%} "
            f"| {overall['ic_pos_ratio']:.1%} "
            f"| {overall['top5_turnover']:.2f} "
            f"| {pass_mark} |"
        )

        summary_rows.append({
            "horizon": horizon,
            "icir": overall["rank_icir"],
            "top5_annual": top5_annual,
            "all_pass": status["all_pass"],
        })

    lines.append("")

    # Year-by-year breakdown
    lines.append("## Year-by-Year Stability")
    lines.append("")
    lines.append("ICIR by year for each horizon:")
    lines.append("")

    # Get all years
    all_years = set()
    for result in all_results.values():
        all_years.update(result["by_year"].keys())
    all_years = sorted(all_years)

    # Build year-by-year table
    lines.append("| Horizon | " + " | ".join([str(y) for y in all_years]) + " |")
    lines.append("|---------|" + "|".join(["--------"] * len(all_years)) + "|")

    for horizon in sorted(all_results.keys(), key=lambda h: int(h.replace("d", ""))):
        result = all_results[horizon]
        yearly = result["by_year"]

        row_values = []
        for year in all_years:
            if year in yearly:
                icir = yearly[year]["rank_icir"]
                row_values.append(f"{icir:+.2f}")
            else:
                row_values.append("—")

        lines.append(f"| {horizon:>7s} | " + " | ".join(row_values) + " |")

    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")

    passing_horizons = [row for row in summary_rows if row["all_pass"]]

    if not passing_horizons:
        lines.append("⚠️ **No horizons meet all success criteria.**")
        lines.append("")
        lines.append("**Suggested Actions:**")
        lines.append("1. Review feature quality and data integrity")
        lines.append("2. Consider lowering success thresholds if realistic")
        lines.append("3. Proceed with best-performing horizon and document limitations")
        lines.append("")

        # Show best horizon
        best = max(summary_rows, key=lambda x: x["icir"])
        lines.append(f"**Best Horizon:** {best['horizon']} (ICIR = {best['icir']:.3f})")
    else:
        lines.append(f"✅ **{len(passing_horizons)} horizon(s) meet all success criteria:**")
        lines.append("")

        # Sort by ICIR descending
        passing_horizons.sort(key=lambda x: x["icir"], reverse=True)

        for i, row in enumerate(passing_horizons, 1):
            lines.append(f"{i}. **{row['horizon']}**: ICIR = {row['icir']:.3f}, Top5 = {row['top5_annual']:+.1%} annual")

        lines.append("")
        lines.append("**Recommended for Stage 2 (Feature Screening):**")
        best = passing_horizons[0]
        lines.append(f"- **{best['horizon']}** (highest ICIR)")
        lines.append("")

    # Detailed yearly breakdown section
    lines.append("## Detailed Year-by-Year Metrics")
    lines.append("")

    for horizon in sorted(all_results.keys(), key=lambda h: int(h.replace("d", ""))):
        result = all_results[horizon]
        lines.append(f"### {horizon}")
        lines.append("")
        lines.append("| Year | Rank IC | ICIR | Top5 Excess | IC Pos % | Days |")
        lines.append("|------|---------|------|-------------|----------|------|")

        for year in sorted(result["by_year"].keys()):
            metrics = result["by_year"][year]
            lines.append(
                f"| {year} "
                f"| {metrics['rank_ic']:+.4f} "
                f"| {metrics['rank_icir']:+.3f} "
                f"| {metrics['top5_excess']:+.4f} "
                f"| {metrics['ic_pos_ratio']:.1%} "
                f"| {metrics['n_days']} |"
            )

        lines.append("")

    # Go/No-Go decision
    lines.append("## Go/No-Go Decision")
    lines.append("")

    if passing_horizons:
        lines.append("**✅ GO** - Proceed to Stage 2 (Feature Screening)")
        lines.append("")
        lines.append(f"Use **{passing_horizons[0]['horizon']}** label for feature screening.")
    else:
        lines.append("**⚠️ CONDITIONAL GO** - Proceed with caution")
        lines.append("")
        best = max(summary_rows, key=lambda x: x["icir"])
        lines.append(f"Use **{best['horizon']}** label (best available) for Stage 2.")
        lines.append("Document performance limitations and consider feature engineering improvements.")

    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare label selection results")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/models/v3_0_1_label_selection",
        help="Directory containing model predictions",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/configs/v3_0_1_label_selection.yaml",
        help="Config file with success thresholds",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/results/v3_0_1_label_metrics.json",
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=REPO_ROOT / "v3_pipeline/reports/v3_label_comparison.md",
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    # Load config for thresholds
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    thresholds = config["evaluation"]["success_threshold"]

    # Find all prediction files
    pred_files = sorted(args.models_dir.glob("pred_*.parquet"))
    if not pred_files:
        logger.error(f"No prediction files found in {args.models_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pred_files)} prediction files")

    # Evaluate each horizon
    all_results = {}
    for pred_path in pred_files:
        # Extract horizon from filename (e.g., "pred_10d.parquet" -> "10d")
        horizon = pred_path.stem.replace("pred_", "")

        try:
            pred_df = pd.read_parquet(pred_path)
            result = evaluate_horizon_by_year(pred_df, horizon)
            all_results[horizon] = result
        except Exception as e:
            logger.error(f"Failed to evaluate {horizon}: {e}", exc_info=True)
            continue

    if not all_results:
        logger.error("No horizons evaluated successfully")
        sys.exit(1)

    # Save JSON results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Metrics saved to {args.output_json}")

    # Generate markdown report
    generate_markdown_report(all_results, thresholds, args.output_report)

    logger.info("=" * 80)
    logger.info("Label comparison complete!")
    logger.info(f"Report: {args.output_report}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
