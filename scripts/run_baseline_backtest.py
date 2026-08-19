#!/usr/bin/env python3
"""
v8/v12 原参数重跑脚本（Issue #12 验收第 3 条）。

功能
----
使用 walk-forward 重训后的模型（MODEL_DIR），以原始固定策略参数（param35 / param2）
重跑 v8 与 v12 回测，得到诚实基线数字。

参数不得重新在评估段上搜索。本脚本只跑 param35（v8）和 param2（v12）。

用法
----
    uv run python scripts/run_baseline_backtest.py [选项]

选项
----
    --strategy  v8|v12|all      要跑的策略（默认：all）
    --start-date YYYY-MM-DD     回测起始（默认：全量）
    --end-date   YYYY-MM-DD     回测截止（默认：全量）
    --capital N                 初始资金（默认：248526）
    --output-dir PATH           结果输出目录（默认：output/baseline_results）

输出
----
    output/baseline_results/v8_param35_result.json
    output/baseline_results/v12_param2_result.json
    output/baseline_results/summary.json

注意：需要先完成 run_feature_pipeline.py + run_walk_forward_train.py。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _log_result_summary(label: str, result: dict) -> None:
    """安全打印回测结果摘要（容忍 missing key）。"""
    def _f(key: str) -> str:
        v = result.get(key)
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    logger.info(
        f"{label} → 收益率={_f('return_rate')}  "
        f"回撤={_f('max_drawdown')}  "
        f"夏普={_f('sharpe_ratio')}  "
        f"交易数={result.get('total_trades', 'N/A')}"
    )


def run_v8(
    capital: float,
    start_date: str | None,
    end_date: str | None,
    output_dir: Path,
) -> dict:
    """跑 v8 param35 回测，返回结果 dict。"""
    from config.settings import DATASET_DIR, MODEL_DIR
    from comm_fun import model_config
    from grid_trading_simulation_v8 import data_process, simple_run

    logger.info("=== v8 param35 回测开始 ===")
    params = model_config.STRATEGY_PARAMS_CANDIDATES_V8["param35"]
    logger.info(f"策略参数：{params}")

    processed_data = data_process(
        dataset_dir=DATASET_DIR,
        required_files=["test_set.csv", "validation_set.csv"],
        start_date=start_date,
        end_date=end_date,
    )
    if processed_data is None or len(processed_data) == 0:
        raise RuntimeError("v8 数据加载失败，请检查 DATASET_DIR 中是否有 test_set.csv / validation_set.csv")

    _, result_df = simple_run(
        initial_capital=capital,
        strategy_name="param35",
        strategy_params=params,
        full_data=processed_data,
    )
    # simple_run v8 返回 (txt_list, DataFrame)
    result = result_df.iloc[0].to_dict() if result_df is not None and not result_df.empty else {}
    result["version"] = "v8"
    result["param_key"] = "param35"
    result["initial_capital"] = capital

    out_file = output_dir / "v8_param35_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"v8 结果已保存至 {out_file}")
    _log_result_summary("v8 param35", result)
    return result


def run_v12(
    capital: float,
    start_date: str | None,
    end_date: str | None,
    output_dir: Path,
) -> dict:
    """跑 v12 param2 回测，返回结果 dict。"""
    from config.settings import DATASET_DIR
    from comm_fun import model_config
    from grid_trading_simulation_v12 import (
        data_process,
        simple_run,
        _preload_st_cache,
        PrecomputedATR,
    )

    logger.info("=== v12 param2 回测开始 ===")
    params = model_config.STRATEGY_PARAMS_CANDIDATES_V12["param2"]
    logger.info(f"策略参数：{params}")

    processed_data = data_process(
        dataset_dir=DATASET_DIR,
        required_files=["test_set.csv", "validation_set.csv"],
        start_date=start_date,
        end_date=end_date,
    )
    if processed_data is None or len(processed_data) == 0:
        raise RuntimeError("v12 数据加载失败")

    # 价格字典（ATR 缓存需要）
    prices_df_dict: dict = {}
    if "code" in processed_data.columns and "date" in processed_data.columns:
        for code, group in processed_data.groupby("code"):
            g = group.sort_values("date").set_index("date")
            prices_df_dict[code] = g[["open", "high", "low", "close", "volume"]]

    trade_dates = sorted(processed_data["date"].dt.strftime("%Y%m%d").unique())
    st_preloaded = _preload_st_cache(trade_dates)
    atr_cache = PrecomputedATR(prices_df_dict, trade_dates, lookbacks=[7, 10, 14, 21])

    result_df = simple_run(
        initial_capital=capital,
        strategy_name="param2",
        strategy_params=params,
        full_data=processed_data,
        prices_df=prices_df_dict,
        atr_cache=atr_cache,
        st_preloaded=st_preloaded,
    )
    result = result_df.iloc[0].to_dict() if result_df is not None and not result_df.empty else {}
    result["version"] = "v12"
    result["param_key"] = "param2"
    result["initial_capital"] = capital

    out_file = output_dir / "v12_param2_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"v12 结果已保存至 {out_file}")
    _log_result_summary("v12 param2", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="v8/v12 原参数诚实基线重跑")
    parser.add_argument("--strategy", choices=["v8", "v12", "all"], default="all")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--capital", type=float, default=248526.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="结果输出目录（默认：output/baseline_results）",
    )
    args = parser.parse_args()

    from config.settings import RESULT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else RESULT_DIR / "baseline_results"

    results: dict = {}

    if args.strategy in ("v8", "all"):
        try:
            results["v8"] = run_v8(args.capital, args.start_date, args.end_date, output_dir)
        except Exception as exc:
            logger.error(f"v8 回测失败：{exc}", exc_info=True)
            results["v8"] = {"error": str(exc)}

    if args.strategy in ("v12", "all"):
        try:
            results["v12"] = run_v12(args.capital, args.start_date, args.end_date, output_dir)
        except Exception as exc:
            logger.error(f"v12 回测失败：{exc}", exc_info=True)
            results["v12"] = {"error": str(exc)}

    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"汇总结果已保存至 {summary_path}")
    logger.info("=== 基线重跑完成 ===")


if __name__ == "__main__":
    main()
