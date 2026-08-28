#!/usr/bin/env python3
"""在已有 OOS 预测上重算排名口径指标（不重训，纯计算）。

对 label 网格各格（或任意含 oos_predictions.parquet 的切分目录）：
从缓存取原始未来收益列，与预测按 (timestamp, symbol) 合并，
调用 src/rank_metrics.daily_rank_metrics 输出排名口径指标。

用法
----
    uv run python scripts/run_rank_metrics.py --horizons 3,5,10,15,20,25,30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rank_metrics import daily_rank_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="已有预测重算排名口径指标")
    parser.add_argument("--horizons", default="3,5,10,15,20,25,30")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--score-start", default="2022-01-01")
    parser.add_argument("--score-end", default="2025-07-31")
    args = parser.parse_args()

    from config.settings import DAILY_FEATURE_DIR, MODEL_DIR

    horizons = [int(x) for x in args.horizons.split(",")]
    tag = f"{args.train_end}_{args.score_start}_{args.score_end}"
    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"

    # 一次性取验证窗口的原始收益列
    ret_cols = [f"future_return_{d}d" for d in horizons]
    dataset = ds.dataset(cache_path, format="parquet")
    filt = (ds.field("timestamp") >= pa.scalar(args.score_start)) & (
        ds.field("timestamp") <= pa.scalar(args.score_end)
    )
    rets = dataset.scanner(columns=["timestamp", "symbol"] + ret_cols, filter=filt).to_table().to_pandas()
    logger.info(f"缓存原始收益列：{len(rets)} 行 × {len(ret_cols)} 列")

    results: dict[str, dict] = {}
    for d in horizons:
        cell = MODEL_DIR / f"single_{tag}_label{d}d" / "oos_predictions.parquet"
        if not cell.exists():
            logger.warning(f"缺少 {cell}，跳过 {d}d")
            continue
        pred = pd.read_parquet(cell)
        merged = pred.merge(rets, on=["timestamp", "symbol"], how="left")
        coverage = merged[f"future_return_{d}d"].notna().mean()
        if coverage < 0.99:
            logger.warning(f"{d}d 合并覆盖率仅 {coverage:.2%}")
        future = f"future_return_{d}d"
        results[future] = daily_rank_metrics(merged, future)
        m = results[future]
        logger.info(
            f"{d}d: RankIC={m['daily_rank_ic']:.4f} ICIR={m['daily_rank_icir']:.3f} "
            f"top5超额={m['daily_top5_excess_ret']:+.4f} 换手={m['top5_turnover']:.3f} "
            f"Q5-Q1={m['quantile_5_1_spread']:+.4f}"
        )

    exp_dir = REPO_ROOT / "experiments"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / f"rank_metrics_{tag}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )

    table = pd.DataFrame(results)
    metric_rows = [r for r in table.index if not r.startswith("quantile_q") and r != "ic_ndays"]
    logger.info(f"\n=== 排名口径对照表 ===\n{table.loc[metric_rows].to_string(float_format=lambda v: f'{v:.4f}')}")
    q_rows = [r for r in table.index if r.startswith("quantile_q")]
    logger.info(f"\n=== 分层各档平均收益 ===\n{table.loc[q_rows].to_string(float_format=lambda v: f'{v:+.4f}')}")
    logger.info(f"结果已写入 {exp_dir}/rank_metrics_{tag}.json")


if __name__ == "__main__":
    main()
