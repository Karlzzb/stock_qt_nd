#!/usr/bin/env python3
"""
全量特征重算脚本（Issue #12 验收第 1 条）。

功能
----
使用可信 v2 特征管线（无泄露）重算全量特征。
- 行情数据从 DAILY_PARQUET_DIR 读取（Parquet 格式，Issue #5 产物）。
- float64 精度全程计算，落盘时降为 float32。
- 缓存指纹自动校验：版本号变化时旧缓存自动失效。

用法
----
    uv run python scripts/run_feature_pipeline.py [选项]

常用选项
--------
    --start-date YYYY-MM-DD   起始日期（默认：2010-01-01）
    --end-date   YYYY-MM-DD   截止日期（默认：今天）
    --workers N               并发进程数（默认：CPU 核数 - 1，最少 1）
    --force-rebuild           清除所有 .fp 指纹文件，强制重算所有日期

注意
----
本脚本不进 CI，需要真实数据与足够内存（~8 GB）。
预计耗时：全量 5881 只 × ~0.05 s/只/天 × 4000 天 ≈ 数小时（24 并发）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_price_data_from_parquet(
    parquet_dir: Path,
    start_date: str = "2009-01-01",
    end_date: str | None = None,
) -> dict:
    """从 DAILY_PARQUET_DIR 加载行情数据，返回与 feature_pipeline_v2.load_price_data 相同格式的 dict。

    Returns
    -------
    dict[str, pd.DataFrame]
        键：ts_code；值：以 trade_date 为 DatetimeIndex，列为 [open, high, low, close, volume] 的 float64 DataFrame。
    """
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"DAILY_PARQUET_DIR ({parquet_dir}) 下无 Parquet 文件，"
            "请先运行 scripts/fetch_daily_data.py 拉取数据。"
        )

    logger.info(f"共发现 {len(parquet_files)} 只股票的 Parquet 文件，开始加载…")
    stocks: dict = {}

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    for pq_file in parquet_files:
        ts_code = pq_file.stem  # e.g. "000001.SZ"
        try:
            df = pd.read_parquet(pq_file, columns=["trade_date", "open", "high", "low", "close", "vol"])
            df = df.rename(columns={"vol": "volume"})
            # 保证 float64
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype("float64")
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            df["symbol"] = ts_code

            if start_dt is not None:
                df = df[df.index >= start_dt]
            if end_dt is not None:
                df = df[df.index <= end_dt]

            if len(df) > 0:
                stocks[ts_code] = df
        except Exception as exc:
            logger.warning(f"[{ts_code}] 加载失败：{exc}")

    logger.info(f"成功加载 {len(stocks)} 只股票。")
    return stocks


def _clear_fingerprints(feature_dir: Path) -> None:
    """删除所有 .fp 指纹 sidecar 文件，触发缓存失效重算。"""
    fp_files = list(feature_dir.glob("*.fp"))
    for fp in fp_files:
        fp.unlink(missing_ok=True)
    logger.info(f"已清除 {len(fp_files)} 个指纹文件，所有日期将强制重算。")


def main() -> None:
    parser = argparse.ArgumentParser(description="全量特征重算（Issue #12 验收第 1 条）")
    parser.add_argument("--start-date", default="2010-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="截止日期 YYYY-MM-DD（默认：今天）")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="并发进程数（默认：CPU 核数 - 1）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="清除所有 .fp 指纹文件，强制重算所有日期",
    )
    args = parser.parse_args()

    from config.settings import DAILY_PARQUET_DIR, DAILY_FEATURE_DIR
    from feature_pipeline_v2 import process_stocks_batch_parallel_optimized

    logger.info("=== run_feature_pipeline.py 启动 ===")
    logger.info(f"  Parquet 目录 : {DAILY_PARQUET_DIR}")
    logger.info(f"  Feature 目录 : {DAILY_FEATURE_DIR}")
    logger.info(f"  起始日期     : {args.start_date}")
    logger.info(f"  截止日期     : {args.end_date or '今天'}")
    logger.info(f"  并发进程数   : {args.workers}")
    logger.info(f"  强制重算     : {args.force_rebuild}")

    if args.force_rebuild:
        _clear_fingerprints(DAILY_FEATURE_DIR)

    logger.info("正在加载全量行情数据到主进程内存，请稍候…")
    full_stocks = load_price_data_from_parquet(
        DAILY_PARQUET_DIR,
        start_date="2009-01-01",  # 加载足够历史以支持特征计算
        end_date=None,
    )

    logger.info("开始批量特征计算…")
    process_stocks_batch_parallel_optimized(
        full_stocks,
        batch_size=100,
        max_workers=args.workers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info("=== 全量特征重算完成 ===")


if __name__ == "__main__":
    main()
