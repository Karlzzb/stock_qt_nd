#!/usr/bin/env python3
"""
从 feature_cache_all.parquet 生成 test_set.csv / validation_set.csv（Issue #12 AC3 前置步骤）。

背景
----
原 data_process.py 通过逐个读取 8766 个 realistic_features_*.csv 生成三个数据集，
全量 22M 行 × 678 列 float64 远超内存（>100GB），不可行。
本脚本改为从 AC1 产出的全量 parquet 缓存流式生成，数据内容与 daily CSV 完全一致
（缓存本身就是由这些 CSV 构建的）。

split 逻辑与 data_process.py main() 完全一致：
按日期聚合行数 → 累计行数 60% / 80% 处取最近日期 → test=(60%,80%], val=(80%,100%]。

只写模拟回测实际需要的列：
  timestamp, symbol, LABEL_COL + lgbm_features.json（模型实际使用的 138 列）。
旧管线中的 sh_* / divergence_* 等 21 列在特征重构后已不存在于数据源，
模型也不使用它们（详见 predictor_model_v2._prepare_features 注释）。

用法
----
    uv run python scripts/generate_test_val_sets.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow.dataset as ds
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from comm_fun import model_config
    from config.settings import DAILY_FEATURE_DIR, DATASET_DIR, MODEL_DIR

    cache_path = DAILY_FEATURE_DIR.parent / "feature_cache_all.parquet"
    lgbm_feats = json.loads((MODEL_DIR / "lgbm_features.json").read_text())
    out_cols = ["timestamp", "symbol", model_config.LABEL_COL] + lgbm_feats

    dataset = ds.dataset(cache_path, format="parquet")

    # ---- 第一步：只扫 timestamp 列，按日期聚合计数，复刻 60/20/20 split ----
    import pandas as pd

    ts = dataset.scanner(columns=["timestamp"], batch_size=500_000).to_table().to_pandas()
    date_counts = ts["timestamp"].value_counts().sort_index()
    del ts
    cumulative = date_counts.cumsum()
    total_rows = int(cumulative.iloc[-1])
    train_end_date = date_counts.index[(cumulative - int(total_rows * 0.6)).abs().argmin()]
    val_end_date = date_counts.index[(cumulative - int(total_rows * 0.8)).abs().argmin()]
    logger.info(
        f"总行数={total_rows}，train_end={train_end_date}，val_end={val_end_date}"
    )

    # ---- 第二步：流式扫描所需列，按日期区间写两个 CSV ----
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    test_path = DATASET_DIR / "test_set.csv"
    val_path = DATASET_DIR / "validation_set.csv"

    import pyarrow as pa

    written_test = written_val = 0
    header_test = header_val = False
    scanner = dataset.scanner(columns=out_cols, batch_size=500_000)
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        ts_col = df["timestamp"]
        test_mask = (ts_col > train_end_date) & (ts_col <= val_end_date)
        val_mask = ts_col > val_end_date
        if test_mask.any():
            df.loc[test_mask].to_csv(
                test_path, mode="a", index=False, header=not header_test
            )
            header_test = True
            written_test += int(test_mask.sum())
        if val_mask.any():
            df.loc[val_mask].to_csv(
                val_path, mode="a", index=False, header=not header_val
            )
            header_val = True
            written_val += int(val_mask.sum())
        del df

    logger.info(f"test_set.csv: {written_test} 行 → {test_path}")
    logger.info(f"validation_set.csv: {written_val} 行 → {val_path}")
    expected_test = int(date_counts[(date_counts.index > train_end_date) & (date_counts.index <= val_end_date)].sum())
    expected_val = int(date_counts[date_counts.index > val_end_date].sum())
    assert written_test == expected_test, f"test 行数不符：{written_test} != {expected_test}"
    assert written_val == expected_val, f"val 行数不符：{written_val} != {expected_val}"
    logger.info("=== test/val 数据集生成完成 ===")


if __name__ == "__main__":
    main()
