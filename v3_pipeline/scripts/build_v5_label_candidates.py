#!/usr/bin/env python3
"""Build V5 label-candidate cache: add open_exec / mfr label families.

Reads raw daily parquets (stock_data/daily/*.parquet), computes per-symbol
labels for the open_exec and mfr families across all horizons, adds
cross-sectional rank versions, and merges onto the V4 clean cache on
(symbol, date). No rows are dropped from the base cache (left join).

Output: v3_pipeline/feature_cache_v5_labels.parquet

Usage:
    python v3_pipeline/scripts/build_v5_label_candidates.py [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from v3_pipeline.src.label_candidates import max_forward_return, open_exec_return
from v3_pipeline.src.ranking_labels import compute_cross_sectional_ranks

HORIZONS = [3, 5, 10, 15, 20, 25, 30, 45, 60]
DAILY_DIR = REPO_ROOT / "stock_data" / "daily"
BASE_CACHE = REPO_ROOT / "v3_pipeline" / "feature_cache_v4_clean.parquet"
OUT_CACHE = REPO_ROOT / "v3_pipeline" / "feature_cache_v5_labels.parquet"


def compute_labels_one_file(path: Path) -> pd.DataFrame | None:
    """Compute all label columns for one symbol's daily parquet.

    Returns DataFrame with (ts_code, date, <label cols>) or None on failure.
    Index parquets lack a ts_code column; fall back to the file stem.
    """
    try:
        d = pd.read_parquet(path)
        if "ts_code" not in d.columns:
            d["ts_code"] = path.stem
        if len(d) == 0:
            return None
        d = d.sort_values("trade_date").reset_index(drop=True)
        out = pd.DataFrame({
            "ts_code": d["ts_code"],
            "date": pd.to_datetime(d["trade_date"]).dt.strftime("%Y-%m-%d"),
        })
        for h in HORIZONS:
            out[f"open_exec_return_{h}d"] = open_exec_return(d, h).to_numpy()
            out[f"mfr_return_{h}d"] = max_forward_return(d, h).to_numpy()
        return out
    except Exception as exc:  # noqa: BLE001 - log and skip bad files
        print(f"  [WARN] {path.name}: {exc}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    files = sorted(DAILY_DIR.glob("*.parquet"))
    print(f"Computing labels for {len(files)} symbols, horizons={HORIZONS}", flush=True)

    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(compute_labels_one_file, f): f for f in files}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                parts.append(r)
            done += 1
            if done % 500 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files", flush=True)

    labels = pd.concat(parts, ignore_index=True)
    print(f"Label panel: {len(labels)} rows x {len(labels.columns)} cols", flush=True)

    # Cross-sectional ranks per date for each raw label
    for h in HORIZONS:
        for fam in ("open_exec_return", "mfr_return"):
            raw = f"{fam}_{h}d"
            labels[f"rank_{raw}"] = compute_cross_sectional_ranks(
                labels, raw, timestamp_col="date"
            )
    print("Ranks computed", flush=True)

    # Merge onto V4 clean cache (left join, no row deletion)
    base = pd.read_parquet(BASE_CACHE)
    print(f"Base cache: {len(base)} rows x {len(base.columns)} cols", flush=True)
    base = base.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    base["date"] = base["timestamp"].astype(str).str[:10]

    merged = base.merge(
        labels, how="left", left_on=["symbol", "date"], right_on=["ts_code", "date"]
    )
    merged = merged.drop(columns=["ts_code"])
    match = merged["open_exec_return_10d"].notna().mean()
    print(f"Merged: {len(merged)} rows x {len(merged.columns)} cols, match={match:.2%}", flush=True)

    merged.to_parquet(OUT_CACHE, index=False)
    print(f"Saved -> {OUT_CACHE}", flush=True)


if __name__ == "__main__":
    main()
