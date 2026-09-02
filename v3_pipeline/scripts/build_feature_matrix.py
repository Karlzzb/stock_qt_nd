#!/usr/bin/env python3
"""特征矩阵构建驱动: 对主池/备池 div 组事件产出特征矩阵 parquet + feature_dictionary.csv.

用法: python build_feature_matrix.py [--workers 24] [--sample 0]
输出:
  v3_pipeline/reports/feature_matrix/main_pool_features.parquet   (m_fractal15_full, 8158 事件)
  v3_pipeline/reports/feature_matrix/backup_pool_features.parquet (m_zigzag05_nofilter, 37012 事件)
  v3_pipeline/reports/feature_matrix/feature_dictionary.csv
"""
import argparse
import glob
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
import feature_engine as fe  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
SCAN_DIR = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
POOLS = {"main": "m_fractal15_full", "backup": "m_zigzag05_nofilter"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample", type=int, default=0, help="只取前 N 个文件 (冒烟)")
    args = ap.parse_args()
    t0 = time.time()

    # ---- 指数 (IVOL60 与市场特征用)
    idx_sh = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    idx_sz = fe.load_index_df(fe.DATA_DIR / "399001.SZ.parquet")
    idx_dates = idx_sh["_days"].to_numpy(np.int32)
    c_idx = pd.Series(idx_sh["close"].to_numpy(np.float64))
    idx_r = (c_idx / c_idx.shift(1) - 1.0).to_numpy(np.float64)

    # ---- 事件表 (只读 events.parquet; 引擎绝不触碰 labels.parquet)
    events = {}
    per_stock = defaultdict(lambda: defaultdict(list))
    for pool, name in POOLS.items():
        ev = pd.read_parquet(SCAN_DIR / name / "events.parquet").sort_values("event_id")
        ev = ev.reset_index(drop=True)
        n_raw = len(ev)
        # 指数伪股 (000001.SH/399001.SZ) 产生的事件非可交易股票, 剔除 (event_id 保留原值)
        ev = ev[~ev["ts_code"].isin(fe.INDEX_CODES)].reset_index(drop=True)
        print(f"[{pool}] 剔除指数伪股事件 {n_raw - len(ev)} 行 ({n_raw} -> {len(ev)})",
              flush=True)
        events[pool] = ev
        sig_days = fe._to_days(ev["date"])
        for row, sd in zip(ev.itertuples(), sig_days):
            per_stock[row.ts_code][pool].append(
                (int(row.event_id), int(row.sig_idx), int(sd),
                 int(row.low_date), int(row.prev_low_date)))
    print(f"事件: main={len(events['main'])} backup={len(events['backup'])}", flush=True)

    # ---- 全 universe 并行计算
    files = sorted(glob.glob(str(fe.DATA_DIR / "*.parquet")))
    if args.sample:
        files = files[: args.sample]
    tasks = []
    for sid, path in enumerate(files):
        code = Path(path).stem
        tasks.append((path, code, sid, dict(per_stock.get(code, {})), idx_dates, idx_r))
    print(f"加载并计算 {len(tasks)} 只股票 (workers={args.workers}) ...", flush=True)
    panels, pool_rows = [], {p: [] for p in POOLS}
    n_err, n_anchor_miss = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(fe.process_stock, tasks, chunksize=16)):
            if r["error"]:
                n_err += 1
                continue
            n_anchor_miss += r["n_anchor_miss"]
            if r["panel"] is not None:
                panels.append(r["panel"])
            for pool, blk in r["rows"].items():
                pool_rows[pool].append(blk)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    panel = pd.concat(panels, ignore_index=True)
    print(f"面板 {len(panel)} 行, 跳过 {n_err} 只, 锚点缺失 {n_anchor_miss} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 市场特征框 (date 级)
    market, _ = fe.build_market_frame(panel, idx_sh, idx_sz)
    print(f"市场特征框 {market.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 装配两池矩阵
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pool in POOLS:
        rows = pd.concat(pool_rows[pool]).sort_index()
        mat = fe.assemble_pool(events[pool], rows, market, panel)
        path = OUT_DIR / f"{pool}_pool_features.parquet"
        mat.to_parquet(path, index=False)
        feat_cols = [c for c in mat.columns if c not in fe.META_COLS]
        nan_share = float(mat[feat_cols].isna().mean().mean())
        print(f"[{pool}] 矩阵 {mat.shape} -> {path} (NaN 占比 {nan_share:.3f}, "
              f"{time.time()-t0:.0f}s)", flush=True)

    dic = fe.feature_dictionary()
    dic_path = OUT_DIR / "feature_dictionary.csv"
    dic.to_csv(dic_path, index=False)
    print(f"特征字典 {dic.shape} -> {dic_path}", flush=True)
    print(f"完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
