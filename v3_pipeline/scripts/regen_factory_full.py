#!/usr/bin/env python3
"""特征工厂全池重生成（issue #22 来源 2）。

背景: 既有缓存 reports/feature_factory/cache/factory_features_{main,backup}.parquet
只覆盖干净 train/val 子集（主 5702/8154、备 25551/36986，见 run_feature_factory
build_universe）。本脚本对全池事件键重生成，口径与先生成阶段逐位一致
（feature_factory.compute_stock_factory 为纯因果滚动算子，已经截断对拍验证，
见 run_feature_factory._truncation_check）。

输入: v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet (事件键)
输出: v3_pipeline/reports/feature_master/cache/factory_full_{main,backup}.parquet
      v3_pipeline/reports/feature_master/cache/factory_full_results.json

用法: python v3_pipeline/scripts/regen_factory_full.py [--workers 24] [--force]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_engine as fe  # noqa: E402
import feature_factory as ff  # noqa: E402

FM_DIR = REPO / "v3_pipeline" / "reports" / "feature_matrix"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "feature_master" / "cache"
POOLS = ("main", "backup")
PROGRESS = OUT_DIR / "factory_full_progress.log"

_RIDX = None


def _gen_init(ridx):
    global _RIDX
    _RIDX = ridx


def load_ridx_map():
    """指数日收益映射（与 run_feature_factory.load_ridx_map 同口径）。"""
    idx = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    c = idx["close"].to_numpy(np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    return {int(d): float(v) for d, v in zip(idx["_days"].to_numpy(np.int64), r)}


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _gen_worker(task):
    """单股全历史工厂特征 -> 两池事件日行。与 run_feature_factory._gen_worker 同构。"""
    code, ev = task  # ev: {pool: [Timestamp, ...]}
    try:
        path = fe.DATA_DIR / f"{code}.parquet"
        if not path.exists():
            return code, None, "缺文件"
        df = fe.load_stock_df(path)
        cols, _ = ff.compute_stock_factory(df, _RIDX)
        days_i = df["trade_date"].to_numpy("datetime64[D]").astype(np.int64)
        pos = {int(d): i for i, d in enumerate(days_i)}
        names = list(cols.keys())
        mat = np.column_stack([cols[n] for n in names])
        out = {}
        for pool, dates in ev.items():
            rows, kept = [], []
            for d in dates:
                p = pos.get(int(np.datetime64(d, "D").astype(np.int64)))
                if p is not None:
                    rows.append(p)
                    kept.append(d)
            if kept:
                out[pool] = (kept, mat[rows])
        return code, (out, names), None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # ---- 全池事件键（特征矩阵键 = 事件表剔指数伪股）
    per_stock = defaultdict(lambda: defaultdict(list))
    pool_sizes = {}
    for pool in POOLS:
        fm = pd.read_parquet(FM_DIR / f"{pool}_pool_features.parquet",
                             columns=["ts_code", "date"])
        fm["date"] = pd.to_datetime(fm["date"])
        assert not fm.duplicated(["ts_code", "date"]).any()
        pool_sizes[pool] = len(fm)
        for code, d in zip(fm["ts_code"], fm["date"]):
            per_stock[code][pool].append(d)
    log(f"全池事件键: main={pool_sizes['main']} backup={pool_sizes['backup']}, "
        f"涉股 {len(per_stock)}")

    out_paths = {p: OUT_DIR / f"factory_full_{p}.parquet" for p in POOLS}
    if all(p.exists() for p in out_paths.values()) and not args.force:
        log("缓存命中, 跳过生成 (--force 重跑)")
        return

    ridx = load_ridx_map()
    tasks = sorted((code, dict(ev)) for code, ev in per_stock.items())
    log(f"逐股生成 {len(tasks)} 只 (workers={args.workers}) ...")
    names = None
    blocks = {p: [] for p in POOLS}
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_gen_init, initargs=(ridx,)) as ex:
        for i, (code, res, err) in enumerate(
                ex.map(_gen_worker, tasks, chunksize=8)):
            if err:
                errors[code] = err
                continue
            out, stock_names = res
            if names is None:
                names = stock_names
            assert stock_names == names, f"{code} 工厂列集与全局不一致"
            for pool, (kept, mat) in out.items():
                blk = pd.DataFrame(mat, columns=names)
                blk["ts_code"] = code
                blk["date"] = pd.to_datetime(kept)
                blocks[pool].append(blk)
            if (i + 1) % 500 == 0:
                log(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")

    results = {"errors": errors, "pools": {}}
    for pool in POOLS:
        df = pd.concat(blocks[pool], ignore_index=True)
        df = df[["ts_code", "date"] + [c for c in df.columns
                                       if c not in ("ts_code", "date")]]
        df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
        df.to_parquet(out_paths[pool], index=False)
        results["pools"][pool] = {
            "rows": int(len(df)), "cols": int(df.shape[1]),
            "events_missing": int(pool_sizes[pool] - len(df)),
            "path": str(out_paths[pool].relative_to(REPO)),
        }
        log(f"[{pool}] {df.shape} -> {out_paths[pool]} "
            f"(缺 {pool_sizes[pool] - len(df)})")
    rpath = OUT_DIR / "factory_full_results.json"
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
