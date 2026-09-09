#!/usr/bin/env python3
"""来源 2:特征工厂 v6 全事件键重生成(战役 #47 M2,issue #49)。

机器来源:v3_pipeline/scripts/regen_factory_full.py 改造(#44 §2.4 来源 2,
适配等级 B=改键+单池)。改造点:
  - 键源: v5 双池特征矩阵键 -> v6 事件表 96,577 键(单池,池无关);
  - 输出: 单一 cache/s2_factory_full.parquet(v5 为 main/backup 双产物);
  - 缓存命中逻辑删除(M2 纪律: 确定性双跑,每次必重算,--out 区分产物);
  - 逐股 compute_stock_factory 全历史计算后切片事件日(纯因果滚动算子,
    v5 截断对拍结论沿用,见 run_feature_factory._truncation_check)。

输入:
  事件表   experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet(只读)
  原始日线 stock_data/daily/*.parquet(只读,fe.load_stock_df 口径=v5 原口径;
           实测全市场 0 文件存在 NaN close/重复 trade_date,与 scanner 口径等价)
输出:
  cache/s2_factory_full.parquet   (96,577 行 × ~1,605 列,键 ts_code+date)
  cache/s2_results.json           (台账:缺行/耗时/列数)

用法: python run_factory_v6.py [--workers 24] [--out cache/s2_factory_full.parquet]
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
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_engine as fe  # noqa: E402
import feature_factory as ff  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

_RIDX = None


def _gen_init(ridx):
    global _RIDX
    _RIDX = ridx


def load_ridx_map():
    """指数日收益映射(与 regen_factory_full.load_ridx_map 同口径)。"""
    idx = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    c = idx["close"].to_numpy(np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    return {int(d): float(v) for d, v in zip(idx["_days"].to_numpy(np.int64), r)}


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [s2] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _gen_worker(task):
    """单股全历史工厂特征 -> 事件日行(与 regen_factory_full._gen_worker 同构)。"""
    code, dates = task
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
        rows, kept = [], []
        for d in dates:
            p = pos.get(int(np.datetime64(d, "D").astype(np.int64)))
            if p is not None:
                rows.append(p)
                kept.append(d)
        if not kept:
            return code, None, "事件日全缺"
        return code, (kept, mat[rows], names), None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=str, default=str(CACHE_DIR / "s2_factory_full.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- v6 事件键(单池)
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date"])
    ev = ev.rename(columns={"event_date": "date"})
    ev["date"] = pd.to_datetime(ev["date"])
    assert not ev.duplicated(["ts_code", "date"]).any(), "事件键不唯一"
    per_stock = defaultdict(list)
    for code, d in zip(ev["ts_code"], ev["date"]):
        per_stock[code].append(d)
    log(f"事件键 {len(ev)},涉股 {len(per_stock)}")

    ridx = load_ridx_map()
    tasks = sorted((code, dates) for code, dates in per_stock.items())
    log(f"逐股生成 {len(tasks)} 只 (workers={args.workers}) ...")
    names = None
    blocks = []
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_gen_init, initargs=(ridx,)) as ex:
        for i, (code, res, err) in enumerate(
                ex.map(_gen_worker, tasks, chunksize=8)):
            if err:
                errors[code] = err
                continue
            kept, mat, stock_names = res
            if names is None:
                names = stock_names
            assert stock_names == names, f"{code} 工厂列集与全局不一致"
            blk = pd.DataFrame(mat, columns=names)
            blk["ts_code"] = code
            blk["date"] = pd.to_datetime(kept)
            blocks.append(blk)
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")

    df = pd.concat(blocks, ignore_index=True)
    df = df[["ts_code", "date"] + [c for c in df.columns
                                   if c not in ("ts_code", "date")]]
    df = df.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)
    assert not df.duplicated(["ts_code", "date"]).any()
    n_missing = int(len(ev) - len(df))
    # 缺行即停线: 事件日必在个股日线内(M1 已锁),缺一行都是 bug
    assert n_missing == 0, f"事件键缺行 {n_missing}: {sorted(errors)[:10]}"
    fe.assert_no_inf(df[[c for c in df.columns if c not in ("ts_code", "date")]])

    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    log(f"来源 2 矩阵 {df.shape} -> {out_path.name} ({time.time() - t0:.0f}s)")

    results = {
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "feature_cols": int(df.shape[1] - 2),
        "events_missing": n_missing, "errors": errors,
        "elapsed_sec": time.time() - t0,
    }
    rpath = CACHE_DIR / "s2_results.json" if out_path.name == "s2_factory_full.parquet" \
        else out_path.with_suffix(".results.json")
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
