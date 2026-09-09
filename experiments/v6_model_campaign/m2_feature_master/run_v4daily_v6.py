#!/usr/bin/env python3
"""来源 3:V4 日频特征 v6 事件日快照重建驱动(战役 #47 M2,issue #49)。

机器来源:v3_pipeline/scripts/rebuild_v4_daily_snapshot.py 改造(#44 §2.4 来源 3,
适配等级 C=Pass A 重跑)。改造点:
  - 键源: v5 双池 -> v6 事件表 96,577 键(单池,池无关);
  - **Pass A 落全历史行**(池无关,#44 核心建议):parts 目录为全新的
    cache/v4daily_parts_fullhist/,启动时断言目录不存在或为空——旧 parts 的
    缓存命中陷阱(#44 §1:文件存在即命中、静默缺行)以全新目录规避,
    本驱动不含任何"已存在即跳过"逻辑;
  - Pass B 日期块横截面装配与 v5 逐字同构(250 日一块,单池);
  - 前缀稳定性抽检改在 v6 事件日上(12 股 × 3 日,seed 42,v5 同口径)。

口径与偏离登记全部继承 v3_pipeline/src/v4_daily_snapshot.py 模块 docstring(7 条),
本驱动零改动计算本体。

输入:
  事件表   experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet(只读)
  原始日线 stock_data/daily/*.parquet(只读)
输出:
  cache/s3_v4daily_snapshot.parquet      (96,577 键左覆盖,缺行 NaN 保留并计数落盘)
  cache/s3_results.json                  (台账)
中间产物:
  cache/v4daily_parts_fullhist/{code}.parquet  逐股全历史特征行(约 21 GiB,池无关)

用法: python run_v4daily_v6.py [--workers 24] [--sample 0]
            [--parts-dir cache/v4daily_parts_fullhist] [--out cache/s3_v4daily_snapshot.parquet]
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import src.feature_pipeline_v2 as fp2  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
CACHE_DIR = SCRIPT_DIR / "cache"
DAILY_DIR = REPO / "stock_data" / "daily"
INDEX_CODES = ("000001.SH", "399001.SZ")
PROGRESS = SCRIPT_DIR / "progress.log"

DATE_CHUNK = 250  # 横截面排名的日期块大小(v5 原值)

# 前缀稳定性抽检规模(确定性种子,v5 原值)
N_ASSERT_STOCKS = 12
ASSERT_DAYS_PER_STOCK = 3
ASSERT_SEED = 42


def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [s3] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_event_keys():
    """v6 事件键 + 事件日并集(按股)。"""
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date"])
    ev = ev.rename(columns={"event_date": "date"})
    ev["date"] = pd.to_datetime(ev["date"])
    assert not ev.duplicated(["ts_code", "date"]).any(), "事件键不唯一"
    per_stock = defaultdict(set)
    for code, d in zip(ev["ts_code"], ev["date"]):
        per_stock[code].add(d)
    return ev, per_stock


def market_rank_frame(event_dates):
    """rank_return / rank_volume 的全市场当日横截面百分位(V2 原口径,v5 逐字)。"""
    frames = []
    files = sorted(DAILY_DIR.glob("*.parquet"))
    for path in files:
        code = path.stem
        if code in INDEX_CODES:
            continue  # 指数不参与个股横截面排名
        df = pd.read_parquet(path, columns=["trade_date", "close", "vol"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date")
        df["pct_change"] = df["close"].pct_change()
        df = df[df["trade_date"].isin(event_dates)]
        if len(df):
            frames.append(pd.DataFrame({"symbol": code, "timestamp": df["trade_date"],
                                        "pct_change": df["pct_change"].to_numpy(),
                                        "volume": df["vol"].to_numpy()}))
    big = pd.concat(frames, ignore_index=True)
    grp = big.groupby("timestamp")
    big["rank_return"] = grp["pct_change"].rank(pct=True)
    big["rank_volume"] = grp["volume"].rank(pct=True)
    big[["rank_return", "rank_volume"]] = big[["rank_return", "rank_volume"]].fillna(0.0)
    return big[["symbol", "timestamp", "rank_return", "rank_volume"]]


def _worker(task):
    """Pass A: 单股全历史特征 -> 全历史行落 parts(池无关;无缓存命中逻辑)。"""
    code, parts_dir = task
    try:
        path = DAILY_DIR / f"{code}.parquet"
        if not path.exists():
            return code, 0, "缺文件"
        df = v4s.load_stock_v2(path, code)
        feat = v4s.compute_stock_features(df, pipe=fp2.FeaturePipeline(None, None))
        if feat is None:
            return code, 0, "历史不足100行"
        feat = feat.drop(columns=[c for c in v4s.MARKET_RANK_COLS if c in feat.columns])
        feat.to_parquet(Path(parts_dir) / f"{code}.parquet", index=False)
        return code, len(feat), None
    except Exception as e:  # noqa: BLE001
        return code, 0, repr(e)


def market_features_for_dates(dates):
    """对全部事件日逐日算大盘特征(V2 原函数,指数截断语义内建)。"""
    df_sh = v4s.load_index_v2(DAILY_DIR / "000001.SH.parquet")
    df_sz = v4s.load_index_v2(DAILY_DIR / "399001.SZ.parquet")
    pipe = fp2.FeaturePipeline(None, None)
    frames = [pipe._calculate_market_features(d, df_sh=df_sh, df_sz=df_sz)
              for d in sorted(pd.to_datetime(list(dates)))]
    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out.reset_index()


def prefix_stability_assert(per_stock):
    """快照口径前缀稳定性: 抽样股票 x v6 事件日, 全历史值与截断重算逐位一致。"""
    rng = np.random.default_rng(ASSERT_SEED)
    stocks = sorted(per_stock)
    picks = rng.choice(len(stocks), size=min(N_ASSERT_STOCKS, len(stocks)),
                       replace=False)
    pipe = fp2.FeaturePipeline(None, None)
    n_checked, mismatches = 0, []
    for i in picks:
        code = stocks[i]
        df = v4s.load_stock_v2(DAILY_DIR / f"{code}.parquet", code)
        feat_full = v4s.compute_stock_features(df, pipe=pipe)
        if feat_full is None:
            continue
        days = sorted(d for d in per_stock[code]
                      if d in set(feat_full["timestamp"]))
        if not days:
            continue
        day_picks = rng.choice(len(days),
                               size=min(ASSERT_DAYS_PER_STOCK, len(days)),
                               replace=False)
        for jj in day_picks:
            T = days[jj]
            ref = v4s.prefix_recompute_at(DAILY_DIR / f"{code}.parquet", code, T,
                                          pipe=pipe)
            if ref is None:
                continue
            row_full = feat_full[feat_full["timestamp"] == T]
            cmp_cols = [c for c in feat_full.columns
                        if c not in v4s.MARKET_RANK_COLS + v4s.KEY_COLS]
            a = row_full[cmp_cols].iloc[0].to_numpy(np.float64)
            b = ref[cmp_cols].iloc[0].to_numpy(np.float64)
            n_checked += 1
            if not np.allclose(a, b, rtol=1e-9, atol=0, equal_nan=True):
                bad = [cmp_cols[k] for k in np.where(
                    ~np.isclose(a, b, rtol=1e-9, atol=0, equal_nan=True))[0]]
                mismatches.append({"ts_code": code, "date": str(T.date()),
                                   "cols": bad[:10]})
    return {"n_checked": n_checked, "mismatches": mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample", type=int, default=0, help="Pass A 只取前 N 只股票 (冒烟)")
    ap.add_argument("--parts-dir", type=str,
                    default=str(CACHE_DIR / "v4daily_parts_fullhist"))
    ap.add_argument("--out", type=str,
                    default=str(CACHE_DIR / "s3_v4daily_snapshot.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    parts_dir = Path(args.parts_dir)
    # 缓存命中陷阱物理规避: parts 目录必须全新(不存在或为空)
    if parts_dir.exists():
        stale = list(parts_dir.glob("*.parquet"))
        assert not stale, f"parts 目录非空({len(stale)} 文件),疑缓存命中陷阱: {parts_dir}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    keys, per_stock = load_event_keys()
    event_dates = set(keys["date"])
    log(f"事件键 {len(keys)};涉股 {len(per_stock)},事件日并集 {len(event_dates)}")

    # ---- 前缀稳定性断言(先验后算,口径级回归守卫;v6 事件日上)
    log("前缀稳定性抽检 ...")
    stab = prefix_stability_assert(per_stock)
    log(f"抽检 {stab['n_checked']} 个 (股票,日) 单元,不一致 {len(stab['mismatches'])}")
    assert stab["n_checked"] > 0, "前缀稳定性抽检为空"
    assert not stab["mismatches"], f"前缀稳定性不一致: {stab['mismatches'][:3]}"

    # ---- 全市场横截面排名 pass (rank_return/rank_volume)
    log("全市场 rank_return/rank_volume pass ...")
    rank_df = market_rank_frame(event_dates)
    log(f"排名面板 {rank_df.shape} ({time.time() - t0:.0f}s)")

    # ---- Pass A: 逐股特征链,全历史行落盘(池无关)
    all_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet")
                       if p.stem not in INDEX_CODES)
    tasks = [(code, str(parts_dir)) for code in all_codes]
    if args.sample:
        tasks = tasks[: args.sample]
    log(f"Pass A 逐股特征链 {len(tasks)} 只(全历史落盘,workers={args.workers}) ...")
    n_rows, errors = 0, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, n, err) in enumerate(ex.map(_worker, tasks, chunksize=8)):
            if err:
                errors[code] = err
            else:
                n_rows += n
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    n_parts = len(list(parts_dir.glob("*.parquet")))
    log(f"Pass A 完成: 全历史行 {n_rows},parts {n_parts} 只,失败/跳过 {len(errors)} "
        f"({time.time() - t0:.0f}s)")

    # ---- 大盘特征
    mkt = market_features_for_dates(event_dates)
    log(f"大盘特征 {mkt.shape} ({time.time() - t0:.0f}s)")

    # ---- Pass B: 日期块横截面排名 + 事件单元格抽取(单池,v5 同构)
    dates_sorted = sorted(event_dates)
    chunks = [dates_sorted[i: i + DATE_CHUNK]
              for i in range(0, len(dates_sorted), DATE_CHUNK)]
    part_files = sorted(parts_dir.glob("*.parquet"))
    log(f"Pass B 横截面排名: {len(chunks)} 个日期块 x {len(part_files)} 只股 ...")
    pipe = fp2.FeaturePipeline(None, None)
    blocks = []
    ev_keys = pd.MultiIndex.from_arrays([keys["ts_code"], keys["date"]])
    for ci, cdates in enumerate(chunks):
        cset = set(cdates)
        frames = []
        for pf in part_files:
            d = pd.read_parquet(pf)
            d = d[d["timestamp"].isin(cset)]
            if len(d):
                frames.append(d)
        if not frames:
            continue
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.merge(mkt, on="timestamp", how="left")
        panel = panel.merge(rank_df, on=["symbol", "timestamp"], how="left")
        panel[["rank_return", "rank_volume"]] = \
            panel[["rank_return", "rank_volume"]].fillna(0.0)
        panel = pipe._calculate_cross_features(panel)
        panel_idx = pd.MultiIndex.from_arrays([panel["symbol"], panel["timestamp"]])
        mask = panel_idx.isin(ev_keys)
        if mask.any():
            blocks.append(panel.loc[mask])
        log(f"  块 {ci + 1}/{len(chunks)}: 面板 {panel.shape} ({time.time() - t0:.0f}s)")
        del panel, frames

    # ---- 装配落盘
    df = pd.concat(blocks, ignore_index=True)
    df = v4s.add_calendar_features(df)
    df = df.rename(columns={"timestamp": "date", "symbol": "ts_code"})
    df = df.sort_values(["ts_code", "date"], kind="mergesort").reset_index(drop=True)
    assert not df.duplicated(["ts_code", "date"]).any()
    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    n_missed = int(len(keys) - len(df))
    log(f"来源 3 快照 {df.shape} -> {out_path.name} (缺快照事件 {n_missed},NaN 保留)")

    results = {
        "prefix_stability": stab,
        "pass_a": {"fullhist_rows": n_rows, "parts_files": n_parts,
                   "errors": errors},
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "events_missing_snapshot": n_missed,
        "elapsed_sec": time.time() - t0,
    }
    rpath = CACHE_DIR / "s3_results.json" \
        if out_path.resolve() == (CACHE_DIR / "s3_v4daily_snapshot.parquet").resolve() \
        else out_path.with_suffix(".results.json")
    rpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    log(f"结果 -> {rpath}; 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
