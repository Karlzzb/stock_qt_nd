#!/usr/bin/env python3
"""标签赛标签表构建驱动（issue #26）：两池各落一份十九列标签表。

流程:
  1. 载两池事件表（m_scan 锁定产物），取 (ts_code, date) 键集与股票并集;
  2. 狙击标签 hit_N20_k2.0 经 tep.load_div_labels 从各池 labels.parquet
     div 组逐位对齐装载（口径同 #25 冒烟）;
  3. 十八个收益二分类标签（cur/open_exec 两族各九视野，收益大于零）按股票
     在 stock_data/daily 全序列上计算（label_race.compute_return_labels，
     复用 label_candidates 因果算子），再按 (ts_code, date) 合并到各池事件;
  4. 因果性断言（真实数据）：抽样股票截掉序列头部 200 行重算，交集日期
     标签逐位一致（标签只看未来，过去截断不得引起漂移）;
  5. 键缺失（事件股票无日线文件/日期无 bar）计数落台账，标签 NaN 保留
     （尾部截断/停牌顺延），剔除时机在跑训驱动按段登记。

输出:
  v3_pipeline/reports/label_race/labels_race_{main,backup}.parquet
  v3_pipeline/reports/label_race/labels_build_results.json   台账

用法: python v3_pipeline/scripts/build_race_labels.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

LAB_DIR = REPO / "v3_pipeline" / "reports" / "divergence_lab" / "m_scan"
POOL_LAB = {"main": "m_fractal15_full", "backup": "m_zigzag05_nofilter"}
DAILY_DIR = REPO / "stock_data" / "daily"
OUT_DIR = REPO / "v3_pipeline" / "reports" / "label_race"

TRUNC_CHECK_N_SYMBOLS = 20
TRUNC_HEAD_ROWS = 200


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "labels_build_progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _labels_one_symbol(ts_code: str) -> pd.DataFrame | None:
    path = DAILY_DIR / f"{ts_code}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close"])
    if len(d) == 0:
        return None
    out = lr.compute_return_labels(d)
    out.insert(0, "ts_code", ts_code)
    return out


def _truncation_check(symbols: list[str]) -> dict:
    """因果性断言：抽样股票截头部 TRUNC_HEAD_ROWS 行重算，交集日期逐位一致。"""
    rng = np.random.default_rng(20260903)
    sample = list(rng.choice(symbols, size=min(TRUNC_CHECK_N_SYMBOLS, len(symbols)),
                             replace=False))
    checked = 0
    for ts in sample:
        path = DAILY_DIR / f"{ts}.parquet"
        d = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close"])
        if len(d) <= TRUNC_HEAD_ROWS + 80:
            continue
        full = lr.compute_return_labels(d)
        cut = lr.compute_return_labels(d.iloc[TRUNC_HEAD_ROWS:].copy())
        m = full.merge(cut, on="date", suffixes=("_a", "_b"))
        assert len(m) == len(cut), f"{ts} 截断后日期错位"
        for c in lr.candidate_labels()[1:]:
            a, b = m[f"{c}_a"].to_numpy(), m[f"{c}_b"].to_numpy()
            assert np.array_equal(a, b, equal_nan=True), \
                f"{ts} 标签 {c} 头部截断后漂移（因果性破坏）"
        checked += 1
    return {"sampled": len(sample), "checked": checked, "head_rows": TRUNC_HEAD_ROWS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    log("[标签构建] 开工: 两池十九列标签表")

    # ---------------------------------------------------------- 1. 事件键与股票并集
    events, sniper = {}, {}
    symbols: set[str] = set()
    for pool, lab in POOL_LAB.items():
        ev = pd.read_parquet(LAB_DIR / lab / "events.parquet",
                             columns=["ts_code", "date"])
        ev["date"] = pd.to_datetime(ev["date"])
        assert not ev.duplicated(["ts_code", "date"]).any(), f"{pool} 事件键不唯一"
        events[pool] = ev
        sniper[pool] = tep.load_div_labels(LAB_DIR / lab / "events.parquet",
                                           LAB_DIR / lab / "labels.parquet",
                                           lr.SNIPER_LABEL)
        symbols |= set(ev["ts_code"])
        log(f"[阶段1] {pool} 池事件 {len(ev)} 行, 狙击标签 {len(sniper[pool])} 行")
    symbols = sorted(symbols)
    missing_files = [s for s in symbols if not (DAILY_DIR / f"{s}.parquet").exists()]
    log(f"[阶段1] 股票并集 {len(symbols)} 只, 缺日线文件 {len(missing_files)} 只")

    # ---------------------------------------------------------- 2. 因果性断言（先断言后全量）
    trunc = _truncation_check(symbols)
    log(f"[阶段2] 因果性断言通过: 抽样 {trunc['sampled']} 只, "
        f"实测 {trunc['checked']} 只截头部 {trunc['head_rows']} 行重算逐位一致")

    # ---------------------------------------------------------- 3. 全量收益标签（按股票并行）
    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_labels_one_symbol, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                parts.append(r)
            done += 1
            if done % 500 == 0 or done == len(symbols):
                log(f"[阶段3] 收益标签进度 {done}/{len(symbols)}")
    ret_all = pd.concat(parts, ignore_index=True)
    assert not ret_all.duplicated(["ts_code", "date"]).any(), "收益标签键不唯一"
    log(f"[阶段3] 收益标签全量 {len(ret_all)} 行 x {ret_all.shape[1]} 列")

    # ---------------------------------------------------------- 4. 合并到各池事件
    results = {"issue": 26, "symbols": len(symbols),
               "missing_daily_files": missing_files, "truncation_check": trunc,
               "pools": {}}
    for pool in POOL_LAB:
        lab = events[pool].merge(ret_all, on=["ts_code", "date"], how="left",
                                 validate="one_to_one")
        lab = lab.merge(sniper[pool], on=["ts_code", "date"], how="left",
                        validate="one_to_one")
        assert lab.shape[1] == 2 + 19, f"{pool} 标签表列数异常: {lab.shape[1]}"
        key_miss = lab[lr.return_label_name("cur", 3)].isna() & \
            ~lab["ts_code"].isin(set(ret_all["ts_code"]))
        stats = {"n_events": int(len(lab)),
                 "key_missing_events": int(key_miss.sum()),
                 "label_nan_counts": {c: int(lab[c].isna().sum())
                                      for c in lr.candidate_labels()}}
        out = OUT_DIR / f"labels_race_{pool}.parquet"
        lab.to_parquet(out, index=False)
        results["pools"][pool] = stats
        log(f"[阶段4] {pool} 池标签表落盘 {out.name}: {len(lab)} 行, "
            f"键缺失 {stats['key_missing_events']} 事件")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with (OUT_DIR / "labels_build_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 耗时 {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
