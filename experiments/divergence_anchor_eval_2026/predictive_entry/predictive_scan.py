#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预判金叉前夜候选扫描器 + 三条数学不变量全市场验证（预登记见同目录 README.md）。

口径钉死（README 第 1~4 节）：
- talib.MACD(close)（12/26/9，不复权）+ 自复算内部 EMA12/EMA26（快 EMA 与慢 EMA 同在行号 25
  播种：快=SMA(close[14:26])、慢=SMA(close[0:26])；DEA 行号 33 以 DIF_int[25:34] SMA 播种）。
- A = EMA12×(11/13) − EMA26×(25/27)，B = 2/13 − 2/27，c* = (DEA_t − A)/B。
- 硬不变量（任一失败即停）：
  1. 真实金叉日 t+1：close[t+1] > c*；
  2. 非金叉日 t+1 且 DIF_t ≤ DEA_t：close[t+1] ≤ c* + 1e-9；
  3. |A + B·close[t+1] − talib DIF[t+1]| < 1e-9（含复算 DIF/DEA vs talib < 1e-9）。
- 前夜候选（eve t ∈ 2026-01-01..2026-08-31）：DIF_t ≤ DEA_t、t+1 在数据内、存在最近两金叉
  C(k−2)/C(k−1)、dif_lift_star = DEA_t − DIF[C(k−1)] ≥ 0.001，加变体价格条件（README 3/4 节）。

产物：candidates.parquet、predictive_scan.log。全程禁网络；仓库内除本目录外只读。
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import talib
from scipy.signal import lfilter

REPO = "/home/karl/repos/personal/stock_qt_nd"
DATA_DIR = os.path.join(REPO, "stock_data", "daily")
OUT_DIR = os.path.join(REPO, "experiments", "divergence_anchor_eval_2026", "predictive_entry")
LOG_PATH = os.path.join(OUT_DIR, "predictive_scan.log")

DATE_LO = pd.Timestamp("2026-01-01")
DATE_HI = pd.Timestamp("2026-08-31")
MIN_ROWS = 100
DIF_LIFT_MIN = 0.001
TOL = 1e-9

B = 2.0 / 13.0 - 2.0 / 27.0  # = 28/351 > 0


def macd_internal(close: np.ndarray):
    """复算 talib.MACD 内部 EMA12/EMA26/DIF/DEA（README 1.1 节种子口径）。"""
    n = len(close)
    if n < 34:
        return None
    e12 = np.full(n, np.nan)
    e26 = np.full(n, np.nan)
    e12[25] = close[14:26].mean()
    e26[25] = close[0:26].mean()
    t12, _ = lfilter([2.0 / 13.0], [1.0, -(11.0 / 13.0)], close[26:], zi=[(11.0 / 13.0) * e12[25]])
    t26, _ = lfilter([2.0 / 27.0], [1.0, -(25.0 / 27.0)], close[26:], zi=[(25.0 / 27.0) * e26[25]])
    e12[26:] = t12
    e26[26:] = t26
    dif_i = e12 - e26
    dea = np.full(n, np.nan)
    dea[33] = dif_i[25:34].mean()
    t9, _ = lfilter([0.2], [1.0, -0.8], dif_i[34:], zi=[0.8 * dea[33]])
    dea[34:] = t9
    dif_out = np.where(np.arange(n) >= 33, dif_i, np.nan)
    return e12, e26, dif_out, dea


def confirmed_lows_causal(close: np.ndarray, lo: int, hi: int, t_max: int):
    """(lo, hi] 内（行号 lo+1..hi）eve 日 t_max 可知的确认局部低点（左3右2，s+2<=t_max）。

    沿用 scan_v2 的贪心最小间隔过滤（相邻保留低点间隔 >= 3，保留先出现者）。
    """
    cands = []
    for s in range(lo + 1, hi + 1):
        if s - 3 < 0 or s + 2 > t_max:
            continue
        if close[s] <= close[s - 3:s].min() and close[s] < close[s + 1:s + 3].min():
            cands.append(s)
    kept = []
    for s in cands:
        if not kept or s - kept[-1] >= 3:
            kept.append(s)
    return kept


def anchor_causal(close: np.ndarray, lo: int, hi: int, t_max: int):
    """scan_v2 find_anchor 的因果截断版：窗口 (lo, hi]（调用方已截 hi）、确认日 <= t_max。"""
    if hi <= lo:
        return None
    lows = confirmed_lows_causal(close, lo, hi, t_max)
    if lows:
        best = lows[0]
        for s in lows[1:]:
            if close[s] < close[best]:
                best = s
        return best
    seg = close[lo + 1:hi + 1]
    return lo + 1 + int(np.argmin(seg))  # 并列取最早


def process_stock(path: str):
    """返回 (ts_code|None, candidates, inv_counts dict, calibration dict|None, err|None)。"""
    try:
        df = pd.read_parquet(path)
        cols = set(df.columns)
        if "ts_code" not in cols or "vol" not in cols:
            return None, [], None, None, None
        if len(df) < MIN_ROWS:
            return None, [], None, None, None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        close = df["close"].to_numpy(dtype=np.float64)
        dates = df["trade_date"].to_numpy()  # datetime64[ns]
        n = len(df)
        ts_code = str(df["ts_code"].iloc[0])

        dif_t, dea_t, _ = talib.MACD(close)
        if np.isnan(dif_t).all():
            return None, [], None, None, None
        internal = macd_internal(close)
        if internal is None:
            return None, [], None, None, None
        e12, e26, dif_i, dea_i = internal

        valid = ~(np.isnan(dif_t) | np.isnan(dea_t))
        # 复算 vs talib（不变量 3 的一部分）
        d_dif = np.abs(dif_i[valid] - dif_t[valid]).max() if valid.any() else 0.0
        d_dea = np.abs(dea_i[valid] - dea_t[valid]).max() if valid.any() else 0.0
        if not (d_dif < TOL and d_dea < TOL):
            return None, [], None, None, (
                f"{ts_code}: internal replication mismatch d_dif={d_dif} d_dea={d_dea}")

        A = e12 * (11.0 / 13.0) - e26 * (25.0 / 27.0)

        # ---------------- 不变量 1/2/3（全历史逐日） ----------------
        n1 = f1 = n2 = f2 = n3 = f3 = 0
        cross = np.zeros(n, dtype=bool)
        cross[1:] = (valid[:-1] & valid[1:]
                     & (dif_t[:-1] <= dea_t[:-1])
                     & (dif_t[1:] > dea_t[1:]))
        cstar_prev = (dea_t[:-1] - A[:-1]) / B  # 用 t 日状态算的 c*（对 t+1 日）
        prev_state_ok = valid[:-1] & (dif_t[:-1] <= dea_t[:-1]) & ~np.isnan(A[:-1])

        idx1 = np.flatnonzero(cross[1:] & prev_state_ok)
        n1 = len(idx1)
        f1 = int((close[idx1 + 1] <= cstar_prev[idx1]).sum())
        idx2 = np.flatnonzero(~cross[1:] & prev_state_ok & valid[1:])
        n2 = len(idx2)
        f2 = int((close[idx2 + 1] > cstar_prev[idx2] + TOL).sum())
        idx3 = np.flatnonzero(~np.isnan(A[:-1]) & valid[1:])
        n3 = len(idx3)
        dif_lin = A[idx3] + B * close[idx3 + 1]
        f3 = int((np.abs(dif_lin - dif_t[idx3 + 1]) >= TOL).sum())
        inv = dict(n1=n1, f1=f1, n2=n2, f2=f2, n3=n3, f3=f3)
        if f1 or f2 or f3:
            return None, [], inv, None, (
                f"{ts_code}: INVARIANT FAIL f1={f1}/{n1} f2={f2}/{n2} f3={f3}/{n3}")

        crosses = np.flatnonzero(cross)

        # ---------------- 校准抽查字段（600283.SH eve 2026-07-28） ----------------
        calib = None
        if ts_code == "600283.SH":
            hit = np.flatnonzero(dates == np.datetime64("2026-07-28"))
            if len(hit):
                i = int(hit[0])
                calib = dict(
                    eve_date="2026-07-28", c_star=float((dea_t[i] - A[i]) / B),
                    dif_star=float(dea_t[i]), close_t=float(close[i]),
                    margin_ratio=float((dea_t[i] - A[i]) / B / close[i] - 1.0),
                    next_date=str(pd.Timestamp(dates[i + 1]).date()) if i + 1 < n else None,
                    next_close=float(close[i + 1]) if i + 1 < n else None,
                    next_dif=float(dif_t[i + 1]) if i + 1 < n else None,
                    next_dea=float(dea_t[i + 1]) if i + 1 < n else None,
                    next_is_cross=bool(cross[i + 1]) if i + 1 < n else None,
                )

        # ---------------- 前夜候选枚举 ----------------
        cands = []
        # eve 行号范围：date 落在窗口内
        in_win = (dates >= np.datetime64(DATE_LO)) & (dates <= np.datetime64(DATE_HI))
        for t in np.flatnonzero(in_win):
            if t + 1 > n - 1:
                continue
            if not (valid[t] and dif_t[t] <= dea_t[t]) or np.isnan(A[t]):
                continue
            prev = crosses[crosses <= t]
            if len(prev) < 2:
                continue
            c1, c2 = int(prev[-1]), int(prev[-2])  # C(k-1), C(k-2)
            dif_lift_star = dea_t[t] - dif_t[c1]
            if dif_lift_star < DIF_LIFT_MIN:
                continue
            c_star = (dea_t[t] - A[t]) / B
            base = dict(
                ts_code=ts_code,
                eve_date=pd.Timestamp(dates[t]),
                target_date=pd.Timestamp(dates[t + 1]),
                eve_close=float(close[t]),
                c_star=float(c_star),
                margin_ratio=float(c_star / close[t] - 1.0),
                dif_star=float(dea_t[t]),
                dif_lift_star=float(dif_lift_star),
                cross_prev1_date=pd.Timestamp(dates[c1]),
                cross_prev1_dif=float(dif_t[c1]),
                cross_prev2_date=pd.Timestamp(dates[c2]),
            )
            # 变体1：min close (c1, t] < min close (c2, c1]
            min_prev = close[c2 + 1:c1 + 1].min()
            min_cur = close[c1 + 1:t + 1].min()
            if min_cur < min_prev:
                r = dict(base)
                r.update(variant="v1", v1_min_prev=float(min_prev), v1_min_cur=float(min_cur))
                cands.append(r)
            # 变体2：provisional 锚（窗口 (c1, t]，确认至 t-2）< anchor(k-1)（因果截断版）
            a_prev = anchor_causal(close, c2, min(c1 + 5, t), t)
            a_cur = anchor_causal(close, c1, t, t)
            if a_prev is not None and a_cur is not None and close[a_cur] < close[a_prev]:
                r = dict(base)
                r.update(variant="v2",
                         prov_anchor_date=pd.Timestamp(dates[a_cur]),
                         prov_anchor_close=float(close[a_cur]),
                         anchor_prev_date=pd.Timestamp(dates[a_prev]),
                         anchor_prev_close=float(close[a_prev]))
                cands.append(r)
        return ts_code, cands, inv, calib, None
    except Exception:
        return None, [], None, None, traceback.format_exc()


def worker(path):
    return process_stock(path)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"PREDICTIVE SCAN START window=[{DATE_LO.date()}..{DATE_HI.date()}]")
    files = sorted(
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")
    )
    log(f"文件总数: {len(files)}")

    all_cands = []
    calib = None
    inv_tot = dict(n1=0, f1=0, n2=0, f2=0, n3=0, f3=0)
    n_stock = n_skip = n_err = 0
    errs = []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for i, (ts_code, cands, inv, cal, err) in enumerate(
                pool.imap_unordered(worker, files, chunksize=32)):
            if err is not None:
                n_err += 1
                errs.append(err)
            elif ts_code is None:
                n_skip += 1
            else:
                n_stock += 1
                all_cands.extend(cands)
                if inv:
                    for k in inv_tot:
                        inv_tot[k] += inv[k]
                if cal is not None:
                    calib = cal
            if (i + 1) % 500 == 0:
                log(f"heartbeat: {i + 1}/{len(files)} files, candidates={len(all_cands)}, "
                    f"{time.time() - t0:.0f}s")

    log(f"扫描完成: 个股 {n_stock}, 跳过 {n_skip}, 异常 {n_err}, 候选 {len(all_cands)}, "
        f"耗时 {time.time() - t0:.0f}s")
    for e in errs[:5]:
        log("异常样例:\n" + e)

    # ---------------- 不变量总账（不过即停） ----------------
    log(f"不变量1(真实金叉日 close>c*): 验证 {inv_tot['n1']} 条, 失败 {inv_tot['f1']}")
    log(f"不变量2(非金叉日且DIF<=DEA close<=c*+1e-9): 验证 {inv_tot['n2']} 条, 失败 {inv_tot['f2']}")
    log(f"不变量3(线性重算DIF vs talib <1e-9): 验证 {inv_tot['n3']} 条, 失败 {inv_tot['f3']}")

    df = pd.DataFrame(all_cands)
    if len(df):
        df = df.sort_values(["eve_date", "ts_code", "variant"]).reset_index(drop=True)
    pq_path = os.path.join(OUT_DIR, "candidates.parquet")
    df.to_parquet(pq_path, index=False)
    log(f"candidates.parquet rows={len(df)} -> {pq_path}")
    if len(df):
        log("候选按变体: " + str(df["variant"].value_counts().to_dict()))
        log("候选按月: " + str(df["eve_date"].dt.strftime("%Y-%m").value_counts()
                                 .sort_index().to_dict()))

    log(f"校准抽查 600283.SH: {json.dumps(calib, ensure_ascii=False)}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    if n_err or inv_tot["f1"] or inv_tot["f2"] or inv_tot["f3"]:
        log("HARD FAIL: 存在异常或不变量失败，停止")
        sys.exit(1)
    log(f"PREDICTIVE SCAN DONE ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
