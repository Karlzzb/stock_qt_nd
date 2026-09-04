#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""金叉对金叉 MACD 底背离扫描器 —— 变体 2（右侧确认低点精确锚定版）。

口径（钉死）：
- DIF, DEA = talib.MACD(close) 前两个返回值（12/26/9），个股全历史一次性计算，不复权 close。
- 金叉日 t：DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]（两者均非 NaN）。
- 对每个金叉 C(k)（k>=2），低点搜索窗口 W(k) = (C(k-1), C(k)+5个交易日]（左开右闭，按行号，末尾截断）。
- 窗口内确认局部低点（左3右2，按行号）：close[t] <= min(close[t-3..t-1]) 且 close[t] < min(close[t+1..t+2])；
  通过候选按时间贪心做最小间隔过滤（相邻保留低点间隔 >= 3 个交易日，保留先出现者）。
- anchor(k) = W(k) 内收盘价最低的确认局部低点；无确认低点则回退为 W(k) 内最低收盘价所在日（并列取最早）。
- 对相邻金叉对 (C(k-1), C(k))（需存在 C(k-2)，即至少 3 次金叉）：
    1) dif_lift = DIF[C(k)] - DIF[C(k-1)] >= 0.001
    2) close[anchor(k)] < close[anchor(k-1)]
  两条件满足则事件：event_date = max(C(k), anchor(k)+2交易日)（anchor+2 超出末尾则丢弃）。
只保留 event_date ∈ [2026-01-01, 2026-08-31]。
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

REPO = "/home/karl/repos/personal/stock_qt_nd"
DATA_DIR = os.path.join(REPO, "stock_data", "daily")
OUT_DIR = os.path.join(REPO, "experiments", "divergence_anchor_eval_2026")

DATE_LO = pd.Timestamp("2026-01-01")
DATE_HI = pd.Timestamp("2026-08-31")

DIF_LIFT_MIN = 0.001
MIN_ROWS = 100


def process_stock(path):
    """返回 (ts_code 或 None, events list, err 或 None)。"""
    try:
        df = pd.read_parquet(path)
        cols = set(df.columns)
        if "ts_code" not in cols or "vol" not in cols:
            return None, [], None  # 指数等非个股文件，静默跳过
        if len(df) < MIN_ROWS:
            return None, [], None
        df = df.sort_values("trade_date").reset_index(drop=True)
        close = df["close"].to_numpy(dtype=np.float64)
        dates = df["trade_date"].to_numpy()
        n = len(df)
        ts_code = str(df["ts_code"].iloc[0])

        dif, dea, _ = talib.MACD(close)
        # 金叉：DIF 上穿 DEA
        valid = ~(np.isnan(dif) | np.isnan(dea))
        cross_mask = np.zeros(n, dtype=bool)
        cross_mask[1:] = (
            valid[:-1] & valid[1:]
            & (dif[:-1] <= dea[:-1])
            & (dif[1:] > dea[1:])
        )
        crosses = np.flatnonzero(cross_mask)
        if len(crosses) < 3:
            return ts_code, [], None

        def confirmed_lows(lo_idx, hi_idx):
            """窗口 (lo_idx, hi_idx] 内（按行号闭区间 lo_idx+1..hi_idx）的确认局部低点。"""
            cands = []
            for t in range(lo_idx + 1, hi_idx + 1):
                if t - 3 < 0 or t + 2 > n - 1:
                    continue
                if close[t] <= close[t - 3:t].min() and close[t] < close[t + 1:t + 3].min():
                    cands.append(t)
            kept = []
            for t in cands:
                if not kept or t - kept[-1] >= 3:
                    kept.append(t)
            return kept

        def find_anchor(k):
            """W(k) = (C(k-1), C(k)+5] 的锚定低点行号。"""
            lo = crosses[k - 1]
            hi = min(crosses[k] + 5, n - 1)
            if hi <= lo:
                return None
            lows = confirmed_lows(lo, hi)
            if lows:
                # 收盘价最低的确认低点；并列取最早
                best = lows[0]
                for t in lows[1:]:
                    if close[t] < close[best]:
                        best = t
                return best
            seg = close[lo + 1:hi + 1]
            return lo + 1 + int(np.argmin(seg))  # argmin 并列取最早

        events = []
        anchors = {}
        for k in range(1, len(crosses)):
            anchors[k] = find_anchor(k)
        for k in range(2, len(crosses)):  # 需存在 C(k-2)
            a_prev, a_cur = anchors.get(k - 1), anchors.get(k)
            if a_prev is None or a_cur is None:
                continue
            dif_lift = dif[crosses[k]] - dif[crosses[k - 1]]
            if dif_lift < DIF_LIFT_MIN:
                continue
            if not (close[a_cur] < close[a_prev]):
                continue
            if a_cur + 2 > n - 1:
                continue  # anchor+2 超出数据末尾，丢弃
            event_idx = max(crosses[k], a_cur + 2)
            events.append({
                "ts_code": ts_code,
                "event_date": pd.Timestamp(dates[event_idx]),
                "anchor_date": pd.Timestamp(dates[a_cur]),
                "anchor_close": float(close[a_cur]),
                "cross_prev_date": pd.Timestamp(dates[crosses[k - 1]]),
                "cross_prev_dif": float(dif[crosses[k - 1]]),
                "cross_date": pd.Timestamp(dates[crosses[k]]),
                "cross_dif": float(dif[crosses[k]]),
                "dif_lift": float(dif_lift),
                "anchor_prev_date": pd.Timestamp(dates[a_prev]),
                "anchor_prev_close": float(close[a_prev]),
            })
        return ts_code, events, None
    except Exception:
        return None, [], traceback.format_exc()


def worker(path):
    return process_stock(path)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "events_v2.log")
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    files = sorted(
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")
    )
    log(f"文件总数: {len(files)}")

    all_events = []
    n_stock, n_skip, n_err = 0, 0, 0
    errs = []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for ts_code, events, err in pool.imap_unordered(worker, files, chunksize=32):
            if err is not None:
                n_err += 1
                errs.append(err)
            elif ts_code is None:
                n_skip += 1
            else:
                n_stock += 1
                all_events.extend(events)

    elapsed_scan = time.time() - t0
    ev = pd.DataFrame(all_events)
    if len(ev):
        ev = ev.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    log(f"扫描完成: 个股 {n_stock} 只, 跳过(指数/行数不足) {n_skip} 个, 异常 {n_err} 个, "
        f"原始事件 {len(ev)} 起, 扫描耗时 {elapsed_scan:.1f}s")
    for e in errs[:5]:
        log("异常样例:\n" + e)

    n_all = len(ev)
    if len(ev):
        ev = ev[(ev["event_date"] >= DATE_LO) & (ev["event_date"] <= DATE_HI)].reset_index(drop=True)
    log(f"2026-01-01..2026-08-31 窗口过滤: {n_all} -> {len(ev)}")

    pq_path = os.path.join(OUT_DIR, "events_v2.parquet")
    ev.to_parquet(pq_path, index=False)

    # ---------- 校准硬断言 ----------
    checks = []

    def get(code, cross_date):
        m = ev[(ev["ts_code"] == code) & (ev["cross_date"] == pd.Timestamp(cross_date))]
        return m.iloc[0] if len(m) else None

    r = get("600283.SH", "2026-07-29")
    ok = r is not None and r["anchor_date"] == pd.Timestamp("2026-07-14") and abs(r["anchor_close"] - 7.57) < 0.02
    checks.append(("断言1: 600283.SH cross=2026-07-29 anchor=2026-07-14 close≈7.57", ok,
                   None if r is None else dict(r)))

    r = get("002230.SZ", "2026-07-31")
    ok = r is not None and r["anchor_date"] == pd.Timestamp("2026-07-24") and abs(r["anchor_close"] - 38.88) < 0.05
    checks.append(("断言2: 002230.SZ cross=2026-07-31 anchor=2026-07-24 close≈38.88", ok,
                   None if r is None else dict(r)))

    r = get("601212.SH", "2026-07-14")
    ok = (r is not None and r["anchor_date"] == pd.Timestamp("2026-07-17")
          and abs(r["anchor_close"] - 4.47) < 0.02
          and r["event_date"] == pd.Timestamp("2026-07-21"))
    checks.append(("断言3: 601212.SH cross=2026-07-14 anchor=2026-07-17 close≈4.47 event=2026-07-21", ok,
                   None if r is None else dict(r)))

    # 阴性断言：指定金叉对不得产生事件（cross_prev_date -> cross_date）
    def pair_absent(code, prev_date, cur_date):
        m = ev[(ev["ts_code"] == code)
               & (ev["cross_prev_date"] == pd.Timestamp(prev_date))
               & (ev["cross_date"] == pd.Timestamp(cur_date))]
        return len(m) == 0

    checks.append(("断言4a(阴性): 600283.SH 对(2026-04-15→2026-07-01) 无事件",
                   pair_absent("600283.SH", "2026-04-15", "2026-07-01"), None))
    checks.append(("断言4b(阴性): 002230.SZ 对(2026-06-24→2026-07-02) 无事件",
                   pair_absent("002230.SZ", "2026-06-24", "2026-07-02"), None))

    n_pass = 0
    for name, ok, detail in checks:
        log(f"{'PASS' if ok else 'FAIL'}  {name}")
        if detail is not None:
            log(f"      实际: {detail}")
        n_pass += int(ok)
    log(f"校准断言: {n_pass}/{len(checks)} 通过")

    elapsed = time.time() - t0
    monthly = {}
    if len(ev):
        monthly = ev["event_date"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    summary = {
        "total_events": int(len(ev)),
        "n_stocks_with_events": int(ev["ts_code"].nunique()) if len(ev) else 0,
        "monthly_events": {k: int(v) for k, v in monthly.items()},
        "n_stocks_scanned": n_stock,
        "n_files_skipped": n_skip,
        "n_errors": n_err,
        "elapsed_sec": round(elapsed, 1),
        "calibration_pass": f"{n_pass}/{len(checks)}",
    }
    with open(os.path.join(OUT_DIR, "events_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"总耗时 {elapsed:.1f}s；输出: {pq_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    if n_pass != len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
