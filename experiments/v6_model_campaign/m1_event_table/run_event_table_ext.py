#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1 事件表扩展与扫描器 —— 预登记 = 同目录 README.md(先于跑数落盘,勿改)。

实现口径:
- 扫描:逐字复刻 divergence_anchor_eval_2026/scan_v1.py 的单股口径(金叉三元组、
  DIF 抬升、区间最低价锚定、个股过滤),唯一改动 = 去掉 event_date 2026 窗口过滤
  (与 frozen 参照 run_seeds.py::scan_one_v1 一致),并在事件行上补落 README
  第三节预登记的 16 个新列。信号定义一个字不改。
- 产物:events_ext_v1.parquet(96,577 行 × 25 列 = 旧 9 列 + 新 16 列)。
- 验收:A(与 frozen events_history_v1.parquet 对账,共有列逐位一致)、
  B(截断历史重算逐位一致)、D(NaN 披露 + warm-up)、E(两次全量扫描逐位一致)。
  #42 §7 十一断言在 selfcheck_causality.py。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
FROZEN_EVENTS = Path(
    os.path.join(REPO, "experiments", "divergence_seed_trial_history",
                 "events_history_v1.parquet"))
OUT_DIR = Path(os.path.join(REPO, "experiments", "v6_model_campaign",
                            "m1_event_table"))
LOG_PATH = OUT_DIR / "progress.log"

MIN_ROWS = 100
DIF_LIFT_MIN = 0.001
SAMPLE_SEED = 20260909

OLD_COLS = ["ts_code", "event_date", "anchor_date", "anchor_close",
            "cross_prev_date", "cross_prev_dif", "cross_date", "cross_dif",
            "dif_lift"]
NEW_COLS = ["cross_prev2_row", "cross_prev2_date", "cross_prev2_dif",
            "cross_prev2_dea", "cross_prev_row", "cross_prev_dea",
            "event_row", "event_close", "event_vol", "event_amount",
            "cross_dea", "min_prev_close", "min_prev_row", "min_prev_date",
            "anchor_row", "anchor_bars"]
ALL_COLS = OLD_COLS + NEW_COLS

TRUNC_N_STOCKS = 20
TRUNC_FRACS = (0.4, 0.6, 0.8)
TRUNC_MIN_EVENTS = 5      # 合格池:事件数 >= 5
TRUNC_MIN_ROWS = 300      # 合格池:历史行数 >= 300


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 扫描(逐字复刻 + 加列)
def scan_df(df: pd.DataFrame):
    """scan_v1.py::scan_one 逐字口径(区间最低价锚定,去 2026 窗口),
    输入为已读入的 DataFrame;返回 (ts_code, events, skip_reason)。

    与口径源头的唯一差异:补落 16 个新列(全部为行号 <= j 的行派生量,
    不改变事件判定);amount 列缺失时 event_amount 记 NaN(README 披露 5,
    实测 0 个股文件触发)。
    """
    if "ts_code" not in df.columns or "vol" not in df.columns:
        return None, [], "schema"
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None, [], "short"

    close = df["close"].to_numpy(dtype=np.float64)
    dates = df["trade_date"].to_numpy()
    vol = df["vol"].to_numpy(dtype=np.float64)
    if "amount" in df.columns:
        amount = df["amount"].to_numpy(dtype=np.float64)
    else:
        amount = np.full(len(df), np.nan)
    dif, dea, _ = talib.MACD(close)  # 默认 12/26/9
    if np.isnan(dif).all():
        return None, [], "nan"

    # 金叉:DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]
    prev_le = (dif[:-1] <= dea[:-1]) & ~np.isnan(dif[:-1]) & ~np.isnan(dea[:-1])
    now_gt = (dif[1:] > dea[1:]) & ~np.isnan(dif[1:]) & ~np.isnan(dea[1:])
    crosses = np.nonzero(prev_le & now_gt)[0] + 1  # 行号(t)

    ts_code = str(df["ts_code"].iloc[0])
    events: list[dict] = []
    # 需要 C(k-2) 存在,即至少 3 次金叉,从第 3 个金叉起判定(k 从 2 起,0-based)
    for k in range(2, len(crosses)):
        c_km2, c_km1, c_k = crosses[k - 2], crosses[k - 1], crosses[k]
        dif_lift = dif[c_k] - dif[c_km1]
        if dif_lift < DIF_LIFT_MIN:
            continue
        # 左开右闭:(c_km2, c_km1] 与 (c_km1, c_k]
        seg_prev = close[c_km2 + 1 : c_km1 + 1]
        seg_cur = close[c_km1 + 1 : c_k + 1]
        if len(seg_prev) == 0 or len(seg_cur) == 0:
            continue
        idx_prev_rel = int(np.argmin(seg_prev))  # 并列取最早
        min_prev = seg_prev[idx_prev_rel]
        idx_cur_rel = int(np.argmin(seg_cur))    # 并列取最早
        min_cur = seg_cur[idx_cur_rel]
        if not (min_prev > min_cur):
            continue
        min_prev_idx = c_km2 + 1 + idx_prev_rel
        anchor_idx = c_km1 + 1 + idx_cur_rel
        events.append(
            {
                # ---- 旧 9 列(逐字口径,不改) ----
                "ts_code": ts_code,
                "event_date": pd.Timestamp(dates[c_k]),
                "anchor_date": pd.Timestamp(dates[anchor_idx]),
                "anchor_close": float(close[anchor_idx]),
                "cross_prev_date": pd.Timestamp(dates[c_km1]),
                "cross_prev_dif": float(dif[c_km1]),
                "cross_date": pd.Timestamp(dates[c_k]),
                "cross_dif": float(dif[c_k]),
                "dif_lift": float(dif_lift),
                # ---- 新 16 列(README 第三节预登记) ----
                "cross_prev2_row": int(c_km2),
                "cross_prev2_date": pd.Timestamp(dates[c_km2]),
                "cross_prev2_dif": float(dif[c_km2]),
                "cross_prev2_dea": float(dea[c_km2]),
                "cross_prev_row": int(c_km1),
                "cross_prev_dea": float(dea[c_km1]),
                "event_row": int(c_k),
                "event_close": float(close[c_k]),
                "event_vol": float(vol[c_k]),
                "event_amount": float(amount[c_k]),
                "cross_dea": float(dea[c_k]),
                "min_prev_close": float(min_prev),
                "min_prev_row": int(min_prev_idx),
                "min_prev_date": pd.Timestamp(dates[min_prev_idx]),
                "anchor_row": int(anchor_idx),
                "anchor_bars": int(c_k - anchor_idx),
            }
        )
    return ts_code, events, None


def scan_one(path: str):
    try:
        return scan_df(pd.read_parquet(path))
    except Exception as e:  # noqa: BLE001
        return None, [], f"error:{e}"


def run_scan(tag: str, files: list[str]) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    stats = {"schema": 0, "short": 0, "nan": 0, "error": 0}
    n_stocks = 0
    all_events: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, (ts_code, evs, reason) in enumerate(
            pool.imap_unordered(scan_one, files, chunksize=16)
        ):
            if reason:
                key = reason.split(":")[0]
                stats[key if key in stats else "error"] += 1
            if ts_code is not None:
                n_stocks += 1
            all_events.extend(evs)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: scan_{tag} {i + 1}/{len(files)} 文件 "
                    f"({time.time() - t0:.0f}s)")
    ev = pd.DataFrame(all_events, columns=ALL_COLS)
    if len(ev) > 0:
        ev = ev.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    log(f"[scan_{tag}] 个股 {n_stocks}/{len(files)},跳过 {stats},"
        f"全历史事件 {len(ev)} 起 ({time.time() - t0:.0f}s)")
    return ev, {"stats": stats, "n_stocks": n_stocks,
                "elapsed_sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- 逐位比较工具
def frames_bitwise_equal(a: pd.DataFrame, b: pd.DataFrame,
                         cols: list[str]) -> tuple[bool, float, str]:
    """逐位比较两表指定列。返回 (ok, 浮点最大绝对差, 失败列)。"""
    max_diff = 0.0
    for col in cols:
        xa, xb = a[col], b[col]
        if xa.dtype.kind == "f":
            va, vb = xa.to_numpy(), xb.to_numpy()
            if not np.array_equal(va, vb, equal_nan=True):
                d = np.abs(va - vb)
                d = d[np.isfinite(d)]
                max_diff = max(max_diff, float(d.max()) if len(d) else np.nan)
                return False, max_diff, col
        else:
            if not bool((xa == xb).all()):
                return False, max_diff, col
    return True, max_diff, ""


# ---------------------------------------------------------------- 验收 A:frozen 对账
def acceptance_a(ev: pd.DataFrame) -> dict:
    ref = pd.read_parquet(FROZEN_EVENTS)
    out: dict = {"n_new": int(len(ev)), "n_ref": int(len(ref))}
    key_new = ev[["ts_code", "event_date"]].drop_duplicates()
    key_ref = ref[["ts_code", "event_date"]].drop_duplicates()
    out["key_unique_new"] = bool(len(key_new) == len(ev))
    out["key_unique_ref"] = bool(len(key_ref) == len(ref))
    out["row_count_match"] = bool(len(ev) == len(ref) == 96577)
    # 行序逐行比较(双侧均按 event_date, ts_code 升序,键唯一故行序唯一确定)
    ref_sorted = ref.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    out["ref_sorted_row_order"] = bool((ref[["event_date", "ts_code"]].reset_index(drop=True)
                                        .equals(ref_sorted[["event_date", "ts_code"]])))
    ok_order, diff_order, col_order = frames_bitwise_equal(
        ev, ref_sorted, OLD_COLS)
    out["row_order_bitwise_ok"] = bool(ok_order)
    out["row_order_max_abs_float_diff"] = diff_order
    if not ok_order:
        out["row_order_fail_col"] = col_order
    # 按键对齐比较(稳健性,与行序无关)
    m = ev.merge(ref, on=["ts_code", "event_date"], how="outer", indicator=True)
    out["join_both"] = int((m["_merge"] == "both").sum())
    out["join_left_only"] = int((m["_merge"] == "left_only").sum())
    out["join_right_only"] = int((m["_merge"] == "right_only").sum())
    join_ok = out["join_left_only"] == 0 and out["join_right_only"] == 0
    out["join_key_match"] = bool(join_ok)
    max_diff = 0.0
    join_cols_ok = True
    fail_col = ""
    if join_ok:
        mb = m[m["_merge"] == "both"]
        for col in OLD_COLS:
            if col in ("ts_code", "event_date"):
                continue
            xa, xb = mb[f"{col}_x"], mb[f"{col}_y"]
            if xa.dtype.kind == "f":
                va, vb = xa.to_numpy(), xb.to_numpy()
                if not np.array_equal(va, vb, equal_nan=True):
                    d = np.abs(va - vb)
                    d = d[np.isfinite(d)]
                    max_diff = max(max_diff, float(d.max()) if len(d) else np.nan)
                    join_cols_ok = False
                    fail_col = col
            else:
                if not bool((xa == xb).all()):
                    join_cols_ok = False
                    fail_col = col
    out["join_cols_bitwise_ok"] = bool(join_cols_ok)
    out["join_max_abs_float_diff"] = max_diff
    if fail_col:
        out["join_fail_col"] = fail_col
    out["ok"] = bool(out["row_count_match"] and out["key_unique_new"]
                     and out["key_unique_ref"] and ok_order and join_ok
                     and join_cols_ok)
    log(f"[验收A] 行数 {out['n_new']} vs frozen {out['n_ref']};"
        f"键唯一(新/旧)={out['key_unique_new']}/{out['key_unique_ref']};"
        f"行序逐位={'PASS' if ok_order else 'FAIL'};"
        f"键对齐逐位={'PASS' if join_cols_ok else 'FAIL'};"
        f"浮点最大绝对差 {max(out['row_order_max_abs_float_diff'], max_diff):.2e}"
        f" -> {'PASS' if out['ok'] else 'FAIL'}")
    return out


# ---------------------------------------------------------------- 验收 B:截断历史重算
def _trunc_worker(task: dict) -> dict:
    ts_code = task["ts_code"]
    df = pd.read_parquet(task["path"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    full_events = task["full_events"]  # 该股全量表事件(list[dict])
    rows: list[dict] = []
    for t in task["trunc_rows"]:
        _, evs, reason = scan_df(df.iloc[: t + 1].copy())
        if reason is not None:
            rows.append({"ts_code": ts_code, "trunc_row": t, "ok": False,
                         "reason": f"trunc scan skipped: {reason}"})
            continue
        trunc_ev = pd.DataFrame(evs, columns=ALL_COLS)
        ref = pd.DataFrame([e for e in full_events if e["event_row"] <= t],
                           columns=ALL_COLS)
        if len(trunc_ev) != len(ref):
            rows.append({"ts_code": ts_code, "trunc_row": t, "ok": False,
                         "n_trunc": len(trunc_ev), "n_ref": len(ref),
                         "reason": "事件集合行数不一致"})
            continue
        if len(trunc_ev) == 0:
            rows.append({"ts_code": ts_code, "trunc_row": t, "ok": True,
                         "n_events": 0, "max_abs_float_diff": 0.0})
            continue
        trunc_ev = trunc_ev.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
        ref = ref.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
        ok, max_diff, fail_col = frames_bitwise_equal(trunc_ev, ref, ALL_COLS)
        rows.append({"ts_code": ts_code, "trunc_row": t, "ok": bool(ok),
                     "n_events": int(len(ref)),
                     "max_abs_float_diff": max_diff,
                     **({} if ok else {"fail_col": fail_col})})
    return {"ts_code": ts_code, "rows": rows}


def acceptance_b(ev: pd.DataFrame, files: list[str]) -> dict:
    t0 = time.time()
    path_by_code = {Path(f).stem: f for f in files}
    ev_counts = ev.groupby("ts_code").size()
    # 合格池:事件数 >= 5 且历史行数 >= 300
    eligible: list[str] = []
    nrows_cache: dict[str, int] = {}
    for code, cnt in ev_counts.items():
        if cnt < TRUNC_MIN_EVENTS or code not in path_by_code:
            continue
        nr = pq.read_metadata(path_by_code[code]).num_rows
        nrows_cache[code] = nr
        if nr >= TRUNC_MIN_ROWS:
            eligible.append(code)
    rng = np.random.default_rng(SAMPLE_SEED)
    picked = sorted(rng.choice(np.array(eligible, dtype=object),
                               size=TRUNC_N_STOCKS, replace=False).tolist())
    ev_by_code: dict[str, list[dict]] = {}
    for r in ev.itertuples(index=False):
        ev_by_code.setdefault(r.ts_code, []).append(r._asdict())
    tasks = []
    for code in picked:
        n = nrows_cache[code]
        trunc_rows = sorted({max(MIN_ROWS, int(round(n * f)))
                             for f in TRUNC_FRACS})
        trunc_rows = [t for t in trunc_rows if t < n]
        tasks.append({"ts_code": code, "path": path_by_code[code],
                      "trunc_rows": trunc_rows,
                      "full_events": ev_by_code[code]})
    log(f"[验收B] 合格池 {len(eligible)} 股,抽样 {len(picked)} 股"
        f"(种子 {SAMPLE_SEED}),截断行 {[t['trunc_rows'] for t in tasks][:3]}..."
        f"(前三股示例)")
    results: list[dict] = []
    with mp.Pool(processes=min(8, max(1, mp.cpu_count() - 1))) as pool:
        for i, res in enumerate(pool.imap_unordered(_trunc_worker, tasks)):
            results.append(res)
            log(f"heartbeat: 截断重算 {i + 1}/{len(tasks)} 股 "
                f"({time.time() - t0:.0f}s)")
    detail = [row for res in results for row in res["rows"]]
    n_ok = sum(1 for r in detail if r["ok"])
    n_events = sum(int(r.get("n_events", 0)) for r in detail if r["ok"])
    max_diff = max([float(r.get("max_abs_float_diff", 0.0))
                    for r in detail if r["ok"]] or [0.0])
    out = {"seed": SAMPLE_SEED, "n_stocks": len(picked),
           "n_truncs": len(detail), "n_ok": n_ok,
           "n_events_compared": n_events,
           "max_abs_float_diff": max_diff,
           "stocks": picked, "detail": detail,
           "ok": bool(n_ok == len(detail) and len(detail) >=
                      TRUNC_N_STOCKS * 3)}
    log(f"[验收B] 截断重算 {n_ok}/{len(detail)} 组通过,"
        f"对比事件 {n_events} 起,浮点最大绝对差 {max_diff:.2e}"
        f" -> {'PASS' if out['ok'] else 'FAIL'}")
    return out


# ---------------------------------------------------------------- 验收 D(扫描侧):NaN + warm-up
def acceptance_d_scan(ev: pd.DataFrame) -> dict:
    nan_ratio = {c: float(ev[c].isna().mean()) for c in NEW_COLS}
    nan_count = {c: int(ev[c].isna().sum()) for c in NEW_COLS}
    macd_cols = ["cross_prev2_dif", "cross_prev2_dea", "cross_prev_dif",
                 "cross_prev_dea", "cross_dif", "cross_dea", "dif_lift"]
    warmup_ok = bool(ev[macd_cols].notna().all().all())
    out = {"nan_ratio": nan_ratio, "nan_count": nan_count,
           "warmup_macd_cols_all_non_nan": warmup_ok,
           "ok": warmup_ok}
    log(f"[验收D-扫描侧] warm-up 段 MACD 列全非 NaN={warmup_ok};"
        f"新列 NaN 计数={ {c: v for c, v in nan_count.items() if v} or '全 0' }"
        f" -> {'PASS' if warmup_ok else 'FAIL'}")
    return out


def main() -> None:
    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("M1 EVENT TABLE EXT START | 预登记=README.md(冻结) | "
        "预计总时长约 10-15 分钟(扫描两次 ~4min×2,验收 ~2min)")

    files = sorted(str(p) for p in DATA_DIR.glob("*.parquet"))
    log(f"[init] 个股 parquet {len(files)} 个;CPU {mp.cpu_count()}")

    # ---------------- 阶段 1:全量扫描两次(验收 E 确定性) ----------------
    ev1, meta1 = run_scan("run1", files)
    ev2, meta2 = run_scan("run2", files)
    det_ok, det_diff, det_col = frames_bitwise_equal(ev1, ev2, ALL_COLS)
    det = {"ok": bool(det_ok and len(ev1) == len(ev2)),
           "n_run1": int(len(ev1)), "n_run2": int(len(ev2)),
           "max_abs_float_diff": det_diff}
    if not det_ok:
        det["fail_col"] = det_col
    log(f"[验收E] 两次全量扫描逐位一致={det_ok},"
        f"行数 {len(ev1)}/{len(ev2)},浮点最大绝对差 {det_diff:.2e}"
        f" -> {'PASS' if det['ok'] else 'FAIL'}")
    ev = ev1

    # ---------------- 阶段 2:落盘 ----------------
    ev.to_parquet(OUT_DIR / "events_ext_v1.parquet", index=False)
    log(f"[dump] events_ext_v1.parquet {len(ev)} 行 × {len(ev.columns)} 列")

    # ---------------- 阶段 3:验收 A/B/D ----------------
    res_a = acceptance_a(ev)
    res_b = acceptance_b(ev, files)
    res_d = acceptance_d_scan(ev)

    checks_all = res_a["ok"] and res_b["ok"] and res_d["ok"] and det["ok"]
    results = {
        "experiment": "M1 事件表扩展与扫描器",
        "prereadme": "README.md 先于跑数落盘(冻结)",
        "columns": {"old": OLD_COLS, "new": NEW_COLS, "all": ALL_COLS},
        "scan_meta": {"run1": meta1, "run2": meta2},
        "acceptance_A_frozen_reconcile": res_a,
        "acceptance_B_truncation": {k: v for k, v in res_b.items()
                                    if k != "detail"},
        "acceptance_B_detail": res_b["detail"],
        "acceptance_D_nan_warmup": res_d,
        "acceptance_E_determinism": det,
        "checks_all_pass": bool(checks_all),
        "duration_sec": round(time.time() - t_all, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_DIR / "m1_scan_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] m1_scan_results.json")
    log(f"M1 SCAN DONE ({time.time() - t_all:.0f}s) "
        f"扫描侧验收总评={'ALL PASS' if checks_all else 'HAS FAIL'}")


if __name__ == "__main__":
    main()
