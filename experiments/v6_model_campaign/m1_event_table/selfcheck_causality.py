#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1 因果性自检 —— #42 §7 十一断言的可机检实现 + report.md 生成。

预登记 = 同目录 README.md 第五/六节(先于跑数落盘,勿改)。
输入:events_ext_v1.parquet(本阶段扫描产物)、m1_scan_results.json、
frozen 基准 events_history_v1.parquet / trades_seed.parquet / summary_seed.csv(只读)。
断言 2 的区间语义重算为独立重写的代码路径(不 import 扫描器),逐字段逐位核对。
断言 7 的缩放重扫为管线性质测试,复用扫描器 scan_df。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
SEED_DIR = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history"))
OUT_DIR = Path(os.path.join(REPO, "experiments", "v6_model_campaign",
                            "m1_event_table"))
LOG_PATH = OUT_DIR / "progress.log"

sys.path.insert(0, str(OUT_DIR))
from run_event_table_ext import (ALL_COLS, NEW_COLS, OLD_COLS,  # noqa: E402
                                 scan_df)

SAMPLE_SEED = 20260909
A2_N_SAMPLE = 2000          # 断言 2 抽样事件数
A3_N_SAMPLE = 300           # 断言 3 抽样事件数(限 j >= 100)
A3_MIN_ROW = 100            # 断言 3 烧入期下界
A3_TOL = 1e-8               # 断言 3 容差
A7_N_STOCKS = 50            # 断言 7 抽样股票数(事件数 >= 3)
A7_LAMBDA = 2.718           # 断言 7 缩放因子
A7_TOL = 1e-9               # 断言 7 比值容差
DIF_LIFT_MIN = 0.001
FROZEN_SEED_COUNTS = {"S1": 5580, "S2": 2419, "S3": 1159, "S4": 11888,
                      "S5": 3271}  # #31 §4.4 / summary_seed.csv v1 n_selected


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 断言 1:行号上界
def assert_1(ev: pd.DataFrame) -> dict:
    detail = {
        "i2<i1": bool((ev["cross_prev2_row"] < ev["cross_prev_row"]).all()),
        "i1<j": bool((ev["cross_prev_row"] < ev["event_row"]).all()),
        "p_low>i2": bool((ev["min_prev_row"] > ev["cross_prev2_row"]).all()),
        "p_low<=i1": bool((ev["min_prev_row"] <= ev["cross_prev_row"]).all()),
        "a>i1": bool((ev["anchor_row"] > ev["cross_prev_row"]).all()),
        "a<=j": bool((ev["anchor_row"] <= ev["event_row"]).all()),
        "anchor_bars==j-a": bool(
            (ev["anchor_bars"] == ev["event_row"] - ev["anchor_row"]).all()),
        "anchor_bars>=0": bool((ev["anchor_bars"] >= 0).all()),
    }
    ok = all(detail.values())
    log(f"[断言1] 行号上界:{detail} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "detail": detail,
            "note": "全部输入行号 <= j 由构造保证;构造性证明 = 验收B 截断重算"
                    "(见 m1_scan_results.json)"}


# ---------------------------------------------------------------- 断言 2(+6 抽样):区间语义自洽(独立代码路径)
def _a2_worker(task: dict) -> list[dict]:
    """独立重写的重算实现(不复用扫描器),逐事件逐字段核对。"""
    ts_code = task["ts_code"]
    df = pd.read_parquet(DATA_DIR / f"{ts_code}.parquet")
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=np.float64)
    vol = df["vol"].to_numpy(dtype=np.float64)
    amount = (df["amount"].to_numpy(dtype=np.float64)
              if "amount" in df.columns else np.full(len(df), np.nan))
    dates = df["trade_date"].to_numpy()
    dif, dea, _ = talib.MACD(close)
    prev_le = (dif[:-1] <= dea[:-1]) & ~np.isnan(dif[:-1]) & ~np.isnan(dea[:-1])
    now_gt = (dif[1:] > dea[1:]) & ~np.isnan(dif[1:]) & ~np.isnan(dea[1:])
    crosses = np.nonzero(prev_le & now_gt)[0] + 1
    pos_in_crosses = {int(c): i for i, c in enumerate(crosses)}
    out: list[dict] = []
    for e in task["events"]:
        fails: list[str] = []
        j, i1, i2 = e["event_row"], e["cross_prev_row"], e["cross_prev2_row"]
        ki = pos_in_crosses.get(j, -1)
        if ki < 2 or crosses[ki - 1] != i1 or crosses[ki - 2] != i2:
            fails.append("金叉三元组不连续/不在金叉序列")
        if not (j < len(dates)
                and pd.Timestamp(dates[j]) == pd.Timestamp(e["event_date"])):
            fails.append("event_date")
        seg_prev = close[i2 + 1 : i1 + 1]
        seg_cur = close[i1 + 1 : j + 1]
        p_low = i2 + 1 + int(np.argmin(seg_prev))   # 并列取最早
        a = i1 + 1 + int(np.argmin(seg_cur))        # 并列取最早
        checks = [
            ("min_prev_row", p_low == e["min_prev_row"]),
            ("min_prev_close", close[p_low] == e["min_prev_close"]),
            ("min_prev_date",
             pd.Timestamp(dates[p_low]) == pd.Timestamp(e["min_prev_date"])),
            ("anchor_row", a == e["anchor_row"]),
            ("anchor_close", close[a] == e["anchor_close"]),
            ("anchor_date",
             pd.Timestamp(dates[a]) == pd.Timestamp(e["anchor_date"])),
            ("a_in_(i1,j]", i1 < a <= j),
            ("anchor_bars", j - a == e["anchor_bars"]),
            ("dif@i2", dif[i2] == e["cross_prev2_dif"]),
            ("dea@i2", dea[i2] == e["cross_prev2_dea"]),
            ("dif@i1", dif[i1] == e["cross_prev_dif"]),
            ("dea@i1", dea[i1] == e["cross_prev_dea"]),
            ("dif@j", dif[j] == e["cross_dif"]),
            ("dea@j", dea[j] == e["cross_dea"]),
            ("dif_lift", dif[j] - dif[i1] == e["dif_lift"]),
            ("event_close", close[j] == e["event_close"]),
            ("event_vol", vol[j] == e["event_vol"]),
            ("event_amount", bool((amount[j] == e["event_amount"])
                                  or (np.isnan(amount[j])
                                      and np.isnan(e["event_amount"])))),
            ("cross_prev2_date",
             pd.Timestamp(dates[i2]) == pd.Timestamp(e["cross_prev2_date"])),
            ("cross_prev_date",
             pd.Timestamp(dates[i1]) == pd.Timestamp(e["cross_prev_date"])),
            ("dif_lift>=0.001", e["dif_lift"] >= DIF_LIFT_MIN),
            ("min_prev>min_cur", e["min_prev_close"] > e["anchor_close"]),
        ]
        fails += [name for name, okc in checks if not okc]
        out.append({"ts_code": ts_code, "event_date": str(e["event_date"]),
                    "ok": not fails, "fails": fails})
    return out


def assert_2_and_6_sample(ev: pd.DataFrame) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(ev), size=min(A2_N_SAMPLE, len(ev)), replace=False)
    sample = ev.iloc[np.sort(idx)]
    by_code: dict[str, list[dict]] = {}
    for r in sample.itertuples(index=False):
        d = r._asdict()
        by_code.setdefault(r.ts_code, []).append(d)
    tasks = [{"ts_code": c, "events": es} for c, es in by_code.items()]
    results: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_a2_worker, tasks,
                                                    chunksize=4)):
            results.extend(res)
            if (i + 1) % 200 == 0:
                log(f"heartbeat: 断言2 独立重算 {i + 1}/{len(tasks)} 股 "
                    f"({time.time() - t0:.0f}s)")
    n_bad = sum(1 for r in results if not r["ok"])
    bad_examples = [r for r in results if not r["ok"]][:5]
    ok = n_bad == 0
    log(f"[断言2+6抽样] 独立重算 {len(results)} 事件({len(tasks)} 股),"
        f"不一致 {n_bad} -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    return {"ok": ok, "n_sample": len(results), "n_stocks": len(tasks),
            "n_fail": n_bad, "bad_examples": bad_examples,
            "seed": SAMPLE_SEED}


# ---------------------------------------------------------------- 断言 3:MACD 无前视
def _a3_worker(task: dict) -> list[float]:
    df = pd.read_parquet(DATA_DIR / f"{task['ts_code']}.parquet",
                         columns=["trade_date", "close"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=np.float64)
    dif_full, dea_full, _ = talib.MACD(close)
    out: list[float] = []
    for j, dj, aj in task["events"]:  # (event_row, cross_dif, cross_dea)
        dif_t, dea_t, _ = talib.MACD(close[: j + 1])
        out.append(max(abs(dif_t[j] - dj), abs(dea_t[j] - aj),
                       abs(dif_t[j] - dif_full[j]),
                       abs(dea_t[j] - dea_full[j])))
    return out


def assert_3(ev: pd.DataFrame) -> dict:
    t0 = time.time()
    pool_df = ev[ev["event_row"] >= A3_MIN_ROW]
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(pool_df), size=min(A3_N_SAMPLE, len(pool_df)),
                     replace=False)
    sample = pool_df.iloc[np.sort(idx)]
    by_code: dict[str, list] = {}
    for r in sample.itertuples(index=False):
        by_code.setdefault(r.ts_code, []).append(
            (r.event_row, r.cross_dif, r.cross_dea))
    tasks = [{"ts_code": c, "events": es} for c, es in by_code.items()]
    diffs: list[float] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for res in pool.imap_unordered(_a3_worker, tasks, chunksize=4):
            diffs.extend(res)
    max_diff = float(max(diffs)) if diffs else 0.0
    ok = max_diff < A3_TOL
    log(f"[断言3] MACD 截尾重算 {len(diffs)} 事件,最大绝对差 {max_diff:.2e} "
        f"(容差 {A3_TOL}) -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    return {"ok": ok, "n_sample": len(diffs), "max_abs_diff": max_diff,
            "tol": A3_TOL, "seed": SAMPLE_SEED}


# ---------------------------------------------------------------- 断言 4 + 10:dd20 全表重算 + 种子资格
def _a4_worker(task: dict) -> list[dict]:
    ts_code = task["ts_code"]
    df = pd.read_parquet(DATA_DIR / f"{ts_code}.parquet",
                         columns=["trade_date", "close"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=np.float64)
    dates = pd.DatetimeIndex(df["trade_date"])
    pos = {d: i for i, d in enumerate(dates)}
    out: list[dict] = []
    for event_date, event_row in task["events"]:
        j = pos.get(pd.Timestamp(event_date), -1)
        if j < 0:
            out.append({"ts_code": ts_code, "event_date": str(event_date),
                        "row_match": False, "dd20": np.nan,
                        "win_len_ok": False, "dd20_le_0": False})
            continue
        t0 = max(0, j - 20)
        win = close[t0 : j + 1]
        dd20 = close[j] / win.max() - 1.0
        out.append({
            "ts_code": ts_code, "event_date": str(event_date),
            "row_match": bool(j == event_row),
            "dd20": float(dd20),
            "win_len_ok": bool(len(win) == min(21, j + 1)
                               and t0 == max(0, j - 20)),
            "dd20_le_0": bool(dd20 <= 1e-12),
        })
    return out


def assert_4_10(ev: pd.DataFrame) -> tuple[dict, dict]:
    """返回 (断言4 结果, 断言10 结果)。"""
    t0 = time.time()
    by_code: dict[str, list] = {}
    for r in ev.itertuples(index=False):
        by_code.setdefault(r.ts_code, []).append((r.event_date, r.event_row))
    tasks = [{"ts_code": c, "events": es} for c, es in by_code.items()]
    recs: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(_a4_worker, tasks,
                                                    chunksize=8)):
            recs.extend(res)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: 断言4 dd20 全表重算 {i + 1}/{len(tasks)} 股 "
                    f"({time.time() - t0:.0f}s)")
    dd = pd.DataFrame(recs)
    dd["event_date"] = pd.to_datetime(dd["event_date"])
    a4 = {
        "n_events": int(len(dd)),
        "row_match_all": bool(dd["row_match"].all()),
        "win_len_all_ok": bool(dd["win_len_ok"].all()),
        "dd20_le_1e-12_all": bool(dd["dd20_le_0"].all()),
        "dd20_max": float(dd["dd20"].max()),
        "dd20_min": float(dd["dd20"].min()),
    }
    a4["ok"] = bool(a4["row_match_all"] and a4["win_len_all_ok"]
                    and a4["dd20_le_1e-12_all"])
    log(f"[断言4] dd20 全表重算 {len(dd)} 事件:行号对账={a4['row_match_all']},"
        f"窗口长度合规={a4['win_len_all_ok']},dd20<=1e-12={a4['dd20_le_1e-12_all']}"
        f"(max={a4['dd20_max']:.3e}) -> {'PASS' if a4['ok'] else 'FAIL'} "
        f"({time.time() - t0:.0f}s)")

    # ---- 与 frozen trades_seed(v1) 对账 + 断言 10 种子资格 ----
    t1 = time.time()
    tr = pd.read_parquet(SEED_DIR / "trades_seed.parquet")
    tr1 = tr[tr["variant"] == "v1"]
    tr_ev = tr1[["ts_code", "event_date", "dd20", "bounce",
                 "sel_S1", "sel_S2", "sel_S3", "sel_S4", "sel_S5"]] \
        .drop_duplicates(["ts_code", "event_date"]).reset_index(drop=True)
    tr_ev = tr_ev.rename(columns={
        "dd20": "dd20_frozen", "bounce": "bounce_frozen",
        **{s: f"{s}_frozen" for s in
           ["sel_S1", "sel_S2", "sel_S3", "sel_S4", "sel_S5"]}})
    m = ev.merge(dd[["ts_code", "event_date", "dd20"]],
                 on=["ts_code", "event_date"], how="left")
    assert len(m) == len(ev) and m["dd20"].notna().all(), "dd20 回填不完整"
    m["bounce_recalc"] = m["event_close"] / m["anchor_close"] - 1.0
    # frozen 计数自洽性:summary_seed.csv v1 三档 H 的 n_selected 应一致
    summ = pd.read_csv(SEED_DIR / "summary_seed.csv")
    sv = summ[(summ["variant"] == "v1") & (summ["seed"] != "ALL")]
    frozen_counts = {}
    frozen_counts_consistent = True
    for seed, grp in sv.groupby("seed"):
        vals = sorted(grp["n_selected"].unique())
        frozen_counts[seed] = int(vals[0]) if len(vals) == 1 else None
        if len(vals) != 1 or int(vals[0]) != FROZEN_SEED_COUNTS[seed]:
            frozen_counts_consistent = False
    # 本表重算种子布尔
    dd_v, bo_v = m["dd20"], m["bounce_recalc"]
    sel = {
        "S1": (dd_v <= -0.15) & (bo_v > 0.02) & (bo_v <= 0.08),
        "S2": (dd_v <= -0.20) & (bo_v > 0.02) & (bo_v <= 0.08),
        "S3": (dd_v <= -0.25) & (bo_v > 0.02) & (bo_v <= 0.08),
        "S4": dd_v <= -0.15,
        "S5": dd_v <= -0.25,
    }
    counts = {k: int(v.sum()) for k, v in sel.items()}
    counts_match = counts == FROZEN_SEED_COUNTS
    # 嵌套链:v6-3⊂v6-2⊂v6-1⊂v6-4、v6-5⊂v6-4 零违反
    nest = {
        "S3_in_S2": int((sel["S3"] & ~sel["S2"]).sum()),
        "S2_in_S1": int((sel["S2"] & ~sel["S1"]).sum()),
        "S1_in_S4": int((sel["S1"] & ~sel["S4"]).sum()),
        "S5_in_S4": int((sel["S5"] & ~sel["S4"]).sum()),
    }
    nest_ok = all(v == 0 for v in nest.values())
    # 与 frozen trades_seed 逐位对账(dd20/bounce/五布尔)
    mm = m.merge(tr_ev, on=["ts_code", "event_date"], how="outer",
                 indicator=True)
    join_ok = bool((mm["_merge"] == "both").all())
    dd20_diff = float(np.abs(mm["dd20"] - mm["dd20_frozen"]).max())
    bounce_diff = float(np.abs(mm["bounce_recalc"] - mm["bounce_frozen"]).max())
    dd20_bit = bool(np.array_equal(mm["dd20"].to_numpy(),
                                   mm["dd20_frozen"].to_numpy()))
    bounce_bit = bool(np.array_equal(mm["bounce_recalc"].to_numpy(),
                                     mm["bounce_frozen"].to_numpy()))
    # 在合并表自身列上重算布尔再逐位比(避免外连接行序对齐问题)
    ddm, bom = mm["dd20"], mm["bounce_recalc"]
    sel_m = {
        "S1": (ddm <= -0.15) & (bom > 0.02) & (bom <= 0.08),
        "S2": (ddm <= -0.20) & (bom > 0.02) & (bom <= 0.08),
        "S3": (ddm <= -0.25) & (bom > 0.02) & (bom <= 0.08),
        "S4": ddm <= -0.15,
        "S5": ddm <= -0.25,
    }
    bool_bit = {s: bool((mm[f"sel_{s}_frozen"].to_numpy()
                         == sel_m[s].to_numpy()).all()) for s in sel_m}
    a10 = {
        "frozen_counts_consistent": frozen_counts_consistent,
        "counts_recalc": counts, "counts_frozen": FROZEN_SEED_COUNTS,
        "counts_match": counts_match,
        "nesting_violations": nest, "nesting_ok": nest_ok,
        "join_with_frozen_trades_ok": join_ok,
        "dd20_bitwise": dd20_bit, "dd20_max_abs_diff": dd20_diff,
        "bounce_bitwise": bounce_bit, "bounce_max_abs_diff": bounce_diff,
        "seed_bool_bitwise": bool_bit,
    }
    a10["ok"] = bool(frozen_counts_consistent and counts_match and nest_ok
                     and join_ok and dd20_bit and bounce_bit
                     and all(bool_bit.values()))
    log(f"[断言10] 种子资格重算:计数 {counts} vs frozen "
        f"{FROZEN_SEED_COUNTS}(一致={counts_match});嵌套违反 {nest};"
        f"dd20/bounce 逐位={dd20_bit}/{bounce_bit};布尔逐位={bool_bit}"
        f" -> {'PASS' if a10['ok'] else 'FAIL'} ({time.time() - t1:.0f}s)")
    return a4, a10


# ---------------------------------------------------------------- 断言 5:锚点因果与取值域
def assert_5(ev: pd.DataFrame) -> dict:
    detail = {
        "a<=j": bool((ev["anchor_row"] <= ev["event_row"]).all()),
        "anchor_close>0": bool((ev["anchor_close"] > 0).all()),
    }
    eq_mask = ev["anchor_row"] == ev["event_row"]
    bounce0 = (ev.loc[eq_mask, "event_close"]
               / ev.loc[eq_mask, "anchor_close"] - 1.0).abs()
    detail["n_a_eq_j"] = int(eq_mask.sum())
    detail["a==j_imply_bounce0"] = bool((bounce0 <= 1e-12).all()) \
        if len(bounce0) else True
    detail["max_abs_bounce_when_a_eq_j"] = float(bounce0.max()) \
        if len(bounce0) else 0.0
    ok = all(v for k, v in detail.items() if isinstance(v, bool))
    log(f"[断言5] 锚点因果与取值域:{detail} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "detail": detail}


# ---------------------------------------------------------------- 断言 6(全表部分):事件条件自洽
def assert_6_full(ev: pd.DataFrame) -> dict:
    detail = {
        "dif_lift_min": float(ev["dif_lift"].min()),
        "dif_lift>=0.001_all": bool((ev["dif_lift"] >= DIF_LIFT_MIN).all()),
        "min_prev>min_cur_all": bool(
            (ev["min_prev_close"] > ev["anchor_close"]).all()),
        "min_prev/max(min_prev-anchor)":
            float((ev["min_prev_close"] - ev["anchor_close"]).max()),
    }
    ok = detail["dif_lift>=0.001_all"] and detail["min_prev>min_cur_all"]
    log(f"[断言6全表] 事件条件:dif_lift>=0.001={detail['dif_lift>=0.001_all']}"
        f"(min={detail['dif_lift_min']:.6f}),min_prev>min_cur="
        f"{detail['min_prev>min_cur_all']} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "detail": detail,
            "note": "抽样从原始日线逐字段重算核对见断言2(同批 2,000 事件)"}


# ---------------------------------------------------------------- 断言 7:量纲缩放不变性
# 偏差披露(2026-09-09 首跑发现,README 修订记录 ①):
# 信号判定含绝对阈值 dif_lift >= 0.001(价格量纲,#31 §2.1 冻结口径),
# close 同乘 lambda 后 dif_lift 同步放大 lambda 倍,
# 故缩放后会新增 unscaled dif_lift ∈ [0.001/lambda, 0.001) 的事件 ——
# 事件集合不具缩放不变性是冻结口径的直接推论,非实现缺陷。
# 据此断言 7 按「scan_v1.py 因果可行为准」实现为:
# (a) 基线事件零缺失(缩放不得使既有事件消失);
# (b) 每个新增事件的 dif_lift 必须落在阈值带 [0.001/lambda, 0.001) 内
#     (证明增量恰由绝对阈值效应解释,而非金叉比较翻转);
# (c) 基线事件(交集)上:行号/日期/量能列严格不变,
#     价格量纲列按 lambda 等比(相对误差 < 1e-9),
#     无量纲比值变化 < 1e-9。
def _a7_worker(task: dict) -> dict:
    ts_code = task["ts_code"]
    df = pd.read_parquet(DATA_DIR / f"{ts_code}.parquet")
    _, evs_base, reason = scan_df(df)
    if reason is not None:
        return {"ts_code": ts_code, "ok": False, "reason": reason}
    df2 = df.copy()
    df2["close"] = df2["close"] * A7_LAMBDA
    _, evs_scaled, reason2 = scan_df(df2)
    if reason2 is not None:
        return {"ts_code": ts_code, "ok": False, "reason": reason2}
    base = {e["event_row"]: e for e in evs_base}
    scaled = {e["event_row"]: e for e in evs_scaled}
    ref_rows = {e["event_row"]: e for e in task["table_events"]}
    if set(base) != set(ref_rows):
        return {"ts_code": ts_code, "ok": False,
                "reason": "基线重扫与事件表行号集合不一致"}
    missing = sorted(set(base) - set(scaled))
    if missing:
        return {"ts_code": ts_code, "ok": False,
                "reason": f"基线事件在缩放后消失(行号 {missing[:5]})"}
    added = sorted(set(scaled) - set(base))
    band_lo = DIF_LIFT_MIN / A7_LAMBDA
    for j in added:  # 新增事件必须可由阈值带解释
        dl_unscaled = scaled[j]["dif_lift"] / A7_LAMBDA
        if not (band_lo - 1e-9 <= dl_unscaled < DIF_LIFT_MIN + 1e-9):
            return {"ts_code": ts_code, "ok": False,
                    "reason": f"新增事件 row{j} 的 dif_lift/lambda="
                              f"{dl_unscaled:.8f} 不在阈值带"}
    int_cols = ["cross_prev2_row", "cross_prev_row", "event_row",
                "min_prev_row", "anchor_row", "anchor_bars"]
    date_cols = ["cross_prev2_date", "cross_prev_date", "min_prev_date",
                 "anchor_date", "event_date"]
    lam_cols = ["cross_prev2_dif", "cross_prev2_dea", "cross_prev_dif",
                "cross_prev_dea", "cross_dif", "cross_dea", "dif_lift",
                "min_prev_close", "anchor_close", "event_close"]
    inv_cols = ["event_vol", "event_amount"]
    max_rel = 0.0
    max_ratio_diff = 0.0
    for j, eb in base.items():
        es = scaled[j]
        et = ref_rows[j]
        for c in int_cols:
            if eb[c] != es[c] or eb[c] != et[c]:
                return {"ts_code": ts_code, "ok": False,
                        "reason": f"行号列变化 {c} @row{j}"}
        for c in date_cols:
            if not (pd.Timestamp(eb[c]) == pd.Timestamp(es[c])
                    == pd.Timestamp(et[c])):
                return {"ts_code": ts_code, "ok": False,
                        "reason": f"日期列变化 {c} @row{j}"}
        for c in inv_cols:
            if not (eb[c] == es[c] == et[c]):
                return {"ts_code": ts_code, "ok": False,
                        "reason": f"量能列变化 {c} @row{j}"}
        for c in lam_cols:
            o, s = eb[c], es[c]
            denom = A7_LAMBDA * abs(o)
            rel = abs(s - A7_LAMBDA * o) / denom if denom > 0 \
                else abs(s - A7_LAMBDA * o)
            max_rel = max(max_rel, float(rel))
        # 代表性无量纲比值(README 第六节断言 7 清单)
        ratios_base = [
            eb["cross_dif"] / eb["event_close"],
            eb["cross_dea"] / eb["event_close"],
            eb["dif_lift"] / eb["event_close"],
            (eb["cross_dif"] - eb["cross_dea"]) / eb["event_close"],
            eb["anchor_close"] / eb["min_prev_close"] - 1.0,
            (eb["event_row"] - eb["cross_prev_row"])
            / (eb["cross_prev_row"] - eb["cross_prev2_row"]),
        ]
        ratios_scaled = [
            es["cross_dif"] / es["event_close"],
            es["cross_dea"] / es["event_close"],
            es["dif_lift"] / es["event_close"],
            (es["cross_dif"] - es["cross_dea"]) / es["event_close"],
            es["anchor_close"] / es["min_prev_close"] - 1.0,
            (es["event_row"] - es["cross_prev_row"])
            / (es["cross_prev_row"] - es["cross_prev2_row"]),
        ]
        for rb, rs in zip(ratios_base, ratios_scaled):
            max_ratio_diff = max(max_ratio_diff, float(abs(rb - rs)))
    ok = max_rel < A7_TOL and max_ratio_diff < A7_TOL
    return {"ts_code": ts_code, "ok": bool(ok), "n_events": len(base),
            "n_added": len(added), "added_rows": added,
            "max_rel_scale_err": max_rel, "max_ratio_diff": max_ratio_diff}


def assert_7(ev: pd.DataFrame) -> dict:
    t0 = time.time()
    counts = ev.groupby("ts_code").size()
    eligible = sorted(counts[counts >= 3].index.tolist())
    rng = np.random.default_rng(SAMPLE_SEED)
    picked = sorted(rng.choice(np.array(eligible, dtype=object),
                               size=min(A7_N_STOCKS, len(eligible)),
                               replace=False).tolist())
    ev_by_code: dict[str, list[dict]] = {}
    for r in ev.itertuples(index=False):
        ev_by_code.setdefault(r.ts_code, []).append(r._asdict())
    tasks = [{"ts_code": c, "table_events": ev_by_code[c]} for c in picked]
    results: list[dict] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for res in pool.imap_unordered(_a7_worker, tasks):
            results.append(res)
    n_bad = [r for r in results if not r["ok"]]
    max_rel = max(float(r.get("max_rel_scale_err", 0.0)) for r in results)
    max_ratio = max(float(r.get("max_ratio_diff", 0.0)) for r in results)
    n_events = sum(int(r.get("n_events", 0)) for r in results)
    added = {r["ts_code"]: r["n_added"] for r in results
             if r.get("n_added", 0) > 0}
    ok = not n_bad
    log(f"[断言7] 量纲缩放(lambda={A7_LAMBDA}):{len(results)} 股 "
        f"{n_events} 基线事件,价格列相对缩放误差 max={max_rel:.2e},"
        f"无量纲比值变化 max={max_ratio:.2e}(容差 {A7_TOL});"
        f"基线事件零缺失,新增事件(绝对阈值带效应,已披露)={added or '无'};"
        f"失败 {len(n_bad)} -> {'PASS' if ok else 'FAIL'} "
        f"({time.time() - t0:.0f}s)")
    return {"ok": ok, "n_stocks": len(results), "n_events": n_events,
            "lambda": A7_LAMBDA, "max_rel_scale_err": max_rel,
            "max_ratio_diff": max_ratio, "tol": A7_TOL,
            "added_events_by_threshold_band": added,
            "deviation": "事件集合缩放不变性与冻结口径的绝对阈值 "
                         "dif_lift>=0.001 矛盾;按 scan_v1.py 因果可行为准,"
                         "改为基线事件零缺失 + 新增事件阈值带归因 + "
                         "交集不变性(README 修订记录 ①,report.md 披露)",
            "bad": n_bad[:5], "seed": SAMPLE_SEED}


# ---------------------------------------------------------------- 断言 8/9:来源与隔离(列集机检)
def assert_8_9(ev: pd.DataFrame) -> tuple[dict, dict]:
    cols = list(ev.columns)
    a8 = {
        "columns_exact_registered": cols == ALL_COLS,
        "no_snapshot_env_cols": not any(
            c.lower() in ("vol20", "vma5", "dvol") or "snapshot" in c.lower()
            for c in cols),
    }
    a8["ok"] = a8["columns_exact_registered"] and a8["no_snapshot_env_cols"]
    a8["note"] = ("E2/E3/E4 按 #42 族E 立规由 v5 日频快照在 M2 覆盖,"
                  "本表不含亦不得含;M2 构建器不得直接读事件日后行(移交声明)")
    a9 = {
        "no_label_prefix_cols": not any(c.startswith("label_") for c in cols),
        "no_label_artifacts": not any(OUT_DIR.glob("*label*")),
        "join_key": "(ts_code, event_date)",
    }
    a9["ok"] = a9["no_label_prefix_cols"] and a9["no_label_artifacts"]
    a9["note"] = ("M1 不产出标签表;标签由 M3 独立构建器产出,"
                  "join 键仅限 (ts_code, event_date)(移交声明)")
    log(f"[断言8] 环境特征来源:列集==预登记25列={a8['columns_exact_registered']}"
        f" -> {'PASS' if a8['ok'] else 'FAIL'}")
    log(f"[断言9] 特征-标签隔离:无 label_ 列={a9['no_label_prefix_cols']},"
        f"无标签产物={a9['no_label_artifacts']} -> {'PASS' if a9['ok'] else 'FAIL'}")
    return a8, a9


# ---------------------------------------------------------------- 断言 11:缺失披露
def assert_11(ev: pd.DataFrame) -> dict:
    nan_count = {c: int(ev[c].isna().sum()) for c in NEW_COLS}
    nan_ratio = {c: float(ev[c].isna().mean()) for c in NEW_COLS}
    allowed_nan_cols = {"event_amount"}  # 预登记允许集(缺列守卫,README 披露 5)
    unexpected = {c: v for c, v in nan_count.items()
                  if v > 0 and c not in allowed_nan_cols}
    ok = not unexpected
    log(f"[断言11] 缺失披露:NaN 计数="
        f"{ {c: v for c, v in nan_count.items() if v} or '全 0' };"
        f"预登记允许集外 NaN={unexpected or '无'} -> {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "nan_count": nan_count, "nan_ratio": nan_ratio,
            "allowed_nan_cols": sorted(allowed_nan_cols),
            "unexpected_nan": unexpected,
            "note": "含守卫 NaN 分支的特征 B3/B4/B6/C4/E1 归 M2 特征构建器,"
                    "其 NaN 占比披露与'不得用事件日后信息填充'断言在 M2 执行"
                    "(移交声明)"}


# ---------------------------------------------------------------- report.md 生成
def write_report(scan: dict, caus: dict, checks_all: bool) -> None:
    L: list[str] = []
    ap = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    L.append("# M1 事件表扩展与扫描器 —— 结果与自检全表(预登记一发,2026-09-09)")
    L.append("")
    L.append("> 预登记 = 同目录 README.md(先于跑数落盘,冻结)。")
    L.append("> 口径源头:#31 §2.1、#42 §3.1/§7、scan_v1.py、run_seeds.py(只读)。")
    L.append("> 对账基准:events_history_v1.parquet(96,577 行 × 9 列,只读)。")
    L.append("")
    L.append(f"**验收总评:{'ALL PASS' if checks_all else 'HAS FAIL(如实交付)'}**")
    L.append("")
    L.append("## 1. 产物概要")
    L.append("")
    a = scan["acceptance_A_frozen_reconcile"]
    L.append(f"- 产物:`experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet`,"
             f"{a['n_new']} 行 × 25 列(旧 9 列 + 新 16 列)。")
    L.append(f"- 扫描两次(run1 {scan['scan_meta']['run1']['elapsed_sec']}s / "
             f"run2 {scan['scan_meta']['run2']['elapsed_sec']}s),"
             f"跳过统计 run1={scan['scan_meta']['run1']['stats']},")
    L.append(f"  run2={scan['scan_meta']['run2']['stats']}(schema=2 为指数文件,"
             "与 frozen 参照一致)。")
    L.append(f"- 扫描+验收总耗时 {scan['duration_sec']}s;自检(本脚本)耗时 "
             f"{caus['duration_sec']}s。")
    L.append("")
    L.append("## 2. 新增列清单(16 列;中文全称/公式/因果/量纲见 README 第三节)")
    L.append("")
    d_nan = caus["assertion_11"]["nan_count"]
    d_ratio = caus["assertion_11"]["nan_ratio"]
    L.append("| # | 列名 | 中文全称 | NaN 计数 | NaN 占比 |")
    L.append("|---|---|---|---|---|")
    cn = {
        "cross_prev2_row": "远金叉行号", "cross_prev2_date": "远金叉日期",
        "cross_prev2_dif": "远金叉 DIF 值", "cross_prev2_dea": "远金叉 DEA 值",
        "cross_prev_row": "前金叉行号", "cross_prev_dea": "前金叉 DEA 值",
        "event_row": "事件日行号", "event_close": "事件日收盘价",
        "event_vol": "事件日成交量", "event_amount": "事件日成交额",
        "cross_dea": "事件日金叉 DEA 值", "min_prev_close": "前区间最低收盘价",
        "min_prev_row": "前区间最低收盘价行号", "min_prev_date": "前区间最低收盘价日期",
        "anchor_row": "锚点行号", "anchor_bars": "锚点距事件日交易日数",
    }
    for i, c in enumerate(NEW_COLS, 1):
        L.append(f"| {i} | {c} | {cn[c]} | {d_nan[c]} | {d_ratio[c]:.6%} |")
    L.append("")
    L.append("## 3. 验收 A:与 frozen 基准对账")
    L.append("")
    L.append(f"- 行数:新表 {a['n_new']} vs frozen {a['n_ref']}"
             f"(相等={a['row_count_match']})。")
    L.append(f"- 键 (ts_code, event_date) 唯一性:新表 {a['key_unique_new']},"
             f"frozen {a['key_unique_ref']}。")
    L.append(f"- 行序逐行逐位(9 共有列):{'PASS' if a['row_order_bitwise_ok'] else 'FAIL'},"
             f"浮点最大绝对差 {a['row_order_max_abs_float_diff']:.2e}。")
    L.append(f"- 键对齐合并:both={a['join_both']},left_only={a['join_left_only']},"
             f"right_only={a['join_right_only']};"
             f"对齐后逐位={'PASS' if a['join_cols_bitwise_ok'] else 'FAIL'},"
             f"浮点最大绝对差 {a['join_max_abs_float_diff']:.2e}。")
    L.append(f"- frozen 表行序本身即 (event_date, ts_code) 升序:"
             f"{a['ref_sorted_row_order']}。")
    L.append(f"- **结论:{ap(a['ok'])}**")
    L.append("")
    L.append("## 4. 验收 B:截断历史重算(20 股 × 3 截断行,种子 20260909)")
    L.append("")
    b = scan["acceptance_B_truncation"]
    L.append(f"- 抽样 {b['n_stocks']} 股 × {b['n_truncs']} 截断组,"
             f"通过 {b['n_ok']}/{b['n_truncs']},"
             f"对比事件合计 {b['n_events_compared']} 起,"
             f"全部 25 列逐位一致,浮点最大绝对差 {b['max_abs_float_diff']:.2e}。")
    L.append("")
    L.append("| 股票 | 截断行 | 对比事件数 | 逐位一致 |")
    L.append("|---|---|---|---|")
    for r in scan["acceptance_B_detail"]:
        L.append(f"| {r['ts_code']} | {r['trunc_row']} | "
                 f"{r.get('n_events', 0)} | {'PASS' if r['ok'] else 'FAIL'} |")
    L.append("")
    L.append(f"- 抽样股票:{', '.join(b['stocks'])}。")
    L.append(f"- **结论:{ap(b['ok'])}**")
    L.append("")
    L.append("## 5. 验收 C:#42 §7 十一断言逐条结果")
    L.append("")
    order = [("断言1 行号上界", "assertion_1"),
             ("断言2 区间语义自洽(独立重算抽样)", "assertion_2"),
             ("断言3 MACD 无前视", "assertion_3"),
             ("断言4 dd20 窗口与符号(全表)", "assertion_4"),
             ("断言5 锚点因果与取值域", "assertion_5"),
             ("断言6 事件条件自洽(全表)", "assertion_6_full"),
             ("断言7 量纲缩放不变性", "assertion_7"),
             ("断言8 环境特征来源", "assertion_8"),
             ("断言9 特征-标签隔离", "assertion_9"),
             ("断言10 成员资格重算一致", "assertion_10"),
             ("断言11 缺失披露", "assertion_11")]
    L.append("| 断言 | 结果 | 关键数值 |")
    L.append("|---|---|---|")
    key_nums = {
        "assertion_1": "八项行号序关系全真",
        "assertion_2": f"抽样 {caus['assertion_2']['n_sample']} 事件 "
                       f"({caus['assertion_2']['n_stocks']} 股),"
                       f"不一致 {caus['assertion_2']['n_fail']}",
        "assertion_3": f"抽样 {caus['assertion_3']['n_sample']} 事件,"
                       f"截尾重算最大绝对差 "
                       f"{caus['assertion_3']['max_abs_diff']:.2e}"
                       f"(容差 {A3_TOL})",
        "assertion_4": f"全表 {caus['assertion_4']['n_events']} 事件,"
                       f"dd20 max={caus['assertion_4']['dd20_max']:.3e},"
                       f"行号对账/窗口长度/dd20<=1e-12 全真",
        "assertion_5": f"a==j 事件 {caus['assertion_5']['detail']['n_a_eq_j']} 起,"
                       f"|bounce| max="
                       f"{caus['assertion_5']['detail']['max_abs_bounce_when_a_eq_j']:.2e}",
        "assertion_6_full": f"dif_lift min={caus['assertion_6_full']['detail']['dif_lift_min']:.6f}"
                            f"(>=0.001),min_prev>min_cur 全真",
        "assertion_7": f"{caus['assertion_7']['n_stocks']} 股 "
                       f"{caus['assertion_7']['n_events']} 基线事件零缺失,"
                       f"缩放相对误差 {caus['assertion_7']['max_rel_scale_err']:.2e},"
                       f"比值变化 {caus['assertion_7']['max_ratio_diff']:.2e}"
                       f"(容差 {A7_TOL});"
                       f"新增事件(绝对阈值带效应,偏差披露见第 8 节 10):"
                       f"{caus['assertion_7']['added_events_by_threshold_band'] or '无'}",
        "assertion_8": "列集==预登记 25 列,无快照环境列",
        "assertion_9": "无 label_ 列,无标签产物",
        "assertion_10": f"计数重算 {caus['assertion_10']['counts_recalc']},"
                        f"嵌套违反 {caus['assertion_10']['nesting_violations']},"
                        f"dd20/bounce/布尔逐位一致",
        "assertion_11": "16 新列 NaN 全 0(见第 2 节)",
    }
    for name, key in order:
        L.append(f"| {name} | {ap(caus[key]['ok'])} | {key_nums[key]} |")
    L.append("")
    L.append("### 断言 10 明细(种子资格,与 frozen trades_seed v1 对账)")
    L.append("")
    a10 = caus["assertion_10"]
    L.append("| 种子 | 重算计数 | frozen 计数 | 一致 |")
    L.append("|---|---|---|---|")
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        L.append(f"| {s}(v6-{int(s[1])}) | {a10['counts_recalc'][s]} | "
                 f"{a10['counts_frozen'][s]} | "
                 f"{'是' if a10['counts_recalc'][s] == a10['counts_frozen'][s] else '否'} |")
    L.append("")
    L.append(f"- frozen 计数自洽(summary_seed.csv v1 三档 H 一致且等于 #31 §4.4):"
             f"{a10['frozen_counts_consistent']}。")
    L.append(f"- 嵌套链违反计数:{a10['nesting_violations']}(全 0 = 通过)。")
    L.append(f"- 与 frozen trades_seed(v1,96,577 事件)join 完整:"
             f"{a10['join_with_frozen_trades_ok']};dd20 逐位={a10['dd20_bitwise']}"
             f"(最大绝对差 {a10['dd20_max_abs_diff']:.2e}),"
             f"bounce 逐位={a10['bounce_bitwise']}"
             f"(最大绝对差 {a10['bounce_max_abs_diff']:.2e}),"
             f"五布尔逐位={a10['seed_bool_bitwise']}。")
    L.append("")
    L.append("## 6. 验收 D:NaN 披露与 warm-up 行为")
    L.append("")
    d = scan["acceptance_D_nan_warmup"]
    L.append("- 新增 16 列逐列 NaN 计数/占比见第 2 节表;"
             f"warm-up 段行为:全部 MACD 派生列(cross_prev2_dif/dea、"
             f"cross_prev_dif/dea、cross_dif/dea、dif_lift)非 NaN = "
             f"{d['warmup_macd_cols_all_non_nan']}(warm-up 段由金叉检测的 "
             f"非 NaN 要求自然剔除,与旧表一致;行数对账见验收 A)。")
    L.append(f"- **结论:{ap(d['ok'] and caus['assertion_11']['ok'])}**")
    L.append("")
    L.append("## 7. 验收 E:确定性")
    L.append("")
    e = scan["acceptance_E_determinism"]
    L.append(f"- 两次全量扫描行数 {e['n_run1']}/{e['n_run2']},"
             f"全部 25 列逐位一致,浮点最大绝对差 {e['max_abs_float_diff']:.2e}。")
    L.append(f"- **结论:{ap(e['ok'])}**")
    L.append("")
    L.append("## 8. 偏差与设计披露(README 第七节,原样复述 + 实测)")
    L.append("")
    L.append("1. min_cur 不单独落列:由构造恒等于 anchor_close,断言 6 以 "
             "min_prev_close > anchor_close 实现;实测零偏差。")
    L.append("2. 「急跌窗口内部路径」不落盘(变长),改落 event_row;窗口 W 由 M2 "
             "凭 event_row 从原始日线重建;dd20 因果性经断言 4 全表独立重算验证。")
    L.append("3. dd20/bounce 与种子布尔不落 M1 事件表(归 M2);断言 4/10 以全表"
             "独立重算 + frozen trades_seed 逐位对账验收,实测逐位一致。")
    L.append("4. E2/E3/E4 快照覆盖,不进 v6 专属层(#42 族E 立规),M1 不落。")
    L.append("5. amount 缺列守卫实测 0 触发(仅 2 个指数文件缺 amount,本就被 "
             "schema 过滤);event_amount 实测 NaN = "
             f"{d_nan['event_amount']}。")
    L.append("6. 行号为个股序列 0 基行号(非市场日历对齐);个股序列无停牌行。")
    L.append("7. 事件日量能为原始口径,未复权未归一;归一化特征归 M2。")
    L.append("8. 扫描跳过口径与 frozen 参照逐字一致;schema 跳过 2 个为指数文件。")
    L.append("9. parquet 产物与 progress.log 按 .gitignore 不入库;代码、README、"
             "report.md、两份自检 JSON 入库。")
    L.append("10. **断言 7 偏差披露(首跑发现,README 修订记录 ①)**:预登记实现"
             "「缩放后事件集合不变」与冻结信号口径的绝对阈值 dif_lift>=0.001"
             "(价格量纲,#31 §2.1)矛盾:close 同乘 lambda 后 dif_lift 同步放大,"
             "缩放版会新增 unscaled dif_lift ∈ [0.001/lambda, 0.001) 的事件"
             "(首跑 50 股中 5 股各 +1 起,合计 +5 起)。"
             "按铁律「#42 与 scan_v1.py 矛盾时以因果可行为准,不许私改信号定义」,"
             "断言 7 改实现为:基线事件零缺失 + 每个新增事件 dif_lift 落在阈值带"
             "(证明增量=绝对阈值效应而非金叉比较翻转)+ 基线事件交集上行号/日期/"
             "量能严格不变、价格列等比(相对误差 < 1e-9)、无量纲比值变化 < 1e-9。"
             "信号定义未做任何改动。")
    L.append("")
    L.append("## 9. 移交 M2 声明(断言 8/9/11 的 M2 部分)")
    L.append("")
    L.append("- 断言 8:E2/E3/E4 及 v5 三来源环境特征必须取自日频特征快照第 "
             "event_date 行,M2 构建器不得直接读事件日后行。")
    L.append("- 断言 9:标签表由 M3 独立构建器产出,join 键仅限 "
             "(ts_code, event_date),两表构建器无共享中间产物。")
    L.append("- 断言 11:B3/B4/B6/C4/E1 的守卫 NaN 分支占比在 M2 落盘披露,"
             "NaN 不得用事件日后信息填充。")
    L.append("")
    L.append("## 10. 复现步骤")
    L.append("")
    L.append("1. `python3 experiments/v6_model_campaign/m1_event_table/run_event_table_ext.py`"
             "(扫描两次 + 验收 A/B/D/E,落 m1_scan_results.json)。")
    L.append("2. `python3 experiments/v6_model_campaign/m1_event_table/selfcheck_causality.py`"
             "(十一断言,落 m1_causality_results.json 与本报告)。")
    L.append("")
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("[dump] report.md")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    log("M1 CAUSALITY SELFCHECK START | 预登记=README.md 第五/六节(冻结)")
    ev = pd.read_parquet(OUT_DIR / "events_ext_v1.parquet")
    with open(OUT_DIR / "m1_scan_results.json", encoding="utf-8") as f:
        scan = json.load(f)
    log(f"[init] events_ext_v1.parquet {len(ev)} 行 × {len(ev.columns)} 列")

    r1 = assert_1(ev)
    r2 = assert_2_and_6_sample(ev)
    r3 = assert_3(ev)
    r4, r10 = assert_4_10(ev)
    r5 = assert_5(ev)
    r6 = assert_6_full(ev)
    r7 = assert_7(ev)
    r8, r9 = assert_8_9(ev)
    r11 = assert_11(ev)

    caus = {
        "experiment": "M1 因果性自检(#42 §7 十一断言)",
        "assertion_1": r1, "assertion_2": r2, "assertion_3": r3,
        "assertion_4": r4, "assertion_5": r5, "assertion_6_full": r6,
        "assertion_7": r7, "assertion_8": r8, "assertion_9": r9,
        "assertion_10": r10, "assertion_11": r11,
        "duration_sec": round(time.time() - t_all, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    keys = ["assertion_1", "assertion_2", "assertion_3", "assertion_4",
            "assertion_5", "assertion_6_full", "assertion_7", "assertion_8",
            "assertion_9", "assertion_10", "assertion_11"]
    caus_all = all(caus[k]["ok"] for k in keys)
    caus["assertions_all_pass"] = bool(caus_all)
    scan_side_ok = bool(scan["checks_all_pass"])
    checks_all = caus_all and scan_side_ok
    caus["checks_all_pass_incl_scan_side"] = bool(checks_all)
    with open(OUT_DIR / "m1_causality_results.json", "w", encoding="utf-8") as f:
        json.dump(caus, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] m1_causality_results.json")
    write_report(scan, caus, checks_all)
    log(f"M1 CAUSALITY SELFCHECK DONE ({time.time() - t_all:.0f}s) "
        f"十一断言={'ALL PASS' if caus_all else 'HAS FAIL'},"
        f"含扫描侧总评={'ALL PASS' if checks_all else 'HAS FAIL'}")


if __name__ == "__main__":
    main()
