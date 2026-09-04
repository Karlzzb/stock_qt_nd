#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIF 拐点提前事件（实验B）扫描器 + 自检闸门 + 前瞻收益汇总（预登记见同目录 README.md）。

流程（单脚本一次跑完，闸门不过则停）：
1. 每股复算 talib.MACD 内部种子口径的 DIF/DEA（直接 import 复用
   predictive_entry/predictive_scan.py 的 macd_internal，不复制实现），并与 talib.MACD
   直接输出对拍：任何一只股票偏差 > 1e-9 即硬失败。
2. 自检闸门二：用复算值完整复现既有金叉对金叉事件（金叉检测 + dif_lift >= 0.001 +
   变体 1 价格条件 + 事件日窗口过滤），(ts_code, cross_date) 集合必须与
   events_v1.parquet 零容差一致。
3. 提前事件扫描：对每个金叉 C_prev（需存在再前金叉 C_prev2），在区间 (C_prev, C_next)
   （尾区间到数据末尾）内逐日监控三条件（价格新低 / DIF 拐点没新低 / 自拐点连升 j 日），
   首日 t = τ + j 开火，每区间每配置最多一次；B1 锚点为区间 (C_prev, 开火基日] 最低收盘价日，
   B2 锚点为同窗口左 3 右 2 确认局部低点（贪心间隔 >= 3，沿用既有变体 2 规则），
   B2 事件日 = max(开火基日, 锚点日 + 2)。
4. 汇总：前瞻收益（收盘对收盘不含成本）+5/+10/+20/+30/+60 均值/中位/胜率，
   +10/+20 按 event_date 聚类的 cluster-robust t；基线行引用 forward_returns.json；
   既有变体 1/2 的事件日距锚点距离由本脚本按事件表与行情重算。

全程禁网络；仓库内除本目录外只读。
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

# 复用 predictive_entry 实验的 MACD 内部种子口径复算实现（README 第 6 节钉死：import 复用）。
_PREDICTIVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "predictive_entry")
sys.path.insert(0, os.path.abspath(_PREDICTIVE_DIR))
from predictive_scan import macd_internal  # noqa: E402

REPO = "/home/karl/repos/personal/stock_qt_nd"
DATA_DIR = os.path.join(REPO, "stock_data", "daily")
EXP_DIR = os.path.join(REPO, "experiments", "divergence_anchor_eval_2026")
OUT_DIR = os.path.join(EXP_DIR, "early_event_dif_turn")
LOG_PATH = os.path.join(OUT_DIR, "scan_early_event.log")

DATE_LO = pd.Timestamp("2026-01-01")
DATE_HI = pd.Timestamp("2026-08-31")
MIN_ROWS = 100
DIF_LIFT_MIN = 0.001
TOL = 1e-9
HORIZONS = [5, 10, 20, 30, 60]
JS = (1, 2, 3)

EVENT_COLUMNS = [
    "ts_code", "event_date", "anchor_date", "anchor_close", "anchor_kind",
    "cross_prev_date", "cross_prev_dif", "cross_prev2_date",
    "tau_date", "tau_dif", "dif_floor", "fire_base_date", "j",
    "dist_event_anchor_trade_days", "completed", "next_cross_date",
    "lead_days_vs_next_cross", "dif_lift_next", "became_v1_event",
    "fwd_ret_5", "fwd_ret_10", "fwd_ret_20", "fwd_ret_30", "fwd_ret_60",
]

# 既有事件日 +10 日均值（%），引用自 forward_returns.json（判活线第三条的对照基准）。
EXISTING_EVENT_MEAN10 = {"b1": None, "b2": None}  # 运行时从 forward_returns.json 读入


def strict_up_run_lengths(dif: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """每个位置 t 上、以 t 结尾的 DIF 严格连升天数（DIF[t] > DIF[t-1] 计一步）。"""
    n = len(dif)
    inc = np.zeros(n, dtype=bool)
    inc[1:] = valid[1:] & valid[:-1] & (dif[1:] > dif[:-1])
    idx = np.arange(n)
    last_not_inc = np.maximum.accumulate(np.where(inc, -1, idx))
    return np.where(inc, idx - last_not_inc, 0)


def b2_anchor(close: np.ndarray, lo: int, fire_base: int, n: int):
    """窗口 (lo, fire_base] 内的锚点（确认规则与既有变体 2 一致：左 3 右 2、间隔 >= 3）。

    返回 (锚点行号, 锚点类型中文全称)。
    """
    cands = []
    for s in range(lo + 1, fire_base + 1):
        if s - 3 < 0 or s + 2 > n - 1:
            continue
        if close[s] <= close[s - 3:s].min() and close[s] < close[s + 1:s + 3].min():
            cands.append(s)
    kept = []
    for s in cands:
        if not kept or s - kept[-1] >= 3:
            kept.append(s)
    if kept:
        best = kept[0]
        for s in kept[1:]:
            if close[s] < close[best]:
                best = s
        return best, "确认局部低点"
    seg = close[lo + 1:fire_base + 1]
    return lo + 1 + int(np.argmin(seg)), "回退区间最低收盘价"


def process_stock(path: str):
    """返回 (path, ts_code 或 None, 复现的变体1金叉对列表, 复算偏差元组, 事件列表,
    B2出窗丢弃数, 错误串或 None)。"""
    try:
        df = pd.read_parquet(path)
        cols = set(df.columns)
        if "ts_code" not in cols or "vol" not in cols:
            return path, None, [], None, [], 0, None
        if len(df) < MIN_ROWS:
            return path, None, [], None, [], 0, None
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        close = df["close"].to_numpy(dtype=np.float64)
        dates = df["trade_date"].to_numpy()
        n = len(df)
        ts_code = str(df["ts_code"].iloc[0])

        dif_t, dea_t, _ = talib.MACD(close)
        if np.isnan(dif_t).all():
            return path, None, [], None, [], 0, None
        internal = macd_internal(close)
        if internal is None:
            return path, None, [], None, [], 0, None
        _, _, dif, dea = internal

        valid = ~(np.isnan(dif_t) | np.isnan(dea_t))
        d_dif = float(np.abs(dif[valid] - dif_t[valid]).max()) if valid.any() else 0.0
        d_dea = float(np.abs(dea[valid] - dea_t[valid]).max()) if valid.any() else 0.0
        diffs = (d_dif, d_dea)

        # 金叉检测（用复算值；与 talib 输出对拍由偏差闸门兜底）
        cross = np.zeros(n, dtype=bool)
        cross[1:] = (valid[:-1] & valid[1:]
                     & (dif[:-1] <= dea[:-1])
                     & (dif[1:] > dea[1:]))
        crosses = np.flatnonzero(cross)

        # ---------- 自检闸门二：复现既有变体 1 金叉对 ----------
        v1_pairs = []
        for k in range(2, len(crosses)):
            c_km2, c_km1, c_k = int(crosses[k - 2]), int(crosses[k - 1]), int(crosses[k])
            dif_lift = dif[c_k] - dif[c_km1]
            if dif_lift < DIF_LIFT_MIN:
                continue
            seg_prev = close[c_km2 + 1:c_km1 + 1]
            seg_cur = close[c_km1 + 1:c_k + 1]
            if len(seg_prev) == 0 or len(seg_cur) == 0:
                continue
            if not (seg_prev.min() > seg_cur.min()):
                continue
            d = pd.Timestamp(dates[c_k])
            if DATE_LO <= d <= DATE_HI:
                v1_pairs.append((ts_code, d.strftime("%Y-%m-%d")))

        # ---------- 提前事件扫描 ----------
        up_run = strict_up_run_lengths(dif, valid)
        events = []
        dropped_datatail = 0
        for k in range(1, len(crosses)):
            c_prev = int(crosses[k])
            c_prev2 = int(crosses[k - 1])
            c_next = int(crosses[k + 1]) if k + 1 < len(crosses) else None
            seg_start = c_prev + 1
            seg_end = (c_next - 1) if c_next is not None else (n - 1)
            if seg_end < seg_start:
                continue
            min_prev_price = float(close[c_prev2 + 1:c_prev + 1].min())
            dif_floor = float(dif[c_prev] + DIF_LIFT_MIN)
            run_min_dif = np.inf
            tau = -1
            run_min_close = np.inf
            fire = {1: -1, 2: -1, 3: -1}
            for t in range(seg_start, seg_end + 1):
                d = dif[t]
                if d < run_min_dif:
                    run_min_dif = d
                    tau = t
                c = close[t]
                if c < run_min_close:
                    run_min_close = c
                if run_min_dif >= dif_floor and run_min_close < min_prev_price:
                    for j in JS:
                        if fire[j] < 0 and tau == t - j and up_run[t] >= j:
                            fire[j] = t
                if all(v >= 0 for v in fire.values()):
                    break
            for j in JS:
                f = fire[j]
                if f < 0:
                    continue
                base_common = dict(
                    ts_code=ts_code,
                    cross_prev_date=pd.Timestamp(dates[c_prev]),
                    cross_prev_dif=float(dif[c_prev]),
                    cross_prev2_date=pd.Timestamp(dates[c_prev2]),
                    tau_date=pd.Timestamp(dates[tau]),
                    tau_dif=float(dif[tau]),
                    dif_floor=dif_floor,
                    fire_base_date=pd.Timestamp(dates[f]),
                    j=j,
                    completed=c_next is not None,
                    next_cross_idx=c_next,
                    next_cross_date=(pd.Timestamp(dates[c_next])
                                     if c_next is not None else pd.NaT),
                    dif_lift_next=(float(dif[c_next] - dif[c_prev])
                                   if c_next is not None else np.nan),
                )
                base_common["became_v1_event"] = bool(
                    c_next is not None and base_common["dif_lift_next"] >= DIF_LIFT_MIN)
                # ---- 风味 B1：锚点 = 区间 (C_prev, 开火基日] 最低收盘价日，事件日 = 开火基日 ----
                seg = close[c_prev + 1:f + 1]
                a1 = c_prev + 1 + int(np.argmin(seg))  # 并列取最早
                rec = dict(base_common)
                rec.update(flavor="b1", anchor_date=pd.Timestamp(dates[a1]),
                           anchor_close=float(close[a1]), anchor_kind="区间最低收盘价",
                           event_idx=f, anchor_idx=a1)
                events.append(rec)
                # ---- 风味 B2：确认局部低点锚，事件日 = max(开火基日, 锚点日 + 2) ----
                a2, kind = b2_anchor(close, c_prev, f, n)
                e2 = max(f, a2 + 2)
                if e2 > n - 1:
                    dropped_datatail += 1
                else:
                    rec = dict(base_common)
                    rec.update(flavor="b2", anchor_date=pd.Timestamp(dates[a2]),
                               anchor_close=float(close[a2]), anchor_kind=kind,
                               event_idx=e2, anchor_idx=a2)
                    events.append(rec)

        # ---------- 事件窗口过滤 + 前瞻收益 ----------
        out_events = []
        for rec in events:
            e = rec.pop("event_idx")
            aidx = rec.pop("anchor_idx")
            nxt_idx = rec.pop("next_cross_idx")
            event_date = pd.Timestamp(dates[e])
            if not (DATE_LO <= event_date <= DATE_HI):
                continue
            rec["event_date"] = event_date
            rec["dist_event_anchor_trade_days"] = int(e - aidx)
            rec["lead_days_vs_next_cross"] = (int(nxt_idx - e)
                                              if nxt_idx is not None else np.nan)
            for h in HORIZONS:
                rec[f"fwd_ret_{h}"] = (float(close[e + h] / close[e] - 1.0)
                                       if e + h <= n - 1 else np.nan)
            out_events.append(rec)
        return path, ts_code, v1_pairs, diffs, out_events, dropped_datatail, None
    except Exception:
        return path, None, [], None, [], 0, traceback.format_exc()


def cluster_t(rets: np.ndarray, clusters: np.ndarray):
    """按 event_date 聚类的单样本 cluster-robust t（均值检验）。"""
    x = np.asarray(rets, dtype=np.float64)
    g = np.asarray(clusters)
    n_ = len(x)
    if n_ < 3:
        return np.nan
    xbar = x.mean()
    s = pd.DataFrame({"r": x - xbar, "g": g}).groupby("g")["r"].sum().to_numpy()
    g_count = len(s)
    if g_count < 2:
        return np.nan
    var = (g_count / (g_count - 1.0)) * float((s ** 2).sum()) / (n_ ** 2)
    se = np.sqrt(var)
    return float(xbar / se) if se > 0 else np.nan


def summarize_horizon(df: pd.DataFrame, h: int):
    """单档前瞻收益统计：样本数、均值%、中位%、胜率%（不足 N 日的已存 NaN，剔除）。"""
    r = df[f"fwd_ret_{h}"].dropna().to_numpy(dtype=np.float64)
    if len(r) == 0:
        return {"n": 0, "mean": None, "median": None, "win": None}, r
    return {"n": int(len(r)), "mean": float(r.mean() * 100),
            "median": float(np.median(r) * 100),
            "win": float((r > 0).mean() * 100)}, r


def existing_distance_worker(task):
    """单股：计算既有事件表中 (事件日, 锚点日) 对的交易日距离列表。"""
    p, pairs = task
    try:
        d = pd.read_parquet(p, columns=["trade_date"])
        pos = {pd.Timestamp(x): i for i, x in enumerate(pd.to_datetime(d["trade_date"]))}
        out = []
        for ev_d, an_d in pairs:
            i, j = pos.get(pd.Timestamp(ev_d)), pos.get(pd.Timestamp(an_d))
            if i is not None and j is not None:
                out.append(i - j)
        return out
    except Exception:
        return []


def existing_event_distances(events_path: str, log):
    """重算既有事件表的事件日距锚点交易日距离（读 trade_date 列定位行号）。"""
    ev = pd.read_parquet(events_path)
    if len(ev) == 0:
        return np.array([])
    tasks = []
    for code, g in ev.groupby("ts_code"):
        tasks.append((os.path.join(DATA_DIR, f"{code}.parquet"),
                      list(zip(g["event_date"], g["anchor_date"]))))

    dists = []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for i, out in enumerate(
                pool.imap_unordered(existing_distance_worker, tasks, chunksize=64)):
            dists.extend(out)
            if (i + 1) % 500 == 0:
                log(f"心跳: 既有事件距离重算 {i + 1}/{len(tasks)}")
    return np.asarray(dists, dtype=np.float64)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"实验B扫描启动 窗口=[{DATE_LO.date()}..{DATE_HI.date()}]")
    files = sorted(
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    log(f"文件总数: {len(files)}")

    # ---------------- 第一阶段：逐股并行（复算对拍 + 复现 + 扫描） ----------------
    all_events = []
    v1_pairs_got = set()
    diff_by_path = {}
    n_stock = n_skip = n_err = 0
    dropped_datatail_total = 0
    hard_diff_fail = []
    errs = []
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for i, (path, ts_code, v1_pairs, diffs, evs, dropped_tail, err) in enumerate(
                pool.imap_unordered(process_stock, files, chunksize=32)):
            if err is not None:
                n_err += 1
                errs.append(err)
            elif ts_code is None:
                n_skip += 1
            else:
                n_stock += 1
                v1_pairs_got.update(v1_pairs)
                diff_by_path[path] = diffs
                if diffs[0] > TOL or diffs[1] > TOL:
                    hard_diff_fail.append((ts_code, diffs[0], diffs[1]))
                all_events.extend(evs)
                dropped_datatail_total += dropped_tail
            if (i + 1) % 500 == 0:
                log(f"心跳: {i + 1}/{len(files)} 文件, 事件 {len(all_events)}, "
                    f"{time.time() - t0:.0f}s")
    log(f"第一阶段完成: 个股 {n_stock}, 跳过 {n_skip}, 异常 {n_err}, "
        f"事件(窗口内) {len(all_events)}, B2出窗丢弃 {dropped_datatail_total}, "
        f"耗时 {time.time() - t0:.0f}s")
    for e in errs[:5]:
        log("异常样例:\n" + e)

    # ---------------- 自检闸门 ----------------
    gate = {}
    # 闸门一：确定性 20 股抽样（等距取样，跳过未处理文件直至凑满 20 股），复算 vs talib 偏差 <= 1e-9
    step = max(1, len(files) // 20)
    sampled = []
    for i in range(0, len(files), step):
        p = files[i]
        if p in diff_by_path:
            sampled.append((p, diff_by_path[p]))
        if len(sampled) >= 20:
            break
    max_d_dif = max((d[0] for _, d in sampled), default=np.nan)
    max_d_dea = max((d[1] for _, d in sampled), default=np.nan)
    gate1_pass = (len(hard_diff_fail) == 0 and n_err == 0
                  and max_d_dif <= TOL and max_d_dea <= TOL)
    gate["dif_dea_recompute"] = {
        "sampled_n": len(sampled), "sampled_max_abs_diff_dif": max_d_dif,
        "sampled_max_abs_diff_dea": max_d_dea,
        "full_market_n_stocks_over_tol": len(hard_diff_fail),
        "n_errors": n_err, "pass": bool(gate1_pass),
    }
    log(f"闸门一(复算对拍): 抽样 {len(sampled)} 股, 最大偏差 DIF={max_d_dif:.3e} "
        f"DEA={max_d_dea:.3e}, 全市场超差股数 {len(hard_diff_fail)}, 异常 {n_err} -> "
        f"{'PASS' if gate1_pass else 'FAIL'}")

    # 闸门二：零容差复现 events_v1
    ev1 = pd.read_parquet(os.path.join(EXP_DIR, "events_v1.parquet"))
    v1_pairs_expected = set(
        (str(r.ts_code), pd.Timestamp(r.cross_date).strftime("%Y-%m-%d"))
        for r in ev1.itertuples())
    missing = sorted(v1_pairs_expected - v1_pairs_got)
    extra = sorted(v1_pairs_got - v1_pairs_expected)
    gate2_pass = (len(missing) == 0 and len(extra) == 0)
    gate["v1_reproduction"] = {
        "expected_pairs": len(v1_pairs_expected), "reproduced_pairs": len(v1_pairs_got),
        "missing": len(missing), "extra": len(extra),
        "missing_sample": missing[:5], "extra_sample": extra[:5],
        "pass": bool(gate2_pass),
    }
    log(f"闸门二(零容差复现 events_v1): 期望 {len(v1_pairs_expected)} 对, "
        f"复现 {len(v1_pairs_got)} 对, 缺失 {len(missing)}, 多余 {len(extra)} -> "
        f"{'PASS' if gate2_pass else 'FAIL'}")

    if not (gate1_pass and gate2_pass):
        log("自检闸门未过，停止：不出事件表与收益表")
        verdict = {"gate": gate, "verdict": "自检闸门未过，管线 bug，停止",
                   "elapsed_sec": round(time.time() - t0, 1)}
        with open(os.path.join(OUT_DIR, "verdict.json"), "w", encoding="utf-8") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        sys.exit(1)

    # ---------------- 事件表落盘 ----------------
    ev_all = pd.DataFrame(all_events)
    if len(ev_all) == 0:
        ev_all = pd.DataFrame(columns=["flavor", "j"] + EVENT_COLUMNS)
    frames = {}
    for flavor in ("b1", "b2"):
        for j in JS:
            key = f"{flavor}_j{j}"
            sub = ev_all[ev_all["flavor"] == flavor]
            sub = sub[sub["j"] == j].drop(columns=["flavor"])
            if len(sub):
                sub = sub.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
            sub = sub.reindex(columns=EVENT_COLUMNS)
            frames[key] = sub
            pq = os.path.join(OUT_DIR, f"events_{key}.parquet")
            sub.to_parquet(pq, index=False)
            log(f"{pq} 行数={len(sub)}")

    # ---------------- 基线引用与既有事件距离重算 ----------------
    with open(os.path.join(EXP_DIR, "forward_returns.json"), encoding="utf-8") as f:
        fwd_ref = json.load(f)
    EXISTING_EVENT_MEAN10["b1"] = fwd_ref["v1"]["event"]["10"]["mean"]
    EXISTING_EVENT_MEAN10["b2"] = fwd_ref["v2"]["event"]["10"]["mean"]
    log(f"判活线对照(+10均值%): 既有变体1事件日 {EXISTING_EVENT_MEAN10['b1']:.4f}, "
        f"既有变体2事件日 {EXISTING_EVENT_MEAN10['b2']:.4f}")

    dist_v1 = existing_event_distances(os.path.join(EXP_DIR, "events_v1.parquet"), log)
    dist_v2 = existing_event_distances(os.path.join(EXP_DIR, "events_v2.parquet"), log)
    log(f"既有变体1 事件日距锚点距离: 中位 {np.median(dist_v1):.0f}, "
        f"均值 {dist_v1.mean():.2f} (n={len(dist_v1)})")
    log(f"既有变体2 事件日距锚点距离: 中位 {np.median(dist_v2):.0f}, "
        f"均值 {dist_v2.mean():.2f} (n={len(dist_v2)})")

    # ---------------- 汇总表（配置为行） ----------------
    rows = []
    verdict_rows = {}
    for key, sub in frames.items():
        flavor = key.split("_")[0]
        row = {"配置": key, "数据来源": "本实验扫描"}
        row["事件数"] = int(len(sub))
        row["覆盖股票数"] = int(sub["ts_code"].nunique()) if len(sub) else 0
        if len(sub):
            row["后续金叉完成率%"] = float(sub["completed"].mean() * 100)
            row["升级为变体1事件率%"] = float(sub["became_v1_event"].mean() * 100)
            row["事件日距锚点距离中位"] = float(sub["dist_event_anchor_trade_days"].median())
            row["事件日距锚点距离均值"] = float(sub["dist_event_anchor_trade_days"].mean())
            row["事件日不早于后续金叉笔数"] = int(
                (sub["lead_days_vs_next_cross"] <= 0).sum())
        else:
            row["后续金叉完成率%"] = row["升级为变体1事件率%"] = None
            row["事件日距锚点距离中位"] = row["事件日距锚点距离均值"] = None
            row["事件日不早于后续金叉笔数"] = 0
        metrics = {}
        for h in HORIZONS:
            s, r = summarize_horizon(sub, h)
            row[f"+{h}日样本数"] = s["n"]
            row[f"+{h}日均值%"] = s["mean"]
            row[f"+{h}日中位%"] = s["median"]
            row[f"+{h}日胜率%"] = s["win"]
            metrics[h] = s
            if h in (10, 20) and len(sub):
                mask = sub[f"fwd_ret_{h}"].notna()
                t_val = cluster_t(
                    sub.loc[mask, f"fwd_ret_{h}"].to_numpy(dtype=np.float64),
                    sub.loc[mask, "event_date"].to_numpy())
                row[f"+{h}日聚类t"] = t_val
                metrics[f"t{h}"] = t_val
            elif h in (10, 20):
                row[f"+{h}日聚类t"] = None
                metrics[f"t{h}"] = None
        rows.append(row)

        m10 = metrics[10]["mean"]
        t10 = metrics.get("t10")
        comp = row["后续金叉完成率%"]
        threshold = EXISTING_EVENT_MEAN10[flavor] + 2.0
        checks = {
            "+10均值>0": bool(m10 is not None and m10 > 0),
            "+10聚类t>=2": bool(t10 is not None and not np.isnan(t10) and t10 >= 2),
            f"+10均值>=同风味既有+2pp(阈值{threshold:.4f}%)": bool(
                m10 is not None and m10 >= threshold),
            "后续金叉完成率>=70%": bool(comp is not None and comp >= 70.0),
        }
        alive = all(checks.values())
        verdict_rows[key] = {
            "事件数": row["事件数"], "覆盖股票数": row["覆盖股票数"],
            "后续金叉完成率%": comp,
            "+10均值%": m10, "+10聚类t": t10,
            "+20均值%": metrics[20]["mean"], "+20聚类t": metrics.get("t20"),
            "判活检查": checks, "有戏": alive,
        }
        rows[-1]["判活"] = "有戏" if alive else "不过线"

    # 基线行：全市场无条件前瞻收益（引用 forward_returns.json baseline 块）
    base_row = {"配置": "基线_全市场无条件", "数据来源": "forward_returns.json baseline 块",
                "事件数": None, "覆盖股票数": None, "后续金叉完成率%": None,
                "升级为变体1事件率%": None, "事件日距锚点距离中位": None,
                "事件日距锚点距离均值": None, "事件日不早于后续金叉笔数": None,
                "判活": "—"}
    for h in HORIZONS:
        s = fwd_ref["baseline"][str(h)]
        base_row[f"+{h}日样本数"] = s["n"]
        base_row[f"+{h}日均值%"] = s["mean"]
        base_row[f"+{h}日中位%"] = s["median"]
        base_row[f"+{h}日胜率%"] = s["win"]
        base_row[f"+{h}日聚类t"] = None
    rows.append(base_row)

    # 基线行：既有变体 1 / 变体 2 事件日前瞻收益（引用 forward_returns.json，距离重算）
    for tag, vname, dists, ev_path in (
            ("既有变体1事件日", "v1", dist_v1, "events_v1.parquet"),
            ("既有变体2事件日", "v2", dist_v2, "events_v2.parquet")):
        ev_ref = pd.read_parquet(os.path.join(EXP_DIR, ev_path))
        r = {"配置": tag,
             "数据来源": f"forward_returns.json {vname}.event 块; 距离为本脚本重算",
             "事件数": int(len(ev_ref)),
             "覆盖股票数": int(ev_ref["ts_code"].nunique()),
             "后续金叉完成率%": None, "升级为变体1事件率%": None,
             "事件日距锚点距离中位": float(np.median(dists)) if len(dists) else None,
             "事件日距锚点距离均值": float(dists.mean()) if len(dists) else None,
             "事件日不早于后续金叉笔数": None, "判活": "—"}
        for h in HORIZONS:
            s = fwd_ref[vname]["event"][str(h)]
            r[f"+{h}日样本数"] = s["n"]
            r[f"+{h}日均值%"] = s["mean"]
            r[f"+{h}日中位%"] = s["median"]
            r[f"+{h}日胜率%"] = s["win"]
            r[f"+{h}日聚类t"] = None
        rows.append(r)

    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "forward_table.csv")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"汇总表 -> {csv_path}")

    alive_keys = [k for k, v in verdict_rows.items() if v["有戏"]]
    verdict = {
        "gate": gate,
        "判活线": {
            "规则": "+10均值>0 且 +10聚类t>=2 且 +10均值>=同风味既有事件日+10均值+2pp "
                    "且 后续金叉完成率>=70%",
            "对照阈值%": {"b1": EXISTING_EVENT_MEAN10["b1"] + 2.0,
                         "b2": EXISTING_EVENT_MEAN10["b2"] + 2.0},
        },
        "逐行": verdict_rows,
        "总裁决": ("过线配置: " + ",".join(alive_keys)) if alive_keys else "六行全部不过线，判死",
        "披露": ["单段 2026 沙盒结果，不构成采纳定义的充分条件",
                 "本实验为新触发器定义，不是既有金叉对金叉定义的优化",
                 f"B2 锚点确认超出数据末尾丢弃 {dropped_datatail_total} 起（不计入完成率分母）",
                 "后续金叉完成率受数据末尾截断影响：尾区间内开火的事件机械性记为未完成"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    log(f"总裁决: {verdict['总裁决']}")
    log(f"verdict.json -> {os.path.join(OUT_DIR, 'verdict.json')}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"全部完成 耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
