"""
阶段底标注测量：变体1（区间最低价锚定） vs 变体2（右侧确认精确锚定）
裁判 = 阶段底（±60 / ±120 交易日窗口），配方沿用 ../evaluate.py，仅窗口参数不同。

口径见本目录 README.md（预登记，2026-09-04）。

输入：../events_v1.parquet / ../events_v2.parquet / ../ground_truth_lows_2026.parquet /
      ../eval_summary.json（±20 既有结果，仅抄入对照列）
输出：eval_summary_stage.json + report.md + stdout 对照表；心跳写 progress.log
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/home/karl/repos/personal/stock_qt_nd/stock_data/daily")
EXP_DIR = Path("/home/karl/repos/personal/stock_qt_nd/experiments/divergence_anchor_eval_2026")
OUT_DIR = EXP_DIR / "stage_bottom_eval"
LOG = OUT_DIR / "progress.log"

GT_START = pd.Timestamp("2026-01-01")
WINDOWS = [60, 120]
MIN_ROWS = 100
DIST_THRESHOLDS = [3, 5, 10, 15, 20, 30]
R = 100
SEED = 20260903
CROSS_LOCAL_TH = 5   # 交叉表局部底阈值（沿用 eval_summary 口径常用档）
CROSS_STAGE_TH = 10  # 交叉表阶段底阈值
INF = 10 ** 9


def log(msg: str):
    line = f"{datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Phase 1：单遍读盘，同时建 W=60 / W=120 双窗口阶段底基准 + 日期位置表
# ---------------------------------------------------------------------------

def gt_worker(path_str: str):
    """返回 (ts_code, {W: [(date, close)]}, dates) 或 None。配方同 ../evaluate.py gt_worker。"""
    path = Path(path_str)
    try:
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "close"])
    except Exception:
        return None
    if "ts_code" not in df.columns or len(df) < MIN_ROWS:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["trade_date"]).to_numpy()
    n = len(df)
    lows_by_w: dict[int, list] = {}
    for W in WINDOWS:
        lows = []
        for i in range(W, n - W):
            c = close[i]
            win = close[i - W: i + W + 1]
            if c == win.min() and c < close[i - W: i].min():
                d = dates[i]
                if d >= GT_START.to_datetime64():
                    lows.append((d, float(c)))
        lows_by_w[W] = lows
    ts_code = str(df["ts_code"].iloc[0])
    return ts_code, lows_by_w, dates


def build_ground_truth(files: list[str]):
    lows_rows = {W: [] for W in WINDOWS}
    date_index: dict[str, np.ndarray] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(gt_worker, files, chunksize=64):
            if res is None:
                continue
            ts_code, lows_by_w, dates = res
            date_index[ts_code] = dates
            for W in WINDOWS:
                for d, c in lows_by_w[W]:
                    lows_rows[W].append((ts_code, d, c))
    gts = {}
    for W in WINDOWS:
        gt = pd.DataFrame(lows_rows[W], columns=["ts_code", "low_date", "low_close"])
        gt.to_parquet(OUT_DIR / f"stage_gt_w{W}.parquet", index=False)
        gts[W] = gt
        log(f"[gt] W={W} 真阶段底 {len(gt)} 个 / 股票 {gt['ts_code'].nunique()} 只")
    log(f"[gt] 双窗口基准建成，耗时 {time.time()-t0:.1f}s")
    return gts, date_index


# ---------------------------------------------------------------------------
# Phase 2：距离 / 覆盖率 / 随机基线（配方同 ../evaluate.py 与 ../random_baseline.py）
# ---------------------------------------------------------------------------

def score_window_variant(events: pd.DataFrame, gt: pd.DataFrame,
                         date_index: dict[str, np.ndarray], W: int,
                         keep_stage_dist: bool = False) -> dict:
    gt_by_sym = {sym: g.sort_values("low_date").reset_index(drop=True)
                 for sym, g in gt.groupby("ts_code")}
    ev_by_sym = {sym: g for sym, g in events.groupby("ts_code")}

    dist: list[int] = []
    premium: list[float] = []
    truncated_events = 0   # 事件日 > 末尾-W，截断不可评
    no_gt_events = 0       # 该股该窗口无真阶段底
    low_min_dist: list[int] = []
    scored_keys: list[tuple] = []  # (sym, event_date, anchor_date) 与 dist 对齐

    # 覆盖率：每个真阶段底到最近可评分锚点的距离
    for sym, g in gt_by_sym.items():
        dates = date_index.get(sym)
        if dates is None:
            continue
        pos = {d: i for i, d in enumerate(dates)}
        low_pos = np.array([pos[d] for d in g["low_date"].to_numpy()])
        ev = ev_by_sym.get(sym)
        anchor_pos = np.array([], dtype=np.int64)
        if ev is not None:
            cutoff = len(dates) - 1 - W
            aps = []
            for e_date, a_date in zip(ev["event_date"].to_numpy(), ev["anchor_date"].to_numpy()):
                ep = pos.get(pd.Timestamp(e_date).to_datetime64(), -1)
                if ep < 0 or ep > cutoff:
                    continue
                ap = pos.get(pd.Timestamp(a_date).to_datetime64(), -1)
                if ap >= 0:
                    aps.append(ap)
            anchor_pos = np.array(aps, dtype=np.int64)
        for lp in low_pos:
            if len(anchor_pos):
                low_min_dist.append(int(np.abs(anchor_pos - lp).min()))
            else:
                low_min_dist.append(INF)

    for sym, ev in ev_by_sym.items():
        dates = date_index.get(sym)
        if dates is None:
            truncated_events += len(ev)
            continue
        pos = {d: i for i, d in enumerate(dates)}
        cutoff = len(dates) - 1 - W
        g = gt_by_sym.get(sym)
        has_gt = g is not None and len(g) > 0
        if has_gt:
            low_pos = np.array([pos[d] for d in g["low_date"].to_numpy()])
            low_close = g["low_close"].to_numpy()
            low_dates = g["low_date"].to_numpy()
        n_no_gt_this = 0
        for e_date, a_date, a_close in zip(ev["event_date"].to_numpy(),
                                           ev["anchor_date"].to_numpy(),
                                           ev["anchor_close"].to_numpy()):
            ep = pos.get(pd.Timestamp(e_date).to_datetime64(), -1)
            if ep < 0 or ep > cutoff:
                truncated_events += 1
                continue
            if not has_gt:
                n_no_gt_this += 1
                continue
            ap = pos.get(pd.Timestamp(a_date).to_datetime64(), -1)
            if ap < 0:
                truncated_events += 1
                continue
            d = np.abs(low_pos - ap)
            j = int(np.argmin(d))
            ties = np.flatnonzero(d == d[j])  # 并列取日期更早者（确定性）
            if len(ties) > 1:
                j = int(ties[np.argsort(low_dates[ties])[0]])
            dist.append(int(d[j]))
            premium.append(float(a_close / low_close[j] - 1.0))
            scored_keys.append((sym, pd.Timestamp(e_date), pd.Timestamp(a_date)))
        no_gt_events += n_no_gt_this

    dist_arr = np.array(dist)
    prem_arr = np.array(premium)
    low_min_arr = np.array(low_min_dist)
    n_lows = len(low_min_arr)

    result = {
        "window": W,
        "events_total": int(len(events)),
        "symbols_with_events": int(events["ts_code"].nunique()),
        "events_truncated": int(truncated_events),
        "events_no_gt_low": int(no_gt_events),
        "events_scored": int(len(dist_arr)),
        "dist_median": float(np.median(dist_arr)) if len(dist_arr) else None,
        "dist_mean": float(np.mean(dist_arr)) if len(dist_arr) else None,
        "premium_median_pct": float(np.median(prem_arr) * 100) if len(prem_arr) else None,
        "premium_mean_pct": float(np.mean(prem_arr) * 100) if len(prem_arr) else None,
        "gt_lows_total": int(n_lows),
    }
    for th in DIST_THRESHOLDS:
        result[f"dist_le{th}_pct"] = float(np.mean(dist_arr <= th) * 100) if len(dist_arr) else None
        result[f"capture_le{th}_pct"] = float(np.mean(low_min_arr <= th) * 100) if n_lows else None
        result[f"captured_le{th}_count"] = int(np.sum(low_min_arr <= th))
    if keep_stage_dist:
        result["_scored"] = (scored_keys, dist_arr)
    return result


def random_baseline(events: pd.DataFrame, gt: pd.DataFrame,
                    date_index: dict[str, np.ndarray], W: int,
                    events_scored: int, rng: np.random.Generator) -> dict:
    """同 ../random_baseline.py 配方：每股有效区间 [2026-01-01, 末尾-W) 均匀随机 m 锚，R 次取均值。
    m = 该股该窗口可评分事件数（截断后口径）。输出随机命中率(dist)与随机覆盖率(capture)。"""
    gt_by_sym = {sym: g.sort_values("low_date").reset_index(drop=True)
                 for sym, g in gt.groupby("ts_code")}

    # 每股可评分事件数 m
    m_by_sym: dict[str, int] = {}
    for sym, ev in events.groupby("ts_code"):
        dates = date_index.get(sym)
        if dates is None:
            continue
        pos = {d: i for i, d in enumerate(dates)}
        cutoff = len(dates) - 1 - W
        m = 0
        for e_date in ev["event_date"].to_numpy():
            ep = pos.get(pd.Timestamp(e_date).to_datetime64(), -1)
            if 0 <= ep <= cutoff:
                m += 1
        if m:
            m_by_sym[sym] = m

    captured_total = {th: 0.0 for th in DIST_THRESHOLDS}
    hit_total = {th: 0.0 for th in DIST_THRESHOLDS}
    draw_total = 0
    n_lows = 0
    for sym, g in gt_by_sym.items():
        dates = date_index.get(sym)
        if dates is None:
            continue
        pos = {d: i for i, d in enumerate(dates)}
        low_pos = np.array([pos[d.to_datetime64()] for d in g["low_date"]])
        m = m_by_sym.get(sym, 0)
        n_lows += len(low_pos)
        if m == 0:
            continue
        hi = len(dates) - W
        lo = int(np.searchsorted(dates, np.datetime64("2026-01-01")))
        if hi <= lo:
            continue
        draws = rng.integers(lo, hi, size=(R, m))
        # 每个低点 ±th 内是否有随机锚（覆盖率）
        cap_r = {th: np.zeros(len(low_pos)) for th in DIST_THRESHOLDS}
        # 每个随机锚到最近低点的距离（命中率）
        hit_r = {th: 0.0 for th in DIST_THRESHOLDS}
        for r in range(R):
            dmat = np.abs(draws[r][:, None] - low_pos[None, :])
            d_low = dmat.min(axis=0)
            d_anchor = dmat.min(axis=1)
            for th in DIST_THRESHOLDS:
                cap_r[th] += (d_low <= th)
                hit_r[th] += float(np.sum(d_anchor <= th))
        for th in DIST_THRESHOLDS:
            captured_total[th] += (cap_r[th] / R).sum()
            hit_total[th] += hit_r[th] / R
        draw_total += m

    out = {"n_lows": int(n_lows), "random_anchors_per_rep_total": int(draw_total)}
    for th in DIST_THRESHOLDS:
        out[f"random_capture_le{th}_pct"] = (
            float(captured_total[th] / n_lows * 100) if n_lows else None)
        out[f"random_dist_le{th}_pct"] = (
            float(hit_total[th] / draw_total * 100) if draw_total else None)
    return out


# ---------------------------------------------------------------------------
# Phase 3：2×2 交叉表（±20 局部底 ≤5 天 × ±60 阶段底 ≤10 天）
# ---------------------------------------------------------------------------

def cross_table(events: pd.DataFrame, local_gt: pd.DataFrame, stage_gt: pd.DataFrame,
                date_index: dict[str, np.ndarray], W: int) -> dict:
    local_by_sym = {sym: g.sort_values("low_date").reset_index(drop=True)
                    for sym, g in local_gt.groupby("ts_code")}
    stage_by_sym = {sym: g.sort_values("low_date").reset_index(drop=True)
                    for sym, g in stage_gt.groupby("ts_code")}

    cells = {"local_le5_stage_le10": 0, "local_le5_stage_gt10": 0,
             "local_gt5_stage_le10": 0, "local_gt5_stage_gt10": 0}
    n_in = n_trunc = n_no_local = n_no_stage = 0
    for sym, ev in events.groupby("ts_code"):
        dates = date_index.get(sym)
        if dates is None:
            n_trunc += len(ev)
            continue
        pos = {d: i for i, d in enumerate(dates)}
        cutoff = len(dates) - 1 - W
        lg = local_by_sym.get(sym)
        sg = stage_by_sym.get(sym)
        local_pos = (np.array([pos[d] for d in lg["low_date"].to_numpy()])
                     if lg is not None and len(lg) else None)
        stage_pos = (np.array([pos[d] for d in sg["low_date"].to_numpy()])
                     if sg is not None and len(sg) else None)
        for e_date, a_date in zip(ev["event_date"].to_numpy(), ev["anchor_date"].to_numpy()):
            ep = pos.get(pd.Timestamp(e_date).to_datetime64(), -1)
            if ep < 0 or ep > cutoff:
                n_trunc += 1
                continue
            ap = pos.get(pd.Timestamp(a_date).to_datetime64(), -1)
            if ap < 0:
                n_trunc += 1
                continue
            if local_pos is None:
                n_no_local += 1
                continue
            if stage_pos is None:
                n_no_stage += 1
                continue
            dl = int(np.abs(local_pos - ap).min())
            ds = int(np.abs(stage_pos - ap).min())
            n_in += 1
            if dl <= CROSS_LOCAL_TH and ds <= CROSS_STAGE_TH:
                cells["local_le5_stage_le10"] += 1
            elif dl <= CROSS_LOCAL_TH:
                cells["local_le5_stage_gt10"] += 1
            elif ds <= CROSS_STAGE_TH:
                cells["local_gt5_stage_le10"] += 1
            else:
                cells["local_gt5_stage_gt10"] += 1
    return {"window": W, "cells": cells, "events_in_table": n_in,
            "events_truncated": n_trunc, "events_no_local_low": n_no_local,
            "events_no_stage_low": n_no_stage}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_all = time.time()
    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    log(f"[load] 日线文件 {len(files)} 个，开始单遍建双窗口阶段底基准")

    gts, date_index = build_ground_truth(files)

    events = {name: pd.read_parquet(EXP_DIR / f"events_{name}.parquet") for name in ["v1", "v2"]}
    local_gt = pd.read_parquet(EXP_DIR / "ground_truth_lows_2026.parquet")
    local_gt["low_date"] = pd.to_datetime(local_gt["low_date"])
    prev = json.loads((EXP_DIR / "eval_summary.json").read_text(encoding="utf-8"))

    rng = np.random.default_rng(SEED)
    results: dict = {}
    for W in WINDOWS:
        for name in ["v1", "v2"]:
            t0 = time.time()
            r = score_window_variant(events[name], gts[W], date_index, W)
            log(f"[score] W={W} {name}: 可评 {r['events_scored']}/{r['events_total']}，"
                f"截断 {r['events_truncated']}，无基准 {r['events_no_gt_low']}，"
                f"耗时 {time.time()-t0:.1f}s")
            t0 = time.time()
            rb = random_baseline(events[name], gts[W], date_index, W,
                                 r["events_scored"], rng)
            r.update(rb)
            log(f"[baseline] W={W} {name}: 随机基线完成（R={R}），耗时 {time.time()-t0:.1f}s")
            results[f"w{W}_{name}"] = r

    crosses = {}
    for name in ["v1", "v2"]:
        crosses[name] = cross_table(events[name], local_gt, gts[60], date_index, 60)
        log(f"[cross] {name} 2×2 交叉表完成：{crosses[name]['cells']}")

    out = {
        "ground_truth": ("close[t] == min(close[t-W..t+W]) 且严格低于前W日；"
                         "t∈[2026-01-01, 数据末尾-W交易日]；W∈{60,120}"),
        "preregistered_criterion": ("±60 窗口 dist_le10_pct / 随机基线 dist_le10_pct："
                                    "≥2倍=有；1.2~2倍=弱；≤1.2倍=无。±120 仅参考。"),
        "thresholds": DIST_THRESHOLDS,
        "random_reps": R,
        "random_seed": SEED,
        "cross_table_thresholds": {"local_le": CROSS_LOCAL_TH, "stage_le": CROSS_STAGE_TH},
        "results": results,
        "cross_table_w60": crosses,
    }
    # 去掉内部键
    for k in results:
        results[k].pop("_scored", None)
    (OUT_DIR / "eval_summary_stage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[done] 写入 eval_summary_stage.json；总耗时 {time.time()-t_all:.1f}s")

    write_report(out, prev)
    log("[done] 写入 report.md")


def fmt(v, spec):
    return spec.format(v) if v is not None else "—"


def write_report(out: dict, prev: dict):
    res = out["results"]
    lines = []
    lines.append("# 阶段底标注测量报告（预登记纯测量，2026-09-04）")
    lines.append("")
    lines.append(f"Ground truth：{out['ground_truth']}")
    lines.append("")
    lines.append(f"判据：{out['preregistered_criterion']}")
    lines.append("")
    lines.append("## 1. 样本与截断披露（配置为行）")
    lines.append("")
    lines.append("| 配置 | 事件总数 | 截断不可评 | 无基准低点 | 可评分 | 可评占比(%) | 真阶段底总数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, label in [("w60_v1", "W60×变体1"), ("w60_v2", "W60×变体2"),
                       ("w120_v1", "W120×变体1"), ("w120_v2", "W120×变体2")]:
        r = res[key]
        pct = r["events_scored"] / r["events_total"] * 100
        lines.append(f"| {label} | {r['events_total']} | {r['events_truncated']} | "
                     f"{r['events_no_gt_low']} | {r['events_scored']} | {pct:.1f} | "
                     f"{r['gt_lows_total']} |")
    lines.append("")
    lines.append("## 2. 距离与覆盖指标（配置为行，随机基线同列并排）")
    lines.append("")
    header = ["配置", "dist中位", "dist均值", "溢价中位(%)", "溢价均值(%)"]
    for th in DIST_THRESHOLDS:
        header += [f"dist≤{th}(%)", f"随机≤{th}(%)", f"倍数≤{th}"]
    for th in DIST_THRESHOLDS:
        header += [f"capture≤{th}(%)", f"随机cap≤{th}(%)"]
    lines.append("|" + "|".join(header) + "|")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    verdicts = {}
    for key, label in [("w60_v1", "W60×变体1"), ("w60_v2", "W60×变体2"),
                       ("w120_v1", "W120×变体1"), ("w120_v2", "W120×变体2")]:
        r = res[key]
        row = [label, fmt(r["dist_median"], "{:.1f}"), fmt(r["dist_mean"], "{:.2f}"),
               fmt(r["premium_median_pct"], "{:.2f}"), fmt(r["premium_mean_pct"], "{:.2f}")]
        for th in DIST_THRESHOLDS:
            a = r[f"dist_le{th}_pct"]
            b = r[f"random_dist_le{th}_pct"]
            ratio = (a / b) if (a is not None and b) else None
            if th == 10:
                verdicts[key] = (a, b, ratio)
            row += [fmt(a, "{:.1f}"), fmt(b, "{:.1f}"), fmt(ratio, "{:.2f}")]
        for th in DIST_THRESHOLDS:
            row += [fmt(r[f"capture_le{th}_pct"], "{:.1f}"),
                    fmt(r[f"random_capture_le{th}_pct"], "{:.1f}")]
        lines.append("|" + "|".join(row) + "|")
    lines.append("")
    lines.append("（倍数≤N = 变体 dist_leN_pct / 随机基线 dist_leN_pct；随机基线 R="
                 f"{out['random_reps']}，种子 {out['random_seed']}，"
                 "每股有效区间均匀随机 m 锚，m=该股该窗口可评分事件数。）")
    lines.append("")
    lines.append("## 3. 判据宣判（±60 窗口 dist_le10_pct vs 随机基线）")
    lines.append("")
    lines.append("| 配置 | dist≤10(%) | 随机基线(%) | 倍数 | 宣判 |")
    lines.append("|---|---|---|---|---|")
    for key, label in [("w60_v1", "W60×变体1"), ("w60_v2", "W60×变体2")]:
        a, b, ratio = verdicts[key]
        if ratio is None:
            verdict = "不可判"
        elif ratio >= 2.0:
            verdict = "有阶段底标注能力"
        elif ratio > 1.2:
            verdict = "弱"
        else:
            verdict = "无（A 方向封档依据）"
        lines.append(f"| {label} | {a:.1f} | {b:.1f} | {ratio:.2f} | {verdict} |")
    lines.append("")
    lines.append("±120 窗口因样本截断只作参考读数，不参与宣判。")
    lines.append("")
    lines.append("## 4. 2×2 交叉表（W60 可评分事件：锚点距±20局部底≤5天 × 距±60阶段底≤10天）")
    lines.append("")
    for name, label in [("v1", "变体1"), ("v2", "变体2")]:
        c = out["cross_table_w60"][name]
        cells = c["cells"]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| | 阶段底≤10天 | 阶段底>10天 | 行合计 |")
        lines.append("|---|---|---|---|")
        a = cells["local_le5_stage_le10"]; b = cells["local_le5_stage_gt10"]
        d = cells["local_gt5_stage_le10"]; e = cells["local_gt5_stage_gt10"]
        lines.append(f"| 局部底≤5天 | {a} | {b} | {a+b} |")
        lines.append(f"| 局部底>5天 | {d} | {e} | {d+e} |")
        lines.append(f"| 列合计 | {a+d} | {b+e} | {a+b+d+e} |")
        lines.append("")
        lines.append(f"入表事件 {c['events_in_table']}；截断 {c['events_truncated']}；"
                     f"无局部底基准 {c['events_no_local_low']}；"
                     f"无阶段底基准 {c['events_no_stage_low']}。")
        lines.append("")
        tot_local = a + b
        if tot_local:
            lines.append(f"v6 逮住的局部底（锚点距局部底≤5天）共 {tot_local} 个，"
                         f"其中贴阶段底（≤10天）{a} 个（{a/tot_local*100:.1f}%），"
                         f"下跌中继（>10天）{b} 个（{b/tot_local*100:.1f}%）。")
            lines.append("")
    lines.append("## 5. 与 ±20 局部底既有结果并排对照（eval_summary.json 原样抄入）")
    lines.append("")
    lines.append("| 指标 | ±20变体1 | ±20变体2 | ±60变体1 | ±60变体2 | ±120变体1 | ±120变体2 |")
    lines.append("|---|---|---|---|---|---|---|")
    common_th = [3, 5, 10, 15, 20]
    for th in common_th:
        pk = f"dist_le{th}_pct"
        row = [f"dist≤{th}(%)",
               fmt(prev["v1"].get(pk), "{:.1f}"), fmt(prev["v2"].get(pk), "{:.1f}"),
               fmt(res["w60_v1"][pk], "{:.1f}"), fmt(res["w60_v2"][pk], "{:.1f}"),
               fmt(res["w120_v1"][pk], "{:.1f}"), fmt(res["w120_v2"][pk], "{:.1f}")]
        lines.append("|" + "|".join(row) + "|")
    for th in common_th:
        pk = f"capture_le{th}_pct"
        row = [f"capture≤{th}(%)",
               fmt(prev["v1"].get(pk), "{:.1f}"), fmt(prev["v2"].get(pk), "{:.1f}"),
               fmt(res["w60_v1"][pk], "{:.1f}"), fmt(res["w60_v2"][pk], "{:.1f}"),
               fmt(res["w120_v1"][pk], "{:.1f}"), fmt(res["w120_v2"][pk], "{:.1f}")]
        lines.append("|" + "|".join(row) + "|")
    lines.append("")
    lines.append("注：±20 列为 2026-09-03 既有结果（窗口半径20，基准低点总数 12221，"
                 "可评分事件 v1=5621/v2=5171）；±60/±120 列为本实验同配方换窗重打分，"
                 "样本截断不同，分母不同，直接比数需谨慎。")
    lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
