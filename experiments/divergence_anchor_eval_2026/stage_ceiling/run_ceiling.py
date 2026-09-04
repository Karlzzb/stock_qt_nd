#!/usr/bin/env python3
"""阶段底收益天花板测量（实验A生死闸门）—— 预登记 = 同目录 README.md（先于跑数落盘）。

测量"完美识别锚点贴阶段底"的收益上界：分组标签为事后信息（上界假设），
入场/持有/出场完全因果。非可交易策略回测（README 显著警告）。

口径要点（逐条对应 README）：
  分组：每股日线序列 = load_market_data 返回的该股序列（含引擎自带 60 自然日前向缓冲，
  实际自 2025-11-02 起），按位置计下标；i_ev < len(序列) - W 才可评分；
  dist = min over 该股该窗口全部阶段底低点 |i_anchor - i_low|；dist<=10 near，否则 far。
  入场：事件日下一交易日（日历）开盘买；无报价/开盘涨停拒买不递补；整手/现金逐条复刻冻结引擎。
  出场：入场日记第 1 日，第 H 个交易日（日历口径，停牌日计入持有日但不评估）收盘卖；
  收盘跌停顺延至下一非跌停收盘日；数据耗尽 -> incomplete。
  成本：buy_cost/sell_costs/SLIPPAGE/BOARD_LOT 直接调冻结引擎原语。

产物：trades_ceiling.parquet（每事件 x 变体 x W x H 一行）、summary_ceiling.csv（24 行）、
verdict.json、report.md、progress.log（心跳）。
全程禁网络；除本目录外仓库只读。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))

import strategy_engine as se  # noqa: E402  v1 冻结引擎（只读复用原语：加载/成本/常量）

EXP_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026")
OUT_DIR = os.path.join(EXP_DIR, "stage_ceiling")
EVENTS = {
    "v1": os.path.join(EXP_DIR, "events_v1.parquet"),
    "v2": os.path.join(EXP_DIR, "events_v2.parquet"),
}
GT = {
    60: os.path.join(EXP_DIR, "stage_bottom_eval", "stage_gt_w60.parquet"),
    120: os.path.join(EXP_DIR, "stage_bottom_eval", "stage_gt_w120.parquet"),
}
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

START, END = "2026-01-01", "2026-08-31"
BUDGET = 100_000.0          # 每笔固定名义本金（与事件层/冻结引擎一致）
W_GRID = [60, 120]
H_GRID = [60, 120]
NEAR_TH = 10                # near/far 分界（dist <= 10 -> near）
TOL = 1e-9                  # 价格比较容差（= se.PRICE_TOL）
SAMPLE_SEED = 20260904      # 自检 3 抽样种子（预登记）
# stage_bottom_eval 既有可评分数（自检 2 对齐披露用）
REF_SCORABLE = {("v1", 60): 410, ("v2", 60): 402, ("v1", 120): 19, ("v2", 120): 17}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 数据结构
class StockBars:
    """单股行情对齐到交易日历的 numpy 数组（缺日 = NaN）。日历下标系，仅用于模拟。"""

    __slots__ = ("open_", "high", "low", "close", "up_lim", "dn_lim")

    def __init__(self, n: int) -> None:
        self.open_ = np.full(n, np.nan)
        self.high = np.full(n, np.nan)
        self.low = np.full(n, np.nan)
        self.close = np.full(n, np.nan)
        self.up_lim = np.full(n, np.nan)   # 涨停价；NaN = 当日无约束（引擎同款缺省）
        self.dn_lim = np.full(n, np.nan)


def build_bars(md: se.MarketData, cal_dt: pd.DatetimeIndex, codes: list[str]) -> dict:
    """日线向量化对齐 + 涨跌停表一次性对齐成 DataFrame 再按列取（禁逐股逐日 to_dict）。"""
    n = len(cal_dt)
    bars: dict[str, StockBars] = {}
    for code in codes:
        sb = StockBars(n)
        df = md.daily.get(code)
        if df is not None:
            loc = cal_dt.get_indexer(df.index)
            mask = loc >= 0
            idx = loc[mask]
            sub = df.iloc[np.flatnonzero(mask)]
            sb.open_[idx] = sub["open"].to_numpy(dtype=float)
            sb.high[idx] = sub["high"].to_numpy(dtype=float)
            sb.low[idx] = sub["low"].to_numpy(dtype=float)
            sb.close[idx] = sub["close"].to_numpy(dtype=float)
        bars[code] = sb
    # 涨跌停：开工时一次性对齐成两个 DataFrame（index=日历，columns=ts_code）
    up_df = pd.DataFrame({d: lim["up_limit"] for d, lim in md.limits.items()}).T
    dn_df = pd.DataFrame({d: lim["down_limit"] for d, lim in md.limits.items()}).T
    up_df = up_df.reindex(cal_dt)
    dn_df = dn_df.reindex(cal_dt)
    for code in codes:
        if code in up_df.columns:
            bars[code].up_lim = up_df[code].to_numpy(dtype=float)
            bars[code].dn_lim = dn_df[code].to_numpy(dtype=float)
    return bars


# ---------------------------------------------------------------- 分组（序列下标系）
def classify_events(ev: pd.DataFrame, W: int, posmap: dict, serlen: dict,
                    gtpos: dict) -> pd.DataFrame:
    """逐事件分组。返回列：group(near/far/NA)、dist、cls（分类状态）。
    cls 取值：scorable / unscorable_no_series / unscorable_no_event_row /
    unscorable_truncated / unscorable_anchor / no_gt。"""
    groups: list = []
    dists: list = []
    clss: list = []
    for r in ev.itertuples(index=False):
        code = r.ts_code
        pm = posmap.get(code)
        if pm is None:
            groups.append(None); dists.append(np.nan); clss.append("unscorable_no_series")
            continue
        i_ev = pm.get(r.event_date, -1)
        if i_ev < 0:
            groups.append(None); dists.append(np.nan); clss.append("unscorable_no_event_row")
            continue
        if i_ev >= serlen[code] - W:
            groups.append(None); dists.append(np.nan); clss.append("unscorable_truncated")
            continue
        i_anchor = pm.get(r.anchor_date, -1)
        if i_anchor < 0:
            groups.append(None); dists.append(np.nan); clss.append("unscorable_anchor")
            continue
        lows = gtpos.get((W, code))
        if lows is None or len(lows) == 0:
            groups.append(None); dists.append(np.nan); clss.append("no_gt")
            continue
        dist = int(np.abs(lows - i_anchor).min())
        groups.append("near" if dist <= NEAR_TH else "far")
        dists.append(dist)
        clss.append("scorable")
    out = ev[["ts_code", "event_date", "anchor_date"]].copy()
    out["group"] = groups
    out["dist"] = dists
    out["cls"] = clss
    return out


# ---------------------------------------------------------------- 逐笔模拟（因果）
def simulate(sb: StockBars, di_ev: int, H: int, cal_arr: list) -> dict:
    """入场：事件日下一交易日开盘买（无报价/涨停拒买不递补，整手现金逐条复刻冻结引擎）。
    出场：入场日记第 1 日，第 H 个交易日（日历口径）收盘卖，跌停顺延，耗尽 incomplete。"""
    n_cal = len(cal_arr)
    if di_ev + 1 >= n_cal:
        return dict(status="dropped_no_next_day")
    di_e = di_ev + 1
    o = sb.open_[di_e]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = sb.up_lim[di_e]
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    px = o * (1.0 + se.SLIPPAGE)
    sh = int(BUDGET / px / se.BOARD_LOT) * se.BOARD_LOT
    if sh < se.BOARD_LOT:
        sh = se.BOARD_LOT
    comm = se.buy_cost(sh, px)
    while sh > 0 and sh * px + comm > BUDGET + 1e-6:
        sh -= se.BOARD_LOT
        comm = se.buy_cost(sh, px) if sh > 0 else 0.0
    if sh <= 0:
        return dict(status="dropped_cash")
    base = dict(entry_date=cal_arr[di_e], entry_raw=o, entry_exec=px,
                shares=sh, buy_comm=comm)
    deferred = 0
    for di in range(di_e + H - 1, n_cal):
        c = sb.close[di]
        if not np.isfinite(c):
            continue  # 停牌日计入持有日但不评估（冻结引擎 held 口径）
        dn = sb.dn_lim[di]
        if np.isfinite(dn) and c <= dn + TOL:
            deferred += 1
            continue  # 跌停顺延至下一非跌停收盘日
        xs = c * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(sh, xs, cal_arr[di])
        gross_ret = xs / px - 1.0
        net_pnl = sh * (xs - px) - comm - xcomm - stamp
        net_ret = net_pnl / (sh * px + comm)
        return dict(status="closed", exit_date=cal_arr[di], exit_raw=c, exit_exec=xs,
                    sell_comm=xcomm, stamp=stamp, gross_ret=gross_ret,
                    net_ret=net_ret, net_pnl=net_pnl,
                    held_days=di - di_e + 1, deferred_days=deferred, **base)
    return dict(status="incomplete", deferred_days=deferred, **base)


# ---------------------------------------------------------------- 聚类稳健 t（Liang-Zeger）
def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    """单样本均值的聚类稳健 t。预登记口径：
    得分 s_i = x_i - x̄；簇得分和 S_c；var(x̄) = [G/(G-1)] x Σ S_c² / n²。"""
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    df = pd.DataFrame({"s": s, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    log(f"STAGE CEILING START window=[{START}..{END}] budget={BUDGET:.0f} "
        f"W={W_GRID} H={H_GRID} near_th={NEAR_TH} | 主格=变体 x W60 x near x H60 | "
        f"预计总时长约2-3分钟（参照 strength_follow 同机数据加载量级）")

    events = {}
    for name, path in EVENTS.items():
        ev = pd.read_parquet(path)
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        ev["anchor_date"] = pd.to_datetime(ev["anchor_date"])
        assert not ev.duplicated(["ts_code", "event_date"]).any(), f"{name} 事件键重复"
        events[name] = ev
        log(f"[load] {name}: 事件 {len(ev)} 行 / {ev['ts_code'].nunique()} 股")
    gts = {}
    for W, path in GT.items():
        gt = pd.read_parquet(path)
        gt["low_date"] = pd.to_datetime(gt["low_date"])
        gts[W] = gt
        log(f"[load] 阶段底基准 W={W}: {len(gt)} 低点 / {gt['ts_code'].nunique()} 股")

    codes = sorted(set(events["v1"]["ts_code"]) | set(events["v2"]["ts_code"]))
    t0 = time.time()
    md = se.load_market_data(codes, START, END, REPO_ROOT, log_path=LOG_PATH)
    cal_arr = list(md.calendar)
    cal_dt = pd.DatetimeIndex(cal_arr)
    cal_index = {d: i for i, d in enumerate(cal_arr)}
    cal_set = set(cal_arr)
    assert md.limit_missing_dates == 0, "本预登记假设 stk_limit 窗口内零缺日"
    log(f"[load] 行情就绪：{len(md.daily)} 股 / 日历 {len(cal_arr)} 交易日 "
        f"（{time.time() - t0:.0f}s）")

    # 事件日必须在日历内（交易模拟前提）
    for name in EVENTS:
        ev = events[name]
        in_cal = ev["event_date"].isin(cal_set)
        n_out = int((~in_cal).sum())
        if n_out:
            log(f"[warn] {name}: {n_out} 事件事件日不在日历内，剔除")
        events[name] = ev[in_cal].reset_index(drop=True)

    t0 = time.time()
    bars = build_bars(md, cal_dt, codes)
    log(f"[align] 日历对齐数组与涨跌停矩阵就绪（{time.time() - t0:.0f}s）")

    # 每股序列位置表（含引擎 60 自然日前向缓冲；分组口径见 README）
    posmap: dict[str, dict] = {}
    serlen: dict[str, int] = {}
    for code, df in md.daily.items():
        posmap[code] = {d: i for i, d in enumerate(df.index)}
        serlen[code] = len(df)
    log(f"[align] 每股序列位置表就绪：{len(posmap)} 股；"
        f"序列起点样例={min(df.index[0] for df in md.daily.values()).date()}")

    # 阶段底低点 -> 该股序列下标（posmap 仅含事件股；非事件股的低点不参与任何评分）
    event_code_set = set(codes)
    gtpos: dict[tuple, np.ndarray] = {}
    n_gt_missed_event_stock = 0   # 事件股上 low_date 落出序列（预期 0，披露）
    n_gt_skipped_non_event = 0    # 非事件股的低点（本实验用不到，披露）
    for W, gt in gts.items():
        for code, g in gt.groupby("ts_code"):
            pm = posmap.get(code)
            if pm is None:
                if code in event_code_set:
                    n_gt_missed_event_stock += len(g)
                else:
                    n_gt_skipped_non_event += len(g)
                continue
            pts = [pm[d] for d in g["low_date"] if d in pm]
            n_gt_missed_event_stock += len(g) - len(pts)
            if pts:
                gtpos[(W, code)] = np.array(sorted(pts), dtype=np.int64)
    log(f"[align] 阶段底下标映射就绪：{len(gtpos)} (W,股)；"
        f"事件股落出序列低点 {n_gt_missed_event_stock} 个（预期 0）；"
        f"非事件股低点 {n_gt_skipped_non_event} 个（不参与评分）")

    # ---------------- 分组（变体 x W）+ 模拟（变体 x 事件 x H，W 间复用） ----------------
    all_rows: list[pd.DataFrame] = []
    disc_rows: list[dict] = []
    cls_store: dict[tuple, pd.DataFrame] = {}
    for name in ["v1", "v2"]:
        ev = events[name]
        t0 = time.time()
        cls_by_w = {W: classify_events(ev, W, posmap, serlen, gtpos) for W in W_GRID}
        for W in W_GRID:
            cls_store[(name, W)] = cls_by_w[W]
            c = cls_by_w[W]
            vc = c["cls"].value_counts().to_dict()
            n_near = int((c["group"] == "near").sum())
            n_far = int((c["group"] == "far").sum())
            n_pool = n_near + n_far
            ref = REF_SCORABLE[(name, W)]
            disc_rows.append(dict(
                variant=name, W=W, events_total=len(c),
                unscorable_no_series=vc.get("unscorable_no_series", 0),
                unscorable_no_event_row=vc.get("unscorable_no_event_row", 0),
                unscorable_truncated=vc.get("unscorable_truncated", 0),
                unscorable_anchor=vc.get("unscorable_anchor", 0),
                no_gt=vc.get("no_gt", 0),
                near=n_near, far=n_far, pool_scorable=n_pool,
                ref_scorable=ref, diff_vs_ref=n_pool - ref))
            log(f"[classify] {name} W={W}: 可评分 {n_pool}（near {n_near} / far {n_far}）"
                f" | 截断 {vc.get('unscorable_truncated', 0)}"
                f" anchor落出 {vc.get('unscorable_anchor', 0)}"
                f" 无基准 {vc.get('no_gt', 0)}"
                f" 无序列 {vc.get('unscorable_no_series', 0)}"
                f" 事件日无行 {vc.get('unscorable_no_event_row', 0)}"
                f" | 对照 stage_bottom_eval {ref}（差 {n_pool - ref}）")
        # 模拟：任一 W 入组（near/far）的事件，逐事件逐 H 模拟一次
        grouped_mask = np.zeros(len(ev), dtype=bool)
        for W in W_GRID:
            grouped_mask |= cls_by_w[W]["group"].notna().to_numpy()
        di_evs = ev["event_date"].map(cal_index).to_numpy(dtype=int)
        codes_arr = ev["ts_code"].to_numpy()
        sim: dict[tuple, dict] = {}
        n_sim = 0
        for i in np.flatnonzero(grouped_mask):
            sb = bars[codes_arr[i]]
            for H in H_GRID:
                sim[(i, H)] = simulate(sb, int(di_evs[i]), H, cal_arr)
            n_sim += 1
        log(f"[simulate] {name}: {n_sim} 事件 x {len(H_GRID)} H 模拟完成"
            f"（{time.time() - t0:.0f}s）")
        # 展开成行：事件 x W x H
        for W in W_GRID:
            c = cls_by_w[W]
            recs = []
            for i in range(len(ev)):
                grp = c["group"].iat[i]
                row = dict(variant=name, W=W, ts_code=codes_arr[i],
                           event_date=ev["event_date"].iat[i],
                           anchor_date=ev["anchor_date"].iat[i],
                           group=grp if pd.notna(grp) else None,
                           dist=c["dist"].iat[i])
                if pd.isna(grp):
                    row["status"] = c["cls"].iat[i]
                    for H in H_GRID:
                        recs.append(dict(row, H=H))
                else:
                    for H in H_GRID:
                        recs.append(dict(row, H=H, **sim[(i, H)]))
            all_rows.append(pd.DataFrame(recs))
        log(f"[rows] {name}: 行展开完成（累计 {time.time() - t0:.0f}s）")

    full = pd.concat(all_rows, ignore_index=True)
    out_pq = os.path.join(OUT_DIR, "trades_ceiling.parquet")
    full.to_parquet(out_pq, index=False)
    log(f"[dump] trades_ceiling.parquet 行数={len(full)} -> {out_pq}")

    # ---------------- 自检 1：因果断言 ----------------
    entered = full[full["status"].isin({"closed", "incomplete"})]
    n_viol_entry = int((entered["entry_date"] <= entered["event_date"]).sum())
    closed_all = full[full["status"] == "closed"]
    n_viol_exit = int((closed_all["exit_date"] <= closed_all["entry_date"]).sum())
    check1_ok = (n_viol_entry == 0 and n_viol_exit == 0)
    log(f"[check1] 因果断言：entry<=event 违反 {n_viol_entry}，"
        f"exit<=entry 违反 {n_viol_exit} -> {'PASS' if check1_ok else 'FAIL'}")

    # ---------------- 汇总（配置为行，24 格） ----------------
    sum_rows: list[dict] = []
    for name in ["v1", "v2"]:
        for W in W_GRID:
            sub = full[(full["variant"] == name) & (full["W"] == W)]
            for grp in ["near", "far", "pool"]:
                gsub = sub[sub["group"] == grp] if grp != "pool" \
                    else sub[sub["group"].isin(["near", "far"])]
                for H in H_GRID:
                    gdf = gsub[gsub["H"] == H]
                    st = gdf["status"].value_counts().to_dict()
                    n_scorable = len(gdf)
                    n_closed = int(st.get("closed", 0))
                    n_incomplete = int(st.get("incomplete", 0))
                    n_traded = n_closed + n_incomplete
                    cl = gdf[gdf["status"] == "closed"]
                    ret = cl["net_ret"].to_numpy(dtype=float)
                    ct = cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) \
                        if n_closed else np.nan
                    sum_rows.append(dict(
                        variant=name, W=W, group=grp, H=H,
                        n_scorable=n_scorable, n_traded=n_traded, n_closed=n_closed,
                        n_incomplete=n_incomplete,
                        n_dropped_limitup=int(st.get("dropped_limitup", 0)),
                        n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
                        n_dropped_cash=int(st.get("dropped_cash", 0)),
                        net_mean=float(ret.mean()) if n_closed else np.nan,
                        net_median=float(np.median(ret)) if n_closed else np.nan,
                        win_rate=float((ret > 0).mean()) if n_closed else np.nan,
                        cluster_t=ct,
                    ))
    summary = pd.DataFrame(sum_rows)
    sum_path = os.path.join(OUT_DIR, "summary_ceiling.csv")
    summary.to_csv(sum_path, index=False, float_format="%.6f")
    log(f"[dump] summary_ceiling.csv 24 行 -> {sum_path}")

    # near - far 对照差值（披露）
    nf_rows = []
    for name in ["v1", "v2"]:
        for W in W_GRID:
            for H in H_GRID:
                r_near = summary[(summary["variant"] == name) & (summary["W"] == W)
                                 & (summary["group"] == "near") & (summary["H"] == H)].iloc[0]
                r_far = summary[(summary["variant"] == name) & (summary["W"] == W)
                                & (summary["group"] == "far") & (summary["H"] == H)].iloc[0]
                nf_rows.append(dict(
                    variant=name, W=W, H=H,
                    near_net_mean=r_near["net_mean"], far_net_mean=r_far["net_mean"],
                    near_minus_far=(r_near["net_mean"] - r_far["net_mean"])
                    if np.isfinite(r_near["net_mean"]) and np.isfinite(r_far["net_mean"])
                    else np.nan))

    # ---------------- 自检 2：计数守恒 + 与 stage_bottom_eval 对齐 ----------------
    check2_rows = []
    check2_ok = True
    for d in disc_rows:
        cons_ok = d["near"] + d["far"] == d["pool_scorable"]
        if not cons_ok:
            check2_ok = False
        check2_rows.append(dict(d, conservation_ok=cons_ok))
    # 差异来源分解：ref（全历史序列）可评而本实验（load_market_data 序列，起点 2025-11-02）
    # 不可评的唯一预期差源 = anchor_date 落出序列且该股该窗口有基准低点的事件；
    # anchor 落出且无基准的事件在 ref 口径计入 no_gt，不影响可评分数。
    for d in check2_rows:
        name, W = d["variant"], d["W"]
        c = cls_store[(name, W)]
        ao = c[c["cls"] == "unscorable_anchor"]
        n_ao_gt = int(ao["ts_code"].map(lambda x: (W, x) in gtpos).sum())
        d["anchor_out_with_gt"] = n_ao_gt
        d["anchor_out_no_gt"] = int(len(ao)) - n_ao_gt
        d["diff_residual"] = d["diff_vs_ref"] + n_ao_gt  # 预期 0
        if d["diff_residual"] != 0:
            check2_ok = False
    log(f"[check2] 计数守恒与对齐：{'PASS' if check2_ok else 'FAIL'}；"
        f"差异分解=" + json.dumps(
            [{k: r[k] for k in ("variant", "W", "pool_scorable", "ref_scorable",
                                "diff_vs_ref", "anchor_out_with_gt",
                                "anchor_out_no_gt", "diff_residual")} for r in check2_rows],
            ensure_ascii=False))

    # ---------------- 自检 3：随机抽 5 笔完整生命周期 ----------------
    rng = np.random.default_rng(SAMPLE_SEED)
    pool_closed = full[full["status"] == "closed"].drop_duplicates(
        ["variant", "ts_code", "event_date", "H"])
    pick = pool_closed.iloc[rng.choice(len(pool_closed), size=5, replace=False)]
    samples = []
    for r in pick.itertuples(index=False):
        samples.append(dict(
            variant=r.variant, W=int(r.W), H=int(r.H), ts_code=r.ts_code,
            event_date=str(r.event_date.date()), anchor_date=str(r.anchor_date.date()),
            group=r.group, dist=int(r.dist),
            entry_date=str(r.entry_date.date()), entry_raw=float(r.entry_raw),
            entry_exec=float(r.entry_exec), shares=int(r.shares),
            exit_date=str(r.exit_date.date()), exit_raw=float(r.exit_raw),
            exit_exec=float(r.exit_exec),
            gross_ret=float(r.gross_ret), buy_comm=float(r.buy_comm),
            sell_comm=float(r.sell_comm), stamp=float(r.stamp),
            net_pnl=float(r.net_pnl), net_ret=float(r.net_ret),
            held_days=int(r.held_days), deferred_days=int(r.deferred_days)))
        log(f"[check3-sample] {r.variant} {r.ts_code} ev={r.event_date.date()} "
            f"grp={r.group} dist={int(r.dist)} H={r.H} "
            f"入{r.entry_date.date()}@{r.entry_raw:.3f} 出{r.exit_date.date()}@{r.exit_raw:.3f} "
            f"净{r.net_ret:+.4f}")

    # ---------------- 主格宣判 ----------------
    verdicts = {}
    for name in ["v1", "v2"]:
        r = summary[(summary["variant"] == name) & (summary["W"] == 60)
                    & (summary["group"] == "near") & (summary["H"] == 60)].iloc[0]
        exists = (r["n_closed"] > 0 and r["net_mean"] > 0
                  and np.isfinite(r["cluster_t"]) and r["cluster_t"] >= 2.0)
        verdicts[name] = dict(
            cell="W60 x near x H60",
            n_scorable=int(r["n_scorable"]), n_traded=int(r["n_traded"]),
            n_closed=int(r["n_closed"]), n_incomplete=int(r["n_incomplete"]),
            net_mean=float(r["net_mean"]), net_median=float(r["net_median"]),
            win_rate=float(r["win_rate"]),
            cluster_t=float(r["cluster_t"]) if np.isfinite(r["cluster_t"]) else None,
            ceiling_exists=bool(exists),
            verdict=("天花板存在（A 值得进入识别规则设计）" if exists
                     else "天花板不存在（A 封档依据）"))
        log(f"[verdict] {name} 主格 W60xnearxH60: n={verdicts[name]['n_closed']} "
            f"净笔均={verdicts[name]['net_mean']:+.4f} cluster_t={verdicts[name]['cluster_t']} "
            f"-> {verdicts[name]['verdict']}")
    n_ceiling = sum(1 for v in verdicts.values() if v["ceiling_exists"])
    overall = ("两变体主格均未见天花板 -> A 方案（背离降级为状态标签吃阶段反转）封档依据成立"
               if n_ceiling == 0 else
               f"{n_ceiling}/2 变体主格天花板存在 -> A 值得进入识别规则设计（必要条件，非充分）")

    # ---------------- 自检 4：脚本独立性声明（静态成立，写入报告） ----------------
    check4_ok = True

    checks = dict(
        check1_causality=dict(ok=check1_ok, n_entry_violations=n_viol_entry,
                              n_exit_violations=n_viol_exit),
        check2_conservation=dict(ok=check2_ok, rows=check2_rows),
        check3_samples=dict(ok=True, seed=SAMPLE_SEED, samples=samples),
        check4_independence=dict(
            ok=check4_ok,
            note="仅 import strategy_engine 原语（load_market_data/buy_cost/sell_costs/"
                 "SLIPPAGE/BOARD_LOT/PRICE_TOL），模拟循环本脚本独立实现，"
                 "未复用任何既有回测实现。"),
    )
    all_ok = all(v["ok"] for v in checks.values())
    log(f"[checks] 四项自检总评：{'ALL PASS' if all_ok else 'HAS FAIL'}")

    # ---------------- verdict.json ----------------
    vout = dict(
        experiment="stage_ceiling（阶段底收益天花板测量，A 方案生死闸门）",
        warning=("分组标签为事后信息（上界假设），入场/持有/出场完全因果；"
                 "本实验不是可交易策略回测。完美识别都不赚钱 => A 封档。"),
        prereadme="README.md 先于任何跑数落盘",
        main_criterion="主格 = 变体 x W60 x near x H60：净笔均 > 0 且 cluster_t >= 2 -> 天花板存在",
        verdicts=verdicts,
        secondary_support={name: summary[(summary["variant"] == name) & (summary["W"] == 60)
                                         & (summary["group"] == "near")
                                         & (summary["H"] == 120)][
            ["n_scorable", "n_closed", "net_mean", "net_median", "win_rate",
             "cluster_t"]].iloc[0].to_dict() for name in ["v1", "v2"]},
        near_minus_far=nf_rows,
        disclosure=disc_rows,
        checks=checks,
        overall=overall,
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    vpath = os.path.join(OUT_DIR, "verdict.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vout, f, ensure_ascii=False, indent=2, default=str)
    log(f"[dump] verdict.json -> {vpath}")

    # ---------------- report.md ----------------
    write_report(summary, verdicts, nf_rows, check2_rows, samples, checks, all_ok, overall)
    log(f"[dump] report.md -> {os.path.join(OUT_DIR, 'report.md')}")
    log(f"STAGE CEILING DONE ({time.time() - t_all:.0f}s)")

    pd.set_option("display.width", 300)
    print("\n===== SUMMARY（配置为行，24 格全出数） =====")
    print(summary.to_string(index=False))
    print("\n===== 主格宣判 =====")
    for name, v in verdicts.items():
        print(f"{name}: {v['verdict']} | n={v['n_closed']} "
              f"净笔均={v['net_mean']:+.4f} cluster_t={v['cluster_t']}")
    print(overall)


def write_report(summary, verdicts, nf_rows, check2_rows, samples, checks, all_ok, overall):
    L = []
    L.append("# 阶段底收益天花板测量报告（预登记纯测量，2026-09-04）")
    L.append("")
    L.append("> **显著警告**：分组标签（near/far）是事后信息（上界假设），"
             "入场/持有/出场完全因果。本实验不是可交易策略回测。")
    L.append("> 若完美识别（上界）扣完成本都不赚钱，则任何因果识别规则都不可能赚钱，A 方案封档。")
    L.append("")
    L.append(f"主判据：主格 = 变体 × W60 × near × H60；净笔均 > 0 且 cluster_t ≥ 2 → 天花板存在。")
    L.append("")
    L.append("## 1. 主格宣判（预登记判据，出数后未移动球门）")
    L.append("")
    L.append("| 变体 | n_scorable | n_traded | n_closed | n_incomplete | 净笔均 | 净中位 | 胜率 | cluster_t | 宣判 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in ["v1", "v2"]:
        v = verdicts[name]
        ct = f"{v['cluster_t']:.3f}" if v["cluster_t"] is not None else "NaN"
        L.append(f"| {name} | {v['n_scorable']} | {v['n_traded']} | {v['n_closed']} | "
                 f"{v['n_incomplete']} | {v['net_mean']:+.4f} | {v['net_median']:+.4f} | "
                 f"{v['win_rate']:.3f} | {ct} | {v['verdict']} |")
    L.append("")
    L.append(f"**总评**：{overall}。")
    L.append("")
    L.append("## 2. 全格子表（配置为行，24 格全出数）")
    L.append("")
    L.append("| 变体 | W | 组 | H | n_scorable | n_traded | n_closed | n_incomplete | "
             "drop涨停 | drop无报价 | drop现金 | 净笔均 | 净中位 | 胜率 | cluster_t |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.itertuples(index=False):
        nm = f"{r.net_mean:+.4f}" if np.isfinite(r.net_mean) else "NaN"
        nd = f"{r.net_median:+.4f}" if np.isfinite(r.net_median) else "NaN"
        wr = f"{r.win_rate:.3f}" if np.isfinite(r.win_rate) else "NaN"
        ct = f"{r.cluster_t:.3f}" if np.isfinite(r.cluster_t) else "NaN"
        L.append(f"| {r.variant} | {r.W} | {r.group} | {r.H} | {r.n_scorable} | {r.n_traded} | "
                 f"{r.n_closed} | {r.n_incomplete} | {r.n_dropped_limitup} | "
                 f"{r.n_dropped_no_quote} | {r.n_dropped_cash} | {nm} | {nd} | {wr} | {ct} |")
    L.append("")
    L.append("## 3. near − far 对照（披露）")
    L.append("")
    L.append("| 变体 | W | H | near 净笔均 | far 净笔均 | near−far |")
    L.append("|---|---|---|---|---|---|")
    for r in nf_rows:
        fmt = lambda x: f"{x:+.4f}" if np.isfinite(x) else "NaN"  # noqa: E731
        L.append(f"| {r['variant']} | {r['W']} | {r['H']} | {fmt(r['near_net_mean'])} | "
                 f"{fmt(r['far_net_mean'])} | {fmt(r['near_minus_far'])} |")
    L.append("")
    L.append("## 4. 分组与截断披露（每变体 × W）")
    L.append("")
    L.append("| 变体 | W | 事件总数 | 截断不可评 | anchor落出序列 | 无基准低点(no_gt) | "
             "无序列 | 事件日无行 | near | far | pool可评分 | 对照stage_bottom_eval | 差值 | "
             "anchor落出且有基准 | anchor落出且无基准 | 残差 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in check2_rows:
        L.append(f"| {d['variant']} | {d['W']} | {d['events_total']} | "
                 f"{d['unscorable_truncated']} | {d['unscorable_anchor']} | {d['no_gt']} | "
                 f"{d['unscorable_no_series']} | {d['unscorable_no_event_row']} | "
                 f"{d['near']} | {d['far']} | {d['pool_scorable']} | {d['ref_scorable']} | "
                 f"{d['diff_vs_ref']} | {d['anchor_out_with_gt']} | {d['anchor_out_no_gt']} | "
                 f"{d['diff_residual']} |")
    L.append("")
    L.append("差值解释口径：stage_bottom_eval 用全历史序列，本实验用 load_market_data 序列"
             "（引擎自带 60 自然日前向缓冲，起点 2025-11-02）；anchor_date 早于 2025-11-02 的事件"
             "在彼处可评分、此处记 unscorable_anchor。")
    L.append("其中该股该窗口有基准低点者影响可评分数（残差 = 差值 + 此类事件数，应为 0）；"
             "无基准低点者在彼处计入 no_gt，不影响可评分数。")
    L.append("")
    L.append("## 5. 自检结果")
    L.append("")
    c1 = checks["check1_causality"]
    L.append(f"1. 因果断言：entry_date > event_date 违反 {c1['n_entry_violations']} 笔，"
             f"exit_date > entry_date 违反 {c1['n_exit_violations']} 笔 → "
             f"{'PASS' if c1['ok'] else 'FAIL'}（违反数必须为 0）。")
    c2ok = checks["check2_conservation"]["ok"]
    L.append(f"2. 计数守恒：每（变体 × W）near+far = pool 成立 → {'PASS' if c2ok else 'FAIL'}；"
             "与 stage_bottom_eval 对齐见第 4 节（差值逐笔归因于 anchor 落出序列，残差见表）。")
    L.append(f"3. 随机抽 5 笔完整生命周期（种子 {checks['check3_samples']['seed']}，供人工抽查）：")
    L.append("")
    L.append("| 变体 | 代码 | 事件日 | 锚日 | 组 | dist | H | 入场日 | 入场价(原始/执行) | 股数 | "
             "出场日 | 出场价(原始/执行) | 毛收益 | 买佣 | 卖佣 | 印花税 | 净盈亏 | 净收益 | 持有日 | 顺延日 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in samples:
        L.append(f"| {s['variant']} | {s['ts_code']} | {s['event_date']} | {s['anchor_date']} | "
                 f"{s['group']} | {s['dist']} | {s['H']} | {s['entry_date']} | "
                 f"{s['entry_raw']:.3f}/{s['entry_exec']:.3f} | {s['shares']} | {s['exit_date']} | "
                 f"{s['exit_raw']:.3f}/{s['exit_exec']:.3f} | {s['gross_ret']:+.4f} | "
                 f"{s['buy_comm']:.2f} | {s['sell_comm']:.2f} | {s['stamp']:.2f} | "
                 f"{s['net_pnl']:+.2f} | {s['net_ret']:+.4f} | {s['held_days']} | "
                 f"{s['deferred_days']} |")
    L.append("")
    L.append(f"4. 独立性：{checks['check4_independence']['note']} → PASS。")
    L.append("")
    L.append(f"**自检总评：{'ALL PASS' if all_ok else 'HAS FAIL'}**")
    L.append("")
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
