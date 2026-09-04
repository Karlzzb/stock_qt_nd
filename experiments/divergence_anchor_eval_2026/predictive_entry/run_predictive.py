#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预判金叉挂买两档设计逐笔模拟 + 基线重算对拍 + 汇总（预登记见同目录 README.md）。

- 读 candidates.parquet（predictive_scan.py 产物，含未过边际闸门全部候选）。
- 设计1（目标日开盘市价买 ×1.001）/ 设计2（c* 限价单，不加滑点）。
- margin 网格 {0.01 主档, 0.0, 0.03}：仅当 c* ≤ eve_close×(1−m) 出手，否则 skipped_margin。
- 确认单出场 E1-H12 / A13，口径逐条对齐 backtest 事件层（README 第 6 节）。
- 失败单 t+2 开盘认错卖出（×0.999，跌停顺延，末端 incomplete 剔除）。
- 基线：events_v1/v2 全量 × {A13, E1-H12}，直接 import 复用 backtest/event_study.py 的
  simulate_event（不复制实现），并与既有 event_study.parquet 对拍（|Δret|<1e-9 比例须 >99%）。

产物：trades_predictive.parquet、summary_predictive.csv、run_predictive.log。
全程禁网络；仓库内除本目录外只读。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
BT_DIR = os.path.join(REPO_ROOT, "experiments", "divergence_anchor_eval_2026", "backtest")
OUT_DIR = os.path.join(REPO_ROOT, "experiments", "divergence_anchor_eval_2026", "predictive_entry")
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))
sys.path.insert(0, BT_DIR)

import strategy_engine as se  # noqa: E402  冻结引擎：常量/成本函数/load_market_data（只读复用）
import event_study as es_mod  # noqa: E402  基线逐笔函数同源复用（含 CONFIGS/BUDGET）

LOG_PATH = os.path.join(OUT_DIR, "run_predictive.log")
EVENTS = {
    "v1": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v1.parquet"),
    "v2": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v2.parquet"),
}
EVENT_STUDY_PQ = os.path.join(BT_DIR, "event_study.parquet")

START, END = "2026-01-01", "2026-08-31"
BENCH = "000905.SH"
BUDGET = es_mod.BUDGET          # 100,000 元/笔
M_MAIN = 0.01                   # 主边际
M_GRID = (0.01, 0.0, 0.03)      # 主档 + 敏感性
TOL = se.PRICE_TOL
EXITS = ("E1-H12", "A13")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def round_shares(exec_price: float):
    """整手规则（对齐引擎/事件层）：返回 (shares, commission) 或 None（dropped_cash）。"""
    shares = int(BUDGET / exec_price / se.BOARD_LOT) * se.BOARD_LOT
    if shares < se.BOARD_LOT:
        shares = se.BOARD_LOT
    comm = se.buy_cost(shares, exec_price)
    while shares > 0 and shares * exec_price + comm > BUDGET + 1e-6:
        shares -= se.BOARD_LOT
        comm = se.buy_cost(shares, exec_price) if shares > 0 else 0.0
    if shares <= 0:
        return None
    return shares, comm


def sell_leg(shares: int, entry_price: float, entry_commission: float,
             raw_exit: float, day: pd.Timestamp):
    exec_sell = raw_exit * (1.0 - se.SLIPPAGE)
    xcomm, stamp = se.sell_costs(shares, exec_sell, day)
    net_pnl = shares * (exec_sell - entry_price) - (entry_commission + xcomm + stamp)
    ret = net_pnl / (shares * entry_price + entry_commission)
    return exec_sell, xcomm, stamp, net_pnl, ret


def down_limited(md, ts_code: str, day: pd.Timestamp, close_px: float) -> bool:
    lim = md.limits.get(day)
    if lim is not None and ts_code in lim.index:
        return close_px <= float(lim.loc[ts_code, "down_limit"]) + TOL
    return False


def simulate_confirmed_exit(ts_code, row, di_entry, entry_day, entry_price, shares,
                            entry_commission, exit_name, md, cal_arr):
    """确认单出场：E1-H12 / A13（逐条对齐事件层口径，README 第 6 节）。"""
    deferred = 0
    for di in range(di_entry + 1, len(cal_arr)):
        day = cal_arr[di]
        bar = row.loc[day] if day in row.index else None
        if bar is None:
            continue  # 无行情日不评估
        held = di - di_entry + 1
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        raw_exit = reason = None
        if exit_name == "A13":
            tp_b = entry_price * 1.25
            sl_b = entry_price * 0.86
            tp_hit = h >= tp_b - TOL
            sl_hit = l <= sl_b + TOL
            if tp_hit and not sl_hit:
                raw_exit = o if o >= tp_b - TOL else tp_b
                reason = "tp"
            elif sl_hit:
                raw_exit = o if o <= sl_b + TOL else sl_b
                reason = "sl"
            elif held >= 12:
                raw_exit = c
                reason = "horizon"
        else:  # E1-H12
            if held >= 12:
                raw_exit = c
                reason = "horizon"
        if raw_exit is None:
            continue
        if down_limited(md, ts_code, day, c):
            deferred += 1
            continue  # 跌停顺延
        exec_sell, xcomm, stamp, net_pnl, ret = sell_leg(
            shares, entry_price, entry_commission, raw_exit, day)
        return dict(status="closed", exit_date=day, exit_reason=reason,
                    exit_raw_price=raw_exit, exit_exec_price=exec_sell,
                    exit_commission=xcomm, stamp_tax=stamp, net_pnl=net_pnl,
                    ret=ret, held_days=held, deferred_days=deferred)
    return dict(status="incomplete", deferred_days=deferred)


def simulate_fail_exit(ts_code, row, di_entry, entry_price, shares,
                       entry_commission, md, cal_arr):
    """失败单：t+2（日历次日）开盘认错卖出，跌停顺延，无行情日跳过。"""
    deferred = 0
    for di in range(di_entry + 1, len(cal_arr)):
        day = cal_arr[di]
        bar = row.loc[day] if day in row.index else None
        if bar is None:
            continue
        o, c = float(bar["open"]), float(bar["close"])
        if down_limited(md, ts_code, day, c):
            deferred += 1
            continue
        exec_sell, xcomm, stamp, net_pnl, ret = sell_leg(
            shares, entry_price, entry_commission, o, day)
        return dict(status="closed", exit_date=day, exit_reason="fail_next_open",
                    exit_raw_price=o, exit_exec_price=exec_sell,
                    exit_commission=xcomm, stamp_tax=stamp, net_pnl=net_pnl,
                    ret=ret, held_days=di - di_entry + 1, deferred_days=deferred)
    return dict(status="incomplete", deferred_days=deferred)


def simulate_candidate(cand, design: str, md, cal_index, cal_arr):
    """单候选 × 单设计的入场+确认判定+出场（出场结果按 exit 分列）。

    返回 dict：entry_status ∈ {confirmed, failed, unfilled, dropped_limitup,
    dropped_cash, dropped_no_calendar}；confirmed 时 exits={exit_name: 结果}；
    failed 时 fail=结果。
    """
    ts_code = cand["ts_code"]
    target = cand["target_date"]
    c_star = cand["c_star"]
    if target not in cal_index:
        return dict(entry_status="dropped_no_calendar")
    row = md.daily.get(ts_code)
    bar = row.loc[target] if (row is not None and target in row.index) else None
    if bar is None:
        return dict(entry_status="dropped_no_quote")
    o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

    # 涨停锁死判定（两设计共用）
    lim = md.limits.get(target)
    limitup = False
    if lim is not None and ts_code in lim.index:
        limitup = o >= float(lim.loc[ts_code, "up_limit"]) - TOL

    if design == "d1_open":
        if limitup:
            return dict(entry_status="dropped_limitup")
        entry_price = o * (1.0 + se.SLIPPAGE)
    else:  # d2_limit
        if limitup:
            return dict(entry_status="unfilled")
        if o <= c_star:
            entry_price = o          # 限价单以更优开盘价成交，不加滑点
        elif l <= c_star:
            entry_price = c_star     # 按临界价成交，不加滑点
        else:
            return dict(entry_status="unfilled")

    rs = round_shares(entry_price)
    if rs is None:
        return dict(entry_status="dropped_cash")
    shares, entry_commission = rs
    di_entry = cal_index[target]
    base = dict(entry_date=target, entry_price=entry_price, shares=shares,
                entry_commission=entry_commission)

    if c > c_star:  # 金叉成立
        exits = {ex: simulate_confirmed_exit(ts_code, row, di_entry, target,
                                             entry_price, shares, entry_commission,
                                             ex, md, cal_arr)
                 for ex in EXITS}
        return dict(entry_status="confirmed", exits=exits, **base)
    fail = simulate_fail_exit(ts_code, row, di_entry, entry_price, shares,
                              entry_commission, md, cal_arr)
    return dict(entry_status="failed", fail=fail, **base)


def main() -> None:
    t_all = time.time()
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log(f"PREDICTIVE RUN START window=[{START}..{END}] bench={BENCH} "
        f"budget={BUDGET:.0f} margins={M_GRID} main_m={M_MAIN}")

    cand = pd.read_parquet(os.path.join(OUT_DIR, "candidates.parquet"))
    log(f"candidates loaded: {len(cand)} "
        f"(by variant: {cand['variant'].value_counts().to_dict()})")

    # 基准收盘序列
    bdf = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index", f"{BENCH}.parquet"),
                          columns=["trade_date", "close"])
    bdf["trade_date"] = pd.to_datetime(bdf["trade_date"])
    bench_close = bdf.set_index("trade_date")["close"].astype(float).sort_index()

    # ---------------- 行情加载（候选股 ∪ 事件股） ----------------
    ev_tables = {}
    for v, p in EVENTS.items():
        e = pd.read_parquet(p)
        e["event_date"] = pd.to_datetime(e["event_date"])
        e["cross_date"] = pd.to_datetime(e["cross_date"])
        ev_tables[v] = e
    codes = sorted(set(cand["ts_code"]).union(
        *[set(e["ts_code"]) for e in ev_tables.values()]))
    log(f"loading market data for {len(codes)} stocks ...")
    md = se.load_market_data(codes, START, END, REPO_ROOT, log_path=LOG_PATH)
    cal_arr = list(md.calendar)
    cal_index = {d: i for i, d in enumerate(cal_arr)}
    log(f"market data loaded: daily={len(md.daily)} calendar={len(cal_arr)} "
        f"limit_missing_days={md.limit_missing_dates} ({time.time() - t_all:.0f}s)")

    # ---------------- 基线重算 + 对拍 ----------------
    log("BASELINE recompute start (events_v1/v2 × {A13, E1-H12}, simulate_event 同源复用)")
    es_old = pd.read_parquet(EVENT_STUDY_PQ)
    es_old["event_date"] = pd.to_datetime(es_old["event_date"])
    baseline_rows = []
    cmp_report = []
    for v, e in ev_tables.items():
        sig = f"events_{v}"
        for cfg in ("A13", "E1-H12"):
            spec = es_mod.CONFIGS[cfg]
            recs = []
            for r in e.itertuples():
                res = es_mod.simulate_event(r.ts_code, r.event_date, spec, md,
                                            cal_index, cal_arr, {}, None)
                res.update(ts_code=r.ts_code, event_date=r.event_date)
                recs.append(res)
            new = pd.DataFrame(recs)
            old = es_old[(es_old["signal"] == sig) & (es_old["config"] == cfg)]
            n_cnt_ok = len(new) == len(old)
            old_idx = old.set_index(["ts_code", "event_date"])
            n_closed_new = int((new["status"] == "closed").sum())
            n_closed_old = int((old["status"] == "closed").sum())
            diffs = []
            mismatch = []
            for r in new[new["status"] == "closed"].itertuples():
                key = (r.ts_code, pd.Timestamp(r.event_date))
                if key not in old_idx.index:
                    mismatch.append((key, "old_missing"))
                    continue
                o_ret = old_idx.loc[key, "ret"]
                if pd.isna(o_ret):
                    mismatch.append((key, "old_not_closed"))
                    continue
                d = abs(float(r.ret) - float(o_ret))
                diffs.append(d)
                if d >= 1e-9:
                    mismatch.append((key, f"ret_diff={d}"))
            frac = (np.mean(np.array(diffs) < 1e-9) if diffs else np.nan)
            cmp_report.append(dict(signal=sig, config=cfg, rows_new=len(new),
                                   rows_old=len(old), count_match=n_cnt_ok,
                                   closed_new=n_closed_new, closed_old=n_closed_old,
                                   ret_match_frac=frac, n_mismatch=len(mismatch)))
            log(f"BASELINE CMP {sig}__{cfg}: rows {len(new)} vs {len(old)} "
                f"(match={n_cnt_ok}), closed {n_closed_new} vs {n_closed_old}, "
                f"|Δret|<1e-9 比例={frac:.6f}, mismatch={mismatch[:5]}")
            cl = new[new["status"] == "closed"].copy()
            b_in = cl["entry_date"].map(bench_close)
            b_out = cl["exit_date"].map(bench_close)
            cl["bench_ret"] = (b_out / b_in - 1.0).to_numpy()
            cl["excess"] = cl["ret"] - cl["bench_ret"]
            cl["signal"] = v
            cl["config"] = cfg
            baseline_rows.append(cl)
    baseline = pd.concat(baseline_rows, ignore_index=True)
    log(f"BASELINE recompute done ({time.time() - t_all:.0f}s)")

    # ---------------- 预判逐笔模拟 ----------------
    log("PREDICTIVE simulate start")
    t0 = time.time()
    trade_rows = []
    # 每 (variant, design, margin) 披露计数
    disc = {}
    n_cand = len(cand)
    for i, c in enumerate(cand.itertuples()):
        cd = c._asdict()
        for design in ("d1_open", "d2_limit"):
            res = simulate_candidate(cd, design, md, cal_index, cal_arr)
            for m in M_GRID:
                gate = cd["c_star"] <= cd["eve_close"] * (1.0 - m)
                key = (cd["variant"], design, m)
                d = disc.setdefault(key, dict(total=0, skipped_margin=0, unfilled=0,
                                              dropped_limitup=0, dropped_cash=0,
                                              dropped_no_calendar=0, dropped_no_quote=0,
                                              confirmed=0, failed=0))
                d["total"] += 1
                if not gate:
                    d["skipped_margin"] += 1
                    continue
                st = res["entry_status"]
                if st in d:
                    d[st] += 1
                common = dict(variant=cd["variant"], design=design, margin=m,
                              ts_code=cd["ts_code"], eve_date=cd["eve_date"],
                              target_date=cd["target_date"], c_star=cd["c_star"],
                              margin_ratio=cd["margin_ratio"],
                              dif_lift_star=cd["dif_lift_star"])
                if st == "confirmed":
                    for ex in EXITS:
                        r = dict(common)
                        r.update(exit=ex, confirmed=True,
                                 entry_date=res["entry_date"],
                                 entry_price=res["entry_price"],
                                 shares=res["shares"],
                                 entry_commission=res["entry_commission"])
                        r.update(res["exits"][ex])
                        trade_rows.append(r)
                elif st == "failed":
                    for ex in EXITS:  # 失败单与出场规则无关，按 exit 冗余成行（README 9.3）
                        r = dict(common)
                        r.update(exit=ex, confirmed=False,
                                 entry_date=res["entry_date"],
                                 entry_price=res["entry_price"],
                                 shares=res["shares"],
                                 entry_commission=res["entry_commission"])
                        r.update(res["fail"])
                        trade_rows.append(r)
                else:  # unfilled / dropped_*：仅披露计数，无交易
                    r = dict(common)
                    r.update(exit=None, confirmed=None, status=st)
                    trade_rows.append(r)
        if (i + 1) % 20000 == 0:
            log(f"heartbeat: {i + 1}/{n_cand} candidates, trades_rows={len(trade_rows)}, "
                f"{time.time() - t0:.0f}s")
    trades = pd.DataFrame(trade_rows)
    log(f"PREDICTIVE simulate done rows={len(trades)} ({time.time() - t0:.0f}s)")

    # 超额（closed 行）
    closed_mask = trades["status"] == "closed"
    b_in = trades.loc[closed_mask, "entry_date"].map(bench_close)
    b_out = trades.loc[closed_mask, "exit_date"].map(bench_close)
    assert b_in.notna().all() and b_out.notna().all(), "基准序列未覆盖持有窗"
    trades.loc[closed_mask, "bench_ret"] = (b_out / b_in - 1.0).to_numpy()
    trades.loc[closed_mask, "excess"] = (trades.loc[closed_mask, "ret"]
                                         - trades.loc[closed_mask, "bench_ret"])

    pq_path = os.path.join(OUT_DIR, "trades_predictive.parquet")
    trades.to_parquet(pq_path, index=False)
    log(f"trades_predictive.parquet rows={len(trades)} -> {pq_path}")

    # ---------------- 入场价改善配对（同一金叉日对齐基线） ----------------
    # 基线入场执行价映射：(variant, ts_code, cross_date) -> entry_price（含 ×1.001）
    base_entry = {}
    for v, e in ev_tables.items():
        sub = baseline[baseline["signal"] == v].set_index(["ts_code", "event_date"])
        for r in e.itertuples():
            key = (r.ts_code, pd.Timestamp(r.event_date))
            if key in sub.index:
                ep = sub.loc[key, "entry_price"]
                if isinstance(ep, pd.Series):
                    ep = ep.iloc[0]
                if pd.notna(ep):
                    base_entry[(v, r.ts_code, pd.Timestamp(r.cross_date))] = float(ep)
    pair_stats = {}
    for (v, design, m), g in trades[(trades["confirmed"] == True)  # noqa: E712
                                    & trades["entry_price"].notna()].groupby(
                                        ["variant", "design", "margin"]):
        imps = []
        n_unpaired = 0
        for r in g.drop_duplicates(["ts_code", "target_date"]).itertuples():
            be = base_entry.get((v, r.ts_code, pd.Timestamp(r.target_date)))
            if be is None:
                n_unpaired += 1
                continue
            imps.append((be - r.entry_price) / be)
        pair_stats[(v, design, m)] = dict(
            n_paired=len(imps), n_unpaired=n_unpaired,
            improve_mean=float(np.mean(imps)) if imps else np.nan,
            improve_median=float(np.median(imps)) if imps else np.nan)

    # ---------------- 汇总表 ----------------
    def agg(g: pd.DataFrame):
        cl = g[g["status"] == "closed"]
        ret = cl["ret"].to_numpy()
        n = len(ret)
        sd = ret.std(ddof=1) if n > 1 else np.nan
        return dict(n_closed=n,
                    ret_mean=float(ret.mean()) if n else np.nan,
                    win_rate=float((ret > 0).mean()) if n else np.nan,
                    excess_mean=float(cl["excess"].mean()) if n else np.nan,
                    t_ret=float(ret.mean() / (sd / np.sqrt(n)))
                    if n > 1 and sd > 0 else np.nan)

    sum_rows = []
    for (v, design, ex, m), g in trades.groupby(["variant", "design", "exit", "margin"],
                                                dropna=False):
        if pd.isna(ex):
            continue
        d = disc[(v, design, m)]
        conf = g[g["confirmed"] == True]  # noqa: E712
        fail = g[g["confirmed"] == False]  # noqa: E712
        a_conf, a_fail = agg(conf), agg(fail)
        comb = g[g["status"] == "closed"]
        a_comb = agg(g)
        ps = pair_stats.get((v, design, m), {})
        sum_rows.append(dict(
            variant=v, design=design, exit=ex, margin=m,
            total_candidates=d["total"], skipped_margin=d["skipped_margin"],
            unfilled=d["unfilled"], dropped_limitup=d["dropped_limitup"],
            dropped_cash=d["dropped_cash"],
            n_filled=d["confirmed"] + d["failed"],
            n_confirmed=d["confirmed"], n_failed=d["failed"],
            confirm_rate=(d["confirmed"] / (d["confirmed"] + d["failed"])
                          if d["confirmed"] + d["failed"] else np.nan),
            conf_ret_mean=a_conf["ret_mean"],
            fail_ret_mean=a_fail["ret_mean"],
            comb_ret_mean=a_comb["ret_mean"],
            comb_win_rate=a_comb["win_rate"],
            comb_excess_mean=a_comb["excess_mean"],
            comb_t_ret=a_comb["t_ret"],
            n_incomplete=int((g["status"] == "incomplete").sum()),
            n_deferred=int((g["deferred_days"].fillna(0) > 0).sum()),
            n_paired=ps.get("n_paired"), n_unpaired=ps.get("n_unpaired"),
            entry_improve_mean=ps.get("improve_mean"),
            entry_improve_median=ps.get("improve_median"),
        ))
    # 基线行
    for (v, cfg), g in baseline.groupby(["signal", "config"]):
        ret = g["ret"].to_numpy()
        n = len(ret)
        sd = ret.std(ddof=1) if n > 1 else np.nan
        sum_rows.append(dict(
            variant=v, design="baseline_next_open", exit=cfg, margin=np.nan,
            n_filled=n, n_confirmed=n, n_failed=0,
            comb_ret_mean=float(ret.mean()), comb_win_rate=float((ret > 0).mean()),
            comb_excess_mean=float(g["excess"].mean()),
            comb_t_ret=float(ret.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
        ))
    summary = pd.DataFrame(sum_rows)
    sum_path = os.path.join(OUT_DIR, "summary_predictive.csv")
    summary.to_csv(sum_path, index=False, float_format="%.6f")
    log(f"summary_predictive.csv rows={len(summary)} -> {sum_path}")

    cmp_df = pd.DataFrame(cmp_report)
    log("对拍总账: " + str(cmp_df[["signal", "config", "count_match",
                                   "ret_match_frac", "n_mismatch"]].to_dict("records")))
    all_frac = [r["ret_match_frac"] for r in cmp_report]
    log(f"BASELINE CROSS-CHECK: 全部一致率>99% = {all(f > 0.99 for f in all_frac)}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    main_tab = summary[(summary["margin"] == M_MAIN) | (summary["design"] == "baseline_next_open")]
    sens_tab = summary[summary["margin"].isin([0.0, 0.03])]
    log("===== MAIN TABLE (m=1% + baseline) =====\n" + main_tab.to_string(index=False))
    log("===== SENSITIVITY (m=0%, 3%) =====\n"
        + sens_tab[["variant", "design", "exit", "margin", "n_filled", "confirm_rate",
                    "comb_ret_mean"]].to_string(index=False))
    log(f"PREDICTIVE RUN DONE ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
