#!/usr/bin/env python3
"""事件层全池研究（README.md「预登记增补 2026-09-04」B/C/D 节，预登记先于跑数）。

两变体全部事件 × 三档出场（A13 / B15 / E1-H12）逐笔独立模拟，每笔名义本金 100,000 元。
逐笔模拟独立实现（不调用组合引擎主循环），屏障/顺延/成本/整手规则逐条对齐引擎口径
（对齐清单 = README 增补 D 节 1~18 条）。

对拍（C 节）：组合层 N=10 每笔成交 trades 与事件层同 (signal,config,ts_code,event_date)
结果比对——第一层以组合层实际股数钉住重放，全字段逐位断言（容差 1e-9），任一不符即停；
第二层披露固定预算股数与组合层股数不一致笔数，并在股数一致子集上断言 net_pnl 逐位一致。

产物：event_study.parquet（全量含 dropped_*/incomplete 行）、event_study_summary.csv、
event_study_terciles.csv（E1-H12 dif_lift 组内三分位档）、event_study.log。
全程禁网络；除本目录外仓库只读。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))

import strategy_engine as se  # noqa: E402  v1 冻结引擎（只读复用常量/成本函数/load_market_data）
import strategy_engine_v3 as se3  # noqa: E402  v3 冻结引擎（只读复用 atr_series/vol_band/ExitSpec）
from run_strategy_tuning import build_mkt_atr  # noqa: E402  类 B 全市场 ATR 均值（同源）

BT_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/backtest")
EVENTS = {
    "events_v1": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v1.parquet"),
    "events_v2": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v2.parquet"),
}
LOG_PATH = os.path.join(BT_DIR, "event_study.log")
RUNS_N10 = os.path.join(BT_DIR, "runs_n10")

START, END = "2026-01-01", "2026-08-31"
BENCH = "000905.SH"
N_SLOTS = 10
BUDGET = 100_000.0        # 事件层每笔名义本金（= 组合层 N=10 单仓量级）
VOL_LB = 21
XTOL = 1e-9               # 对拍逐位容差

CONFIGS = {
    "A13": se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12),
    "B15": se3.ExitSpec.vol_adaptive(tp=0.25, sl=-0.14, horizon=12, vol_lookback=VOL_LB,
                                     vol_high_thresh=1.8, vol_low_thresh=0.6,
                                     vol_profit_mult=1.5, vol_stop_mult=1.1,
                                     low_vol_profit_mult=1.0),
    "E1-H12": se3.ExitSpec.horizon_only(horizon=12),
}

# 对拍逐字段清单（第一层）
CMP_FIELDS = ["entry_date", "exit_date", "exit_reason", "entry_price", "exit_raw_price",
              "exit_exec_price", "entry_commission", "exit_commission", "stamp_tax",
              "net_pnl"]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 逐笔独立模拟
def simulate_event(ts_code: str, event_date: pd.Timestamp, spec: se3.ExitSpec,
                   md: se.MarketData, cal_index: dict, cal_arr: list,
                   stock_atr: dict, mkt_atr: pd.Series | None,
                   shares_override: int | None = None,
                   fb_counter: list | None = None) -> dict:
    """单事件逐笔模拟。规则与引擎逐条对齐（README 增补 D 节）。

    shares_override 非空时股数钉住（对拍第一层用），现金约束不再适用
    （组合层实际成交即已满足其现金约束）；否则预算/现金 = BUDGET 固定。
    返回 status ∈ {closed, dropped_no_next_day, dropped_no_quote, dropped_limitup,
    dropped_no_close_T, dropped_cash, incomplete}。
    """
    cls = spec.strategy_class
    if event_date not in cal_index:
        return dict(status="dropped_event_not_in_calendar")
    di_ev = cal_index[event_date]
    if di_ev + 1 >= len(cal_arr):
        return dict(status="dropped_no_next_day")
    entry_day = cal_arr[di_ev + 1]
    assert entry_day > event_date, "leakage guard: entry not after event"

    row = md.daily.get(ts_code)
    bar = row.loc[entry_day] if (row is not None and entry_day in row.index) else None
    if bar is None:
        return dict(status="dropped_no_quote")
    open_px = float(bar["open"])
    lim = md.limits.get(entry_day)
    if lim is not None and ts_code in lim.index:
        up_lim = float(lim.loc[ts_code, "up_limit"])
        if open_px >= up_lim - se.PRICE_TOL:
            return dict(status="dropped_limitup")
    t_close = float(row.loc[event_date, "close"]) if event_date in row.index else None
    if t_close is None:
        return dict(status="dropped_no_close_T")

    exec_price = open_px * (1.0 + se.SLIPPAGE)
    if shares_override is not None:
        shares = int(shares_override)
        comm = se.buy_cost(shares, exec_price)
    else:
        cash = BUDGET
        shares = int(BUDGET / exec_price / se.BOARD_LOT) * se.BOARD_LOT
        if shares < se.BOARD_LOT:
            shares = se.BOARD_LOT
        comm = se.buy_cost(shares, exec_price)
        while shares > 0 and shares * exec_price + comm > cash + 1e-6:
            shares -= se.BOARD_LOT
            comm = se.buy_cost(shares, exec_price) if shares > 0 else 0.0
        if shares <= 0:
            return dict(status="dropped_cash")
    entry_commission = comm

    di_entry = cal_index[entry_day]
    deferred = 0
    n_cal = len(cal_arr)
    for di in range(di_entry + 1, n_cal):
        day = cal_arr[di]
        bar = row.loc[day] if day in row.index else None
        if bar is None:
            continue  # 无行情日不评估（对齐 D-14）
        held = di - di_entry + 1
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))

        raw_exit = None
        reason = None
        if cls in ("fixed_tp_sl", "vol_adaptive"):
            if cls == "fixed_tp_sl":
                tp_eff, sl_eff = spec.tp, spec.sl
            else:
                # 类 B：ref_day = 前一交易日（<=day-1 行情，对齐 D-12）
                ref_day = cal_arr[di - 1]
                s_atr = (stock_atr or {}).get(ts_code)
                a_stock = None
                if s_atr is not None and ref_day in s_atr.index:
                    v = float(s_atr.loc[ref_day])
                    if np.isfinite(v) and v > 0:
                        a_stock = v
                a_mkt = None
                if mkt_atr is not None and ref_day in mkt_atr.index:
                    v = float(mkt_atr.loc[ref_day])
                    if np.isfinite(v) and v > 0:
                        a_mkt = v
                if a_stock is None or a_mkt is None:
                    if fb_counter is not None:
                        fb_counter[0] += 1
                    tp_eff, sl_eff = spec.tp, spec.sl
                else:
                    tp_eff, sl_eff = se3.vol_band(a_stock / a_mkt, spec)
            tp_b = exec_price * (1.0 + tp_eff)
            sl_b = exec_price * (1.0 + sl_eff)
            tp_hit = h >= tp_b - se.PRICE_TOL
            sl_hit = l <= sl_b + se.PRICE_TOL
            if tp_hit and not sl_hit:
                raw_exit = o if o >= tp_b - se.PRICE_TOL else tp_b
                reason = "tp"
            elif sl_hit:
                raw_exit = o if o <= sl_b + se.PRICE_TOL else sl_b
                reason = "sl"
            elif held >= spec.horizon:
                raw_exit = c
                reason = "horizon"
        else:  # E1
            if held >= spec.horizon:
                raw_exit = c
                reason = "horizon"

        if raw_exit is None:
            continue
        lim = md.limits.get(day)
        if lim is not None and ts_code in lim.index:
            dn = float(lim.loc[ts_code, "down_limit"])
            if c <= dn + se.PRICE_TOL:
                deferred += 1
                continue  # 跌停顺延（对齐 D-13）
        exec_sell = raw_exit * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(shares, exec_sell, day)
        gross_pnl = shares * (exec_sell - exec_price)
        total_cost = entry_commission + xcomm + stamp
        net_pnl = gross_pnl - total_cost
        ret = net_pnl / (shares * exec_price + entry_commission)
        return dict(status="closed", entry_date=entry_day, entry_price=exec_price,
                    shares=shares, entry_commission=entry_commission,
                    exit_date=day, exit_reason=reason, exit_raw_price=raw_exit,
                    exit_exec_price=exec_sell, exit_commission=xcomm, stamp_tax=stamp,
                    gross_pnl=gross_pnl, total_cost=total_cost, net_pnl=net_pnl,
                    ret=ret, held_days=held, deferred_days=deferred)
    return dict(status="incomplete", entry_date=entry_day, entry_price=exec_price,
                shares=shares, entry_commission=entry_commission, deferred_days=deferred)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    log(f"EVENT STUDY START window=[{START}..{END}] bench={BENCH} budget={BUDGET:.0f} "
        f"configs={list(CONFIGS)} signals={list(EVENTS)}")

    # 基准收盘序列（超额用：close(entry) -> close(exit)）
    bdf = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index", f"{BENCH}.parquet"),
                          columns=["trade_date", "close"])
    bdf["trade_date"] = pd.to_datetime(bdf["trade_date"])
    bench_close = bdf.set_index("trade_date")["close"].astype(float).sort_index()

    all_rows: list[dict] = []
    disc_rows: list[dict] = []   # 每组披露计数
    mkt_atr_cache: pd.Series | None = None
    n10_trades: dict[tuple[str, str], pd.DataFrame] = {}
    n10_open: dict[tuple[str, str], pd.DataFrame] = {}

    for sig_name, ev_path in EVENTS.items():
        ev = pd.read_parquet(ev_path)
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        assert ev["dif_lift"].notna().all() and (ev["dif_lift"] > 0).all()
        assert not ev.duplicated(["ts_code", "event_date"]).any()
        md = se.load_market_data(ev["ts_code"].unique().tolist(), START, END,
                                 REPO_ROOT, log_path=LOG_PATH)
        cal_arr = list(md.calendar)
        cal_index = {d: i for i, d in enumerate(cal_arr)}
        ev = ev[ev["event_date"].isin(set(cal_arr))].reset_index(drop=True)
        log(f"{sig_name}: events_in_calendar={len(ev)} calendar_days={len(cal_arr)}")

        if mkt_atr_cache is None:
            log(f"构建全市场 ATR 均值序列（类 B，LB={VOL_LB}）...")
            mkt_atr_cache = build_mkt_atr(cal_arr, VOL_LB, LOG_PATH)
            assert mkt_atr_cache.notna().all()
        stock_atr = {c: se3.atr_series(df, VOL_LB) for c, df in md.daily.items()}

        for cfg_name, spec in CONFIGS.items():
            t0 = time.time()
            fb = [0]  # vol_fallback_mid 计数
            recs: list[dict] = []
            for r in ev.itertuples():
                res = simulate_event(r.ts_code, r.event_date, spec, md, cal_index,
                                     cal_arr, stock_atr,
                                     mkt_atr_cache if cfg_name == "B15" else None,
                                     fb_counter=fb)
                res.update(signal=sig_name, config=cfg_name, ts_code=r.ts_code,
                           event_date=r.event_date, dif_lift=float(r.dif_lift))
                recs.append(res)
            gdf = pd.DataFrame(recs)
            closed = gdf[gdf["status"] == "closed"].copy()
            assert (closed["entry_date"] > closed["event_date"]).all(), \
                f"{sig_name}__{cfg_name}: 事件层存在 entry <= event（泄漏）"
            # 超额：bench close(entry) -> close(exit)
            b_in = closed["entry_date"].map(bench_close)
            b_out = closed["exit_date"].map(bench_close)
            assert b_in.notna().all() and b_out.notna().all(), "基准序列未覆盖持有窗"
            closed["bench_ret"] = (b_out / b_in - 1.0).to_numpy()
            closed["excess"] = closed["ret"] - closed["bench_ret"]
            gdf.loc[closed.index, ["bench_ret", "excess"]] = closed[["bench_ret", "excess"]]
            all_rows.append(gdf)

            st = gdf["status"].value_counts().to_dict()
            disc = dict(signal=sig_name, config=cfg_name, total=len(gdf),
                        closed=int(st.get("closed", 0)),
                        incomplete=int(st.get("incomplete", 0)),
                        dropped_limitup=int(st.get("dropped_limitup", 0)),
                        dropped_no_quote=int(st.get("dropped_no_quote", 0)),
                        dropped_no_next_day=int(st.get("dropped_no_next_day", 0)),
                        dropped_no_close_T=int(st.get("dropped_no_close_T", 0)),
                        dropped_cash=int(st.get("dropped_cash", 0)),
                        vol_fallback_mid=fb[0],
                        deferred_events=int((gdf["deferred_days"].fillna(0) > 0).sum()),
                        limit_missing_days=md.limit_missing_dates)
            disc_rows.append(disc)
            log(f"{sig_name}__{cfg_name} done {time.time() - t0:.0f}s | "
                f"closed={disc['closed']} incomplete={disc['incomplete']} "
                f"dropped_limitup={disc['dropped_limitup']} "
                f"dropped_no_quote={disc['dropped_no_quote']} "
                f"dropped_cash={disc['dropped_cash']} fb_mid={fb[0]}")

            # 装入组合层 N=10 trades 供对拍
            run_dir = os.path.join(RUNS_N10, f"{sig_name}__{cfg_name}")
            tr = pd.read_parquet(os.path.join(run_dir, "trades.parquet"))
            tr["event_date"] = pd.to_datetime(tr["event_date"])
            n10_trades[(sig_name, cfg_name)] = tr
            n10_open[(sig_name, cfg_name)] = pd.read_parquet(
                os.path.join(run_dir, "open_positions.parquet"))

    full = pd.concat(all_rows, ignore_index=True)
    out_pq = os.path.join(BT_DIR, "event_study.parquet")
    full.to_parquet(out_pq, index=False)
    log(f"event_study.parquet rows={len(full)} -> {out_pq}")

    # ---------------------------------------------------------------- 汇总表
    sum_rows: list[dict] = []
    for (sig, cfg), gdf in full.groupby(["signal", "config"], sort=False):
        cl = gdf[gdf["status"] == "closed"]
        n = len(cl)
        ret, exc = cl["ret"].to_numpy(), cl["excess"].to_numpy()
        sd = ret.std(ddof=1) if n > 1 else np.nan
        sde = exc.std(ddof=1) if n > 1 else np.nan
        sum_rows.append(dict(
            signal=sig, config=cfg, n=n,
            ret_mean=float(ret.mean()), ret_median=float(np.median(ret)),
            win_rate=float((ret > 0).mean()),
            excess_mean=float(exc.mean()),
            t_ret=float(ret.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
            t_excess=float(exc.mean() / (sde / np.sqrt(n))) if n > 1 and sde > 0 else np.nan,
            bench_ret_mean=float(cl["bench_ret"].mean()),
            held_days_mean=float(cl["held_days"].mean()),
        ))
    summary = pd.DataFrame(sum_rows)
    summary_path = os.path.join(BT_DIR, "event_study_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    log(f"event_study_summary.csv -> {summary_path}")

    # ---------------------------------------------------------------- E1-H12 dif_lift 三分位档
    ter_rows: list[dict] = []
    for sig in EVENTS:
        cl = full[(full["signal"] == sig) & (full["config"] == "E1-H12")
                  & (full["status"] == "closed")].copy()
        cl["tercile"] = pd.qcut(cl["dif_lift"], 3, labels=["low", "mid", "high"])
        for ter, g in cl.groupby("tercile", observed=True):
            ter_rows.append(dict(
                signal=sig, tercile=str(ter), n=len(g),
                dif_lift_min=float(g["dif_lift"].min()),
                dif_lift_max=float(g["dif_lift"].max()),
                ret_mean=float(g["ret"].mean()),
                ret_median=float(g["ret"].median()),
                win_rate=float((g["ret"] > 0).mean()),
                excess_mean=float(g["excess"].mean()),
            ))
    terciles = pd.DataFrame(ter_rows)
    ter_path = os.path.join(BT_DIR, "event_study_terciles.csv")
    terciles.to_csv(ter_path, index=False, float_format="%.6f")
    log(f"event_study_terciles.csv -> {ter_path}")

    # ---------------------------------------------------------------- 对拍（C 节）
    log("CROSS-CHECK START: 组合层 N=10 trades vs 事件层")
    n_mismatch_t1 = 0
    n_shares_diff = 0
    n_shares_same_checked = 0
    reverse_disc: list[dict] = []   # 事件层 dropped 而组合层成交
    open_vs_incomplete: list[dict] = []
    for (sig, cfg), tr in n10_trades.items():
        spec = CONFIGS[cfg]
        ev = pd.read_parquet(EVENTS[sig])
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        md = se.load_market_data(ev["ts_code"].unique().tolist(), START, END, REPO_ROOT)
        cal_arr = list(md.calendar)
        cal_index = {d: i for d, i in zip(cal_arr, range(len(cal_arr)))}
        stock_atr = {c: se3.atr_series(df, VOL_LB) for c, df in md.daily.items()} \
            if cfg == "B15" else {}
        esub = full[(full["signal"] == sig) & (full["config"] == cfg)] \
            .set_index(["ts_code", "event_date"])

        for t in tr.itertuples():
            key = (t.ts_code, pd.Timestamp(t.event_date))
            assert key in esub.index, f"{sig}__{cfg} 组合成交 {key} 事件层缺行"
            erow = esub.loc[key]
            # 反向披露：事件层未成交但组合层成交
            if erow["status"] != "closed":
                reverse_disc.append(dict(signal=sig, config=cfg, ts_code=t.ts_code,
                                         event_date=str(pd.Timestamp(t.event_date).date()),
                                         event_status=erow["status"]))
                continue
            # 第一层：钉住组合层实际股数重放，全字段逐位断言
            rep = simulate_event(t.ts_code, pd.Timestamp(t.event_date), spec, md,
                                 cal_index, cal_arr, stock_atr, mkt_atr_cache,
                                 shares_override=int(t.shares))
            assert rep["status"] == "closed", \
                f"{sig}__{cfg} {key} 钉股数重放未闭合: {rep['status']}"
            for f in CMP_FIELDS:
                a, b = rep[f], getattr(t, f)
                if f in ("entry_date", "exit_date"):
                    ok = pd.Timestamp(a) == pd.Timestamp(b)
                elif f == "exit_reason":
                    ok = a == b
                else:
                    ok = abs(float(a) - float(b)) <= XTOL
                if not ok:
                    n_mismatch_t1 += 1
                    log(f"TIER1 MISMATCH {sig}__{cfg} {key} field={f}: "
                        f"event={a} vs portfolio={b}")
            # 第二层：固定预算股数 vs 组合层股数
            if int(erow["shares"]) != int(t.shares):
                n_shares_diff += 1
            else:
                n_shares_same_checked += 1
                assert abs(float(erow["net_pnl"]) - float(t.net_pnl)) <= XTOL, \
                    f"{sig}__{cfg} {key} 股数一致子集 net_pnl 不符"
        # 期末未平仓持仓 ↔ 事件层 incomplete 披露
        op = n10_open[(sig, cfg)]
        for p in op.itertuples():
            key = (p.ts_code, pd.Timestamp(p.event_date))
            if key in esub.index:
                open_vs_incomplete.append(dict(
                    signal=sig, config=cfg, ts_code=p.ts_code,
                    event_date=str(pd.Timestamp(p.event_date).date()),
                    event_status=esub.loc[key, "status"]))

    assert n_mismatch_t1 == 0, f"对拍第一层存在 {n_mismatch_t1} 处字段不一致，停止"
    log(f"CROSS-CHECK DONE: tier1 全量逐位一致（组合成交笔数="
        f"{sum(len(t) for t in n10_trades.values())}，mismatch=0）; "
        f"tier2 股数不一致笔数={n_shares_diff}，股数一致子集逐位断言笔数={n_shares_same_checked}")
    if reverse_disc:
        rd = pd.DataFrame(reverse_disc)
        rd_path = os.path.join(BT_DIR, "crosscheck_reverse_disclosure.csv")
        rd.to_csv(rd_path, index=False)
        log(f"反向披露：事件层未闭合而组合层成交 {len(rd)} 笔 -> {rd_path}")
        log(rd.to_string(index=False))
    else:
        log("反向披露：0 笔（组合层成交笔在事件层全部闭合）")
    if open_vs_incomplete:
        ov = pd.DataFrame(open_vs_incomplete)
        log("期末未平仓持仓的事件层 status 分布: "
            + str(ov["event_status"].value_counts().to_dict()))

    disc_df = pd.DataFrame(disc_rows)
    disc_path = os.path.join(BT_DIR, "event_study_disclosure.csv")
    disc_df.to_csv(disc_path, index=False)
    log(f"event_study_disclosure.csv -> {disc_path}")
    log(f"EVENT STUDY DONE ({time.time() - t_all:.0f}s)")

    print("\n===== EVENT STUDY SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== E1-H12 dif_lift TERCILES =====")
    print(terciles.to_string(index=False))
    print("\n===== DISCLOSURE =====")
    print(disc_df.to_string(index=False))


if __name__ == "__main__":
    main()
