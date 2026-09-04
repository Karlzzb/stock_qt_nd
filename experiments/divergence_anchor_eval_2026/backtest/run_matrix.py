#!/usr/bin/env python3
"""背离信号两变体 × 三出场 预登记回测矩阵执行器（README.md 为预登记，先写后跑）。

矩阵：{events_v1, events_v2} × {A13, B15, E1-H12}，n_slots=3，
窗口 2026-01-01..2026-08-31，基准 000905.SH（index_metrics 口径，同 T8/T9）。
适配器直接构造引擎所需事件 DataFrame（ts_code / event_date / prob←dif_lift），
不调用 load_score_events。硬断言：prob 无缺失且全正、(ts_code,event_date) 无重复、
每个 run trades 非空且全部 entry_date > event_date。
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

import strategy_engine as se  # noqa: E402  v1 冻结引擎（只读复用常量与 load_market_data）
import strategy_engine_v3 as se3  # noqa: E402  v3 冻结引擎（只读调用）
from run_strategy_family import compute_metrics, index_metrics  # noqa: E402  T8/T9 同口径
from run_strategy_tuning import build_mkt_atr  # noqa: E402  类 B 全市场 ATR 均值（T8/T9 同源）

BT_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/backtest")
EVENTS = {
    "events_v1": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v1.parquet"),
    "events_v2": os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026/events_v2.parquet"),
}
LOG_PATH = os.path.join(BT_DIR, "run_matrix.log")

START, END = "2026-01-01", "2026-08-31"
BENCH = "000905.SH"
N_SLOTS = 3
VOL_LB = 21

CONFIGS = {
    "A13": se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12),
    "B15": se3.ExitSpec.vol_adaptive(tp=0.25, sl=-0.14, horizon=12, vol_lookback=VOL_LB,
                                     vol_high_thresh=1.8, vol_low_thresh=0.6,
                                     vol_profit_mult=1.5, vol_stop_mult=1.1,
                                     low_vol_profit_mult=1.0),
    "E1-H12": se3.ExitSpec.horizon_only(horizon=12),
}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_signal_events(path: str, name: str) -> pd.DataFrame:
    """构造引擎所需事件 DataFrame：ts_code / event_date / prob←dif_lift（硬断言齐全）。"""
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    assert df["dif_lift"].notna().all(), f"{name}: dif_lift 存在缺失"
    assert (df["dif_lift"] > 0).all(), f"{name}: dif_lift 非全正"
    assert not df.duplicated(["ts_code", "event_date"]).any(), f"{name}: (ts_code,event_date) 重复"
    ev = df[["ts_code", "event_date"]].copy()
    ev["prob"] = df["dif_lift"].astype(float)
    ev = ev.sort_values(["event_date", "ts_code"]).reset_index(drop=True)
    log(f"{name}: events={len(ev)} unique_stocks={ev['ts_code'].nunique()} "
        f"event_date=[{ev['event_date'].min().date()}..{ev['event_date'].max().date()}] "
        f"prob_min={ev['prob'].min():.6f}")
    return ev


def main() -> None:
    t_all = time.time()
    log(f"MATRIX RUN START window=[{START}..{END}] bench={BENCH} "
        f"configs={list(CONFIGS)} signals={list(EVENTS)}")

    bench = index_metrics(BENCH, START, END)
    assert np.isfinite(bench["ann_return"]), "基准序列未覆盖回测窗"
    log(f"基准 {BENCH} [{START}..{END}]: ann={bench['ann_return']:+.4f} mdd={bench['max_dd']:.4f}")

    summary_rows: list[dict] = []
    mkt_atr_cache: pd.Series | None = None

    for sig_name, ev_path in EVENTS.items():
        events = load_signal_events(ev_path, sig_name)
        md = se.load_market_data(events["ts_code"].unique().tolist(), START, END,
                                 REPO_ROOT, log_path=LOG_PATH)
        events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
        log(f"{sig_name}: events_in_calendar={len(events)} calendar_days={len(md.calendar)}")

        # 类 B 数据：日历跨信号变体一致（取自 000905.SH 指数日），mkt_atr 只构建一次
        if mkt_atr_cache is None:
            log(f"构建全市场 ATR 均值序列（类 B，LB={VOL_LB}，窗口日历）...")
            mkt_atr_cache = build_mkt_atr(md.calendar, VOL_LB, LOG_PATH)
            assert mkt_atr_cache.notna().all(), "mkt_atr 在回测日历上存在缺失"
        stock_atr = {c: se3.atr_series(df, VOL_LB) for c, df in md.daily.items()}

        for cfg_name, spec in CONFIGS.items():
            t0 = time.time()
            kwargs = {}
            if spec.strategy_class == "vol_adaptive":
                kwargs = dict(mkt_atr=mkt_atr_cache, stock_atr=stock_atr)
            out_dir = os.path.join(BT_DIR, "runs", f"{sig_name}__{cfg_name}")
            res = se3.run_config_v3(events, md, n_slots=N_SLOTS, exit_spec=spec,
                                    start=START, end=END, out_dir=out_dir,
                                    log_path=LOG_PATH, **kwargs)
            trades = res["trades"]
            # ---- 自检硬断言：trades 非空且全部 entry_date > event_date ----
            assert not trades.empty, f"{sig_name}__{cfg_name}: trades 为空"
            assert (trades["entry_date"] > trades["event_date"]).all(), \
                f"{sig_name}__{cfg_name}: 存在 entry_date <= event_date 的成交（泄漏）"

            m = compute_metrics(res["equity"], trades, res["stats"])
            s = res["stats"]
            excess_pp = (m["ann_return"] - bench["ann_return"]) * 100.0
            row = dict(
                signal=sig_name, config=cfg_name,
                ann_return=m["ann_return"], excess_ann_pp=excess_pp,
                sharpe=m["sharpe"], max_dd=m["max_dd"], calmar=m["calmar"],
                n_trades=m["n_trades"], turnover=m["turnover"],
                utilization=m["utilization"],
                exits_tp=s["exits_tp"], exits_sl=s["exits_sl"],
                exits_horizon=s["exits_horizon"],
                open_at_end=s["open_at_end"],
                vol_fallback_mid=s["vol_fallback_mid"],
                dropped_slot_full=s["dropped_slot_full"],
                dropped_limitup=s["dropped_limitup"],
                dropped_no_quote=s["dropped_no_quote"],
                dropped_cash=s["dropped_cash"],
                deferred_exits=s["deferred_exits"],
                final_equity=m["final_equity"],
                bench_ann=bench["ann_return"],
            )
            summary_rows.append(row)
            log(f"{sig_name}__{cfg_name} done {time.time() - t0:.0f}s | "
                f"ann={m['ann_return']:+.4f} excess={excess_pp:+.2f}pp "
                f"sharpe={m['sharpe']:.3f} mdd={m['max_dd']:.4f} trades={m['n_trades']} "
                f"open_at_end={s['open_at_end']}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    log(f"MATRIX RUN DONE ({time.time() - t_all:.0f}s) -> {summary_path}")

    # ---- 打印汇总表（行=配置）----
    show = summary[["signal", "config", "ann_return", "excess_ann_pp", "sharpe",
                    "max_dd", "calmar", "n_trades", "turnover", "utilization",
                    "exits_tp", "exits_sl", "exits_horizon", "open_at_end",
                    "bench_ann"]].copy()
    for c in ("ann_return", "sharpe", "max_dd", "calmar", "turnover", "utilization",
              "bench_ann"):
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    show["excess_ann_pp"] = show["excess_ann_pp"].map(lambda v: f"{v:+.2f}")
    print("\n===== SUMMARY =====")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
