#!/usr/bin/env python3
"""T8 三类策略验证段调优 runner —— issue #28 预登记网格执行（2026-09-03 预登记评论）。

执行清单（全部验证段 2019-01-01..2022-10-31，merged pool 定版分数信号源）：
 1. 类 A 固定止盈止损 20 配置（v8 中心 +25%/-14%/H16/N3）。
 2. 类 B 波动率自适应 20 配置（v12 param2 中心，ATR 分档每日 <=t-1 重算）。
 3. 类 C 分数衰减退出 16 配置（日频打分面板重算，top_k/margin/H/N 小网格）。
 4. 裁决：每类最优 = 净年化超额（对 000905.SH 同期年化）最高，平局夏普高者，再平局配置号小者。
 5. 类 C 出场行为断言：全部 C 配置交易逐笔独立重放（出场原因/日期/价格由原始数据重算），
    三分支（rank_out/score_drop/horizon）全网格合计均须真实触发。
 6. 落盘：每类 CSV（配置为行）+ 最优配置 runs 明细 + tuning_summary.json + 本脚本日志。

防多重比较纪律：网格即预登记全集，运行后不做任何追加配置。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO_ROOT, "v3_pipeline", "scripts"))

import strategy_engine as se  # noqa: E402
import strategy_engine_v3 as se3  # noqa: E402
from run_strategy_family import compute_metrics, index_metrics  # noqa: E402

OUT_ROOT = os.path.join(REPO_ROOT, "v3_pipeline/reports/strategy_tuning")
PROGRESS_LOG = os.path.join(OUT_ROOT, "tuning_progress.log")
PANEL_PATH = os.path.join(OUT_ROOT, "daily_score_panel_20190101_20221031.parquet")

START, END = "2019-01-01", "2022-10-31"
BENCH = "000905.SH"
VOL_LOOKBACKS = (14, 21)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 预登记网格
def grid_a() -> list[dict]:
    """类 A 固定止盈止损 20 配置（预登记评论 A1-A20）。"""
    cfgs = []
    i = 0
    for tp in (0.20, 0.25, 0.30, 0.35):
        for sl in (-0.10, -0.14, -0.18):
            i += 1
            cfgs.append(dict(name=f"A{i}", tp=tp, sl=sl, horizon=16, n_slots=3))
    i += 1
    cfgs.append(dict(name=f"A{i}", tp=0.25, sl=-0.14, horizon=12, n_slots=3))
    i += 1
    cfgs.append(dict(name=f"A{i}", tp=0.25, sl=-0.14, horizon=20, n_slots=3))
    i += 1
    cfgs.append(dict(name=f"A{i}", tp=0.25, sl=-0.14, horizon=16, n_slots=2))
    i += 1
    cfgs.append(dict(name=f"A{i}", tp=0.25, sl=-0.14, horizon=16, n_slots=5))
    for tp, sl, h in ((0.20, -0.10, 12), (0.35, -0.18, 20),
                      (0.20, -0.18, 20), (0.35, -0.10, 12)):
        i += 1
        cfgs.append(dict(name=f"A{i}", tp=tp, sl=sl, horizon=h, n_slots=3))
    assert i == 20
    return cfgs


V12_CENTER = dict(tp=0.25, sl=-0.14, horizon=16, n_slots=3, vol_lookback=21,
                  vol_high_thresh=1.8, vol_low_thresh=0.6, vol_profit_mult=1.5,
                  vol_stop_mult=1.1, low_vol_profit_mult=1.0)


def grid_b() -> list[dict]:
    """类 B 波动率自适应 20 配置（预登记评论 B1-B20）。"""
    perturb = [
        dict(vol_lookback=14),
        dict(vol_high_thresh=2.5),
        dict(vol_low_thresh=0.4),
        dict(vol_low_thresh=0.8),
        dict(vol_profit_mult=1.2),
        dict(vol_profit_mult=2.0),
        dict(vol_stop_mult=1.5),
        dict(low_vol_profit_mult=0.8),
        dict(tp=0.20),
        dict(tp=0.30),
        dict(tp=0.35),
        dict(sl=-0.10),
        dict(sl=-0.18),
        dict(horizon=12),
        dict(horizon=20),
        dict(n_slots=5),
        dict(vol_profit_mult=2.0, vol_high_thresh=2.5),
        dict(tp=0.35, vol_profit_mult=2.0, vol_high_thresh=2.5),
        dict(vol_lookback=14, vol_low_thresh=0.4, vol_profit_mult=1.2),
    ]
    cfgs = [dict(name="B1", **V12_CENTER)]
    for i, p in enumerate(perturb, start=2):
        cfgs.append(dict(name=f"B{i}", **{**V12_CENTER, **p}))
    assert len(cfgs) == 20
    return cfgs


def grid_c() -> list[dict]:
    """类 C 分数衰减退出 16 配置（预登记评论 C1-C16）。"""
    cfgs = []
    i = 0
    for h in (5, 10, 15, 20, 25, 30):
        for n in (3, 5):
            i += 1
            cfgs.append(dict(name=f"C{i}", horizon=h, n_slots=n,
                             top_k=5, score_margin=0.0))
    i += 1
    cfgs.append(dict(name=f"C{i}", horizon=15, n_slots=3, top_k=5, score_margin=0.05))
    i += 1
    cfgs.append(dict(name=f"C{i}", horizon=15, n_slots=3, top_k=5, score_margin=0.10))
    i += 1
    cfgs.append(dict(name=f"C{i}", horizon=15, n_slots=3, top_k=3, score_margin=0.0))
    i += 1
    cfgs.append(dict(name=f"C{i}", horizon=15, n_slots=3, top_k=8, score_margin=0.0))
    assert i == 16
    return cfgs


# ---------------------------------------------------------------- 类 B 数据：全市场 ATR 均值
def build_mkt_atr(calendar: list[pd.Timestamp], lookback: int,
                  log_path: str | None = None) -> pd.Series:
    """全市场逐日 ATR(lookback) 均值。轻量独立加载全部日线（窗口切片）。"""
    t0 = time.time()
    lo = calendar[0] - pd.Timedelta(days=90)
    hi = calendar[-1]
    sums: dict[pd.Timestamp, float] = {}
    cnts: dict[pd.Timestamp, int] = {}
    files = sorted(os.listdir(os.path.join(REPO_ROOT, "stock_data/daily")))
    for i, fn in enumerate(files):
        code = fn[:-8]
        df = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/daily", fn),
                             columns=["trade_date", "open", "high", "low", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= lo) & (df["trade_date"] <= hi)]
        if len(df) < lookback:
            continue
        df = df.set_index("trade_date").sort_index()
        a = se3.atr_series(df, lookback)
        a = a.dropna()
        for d, v in a.items():
            if v > 0:
                sums[d] = sums.get(d, 0.0) + float(v)
                cnts[d] = cnts.get(d, 0) + 1
        if (i + 1) % 1000 == 0:
            se._log(log_path, f"  mkt_atr lb={lookback} {i + 1}/{len(files)} "
                              f"({time.time() - t0:.0f}s)")
    out = pd.Series({d: sums[d] / cnts[d] for d in sums}).sort_index()
    out = out.reindex(calendar)
    se._log(log_path, f"mkt_atr lb={lookback} days_valid={out.notna().sum()} "
                      f"({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------- 类 C 出场行为独立重放断言
def replay_score_decay_assertions(trades: pd.DataFrame, cfg: dict,
                                  events: pd.DataFrame, panel: pd.DataFrame,
                                  calendar: list[pd.Timestamp],
                                  repo_root: str,
                                  open_positions: pd.DataFrame | None = None) -> dict:
    """逐笔独立重放类 C 出场：不读引擎内部状态，由 原始日线+scores_final+面板 重算。

    对每笔交易断言：
      1. exit_date > entry_date（T+1）。
      2. horizon：exit_date 为 entry_date 起第 H 个交易日，exit_raw_price=当日收盘。
      3. rank_out/score_drop：决策日 D = exit_date 前该股最后一个有行情的交易日
         （停牌日无面板分，引擎按 missing-score hold 跳过评估，出场意图顺延至下一交易日开盘）；
         候选集 = D 日新鲜信号 prob ∪ D 日全部在持股票（面板分）；
         rank_out -> 该股不在前 top_k；score_drop -> score_D < buy_prob×(1-margin)；
         exit_raw_price = exit_date 开盘价（独立读原始日线）。
    buy_prob 自 scores_final 原始事件行读取（不用引擎记录值）。

    在持集合重建口径（与引擎 step-2 评估时刻的 positions 逐一对齐）：
      - 已平仓交易 q：entry_date <= D 且（exit_date > D，或 exit_date == D 且 exit_reason=='horizon'
        —— 收盘到期卖出在 D 日评估时仍在仓；rank_out/score_drop 为 D 日开盘执行，评估前已离场）。
      - 窗口末仍在仓持仓（open_positions，含停牌僵尸仓，永不出现于 trades）：
        entry_date <= D 即在持。T9 测试段实锤案例：600781.SH 入场日即最后行情日，
        僵尸仓进入候选集改变 top_k 排名（见 t9_report.md）。
    """
    sp = panel.set_index(["ts_code", "date"])["prob"]
    ev_prob = events.set_index(["ts_code", "event_date"])["prob"]
    by_date = {d: g for d, g in events.groupby("event_date")}
    code_days = {c: g["date"].to_numpy() for c, g in panel.groupby("ts_code")}
    cal_index = {d: i for i, d in enumerate(calendar)}
    top_k, margin, horizon = cfg["top_k"], cfg["score_margin"], cfg["horizon"]
    if open_positions is None:
        open_positions = pd.DataFrame(
            columns=["ts_code", "event_date", "entry_date", "buy_prob"])

    n_checked, mismatches = 0, []
    daily_cache: dict[str, pd.DataFrame] = {}

    def raw_bar(code, day):
        if code not in daily_cache:
            df = pd.read_parquet(os.path.join(repo_root, "stock_data/daily",
                                              f"{code}.parquet"),
                                 columns=["trade_date", "open", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            daily_cache[code] = df.set_index("trade_date").sort_index()
        df = daily_cache[code]
        return df.loc[day] if day in df.index else None

    for tr in trades.itertuples():
        n_checked += 1
        tag = f"{tr.ts_code} {tr.entry_date.date()}->{tr.exit_date.date()}"
        if not tr.exit_date > tr.entry_date:
            mismatches.append((tag, "exit not after entry"))
            continue
        held = cal_index[tr.exit_date] - cal_index[tr.entry_date] + 1
        if tr.exit_reason == "horizon":
            if held != horizon:
                mismatches.append((tag, f"held {held} != H {horizon}"))
                continue
            bar = raw_bar(tr.ts_code, tr.exit_date)
            if bar is None or abs(float(bar["close"]) - tr.exit_raw_price) > 1e-9:
                mismatches.append((tag, "horizon exit price != close"))
            continue
        # rank_out / score_drop：决策日 = 出场日前该股最后有行情的交易日（停牌跳过评估）
        days = code_days.get(tr.ts_code)
        if days is None:
            mismatches.append((tag, "no panel days for stock"))
            continue
        prior = days[days < np.datetime64(tr.exit_date)]
        if not len(prior):
            mismatches.append((tag, "no bar day before exit"))
            continue
        dec_day = pd.Timestamp(prior[-1])
        s_t = sp.get((tr.ts_code, dec_day), np.nan)
        if not np.isfinite(s_t):
            mismatches.append((tag, "panel score missing at decision day"))
            continue
        buy_prob = float(ev_prob.get((tr.ts_code, tr.event_date), np.nan))
        if not np.isfinite(buy_prob):
            mismatches.append((tag, "buy prob missing in scores_final"))
            continue
        # 候选集重建：D 日新鲜信号 + D 日在持股票（含本股，按 ts_code 去重取高）
        cand: dict[str, float] = {}

        def _merge(code: str, score: float) -> None:
            if np.isfinite(score):
                cand[code] = max(cand.get(code, -np.inf), float(score))

        if dec_day in by_date:
            for r in by_date[dec_day].itertuples():
                _merge(r.ts_code, float(r.prob))
        for q in trades.itertuples():
            # D 日评估时在持：已入场；出场日晚于 D，或 D 日收盘到期卖出（评估时仍在仓）
            if q.entry_date <= dec_day and (q.exit_date > dec_day or
                                            (q.exit_date == dec_day
                                             and q.exit_reason == "horizon")):
                _merge(q.ts_code, sp.get((q.ts_code, dec_day), np.nan))
        for q in open_positions.itertuples():
            # 窗口末仍在仓持仓（含僵尸仓）：永不平仓故不在 trades，D 日已入场即在持
            if q.entry_date <= dec_day:
                _merge(q.ts_code, sp.get((q.ts_code, dec_day), np.nan))
        ranked = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
        top_codes = [c for c, _ in ranked[:top_k]]
        threshold = buy_prob * (1.0 - margin)
        if tr.exit_reason == "rank_out":
            ok = tr.ts_code not in top_codes
        elif tr.exit_reason == "score_drop":
            ok = tr.ts_code in top_codes and s_t < threshold
        else:
            ok = False
        if not ok:
            mismatches.append((tag, f"reason {tr.exit_reason} not reproduced"))
            continue
        bar = raw_bar(tr.ts_code, tr.exit_date)
        if bar is None or abs(float(bar["open"]) - tr.exit_raw_price) > 1e-9:
            mismatches.append((tag, "exit price != decision+1 open"))
    return {"n_trades": len(trades), "n_checked": n_checked,
            "n_mismatch": len(mismatches), "mismatches": mismatches[:10]}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    log("T8 RUN START（issue #28 预登记网格：A20 + B20 + C16，验证段）")

    events = se3.load_score_events(START, END, REPO_ROOT, PROGRESS_LOG)
    md = se.load_market_data(events["ts_code"].unique().tolist(), START, END,
                             REPO_ROOT, log_path=PROGRESS_LOG)
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    bench = index_metrics(BENCH, START, END)
    log(f"基准 {BENCH}: ann={bench['ann_return']:+.4f} mdd={bench['max_dd']:.4f}")

    log("构建全市场 ATR 均值序列（类 B，LB=14/21）...")
    mkt_atr = {lb: build_mkt_atr(md.calendar, lb, PROGRESS_LOG)
               for lb in VOL_LOOKBACKS}
    stock_atr = {lb: {c: se3.atr_series(df, lb) for c, df in md.daily.items()}
                 for lb in VOL_LOOKBACKS}

    log("加载日频打分面板（类 C）...")
    panel = pd.read_parquet(PANEL_PATH, columns=["ts_code", "date", "prob"])
    panel["date"] = pd.to_datetime(panel["date"])

    summary: dict[str, list[dict]] = {"A": [], "B": [], "C": []}

    def run_row(cls: str, cfg: dict, spec: se3.ExitSpec, **kwargs) -> tuple[dict, dict]:
        t0 = time.time()
        res = se3.run_backtest_v3(events, md, n_slots=cfg["n_slots"], exit_spec=spec,
                                  **kwargs)
        m = compute_metrics(res["equity"], res["trades"], res["stats"])
        excess = m.get("ann_return", np.nan) - bench["ann_return"]
        row = dict(config=cfg["name"], strategy_class=cls,
                   **{k: v for k, v in cfg.items() if k != "name"},
                   ann_return=m.get("ann_return"), excess_ann=excess,
                   max_dd=m.get("max_dd"), sharpe=m.get("sharpe"),
                   calmar=m.get("calmar"), turnover=m.get("turnover"),
                   n_trades=m.get("n_trades"), coverage=m.get("coverage"),
                   utilization=m.get("utilization"),
                   exits_tp=res["stats"].get("exits_tp"),
                   exits_sl=res["stats"].get("exits_sl"),
                   exits_horizon=res["stats"].get("exits_horizon"),
                   exits_rank_out=res["stats"].get("exits_rank_out"),
                   exits_score_drop=res["stats"].get("exits_score_drop"),
                   elapsed_s=round(time.time() - t0, 2))
        log(f"{cls} {cfg['name']} done {row['elapsed_s']}s | ann={m.get('ann_return', float('nan')):+.4f} "
            f"excess={excess:+.4f} sharpe={m.get('sharpe', float('nan')):.3f} "
            f"trades={m.get('n_trades', 0)}")
        return row, res

    # ---- 类 A ----
    for cfg in grid_a():
        spec = se3.ExitSpec.fixed_tp_sl(tp=cfg["tp"], sl=cfg["sl"], horizon=cfg["horizon"])
        row, _ = run_row("A", cfg, spec)
        summary["A"].append(row)
    log("类 A 20/20 完成")

    # ---- 类 B ----
    for cfg in grid_b():
        lb = cfg["vol_lookback"]
        spec = se3.ExitSpec.vol_adaptive(
            tp=cfg["tp"], sl=cfg["sl"], horizon=cfg["horizon"], vol_lookback=lb,
            vol_high_thresh=cfg["vol_high_thresh"], vol_low_thresh=cfg["vol_low_thresh"],
            vol_profit_mult=cfg["vol_profit_mult"], vol_stop_mult=cfg["vol_stop_mult"],
            low_vol_profit_mult=cfg["low_vol_profit_mult"])
        row, _ = run_row("B", cfg, spec, mkt_atr=mkt_atr[lb],
                         stock_atr=stock_atr[lb])
        summary["B"].append(row)
    log("类 B 20/20 完成")

    # ---- 类 C ----
    c_results: dict[str, dict] = {}
    for cfg in grid_c():
        spec = se3.ExitSpec.score_decay(horizon=cfg["horizon"], top_k=cfg["top_k"],
                                        score_margin=cfg["score_margin"])
        row, res = run_row("C", cfg, spec, score_panel=panel)
        summary["C"].append(row)
        c_results[cfg["name"]] = (cfg, res)
    log("类 C 16/16 完成")

    # ---- 落盘三类表（配置为行）----
    tables = {}
    for cls in "ABC":
        df = pd.DataFrame(summary[cls])
        path = os.path.join(OUT_ROOT, f"tuning_class_{cls}.csv")
        df.to_csv(path, index=False)
        tables[cls] = df
        log(f"类 {cls} 表落盘 {path} ({len(df)} 配置)")

    # ---- 裁决（预登记规则：excess_ann 降序 → sharpe 降序 → 配置号小者）----
    adjudication = {}
    for cls in "ABC":
        df = tables[cls]
        cfg_num = df["config"].str.extract(r"(\d+)")[0].astype(int)
        best = df.assign(_n=cfg_num).sort_values(
            ["excess_ann", "sharpe", "_n"], ascending=[False, False, True]).iloc[0]
        adjudication[cls] = {"config": best["config"],
                             "excess_ann": float(best["excess_ann"]),
                             "sharpe": float(best["sharpe"]),
                             "ann_return": float(best["ann_return"])}
        log(f"类 {cls} 最优: {best['config']} excess={best['excess_ann']:+.4f} "
            f"sharpe={best['sharpe']:.3f}")

    # ---- 最优配置 runs 明细落盘（三类各一）----
    best_specs = {
        "A": lambda c: se3.ExitSpec.fixed_tp_sl(tp=c["tp"], sl=c["sl"],
                                                horizon=c["horizon"]),
        "B": lambda c: se3.ExitSpec.vol_adaptive(
            tp=c["tp"], sl=c["sl"], horizon=c["horizon"],
            vol_lookback=int(c["vol_lookback"]),
            vol_high_thresh=c["vol_high_thresh"], vol_low_thresh=c["vol_low_thresh"],
            vol_profit_mult=c["vol_profit_mult"], vol_stop_mult=c["vol_stop_mult"],
            low_vol_profit_mult=c["low_vol_profit_mult"]),
        "C": lambda c: se3.ExitSpec.score_decay(horizon=int(c["horizon"]),
                                                top_k=int(c["top_k"]),
                                                score_margin=c["score_margin"]),
    }
    for cls in "ABC":
        name = adjudication[cls]["config"]
        cfg = next(c for c in tables[cls].to_dict("records") if c["config"] == name)
        spec = best_specs[cls](cfg)
        kwargs = {}
        if cls == "B":
            kwargs = dict(mkt_atr=mkt_atr[int(cfg["vol_lookback"])],
                          stock_atr=stock_atr[int(cfg["vol_lookback"])])
        if cls == "C":
            kwargs = dict(score_panel=panel)
        out_dir = os.path.join(OUT_ROOT, "runs", f"val_{name}")
        se3.run_config_v3(events, md, n_slots=int(cfg["n_slots"]), exit_spec=spec,
                          start=START, end=END, out_dir=out_dir,
                          log_path=PROGRESS_LOG, **kwargs)
        log(f"最优配置 {name} 明细落盘 {out_dir}")

    # ---- 类 C 出场行为断言（全网格合计 + 最优配置逐笔重放）----
    cdf = tables["C"]
    branch_totals = dict(rank_out=int(cdf["exits_rank_out"].sum()),
                         score_drop=int(cdf["exits_score_drop"].sum()),
                         horizon=int(cdf["exits_horizon"].sum()))
    branches_ok = all(v > 0 for v in branch_totals.values())
    log(f"类 C 三分支全网格合计: {branch_totals} -> {'PASS' if branches_ok else 'FAIL'}")

    best_c = adjudication["C"]["config"]
    cfg_c, res_c = c_results[best_c]
    replay = replay_score_decay_assertions(res_c["trades"], cfg_c, events, panel,
                                           md.calendar, REPO_ROOT)
    replay_ok = replay["n_mismatch"] == 0 and replay["n_trades"] > 0
    log(f"类 C 最优 {best_c} 逐笔重放: {replay['n_checked']} 笔, "
        f"不一致 {replay['n_mismatch']} -> {'PASS' if replay_ok else 'FAIL'}")

    summary_json = {
        "issue": 28, "window": [START, END], "benchmark": BENCH,
        "benchmark_ann": bench["ann_return"],
        "adjudication": adjudication,
        "class_c_branch_totals": branch_totals,
        "class_c_branch_assertion": "PASS" if branches_ok else "FAIL",
        "class_c_replay": {**replay, "pass": replay_ok},
        "elapsed_s": round(time.time() - t_all, 1),
    }
    with open(os.path.join(OUT_ROOT, "tuning_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2, default=str)
    log(f"T8 RUN DONE ({time.time() - t_all:.0f}s) summary -> tuning_summary.json")

    if not (branches_ok and replay_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
