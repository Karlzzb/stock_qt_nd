#!/usr/bin/env python3
"""T9 测试段终审 runner —— issue #29 预登记评论 id 5521655421（2026-09-03）执行。

纪律约束（预登记）：
  1. 终审对象全冻结：A13（tp .25/sl -.14/H12/N3）、B15（v12 中心+H12）、C15（H15/N3/top_k3/margin0）。
  2. 每类配置在测试段（2022-11-01..2026-08-31）只跑一次；本脚本内置一次性护栏：
     裁决 JSON 已存在则拒绝运行，除非显式 --allow-rerun 并登记理由（仅工程正确性缺陷）。
  3. 过线标准（逐类独立）：净年化超额 > 基准 +15pp 且 Sharpe > 0.5。
  4. 类 C 全部交易逐笔独立重放断言（复用 T8 replay_score_decay_assertions，零不一致方可出数）。
  5. 测试段面板四断言先行（构建器 results json 全 PASS 才允许消费）。

落盘：v3_pipeline/reports/test_adjudication/
  runs/test_{A13,B15,C15}/（equity_curve.parquet / trades.parquet / stats.json）
  adjudication.json（指标 + 过线裁决 + 断言台账）
  t9_progress.log（阶段日志与心跳）
"""
from __future__ import annotations

import argparse
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
from run_strategy_tuning import (  # noqa: E402
    V12_CENTER, build_mkt_atr, replay_score_decay_assertions,
)

OUT_ROOT = os.path.join(REPO_ROOT, "v3_pipeline/reports/test_adjudication")
PROGRESS_LOG = os.path.join(OUT_ROOT, "t9_progress.log")
PANEL_PATH = os.path.join(
    REPO_ROOT, "v3_pipeline/reports/strategy_tuning",
    "daily_score_panel_20221101_20260831.parquet")
PANEL_RESULTS = PANEL_PATH.replace(".parquet", "_results.json")

START, END = "2022-11-01", "2026-08-31"
BENCH = "000905.SH"
EXCESS_THRESHOLD = 0.15
SHARPE_THRESHOLD = 0.5
PREREG_COMMENT_ID = 5521655421

# 预登记冻结三配置（T8 验证段裁决产出，参数一字不动）
CONFIGS = {
    "A13": dict(cls="A", n_slots=3,
                spec=se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12)),
    "B15": dict(cls="B", n_slots=3,
                spec=se3.ExitSpec.vol_adaptive(
                    tp=V12_CENTER["tp"], sl=V12_CENTER["sl"], horizon=12,
                    vol_lookback=V12_CENTER["vol_lookback"],
                    vol_high_thresh=V12_CENTER["vol_high_thresh"],
                    vol_low_thresh=V12_CENTER["vol_low_thresh"],
                    vol_profit_mult=V12_CENTER["vol_profit_mult"],
                    vol_stop_mult=V12_CENTER["vol_stop_mult"],
                    low_vol_profit_mult=V12_CENTER["low_vol_profit_mult"])),
    "C15": dict(cls="C", n_slots=3,
                spec=se3.ExitSpec.score_decay(horizon=15, top_k=3, score_margin=0.0)),
}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_panel_assertions() -> dict:
    """消费测试段面板前核验构建器四项断言全 PASS。"""
    if not os.path.exists(PANEL_RESULTS):
        raise SystemExit(f"测试段面板 results 缺失: {PANEL_RESULTS}（先跑 build_daily_score_panel.py）")
    with open(PANEL_RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    fails = [a["name"] for a in res["assertions"] if not a["pass"]]
    if fails:
        raise SystemExit(f"测试段面板断言未全过: {fails}")
    return {"path": PANEL_PATH, "window": res["window"],
            "panel_rows": res["panel_rows"], "n_events": res["n_events"],
            "assertions": {a["name"]: a["pass"] for a in res["assertions"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-rerun", default=None, metavar="REASON",
                    help="一次性护栏旁路：仅在工程正确性缺陷修复后重跑，理由必填并登记台账")
    args = ap.parse_args()
    adj_path = os.path.join(OUT_ROOT, "adjudication.json")
    rerun_of = None
    if os.path.exists(adj_path):
        if not args.allow_rerun:
            raise SystemExit(f"一次性护栏：{adj_path} 已存在，拒绝重跑。"
                             "如属工程缺陷修复，用 --allow-rerun '理由' 显式旁路并登记。")
        with open(adj_path, encoding="utf-8") as f:
            rerun_of = json.load(f).get("run_ts")
        log(f"!! 护栏旁路重跑（理由: {args.allow_rerun}；首次运行时间 {rerun_of}）")

    t_all = time.time()
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # 尝试史自阶段日志自举：每次 "T9 RUN START" 一行，含未写裁决文件的失败尝试
    prior_starts = 0
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG, encoding="utf-8") as f:
            prior_starts = sum(1 for ln in f if "T9 RUN START" in ln)
    attempt = prior_starts + 1
    log(f"T9 RUN START attempt#{attempt}（issue #29 预登记评论 {PREREG_COMMENT_ID}："
        f"A13/B15/C15 测试段一次性终审）")

    panel_meta = check_panel_assertions()
    log(f"测试段面板四断言全过: {panel_meta['assertions']} rows={panel_meta['panel_rows']}")

    events = se3.load_score_events(START, END, REPO_ROOT, PROGRESS_LOG)
    md = se.load_market_data(events["ts_code"].unique().tolist(), START, END,
                             REPO_ROOT, log_path=PROGRESS_LOG)
    events = events[events["event_date"].isin(set(md.calendar))].reset_index(drop=True)
    bench = index_metrics(BENCH, START, END)
    log(f"基准 {BENCH} [{START}..{END}]: ann={bench['ann_return']:+.4f} "
        f"mdd={bench['max_dd']:.4f}")

    log("构建全市场 ATR 均值序列（类 B，LB=21，测试段窗口）...")
    lb = V12_CENTER["vol_lookback"]
    mkt_atr = build_mkt_atr(md.calendar, lb, PROGRESS_LOG)
    stock_atr = {c: se3.atr_series(df, lb) for c, df in md.daily.items()}

    log("加载测试段日频打分面板（类 C）...")
    panel = pd.read_parquet(PANEL_PATH, columns=["ts_code", "date", "prob"])
    panel["date"] = pd.to_datetime(panel["date"])

    results: dict[str, dict] = {}
    c15_trades = None
    c15_open = None
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        kwargs = {}
        if cfg["cls"] == "B":
            kwargs = dict(mkt_atr=mkt_atr, stock_atr=stock_atr)
        if cfg["cls"] == "C":
            kwargs = dict(score_panel=panel)
        out_dir = os.path.join(OUT_ROOT, "runs", f"test_{name}")
        res = se3.run_config_v3(events, md, n_slots=cfg["n_slots"],
                                exit_spec=cfg["spec"], start=START, end=END,
                                out_dir=out_dir, log_path=PROGRESS_LOG, **kwargs)
        m = compute_metrics(res["equity"], res["trades"], res["stats"])
        excess = m.get("ann_return", np.nan) - bench["ann_return"]
        passed = bool(excess > EXCESS_THRESHOLD and m.get("sharpe", np.nan) > SHARPE_THRESHOLD)
        results[name] = {
            "strategy_class": cfg["cls"], "spec": cfg["spec"].__dict__,
            "n_slots": cfg["n_slots"],
            "ann_return": m.get("ann_return"), "excess_ann": excess,
            "max_dd": m.get("max_dd"), "sharpe": m.get("sharpe"),
            "calmar": m.get("calmar"), "n_trades": m.get("n_trades"),
            "turnover": m.get("turnover"), "utilization": m.get("utilization"),
            "yearly_returns": m.get("yearly_returns"),
            "pass_threshold": passed,
        }
        log(f"{name} done {time.time() - t0:.0f}s | ann={m.get('ann_return', float('nan')):+.4f} "
            f"excess={excess:+.4f} sharpe={m.get('sharpe', float('nan')):.3f} "
            f"trades={m.get('n_trades', 0)} -> {'过线' if passed else '未过线'}")
        if name == "C15":
            c15_trades = res["trades"]
            c15_open = res["open_positions"]

    # ---- 类 C 逐笔独立重放断言（在持集合 = trades ∪ open_positions，含停牌僵尸仓）----
    c15_spec = CONFIGS["C15"]["spec"]
    replay = replay_score_decay_assertions(
        c15_trades, dict(top_k=c15_spec.top_k, score_margin=c15_spec.score_margin,
                         horizon=c15_spec.horizon),
        events, panel, md.calendar, REPO_ROOT, open_positions=c15_open)
    replay_ok = replay["n_mismatch"] == 0 and replay["n_trades"] > 0
    log(f"类 C C15 逐笔重放: {replay['n_checked']} 笔, 不一致 {replay['n_mismatch']} "
        f"-> {'PASS' if replay_ok else 'FAIL'}")
    if not replay_ok:
        raise SystemExit(f"类 C 逐笔重放断言失败: {replay['mismatches'][:3]}")

    adjudication = {
        "issue": 29, "preregistration_comment_id": PREREG_COMMENT_ID,
        "run_ts": run_ts, "window": [START, END], "benchmark": BENCH,
        "benchmark_ann": bench["ann_return"], "benchmark_mdd": bench["max_dd"],
        "thresholds": {"excess_ann_gt": EXCESS_THRESHOLD,
                       "sharpe_gt": SHARPE_THRESHOLD},
        "results": results,
        "class_c_replay": {**replay, "pass": replay_ok},
        "panel_assertions": panel_meta["assertions"],
        "one_shot_guard": {"attempt": attempt,
                           "prior_run_starts": prior_starts,
                           "completed_rerun": rerun_of is not None,
                           "rerun_reason": args.allow_rerun,
                           "rerun_of": rerun_of},
        "elapsed_s": round(time.time() - t_all, 1),
    }
    with open(adj_path, "w", encoding="utf-8") as f:
        json.dump(adjudication, f, ensure_ascii=False, indent=2, default=str)
    n_pass = sum(1 for r in results.values() if r["pass_threshold"])
    log(f"T9 RUN DONE ({time.time() - t_all:.0f}s) 过线 {n_pass}/3 -> {adj_path}")


if __name__ == "__main__":
    main()
