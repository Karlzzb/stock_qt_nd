#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1(最早背离设计)按 #31 口径重跑统计 —— 预登记 = 同目录 README.md(先于跑数落盘,冻结,勿改)。

实现口径:
- V1 事件池:直接复用 issue #2 产物 v3_pipeline/reports/divergence_event_study/events.parquet
  (逐日截断因果模拟,1,033,193 事件;自检 b 独立抽样对拍后才放行)。
- 事件索引:events.parquet 的 t_idx/prev_idx 基于事件研究预处理序列
  (dropna(close) → drop_duplicates(trade_date) → sort_values(trade_date));
  模拟 worker 用同一预处理,逐事件断言序列[t_idx] 日期 == 事件日。
- 交易模拟:逐条复刻 #31 §2.3,直接调用 run_seeds.py 的 _simulate_one / load_limits / cluster_t(零改动)。
- 自检 a:用 #31 events_history_v1.parquet 以复用原语重跑 v6-1/H20,逐笔对拍 + 净笔均复现 +4.89%。
- v6 数字:引用 #31 落盘(summary_seed.csv / trades_seed.parquet 重算并与已发表数字核对,自检 d),不重跑 v6。
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO = "/home/karl/repos/personal/stock_qt_nd"
sys.path.insert(0, os.path.join(REPO, "experiments", "divergence_seed_trial_history"))
import run_seeds as rs  # noqa: E402  复用:模拟原语/涨跌停加载/cluster_t/成本引擎

sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))
import divergence_event_study as des  # noqa: E402  复用:逐日截断因果模拟 simulate_events_idx

SEED_DIR = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history"))
OUT_DIR = Path(os.path.join(REPO, "experiments", "v1_seed_recompare"))
LOG_PATH = OUT_DIR / "progress.log"
V1_EVENTS = Path(os.path.join(
    REPO, "v3_pipeline", "reports", "divergence_event_study", "events.parquet"))
V1_STATS = Path(os.path.join(
    REPO, "v3_pipeline", "reports", "divergence_event_study", "stats.json"))
CAL_PATH = Path(os.path.join(REPO, "stock_data", "daily", "000001.SH.parquet"))

rs.LOG_PATH = LOG_PATH  # 复用函数的日志改写到本实验 progress.log,不污染 #31 档案

H_LIST = [10, 20, 25]
CONFIGS = ["ALL", "v6-1", "v6-2", "v6-3", "v6-4", "v6-5"]  # V1 版;sel 列映射见 SEL_COL
SEL_COL = {"ALL": "sel_ALL", "v6-1": "sel_S1", "v6-2": "sel_S2",
           "v6-3": "sel_S3", "v6-4": "sel_S4", "v6-5": "sel_S5"}
SAMPLE_SEED = 20260907
CHECK_B_STOCKS = 6
CHECK_B_NONEVENT_DAYS = 150
COV_LEVELS = [90, 85, 75, 50, 25, 10, 5]  # "X%信号涨幅>" 档位

logging.getLogger().setLevel(logging.WARNING)  # divergence_detector 模块 import 时把 root 调到 INFO,压回


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 共用:涨跌停按需加载(口径同 run_seeds 主流程)
def load_limits_for(ev_dates: pd.Series, codes: set):
    """返回 (limits_by_code, n_missing, n_needed, n_total)。口径逐行同 run_seeds.main 阶段 2。"""
    limit_files = sorted(p.stem for p in rs.LIMIT_DIR.glob("*.parquet"))
    lf_dates = pd.to_datetime(pd.Series(limit_files), format="%Y%m%d")
    ev_arr = ev_dates.sort_values().to_numpy()
    lo = lf_dates - pd.Timedelta(days=rs.LIMIT_LOOKAHEAD_DAYS)
    idx_hi = np.searchsorted(ev_arr, lf_dates.to_numpy(), side="right")
    idx_lo = np.searchsorted(ev_arr, lo.to_numpy(), side="left")
    needed_dates = [d for d, m in zip(limit_files, idx_hi > idx_lo) if m]
    limits_by_code, n_missing = rs.load_limits(needed_dates, codes)
    return limits_by_code, n_missing, len(needed_dates), len(limit_files)


def run_pool_sim(tasks, worker, tag: str, chunksize: int = 4):
    """mp.Pool 跑模拟,返回 (records list, fallback_loads, extra_sum)。worker 返回 (recs, fb, extra)。"""
    t0 = time.time()
    recs: list[dict] = []
    fb_total = 0
    extra_total = 0
    n_done = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 1),
                 initializer=rs._sim_worker_init) as pool:
        for r, fb, extra in pool.imap_unordered(worker, tasks, chunksize=chunksize):
            recs.extend(r)
            fb_total += fb
            extra_total += extra
            n_done += 1
            if n_done % 500 == 0:
                log(f"heartbeat: {tag} 模拟 {n_done}/{len(tasks)} 股,"
                    f"累计记录 {len(recs)} ({time.time() - t0:.0f}s)")
    log(f"[{tag}] 模拟完成:{len(recs)} 行 ({time.time() - t0:.0f}s)")
    return recs, fb_total, extra_total


# ---------------------------------------------------------------- 自检 a:模拟原语保真(#31 v6-1/H20 重跑)
def _check_a_worker(task: dict):
    """包装 rs.simulate_stock,补 extra 返回值位。"""
    recs, fb = rs.simulate_stock(task)
    return recs, fb, 0


def check_a() -> dict:
    t0 = time.time()
    saved_variants, saved_h = rs.VARIANTS, rs.H_LIST
    rs.VARIANTS = ["v1"]
    rs.H_LIST = [20]
    try:
        ev31 = pd.read_parquet(SEED_DIR / "events_history_v1.parquet")
        assert len(ev31) == 96577, len(ev31)
        ev_by_code: dict[str, dict] = {}
        for r in ev31.itertuples(index=False):
            ev_by_code.setdefault(r.ts_code, {}).setdefault("v1", []).append(
                dict(event_date=r.event_date, anchor_close=r.anchor_close))
        limits_by_code, n_missing, _, _ = load_limits_for(
            ev31["event_date"], set(ev31["ts_code"].unique()))
        tasks = [dict(ts_code=c, events=e, limits=limits_by_code.get(c))
                 for c, e in sorted(ev_by_code.items())]
        recs, fb, _ = run_pool_sim(tasks, _check_a_worker, "checkA")
    finally:
        rs.VARIANTS, rs.H_LIST = saved_variants, saved_h
    df = pd.DataFrame(recs)

    # 逐笔对拍 #31 trades_seed.parquet(v1/H20 子集)
    t31 = pd.read_parquet(SEED_DIR / "trades_seed.parquet")
    t31 = t31[(t31["variant"] == "v1") & (t31["H"] == 20)].reset_index(drop=True)
    mine = df.sort_values(["ts_code", "event_date"]).reset_index(drop=True)
    ref = t31.sort_values(["ts_code", "event_date"]).reset_index(drop=True)
    detail: dict = {"n_mine": int(len(mine)), "n_ref": int(len(ref))}
    ok = len(mine) == len(ref)
    if ok:
        same_key = bool(((mine["ts_code"] == ref["ts_code"])
                         & (mine["event_date"] == ref["event_date"])).all())
        same_status = bool((mine["status"] == ref["status"]).all())
        both_closed = (mine["status"] == "closed") & (ref["status"] == "closed")
        dd = np.abs(mine.loc[both_closed, "net_ret"].to_numpy()
                    - ref.loc[both_closed, "net_ret"].to_numpy())
        max_diff = float(dd.max()) if len(dd) else 0.0
        ok = same_key and same_status and max_diff == 0.0
        detail.update(same_key=same_key, same_status=same_status,
                      max_abs_net_ret_diff=max_diff,
                      n_closed=int(both_closed.sum()))
    # 净笔均复现 +4.89%(summary_seed.csv 六位小数)
    sel = (df["dd20"] <= -0.15) & (df["bounce"] > 0.02) & (df["bounce"] <= 0.08)
    cl = df[sel & (df["status"] == "closed")]
    mean = float(cl["net_ret"].mean())
    ref_csv = pd.read_csv(SEED_DIR / "summary_seed.csv")
    ref_mean = float(ref_csv[(ref_csv["variant"] == "v1") & (ref_csv["seed"] == "S1")
                             & (ref_csv["H"] == 20)]["net_mean"].iloc[0])
    detail.update(n_closed_s1=int(len(cl)), net_mean=mean, ref_net_mean=ref_mean,
                  abs_diff_vs_csv=abs(mean - ref_mean),
                  pct_2dp=round(mean * 100, 2))
    ok = ok and abs(mean - ref_mean) < 1e-6 and round(mean * 100, 2) == 4.89
    detail["ok"] = bool(ok)
    log(f"[checkA] 模拟原语保真:逐笔 max_abs_net_ret_diff={detail.get('max_abs_net_ret_diff')},"
        f"v6-1/H20 n_closed={len(cl)} 净笔均={mean:+.6f}(参照 {ref_mean:+.6f},"
        f"两位百分 {round(mean * 100, 2):+.2f}%) -> {'PASS' if ok else 'FAIL'} "
        f"({time.time() - t0:.0f}s)")
    return detail


# ---------------------------------------------------------------- V1 事件池加载与断言
def load_preprocessed(code: str) -> pd.DataFrame:
    """与 divergence_event_study.load_stock 逐行一致的预处理(dropna close → dedup date → sort)。"""
    df = pd.read_parquet(rs.DATA_DIR / f"{code}.parquet",
                         columns=["trade_date", "open", "close"])
    df = (df.dropna(subset=["close"]).drop_duplicates("trade_date")
          .sort_values("trade_date").reset_index(drop=True))
    return df


def load_v1_pool() -> pd.DataFrame:
    ev = pd.read_parquet(V1_EVENTS)
    meta = json.load(open(V1_STATS))["meta"]
    a1 = int(len(ev)) == 1033193 == int(meta["n_events"])
    a2 = not bool(ev.duplicated(["ts_code", "date"]).any())
    a3 = bool(ev["compare_rank"].isin([1, 2]).all())
    a4 = int(meta["val_mismatch"]) == 0
    log(f"[pool] events.parquet {len(ev)} 行,涉及 {ev['ts_code'].nunique()} 股,"
        f"事件日 {ev['date'].min().date()} ~ {ev['date'].max().date()};"
        f"断言: 行数一致={a1}, (ts_code,date)唯一={a2}, rank∈{{1,2}}={a3},"
        f" issue#2对拍val_mismatch=0={a4}")
    assert a1 and a2 and a3 and a4, "V1 事件池硬断言失败"
    return ev


# ---------------------------------------------------------------- 自检 b:V1 事件因果对拍
def check_b(ev: pd.DataFrame) -> dict:
    t0 = time.time()
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, REPO)  # divergence_detector -> comm_fun -> config.settings 需要仓库根
    from divergence_detector import DivergenceDetector  # noqa: E402

    det = DivergenceDetector()
    rng = np.random.default_rng(SAMPLE_SEED)
    cnt = ev.groupby("ts_code").size()
    cand = np.array(sorted(cnt[(cnt >= 100) & (cnt <= 400)].index))
    picks = sorted(rng.choice(cand, size=CHECK_B_STOCKS, replace=False).tolist())
    log(f"[checkB] 抽样 {len(picks)} 股(种子 {SAMPLE_SEED},候选事件数 100~400 的 {len(cand)} 股):"
        f" {picks}")
    total_mismatch = 0
    per_stock = []
    for code in picks:
        df = load_preprocessed(code)
        close = df["close"].to_numpy(np.float64)
        n = len(df)
        macd = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        sim = sorted((int(t), int(r), int(p)) for t, r, p in
                     des.simulate_events_idx(close, macd))
        sub = ev[ev["ts_code"] == code].sort_values("t_idx")
        ref = [(int(r.t_idx), int(r.compare_rank), int(r.prev_idx))
               for r in sub.itertuples(index=False)]
        sim_equal = sim == ref
        # 真实 V1 检测器逐日截断重算(全部事件日 + 抽样非事件日),日级布尔对拍
        idx = pd.DatetimeIndex(df["trade_date"])
        dfd = pd.DataFrame({"close": close, "macd": macd,
                            "volume": np.ones(n)}, index=idx)
        ev_days = {int(t) for t in sub["t_idx"]}
        non = np.setdiff1d(np.arange(des.MIN_LEN - 1, n),
                           np.fromiter(ev_days, int, len(ev_days)))
        sample = rng.choice(non, size=min(CHECK_B_NONEVENT_DAYS, len(non)),
                            replace=False)
        checks = sorted(ev_days) + sorted(int(t) for t in sample)
        mism = 0
        for t in checks:
            res = det.detect_daily_divergence(dfd.iloc[: t + 1], code,
                                              idx[t].date())
            if (len(res) > 0) != (t in ev_days):
                mism += 1
                log(f"  MISMATCH {code} t={t} date={idx[t].date()} "
                    f"detector={len(res) > 0} pool={t in ev_days}")
        total_mismatch += mism
        per_stock.append(dict(ts_code=code, n_rows=n, n_events=len(ref),
                              sim_equal=bool(sim_equal),
                              detector_days_checked=len(checks),
                              detector_mismatch=mism))
        log(f"  [checkB] {code}: 行数 {n},事件 {len(ref)},"
            f"因果模拟逐行一致={sim_equal},检测器对拍 {len(checks)} 日 mism={mism} "
            f"({time.time() - t0:.0f}s)")
    ok = total_mismatch == 0 and all(s["sim_equal"] for s in per_stock)
    detail = dict(ok=bool(ok), seed=SAMPLE_SEED, stocks=per_stock,
                  total_detector_mismatch=int(total_mismatch))
    log(f"[checkB] V1 事件因果对拍 -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
    return detail


# ---------------------------------------------------------------- V1 事件级模拟(复用 rs._simulate_one)
def simulate_v1_stock(task: dict):
    """单股:预处理与事件研究逐行一致;逐事件算 dd20/bounce(anchor=close[prev_idx]),逐 H 模拟。"""
    code = task["ts_code"]
    lim = task["limits"]
    df = load_preprocessed(code)
    open_ = df["open"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    dts = pd.DatetimeIndex(df["trade_date"])
    d_int = (dts.year * 10000 + dts.month * 100 + dts.day).to_numpy(np.int32)
    recs: list[dict] = []
    n_idx_mismatch = 0
    for t_idx, prev_idx, rank, ev_date in task["events"]:
        if dts[t_idx] != ev_date:
            n_idx_mismatch += 1
            continue
        close_ev = float(close[t_idx])
        anchor = float(close[prev_idx])
        dd20 = close_ev / float(close[max(0, t_idx - 20):t_idx + 1].max()) - 1.0
        bounce = close_ev / anchor - 1.0
        base = dict(ts_code=code, event_date=ev_date, t_idx=int(t_idx),
                    prev_idx=int(prev_idx), compare_rank=int(rank),
                    anchor_close=anchor, close_ev=close_ev,
                    dd20=float(dd20), bounce=float(bounce))
        for H in H_LIST:
            recs.append({**base, "H": H,
                         **rs._simulate_one(open_, close, dts, d_int, lim,
                                            code, int(t_idx), H)})
    fb = rs._G["fallback_loads"]
    rs._G["fallback_loads"] = 0
    return recs, fb, n_idx_mismatch


# ---------------------------------------------------------------- 统计(#31 §2.4 同口径)
def summary_rows(trades: pd.DataFrame, variant_name: str,
                 sel_map: dict, universe: int) -> tuple[pd.DataFrame, dict]:
    """逐配置 × H 出 #31 §2.4 全套指标。返回 (summary df, yearly dict)。"""
    rows: list[dict] = []
    yearly: dict[tuple, dict] = {}
    for cfg in sel_map:
        sel_col = sel_map[cfg]
        for H in H_LIST:
            g = trades[(trades["H"] == H) & trades[sel_col]]
            st = g["status"].value_counts().to_dict()
            cl = g[g["status"] == "closed"]
            ret = cl["net_ret"].to_numpy(dtype=float)
            n_closed = len(cl)
            ct = rs.cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) \
                if n_closed else np.nan
            yr_stats = {}
            if n_closed:
                yr = cl["entry_date"].dt.year
                grp = cl.groupby(yr)["net_ret"]
                ymean, ycount = grp.mean(), grp.count()
                yr_stats = {int(y): (float(ymean.loc[y]), int(ycount.loc[y]))
                            for y in ymean.index}
                qual = [y for y, (m, c) in yr_stats.items() if c >= 30]
                win_year_share = (float(np.mean([yr_stats[y][0] > 0 for y in qual]))
                                  if qual else np.nan)
            else:
                win_year_share = np.nan
            if n_closed:
                pnl_by_day = cl.groupby(cl["entry_date"].dt.date)["net_pnl"].sum()
                total_pnl = float(pnl_by_day.sum())
                top5 = float(pnl_by_day.sort_values(ascending=False).head(5).sum())
                date_conc = top5 / total_pnl if total_pnl > 0 else np.nan
            else:
                date_conc = np.nan
            rows.append(dict(
                variant=variant_name, seed=cfg, H=H,
                n_universe=universe, n_selected=int(len(g)), n_closed=n_closed,
                n_dropped_limitup=int(st.get("dropped_limitup", 0)),
                n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
                n_dropped_cash=int(st.get("dropped_cash", 0)),
                n_truncated=int(st.get("truncated_no_next", 0)
                                + st.get("truncated_exhausted", 0)),
                net_mean=float(ret.mean()) if n_closed else np.nan,
                net_median=float(np.median(ret)) if n_closed else np.nan,
                win_rate=float((ret > 0).mean()) if n_closed else np.nan,
                share_gt_1pct=float((ret > 0.01).mean()) if n_closed else np.nan,
                cluster_t=ct, win_year_share=win_year_share, date_conc=date_conc))
            yearly[(variant_name, cfg, H)] = yr_stats
    return pd.DataFrame(rows), yearly


def five_lines(row: pd.Series) -> dict:
    crit = dict(
        c1_n_closed_ge_300=bool(row["n_closed"] >= 300),
        c2_net_mean_pos=bool(np.isfinite(row["net_mean"]) and row["net_mean"] > 0),
        c3_cluster_t_ge_2=bool(np.isfinite(row["cluster_t"]) and row["cluster_t"] >= 2),
        c4_win_year_ge_60=bool(np.isfinite(row["win_year_share"])
                               and row["win_year_share"] >= 0.6),
        c5_date_conc_le_50=bool(np.isfinite(row["date_conc"])
                                and row["date_conc"] <= 0.5))
    return dict(checks=crit, passed=bool(all(crit.values())))


def dist_row(ret: np.ndarray) -> dict:
    """覆盖率→门槛:'X%信号涨幅>' = 涨幅最高的 X% 信号都超过该值(= 净收益第 (100-X) 百分位)。"""
    out = {}
    for cov in COV_LEVELS:
        out[f"cov{cov}"] = float(np.percentile(ret, 100 - cov)) if len(ret) else np.nan
    out["mean"] = float(ret.mean()) if len(ret) else np.nan
    out["best"] = float(ret.max()) if len(ret) else np.nan
    out["worst"] = float(ret.min()) if len(ret) else np.nan
    return out


# ---------------------------------------------------------------- 信号时间分布(事件级,#31 §4.4/§4.7 维度1/6b 同口径)
def temporal_stats(frame: pd.DataFrame, sel_map: dict, cal: pd.DatetimeIndex) -> dict:
    """frame:每事件一行(H=20 切片)。返回 {cfg: dict}。

    月跨度口径(#31 §4.4 反推核对一致):跨度 = 种子并集(sel_map 中除 ALL 外的布尔或)
    首末事件月(含两端),全表共享;月均 = 总数/跨度;零信号月占比 = (跨度 − 该行有信号月数)/跨度。
    v6 侧复算:跨度 398(1993-07~2026-08),与 #31 §4.4 全表逐格一致(自检 d)。"""
    cal_pos = {d: i for i, d in enumerate(cal)}
    pos = frame["event_date"].map(cal_pos)
    in_cal = pos.notna().to_numpy()
    # 种子并集首末事件月
    seed_cols = [c for c in sel_map.values() if c != "sel_ALL"]
    union = np.logical_or.reduce([frame[c].to_numpy() for c in seed_cols]) \
        if seed_cols else frame["sel_ALL"].to_numpy()
    u_dates = frame.loc[union, "event_date"]
    if len(u_dates):
        u_months = u_dates.dt.to_period("M")
        first_m, last_m = u_months.min(), u_months.max()
    else:  # 并集为空(防御):退化为全池首末月
        all_months = frame["event_date"].dt.to_period("M")
        first_m, last_m = all_months.min(), all_months.max()
    months_span = (last_m.year - first_m.year) * 12 + (last_m.month - first_m.month) + 1
    out: dict = {}
    for cfg, sel_col in sel_map.items():
        sel = frame[sel_col].to_numpy()
        sub_dates = frame.loc[sel, "event_date"]
        n_total = int(len(sub_dates))
        if n_total == 0:
            out[cfg] = dict(n_total=0)
            continue
        per_month = sub_dates.dt.to_period("M")
        mc = per_month.value_counts()
        dc = sub_dates.value_counts()  # 逐日信号数(全历史)
        # 逐日维度(#31 §4.7 维度1:零信号日占比仅按落入日历的有信号日计)
        sel_dates = sub_dates.to_numpy()
        n_days_with = int(dc.size)
        n_days_in_cal = int(pd.Index(sel_dates[in_cal[sel]]).nunique()) \
            if in_cal[sel].any() else 0
        zero_day_share = 1.0 - n_days_in_cal / len(cal)
        day_counts = dc.to_numpy(dtype=float)
        # 逐年
        yr_sig = sub_dates.dt.year.value_counts()
        in_cal_dates = pd.Index(sel_dates[in_cal[sel]]) if in_cal[sel].any() \
            else pd.Index([], dtype="datetime64[ns]")
        yr_days = pd.Series(in_cal_dates).groupby(in_cal_dates.year).nunique() \
            if len(in_cal_dates) else pd.Series(dtype=int)
        out[cfg] = dict(
            n_total=n_total,
            per_year_avg=n_total / 35.0,
            per_month_avg=n_total / months_span,
            zero_month_share=1.0 - float(len(mc)) / months_span,
            max_month=int(mc.max()), max_day=int(dc.max()),
            n_days_with=n_days_with,
            n_days_pre_cal=n_days_with - n_days_in_cal,
            zero_day_share=zero_day_share,
            day_med=float(np.percentile(day_counts, 50)),
            day_p95=float(np.percentile(day_counts, 95)),
            day_max=int(day_counts.max()),
            yearly_signals={int(y): int(c) for y, c in yr_sig.items()},
            yearly_days={int(y): int(d) for y, d in yr_days.items()},
        )
    return out


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("V1 SEED RECOMPARE START | 预登记=README.md(冻结) | "
        "实测全程约 3.5 分钟(自检a ~15s,自检b ~2.5min,涨涨停 ~10s×2,"
        "V1 模拟 3.1M 笔 ~25s,统计+报告 ~15s;27 进程)")

    # ---------------- 自检 a:模拟原语保真 ----------------
    check_a_detail = check_a()
    if not check_a_detail["ok"]:
        log("FATAL: 自检 a 失败,终止(不过则失败)")
        _dump_abort(dict(check_a=check_a_detail))
        sys.exit(1)

    # ---------------- V1 事件池加载与硬断言 ----------------
    ev = load_v1_pool()
    n_pool = len(ev)

    # ---------------- 自检 b:V1 事件因果对拍 ----------------
    check_b_detail = check_b(ev)
    if not check_b_detail["ok"]:
        log("FATAL: 自检 b 失败,终止(不过则失败)")
        _dump_abort(dict(check_a=check_a_detail, check_b=check_b_detail))
        sys.exit(1)

    # ---------------- 涨跌停按需加载(V1 全池) ----------------
    limits_by_code, n_limit_missing, n_limit_needed, n_limit_total = load_limits_for(
        ev["date"], set(ev["ts_code"].unique()))

    # ---------------- V1 事件级模拟 ----------------
    ev_sorted = ev.sort_values(["ts_code", "t_idx"])
    tasks = []
    for code, grp in ev_sorted.groupby("ts_code", sort=True):
        events = [(int(r.t_idx), int(r.prev_idx), int(r.compare_rank), r.date)
                  for r in grp.itertuples(index=False)]
        tasks.append(dict(ts_code=code, events=events,
                          limits=limits_by_code.get(code)))
    log(f"[simulate] 待模拟个股 {len(tasks)} 只,事件 {n_pool} 起 × H{H_LIST}")
    recs, fb_total, idx_mismatch = run_pool_sim(tasks, simulate_v1_stock, "V1")
    log(f"[simulate] t_idx 日期逐事件断言违例 {idx_mismatch} 起(须为 0)")
    trades = pd.DataFrame(recs)
    trades = trades.sort_values(["ts_code", "event_date", "H"]).reset_index(drop=True)

    # 种子选择布尔列(#31 §2.2 逐字)
    dd, bo = trades["dd20"], trades["bounce"]
    trades["sel_S1"] = (dd <= -0.15) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S2"] = (dd <= -0.20) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S3"] = (dd <= -0.25) & (bo > 0.02) & (bo <= 0.08)
    trades["sel_S4"] = dd <= -0.15
    trades["sel_S5"] = dd <= -0.25
    trades["sel_ALL"] = True
    trades.to_parquet(OUT_DIR / "trades_v1.parquet", index=False)
    log(f"[dump] trades_v1.parquet {len(trades)} 行;"
        f" bounce 全池范围 [{trades['bounce'].min():+.6f}, {trades['bounce'].max():+.6f}]"
        f"(README 披露 1:V1 池 bounce 恒 < 0 -> max 须 < 0)")
    vc = trades.groupby("H")["status"].value_counts()
    log(f"[simulate] 状态分布:\n{vc.to_string()}")

    # ---------------- 统计:V1 18 行 ----------------
    summary, yearly = summary_rows(trades, "V1", SEL_COL, n_pool)
    summary.to_csv(OUT_DIR / "summary_v1.csv", index=False, float_format="%.6f")
    log(f"[dump] summary_v1.csv {len(summary)} 行")

    # 收益分布 + >+1% 占比(closed 口径)
    dist_rows: list[dict] = []
    for cfg, sel_col in SEL_COL.items():
        for H in H_LIST:
            cl = trades[(trades["H"] == H) & trades[sel_col]
                        & (trades["status"] == "closed")]
            ret = cl["net_ret"].to_numpy(dtype=float)
            dist_rows.append(dict(variant="V1", seed=cfg, H=H, n=len(ret),
                                  **dist_row(ret)))
    dist_v1 = pd.DataFrame(dist_rows)

    # 信号时间分布(事件级,H=20 切片)
    cal = pd.DatetimeIndex(pd.read_parquet(CAL_PATH, columns=["trade_date"])
                           ["trade_date"]).sort_values()
    assert len(cal) == 8000, len(cal)
    frame = trades[trades["H"] == 20]
    temporal_v1 = temporal_stats(frame, SEL_COL, cal)
    log("[stats] V1 分布/时间分布完成")

    # ---------------- v6 引用数(#31 落盘,不重跑) ----------------
    sum31 = pd.read_csv(SEED_DIR / "summary_seed.csv")
    sum31_v1 = sum31[sum31["variant"] == "v1"].reset_index(drop=True)
    t31 = pd.read_parquet(SEED_DIR / "trades_seed.parquet")
    t31f = t31[t31["variant"] == "v1"]
    sel_map_v6 = SEL_COL  # 列名相同
    dist31_rows: list[dict] = []
    for cfg, sel_col in sel_map_v6.items():
        for H in H_LIST:
            cl = t31f[(t31f["H"] == H) & t31f[sel_col]
                      & (t31f["status"] == "closed")]
            ret = cl["net_ret"].to_numpy(dtype=float)
            dist31_rows.append(dict(variant="v6", seed=cfg, H=H, n=len(ret),
                                    **dist_row(ret)))
    dist_v6 = pd.DataFrame(dist31_rows)
    # >+1% 占比并入 v6 引用表(seed 码 S1..S5/ALL)
    seed31_code = {"v6-1": "S1", "v6-2": "S2", "v6-3": "S3",
                   "v6-4": "S4", "v6-5": "S5", "ALL": "ALL"}
    share_rows = []
    for cfg, sel_col in sel_map_v6.items():
        for H in H_LIST:
            cl = t31f[(t31f["H"] == H) & t31f[sel_col]
                      & (t31f["status"] == "closed")]
            ret = cl["net_ret"].to_numpy(dtype=float)
            share_rows.append(dict(seed=seed31_code[cfg], H=H,
                                   share_gt_1pct=float((ret > 0.01).mean())
                                   if len(ret) else np.nan))
    sum31_v1 = sum31_v1.merge(pd.DataFrame(share_rows), on=["seed", "H"],
                              how="left")
    frame31 = t31f[t31f["H"] == 20]
    temporal_v6 = temporal_stats(frame31, sel_map_v6, cal)
    log("[stats] v6 引用数重算完成(源:#31 落盘 parquet/csv)")

    # ---------------- 自检 d:v6 引用数与 #31 已发表数字核对 ----------------
    check_d = check_d_against_31(dist_v6, temporal_v6, sum31_v1)
    log(f"[checkD] v6 引用数核对:{check_d['n_checked']} 格,"
        f"失配 {check_d['n_mismatch']} -> {'PASS' if check_d['ok'] else 'FAIL'}")

    # ---------------- 自检 c:计数/因果/守恒 ----------------
    closed = trades[trades["status"] == "closed"]
    n_viol_entry = int((closed["entry_date"] <= closed["event_date"]).sum())
    n_viol_exit = int((closed["exit_date"] <= closed["entry_date"]).sum())
    c3a = bool((summary["n_selected"] <= summary["n_universe"]).all())
    all_rows = summary[summary["seed"] == "ALL"]
    c3b = bool((all_rows["n_selected"] == all_rows["n_universe"]).all())
    cons = (summary["n_closed"] + summary["n_dropped_limitup"]
            + summary["n_dropped_no_quote"] + summary["n_dropped_cash"]
            + summary["n_truncated"])
    c3c = bool((cons == summary["n_selected"]).all())
    c3d = bool(len(trades) == 3 * n_pool)
    c3e = idx_mismatch == 0
    check_c = dict(ok=bool(n_viol_entry == 0 and n_viol_exit == 0
                           and c3a and c3b and c3c and c3d and c3e),
                   n_entry_viol=n_viol_entry, n_exit_viol=n_viol_exit,
                   sel_le_uni=c3a, all_eq_uni=c3b, sum_eq=c3c,
                   rows_eq_3x_pool=c3d, idx_assert_zero=c3e)
    log(f"[checkC] 因果/守恒:entry<=event {n_viol_entry},exit<=entry {n_viol_exit},"
        f"sel<=uni={c3a},ALL=uni={c3b},分量加总={c3c},行数=3×池={c3d},"
        f"索引断言0违例={c3e} -> {'PASS' if check_c['ok'] else 'FAIL'}")

    # ---------------- 参照五线(不设硬线,只出数) ----------------
    verdicts: dict = {}
    for cfg in CONFIGS:
        per_H = {}
        for H in H_LIST:
            row = summary[(summary["seed"] == cfg) & (summary["H"] == H)].iloc[0]
            fl = five_lines(row)
            per_H[str(H)] = dict(
                n_closed=int(row["n_closed"]),
                net_mean=float(row["net_mean"]) if np.isfinite(row["net_mean"]) else None,
                cluster_t=float(row["cluster_t"]) if np.isfinite(row["cluster_t"]) else None,
                win_year_share=(float(row["win_year_share"])
                                if np.isfinite(row["win_year_share"]) else None),
                date_conc=float(row["date_conc"]) if np.isfinite(row["date_conc"]) else None,
                **fl)
        verdicts[cfg] = dict(per_H=per_H,
                             any_H_pass=any(v["passed"] for v in per_H.values()))
        log(f"[五线参照] V1 {cfg}: "
            + " ".join(f"H{h}:{'过' if v['passed'] else '否'}"
                       for h, v in per_H.items()))

    checks_all = bool(check_a_detail["ok"] and check_b_detail["ok"]
                      and check_c["ok"] and check_d["ok"])

    # ---------------- verdict.json ----------------
    vout = dict(
        experiment="V1(最早背离设计)按 #31 口径重跑统计(issue #34)",
        prereadme="README.md 先于任何跑数落盘(冻结)",
        event_pool=dict(source=str(V1_EVENTS), n_events=n_pool,
                        semantics="issue #2 逐日截断因果模拟产物,val_mismatch=0"),
        checks=dict(check_a_sim_fidelity=check_a_detail,
                    check_b_causality=check_b_detail,
                    check_c_conservation=check_c,
                    check_d_v6_reference=check_d),
        checks_all_pass=checks_all,
        five_line_reference=dict(criteria="n>=300 / 净笔均>0 / cluster_t>=2 / "
                                 "盈利年>=60% / 集中度<=50%(同一 H 五线全满;本票不设硬线)",
                                 per_config=verdicts),
        disclosure=[
            "V1 事件日即低点日 -> bounce 恒 < 0 -> V1 版 v6-1/2/3 结构性 0 选中(README 披露 1)",
            "事件池复用 issue #2 产物(逐日截断因果,抽样对拍见自检 b)",
            "退市股事件后数据不足记 truncated(按没买成处理,结论偏乐观方向)",
            "stk_limit 2007-01-04 起,缺文件日=无涨跌停约束",
            "事件索引基于事件研究预处理序列(dropna close/dedup date/sort),逐事件日期断言 0 违例",
        ],
        n_limit_missing_days=int(n_limit_missing),
        n_limit_files_needed=int(n_limit_needed),
        n_limit_files_total=int(n_limit_total),
        n_fallback_loads=int(fb_total),
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(vout, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict.json")

    # ---------------- report.md ----------------
    write_report(summary, dist_v1, temporal_v1, sum31_v1, dist_v6, temporal_v6,
                 verdicts, vout["checks"], checks_all, cal)
    log("[dump] report.md")
    log(f"V1 SEED RECOMPARE DONE ({time.time() - t_all:.0f}s) "
        f"自检总评={'ALL PASS' if checks_all else 'HAS FAIL'}")

    pd.set_option("display.width", 300)
    print("\n===== V1 SUMMARY(18 行全出数) =====")
    print(summary.to_string(index=False))


def _dump_abort(checks: dict) -> None:
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(dict(aborted=True, checks=checks,
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------- 自检 d:v6 引用数核对(#31 已发表数字)
def check_d_against_31(dist_v6: pd.DataFrame, temporal_v6: dict,
                       sum31_v1: pd.DataFrame) -> dict:
    """从 #31 落盘重算的 v6 数字 vs issue #31 已发表表格(§4.2/§4.3/§4.4/§4.7 维度1),逐格核对。"""
    mismatches: list[str] = []
    n_checked = 0

    def ck(name: str, got: float, want: float, tol: float):
        nonlocal n_checked
        n_checked += 1
        if not (np.isfinite(got) and abs(got - want) <= tol):
            mismatches.append(f"{name}: got {got}, want {want}(tol {tol})")

    # §4.2 关键行(H=20,净笔均 2 位百分容差 0.005pp;来自 summary CSV 直接核对六位值)
    pub_mean = {"v6-1": 0.0489, "v6-2": 0.0600, "v6-3": 0.0653,
                "v6-4": 0.0568, "v6-5": 0.0964, "ALL": 0.0158}
    for seed, want in pub_mean.items():
        got = float(sum31_v1[(sum31_v1["seed"] == {"v6-1": "S1", "v6-2": "S2",
                                                  "v6-3": "S3", "v6-4": "S4",
                                                  "v6-5": "S5", "ALL": "ALL"}[seed])
                             & (sum31_v1["H"] == 20)]["net_mean"].iloc[0])
        ck(f"§4.2 {seed}/H20 净笔均", got, want, 0.00005)
    # §4.3 H=20 分布七档(1 位百分容差 0.05%)
    pub_dist = {
        "v6-1": [-15.9, -12.4, -6.6, 4.3, 15.1, 26.5, 34.5],
        "v6-2": [-17.1, -13.2, -7.4, 4.8, 17.4, 30.6, 38.3],
        "v6-3": [-17.7, -13.2, -8.0, 5.0, 18.6, 32.3, 41.3],
        "v6-4": [-17.1, -12.6, -6.2, 4.9, 16.3, 28.5, 36.8],
        "v6-5": [-16.1, -11.0, -3.9, 9.1, 21.8, 34.8, 43.3],
        "ALL": [-14.4, -11.2, -6.9, 0.3, 8.3, 18.4, 26.7],
    }
    for seed, wants in pub_dist.items():
        row = dist_v6[(dist_v6["seed"] == seed) & (dist_v6["H"] == 20)].iloc[0]
        for cov, want in zip(COV_LEVELS, wants):
            ck(f"§4.3 {seed}/H20 cov{cov}", row[f"cov{cov}"] * 100, want, 0.05)
    # §4.4 汇总行(ALL)
    t_all = temporal_v6["ALL"]
    ck("§4.4 ALL 总数", t_all["n_total"], 96577, 0)
    ck("§4.4 ALL 月均", t_all["per_month_avg"], 242.7, 0.05)
    ck("§4.4 ALL 零信号月占比", t_all["zero_month_share"] * 100, 1.5, 0.05)
    ck("§4.4 ALL 单月最多", t_all["max_month"], 2158, 0)
    ck("§4.4 ALL 单日最多", t_all["max_day"], 488, 0)
    ck("§4.4 v6-5 总数", temporal_v6["v6-5"]["n_total"], 3271, 0)
    ck("§4.4 v6-5 单月最多", temporal_v6["v6-5"]["max_month"], 985, 0)
    # §4.7 维度1(v6-4)
    t4 = temporal_v6["v6-4"]
    ck("维度1 v6-4 有信号日数", t4["n_days_with"], 2234, 0)
    ck("维度1 v6-4 零信号日占比", t4["zero_day_share"] * 100, 72.11, 0.005)
    ck("维度1 v6-4 50%日信号数>", t4["day_med"], 2, 0)
    ck("维度1 v6-4 5%日>", t4["day_p95"], 19, 0.05)
    ck("维度1 v6-4 max", t4["day_max"], 223, 0)
    ck("维度1 v6-3 5%日>", temporal_v6["v6-3"]["day_p95"], 13.4, 0.05)
    # §4.3 净收益>+1% 占比(1 位百分容差 0.05pp;sum31_v1 已并入 share_gt_1pct)
    seed31_code = {"v6-1": "S1", "v6-5": "S5", "ALL": "ALL"}
    pub_share = {("v6-1", 10): 51.1, ("v6-1", 20): 58.1, ("v6-1", 25): 56.6,
                 ("v6-5", 10): 51.7, ("v6-5", 20): 67.4, ("v6-5", 25): 62.7,
                 ("ALL", 10): 43.2, ("ALL", 20): 47.4, ("ALL", 25): 47.4}
    for (seed, H), want in pub_share.items():
        got = float(sum31_v1[(sum31_v1["seed"] == seed31_code[seed])
                             & (sum31_v1["H"] == H)]["share_gt_1pct"].iloc[0]) * 100
        ck(f"§4.3 {seed}/H{H} 净>+1%占比", got, want, 0.05)
    return dict(ok=len(mismatches) == 0, n_checked=n_checked,
                n_mismatch=len(mismatches), mismatches=mismatches)


# ---------------------------------------------------------------- report.md
def fp(x, nd=2, sign=True):
    """小数收益 → '+4.89%' 字符串;NaN → '—'。"""
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%"


def fn(x, nd=3):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def write_report(summary, dist_v1, temporal_v1, sum31_v1, dist_v6, temporal_v6,
                 verdicts, checks, checks_all, cal) -> None:
    L: list[str] = []
    seed31 = {"v6-1": "S1", "v6-2": "S2", "v6-3": "S3", "v6-4": "S4",
              "v6-5": "S5", "ALL": "ALL"}

    def row31(seed: str, H: int) -> pd.Series:
        return sum31_v1[(sum31_v1["seed"] == seed31[seed])
                        & (sum31_v1["H"] == H)].iloc[0]

    def row_v1(cfg: str, H: int) -> pd.Series:
        return summary[(summary["seed"] == cfg) & (summary["H"] == H)].iloc[0]

    def drow(df: pd.DataFrame, seed: str, H: int) -> pd.Series:
        return df[(df["seed"] == seed) & (df["H"] == H)].iloc[0]

    L.append("# V1(最早背离设计)按 #31 口径重跑统计报告(issue #34,预登记一发,2026-09-07)")
    L.append("")
    L.append("> 口径:README.md(冻结);事件池 = issue #2 产物 events.parquet(逐日截断因果,1,033,193 事件)。")
    L.append("> 交易机制/统计指标逐字 #31 §2.3/§2.4;v6 数字引用 #31 落盘结果,不重跑。")
    L.append("> 披露:V1 事件日即低点日 → bounce 恒 < 0 → V1 版 v6-1/2/3 结构性 0 选中(原样出数)。")
    L.append("> 本票不设硬线(#34:由用户看表拍板),五线判活仅作参照出数。")
    L.append("")
    L.append(f"**自检总评:{'ALL PASS' if checks_all else 'HAS FAIL(如实交付)'}**")
    L.append("")

    # ---- 1. 自检
    L.append("## 1. 自检结果(README 第 7 节,全过才可出报告)")
    L.append("")
    ca = checks["check_a_sim_fidelity"]
    L.append(f"- **自检 a(模拟原语保真)**:#31 v1/H20 全格重跑,逐笔 net_ret max abs diff = "
             f"{ca.get('max_abs_net_ret_diff')}(对拍覆盖 closed {ca.get('n_closed')} 笔);"
             f"v6-1 子集 n_closed={ca.get('n_closed_s1')},"
             f"净笔均 {ca.get('net_mean'):+.6f} vs #31 CSV {ca.get('ref_net_mean'):+.6f}"
             f"(两位百分 {ca.get('pct_2dp'):+.2f}% = +4.89%) -> "
             f"{'PASS' if ca['ok'] else 'FAIL'}")
    cb = checks["check_b_causality"]
    L.append(f"- **自检 b(V1 事件因果对拍)**:抽样 {len(cb['stocks'])} 股(种子 {cb['seed']}),"
             f"因果模拟逐行一致 + 真实检测器逐日对拍总失配 {cb['total_detector_mismatch']} -> "
             f"{'PASS' if cb['ok'] else 'FAIL'}")
    for s in cb["stocks"]:
        L.append(f"  - {s['ts_code']}:行数 {s['n_rows']},事件 {s['n_events']},"
                 f"模拟逐行一致={s['sim_equal']},检测器对拍 {s['detector_days_checked']} 日 "
                 f"mism={s['detector_mismatch']}")
    cc = checks["check_c_conservation"]
    L.append(f"- **自检 c(因果/守恒)**:entry<=event {cc['n_entry_viol']} 笔,"
             f"exit<=entry {cc['n_exit_viol']} 笔;n_selected<=n_universe={cc['sel_le_uni']},"
             f"ALL 行相等={cc['all_eq_uni']},分量加总={cc['sum_eq']},"
             f"trades 行数=3×事件池={cc['rows_eq_3x_pool']},"
             f"事件索引断言 0 违例={cc['idx_assert_zero']} -> "
             f"{'PASS' if cc['ok'] else 'FAIL'}")
    cd = checks["check_d_v6_reference"]
    L.append(f"- **自检 d(v6 引用数核对)**:从 #31 落盘重算 {cd['n_checked']} 格"
             f"(§4.2 净笔均/§4.3 分布七档/§4.4 汇总/§4.7 维度1),失配 {cd['n_mismatch']} -> "
             f"{'PASS' if cd['ok'] else 'FAIL'}")
    if cd["mismatches"]:
        for m in cd["mismatches"]:
            L.append(f"  - 失配:{m}")
    L.append("")

    # ---- 2. V1 18 行全表
    L.append("## 2. V1 全表(6 配置 × 3 档 H = 18 行,配置为行,全出数;净笔均/净中位/胜率为小数)")
    L.append("")
    L.append("| 配置 | H | n_universe | n_selected | n_closed | drop涨停 | drop无报价 | drop现金 | n_truncated | 净笔均 | 净中位 | 胜率 | 净>+1%占比 | cluster_t | 盈利年占比 | 日期集中度 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.itertuples(index=False):
        L.append(f"| V1-{r.seed} | {r.H} | {r.n_universe} | {r.n_selected} | {r.n_closed} | "
                 f"{r.n_dropped_limitup} | {r.n_dropped_no_quote} | {r.n_dropped_cash} | "
                 f"{r.n_truncated} | {fp(r.net_mean, 4)} | {fp(r.net_median, 4)} | "
                 f"{fn(r.win_rate)} | {fn(r.share_gt_1pct)} | {fn(r.cluster_t)} | "
                 f"{fn(r.win_year_share)} | {fn(r.date_conc)} |")
    L.append("")
    L.append("注:V1 版 v6-1/2/3 因 bounce 恒 < 0 结构性 0 选中(README 披露 1),相关指标为 —。")
    L.append("")

    # ---- 3. 参照五线
    L.append("## 3. 参照五线(#31 同款判活线,仅出数不设硬线;五线编码 1=n≥300 2=净笔均>0 3=cluster_t≥2 4=盈利年≥60% 5=集中度≤50%)")
    L.append("")
    L.append("| 配置 | H | n_closed | 净笔均 | cluster_t | 盈利年占比 | 日期集中度 | 五线(12345) | 过线 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        for h, pv in verdicts[cfg]["per_H"].items():
            crit = "".join("1" if k else "0" for k in pv["checks"].values())
            nm = f"{pv['net_mean'] * 100:+.2f}%" if pv["net_mean"] is not None else "—"
            L.append(f"| V1-{cfg} | {h} | {pv['n_closed']} | {nm} | "
                     f"{fn(pv['cluster_t'])} | {fn(pv['win_year_share'])} | "
                     f"{fn(pv['date_conc'])} | {crit} | {'过' if pv['passed'] else '否'} |")
    L.append("")

    # ---- 4. 并表对比(H=20 主表)
    L.append("## 4. 并表对比(H=20 主表;v6 行 = #31 落盘引用,自检 d 已核对)")
    L.append("")
    L.append("**池价值(全池行)与池内筛选增益(种子行 − 全池行)分开表述(纪律)。**")
    L.append("")
    L.append("| 池 | 配置 | n_closed | 净笔均 | 净中位 | 胜率 | 净>+1%占比 | cluster_t | 盈利年占比 | 日期集中度 | 净笔均−本池ALL |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for pool, getter, allseed in (("V1", row_v1, "ALL"), ("v6", row31, "ALL")):
        base_mean = getter(allseed, 20)["net_mean"]
        for cfg in ["ALL", "v6-1", "v6-2", "v6-3", "v6-4", "v6-5"]:
            r = getter(cfg, 20)
            nm = r["net_mean"]
            delta = nm - base_mean if np.isfinite(nm) and np.isfinite(base_mean) else np.nan
            L.append(f"| {pool} | {cfg} | {int(r['n_closed'])} | {fp(nm)} | "
                     f"{fp(r['net_median'])} | {fn(r['win_rate'])} | "
                     f"{fn(r['share_gt_1pct'])} | {fn(r['cluster_t'])} | "
                     f"{fn(r['win_year_share'])} | {fn(r['date_conc'])} | "
                     f"{fp(delta, 2)} |")
    L.append("")
    L.append("读法:V1 全池行 = V1 裸信号池价值;V1 版 v6-4/v6-5 行 = V1 池内急跌筛选增益;"
             "v6 行原样引用 #31(其池 = 背离v6 金叉对金叉信号池)。")
    L.append("")

    # ---- 4b. 全 H 并表
    L.append("### 4b. 全 H 并表(净笔均,扣完成本,closed 口径)")
    L.append("")
    L.append("| 池 | 配置 | H10 净笔均(n) | H20 净笔均(n) | H25 净笔均(n) |")
    L.append("|---|---|---|---|---|")
    for pool, getter in (("V1", row_v1), ("v6", row31)):
        for cfg in CONFIGS:
            cells = []
            for H in H_LIST:
                r = getter(cfg, H)
                nm = r["net_mean"]
                cells.append(f"{fp(nm)}({int(r['n_closed'])})"
                             if np.isfinite(nm) else f"—({int(r['n_closed'])})")
            L.append(f"| {pool} | {cfg} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # ---- 5. 收益分布
    L.append("## 5. 收益分布(净,closed 口径;'X%信号涨幅>' = 涨幅最高的 X% 信号都超过该值)")
    L.append("")
    for H in H_LIST:
        L.append(f"### H={H}")
        L.append("")
        L.append("| 池 | 配置 | n | 均值 | 90%信号涨幅> | 85%信号涨幅> | 75%信号涨幅> | 50%信号涨幅> | 25%信号涨幅> | 10%信号涨幅> | 5%信号涨幅> | 最好 | 最差 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pool, df in (("V1", dist_v1), ("v6", dist_v6)):
            for cfg in CONFIGS:
                r = drow(df, cfg, H)
                if r["n"] == 0:
                    L.append(f"| {pool} | {cfg} | 0 | — | — | — | — | — | — | — | — | — | — |")
                    continue
                L.append(f"| {pool} | {cfg} | {int(r['n'])} | {fp(r['mean'])} | "
                         + " | ".join(fp(r[f"cov{c}"], 1) for c in COV_LEVELS)
                         + f" | {fp(r['best'], 1)} | {fp(r['worst'], 1)} |")
        L.append("")
    L.append("v6 行与 #31 §4.3 已发表数字一致(自检 d 逐格核对,容差 = 发表值一位百分舍入 ±0.05pp)。")
    L.append("")

    # ---- 6. 净收益>+1% 占比
    L.append("## 6. 净收益 > +1% 的信号占比(closed 口径,扣完成本,分母 = n_closed)")
    L.append("")
    L.append("| 池 | 配置 | H=10 | H=20 | H=25 |")
    L.append("|---|---|---|---|---|")
    for pool, getter in (("V1", row_v1), ("v6", row31)):
        for cfg in CONFIGS:
            cells = []
            for H in H_LIST:
                v = getter(cfg, H)["share_gt_1pct"]
                cells.append(fn(v * 100, 1) + "%" if np.isfinite(v) else "—")
            L.append(f"| {pool} | {cfg} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # ---- 7. 信号时间分布
    L.append("## 7. 信号时间分布(事件级,与 H 无关;市场日历 = 上证指数日线 8000 交易日,1993-10-08 ~ 2026-08-31)")
    L.append("")
    L.append("### 7.1 汇总(#31 §4.4 同口径:年均=总数/35;月均=总数/种子并集首末事件月跨度月数(含两端,全表共享);零信号月占比=(跨度−有信号月数)/跨度)")
    L.append("")
    L.append("| 池 | 配置 | 35年总数 | 年均 | 月均 | 零信号月占比 | 单月最多 | 单日最多 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pool, tab in (("V1", temporal_v1), ("v6", temporal_v6)):
        for cfg in CONFIGS:
            t = tab[cfg]
            if t.get("n_total", 0) == 0:
                L.append(f"| {pool} | {cfg} | 0 | — | — | — | — | — |")
                continue
            L.append(f"| {pool} | {cfg} | {t['n_total']} | {t['per_year_avg']:.0f} | "
                     f"{t['per_month_avg']:.1f} | {t['zero_month_share'] * 100:.1f}% | "
                     f"{t['max_month']} | {t['max_day']} |")
    L.append("")
    L.append("### 7.2 逐日信号数(#31 §4.7 维度1 同口径;零信号日占比分母 = 日历 8000 交易日,仅落入日历的有信号日计入;'X%日>' = 有信号日中日信号数超过该值的天数恰占 X%)")
    L.append("")
    L.append("| 池 | 配置 | 事件总数 | 有信号日数 | 零信号日占比 | 50%日信号数> | 5%日> | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pool, tab in (("V1", temporal_v1), ("v6", temporal_v6)):
        for cfg in CONFIGS:
            t = tab[cfg]
            if t.get("n_total", 0) == 0:
                L.append(f"| {pool} | {cfg} | 0 | 0 | — | — | — | — |")
                continue
            L.append(f"| {pool} | {cfg} | {t['n_total']} | {t['n_days_with']} | "
                     f"{t['zero_day_share'] * 100:.2f}% | {t['day_med']:g} | "
                     f"{t['day_p95']:g} | {t['day_max']} |")
    L.append("")
    pre_cal = {cfg: temporal_v1[cfg].get("n_days_pre_cal", 0) for cfg in CONFIGS}
    L.append(f"V1 池日历起点前有信号日数(计数披露,不入零信号日占比):{pre_cal}")
    L.append("")

    # 7.3 逐年
    L.append("### 7.3 逐年信号数与有信号日数(V1;有信号日数与'占交易日%'只计落入市场日历的日,日历起点前的信号日计数见 7.2 注;1992/1993/2026 为不完整日历年)")
    L.append("")
    years = list(range(1992, 2027))
    cal_years = pd.Series(cal.year).value_counts()
    L.append("| 年份 | 交易日数 | 全池信号数 | v6-4信号数 | v6-5信号数 | 全池有信号日 | 全池占交易日% | v6-4有信号日 | v6-4占交易日% | v6-5有信号日 | v6-5占交易日% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for y in years:
        ndays = int(cal_years.get(y, 0))
        row = [str(y), str(ndays) if ndays else "—"]
        for cfg in ("ALL", "v6-4", "v6-5"):
            t = temporal_v1[cfg]
            row.append(str(t.get("yearly_signals", {}).get(y, 0)))
        for cfg in ("ALL", "v6-4", "v6-5"):
            t = temporal_v1[cfg]
            d = t.get("yearly_days", {}).get(y, 0)
            pct = f"{d / ndays * 100:.1f}%" if ndays else "—"
            row.append(str(d))
            row.append(pct)
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("V1 版 v6-1/2/3 逐年恒 0(结构性空池),不列表;v6 侧逐年全表见 #31 §4.7 维度 6(本票不重跑)。")
    L.append("")

    # ---- 8. 结语
    L.append("## 8. 结语(只出数,不评议 V1 优劣,不分析背离v6 底层信号)")
    L.append("")
    L.append("- 本报告全部数字可直接与 #31 §4.2/§4.3/§4.4/§4.7 并排读;v6 行来自 #31 落盘,自检 d 逐格核对一致。")
    L.append("- V1 版 v6-1/2/3 结构性空池是 V1 信号定义(事件日=低点日 ⇒ bounce 恒 < 0)的直接推论,非数据问题。")
    L.append("- 拍板权在用户(issue #34 不设硬线)。")
    L.append("")
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
