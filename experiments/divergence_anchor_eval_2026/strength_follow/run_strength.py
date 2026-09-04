#!/usr/bin/env python3
"""信号后追强势入场(实验B)—— 信号定义不动,只改入场规则(预登记 = 同目录 README.md)。

机制与成本逐条镜像事件层研究 backtest/event_study.py 与实验A pullback_entry/run_pullback.py;
成本/整手/滑点/涨跌停常量与函数直接调用 v1 冻结引擎 strategy_engine(只读),保证逐位一致;
A13 出场语义对齐 strategy_engine_v3 ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=12) 冻结定义。

入场差异(本实验唯一变量,事件日 e 收盘后信号已知,entry_close_ref = e 日收盘):
  B1 不破位确认:窗 e+1..e+k 内 min(low) 从未 < ref×(1-d) − 1e-9 → 第 e+k+1 日开盘买;
     破位 → skipped_breach;窗内零有行情日 → skipped_no_window_data。
  B2 突破确认:H_ref = max(high) over e+1..e+k;其后 5 个交易日内首个 close > H_ref + 1e-9
     触发,次日开盘买;未触发 → skipped_no_trigger。
  对照 k=0:T+1 开盘必买、E1-H12 出场,逐条复刻事件层 simulate_event 入场段(自检闸门)。
入场执行共用:T+1 式开盘买、涨停拒买不递补、无行情不递补、整手/现金逐条复刻事件层。
出场(入场日记持有第 1 日,次一交易日起评估):E1-H12 裸持 12 日收盘卖;
A13 止盈×1.25/止损×0.86/最长 12 日,日内触及语义与事件层逐字一致。

产物:trades_strength.parquet(逐笔全量含 skipped_*/dropped_*/incomplete 行)、
summary_strength.csv(36 格 + 2 对照行,配置为行)、verdict.json、progress.log(心跳)。
全程禁网络;除本目录外仓库只读。
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

import strategy_engine as se  # noqa: E402  v1 冻结引擎(只读复用常量/成本函数/数据加载)

EXP_DIR = os.path.join(REPO_ROOT, "experiments/divergence_anchor_eval_2026")
OUT_DIR = os.path.join(EXP_DIR, "strength_follow")
EVENTS = {
    "events_v1": os.path.join(EXP_DIR, "events_v1.parquet"),
    "events_v2": os.path.join(EXP_DIR, "events_v2.parquet"),
}
EVENT_STUDY_PQ = os.path.join(EXP_DIR, "backtest", "event_study.parquet")
LOG_PATH = os.path.join(OUT_DIR, "progress.log")

START, END = "2026-01-01", "2026-08-31"
BENCH = "000905.SH"
BUDGET = 100_000.0          # 每笔固定名义本金(与事件层一致)
K_GRID = [3, 5, 8]
D_GRID = [0.00, 0.03]
EXIT_RULES = ["E1-H12", "A13"]
HOLD = 12                   # E1-H12 与 A13 共用最长持有(冻结)
TP, SL = 0.25, -0.14        # A13 冻结止盈止损
TRIG_DAYS = 5               # B2 触发窗长度(交易日)
TOL = se.PRICE_TOL          # 1e-9
XTOL = 1e-9                 # 自检逐位容差
BASELINE = {"events_v1": -0.01280, "events_v2": -0.00849}  # 判活线③登记基线(写死)
BASELINE_ES_RECOMP = {"events_v1": -0.012869, "events_v2": -0.008536}  # 交叉披露值


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 数据结构
class StockBars:
    """单股行情对齐到交易日历的 numpy 数组(缺日 = NaN),加速逐日扫描。"""

    __slots__ = ("open_", "high", "low", "close", "up_lim", "dn_lim")

    def __init__(self, n: int) -> None:
        self.open_ = np.full(n, np.nan)
        self.high = np.full(n, np.nan)
        self.low = np.full(n, np.nan)
        self.close = np.full(n, np.nan)
        self.up_lim = np.full(n, np.nan)   # 涨停价;NaN = 当日无约束(引擎同款缺省)
        self.dn_lim = np.full(n, np.nan)


def build_bars(md: se.MarketData, cal_index: dict, codes: list[str]) -> dict:
    """把 MarketData 的日线与涨跌停表对齐成日历下标数组,语义与引擎查找逐条一致。"""
    n = len(cal_index)
    cal_dt = pd.DatetimeIndex(list(cal_index.keys()))
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
    for d, lim in md.limits.items():
        di = cal_index[d]
        up = lim["up_limit"].to_dict()
        dn = lim["down_limit"].to_dict()
        for code in codes:
            u = up.get(code)
            if u is not None:
                bars[code].up_lim[di] = u
                bars[code].dn_lim[di] = dn[code]
    return bars


# ---------------------------------------------------------------- 入场执行与出场(共用段)
def exec_entry(sb: StockBars, di_entry: int) -> dict:
    """入场日开盘买入执行段(无行情拒买/涨停拒买不递补/整手现金,逐条复刻事件层)。"""
    o = sb.open_[di_entry]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = sb.up_lim[di_entry]
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    exec_price = o * (1.0 + se.SLIPPAGE)
    shares = int(BUDGET / exec_price / se.BOARD_LOT) * se.BOARD_LOT
    if shares < se.BOARD_LOT:
        shares = se.BOARD_LOT
    comm = se.buy_cost(shares, exec_price)
    while shares > 0 and shares * exec_price + comm > BUDGET + 1e-6:
        shares -= se.BOARD_LOT
        comm = se.buy_cost(shares, exec_price) if shares > 0 else 0.0
    if shares <= 0:
        return dict(status="dropped_cash")
    return dict(status="entered", entry_price_raw=o, entry_exec_price=exec_price,
                shares=shares, entry_commission=comm)


def exit_scan(sb: StockBars, di_fill: int, exec_price: float, shares: int,
              entry_commission: float, exit_rule: str, cal_arr: list) -> dict:
    """出场段(成交日记持有第 1 日,次一交易日起评估;语义与事件层逐字一致)。"""
    tp_b = exec_price * (1.0 + TP) if exit_rule == "A13" else None
    sl_b = exec_price * (1.0 + SL) if exit_rule == "A13" else None
    deferred = 0
    n_cal = len(cal_arr)
    for di in range(di_fill + 1, n_cal):
        c = sb.close[di]
        if not np.isfinite(c):
            continue  # 无行情日不评估、不触发、持仓沿用
        held = di - di_fill + 1
        o, h, l = sb.open_[di], sb.high[di], sb.low[di]
        raw_exit = None
        reason = None
        if exit_rule == "E1-H12":
            if held >= HOLD:
                raw_exit, reason = c, "horizon"
        else:  # A13:同日双触发保守取止损;开盘越屏障按开盘;到期收盘卖
            tp_hit = h >= tp_b - TOL
            sl_hit = l <= sl_b + TOL
            if tp_hit and not sl_hit:
                raw_exit = o if o >= tp_b - TOL else tp_b
                reason = "tp"
            elif sl_hit:
                raw_exit = o if o <= sl_b + TOL else sl_b
                reason = "sl"
            elif held >= HOLD:
                raw_exit, reason = c, "horizon"
        if raw_exit is None:
            continue
        dn = sb.dn_lim[di]
        if np.isfinite(dn) and c <= dn + TOL:
            deferred += 1
            continue  # 跌停顺延至首个可卖日
        exec_sell = raw_exit * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(shares, exec_sell, cal_arr[di])
        net_pnl = shares * (exec_sell - exec_price) - (entry_commission + xcomm + stamp)
        ret = net_pnl / (shares * exec_price + entry_commission)
        return dict(status="closed", entry_date=cal_arr[di_fill],
                    exit_date=cal_arr[di], exit_reason=reason,
                    exit_raw_price=raw_exit, exit_exec_price=exec_sell,
                    exit_commission=xcomm, stamp_tax=stamp, net_pnl=net_pnl,
                    ret=ret, held_days=held, deferred_days=deferred)
    return dict(status="incomplete", entry_date=cal_arr[di_fill],
                deferred_days=deferred)


# ---------------------------------------------------------------- 逐笔模拟(三 flavor)
def simulate_control(sb: StockBars, di_ev: int, cal_arr: list) -> dict:
    """k=0 对照:T+1 开盘必买、E1-H12 出场。
    判定次序逐条复刻事件层 simulate_event 入场段:
    no_next_day → no_quote → limitup → no_close_T → cash。"""
    n_cal = len(cal_arr)
    if di_ev + 1 >= n_cal:
        return dict(status="dropped_no_next_day")
    di_entry = di_ev + 1
    if not np.isfinite(sb.open_[di_entry]):
        return dict(status="dropped_no_quote")
    up = sb.up_lim[di_entry]
    if np.isfinite(up) and sb.open_[di_entry] >= up - TOL:
        return dict(status="dropped_limitup")
    if not np.isfinite(sb.close[di_ev]):
        return dict(status="dropped_no_close_T")
    ent = exec_entry(sb, di_entry)
    if ent["status"] != "entered":
        return ent  # dropped_cash(quote/limitup 已在上面按事件层次序判过)
    out = exit_scan(sb, di_entry, ent["entry_exec_price"], ent["shares"],
                    ent["entry_commission"], "E1-H12", cal_arr)
    out.update(entry_price_raw=ent["entry_price_raw"],
               entry_exec_price=ent["entry_exec_price"], shares=ent["shares"],
               entry_commission=ent["entry_commission"])
    return out


def simulate_b1(sb: StockBars, di_ev: int, k: int, d: float,
                exit_rule: str, cal_arr: list) -> dict:
    """B1 不破位确认。判定次序(预登记 §3):
    no_next_day → no_close_T → 窗扫描(skipped_*) → no_quote → limitup → cash。"""
    n_cal = len(cal_arr)
    if di_ev + k + 1 >= n_cal:
        return dict(status="dropped_no_next_day")
    ref = sb.close[di_ev]
    if not np.isfinite(ref):
        return dict(status="dropped_no_close_T")
    thr = ref * (1.0 - d)
    lo = sb.low[di_ev + 1: di_ev + k + 1]
    finite = np.isfinite(lo)
    n_bars = int(finite.sum())
    if n_bars == 0:
        return dict(status="skipped_no_window_data", breach_threshold=thr,
                    n_window_bars=0)
    w_low = float(np.min(lo[finite]))
    diag = dict(W_low=w_low, breach_threshold=thr, n_window_bars=n_bars)
    if w_low < thr - TOL:
        return dict(status="skipped_breach", **diag)
    di_entry = di_ev + k + 1
    ent = exec_entry(sb, di_entry)
    if ent["status"] != "entered":
        ent.update(diag)
        return ent
    out = exit_scan(sb, di_entry, ent["entry_exec_price"], ent["shares"],
                    ent["entry_commission"], exit_rule, cal_arr)
    out.update(entry_price_raw=ent["entry_price_raw"],
               entry_exec_price=ent["entry_exec_price"], shares=ent["shares"],
               entry_commission=ent["entry_commission"], **diag)
    return out


def simulate_b2(sb: StockBars, di_ev: int, k: int,
                exit_rule: str, cal_arr: list) -> dict:
    """B2 突破确认。判定次序(预登记 §3):
    no_next_day(参考窗截断) → 参考窗扫描(no_window_data) → 触发窗扫描
    (no_trigger / 触发) → no_next_day(触发日无次日) → no_quote → limitup → cash。"""
    n_cal = len(cal_arr)
    if di_ev + k >= n_cal:
        return dict(status="dropped_no_next_day")
    hi = sb.high[di_ev + 1: di_ev + k + 1]
    finite = np.isfinite(hi)
    n_bars = int(finite.sum())
    if n_bars == 0:
        return dict(status="skipped_no_window_data", n_window_bars=0)
    h_ref = float(np.max(hi[finite]))
    diag = dict(H_ref=h_ref, n_window_bars=n_bars)
    di_trig = -1
    di_scan_end = min(di_ev + k + TRIG_DAYS, n_cal - 1)
    for di in range(di_ev + k + 1, di_scan_end + 1):
        c = sb.close[di]
        if not np.isfinite(c):
            continue  # 无行情日不评估,触发窗照常流逝
        if c > h_ref + TOL:
            di_trig = di
            break
    if di_trig < 0:
        return dict(status="skipped_no_trigger",
                    trig_window_truncated=bool(di_ev + k + TRIG_DAYS > n_cal - 1),
                    **diag)
    diag["trigger_date"] = cal_arr[di_trig]
    if di_trig + 1 >= n_cal:
        return dict(status="dropped_no_next_day", **diag)
    di_entry = di_trig + 1
    ent = exec_entry(sb, di_entry)
    if ent["status"] != "entered":
        ent.update(diag)
        return ent
    out = exit_scan(sb, di_entry, ent["entry_exec_price"], ent["shares"],
                    ent["entry_commission"], exit_rule, cal_arr)
    out.update(entry_price_raw=ent["entry_price_raw"],
               entry_exec_price=ent["entry_exec_price"], shares=ent["shares"],
               entry_commission=ent["entry_commission"], **diag)
    return out


# ---------------------------------------------------------------- 聚类稳健 t(Liang-Zeger)
def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    """单样本均值的聚类稳健 t。预登记口径(README 第 6 节):
    得分 s_i = x_i − x̄;簇得分和 S_c;var(x̄) = [G/(G−1)] × Σ S_c² / n²。"""
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
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log(f"STRENGTH FOLLOW START window=[{START}..{END}] bench={BENCH} budget={BUDGET:.0f} "
        f"预计总时长约5分钟(参照实验A同机18配置22秒;本实验38配置+两次数据加载) | "
        f"B1: k={K_GRID}×d={D_GRID} B2: k={K_GRID} trig_days={TRIG_DAYS} "
        f"exits={EXIT_RULES} + k=0 对照(自检闸门)")

    # 基准收盘序列(超额口径:收盘(入场日) → 收盘(出场日),与事件层一致)
    bdf = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index", f"{BENCH}.parquet"),
                          columns=["trade_date", "close"])
    bdf["trade_date"] = pd.to_datetime(bdf["trade_date"])
    bench_close = bdf.set_index("trade_date")["close"].astype(float).sort_index()

    # 事件层 E1-H12 closed 检索表(被跳过子集前向收益刻画 + 自检闸门对照)
    es = pd.read_parquet(EVENT_STUDY_PQ)
    es["event_date"] = pd.to_datetime(es["event_date"])
    es_e1 = es[(es["config"] == "E1-H12")]
    es_e1_closed = es_e1[es_e1["status"] == "closed"]
    es_ret_map = {(s, c, d): r for s, c, d, r in zip(
        es_e1_closed["signal"], es_e1_closed["ts_code"],
        es_e1_closed["event_date"], es_e1_closed["ret"])}
    log(f"event_study.parquet 载入:总行={len(es)} E1-H12 行={len(es_e1)} "
        f"closed={len(es_e1_closed)}(自检对照与跳过子集刻画用)")

    all_rows: list[pd.DataFrame] = []

    for sig_name, ev_path in EVENTS.items():
        t_sig = time.time()
        ev = pd.read_parquet(ev_path)
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        assert ev["dif_lift"].notna().all() and (ev["dif_lift"] > 0).all()
        assert not ev.duplicated(["ts_code", "event_date"]).any()
        md = se.load_market_data(ev["ts_code"].unique().tolist(), START, END,
                                 REPO_ROOT, log_path=LOG_PATH)
        cal_arr = list(md.calendar)
        cal_index = {d: i for i, d in enumerate(cal_arr)}
        ev = ev[ev["event_date"].isin(set(cal_arr))].reset_index(drop=True)
        assert md.limit_missing_dates == 0, "本预登记假设 stk_limit 窗口内零缺日"
        log(f"{sig_name}: 事件数={len(ev)} 日历交易日={len(cal_arr)};构建对齐行情数组...")
        bars = build_bars(md, cal_index, ev["ts_code"].unique().tolist())
        di_evs = ev["event_date"].map(cal_index).to_numpy(dtype=int)
        codes = ev["ts_code"].to_numpy()
        log(f"{sig_name}: 数组就绪({time.time() - t_sig:.0f}s),开始 19 配置模拟")

        # 配置清单:k=0 对照先行(自检闸门载体),再 B1 网格,再 B2 网格
        configs = [("control", 0, np.nan, "E1-H12")]
        for k in K_GRID:
            for d in D_GRID:
                for ex in EXIT_RULES:
                    configs.append(("B1", k, d, ex))
        for k in K_GRID:
            for ex in EXIT_RULES:
                configs.append(("B2", k, np.nan, ex))
        assert len(configs) == 19

        for flavor, k, d, exit_rule in configs:
            t0 = time.time()
            recs: list[dict] = []
            for i in range(len(ev)):
                sb = bars[codes[i]]
                di_ev = int(di_evs[i])
                if flavor == "control":
                    res = simulate_control(sb, di_ev, cal_arr)
                elif flavor == "B1":
                    res = simulate_b1(sb, di_ev, k, float(d), exit_rule, cal_arr)
                else:
                    res = simulate_b2(sb, di_ev, k, exit_rule, cal_arr)
                res.update(signal=sig_name, flavor=flavor, k=k,
                           d=None if np.isnan(d) else float(d), exit_rule=exit_rule,
                           ts_code=codes[i], event_date=ev["event_date"].iat[i],
                           dif_lift=float(ev["dif_lift"].iat[i]))
                recs.append(res)
            gdf = pd.DataFrame(recs)
            entered = gdf[gdf["status"].isin({"closed", "incomplete"})]
            assert (entered["entry_date"] > entered["event_date"]).all(), \
                f"{sig_name} {flavor} k={k} {exit_rule}: 存在 entry_date <= event_date(泄漏)"
            closed = gdf[gdf["status"] == "closed"]
            if len(closed):
                b_in = closed["entry_date"].map(bench_close)
                b_out = closed["exit_date"].map(bench_close)
                assert b_in.notna().all() and b_out.notna().all(), "基准未覆盖持有窗"
                gdf.loc[closed.index, "bench_ret"] = (b_out / b_in - 1.0).to_numpy()
                gdf.loc[closed.index, "excess"] = \
                    (closed["ret"] - (b_out / b_in - 1.0)).to_numpy()
            all_rows.append(gdf)
            st = gdf["status"].value_counts().to_dict()
            d_lab = "-" if np.isnan(d) else f"{d:.2f}"
            log(f"{sig_name} {flavor} k={k} d={d_lab} {exit_rule} 完成 "
                f"{time.time() - t0:.0f}s | closed={st.get('closed', 0)} "
                f"incomplete={st.get('incomplete', 0)} "
                f"skipped_breach={st.get('skipped_breach', 0)} "
                f"skipped_no_trigger={st.get('skipped_no_trigger', 0)} "
                f"skipped_no_window_data={st.get('skipped_no_window_data', 0)} "
                f"dropped_limitup={st.get('dropped_limitup', 0)} "
                f"dropped_no_quote={st.get('dropped_no_quote', 0)} "
                f"dropped_no_next_day={st.get('dropped_no_next_day', 0)} "
                f"dropped_cash={st.get('dropped_cash', 0)}")
        log(f"{sig_name} 全部 19 配置完成,累计 {time.time() - t_sig:.0f}s")

    full = pd.concat(all_rows, ignore_index=True)
    out_pq = os.path.join(OUT_DIR, "trades_strength.parquet")
    full.to_parquet(out_pq, index=False)
    log(f"trades_strength.parquet 行数={len(full)} -> {out_pq}")

    # ---------------------------------------------------------------- 自检闸门(零容差)
    log("SELF-CHECK START: k=0 对照 对拍 event_study.parquet E1-H12")
    gate = {}
    gate_ok = True
    for sig_name in EVENTS:
        mine = full[(full["signal"] == sig_name) & (full["flavor"] == "control")]
        ref = es_e1[es_e1["signal"] == sig_name]
        a = mine.set_index(["ts_code", "event_date"])
        b = ref.set_index(["ts_code", "event_date"])
        assert len(a) == len(b) and a.index.sort_values().equals(
            b.index.sort_values()), \
            f"{sig_name} control: 事件集合与 event_study 不一致"
        b = b.reindex(a.index)
        n_status_mm = int((a["status"] != b["status"]).sum())
        both = a.join(b, lsuffix="_mine", rsuffix="_ref", how="inner")
        cl = both[both["status_mine"] == "closed"]
        ret_diff = (cl["ret_mine"] - cl["ret_ref"]).abs()
        exc_diff = (cl["excess_mine"] - cl["excess_ref"]).abs()
        n_ret_mm = int((ret_diff > XTOL).sum())
        n_exc_mm = int((exc_diff > XTOL).sum())
        n_date_mm = int((cl["entry_date_mine"] != cl["entry_date_ref"]).sum()
                        + (cl["exit_date_mine"] != cl["exit_date_ref"]).sum())
        # 任务书口径:status=closed 笔数一致
        n_closed_mine = int((a["status"] == "closed").sum())
        n_closed_ref = int((b["status"] == "closed").sum())
        gate[sig_name] = dict(
            n_events=len(a), n_closed_mine=n_closed_mine, n_closed_ref=n_closed_ref,
            n_status_mismatch=n_status_mm, n_ret_mismatch=n_ret_mm,
            n_excess_mismatch=n_exc_mm,
            max_ret_diff=float(ret_diff.max()) if len(cl) else None,
            max_excess_diff=float(exc_diff.max()) if len(cl) else None,
            n_date_mismatch=n_date_mm)
        log(f"对拍 {sig_name} control vs event_study E1-H12: "
            f"closed {n_closed_mine} vs {n_closed_ref} status 不一致={n_status_mm} "
            f"ret 超容差={n_ret_mm} excess 超容差={n_exc_mm} 日期不一致={n_date_mm}")
        if (n_status_mm or n_ret_mm or n_exc_mm or n_date_mm
                or n_closed_mine != n_closed_ref):
            gate_ok = False
    gate["passed"] = gate_ok
    if not gate_ok:
        log("SELF-CHECK FAILED: k=0 对照与 event_study 存在逐笔不一致,停止,不进入判读")
        _write_verdict(gate, None, None, t_all)
        sys.exit(2)
    log("SELF-CHECK PASSED: k=0 对照与 event_study E1-H12 全量逐笔一致"
        "(status 全同,closed 笔数一致,ret/excess 容差 1e-9)")

    # ---------------------------------------------------------------- 汇总(配置为行)
    sum_rows: list[dict] = []
    for (sig, flavor, k, d_key, ex), gdf in full.groupby(
            ["signal", "flavor", "k", "d", "exit_rule"], sort=False, dropna=False):
        d_out = float(d_key) if pd.notna(d_key) else np.nan
        st = gdf["status"].value_counts().to_dict()
        n_sig = len(gdf)
        n_closed = int(st.get("closed", 0))
        n_incomplete = int(st.get("incomplete", 0))
        n_trades = n_closed + n_incomplete
        cl = gdf[gdf["status"] == "closed"]
        ret = cl["ret"].to_numpy(dtype=float)
        exc = cl["excess"].to_numpy(dtype=float)
        ct = cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) if n_closed else np.nan
        ret_mean = float(ret.mean()) if n_closed else np.nan
        # 被跳过子集刻画:规则跳过子集在 event_study E1-H12 closed 上的前向收益均值
        if flavor == "B1":
            skip_mask = gdf["status"] == "skipped_breach"
        elif flavor == "B2":
            skip_mask = gdf["status"] == "skipped_no_trigger"
        else:
            skip_mask = pd.Series(False, index=gdf.index)
        skipped = gdf[skip_mask]
        sk_rets = [es_ret_map[(sig, r.ts_code, r.event_date)]
                   for r in skipped.itertuples()
                   if (sig, r.ts_code, r.event_date) in es_ret_map]
        sk_fwd_mean = float(np.mean(sk_rets)) if sk_rets else np.nan
        # 窗内缺行情日披露(观察/参考窗有缺日但仍完成评估的笔数)
        n_window_gap = int((gdf["n_window_bars"].fillna(k) < k).sum()) \
            if flavor in ("B1", "B2") else 0
        n_trig_trunc = int(gdf["trig_window_truncated"].fillna(False).sum()) \
            if flavor == "B2" else 0
        exc_vs_base = (ret_mean - BASELINE[sig]) if n_closed else np.nan
        if flavor == "control":
            verdict = "control"
        else:
            alive = (n_closed > 0 and ret_mean > 0
                     and np.isfinite(ct) and ct >= 2.0
                     and exc_vs_base >= 0.02)
            verdict = "alive" if alive else "dead"
        sum_rows.append(dict(
            signal=sig, flavor=flavor, k=int(k), d=d_out, exit=ex,
            n_signals=n_sig,
            trade_rate=n_trades / n_sig if n_sig else np.nan,
            n_trades=n_trades,
            ret_mean=ret_mean,
            ret_median=float(np.median(ret)) if n_closed else np.nan,
            win_rate=float((ret > 0).mean()) if n_closed else np.nan,
            excess_vs_baseline=exc_vs_base,
            cluster_t=ct,
            skipped_subset_fwd_mean=sk_fwd_mean,
            verdict=verdict,
            # ---- 以下为披露列 ----
            skipped_n=int(skip_mask.sum()),
            skipped_matched_n=len(sk_rets),
            n_closed=n_closed, n_incomplete=n_incomplete,
            n_skipped_breach=int(st.get("skipped_breach", 0)),
            n_skipped_no_trigger=int(st.get("skipped_no_trigger", 0)),
            n_skipped_no_window_data=int(st.get("skipped_no_window_data", 0)),
            n_dropped_no_next_day=int(st.get("dropped_no_next_day", 0)),
            n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
            n_dropped_limitup=int(st.get("dropped_limitup", 0)),
            n_dropped_no_close_T=int(st.get("dropped_no_close_T", 0)),
            n_dropped_cash=int(st.get("dropped_cash", 0)),
            n_window_gap=n_window_gap, n_trig_window_truncated=n_trig_trunc,
            excess_mean=float(exc.mean()) if n_closed else np.nan,
            bench_ret_mean=float(cl["bench_ret"].mean()) if n_closed else np.nan,
            held_days_mean=float(cl["held_days"].mean()) if n_closed else np.nan,
        ))
    summary = pd.DataFrame(sum_rows)
    sum_path = os.path.join(OUT_DIR, "summary_strength.csv")
    summary.to_csv(sum_path, index=False, float_format="%.6f")
    log(f"summary_strength.csv -> {sum_path}")

    grid_rows = [r for r in sum_rows if r["flavor"] != "control"]
    n_alive = sum(1 for r in grid_rows if r["verdict"] == "alive")
    log(f"判活:36 格中过线 {n_alive} 格;" + ("存在判活格" if n_alive else "全格不过线 = B 方向判死"))
    _write_verdict(gate, summary, grid_rows, t_all)
    log(f"STRENGTH FOLLOW DONE ({time.time() - t_all:.0f}s)")

    pd.set_option("display.width", 300)
    show_cols = ["signal", "flavor", "k", "d", "exit", "n_signals", "trade_rate",
                 "n_trades", "ret_mean", "ret_median", "win_rate",
                 "excess_vs_baseline", "cluster_t", "skipped_subset_fwd_mean", "verdict"]
    print("\n===== SUMMARY(配置为行;control 为自检/对照行)=====")
    print(summary[show_cols].to_string(index=False))


def _write_verdict(gate: dict, summary: pd.DataFrame | None,
                   grid_rows: list[dict] | None, t_all: float) -> None:
    best_worst = None
    if grid_rows:
        best_worst = {}
        for sig in {r["signal"] for r in grid_rows}:
            rows = [r for r in grid_rows if r["signal"] == sig
                    and np.isfinite(r["ret_mean"] if r["ret_mean"] is not None else np.nan)]
            if rows:
                b = max(rows, key=lambda r: r["ret_mean"])
                w = min(rows, key=lambda r: r["ret_mean"])
                key = lambda r: f"{r['flavor']}_k{r['k']}_d{r['d']}_{r['exit']}"
                best_worst[sig] = dict(
                    best=dict(cell=key(b), ret_mean=b["ret_mean"],
                              cluster_t=b["cluster_t"], trade_rate=b["trade_rate"]),
                    worst=dict(cell=key(w), ret_mean=w["ret_mean"],
                               cluster_t=w["cluster_t"], trade_rate=w["trade_rate"]))
    out = dict(
        experiment="strength_follow(实验B:信号后追强势入场)",
        prereadme="README.md 先于任何跑数落盘",
        baseline_registered=BASELINE,
        baseline_event_study_recomputed=BASELINE_ES_RECOMP,
        gate=gate,
        verdicts=([dict(signal=r["signal"], flavor=r["flavor"], k=r["k"], d=r["d"],
                        exit=r["exit"], ret_mean=r["ret_mean"],
                        excess_vs_baseline=r["excess_vs_baseline"],
                        cluster_t=r["cluster_t"], trade_rate=r["trade_rate"],
                        skipped_subset_fwd_mean=r["skipped_subset_fwd_mean"],
                        verdict=r["verdict"]) for r in grid_rows]
                  if grid_rows else None),
        best_worst_per_signal=best_worst,
        n_alive=(sum(1 for r in grid_rows if r["verdict"] == "alive")
                 if grid_rows else None),
        overall=("判死:36 格全部不过线,B 方向(信号后追强势入场)判死,阴性结果原样交付"
                 if grid_rows is not None
                 and not any(r["verdict"] == "alive" for r in grid_rows)
                 else "存在判活格" if grid_rows is not None else "自检未过,未进入判读"),
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    path = os.path.join(OUT_DIR, "verdict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"verdict.json -> {path}")


if __name__ == "__main__":
    main()
