#!/usr/bin/env python3
"""事件后回踩挂单入场(实验A)—— 信号定义不动,只改入场规则(预登记 = 同目录 README.md)。

机制与成本逐条镜像事件层研究 backtest/event_study.py(其对齐清单 = backtest/README.md
增补 D 节 1~18 条);成本/整手/滑点/涨跌停常量与函数直接调用 v1 冻结引擎
strategy_engine(只读),保证逐位一致。

入场差异(本实验唯一变量):事件日次一交易日起挂限价买单,限价 = 锚点收盘 × (1+m),
有效期 20 个交易日;当日最低价 ≤ 限价则成交,成交原始价 = min(当日开盘价, 限价);
涨停日(开盘价 ≥ 涨停价 − 1e-9,涨停价取自 stock_data/stk_limit,引擎同源)不成交、
挂单留效顺延;过期未成交记 no_fill(未成交,不是亏损)。
对照变体 m=+∞:事件日次一交易日开盘必成交,逐条复刻事件层入场段(含涨停拒买不递补、
无行情剔除),用于与 event_study.parquet 逐笔对拍(自检闸门)。

出场三档(自成交日起算,成交日记持有第 1 日,次一交易日起方可卖):
  E1-H12    裸持 12 个交易日收盘卖(引擎 E1 模式);
  A13       止盈 = 买入执行价 × 1.25,止损 = 买入执行价 × 0.86,最长 12 日;
  ANCHOR-SL 止盈 = 买入执行价 × 1.25,止损价 = 锚点收盘 × 0.97(绝对价),最长 12 日。
日内触及语义(容差 1e-9、同日双触发保守取止损、开盘价越屏障按开盘价、
跌停顺延、无行情日不评估)与事件层逐字一致。

产物:trades_pullback.parquet(逐笔全量含 no_fill/dropped_*/incomplete 行)、
summary_pullback.csv(12 行 + 6 对照行)、verdict.json、run_pullback.log。
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
OUT_DIR = os.path.join(EXP_DIR, "pullback_entry")
EVENTS = {
    "events_v1": os.path.join(EXP_DIR, "events_v1.parquet"),
    "events_v2": os.path.join(EXP_DIR, "events_v2.parquet"),
}
EVENT_STUDY_PQ = os.path.join(EXP_DIR, "backtest", "event_study.parquet")
LOG_PATH = os.path.join(OUT_DIR, "run_pullback.log")

START, END = "2026-01-01", "2026-08-31"
BENCH = "000905.SH"
BUDGET = 100_000.0          # 每笔固定名义本金(与事件层一致)
VALID_DAYS = 20             # 挂单有效期(交易日,自事件日次一交易日起含当日)
M_GRID = [0.00, 0.03]       # 限价档位;对照变体 m=+∞ 单独常量化
M_CTRL = np.inf
EXIT_RULES = ["E1-H12", "A13", "ANCHOR-SL"]
TOL = se.PRICE_TOL          # 1e-9
XTOL = 1e-9                 # 对拍逐位容差


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
            loc = cal_dt.get_indexer(df.index)   # 日历外日期 = -1(加载窗前后各多 60 自然日)
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


# ---------------------------------------------------------------- 逐笔模拟
def simulate_pullback(sb: StockBars, di_ev: int, anchor_close: float, m: float,
                      exit_rule: str, cal_arr: list,
                      day_has_limit_file: np.ndarray) -> dict:
    """单事件逐笔模拟。入场为限价挂单(m 有限)或次一交易日开盘必成交(m=+∞ 对照)。

    返回 status ∈ {closed, no_fill, incomplete, dropped_no_next_day,
    dropped_no_close_T, dropped_limitup, dropped_no_quote, dropped_cash}。
    对照路径(m=+∞)的状态机与事件层 simulate_event 入场段逐条同序:
    无行情 → 涨停拒买 → 事件日收盘存在性校验。
    """
    n_cal = len(cal_arr)
    if di_ev + 1 >= n_cal:
        return dict(status="dropped_no_next_day")
    is_ctrl = np.isinf(m)
    limit_px = np.inf if is_ctrl else anchor_close * (1.0 + m)

    if not is_ctrl:
        # 挂单路径:事件日收盘存在性校验(引擎同款),再进入有效期扫描
        if not np.isfinite(sb.close[di_ev]):
            return dict(status="dropped_no_close_T")

    di_fill = -1
    raw_fill = np.nan
    di_last = n_cal - 1 if is_ctrl else min(di_ev + VALID_DAYS, n_cal - 1)
    for di in range(di_ev + 1, di_last + 1):
        o = sb.open_[di]
        if not np.isfinite(o):
            if is_ctrl:
                return dict(status="dropped_no_quote")
            continue  # 无行情日不成交,挂单留效
        up = sb.up_lim[di]
        if np.isfinite(up):
            if o >= up - TOL:
                if is_ctrl:
                    return dict(status="dropped_limitup")
                continue  # 涨停日不可买,挂单留效顺延
        elif day_has_limit_file[di]:
            pass  # 当日涨跌停表存在但该股不在表中 = 无约束(引擎同款缺省)
        else:
            # 兜底代理规则(任务书口径):仅 stk_limit 当日表整体缺失时启用;
            # 本窗口缺日=0,不触发
            pre = sb.close[di - 1] if di - 1 >= 0 else np.nan
            if np.isfinite(pre):
                one_way = (sb.low[di] == sb.high[di]) and (sb.close[di] / pre - 1.0 >= 0.098)
                if o >= pre * 1.098 or one_way:
                    if is_ctrl:
                        return dict(status="dropped_limitup")
                    continue
        if is_ctrl:
            # 对照路径:事件日收盘存在性校验次序与事件层一致(涨停判定之后)
            if not np.isfinite(sb.close[di_ev]):
                return dict(status="dropped_no_close_T")
            raw_fill = o
            di_fill = di
            break
        if sb.low[di] <= limit_px:
            raw_fill = min(o, limit_px)  # 开盘跳空低于限价按开盘价成交
            di_fill = di
            break
    if di_fill < 0:
        return dict(status="no_fill")

    # 整手与现金(逐条复刻引擎/事件层:不足一手买一手,现金不足逐手递减)
    exec_price = raw_fill * (1.0 + se.SLIPPAGE)
    shares = int(BUDGET / exec_price / se.BOARD_LOT) * se.BOARD_LOT
    if shares < se.BOARD_LOT:
        shares = se.BOARD_LOT
    comm = se.buy_cost(shares, exec_price)
    while shares > 0 and shares * exec_price + comm > BUDGET + 1e-6:
        shares -= se.BOARD_LOT
        comm = se.buy_cost(shares, exec_price) if shares > 0 else 0.0
    if shares <= 0:
        return dict(status="dropped_cash")
    entry_commission = comm

    # 出场(成交日记持有第 1 日,次一交易日起评估;语义与事件层逐字一致)
    tp_b = exec_price * 1.25 if exit_rule in ("A13", "ANCHOR-SL") else None
    sl_b = (exec_price * 0.86 if exit_rule == "A13"
            else anchor_close * 0.97 if exit_rule == "ANCHOR-SL" else None)
    deferred = 0
    for di in range(di_fill + 1, n_cal):
        c = sb.close[di]
        if not np.isfinite(c):
            continue  # 无行情日不评估、不触发、持仓沿用
        held = di - di_fill + 1
        o, h, l = sb.open_[di], sb.high[di], sb.low[di]
        raw_exit = None
        reason = None
        if exit_rule == "E1-H12":
            if held >= 12:
                raw_exit, reason = c, "horizon"
        else:
            tp_hit = h >= tp_b - TOL
            sl_hit = l <= sl_b + TOL
            if tp_hit and not sl_hit:
                raw_exit = o if o >= tp_b - TOL else tp_b
                reason = "tp"
            elif sl_hit:
                raw_exit = o if o <= sl_b + TOL else sl_b
                reason = "sl"
            elif held >= 12:
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
        return dict(status="closed", fill_date=cal_arr[di_fill], fill_price=raw_fill,
                    entry_exec_price=exec_price, shares=shares,
                    entry_commission=entry_commission, exit_date=cal_arr[di],
                    exit_reason=reason, exit_raw_price=raw_exit, exit_price=exec_sell,
                    exit_commission=xcomm, stamp_tax=stamp, net_pnl=net_pnl, ret=ret,
                    held_days=held, deferred_days=deferred)
    return dict(status="incomplete", fill_date=cal_arr[di_fill], fill_price=raw_fill,
                entry_exec_price=exec_price, shares=shares,
                entry_commission=entry_commission, deferred_days=deferred)


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
    log(f"PULLBACK ENTRY START window=[{START}..{END}] bench={BENCH} budget={BUDGET:.0f} "
        f"m_grid={M_GRID}+[inf 对照] exits={EXIT_RULES} valid_days={VALID_DAYS}")

    # 基准收盘序列(超额口径:收盘(成交日) → 收盘(出场日),与事件层一致)
    bdf = pd.read_parquet(os.path.join(REPO_ROOT, "stock_data/index", f"{BENCH}.parquet"),
                          columns=["trade_date", "close"])
    bdf["trade_date"] = pd.to_datetime(bdf["trade_date"])
    bench_close = bdf.set_index("trade_date")["close"].astype(float).sort_index()

    all_rows: list[pd.DataFrame] = []
    grid = [(sig, m, ex) for sig in EVENTS for m in M_GRID + [M_CTRL] for ex in EXIT_RULES]

    for sig_name, ev_path in EVENTS.items():
        t_sig = time.time()
        ev = pd.read_parquet(ev_path)
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        ev["anchor_date"] = pd.to_datetime(ev["anchor_date"])
        assert ev["dif_lift"].notna().all() and (ev["dif_lift"] > 0).all()
        assert not ev.duplicated(["ts_code", "event_date"]).any()
        md = se.load_market_data(ev["ts_code"].unique().tolist(), START, END,
                                 REPO_ROOT, log_path=LOG_PATH)
        cal_arr = list(md.calendar)
        cal_index = {d: i for i, d in enumerate(cal_arr)}
        ev = ev[ev["event_date"].isin(set(cal_arr))].reset_index(drop=True)
        assert md.limit_missing_dates == 0, "本预登记假设 stk_limit 窗口内零缺日"
        day_has_limit_file = np.array(
            [d in md.limits for d in cal_arr], dtype=bool)
        log(f"{sig_name}: 事件数={len(ev)} 日历交易日={len(cal_arr)};构建对齐行情数组...")
        bars = build_bars(md, cal_index, ev["ts_code"].unique().tolist())
        di_evs = ev["event_date"].map(cal_index).to_numpy(dtype=int)
        anchors = ev["anchor_close"].to_numpy(dtype=float)
        codes = ev["ts_code"].to_numpy()
        log(f"{sig_name}: 数组就绪({time.time() - t_sig:.0f}s),开始 9 配置模拟")

        for m in M_GRID + [M_CTRL]:
            for exit_rule in EXIT_RULES:
                t0 = time.time()
                recs: list[dict] = []
                m_lab = "inf" if np.isinf(m) else f"{m:.2f}"
                for i in range(len(ev)):
                    res = simulate_pullback(bars[codes[i]], int(di_evs[i]),
                                            float(anchors[i]), m, exit_rule, cal_arr,
                                            day_has_limit_file)
                    res.update(signal=sig_name, m=m_lab, exit_rule=exit_rule,
                               ts_code=codes[i], event_date=ev["event_date"].iat[i],
                               anchor_close=float(anchors[i]),
                               dif_lift=float(ev["dif_lift"].iat[i]))
                    recs.append(res)
                gdf = pd.DataFrame(recs)
                closed = gdf[gdf["status"] == "closed"]
                assert (closed["fill_date"] > closed["event_date"]).all(), \
                    f"{sig_name} m={m_lab} {exit_rule}: 存在 fill_date <= event_date(泄漏)"
                if len(closed):
                    b_in = closed["fill_date"].map(bench_close)
                    b_out = closed["exit_date"].map(bench_close)
                    assert b_in.notna().all() and b_out.notna().all(), "基准未覆盖持有窗"
                    gdf.loc[closed.index, "bench_ret"] = (b_out / b_in - 1.0).to_numpy()
                    gdf.loc[closed.index, "excess"] = \
                        (closed["ret"] - (b_out / b_in - 1.0)).to_numpy()
                all_rows.append(gdf)
                st = gdf["status"].value_counts().to_dict()
                log(f"{sig_name} m={m_lab} {exit_rule} 完成 {time.time() - t0:.0f}s | "
                    f"closed={st.get('closed', 0)} no_fill={st.get('no_fill', 0)} "
                    f"incomplete={st.get('incomplete', 0)} "
                    f"dropped_limitup={st.get('dropped_limitup', 0)} "
                    f"dropped_no_quote={st.get('dropped_no_quote', 0)} "
                    f"dropped_cash={st.get('dropped_cash', 0)}")
        log(f"{sig_name} 全部配置完成,累计 {time.time() - t_sig:.0f}s")

    full = pd.concat(all_rows, ignore_index=True)
    out_pq = os.path.join(OUT_DIR, "trades_pullback.parquet")
    full.to_parquet(out_pq, index=False)
    log(f"trades_pullback.parquet 行数={len(full)} -> {out_pq}")

    # ---------------------------------------------------------------- 自检闸门
    log("SELF-CHECK START: m=+∞ 对照 对拍 event_study.parquet")
    es = pd.read_parquet(EVENT_STUDY_PQ)
    es["event_date"] = pd.to_datetime(es["event_date"])
    gate = {}
    gate_ok = True
    for sig_name in EVENTS:
        for my_rule, es_cfg in (("E1-H12", "E1-H12"), ("A13", "A13")):
            mine = full[(full["signal"] == sig_name) & (full["m"] == "inf")
                        & (full["exit_rule"] == my_rule)]
            ref = es[(es["signal"] == sig_name) & (es["config"] == es_cfg)]
            a = mine.set_index(["ts_code", "event_date"])
            b = ref.set_index(["ts_code", "event_date"])
            assert len(a) == len(b) and a.index.sort_values().equals(
                b.index.sort_values()), \
                f"{sig_name} {my_rule}: 事件集合与 event_study 不一致"
            b = b.reindex(a.index)
            n_status_mm = int((a["status"] != b["status"]).sum())
            # join 后缀仅作用于重名列;event_study 独有的 entry_date 保持原名
            both = a.join(b, lsuffix="_mine", rsuffix="_ref", how="inner")
            cl = both[both["status_mine"] == "closed"]
            ret_diff = (cl["ret_mine"] - cl["ret_ref"]).abs()
            n_ret_mm = int((ret_diff > XTOL).sum())
            n_date_mm = int((cl["fill_date"] != cl["entry_date"]).sum()
                            + (cl["exit_date_mine"] != cl["exit_date_ref"]).sum())
            gate[f"{sig_name}__{my_rule}"] = dict(
                n_events=len(a), n_status_mismatch=n_status_mm,
                n_closed=int(len(cl)), n_ret_mismatch=n_ret_mm,
                max_ret_diff=float(ret_diff.max()) if len(cl) else None,
                n_date_mismatch=n_date_mm)
            log(f"对拍 {sig_name} {my_rule} vs event_study {es_cfg}: "
                f"status 不一致={n_status_mm} ret 超容差={n_ret_mm} "
                f"日期不一致={n_date_mm} (closed={len(cl)})")
            if n_status_mm or n_ret_mm or n_date_mm:
                gate_ok = False
    gate["passed"] = gate_ok
    if not gate_ok:
        log("SELF-CHECK FAILED: 对照变体与 event_study 存在逐笔不一致,停止,不进入判读")
        _write_verdict(gate, None, None, t_all)
        sys.exit(2)
    log("SELF-CHECK PASSED: 对照变体与 event_study 全量逐笔一致(status 全同,ret 容差 1e-9)")

    # ---------------------------------------------------------------- 汇总(配置为行)
    zero_status = {"no_fill", "dropped_limitup", "dropped_no_quote", "dropped_cash"}
    sum_rows: list[dict] = []
    for (sig, m_lab, ex), gdf in full.groupby(["signal", "m", "exit_rule"], sort=False):
        st = gdf["status"].value_counts().to_dict()
        n_sig = len(gdf)
        n_closed = int(st.get("closed", 0))
        n_incomplete = int(st.get("incomplete", 0))
        # 有效信号分母:剔除日历/数据不可执行与窗口末端未闭合(与事件层剔除口径一致)
        n_valid = n_sig - int(st.get("dropped_no_next_day", 0)) \
            - int(st.get("dropped_no_close_T", 0)) - n_incomplete
        cl = gdf[gdf["status"] == "closed"]
        ret = cl["ret"].to_numpy(dtype=float)
        exc = cl["excess"].to_numpy(dtype=float)
        # 按信号分母序列:成交 = ret/excess,未成交 = 0;聚类键 = 成交日(未成交取事件日)
        xs = np.where(gdf["status"] == "closed", gdf["ret"].fillna(0.0), 0.0)
        xe = np.where(gdf["status"] == "closed", gdf["excess"].fillna(0.0), 0.0)
        in_denom = ~(gdf["status"].isin({"dropped_no_next_day", "dropped_no_close_T",
                                         "incomplete"}))
        xs, xe = xs[in_denom.to_numpy()], xe[in_denom.to_numpy()]
        clus = np.where(gdf["status"] == "closed",
                        gdf["fill_date"].astype(str), gdf["event_date"].astype(str))
        clus = clus[in_denom.to_numpy()]
        exp_ret = float(xs.sum() / n_valid) if n_valid > 0 else np.nan
        exp_exc = float(xe.sum() / n_valid) if n_valid > 0 else np.nan
        sum_rows.append(dict(
            signal=sig, m=m_lab, exit_rule=ex, n_signals=n_sig,
            n_valid=n_valid, n_filled=n_closed,
            fill_rate=n_closed / n_valid if n_valid > 0 else np.nan,
            ret_mean=float(ret.mean()) if n_closed else np.nan,
            ret_median=float(np.median(ret)) if n_closed else np.nan,
            win_rate=float((ret > 0).mean()) if n_closed else np.nan,
            excess_mean=float(exc.mean()) if n_closed else np.nan,
            exp_ret_per_signal=exp_ret,
            exp_excess_per_signal=exp_exc,
            t_cluster_ret=cluster_t(xs, clus),
            t_cluster_excess=cluster_t(xe, clus),
            held_days_mean=float(cl["held_days"].mean()) if n_closed else np.nan,
            n_no_fill=int(st.get("no_fill", 0)),
            n_incomplete=n_incomplete,
            n_dropped_limitup=int(st.get("dropped_limitup", 0)),
            n_dropped_no_quote=int(st.get("dropped_no_quote", 0)),
            n_dropped_cash=int(st.get("dropped_cash", 0)),
            n_dropped_no_next_day=int(st.get("dropped_no_next_day", 0)),
            fill_days_mean=float(
                (pd.to_datetime(cl["fill_date"]) - pd.to_datetime(cl["event_date"]))
                .dt.days.mean()) if n_closed else np.nan,
        ))
    summary = pd.DataFrame(sum_rows)
    # 判活:exp_ret > 0 且 聚类 t ≥ 2 且 比同信号同出场对照行(m=inf)高 ≥ +1pp
    base = summary[summary["m"] == "inf"].set_index(["signal", "exit_rule"])
    verdicts: list[dict] = []
    for r in sum_rows:
        if r["m"] == "inf":
            continue
        b = base.loc[(r["signal"], r["exit_rule"])]
        delta_pp = (r["exp_ret_per_signal"] - float(b["exp_ret_per_signal"])) * 100.0
        alive = (r["exp_ret_per_signal"] > 0
                 and np.isfinite(r["t_cluster_ret"]) and r["t_cluster_ret"] >= 2.0
                 and delta_pp >= 1.0)
        verdicts.append(dict(signal=r["signal"], m=r["m"], exit_rule=r["exit_rule"],
                             exp_ret=r["exp_ret_per_signal"],
                             baseline_exp_ret=float(b["exp_ret_per_signal"]),
                             delta_vs_baseline_pp=delta_pp,
                             t_cluster=r["t_cluster_ret"], alive=bool(alive)))
    summary["delta_vs_baseline_pp"] = summary.apply(
        lambda r: np.nan if r["m"] == "inf" else
        (r["exp_ret_per_signal"]
         - float(base.loc[(r["signal"], r["exit_rule"]), "exp_ret_per_signal"])) * 100.0,
        axis=1)
    summary["alive"] = summary.apply(
        lambda r: None if r["m"] == "inf" else bool(
            r["exp_ret_per_signal"] > 0 and np.isfinite(r["t_cluster_ret"])
            and r["t_cluster_ret"] >= 2.0 and r["delta_vs_baseline_pp"] >= 1.0), axis=1)
    sum_path = os.path.join(OUT_DIR, "summary_pullback.csv")
    summary.to_csv(sum_path, index=False, float_format="%.6f")
    log(f"summary_pullback.csv -> {sum_path}")

    n_alive = sum(1 for v in verdicts if v["alive"])
    log(f"判活:12 行中过线 {n_alive} 行;" + ("存在有戏行" if n_alive else "全部不过线 = 判死"))
    _write_verdict(gate, summary, verdicts, t_all)
    log(f"PULLBACK ENTRY DONE ({time.time() - t_all:.0f}s)")

    pd.set_option("display.width", 250)
    print("\n===== SUMMARY(配置为行;m=inf 为对照/基线行)=====")
    print(summary.to_string(index=False))
    print("\n===== VERDICTS(12 行判活)=====")
    print(pd.DataFrame(verdicts).to_string(index=False))


def _write_verdict(gate: dict, summary: pd.DataFrame | None,
                   verdicts: list[dict] | None, t_all: float) -> None:
    out = dict(
        experiment="pullback_entry(实验A:事件后回踩挂单入场)",
        prereadme="README.md 先于任何跑数落盘",
        gate=gate,
        verdicts=verdicts,
        n_alive=sum(1 for v in verdicts if v["alive"]) if verdicts else None,
        overall=("判死:12 行全部不过线,阴性结果原样交付" if verdicts is not None
                 and not any(v["alive"] for v in verdicts)
                 else "存在过线行" if verdicts is not None else "自检未过,未进入判读"),
        duration_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    path = os.path.join(OUT_DIR, "verdict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"verdict.json -> {path}")


if __name__ == "__main__":
    main()
