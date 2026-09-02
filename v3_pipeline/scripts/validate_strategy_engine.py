"""strategy_engine 独立校验脚本 —— issue #18 冻结前验收。

四项校验（全过才冻结）：
 1. 手工复算 >= 3 笔交易（E1/E2/E3 至少各 1 笔）：用原始日线/涨跌停表独立重算
    成交价/成本/损益，逐分对账（容差 0.01 元）；并对小规模实跑的全部交易做完整重放。
 2. 无泄漏审计：(a) 所有买入日 > 事件日；(b) E2/E3 触发判断只用持有期内行情；
    (c) 取舍特征日期 == 事件日 T。
 3. 合成边界测试：一字涨停拒买、跌停顺延卖出、满仓信号跳过、无行情放弃、现金不足放弃。
 4. 冒烟：主池 val 段（2019-01~2022-10）S0/E1(H=20)/N=10 跑通出曲线，
    事件覆盖率与资金利用率写入日志。
附：S0 固定 seed 可复现性检查（同配置连跑两次交易明细必须一致）。

产物：v3_pipeline/reports/strategy_real_trading/validation_report.md（每句一行）
     v3_pipeline/reports/strategy_real_trading/smoke_S0_E1_H20_N10/{equity_curve,trades}.parquet
退出码：全部通过 0，任一失败 1。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_engine as se  # noqa: E402

REPO = "/home/karl/repos/personal/stock_qt_nd"
ART = os.path.join(REPO, "v3_pipeline/reports/strategy_real_trading")
LOG = os.path.join(ART, "progress.log")
REPORT = os.path.join(ART, "validation_report.md")

CHECKS: list[tuple[str, bool, str]] = []   # (名称, 是否通过, 细节)
REPORT_LINES: list[str] = []


def log(msg: str) -> None:
    se._log(LOG, msg)


def record(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    log(f"CHECK [{'PASS' if passed else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- 独立数据访问（不经引擎）
class RawData:
    """校验方独立读取原始 parquet，避免与引擎共用加载逻辑。"""

    def __init__(self) -> None:
        idx = pd.read_parquet(os.path.join(REPO, "stock_data/index/000905.SH.parquet"),
                              columns=["trade_date"])
        idx["trade_date"] = pd.to_datetime(idx["trade_date"])
        self.calendar = sorted(pd.Timestamp(d) for d in idx["trade_date"].unique())
        self._daily_cache: dict[str, pd.DataFrame] = {}
        self._limit_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}
        fm = pd.read_parquet(os.path.join(
            REPO, "v3_pipeline/reports/feature_matrix/main_pool_features.parquet"),
            columns=["ts_code", "date", "ATRN", "RET20", "sig_idx"])
        fm["date"] = pd.to_datetime(fm["date"])
        self.features = fm.set_index(["ts_code", "date"])

    def daily(self, code: str) -> pd.DataFrame:
        if code not in self._daily_cache:
            df = pd.read_parquet(os.path.join(REPO, "stock_data/daily", f"{code}.parquet"),
                                 columns=["trade_date", "open", "high", "low", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            self._daily_cache[code] = df.set_index("trade_date").sort_index()
        return self._daily_cache[code]

    def limits(self, day: pd.Timestamp) -> pd.DataFrame | None:
        if day not in self._limit_cache:
            fp = os.path.join(REPO, "stock_data/stk_limit", day.strftime("%Y%m%d") + ".parquet")
            self._limit_cache[day] = pd.read_parquet(fp).set_index("ts_code") \
                if os.path.exists(fp) else None
        return self._limit_cache[day]


def money_eq(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


def recompute_trade(tr: pd.Series, exit_rule: str, horizon: int, raw: RawData) -> dict:
    """对一笔交易做完全独立的重放复算。任何不一致抛 AssertionError。"""
    code = tr["ts_code"]
    d_event = pd.Timestamp(tr["event_date"])
    d_entry = pd.Timestamp(tr["entry_date"])
    d_exit = pd.Timestamp(tr["exit_date"])
    df = raw.daily(code)

    # ---- 入场日：事件日之后的首个交易所交易日（无泄漏口径）----
    ci = raw.calendar.index(d_event)
    exp_entry_day = raw.calendar[ci + 1]
    assert d_entry == exp_entry_day, f"entry {d_entry} != next trading day {exp_entry_day}"
    assert d_event < d_entry < d_exit or (d_event < d_entry and d_entry <= d_exit)
    bar = df.loc[d_entry]
    exp_entry = float(bar["open"]) * (1.0 + se.SLIPPAGE)
    assert abs(tr["entry_price"] - exp_entry) < 1e-6 * exp_entry, \
        f"entry price {tr['entry_price']} != {exp_entry}"
    lim = raw.limits(d_entry)
    if lim is not None and code in lim.index:
        assert float(bar["open"]) < float(lim.loc[code, "up_limit"]) - se.PRICE_TOL, \
            "entry day open >= up_limit should have been rejected"
    assert tr["shares"] % se.BOARD_LOT == 0 and tr["shares"] > 0
    exp_comm = max(se.COMMISSION_MIN, tr["shares"] * exp_entry * se.COMMISSION_RATE)
    assert money_eq(tr["entry_commission"], exp_comm), "entry commission mismatch"

    # ---- ATR 与屏障：ATRN(事件日特征) × T 日收盘（特征日期==T 的独立核验）----
    atrn = float(raw.features.loc[(code, d_event), "ATRN"])
    t_close = float(df.loc[d_event, "close"])
    atr = atrn * t_close
    assert money_eq(tr["atr"], atr, tol=1e-4), f"atr {tr['atr']} != {atr}"
    tp = exp_entry + se.TP_ATR_MULT * atr
    sl = exp_entry - se.SL_ATR_MULT * atr
    assert money_eq(tr["tp_barrier"], tp, tol=1e-4) and money_eq(tr["sl_barrier"], sl, tol=1e-4)

    # ---- 出场重放：只用 (entry_date, exit_date] 内行情 ----
    ei = raw.calendar.index(d_entry)
    found = None
    deferred = 0
    for day in raw.calendar[ei + 1:]:
        if day > d_exit:  # 不允许越过引擎记录的退出日
            break
        if day not in df.index:
            continue
        b = df.loc[day]
        o, h, l, c = (float(b[k]) for k in ("open", "high", "low", "close"))
        held = raw.calendar.index(day) - ei + 1
        raw_px, reason = None, None
        if exit_rule == "E2" and h >= tp - se.PRICE_TOL:
            raw_px, reason = (o if o >= tp - se.PRICE_TOL else tp), "E2"
        elif exit_rule == "E3" and l <= sl + se.PRICE_TOL:
            raw_px, reason = (o if o <= sl + se.PRICE_TOL else sl), "E3"
        elif held >= horizon:
            raw_px, reason = c, "E1"
        if raw_px is None:
            continue
        lim = raw.limits(day)
        if lim is not None and code in lim.index and \
                c <= float(lim.loc[code, "down_limit"]) + se.PRICE_TOL:
            deferred += 1
            continue
        found = (day, raw_px, reason, held, deferred)
        break
    assert found is not None, f"replay found no exit for {code} event {d_event}"
    day, raw_px, reason, held, deferred = found
    assert day == d_exit, f"exit day {day} != {d_exit}"
    assert reason == tr["exit_reason"], f"reason {reason} != {tr['exit_reason']}"
    assert abs(raw_px - tr["exit_raw_price"]) < 1e-6 * raw_px, "raw exit price mismatch"
    assert held == tr["held_days"], f"held_days {held} != {tr['held_days']}"
    assert deferred == tr["deferred_days"], "deferred_days mismatch"

    exp_exec = raw_px * (1.0 - se.SLIPPAGE)
    assert money_eq(tr["exit_exec_price"], exp_exec), "exit exec price mismatch"
    gross_amt = tr["shares"] * exp_exec
    exp_comm2 = max(se.COMMISSION_MIN, gross_amt * se.COMMISSION_RATE)
    exp_stamp = gross_amt * (se.STAMP_TAX_OLD if day < se.STAMP_TAX_SWITCH else se.STAMP_TAX_NEW)
    assert money_eq(tr["exit_commission"], exp_comm2), "exit commission mismatch"
    assert money_eq(tr["stamp_tax"], exp_stamp), "stamp tax mismatch"
    exp_gross_pnl = tr["shares"] * (exp_exec - exp_entry)
    exp_cost = exp_comm + exp_comm2 + exp_stamp
    assert money_eq(tr["gross_pnl"], exp_gross_pnl), "gross pnl mismatch"
    assert money_eq(tr["total_cost"], exp_cost), "total cost mismatch"
    assert money_eq(tr["net_pnl"], exp_gross_pnl - exp_cost), "net pnl mismatch"
    return dict(entry=exp_entry, shares=int(tr["shares"]), exit=exp_exec,
                reason=reason, held=held, deferred=deferred,
                net=exp_gross_pnl - exp_cost)


# ---------------------------------------------------------------- 合成边界测试
def _syn_md() -> tuple[se.MarketData, list[pd.Timestamp]]:
    days = [pd.Timestamp(d) for d in
            ("2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
             "2020-01-08", "2020-01-09", "2020-01-10")]

    def mk(rows: dict[pd.Timestamp, tuple[float, float, float, float]]) -> pd.DataFrame:
        df = pd.DataFrame(
            [(d, *v) for d, v in rows.items()],
            columns=["trade_date", "open", "high", "low", "close"],
        ).set_index("trade_date").sort_index()
        return df

    d1, d2, d3, d4 = days[0], days[1], days[2], days[3]
    daily = {
        # AAA：d2 开盘 == 涨停（一字涨停拒买）
        "AAA": mk({d1: (10.0, 10.2, 9.8, 10.0), d2: (11.0, 11.0, 11.0, 11.0),
                   d3: (11.0, 11.2, 10.8, 11.0), d4: (11.0, 11.1, 10.9, 11.0)}),
        # BBB：正常买入，d3 收盘 <= 跌停（顺延），d4 恢复可卖
        "BBB": mk({d1: (10.0, 10.2, 9.8, 10.0), d2: (10.0, 10.1, 9.6, 9.5),
                   d3: (9.0, 9.1, 8.5, 8.5), d4: (8.8, 9.2, 8.7, 9.0),
                   days[4]: (9.0, 9.1, 8.9, 9.0)}),
        # CCC：正常买入、正常 E1 退出
        "CCC": mk({d1: (20.0, 20.2, 19.8, 20.0), d2: (20.0, 20.5, 19.9, 20.2),
                   d3: (20.5, 21.2, 20.4, 21.0), d4: (21.0, 21.3, 20.8, 21.1),
                   days[4]: (21.1, 21.2, 21.0, 21.1)}),
        # DDD：d2 无行情（停牌），入场应放弃
        "DDD": mk({d1: (5.0, 5.1, 4.9, 5.0), d3: (5.0, 5.1, 4.9, 5.0)}),
    }
    lim_rows = {
        d1: {"AAA": (11.0, 9.0), "BBB": (11.0, 9.0), "CCC": (22.0, 18.0), "DDD": (5.5, 4.5)},
        d2: {"AAA": (11.0, 9.0), "BBB": (11.0, 9.0), "CCC": (22.0, 18.0), "DDD": (5.5, 4.5)},
        d3: {"AAA": (12.1, 9.9), "BBB": (10.45, 8.55), "CCC": (22.22, 18.18), "DDD": (5.5, 4.5)},
        d4: {"AAA": (12.1, 9.9), "BBB": (9.35, 7.65), "CCC": (23.1, 18.9), "DDD": (5.5, 4.5)},
        days[4]: {"BBB": (9.9, 8.1), "CCC": (23.21, 18.99)},
        days[5]: {"BBB": (9.9, 8.1), "CCC": (23.21, 18.99)},
        days[6]: {"BBB": (9.9, 8.1), "CCC": (23.21, 18.99)},
    }
    limits = {d: pd.DataFrame([(k, *v) for k, v in rows.items()],
                              columns=["ts_code", "up_limit", "down_limit"]
                              ).set_index("ts_code")
              for d, rows in lim_rows.items()}
    return se.MarketData(daily=daily, limits=limits, calendar=days), days


def _syn_events(rows: list[tuple[str, int]]) -> pd.DataFrame:
    d1 = pd.Timestamp("2020-01-02")
    return pd.DataFrame(
        [(c, d1, 0.05, -0.10, float(i)) for c, i in rows],
        columns=["ts_code", "event_date", "ATRN", "RET20", "sig_idx"],
    )


def test_synthetic() -> None:
    md, days = _syn_md()
    d2, d3, d4 = days[1], days[2], days[3]

    # 场景 A：S1 取前 2（DDD、AAA）；DDD 无行情放弃、AAA 涨停放弃；BBB/CCC 满仓跳过
    ev = _syn_events([("DDD", 0), ("AAA", 1), ("BBB", 2), ("CCC", 3)])
    r = se.run_backtest(ev, md, n_slots=2, selection="S1", exit_rule="E1", horizon=2)
    s = r["stats"]
    assert len(r["trades"]) == 0, "scenario A should have no trades"
    assert s["dropped_no_quote"] == 1 and s["dropped_limitup"] == 1, \
        f"scenario A drops wrong: {s}"
    assert s["dropped_slot_full"] == 2, f"scenario A slot-full count wrong: {s}"
    assert (r["equity"]["equity"] == se.INIT_CAPITAL).all(), "cash untouched expected"

    # 场景 B：BBB+CCC 入场；BBB 卖出日跌停顺延一日；CCC 按期 E1 退出；T+1 不可卖
    ev = _syn_events([("BBB", 1), ("CCC", 2)])
    r = se.run_backtest(ev, md, n_slots=2, selection="S1", exit_rule="E1", horizon=2)
    tr = r["trades"].set_index("ts_code")
    assert set(tr.index) == {"BBB", "CCC"}, f"scenario B trades wrong: {tr.index.tolist()}"
    b, c = tr.loc["BBB"], tr.loc["CCC"]
    assert b["entry_date"] == d2 and b["exit_date"] == d4 and b["exit_reason"] == "E1", \
        f"BBB deferral wrong: {b[['entry_date','exit_date','exit_reason']].to_dict()}"
    assert b["deferred_days"] == 1 and abs(b["exit_raw_price"] - 9.0) < 1e-9
    assert c["entry_date"] == d2 and c["exit_date"] == d3 and c["exit_reason"] == "E1"
    assert (r["trades"]["exit_date"] > r["trades"]["entry_date"]).all(), "T+1 sell rule violated"
    exp_exec = 21.0 * (1 - se.SLIPPAGE)
    assert money_eq(c["exit_exec_price"], exp_exec), "CCC exec price wrong"

    # 场景 C：现金不足一手 -> 放弃
    ev = _syn_events([("BBB", 1)])
    r = se.run_backtest(ev, md, n_slots=1, selection="S1", exit_rule="E1", horizon=2,
                        init_capital=500.0)
    assert r["stats"]["dropped_cash"] == 1 and len(r["trades"]) == 0, \
        f"scenario C wrong: {r['stats']}"


# ---------------------------------------------------------------- 主流程
def main() -> int:
    t_start = time.time()
    log("validate_strategy_engine START")

    # ---------- 合成边界测试（校验 3）----------
    try:
        test_synthetic()
        record("synthetic_edge_tests", True,
               "一字涨停拒买/跌停顺延/满仓跳过/无行情放弃/现金不足 全部符合规格")
    except AssertionError as e:
        record("synthetic_edge_tests", False, str(e))

    # ---------- 小规模实跑（校验 1/2 素材）：2019 全年 E1/E2/E3 ----------
    events_2019 = se.load_events("main", "2019-01-01", "2019-12-31", REPO, LOG)
    md2019 = se.load_market_data(events_2019["ts_code"].unique().tolist(),
                                 "2019-01-01", "2019-12-31", REPO, log_path=LOG)
    events_2019 = events_2019[events_2019["event_date"].isin(set(md2019.calendar))]
    runs = {}
    for rule in ("E1", "E2", "E3"):
        runs[rule] = se.run_backtest(events_2019, md2019, n_slots=10, selection="S0",
                                     exit_rule=rule, horizon=20, seed=42, log_path=LOG)

    # S0 seed 可复现性
    r_again = se.run_backtest(events_2019, md2019, n_slots=10, selection="S0",
                              exit_rule="E1", horizon=20, seed=42)
    rep_ok = runs["E1"]["trades"].equals(r_again["trades"]) and \
        runs["E1"]["equity"].equals(r_again["equity"])
    record("s0_seed_reproducible", rep_ok, "同 seed 连跑两次交易明细与资金曲线完全一致")

    raw = RawData()

    # ---------- 校验 1：手工复算（抽样 3 笔逐分对账 + 全量重放）----------
    try:
        sample_details = []
        for rule in ("E1", "E2", "E3"):
            tr_df = runs[rule]["trades"]
            pick = tr_df[tr_df["exit_reason"] == rule]
            assert not pick.empty, f"no {rule} trade to sample in 2019 run"
            rec = recompute_trade(pick.iloc[0], rule, 20, raw)
            sample_details.append(
                f"{rule} 样本 {pick.iloc[0]['ts_code']} 入场 {rec['entry']:.3f} x {rec['shares']}股 "
                f"出场 {rec['exit']:.3f}（{rec['reason']}，持有 {rec['held']} 日，顺延 {rec['deferred']} 日）"
                f"净利 {rec['net']:.2f} 元 —— 逐分对账一致")
        n_total = 0
        for rule in ("E1", "E2", "E3"):
            for _, tr in runs[rule]["trades"].iterrows():
                recompute_trade(tr, rule, 20, raw)
                n_total += 1
        record("hand_recalc", True,
               f"3 笔抽样（E1/E2/E3 各 1）逐分对账通过；{n_total} 笔全量重放全部一致")
        REPORT_LINES.extend(sample_details)
    except AssertionError as e:
        record("hand_recalc", False, str(e))

    # ---------- 校验 2：无泄漏审计 ----------
    try:
        # (a) 所有买入日 > 事件日
        for rule, r in runs.items():
            tr = r["trades"]
            assert (tr["entry_date"] > tr["event_date"]).all(), f"{rule}: entry <= event"
            assert (tr["exit_date"] > tr["entry_date"]).all(), f"{rule}: exit <= entry (T+1)"
        # (b) E2/E3 触发判断只用持有期内行情：全量重放（校验 1 已逐笔重建触发日行情，
        #     且重放窗口硬限制在 (entry_date, exit_date]）在此显式复验触发日屏障关系
        n_barrier = 0
        for rule in ("E2", "E3"):
            for _, tr in runs[rule]["trades"].iterrows():
                if tr["exit_reason"] != rule:
                    continue
                b = raw.daily(tr["ts_code"]).loc[tr["exit_date"]]
                if rule == "E2":
                    assert float(b["high"]) >= tr["tp_barrier"] - se.PRICE_TOL
                else:
                    assert float(b["low"]) <= tr["sl_barrier"] + se.PRICE_TOL
                assert tr["entry_date"] < tr["exit_date"]
                n_barrier += 1
        # (c) 取舍特征日期 == T：独立重合并，逐事件核对
        ex = pd.read_parquet(os.path.join(
            REPO, "v3_pipeline/reports/pool_cleaning/excluded_events_main.parquet"),
            columns=["ts_code", "date", "f_any"])
        ev = ex.loc[~ex["f_any"], ["ts_code", "date"]]
        ev["date"] = pd.to_datetime(ev["date"])
        fm = raw.features.reset_index()
        m = ev.merge(fm, on=["ts_code", "date"], how="left", validate="one_to_one")
        assert m[["ATRN", "RET20", "sig_idx"]].notna().all().all(), "feature date != event date"
        eng = se.load_events("main", "2019-01-01", "2019-12-31", REPO)
        assert len(eng) == len(m[(m["date"] >= "2019-01-01") & (m["date"] <= "2019-12-31")])
        record("leakage_audit", True,
               f"(a) 买入日>事件日 全过；(b) {n_barrier} 笔屏障触发只用持有期内行情；"
               f"(c) 特征日期==事件日 独立重合并一致")
    except AssertionError as e:
        record("leakage_audit", False, str(e))

    # ---------- 校验 4：冒烟（主池 val 段 S0/E1(H=20)/N=10）----------
    smoke_dir = os.path.join(ART, "smoke_S0_E1_H20_N10")
    try:
        smoke = se.run_config(pool="main", selection="S0", exit_rule="E1", horizon=20,
                              n_slots=10, start="2019-01-01", end="2022-10-31", seed=42,
                              repo_root=REPO, out_dir=smoke_dir, log_path=LOG)
        eq, tr, st = smoke["equity"], smoke["trades"], smoke["stats"]
        assert len(eq) > 500 and len(tr) > 100, "smoke output too thin"
        assert eq["equity"].notna().all() and (eq["equity"] > 0).all()
        log(f"SMOKE coverage={st['coverage']:.4f} utilization={st['capital_utilization']:.4f} "
            f"equity_head={eq['equity'].iloc[0]:.2f} equity_tail={eq['equity'].iloc[-1]:.2f} "
            f"trades={len(tr)} limit_missing_days={st['limit_missing_days']}")
        record("smoke_main_val_S0_E1_H20_N10", True,
               f"覆盖率={st['coverage']:.4f} 利用率={st['capital_utilization']:.4f} "
               f"曲线 {eq['equity'].iloc[0]:.0f} -> {eq['equity'].iloc[-1]:.0f} "
               f"交易 {len(tr)} 笔 缺涨跌停表 {st['limit_missing_days']} 天")
    except (AssertionError, Exception) as e:  # noqa: BLE001
        record("smoke_main_val_S0_E1_H20_N10", False, repr(e))

    # ---------- 汇总报告 ----------
    all_pass = all(ok for _, ok, _ in CHECKS)
    lines = ["# strategy_engine 校验报告（issue #18 冻结验收）", ""]
    lines.append(f"校验时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，总耗时 {time.time() - t_start:.1f} 秒。")
    lines.append(f"总体结论：{'全部通过，引擎冻结。' if all_pass else '存在失败项，禁止冻结。'}")
    lines.append("")
    for name, ok, detail in CHECKS:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}：{detail}")
    if REPORT_LINES:
        lines.append("")
        lines.append("## 手工复算抽样明细（逐分对账，容差 0.01 元）")
        lines.extend(f"- {s}" for s in REPORT_LINES)
    lines.append("")
    lines.append("## 口径备注")
    lines.append("- 单仓预算 = 信号日 T 收盘权益 / N，T+1 开盘买入，股数向下取整到 100 股。")
    lines.append("- 屏障价以含滑点入场执行价为基准，ATR = 事件日特征 ATRN × T 日收盘价。")
    lines.append("- 卖出日收盘价 <= 跌停价即视为不可卖并顺延，缺涨跌停表的日期视为无约束并计入 warning。")
    lines.append("- E4 同日止盈止损双触发时保守取止损。")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"validate_strategy_engine DONE all_pass={all_pass} "
        f"elapsed={time.time() - t_start:.1f}s report={REPORT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
