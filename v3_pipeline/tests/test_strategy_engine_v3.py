#!/usr/bin/env python3
"""strategy_engine_v3 单元测试（issue #28）。

合成小行情已知值测试：三类出场的触发/成交/顺延语义、T+1 与涨停拒买继承、
类 B 分档函数、类 C 候选集与意图执行，以及与 v1 冻结引擎的 E1 逐位对拍回归。
测试不触磁盘（MarketData 手工构造）；真实数据回归在网格运行脚本侧做抽样对拍。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))

import strategy_engine as se  # noqa: E402
import strategy_engine_v3 as se3  # noqa: E402

CAL = list(pd.date_range("2021-01-04", periods=30, freq="B"))  # 30 个交易日


def make_md(bars: dict[str, pd.DataFrame], limits: dict | None = None) -> se.MarketData:
    """bars: code -> DataFrame(index=date, cols open/high/low/close)。"""
    return se.MarketData(daily=bars, limits=limits or {}, calendar=CAL,
                         limit_missing_dates=0)


def make_events(rows: list[tuple[str, str, float]], with_sig_idx=False) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts_code", "event_date", "prob"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["event_id"] = [f"e{i}" for i in range(len(df))]
    if with_sig_idx:
        df["sig_idx"] = range(len(df))
    return df


def flat_bars(code_days: dict[str, list[int]], price=10.0) -> dict[str, pd.DataFrame]:
    """每只股票在给定日历日下标上放固定价 bar（open=high=low=close=price）。"""
    out = {}
    for code, days in code_days.items():
        idx = [CAL[i] for i in days]
        out[code] = pd.DataFrame(
            {"open": price, "high": price, "low": price, "close": price}, index=idx)
    return out


# ---------------------------------------------------------------- 类 A：固定止盈止损
class TestFixedTpSl:
    def test_tp_hit_at_barrier(self):
        # 事件日 CAL[0]，T+1=CAL[1] 开盘 10 买入；CAL[2] high 触及 +25% 屏障
        bars = flat_bars({"AAA": list(range(30))})
        b = bars["AAA"]
        b.iloc[1:] = 10.0
        b.loc[CAL[2], "high"] = 13.0  # >= 10*1.001*1.25 = 12.5125
        b.loc[CAL[2], "open"] = 10.0
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "tp"
        assert tr["exit_date"] == CAL[2]
        assert tr["exit_raw_price"] == pytest.approx(10.0 * 1.001 * 1.25)

    def test_tp_gap_open_exits_at_open(self):
        bars = flat_bars({"AAA": list(range(30))})
        bars["AAA"].loc[CAL[2], ["open", "high"]] = 13.0  # open 越过屏障
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "tp"
        assert tr["exit_raw_price"] == pytest.approx(13.0)

    def test_sl_hit_and_double_trigger_conservative(self):
        bars = flat_bars({"AAA": list(range(30))})
        # 同日高低双触及 -> 保守取止损
        bars["AAA"].loc[CAL[2], "high"] = 13.0
        bars["AAA"].loc[CAL[2], "low"] = 8.0
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "sl"
        assert tr["exit_raw_price"] == pytest.approx(10.0 * 1.001 * 0.86)

    def test_horizon_exit_at_close(self):
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.99, sl=-0.99, horizon=5)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        # 买入 CAL[1]（第 1 日），满 5 日 = CAL[5] 收盘卖
        assert tr["exit_reason"] == "horizon"
        assert tr["exit_date"] == CAL[5]
        assert tr["held_days"] == 5

    def test_t1_no_sell_on_entry_day(self):
        # 买入当日 high 即触及屏障，不可卖；次日才触发
        bars = flat_bars({"AAA": list(range(30))})
        bars["AAA"].loc[CAL[1], "high"] = 20.0  # 买入当日
        bars["AAA"].loc[CAL[2], "high"] = 20.0
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        assert tr["exit_date"] == CAL[2]

    def test_limit_down_deferral(self):
        bars = flat_bars({"AAA": list(range(30))})
        bars["AAA"].loc[CAL[2], "high"] = 13.0  # tp 触发
        bars["AAA"].loc[CAL[2], "close"] = 8.5  # 但收盘跌停
        bars["AAA"].loc[CAL[3], "high"] = 13.0  # 顺延后次日仍触及 -> 成交
        md = make_md(bars, limits={
            CAL[2]: pd.DataFrame({"up_limit": 15.0, "down_limit": 8.5},
                                 index=pd.Index(["AAA"], name="ts_code")),
        })
        ev = make_events([("AAA", "2021-01-04", 0.9)])
        spec = se3.ExitSpec.fixed_tp_sl(tp=0.25, sl=-0.14, horizon=16)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        tr = res["trades"].iloc[0]
        assert tr["exit_date"] == CAL[3]  # 顺延一日
        assert tr["deferred_days"] == 1
        assert res["stats"]["deferred_exits"] == 1


# ---------------------------------------------------------------- 类 B：波动率自适应
class TestVolAdaptive:
    def test_vol_band_thresholds(self):
        spec = se3.ExitSpec.vol_adaptive(tp=0.25, sl=-0.14, horizon=16, vol_lookback=21,
                                         vol_high_thresh=1.8, vol_low_thresh=0.6,
                                         vol_profit_mult=1.5, vol_stop_mult=1.1,
                                         low_vol_profit_mult=1.0)
        assert se3.vol_band(2.0, spec) == (0.375, pytest.approx(-0.154))
        assert se3.vol_band(0.5, spec) == (0.25, -0.14)
        assert se3.vol_band(1.0, spec) == (0.25, -0.14)
        assert se3.vol_band(None, spec) == (0.25, -0.14)  # 缺失回落中档
        assert se3.vol_band(1.8, spec)[0] == 0.375        # 高端点含
        assert se3.vol_band(0.6, spec)[0] == 0.25         # 低端点含

    def test_atr_uses_prior_day_only(self):
        # 引擎内 _band_at 用 <=t-1 的 ATR：前 25 日温和波动（ATR=0.1），t 日极端行情。
        # 市场 ATR 固定 0.2 -> 按 t-1 口径 vol_mult=0.5（低波档，tp 屏障 12.5125）；
        # 若误用 t 日 ATR（剧震 TR=7 -> ATR≈0.429，vol_mult≈2.14 -> 高波档，
        # tp 屏障 13.7638），成交价不同可被区分。
        n = 40
        cal = list(pd.date_range("2021-01-04", periods=n, freq="B"))
        close = np.full(n, 10.0)
        high = np.full(n, 10.05)
        low = np.full(n, 9.95)
        open_ = np.full(n, 10.0)
        high[25] = 16.0   # 触及两档 tp 屏障；低档 12.5125 成交 vs 高档 13.7638 成交
        low[25] = 9.0     # 不触及任何 sl 屏障（8.6086 / 8.4685）
        bars = {"AAA": pd.DataFrame({"open": open_, "high": high, "low": low,
                                     "close": close}, index=cal)}
        md = se.MarketData(daily=bars, limits={}, calendar=cal, limit_missing_dates=0)
        ev = make_events([("AAA", "2021-01-04", 0.9)])  # 事件 cal[0]，入场 cal[1]
        spec = se3.ExitSpec.vol_adaptive(tp=0.25, sl=-0.14, horizon=30, vol_lookback=21,
                                         vol_high_thresh=1.8, vol_low_thresh=0.6,
                                         vol_profit_mult=1.5, vol_stop_mult=1.1,
                                         low_vol_profit_mult=1.0)
        stock_atr = {"AAA": se3.atr_series(bars["AAA"], 21)}
        mkt_atr = pd.Series(0.2, index=cal)  # 恒为温和期个股 ATR(0.1) 的 2 倍
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec,
                                  mkt_atr=mkt_atr, stock_atr=stock_atr)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "tp"
        assert tr["exit_date"] == cal[25]
        assert tr["exit_raw_price"] == pytest.approx(10.0 * 1.001 * 1.25)


# ---------------------------------------------------------------- 类 C：分数衰减退出
class TestScoreDecay:
    def _panel(self, rows: list[tuple[str, str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["ts_code", "date", "prob"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_rank_out_exit_next_open(self):
        # AAA 买入后，CAL[3] 出现 5 个更高分新鲜信号 -> 跌出前 5 -> CAL[4] 开盘卖
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        for i, c in enumerate("BCDEF"):
            bars[c] = flat_bars({"X": list(range(30))})["X"]
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.50)] +
                         [(c, "2021-01-07", 0.90) for c in "BCDEF"])  # CAL[3]
        # 面板：AAA 持仓期分数恒定 0.50
        panel = self._panel([("AAA", d.strftime("%Y-%m-%d"), 0.50) for d in CAL])
        spec = se3.ExitSpec.score_decay(horizon=20, top_k=5, score_margin=0.0)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "rank_out"
        assert tr["exit_date"] == CAL[4]            # 意图 CAL[3] 收盘产生，次日开盘执行
        assert tr["exit_raw_price"] == pytest.approx(10.0)

    def test_score_drop_exit(self):
        # 无新鲜信号竞争，AAA 分数跌破买入阈值 -> score_drop
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.80)])
        panel = self._panel(
            [("AAA", "2021-01-05", 0.80)] +          # 买入当日不评估
            [("AAA", d.strftime("%Y-%m-%d"), 0.70) for d in CAL[2:]])  # < 0.80
        spec = se3.ExitSpec.score_decay(horizon=20, top_k=5, score_margin=0.0)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "score_drop"
        assert tr["exit_date"] == CAL[3]            # CAL[2] 触发，CAL[3] 开盘卖

    def test_margin_relaxes_threshold(self):
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.80)])
        panel = self._panel([("AAA", d.strftime("%Y-%m-%d"), 0.77) for d in CAL])
        # margin=0.05 -> 阈值 0.76，0.77 不跌破 -> 兜底 horizon 出场
        spec = se3.ExitSpec.score_decay(horizon=6, top_k=5, score_margin=0.05)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "horizon"
        assert tr["held_days"] == 6

    def test_horizon_backstop_when_score_alive(self):
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.80)])
        panel = self._panel([("AAA", d.strftime("%Y-%m-%d"), 0.90) for d in CAL])
        spec = se3.ExitSpec.score_decay(horizon=8, top_k=5, score_margin=0.0)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "horizon"
        assert tr["exit_date"] == CAL[8]            # 买入 CAL[1]，满 8 日收盘卖

    def test_intent_deferred_on_limit_down(self):
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        bars["AAA"].loc[CAL[3], "close"] = 8.5      # 意图执行日跌停 -> 顺延
        md = make_md(bars, limits={
            CAL[3]: pd.DataFrame({"up_limit": 15.0, "down_limit": 8.5},
                                 index=pd.Index(["AAA"], name="ts_code")),
        })
        ev = make_events([("AAA", "2021-01-04", 0.80)])
        panel = self._panel([("AAA", d.strftime("%Y-%m-%d"), 0.70) for d in CAL])
        spec = se3.ExitSpec.score_decay(horizon=20, top_k=5, score_margin=0.0)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "score_drop"
        assert tr["exit_date"] == CAL[4]
        assert tr["deferred_days"] == 1

    def test_score_lookup_missing_holds(self):
        # 面板缺 (AAA, 某日)：当日不评估，持仓保留至兜底
        bars = flat_bars({"AAA": list(range(30))}, price=10.0)
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.80)])
        panel = self._panel([("AAA", d.strftime("%Y-%m-%d"), 0.90)
                             for d in CAL if d != CAL[2]])
        spec = se3.ExitSpec.score_decay(horizon=6, top_k=5, score_margin=0.0)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec, score_panel=panel)
        tr = res["trades"].iloc[0]
        assert tr["exit_reason"] == "horizon"
        assert res["stats"]["score_lookup_missing"] >= 1


# ---------------------------------------------------------------- 信号源与取舍
class TestScoreSignals:
    def test_score_priority_selection(self):
        bars = flat_bars({c: list(range(30)) for c in ["AAA", "BBB", "CCC"]})
        md = make_md(bars)
        ev = make_events([("AAA", "2021-01-04", 0.50),
                          ("BBB", "2021-01-04", 0.90),
                          ("CCC", "2021-01-04", 0.70)])
        spec = se3.ExitSpec.horizon_only(horizon=5)
        res = se3.run_backtest_v3(ev, md, n_slots=2, exit_spec=spec)
        entered = set(res["trades"]["ts_code"])
        assert entered == {"BBB", "CCC"}            # prob 降序取前 2

    def test_entry_limit_up_rejected(self):
        bars = flat_bars({"AAA": list(range(30))})
        bars["AAA"].loc[CAL[1], "open"] = 11.0      # T+1 开盘涨停
        md = make_md(bars, limits={
            CAL[1]: pd.DataFrame({"up_limit": 11.0, "down_limit": 9.0},
                                 index=pd.Index(["AAA"], name="ts_code")),
        })
        ev = make_events([("AAA", "2021-01-04", 0.90)])
        spec = se3.ExitSpec.horizon_only(horizon=5)
        res = se3.run_backtest_v3(ev, md, n_slots=3, exit_spec=spec)
        assert res["trades"].empty
        assert res["stats"]["dropped_limitup"] == 1


# ---------------------------------------------------------------- v1 回归：E1 逐位对拍
class TestV1Regression:
    def test_e1_matches_v1_bitwise(self):
        rng = np.random.RandomState(7)
        n = 60
        cal = list(pd.date_range("2021-03-01", periods=n, freq="B"))
        bars = {}
        for c in ["AAA", "BBB", "CCC", "DDD"]:
            close = 10 * np.cumprod(1 + rng.normal(0, 0.02, n))
            open_ = close * (1 + rng.normal(0, 0.005, n))
            high = np.maximum(open_, close) * 1.01
            low = np.minimum(open_, close) * 0.99
            bars[c] = pd.DataFrame({"open": open_, "high": high, "low": low,
                                    "close": close}, index=cal)
        md = se.MarketData(daily=bars, limits={}, calendar=cal, limit_missing_dates=0)
        rows = [("AAA", cal[0], 0.9), ("BBB", cal[1], 0.8), ("CCC", cal[2], 0.7),
                ("DDD", cal[2], 0.6), ("AAA", cal[10], 0.95), ("BBB", cal[20], 0.85)]
        ev3 = pd.DataFrame({"ts_code": [r[0] for r in rows],
                            "event_date": [r[1] for r in rows],
                            "prob": [r[2] for r in rows],
                            "sig_idx": range(len(rows))})
        # v1 需要 ATRN/RET20/sig_idx 列
        ev1 = ev3.copy()
        ev1["ATRN"] = 0.03
        ev1["RET20"] = -0.1
        r1 = se.run_backtest(ev1, md, n_slots=2, selection="S1", exit_rule="E1",
                             horizon=10, seed=42)
        r3 = se3.run_backtest_v3(ev3, md, n_slots=2,
                                 exit_spec=se3.ExitSpec.horizon_only(10),
                                 selection="sig_idx")
        cols = ["ts_code", "event_date", "entry_date", "entry_price", "shares",
                "exit_date", "exit_reason", "exit_raw_price", "exit_exec_price",
                "net_pnl", "held_days"]
        t1 = r1["trades"][cols].reset_index(drop=True)
        t3 = r3["trades"][cols].reset_index(drop=True)
        assert t1["exit_reason"].eq("E1").all()
        assert t3["exit_reason"].eq("horizon").all()
        pd.testing.assert_frame_equal(
            t1.drop(columns=["exit_reason"]), t3.drop(columns=["exit_reason"]),
            check_exact=True)
        # 权益曲线逐位一致
        pd.testing.assert_frame_equal(
            r1["equity"][["date", "cash", "market_value", "equity", "n_positions"]],
            r3["equity"][["date", "cash", "market_value", "equity", "n_positions"]],
            check_exact=True)
