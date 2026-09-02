#!/usr/bin/env python3
"""divergence_stability_check 的单元测试。

接缝: 背离检测配置 -> 事件表。只验外部行为:
  - 前缀稳定性: 合成数据上, 截断重跑与全量跑在截断窗口内的事件完全一致;
  - regime 截断等价: 全量 regime 的前缀切片 == 对截断后股票群重算 regime;
  - 截断网格划分: 网格窗口不重不漏地覆盖全部事件日;
  - 重合率指标: 事件集合交集数学正确。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import divergence_lab as lab  # noqa: E402
import divergence_stability_check as dsc  # noqa: E402


def _cfg_fractal():
    return lab.load_config(str(SCRIPTS.parents[1] /
                             "v3_pipeline/configs/divergence_lab/m_scan/m_fractal15_full.json"))


def _cfg_zigzag():
    return lab.load_config(str(SCRIPTS.parents[1] /
                             "v3_pipeline/configs/divergence_lab/m_scan/m_zigzag05_nofilter.json"))


def _random_walk(n, seed, start=20.0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.02, n)
    return start * np.cumprod(1.0 + ret)


# ---------------------------------------------------------------- 前缀稳定性(合成)
@pytest.mark.parametrize("cfg_fn", [_cfg_fractal, _cfg_zigzag])
def test_prefix_stability_synthetic(cfg_fn):
    cfg = cfg_fn()
    n = 800
    close = _random_walk(n, seed=7)
    vol = np.random.default_rng(8).uniform(500, 2000, n)
    # 合成日序: 连续自然日即可(检测只看相对位置)
    dates = (np.arange(n) + 19000).astype(np.int32)

    full = dsc.detect_events_from_arrays(close, vol, cfg)
    grid = dsc.truncation_grid(dates, step=100)
    assert grid[-1] == dates[-1]

    mismatches = dsc.prefix_mismatches(full, dates, lambda t: dsc.detect_events_from_arrays(
        close[: t + 1], vol[: t + 1], cfg), grid)
    assert mismatches == []


def test_detect_events_matches_lab_load_stock(tmp_path):
    """detect_events_from_arrays 必须与 divergence_lab.load_stock 的检测逻辑一致。"""
    n = 400
    close = _random_walk(n, seed=11)
    vol = np.random.default_rng(12).uniform(500, 2000, n)
    pd_dates = (np.datetime64("2020-01-01") + np.arange(n)).astype("datetime64[D]")
    df = pd.DataFrame({"trade_date": pd_dates, "close": close, "vol": vol,
                       "open": close, "high": close, "low": close})
    fp = tmp_path / "SYNTH.parquet"
    df.to_parquet(fp, index=False)

    cfg = _cfg_fractal()
    ref = lab.load_stock((str(fp), cfg))
    assert "error" not in ref
    got = dsc.detect_events_from_arrays(close, vol, cfg)
    for key in ("sig", "low", "prev", "rank", "form"):
        np.testing.assert_array_equal(got[key], ref["events"][key])


# ---------------------------------------------------------------- regime 截断等价
def test_regime_truncation_equiv():
    # 5 只合成股, 其中一只上市晚(前半段无数据), 一只提前退市(后半段无数据)
    window, th = 20, 0.05
    base_dates = (np.datetime64("2018-01-01") + np.arange(300)).astype("datetime64[D]") \
        .astype(np.int32)
    stocks = []
    for i in range(5):
        lo, hi = 0, 300
        if i == 3:
            lo = 150  # 晚上市
        if i == 4:
            hi = 150  # 早退市
        d = base_dates[lo:hi]
        close = _random_walk(hi - lo, seed=100 + i)
        cf = close / close[0]
        stocks.append({"dates": d, "cf": cf})
    all_dates, reg_full = lab.build_market_regime(stocks, window, th)

    t_cut = base_dates[199]  # 截断到第 200 天
    trunc = [{"dates": s["dates"][s["dates"] <= t_cut],
              "cf": s["cf"][s["dates"] <= t_cut]} for s in stocks]
    trunc = [s for s in trunc if len(s["dates"]) > 1]
    all_dates_t, reg_t = lab.build_market_regime(trunc, window, th)

    # 在共享日期轴上逐日比较(截断轴不含 >t_cut 的日期)
    pos_full = np.searchsorted(all_dates, all_dates_t)
    assert np.array_equal(all_dates[pos_full], all_dates_t)
    np.testing.assert_array_equal(reg_full[pos_full], reg_t)


# ---------------------------------------------------------------- 截断网格
def test_prefix_end_with_suspension_gap():
    """个股停牌导致 T 不在其交易日轴上时, 前缀必须回退到最后一个 <= T 的交易日。"""
    dates = np.array([100, 101, 102, 105, 106], np.int32)  # 103/104 停牌
    assert dsc.prefix_end(dates, 102) == 2
    assert dsc.prefix_end(dates, 103) == 2  # 停牌日回退, 不纳入 105
    assert dsc.prefix_end(dates, 104) == 2
    assert dsc.prefix_end(dates, 105) == 3
    assert dsc.prefix_end(dates, 99) == -1  # 上市前


def test_truncation_grid_covers_all():
    dates = (np.datetime64("2019-01-02") + np.arange(500)).astype("datetime64[D]") \
        .astype(np.int32)
    grid = dsc.truncation_grid(dates, step=63)
    assert grid[0] < dates[0] + 63 + 1
    assert grid[-1] == dates[-1]
    assert np.all(np.diff(grid) > 0)
    # 窗口划分不重不漏: 每个日期恰好落入一个 (prev, T] 窗口
    for d in (dates[0], dates[250], dates[-1]):
        hits = sum(1 for i in range(1, len(grid)) if grid[i - 1] < d <= grid[i])
        first = 1 if d <= grid[0] else 0
        assert hits + first == 1


# ---------------------------------------------------------------- 重合率指标
def test_overlap_metrics():
    base = [("A", 10), ("B", 20), ("C", 30)]
    pert = [("A", 10), ("C", 30), ("D", 40), ("E", 50)]
    m = dsc.overlap_metrics(base, pert)
    assert m["n_base"] == 3 and m["n_pert"] == 4 and m["n_inter"] == 2
    assert m["recall_base"] == pytest.approx(2 / 3)
    assert m["recall_pert"] == pytest.approx(2 / 4)
