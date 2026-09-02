#!/usr/bin/env python3
"""divergence_lab 因果性与正确性测试。

硬性要求: 对 3 只真实股票, 用暴力逐日截断法(每天 t 只用 close[:t+1] 重跑检测,
收集信号日恰好为 t 的事件)与库实现的全量向量化/流式结果全量对拍, 0 mismatch。
覆盖 fractal 与 zigzag 两种新识别法; legacy 与 V1 因果模拟函数全量对拍。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba import njit

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import divergence_event_study as des  # noqa: E402
import divergence_lab as lab  # noqa: E402

REPO = SCRIPTS.parents[1]
DATA = REPO / "stock_data" / "daily"
TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]


def _close(symbol):
    df = pd.read_parquet(DATA / f"{symbol}.parquet")
    df = df.dropna(subset=["close"]).drop_duplicates("trade_date").sort_values("trade_date")
    return df["close"].to_numpy(np.float64)


# ---------------------------------------------------------------- 暴力截断实现(独立代码路径)
def brute_fractal_sigdays(p, order):
    """逐日截断: 第 t 日信号 <=> t-order 是 p[t-2k .. t] 的严格最小值。"""
    n = len(p)
    hits = []
    for t in range(2 * order, n):
        w = p[t - 2 * order: t + 1]
        c = w[order]
        if np.all(c < np.concatenate([w[:order], w[order + 1:]])):
            hits.append(t)
    return set(hits)


@njit(cache=True)
def _brute_zigzag_prefix(p, pct):
    """独立流式实现: 在前缀 p 上重跑 zigzag, 返回最后一个被确认低点的确认日(无则 -1)。
    与库实现逐日对拍: 第 t 日有信号 <=> 在 p[:t+1] 上最后一次确认恰好发生在 t。"""
    n = len(p)
    hi, lo = p[0], p[0]
    hi_i, lo_i = 0, 0
    state = 0  # 0=未定向, 1=上行, 2=下行
    last_confirm = -1
    for i in range(1, n):
        x = p[i]
        if state == 0:
            if x < lo:
                lo, lo_i = x, i
            if x > hi:
                hi, hi_i = x, i
            if lo_i < i and x >= lo * (1.0 + pct):
                last_confirm = i
                state = 1
                hi, hi_i = x, i
            elif hi_i < i and x <= hi * (1.0 - pct):
                state = 2
                lo, lo_i = x, i
        elif state == 1:
            if x > hi:
                hi, hi_i = x, i
            elif hi_i < i and x <= hi * (1.0 - pct):
                state = 2
                lo, lo_i = x, i
        else:
            if x < lo:
                lo, lo_i = x, i
            elif lo_i < i and x >= lo * (1.0 + pct):
                last_confirm = i
                state = 1
                hi, hi_i = x, i
    return last_confirm


def brute_zigzag_sigdays(p, pct):
    n = len(p)
    hits = set()
    for t in range(1, n):
        if _brute_zigzag_prefix(p[: t + 1], pct) == t:
            hits.add(t)
    return hits


# ---------------------------------------------------------------- 因果对拍: fractal
@pytest.mark.parametrize("symbol", TEST_STOCKS)
@pytest.mark.parametrize("order", [5, 10])
def test_fractal_causality(symbol, order):
    p = _close(symbol)
    low_idx, sig_idx = lab.fractal_lows(p, order)
    assert len(low_idx) > 0, "测试股票应存在分形低点"
    # 信号日 = 低点确认日, 必须在样本内且不与低点同日
    assert np.all(sig_idx == low_idx + order)
    assert np.all(sig_idx < len(p))
    lib_days = set(sig_idx.tolist())
    brute_days = brute_fractal_sigdays(p, order)
    assert lib_days == brute_days, (
        f"{symbol} order={order} mismatch: lib-only={sorted(lib_days - brute_days)[:5]}, "
        f"brute-only={sorted(brute_days - lib_days)[:5]}")


# ---------------------------------------------------------------- 因果对拍: zigzag
@pytest.mark.parametrize("symbol", TEST_STOCKS)
@pytest.mark.parametrize("pct", [0.03, 0.05])
def test_zigzag_causality(symbol, pct):
    p = _close(symbol)
    low_idx, sig_idx = lab.zigzag_lows(p, pct)
    assert len(low_idx) > 0, "测试股票应存在 zigzag 低点"
    assert np.all(sig_idx > low_idx)  # 确认日严格晚于低点日
    assert np.all(sig_idx < len(p))
    lib_days = set(sig_idx.tolist())
    brute_days = brute_zigzag_sigdays(p, pct)
    assert lib_days == brute_days, (
        f"{symbol} pct={pct} mismatch: lib-only={sorted(lib_days - brute_days)[:5]}, "
        f"brute-only={sorted(brute_days - lib_days)[:5]}")


# ---------------------------------------------------------------- legacy 复刻 V1
@pytest.mark.parametrize("symbol", TEST_STOCKS)
def test_legacy_matches_v1(symbol):
    import talib
    p = _close(symbol)
    cfg = lab._deep_merge(lab.DEFAULT_CONFIG, {"lows": {"method": "legacy"}})
    r = lab.load_stock((str(DATA / f"{symbol}.parquet"), cfg))
    assert "error" not in r
    dif = talib.MACD(p, 12, 26, 9)[0]
    ref = des.simulate_events_idx(p, dif)
    got = list(zip(r["events"]["sig"].tolist(), r["events"]["rank"].tolist(),
                   r["events"]["prev"].tolist()))
    assert got == [(t, rank, prev) for t, rank, prev in ref]


# ---------------------------------------------------------------- 低点序列与背离规则(合成数据)
def _dcfg(**kw):
    d = dict(lab.DEFAULT_CONFIG["divergence"])
    d.update(kw)
    return d


def test_min_sep_filter():
    low_idx = np.array([10, 12, 20, 40])
    sig_idx = low_idx + 5
    li, si = lab.apply_min_sep(low_idx, sig_idx, 5)
    assert li.tolist() == [10, 20, 40]
    assert si.tolist() == [15, 25, 45]


def _arrays():
    close = np.full(200, 100.0)
    dif = np.zeros(200)
    vol = np.full(200, 1000.0)
    return close, dif, vol


def test_divergence_rules_synthetic():
    close, dif, vol = _arrays()
    low_idx = np.array([50, 100, 150])
    sig_idx = low_idx + 5
    close[50], close[100], close[150] = 10.0, 9.0, 8.5   # 依次新低
    dif[50], dif[100], dif[150] = -1.0, -0.5, -0.2       # 依次抬高

    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(min_change=0.001, lookback=2), warmup=0)
    assert ev["sig"].tolist() == [105, 155]
    assert ev["rank"].tolist() == [1, 1]

    # min_decline=0.08: 9.0 < 10*0.92=9.2 成立; 8.5 < 9*0.92=8.28 不成立(lookback=1 隔离)
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(min_decline=0.08, lookback=1), warmup=0)
    assert ev["sig"].tolist() == [105]
    # lookback=2 时 155 可回看到低点 50 (8.5 < 10*0.92=9.2 成立, rank=2)
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(min_decline=0.08, lookback=2), warmup=0)
    assert ev["sig"].tolist() == [105, 155] and ev["rank"].tolist() == [1, 2]

    # below_zero: dif[150]=-0.2<0 仍满足; 改成正值则不满足
    dif2 = dif.copy(); dif2[150] = 0.3
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif2, dif2, vol,
                                      _dcfg(below_zero=True), warmup=0)
    assert ev["sig"].tolist() == [105]

    # volume_confirm: 次低缩量才保留
    vol[100] = 900.0; vol[150] = 1200.0
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(volume_confirm=True), warmup=0)
    assert ev["sig"].tolist() == [105]
    vol[100] = 1000.0; vol[150] = 1000.0

    # multi=3: 三个低点两两背离 => 只有第 3 个低点成事件
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(multi=3), warmup=0)
    assert ev["sig"].tolist() == [155]

    # 指标未抬高 => 无事件
    dif3 = dif.copy(); dif3[100] = dif3[50]
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif3, dif3, vol,
                                      _dcfg(), warmup=0)
    assert 105 not in ev["sig"].tolist()


def test_volume_ratio():
    close, dif, vol = _arrays()
    low_idx = np.array([50, 100, 150])
    sig_idx = low_idx + 5
    close[50], close[100], close[150] = 10.0, 9.0, 8.5   # 依次新低
    dif[50], dif[100], dif[150] = -1.0, -0.5, -0.2       # 依次抬高
    vol[50], vol[100], vol[150] = 1000.0, 950.0, 820.0

    # ratio=1.0 与 volume_confirm=true 等价
    ev_a = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                        _dcfg(volume_confirm=True), warmup=0)
    ev_b = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                        _dcfg(volume_ratio=1.0), warmup=0)
    assert ev_a["sig"].tolist() == ev_b["sig"].tolist() == [105, 155]
    # ratio=0.9 更严: 950 < 1000*0.9=900 不成立, 820 < 950*0.9=855 成立
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(volume_ratio=0.9), warmup=0)
    assert ev["sig"].tolist() == [155]
    # ratio=0.8: 820 < 950*0.8=760 不成立 => 空 (严格子集关系)
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(volume_ratio=0.8), warmup=0)
    assert ev["sig"].tolist() == []
    # volume_ratio 优先于 volume_confirm: ratio=1.0 放宽时 confirm=false 不影响
    ev = lab.detect_divergence_events(low_idx, sig_idx, close, dif, dif, vol,
                                      _dcfg(volume_confirm=False, volume_ratio=0.85), warmup=0)
    assert ev["sig"].tolist() == [155]


# ---------------------------------------------------------------- 标签(合成数据)
def _stock(close, open_=None, high=None, pct=None, atr=None):
    n = len(close)
    close = np.asarray(close, np.float32)
    if open_ is None:
        open_ = close.copy()
    if high is None:
        high = np.asarray(close, np.float32)
    if pct is None:
        c64 = close.astype(np.float64)
        pct = np.concatenate([[np.nan], c64[1:] / c64[:-1] * 100 - 100])
    cf = np.cumprod(1.0 + np.where(np.isfinite(pct), pct, 0.0) / 100.0)
    if atr is None:
        atr = np.full(n, 1.0, np.float32)
    return dict(close=close, open=np.asarray(open_, np.float32),
                high=np.asarray(high, np.float32),
                cf=cf.astype(np.float32), atr=np.asarray(atr, np.float32))


def _cfg(**label_kw):
    labels = dict(fixed=None, dynamic=None, sniper=None, mfe=None)
    labels.update(label_kw)
    return lab._deep_merge(lab.DEFAULT_CONFIG, {"labels": labels})


def test_label_fixed_close_T():
    close = np.arange(100, 130, dtype=float)  # 每日约等差
    pct = np.concatenate([[np.nan], close[1:] / close[:-1] * 100 - 100])
    st = _stock(close, pct=pct)
    cfg = _cfg(fixed=[5, 10])
    out = lab.compute_labels(st, [0, 19], None, cfg)
    c64 = close.astype(np.float64)
    assert out["ret_h5"][0] == pytest.approx(c64[5] / c64[0] - 1, rel=1e-5)
    assert out["ret_h10"][1] == pytest.approx(np.prod(1 + pct[20:30] / 100) - 1, rel=1e-5)
    out2 = lab.compute_labels(st, [25], None, cfg)
    assert np.isnan(out2["ret_h10"][0])  # 25+10 > 29 数据不足


def test_label_fixed_open_T1():
    close = np.full(30, 10.0)
    open_ = np.full(30, 8.0)
    pct = np.zeros(30); pct[0] = np.nan
    st = _stock(close, open_=open_, pct=pct)
    cfg = lab._deep_merge(_cfg(fixed=[5]), {"entry": "open_T1"})
    out = lab.compute_labels(st, [0], None, cfg)
    # T+1 开盘 8 买入, 之后每日 pct=0 => 收益 = 10/8 - 1
    assert out["ret_h5"][0] == pytest.approx(10 / 8 - 1, rel=1e-5)


def test_label_dynamic():
    close = np.arange(100.0, 200.0)
    st = _stock(close)
    cfg = _cfg(dynamic={"c": 0.5, "cap": 60})
    out = lab.compute_labels(st, [0, 10], [20, 200], cfg)  # h=10 / h=min(100,60)=60
    c64 = close.astype(np.float64)
    assert out["dyn"][0] == pytest.approx(c64[10] / c64[0] - 1, rel=1e-5)
    assert out["dyn"][1] == pytest.approx(c64[70] / c64[10] - 1, rel=1e-5)


def test_label_sniper_and_mfe():
    n = 50
    close = np.full(n, 10.0)
    open_ = np.full(n, 10.0)
    high = np.full(n, 10.0)
    high[12] = 13.5   # 信号日 t=10, 入场日 11, 窗口 [11..10+N]
    st = _stock(close, open_=open_, high=high, atr=np.full(n, 1.0, np.float32))
    cfg = lab._deep_merge(_cfg(sniper={"Ns": [10, 40], "ks": [2.0, 3.0]}, mfe=[10]),
                          {"entry": "open_T1"})
    out = lab.compute_labels(st, [10], [10], cfg)
    # 目标 = 10 + k*ATR(=1): k=2 -> 12 (13.5 触及), k=3 -> 13 (13.5 触及), N=10 窗口 [11..20] 含 12
    assert out["hit_N10_k2.0"][0] == 1.0
    assert out["hit_N10_k3.0"][0] == 1.0
    high2 = high.copy(); high2[12] = 12.5
    st2 = _stock(close, open_=open_, high=high2, atr=np.full(n, 1.0, np.float32))
    out2 = lab.compute_labels(st2, [10], [10], cfg)
    assert out2["hit_N10_k2.0"][0] == 1.0   # 12.5 >= 12
    assert out2["hit_N10_k3.0"][0] == 0.0   # 12.5 < 13
    assert out["mfe_h10"][0] == pytest.approx(13.5 / 10 - 1, rel=1e-5)
    # 数据不足: 信号日 45, N=10 => 45+10 > 49
    out3 = lab.compute_labels(st, [45], [10], cfg)
    assert np.isnan(out3["hit_N10_k2.0"][0])
    assert np.isnan(out3["mfe_h10"][0])


def test_label_columns_and_canon():
    cfg = lab._deep_merge(lab.DEFAULT_CONFIG,
                          {"labels": {"fixed": [5, 10, 30], "dynamic": {"c": 1.0, "cap": 60},
                                      "sniper": {"Ns": [20], "ks": [2.0]}, "mfe": [10]}})
    cols = lab.label_columns(cfg)
    assert cols == ["ret_h5", "ret_h10", "ret_h30", "dyn", "hit_N20_k2.0", "mfe_h10"]
    canon = lab._canon_cols(cols)
    assert canon["hit"] == "hit_N20_k2.0" and canon["mfe"] == "mfe_h10" and canon["r5"] == "ret_h5"
