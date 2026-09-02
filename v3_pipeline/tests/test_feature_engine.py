#!/usr/bin/env python3
"""feature_engine 泄漏防线与正确性测试 (主表第 7 章).

硬性对拍: 每个特征抽样真实股票 × 日期, 将输入截断至 T 重算, 与全历史计算在 T 处比对
(rtol<=1e-6); 市场特征框同法对拍. 另含列名黑名单、标签命名空间隔离、边界
(新股不足窗口/停牌缺失/除权日) 与若干手算真值测试.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import feature_engine as fe  # noqa: E402

REPO = SRC.parents[1]
DATA = REPO / "stock_data" / "daily"
TEST_STOCKS = ["600519.SH", "000001.SZ", "000002.SZ"]
N_TRUNC_DATES = 50
RTOL = 1e-6


# ================================================================ 构造工具
def _real_stock(code):
    df = fe.load_stock_df(DATA / f"{code}.parquet")
    return df


def _real_ctx(code):
    df = _real_stock(code)
    return df, fe.compute_stock_features(df, code)


def _make_df(n=300, code="000001.SZ", seed=0, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    close = 10.0 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    pre_close = np.concatenate([[np.nan], close[:-1]])
    pct = (close / pre_close - 1.0) * 100.0
    vol = rng.uniform(1e4, 1e5, n)
    return pd.DataFrame({
        "ts_code": code, "trade_date": pd.bdate_range(start, periods=n),
        "open": open_, "high": high, "low": low, "close": close,
        "pre_close": pre_close, "change": close - pre_close, "pct_chg": pct,
        "vol": vol, "amount": vol * close / 10.0,
    })


def _assert_series_equal_at(full_val, trunc_val, code, col, t, mismatches):
    a, b = full_val, trunc_val
    if np.isnan(a) and np.isnan(b):
        return
    if not np.isclose(a, b, rtol=RTOL, atol=1e-12, equal_nan=True):
        mismatches.append((code, col, int(t), float(a), float(b)))


# ================================================================ 7.1 列名黑名单
def test_blacklist_patterns_trigger():
    bad = [
        "stop_loss_return_5", "stop_loss_x",          # ^stop_loss_
        "future_ret",                                  # ^future_
        "next_open",                                   # ^next_
        "label_hit", "label",                          # ^label
        "mfr_20",                                      # ^mfr_
        "cur_return", "max_forward_return", "open_exec_return", "open_exec_return_20",
        "rank_future_ic", "rank_open_exec_20",
        "ret_h10", "ret_h5",                           # ^ret_h\d+$
        "hit_N20_k2.0",                                # ^hit_N
        "mfe_h10", "mae_1", "tmfe", "tmae", "tmfe_h5", "tmae_1",
        "dyn_ret",                                     # ^dyn_
        "entry_date",
        "rank_ret20", "rank_x",                        # ^rank_
    ]
    for name in bad:
        with pytest.raises(AssertionError, match="黑名单"):
            fe.assert_no_blacklisted([name])


def test_label_namespace_isolated():
    # 标签列 (divergence_lab/label_candidates/ranking_labels 命名空间) 不得出现在特征矩阵
    for name in ["ret_h10", "ret_h30", "hit_N20_k2.0", "hit_N40_k3.0", "mfe_h10",
                 "dyn", "cur_return", "max_forward_return"]:
        with pytest.raises(AssertionError, match="标签"):
            fe.assert_no_label_columns([name])


def test_blacklist_case_insensitive():
    with pytest.raises(AssertionError):
        fe.assert_no_blacklisted(["RET_H10"])
    with pytest.raises(AssertionError):
        fe.assert_no_blacklisted(["Rank_future_x"])


def test_registry_passes_blacklist_and_whitelist():
    fe.assert_no_blacklisted(list(fe.FEATURE_REGISTRY))
    fe.assert_no_label_columns(list(fe.FEATURE_REGISTRY))
    fe.assert_registry_inputs_whitelisted()
    assert len(fe.BLACKLIST_PATTERNS) >= 14


def test_registry_spec_counts():
    p0 = {s.feature for s in fe.FEATURE_REGISTRY.values() if s.layer == "P0"}
    p1 = {s.feature for s in fe.FEATURE_REGISTRY.values() if s.layer == "P1"}
    assert len(p0) == 60
    assert len(p1) == 81
    # 事件结构特征标记
    for c in ("RSI_DIV", "DIV_COUNT_120", "DIV_SPAN_BARS", "DAYS_SINCE_L2",
              "DIV_HIST_AREA_SHRINK", "DIV_GOLDEN_CROSS_STATE", "REBOUND_FROM_L2",
              "DIV_PRICE_NEWLOW_DEPTH", "DIV_DIF_LIFT"):
        assert fe.FEATURE_REGISTRY[c].event_only


def test_meta_cols_pass_blacklist():
    fe.assert_no_blacklisted(fe.META_COLS)


def test_generator_registry_interface():
    g = fe.GeneratorRegistry()
    spec = g.register("ROC20", "C~/Ref(C~,20)-1")
    assert spec.layer == "G" and spec.formula == "ROC20" or spec.formula
    assert "ROC20" in g.specs
    with pytest.raises(AssertionError):
        g.register("future_roc20", "x")


# ================================================================ 7.2 截断历史重算对拍 (股内特征)
@pytest.mark.parametrize("code", TEST_STOCKS)
def test_truncation_recompute_stock_features(code):
    df, (feats, ctx) = _real_ctx(code)
    n = len(df)
    rng = np.random.default_rng(42)
    ts = np.sort(rng.choice(np.arange(30, n), size=min(N_TRUNC_DATES, n - 30), replace=False))
    mismatches = []
    for t in ts:
        df_t = df.iloc[: t + 1]
        feats_t, _ = fe.compute_stock_features(df_t, code)
        for col in feats.columns:
            _assert_series_equal_at(feats[col].iloc[t], feats_t[col].iloc[t],
                                    code, col, t, mismatches)
    assert not mismatches, f"{code} 截断对拍失败 {len(mismatches)} 处: {mismatches[:5]}"


@pytest.mark.parametrize("code", TEST_STOCKS)
def test_truncation_recompute_event_features(code):
    df, (feats, ctx) = _real_ctx(code)
    n = len(df)
    rng = np.random.default_rng(7)
    ts = np.sort(rng.choice(np.arange(300, n), size=min(30, n - 300), replace=False))
    mismatches = []
    for t in ts:
        i1, i2 = t - 40, t - 15
        sig = np.array([t])
        full = fe.compute_event_features(ctx, sig, np.array([i1]), np.array([i2]), sig)
        df_t = df.iloc[: t + 1]
        _, ctx_t = fe.compute_stock_features(df_t, code)
        tr = fe.compute_event_features(ctx_t, sig, np.array([i1]), np.array([i2]), sig)
        for col in fe.EVENT_FEATURE_COLS:
            _assert_series_equal_at(full[col][0], tr[col][0], code, col, t, mismatches)
    assert not mismatches, f"{code} 事件特征截断对拍失败: {mismatches[:5]}"


# ================================================================ 7.2 截断对拍 (市场特征框)
def _mini_panel():
    idx_sh = fe.load_index_df(DATA / "000001.SH.parquet")
    idx_sz = fe.load_index_df(DATA / "399001.SZ.parquet")
    parts = []
    for sid, code in enumerate(TEST_STOCKS):
        df = _real_stock(code)
        _, ctx = fe.compute_stock_features(df, code)
        parts.append(pd.DataFrame({"sid": sid, "date": ctx["days"], "is_index": False,
                                   **ctx["panel"]}))
    return pd.concat(parts, ignore_index=True), idx_sh, idx_sz


def test_truncation_recompute_market_frame():
    panel, idx_sh, idx_sz = _mini_panel()
    market, dates = fe.build_market_frame(panel, idx_sh, idx_sz)
    rng = np.random.default_rng(11)
    picks = rng.choice(np.arange(300, len(dates)), size=20, replace=False)
    mismatches = []
    for di in picks:
        t_day = int(dates[di])
        panel_t = panel[panel["date"] <= t_day]
        sh_t = idx_sh[idx_sh["_days"] <= t_day]
        sz_t = idx_sz[idx_sz["_days"] <= t_day]
        market_t, dates_t = fe.build_market_frame(panel_t, sh_t, sz_t)
        assert dates_t[-1] == t_day
        for col in market.columns:
            a, b = market[col].iloc[di], market_t[col].iloc[-1]
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isclose(a, b, rtol=RTOL, atol=1e-12, equal_nan=True):
                mismatches.append((col, t_day, float(a), float(b)))
    assert not mismatches, f"市场特征截断对拍失败: {mismatches[:5]}"


# ================================================================ 边界: 新股不足窗口
def test_new_listing_short_history():
    df = _make_df(n=40)
    feats, ctx = fe.compute_stock_features(df, "000001.SZ")
    assert len(feats) == 40
    assert np.isfinite(feats["RET5"].iloc[-1])          # 短窗口有值
    assert feats["BIAS60"].isna().all()                  # 长窗口全 NaN, 不报错
    assert feats["MA200_RATIO"].isna().all()
    assert feats["RET120_20"].isna().all()
    assert (feats["NEW_LISTING"] == 1.0).all()
    ev = fe.compute_event_features(ctx, np.array([39]), np.array([10]), np.array([30]),
                                   np.array([39]))
    assert np.isfinite(ev["DIV_SPAN_BARS"][0])


# ================================================================ 边界: 停牌缺失 (bar 口径滚动)
def test_suspension_gap_bar_based():
    df = _make_df(n=300, seed=3)
    keep = np.ones(len(df), bool)
    keep[100:130] = False  # 停牌 30 根
    df2 = df[keep].reset_index(drop=True)
    feats2, ctx2 = fe.compute_stock_features(df2, "000001.SZ")
    # CF 链只依赖可得 bar: RET20 = CF[i]/CF[i-20]-1 精确成立
    cf = ctx2["ca"]
    i = 200
    assert feats2["RET20"].iloc[i] == pytest.approx(cf[i] / cf[i - 20] - 1.0, rel=1e-12)
    # 截断对拍在缺行序列上依然成立
    t = 250
    feats_t, _ = fe.compute_stock_features(df2.iloc[: t + 1], "000001.SZ")
    for col in ("RET20", "VOL20", "BIAS20", "RSI14", "MACD_HIST", "ATRN"):
        a, b = feats2[col].iloc[t], feats_t[col].iloc[t]
        assert np.isclose(a, b, rtol=RTOL, atol=1e-12, equal_nan=True), col


# ================================================================ 边界: 除权日 (pct_chg 链连续, 原始价跳变)
def test_split_adjustment_chain():
    n = 120
    df = _make_df(n=n, seed=5)
    k = 60
    # k 日起 1拆2: 价格全减半, 但 pct_chg 仍按真实收益 (约 0) 给出
    for col in ("open", "high", "low", "close"):
        df.loc[k:, col] = df.loc[k:, col] / 2.0
    df.loc[k:, "pre_close"] = np.concatenate([[np.nan], df["close"].to_numpy()[k:-1]])
    df.loc[k, "pre_close"] = df["close"].to_numpy()[k - 1] / 2.0  # 除权后前收盘
    df.loc[k:, "pct_chg"] = (df["close"].to_numpy()[k:]
                             / df["pre_close"].to_numpy()[k:] - 1.0) * 100.0
    df.loc[k, "pct_chg"] = 0.1  # 除权日真实涨幅 ~0
    feats, ctx = fe.compute_stock_features(df, "000001.SZ")
    # CF 链在除权日连续: RET1[k] 为真实收益而非 -50%
    assert feats["RET1"].iloc[k] == pytest.approx(0.001, rel=1e-9)
    assert abs(feats["RET20"].iloc[k + 5]) < 0.2  # 复权收益链无 -50% 跳变
    # LOG_PRICE 保留原始价失真 (主表 1.2 标注口径)
    assert feats["LOG_PRICE"].iloc[k - 1] - feats["LOG_PRICE"].iloc[k] > 0.5
    # ATRN 用复权价, 除权日无数量级跳变
    assert feats["ATRN"].iloc[k] < 3.0 * feats["ATRN"].iloc[k - 1]


# ================================================================ 手算真值
def test_known_values_kline_and_limit():
    df = _make_df(n=300, seed=9)
    # 一字板: H==L==O==C
    df.loc[100, ["open", "high", "low", "close"]] = 12.0
    feats, _ = fe.compute_stock_features(df, "000001.SZ")
    assert feats["CLOSE_VS_HIGH"].iloc[100] == 0.5
    assert feats["UPPER_SHADOW_RATIO"].iloc[100] == 0.0
    # 板块涨停幅度
    days = fe._to_days(df["trade_date"])
    assert fe.limit_pct_array("688001.SH", days)[0] == 20.0
    assert fe.limit_pct_array("830799.BJ", days)[0] == 30.0
    cut = np.datetime64("2020-08-24").astype("datetime64[D]").astype(np.int32)
    d2 = np.array([cut - 1, cut], np.int32)
    assert fe.limit_pct_array("300001.SZ", d2).tolist() == [10.0, 20.0]
    assert fe.limit_pct_array("600519.SH", d2).tolist() == [10.0, 10.0]


def test_known_values_operators():
    x = np.arange(50, dtype=np.float64) * 2.0 + 1.0
    assert fe._slope(x, 10)[-1] == pytest.approx(2.0, rel=1e-12)
    r = fe._ts_pct_rank(np.arange(30, dtype=np.float64), 10)
    assert r[-1] == pytest.approx(0.95)          # 单调升, (less+0.5*ties)/w = (9+0.5)/10
    assert np.isnan(r[8]) and np.isfinite(r[9])  # 全窗口才出值
    d = fe._idx_extreme(np.arange(30, dtype=np.float64), 20, "max")
    assert d[-1] == 0.0                          # 最大值就在今天
    streak = fe._streak(np.array([False, True, True, False, True]))
    assert streak.tolist() == [0, 1, 2, 0, 1]


def test_attach_ret20_csr_known():
    panel = pd.DataFrame({
        "sid": [0, 1, 2, 0, 1], "date": [10, 10, 10, 11, 11],
        "is_index": False, "r1": 0.0, "ret5": 0.0,
        "ret20": [-0.1, 0.0, 0.2, 0.05, np.nan],
        "above_ma20": 1.0, "newlow250": 0.0, "amount": 1.0, "limit_up": 0.0,
    })
    csr = fe.attach_ret20_csr(panel, np.array([10, 10, 10, 11, 11], np.int32),
                              np.array([-0.1, 0.0, 0.2, 0.05, 0.01]))
    assert csr[:4].tolist() == pytest.approx([0 / 3 + 1 / 6, 1 / 3 + 1 / 6, 2 / 3 + 1 / 6, 0.5])
    assert csr[4] == 0.0     # 0.01 低于当日唯一有效值 0.05 -> 分位 0
    assert np.isnan(fe.attach_ret20_csr(panel, np.array([11], np.int32), np.array([np.nan]))[0])


# ================================================================ 端到端小装配: 标签隔离 + 黑名单 + 行对齐
def test_assemble_pool_end_to_end_mini():
    panel, idx_sh, idx_sz = _mini_panel()
    market, _ = fe.build_market_frame(panel, idx_sh, idx_sz)
    idx_dates = idx_sh["_days"].to_numpy(np.int32)
    c_idx = idx_sh["close"].to_numpy(np.float64)
    idx_r = np.concatenate([[np.nan], c_idx[1:] / c_idx[:-1] - 1.0])
    ev_frames = {}
    for sid, code in enumerate(TEST_STOCKS):
        df = _real_stock(code)
        n = len(df)
        days = fe._to_days(df["trade_date"])
        events = {"main": [(0, n - 1, int(days[-1]), int(days[n - 30]), int(days[n - 15]))]}
        r = fe.process_stock((DATA / f"{code}.parquet", code, sid, events, idx_dates, idx_r))
        assert r["error"] is None
        ev_frames[code] = r["rows"]["main"]
    rows = pd.concat(ev_frames.values()).sort_index()
    # 每个股票同一 event_id=0 会冲突 -> 改为逐股 event_id
    rows.index = pd.Index(range(len(rows)), name="event_id")
    ev_meta = []
    for sid, code in enumerate(TEST_STOCKS):
        df = _real_stock(code)
        ev_meta.append({"event_id": sid, "ts_code": code, "date": df["trade_date"].iloc[-1],
                        "sig_idx": len(df) - 1})
    events_df = pd.DataFrame(ev_meta)
    mat = fe.assemble_pool(events_df, rows, market, panel)
    assert len(mat) == len(TEST_STOCKS)
    fe.assert_no_blacklisted(mat.columns)
    fe.assert_no_label_columns(mat.columns)
    feat_cols = [c for c in mat.columns if c not in fe.META_COLS]
    assert len(feat_cols) == len(fe.FEATURE_REGISTRY)
    csr = mat["RET20_CSR"].to_numpy()
    assert ((csr >= 0) & (csr <= 1)).all()


# ================================================================ inf 护栏 (复核条件项 1)
def test_assert_no_inf_trigger():
    frame = pd.DataFrame({"A": [1.0, np.inf], "B": [np.nan, 2.0]})
    with pytest.raises(AssertionError, match="inf"):
        fe.assert_no_inf(frame)
    fe.assert_no_inf(pd.DataFrame({"A": [1.0, np.nan], "B": [-1e30, 0.0]}))


def test_limit_lock_stock_no_inf_regression():
    """一字板回归 (真实案例): 000556.SZ 2001-11-02 前后连续一字板, H/L 窗口常数,
    rolling.corr 曾产出 inf (全历史 31 处); 修复后全部特征列 0 inf, 常数窗 corr 为 NaN."""
    df = _real_stock("000556.SZ")
    feats, ctx = fe.compute_stock_features(df, "000556.SZ")
    fe.assert_no_inf(feats)
    days = ctx["days"]
    i = int(np.searchsorted(days, np.datetime64("2001-11-02").astype("datetime64[D]").astype(np.int32)))
    assert days[i] == np.datetime64("2001-11-02").astype("datetime64[D]").astype(np.int32)
    assert np.isnan(feats["HLV_DIV10"].iloc[i])          # 修复前为 inf
    for col in ("CPV10", "CPV20", "CPV_VWAP10", "RVC20"):
        v = feats[col].iloc[i]
        assert np.isnan(v) or np.isfinite(v)
    # 面板列同样无 inf
    for k, v in ctx["panel"].items():
        assert not np.isinf(v).any(), k


def test_assemble_rejects_unregistered_columns():
    panel, idx_sh, idx_sz = _mini_panel()
    market, _ = fe.build_market_frame(panel, idx_sh, idx_sz)
    idx_dates = idx_sh["_days"].to_numpy(np.int32)
    c_idx = idx_sh["close"].to_numpy(np.float64)
    idx_r = np.concatenate([[np.nan], c_idx[1:] / c_idx[:-1] - 1.0])
    df0 = _real_stock("000001.SZ")
    days0 = fe._to_days(df0["trade_date"])
    events = {"main": [(0, len(df0) - 1, int(days0[-1]), int(days0[-30]), int(days0[-15]))]}
    r = fe.process_stock((DATA / "000001.SZ.parquet", "000001.SZ", 0, events, idx_dates, idx_r))
    rows = r["rows"]["main"].copy()
    rows["NOT_A_FEATURE"] = 1.0  # 未注册注入列
    events_df = pd.DataFrame([{"event_id": 0, "ts_code": "000001.SZ",
                               "date": df0["trade_date"].iloc[-1], "sig_idx": len(df0) - 1}])
    with pytest.raises(AssertionError, match="未注册注入列"):
        fe.assemble_pool(events_df, rows, market, panel)
