#!/usr/bin/env python3
"""v4_daily_snapshot 的单元测试（接缝: 原始日线 -> 事件日 V4 特征快照）。

只验外部行为:
  - 逐股特征链在合成数据上的已知值抽查;
  - 前缀稳定性: 全历史计算在 T 日的取值 == 仅用不晚于 T 数据重算的取值（漂移审计核心）;
  - 输出不含泄漏/标签族列;
  - 日历/大盘/横截面特征的已知值;
  - 快照取行语义。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import src.feature_pipeline_v2 as fp2  # noqa: E402
import v4_daily_snapshot as v4s  # noqa: E402


def _synthetic_stock(n=400, seed=7, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    close = 20.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, n))
    open_ = close * (1.0 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, n))
    vol = rng.uniform(5000, 20000, n)
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({"timestamp": dates, "symbol": "SYNTH", "open": open_,
                         "high": high, "low": low, "close": close, "volume": vol})


@pytest.fixture(scope="module")
def feat_full():
    df = _synthetic_stock()
    pipe = fp2.FeaturePipeline(None, None)
    return v4s.compute_stock_features(df, pipe=pipe)


# ---------------------------------------------------------------- 已知值抽查
def test_known_values(feat_full):
    df = _synthetic_stock()
    f = feat_full
    # ma_5 是 close 的 5 日滚动均值
    np.testing.assert_allclose(
        f["ma_5"].to_numpy()[10:], df["close"].rolling(5).mean().to_numpy()[10:],
        rtol=1e-12)
    # pct_change 是 close 的一阶日收益
    np.testing.assert_allclose(
        f["pct_change"].to_numpy()[1:], df["close"].pct_change().to_numpy()[1:],
        rtol=1e-12, equal_nan=True)
    # ret_intraday = close/open - 1
    np.testing.assert_allclose(
        f["ret_intraday"].to_numpy()[5:],
        (df["close"] / df["open"] - 1.0).to_numpy()[5:], rtol=1e-12)


def test_no_forbidden_columns(feat_full):
    import re
    pats = [re.compile(p) for p in v4s.FORBIDDEN_PATTERNS]
    bad = [c for c in feat_full.columns if any(p.match(c) for p in pats)]
    assert bad == []


# ---------------------------------------------------------------- 前缀稳定性(核心)
@pytest.mark.parametrize("t_pos", [150, 250, 399])
def test_prefix_stability(feat_full, t_pos):
    df = _synthetic_stock()
    T = df["timestamp"].iloc[t_pos]
    pipe = fp2.FeaturePipeline(None, None)
    feat_prefix = v4s.compute_stock_features(df.iloc[: t_pos + 1].reset_index(drop=True),
                                             pipe=pipe)
    row_full = feat_full[feat_full["timestamp"] == T]
    row_pref = feat_prefix[feat_prefix["timestamp"] == T]
    assert len(row_full) == 1 and len(row_pref) == 1
    cmp_cols = [c for c in feat_full.columns
                if c not in v4s.MARKET_RANK_COLS + v4s.KEY_COLS]
    a = row_full[cmp_cols].iloc[0].to_numpy(np.float64)
    b = row_pref[cmp_cols].iloc[0].to_numpy(np.float64)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=0, equal_nan=True,
                               err_msg=f"前缀不稳定 @ {T.date()}")


# ---------------------------------------------------------------- 日历特征
def test_calendar_features():
    df = pd.DataFrame({"timestamp": pd.to_datetime(
        ["2021-03-01", "2021-03-31", "2021-12-31"])})
    out = v4s.add_calendar_features(df)
    assert out["day_of_month"].tolist() == [1, 31, 31]
    assert out["is_month_start"].tolist() == [1, 0, 0]
    assert out["is_month_end"].tolist() == [0, 1, 1]
    assert out["is_quarter_end"].tolist() == [0, 1, 1]
    assert out["quarter"].tolist() == [1, 1, 4]


# ---------------------------------------------------------------- 大盘特征
def test_market_features_known_values():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    base = np.linspace(3000, 3100, 60)
    idx = pd.DataFrame({"open": base * 0.999, "high": base * 1.01, "low": base * 0.99,
                        "close": base, "volume": np.full(60, 1e5)}, index=dates)
    pipe = fp2.FeaturePipeline(None, None)
    T = dates[40]
    m = pipe._calculate_market_features(T, df_sh=idx, df_sz=idx)
    row = m.loc[T]
    expected_chg = (base[40] - base[40] * 0.999) / (base[40] * 0.999)
    assert row["sh_price_change"] == pytest.approx(expected_chg)
    assert row["sz_price_change"] == pytest.approx(expected_chg)
    assert row["sh_sz_sync_direction"] == 1
    # 前缀一致: 截断指数数据重算同值
    m2 = pipe._calculate_market_features(T, df_sh=idx.loc[:T], df_sz=idx.loc[:T])
    assert row["sh_volume_ratio"] == pytest.approx(m2.loc[T]["sh_volume_ratio"])


# ---------------------------------------------------------------- 横截面特征
def test_cross_features_complementary():
    day = pd.DataFrame({
        "timestamp": [pd.Timestamp("2021-06-01")] * 3,
        "symbol": ["A", "B", "C"],
        "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0],
        "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0],
        "volume": [100.0, 200.0, 300.0],
        "feat_x": [10.0, 20.0, 30.0],
        "flag_i": [0, 1, 1],
    })
    pipe = fp2.FeaturePipeline(None, None)
    out = pipe._calculate_cross_features(day)
    assert out["cs_n"].tolist() == [3, 3, 3]
    # open/close/high/low 被排除, volume/feat_x/flag_i 参与排名
    assert "open_rankpct" not in out.columns
    assert "volume_rankpct" in out.columns
    np.testing.assert_allclose(sorted(out["feat_x_rankpct"]), [1 / 3, 2 / 3, 1.0])
    # 同日群体外无泄漏: 加一天不同群体不影响前一天的排名
    day2 = pd.concat([day, day.assign(timestamp=pd.Timestamp("2021-06-02"),
                                      feat_x=[99.0, 1.0, 50.0])])
    out2 = pipe._calculate_cross_features(day2)
    first = out2[out2["timestamp"] == pd.Timestamp("2021-06-01")]
    np.testing.assert_allclose(first["feat_x_rankpct"].to_numpy(),
                               out["feat_x_rankpct"].to_numpy())


# ---------------------------------------------------------------- 快照取行
def test_snapshot_rows(feat_full):
    dates = feat_full["timestamp"].iloc[[100, 200]]
    snap = v4s.snapshot_rows(feat_full, dates)
    assert len(snap) == 2
    assert set(snap["timestamp"]) == set(dates)
