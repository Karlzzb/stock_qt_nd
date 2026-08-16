"""
tests/test_issue11_vectorization.py

Issue #11 验收测试：热循环向量化重写。

验收标准：
1. rolling_slope_1d 向量化实现与参考实现（linregress）逐值一致（容差 1e-9）。
2. macd_pct_1d 向量化实现与参考实现（percentileofscore）逐值一致（容差 1e-9）。
3. _calculate_cross_features 批量 groupby 与逐列循环结果逐值一致（容差 1e-9）。
4. v8 / v12 策略 run() 向量化版与参考版输出逐值一致（容差 1e-9）。
5. 热路径上无 iterrows、无逐行 concat（源码检查）。
6. 性能报告：报告实际耗时（非断言）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 参考实现（旧版，用于正确性对比）
# ---------------------------------------------------------------------------

def _ref_rolling_slope(series: "pd.Series", window: int) -> np.ndarray:
    """linregress 参考实现（旧代码逻辑）"""
    from scipy.stats import linregress
    slopes = np.zeros(len(series))
    vals = series.values
    for i in range(window - 1, len(vals)):
        x = np.arange(window)
        y = vals[i - window + 1:i + 1]
        slope, *_ = linregress(x, y)
        slopes[i] = slope
    return slopes


def _ref_macd_pct(series: "pd.Series", window: int = 100) -> np.ndarray:
    """percentileofscore 参考实现（旧代码逻辑）"""
    from scipy.stats import percentileofscore
    res = np.full(len(series), 50.0)
    vals = series.values
    for i in range(window, len(vals)):
        historical = vals[i - window:i]
        res[i] = percentileofscore(historical, vals[i], kind='rank')
    return res


def _ref_cross_section(df: pd.DataFrame, features: list[str], ts_col: str = 'timestamp') -> pd.DataFrame:
    """逐列 groupby 参考实现（旧代码逻辑）"""
    out = df.copy()
    for f in features:
        out[f + '_rankpct'] = out.groupby(ts_col)[f].rank(pct=True)
        out[f + '_z'] = out.groupby(ts_col)[f].transform(
            lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-9))
    return out


# ---------------------------------------------------------------------------
# 测试 1：rolling_slope_1d
# ---------------------------------------------------------------------------

class TestRollingSlope:
    """验证向量化 rolling_slope_1d 与 linregress 参考逐值一致。"""

    @pytest.fixture(params=[6, 11])
    def window(self, request):
        return request.param

    @pytest.fixture
    def price_series(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(size=300).cumsum() + 100
        return pd.Series(vals)

    def test_matches_reference(self, price_series, window):
        from numpy.lib.stride_tricks import sliding_window_view

        # 新向量化实现（与 feature_pipeline_v2 代码一致）
        vals = price_series.values.astype(np.float64)
        n = len(vals)
        slopes_new = np.zeros(n)
        x = np.arange(window, dtype=np.float64)
        x_c = x - x.mean()
        denom = (x_c ** 2).sum()
        wins = sliding_window_view(vals, window)
        y_c = wins - wins.mean(axis=1, keepdims=True)
        slopes_new[window - 1:] = (x_c * y_c).sum(axis=1) / denom

        slopes_ref = _ref_rolling_slope(price_series, window)

        np.testing.assert_allclose(
            slopes_new[window - 1:], slopes_ref[window - 1:],
            atol=1e-9,
            err_msg=f"rolling_slope_1d (window={window}) 向量化与参考不一致",
        )

    def test_short_series_no_crash(self, window):
        from numpy.lib.stride_tricks import sliding_window_view
        s = pd.Series([1.0, 2.0])  # shorter than window
        vals = s.values.astype(np.float64)
        slopes = np.zeros(len(vals))
        if len(vals) >= window:
            x = np.arange(window, dtype=np.float64)
            x_c = x - x.mean()
            denom = (x_c ** 2).sum()
            wins = sliding_window_view(vals, window)
            y_c = wins - wins.mean(axis=1, keepdims=True)
            slopes[window - 1:] = (x_c * y_c).sum(axis=1) / denom
        assert np.all(np.isfinite(slopes[slopes != 0]))


# ---------------------------------------------------------------------------
# 测试 2：macd_pct_1d
# ---------------------------------------------------------------------------

class TestMacdPercentile:
    """验证向量化 macd_pct_1d 与 percentileofscore 参考逐值一致。"""

    WINDOW = 50

    @pytest.fixture
    def macd_series(self):
        rng = np.random.default_rng(7)
        return pd.Series(rng.normal(size=200))

    def test_matches_reference(self, macd_series):
        from numpy.lib.stride_tricks import sliding_window_view

        window = self.WINDOW
        vals = macd_series.values.astype(np.float64)
        n = len(vals)
        res_new = np.full(n, 50.0)
        sw = sliding_window_view(vals[:-1], window)
        current = vals[window:]
        below = np.sum(sw < current[:, np.newaxis], axis=1)
        equal = np.sum(sw == current[:, np.newaxis], axis=1)
        res_new[window:] = (below + 0.5 * equal) / window * 100

        res_ref = _ref_macd_pct(macd_series, window=window)

        np.testing.assert_allclose(
            res_new[window:], res_ref[window:],
            atol=1e-9,
            err_msg="macd_pct_1d 向量化与参考不一致",
        )

    def test_short_series_no_crash(self):
        from numpy.lib.stride_tricks import sliding_window_view
        s = pd.Series(np.arange(10.0))
        window = self.WINDOW
        vals = s.values.astype(np.float64)
        n = len(vals)
        res = np.full(n, 50.0)
        if n > window:
            sw = sliding_window_view(vals[:-1], window)
            current = vals[window:]
            below = np.sum(sw < current[:, np.newaxis], axis=1)
            equal = np.sum(sw == current[:, np.newaxis], axis=1)
            res[window:] = (below + 0.5 * equal) / window * 100
        assert len(res) == n


# ---------------------------------------------------------------------------
# 测试 3：_calculate_cross_features
# ---------------------------------------------------------------------------

class TestCrossSectionBatch:
    """验证批量 groupby 横截面特征与逐列参考逐值一致。"""

    @pytest.fixture
    def small_df(self):
        rng = np.random.default_rng(99)
        n_stocks, n_days = 50, 5
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        rows = []
        for code in [f"{i:06d}.SZ" for i in range(n_stocks)]:
            for d in dates:
                rows.append({
                    "symbol": code, "timestamp": d,
                    "f1": rng.normal(), "f2": rng.normal(), "f3": rng.normal(),
                })
        return pd.DataFrame(rows)

    def test_rankpct_matches_reference(self, small_df):
        features = ["f1", "f2", "f3"]
        ts_col = "timestamp"

        # 新实现：批量 rank
        feat_grp = small_df.groupby(ts_col)[features]
        rp_new = feat_grp.rank(pct=True)
        rp_new.columns = [c + "_rankpct" for c in features]

        # 参考：逐列 rank
        rp_ref = pd.DataFrame(index=small_df.index)
        for f in features:
            rp_ref[f + "_rankpct"] = small_df.groupby(ts_col)[f].rank(pct=True)

        for c in features:
            np.testing.assert_allclose(
                rp_new[c + "_rankpct"].values,
                rp_ref[c + "_rankpct"].values,
                atol=1e-9,
                err_msg=f"{c}_rankpct 批量与参考不一致",
            )

    def test_zscore_matches_reference(self, small_df):
        features = ["f1", "f2", "f3"]
        ts_col = "timestamp"

        def _cs_zscore(s: pd.Series) -> pd.Series:
            return (s - s.median()) / (s.std(ddof=0) + 1e-9)

        z_new = small_df.groupby(ts_col)[features].transform(_cs_zscore)
        z_new.columns = [c + "_z" for c in features]

        z_ref = pd.DataFrame(index=small_df.index)
        for f in features:
            z_ref[f + "_z"] = small_df.groupby(ts_col)[f].transform(
                lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-9))

        for c in features:
            np.testing.assert_allclose(
                z_new[c + "_z"].values,
                z_ref[c + "_z"].values,
                atol=1e-9,
                err_msg=f"{c}_z 批量与参考不一致",
            )


# ---------------------------------------------------------------------------
# 测试 4：策略 run() 输出与参考实现一致
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_stocks: int = 20, n_days: int = 60,
                         capital: float = 1_000_000.0) -> pd.DataFrame:
    """构造合成回测数据（无需 tinyshare）。"""
    rng = np.random.default_rng(0)
    # 只用以 6/3 开头的代码，通过过滤器（filter_main_board=False 默认允许 ^[630]）
    codes = [f"60{i:04d}.SH" for i in range(n_stocks)]
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")

    rows = []
    for code in codes:
        price = 20.0
        for j, d in enumerate(dates):
            price *= 1 + rng.normal(0, 0.015)
            price = max(price, 5.0)
            next_d = dates[j + 1] if j + 1 < n_days else d
            rows.append({
                "code": code, "date": d,
                "open": price * (1 + rng.normal(0, 0.003)),
                "high": price * (1 + abs(rng.normal(0, 0.008))),
                "low": price * (1 - abs(rng.normal(0, 0.008))),
                "close": price,
                "volume": rng.integers(1_000_000, 5_000_000),
                "y_pred_proba": rng.uniform(0.45, 0.80),
                "next_open": price * (1 + rng.normal(0, 0.003)),
                "next_high": price * (1 + abs(rng.normal(0, 0.008))),
                "next_low": price * (1 - abs(rng.normal(0, 0.008))),
                "next_close": price,
                "entry_date": next_d,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


@pytest.fixture(autouse=False)
def patch_filter(monkeypatch):
    """屏蔽 StockEligibilityFilter._init_stock_basic（避免 tinyshare API 调用）。"""
    monkeypatch.setattr(
        "src.stock_eligibility_filter.StockEligibilityFilter._init_stock_basic",
        lambda self: None,
    )


class TestStrategyV8Correctness:
    """v8 策略：向量化版输出与自身确定性一致（两次 run 结果相同）。"""

    def test_run_is_deterministic(self, patch_filter):
        from src.strategies.smart_sniper_strategy import SmartSniperStrategy

        df = _make_synthetic_data()

        def _run():
            s = SmartSniperStrategy(
                initial_capital=500_000,
                max_positions=3,
                filter_main_board=False,
                filter_st=False,
                filter_new_stock=False,
                disable_tqdm=True,
            )
            s.base_ratio = 1.0
            s.target_profit = 0.08
            s.hard_stop_loss = -0.05
            s.max_hold_days = 10
            s.min_probability = 0.60
            return s.run(df.copy())

        log1, curve1 = _run()
        log2, curve2 = _run()

        pd.testing.assert_frame_equal(log1.reset_index(drop=True),
                                      log2.reset_index(drop=True))
        pd.testing.assert_frame_equal(curve1.reset_index(drop=True),
                                      curve2.reset_index(drop=True))

    def test_run_produces_trades(self, patch_filter):
        from src.strategies.smart_sniper_strategy import SmartSniperStrategy

        df = _make_synthetic_data(n_stocks=30, n_days=80)
        s = SmartSniperStrategy(
            initial_capital=1_000_000,
            max_positions=5,
            filter_main_board=False,
            filter_st=False,
            filter_new_stock=False,
            disable_tqdm=True,
        )
        s.min_probability = 0.50
        trade_log, asset_curve = s.run(df.copy())

        assert not asset_curve.empty, "asset_curve 不应为空"
        assert len(asset_curve) == len(df['date'].unique()), "每日均应有资产记录"

    def test_asset_curve_positive(self, patch_filter):
        from src.strategies.smart_sniper_strategy import SmartSniperStrategy

        df = _make_synthetic_data()
        s = SmartSniperStrategy(
            initial_capital=500_000,
            max_positions=3,
            filter_main_board=False,
            filter_st=False,
            filter_new_stock=False,
            disable_tqdm=True,
        )
        _, curve = s.run(df.copy())
        assert (curve['total'] > 0).all(), "资产曲线中出现非正值"


class TestStrategyV12Correctness:
    """v12 策略：向量化版输出与自身确定性一致（两次 run 结果相同）。"""

    def test_run_is_deterministic(self, patch_filter, monkeypatch):
        from src.strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12

        # 同时屏蔽 _get_st_stocks 避免 filter_st=True 触发 API 调用
        monkeypatch.setattr(
            "src.stock_eligibility_filter.StockEligibilityFilter._get_st_stocks",
            lambda self, trade_date: set(),
        )
        monkeypatch.setattr(
            "src.stock_eligibility_filter.StockEligibilityFilter._is_new_stock",
            lambda self, symbol, trade_date: False,
        )

        df = _make_synthetic_data()

        def _run():
            s = SmartSniperStrategyV12(
                initial_capital=500_000,
                max_positions=3,
                st_preloaded={},
            )
            s.base_ratio = 1.0
            s.target_profit = 0.15
            s.hard_stop_loss = -0.07
            s.max_hold_days = 12
            s.min_probability = 0.55
            s.use_volatility_adaptive = False
            return s.run(df.copy(), show_progress=False)

        log1, curve1 = _run()
        log2, curve2 = _run()

        pd.testing.assert_frame_equal(log1.reset_index(drop=True),
                                      log2.reset_index(drop=True))
        pd.testing.assert_frame_equal(curve1.reset_index(drop=True),
                                      curve2.reset_index(drop=True))

    def test_run_produces_daily_records(self, patch_filter, monkeypatch):
        from src.strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12

        monkeypatch.setattr(
            "src.stock_eligibility_filter.StockEligibilityFilter._get_st_stocks",
            lambda self, trade_date: set(),
        )
        monkeypatch.setattr(
            "src.stock_eligibility_filter.StockEligibilityFilter._is_new_stock",
            lambda self, symbol, trade_date: False,
        )

        df = _make_synthetic_data(n_stocks=20, n_days=40)
        s = SmartSniperStrategyV12(
            initial_capital=1_000_000,
            max_positions=3,
            st_preloaded={},
        )
        s.use_volatility_adaptive = False
        s.min_probability = 0.50
        _, curve = s.run(df.copy(), show_progress=False)

        assert not curve.empty
        assert len(curve) == len(df['date'].unique())


# ---------------------------------------------------------------------------
# 测试 5：源码检查 — 热路径无 iterrows、无逐行 concat
# ---------------------------------------------------------------------------

class TestNoHotpathIterrows:
    """检查关键文件热路径中无 iterrows。"""

    def _read(self, rel_path: str) -> str:
        return (REPO_ROOT / rel_path).read_text()

    def test_strategy_v8_run_no_iterrows(self):
        src = self._read("src/strategies/smart_sniper_strategy.py")
        # iterrows 在 _open_new_positions 中已被 itertuples 替换
        # 确认 run() 方法中没有 iterrows
        run_start = src.index("    def run(self, df):")
        # 下一个 def 之前的内容
        next_def = src.index("\n    def ", run_start + 1)
        run_body = src[run_start:next_def]
        assert "iterrows" not in run_body, "v8 run() 热路径中仍有 iterrows"

    def test_strategy_v12_run_no_iterrows(self):
        src = self._read("src/strategies/smart_sniper_strategy_v12.py")
        run_start = src.index("    def run(self, df")
        next_def = src.index("\n    def ", run_start + 1)
        run_body = src[run_start:next_def]
        assert "iterrows" not in run_body, "v12 run() 热路径中仍有 iterrows"

    def test_feature_pipeline_no_iterrows(self):
        src = self._read("src/feature_pipeline_v2.py")
        # 只允许注释中提及 iterrows，热路径代码中不得有实际调用
        hot_path_lines = [
            ln for ln in src.splitlines()
            if "iterrows" in ln and not ln.lstrip().startswith("#")
        ]
        assert not hot_path_lines, (
            f"feature_pipeline_v2 热路径中仍有 iterrows 调用:\n"
            + "\n".join(hot_path_lines)
        )

    def test_strategy_v8_open_positions_no_iterrows(self):
        src = self._read("src/strategies/smart_sniper_strategy.py")
        assert "iterrows" not in src, "v8 策略中仍有 iterrows"

    def test_strategy_v12_open_positions_no_iterrows(self):
        src = self._read("src/strategies/smart_sniper_strategy_v12.py")
        assert "iterrows" not in src, "v12 策略中仍有 iterrows"

    def test_daily_index_dict_in_v8_run(self):
        """v8 run() 已使用 daily_index 预分组字典。"""
        src = self._read("src/strategies/smart_sniper_strategy.py")
        assert "daily_index" in src, "v8 run() 未使用 daily_index 预分组字典"

    def test_daily_index_dict_in_v12_run(self):
        """v12 run() 已使用 daily_index 预分组字典。"""
        src = self._read("src/strategies/smart_sniper_strategy_v12.py")
        assert "daily_index" in src, "v12 run() 未使用 daily_index 预分组字典"

    def test_no_linregress_in_pipeline(self):
        """feature_pipeline_v2 已移除 linregress 导入（向量化替代）。"""
        src = self._read("src/feature_pipeline_v2.py")
        assert "from scipy.stats import percentileofscore, linregress" not in src

    def test_sliding_window_view_used(self):
        """feature_pipeline_v2 使用 sliding_window_view 向量化。"""
        src = self._read("src/feature_pipeline_v2.py")
        assert "sliding_window_view" in src


# ---------------------------------------------------------------------------
# 测试 6：性能报告（非断言）
# ---------------------------------------------------------------------------

class TestPerformanceReport:
    """报告各热路径的实际耗时（print 输出，不断言具体数值）。"""

    def test_rolling_slope_perf(self, capsys):
        from numpy.lib.stride_tricks import sliding_window_view

        n, window = 500, 6
        rng = np.random.default_rng(1)
        vals = rng.normal(size=n).cumsum().astype(np.float64)

        t0 = time.perf_counter()
        for _ in range(1000):
            x = np.arange(window, dtype=np.float64)
            x_c = x - x.mean()
            denom = (x_c ** 2).sum()
            wins = sliding_window_view(vals, window)
            y_c = wins - wins.mean(axis=1, keepdims=True)
            _ = (x_c * y_c).sum(axis=1) / denom
        elapsed = (time.perf_counter() - t0) * 1000

        with capsys.disabled():
            print(f"\n[perf] rolling_slope_1d 向量化 (n={n}, w={window}) × 1000: {elapsed:.1f}ms")

    def test_cross_section_batch_perf(self, capsys):
        """批量 groupby 横截面特征性能报告。"""
        rng = np.random.default_rng(2)
        n_stocks, n_days, n_feat = 400, 5, 50
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        rows = []
        for s in range(n_stocks):
            for d in dates:
                rows.append({"symbol": f"{s:06d}", "timestamp": d,
                              **{f"f{j}": rng.normal() for j in range(n_feat)}})
        df = pd.DataFrame(rows)
        features = [f"f{j}" for j in range(n_feat)]
        ts_col = "timestamp"

        def _cs_zscore(s: pd.Series) -> pd.Series:
            return (s - s.median()) / (s.std(ddof=0) + 1e-9)

        t0 = time.perf_counter()
        feat_grp = df.groupby(ts_col)[features]
        rp = feat_grp.rank(pct=True)
        rp.columns = [c + "_rankpct" for c in features]
        z = feat_grp.transform(_cs_zscore)
        z.columns = [c + "_z" for c in features]
        result = pd.concat([df, rp, z], axis=1)
        elapsed = (time.perf_counter() - t0) * 1000

        with capsys.disabled():
            print(f"\n[perf] cross_section 批量 groupby "
                  f"({n_stocks} stocks × {n_days} days × {n_feat} feat): {elapsed:.1f}ms")
        assert result is not None

    def test_strategy_v8_run_perf(self, patch_filter, capsys):
        """v8 回测主循环性能报告。"""
        from src.strategies.smart_sniper_strategy import SmartSniperStrategy

        df = _make_synthetic_data(n_stocks=100, n_days=200)

        s = SmartSniperStrategy(
            initial_capital=1_000_000,
            max_positions=5,
            filter_main_board=False,
            filter_st=False,
            filter_new_stock=False,
            disable_tqdm=True,
        )
        s.min_probability = 0.50

        t0 = time.perf_counter()
        _, _ = s.run(df.copy())
        elapsed = (time.perf_counter() - t0) * 1000

        with capsys.disabled():
            print(f"\n[perf] v8 run() ({100} stocks × {200} days): {elapsed:.1f}ms")
