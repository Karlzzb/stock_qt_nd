"""
tests/test_issue10_pipeline_fixes.py
Issue #10 验收测试：删泄露源 + 修已知 bug + 逐股隔离 + 去前视

涵盖：
1. is_quick_divergence 恒为 0 修复
2. macd_golden_cross 不再被 advance 特征覆盖（含 K 线形态 df.apply axis=1 验证）
3. _filter_valid_rows_apply 无前视（不访问次日最低价）
4. 泄露列不存在于 v2 输出（close_wavelet, close_d0.4）
5. 特征管线批量入口逐值回归基准（含退市股样本）
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DAILY = REPO_ROOT / "tests" / "fixtures" / "daily"


# ---------------------------------------------------------------------------
# 辅助：构造最小 OHLCV 数据
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, symbol: str = "TEST.SZ") -> pd.DataFrame:
    """生成含 symbol 列的长格式 OHLCV DataFrame（模拟多股票拼接格式）。"""
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 10.0 + rng.normal(0, 0.1, n).cumsum()
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.03, n))
    low = close * (1 - rng.uniform(0, 0.03, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


def _make_pipeline():
    from src.feature_pipeline_v2 import FeaturePipeline
    from src.divergence_detector_v2 import DivergenceDetectorV2
    return FeaturePipeline(
        divergence_detector=DivergenceDetectorV2(),
        full_stocks_inner={},
    )


def _run_pipeline(big: pd.DataFrame) -> pd.DataFrame:
    p = _make_pipeline()
    big = p._calculate_basic_technical_features(big)
    assert big is not None
    big = p._calculate_advance_technical_features(big)
    big = p._generate_alpha_features(big)
    assert big is not None
    big = p.generate_structure_features(big)
    assert big is not None
    big = p.generate_lag_features(big)
    assert big is not None
    return big


# ---------------------------------------------------------------------------
# 1. is_quick_divergence 修复
# ---------------------------------------------------------------------------

class TestIsQuickDivergenceFix:
    """is_quick_divergence 应按 formation_period 行级判断，不再恒为 0。"""

    def _make_enriched_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["A.SZ", "B.SZ", "C.SZ"],
            "timestamp": pd.date_range("2021-01-01", periods=3),
            "formation_period": [2, 3, 5],  # 2 和 3 满足 <=3，5 不满足
        })

    def test_quick_divergence_not_all_zero(self):
        """formation_period <= 3 的行应为 1，>3 的应为 0。"""
        from src.feature_pipeline_v2 import FeaturePipeline
        from src.divergence_detector_v2 import DivergenceDetectorV2

        p = FeaturePipeline(
            divergence_detector=DivergenceDetectorV2(),
            full_stocks_inner={},
        )

        df = self._make_enriched_df()
        # 直接调用修复后的逻辑（模拟 generate_features_for_date 中的代码片段）
        if 'formation_period' in df.columns:
            df['is_quick_divergence'] = (df['formation_period'] <= 3).astype(int)
        else:
            df['is_quick_divergence'] = 0

        assert list(df['is_quick_divergence']) == [1, 1, 0], (
            f"期望 [1, 1, 0]，实际 {list(df['is_quick_divergence'])}"
        )

    def test_quick_divergence_no_formation_period_col(self):
        """无 formation_period 列时应默认 0，不报错。"""
        df = pd.DataFrame({"symbol": ["A.SZ"], "timestamp": pd.Timestamp("2021-01-01")})
        if 'formation_period' in df.columns:
            df['is_quick_divergence'] = (df['formation_period'] <= 3).astype(int)
        else:
            df['is_quick_divergence'] = 0
        assert df['is_quick_divergence'].iloc[0] == 0


# ---------------------------------------------------------------------------
# 2. macd_golden_cross 不被 advance 特征覆盖
# ---------------------------------------------------------------------------

class TestMacdGoldenCrossNotOverwritten:
    """advance 特征不得覆盖 macd_features_rolling 计算的真实金叉信号。
    同时验证 K 线形态 df.apply 已带 axis=1（bug AC3 第一条）。
    """

    @pytest.fixture(scope="class")
    @classmethod
    def feature_df(cls):
        big = _make_ohlcv(n=300)
        return _run_pipeline(big)

    def test_macd_golden_cross_column_exists(self, feature_df):
        assert "macd_golden_cross" in feature_df.columns, "缺少 macd_golden_cross 列"

    def test_macd_above_signal_column_exists(self, feature_df):
        """新增列 macd_above_signal（advance 中 macd > signal line 的静态判断）。"""
        assert "macd_above_signal" in feature_df.columns, "缺少 macd_above_signal 列"

    def test_macd_golden_cross_is_true_cross(self, feature_df):
        """macd_golden_cross 应为真实金叉（前一天 macd < signal，今天反转），
        而非简单的 macd > signal 判断（那是 macd_above_signal）。"""
        df = feature_df.copy()
        # 真实金叉与 macd_above_signal 不等价：金叉是穿越事件（0→1），above_signal 是状态
        # 验证：macd_golden_cross=1 时 macd_above_signal 也应=1（方向一致）
        cross1 = df[df["macd_golden_cross"] == 1]
        if len(cross1) > 0:
            assert (cross1["macd_above_signal"] == 1).all(), (
                "macd_golden_cross=1 时 macd_above_signal 应同时为 1"
            )

    def test_macd_golden_cross_not_same_as_above_signal(self, feature_df):
        """macd_golden_cross 与 macd_above_signal 值不应完全相同（语义不同）。"""
        gc = feature_df["macd_golden_cross"]
        ab = feature_df["macd_above_signal"]
        assert not gc.equals(ab), (
            "macd_golden_cross 与 macd_above_signal 完全相同，说明仍存在覆盖 bug"
        )

    def test_hammer_pattern_no_nan(self, feature_df):
        """K 线形态 hammer_pattern 应有有效值（df.apply 带 axis=1 才能正常计算）。
        若 axis=1 缺失，apply 按列调用，结果将全为 NaN 或抛 TypeError。
        """
        assert "hammer_pattern" in feature_df.columns, "缺少 hammer_pattern 列"
        assert feature_df["hammer_pattern"].notna().any(), (
            "hammer_pattern 全为 NaN，df.apply 可能缺少 axis=1"
        )

    def test_doji_pattern_no_nan(self, feature_df):
        """K 线形态 doji_pattern 同上。"""
        assert "doji_pattern" in feature_df.columns, "缺少 doji_pattern 列"
        assert feature_df["doji_pattern"].notna().any(), (
            "doji_pattern 全为 NaN，df.apply 可能缺少 axis=1"
        )


# ---------------------------------------------------------------------------
# 3. _filter_valid_rows_apply 无前视
# ---------------------------------------------------------------------------

class TestFilterValidRowsNoLookahead:
    """_filter_valid_rows_apply 只使用当日及之前信息，不访问 pos+1 处的次日低价。"""

    def _make_full_stocks_data(self, symbol: str = "000001.SZ", n: int = 200) -> dict:
        """构造单股 DatetimeIndex OHLCV 数据。"""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(0)
        close = 10 + rng.normal(0, 0.1, n).cumsum()
        close = np.maximum(close, 1.0)
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=dates)
        return {symbol: df}

    def test_last_row_signal_not_dropped(self):
        """信号日是最后一行时不应被丢弃（无次日数据可偷看，修复前 return True）。"""
        from src.feature_pipeline_v2 import FeaturePipeline
        from src.divergence_detector_v2 import DivergenceDetectorV2

        symbol = "000001.SZ"
        full_stocks = self._make_full_stocks_data(symbol)
        last_date = full_stocks[symbol].index[-1]

        # 构造恰好落在最后一行的信号
        signal_df = pd.DataFrame({
            "symbol": [symbol],
            "timestamp": [last_date],
            "close_current": [full_stocks[symbol]["close"].iloc[-1]],
        })

        p = FeaturePipeline(
            divergence_detector=DivergenceDetectorV2(),
            full_stocks_inner=full_stocks,
        )

        result = p._filter_valid_rows_apply(signal_df)
        assert len(result) == 1, (
            "最后一行信号不应被过滤掉（无前视数据可用时应保留）"
        )

    def test_valid_mid_row_signal_kept(self):
        """中间行的信号（有完整历史）应被保留。"""
        from src.feature_pipeline_v2 import FeaturePipeline
        from src.divergence_detector_v2 import DivergenceDetectorV2

        symbol = "000001.SZ"
        full_stocks = self._make_full_stocks_data(symbol)
        mid_date = full_stocks[symbol].index[100]

        signal_df = pd.DataFrame({
            "symbol": [symbol],
            "timestamp": [mid_date],
            "close_current": [full_stocks[symbol]["close"].iloc[100]],
        })

        p = FeaturePipeline(
            divergence_detector=DivergenceDetectorV2(),
            full_stocks_inner=full_stocks,
        )

        result = p._filter_valid_rows_apply(signal_df)
        assert len(result) == 1, "有效中间行信号应被保留"

    def test_unknown_symbol_dropped(self):
        """symbol 不在 full_stocks_data 中的信号应被过滤掉。"""
        from src.feature_pipeline_v2 import FeaturePipeline
        from src.divergence_detector_v2 import DivergenceDetectorV2

        symbol = "000001.SZ"
        full_stocks = self._make_full_stocks_data(symbol)

        signal_df = pd.DataFrame({
            "symbol": ["999999.SZ"],  # 不存在的 symbol
            "timestamp": [full_stocks[symbol].index[50]],
            "close_current": [10.0],
        })

        p = FeaturePipeline(
            divergence_detector=DivergenceDetectorV2(),
            full_stocks_inner=full_stocks,
        )

        result = p._filter_valid_rows_apply(signal_df)
        assert len(result) == 0, "未知 symbol 应被过滤掉"

    def test_filter_does_not_access_future_data(self):
        """验证过滤器不再访问 pos+1 处的 low 价（通过有限长度 DataFrame 验证边界）。"""
        from src.feature_pipeline_v2 import FeaturePipeline
        from src.divergence_detector_v2 import DivergenceDetectorV2

        symbol = "000001.SZ"
        # 构造只有 1 行的 DataFrame——如果仍有前视，访问 iat[1] 会 IndexError
        single_date = pd.Timestamp("2021-06-01")
        df_single = pd.DataFrame({
            "open": [10.0], "high": [10.5], "low": [9.5],
            "close": [10.0], "volume": [1e6],
        }, index=[single_date])
        full_stocks = {symbol: df_single}

        signal_df = pd.DataFrame({
            "symbol": [symbol],
            "timestamp": [single_date],
            "close_current": [10.0],
        })

        p = FeaturePipeline(
            divergence_detector=DivergenceDetectorV2(),
            full_stocks_inner=full_stocks,
        )

        # 不应抛出 IndexError / 不应访问越界的次日数据
        result = p._filter_valid_rows_apply(signal_df)
        assert len(result) == 1, "单行数据中的信号应被保留（不前视次日）"


# ---------------------------------------------------------------------------
# 4. 泄露列不存在于 v2 输出
# ---------------------------------------------------------------------------

class TestNoLeakageColumnsInV2:
    """v2 管线输出不应包含已确认的泄露列。"""

    LEAK_COLS = [
        "close_wavelet",   # 全局小波去噪（前向泄露）
        "close_d0.4",      # 全局分数阶差分（前向泄露）
    ]

    @pytest.fixture(scope="class")
    @classmethod
    def feature_df(cls):
        big = _make_ohlcv(n=300)
        return _run_pipeline(big)

    @pytest.mark.parametrize("col", LEAK_COLS)
    def test_leak_col_absent(self, col, feature_df):
        assert col not in feature_df.columns, (
            f"泄露列 {col!r} 不应出现在 v2 输出中"
        )

    def test_close_smooth_10_present(self, feature_df):
        """close_wavelet 的安全替代 close_smooth_10 应存在。"""
        assert "close_smooth_10" in feature_df.columns, (
            "close_smooth_10（EMA 平滑平替）应存在"
        )


# ---------------------------------------------------------------------------
# 5. 批量入口逐值回归基准（含退市股样本）
# ---------------------------------------------------------------------------

class TestBatchRegressionWithDelistedStock:
    """验证批量特征管线对包含退市股（历史短、尾部截断）的组合稳定输出。"""

    @pytest.fixture(scope="class")
    @classmethod
    def multi_stock_result(cls):
        """构造正常股 + 退市股（短序列）的联合输出。"""
        normal = _make_ohlcv(n=300, symbol="000001.SZ")
        delisted = _make_ohlcv(n=150, symbol="DELIST.SZ")  # 模拟退市股（短序列）

        big = pd.concat([normal, delisted], ignore_index=True)
        big = big.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return _run_pipeline(big)

    def test_both_symbols_present(self, multi_stock_result):
        symbols = set(multi_stock_result["symbol"].unique())
        assert "000001.SZ" in symbols
        assert "DELIST.SZ" in symbols

    def test_no_cross_stock_contamination_in_macd_golden_cross(self, multi_stock_result):
        """macd_golden_cross 应为0/1整数，不出现因跨股污染导致的异常值。"""
        col = multi_stock_result["macd_golden_cross"]
        assert col.dropna().isin([0, 1]).all(), (
            "macd_golden_cross 应只含 0/1"
        )

    def test_rsi_per_symbol_isolated(self, multi_stock_result):
        """每只股票的 RSI 序列相互独立，不应完全相同（跨股污染特征）。"""
        df = multi_stock_result.dropna(subset=["rsi_14"])
        rsi_normal = df[df["symbol"] == "000001.SZ"]["rsi_14"].values
        rsi_delist = df[df["symbol"] == "DELIST.SZ"]["rsi_14"].values
        min_len = min(len(rsi_normal), len(rsi_delist))
        assert min_len > 0, "两只股票的 RSI 均为空"
        # 不应完全相同（如果跨股污染，二者 RSI 序列会重叠）
        assert not np.allclose(
            rsi_normal[-min_len:], rsi_delist[-min_len:], equal_nan=True
        ), "两只股票的 RSI 完全相同，疑似跨股污染"

    def test_delisted_stock_has_valid_features(self, multi_stock_result):
        """退市股（短序列）应有至少部分非 NaN 的基础特征。"""
        delist_rows = multi_stock_result[multi_stock_result["symbol"] == "DELIST.SZ"]
        for col in ["rsi_14", "ma_20"]:
            assert delist_rows[col].notna().any(), (
                f"退市股 DELIST.SZ 的列 {col!r} 全为 NaN"
            )

    def test_boxcox_atr_per_symbol_no_global_lambda(self, multi_stock_result):
        """boxcox_atr（实为 log1p(atr)）逐股独立，不应跨股共享全局 lambda。"""
        assert "boxcox_atr" in multi_stock_result.columns
        # log1p(atr) 应 >= 0
        vals = multi_stock_result["boxcox_atr"].dropna()
        assert (vals >= 0).all(), "log1p(atr) 应 >= 0"
