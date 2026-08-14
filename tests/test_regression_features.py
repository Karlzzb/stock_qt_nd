"""
tests/test_regression_features.py
特征管线确定性回归测试（Issue #7 验收标准）。

测试缝：FeaturePipeline 批量特征计算入口（纯内存，不依赖文件 I/O）。
断言方式：固定输入 → 逐值断言输出（对比 tests/fixtures/baselines/feature_pipeline.json）。

基准更新：当泄露修复或特征逻辑合理变更后，需显式运行
    .venv/bin/python scripts/update_test_baselines.py
切勿在测试中放宽断言或在代码中跳过失败断言。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DAILY = REPO_ROOT / "tests" / "fixtures" / "daily"
BASELINES_DIR = REPO_ROOT / "tests" / "fixtures" / "baselines"
BASELINE_PATH = BASELINES_DIR / "feature_pipeline.json"


# ---------------------------------------------------------------------------
# 辅助：特征管线批量入口（测试缝）
# ---------------------------------------------------------------------------

def _build_long_df(stock_dict: dict) -> pd.DataFrame:
    """将 {symbol: DatetimeIndex OHLCV_DataFrame} 转换为多股票长格式 DataFrame。"""
    frames = []
    for symbol, df in stock_dict.items():
        tmp = df.reset_index().copy()
        date_col = tmp.columns[0]
        tmp.rename(columns={date_col: "timestamp"}, inplace=True)
        tmp["symbol"] = symbol
        frames.append(tmp)
    big = pd.concat(frames, ignore_index=True)
    big["timestamp"] = pd.to_datetime(big["timestamp"])
    return big.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def compute_features(stock_dict: dict) -> pd.DataFrame:
    """
    运行完整特征计算流水线，返回带所有特征的 DataFrame。

    这是 Issue #7 要求的「特征管线批量入口」测试缝：
    只关心外部行为（输入 OHLCV → 输出特征 DataFrame），不依赖文件 I/O。
    """
    from src.feature_pipeline_v2 import FeaturePipeline
    from src.divergence_detector_v2 import DivergenceDetectorV2

    pipeline = FeaturePipeline(
        divergence_detector=DivergenceDetectorV2(),
        full_stocks_inner={},
    )

    big = _build_long_df(stock_dict)
    big = pipeline._calculate_basic_technical_features(big)
    assert big is not None, "基础特征计算失败（数据量不足 100 行？）"

    big = pipeline._calculate_advance_technical_features(big)
    big = pipeline._generate_alpha_features(big)
    assert big is not None, "alpha 特征计算失败"

    big = pipeline.generate_structure_features(big)
    assert big is not None, "结构特征计算失败"

    big = pipeline.generate_lag_features(big)
    assert big is not None, "lag 特征计算失败"

    return big


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline() -> dict:
    """加载特征管线基准 JSON。"""
    assert BASELINE_PATH.exists(), (
        f"基准文件不存在: {BASELINE_PATH}\n"
        "请运行: .venv/bin/python scripts/update_test_baselines.py"
    )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def feature_result(baseline) -> pd.DataFrame:
    """加载 fixture、运行特征管线，返回计算结果（模块级缓存避免重复计算）。"""
    fixture_name = Path(baseline["fixture"]).name
    df = pd.read_parquet(FIXTURES_DAILY / fixture_name)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    symbol = baseline["symbol"]
    return compute_features({symbol: df})


@pytest.fixture(scope="module")
def target_row(feature_result, baseline) -> pd.Series:
    """返回基准中 target_date 对应的行。"""
    symbol = baseline["symbol"]
    target_date = pd.Timestamp(baseline["target_date"]).date()
    rows = feature_result[
        (feature_result["symbol"] == symbol)
        & (feature_result["timestamp"].dt.date == target_date)
    ]
    assert len(rows) == 1, (
        f"在 {target_date} 找到 {len(rows)} 行（期望 1 行）"
    )
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# 基本健全性检查
# ---------------------------------------------------------------------------

class TestFeaturePipelineBasic:
    def test_fixture_loads_offline(self):
        """fixture 数据离线可读（不依赖网络）。"""
        path = FIXTURES_DAILY / "000001.SZ.parquet"
        assert path.exists(), f"fixture 文件不存在: {path}"
        df = pd.read_parquet(path)
        assert len(df) >= 100, "fixture 行数不足 100"
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_pipeline_returns_dataframe(self, feature_result):
        """特征管线返回非空 DataFrame。"""
        assert isinstance(feature_result, pd.DataFrame)
        assert len(feature_result) > 0

    def test_symbol_column_preserved(self, feature_result):
        """输出保留 symbol 列（不被 groupby.apply 丢弃）。"""
        assert "symbol" in feature_result.columns

    def test_timestamp_column_present(self, feature_result):
        """输出包含 timestamp 列。"""
        assert "timestamp" in feature_result.columns

    def test_basic_talib_features_present(self, feature_result):
        """talib 基础技术指标列存在。"""
        for col in ["rsi_14", "macd", "ma_20", "obv", "atr"]:
            assert col in feature_result.columns, f"缺少列: {col}"

    def test_structure_features_present(self, feature_result):
        """结构特征列存在。"""
        for col in ["vol_gk", "illiq", "ret_overnight", "ret_intraday"]:
            assert col in feature_result.columns, f"缺少列: {col}"

    def test_lag_features_present(self, feature_result):
        """lag 特征列存在。"""
        for col in ["close_lag_5", "volume_lag_10", "return_lag_1"]:
            assert col in feature_result.columns, f"缺少列: {col}"

    def test_no_all_nan_basic_columns(self, feature_result):
        """基础列不能全为 NaN（至少有一行有效值）。"""
        for col in ["rsi_14", "macd", "ma_20"]:
            assert feature_result[col].notna().any(), f"列 {col} 全为 NaN"


# ---------------------------------------------------------------------------
# 确定性回归断言（逐值对比基准）
# ---------------------------------------------------------------------------

class TestFeaturePipelineRegression:
    def test_baseline_file_exists(self):
        """基准文件必须存在（防止静默跳过）。"""
        assert BASELINE_PATH.exists(), (
            "基准文件缺失，请运行 scripts/update_test_baselines.py"
        )

    def test_all_asserted_features_exist(self, feature_result, baseline):
        """基准中列出的所有特征在输出 DataFrame 中均存在。"""
        for assertion in baseline["assertions"]:
            feat = assertion["feature"]
            assert feat in feature_result.columns, (
                f"基准断言特征 {feat!r} 在输出中不存在"
            )

    @pytest.mark.parametrize(
        "feat,expected,abs_tol",
        [
            pytest.param(
                a["feature"],
                a["value"],
                # 不同量级的特征用不同容差：大数值用相对容差思路但保持绝对 tol
                1e-3 if abs(a["value"]) > 100 else 1e-4,
                id=a["feature"],
            )
            for a in json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["assertions"]
            if BASELINE_PATH.exists()
        ]
        if BASELINE_PATH.exists()
        else [],
    )
    def test_feature_value_matches_baseline(
        self,
        feat: str,
        expected: float,
        abs_tol: float,
        target_row: pd.Series,
    ):
        """
        固定输入 → 特征值与基准逐值对比。

        失败意味着特征逻辑或数据发生了变化。
        如果变化是预期的（泄露修复、算法改进），请显式更新基准：
            .venv/bin/python scripts/update_test_baselines.py
        """
        actual = float(target_row[feat])
        assert not np.isnan(actual), (
            f"特征 {feat!r} 值为 NaN，期望 {expected}"
        )
        assert abs(actual - expected) <= abs_tol, (
            f"特征 {feat!r} 回归：期望 {expected:.8f}，实际 {actual:.8f}，"
            f"偏差 {abs(actual - expected):.2e}（容差 {abs_tol:.0e}）\n"
            f"如果此变化是预期的，请运行 scripts/update_test_baselines.py 更新基准。"
        )
