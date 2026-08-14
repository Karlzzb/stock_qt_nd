"""
tests/test_regression_divergence.py
背离检测器确定性回归测试（Issue #7 验收标准）。

测试缝：DivergenceDetectorV2.detect_daily_divergence（背离检测器批量入口）。
断言方式：固定输入 → 逐值断言输出（对比 tests/fixtures/baselines/divergence_detector.json）。

基准更新：当锚点漂移修复或检测逻辑合理变更后，需显式运行
    .venv/bin/python scripts/update_test_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

REPO_ROOT = Path(__file__).parent.parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DAILY = REPO_ROOT / "tests" / "fixtures" / "daily"
BASELINES_DIR = REPO_ROOT / "tests" / "fixtures" / "baselines"
BASELINE_PATH = BASELINES_DIR / "divergence_detector.json"


# ---------------------------------------------------------------------------
# 辅助：加载 fixture 并计算 MACD
# ---------------------------------------------------------------------------

def _load_with_macd(fixture_name: str) -> pd.DataFrame:
    """加载 OHLCV fixture，计算并附加 macd 列，返回带 DatetimeIndex 的 DataFrame。"""
    path = FIXTURES_DAILY / fixture_name
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    close = df["close"].values.astype(np.float64)
    macd_vals, _, _ = talib.MACD(close)
    df = df.copy()
    df["macd"] = macd_vals
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline() -> dict:
    """加载背离检测器基准 JSON。"""
    assert BASELINE_PATH.exists(), (
        f"基准文件不存在: {BASELINE_PATH}\n"
        "请运行: .venv/bin/python scripts/update_test_baselines.py"
    )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def divergence_data(baseline) -> pd.DataFrame:
    """加载 div_trigger fixture，附加 MACD 列。"""
    fixture_name = Path(baseline["fixture"]).name
    return _load_with_macd(fixture_name)


@pytest.fixture(scope="module")
def detection_result(divergence_data, baseline):
    """调用背离检测器，返回在 target_date 的检测结果。"""
    from src.divergence_detector_v2 import DivergenceDetectorV2

    detector = DivergenceDetectorV2()
    target_date = pd.Timestamp(baseline["target_date"]).date()
    symbol = baseline["symbol"]
    return detector.detect_daily_divergence(divergence_data, symbol, target_date)


# ---------------------------------------------------------------------------
# 基本健全性检查
# ---------------------------------------------------------------------------

class TestDivergenceDetectorBasic:
    def test_fixture_loads_offline(self):
        """fixture 数据离线可读（不依赖网络）。"""
        path = FIXTURES_DAILY / "div_trigger.parquet"
        assert path.exists(), f"fixture 文件不存在: {path}"
        df = pd.read_parquet(path)
        assert len(df) >= 100, "fixture 行数不足 100"

    def test_divergence_detected_at_target_date(self, detection_result, baseline):
        """在已知背离日 target_date 应至少检测到一个背离点。"""
        min_count = baseline.get("expected_count_min", 1)
        assert len(detection_result) >= min_count, (
            f"期望 >= {min_count} 个背离点，实际检测到 {len(detection_result)} 个"
        )

    def test_result_is_dataframe(self, detection_result):
        """检测结果类型为 DataFrame。"""
        assert isinstance(detection_result, pd.DataFrame)

    def test_result_has_required_columns(self, detection_result):
        """检测结果包含必需的字段。"""
        required = [
            "close_current", "close_previous",
            "macd_current", "macd_previous",
            "price_decline_pct", "macd_increase_pct",
        ]
        for col in required:
            assert col in detection_result.columns, (
                f"检测结果缺少字段: {col}"
            )

    def test_price_new_low_invariant(self, detection_result):
        """背离的核心不变量：所有检测到的点，close_current < close_previous。"""
        assert (detection_result["close_current"] < detection_result["close_previous"]).all(), (
            "检测到的背离点存在 close_current >= close_previous，违反背离定义"
        )

    def test_macd_higher_invariant(self, detection_result):
        """背离的核心不变量：所有检测到的点，macd_current > macd_previous。"""
        assert (detection_result["macd_current"] > detection_result["macd_previous"]).all(), (
            "检测到的背离点存在 macd_current <= macd_previous，违反背离定义"
        )

    def test_price_decline_pct_negative(self, detection_result):
        """price_decline_pct 应为负值（价格下跌）。"""
        assert (detection_result["price_decline_pct"] < 0).all(), (
            "price_decline_pct 应为负值"
        )

    def test_macd_increase_pct_positive(self, detection_result):
        """macd_increase_pct 应为正值（MACD 相对改善）。"""
        assert (detection_result["macd_increase_pct"] > 0).all(), (
            "macd_increase_pct 应为正值"
        )


# ---------------------------------------------------------------------------
# 确定性回归断言（逐值对比基准）
# ---------------------------------------------------------------------------

class TestDivergenceDetectorRegression:
    def test_baseline_file_exists(self):
        """基准文件必须存在。"""
        assert BASELINE_PATH.exists(), (
            "基准文件缺失，请运行 scripts/update_test_baselines.py"
        )

    def test_detection_count_at_least_expected(self, detection_result, baseline):
        """检测到的背离点数量不少于基准值。"""
        min_count = baseline.get("expected_count_min", 1)
        assert len(detection_result) >= min_count, (
            f"背离点数量回归：期望 >= {min_count}，实际 {len(detection_result)}"
        )

    @pytest.mark.parametrize(
        "field,expected",
        [
            pytest.param(k, v, id=k)
            for k, v in (
                json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
                .get("first_divergence", {})
                .items()
            )
            if v is not None
        ]
        if BASELINE_PATH.exists()
        else [],
    )
    def test_first_divergence_field_matches_baseline(
        self,
        field: str,
        expected: float,
        detection_result: pd.DataFrame,
        baseline: dict,
    ):
        """
        第一个背离点的各字段值与基准逐值对比。

        失败意味着背离检测逻辑（波谷检测、锚点、计算公式）发生了变化。
        如果变化是预期的（锚点漂移修复），请显式更新基准：
            .venv/bin/python scripts/update_test_baselines.py
        """
        abs_tol = baseline.get("abs_tol", 1e-5)
        assert not detection_result.empty, "检测结果为空，无法验证第一个背离点"

        first = detection_result.iloc[0]
        assert field in first.index, (
            f"字段 {field!r} 在检测结果中不存在"
        )

        actual = float(first[field])
        assert not np.isnan(actual), (
            f"字段 {field!r} 值为 NaN，期望 {expected}"
        )
        assert abs(actual - expected) <= abs_tol, (
            f"背离字段 {field!r} 回归：期望 {expected:.8f}，实际 {actual:.8f}，"
            f"偏差 {abs(actual - expected):.2e}（容差 {abs_tol:.0e}）\n"
            f"如果此变化是预期的，请运行 scripts/update_test_baselines.py 更新基准。"
        )

    def test_divergence_deterministic_across_calls(self, divergence_data, baseline):
        """多次调用同一输入，结果完全相同（确定性）。"""
        from src.divergence_detector_v2 import DivergenceDetectorV2

        detector = DivergenceDetectorV2()
        target_date = pd.Timestamp(baseline["target_date"]).date()
        symbol = baseline["symbol"]

        result1 = detector.detect_daily_divergence(divergence_data, symbol, target_date)
        result2 = detector.detect_daily_divergence(divergence_data, symbol, target_date)

        assert len(result1) == len(result2), "多次调用结果行数不同（非确定性）"
        if len(result1) > 0:
            for col in ["close_current", "macd_current", "price_decline_pct"]:
                if col in result1.columns:
                    pd.testing.assert_series_equal(
                        result1[col].reset_index(drop=True),
                        result2[col].reset_index(drop=True),
                        check_names=False,
                        rtol=1e-10,
                    )
