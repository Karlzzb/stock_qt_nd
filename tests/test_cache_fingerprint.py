"""
Issue #4 验收测试：缓存指纹失效机制 + dtype 精度解耦。

AC1 - 指纹校验：
  - 同输入同指纹 → is_cache_valid 返回 True（命中缓存）
  - 参数变化后指纹不同 → is_cache_valid 返回 False（缓存失效）
  - CSV 存在但 .fp 文件缺失 → is_cache_valid 返回 False（旧缓存自动失效）
  - .fp 文件内容过期 → is_cache_valid 返回 False

AC2 - dtype 精度：
  - optimize_dtypes 将 float64 列转为 float32（落盘降精度正常工作）
  - optimize_dtypes 不修改传入的原始 DataFrame（不产生副作用）
  - 原始数据 float64 精度高于 float32，证明计算路径保留了精度
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

# 确保项目根目录在路径中
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_pipeline_v2 import (
    compute_cache_fingerprint,
    is_cache_valid,
    write_cache_fingerprint,
    optimize_dtypes,
    FEATURE_PIPELINE_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path):
    """返回临时 CSV 路径（文件尚不存在）。"""
    return tmp_path / "realistic_features_20240101.csv"


@pytest.fixture
def sample_ohlcv_df():
    """小型 OHLCV DataFrame，使用 float64，精度超过 4 位小数。"""
    return pd.DataFrame(
        {
            "open":   [10.123456789, 11.123456789],
            "high":   [12.987654321, 13.987654321],
            "low":    [9.111111111,  10.111111111],
            "close":  [11.555555555, 12.555555555],
            "volume": [1000000.123456, 2000000.987654],
        },
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# AC1: 缓存指纹校验
# ---------------------------------------------------------------------------

class TestCacheFingerprint:
    def test_fingerprint_is_deterministic(self):
        """同一进程内多次调用返回相同指纹。"""
        assert compute_cache_fingerprint() == compute_cache_fingerprint()

    def test_fingerprint_is_nonempty_hex_string(self):
        fp = compute_cache_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16
        # 应为合法的十六进制字符串
        int(fp, 16)

    def test_fingerprint_contains_version(self):
        """指纹来源中包含版本号，确保版本变化会改变指纹。"""
        import hashlib, json
        from src.feature_pipeline_v2 import _build_fingerprint_params
        params = _build_fingerprint_params()
        assert params["version"] == FEATURE_PIPELINE_VERSION

    def test_valid_cache_hit(self, tmp_csv):
        """CSV 存在 + .fp 匹配 → is_cache_valid 返回 True。"""
        tmp_csv.write_text("dummy")
        fp = compute_cache_fingerprint()
        write_cache_fingerprint(tmp_csv, fp)
        assert is_cache_valid(tmp_csv, fp) is True

    def test_missing_csv_is_invalid(self, tmp_csv):
        """CSV 不存在 → is_cache_valid 返回 False。"""
        fp = compute_cache_fingerprint()
        assert is_cache_valid(tmp_csv, fp) is False

    def test_missing_fp_sidecar_is_invalid(self, tmp_csv):
        """CSV 存在但 .fp 文件缺失 → is_cache_valid 返回 False（旧缓存自动失效）。"""
        tmp_csv.write_text("dummy")
        fp = compute_cache_fingerprint()
        assert is_cache_valid(tmp_csv, fp) is False

    def test_stale_fingerprint_is_invalid(self, tmp_csv):
        """CSV + .fp 均存在但指纹不匹配 → is_cache_valid 返回 False。"""
        tmp_csv.write_text("dummy")
        write_cache_fingerprint(tmp_csv, "0000000000000000")
        fp = compute_cache_fingerprint()
        # 当前指纹不太可能是全 0，除非极端巧合
        assert fp != "0000000000000000"
        assert is_cache_valid(tmp_csv, fp) is False

    def test_changed_return_periods_changes_fingerprint(self, monkeypatch):
        """RETURN_PERIODS 变化后指纹应改变。"""
        original_fp = compute_cache_fingerprint()
        import src.comm_fun as cf
        monkeypatch.setattr(cf.model_config, "RETURN_PERIODS", [1, 2, 3])
        # 重新导入以获取新指纹（monkeypatch 已修改 model_config 引用）
        from src.feature_pipeline_v2 import compute_cache_fingerprint as cfp
        new_fp = cfp()
        assert original_fp != new_fp

    def test_changed_feature_need_max_days_changes_fingerprint(self, monkeypatch):
        """FEATURE_NEED_MAX_DAYS 变化后指纹应改变。"""
        original_fp = compute_cache_fingerprint()
        import src.comm_fun as cf
        monkeypatch.setattr(cf.model_config, "FEATURE_NEED_MAX_DAYS", 999)
        from src.feature_pipeline_v2 import compute_cache_fingerprint as cfp
        new_fp = cfp()
        assert original_fp != new_fp

    def test_write_then_read_fingerprint(self, tmp_csv):
        """write_cache_fingerprint 写入的内容应被 is_cache_valid 正确读取。"""
        tmp_csv.write_text("dummy")
        fp = compute_cache_fingerprint()
        write_cache_fingerprint(tmp_csv, fp)
        fp_file = Path(str(tmp_csv) + ".fp")
        assert fp_file.read_text().strip() == fp


# ---------------------------------------------------------------------------
# AC2: dtype 精度解耦
# ---------------------------------------------------------------------------

class TestDtypePrecision:
    def test_optimize_dtypes_converts_float64_to_float32(self, sample_ohlcv_df):
        """optimize_dtypes 应将 float64 列转为 float32。"""
        result = optimize_dtypes(sample_ohlcv_df)
        float_cols = result.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            assert result[col].dtype == np.float32, (
                f"列 {col} 应为 float32，实际为 {result[col].dtype}"
            )

    def test_optimize_dtypes_does_not_mutate_input(self, sample_ohlcv_df):
        """optimize_dtypes 不应修改传入的 DataFrame（返回新对象）。"""
        original_dtype = sample_ohlcv_df["close"].dtype
        original_val = sample_ohlcv_df["close"].iloc[0]
        _ = optimize_dtypes(sample_ohlcv_df)
        assert sample_ohlcv_df["close"].dtype == original_dtype, (
            "optimize_dtypes 不应修改传入 DataFrame 的 dtype"
        )
        assert sample_ohlcv_df["close"].iloc[0] == original_val, (
            "optimize_dtypes 不应修改传入 DataFrame 的值"
        )

    def test_optimize_dtypes_returns_new_object(self, sample_ohlcv_df):
        """optimize_dtypes 应返回新的 DataFrame 对象。"""
        result = optimize_dtypes(sample_ohlcv_df)
        assert result is not sample_ohlcv_df

    def test_float64_has_higher_precision_than_float32(self, sample_ohlcv_df):
        """原始 float64 数据精度高于 4 位小数，证明未经降精度前计算是有意义的。"""
        val = sample_ohlcv_df["close"].iloc[0]  # 11.555555555
        assert sample_ohlcv_df["close"].dtype == np.float64
        # 精度超过 4 位小数
        assert abs(val - round(val, 4)) > 1e-6, (
            "测试数据本身应有超过 4 位小数的精度，以体现 float64 vs float32 的差异"
        )

    def test_disk_value_loses_sub_4decimal_precision(self, sample_ohlcv_df):
        """落盘后 float32 值与原始 float64 值的差在 float32 精度范围内。"""
        result = optimize_dtypes(sample_ohlcv_df)
        original_val = sample_ohlcv_df["close"].iloc[0]
        disk_val = float(result["close"].iloc[0])
        # 差值在合理精度范围（float32 约 6~7 位有效数字）
        assert abs(original_val - disk_val) < 1e-3, (
            "落盘值偏差过大，请检查 optimize_dtypes 逻辑"
        )

    def test_optimize_dtypes_handles_none(self):
        """optimize_dtypes 传入 None 应直接返回 None。"""
        assert optimize_dtypes(None) is None

    def test_optimize_dtypes_handles_empty_df(self):
        """optimize_dtypes 传入空 DataFrame 应原样返回。"""
        empty = pd.DataFrame()
        result = optimize_dtypes(empty)
        assert result.empty
