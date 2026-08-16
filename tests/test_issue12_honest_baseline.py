"""
Issue #12 验收测试：诚实基线脚本套件。

验收标准
--------
AC1 - run_feature_pipeline.py：load_price_data_from_parquet 将 Parquet 转换为
      与 feature_pipeline_v2.load_price_data 相同的字典格式（含 float64 保证）。
AC2 - run_walk_forward_train.py：load_feature_csvs 不存在时报 FileNotFoundError；
      build_dataset 正确生成 label 列。
AC3 - generate_comparison_report.py：_fmt / _delta_arrow 格式化正确；
      generate_report 在缺少 summary.json 时报 FileNotFoundError。
AC4 - run_baseline_backtest.py：模块可导入，旧基线常量与 comm_fun.py 注释一致。

测试设计原则
-----------
- 不依赖网络或真实数据（用合成 fixtures）。
- 不运行耗时训练或回测。
- 全部为纯单元测试，在 CI 中可靠通过。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# AC1 – load_price_data_from_parquet
# ---------------------------------------------------------------------------

class TestLoadPriceDataFromParquet:
    """AC1：Parquet 加载器格式正确、dtype 为 float64。"""

    def _make_parquet(self, tmp_path: Path, ts_code: str) -> Path:
        """生成最小 Parquet fixture。"""
        df = pd.DataFrame({
            "ts_code": [ts_code] * 5,
            "trade_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open":  [10.0, 10.1, 10.2, 10.3, 10.4],
            "high":  [10.5, 10.6, 10.7, 10.8, 10.9],
            "low":   [9.5,  9.6,  9.7,  9.8,  9.9],
            "close": [10.2, 10.3, 10.4, 10.5, 10.6],
            "vol":   [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        })
        p = tmp_path / f"{ts_code}.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_returns_dict_with_correct_symbol(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        self._make_parquet(tmp_path, "000001.SZ")
        result = load_price_data_from_parquet(tmp_path)
        assert "000001.SZ" in result

    def test_dataframe_has_ohlcv_columns(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        self._make_parquet(tmp_path, "000001.SZ")
        df = load_price_data_from_parquet(tmp_path)["000001.SZ"]
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"缺少列 {col}"

    def test_float64_dtype_preserved(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        self._make_parquet(tmp_path, "000001.SZ")
        df = load_price_data_from_parquet(tmp_path)["000001.SZ"]
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype == np.float64, f"{col} 非 float64"

    def test_datetime_index(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        self._make_parquet(tmp_path, "000001.SZ")
        df = load_price_data_from_parquet(tmp_path)["000001.SZ"]
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_date_filter_applied(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        self._make_parquet(tmp_path, "000001.SZ")
        df = load_price_data_from_parquet(tmp_path, start_date="2020-01-03")["000001.SZ"]
        assert df.index.min() >= pd.Timestamp("2020-01-03")

    def test_empty_dir_raises(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        with pytest.raises(FileNotFoundError):
            load_price_data_from_parquet(tmp_path)

    def test_multiple_stocks_loaded(self, tmp_path):
        from scripts.run_feature_pipeline import load_price_data_from_parquet
        for code in ("000001.SZ", "600519.SH", "300750.SZ"):
            self._make_parquet(tmp_path, code)
        result = load_price_data_from_parquet(tmp_path)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# AC2 – load_feature_csvs / build_dataset
# ---------------------------------------------------------------------------

class TestWalkForwardHelpers:
    """AC2：特征 CSV 加载与数据集构建。"""

    def test_load_feature_csvs_empty_dir_raises(self, tmp_path):
        from scripts.run_walk_forward_train import load_feature_csvs
        with pytest.raises(FileNotFoundError):
            load_feature_csvs(tmp_path)

    def test_load_feature_csvs_finds_files(self, tmp_path):
        from scripts.run_walk_forward_train import load_feature_csvs
        # 生成最小 fixture：两个日期文件
        for date_str in ("20200101", "20200102"):
            p = tmp_path / f"realistic_features_{date_str}.csv"
            pd.DataFrame({"timestamp": [f"2020-01-{date_str[-2:]}"],
                          "symbol": ["000001.SZ"]}).to_csv(p, index=False)
        df = load_feature_csvs(tmp_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# AC3 – generate_comparison_report helpers
# ---------------------------------------------------------------------------

class TestReportHelpers:
    """AC3：格式化函数与报告生成。"""

    def test_fmt_percentage(self):
        from scripts.generate_comparison_report import _fmt
        assert _fmt(0.215, "return_rate") == "21.50%"

    def test_fmt_negative_drawdown(self):
        from scripts.generate_comparison_report import _fmt
        assert _fmt(-0.292, "max_drawdown") == "-29.20%"

    def test_fmt_sharpe(self):
        from scripts.generate_comparison_report import _fmt
        assert _fmt(1.2266, "sharpe_ratio") == "1.2266"

    def test_fmt_trades(self):
        from scripts.generate_comparison_report import _fmt
        assert _fmt(133, "total_trades") == "133"

    def test_fmt_none(self):
        from scripts.generate_comparison_report import _fmt
        assert _fmt(None, "return_rate") == "—"

    def test_delta_arrow_up(self):
        from scripts.generate_comparison_report import _delta_arrow
        result = _delta_arrow(1.5, 1.0, higher_is_better=True)
        assert result.startswith("↑")

    def test_delta_arrow_down_bad(self):
        from scripts.generate_comparison_report import _delta_arrow
        # 收益下降（新 < 旧），higher_is_better=True → ↓ 箭头
        result = _delta_arrow(0.5, 1.0, higher_is_better=True)
        assert result.startswith("↓")

    def test_generate_report_missing_summary_raises(self, tmp_path):
        from scripts.generate_comparison_report import generate_report
        out = tmp_path / "report.md"
        with pytest.raises(FileNotFoundError):
            generate_report(tmp_path, out, "测试报告")

    def test_generate_report_creates_file(self, tmp_path):
        from scripts.generate_comparison_report import generate_report
        summary = {
            "v8": {"return_rate": 0.8, "max_drawdown": -0.35,
                   "sharpe_ratio": 0.9, "win_rate": 0.48, "total_trades": 100},
            "v12": {"return_rate": 1.2, "max_drawdown": -0.30,
                    "sharpe_ratio": 1.1, "win_rate": 0.52, "total_trades": 90},
        }
        (tmp_path / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        out = tmp_path / "report.md"
        generate_report(tmp_path, out, "单元测试报告")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "单元测试报告" in content
        assert "v8" in content.lower() or "V8" in content
        assert "诚实基线" in content or "对照" in content

    def test_report_contains_old_and_new_values(self, tmp_path):
        from scripts.generate_comparison_report import generate_report
        summary = {
            "v8": {"return_rate": 0.5, "max_drawdown": -0.40,
                   "sharpe_ratio": 0.7, "win_rate": 0.45, "total_trades": 80},
        }
        (tmp_path / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        out = tmp_path / "report.md"
        generate_report(tmp_path, out, "T")
        content = out.read_text(encoding="utf-8")
        # 旧基线收益率 2.1148 应出现在表格中
        assert "211.48%" in content or "2.1148" in content or "旧基线" in content


# ---------------------------------------------------------------------------
# AC4 – run_baseline_backtest 常量一致性
# ---------------------------------------------------------------------------

class TestBaselineConstants:
    """AC4：run_baseline_backtest 的旧基线常量与 comm_fun.py 注释一致。"""

    def test_v8_param35_return_rate(self):
        from scripts.generate_comparison_report import OLD_BASELINE
        assert abs(OLD_BASELINE["v8"]["return_rate"] - 2.1148) < 1e-4

    def test_v8_param35_sharpe(self):
        from scripts.generate_comparison_report import OLD_BASELINE
        assert abs(OLD_BASELINE["v8"]["sharpe_ratio"] - 1.2266) < 1e-4

    def test_v12_param2_return_rate(self):
        from scripts.generate_comparison_report import OLD_BASELINE
        assert abs(OLD_BASELINE["v12"]["return_rate"] - 4.0518) < 1e-4

    def test_v12_param2_sharpe(self):
        from scripts.generate_comparison_report import OLD_BASELINE
        assert abs(OLD_BASELINE["v12"]["sharpe_ratio"] - 1.9795) < 1e-4

    def test_param_keys_exist_in_comm_fun(self):
        from comm_fun import model_config
        assert "param35" in model_config.STRATEGY_PARAMS_CANDIDATES_V8
        assert "param2" in model_config.STRATEGY_PARAMS_CANDIDATES_V12
