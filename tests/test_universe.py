"""
universe.py 单元测试（Issue #5 验收标准）。

验证：
- 给定日期返回当日在市股票集合
- 退市股在退市当天及之后不出现
- 上市当天出现，上市前不出现
- 在市股票（delist_date 为空）始终出现
- 从 Parquet 序列化/反序列化后结果一致
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.universe import PointInTimeUniverse, _to_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """小型合成股票列表：
    - 000001.SZ : 长期在市，无退市日期
    - 000002.SZ : 在市（2010-01-01 上市，无退市）
    - 000999.SZ : 已退市（2015-06-01 上市，2020-03-15 退市）
    - 001001.SZ : 已退市（2005-01-01 上市，2019-12-31 退市）
    """
    return pd.DataFrame(
        {
            "ts_code":     ["000001.SZ", "000002.SZ", "000999.SZ", "001001.SZ"],
            "name":        ["平安银行",   "万科A",     "退市A",     "退市B"],
            "list_date":   ["19910403",   "20100101",  "20150601",  "20050101"],
            "delist_date": [None,         None,        "20200315",  "20191231"],
        }
    )


@pytest.fixture
def pit(sample_df) -> PointInTimeUniverse:
    return PointInTimeUniverse(sample_df)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

class TestToDates:
    def test_yyyymmdd_string(self):
        assert _to_date("20200101") == date(2020, 1, 1)

    def test_iso_string(self):
        assert _to_date("2020-01-01") == date(2020, 1, 1)

    def test_date_passthrough(self):
        d = date(2020, 6, 1)
        assert _to_date(d) is d

    def test_pandas_timestamp(self):
        assert _to_date(pd.Timestamp("2020-01-01")) == date(2020, 1, 1)


# ---------------------------------------------------------------------------
# 核心行为：get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    def test_live_stock_always_present(self, pit):
        """在市股票（delist_date=None）在任意查询日均出现。"""
        assert "000001.SZ" in pit.get_universe("2024-08-01")
        assert "000001.SZ" in pit.get_universe("1993-01-01")

    def test_stock_appears_on_list_date(self, pit):
        """上市当天出现。"""
        # 000002.SZ 上市于 2010-01-01
        assert "000002.SZ" in pit.get_universe("2010-01-01")

    def test_stock_absent_before_list_date(self, pit):
        """上市前不出现。"""
        assert "000002.SZ" not in pit.get_universe("2009-12-31")

    def test_delisted_stock_absent_on_delist_date(self, pit):
        """退市当天不出现（退市日 > query_date 才计入）。"""
        # 000999.SZ 退市于 2020-03-15
        assert "000999.SZ" not in pit.get_universe("2020-03-15")

    def test_delisted_stock_absent_after_delist_date(self, pit):
        """退市后不出现。"""
        assert "000999.SZ" not in pit.get_universe("2020-03-16")
        assert "000999.SZ" not in pit.get_universe("2024-01-01")

    def test_delisted_stock_present_before_delist_date(self, pit):
        """退市前一天仍出现。"""
        assert "000999.SZ" in pit.get_universe("2020-03-14")

    def test_full_universe_on_early_date(self, pit):
        """2004 年时只有 000001.SZ 在市（001001 2005 年上市，其余更晚）。"""
        universe = pit.get_universe("2004-06-01")
        assert "000001.SZ" in universe
        assert "001001.SZ" not in universe
        assert "000002.SZ" not in universe
        assert "000999.SZ" not in universe

    def test_all_live_on_date_between_listings(self, pit):
        """2016 年时 4 只股票中有 3 只在市（001001 在市、000999 在市、两只始终在市）。"""
        universe = pit.get_universe("2016-01-01")
        assert "000001.SZ" in universe
        assert "000002.SZ" in universe
        assert "000999.SZ" in universe
        assert "001001.SZ" in universe

    def test_returns_set_type(self, pit):
        result = pit.get_universe("2020-01-01")
        assert isinstance(result, set)

    def test_yyyymmdd_string_input(self, pit):
        """接受 YYYYMMDD 格式字符串。"""
        assert "000001.SZ" in pit.get_universe("20200101")

    def test_001001_absent_after_20191231(self, pit):
        """001001.SZ 退市于 2019-12-31，当天及之后不出现。"""
        assert "001001.SZ" not in pit.get_universe("2019-12-31")
        assert "001001.SZ" not in pit.get_universe("2020-01-01")

    def test_001001_present_before_delist(self, pit):
        assert "001001.SZ" in pit.get_universe("2019-12-30")


# ---------------------------------------------------------------------------
# get_universe_df
# ---------------------------------------------------------------------------

class TestGetUniverseDf:
    def test_returns_dataframe(self, pit):
        df = pit.get_universe_df("2020-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_df_columns(self, pit):
        df = pit.get_universe_df("2020-01-01")
        assert "ts_code" in df.columns

    def test_df_consistent_with_set(self, pit):
        d = "2020-01-01"
        codes_set = pit.get_universe(d)
        codes_df = set(pit.get_universe_df(d)["ts_code"].tolist())
        assert codes_set == codes_df


# ---------------------------------------------------------------------------
# 元信息属性
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_total_stocks(self, pit):
        assert pit.total_stocks == 4

    def test_live_stocks(self, pit):
        assert pit.live_stocks == 2

    def test_delisted_stocks(self, pit):
        assert pit.delisted_stocks == 2

    def test_repr(self, pit):
        r = repr(pit)
        assert "PointInTimeUniverse" in r
        assert "total=4" in r


# ---------------------------------------------------------------------------
# 序列化 / 反序列化
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_preserves_universe(self, pit, tmp_path):
        path = tmp_path / "universe.parquet"
        pit.save(path)
        assert path.exists()

        pit2 = PointInTimeUniverse.load(path)
        d = "2020-01-01"
        assert pit.get_universe(d) == pit2.get_universe(d)

    def test_save_and_load_preserves_delisted_exclusion(self, pit, tmp_path):
        path = tmp_path / "universe.parquet"
        pit.save(path)
        pit2 = PointInTimeUniverse.load(path)
        # 退市股退市后仍被排除
        assert "000999.SZ" not in pit2.get_universe("2020-12-01")
        assert "000999.SZ" in pit2.get_universe("2020-03-14")


# ---------------------------------------------------------------------------
# 边界条件与健壮性
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_delist_date_column(self):
        """DataFrame 完全没有 delist_date 列时，所有上市后股票均视为在市。"""
        df = pd.DataFrame(
            {"ts_code": ["A.SH", "B.SH"], "list_date": ["20000101", "20100101"]}
        )
        pit = PointInTimeUniverse(df)
        assert "A.SH" in pit.get_universe("2024-01-01")
        assert "B.SH" in pit.get_universe("2024-01-01")
        assert "B.SH" not in pit.get_universe("2009-12-31")

    def test_empty_df_returns_empty_set(self):
        df = pd.DataFrame({"ts_code": [], "list_date": [], "delist_date": []})
        pit = PointInTimeUniverse(df)
        assert pit.get_universe("2020-01-01") == set()

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"ts_code": ["A.SH"]})
        with pytest.raises(ValueError, match="list_date"):
            PointInTimeUniverse(df)

    def test_corrupt_list_date_row_dropped(self):
        """list_date 无法解析的行被静默丢弃，不影响其余行。"""
        df = pd.DataFrame(
            {
                "ts_code":     ["GOOD.SH", "BAD.SH"],
                "list_date":   ["20200101",  "not-a-date"],
                "delist_date": [None, None],
            }
        )
        pit = PointInTimeUniverse(df)
        assert "GOOD.SH" in pit.get_universe("2021-01-01")
        assert "BAD.SH" not in pit.get_universe("2021-01-01")
