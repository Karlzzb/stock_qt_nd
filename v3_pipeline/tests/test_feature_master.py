#!/usr/bin/env python3
"""feature_master 的单元测试（接缝: 三来源特征 -> 事件×特征主表）。

只验外部行为:
  - 泄漏排除模式与白名单;
  - 三源合并的键对齐与碰撞处理（值同去重、值异报错）;
  - 段标签口径;
  - 精确成对相关与 |ρ|>=0.999 去重数学。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_master as fm  # noqa: E402


def _events():
    return pd.DataFrame({
        "event_id": [0, 1, 2],
        "ts_code": ["A", "A", "B"],
        "date": pd.to_datetime(["2010-05-04", "2020-03-02", "2023-01-09"]),
        "sig_idx": [100, 200, 150],
        "low_date": [19000, 19500, 19800],
        "prev_low_date": [18900, 19400, 19700],
        "compare_rank": [1, 2, 1],
        "formation": [5, 8, 3],
        "regime": ["up", "up", "up"],
        "above_ma200": [True, True, False],
    })


def _s1():
    return pd.DataFrame({
        "event_id": [0, 1, 2],
        "ts_code": ["A", "A", "B"],
        "date": pd.to_datetime(["2010-05-04", "2020-03-02", "2023-01-09"]),
        "sig_idx": [100, 200, 150],
        "ATRN": [0.03, 0.04, 0.05],
        "RET5": [0.01, -0.02, 0.03],
    })


def _s2():
    return pd.DataFrame({
        "ts_code": ["A", "A", "B"],
        "date": pd.to_datetime(["2010-05-04", "2020-03-02", "2023-01-09"]),
        "MEAN_OPEN_3": [10.0, 11.0, 12.0],
    })


# ---------------------------------------------------------------- 泄漏排除
def test_exclusion_patterns():
    cols = ["stop_loss_return_3d", "future_return_5d", "label_x", "mfr_return_3d",
            "rank_future_return_3d", "open_exec_return_5d", "hit_N20_k2.0",
            "dyn_exit", "ATRN", "rank_return", "rank_volume", "close_rankpct"]
    bad = fm.excluded_columns(cols)
    assert set(bad) == {"stop_loss_return_3d", "future_return_5d", "label_x",
                        "mfr_return_3d", "rank_future_return_3d",
                        "open_exec_return_5d", "hit_N20_k2.0", "dyn_exit"}
    # 白名单与 _rankpct 后缀不受影响
    fm.assert_no_leakage(["ATRN", "rank_return", "rank_volume", "close_rankpct",
                          "close_z"])
    with pytest.raises(AssertionError):
        fm.assert_no_leakage(["stop_loss_return_3d"])


# ---------------------------------------------------------------- 段标签
def test_segment_of():
    dates = pd.to_datetime(["1999-06-01", "2010-05-04", "2018-12-28",
                            "2020-03-02", "2022-10-31", "2023-01-09"])
    seg = fm.segment_of(dates)
    assert list(seg) == ["pre2001", "train", "embargo", "val", "embargo", "test"]


# ---------------------------------------------------------------- 合并
def test_merge_sources_basic():
    df, src_of, coll = fm.merge_sources(_events(), _s1(), _s2(), None)
    assert len(df) == 3
    assert df["ATRN"].notna().all() and df["MEAN_OPEN_3"].notna().all()
    assert src_of["ATRN"] == "s1" and src_of["MEAN_OPEN_3"] == "s2"
    assert coll == []
    assert "seg" in df.columns


def test_merge_collision_same_values_dedup():
    s2 = _s2().rename(columns={"MEAN_OPEN_3": "ATRN"})
    s2["ATRN"] = [0.03, 0.04, 0.05]  # 与 s1 完全相同
    df, src_of, coll = fm.merge_sources(_events(), _s1(), s2, None)
    assert df["ATRN"].tolist() == [0.03, 0.04, 0.05]
    assert coll == [{"column": "ATRN", "kept_source": "s1", "dropped_source": "s2"}]


def test_merge_collision_different_values_raises():
    s2 = _s2().rename(columns={"MEAN_OPEN_3": "ATRN"})
    s2["ATRN"] = [9.9, 9.9, 9.9]
    with pytest.raises(ValueError, match="列名碰撞"):
        fm.merge_sources(_events(), _s1(), s2, None)


def test_merge_collision_nan_asymmetry_raises():
    """一侧 NaN、一侧有值 = 值异（不允许静默通过）。"""
    s2 = _s2().rename(columns={"MEAN_OPEN_3": "ATRN"})
    s2["ATRN"] = [np.nan, 0.04, 0.05]  # 第一行 s1 有值、s2 缺失
    with pytest.raises(ValueError, match="列名碰撞"):
        fm.merge_sources(_events(), _s1(), s2, None)
    # 两侧同位置同 NaN = 值同, 去重不报错
    s2b = _s2().rename(columns={"MEAN_OPEN_3": "ATRN"})
    s2b["ATRN"] = [np.nan, 0.04, 0.05]
    s1b = _s1()
    s1b.loc[0, "ATRN"] = np.nan
    df, _, coll = fm.merge_sources(_events(), s1b, s2b, None)
    assert len(coll) == 1 and df["ATRN"].isna().sum() == 1


def test_merge_missing_rows_give_nan():
    s2 = _s2().iloc[:1]  # 只覆盖第一个事件
    df, _, _ = fm.merge_sources(_events(), _s1(), s2, None)
    assert df["MEAN_OPEN_3"].isna().sum() == 2


# ---------------------------------------------------------------- 去重
def test_pairwise_corr_exact_with_nan():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    b = a * 2.0 + 1e-12 * rng.normal(size=n)  # 完全线性
    c = rng.normal(size=n)
    x = np.column_stack([a, b, c])
    x[::7, 0] = np.nan  # a 有缺失
    x[1::11, 1] = np.nan
    corr = fm.pairwise_corr(x)
    ref = pd.DataFrame(x).corr()  # pandas 默认 pairwise complete
    np.testing.assert_allclose(corr, ref.to_numpy(), atol=1e-10)


def test_dedup_math():
    rng = np.random.default_rng(1)
    n = 1000
    base = rng.normal(size=n)
    df = pd.DataFrame({
        "event_id": range(n), "ts_code": "A",
        "date": pd.date_range("2010-01-01", periods=n),
        "sig_idx": 0, "low_date": 0, "prev_low_date": 0, "compare_rank": 0,
        "formation": 0, "regime": "up", "above_ma200": True, "seg": "train",
        "f1": base,
        "f2": base * 3.0,                 # 与 f1 完全相关 -> 剔
        "f3": base + 1e-4 * rng.normal(size=n),  # |ρ|>0.999 -> 剔
        "f4": rng.normal(size=n),         # 独立 -> 留
    })
    feats = ["f1", "f2", "f3", "f4"]
    src_of = {"f1": "s1", "f2": "s2", "f3": "s2", "f4": "s3"}
    mask = np.ones(n, bool)
    keep, records, corr = fm.dedup_by_correlation(df, feats, mask, src_of=src_of)
    assert keep == ["f1", "f4"]
    anchored = {r[0]: r[1] for r in records}
    assert anchored == {"f2": "f1", "f3": "f1"}
    fm.assert_dedup_clean(corr, keep, feats)


def test_dedup_priority_prefers_s1():
    rng = np.random.default_rng(2)
    n = 500
    base = rng.normal(size=n)
    df = pd.DataFrame({"g_s2": base, "g_s1": base * 1.0, "g_s3": base + 0.0})
    feats = ["g_s2", "g_s1", "g_s3"]
    src_of = {"g_s2": "s2", "g_s1": "s1", "g_s3": "s3"}
    keep, records, _ = fm.dedup_by_correlation(df, feats, np.ones(n, bool),
                                               src_of=src_of)
    assert keep == ["g_s1"]
    assert {r[0] for r in records} == {"g_s2", "g_s3"}
