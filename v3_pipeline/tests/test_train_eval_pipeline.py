#!/usr/bin/env python3
"""train_eval_pipeline 的单元测试（接缝: 事件×特征主表 + 标签 -> 训练/校准/指标）。

以合成数据验外部行为（无磁盘依赖）:
  - 段界与 30 个交易日隔离带断言的正反例;
  - 五折时间序列折外概率: 覆盖口径、逐位可复现;
  - 校准层已知值、确定性与越界拒绝;
  - 头部五名精确率已知值与平局确定性;
  - 泄漏排除、test 段禁评测与标签装载守卫。
例外: 两个内部接缝（_date_level_folds 的折界不落日内、SquaredLogitCalibrator._design
的 [p, p²] 形态）属验收标准的承重口径，直接钉死。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_master as fm  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

CALENDAR = pd.bdate_range("1990-01-01", "2027-01-01").to_numpy()


def _seg_df(rows):
    """rows: [(ts_code, date)];seg 按权威 segment_of 计算。"""
    df = pd.DataFrame(rows, columns=["ts_code", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["seg"] = fm.segment_of(df["date"])
    return df


# ---------------------------------------------------------------- 段完整性
def test_segment_integrity_pass():
    df = _seg_df([("A", "2018-11-16"), ("B", "2019-01-02"), ("C", "2022-11-01")])
    tep.assert_segment_integrity(df, CALENDAR)


def test_segment_integrity_rejects_seg_mismatch():
    df = _seg_df([("A", "2018-11-16"), ("B", "2019-01-02"), ("C", "2022-11-01")])
    df.loc[0, "seg"] = "val"  # 篡改段标签
    with pytest.raises(AssertionError, match="不一致"):
        tep.assert_segment_integrity(df, CALENDAR)


def test_segment_integrity_rejects_embargo_row_in_train():
    df = _seg_df([("A", "2018-11-16"), ("B", "2019-01-02"), ("C", "2022-11-01")])
    df.loc[0, "date"] = pd.Timestamp("2018-11-20")  # 落入隔离带 1
    df.loc[0, "seg"] = "train"  # 且段标签谎报为 train——同时触发不一致断言
    with pytest.raises(AssertionError):
        tep.assert_segment_integrity(df, CALENDAR)


def test_segment_integrity_rejects_short_embargo():
    # 截断日历使隔离带 1 只剩 10 个交易日
    cal = pd.bdate_range("1990-01-01", "2027-01-01")
    cal = cal[(cal < "2018-11-19") | (cal > "2018-11-30")].to_numpy()
    df = _seg_df([("A", "2018-11-16"), ("B", "2019-01-02"), ("C", "2022-11-01")])
    with pytest.raises(AssertionError, match="个交易日"):
        tep.assert_segment_integrity(df, cal)


def test_segment_integrity_rejects_penetrated_gap():
    # train 末行与 val 首行间隔 < 30 个交易日（且 seg 自洽，绕过断言 1/2）:
    # 用 val 区间内靠近 train 的日期无法做到自洽，改为构造"数据实测间隔"不足:
    # train 行取 2018-11-16，val 行取 2019-01-02，日历来去中间所有交易日。
    df = _seg_df([("A", "2018-11-16"), ("B", "2019-01-02"), ("C", "2022-11-01")])
    cal = pd.bdate_range("1990-01-01", "2027-01-01")
    cal = cal[(cal <= "2018-11-16") | (cal >= "2019-01-02")].to_numpy()
    with pytest.raises(AssertionError):
        tep.assert_segment_integrity(df, cal)


# ---------------------------------------------------------------- 泄漏守卫
def test_feature_columns_excludes_meta_and_rejects_leakage():
    df = pd.DataFrame({"event_id": [0], "ts_code": ["A"], "seg": ["train"],
                       "ATRN": [0.03], "future_return_5d": [0.1]})
    with pytest.raises(AssertionError, match="泄漏"):
        tep.model_feature_columns(df)
    clean = pd.DataFrame({"event_id": [0], "ts_code": ["A"], "seg": ["train"],
                          "ATRN": [0.03], "RET5": [0.01]})
    assert tep.model_feature_columns(clean) == ["ATRN", "RET5"]
    # 非数值非元数据列不得进入建模特征（feature_master 权威口径）
    with_str = clean.assign(note=["x"])
    assert tep.model_feature_columns(with_str) == ["ATRN", "RET5"]


def test_evaluate_segment_rejects_test():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-04"]), "ts_code": ["A"], "event_id": [0],
        "seg": ["test"], "prob": [0.5], "y": [1.0],
    })
    with pytest.raises(AssertionError, match="test"):
        tep.evaluate_segment(df)


# ---------------------------------------------------------------- 五折 OOF
def _toy_train(n_days=60, per_day=4, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=n_days).to_numpy()
    dates = np.repeat(days, per_day)
    n = len(dates)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    logit = X["f0"] * 1.5 - X["f1"] + rng.normal(scale=0.5, size=n)
    y = (logit > np.median(logit)).astype(float)
    return X, y, dates


def test_date_level_folds_respect_day_boundary():
    _, _, dates = _toy_train()
    folds = tep._date_level_folds(dates, 5)
    assert len(folds) == 5
    covered = np.concatenate([va for _, va in folds])
    for tr_idx, va_idx in folds:
        assert dates[tr_idx].max() < dates[va_idx].min()
        assert not set(tr_idx) & set(va_idx)
    # 覆盖口径: 除首块外每行恰被验证一次
    assert len(covered) == len(np.unique(covered))
    first_chunk = np.flatnonzero(dates <= np.unique(dates)[len(np.unique(dates)) // 6 - 1])
    assert set(first_chunk) | set(covered) == set(range(len(dates)))


def test_oof_shape_nan_and_reproducibility():
    X, y, dates = _toy_train()
    kw = dict(n_splits=5, num_boost_round=30, early_stopping_rounds=5)
    oof1, iters1 = tep.time_series_oof(X, y, dates, **kw)
    oof2, iters2 = tep.time_series_oof(X, y, dates, **kw)
    assert oof1.shape == y.shape
    first_chunk_rows = int((dates <= np.unique(dates)[60 // 6 - 1]).sum())
    assert np.isnan(oof1).sum() == first_chunk_rows
    assert ((oof1[~np.isnan(oof1)] > 0) & (oof1[~np.isnan(oof1)] < 1)).all()
    # np.array_equal 默认 NaN!=NaN, 首块折外概率为 NaN, 须 equal_nan=True
    assert np.array_equal(oof1, oof2, equal_nan=True), "同种子同数据折外概率必须逐位一致"
    assert iters1 == iters2
    assert all(i >= 1 for i in iters1)


def test_oof_rejects_unsorted_dates():
    X, y, dates = _toy_train()
    with pytest.raises(AssertionError, match="升序"):
        tep.time_series_oof(X, y, dates[::-1], n_splits=5, num_boost_round=5)


def test_final_num_boost_round():
    assert tep.final_num_boost_round([10, 21, 30]) == 20
    assert tep.final_num_boost_round([1]) == 1
    with pytest.raises(AssertionError):
        tep.final_num_boost_round([])


def test_fit_final_model_deterministic():
    X, y, _ = _toy_train()
    # 玩具数据仅 240 行, 放宽 min_data_in_leaf 否则 LightGBM 无法继续分裂、树数不足
    params = dict(tep.DEFAULT_LGBM_PARAMS, min_data_in_leaf=5)
    b1 = tep.fit_final_model(X, y, num_boost_round=10, params=params)
    b2 = tep.fit_final_model(X, y, num_boost_round=10, params=params)
    assert np.array_equal(b1.predict(X), b2.predict(X))
    assert b1.num_trees() == 10


# ---------------------------------------------------------------- 校准层
def test_calibrator_design_is_p_and_p_squared():
    cal = tep.SquaredLogitCalibrator()
    z = cal._design(np.array([0.2, 0.5]))
    assert z.shape == (2, 2)
    assert np.allclose(z[:, 0], [0.2, 0.5])
    assert np.allclose(z[:, 1], [0.04, 0.25])


def test_calibrator_recovers_direction_and_determinism():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, size=2000)
    y = (p + rng.normal(scale=0.15, size=2000) > 0.5).astype(float)
    c1 = tep.SquaredLogitCalibrator().fit(p, y)
    c2 = tep.SquaredLogitCalibrator().fit(p, y)
    assert c1.coef_.shape == (2,)
    assert c1.coef_[0] > 0  # 概率方向可恢复
    assert np.allclose(c1.coef_, c2.coef_) and c1.intercept_ == c2.intercept_
    out = c1.predict(np.array([0.1, 0.9]))
    assert ((out >= 0) & (out <= 1)).all() and out[1] > out[0]


def test_calibrator_rejects_bad_input():
    cal = tep.SquaredLogitCalibrator()
    with pytest.raises(AssertionError):
        cal._design(np.array([0.5, np.nan]))
    with pytest.raises(AssertionError):
        cal._design(np.array([1.5]))


# ---------------------------------------------------------------- 标签装载
def _write_lab(tmp_path, labels_group):
    ev = pd.DataFrame({"ts_code": ["A", "B"],
                       "date": pd.to_datetime(["2020-01-02", "2020-01-03"])})
    lb = pd.DataFrame({"group": labels_group, "hit_N20_k2.0": [1.0, 0.0]})
    evp, lbp = tmp_path / "events.parquet", tmp_path / "labels.parquet"
    ev.to_parquet(evp, index=False)
    lb.to_parquet(lbp, index=False)
    return evp, lbp


def test_load_div_labels_aligns_div_block(tmp_path):
    evp, lbp = _write_lab(tmp_path, ["div", "div"])
    out = tep.load_div_labels(evp, lbp, "hit_N20_k2.0")
    assert list(out.columns) == ["ts_code", "date", "hit_N20_k2.0"]
    assert out["hit_N20_k2.0"].tolist() == [1.0, 0.0]


def test_load_div_labels_rejects_non_div_head(tmp_path):
    evp, lbp = _write_lab(tmp_path, ["c1", "div"])  # 首行非 div: 逐位对齐破坏
    with pytest.raises(AssertionError, match="div"):
        tep.load_div_labels(evp, lbp, "hit_N20_k2.0")


def test_load_div_labels_rejects_duplicate_keys(tmp_path):
    evp, lbp = _write_lab(tmp_path, ["div", "div"])
    ev = pd.read_parquet(evp)
    ev.loc[1, "date"] = ev.loc[0, "date"]
    ev.loc[1, "ts_code"] = ev.loc[0, "ts_code"]
    ev.to_parquet(evp, index=False)
    with pytest.raises(AssertionError, match="唯一"):
        tep.load_div_labels(evp, lbp, "hit_N20_k2.0")


# ---------------------------------------------------------------- 头部五名精确率
def test_topk_precision_known_value():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 6 + ["2020-01-03"] * 3),
        "ts_code": ["A", "B", "C", "D", "E", "F", "A", "B", "C"],
        "event_id": range(9),
        "prob": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.8, 0.7],
        "y": [1, 1, 0, 0, 1, 0, 0, 1, 1],
    })
    # day1 top5 = A,B,C,D,E -> hits 1,1,0,0,1 = 0.6; day2 全取 3 行 -> 2/3
    out = tep.topk_precision(df, k=5)
    assert out["n_days"] == 2 and out["n_selected"] == 8
    assert out["precision_at_5_dayavg"] == pytest.approx((0.6 + 2 / 3) / 2)
    assert out["precision_at_5_eventavg"] == pytest.approx(5 / 8)


def test_topk_precision_tie_break_deterministic():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4),
        "ts_code": ["D", "C", "B", "A"],
        "event_id": [3, 2, 1, 0],
        "prob": [0.5, 0.5, 0.5, 0.5],
        "y": [0, 0, 1, 1],
    })
    # 概率全同 -> 按 ts_code 升序取前 2: A, B -> 全中
    out = tep.topk_precision(df, k=2)
    assert out["precision_at_2_eventavg"] == 1.0


def test_evaluate_segment_row():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
        "ts_code": list("ABCDEFGH"), "event_id": range(8),
        "seg": ["val"] * 8,
        "prob": [0.9, 0.7, 0.4, 0.2, 0.8, 0.6, 0.3, 0.1],
        "y": [1, 1, 0, 0, 1, 0, 0, 1],
    })
    row = tep.evaluate_segment(df, k=5)
    assert row["n_events"] == 8 and row["n_days"] == 2
    assert row["base_rate"] == pytest.approx(0.5)
    assert row["average_precision"] > 0.5  # 概率方向正确, AP 高于零信息线
