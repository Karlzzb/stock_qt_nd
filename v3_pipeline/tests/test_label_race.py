#!/usr/bin/env python3
"""label_race 的单元测试（接缝: 标签构建 / 网格完整性 / 选配置与裁决 / 单配置跑训）。

以合成数据验外部行为（无磁盘依赖）:
  - 十九候选清单与命名;
  - 收益二分类标签已知值、NaN 保留、头部截断不变（因果性）;
  - 预登记网格完整性（<=50、仅四项超参、全因子无重复）;
  - 合并池 event_id 前缀唯一与段保留;
  - 选配置与裁决的已知值与平局确定性、AP 中位数否决;
  - 单配置跑训端到端（小合成集）与复现性。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import label_race as lr  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402


def _make_daily(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "trade_date": pd.bdate_range("2020-01-01", periods=n),
        "open": open_, "high": high, "low": low, "close": close,
    })


# ---------------------------------------------------------- 候选清单
def test_candidate_labels_nineteen_ordered():
    cands = lr.candidate_labels()
    assert len(cands) == 19
    assert cands[0] == "hit_N20_k2.0"
    assert cands[1:10] == [f"cur_pos_{h}d" for h in lr.RACE_HORIZONS]
    assert cands[10:] == [f"open_exec_pos_{h}d" for h in lr.RACE_HORIZONS]
    assert len(set(cands)) == 19


def test_candidate_cn_full_names():
    for c in lr.candidate_labels():
        cn = lr.candidate_cn(c)
        assert cn and not cn.isascii(), f"{c} 缺中文全名"
    assert "狙击" in lr.candidate_cn("hit_N20_k2.0")
    assert "T 收盘入场" in lr.candidate_cn("cur_pos_3d")
    assert "T+1 开盘入场" in lr.candidate_cn("open_exec_pos_60d")
    assert "60 个交易日" in lr.candidate_cn("open_exec_pos_60d")
    with pytest.raises(AssertionError):
        lr.candidate_cn("cur_pos_7d")


# ---------------------------------------------------------- 收益二分类标签
def test_return_labels_known_values():
    d = _make_daily()
    out = lr.compute_return_labels(d)
    h = 3
    t = 10
    cur_ret = d["close"].iloc[t + 1 + h] / d["close"].iloc[t] - 1
    exe_ret = d["close"].iloc[t + 1 + h] / d["open"].iloc[t + 1] - 1
    assert out[f"cur_pos_{h}d"].iloc[t] == float(cur_ret > 0)
    assert out[f"open_exec_pos_{h}d"].iloc[t] == float(exe_ret > 0)
    assert out.columns[0] == "date"
    assert out.shape[1] == 1 + 18
    assert set(np.nan_to_num(out.drop(columns="date").to_numpy(), nan=2.0).flat) <= {0.0, 1.0, 2.0}


def test_return_labels_nan_preserved_at_tail():
    d = _make_daily(n=60)
    out = lr.compute_return_labels(d)
    # cur/open_exec 在 T+1+h 越界处为 NaN，不得被 (ret > 0) 吞成 0
    assert out["cur_pos_3d"].iloc[-4:].isna().all()
    assert out["open_exec_pos_60d"].iloc[-61:].isna().all()
    assert not out["cur_pos_3d"].iloc[-5:].isna().all()


def test_return_labels_head_truncation_invariant():
    """因果性：截掉序列头部（过去）重算，交集日期标签逐位一致。"""
    d = _make_daily(n=120, seed=7)
    full = lr.compute_return_labels(d)
    cut = lr.compute_return_labels(d.iloc[40:].copy())
    m = full.merge(cut, on="date", suffixes=("_a", "_b"))
    for c in lr.candidate_labels()[1:]:
        a, b = m[f"{c}_a"].to_numpy(), m[f"{c}_b"].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"{c} 头部截断后漂移"


def test_return_labels_past_rows_do_not_matter():
    """改动严格早于 t 的 OHLC，t 处标签不变（cur 入场价取 t 收盘，属"现在"）。"""
    d = _make_daily(n=60, seed=3)
    base = lr.compute_return_labels(d)
    d2 = d.copy()
    d2.loc[:9, ["open", "high", "low", "close"]] *= 1.5
    alt = lr.compute_return_labels(d2)
    for c in lr.candidate_labels()[1:]:
        a = base[c].iloc[10:].to_numpy()
        b = alt[c].iloc[10:].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"{c} 被过去数据污染"


# ---------------------------------------------------------- 预登记网格
def test_grid_integrity():
    assert len(lr.GRID) == 36 and len(lr.GRID) <= 50
    keys = set()
    for g in lr.GRID:
        assert set(g) == set(lr.GRID_KEYS)
        keys.add(tuple(g[k] for k in lr.GRID_KEYS))
    assert len(keys) == 36
    assert {g["num_leaves"] for g in lr.GRID} == {15, 31, 63}
    assert {g["min_data_in_leaf"] for g in lr.GRID} == {50, 100, 200}
    assert {g["learning_rate"] for g in lr.GRID} == {0.05, 0.10}
    assert {g["feature_fraction"] for g in lr.GRID} == {0.6, 0.8}


def test_grid_params_inherits_defaults_and_reproducibility():
    p = lr.grid_params({"num_leaves": 63, "min_data_in_leaf": 50,
                        "learning_rate": 0.10, "feature_fraction": 0.6})
    assert p["num_leaves"] == 63 and p["feature_fraction"] == 0.6
    assert p["bagging_fraction"] == tep.DEFAULT_LGBM_PARAMS["bagging_fraction"]
    assert p["deterministic"] is True and p["num_threads"] == 8
    assert p["seed"] == tep.DEFAULT_LGBM_PARAMS["seed"]
    with pytest.raises(AssertionError):
        lr.grid_params({"max_depth": 4})


# ---------------------------------------------------------- 合并池
def test_build_merged_master_prefix_and_union():
    cols = {"event_id": [0, 1], "ts_code": ["a", "b"],
            "date": pd.to_datetime(["2010-01-04", "2010-01-05"]),
            "seg": ["train", "train"], "f1": [0.1, 0.2]}
    main, backup = pd.DataFrame(cols), pd.DataFrame(cols)
    merged = lr.build_merged_master(main, backup)
    assert len(merged) == 4
    assert set(merged["event_id"]) == {"main_0", "main_1", "backup_0", "backup_1"}
    assert merged["pool"].tolist() == ["main", "main", "backup", "backup"]
    assert merged["seg"].tolist() == ["train"] * 4
    # 键不去重：两池同 (ts_code,date) 各留一行（event_id 前缀区分）
    dup_key = pd.DataFrame({"event_id": [0], "ts_code": ["a"],
                            "date": pd.to_datetime(["2010-01-04"]),
                            "seg": ["train"], "f1": [0.1]})
    m2 = lr.build_merged_master(dup_key, dup_key)
    assert len(m2) == 2 and m2["ts_code"].nunique() == 1


def test_build_merged_master_normalizes_bool_object():
    """备池 downtrend/hammer_signal 为 object(bool, 含 None)（T4 漂移），合并须归一 Int8。"""
    base = {"event_id": [0, 1], "ts_code": ["a", "b"],
            "date": pd.to_datetime(["2010-01-04", "2010-01-05"]),
            "seg": ["train", "train"]}
    main = pd.DataFrame({**base, "downtrend": pd.array([1, 0], dtype="int8")})
    backup = pd.DataFrame({**base,
                           "downtrend": pd.array([True, None], dtype=object)})
    merged = lr.build_merged_master(main, backup)
    assert str(merged["downtrend"].dtype) == "Int8"
    assert merged["downtrend"].dtype.kind == "i"  # 仍在数值特征口径内
    got = merged["downtrend"].tolist()
    assert got[0] == 1 and got[1] == 0 and got[2] == 1 and pd.isna(got[3])
    bad = pd.DataFrame({**base, "downtrend": pd.array([True, "x"], dtype=object)})
    with pytest.raises(AssertionError):
        lr.build_merged_master(main, bad)


# ---------------------------------------------------------- 选配置
def _cfg_rows(vals):
    """vals: [(config_id, p5, ap)]"""
    return pd.DataFrame({
        "config_id": [v[0] for v in vals],
        "val_precision_at_5_dayavg": [v[1] for v in vals],
        "val_average_precision": [v[2] for v in vals],
    })


def test_select_best_config_plain_and_ties():
    m = _cfg_rows([(0, 0.50, 0.60), (1, 0.55, 0.58), (2, 0.52, 0.62)])
    assert lr.select_best_config(m) == 1
    # p5 平局 -> AP 裁决
    m = _cfg_rows([(0, 0.55, 0.58), (1, 0.55, 0.62), (2, 0.50, 0.99)])
    assert lr.select_best_config(m) == 1
    # p5 与 AP 均平局 -> 网格序靠前
    m = _cfg_rows([(2, 0.55, 0.62), (0, 0.55, 0.62), (1, 0.50, 0.50)])
    assert lr.select_best_config(m) == 1  # iloc 位置：config_id=0 在第 1 行


# ---------------------------------------------------------- 裁决
def _summary(p5s, aps):
    return pd.DataFrame({
        "candidate": lr.candidate_labels(),
        "val_precision_at_5_dayavg": p5s,
        "val_average_precision": aps,
    })


def test_adjudicate_winner_and_constraint():
    p5 = [0.40] * 19
    ap = [0.50] * 19
    p5[5], ap[5] = 0.60, 0.55
    r = lr.adjudicate(_summary(p5, ap))
    assert r["winner"] == lr.candidate_labels()[5]
    assert r["ap_constraint_passed"] is True
    assert r["median_val_average_precision"] == pytest.approx(0.50)


def test_adjudicate_ap_below_median_no_winner():
    p5 = [0.40] * 19
    ap = [0.60] * 19
    p5[0], ap[0] = 0.90, 0.10  # 狙击标签 p@5 最高但 AP 远低于中位数
    r = lr.adjudicate(_summary(p5, ap))
    assert r["winner"] is None
    assert r["ap_constraint_passed"] is False
    assert r["winner_val_precision_at_5_dayavg"] == pytest.approx(0.90)


def test_adjudicate_tie_break_by_ap_then_order():
    p5 = [0.50] * 19
    ap = [0.50] * 19
    ap[3], ap[7] = 0.55, 0.56  # p5 全平 -> AP 最高者（候选 7）当选
    r = lr.adjudicate(_summary(p5, ap))
    assert r["winner"] == lr.candidate_labels()[7]
    # p5/AP 全平 -> 候选序靠前（狙击标签）
    r = lr.adjudicate(_summary([0.5] * 19, [0.5] * 19))
    assert r["winner"] == "hit_N20_k2.0"


def test_adjudicate_rejects_wrong_candidate_set():
    s = _summary([0.5] * 19, [0.5] * 19).iloc[:-1]
    with pytest.raises(AssertionError):
        lr.adjudicate(s)


# ---------------------------------------------------------- 单配置跑训（端到端小合成集）
def _synth_frame(n_days=120, per_day=4, seg_lo="2005-01-01", seed=1):
    """合成事件帧：日期跨 train/val 两段，特征与标签带弱信号。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(seg_lo, periods=n_days)
    rows = []
    for i, dt in enumerate(days):
        for j in range(per_day):
            x1, x2 = rng.normal(), rng.normal()
            p = 1 / (1 + np.exp(-(x1 * 0.8 + x2 * 0.5)))
            rows.append({"date": dt, "ts_code": f"S{j:03d}", "event_id": f"e{i}_{j}",
                         "f1": x1, "f2": x2, "y": float(rng.random() < p)})
    df = pd.DataFrame(rows)
    df["seg"] = np.where(df["date"] < "2005-04-01", "train", "val")
    return df


def test_run_single_config_end_to_end_deterministic():
    df = _synth_frame()
    train, val = df[df["seg"] == "train"], df[df["seg"] == "val"]
    params = lr.grid_params(lr.GRID[0])
    row1, art1 = lr.run_single_config(train, val, ["f1", "f2"], "y", params)
    row2, art2 = lr.run_single_config(train, val, ["f1", "f2"], "y", params)
    assert row1 == row2, "同配置复跑指标行不一致"
    assert np.array_equal(art1["oof"], art2["oof"], equal_nan=True)
    for key in ("val_average_precision", "val_precision_at_5_dayavg",
                "train_oof_average_precision", "best_iters", "final_num_boost_round",
                "calib_coef_p", "calib_coef_p2", "calib_intercept"):
        assert key in row1, f"指标行缺键 {key}"
    assert 0.0 <= row1["val_average_precision"] <= 1.0
    assert 0.0 <= row1["val_precision_at_5_dayavg"] <= 1.0
    assert row1["val_n_events"] == len(val)


def test_run_single_config_rejects_test_segment():
    df = _synth_frame()
    bad = df[df["seg"] == "val"].copy()
    bad.loc[bad.index[0], "seg"] = "test"
    with pytest.raises(AssertionError):
        lr.run_single_config(df[df["seg"] == "train"], bad, ["f1", "f2"], "y",
                             lr.grid_params(lr.GRID[0]))
