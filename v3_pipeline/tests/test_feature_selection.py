#!/usr/bin/env python3
"""feature_selection 的单元测试（接缝: 五层精选纯函数 / 当选配置解析 / 分数装配）。

以合成数据验外部行为（无磁盘依赖）:
  - 层1: 重要性排序已知值、零重要性剔除、平局特征名升序;
  - 层2: 符号一致幸存、翻号剔除、常数列与对不足记 0 剔除;
  - 层3: 阈值上下聚簇、代表为层1名次最前者、NaN 相关不判同簇;
  - 层4: K 阶梯已知值与下限断言、拐点已知值与平局取小 K;
  - 当选配置解析: 断言守护（winner 非狙击 / 超参与网格不符即拒绝）;
  - 分数装配: test 段 y 非 NaN 即拒绝、event_id 唯一性。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import feature_selection as fsel  # noqa: E402
import label_race as lr  # noqa: E402


# ---------------------------------------------------------- 层1
def test_layer1_rank_known_values_and_zero_removal():
    shap = np.array([[1.0, -2.0, 0.0],
                     [3.0, 2.0, 0.0]])
    table = fsel.layer1_rank(shap, ["b_feat", "a_feat", "z_feat"])
    # importance: a=2.0, b=2.0, z=0.0; 平局按特征名升序 -> a 先于 b
    assert list(table["feature"]) == ["a_feat", "b_feat", "z_feat"]
    assert list(table["rank"]) == [0, 1, 2]
    assert table["kept"].tolist() == [True, True, False]
    assert np.allclose(table["importance"], [2.0, 2.0, 0.0])


# ---------------------------------------------------------- 层2
def _year_frame(n_per_year: int = 40):
    """四年合成: f_pos 各年正相关, f_flip 2021 起翻号, f_const 2020 年为常数。"""
    rng = np.random.default_rng(7)
    years = np.repeat([2019, 2020, 2021, 2022], n_per_year)
    x_pos = rng.normal(size=len(years))
    s_pos = 2 * x_pos + rng.normal(0, 0.01, len(years))
    x_flip = rng.normal(size=len(years))
    s_flip = np.where(years < 2021, x_flip, -x_flip) + rng.normal(0, 0.01, len(years))
    x_const = rng.normal(size=len(years))
    x_const[years == 2020] = 5.0                      # 2020 年常数 -> 该年符号 0
    s_const = x_const + rng.normal(0, 0.01, len(years))
    X = pd.DataFrame({"f_pos": x_pos, "f_flip": x_flip, "f_const": x_const})
    shap = np.column_stack([s_pos, s_flip, s_const])
    return X, shap, years


def test_layer2_sign_consistency():
    X, shap, years = _year_frame()
    table = fsel.layer2_yearly_signs(X, shap, years, ["f_pos", "f_flip", "f_const"])
    rec = table.set_index("feature")
    assert rec.loc["f_pos", "consistent"]
    assert not rec.loc["f_flip", "consistent"]     # 翻号 -> 漂移剔除
    assert not rec.loc["f_const", "consistent"]    # 2020 常数 -> 符号 0 -> 剔除
    assert rec.loc["f_flip", "sign_2019"] == 1 and rec.loc["f_flip", "sign_2021"] == -1
    assert rec.loc["f_const", "sign_2020"] == 0


def test_layer2_min_pairs_guard():
    rng = np.random.default_rng(3)
    years = np.repeat([2019, 2020, 2021, 2022], 40)
    x = rng.normal(size=160)
    s = x.copy()
    idx19 = np.flatnonzero(years == 2019)
    x[idx19[:12]] = np.nan                  # 2019 年仅 28 有效对 < 30 -> 符号 0
    X = pd.DataFrame({"f": x})
    table = fsel.layer2_yearly_signs(X, s.reshape(-1, 1), years, ["f"])
    assert table.loc[0, "sign_2019"] == 0
    assert not table.loc[0, "consistent"]


def test_layer2_year_coverage_assertion():
    X, shap, years = _year_frame()
    years_bad = years.copy()
    years_bad[0] = 2018                        # 出现预登记外年份 -> 拒绝
    with pytest.raises(AssertionError):
        fsel.layer2_yearly_signs(X, shap, years_bad, ["f_pos", "f_flip", "f_const"])
    missing = np.where(years == 2022, 2021, years)   # 缺 2022 年 -> 拒绝
    with pytest.raises(AssertionError):
        fsel.layer2_yearly_signs(X, shap, missing, ["f_pos", "f_flip", "f_const"])


# ---------------------------------------------------------- 层3
def test_layer3_clustering_threshold_and_representative():
    rng = np.random.default_rng(11)
    n = 500
    base = rng.normal(size=n)
    # hi 与 base 近完全相关 (>0.9), mid 与 base 相关 0.5 (<0.9), ind 独立
    X = pd.DataFrame({
        "base": base,
        "hi": base + rng.normal(0, 0.01, n),
        "mid": 0.5 * base + np.sqrt(0.75) * rng.normal(size=n),
        "ind": rng.normal(size=n),
    })
    rank_of = {"base": 1, "hi": 0, "mid": 2, "ind": 3}   # hi 名次最前
    reps, records, corr = fsel.layer3_clusters(
        X, ["base", "hi", "mid", "ind"], rank_of)
    assert reps[0] == "hi"                    # hi 名次最前 -> 自立为代表
    assert set(reps) == {"hi", "mid", "ind"}  # base 入 hi 簇
    rec = records.set_index("feature")
    assert rec.loc["base", "representative"] == "hi"
    assert rec.loc["base", "corr_with_rep"] > fsel.CORR_CLUSTER_THRESHOLD
    assert rec.loc["mid", "is_representative"]
    assert abs(rec.loc["mid", "corr_with_rep"] - 1.0) < 1e-12


def test_layer3_nan_corr_no_cluster():
    n = 100
    X = pd.DataFrame({"a": np.ones(n), "b": np.arange(n, dtype=float)})
    rank_of = {"a": 0, "b": 1}
    reps, records, _ = fsel.layer3_clusters(X, ["a", "b"], rank_of)
    assert set(reps) == {"a", "b"}            # 常数列相关 NaN -> 不判同簇


# ---------------------------------------------------------- 层4
def test_k_ladder_known_values():
    assert fsel.k_ladder(2060)[-1] == 2060
    assert fsel.k_ladder(2060)[:4] == [5, 10, 15, 20]
    assert fsel.k_ladder(12) == [5, 10, 12]   # 过滤 > n, 并入 n 自身
    assert fsel.k_ladder(5) == [5]
    with pytest.raises(AssertionError):
        fsel.k_ladder(4)


def test_find_elbow_clear_knee():
    curve = pd.DataFrame({
        "k": [5, 10, 20, 40, 80, 160],
        "val_precision_at_5_dayavg": [0.50, 0.56, 0.575, 0.578, 0.579, 0.579],
    })
    out = fsel.find_elbow(curve)
    assert out["k_star"] == 10                # 拐点在 10（其后边际增益骤降）
    assert len(out["distances"]) == 6


def test_find_elbow_tie_breaks_smaller_k_and_last_point():
    # 严格线性 -> 全点距 0 -> argmax 首个 -> 最小 K
    curve = pd.DataFrame({
        "k": [5, 10, 20],
        "val_precision_at_5_dayavg": [0.5, 0.55, 0.6],
    })
    assert fsel.find_elbow(curve)["k_star"] == 5
    # 单调加速（凹向上）-> 最远点为中间或末点; 构造末点最远场景
    curve2 = pd.DataFrame({
        "k": [5, 10, 20],
        "val_precision_at_5_dayavg": [0.5, 0.5, 0.9],
    })
    out2 = fsel.find_elbow(curve2)
    assert out2["k_star"] in (10, 20)         # 确定性即可, 值由几何决定
    # 单点直通
    one = pd.DataFrame({"k": [7], "val_precision_at_5_dayavg": [0.5]})
    assert fsel.find_elbow(one)["k_star"] == 7
    # 非升序拒绝
    bad = pd.DataFrame({"k": [10, 5], "val_precision_at_5_dayavg": [0.5, 0.6]})
    with pytest.raises(AssertionError):
        fsel.find_elbow(bad)


# ---------------------------------------------------------- 当选配置解析
def _summary_row(config_id: int = 13) -> pd.DataFrame:
    cfg = lr.GRID[config_id]
    return pd.DataFrame([{"candidate": lr.SNIPER_LABEL, "config_id": config_id, **cfg}])


def test_resolve_winner_happy_path():
    adj = {"winner": lr.SNIPER_LABEL, "ap_constraint_passed": True}
    label, cid, params = fsel.resolve_winner(_summary_row(13), adj)
    assert label == "hit_N20_k2.0" and cid == 13
    assert params["num_leaves"] == 31 and params["feature_fraction"] == 0.8
    assert params["deterministic"] is True    # 复现性四件套随网格参数带出


def test_resolve_winner_guards():
    with pytest.raises(AssertionError):
        fsel.resolve_winner(_summary_row(), {"winner": "cur_pos_3d",
                                             "ap_constraint_passed": True})
    with pytest.raises(AssertionError):
        fsel.resolve_winner(_summary_row(), {"winner": lr.SNIPER_LABEL,
                                             "ap_constraint_passed": False})
    bad = _summary_row()
    bad.loc[0, "num_leaves"] = 999            # 与预登记网格不符
    with pytest.raises(AssertionError):
        fsel.resolve_winner(bad, {"winner": lr.SNIPER_LABEL,
                                  "ap_constraint_passed": True})


# ---------------------------------------------------------- 分数装配
def test_assemble_scores_test_y_nan_guard():
    keys = pd.DataFrame({
        "ts_code": ["a", "b"], "date": pd.to_datetime(["2022-01-04", "2022-11-01"]),
        "event_id": ["e1", "e2"], "seg": ["val", "test"]})
    prob = np.array([0.7, 0.2])
    with pytest.raises(AssertionError):
        fsel.assemble_scores(keys, prob, np.array([1.0, 0.0]))   # test y 非 NaN
    out = fsel.assemble_scores(keys, prob, np.array([1.0, np.nan]))
    assert out.loc[out["seg"] == "test", "y"].isna().all()
    assert list(out.columns) == ["ts_code", "date", "event_id", "seg", "prob", "y"]


def test_assemble_scores_duplicate_event_id_rejected():
    keys = pd.DataFrame({
        "ts_code": ["a", "a"], "date": pd.to_datetime(["2022-01-04"] * 2),
        "event_id": ["e1", "e1"], "seg": ["val", "val"]})
    with pytest.raises(AssertionError):
        fsel.assemble_scores(keys, np.array([0.1, 0.2]), np.array([0.0, 1.0]))
