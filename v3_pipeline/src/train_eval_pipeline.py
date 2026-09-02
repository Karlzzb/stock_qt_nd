#!/usr/bin/env python3
"""训练评测管线库（issue #25）：LightGBM 二分类 + 逻辑回归校准层（输入 [p, p²]）。

口径（全部预登记，非调参结果）:
  - 切分段沿用 feature_master 的权威定义: train 2001-01-01~2018-12-31 /
    val 2019-01-01~2022-10-31 / 隔离带两段（各 30 个交易日）/ test 2022-11 起。
    段完整性以主表 seg 列与 segment_of 逐行一致 + 隔离带交易日数断言守护。
  - 训练段内五折时间序列切分按"事件日"分块（同日事件不跨折，折界不落日内），
    折外概率（OOF）供校准层拟合；首块（第 1/6 段）无折外概率，校准样本为其余 5 块。
  - 校准层: sklearn LogisticRegression，输入 [p, p²]（p 为 LightGBM 原始概率），
    拟合于 (OOF 概率, 训练段标签)，与一维堆叠恒等形态不同，二次项提供非线性自由度。
  - 终模: 全训练段重训，num_boost_round = 五折 best_iteration 均值（取整）。
  - 指标: 平均精确率（average precision）与头部五名精确率（按事件日截面取
    概率 top-5，日加权为主口径、事件加权为辅口径）。
  - 复现性: LightGBM deterministic=true + 全种子固定 + num_threads 固定，
    同机同数据折外概率逐位一致（冒烟脚本内置复跑断言守护）。

纪律: test 段在标签赛终审前不出任何指标（#20 "测试段每个候选只碰一次"），
本库 evaluate 入口只接受 train/val 段数据。
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

import feature_master as fm

# ------------------------------------------------------------- 预登记常量
N_SPLITS = 5                 # 训练段内时间序列折数
MIN_EMBARGO_TRADING_DAYS = 30
TOP_K = 5
EVAL_SEGMENTS = ("train", "val")

# LightGBM 冒烟固定参数（口径近 v5_label_open_exec.yaml，网格调参属后续票据）
DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    # 复现性四件套
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 20260902,
    "feature_fraction_seed": 20260902,
    "bagging_seed": 20260902,
    "data_random_seed": 20260902,
}
NUM_BOOST_ROUND_CAP = 1000
EARLY_STOPPING_ROUNDS = 50

CALIBRATOR_C = 1.0           # 校准层逻辑回归正则强度（2 特征、千级样本，默认量级）
CALIBRATOR_MAX_ITER = 1000


# ------------------------------------------------------------- 段完整性断言
def assert_segment_integrity(df: pd.DataFrame, calendar: np.ndarray) -> None:
    """段界与隔离带无串段断言（issue #25 AC2）。

    df 须含 date / seg 两列（主表行）；calendar 为交易日历（排序去重的日期数组）。
    四项硬断言:
      1. seg 逐行等于权威 segment_of(date)（段界本身无错位）;
      2. train/val/test 行无一落入隔离带日期区间（物理无串段）;
      3. 每段隔离带在交易日历上不少于 30 个交易日;
      4. 数据实测: train 末行与 val 首行、val 末行与 test 首行之间的
         日历交易日间隔均不少于 30。
    """
    dates = pd.to_datetime(df["date"])
    expect = fm.segment_of(dates)
    got = df["seg"].to_numpy()
    mismatch = int((expect != got).sum())
    assert mismatch == 0, f"seg 列与权威 segment_of 不一致: {mismatch} 行"
    assert set(np.unique(got)) <= {"pre2001", "train", "val", "embargo", "test"}, \
        f"未知段标签: {set(np.unique(got))}"

    cal = pd.to_datetime(pd.Series(calendar)).to_numpy()
    assert cal.size and (cal[:-1] < cal[1:]).all(), "交易日历必须严格递增且无重复"

    for i, (lo, hi) in enumerate(fm.EMBARGO):
        in_band = (dates >= lo) & (dates <= hi)
        bad = df.loc[in_band.to_numpy(), "seg"].isin(["train", "val", "test"]).sum()
        assert bad == 0, f"隔离带 {i}（{lo.date()}~{hi.date()}）内混入 {bad} 行建模型行"
        n_days = int(((cal >= np.datetime64(lo)) & (cal <= np.datetime64(hi))).sum())
        assert n_days >= MIN_EMBARGO_TRADING_DAYS, \
            f"隔离带 {i} 仅 {n_days} 个交易日 < {MIN_EMBARGO_TRADING_DAYS}"

    def _gap_days(seg_a: str, seg_b: str) -> int:
        a_max = dates[got == seg_a].max()
        b_min = dates[got == seg_b].min()
        return int(((cal > np.datetime64(a_max)) & (cal < np.datetime64(b_min))).sum())

    for seg_a, seg_b in (("train", "val"), ("val", "test")):
        assert (got == seg_a).any() and (got == seg_b).any(), f"缺段: {seg_a}/{seg_b}"
        gap = _gap_days(seg_a, seg_b)
        assert gap >= MIN_EMBARGO_TRADING_DAYS, \
            f"{seg_a}~{seg_b} 间仅 {gap} 个交易日 < {MIN_EMBARGO_TRADING_DAYS}（隔离带被穿透）"


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """建模特征列 = feature_master 权威口径（元数据之外、bool 视同数值）+ 泄漏排除断言。"""
    cols = fm.feature_columns(df)
    fm.assert_no_leakage(cols)
    return cols


# ------------------------------------------------------------- 标签装载
def load_div_labels(events_path, labels_path, hit_col: str) -> pd.DataFrame:
    """背离事件狙击标签：labels.parquet 前 len(events) 行必须为 div 组（逐位对齐）。

    返回 (ts_code, date, <hit_col>) 三列，键唯一。
    """
    ev = pd.read_parquet(events_path, columns=["ts_code", "date"])
    lb = pd.read_parquet(labels_path, columns=["group", hit_col])
    div = lb.iloc[: len(ev)]
    assert (div["group"] == "div").all(), "labels.parquet 前 N 行非 div 组，对齐破坏"
    out = pd.DataFrame({
        "ts_code": ev["ts_code"].to_numpy(),
        "date": pd.to_datetime(ev["date"].to_numpy()),
        hit_col: div[hit_col].to_numpy(),
    })
    assert not out.duplicated(["ts_code", "date"]).any(), "标签键 (ts_code,date) 不唯一"
    return out


# ------------------------------------------------------------- 五折 OOF
def _date_level_folds(dates: np.ndarray, n_splits: int):
    """按事件日分块的时间序列折：折界落在日上，同日事件不跨折。

    返回 [(train_idx, val_idx)]，保证 fold 内 max(train 日期) < min(val 日期)。
    """
    uniq = np.unique(dates)
    assert len(uniq) > n_splits, f"事件日数 {len(uniq)} 不足以切 {n_splits} 折"
    folds = []
    for tr_d, va_d in TimeSeriesSplit(n_splits=n_splits).split(uniq):
        tr_lo, tr_hi = uniq[tr_d[0]], uniq[tr_d[-1]]
        va_lo, va_hi = uniq[va_d[0]], uniq[va_d[-1]]
        assert tr_hi < va_lo, "折界穿透：训练日期不早于验证日期"
        tr_idx = np.flatnonzero((dates >= tr_lo) & (dates <= tr_hi))
        va_idx = np.flatnonzero((dates >= va_lo) & (dates <= va_hi))
        folds.append((tr_idx, va_idx))
    return folds


def time_series_oof(X: pd.DataFrame, y: np.ndarray, dates: np.ndarray,
                    n_splits: int = N_SPLITS, params: dict | None = None,
                    num_boost_round: int = NUM_BOOST_ROUND_CAP,
                    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    """训练段五折时间序列折外概率。

    X 须已按 (date, ts_code) 升序；返回 (oof, best_iters)：
    oof 为全长 float64 数组，首块无折外概率记 NaN；best_iters 供终模轮数取均值。
    """
    params = dict(DEFAULT_LGBM_PARAMS if params is None else params)
    dates = np.asarray(dates)
    assert (dates[:-1] <= dates[1:]).all(), "X/dates 必须按日期升序"
    y = np.asarray(y, dtype=np.float64)
    assert len(X) == len(y) == len(dates)

    oof = np.full(len(y), np.nan)
    best_iters: list[int] = []
    for tr_idx, va_idx in _date_level_folds(dates, n_splits):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        oof[va_idx] = booster.predict(X.iloc[va_idx],
                                      num_iteration=booster.best_iteration)
        best_iters.append(booster.best_iteration)
    return oof, best_iters


def final_num_boost_round(best_iters: list[int]) -> int:
    """终模轮数 = 折间 best_iteration 均值取整，下限 1。"""
    assert best_iters, "best_iters 为空"
    return max(int(round(float(np.mean(best_iters)))), 1)


def fit_final_model(X: pd.DataFrame, y: np.ndarray, num_boost_round: int,
                    params: dict | None = None) -> lgb.Booster:
    """全训练段重训终模（无验证集、无早停，轮数由五折均值给定）。"""
    params = dict(DEFAULT_LGBM_PARAMS if params is None else params)
    dtrain = lgb.Dataset(X, label=np.asarray(y, dtype=np.float64))
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


# ------------------------------------------------------------- 校准层 [p, p²]
class SquaredLogitCalibrator:
    """逻辑回归校准层：输入为预测概率 p 及其平方 p²（issue #20 既定形态）。"""

    def __init__(self, C: float = CALIBRATOR_C, max_iter: int = CALIBRATOR_MAX_ITER):
        self._lr = LogisticRegression(C=C, max_iter=max_iter)

    def _design(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        assert np.isfinite(p).all(), "校准层输入含 NaN/inf"
        assert ((p >= 0) & (p <= 1)).all(), "校准层输入越出 [0,1]"
        return np.column_stack([p, p * p])

    def fit(self, p: np.ndarray, y: np.ndarray) -> "SquaredLogitCalibrator":
        self._lr.fit(self._design(p), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._design(p))[:, 1]

    @property
    def coef_(self) -> np.ndarray:
        return self._lr.coef_[0]

    @property
    def intercept_(self) -> float:
        return float(self._lr.intercept_[0])


# ------------------------------------------------------------- 指标
def topk_precision(df: pd.DataFrame, prob_col: str = "prob", label_col: str = "y",
                   k: int = TOP_K) -> dict:
    """头部五名精确率：按事件日截面取概率 top-min(k, 当日事件数)。

    平局按 (prob 降序, ts_code, event_id) 确定性裁决。
    返回日加权（主口径，与 pool_cleaning 基线同口径）与事件加权两个值。
    """
    need = {"date", "ts_code", "event_id", prob_col, label_col}
    assert need <= set(df.columns), f"topk_precision 缺列: {need - set(df.columns)}"
    ranked = df.sort_values(
        ["date", prob_col, "ts_code", "event_id"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    pick = ranked.groupby("date", sort=True).head(k)
    day_hit = pick.groupby("date")[label_col].mean()
    return {
        "n_days": int(day_hit.size),
        "n_selected": int(len(pick)),
        f"precision_at_{k}_dayavg": float(day_hit.mean()),
        f"precision_at_{k}_eventavg": float(pick[label_col].mean()),
    }


def evaluate_segment(df: pd.DataFrame, prob_col: str = "prob",
                     label_col: str = "y", k: int = TOP_K) -> dict:
    """单段指标行：平均精确率 + 头部五名精确率（日加权/事件加权）+ 基数。"""
    assert df["seg"].isin(EVAL_SEGMENTS).all(), "评测只允许 train/val 段（test 终审前禁用）"
    y = df[label_col].to_numpy(dtype=np.float64)
    p = df[prob_col].to_numpy(dtype=np.float64)
    assert np.isfinite(y).all() and np.isfinite(p).all(), "指标输入含 NaN"
    row = {
        "n_events": int(len(df)),
        "base_rate": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
    }
    row.update(topk_precision(df, prob_col=prob_col, label_col=label_col, k=k))
    return row
