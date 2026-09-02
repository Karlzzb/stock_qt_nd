#!/usr/bin/env python3
"""事件×特征主表合并库（issue #22；issue #24 扩展来源 4）。

四来源合并为一张事件日快照特征主表：
  s1 事件级特征词典（feature_matrix，179 列，键 event_id）
  s2 特征工厂（factory_full，1603 列，键 ts_code+date）
  s3 日频特征缓存重建（v4daily_snapshot，键 ts_code+date）
  s4 T3 新特征事件日快照（t3_snapshot，issue #24，键 ts_code+date，可缺省）

纪律：
  - 泄漏列物理剔除：既定排除模式 = V4/V5 训练配置 exclude_patterns
    ∪ feature_engine 14 条黑名单；^rank_ 前缀规则对白名单
    {rank_return, rank_volume} 豁免（V2 因果横截面特征，非标签排名）。
  - 同式列去重：|ρ|≥0.999（两池 train+val 段合并计算，不涉标签），
    保留优先级 s1 > s2 > s3 > s4，同优先级按列名字典序。
  - 列名跨源碰撞：值同去重、值异报错浮出（禁止静默改名）。
"""
import re

import numpy as np
import pandas as pd

# 既定排除模式（v4_0_0_clean.yaml + v5_label_open_exec.yaml + feature_engine 黑名单）
EXCLUDE_PATTERNS = [
    r"^stop_loss_",
    r"^future_",
    r"^next_",
    r"^label",
    r"^mfr_",
    r"^cur_return$",
    r"^max_forward_return$",
    r"^open_exec_return",
    r"^rank_",
    r"^ret_h\d+$",
    r"^hit_N",
    r"^mfe_",
    r"^mae_",
    r"^tmfe",
    r"^tmae",
    r"^dyn_",
    r"^entry_date$",
]
# ^rank_ 豁免: V2 全市场当日横截面百分位特征（因果，t 日收盘可得）
RANK_WHITELIST = frozenset({"rank_return", "rank_volume"})

_EXCLUDE_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_PATTERNS]

# 主表元数据列（非特征）：事件表自带字段 + 段标签
EVENT_META_COLS = ["event_id", "ts_code", "date", "sig_idx", "low_date",
                   "prev_low_date", "compare_rank", "formation", "regime",
                   "above_ma200", "seg"]

KEY = ["ts_code", "date"]

# 切分段（口径 = run_pool_cleaning.py:58-61，隔离带为段界内 30 交易日删除带）
TRAIN_LO, TRAIN_HI = pd.Timestamp("2001-01-01"), pd.Timestamp("2018-12-31")
VAL_LO, VAL_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-10-31")
EMBARGO = [(pd.Timestamp("2018-11-19"), pd.Timestamp("2018-12-28")),
           (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-10-31"))]

DEDUP_THRESHOLD = 0.999
SOURCE_PRIORITY = {"s1": 0, "s2": 1, "s3": 2, "s4": 3}


def segment_of(dates):
    """事件日所属切分段: pre2001/train/val/embargo/test。"""
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    seg = np.full(len(dates), "test", dtype=object)
    seg[dates < TRAIN_LO] = "pre2001"
    seg[(dates >= TRAIN_LO) & (dates <= TRAIN_HI)] = "train"
    seg[(dates >= VAL_LO) & (dates <= VAL_HI)] = "val"
    for lo, hi in EMBARGO:
        seg[(dates >= lo) & (dates <= hi)] = "embargo"
    return seg


def excluded_columns(columns):
    """命中排除模式且不在白名单的列名列表。"""
    return [c for c in columns
            if c not in RANK_WHITELIST and any(rx.match(c) for rx in _EXCLUDE_RE)]


def assert_no_leakage(columns):
    bad = excluded_columns(columns)
    assert not bad, f"主表列命中泄漏排除模式: {bad[:20]}"


def merge_sources(events, s1, s2, s3, s4=None):
    """事件表为底, 左连接各来源。返回 (合并df, 列来源映射, 碰撞记录)。

    events: 锁定事件表（已剔指数伪股）; s1 键 event_id; s2/s3/s4 键 (ts_code,date)。
    s4 = T3 新特征快照（issue #24），可缺省（None 则退化为三来源行为）。
    碰撞列值同则保留高优先级源, 值异则 ValueError。
    """
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    if "seg" not in ev.columns:
        ev["seg"] = segment_of(ev["date"])

    s1 = s1.drop(columns=[c for c in ("ts_code", "date", "sig_idx")
                          if c in s1.columns])
    df = ev.merge(s1, on="event_id", how="left", validate="1:1")
    src_of = {c: "s1" for c in s1.columns if c != "event_id"}

    collisions = []
    for tag, s in (("s2", s2), ("s3", s3), ("s4", s4)):
        if s is None:
            continue
        s = s.copy()
        s["date"] = pd.to_datetime(s["date"])
        overlap = [c for c in s.columns if c not in KEY and c in df.columns]
        keep_cols = KEY[:]
        for c in overlap:
            j = df[[*KEY, c]].merge(s[[*KEY, c]], on=KEY, how="outer",
                                    suffixes=("_a", "_b"))
            a, b = j[f"{c}_a"], j[f"{c}_b"]
            # NaN 不对称（一侧有值一侧缺失）即值异
            asym = (a.isna() != b.isna()).any()
            both = a.notna() & b.notna()
            if both.any():
                if pd.api.types.is_numeric_dtype(a) and \
                        pd.api.types.is_numeric_dtype(b):
                    same_vals = np.allclose(a[both].to_numpy(np.float64),
                                            b[both].to_numpy(np.float64),
                                            rtol=1e-9, atol=0)
                else:
                    same_vals = bool((a[both] == b[both]).all())
            else:
                same_vals = True
            if asym or not same_vals:
                raise ValueError(f"列名碰撞且值不同: {c} (源 {src_of.get(c)} vs {tag})")
            # 值同: 低优先级源的副本不入表
            collisions.append({"column": c, "kept_source": src_of.get(c),
                               "dropped_source": tag})
        keep_cols += [c for c in s.columns if c not in KEY and c not in overlap]
        df = df.merge(s[keep_cols], on=KEY, how="left", validate="m:1")
        for c in keep_cols:
            if c not in KEY:
                src_of[c] = tag
    return df, src_of, collisions


def feature_columns(df):
    """数值特征列（元数据与键之外；bool 视同 0/1 数值）。"""
    return [c for c in df.columns if c not in EVENT_META_COLS
            and df[c].dtype.kind in "fib"]


def pairwise_corr(x, min_pairs=30):
    """精确成对完全观测相关系数矩阵（pairwise-complete, NaN 感知）。

    x: (n_rows, n_cols) float64。对每对 (i,j) 只在两列都非 NaN 的行上计算。
    成对观测数 < min_pairs 时记 NaN（不判重）。
    """
    z = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(np.float64)
    n_ij = m.T @ m                       # 成对观测数
    sx = z.T @ m                         # sx[i,j] = Σ_r x_ri over rows where j present
    sxx = (z * z).T @ m
    sxy = z.T @ z                        # 交叉项（z 在缺失处已为 0）
    with np.errstate(invalid="ignore", divide="ignore"):
        n_safe = np.where(n_ij > 0, n_ij, np.nan)
        mean_i = sx / n_safe             # mean_i[j 视角]: x_i 在 j 完全行上的均值
        mean_j = sx.T / n_safe
        cov = sxy / n_safe - mean_i * mean_j
        var_i = sxx / n_safe - mean_i ** 2
        var_j = sxx.T / n_safe - mean_j ** 2
        denom = np.sqrt(np.clip(var_i, 0, None) * np.clip(var_j, 0, None))
        corr = np.where(denom > 0, cov / denom, np.nan)
    corr[n_ij < min_pairs] = np.nan
    np.fill_diagonal(corr, 1.0)
    return corr


def dedup_by_correlation(df, feat_cols, row_mask, threshold=DEDUP_THRESHOLD,
                         src_of=None):
    """|ρ|>=threshold 同式列去重（贪心: 按源优先级+列名序, 先留后剔）。

    返回 (keep_cols, drop_records)。drop_records: [(dropped, anchor, rho)]。
    """
    if src_of is None:
        src_of = {}
    sub = df.loc[row_mask, feat_cols]
    x = sub.to_numpy(np.float64)
    corr = pairwise_corr(x)
    order = sorted(range(len(feat_cols)),
                   key=lambda i: (SOURCE_PRIORITY.get(src_of.get(feat_cols[i], "s3"), 9),
                                  feat_cols[i]))
    keep_idx, dropped = [], {}
    for i in order:
        if i in dropped:
            continue
        keep_idx.append(i)
        for j in order:
            if j in dropped or j in keep_idx or j == i:
                continue
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                dropped[j] = (feat_cols[i], float(r))
    keep = [feat_cols[i] for i in keep_idx]
    records = [(feat_cols[j],) + v for j, v in dropped.items()]
    return keep, records, corr


def assert_dedup_clean(corr, keep, feat_cols, threshold=DEDUP_THRESHOLD):
    """去重后断言: 保留列两两 |ρ| < threshold。"""
    pos = {c: i for i, c in enumerate(feat_cols)}
    idx = [pos[c] for c in keep]
    sub = corr[np.ix_(idx, idx)]
    offdiag = sub[~np.eye(len(idx), dtype=bool)]
    offdiag = offdiag[np.isfinite(offdiag)]
    if len(offdiag):
        mx = float(np.max(np.abs(offdiag)))
        assert mx < threshold, f"去重后仍存 |ρ|={mx:.6f} >= {threshold}"
