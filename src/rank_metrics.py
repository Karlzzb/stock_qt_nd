"""横截面排名口径评估指标（工业选股系统标准口径）。

模型输出只当"排名分"用，不当概率用。
所有指标均为逐日横截面计算后跨日平均，不使用任何绝对阈值，
因此跨 label、跨模型、跨时间可比——这是优化基准的正式口径。

指标语义（详见 docs 或 issue #16）：
- daily_rank_ic      日 Rank IC 均值（分数排名 vs 未来收益排名的 Spearman 相关）
- daily_rank_icir    IC 均值/标准差（稳定性，类比夏普）
- daily_ic_pos_ratio IC>0 天数占比
- daily_top{n}_hit   每日 top N 的二元达标率，跨日平均
- daily_top5_excess_ret  每日 top5 原始收益 − 当日全样本均值
- top5_turnover      相邻交易日 top5 名单更换比例
- quantile spread    五分层 Q5−Q1 收益差（单调性检验）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ROWS_IC = 5        # 当日不足 5 个信号不算 IC
MIN_ROWS_QUANTILE = 25  # 当日不足 25 个信号不分层


def daily_rank_metrics(
    pred_df: pd.DataFrame,
    ret_col: str,
    label_col: str = "label",
    proba_col: str = "y_pred_proba",
    top_ns=(1, 3, 5, 10),
    n_quantiles: int = 5,
) -> dict:
    """对单份 OOS 预测计算排名口径指标。

    pred_df 需含列：timestamp, symbol, label(0/1), y_pred_proba, ret_col(原始未来收益)。
    """
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    ics: list[float] = []
    top_hits: dict[int, list[float]] = {n: [] for n in top_ns}
    excess5: list[float] = []
    turnovers: list[float] = []
    q_spreads: list[float] = []
    q_level_rets: list[np.ndarray] = []
    prev_top5: set | None = None

    for _, day in df.groupby("date", sort=True):
        day = day.dropna(subset=[ret_col])
        n = len(day)
        if n == 0:
            continue

        if n >= MIN_ROWS_IC:
            ic = day[proba_col].corr(day[ret_col], method="spearman")
            if not np.isnan(ic):
                ics.append(ic)

        for k in top_ns:
            if n >= k:
                top_hits[k].append(float(day.nlargest(k, proba_col)[label_col].mean()))

        if n >= 5:
            top5 = day.nlargest(5, proba_col)
            excess5.append(float(top5[ret_col].mean() - day[ret_col].mean()))
            s5 = set(top5["symbol"])
            if prev_top5 is not None:
                turnovers.append(1.0 - len(s5 & prev_top5) / 5.0)
            prev_top5 = s5

        if n >= MIN_ROWS_QUANTILE:
            q = pd.qcut(day[proba_col], n_quantiles, labels=False, duplicates="drop")
            if q.nunique() == n_quantiles:
                lvl = day.groupby(q)[ret_col].mean().to_numpy()
                q_level_rets.append(lvl)
                q_spreads.append(float(lvl[-1] - lvl[0]))

    ics_arr = np.asarray(ics)
    out: dict[str, float] = {
        "daily_rank_ic": float(ics_arr.mean()) if len(ics_arr) else float("nan"),
        "daily_rank_ic_std": float(ics_arr.std(ddof=1)) if len(ics_arr) > 1 else float("nan"),
        "daily_rank_icir": (
            float(ics_arr.mean() / ics_arr.std(ddof=1)) if len(ics_arr) > 1 else float("nan")
        ),
        "daily_ic_pos_ratio": float((ics_arr > 0).mean()) if len(ics_arr) else float("nan"),
        "ic_ndays": int(len(ics_arr)),
        "daily_top5_excess_ret": float(np.mean(excess5)) if excess5 else float("nan"),
        "top5_turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        "quantile_5_1_spread": float(np.mean(q_spreads)) if q_spreads else float("nan"),
    }
    for k in top_ns:
        out[f"daily_top{k}_hit"] = float(np.mean(top_hits[k])) if top_hits[k] else float("nan")
    if q_level_rets:
        for i, v in enumerate(np.mean(q_level_rets, axis=0), start=1):
            out[f"quantile_q{i}_ret"] = float(v)
    return out
