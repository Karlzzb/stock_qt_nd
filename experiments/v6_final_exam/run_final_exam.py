#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 战役终审脚本(预登记冻结版,规格见 PREREG.md)。
单发:test 段唯一宣判格 = 模型A×种子并集 top30%,#31 五线 + 胜对照 +2pp。
指标函数逐字锚定 experiments/divergence_seed_trial_history/run_seeds.py 存档口径。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
TRADES = REPO / "experiments/divergence_seed_trial_history/trades_seed.parquet"
SCORES_A = REPO / "experiments/v2_on_v6_rerun/cache/scores_v2on6_p7fix.parquet"
SCORES_B = REPO / "experiments/v6_model_campaign/m4_training_selection/scores_v6.parquet"
OUT = REPO / "experiments/v6_final_exam"

# 归档口径原函数复用(run_seeds.py L437-451,import 安全:模块有 __main__ 守卫)
sys.path.insert(0, str(REPO / "experiments/divergence_seed_trial_history"))
from run_seeds import cluster_t  # noqa: E402


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def year_share(cl: pd.DataFrame) -> float:
    """win_year_share,逐字 run_seeds.py L555-569:按入场年分组,年成交>=30 计资格,
    资格年内净笔均>0 的占比;无资格年记 NaN。"""
    if len(cl) == 0:
        return np.nan
    yr = cl["entry_date"].dt.year
    grp = cl.groupby(yr)["net_ret"]
    ymean, ycount = grp.mean(), grp.count()
    qual = [y for y in ymean.index if ycount.loc[y] >= 30]
    if not qual:
        return np.nan
    return float(np.mean([ymean.loc[y] > 0 for y in qual]))


def date_conc(cl: pd.DataFrame) -> float:
    """date_conc,逐字 run_seeds.py L570-577:top5 入场日 net_pnl 合计 / 全部合计(>0 时)。"""
    if len(cl) == 0:
        return np.nan
    pnl_by_day = cl.groupby(cl["entry_date"].dt.date)["net_pnl"].sum()
    total = float(pnl_by_day.sum())
    top5 = float(pnl_by_day.sort_values(ascending=False).head(5).sum())
    return top5 / total if total > 0 else np.nan


def metrics(cl: pd.DataFrame) -> dict:
    ret = cl["net_ret"].to_numpy(dtype=float)
    n = len(cl)
    ct = cluster_t(ret, cl["entry_date"].astype(str).to_numpy()) if n else np.nan
    return dict(
        n_closed=n,
        win_rate=float((ret > 0).mean()) if n else np.nan,
        net_mean=float(ret.mean()) if n else np.nan,
        net_median=float(np.median(ret)) if n else np.nan,
        cluster_t=ct,
        win_year_share=year_share(cl),
        date_conc=date_conc(cl),
    )


def five_lines(m: dict) -> dict:
    """逐字 run_seeds.py L675-686。"""
    return dict(
        c1_n_closed_ge_300=bool(m["n_closed"] >= 300),
        c2_net_mean_pos=bool(np.isfinite(m["net_mean"]) and m["net_mean"] > 0),
        c3_cluster_t_ge_2=bool(np.isfinite(m["cluster_t"]) and m["cluster_t"] >= 2),
        c4_win_year_share_ge_60pct=bool(
            np.isfinite(m["win_year_share"]) and m["win_year_share"] >= 0.6),
        c5_date_conc_le_50pct=bool(np.isfinite(m["date_conc"]) and m["date_conc"] <= 0.5),
    )


def top_x(df: pd.DataFrame, x: float) -> pd.DataFrame:
    """注册排名口径:score 降序,ts_code/event_date 升序,mergesort,head(ceil(n*x))。"""
    s = df.sort_values(["score", "ts_code", "event_date"],
                       ascending=[False, True, True], kind="mergesort")
    return s.head(int(np.ceil(len(s) * x)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(TRADES)
    tr = tr[(tr["variant"] == "v1") & (tr["H"] == 20) & (tr["status"] == "closed")].copy()
    tr["event_date"] = pd.to_datetime(tr["event_date"])
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    assert not tr.duplicated(["ts_code", "event_date"]).any()

    sa = pd.read_parquet(SCORES_A, columns=["ts_code", "date", "seg", "score"])
    sb = pd.read_parquet(SCORES_B, columns=["ts_code", "date", "seg", "score"])
    for s in (sa, sb):
        s["date"] = pd.to_datetime(s["date"])
    sa_t = sa[sa["seg"] == "test"].rename(columns={"score": "score_A"})
    sb_t = sb[sb["seg"] == "test"].rename(columns={"score": "score_B"})

    j = tr.merge(sa_t[["ts_code", "date", "score_A"]],
                 left_on=["ts_code", "event_date"], right_on=["ts_code", "date"],
                 how="inner", validate="1:1")
    j = j.merge(sb_t[["ts_code", "date", "score_B"]],
                left_on=["ts_code", "event_date"], right_on=["ts_code", "date"],
                how="inner", validate="1:1")
    print(f"test 段成交(join 后): {len(j)};并集(sel_S4): {int(j['sel_S4'].sum())}")

    uni = j[j["sel_S4"]]
    slices = {
        "pool_ALL": (j, "info"),
        "union_S4": (uni, "control"),
        "seed_S1": (j[j["sel_S1"]], "info"),
        "seed_S2": (j[j["sel_S2"]], "info"),
        "seed_S3": (j[j["sel_S3"]], "info"),
        "seed_S4": (j[j["sel_S4"]], "info"),
        "seed_S5": (j[j["sel_S5"]], "info"),
        "AxS4_top30": (top_x(uni.rename(columns={"score_A": "score"}), 0.30), "primary"),
        "AxS4_top10": (top_x(uni.rename(columns={"score_A": "score"}), 0.10), "info"),
        "AxS4_top50": (top_x(uni.rename(columns={"score_A": "score"}), 0.50), "info"),
        "Axpool_top5": (top_x(j.rename(columns={"score_A": "score"}), 0.05), "info"),
        "Axpool_top10": (top_x(j.rename(columns={"score_A": "score"}), 0.10), "info"),
        "BxS4_top10": (top_x(uni.rename(columns={"score_B": "score"}), 0.10), "info"),
        "BxS4_top30": (top_x(uni.rename(columns={"score_B": "score"}), 0.30), "info"),
        "BxS4_top50": (top_x(uni.rename(columns={"score_B": "score"}), 0.50), "info"),
    }

    rows, store = [], {}
    for name, (sl, role) in slices.items():
        m = metrics(sl)
        rows.append(dict(slice=name, role=role, **m))
        store[name] = m
        print(f"  {name:14s} n={m['n_closed']:5d} 胜率={m['win_rate']*100:6.2f}% "
              f"净笔均={m['net_mean']*100:+7.3f}% t={m['cluster_t']:+6.2f} "
              f"年占比={m['win_year_share']} 集中度={m['date_conc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_final_exam.csv", index=False, float_format="%.8f")

    pm, cm = store["AxS4_top30"], store["union_S4"]
    crit = five_lines(pm)
    margin = pm["net_mean"] - cm["net_mean"]
    crit["c6_margin_ge_2pp_vs_union"] = bool(margin >= 0.02)
    passed = all(crit.values())
    verdict = dict(
        prereg="experiments/v6_final_exam/PREREG.md",
        primary_slice="AxS4_top30",
        control_slice="union_S4",
        primary_metrics=pm,
        control_metrics=cm,
        margin_pp=float(margin * 100),
        checks=crit,
        verdict="ROUTE_ALIVE" if passed else "CAMPAIGN_DEAD",
        inputs_md5={
            "trades_seed.parquet": md5(TRADES),
            "scores_v2on6_p7fix.parquet": md5(SCORES_A),
            "scores_v6.parquet": md5(SCORES_B),
        },
    )
    (OUT / "verdict_final_exam.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n宣判: {verdict['verdict']} | 边际 {margin*100:+.3f}pp | checks={crit}")


if __name__ == "__main__":
    main()
