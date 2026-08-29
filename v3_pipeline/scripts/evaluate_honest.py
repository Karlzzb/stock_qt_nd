#!/usr/bin/env python3
"""
诚实评估脚本（V4 里程碑口径的唯一事实来源）。

对任一模型目录中的 pred_{h}.parquet（验证集）/ test_pred_{h}.parquet（测试集）
计算排名能力与可交易性指标。所有数字遵循以下修正口径：

1. Rank IC：每个 timestamp 内 pred 与 actual_return 的 Spearman 相关；
   ICIR = mean(IC)/std(IC)；另报 IC>0 占比。
2. 回测窗口不重叠：horizon=h 天时，将排序后的 timestamp 按相位 i%h 分成
   h 条互不重叠的路径，每条路径内部复利，跨路径汇总统计。
   （重叠窗口复利会严重 inflate 胜率/Sharpe/MaxDD——V3 假报告的教训之一。）
3. Sharpe = mean(period_return)/std(period_return) × sqrt(252/h)，
   按不重叠期收益计算，禁止"年化收益/年化波动"伪 Sharpe。
4. 交易成本：单边 0.13%，每期全换手双边 0.26%，直接从期收益扣除。
5. 基准：同期 universe（全部样本）等权平均收益，不扣成本。

用法：
    python v3_pipeline/scripts/evaluate_honest.py v3_pipeline/models/v4_0_0_clean
    python v3_pipeline/scripts/evaluate_honest.py <model_dir> --horizons 3d 10d 30d

pred parquet 需含列：timestamp, symbol, pred, actual_return
（actual_return 为小数收益，如 0.0325 = +3.25%）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST_PER_SIDE = 0.0013          # 单边 0.13%
ROUND_TRIP_COST = 2 * COST_PER_SIDE  # 每期全换手 0.26%
TRADING_DAYS_PER_YEAR = 252
TOP_NS = (1, 5, 10)


def horizon_days(h: str) -> int:
    return int(h.rstrip("d"))


def per_timestamp_ic(df: pd.DataFrame) -> pd.Series:
    """每个 timestamp 的 Spearman(pred, actual_return)。"""
    return df.groupby("timestamp").apply(
        lambda g: g["pred"].corr(g["actual_return"], method="spearman")
        if len(g) >= 10 else np.nan
    )


def spread_stats(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Top-N 与 Bot-N / universe 的毛收益差（每期平均）。"""
    top_spread, top_excess = [], []
    for _, g in df.groupby("timestamp"):
        g = g.sort_values("pred")
        if len(g) < 2 * top_n:
            continue
        bot, top = g["actual_return"].head(top_n).mean(), g["actual_return"].tail(top_n).mean()
        top_spread.append(top - bot)
        top_excess.append(top - g["actual_return"].mean())
    return {
        f"top{top_n}_bot{top_n}_spread": float(np.mean(top_spread)),
        f"top{top_n}_excess_vs_universe": float(np.mean(top_excess)),
    }


def non_overlapping_backtest(df: pd.DataFrame, h_days: int, top_n: int):
    """
    相位切分不重叠回测：h_days 条相位路径，每条内部顺序复利。
    返回 (net_paths, bench_paths)，各为 list[np.ndarray]（每条路径的期收益序列）。
    年化必须按单条路径做再跨路径平均——把 h 条路径的收益池化后当序列复利，
    会把时间跨度放大 h 倍（本脚本早期版本的 bug）。
    """
    ts = np.sort(df["timestamp"].unique())
    by_ts = {t: g for t, g in df.groupby("timestamp")}
    net_paths, bench_paths = [], []
    for phase in range(h_days):
        net, bench = [], []
        for t in ts[phase::h_days]:
            g = by_ts[t].sort_values("pred")
            if len(g) < top_n:
                continue
            gross = g["actual_return"].tail(top_n).mean()
            net.append(gross - ROUND_TRIP_COST)
            bench.append(g["actual_return"].mean())
        if len(net) >= 5:
            net_paths.append(np.array(net))
            bench_paths.append(np.array(bench))
    return net_paths, bench_paths


def annualize(paths: list, h_days: int) -> dict:
    """
    由若干条不重叠相位路径给出年化收益与 Sharpe。
    年化/累计：逐路径计算后取均值；Sharpe/胜率：期收益池化估计分布，
    按单路径频率 252/h 年化。
    """
    if not paths:
        return {"paths": 0}
    periods_per_year = TRADING_DAYS_PER_YEAR / h_days
    anns, cums = [], []
    for p in paths:
        years = len(p) / periods_per_year
        cum = float(np.prod(1 + p))
        cums.append(cum - 1)
        anns.append(cum ** (1 / years) - 1 if years > 0 else np.nan)
    pooled = np.concatenate(paths)
    std = pooled.std(ddof=1)
    sharpe = pooled.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan
    return {
        "paths": len(paths),
        "periods_per_path": int(np.mean([len(p) for p in paths])),
        "cum_return": float(np.mean(cums)),
        "annualized": float(np.mean(anns)),
        "sharpe": float(sharpe),
        "win_rate": float((pooled > 0).mean()),
        "mean_period_return": float(pooled.mean()),
    }


def evaluate_split(pred_path: Path, h: str) -> dict:
    df = pd.read_parquet(pred_path)
    # 列名归一化：训练脚本验证集输出 prediction，测试集输出 pred
    if "pred" not in df.columns and "prediction" in df.columns:
        df = df.rename(columns={"prediction": "pred"})
    h_days = horizon_days(h)
    ic = per_timestamp_ic(df).dropna()
    out = {
        "rows": len(df),
        "timestamps": int(df["timestamp"].nunique()),
        "rank_ic": float(ic.mean()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
        "ic_positive_share": float((ic > 0).mean()),
    }
    out.update(spread_stats(df, top_n=10))
    for n in TOP_NS:
        net_paths, bench_paths = non_overlapping_backtest(df, h_days, n)
        out[f"top{n}_net"] = annualize(net_paths, h_days)
        if n == TOP_NS[0]:
            out["benchmark"] = annualize(bench_paths, h_days)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--horizons", nargs="*", default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    horizons = args.horizons
    if horizons is None:
        horizons = sorted(
            p.stem.removeprefix("test_pred_")
            for p in args.model_dir.glob("test_pred_*.parquet")
        )
    if not horizons:
        sys.exit(f"no test_pred_*.parquet under {args.model_dir}")

    report = {}
    for h in horizons:
        report[h] = {}
        for split, fname in (("val", f"pred_{h}.parquet"), ("test", f"test_pred_{h}.parquet")):
            path = args.model_dir / fname
            if path.exists():
                report[h][split] = evaluate_split(path, h)

    for h, splits in report.items():
        for split, m in splits.items():
            print(f"\n=== {h} {split} === rows={m['rows']:,} ts={m['timestamps']}")
            print(f"  Rank IC {m['rank_ic']:+.4f}  ICIR {m['icir']:+.2f}  "
                  f"IC>0 {m['ic_positive_share']:.1%}")
            print(f"  Top10-Bot10 毛差 {m['top10_bot10_spread']:+.3%}/期  "
                  f"Top10-universe 毛超额 {m['top10_excess_vs_universe']:+.3%}/期")
            for n in TOP_NS:
                r = m[f"top{n}_net"]
                print(f"  Top{n} 净: 年化 {r.get('annualized', np.nan):+.1%}  "
                      f"Sharpe {r.get('sharpe', np.nan):+.2f}  "
                      f"胜率 {r.get('win_rate', np.nan):.1%}  "
                      f"({r.get('paths', 0)} 路径×{r.get('periods_per_path', 0)} 期)")
            b = m["benchmark"]
            print(f"  基准(universe): 年化 {b.get('annualized', np.nan):+.1%}  "
                  f"累计 {b.get('cum_return', np.nan):+.1%}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nsaved -> {args.json_out}")


if __name__ == "__main__":
    main()
