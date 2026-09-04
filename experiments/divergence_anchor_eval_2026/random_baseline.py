"""
随机基线：与变体事件数相同的随机锚点，真低点覆盖率(≤5交易日)的期望值。
每股模拟：在该股 2026 有效区间内均匀随机取 m 个锚点（m=该股各变体事件数），
重复 R 次取均值。固定种子保证可复现。
"""
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path("/home/karl/repos/personal/stock_qt_nd/experiments/divergence_anchor_eval_2026")
DATA_DIR = Path("/home/karl/repos/personal/stock_qt_nd/stock_data/daily")

R = 100
THRESHOLDS = [3, 5, 10, 15, 17, 20]
rng = np.random.default_rng(20260903)

gt = pd.read_parquet(EXP_DIR / "ground_truth_lows_2026.parquet")
gt["low_date"] = pd.to_datetime(gt["low_date"])

results = {}
for name in ["v1", "v2"]:
    ev = pd.read_parquet(EXP_DIR / f"events_{name}.parquet")
    m_by_sym = ev.groupby("ts_code").size().to_dict()

    # 每股需要日期数：从 gt 的 low_date 还原位置需要日期表——重新轻量读取
    from concurrent.futures import ProcessPoolExecutor

    def dates_worker(p):
        try:
            df = pd.read_parquet(p, columns=["ts_code", "trade_date"])
        except Exception:
            return None
        if "ts_code" not in df.columns or len(df) < 100:
            return None
        df = df.sort_values("trade_date")
        return str(df["ts_code"].iloc[0]), pd.to_datetime(df["trade_date"]).to_numpy()

    files = sorted(DATA_DIR.glob("*.parquet"))
    date_index = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(dates_worker, [str(p) for p in files], chunksize=64):
            if res is not None:
                date_index[res[0]] = res[1]

    captured_total = {th: 0.0 for th in THRESHOLDS}
    n_lows = 0
    for sym, g in gt.groupby("ts_code"):
        dates = date_index.get(sym)
        if dates is None:
            continue
        pos = {d: i for i, d in enumerate(dates)}
        low_pos = np.array([pos[d.to_datetime64()] for d in g["low_date"]])
        m = m_by_sym.get(sym, 0)
        n_lows += len(low_pos)
        if m == 0:
            continue
        # 有效锚点区间：与真低点同一有效段（2026-01-01 至 末尾-20交易日）
        hi = len(dates) - 20
        lo = int(np.searchsorted(dates, np.datetime64("2026-01-01")))
        if hi <= lo:
            continue
        # R 次模拟：每次 m 个随机锚点，检查每个低点 ±th 内是否有锚点
        draws = rng.integers(lo, hi, size=(R, m))
        cap_r = {th: np.zeros(len(low_pos)) for th in THRESHOLDS}
        for r in range(R):
            d = np.abs(draws[r][:, None] - low_pos[None, :]).min(axis=0)
            for th in THRESHOLDS:
                cap_r[th] += (d <= th)
        for th in THRESHOLDS:
            captured_total[th] += (cap_r[th] / R).sum()

    results[name] = {
        f"random_capture_le{th}_pct": float(captured_total[th] / n_lows * 100) if n_lows else None
        for th in THRESHOLDS
    } | {"n_lows": int(n_lows)}
    line = "  ".join(f"≤{th}天={results[name][f'random_capture_le{th}_pct']:.1f}%" for th in THRESHOLDS)
    print(f"{name}: 随机基线覆盖率 {line}  (低点数 {n_lows})")

print(results)
