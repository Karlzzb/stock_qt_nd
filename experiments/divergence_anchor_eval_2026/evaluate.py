"""
统一评测：变体1（区间最低价锚定） vs 变体2（右侧确认精确锚定）

口径（预登记，2026-09-03 与用户确认）：
- 真低点基准：每股全历史收盘序列上，交易日 t 满足
    close[t] == min(close[t-20 .. t+20])  且  close[t] < min(close[t-20 .. t-1])
  （±20 交易日窗口内最低；并列取最早——严格低于前窗即排除后者）
  且 t 落在 2026-01-01 之后、且 t 之后仍有 20 个交易日数据（右侧窗口完整）。
- 指标：
  1. 事件数、覆盖股票数
  2. 锚点距最近真低点的交易日距离：中位数/均值/≤3/≤5/≤10 占比
  3. 锚点价相对最近真低点价的溢价：(anchor_close / low_close - 1) 中位数/均值
  4. 真低点覆盖率：距某事件锚点 ≤3 / ≤5 / ≤10 个交易日的真低点占比（“拿到的最低点多”）
- 两变体共用同一份真低点基准、同一股票宇宙（与扫描器一致的 schema 过滤：需 ts_code 且 vol 列、≥100 行）。

输入：events_v1.parquet / events_v2.parquet（同目录）
输出：eval_summary.json + stdout 对照表
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/home/karl/repos/personal/stock_qt_nd/stock_data/daily")
EXP_DIR = Path("/home/karl/repos/personal/stock_qt_nd/experiments/divergence_anchor_eval_2026")

GT_START = pd.Timestamp("2026-01-01")
HALF_WIN = 20  # 真低点判定窗口半径（交易日）
MIN_ROWS = 100
DIST_THRESHOLDS = [3, 5, 10, 15, 17, 20]  # 距离/覆盖率阈值（交易日）


# ---------------------------------------------------------------------------
# Phase 1：每股真低点基准 + 日期位置表（位置表落盘供 Phase 2 复用，避免二次全量读盘）
# ---------------------------------------------------------------------------

def gt_worker(path_str: str):
    """返回 (ts_code, lows: list[(date, close)], dates: list[date]) 或 None。"""
    path = Path(path_str)
    try:
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "close"])
    except Exception:
        return None
    if "ts_code" not in df.columns or len(df) < MIN_ROWS:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(df["trade_date"]).to_numpy()
    n = len(df)
    lows = []
    for i in range(HALF_WIN, n - HALF_WIN):
        c = close[i]
        win = close[i - HALF_WIN: i + HALF_WIN + 1]
        if c == win.min() and c < close[i - HALF_WIN: i].min():
            d = dates[i]
            if d >= GT_START.to_datetime64():
                lows.append((d, float(c)))
    ts_code = str(df["ts_code"].iloc[0])
    return ts_code, lows, dates


def build_ground_truth(files: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    lows_rows = []
    date_index: dict[str, np.ndarray] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(gt_worker, files, chunksize=64):
            if res is None:
                continue
            ts_code, lows, dates = res
            date_index[ts_code] = dates
            for d, c in lows:
                lows_rows.append((ts_code, d, c))
    gt = pd.DataFrame(lows_rows, columns=["ts_code", "low_date", "low_close"])
    print(f"[gt] 真低点 {len(gt)} 个 / 股票 {gt['ts_code'].nunique()} 只 / 耗时 {time.time()-t0:.1f}s")
    return gt, date_index


# ---------------------------------------------------------------------------
# Phase 2：距离与覆盖率
# ---------------------------------------------------------------------------

def nearest_low_metrics(events: pd.DataFrame, gt: pd.DataFrame,
                        date_index: dict[str, np.ndarray]) -> dict:
    gt_by_sym = {sym: g.sort_values("low_date").reset_index(drop=True)
                 for sym, g in gt.groupby("ts_code")}

    dist = []          # 每事件：锚点到最近真低点的交易日距离
    premium = []       # 每事件：锚点价 / 最近真低点价 - 1
    no_gt_events = 0   # 该股 2026 无真低点的事件数
    # 覆盖率：每个真低点到最近锚点的距离
    low_min_dist: list[int] = []

    ev_by_sym = {sym: g for sym, g in events.groupby("ts_code")}

    for sym, g in gt_by_sym.items():
        dates = date_index.get(sym)
        if dates is None:
            continue
        pos = {d: i for i, d in enumerate(dates)}
        low_pos = np.array([pos[d] for d in g["low_date"].to_numpy()])
        ev = ev_by_sym.get(sym)
        if ev is not None:
            anchors = ev["anchor_date"].to_numpy()
            anchor_pos = np.array([pos.get(pd.Timestamp(a).to_datetime64(), -1) for a in anchors])
            anchor_pos = anchor_pos[anchor_pos >= 0]
            # 每个真低点到最近锚点的距离
            for lp in low_pos:
                if len(anchor_pos):
                    low_min_dist.append(int(np.abs(anchor_pos - lp).min()))
                else:
                    low_min_dist.append(10 ** 9)
        else:
            low_min_dist.extend([10 ** 9] * len(low_pos))

    for sym, ev in ev_by_sym.items():
        g = gt_by_sym.get(sym)
        dates = date_index.get(sym)
        if g is None or dates is None or len(g) == 0:
            no_gt_events += len(ev)
            continue
        pos = {d: i for i, d in enumerate(dates)}
        low_pos = np.array([pos[d] for d in g["low_date"].to_numpy()])
        low_close = g["low_close"].to_numpy()
        low_dates = g["low_date"].to_numpy()
        for a_date, a_close in zip(ev["anchor_date"].to_numpy(), ev["anchor_close"].to_numpy()):
            ap = pos.get(pd.Timestamp(a_date).to_datetime64(), -1)
            if ap < 0:
                continue
            d = np.abs(low_pos - ap)
            j = int(np.argmin(d))
            # 距离并列取日期更早的真低点（确定性）
            ties = np.flatnonzero(d == d[j])
            if len(ties) > 1:
                j = int(ties[np.argsort(low_dates[ties])[0]])
            dist.append(int(d[j]))
            premium.append(float(a_close / low_close[j] - 1.0))

    dist = np.array(dist)
    premium = np.array(premium)
    low_min_dist = np.array(low_min_dist)

    n_lows = len(low_min_dist)
    result = {
        "events_total": int(len(events)),
        "symbols_with_events": int(events["ts_code"].nunique()),
        "events_no_gt_low": int(no_gt_events),
        "events_scored": int(len(dist)),
        "dist_median": float(np.median(dist)) if len(dist) else None,
        "dist_mean": float(np.mean(dist)) if len(dist) else None,
        "premium_median_pct": float(np.median(premium) * 100) if len(premium) else None,
        "premium_mean_pct": float(np.mean(premium) * 100) if len(premium) else None,
        "gt_lows_total": int(n_lows),
    }
    for th in DIST_THRESHOLDS:
        result[f"dist_le{th}_pct"] = float(np.mean(dist <= th) * 100) if len(dist) else None
        result[f"capture_le{th}_pct"] = float(np.mean(low_min_dist <= th) * 100) if n_lows else None
        result[f"captured_le{th}_count"] = int(np.sum(low_min_dist <= th))
    return result


def main():
    files = [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))]
    print(f"[load] 文件 {len(files)} 个")
    gt, date_index = build_ground_truth(files)
    gt.to_parquet(EXP_DIR / "ground_truth_lows_2026.parquet", index=False)

    results = {}
    for name in ["v1", "v2"]:
        events = pd.read_parquet(EXP_DIR / f"events_{name}.parquet")
        results[name] = nearest_low_metrics(events, gt, date_index)

    # 对照表（配置为行）
    rows = {
        "事件总数": ("events_total", "{:d}"),
        "覆盖股票数": ("symbols_with_events", "{:d}"),
        "无基准低点事件数": ("events_no_gt_low", "{:d}"),
        "参与距离评分事件数": ("events_scored", "{:d}"),
        "锚点距离中位数(交易日)": ("dist_median", "{:.1f}"),
        "锚点距离均值(交易日)": ("dist_mean", "{:.2f}"),
        "锚点溢价中位数(%)": ("premium_median_pct", "{:.2f}"),
        "锚点溢价均值(%)": ("premium_mean_pct", "{:.2f}"),
        "基准真低点总数": ("gt_lows_total", "{:d}"),
    }
    for th in DIST_THRESHOLDS:
        rows[f"锚点距离≤{th}天占比(%)"] = (f"dist_le{th}_pct", "{:.1f}")
    for th in DIST_THRESHOLDS:
        rows[f"真低点覆盖率≤{th}天(%)"] = (f"capture_le{th}_pct", "{:.1f}")
    for th in DIST_THRESHOLDS:
        rows[f"覆盖真低点个数(≤{th}天)"] = (f"captured_le{th}_count", "{:d}")
    header = f"{'指标':<28}{'变体1(区间最低)':>16}{'变体2(右侧确认)':>16}"
    print("\n" + header)
    print("-" * len(header))
    for label, (key, fmt) in rows.items():
        a = results["v1"][key]
        b = results["v2"][key]
        sa = fmt.format(a) if a is not None else "—"
        sb = fmt.format(b) if b is not None else "—"
        print(f"{label:<28}{sa:>16}{sb:>16}")

    out = {
        "ground_truth": "close[t] == min(close[t-20..t+20]) 且严格低于前20日；t∈[2026-01-01, 数据末尾-20交易日]",
        "v1": results["v1"],
        "v2": results["v2"],
    }
    (EXP_DIR / "eval_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 写入 {EXP_DIR/'eval_summary.json'}")


if __name__ == "__main__":
    main()
