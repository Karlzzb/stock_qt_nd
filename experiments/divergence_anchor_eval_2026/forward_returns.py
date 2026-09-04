"""
N 日后平均涨幅统计：两变体信号的锚点日与事件日前瞻收益。

口径：
- 前瞻收益 = close[t+N] / close[t] - 1（收盘对收盘，不含成本，纯价格行为统计）。
- t 分别取 anchor_date（锚点日）与 event_date（事件日=信号可知晓日）。
- N ∈ {5, 10, 20, 30, 60} 交易日；数据末端不足 N 日的事件该档剔除。
- 基线：全市场 2026 年全部交易日（同样要求 N 日后有数据）的无条件前瞻收益，
  与信号事件同窗同口径，作为"随便挑一天买"的对照。
- 数据：未复权收盘价（两变体与基线同基准，公平）。
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path("/home/karl/repos/personal/stock_qt_nd/experiments/divergence_anchor_eval_2026")
DATA_DIR = Path("/home/karl/repos/personal/stock_qt_nd/stock_data/daily")
NS = [5, 10, 20, 30, 60]
MIN_ROWS = 100


def _load_one(p):
    try:
        df = pd.read_parquet(p, columns=["ts_code", "trade_date", "close"])
    except Exception:
        return None
    if "ts_code" not in df.columns or len(df) < MIN_ROWS:
        return None
    df = df.sort_values("trade_date")
    return (str(df["ts_code"].iloc[0]),
            pd.to_datetime(df["trade_date"]).to_numpy(),
            df["close"].to_numpy(dtype=np.float64))


def load_close_maps():
    """每股：日期→位置、close 数组。返回 {ts_code: (dates_array, close_array)}"""
    files = sorted(DATA_DIR.glob("*.parquet"))
    out = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(_load_one, [str(p) for p in files], chunksize=64):
            if res is not None:
                out[res[0]] = (res[1], res[2])
    return out


def fwd_returns(dates, close, ref_dates, n):
    """ref_dates 中每个日期 t（须在 dates 内）的 N 日前瞻收益；不足 N 日剔除。"""
    pos = {d: i for i, d in enumerate(dates)}
    rets = []
    for d in ref_dates:
        i = pos.get(pd.Timestamp(d).to_datetime64())
        if i is None or i + n >= len(close):
            continue
        rets.append(close[i + n] / close[i] - 1.0)
    return np.array(rets)


def summarize(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"n": 0, "mean": None, "median": None, "win": None}
    return {"n": int(len(rets)),
            "mean": float(np.mean(rets) * 100),
            "median": float(np.median(rets) * 100),
            "win": float(np.mean(rets > 0) * 100)}


def main():
    data = load_close_maps()
    print(f"[load] 股票 {len(data)} 只")

    # 基线：全市场 2026 年全部交易日的无条件前瞻收益
    baseline = {n: [] for n in NS}
    for ts_code, (dates, close) in data.items():
        d2026 = dates[(dates >= np.datetime64("2026-01-01")) &
                      (dates <= np.datetime64("2026-08-31"))]
        for n in NS:
            baseline[n].append(fwd_returns(dates, close, d2026, n))
    baseline = {n: np.concatenate(v) for n, v in baseline.items()}

    results = {}
    for name in ["v1", "v2"]:
        ev = pd.read_parquet(EXP_DIR / f"events_{name}.parquet")
        res = {"anchor": {n: [] for n in NS}, "event": {n: [] for n in NS}}
        for ts_code, g in ev.groupby("ts_code"):
            entry = data.get(ts_code)
            if entry is None:
                continue
            dates, close = entry
            for n in NS:
                res["anchor"][n].append(fwd_returns(dates, close, g["anchor_date"].to_numpy(), n))
                res["event"][n].append(fwd_returns(dates, close, g["event_date"].to_numpy(), n))
        results[name] = {
            ref: {n: summarize(np.concatenate(v)) for n, v in refd.items()}
            for ref, refd in res.items()
        }

    # 出表：每档 N 一行，均值%/中位%/胜率
    def row(label, s):
        if s["n"] == 0:
            return f"{label:<26}{'—':>8}"
        return (f"{label:<26}{s['n']:>8}{s['mean']:>9.2f}{s['median']:>9.2f}{s['win']:>8.1f}")

    for ref, title in [("anchor", "锚点日（买在低点）"), ("event", "事件日（信号可下手日）")]:
        print(f"\n=== {title} 起算 ===")
        print(f"{'口径':<26}{'笔数':>8}{'均值%':>9}{'中位%':>9}{'胜率%':>8}")
        for n in NS:
            print(row(f"  v1  +{n}日", results["v1"][ref][n]))
            print(row(f"  v2  +{n}日", results["v2"][ref][n]))
            b = summarize(baseline[n])
            print(row(f"  基线 +{n}日", b))

    import json
    out = {"baseline": {str(n): summarize(baseline[n]) for n in NS}, **results}
    (EXP_DIR / "forward_returns.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {EXP_DIR/'forward_returns.json'}")


if __name__ == "__main__":
    main()
