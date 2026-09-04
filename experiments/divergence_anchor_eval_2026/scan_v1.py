"""金叉对金叉 MACD 底背离扫描器 —— 变体 1（区间最低价锚定版）。

口径（钉死）：
- DIF, DEA, _ = talib.MACD(close)，默认 12/26/9，按个股全历史一次性计算，不复权 close。
- 金叉日 t：DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]。
- 对相邻金叉对 (C(k-1), C(k))（需 k>=2 且存在 C(k-2)，即至少 3 次金叉）：
  1. dif_lift = DIF[C(k)] - DIF[C(k-1)] >= 0.001；
  2. 区间 (C(k-2), C(k-1)] 最低收盘价 > 区间 (C(k-1), C(k)] 最低收盘价（后者严格新低）；
  3. 两条都满足 -> 事件：event_date = C(k)，anchor_date = 区间 (C(k-1), C(k)]
     最低收盘价所在日（并列取最早），anchor_close = 该日收盘价。
  区间端点语义：左开右闭，按交易日位置（行号）切。
- 只保留 event_date 落在 2026-01-01 至 2026-08-31 的事件。
- 按 schema 过滤：只处理含 ts_code 且含 vol 列的文件；行数 < 100 跳过。
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

DATA_DIR = Path("/home/karl/repos/personal/stock_qt_nd/stock_data/daily")
OUT_DIR = Path("/home/karl/repos/personal/stock_qt_nd/experiments/divergence_anchor_eval_2026")
EVENT_START = pd.Timestamp("2026-01-01")
EVENT_END = pd.Timestamp("2026-08-31")
MIN_ROWS = 100
DIF_LIFT_MIN = 0.001


def scan_one(path: Path) -> tuple[str | None, list[dict], str | None]:
    """扫描单只股票，返回 (ts_code, events, skip_reason)。"""
    df = pd.read_parquet(path)
    if "ts_code" not in df.columns or "vol" not in df.columns:
        return None, [], "schema"
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None, [], "short"

    close = df["close"].to_numpy(dtype=np.float64)
    dates = df["trade_date"].to_numpy()
    dif, dea, _ = talib.MACD(close)  # 默认 12/26/9
    if np.isnan(dif).all():
        return None, [], "nan"

    # 金叉：DIF[t-1] <= DEA[t-1] 且 DIF[t] > DEA[t]
    prev_le = (dif[:-1] <= dea[:-1]) & ~np.isnan(dif[:-1]) & ~np.isnan(dea[:-1])
    now_gt = (dif[1:] > dea[1:]) & ~np.isnan(dif[1:]) & ~np.isnan(dea[1:])
    crosses = np.nonzero(prev_le & now_gt)[0] + 1  # 行号（t）

    ts_code = str(df["ts_code"].iloc[0])
    events: list[dict] = []
    # 需要 C(k-2) 存在，即至少 3 次金叉，从第 3 个金叉起判定（k 从 2 起，0-based）
    for k in range(2, len(crosses)):
        c_km2, c_km1, c_k = crosses[k - 2], crosses[k - 1], crosses[k]
        dif_lift = dif[c_k] - dif[c_km1]
        if dif_lift < DIF_LIFT_MIN:
            continue
        # 左开右闭：(c_km2, c_km1] 与 (c_km1, c_k]
        seg_prev = close[c_km2 + 1 : c_km1 + 1]
        seg_cur = close[c_km1 + 1 : c_k + 1]
        if len(seg_prev) == 0 or len(seg_cur) == 0:
            continue
        min_prev = seg_prev.min()
        idx_cur_rel = int(np.argmin(seg_cur))  # 并列取最早
        min_cur = seg_cur[idx_cur_rel]
        if not (min_prev > min_cur):
            continue
        anchor_idx = c_km1 + 1 + idx_cur_rel
        events.append(
            {
                "ts_code": ts_code,
                "event_date": pd.Timestamp(dates[c_k]),
                "anchor_date": pd.Timestamp(dates[anchor_idx]),
                "anchor_close": float(close[anchor_idx]),
                "cross_prev_date": pd.Timestamp(dates[c_km1]),
                "cross_prev_dif": float(dif[c_km1]),
                "cross_date": pd.Timestamp(dates[c_k]),
                "cross_dif": float(dif[c_k]),
                "dif_lift": float(dif_lift),
            }
        )
    return ts_code, events, None


def worker(path: str):
    try:
        return scan_one(Path(path))
    except Exception as e:  # noqa: BLE001
        return None, [], f"error:{e}"


def main() -> None:
    t0 = time.time()
    files = sorted(str(p) for p in DATA_DIR.glob("*.parquet"))
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[scan_v1] files found: {len(files)}")

    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        results = pool.map(worker, files, chunksize=32)

    n_files = len(results)
    skip_schema = sum(1 for _, _, r in results if r == "schema")
    skip_short = sum(1 for _, _, r in results if r == "short")
    skip_nan = sum(1 for _, _, r in results if r == "nan")
    skip_err = sum(1 for _, _, r in results if r and r.startswith("error:"))
    n_stocks = sum(1 for ts, _, r in results if ts is not None)

    all_events = [e for _, evs, _ in results for e in evs]
    ev = pd.DataFrame(all_events)
    if len(ev) > 0:
        mask = (ev["event_date"] >= EVENT_START) & (ev["event_date"] <= EVENT_END)
        ev = ev.loc[mask].sort_values(["event_date", "ts_code"]).reset_index(drop=True)

    elapsed = time.time() - t0
    log(f"[scan_v1] stocks processed: {n_stocks} / {n_files}")
    log(f"[scan_v1] skipped schema={skip_schema} short={skip_short} nan={skip_nan} error={skip_err}")
    log(f"[scan_v1] events in window 2026-01-01..2026-08-31: {len(ev)}")
    log(f"[scan_v1] elapsed: {elapsed:.1f}s")

    # ---- 校准硬断言 ----
    checks: list[tuple[str, bool, str]] = []

    def pos(code: str, cross: str, anchor: str, anchor_close: float, tag: str) -> None:
        hit = ev[(ev["ts_code"] == code) & (ev["cross_date"] == pd.Timestamp(cross))]
        ok = len(hit) > 0 and bool(
            (hit["anchor_date"] == pd.Timestamp(anchor)).any()
            and np.isclose(hit["anchor_close"], anchor_close, atol=0.005).any()
        )
        detail = ""
        if len(hit) > 0:
            row = hit.iloc[0]
            detail = f"actual anchor_date={row['anchor_date'].date()} anchor_close={row['anchor_close']:.4f}"
        else:
            detail = "no event row"
        checks.append((tag, ok, detail))

    def neg(code: str, prev_cross: str, cross: str, tag: str) -> None:
        hit = ev[
            (ev["ts_code"] == code)
            & (ev["cross_prev_date"] == pd.Timestamp(prev_cross))
            & (ev["cross_date"] == pd.Timestamp(cross))
        ]
        checks.append((tag, len(hit) == 0, f"rows={len(hit)}"))

    pos("600283.SH", "2026-07-29", "2026-07-14", 7.57, "POS1 600283.SH 2026-07-29")
    pos("002230.SZ", "2026-07-31", "2026-07-24", 38.88, "POS2 002230.SZ 2026-07-31")
    pos("601212.SH", "2026-07-14", "2026-07-13", 4.58, "POS3 601212.SH 2026-07-14")
    neg("600283.SH", "2026-04-15", "2026-07-01", "NEG1 600283.SH 2026-04-15->2026-07-01")
    neg("002230.SZ", "2026-06-24", "2026-07-02", "NEG2 002230.SZ 2026-06-24->2026-07-02")

    all_ok = True
    for tag, ok, detail in checks:
        log(f"[calibration] {tag}: {'PASS' if ok else 'FAIL'} ({detail})")
        all_ok = all_ok and ok

    ev.to_parquet(OUT_DIR / "events_v1.parquet", index=False)

    monthly = {}
    if len(ev) > 0:
        monthly = ev["event_date"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    summary = {
        "total_events": int(len(ev)),
        "n_stocks_with_events": int(ev["ts_code"].nunique()) if len(ev) > 0 else 0,
        "monthly_events": {k: int(v) for k, v in monthly.items()},
        "stocks_processed": int(n_stocks),
        "skipped": {"schema": skip_schema, "short": skip_short, "nan": skip_nan, "error": skip_err},
        "elapsed_sec": round(elapsed, 2),
        "calibration_all_pass": all_ok,
    }
    (OUT_DIR / "events_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "events_v1.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not all_ok:
        raise SystemExit("CALIBRATION FAILED")


if __name__ == "__main__":
    main()
