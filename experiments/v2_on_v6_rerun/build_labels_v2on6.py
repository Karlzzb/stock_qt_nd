#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 3:标签表 labels_v2on6.parquet。

预登记军令状 = 同目录 README.md(commit 49b9c5b,冻结);公式唯一来源 = 同目录
anatomy_report.md(解剖报告);v2 原始代码(/tmp/v2_excavation/)只读参考,禁止 import。

README §五 标签规格逐字落实(anatomy §1.3 公式,L425-602 逐字等价):
  - 买入价 = 事件日收盘价(个股行号 j);窗口 = [j+2, j+1+p] 含端点(个股自身交易日行号)。
  - 原版收益:窗口内任一交易日 high ≥ buy×1.15 → 收益 = +0.15,卖出日 = 首触日;
    否则收益 = close[j+1+p]/buy − 1,卖出日 = 窗口末日。
  - 止损版收益(信息列):窗口内逐日,时间优先;日内优先级 open ≤ 0.65buy(按 open 卖)
    → low ≤ 0.65buy(按 0.65buy 卖)→ high ≥ 1.15buy(按 1.15buy 卖);皆未触发 → 期末收盘。
  - 窗口末端超出个股历史 → 该期标签 NaN(不删行)。
  - P2 修正落实:不作 v2 的次日 low 过滤(next_day_low > signal_price 不再置 NaN)。
  - 窗口口径裁定:README §五 止损版行文「自 j+1 起逐日」与 anatomy L67「从买入日后
    第 1 天(= j+2)到第 p 天逐日检查」(v2 L565-592 逐字)存在表述歧义;按军令状
    「公式唯一来源 = anatomy_report.md」裁定为止损版窗口同为 [j+2, j+1+p],
    与原版收益窗口一致(已记入 README 修订记录)。

向量化首触(argmax)实现,28 标签列隔离落盘,绝不进特征表。
event_close 对日线 close 逐位再对账(README §九.4,本驱动独立复核)。

用法:
  python3 build_labels_v2on6.py                   # run1
  python3 build_labels_v2on6.py --rerun-tag run2  # 第二进程重跑(双跑 md5 对账)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
DAILY_DIR = REPO / "stock_data" / "daily"
INDEX_CODES = ("000001.SH", "399001.SZ")
CACHE_DIR = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

EXPECTED_PROFIT = 1.15           # comm_fun.py:L188 逐字
EXPECTED_LOSS = 0.65             # comm_fun.py:L189 逐字(注释写 3% 止损,代码实为 −35%)
RETURN_PERIODS = [3, 5, 10, 15, 20, 25, 30]      # comm_fun.py:L200 逐字

LABEL_COLS = [f"{k}_{p}d" for p in RETURN_PERIODS
              for k in ("future_return", "future_sell_date",
                        "stop_loss_return", "stop_loss_sell_date")]
assert len(LABEL_COLS) == 28

_G: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [build_labels] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _worker(code: str) -> tuple:
    """单股:对其全部事件计算 28 标签列(窗口 [j+2, j+1+p],首触 argmax)。"""
    try:
        df = pd.read_parquet(DAILY_DIR / f"{code}.parquet",
                             columns=["trade_date", "open", "high", "low", "close"])
        df = df.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        assert not df["trade_date"].duplicated().any(), f"{code} trade_date 重复"
        n = len(df)
        dts = df["trade_date"].to_numpy()
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        pos = {pd.Timestamp(d): i for i, d in enumerate(dts)}

        evs = _G["events_by_code"].get(code, [])
        recs = []
        for ev in evs:
            j = pos.get(ev["event_date"])
            assert j is not None, f"{code} 事件日 {ev['event_date']} 不在日线"
            assert j == int(ev["event_row"]), \
                f"{code} {ev['event_date']} 行号 {j} != 事件表 event_row {ev['event_row']}"
            assert c[j] == float(ev["event_close"]), \
                f"{code} {ev['event_date']} close {c[j]!r} != event_close {ev['event_close']!r}"
            buy = c[j]
            target = buy * EXPECTED_PROFIT
            stop = buy * EXPECTED_LOSS
            rec = {"ts_code": code, "event_date": ev["event_date"]}
            for p in RETURN_PERIODS:
                end = j + 1 + p
                fr = fdr = sr = sdr = np.nan
                if end < n:                     # 窗口末端超出历史 → NaN(不删行)
                    ws = slice(j + 2, end + 1)  # 窗口 [j+2, j+1+p] 含端点
                    # ---- 原版收益:首触 high ≥ target → +0.15,否则期末收盘 ----
                    hit = h[ws] >= target
                    if hit.any():
                        k = int(np.argmax(hit))
                        fr = (target - buy) / buy
                        fdr = dts[j + 2 + k]
                    else:
                        fr = (c[end] - buy) / buy
                        fdr = dts[end]
                    # ---- 止损版:逐日时间优先,日内 open → low → high ----
                    co = o[ws] <= stop
                    cl = l[ws] <= stop
                    ch = h[ws] >= target
                    trig = co | cl | ch
                    if trig.any():
                        k = int(np.argmax(trig))
                        if co[k]:
                            sr = (o[j + 2 + k] - buy) / buy
                        elif cl[k]:
                            sr = (stop - buy) / buy
                        else:
                            sr = (target - buy) / buy
                        sdr = dts[j + 2 + k]
                    else:
                        sr = (c[end] - buy) / buy
                        sdr = dts[end]
                rec[f"future_return_{p}d"] = fr
                rec[f"future_sell_date_{p}d"] = fdr
                rec[f"stop_loss_return_{p}d"] = sr
                rec[f"stop_loss_sell_date_{p}d"] = sdr
            recs.append(rec)
        return code, recs, None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="v2on6 标签表构建")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--rerun-tag", default="run1", help="双跑标签(确定性对账用)")
    args = ap.parse_args()
    tag = args.rerun_tag
    t_all = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(EVENTS_PATH)
    assert len(ev) == 96577 and not ev.duplicated(["ts_code", "event_date"]).any()
    events_by_code: dict[str, list] = {}
    for r in ev.itertuples(index=False):
        events_by_code.setdefault(r.ts_code, []).append(
            dict(event_date=r.event_date, event_row=r.event_row,
                 event_close=r.event_close))
    _G["events_by_code"] = events_by_code
    codes = sorted(events_by_code)
    log(f"LABELS BUILD START | tag={tag} | 事件股 {len(codes)} | 事件 {len(ev)} | "
        f"workers={args.workers}")

    t0 = time.time()
    recs_by_code: dict[str, list] = {}
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (code, recs, err) in enumerate(
                ex.map(_worker, codes, chunksize=16)):
            if err:
                errors[code] = err
                log(f"  [labels-error] {code}: {err}")
            else:
                recs_by_code[code] = recs
            if (i + 1) % 1000 == 0:
                log(f"  [labels] {i + 1}/{len(codes)} ({time.time() - t0:.0f}s)")
    assert not errors, f"标签构建失败 {len(errors)} 股: {dict(list(errors.items())[:3])}"

    # ---- 汇总(事件表行序,确定性)----
    lab = pd.DataFrame([rec for code in codes for rec in recs_by_code[code]])
    assert len(lab) == len(ev), f"标签行 {len(lab)} != 事件 {len(ev)}"
    assert not lab.duplicated(["ts_code", "event_date"]).any()
    lab = ev[["ts_code", "event_date"]].merge(lab, on=["ts_code", "event_date"],
                                              how="left", validate="1:1")
    assert len(lab) == len(ev)
    for p in RETURN_PERIODS:
        for k in ("future_sell_date", "stop_loss_sell_date"):
            lab[f"{k}_{p}d"] = pd.to_datetime(lab[f"{k}_{p}d"])

    out_path = CACHE_DIR / f"labels_v2on6_{tag}.parquet"
    lab.to_parquet(out_path, index=False)

    # ---- 覆盖台账:标签 NaN 率(窗口超史)----
    cov = {}
    for p in RETURN_PERIODS:
        cov[f"future_return_{p}d_nan"] = int(lab[f"future_return_{p}d"].isna().sum())
    h = hashlib.md5()
    with open(out_path, "rb") as fh:
        for chunk_b in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk_b)
    ledger = dict(tag=tag, sec=round(time.time() - t_all, 1), n_rows=int(len(lab)),
                  n_codes=len(codes), nan_counts=cov, md5=h.hexdigest(),
                  file=str(out_path))
    with open(CACHE_DIR / f"labels_ledger_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    log(f"LABELS BUILD DONE | tag={tag} | 行 {len(lab)} | md5 {h.hexdigest()} | "
        f"NaN 计数 {cov} | {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
