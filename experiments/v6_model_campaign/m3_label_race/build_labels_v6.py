#!/usr/bin/env python3
"""M3 标签表构建驱动(战役 #47 M3,issue #50;预登记 = 同目录 README.md,冻结)。

机制 = #31 §2.3 事件级直接模拟逐字口径(不沿用 v5 全序列毛利路径),覆盖全部
96,577 事件 × 4 档 H ∈ {10,20,25,60}:
  信号日 T 的下一交易日开盘买(px = open×1.001,0.1% 买入滑点);
  开盘无报价 → dropped_no_quote;开盘价 ≥ 当日涨停价−1e-9 → dropped_limitup(不递补);
  每笔独立 10 万本金整手折算:sh = int(100000/px/100)×100,不足一手取一手,
  while sh>0 且 sh×px+买佣 > 100000+1e-6 减一手,减到 0 → dropped_cash;
  入场日记持有第 1 日;第 H 个交易日(个股序列行号)收盘卖(xs = close×0.999);
  收盘 ≤ 当日跌停价+1e-9 → 顺延至下一非跌停收盘日;数据耗尽(含退市)→ truncated_exhausted;
  事件日无下一行 → truncated_no_next;
  净收益 = (sh×(xs−px) − 买佣 − 卖佣 − 印花税) / (sh×px + 买佣)。
  成本常量复用冻结引擎 strategy_engine(只读):滑点单边 0.1%、佣金双边万2.5 最低5元、
  印花税 2023-08-27 含之前 0.1% / 2023-08-28 起 0.05%、整手 100 股、价格比较容差 1e-9。
  涨跌停 stock_data/stk_limit/YYYYMMDD.parquet(2007-01-04 起),缺文件日视为无约束。

键对齐:事件表 (ts_code, event_date) ↔ 主表元数据列 (event_id, ts_code, date,
event_row, seg) 一对一合并;seg 落盘前以 feature_master.segment_of 独立重算逐行一致断言。
标签构建器不读主表任何特征列(只读上述元数据列与事件表)。

确定性:全量构建两次,两产物 md5 逐位一致才生效(哈希落台账)。

输出(experiments/v6_model_campaign/m3_label_race/):
  labels_v6.parquet          标签表(18 列,schema 见 README §四)
  rebuild_labels_v6.parquet  确定性第二跑副本(自检复核 md5 用)
  label_build_results.json   台账(剔除类别计数/披露/哈希)
  progress.log               心跳

用法: python build_labels_v6.py [--workers 27]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语与常量
import feature_master as fm  # noqa: E402  切分常量唯一权威落点(只读)

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
MASTER_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
DATA_DIR = REPO / "stock_data" / "daily"
LIMIT_DIR = REPO / "stock_data" / "stk_limit"
OUT_DIR = SCRIPT_DIR
LOG_PATH = OUT_DIR / "progress.log"

H_LIST = [10, 20, 25, 60]
BUDGET = 100_000.0
TOL = 1e-9  # = se.PRICE_TOL(价格比较容差,冻结口径)
LIMIT_LOOKAHEAD_DAYS = 40  # #31 同机制:事件日到事件后 40 自然日的 stk_limit 文件
LIMIT_EARLIEST = pd.Timestamp("2007-01-04")  # stk_limit 文件起点(披露口径)
STATUS_ALL = ["closed", "dropped_no_quote", "dropped_limitup", "dropped_cash",
              "truncated_no_next", "truncated_exhausted", "key_missing"]

_G: dict = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [build_labels_v6] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 涨跌停加载(#31 同机制)
def _limit_worker_init(codes):
    _G["limit_codes"] = codes


def _limit_read_one(d: str):
    fp = LIMIT_DIR / f"{d}.parquet"
    if not fp.exists():
        return d, None
    lf = pd.read_parquet(fp)
    lf = lf[lf["ts_code"].isin(_G["limit_codes"])]
    return d, (lf["ts_code"].to_numpy(),
               lf["up_limit"].to_numpy(dtype=np.float64),
               lf["down_limit"].to_numpy(dtype=np.float64))


def load_limits(needed_dates: list[str], needed_codes: set[str], workers: int) -> tuple[dict, int]:
    """加载指定日期的 stk_limit 文件,过滤到有事件的个股。
    返回 ({ts_code: (dates_int32 有序, up, dn)}, 缺文件天数)。"""
    t0 = time.time()
    acc_codes, acc_d, acc_up, acc_dn = [], [], [], []
    missing = 0
    with mp.Pool(processes=workers, initializer=_limit_worker_init,
                 initargs=(frozenset(needed_codes),)) as pool:
        for i, (d, res) in enumerate(pool.imap_unordered(_limit_read_one,
                                                         needed_dates,
                                                         chunksize=16)):
            if res is None:
                missing += 1
            elif len(res[0]):
                codes, ups, dns = res
                acc_codes.append(codes)
                acc_d.append(np.full(len(codes), int(d), dtype=np.int32))
                acc_up.append(ups)
                acc_dn.append(dns)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: stk_limit 加载 {i + 1}/{len(needed_dates)} 文件 "
                    f"({time.time() - t0:.0f}s)")
    big = pd.DataFrame({
        "code": pd.Categorical(np.concatenate(acc_codes)),
        "d": np.concatenate(acc_d),
        "up": np.concatenate(acc_up),
        "dn": np.concatenate(acc_dn),
    }).sort_values(["code", "d"])
    out: dict[str, tuple] = {}
    for code, grp in big.groupby("code", observed=True):
        out[str(code)] = (grp["d"].to_numpy(), grp["up"].to_numpy(),
                          grp["dn"].to_numpy())
    log(f"[limits] 加载 {len(needed_dates) - missing}/{len(needed_dates)} 文件,"
        f"缺文件 {missing} 天,覆盖 {len(out)} 股 ({time.time() - t0:.0f}s)")
    return out, missing


# ---------------------------------------------------------------- 模拟(逐股 worker,#31 §2.3 逐字口径)
def _sim_worker_init():
    _G["fallback_cache"] = {}
    _G["fallback_loads"] = 0


def _lookup_limit(lim, date_int: int, which: int, ts_code: str):
    """lim = (dates_int32, up, dn) 或 None;which: 1=up 2=dn。
    预载数组命中则直接返回;未命中(文件缺失/无该股行/顺延超出 +40 自然日窗口)
    按需回退读当日文件(进程内缓存),仍无则 NaN = 无约束。"""
    if lim is not None:
        dates, up, dn = lim
        i = int(np.searchsorted(dates, date_int))
        if i < len(dates) and dates[i] == date_int:
            return float(up[i] if which == 1 else dn[i])
    cache = _G["fallback_cache"]
    if date_int not in cache:
        fp = LIMIT_DIR / f"{date_int}.parquet"
        if fp.exists():
            lf = pd.read_parquet(fp)
            cache[date_int] = dict(zip(lf["ts_code"],
                                       zip(lf["up_limit"], lf["down_limit"])))
        else:
            cache[date_int] = None
        _G["fallback_loads"] += 1
    day = cache[date_int]
    if day is None:
        return np.nan
    row = day.get(ts_code)
    if row is None:
        return np.nan
    return float(row[0] if which == 1 else row[1])


def _simulate_one(open_, close, dts, d_int, lim, ts_code, j, H) -> dict:
    """#31 §2.3 逐字口径:次日开盘买(涨停/无报价拒买不递补,整手现金),
    入场日记第 1 日,第 H 个交易日(个股序列行号)收盘卖,跌停顺延,耗尽 truncated。"""
    n = len(close)
    if j + 1 >= n:
        return dict(status="truncated_no_next")  # 事件后无下一行(含退市)
    e = j + 1
    o = open_[e]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = _lookup_limit(lim, int(d_int[e]), 1, ts_code)
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    px = o * (1.0 + se.SLIPPAGE)
    sh = int(BUDGET / px / se.BOARD_LOT) * se.BOARD_LOT
    if sh < se.BOARD_LOT:
        sh = se.BOARD_LOT
    comm = se.buy_cost(sh, px)
    while sh > 0 and sh * px + comm > BUDGET + 1e-6:
        sh -= se.BOARD_LOT
        comm = se.buy_cost(sh, px) if sh > 0 else 0.0
    if sh <= 0:
        return dict(status="dropped_cash")
    base = dict(entry_date=dts[e], entry_raw=float(o), entry_exec=float(px),
                shares=int(sh), buy_comm=float(comm))
    deferred = 0
    for r in range(e + H - 1, n):
        c = close[r]
        if not np.isfinite(c):
            continue  # 行计入但不评估(个股序列实际无停牌行,防御保留)
        dn = _lookup_limit(lim, int(d_int[r]), 2, ts_code)
        if np.isfinite(dn) and c <= dn + TOL:
            deferred += 1
            continue
        xs = c * (1.0 - se.SLIPPAGE)
        xcomm, stamp = se.sell_costs(sh, xs, dts[r])
        net_pnl = sh * (xs - px) - comm - xcomm - stamp
        net_ret = net_pnl / (sh * px + comm)
        return dict(status="closed", exit_date=dts[r], exit_raw=float(c),
                    exit_exec=float(xs), sell_comm=float(xcomm), stamp=float(stamp),
                    net_ret=float(net_ret), net_pnl=float(net_pnl),
                    held_rows=int(r - e + 1), deferred_days=int(deferred), **base)
    return dict(status="truncated_exhausted", deferred_days=int(deferred), **base)


def simulate_stock(task: dict) -> tuple[list[dict], int]:
    """单股:加载日线,对该股全部事件逐 H ∈ {10,20,25,60} 模拟。"""
    ts_code = task["ts_code"]
    lim = task["limits"]
    path = DATA_DIR / f"{ts_code}.parquet"
    recs: list[dict] = []
    if not path.exists():
        for ev in task["events"]:
            for H in H_LIST:
                recs.append(dict(ts_code=ts_code, event_date=ev["event_date"], H=H,
                                 status="key_missing"))
        return recs, 0
    df = pd.read_parquet(path, columns=["trade_date", "open", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    df = df.sort_values("trade_date").reset_index(drop=True)
    open_ = df["open"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    dts = pd.DatetimeIndex(df["trade_date"])
    d_int = (dts.year * 10000 + dts.month * 100 + dts.day).to_numpy(dtype=np.int32)
    pos = {d: i for i, d in enumerate(dts)}
    for ev in task["events"]:
        j = pos.get(ev["event_date"], -1)
        if j < 0:
            for H in H_LIST:
                recs.append(dict(ts_code=ts_code, event_date=ev["event_date"], H=H,
                                 status="key_missing"))
            continue
        assert j == ev["event_row"], \
            f"{ts_code} {ev['event_date']} 行号不对齐: 序列 {j} vs 事件表 {ev['event_row']}"
        for H in H_LIST:
            recs.append(dict(ts_code=ts_code, event_date=ev["event_date"], H=H,
                             **_simulate_one(open_, close, dts, d_int, lim,
                                             ts_code, j, H)))
    fb = _G["fallback_loads"]
    _G["fallback_loads"] = 0
    return recs, fb


# ---------------------------------------------------------------- 主流程
def build_labels(workers: int) -> tuple[pd.DataFrame, dict]:
    """全量构建一次。返回 (标签表, 台账)。"""
    t_all = time.time()

    # ---------------- 阶段 1:事件表与主表元数据键对齐 ----------------
    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date", "event_row"])
    assert len(ev) == 96577, f"事件表行数 {len(ev)} != 96577"
    assert not ev.duplicated(["ts_code", "event_date"]).any(), "事件表键不唯一"
    meta = pd.read_parquet(MASTER_PATH,
                           columns=["event_id", "ts_code", "date", "event_row", "seg"])
    assert not meta.duplicated(["ts_code", "date"]).any(), "主表键不唯一"
    assert not meta["event_id"].duplicated().any(), "主表 event_id 不唯一"
    base = ev.merge(meta, left_on=["ts_code", "event_date"], right_on=["ts_code", "date"],
                    how="left", validate="one_to_one")
    assert base["event_id"].notna().all(), "事件表存在主表缺失键"
    assert (base["event_row_x"] == base["event_row_y"]).all(), "event_row 双侧不一致"
    base = base.rename(columns={"event_row_x": "event_row"})[
        ["event_id", "ts_code", "event_date", "event_row", "seg"]]
    seg_recompute = np.asarray(fm.segment_of(base["event_date"]))
    assert (seg_recompute == base["seg"].to_numpy()).all(), "seg 与 segment_of 重算不一致"
    seg_counts = {k: int(v) for k, v in base["seg"].value_counts().items()}
    log(f"[阶段1] 键对齐通过: {len(base)} 行,段计数 {seg_counts}")

    # ---------------- 阶段 2:涨跌停按需加载(#31 同机制,事件日~+40 自然日) ----------------
    all_ev_dates = base["event_date"].sort_values()
    limit_files = sorted(p.stem for p in LIMIT_DIR.glob("*.parquet"))
    lf_dates = pd.to_datetime(pd.Series(limit_files), format="%Y%m%d")
    lo = lf_dates - pd.Timedelta(days=LIMIT_LOOKAHEAD_DAYS)
    ev_arr = all_ev_dates.to_numpy()
    idx_hi = np.searchsorted(ev_arr, lf_dates.to_numpy(), side="right")
    idx_lo = np.searchsorted(ev_arr, lo.to_numpy(), side="left")
    needed_mask = idx_hi > idx_lo
    needed_dates = [d for d, m in zip(limit_files, needed_mask) if m]
    needed_codes = set(base["ts_code"].unique())
    log(f"[limits] 需求日期 {len(needed_dates)}/{len(limit_files)} 文件;"
        f"有事件个股 {len(needed_codes)} 只")
    limits_by_code, n_limit_missing = load_limits(needed_dates, needed_codes, workers)

    # ---------------- 阶段 3:逐股模拟(每事件 × 4 档 H) ----------------
    ev_by_code: dict[str, list] = {}
    for r in base.itertuples(index=False):
        ev_by_code.setdefault(r.ts_code, []).append(
            dict(event_date=r.event_date, event_row=int(r.event_row)))
    tasks = [dict(ts_code=code, events=evs, limits=limits_by_code.get(code))
             for code, evs in sorted(ev_by_code.items())]
    log(f"[simulate] 待模拟个股 {len(tasks)} 只,事件 {len(base)} 起 × H{H_LIST}")

    t0 = time.time()
    trade_recs: list[dict] = []
    fallback_total = 0
    n_done = 0
    with mp.Pool(processes=workers, initializer=_sim_worker_init) as pool:
        for recs, fb in pool.imap_unordered(simulate_stock, tasks, chunksize=4):
            trade_recs.extend(recs)
            fallback_total += fb
            n_done += 1
            if n_done % 500 == 0:
                log(f"heartbeat: 模拟 {n_done}/{len(tasks)} 股,"
                    f"累计记录 {len(trade_recs)} ({time.time() - t0:.0f}s)")
    trades = pd.DataFrame(trade_recs)
    log(f"[simulate] 完成: {len(trades)} 行(事件×H) ({time.time() - t0:.0f}s)")
    assert len(trades) == len(base) * len(H_LIST), "模拟行数不守恒"
    assert not trades.duplicated(["ts_code", "event_date", "H"]).any(), "模拟键不唯一"

    # ---------------- 阶段 4:宽表化(schema 见 README §四) ----------------
    lab = base.copy()
    lab["entry_date"] = pd.NaT
    for H in H_LIST:
        sub = trades[trades["H"] == H][
            ["ts_code", "event_date", "status", "net_ret", "exit_date", "entry_date"]]
        sub = sub.rename(columns={"status": f"status_{H}d",
                                  "net_ret": f"net_ret_{H}d",
                                  "exit_date": f"exit_date_{H}d",
                                  "entry_date": f"_entry_{H}"})
        lab = lab.merge(sub, on=["ts_code", "event_date"], how="left",
                        validate="one_to_one")
        assert lab[f"status_{H}d"].notna().all(), f"H={H} 存在未模拟事件"
    # entry_date 的 H 不变性硬断言(入场不依赖 H),并取成单列
    for H in H_LIST[1:]:
        a = lab["_entry_10"].to_numpy()
        b = lab[f"_entry_{H}"].to_numpy()
        same = (pd.isna(a) & pd.isna(b)) | (a == b)
        assert same.all(), f"entry_date 在 H=10 与 H={H} 间不一致(入口侧口径破坏)"
    lab["entry_date"] = lab["_entry_10"]
    lab = lab.drop(columns=[f"_entry_{H}" for H in H_LIST])
    # NaN 保留原则断言:net_ret 非 NaN ⟺ status == closed
    for H in H_LIST:
        closed = lab[f"status_{H}d"] == "closed"
        assert (lab[f"net_ret_{H}d"].notna() == closed).all(), \
            f"H={H} net_ret 与 status=closed 不一一对应"
        assert (lab[f"exit_date_{H}d"].notna() == closed).all()
    lab = lab.rename(columns={"event_date": "date"})
    lab = lab[["event_id", "ts_code", "date", "event_row", "seg", "entry_date"]
              + [c for H in H_LIST
                 for c in (f"net_ret_{H}d", f"status_{H}d", f"exit_date_{H}d")]]
    assert len(lab) == 96577 and lab.shape[1] == 18, f"标签表形状异常 {lab.shape}"
    assert not lab.duplicated(["ts_code", "date"]).any()
    assert not lab["event_id"].duplicated().any()
    log(f"[阶段4] 标签表成形: {lab.shape[0]} 行 × {lab.shape[1]} 列")

    # ---------------- 阶段 5:剔除类别计数与披露 ----------------
    ledger: dict = {"issue": 50, "seg_counts": seg_counts,
                    "n_events": int(len(lab)), "H_list": H_LIST,
                    "n_limit_files_needed": len(needed_dates),
                    "n_limit_missing_days": int(n_limit_missing),
                    "n_limit_fallback_loads": int(fallback_total),
                    "per_H": {}}
    for H in H_LIST:
        per: dict = {"status_counts_total": {}, "status_counts_by_seg": {},
                     "closed_entry_before_limit_era": {}}
        vc = lab[f"status_{H}d"].value_counts().to_dict()
        per["status_counts_total"] = {k: int(vc.get(k, 0)) for k in STATUS_ALL}
        for seg_name in ("train", "val"):
            sub = lab[lab["seg"] == seg_name]
            vcs = sub[f"status_{H}d"].value_counts().to_dict()
            per["status_counts_by_seg"][seg_name] = {
                k: {"n": int(vcs.get(k, 0)),
                    "share": float(vcs.get(k, 0) / len(sub))}
                for k in STATUS_ALL}
        n_test = int((lab["seg"] == "test").sum())
        assert n_test > 0, "test 段应在场(零触碰只在场断言)"
        closed_mask = lab[f"status_{H}d"] == "closed"
        n_closed = int(closed_mask.sum())
        n_pre = int((lab.loc[closed_mask, "entry_date"] < LIMIT_EARLIEST).sum())
        per["closed_entry_before_limit_era"] = {
            "n": n_pre, "share_of_closed": float(n_pre / n_closed) if n_closed else None,
            "n_closed": n_closed}
        ledger["per_H"][f"H{H}"] = per
    ledger["elapsed_sec"] = round(time.time() - t_all, 1)
    return lab, ledger


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()
    t_all = time.time()
    log(f"开工: 96,577 事件 × H{H_LIST} 事件级直接模拟;workers={args.workers}")

    lab1, ledger = build_labels(args.workers)
    out1 = OUT_DIR / "labels_v6.parquet"
    lab1.to_parquet(out1, index=False)
    md1 = _md5(out1)
    log(f"[确定性] 第一跑落盘 {out1.name} md5={md1}")

    lab2, _ = build_labels(args.workers)
    out2 = OUT_DIR / "rebuild_labels_v6.parquet"
    lab2.to_parquet(out2, index=False)
    md2 = _md5(out2)
    log(f"[确定性] 第二跑落盘 {out2.name} md5={md2}")
    assert md1 == md2, f"确定性双跑 md5 不一致: {md1} vs {md2}"
    assert lab1.equals(lab2), "确定性双跑 DataFrame 不一致"
    log("[确定性] 双跑 md5 逐位一致 -> PASS")

    ledger["determinism"] = {"md5_run1": md1, "md5_run2": md2, "identical": True}
    ledger["total_elapsed_sec"] = round(time.time() - t_all, 1)
    with (OUT_DIR / "label_build_results.json").open("w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2, default=str)
    log(f"[完成] 总耗时 {ledger['total_elapsed_sec']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
