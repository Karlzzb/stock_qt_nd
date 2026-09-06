# 情景补充:限价单回踩买入(2026-09-06 用户点名口径;不改冻结审判结论,纯情景对照)
#
# 与冻结口径(run_seeds.py,预登记=README.md)的唯一差异 = 入场规则:
#   冻结口径:事件日下一交易日开盘买(open×(1+SLIPPAGE);开盘涨停/无报价拒买不递补)。
#   本情景:事件日下一交易日挂限价单,限价 = 信号日收盘价;当日最低价 ≤ 信号收盘价 →
#           按信号收盘价成交(限价单成交于限价,买端滑点 = 0);当日最低价 > 信号收盘价 →
#           买不进(status = no_fill,计数不递补)。
# 其余逐条不变:10 万整手现金约束、入场日记持有第 1 日、第 H 个交易日收盘卖、
# 跌停顺延、卖端滑点、买佣/卖佣/印花税(调 strategy_engine 原语)。
#
# 复用冻结产物:trades_seed.parquet 中 status=='closed' 的行复用其 exit_exec/exit_date
# (卖出路径只依赖入场日 e 与个股序列,与入场价无关),按本情景的新股数重算卖出成本;
# 原口径 dropped_limitup/dropped_cash 的行若在本情景成交,用日线 close + stk_limit
# 文件现算卖出路径(跌停顺延同一逻辑)。
#
# 自检:对原 closed 行用冻结口径(px=open×(1+SLIPPAGE))重算 net_ret,与 parquet 原值
# 逐笔对比,max abs diff 必须 < 1e-9,验证成本重算与冻结引擎一致。
#
# 运行:python3 addendum_close_fill.py(预计 1~3 分钟;心跳写 addendum_close_fill_progress.log)

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "v3_pipeline", "scripts"))  # noqa: E402
import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语与常量

DATA_DIR = Path(os.path.join(REPO, "stock_data", "daily"))
LIMIT_DIR = Path(os.path.join(REPO, "stock_data", "stk_limit"))
OUT_DIR = Path(os.path.join(REPO, "experiments", "divergence_seed_trial_history"))
LOG = OUT_DIR / "addendum_close_fill_progress.log"
MD_PATH = OUT_DIR / "addendum_close_fill.md"
PARQUET_OUT = OUT_DIR / "trades_close_fill.parquet"

BUDGET = 100_000.0
TOL = 1e-9
H_LIST = [10, 20, 25]
SEEDS = ["S1", "S2", "S3", "S4", "S5", "ALL"]

_G: dict = {}


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def _worker_init():
    _G["lim_cache"] = {}


def _lookup_dn(date_int: int, ts_code: str) -> float:
    """跌停价按需回退读当日 stk_limit 文件(进程内缓存);无则 NaN = 无约束。"""
    cache = _G["lim_cache"]
    if date_int not in cache:
        fp = LIMIT_DIR / f"{date_int}.parquet"
        if fp.exists():
            lf = pd.read_parquet(fp)
            cache[date_int] = dict(zip(lf["ts_code"], lf["down_limit"]))
        else:
            cache[date_int] = None
    day = cache[date_int]
    if day is None:
        return np.nan
    v = day.get(ts_code)
    return float(v) if v is not None else np.nan


def _buy_shares(px: float):
    """整手现金约束(与冻结口径逐条一致,仅 px 来源不同)。"""
    sh = int(BUDGET / px / se.BOARD_LOT) * se.BOARD_LOT
    if sh < se.BOARD_LOT:
        sh = se.BOARD_LOT
    comm = se.buy_cost(sh, px)
    while sh > 0 and sh * px + comm > BUDGET + 1e-6:
        sh -= se.BOARD_LOT
        comm = se.buy_cost(sh, px) if sh > 0 else 0.0
    return sh, comm


def process_stock(task: dict):
    """单股:加载日线(trade_date/open/low/close),对该股全部 v1 事件×H 算本情景。"""
    ts_code = task["ts_code"]
    df = pd.read_parquet(DATA_DIR / f"{ts_code}.parquet",
                         columns=["trade_date", "open", "low", "close"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)
    open_ = df["open"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    dts = pd.DatetimeIndex(df["trade_date"])
    d_int = (dts.year * 10000 + dts.month * 100 + dts.day).to_numpy(dtype=np.int32)
    pos = {d: i for i, d in enumerate(dts)}

    recs: list[dict] = []
    check_rows: list[tuple] = []  # (parquet_net_ret, recomputed_frozen_net_ret)
    for row in task["rows"]:
        j = pos.get(row["event_date"], -1)
        if j < 0:
            continue  # 防御:事件由同一文件生成
        H = row["H"]
        base = dict(ts_code=ts_code, event_date=row["event_date"], H=H,
                    sel_S1=row["sel_S1"], sel_S2=row["sel_S2"], sel_S3=row["sel_S3"],
                    sel_S4=row["sel_S4"], sel_S5=row["sel_S5"], sel_ALL=row["sel_ALL"],
                    status_frozen=row["status"])
        if j + 1 >= n:
            recs.append({**base, "status": "truncated_no_next"})
            continue
        e = j + 1
        sig_close = close[j]
        low_e = low[e]

        # ---- 自检:冻结口径重算(仅原 closed 行需要) ----
        if row["status"] == "closed":
            px0 = open_[e] * (1.0 + se.SLIPPAGE)
            sh0, comm0 = _buy_shares(px0)
            xs0 = row["exit_exec"]
            xcomm0, stamp0 = se.sell_costs(sh0, xs0, row["exit_date"])
            net0 = (sh0 * (xs0 - px0) - comm0 - xcomm0 - stamp0) / (sh0 * px0 + comm0)
            check_rows.append((row["net_ret"], net0))

        # ---- 本情景:限价回踩 ----
        if not np.isfinite(low_e) or low_e > sig_close + TOL:
            recs.append({**base, "status": "no_fill"})
            continue
        px = sig_close  # 限价单成交于限价,买端滑点 = 0
        sh, comm = _buy_shares(px)
        if sh <= 0:
            recs.append({**base, "status": "dropped_cash"})
            continue
        entry = dict(entry_date=dts[e], entry_exec=float(px), shares=int(sh),
                     buy_comm=float(comm))

        if row["status"] == "closed":
            # 卖出路径与冻结口径相同,复用 exit_exec/exit_date,按新 sh 重算卖出成本
            xs = row["exit_exec"]
            xcomm, stamp = se.sell_costs(sh, xs, row["exit_date"])
            net_pnl = sh * (xs - px) - comm - xcomm - stamp
            recs.append({**base, "status": "closed", "exit_date": row["exit_date"],
                         "exit_exec": float(xs), "sell_comm": float(xcomm),
                         "stamp": float(stamp), "net_ret": float(net_pnl / (sh * px + comm)),
                         "net_pnl": float(net_pnl), **entry})
            continue
        if row["status"].startswith("truncated"):
            recs.append({**base, "status": row["status"], **entry})
            continue
        # 原口径 dropped_limitup / dropped_cash / dropped_no_quote:现算卖出路径
        deferred = 0
        for r in range(e + H - 1, n):
            c = close[r]
            if not np.isfinite(c):
                continue
            dn = _lookup_dn(int(d_int[r]), ts_code)
            if np.isfinite(dn) and c <= dn + TOL:
                deferred += 1
                continue
            xs = c * (1.0 - se.SLIPPAGE)
            xcomm, stamp = se.sell_costs(sh, xs, dts[r])
            net_pnl = sh * (xs - px) - comm - xcomm - stamp
            recs.append({**base, "status": "closed", "exit_date": dts[r],
                         "exit_exec": float(xs), "sell_comm": float(xcomm),
                         "stamp": float(stamp), "net_ret": float(net_pnl / (sh * px + comm)),
                         "net_pnl": float(net_pnl), "deferred_days": int(deferred), **entry})
            break
        else:
            recs.append({**base, "status": "truncated_exhausted",
                         "deferred_days": int(deferred), **entry})
    return recs, check_rows


def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    """Liang-Zeger 聚类稳健 t(与 run_seeds.py 同一实现)。"""
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    df = pd.DataFrame({"s": s, "c": clusters})
    sums = df.groupby("c")["s"].sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


COVERS = [("90%", 0.10), ("85%", 0.15), ("75%", 0.25), ("50%", 0.50),
          ("25%", 0.75), ("10%", 0.90), ("5%", 0.95)]


def main() -> None:
    t_all = time.time()
    LOG.unlink(missing_ok=True)
    log("CLOSE-FILL 情景 START | 入场=信号日收盘价限价单,其余同冻结口径 | 预计 1~3 分钟")

    trades = pd.read_parquet(OUT_DIR / "trades_seed.parquet")
    v1 = trades[trades.variant == "v1"].reset_index(drop=True)
    log(f"[init] v1 行 {len(v1)}(应为 96577 事件 × 3 H = 289731)")

    need_cols = ["ts_code", "event_date", "H", "status", "exit_date", "exit_exec",
                 "net_ret", "sel_S1", "sel_S2", "sel_S3", "sel_S4", "sel_S5", "sel_ALL"]
    tasks = [{"ts_code": c, "rows": grp[need_cols].to_dict("records")}
             for c, grp in v1.groupby("ts_code")]
    log(f"[init] 个股 {len(tasks)} 只;CPU {mp.cpu_count()}")

    recs: list[dict] = []
    checks: list[tuple] = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 1),
                 initializer=_worker_init) as pool:
        for i, (r, ck) in enumerate(pool.imap_unordered(process_stock, tasks, chunksize=8)):
            recs.extend(r)
            checks.extend(ck)
            if (i + 1) % 500 == 0:
                log(f"heartbeat: {i + 1}/{len(tasks)} 股 ({time.time() - t_all:.0f}s)")

    out = pd.DataFrame(recs)
    out.to_parquet(PARQUET_OUT, index=False)
    log(f"[dump] {PARQUET_OUT.name} {len(out)} 行")

    # ---------------- 自检 ----------------
    ck = np.array(checks)
    max_diff = float(np.abs(ck[:, 0] - ck[:, 1]).max())
    log(f"[自检] 冻结口径重算 vs parquet 原值 max abs diff = {max_diff:.3e} "
        f"({'PASS' if max_diff < 1e-9 else 'FAIL'})")
    assert max_diff < 1e-9, "成本重算与冻结引擎不一致"

    # ---------------- 汇总 ----------------
    frozen = v1  # 冻结口径对照
    MD: list[str] = []
    md = MD.append
    md("# 情景补充:限价单回踩买入 vs 冻结口径(次日开盘买)")
    md("")
    md(f"生成时间:{datetime.datetime.now().isoformat(timespec='seconds')}。")
    md("本附录为 2026-09-06 用户点名的情景对照,不改冻结审判结论;数据源为已冻结产物 + 个股日线。")
    md("")
    md("## 口径")
    md("")
    md("- 事件级框 = `trades_seed.parquet` v1 全部 96577 事件 × H∈{10,20,25};种子口径同冻结(sel_S1~S5,ALL=全池)。")
    md("- **本情景入场**:事件日下一交易日挂限价单,限价 = 信号日收盘价;当日最低价 ≤ 信号收盘价 → 按信号收盘价成交(限价单,买端滑点 = 0);否则 no_fill,计数不递补。")
    md("- **冻结口径入场**(对照):同日开盘价 ×(1+0.1% 滑点);开盘涨停/无报价拒买。")
    md("- 其余逐条一致:10 万整手现金、入场日计持有第 1 日、第 H 日收盘卖、跌停顺延、卖端滑点 0.1%、买佣/卖佣/印花税。")
    md("- 卖出路径复用:原 closed 行的 exit_exec/exit_date 直接复用(卖出路径与入场价无关),按本情景新股数重算卖出成本;原 dropped 行若本情景成交,用日线 close + stk_limit 现算。")
    md(f"- 自检:冻结口径重算 vs parquet 原值 max abs diff = {max_diff:.3e}(PASS)。")
    md("")

    # ---- 成交率 ----
    md("## 成交率(限价回踩是否买得进)")
    md("")
    md("| 种子 | H | 信号数 | 成交笔数 | 买不进(次日最低>信号收盘) | 成交率 | 现金不足 |")
    md("|---|---|---|---|---|---|---|")
    for H in H_LIST:
        for s_ in SEEDS:
            sub = out[(out.H == H) & out[f"sel_{s_}"]]
            n_sel = len(sub)
            n_nofill = int((sub.status == "no_fill").sum())
            n_cash = int((sub.status == "dropped_cash").sum())
            n_fill = n_sel - n_nofill - n_cash
            md(f"| {s_} | {H} | {n_sel} | {n_fill} | {n_nofill} | {n_fill / n_sel:.1%} | {n_cash} |")
    md("")

    # ---- 收益对照 ----
    for H in H_LIST:
        md(f"## 收益对照 H={H}(closed 口径,净收益扣完成本)")
        md("")
        md('覆盖率口径:"X%信号涨幅>" = 涨幅超过该值的信号恰占 X%;均值/最好/最差 = 单笔净收益;'
           '>+1%占比 = 净收益 > +1% 的笔数占 closed 笔数比例。')
        md("")
        md("| 种子 | 口径 | n_closed | 均值 | 90%信号涨幅> | 75%> | 50%> | 25%> | 10%> | 5%> | >+1%占比 | 胜率 | cluster_t | 最好 | 最差 |")
        md("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s_ in SEEDS:
            for label, src in (("本情景", out), ("冻结", frozen)):
                sub = src[(src.H == H) & src[f"sel_{s_}"] & (src.status == "closed")]
                if len(sub) == 0:
                    continue
                clusters = sub["entry_date"].astype(str).to_numpy()
                r = sub["net_ret"].to_numpy() * 100
                ct = cluster_t(sub["net_ret"].to_numpy(), clusters)
                qs = " | ".join(f"{np.quantile(r, q):+.1f}%" for _, q in
                                [("90", 0.10), ("75", 0.25), ("50", 0.50), ("25", 0.75), ("10", 0.90), ("5", 0.95)])
                md(f"| {s_} | {label} | {len(r)} | {r.mean():+.2f}% | {qs} | "
                   f"{(r > 1).mean():.1%} | {(r > 0).mean():.1%} | {ct:.2f} | {r.max():+.1f}% | {r.min():+.1f}% |")
        md("")

    # ---- 两口径差值 ----
    md("## 均值差(本情景 − 冻结,pp)")
    md("")
    md("| 种子 | H=10 | H=20 | H=25 |")
    md("|---|---|---|---|")
    for s_ in SEEDS:
        cells = []
        for H in H_LIST:
            a = out[(out.H == H) & out[f"sel_{s_}"] & (out.status == "closed")]["net_ret"].mean()
            b = frozen[(frozen.H == H) & frozen[f"sel_{s_}"] & (frozen.status == "closed")]["net_ret"].mean()
            cells.append(f"{(a - b) * 100:+.2f}")
        md(f"| {s_} | " + " | ".join(cells) + " |")
    md("")
    md("注:两口径 n_closed 不同(本情景买不进的笔被剔除、原涨停拒买的笔可能成交),均值差含样本构成变化,非纯价格效应。")

    MD_PATH.write_text("\n".join(MD) + "\n", encoding="utf-8")
    log(f"[dump] {MD_PATH.name} 写出完成")
    log(f"ALL DONE ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
