#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 阶段门审判 · 独立证伪式自检(README §四.6)。

纪律:不 import M5 驱动(run_stage_gate_m5.py)任何函数;全部指标/宣判/对账由本文件
自写代码从落盘产物(runs/*.parquet、summary_stage_gate_m5.csv、detcmp_m5.log、
#40 冻结 runs、scores_v6.parquet、labels_v6.parquet、上证指数日线)独立重算。
成本原语只读复用冻结引擎 strategy_engine(与 M3/M4 自检同例)。
另以全新子进程 emit 模式重跑 2 格(M×K3、S4×K3)与落盘 trades 逐位对拍。

用法: python3 selfcheck_m5.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))
import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语

M5 = REPO / "experiments" / "v6_model_campaign" / "m5_stage_gate"
RUNS = M5 / "runs"
LH_RUNS = REPO / "experiments" / "v6_long_hold_trial" / "runs"
SCORES_PATH = REPO / "experiments" / "v6_model_campaign" / "m4_training_selection" / "scores_v6.parquet"
LABELS_PATH = REPO / "experiments" / "v6_model_campaign" / "m3_label_race" / "labels_v6.parquet"
INDEX_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"

SELS6 = ["M", "S1", "S2", "S3", "S4", "S5"]
K_LIST = [3, 5, 10]
CELLS = [f"ALL__H60__{s}__P7K{K}__E1_A6" for s in SELS6 for K in K_LIST]
D_START, D_END = 20200101, 20240521
INIT = 1_000_000.0

RESULTS: dict = {"checks": {}, "all_pass": False}


def record(name: str, ok: bool, **kw) -> None:
    RESULTS["checks"][name] = dict(ok=bool(ok), **kw)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {kw if not ok else ''}", flush=True)


def d_int_series(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s.dt.year * 10000 + s.dt.month * 100
                + s.dt.day).to_numpy(dtype=np.int64)
    if pd.api.types.is_integer_dtype(s):
        return s.to_numpy(dtype=np.int64)
    # 字符串 "YYYY-MM-DD" / "YYYYMMDD"
    return s.astype(str).str.replace("-", "", regex=False).to_numpy(dtype=np.int64)


def cluster_t_own(x: np.ndarray, clusters: np.ndarray) -> float:
    """按入场日聚类的 Liang-Zeger 稳健 t(#31 §2.4 口径)自写实现。

    复刻底座语义:簇按键升序(pandas groupby 排序),簇内按原始行序求和。
    """
    n = len(x)
    if n < 2:
        return np.nan
    xbar = x.mean()
    s = x - xbar
    sums = []
    for c in sorted(set(clusters.tolist())):
        sums.append(float(s[clusters == c].sum()))
    sums = np.asarray(sums)
    g = len(sums)
    if g < 2:
        return np.nan
    var = (g / (g - 1.0)) * float((sums ** 2).sum()) / (n * n)
    if var <= 0:
        return np.nan
    return float(xbar / np.sqrt(var))


# ---------------------------------------------------------------- 1. 指标独立重算 vs summary
def check_metrics() -> dict:
    df = pd.read_csv(M5 / "summary_stage_gate_m5.csv", dtype=str,
                     keep_default_na=False)
    assert list(df["cell_id"]) == CELLS or set(df["cell_id"]) == set(CELLS)
    idx = pd.read_parquet(INDEX_PATH, columns=["trade_date", "close"])
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    dts = d_int_series(idx["trade_date"])
    m = (dts >= D_START) & (dts <= D_END)
    close_w = idx["close"].to_numpy(dtype=np.float64)[m]
    n_days = int(m.sum())
    bench_ann = float((close_w[-1] / close_w[0]) ** (252.0 / (n_days - 1)) - 1.0)

    mism: list[str] = []
    recomputed: dict = {}
    for _, row in df.iterrows():
        cid = row["cell_id"]
        tr = pd.read_parquet(RUNS / cid / "trades.parquet")
        eq = pd.read_parquet(RUNS / cid / "equity_curve.parquet")
        net = tr["net_ret"].to_numpy(dtype=np.float64)
        rec: dict = {}
        rec["n_trades"] = len(tr)
        rec["net_avg"] = float(net.mean()) if len(net) else None
        rec["win_rate"] = float((net > 0).mean()) if len(net) else None
        ct = cluster_t_own(net, tr["entry_date"].dt.strftime("%Y%m%d").to_numpy()) \
            if len(net) else np.nan
        rec["cluster_t"] = None if not np.isfinite(ct) else float(ct)
        eqc = eq["equity_cash"].to_numpy(dtype=np.float64)
        rec["capital_utilization"] = float(
            eq["market_value"].to_numpy(dtype=np.float64).mean() / INIT)
        rec["annualized_cash"] = float(
            (eqc[-1] / INIT) ** (252.0 / (len(eqc) - 1)) - 1.0)
        rec["bench_annualized"] = bench_ann
        rec["excess_cash"] = rec["annualized_cash"] - bench_ann
        if (eqc > 0).all():
            rets = eqc[1:] / eqc[:-1] - 1.0
            rec["sharpe_cash"] = float(rets.mean() / rets.std(ddof=1)
                                       * np.sqrt(252.0))
        else:
            rec["sharpe_cash"] = None
        recomputed[cid] = rec
        for k, v in rec.items():
            ref = row[k]
            if k == "n_trades":
                ok = int(ref) == v
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                ok = ref == ""
            else:
                ok = float(ref) == v  # 字符串精确解析 + 逐位相等(容差 0)
            if not ok:
                mism.append(f"{cid}.{k}: summary={ref!r} 重算={v!r}")
    record("metrics_recompute", not mism, n_cells=len(df), n_mismatch=len(mism),
           mismatch_head=mism[:10])
    return recomputed


# ---------------------------------------------------------------- 2. 主门宣判独立重算
def check_main_gate(rec: dict) -> None:
    per_k: dict = {}
    for K in K_LIST:
        m_avg = rec[f"ALL__H60__M__P7K{K}__E1_A6"]["net_avg"]
        m_ct = rec[f"ALL__H60__M__P7K{K}__E1_A6"]["cluster_t"]
        s_best, s_sel = -np.inf, None
        for s in ("S1", "S2", "S3", "S4", "S5"):
            v = rec[f"ALL__H60__{s}__P7K{K}__E1_A6"]["net_avg"]
            if v is not None and v > s_best:
                s_best, s_sel = v, s
        diff_pp = (m_avg - s_best) * 100.0
        passed = bool(diff_pp >= 2.0 and m_ct is not None and m_ct >= 2.0)
        per_k[str(K)] = dict(diff_pp=round(diff_pp, 6), best_manual_sel=s_sel,
                             passed=passed)
    overall = any(v["passed"] for v in per_k.values())
    verdict = json.load(open(M5 / "verdict_stage_gate_m5.json", encoding="utf-8"))
    vg = verdict["main_gate"]
    mism = []
    for K in K_LIST:
        a, b = per_k[str(K)], vg["per_k"][str(K)]
        if a["passed"] != b["passed"] or a["best_manual_sel"] != b["best_manual_sel"] \
                or abs(a["diff_pp"] - b["diff_pp"]) > 1e-6:
            mism.append(f"K={K}: 重算 {a} vs verdict {b}")
    if overall != vg["passed"]:
        mism.append(f"总宣判不一致: 重算 {overall} vs verdict {vg['passed']}")
    record("main_gate_recompute", not mism, per_k=per_k, overall_passed=overall,
           verdict=vg["verdict"], mismatch=mism)


# ---------------------------------------------------------------- 3. 副门独立重算
def check_secondary() -> None:
    sc = pd.read_parquet(SCORES_PATH)
    lab = pd.read_parquet(LABELS_PATH,
                          columns=["event_id", "status_60d", "net_ret_60d"])
    sv = sc[sc["seg"] == "val"][["event_id", "ts_code", "score"]]
    m = sv.merge(lab, on="event_id", how="left", validate="one_to_one")
    uni = m[m["status_60d"] == "closed"]
    uni = uni.sort_values(["score", "ts_code", "event_id"],
                          ascending=[False, True, True], kind="mergesort")
    n_top = int(np.ceil(0.10 * len(uni)))
    precision = float((uni.head(n_top)["net_ret_60d"] > 0).mean())
    baseline = float((uni["net_ret_60d"] > 0).mean())
    verdict = json.load(open(M5 / "verdict_stage_gate_m5.json", encoding="utf-8"))
    sg = verdict["secondary_gate"]
    ok = (int(len(uni)) == sg["universe"] and n_top == sg["n_top"]
          and precision == sg["precision_top10"] and baseline == sg["pool_baseline"])
    record("secondary_gate_recompute", ok, universe=int(len(uni)), n_top=n_top,
           precision_top10=precision, pool_baseline=baseline,
           lift_pp=round((precision - baseline) * 100, 6))


# ---------------------------------------------------------------- 4. #40 机制一致性独立复核(parquet 重跑)
def check_r40_reconcile(val_keys: set) -> None:
    deg = pd.read_parquet(RUNS / "DEGENERATE__M5__P1_E1A6_S1_H60" / "trades.parquet")
    deg_m = {}
    for r in deg.to_dict("records"):
        d = r["event_date"]
        di = int(d.year * 10000 + d.month * 100 + d.day) if hasattr(d, "year") else int(d)
        deg_m[(r["ts_code"], di)] = r
    fields_f = ["entry_raw", "entry_price", "exit_raw_price", "exit_exec_price"]
    mism: list[str] = []
    n_ref_total = n_exempt = 0
    for s in ("S1", "S2", "S3", "S4", "S5"):
        cid = f"ALL__H60__{s}__P7K3__E1_A6"
        ref = pd.read_parquet(LH_RUNS / cid / "trades.parquet")
        ev_int = d_int_series(ref["event_date"])
        for r, di in zip(ref.to_dict("records"), ev_int):
            key = (r["ts_code"], int(di))
            if key not in val_keys:
                continue
            n_ref_total += 1
            b = deg_m.get(key)
            if b is None:
                cost100 = 100 * float(r["entry_price"]) + se.buy_cost(
                    100, float(r["entry_price"]))
                if cost100 > 100_000.0 + 1e-9:
                    n_exempt += 1
                else:
                    mism.append(f"{cid} {key} 退化格未成交且非 dropped_cash 豁免")
                continue
            for fc in fields_f:
                if not float(r[fc]) == float(b[fc]):
                    mism.append(f"{cid} {key} {fc}: #40={r[fc]!r} M5deg={b[fc]!r}")
            for fc in ("entry_date", "exit_date"):
                va = int(r[fc].year * 10000 + r[fc].month * 100 + r[fc].day)
                vb = b[fc]
                vb = int(vb.year * 10000 + vb.month * 100 + vb.day) \
                    if hasattr(vb, "year") else int(vb)
                if va != vb:
                    mism.append(f"{cid} {key} {fc}: #40={va} M5deg={vb}")
            if str(r["exit_reason"]) != str(b["exit_reason"]):
                mism.append(f"{cid} {key} exit_reason 不一致")
            if int(r["held_rows"]) != int(b["held_rows"]) \
                    or int(r["deferred_days"]) != int(b["deferred_days"]):
                mism.append(f"{cid} {key} held_rows/deferred_days 不一致")
    record("r40_reconcile", not mism, n_ref_val=n_ref_total,
           n_cash_exempt=n_exempt, n_mismatch=len(mism), mismatch_head=mism[:5])


# ---------------------------------------------------------------- 5. 标签对账 + 分数接入独立复核
def check_label_reconcile(val_keys: set) -> None:
    lab = pd.read_parquet(LABELS_PATH)
    lab["date_int"] = d_int_series(lab["date"])
    lab = lab.set_index(["ts_code", "date_int"], verify_integrity=True)
    mism: list[str] = []
    n_trades_total = n_subset = n_entry = 0
    for cid in CELLS:
        tr = pd.read_parquet(RUNS / cid / "trades.parquet")
        ev_int = d_int_series(tr["event_date"]) if len(tr) else np.array([])
        en_int = d_int_series(tr["entry_date"]) if len(tr) else np.array([])
        ex_int = d_int_series(tr["exit_date"]) if len(tr) else np.array([])
        for r, evd, end, exd in zip(tr.to_dict("records"), ev_int, en_int, ex_int):
            n_trades_total += 1
            key = (r["ts_code"], int(evd))
            if key not in val_keys:
                mism.append(f"{cid} {key} 非 val 键(embargo/test 开仓)")
                continue
            lr = lab.loc[key]
            status = str(lr["status_60d"])
            if status in ("dropped_limitup", "dropped_no_quote", "truncated_no_next"):
                mism.append(f"{cid} {key} 标签 {status} 却成交")
            elif status == "dropped_cash":
                cost100 = 100 * r["entry_price"] + se.buy_cost(100, r["entry_price"])
                if not cost100 > 100_000.0 + 1e-6:
                    mism.append(f"{cid} {key} dropped_cash 但 10 万可买一手")
            elif status == "truncated_exhausted":
                if r["exit_reason"] not in ("tp", "sl", "horizon"):
                    mism.append(f"{cid} {key} truncated_exhausted 出场异常")
            ed = lr["entry_date"]
            ed_int = int(ed.year * 10000 + ed.month * 100 + ed.day) \
                if pd.notna(ed) else -1
            n_entry += 1
            if ed_int != int(end):
                mism.append(f"{cid} {key} entry_date 不一致")
                continue
            if (r["exit_reason"] == "horizon" and int(r["held_rows"]) == 60
                    and int(r["deferred_days"]) == 0 and status == "closed"):
                n_subset += 1
                xd = lr["exit_date_60d"]
                xd_int = int(xd.year * 10000 + xd.month * 100 + xd.day)
                if xd_int != int(exd):
                    mism.append(f"{cid} {key} exit_date 不一致")
                px, xs = float(r["entry_price"]), float(r["exit_exec_price"])
                sh = int(100_000.0 / px / se.BOARD_LOT) * se.BOARD_LOT
                if sh < se.BOARD_LOT:
                    sh = se.BOARD_LOT
                comm = se.buy_cost(sh, px)
                while sh > 0 and sh * px + comm > 100_000.0 + 1e-6:
                    sh -= se.BOARD_LOT
                    comm = se.buy_cost(sh, px) if sh > 0 else 0.0
                xcomm, stamp = se.sell_costs(sh, xs, pd.Timestamp(str(int(exd))))
                net_l = (sh * (xs - px) - comm - xcomm - stamp) / (sh * px + comm)
                if not abs(net_l - float(lr["net_ret_60d"])) <= 1e-9:
                    mism.append(f"{cid} {key} net_ret 标签口径重算超差: "
                                f"{net_l!r} vs {lr['net_ret_60d']!r}")
                if int(r["shares"]) == sh:
                    if not abs(float(r["net_ret"]) - float(lr["net_ret_60d"])) <= 1e-9:
                        mism.append(f"{cid} {key} net_ret 同股数超差")
    record("label_reconcile", not mism, n_trades_total=n_trades_total,
           n_entry_checked=n_entry, n_subset=n_subset,
           n_mismatch=len(mism), mismatch_head=mism[:5])


# ---------------------------------------------------------------- 6. 分数接入断言独立复核
def check_score_wiring() -> set:
    sc = pd.read_parquet(SCORES_PATH)
    assert not sc["event_id"].duplicated().any()
    sv = sc[sc["seg"] == "val"]
    n_nan = int(sv["score"].isna().sum())
    lab = pd.read_parquet(LABELS_PATH, columns=["event_id", "ts_code", "date", "seg"])
    lv = lab[lab["seg"] == "val"]
    keys_sc = set(zip(sv["ts_code"], d_int_series(sv["date"])))
    keys_lab = set(zip(lv["ts_code"], d_int_series(lv["date"])))
    ok = (n_nan == 0 and len(sv) == 24405 and keys_sc == keys_lab)
    record("score_wiring", ok, n_val=len(sv), n_score_nan=n_nan,
           keys_equal=keys_sc == keys_lab)
    return keys_sc


# ---------------------------------------------------------------- 7. 确定性:detcmp 日志 + 子进程重跑 2 格
def check_determinism() -> None:
    txt = (M5 / "detcmp_m5.log").read_text(encoding="utf-8")
    log_ok = "PASS 逐位一致" in txt and "不一致格数: 0" in txt
    mism: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(M5 / "run_stage_gate_m5.py"), "--mode", "emit",
               "--only", "ALL__H60__M__P7K3__E1_A6,ALL__H60__S4__P7K3__E1_A6",
               "--emit-dir", td]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        for cid in ("ALL__H60__M__P7K3__E1_A6", "ALL__H60__S4__P7K3__E1_A6"):
            a = pd.read_parquet(RUNS / cid / "trades.parquet")
            b = pd.read_parquet(Path(td) / f"{cid}__trades.parquet")
            b = b.rename(columns={})  # emit 落盘为规范化 int 日期;a 为 datetime,统一后比
            a2 = a.copy()
            for col in ("event_date", "entry_date", "exit_date"):
                if pd.api.types.is_datetime64_any_dtype(a2[col]):
                    a2[col] = d_int_series(a2[col])
                if pd.api.types.is_datetime64_any_dtype(b[col]):
                    b[col] = d_int_series(b[col])
            same = len(a2) == len(b) and all(
                np.array_equal(a2[c].to_numpy(), b[c].to_numpy()) for c in a2.columns)
            if not same:
                mism.append(f"{cid} 子进程重跑与落盘不逐位一致")
    record("determinism", log_ok and not mism, detcmp_log_pass=bool(log_ok),
           emit_rerun_mismatch=mism)


def main() -> None:
    t0 = time.time()
    val_keys = check_score_wiring()
    rec = check_metrics()
    check_main_gate(rec)
    check_secondary()
    check_r40_reconcile(val_keys)
    check_label_reconcile(val_keys)
    check_determinism()
    RESULTS["all_pass"] = all(c["ok"] for c in RESULTS["checks"].values())
    RESULTS["elapsed_sec"] = round(time.time() - t0, 1)
    with open(M5 / "selfcheck_results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"SELFCHECK M5 DONE all_pass={RESULTS['all_pass']} "
          f"({RESULTS['elapsed_sec']}s)", flush=True)
    if not RESULTS["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
