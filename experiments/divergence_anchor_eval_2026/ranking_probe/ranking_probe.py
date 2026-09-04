"""池内排序探针：是否存在任何单因子能把正期望单子从背离事件池中挑出来。

预登记见同目录 README.md（先于本脚本任何跑数落盘）。
口径要点：
- 仅用 event_study.parquet 中 status=="closed" 的行；
- 主目标 = E1-H12 的 ret；副目标 = E1-H12 的 excess；稳健性对照 = A13 的 ret；
- 全部因子仅用 <= event_date 的数据计算，特征函数内置硬断言防泄漏。
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

BASE = "experiments/divergence_anchor_eval_2026"
OUT = os.path.join(BASE, "ranking_probe")
DAILY_DIR = "stock_data/daily"

FACTORS_V1 = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"]
FACTORS_V2_EXTRA = ["f13"]

T_SPEARMON_MIN = 2.9       # Bonferroni p<0.004 (12~13 因子)
T_SLICE_MIN = 3.0
ECON_MIN = 0.01            # +1pp 经济意义门槛
SELF_CHECK_TOL = 1e-6


def _to_ts(s):
    """trade_date 可能是 int(YYYYMMDD) 或 Timestamp，统一为 Timestamp。"""
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s.astype(str), format="%Y%m%d")
    return pd.to_datetime(s)


def compute_symbol_features(ts_code, ev1, ev2):
    """计算单股全部事件特征。仅使用 <= event_date 的行情，内置防泄漏硬断言。"""
    path = os.path.join(DAILY_DIR, f"{ts_code}.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    cols = set(df.columns)
    if not ("ts_code" in cols and "vol" in cols):
        return []
    if len(df) < 100:
        return []
    df = df.sort_values("trade_date").reset_index(drop=True)
    dates = _to_ts(df["trade_date"])
    close = df["close"].to_numpy(dtype=float)
    vol = df["vol"].to_numpy(dtype=float)
    amount = df["amount"].to_numpy(dtype=float) if "amount" in df.columns else np.full(len(df), np.nan)
    ret1 = pd.Series(close).pct_change().to_numpy()
    pos = {d: i for i, d in enumerate(dates)}

    rows = []
    for signal, ev in (("events_v1", ev1), ("events_v2", ev2)):
        if ev is None:
            continue
        for _, e in ev.iterrows():
            event_date = pd.Timestamp(e["event_date"])
            ie = pos.get(event_date)
            if ie is None:
                continue
            # 硬断言：用于特征的行情最大日期 == event_date（防泄漏）
            assert dates.iloc[: ie + 1].max() == event_date, f"{ts_code} {event_date} 泄漏"
            anchor_date = pd.Timestamp(e["anchor_date"])
            ia = pos.get(anchor_date)
            anchor_close = float(e["anchor_close"])

            f1 = float(e["dif_lift"])
            f2 = float(e["cross_dif"])
            f3 = f2 / close[ie] if close[ie] > 0 else np.nan

            icp = pos.get(pd.Timestamp(e["cross_prev_date"]))
            f4 = float(ie - icp) if icp is not None else np.nan
            f5 = float(ie - ia) if ia is not None else np.nan
            f6 = close[ie] / anchor_close - 1.0 if anchor_close > 0 else np.nan

            if ie + 1 >= 250:
                w250 = close[ie - 249 : ie + 1]
                lo, hi = w250.min(), w250.max()
                f7 = (anchor_close - lo) / (hi - lo) if hi > lo else np.nan
            else:
                f7 = np.nan

            if ia is not None:
                w60 = close[max(0, ie - 60) : ia + 1]
                f8 = anchor_close / w60.max() - 1.0 if len(w60) and w60.max() > 0 else np.nan
            else:
                f8 = np.nan

            f9 = close[ie] / close[ie - 20] - 1.0 if ie >= 20 and close[ie - 20] > 0 else np.nan
            f10 = vol[ie] / vol[ie - 20 : ie].mean() if ie >= 20 and vol[ie - 20 : ie].mean() > 0 else np.nan
            f11 = float(np.std(ret1[ie - 19 : ie + 1], ddof=1)) if ie >= 19 else np.nan
            f12 = float(amount[ie])

            row = {
                "signal": signal, "ts_code": ts_code, "event_date": event_date,
                "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
                "f7": f7, "f8": f8, "f9": f9, "f10": f10, "f11": f11, "f12": f12,
            }
            if signal == "events_v2":
                apc = float(e["anchor_prev_close"])
                row["f13"] = anchor_close / apc - 1.0 if apc > 0 else np.nan
            rows.append(row)
    return rows


def _worker(task):
    return compute_symbol_features(*task)


def spearman_t(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 30 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, n
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho**2))
    return rho, t, n


def one_sample_t(a):
    a = a[~np.isnan(a)]
    if len(a) < 30 or a.std(ddof=1) == 0:
        return np.nan
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))


def decile_assign(x):
    """等频十分位，返回 1..10；NaN 保持 NaN。"""
    out = pd.Series(np.nan, index=x.index)
    m = x.notna()
    out[m] = pd.qcut(x[m].rank(method="first"), 10, labels=False) + 1
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    ev1 = pd.read_parquet(os.path.join(BASE, "events_v1.parquet"))
    ev2 = pd.read_parquet(os.path.join(BASE, "events_v2.parquet"))
    es = pd.read_parquet(os.path.join(BASE, "backtest", "event_study.parquet"))
    es = es[es.status == "closed"].copy()
    es["event_date"] = pd.to_datetime(es["event_date"])

    by_code = {}
    for df, key in ((ev1, "ev1"), (ev2, "ev2")):
        for code, g in df.groupby("ts_code"):
            by_code.setdefault(code, {})[key] = g
    tasks = [(code, d.get("ev1"), d.get("ev2")) for code, d in by_code.items()]
    print(f"symbols with events: {len(tasks)}", flush=True)

    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        parts = pool.map(_worker, tasks, chunksize=16)
    rows = [r for part in parts for r in part]
    feat = pd.DataFrame(rows)
    print(f"feature rows: {len(feat)}, elapsed {time.time()-t0:.1f}s", flush=True)
    feat.to_parquet(os.path.join(OUT, "features.parquet"), index=False)

    es1 = es[es.config == "E1-H12"][
        ["signal", "ts_code", "event_date", "ret", "excess"]
    ]
    esa = es[es.config == "A13"][["signal", "ts_code", "event_date", "ret"]].rename(
        columns={"ret": "ret_a13"}
    )
    dfm = feat.merge(es1, on=["signal", "ts_code", "event_date"], how="inner").merge(
        esa, on=["signal", "ts_code", "event_date"], how="inner"
    )
    print(f"joined rows: {len(dfm)}", flush=True)

    decile_rows, spear_rows, slice_rows = [], [], []
    verdict = {"factors": {}, "self_check": {}, "overall": None}
    factor_names = {
        "f1": "dif_lift", "f2": "cross_dif", "f3": "cross_dif/close_event",
        "f4": "金叉间隔交易日数", "f5": "事件距锚交易日数", "f6": "锚点反弹幅度",
        "f7": "锚点52周位置", "f8": "入锚回撤", "f9": "事件前20日动量",
        "f10": "事件日量比", "f11": "20日波动率", "f12": "事件日成交额",
        "f13": "锚对锚跌幅",
    }

    for signal in ["events_v1", "events_v2"]:
        d = dfm[dfm.signal == signal]
        pool_mean = d.ret.mean()
        factors = FACTORS_V1 + (FACTORS_V2_EXTRA if signal == "events_v2" else [])
        for f in factors:
            x = d[f].to_numpy(dtype=float)
            ret = d.ret.to_numpy(dtype=float)
            exc = d.excess.to_numpy(dtype=float)

            dec = decile_assign(d[f])
            for k in range(1, 11):
                g = d[dec == k]
                if len(g) == 0:
                    continue
                decile_rows.append({
                    "signal": signal, "factor": f, "factor_name": factor_names[f],
                    "decile": k, "n": len(g),
                    "ret_mean": g.ret.mean(), "ret_median": g.ret.median(),
                    "win_rate": (g.ret > 0).mean(), "excess_mean": g.excess.mean(),
                })

            rho, t_rho, n_sp = spearman_t(x, ret)

            dm = d.dropna(subset=[f])
            n80 = dm[f].quantile(0.8)
            n90 = dm[f].quantile(0.9)
            n20 = dm[f].quantile(0.2)
            for name, mask in (
                ("top_decile", dm[f] >= n90),
                ("top_quintile", dm[f] >= n80),
                ("bottom_quintile", dm[f] <= n20),
            ):
                g = dm[mask]
                slice_rows.append({
                    "signal": signal, "factor": f, "factor_name": factor_names[f],
                    "slice": name, "n": len(g), "ret_mean": g.ret.mean(),
                    "excess_mean": g.excess.mean(), "ret_t": one_sample_t(g.ret.to_numpy()),
                })

            # 判活
            dec_means = (
                pd.DataFrame({"dec": dec, "ret": d.ret}).dropna()
                .groupby("dec").ret.mean()
            )
            # 十分位均值对档位序号的 Spearman 相关（仅 10 点，不走 spearman_t 的 n>=30 门槛）
            mono_rho = float(
                pd.Series(dec_means.to_numpy()).rank().corr(
                    pd.Series(dec_means.index.to_numpy(dtype=float)).rank()
                )
            )
            direction_ok = bool(
                np.isfinite(rho) and np.isfinite(mono_rho) and np.sign(rho) == np.sign(mono_rho)
            )
            favored = "top_quintile" if (np.isfinite(rho) and rho > 0) else "bottom_quintile"
            fs = [r for r in slice_rows if r["signal"] == signal and r["factor"] == f and r["slice"] == favored][0]
            c1 = bool(np.isfinite(t_rho) and abs(t_rho) >= T_SPEARMON_MIN and direction_ok)
            c2 = bool(np.isfinite(fs["ret_t"]) and fs["ret_mean"] > 0 and fs["ret_t"] >= T_SLICE_MIN)
            c3 = bool(fs["ret_mean"] - pool_mean >= ECON_MIN)
            passed = c1 and c2 and c3

            a13_note = None
            if passed:
                rho_a, t_a, _ = spearman_t(x, d.ret_a13.to_numpy(dtype=float))
                a13_ok = bool(np.isfinite(rho_a) and np.sign(rho_a) == np.sign(rho))
                a13_note = {"rho_a13": rho_a, "t_a13": t_a, "direction_consistent": a13_ok}

            verdict["factors"][f"{signal}/{f}"] = {
                "factor_name": factor_names[f], "n": n_sp, "rho": rho, "t": t_rho,
                "decile_monotone_rho": mono_rho, "direction_ok": direction_ok,
                "favored_end": favored, "favored_quintile_ret_mean": fs["ret_mean"],
                "favored_quintile_t": fs["ret_t"], "pool_ret_mean": pool_mean,
                "favored_minus_pool": fs["ret_mean"] - pool_mean,
                "crit1_spearman": c1, "crit2_favored_positive": c2, "crit3_econ_1pp": c3,
                "passed": passed, "a13_check": a13_note,
            }
            spear_rows.append({
                "signal": signal, "factor": f, "factor_name": factor_names[f],
                "n": n_sp, "rho": rho, "t": t_rho,
            })

    # ---- 自检：f1 三分位精确复现 backtest/event_study_terciles.csv ----
    pub = pd.read_csv(os.path.join(BASE, "backtest", "event_study_terciles.csv"))
    self_ok = True
    details = []
    for signal in ["events_v1", "events_v2"]:
        d = dfm[dfm.signal == signal]
        ter = pd.qcut(d["f1"].rank(method="first"), 3, labels=["low", "mid", "high"])
        g = d.groupby(ter)
        mine = pd.DataFrame({
            "n": g.size(), "ret_mean": g.ret.mean(), "ret_median": g.ret.median(),
            "win_rate": g.apply(lambda s: (s.ret > 0).mean(), include_groups=False),
            "excess_mean": g.excess.mean(),
            "dif_lift_min": g.f1.min(), "dif_lift_max": g.f1.max(),
        })
        for tname in ["low", "mid", "high"]:
            p = pub[(pub.signal == signal) & (pub.tercile == tname)].iloc[0]
            m = mine.loc[tname]
            for col in ["n", "ret_mean", "ret_median", "win_rate", "excess_mean", "dif_lift_min", "dif_lift_max"]:
                if abs(float(p[col]) - float(m[col])) > SELF_CHECK_TOL:
                    self_ok = False
                    details.append(f"{signal}/{tname}/{col}: pub={p[col]} mine={m[col]}")
        # 十分位方向：v1 高档不优于低档超 +0.5pp；v2 D10 最差或次差
        dec = decile_assign(d["f1"])
        dmean = d.groupby(dec).ret.mean()
        if signal == "events_v1":
            hi = dmean.loc[[8, 9, 10]].mean()
            lo = dmean.loc[[1, 2, 3]].mean()
            if hi - lo > 0.005:
                self_ok = False
                details.append(f"v1 f1 高档优于低档 {(hi-lo)*100:.2f}pp")
        else:
            worst_rank = dmean.rank().loc[10.0]
            if worst_rank > 2:
                self_ok = False
                details.append(f"v2 f1 D10 非最差/次差 (rank {worst_rank}/10 ascending)")
    verdict["self_check"] = {"passed": bool(self_ok), "details": details}
    if not self_ok:
        print("SELF-CHECK FAILED:", details, flush=True)

    any_pass = any(v["passed"] for v in verdict["factors"].values())
    verdict["overall"] = {
        "verdict": "有因子过线" if any_pass else "判死：无因子过线",
        "passed_factors": [k for k, v in verdict["factors"].items() if v["passed"]],
        "note": "过线因子仅为探索期线索，必须在战役验证段重验" if any_pass else "2026 沙盒内无可排序结构",
        "elapsed_sec": round(time.time() - t0, 1),
    }

    pd.DataFrame(decile_rows).to_csv(os.path.join(OUT, "decile_tables.csv"), index=False)
    pd.DataFrame(spear_rows).to_csv(os.path.join(OUT, "spearman.csv"), index=False)
    pd.DataFrame(slice_rows).to_csv(os.path.join(OUT, "top_slice.csv"), index=False)
    with open(os.path.join(OUT, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2, default=float)
    print(f"done, elapsed {time.time()-t0:.1f}s, verdict: {verdict['overall']['verdict']}", flush=True)
    return 0 if self_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
