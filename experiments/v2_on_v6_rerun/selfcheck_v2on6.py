#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 7:独立自检。

预登记军令状 = 同目录 README.md(commit 49b9c5b,冻结)§九 验收断言清单。
纪律:不 import 本线任何驱动(build_panel/build_labels/build_master/train_stack/
run_gate)的函数;抽样重算全部自写实现(v3_pipeline 的段界断言与 talib 库本身
不属本线驱动,可用);任一 FAIL 退出码 1。

覆盖(README §九 1-9 机检):
  1 因果/口径抽检:自写重算样本特征(talib 三列 + 若干 rolling/shift 列)对主表;
  2 泄漏:主表/分数表无标签列;主表无 future_/stop_loss_/label_ 列;
  3 段界:assert_segment_integrity 全量 + 段计数 + 训练行 seg 全 train;
  4 行数与键:96,577 守恒、event_id 唯一、(ts_code,event_date) 与事件表互证、
    event_close 抽样对日线 close 逐位、event_row 行号一致;
  5 背离结构族:抽样 500 事件自写重算逐位;
  6 词典:列集 = 主表列集、中文全称非空;
  7 确定性:面板/标签/分数双跑 md5 台账逐位一致;
  8 选择清单:OPTIMIZED/STABLE 落盘、子集关系、条数与台账一致;
  9 门产物:18 格全出数、trades/equity 落盘、verdict 机械一致(自 summary 重算)。
  另:标签抽样自写重算(窗口 [j+2,j+1+p] 首触扫描)对 labels_v2on6 逐位。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import talib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
import train_eval_pipeline as tep  # noqa: E402  v3_pipeline 既有断言(非本线驱动)

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
MASTER_V6_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
DAILY_DIR = REPO / "stock_data" / "daily"
SH_INDEX_PATH = DAILY_DIR / "000001.SH.parquet"
CACHE = SCRIPT_DIR / "cache"
PROGRESS = SCRIPT_DIR / "progress.log"

RETURN_PERIODS = [3, 5, 10, 15, 20, 25, 30]
PREREG_SEGMENT_COUNTS = {"train": 50165, "val": 24405, "test": 17902,
                         "embargo": 3063, "pre2001": 1042}
SEED = 42

RESULTS: list = []


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [selfcheck] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append(dict(check=name, ok=bool(ok), detail=detail))
    log(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}")


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_stock(code: str) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_DIR / f"{code}.parquet",
                         columns=["trade_date", "open", "high", "low", "close",
                                  "vol", "amount"])
    return df.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- 自写标签重算 ----
def label_recompute(df: pd.DataFrame, j: int, p: int) -> tuple:
    """自写标量扫描:窗口 [j+2, j+1+p];原版 15% 首触/期末;止损版时间优先
    日内 open→low→high;窗口末端超史 → 全 NaN。"""
    n = len(df)
    if j + 1 + p >= n:
        return np.nan, pd.NaT, np.nan, pd.NaT
    buy = float(df["close"].iat[j])
    target, stop = buy * 1.15, buy * 0.65
    fr, fdr = None, None
    sr, sdr = None, None
    for k in range(j + 2, j + 2 + p):
        if fr is None and float(df["high"].iat[k]) >= target:
            fr, fdr = (target - buy) / buy, df["trade_date"].iat[k]
        if sr is None:
            o_, l_, h_ = (float(df["open"].iat[k]), float(df["low"].iat[k]),
                          float(df["high"].iat[k]))
            if o_ <= stop:
                sr, sdr = (o_ - buy) / buy, df["trade_date"].iat[k]
            elif l_ <= stop:
                sr, sdr = (stop - buy) / buy, df["trade_date"].iat[k]
            elif h_ >= target:
                sr, sdr = (target - buy) / buy, df["trade_date"].iat[k]
    end = j + 1 + p
    if fr is None:
        fr, fdr = (float(df["close"].iat[end]) - buy) / buy, df["trade_date"].iat[end]
    if sr is None:
        sr, sdr = (float(df["close"].iat[end]) - buy) / buy, df["trade_date"].iat[end]
    return fr, fdr, sr, sdr


# ---------------------------------------------------------------- 自写特征重算 ----
def feature_recompute(df: pd.DataFrame, T: pd.Timestamp) -> dict:
    """自写最小重算(截断 ≤ T):talib 三列 + rolling/shift 派生列,对主表抽样列。"""
    d = df[df["trade_date"] <= T].reset_index(drop=True)
    c = d["close"].to_numpy(np.float64)
    v = d["vol"].to_numpy(np.float64)
    h = d["high"].to_numpy(np.float64)
    l = d["low"].to_numpy(np.float64)
    o = d["open"].to_numpy(np.float64)
    macd, sig, hist = talib.MACD(c)
    rsi14 = talib.RSI(c, timeperiod=14)
    ma60 = talib.MA(c, timeperiod=60)
    obv = talib.OBV(c, v)
    atr = talib.ATR(h, l, c)
    i = len(d) - 1
    out = {
        "macd": macd[i], "macd_signal": sig[i], "macd_hist": hist[i],
        "rsi_14": rsi14[i], "ma_60": ma60[i], "obv": obv[i], "atr": atr[i],
        "volume": v[i],
        "close_lag_3": c[i - 3] if i >= 3 else np.nan,
        "volume_lag_5": v[i - 5] if i >= 5 else np.nan,
        "daily_return": c[i] / c[i - 1] - 1 if i >= 1 else np.nan,
        "amplitude": (h[i] - l[i]) / l[i],
        "high_mean_20": h[i - 19:i + 1].mean() if i >= 19 else np.nan,
        "log_volume": np.log1p(v[i]),
        "intraday_pos": (c[i] - l[i]) / (h[i] - l[i] + 1e-9),
        "ret_overnight": o[i] / c[i - 1] - 1 if i >= 1 else np.nan,
    }
    return out


def main() -> None:
    t_all = time.time()
    log("SELFCHECK START")
    ev = pd.read_parquet(EVENTS_PATH)
    master = pd.read_parquet(CACHE / "master_v2on6_run1.parquet")
    labels = pd.read_parquet(CACHE / "labels_v2on6_run1.parquet")
    scores = pd.read_parquet(CACHE / "scores_v2on6_run1.parquet")
    rng = np.random.default_rng(SEED)

    # ---- §九.4:行数与键 ----
    ok = (len(master) == 96577 and master["event_id"].is_unique
          and not master.duplicated(["ts_code", "event_date"]).any())
    key_m = set(zip(master["ts_code"], master["event_date"]))
    key_e = set(zip(ev["ts_code"], ev["event_date"]))
    ok = ok and key_m == key_e
    record("§九.4 行数与键", ok,
           f"master {len(master)} 行,event_id 唯一,(ts_code,event_date) 与事件表互证")

    m6 = pd.read_parquet(MASTER_V6_PATH, columns=["event_id", "ts_code", "date", "seg"])
    j = master.merge(m6, on="event_id", how="left", validate="1:1",
                     suffixes=("", "_m6"))
    ok = bool(((j["ts_code"] == j["ts_code_m6"]).all()
               and (j["event_date"] == j["date"]).all()
               and (j["seg"] == j["seg_m6"]).all()))
    record("§九.4 event_id/seg 对 master_v6 逐行一致", ok, f"{len(j)} 行")

    # event_close / event_row 抽样逐位(自写装载)
    picks = rng.choice(len(ev), size=500, replace=False)
    bad = []
    cache_df: dict = {}
    for i in picks:
        r = ev.iloc[i]
        code = r["ts_code"]
        if code not in cache_df:
            cache_df[code] = load_stock(code)
        d = cache_df[code]
        pos = int(d["trade_date"].searchsorted(r["event_date"]))
        if not (pos < len(d) and d["trade_date"].iat[pos] == r["event_date"]):
            bad.append(f"{code} {r['event_date']} 不在日线")
            continue
        if pos != int(r["event_row"]):
            bad.append(f"{code} {r['event_date']} 行号 {pos} != {r['event_row']}")
        if float(d["close"].iat[pos]) != float(r["event_close"]):
            bad.append(f"{code} close {d['close'].iat[pos]!r} != {r['event_close']!r}")
    record("§九.4 event_close/event_row 抽样 500 逐位", not bad, f"不一致 {len(bad)}")

    # ---- §九.2:泄漏列扫描 ----
    bad_cols = [c for c in master.columns
                if c.startswith(("future_", "stop_loss_", "label_"))]
    ok = not bad_cols and list(scores.columns) == ["event_id", "ts_code", "date",
                                                   "seg", "score"]
    record("§九.2 泄漏列扫描(主表/分数表)", ok,
           f"主表标签列 {bad_cols};分数表恰 5 列")

    # ---- §九.3:段界 ----
    idx_sh = pd.read_parquet(SH_INDEX_PATH, columns=["trade_date"])
    sh_cal = pd.to_datetime(idx_sh["trade_date"]).sort_values().unique()
    try:
        tep.assert_segment_integrity(
            master.rename(columns={"event_date": "date"})[["date", "seg"]], sh_cal)
        seg_counts = master["seg"].value_counts().to_dict()
        seg_ok = {k: seg_counts.get(k, 0) for k in PREREG_SEGMENT_COUNTS} \
            == PREREG_SEGMENT_COUNTS
        record("§九.3 段界与段计数", seg_ok, f"段计数 {seg_counts}")
    except AssertionError as e:
        record("§九.3 段界与段计数", False, str(e))

    # ---- §九.1:口径抽检(自写重算特征对主表)----
    codes = sorted(ev["ts_code"].unique())
    code_picks = [codes[i] for i in rng.choice(len(codes), size=6, replace=False)]
    cmp_cols = ["macd", "macd_signal", "macd_hist", "rsi_14", "ma_60", "obv", "atr",
                "volume", "close_lag_3", "volume_lag_5", "daily_return", "amplitude",
                "high_mean_20", "log_volume", "intraday_pos", "ret_overnight"]
    bad, n_chk = [], 0
    mi = master.set_index(["ts_code", "event_date"])
    for code in code_picks:
        df = load_stock(code)
        days = sorted(ev.loc[ev["ts_code"] == code, "event_date"])
        for di in rng.choice(len(days), size=min(2, len(days)), replace=False):
            T = days[di]
            got = feature_recompute(df, T)
            row = mi.loc[(code, T)]
            for cname in cmp_cols:
                a, b = float(got[cname]), float(row[cname])
                n_chk += 1
                if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                    bad.append(f"{code} {T.date()} {cname}: 重算 {a!r} != 主表 {b!r}")
    record("§九.1 特征口径抽检(6 股 × 2 日 × 16 列,自写重算)", not bad,
           f"{n_chk} 单元,不一致 {len(bad)};{bad[:3]}")

    # ---- §九.5:背离结构族抽样 500 自写重算 ----
    with open(CACHE / "master_ledger_run1.json", encoding="utf-8") as f:
        enc = json.load(f)["volume_signal_encoder"]
    aux_files = sorted((CACHE / "event_aux_run1").glob("*.parquet"))
    aux = pd.concat([pd.read_parquet(f).assign(ts_code=f.stem)
                     for f in aux_files], ignore_index=True)
    pv = ev[["ts_code", "event_date"]].merge(aux, on=["ts_code", "event_date"],
                                             how="left", validate="1:1")["prev_cross_vol"]
    div_cols = ["close_current", "close_previous", "macd_current", "macd_previous",
                "price_decline_pct", "macd_increase_pct", "compare_rank",
                "formation_period", "is_quick_divergence_y", "divergence_strength",
                "price_macd_ratio", "divergence_magnitude", "confirmation_score",
                "divergence_amount", "volume_signal", "v2j2_macd_increase_pct"]
    amount_by_date = ev.groupby("event_date")["ts_code"].size()
    bad, n_checked = [], 0
    for i in rng.choice(len(ev), size=500, replace=False):
        r = ev.iloc[i]
        ac, mpc = float(r.anchor_close), float(r.min_prev_close)
        cd, cpd, cp2 = float(r.cross_dif), float(r.cross_prev_dif), float(r.cross_prev2_dif)
        lift, bars = float(r.dif_lift), int(r.anchor_bars)
        pdp = (ac - mpc) / mpc
        mip = 0.0 if cpd == 0 else lift / abs(cpd)
        vs = ("bullish" if float(r.event_vol) < float(pv.iloc[i])
              else "bearish" if float(r.event_vol) > 1.5 * float(pv.iloc[i]) else "neutral")
        exp = {
            "close_current": ac, "close_previous": mpc, "macd_current": cd,
            "macd_previous": cpd, "price_decline_pct": pdp, "macd_increase_pct": mip,
            "compare_rank": 1, "formation_period": bars,
            "is_quick_divergence_y": 1 if bars < 3 else 0,
            "divergence_strength": 0.4 * min(1.0, abs(pdp) / 0.1) + 0.6 * min(1.0, mip / 0.1),
            "price_macd_ratio": abs(mip / max(abs(pdp), 1e-6)),
            "divergence_magnitude": (abs(pdp) + abs(mip)) / 2.0,
            "confirmation_score": 1 if (pdp < -0.02 and mip > 0.01) else 0,
            "divergence_amount": int(amount_by_date.loc[r.event_date]),
            "volume_signal": enc[vs],
            "v2j2_macd_increase_pct": 0.0 if cp2 == 0 else (cd - cp2) / abs(cp2),
        }
        row = mi.loc[(r.ts_code, r.event_date)]
        n_checked += 1
        for cname in div_cols:
            e, g = exp[cname], row[cname]
            if isinstance(e, float):
                if not (float(g) == e or (np.isnan(float(g)) and np.isnan(e))):
                    bad.append(f"行{i} {cname}: {g!r} != {e!r}")
            elif int(g) != int(e):
                bad.append(f"行{i} {cname}: {g!r} != {e!r}")
    record("§九.5 背离结构族抽样 500 自写重算", not bad,
           f"{n_checked} 事件 × 16 列,不一致 {len(bad)};{bad[:3]}")

    # ---- 标签抽样自写重算(300 事件 × 7 期)----
    lab_ix = labels.set_index(["ts_code", "event_date"])
    bad, n_chk = [], 0
    for i in rng.choice(len(ev), size=300, replace=False):
        r = ev.iloc[i]
        code = r["ts_code"]
        if code not in cache_df:
            cache_df[code] = load_stock(code)
        df = cache_df[code]
        jpos = int(r["event_row"])
        row = lab_ix.loc[(code, r["event_date"])]
        for p in RETURN_PERIODS:
            fr, fdr, sr, sdr = label_recompute(df, jpos, p)
            for prefix, e, gcol in (("future_return", fr, f"future_return_{p}d"),
                                    ("stop_loss_return", sr, f"stop_loss_return_{p}d")):
                g = row[gcol]
                n_chk += 1
                if np.isnan(e):
                    if not pd.isna(g):
                        bad.append(f"{code} {r['event_date'].date()} {gcol}: 重算 NaN != {g!r}")
                elif not (float(g) == float(e)):
                    bad.append(f"{code} {r['event_date'].date()} {gcol}: {g!r} != {e!r}")
            for prefix, e, gcol in (("fdr", fdr, f"future_sell_date_{p}d"),
                                    ("sdr", sdr, f"stop_loss_sell_date_{p}d")):
                g = row[gcol]
                n_chk += 1
                if pd.isna(e):
                    if not pd.isna(g):
                        bad.append(f"{code} {r['event_date'].date()} {gcol}: 重算 NaT != {g}")
                elif pd.Timestamp(g) != pd.Timestamp(e):
                    bad.append(f"{code} {r['event_date'].date()} {gcol}: {g} != {e}")
    record("标签抽样 300 事件 × 7 期 × 4 列自写重算", not bad,
           f"{n_chk} 单元,不一致 {len(bad)};{bad[:3]}")

    # ---- §九.6:词典 ----
    dic = pd.read_csv(SCRIPT_DIR / "dictionary_v2on6.csv")
    ok = (set(dic["column"]) == set(master.columns)
          and dic["cn_name"].notna().all() and dic["cn_name"].str.len().gt(0).all()
          and not dic["column"].duplicated().any())
    record("§九.6 词典列集与中文全称", ok, f"{len(dic)} 列")

    # ---- §九.7:双跑确定性(台账 md5)----
    det = {}
    for name, f1, f2, key in (
            ("labels", "labels_ledger_run1.json", "labels_ledger_run2.json", "md5"),):
        with open(CACHE / f1) as f:
            det[name] = json.load(f)[key]
        with open(CACHE / f2) as f:
            det[name + "_2"] = json.load(f)[key]
    ok_labels = det["labels"] == det["labels_2"]
    ok_panel = ok_scores = None
    try:
        with open(CACHE / "panel_ledger_run1.json") as f:
            p1 = json.load(f)["eventrows_md5"]
        with open(CACHE / "panel_ledger_run2.json") as f:
            p2 = json.load(f)["eventrows_md5"]
        ok_panel = (p1 == p2)
    except FileNotFoundError:
        ok_panel = "panel run2 未跑(待补)"
    try:
        with open(SCRIPT_DIR / "selection_results_v2on6.json") as f:
            s1 = json.load(f)["scores_md5"]
        ok_scores = (s1 == md5_of(CACHE / "scores_v2on6_run1.parquet"))
    except FileNotFoundError:
        ok_scores = "scores 未出(待补)"
    ok = bool(ok_labels) and ok_panel is True and ok_scores is True
    record("§九.7 双跑 md5(标签/面板/分数)", ok,
           f"labels {'一致' if ok_labels else '不一致'};panel {ok_panel};scores {ok_scores}")

    # ---- §九.8:选择清单 ----
    try:
        opt = json.load(open(SCRIPT_DIR / "optimized_features_v2on6.json"))
        stb = json.load(open(SCRIPT_DIR / "stable_features_v2on6.json"))
        sel = json.load(open(SCRIPT_DIR / "selection_results_v2on6.json"))
        feat_cols = [c for c in master.columns
                     if c not in ("ts_code", "event_date", "event_row", "event_id", "seg")]
        ok = (len(opt) == sel["optimized_v2on6"]["n"]
              and len(stb) == sel["stable_v2on6"]["n"]
              and set(opt) <= set(feat_cols)
              and set(stb) <= (set(feat_cols) | {"pred_lgb", "pred_lgb_squared"})
              and len(opt) > 0 and len(stb) > 0)
        record("§九.8 选择清单程序化落盘", ok,
               f"OPTIMIZED {len(opt)} 条 / STABLE {len(stb)} 条")
    except FileNotFoundError as e:
        record("§九.8 选择清单程序化落盘", False, f"缺文件 {e.filename}")

    # ---- 分数表口径 ----
    n_val = int((scores["seg"] == "val").sum())
    val_non_nan = int(scores.loc[scores["seg"] == "val", "score"].notna().sum())
    tr_nan = int(scores.loc[scores["seg"] == "train", "score"].isna().sum())
    lab_tr_nan = int(labels.merge(master[["ts_code", "event_date", "seg"]],
                                  on=["ts_code", "event_date"], validate="1:1")
                     .query("seg == 'train'")["future_return_15d"].isna().sum())
    in01 = bool(scores["score"].dropna().between(0, 1).all())
    ok = (len(scores) == 96577 and scores["event_id"].is_unique
          and n_val == 24405 and val_non_nan == 24405
          and tr_nan == lab_tr_nan and in01)
    record("分数表口径(96,577/val 全覆盖/train NaN=标签 NaN/值域)",
           ok, f"val {n_val}(非 NaN {val_non_nan}),train NaN {tr_nan} vs {lab_tr_nan}")

    # ---- §九.9:门产物(18 格全出数 + verdict 机械一致)----
    try:
        summ = pd.read_csv(SCRIPT_DIR / "summary_gate_v2on6.csv")
        with open(SCRIPT_DIR / "verdict_gate_v2on6.json", encoding="utf-8") as f:
            verdict = json.load(f)
        n_files = 0
        for cid in summ["cell_id"]:
            d = SCRIPT_DIR / "runs_gate" / cid
            if (d / "trades.parquet").exists() and (d / "equity_curve.parquet").exists():
                n_files += 1
        mg = verdict["main_gate"]["per_k"]
        recalc_ok = True
        for K in ("3", "5", "10"):
            m_avg = float(summ.loc[(summ["sel"] == "M") & (summ["K"] == int(K)),
                                   "net_avg"].iloc[0])
            s_best = max(float(summ.loc[(summ["sel"] == s) & (summ["K"] == int(K)),
                                        "net_avg"].iloc[0])
                         for s in ("S1", "S2", "S3", "S4", "S5"))
            if abs((m_avg - s_best) * 100.0 - mg[K]["diff_pp"]) > 1e-6:
                recalc_ok = False
        ok = (len(summ) == 18 and n_files == 18 and recalc_ok
              and bool(verdict["checks_all_pass"]))
        record("§九.9 门产物 18 格与 verdict 机械一致", ok,
               f"summary {len(summ)} 行,trades/equity {n_files}/18,"
               f"主门重算一致 {recalc_ok},自检总评 {verdict['checks_all_pass']}")
    except FileNotFoundError as e:
        record("§九.9 门产物 18 格与 verdict 机械一致", False, f"缺文件 {e.filename}")

    # ---- 汇总 ----
    n_fail = sum(1 for r in RESULTS if not r["ok"])
    out = dict(sec=round(time.time() - t_all, 1), n_checks=len(RESULTS),
               n_fail=n_fail, all_pass=n_fail == 0, results=RESULTS)
    with open(SCRIPT_DIR / "selfcheck_results_v2on6.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"SELFCHECK DONE | {len(RESULTS)} 项,FAIL {n_fail} | "
        f"{'ALL PASS' if n_fail == 0 else 'HAS FAIL'}({time.time() - t_all:.0f}s)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
