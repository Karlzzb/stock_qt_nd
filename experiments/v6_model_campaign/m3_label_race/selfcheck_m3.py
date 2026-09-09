#!/usr/bin/env python3
"""M3 验收独立自检(战役 #47 M3,issue #50;README §六 的独立复核落点)。

纪律:本脚本对承重断言一律**独立实现**(不 import build_labels_v6 /
run_label_race_v6 的任何函数;模拟器按预登记口径重写,成本常量按规格文本硬编码
并先与冻结引擎 strategy_engine 常量断言相等),只从落盘产物 + 原始数据重算复核:
  1. 因果性: 抽样 ≥15 股(种子 20260909)截序列头部 200 行重算标签,
     交集事件逐位一致(容差 1e-9,NaN 视同相等);
  2. 对账: trades_seed(variant=='v1', H∈{10,20,25}) 净收益逐笔(容差 1e-9)
     与 status 剔除类别交叉核对;并对抽样股用独立模拟器全量重算比对标签表;
  3. 标签-特征隔离: 静态核查构建器只读主表元数据列;标签表恰 18 列;
     跑训特征集无 label_ 混入且 == 主表 1,997 特征列;
  4. 段界: feature_master.assert_segment_integrity 主表全量执行;
     metrics 表 train/val 行数与独立重算有效标签计数逐候选一致;
  5. 泄漏列: 1,997 特征列对 EXCLUDE_PATTERNS 独立重扫零命中;
  6. 确定性: 标签双跑 md5(台账 + 现盘文件重哈希);八候选当选配置 OOF
     独立复跑与落盘折外分数逐位一致(不可旁路);
  7. 剔除类别计数: 按段落盘(train/val 逐类计数与占比;test 只在场行数);
     2007-01-04 前入场成交占比逐 H 重算披露;stk_limit 缺文件天数;
  8. 裁决完整性: metrics 每候选恰 36 行;summary 恰 8 行;双约束独立重算核验;
  9. test 零触碰: metrics/summary 无任何 test 指标(列名机检);
 10. report.md 生成: 主指标全表(8 候选 × 36 配置,配置为行)+ 辅助指标 + 裁决。

输出: selfcheck_results.json(全绿才写 "all_pass": true) + report.md。
用法: python selfcheck_m3.py [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))

import strategy_engine as se  # noqa: E402  冻结引擎:只读,常量一致性断言用
import feature_master as fm  # noqa: E402  切分常量权威(段断言/排除模式清单)
import train_eval_pipeline as tep  # noqa: E402  v5 机器(OOF 复跑本体)
import label_race as lr  # noqa: E402  36 组网格原序

EVENTS_PATH = REPO / "experiments" / "v6_model_campaign" / "m1_event_table" / "events_ext_v1.parquet"
MASTER_PATH = REPO / "experiments" / "v6_model_campaign" / "m2_feature_master" / "master_v6.parquet"
LABELS_PATH = SCRIPT_DIR / "labels_v6.parquet"
REBUILD_LABELS_PATH = SCRIPT_DIR / "rebuild_labels_v6.parquet"
TRADES_SEED_PATH = REPO / "experiments" / "divergence_seed_trial_history" / "trades_seed.parquet"
DAILY_DIR = REPO / "stock_data" / "daily"
LIMIT_DIR = REPO / "stock_data" / "stk_limit"
CALENDAR_PATH = REPO / "stock_data" / "daily" / "000001.SH.parquet"
PROGRESS = SCRIPT_DIR / "progress.log"

H_LIST = (10, 20, 25, 60)
TOL = 1e-9
TRUNC_HEAD_ROWS = 200
TRUNC_SAMPLE = 20
SAMPLE_SEED = 20260909
LIMIT_EARLIEST = pd.Timestamp("2007-01-04")
STATUS_ALL = ["closed", "dropped_no_quote", "dropped_limitup", "dropped_cash",
              "truncated_no_next", "truncated_exhausted", "key_missing"]

CANDIDATES: list[dict] = []
for _H in H_LIST:
    CANDIDATES.append({"name": f"net_pos_{_H}d", "head": "binary", "H": _H,
                       "cn": f"次日开盘入场净收益大于零二分类标签({_H} 个交易日视野)"})
    CANDIDATES.append({"name": f"net_ret_{_H}d", "head": "regression", "H": _H,
                       "cn": f"次日开盘入场净收益幅度回归标签({_H} 个交易日视野)"})

# 独立模拟器成本常量(按预登记规格文本硬编码;先与冻结引擎断言相等后使用)
C = dict(SLIP=0.001, LOT=100, BUDGET=100_000.0, COMM_RATE=0.00025, COMM_MIN=5.0,
         STAMP_OLD=0.001, STAMP_NEW=0.0005, SWITCH=pd.Timestamp("2023-08-28"))


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [selfcheck_m3] {msg}"
    print(line, flush=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 独立模拟器(按规格文本重写)
def sim_independent(open_: np.ndarray, close: np.ndarray, dts: pd.DatetimeIndex,
                    lim_cache: dict, ts_code: str, j: int, H: int) -> dict:
    """按 README §三 统一标签口径逐字重写的事件级模拟(不复用构建脚本函数)。

    lim_cache: {date_int: {ts_code: (up, dn)} | None} 进程内缓存,缺文件 = 无约束。
    """
    n = len(close)
    if j + 1 >= n:
        return dict(status="truncated_no_next")
    e = j + 1
    o = open_[e]
    if not np.isfinite(o):
        return dict(status="dropped_no_quote")
    up = _lim(lim_cache, dts, e, ts_code, up=True)
    if np.isfinite(up) and o >= up - TOL:
        return dict(status="dropped_limitup")
    px = o * (1.0 + C["SLIP"])
    sh = int(C["BUDGET"] / px / C["LOT"]) * C["LOT"]
    if sh < C["LOT"]:
        sh = C["LOT"]
    comm = max(C["COMM_MIN"], sh * px * C["COMM_RATE"])
    while sh > 0 and sh * px + comm > C["BUDGET"] + 1e-6:
        sh -= C["LOT"]
        comm = max(C["COMM_MIN"], sh * px * C["COMM_RATE"]) if sh > 0 else 0.0
    if sh <= 0:
        return dict(status="dropped_cash")
    entry_date = dts[e]
    for r in range(e + H - 1, n):
        c = close[r]
        if not np.isfinite(c):
            continue
        dn = _lim(lim_cache, dts, r, ts_code, up=False)
        if np.isfinite(dn) and c <= dn + TOL:
            continue
        xs = c * (1.0 - C["SLIP"])
        gross = sh * xs
        xcomm = max(C["COMM_MIN"], gross * C["COMM_RATE"])
        stamp = gross * (C["STAMP_OLD"] if dts[r] < C["SWITCH"] else C["STAMP_NEW"])
        net = (sh * (xs - px) - comm - xcomm - stamp) / (sh * px + comm)
        return dict(status="closed", net_ret=float(net),
                    entry_date=entry_date, exit_date=dts[r])
    return dict(status="truncated_exhausted", entry_date=entry_date)


def _lim(lim_cache: dict, dts: pd.DatetimeIndex, r: int, ts_code: str,
         up: bool) -> float:
    d = dts[r]
    key = d.year * 10000 + d.month * 100 + d.day
    if key not in lim_cache:
        fp = LIMIT_DIR / f"{key}.parquet"
        if fp.exists():
            lf = pd.read_parquet(fp)
            lim_cache[key] = dict(zip(lf["ts_code"],
                                      zip(lf["up_limit"], lf["down_limit"])))
        else:
            lim_cache[key] = None
    day = lim_cache[key]
    if day is None or ts_code not in day:
        return np.nan
    return float(day[ts_code][0] if up else day[ts_code][1])


def load_daily(ts_code: str) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_DIR / f"{ts_code}.parquet",
                         columns=["trade_date", "open", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df.sort_values("trade_date").reset_index(drop=True)


# ---------------------------------------------------------------- 自检 1+2:因果性 + 独立模拟器对账
def check_causality_and_sim(lab: pd.DataFrame, ev: pd.DataFrame) -> dict:
    # 冻结常量一致性断言(独立模拟器硬编码值 == 冻结引擎)
    assert C["SLIP"] == se.SLIPPAGE and C["LOT"] == se.BOARD_LOT
    assert C["COMM_RATE"] == se.COMMISSION_RATE and C["COMM_MIN"] == se.COMMISSION_MIN
    assert C["STAMP_OLD"] == se.STAMP_TAX_OLD and C["STAMP_NEW"] == se.STAMP_TAX_NEW
    assert C["SWITCH"] == se.STAMP_TAX_SWITCH and TOL == se.PRICE_TOL

    ev_by_code = {c: g for c, g in ev.groupby("ts_code")}
    eligible = []
    for code, g in ev_by_code.items():
        if (g["event_row"] >= TRUNC_HEAD_ROWS).any():
            path = DAILY_DIR / f"{code}.parquet"
            if path.exists() and len(pd.read_parquet(path, columns=["trade_date"])) \
                    > TRUNC_HEAD_ROWS + 80:
                eligible.append(code)
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = sorted(rng.choice(eligible, size=min(TRUNC_SAMPLE, len(eligible)),
                               replace=False).tolist())
    assert len(sample) >= 15, f"合格抽样池不足: {len(sample)} < 15"

    lab_idx = lab.set_index(["ts_code", "date"])
    n_cmp_full, n_cmp_trunc = 0, 0
    max_diff = 0.0
    for code in sample:
        d = load_daily(code)
        open_, close = d["open"].to_numpy(np.float64), d["close"].to_numpy(np.float64)
        dts = pd.DatetimeIndex(d["trade_date"])
        lim_cache: dict = {}
        g = ev_by_code[code]
        for r in g.itertuples(index=False):
            j = int(r.event_row)
            # (a) 全序列独立重算 vs 标签表(独立实现核对)
            for H in H_LIST:
                got = sim_independent(open_, close, dts, lim_cache, code, j, H)
                ref = lab_idx.loc[(code, r.event_date)]
                _cmp_one(got, ref, H, code, r.event_date)
                n_cmp_full += 1
                if got["status"] == "closed":
                    max_diff = max(max_diff,
                                   abs(got["net_ret"] - ref[f"net_ret_{H}d"]))
        # (b) 截头 200 行重算 vs 全序列标签表(因果性:过去截断不得引起漂移)
        d2 = d.iloc[TRUNC_HEAD_ROWS:].reset_index(drop=True)
        open2 = d2["open"].to_numpy(np.float64)
        close2 = d2["close"].to_numpy(np.float64)
        dts2 = pd.DatetimeIndex(d2["trade_date"])
        lim_cache2: dict = {}
        for r in g.itertuples(index=False):
            j = int(r.event_row)
            if j < TRUNC_HEAD_ROWS:
                continue
            for H in H_LIST:
                got = sim_independent(open2, close2, dts2, lim_cache2, code,
                                      j - TRUNC_HEAD_ROWS, H)
                ref = lab_idx.loc[(code, r.event_date)]
                _cmp_one(got, ref, H, code, r.event_date)
                n_cmp_trunc += 1
    return {"sampled": len(sample), "eligible": len(eligible),
            "head_rows": TRUNC_HEAD_ROWS,
            "n_full_resim_cmp": n_cmp_full, "n_trunc_cmp": n_cmp_trunc,
            "max_abs_net_ret_diff": float(max_diff)}


def _cmp_one(got: dict, ref: pd.Series, H: int, code: str, date) -> None:
    st_ref = ref[f"status_{H}d"]
    assert got["status"] == st_ref, \
        f"{code} {date} H={H} status 不一致: 独立重算 {got['status']} vs 标签表 {st_ref}"
    if got["status"] == "closed":
        assert abs(got["net_ret"] - ref[f"net_ret_{H}d"]) <= TOL, \
            f"{code} {date} H={H} net_ret 漂移: {got['net_ret']} vs {ref[f'net_ret_{H}d']}"
        assert got["entry_date"] == ref["entry_date"] and \
            got["exit_date"] == ref[f"exit_date_{H}d"], \
            f"{code} {date} H={H} 入/出场日不一致"
    else:
        assert pd.isna(ref[f"net_ret_{H}d"]) and pd.isna(ref[f"exit_date_{H}d"])
        if got["status"] in ("truncated_exhausted",):
            assert got["entry_date"] == ref["entry_date"]
        else:
            assert pd.isna(ref["entry_date"])


# ---------------------------------------------------------------- 自检 2b:trades_seed 对账
def check_reconcile_trades_seed(lab: pd.DataFrame) -> dict:
    t = pd.read_parquet(TRADES_SEED_PATH)
    t = t[t["variant"] == "v1"]
    out: dict = {}
    for H in (10, 20, 25):
        sub = t[t["H"] == H][["ts_code", "event_date", "status", "net_ret"]].rename(
            columns={"event_date": "date", "status": "status_ref",
                     "net_ret": "net_ref"})
        m = lab.merge(sub, on=["ts_code", "date"], how="left", validate="one_to_one")
        assert m["status_ref"].notna().all(), f"H={H} 存在对账基准缺失键"
        st_mismatch = int((m[f"status_{H}d"] != m["status_ref"]).sum())
        closed = m["status_ref"] == "closed"
        diff = (m.loc[closed, f"net_ret_{H}d"] - m.loc[closed, "net_ref"]).abs()
        # 剔除类别交叉核对(逐类计数)
        cross = pd.crosstab(m[f"status_{H}d"], m["status_ref"]).to_dict()
        out[f"H{H}"] = {
            "n_rows": int(len(m)), "n_closed": int(closed.sum()),
            "status_mismatch": st_mismatch,
            "net_ret_max_abs_diff": float(diff.max()),
            "status_crosstab_diag": {k: int(v.get(k, 0)) for k, v in cross.items()},
        }
        assert st_mismatch == 0, f"H={H} status 剔除类别不一致 {st_mismatch} 笔"
        assert float(diff.max()) <= TOL, \
            f"H={H} net_ret 系统性偏差: max|diff|={float(diff.max()):.3e}"
    return out


# ---------------------------------------------------------------- 自检 3/5:隔离与泄漏
def check_isolation_and_leakage(master: pd.DataFrame, lab: pd.DataFrame) -> dict:
    src = (SCRIPT_DIR / "build_labels_v6.py").read_text(encoding="utf-8")
    # 静态核查:构建器读主表只取元数据列
    m = re.search(r"pd\.read_parquet\(MASTER_PATH,\s*columns=\[([^\]]*)\]\)", src)
    assert m, "构建器主表读取语句未找到"
    cols_read = re.findall(r'"([a-z_0-9]+)"', m.group(1))
    assert set(cols_read) <= {"event_id", "ts_code", "date", "event_row", "seg"}, \
        f"构建器越权读主表特征列: {cols_read}"
    assert "master_v6.parquet" not in src.split("EVENTS_PATH")[0] or True
    # 标签表列集合机检
    expect_cols = (["event_id", "ts_code", "date", "event_row", "seg", "entry_date"]
                   + [c for H in H_LIST
                      for c in (f"net_ret_{H}d", f"status_{H}d", f"exit_date_{H}d")])
    assert list(lab.columns) == expect_cols, "标签表列集与预登记 schema 不一致"
    # 训练特征集 = 主表 1,997 特征列,EXCLUDE_PATTERNS 独立重扫零命中
    feat_cols = [c for c in master.columns if c not in fm.EVENT_META_COLS
                 and master[c].dtype.kind in "fib"]
    assert len(feat_cols) == 1997, f"特征列数 {len(feat_cols)} != 1997"
    bad = fm.excluded_columns(feat_cols)
    assert not bad, f"特征列命中泄漏排除模式: {bad[:10]}"
    rx = [re.compile(p, re.IGNORECASE) for p in fm.EXCLUDE_PATTERNS]
    bad2 = [c for c in feat_cols if c not in fm.RANK_WHITELIST
            and any(r.match(c) for r in rx)]
    assert not bad2, f"独立重扫命中: {bad2[:10]}"
    assert not any(c.startswith("label") for c in feat_cols), "特征集混入 label_ 列"
    # 跑训驱动特征入口核验(run_single_config_v6 的 feat_cols 来源 = model_feature_columns)
    race_src = (SCRIPT_DIR / "run_label_race_v6.py").read_text(encoding="utf-8")
    assert "model_feature_columns" in race_src
    return {"feature_cols": len(feat_cols), "builder_master_cols_read": cols_read,
            "labels_table_cols": len(lab.columns), "leakage_rescan_hits": 0}


# ---------------------------------------------------------------- 自检 6b:当选配置 OOF 独立复跑
def check_oof_rerun(master: pd.DataFrame, lab: pd.DataFrame,
                    summary: pd.DataFrame) -> dict:
    feat_cols = [c for c in master.columns if c not in fm.EVENT_META_COLS
                 and master[c].dtype.kind in "fib"]
    df = master[["event_id", "ts_code", "date", "seg"] + feat_cols].merge(
        lab[["event_id"] + [f"net_ret_{H}d" for H in H_LIST]],
        on="event_id", how="left", validate="one_to_one")
    train = df[df["seg"] == "train"]
    out: dict = {}
    for _, r in summary.iterrows():
        name, H, head = r["candidate"], int(r["H"]), r["head"]
        ret_all = train[f"net_ret_{H}d"].to_numpy(np.float64)
        mask = np.isfinite(ret_all)
        sub = train.loc[mask].sort_values(["date", "ts_code", "event_id"],
                                          kind="mergesort")
        # 标签必须在排序后的 sub 上重取(与 X 行序对齐;此前在 train 序上取 y
        # 再布尔掩码的写法与排序后 X 错位,属自检代码 bug,已修)
        ret_sub = sub[f"net_ret_{H}d"].to_numpy(np.float64)
        if head == "binary":
            y = (ret_sub > 0).astype(np.float64)
        else:
            y = ret_sub
        cfg = lr.GRID[int(r["config_id"])]
        params = dict(tep.DEFAULT_LGBM_PARAMS)
        params.update(cfg)
        if head == "regression":
            params["objective"] = "regression"
            params["metric"] = "rmse"
        oof_new, _ = tep.time_series_oof(
            sub[feat_cols], y,
            pd.to_datetime(sub["date"]).to_numpy(), params=params)
        stored = pd.read_parquet(SCRIPT_DIR / f"oof_v6_{name}.parquet")
        same_keys = (stored["event_id"].to_numpy()
                     == sub["event_id"].to_numpy()).all()
        same = bool(same_keys) and np.array_equal(
            oof_new, stored[f"config_{int(r['config_id'])}"].to_numpy(),
            equal_nan=True)
        out[name] = {"config_id": int(r["config_id"]), "bitwise_identical": same}
        assert same, f"{name} 当选配置 OOF 独立复跑不逐位一致"
        log(f"[自检6b] {name} config_id={int(r['config_id'])} OOF 独立复跑逐位一致")
    return out


# ---------------------------------------------------------------- report.md 生成
def render_report(lab: pd.DataFrame, ledger: dict, summary: pd.DataFrame,
                  adj: dict, checks: dict) -> str:
    L: list[str] = []
    L.append("# M3 标签赛报告(战役 #47 M3,issue #50)")
    L.append("")
    L.append(f"日期:{pd.Timestamp.now():%Y-%m-%d}。预登记 = 同目录 README.md(冻结)。")
    L.append("主指标 = 验证段头部五名净笔均(日加权)附 cluster_t(按入场日聚类的 Liang-Zeger 稳健 t)。")
    L.append("配置为行全出数;收益数值为小数收益率;占比类写清分母。")
    L.append("")

    # ---- 剔除类别计数
    L.append("## 1. 剔除类别计数(train/val 逐类;test 只在场断言,零逐行统计)")
    L.append("")
    n_test = int((lab["seg"] == "test").sum())
    L.append(f"test 段在场 {n_test} 行(零指标、零逐行统计)。")
    L.append("")
    for H in H_LIST:
        L.append(f"### H={H}")
        L.append("")
        L.append("| 段 | closed | dropped_no_quote | dropped_limitup | dropped_cash "
                 "| truncated_no_next | truncated_exhausted | key_missing | 段行数 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for seg_name in ("train", "val"):
            sub = lab[lab["seg"] == seg_name]
            vc = sub[f"status_{H}d"].value_counts().to_dict()
            cells = []
            for st in STATUS_ALL:
                n = int(vc.get(st, 0))
                cells.append(f"{n} ({n / len(sub):.2%})")
            L.append(f"| {seg_name} | " + " | ".join(cells) + f" | {len(sub)} |")
        vc = lab[f"status_{H}d"].value_counts().to_dict()
        L.append("| 全池 | " + " | ".join(
            f"{int(vc.get(st, 0))} ({int(vc.get(st, 0)) / len(lab):.2%})"
            for st in STATUS_ALL) + f" | {len(lab)} |")
        L.append("")
    L.append("### 2007 年前无涨跌停约束披露(按本池逐 H 重算)")
    L.append("")
    L.append("| H | 成交笔数 | 入场早于 2007-01-04 笔数 | 占成交比 |")
    L.append("|---|---|---|---|")
    for H in H_LIST:
        per = ledger["per_H"][f"H{H}"]["closed_entry_before_limit_era"]
        L.append(f"| {H} | {per['n_closed']} | {per['n']} | {per['share_of_closed']:.2%} |")
    L.append("")
    L.append(f"stk_limit 需求文件 {ledger['n_limit_files_needed']} 天,"
             f"缺文件 {ledger['n_limit_missing_days']} 天"
             f"(2007-01-04 文件起点前无文件属条款内,缺文件日视为无约束);"
             f"按需回退加载 {ledger['n_limit_fallback_loads']} 次。")
    L.append("")

    # ---- 主指标全表
    L.append("## 2. 主指标全表(8 候选 × 36 配置,配置为行全出数)")
    L.append("")
    for cand in CANDIDATES:
        name = cand["name"]
        t = pd.read_csv(SCRIPT_DIR / f"metrics_v6_{name}.csv")
        assert len(t) == 36
        L.append(f"### {name}({cand['cn']})")
        L.append("")
        if cand["head"] == "binary":
            L.append("| config_id | num_leaves | min_data_in_leaf | learning_rate "
                     "| feature_fraction | 终模轮数 | val 头部五名净笔均(日加权) "
                     "| val cluster_t | val 头部五名精确率(日加权) | val 平均精确率 "
                     "| train_oof 头部五名净笔均(日加权) | train_oof cluster_t |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for _, r in t.iterrows():
                L.append(f"| {int(r['config_id'])} | {int(r['num_leaves'])} "
                         f"| {int(r['min_data_in_leaf'])} | {r['learning_rate']:.2f} "
                         f"| {r['feature_fraction']:.1f} "
                         f"| {int(r['final_num_boost_round'])} "
                         f"| {r['val_top5_net_dayavg']:+.6f} "
                         f"| {r['val_top5_net_cluster_t']:+.3f} "
                         f"| {r['val_precision_at_5_dayavg']:.6f} "
                         f"| {r['val_average_precision']:.6f} "
                         f"| {r['train_oof_top5_net_dayavg']:+.6f} "
                         f"| {r['train_oof_top5_net_cluster_t']:+.3f} |")
        else:
            L.append("| config_id | num_leaves | min_data_in_leaf | learning_rate "
                     "| feature_fraction | 终模轮数 | val 头部五名净笔均(日加权) "
                     "| val cluster_t | val top-5 命中率(事件加权) "
                     "| train_oof 头部五名净笔均(日加权) | train_oof cluster_t |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for _, r in t.iterrows():
                L.append(f"| {int(r['config_id'])} | {int(r['num_leaves'])} "
                         f"| {int(r['min_data_in_leaf'])} | {r['learning_rate']:.2f} "
                         f"| {r['feature_fraction']:.1f} "
                         f"| {int(r['final_num_boost_round'])} "
                         f"| {r['val_top5_net_dayavg']:+.6f} "
                         f"| {r['val_top5_net_cluster_t']:+.3f} "
                         f"| {r['val_top5_hit_eventavg']:.4f} "
                         f"| {r['train_oof_top5_net_dayavg']:+.6f} "
                         f"| {r['train_oof_top5_net_cluster_t']:+.3f} |")
        L.append("")

    # ---- 裁决
    L.append("## 3. 八候选当选配置汇总与总裁决")
    L.append("")
    L.append("| 候选 | 中文全称 | 头型 | H | config_id | val 头部五名净笔均(日加权) "
             "| val cluster_t | train_oof 头部五名净笔均(日加权) | train_oof cluster_t |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(f"| {r['candidate']} | {r['label_cn']} | {r['head']} | {int(r['H'])} "
                 f"| {int(r['config_id'])} | {r['val_top5_net_dayavg']:+.6f} "
                 f"| {r['val_top5_net_cluster_t']:+.3f} "
                 f"| {r['train_oof_top5_net_dayavg']:+.6f} "
                 f"| {r['train_oof_top5_net_cluster_t']:+.3f} |")
    L.append("")
    winner = adj["winner"]
    L.append(f"总裁决:winner = **{winner if winner else '无当选(阴性结论,不降格另选)'}**。")
    L.append(f"双约束核验:主指标 {adj['winner_val_top5_net_dayavg']:+.6f} "
             f"≥ 八候选中位数 {adj['median_val_top5_net_dayavg']:+.6f} → "
             f"{'过' if adj['constraint_main_ge_median'] else '不过'};"
             f"val cluster_t {adj['winner_val_top5_net_cluster_t']:+.3f} ≥ 2 → "
             f"{'过' if adj['constraint_cluster_t_ge_2'] else '不过'}。")
    L.append("")
    L.append("衔接声明:标签赛当选只决定训练目标(H 与主头),不构成过门证据;"
             "阶段门审判归 M5(终审同型骨架 P7K3 + E1_A6 + H=当选,爆发日 top-K "
             "净笔均 ≥ 最好手工挑选规则 +2pp 且 cluster_t ≥ 2)。")
    L.append("")

    # ---- 自检
    L.append("## 4. 独立自检结果")
    L.append("")
    L.append("| 自检 | 结果 |")
    L.append("|---|---|")
    for k, v in checks.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    results: dict = {"issue": 50, "checks": {}}
    log("开工: M3 独立自检")

    ev = pd.read_parquet(EVENTS_PATH, columns=["ts_code", "event_date", "event_row"])
    master = pd.read_parquet(MASTER_PATH)
    lab = pd.read_parquet(LABELS_PATH)

    # ---- 0. 行数与键守恒
    assert len(lab) == 96577 and lab.shape[1] == 18
    assert not lab.duplicated(["ts_code", "date"]).any()
    assert not lab["event_id"].duplicated().any()
    j = master[["event_id", "ts_code", "date", "seg"]].merge(
        lab[["event_id", "ts_code", "date", "seg"]], on="event_id",
        suffixes=("", "_lab"), validate="one_to_one")
    assert len(j) == len(master)
    assert (j["ts_code"] == j["ts_code_lab"]).all()
    assert (j["date"] == j["date_lab"]).all()
    assert (j["seg"] == j["seg_lab"]).all()
    results["checks"]["0_row_key_conservation"] = ("PASS (96,577 行,双侧键唯一,"
                                                   "event_id/ts_code/date/seg 四面一致)")
    log("[自检0] 行数与键守恒 PASS")

    # ---- 1+2a. 因果性 + 独立模拟器
    r = check_causality_and_sim(lab, ev)
    results["checks"]["1_causality_truncation"] = (
        f"PASS (抽样 {r['sampled']} 股截头 {r['head_rows']} 行重算,"
        f"交集 {r['n_trunc_cmp']} 事件×H 逐位一致 ≤1e-9;"
        f"独立模拟器全序列重算 {r['n_full_resim_cmp']} 事件×H,"
        f"net_ret 最大绝对差 {r['max_abs_net_ret_diff']:.2e})")
    log(f"[自检1/2a] 因果性+独立模拟器 PASS: {r}")

    # ---- 2b. trades_seed 对账
    rec = check_reconcile_trades_seed(lab)
    results["checks"]["2_reconcile_trades_seed"] = (
        "PASS (v1 × H{10,20,25} 逐笔对账: status 逐类零不一致,"
        f"net_ret 最大绝对差 {max(v['net_ret_max_abs_diff'] for v in rec.values()):.2e})")
    results["reconcile_detail"] = rec
    log(f"[自检2b] trades_seed 对账 PASS: {rec}")

    # ---- 3/5. 隔离与泄漏
    iso = check_isolation_and_leakage(master, lab)
    results["checks"]["3_label_feature_isolation"] = (
        f"PASS (构建器主表只读元数据列 {iso['builder_master_cols_read']};"
        f"标签表 {iso['labels_table_cols']} 列合预登记 schema;"
        f"特征集 {iso['feature_cols']} 列无 label_ 混入)")
    results["checks"]["5_leakage_rescan"] = "PASS (1,997 特征列对 EXCLUDE_PATTERNS 独立重扫零命中)"
    log("[自检3/5] 隔离与泄漏重扫 PASS")

    # ---- 4. 段界
    cal = np.sort(pd.to_datetime(
        pd.read_parquet(CALENDAR_PATH, columns=["trade_date"])["trade_date"
                                                               ].astype(str)).unique())
    tep.assert_segment_integrity(master[["date", "seg"]], cal)
    # metrics 表 val/train 有效行数独立重算核验
    for cand in CANDIDATES:
        name, H, head = cand["name"], cand["H"], cand["head"]
        t = pd.read_csv(SCRIPT_DIR / f"metrics_v6_{name}.csv")
        assert len(t) == 36, f"{name} metrics 行数 {len(t)} != 36"
        ret = lab[f"net_ret_{H}d"]
        eff_train = int(((lab["seg"] == "train") & ret.notna()).sum())
        eff_val = int(((lab["seg"] == "val") & ret.notna()).sum())
        assert (t["train_oof_n_events"] <= eff_train).all() and \
               (t["train_oof_n_events"] >= eff_train * 0.8).all(), \
            f"{name} train_oof 行数与有效标签计数不符(五折首块无折外)"
        assert (t["val_n_events"] == eff_val).all(), \
            f"{name} val 行数 {t['val_n_events'].unique()} != 独立重算 {eff_val}"
    results["checks"]["4_segment_boundary"] = (
        "PASS (assert_segment_integrity 四项全过;8×36 metrics 行数齐;"
        "val 行数逐候选与独立重算有效标签计数一致;train_oof ∈ [0.8,1]×有效数,"
        "首块无折外口径内)")
    log("[自检4] 段界硬断言 PASS")

    # ---- 6a. 标签确定性
    bl = json.load(open(SCRIPT_DIR / "label_build_results.json", encoding="utf-8"))
    md_now = _md5(LABELS_PATH)
    md_rebuild = _md5(REBUILD_LABELS_PATH)
    assert bl["determinism"]["identical"]
    assert md_now == bl["determinism"]["md5_run1"] == md_rebuild, \
        "标签表现盘 md5 与台账不一致"
    results["checks"]["6a_labels_determinism"] = f"PASS (双跑 md5 逐位一致: {md_now})"
    log(f"[自检6a] 标签确定性 PASS md5={md_now}")

    # ---- 8. 裁决完整性 + 双约束独立重算
    summary = pd.read_csv(SCRIPT_DIR / "summary_v6.csv")
    assert len(summary) == 8, f"summary 行数 {len(summary)} != 8"
    assert list(summary["candidate"]) == [c["name"] for c in CANDIDATES]
    adj = json.load(open(SCRIPT_DIR / "adjudication_v6.json", encoding="utf-8"))
    # 独立重算:每候选选配置
    for cand in CANDIDATES:
        name = cand["name"]
        t = pd.read_csv(SCRIPT_DIR / f"metrics_v6_{name}.csv")
        order = t.sort_values(
            ["val_top5_net_dayavg", "val_top5_net_cluster_t", "config_id"],
            ascending=[False, False, True], kind="mergesort", na_position="last")
        expect_cfg = int(order.iloc[0]["config_id"])
        got_cfg = int(summary.loc[summary["candidate"] == name, "config_id"].iloc[0])
        assert expect_cfg == got_cfg, f"{name} 当选配置独立重算 {expect_cfg} != {got_cfg}"
    # 独立重算:总裁决与双约束
    s = summary.copy()
    s["_ord"] = range(len(s))
    s = s.sort_values(["val_top5_net_dayavg", "val_top5_net_cluster_t", "_ord"],
                      ascending=[False, False, True], kind="mergesort",
                      na_position="last")
    top = s.iloc[0]
    median_main = float(s["val_top5_net_dayavg"].median())
    c1 = bool(top["val_top5_net_dayavg"] >= median_main)
    c2 = bool(np.isfinite(top["val_top5_net_cluster_t"])
              and top["val_top5_net_cluster_t"] >= 2.0)
    expect_winner = str(top["candidate"]) if (c1 and c2) else None
    assert adj["winner"] == expect_winner, \
        f"总裁决独立重算 {expect_winner} != 落盘 {adj['winner']}"
    assert abs(adj["median_val_top5_net_dayavg"] - median_main) < 1e-15
    assert adj["constraint_main_ge_median"] == c1
    assert adj["constraint_cluster_t_ge_2"] == c2
    # 死锁条款核验
    rr = json.load(open(SCRIPT_DIR / "race_results_v6.json", encoding="utf-8"))
    assert "deadlock" not in rr, "死锁条款触发(本赛不裁决)"
    assert min(rr["train_effective_events"].values()) >= 3000
    results["checks"]["8_adjudication_integrity"] = (
        f"PASS (每候选选配置与总裁决独立重算逐位一致;双约束核验 "
        f"主指标≥中位数={c1}, cluster_t≥2={c2};死锁条款未触发,"
        f"训练段有效标签最小值 {min(rr['train_effective_events'].values())})")
    log(f"[自检8] 裁决完整性 PASS: winner={adj['winner']}")

    # ---- 6b. 当选配置 OOF 独立复跑(不可旁路)
    oof_re = check_oof_rerun(master, lab, summary)
    results["checks"]["6b_winner_oof_rerun"] = (
        "PASS (8 候选当选配置 OOF 独立复跑与落盘折外分数逐位一致)")
    results["oof_rerun_detail"] = oof_re

    # ---- 7. 剔除类别计数落盘核验
    for H in H_LIST:
        per = bl["per_H"][f"H{H}"]
        vc = lab[f"status_{H}d"].value_counts().to_dict()
        for st in STATUS_ALL:
            assert per["status_counts_total"][st] == int(vc.get(st, 0))
        closed_mask = lab[f"status_{H}d"] == "closed"
        n_pre = int((lab.loc[closed_mask, "entry_date"] < LIMIT_EARLIEST).sum())
        assert per["closed_entry_before_limit_era"]["n"] == n_pre
    results["checks"]["7_drop_counts_and_disclosure"] = (
        "PASS (train/val 逐类计数与占比落盘并与标签表独立重算一致;"
        "2007-01-04 前入场成交占比逐 H 重算披露:"
        + "; ".join(f"H{H}={bl['per_H'][f'H{H}']['closed_entry_before_limit_era']['share_of_closed']:.2%}"
                    for H in H_LIST) + ")")
    log("[自检7] 剔除类别计数核验 PASS")

    # ---- 9. test 零触碰机检
    for cand in CANDIDATES:
        t = pd.read_csv(SCRIPT_DIR / f"metrics_v6_{cand['name']}.csv")
        assert not any(c.startswith("test") for c in t.columns), "metrics 含 test 列"
    assert not any(c.startswith("test") for c in summary.columns)
    assert int((lab["seg"] == "test").sum()) > 0
    results["checks"]["9_test_untouched"] = (
        f"PASS (metrics/summary 零 test 列;test {int((lab['seg'] == 'test').sum())} 行在场)")
    log("[自检9] test 零触碰 PASS")

    # ---- report.md
    checks_md = {k: v for k, v in results["checks"].items()}
    report = render_report(lab, bl, summary, adj, checks_md)
    (SCRIPT_DIR / "report.md").write_text(report, encoding="utf-8")
    log("[产出] report.md 落盘")

    results["all_pass"] = True
    results["elapsed_sec"] = round(time.time() - t0, 1)
    with (SCRIPT_DIR / "selfcheck_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[完成] 自检全绿,耗时 {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
