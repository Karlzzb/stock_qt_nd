#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2-on-v6 复刻线(issue #54) 步骤 6:生死门审判 —— 预登记 = 同目录 README.md §八(commit 49b9c5b,冻结)。

驱动 = 克隆 experiments/v6_model_campaign/m5_stage_gate/run_stage_gate_m5.py,
改动面白名单(README §八):scores 输入路径、输出目录、格 ID 前缀(v2on6)、
verdict/报告文件名、README 引用;**机制零改动**(底座 md5 断言、E1_A6、S1~S5、
P7、cluster_t、退化格对账、标签对账、双跑 detcmp 全部继承;#40 LH_RUNS
参考路径保持无前缀原值)。

底座 = #40 experiments/v6_long_hold_trial/run_long_hold.py 只读 import 复用
(本体零改动,启动时 md5 校验 == a8f1cca6d2bad167899454f896f6d026)。

网格(README §八,18 格):挑选规则 {M=v2-stack 分数排序} ∪ {S1~S5} × K ∈ {3,5,10},
池 = ALL(seg=='val' 24,405 事件,与 M5 同集逐键一致断言),H = 60,仓位 = P7K{K},
出场 = E1_A6(tp=0.25, sl=-0.18,断言守护)。
事件窗口 2020-01-01~2023-12-31(seg=='val' 子集);回测日历延伸至 2023-12-31 后第 90 个
上证指数交易日(2024-05-21,H=60 视野 + 顺延缓冲);硬断言全 18 格 truncated_window == 0。

主门(生死,README §八原文):任一 K 档 M 净笔均 ≥ max(S1~S5 同 K)+2pp 且 M cluster_t ≥ 2
→ 过线(推翻 #52 MODEL_ROUTE_DEAD,模型路线复活);全不过 → v2 路线与 M 路线互证判死。
副门(信息性):precision@top10%(val 有效标签 24,247 起全段 pooled,与 M5 同口径)。

自检(继承 M5 预登记 §四原样):#40 机制一致性对账(S1~S5 × K3 五格,交集逐笔市场机制
字段浮点差 0)、成交单 vs 标签同机制子集对账(容差 1e-9)、模型分数接入断言(键一对一/
val 全覆盖/标签 NaN 事件不得成交(dropped_cash 本金口径例外计数披露)/embargo/test
零开仓)、S4 格与全 18 格双跑逐位一致(detcmp_v2on6.log)、底座 md5 校验、引擎内置断言
原样继承。

用法:
    python3 run_gate_v2on6.py --mode full          # 加载 + 对账 + 18 格双跑 + 宣判 + 报告
    python3 run_gate_v2on6.py --mode emit --only <cid,cid> --emit-dir <dir>  # 自检用:单格重跑落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(REPO / "experiments" / "v6_long_hold_trial"))
sys.path.insert(0, str(REPO / "v3_pipeline" / "scripts"))

import strategy_engine as se  # noqa: E402  冻结引擎:只读复用成本原语(标签制股数重算用)
import run_long_hold as RLH  # noqa: E402  #40 底座:只读 import,本体零改动(README §三)

GATE_DIR = REPO / "experiments" / "v2_on_v6_rerun"            # 白名单:输出目录
SCORES_PATH = GATE_DIR / "cache" / "scores_v2on6_p7fix.parquet"  # 白名单:scores 输入路径(P7 修复链,README 修订记录三轮)
LABELS_PATH = REPO / "experiments" / "v6_model_campaign" / "m3_label_race" / "labels_v6.parquet"  # 标签对账口径继承 M5(同事件集)
LH_RUNS = REPO / "experiments" / "v6_long_hold_trial" / "runs"  # #40 冻结产物(只读;无前缀原值)
CKPT_PATH = GATE_DIR / "checkpoint_pass1.pkl"  # 断点续跑(*.pkl 按 .gitignore 不入库)
CID_PREFIX = "v2on6__"                            # 白名单:格 ID 前缀

BASE_MD5 = "a8f1cca6d2bad167899454f896f6d026"  # run_long_hold.py 冻结校验值(继承 M5 §四.5)

VAL_START = pd.Timestamp("2020-01-01")   # M5 适配:事件富化窗口(README §三.1;seg=='val' 超集)
VAL_END = pd.Timestamp("2023-12-31")
D_START_M5 = 20200101                    # M5 适配:日历起点
D_END_M5 = 20240521                      # M5 适配:日历终点 = 2023-12-31 后第 90 个上证指数交易日(README §二)

SELS6 = ["M", "S1", "S2", "S3", "S4", "S5"]  # M = 模型分数排序(README §一.4)
K_LIST_M5 = [3, 5, 10]
H_M5 = 60
EXIT_M5 = "E1_A6"
SEL_FULL_NAMES = {  # 中文全称(命名全称纪律;S1~S5 同 #40)
    "M": "M 模型分数排序", "S1": "S1 先到先得", "S2": "S2 前20日跌幅最深",
    "S3": "S3 种子强度", "S4": "S4 随机", "S5": "S5 已弹最少+量比最冷"}

_ORIG_SELECT = None  # 底座 select_chosen 原函数引用(M5 适配 §三.3:S1~S5 零重实现)


def log(msg: str) -> None:
    RLH.log(f"[run_gate_v2on6] {msg}")


# ---------------------------------------------------------------- M5 适配 §三.3:模型挑选规则
def v2on6_select_chosen(pg: dict, sig_idx: np.ndarray, n_free: int, rule: str,
                     rng: np.random.RandomState) -> np.ndarray:
    """M5 适配:rule=='M' 走模型分数排序(README §一.4);S1~S5 调用底座原函数(零重实现)。

    M 口径:score 降序;平局 ts_code 升序(code_int 由 ts_code 字典序唯一编码,等价),
    再平局 event_id 升序(防御;事件键 (ts_code, event_date) 唯一,前两级已完备);
    score NaN 排最后(val 段实测零 NaN,防御保留);取前 n_free 名,落选当日放弃不顺延。
    """
    if rule != "M":
        return _ORIG_SELECT(pg, sig_idx, n_free, rule, rng)
    if n_free <= 0 or len(sig_idx) == 0:
        return sig_idx[:0]
    sc = pg["score"][sig_idx]
    sc = np.where(np.isfinite(sc), sc, -np.inf)
    k = np.lexsort((pg["event_id"][sig_idx], pg["code_int"][sig_idx], -sc))
    return sig_idx[k][:n_free]


# ---------------------------------------------------------------- 网格(README §一.4)
def enumerate_grid_v2on6() -> list[dict]:
    ex = RLH.EXIT_BY_NAME[EXIT_M5]
    assert ex["family"] == "E1" and ex["tp"] == 0.25 and ex["sl"] == -0.18, \
        f"E1_A6 参数断言失败: {ex}(README §三.4:断言守护不接受手抄)"
    cells: list[dict] = []
    for sel in SELS6:
        for K in K_LIST_M5:
            cfg = RLH.make_cfg("ALL", H_M5, sel, f"P7K{K}", EXIT_M5)
            cfg["cell_id"] = CID_PREFIX + cfg["cell_id"]   # 白名单:格 ID 前缀
            cells.append(cfg)
    assert len(cells) == 18 and len({c["cell_id"] for c in cells}) == 18
    return cells


# ---------------------------------------------------------------- 加载与适配(README §三)
def load_all() -> None:
    RLH.set_stage("load: 事件富化 + 个股日线(窗口 2020-01-01~2023-12-31)")
    RLH.load_events_and_stocks()  # 底座逐字:富化 dd20/bounce/vol_ratio/atr_pct/entry_row/entry_date
    # ---- M5 适配 §三.2:事件集替换为 seg=='val' 键集 + 一对一合并 event_id/score ----
    sc = pd.read_parquet(SCORES_PATH)
    sv = sc[sc["seg"] == "val"].copy()
    assert len(sv) == 24405, f"scores val 行数 {len(sv)} != 24405"
    assert not sv["event_id"].duplicated().any(), "scores val event_id 重复"
    assert sv["score"].notna().all(), "scores val 存在 NaN(README §四.3:全覆盖断言)"
    sv["date_int"] = (sv["date"].dt.year * 10000 + sv["date"].dt.month * 100
                      + sv["date"].dt.day).astype(np.int32)
    ev = RLH._G["events"]
    ev_f = ev.merge(sv[["event_id", "ts_code", "date_int", "score"]],
                    left_on=["ts_code", "event_date"],
                    right_on=["ts_code", "date_int"],
                    how="inner", validate="one_to_one")
    assert len(ev_f) == 24405, f"val 键合并后 {len(ev_f)} != 24405(键不齐即停)"
    ev_f = ev_f.sort_values(["event_date", "ts_code"], kind="mergesort") \
               .reset_index(drop=True)
    RLH._G["events"] = ev_f
    log(f"[load] 事件集替换为 seg=='val': {len(ev_f)} 起(富化窗口内 "
        f"{len(ev)} 起过滤);score 非 NaN 全覆盖断言通过")
    RLH.load_index_calendar()
    RLH.set_stage("load: 涨跌停(延伸日历)")
    RLH.load_limits()
    RLH.set_stage("build: ALL 池与预计算")
    RLH.build_pools()
    pg = RLH._G["pools"]["ALL"]
    assert len(pg["df"]) == 24405
    # ---- M5 适配 §三.3:pg 增挂 score/event_id 数组(列原序取自 pg['df'])----
    pg["score"] = pg["df"]["score"].to_numpy(dtype=np.float64)
    pg["event_id"] = pg["df"]["event_id"].to_numpy(dtype=np.int64)
    # mkt_atr 不加载(README §三.6:仅 E2 使用,本网格结构性用不到)
    log(f"[load] ALL 池 {pg['n']} 事件,信号日 {len(pg['by_date'])} 天;"
        f"日历 {len(RLH._G['cal'])} 天 [{RLH._G['cal'][0]}..{RLH._G['cal'][-1]}]")


# ---------------------------------------------------------------- 成交单规范化(双跑/落盘统一口径)
def canon_trades(tr: pd.DataFrame) -> pd.DataFrame:
    """trades 规范化:日期列统一为 YYYYMMDD int64(内存 int 与 parquet datetime 双来源对齐)。"""
    if not len(tr):
        return tr.reset_index(drop=True)
    tr = tr.copy()
    for col in ("event_date", "entry_date", "exit_date"):
        v = tr[col]
        if pd.api.types.is_datetime64_any_dtype(v):
            tr[col] = (v.dt.year * 10000 + v.dt.month * 100 + v.dt.day).astype(np.int64)
        else:
            tr[col] = v.astype(np.int64)
    return tr.reset_index(drop=True)


def canon_equity(eq: pd.DataFrame) -> pd.DataFrame:
    eq = eq.copy()
    v = eq["date"]
    if pd.api.types.is_datetime64_any_dtype(v):
        eq["date"] = (v.dt.year * 10000 + v.dt.month * 100 + v.dt.day).astype(np.int64)
    else:
        eq["date"] = v.astype(np.int64)
    return eq.reset_index(drop=True)


def df_bitwise_eq(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy()):
            return False
    return True


# ---------------------------------------------------------------- 第一遍(断点续跑,逐格落盘)
def run_pass1(cells: list[dict]) -> dict:
    ckpt: dict = {}
    if CKPT_PATH.exists():
        with CKPT_PATH.open("rb") as f:
            ckpt = pickle.load(f)
        log(f"[pass1] 断点续跑:已有 {len(ckpt)} 格 checkpoint")
    todo = [c for c in cells if c["cell_id"] not in ckpt]
    RLH.set_stage("pass1: 18 格(断点续跑)", len(ckpt), len(cells))
    if todo:
        cfgs = [{**c, "write": True} for c in todo]
        with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
            for i, res in enumerate(pool.imap_unordered(RLH._cell_worker, cfgs,
                                                        chunksize=1)):
                cid = res["summary"]["cell_id"]
                ckpt[cid] = {"summary": res["summary"], "stats": res["stats"]}
                with CKPT_PATH.open("wb") as f:
                    pickle.dump(ckpt, f)
                RLH._HB["done"] = len(ckpt)
                log(f"[pass1] {cid} 完成 ({len(ckpt)}/{len(cells)})")
    assert len(ckpt) == len(cells)
    return ckpt


# ---------------------------------------------------------------- 第二遍(确定性全量重算,不落盘)
def run_pass2(cells: list[dict]) -> dict:
    RLH.set_stage("pass2: 18 格确定性重算", 0, len(cells))
    out: dict = {}
    cfgs = [{**c, "write": False} for c in cells]
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(RLH._cell_worker, cfgs,
                                                    chunksize=1)):
            out[res["summary"]["cell_id"]] = res
            RLH._HB["done"] = i + 1
    return out


# ---------------------------------------------------------------- 自检 §四.4:双跑逐位对拍
def detcmp_v2on6(cells: list[dict], ckpt: dict, pass2: dict) -> dict:
    RLH.set_stage("detcmp: 全 18 格双跑逐位对拍(含 S4 三档)", 0, len(cells))
    bad: list[str] = []
    lines: list[str] = []
    for i, c in enumerate(cells):
        cid = c["cell_id"]
        su1 = {k: v for k, v in ckpt[cid]["summary"].items() if k != "runtime_sec"}
        st1 = {k: v for k, v in ckpt[cid]["stats"].items() if k != "runtime_sec"}
        r2 = pass2[cid]
        su2 = {k: v for k, v in r2["summary"].items() if k != "runtime_sec"}
        st2 = {k: v for k, v in r2["stats"].items() if k != "runtime_sec"}
        su_ok = RLH._val_eq(su1, su2)
        st_ok = RLH._val_eq(st1, st2)
        tr1 = canon_trades(pd.read_parquet(RLH.RUNS_DIR / cid / "trades.parquet"))
        tr2 = canon_trades(r2["trades"])
        eq1 = canon_equity(pd.read_parquet(RLH.RUNS_DIR / cid / "equity_curve.parquet"))
        eq2 = canon_equity(r2["equity"])
        tr_ok = df_bitwise_eq(tr1, tr2)
        eq_ok = df_bitwise_eq(eq1, eq2)
        if not (su_ok and st_ok and tr_ok and eq_ok):
            bad.append(f"{cid} summary={su_ok} stats={st_ok} trades={tr_ok} equity={eq_ok}")
        lines.append(f"  {'OK' if su_ok and st_ok and tr_ok and eq_ok else 'MISMATCH'} {cid}")
        RLH._HB["done"] = i + 1
    ok = not bad
    with open(RLH.DETCMP_PATH, "w", encoding="utf-8") as f:
        f.write("v2-on-v6 复刻线生死门:全 18 格双跑逐位确定性对拍(继承 M5 预登记 §四.4 原样)\n")
        f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("口径: 两遍全量 18 格,逐格 summary+stats(除 runtime_sec)+trades+equity "
                "严格逐位一致(NaN 与 NaN 视为相等);S4 格(K 三档)同口径覆盖\n")
        f.write(f"结果: {'PASS 逐位一致' if ok else 'FAIL'}\n")
        f.write(f"不一致格数: {len(bad)}\n")
        f.write("\n".join(lines) + "\n")
    log(f"[detcmp] 18 格双跑逐位对拍(含 S4 三档),不一致 {len(bad)} 格"
        f" -> {'PASS' if ok else 'FAIL'}(落 detcmp_v2on6.log)")
    return dict(ok=bool(ok), n_cells=len(cells), n_mismatch=len(bad), bad=bad)


# ---------------------------------------------------------------- 自检 §四.1:#40 机制一致性对账
def reconcile_vs_40(val_keys: set) -> dict:
    """#40 机制一致性对账(README §四.1 + 修订记录 1 加固),两层:

    ① 退化全覆盖层(硬断言):本阶段退化格(P1×E1_A6×S1×H60,仓位上限=无穷、现金约束放开、
    每股预算 10 万,同 #32 退化对账格口径)在验证段窗口接入全部 val 事件;#40 五格
    (S1~S5 × P7K3 × E1_A6 × H60 × ALL)在验证段内的全部已成交单,逐事件必须在退化格
    成交单中在场且市场机制字段(entry/exit 日期与价格、exit_reason、held_rows、
    deferred_days)浮点差 0——每股机制与组合状态无关,故全覆盖可判;
    唯一豁免 = dropped_cash(退化格预算 10 万买不起一手而 #40 P7K3 名义 33.3 万可买,
    按笔重算 100 股成本 > 10 万 + 1e-9 验证),逐笔披露。
    ② 交集层(预登记原口径,披露):#40 五格验证段内成交单与本阶段同配置格成交单的交集
    逐笔对账同字段(窗口不同 ⇒ 组合状态不同 ⇒ 接入集合与股数不同属预期,不比 shares/net_pnl)。
    """
    RLH.set_stage("check-r40: #40 机制一致性对账(退化全覆盖 + 交集)")
    fields_f = ["entry_raw", "entry_price", "exit_raw_price", "exit_exec_price"]
    fields_i = ["held_rows", "deferred_days"]

    def mech_cmp(a: dict, b: dict, key) -> list[str]:
        mism: list[str] = []
        for fc in fields_f:
            if not float(a[fc]) == float(b[fc]):
                mism.append(f"{key} {fc}: #40={a[fc]!r} M5={b[fc]!r}")
        for fc in ("entry_date", "exit_date"):
            va, vb = a[fc], b[fc]
            va = int(va.year * 10000 + va.month * 100 + va.day) \
                if hasattr(va, "year") else int(va)
            vb = int(vb.year * 10000 + vb.month * 100 + vb.day) \
                if hasattr(vb, "year") else int(vb)
            if va != vb:
                mism.append(f"{key} {fc}: #40={va} M5={vb}")
        for fc in fields_i:
            if int(a[fc]) != int(b[fc]):
                mism.append(f"{key} {fc}: #40={a[fc]!r} M5={b[fc]!r}")
        if str(a["exit_reason"]) != str(b["exit_reason"]):
            mism.append(f"{key} exit_reason: #40={a['exit_reason']} M5={b['exit_reason']}")
        return mism

    # ---- 层①:退化全覆盖 ----
    cfg_deg = RLH.make_cfg("ALL", H_M5, "S1", "P1", EXIT_M5)
    cfg_deg["degenerate"] = True
    cfg_deg["cell_id"] = "DEGENERATE__V2ON6__P1_E1A6_S1_H60"
    res_deg = RLH.run_cell(cfg_deg)
    deg_tr = {(r["ts_code"], int(r["event_date"])): r
              for r in res_deg["trades"].to_dict("records")}
    deg_oc = res_deg["outcomes"]
    # 退化格成交单落盘(selfcheck_m5 独立复核从 parquet 重跑全覆盖对账用)
    deg_dir = RLH.RUNS_DIR / cfg_deg["cell_id"]
    deg_dir.mkdir(parents=True, exist_ok=True)
    canon_trades(res_deg["trades"]).to_parquet(deg_dir / "trades.parquet")
    log(f"[check-r40] 退化格跑完: entered={res_deg['stats']['entered']} "
        f"trades={res_deg['stats']['n_trades']}(已落盘 {deg_dir.name}/trades.parquet)")

    per_cell: dict = {}
    all_ok = True
    for s in ("S1", "S2", "S3", "S4", "S5"):
        cid_ref = f"ALL__H60__{s}__P7K3__E1_A6"          # #40 参考格(无前缀原值)
        cid = CID_PREFIX + cid_ref                        # 本线网格格(前缀)
        fp = LH_RUNS / cid_ref / "trades.parquet"
        assert fp.exists(), f"#40 冻结产物缺失: {fp}"
        ref = pd.read_parquet(fp)
        ref["ev_int"] = (ref["event_date"].dt.year * 10000
                         + ref["event_date"].dt.month * 100 + ref["event_date"].dt.day)
        ref = ref[[(c, int(d)) in val_keys
                   for c, d in zip(ref["ts_code"], ref["ev_int"])]]
        got = canon_trades(pd.read_parquet(RLH.RUNS_DIR / cid / "trades.parquet"))
        got_m = {(r["ts_code"], int(r["event_date"])): r
                 for r in got.to_dict("records")}
        mism_full: list[str] = []
        mism_inter: list[str] = []
        n_cash_exempt = 0
        n_inter = 0
        for a in ref.to_dict("records"):
            key = (a["ts_code"], int(a["ev_int"]))
            b = deg_tr.get(key)
            if b is None:
                oc = deg_oc.get(key, {})
                cost100 = 100 * float(a["entry_price"]) + se.buy_cost(
                    100, float(a["entry_price"]))
                if oc.get("status") == "dropped_cash" and cost100 > 100_000.0 + 1e-9:
                    n_cash_exempt += 1  # 预算口径差豁免(披露)
                else:
                    mism_full.append(f"{key} 退化格未成交且非 dropped_cash 豁免"
                                     f"(outcome={oc.get('status')})")
                continue
            mism_full.extend(mech_cmp(a, b, key))
            bm = got_m.get(key)
            if bm is not None:
                n_inter += 1
                mism_inter.extend(mech_cmp(a, bm, key))
        ok = not mism_full and not mism_inter
        all_ok = all_ok and ok
        per_cell[cid] = dict(
            n_ref_val=len(ref), n_full_covered=len(ref) - n_cash_exempt - len(mism_full),
            n_cash_exempt=n_cash_exempt, n_mismatch_full=len(mism_full),
            n_m5_grid=len(got), n_intersection=n_inter, n_mismatch_inter=len(mism_inter),
            mismatch_head=(mism_full + mism_inter)[:5], ok=bool(ok))
        log(f"[check-r40] {cid}: #40 验证段成交 {len(ref)} 笔,退化全覆盖对账不一致 "
            f"{len(mism_full)}(dropped_cash 豁免 {n_cash_exempt} 笔);"
            f"网格交集 {n_inter} 笔不一致 {len(mism_inter)} -> {'PASS' if ok else 'FAIL'}")
        for mline in (mism_full + mism_inter)[:5]:
            log(f"[check-r40-mismatch] {mline}")
    return dict(ok=bool(all_ok), cells=per_cell,
                note=("层①(硬断言):退化格(P1×E1_A6×S1×H60,上限无穷+现金放开+每股预算 10 万)"
                      "全覆盖对账 #40 五格验证段全部成交单,市场机制字段浮点差 0;"
                      "层②(披露):同配置网格交集对账(README §四.1 预登记原口径);"
                      "均不比 shares/net_pnl(组合状态口径差,预期内)"))


# ---------------------------------------------------------------- 自检 §四.2/§四.3:标签对账 + 分数接入
def reconcile_vs_labels(cells: list[dict], labels: pd.DataFrame, val_keys: set) -> dict:
    """每格成交单 vs labels_v6.parquet(README §四.2/§四.3 + 修订记录 2 修正)。

    - 入场侧(全部成交笔):entry_date 与标签 entry_date 逐位一致;
    - 同机制子集(逐笔判定,修订记录 2 扩展):成交笔满足 exit_reason=='horizon' 且
      held_rows==60 且 deferred_days==0 且标签 closed(引擎出场日 = 入场后第 60 个个股
      行情行 = 标签机制)者:按标签口径(10 万本金制股数)以该笔 entry_price/exit_exec_price
      独立重算净收益,与标签 net_ret_60d 容差 1e-9;且 exit_date 与标签 exit_date_60d
      逐位一致;其中引擎 shares == 标签制股数者(本金口径相同),引擎 net_ret 本身与标签
      容差 1e-9(预期逐位);
    - 标签 NaN 事件不得成交(修订记录 2 修正为按引擎自身剔除语义分层):
      入口侧同机制状态(dropped_limitup / dropped_no_quote / truncated_no_next)零成交
      硬断言;dropped_cash 例外 = 10 万本金买不起一手而 P7 名义本金可买(按笔重算验证),
      逐格计数披露;truncated_exhausted(标签纯 H 机制走不到出场 = 数据耗尽)下 E1_A6
      屏障可提前离场,成交合法,断言其 exit_reason ∈ {tp, sl, horizon},逐格计数披露;
    - embargo/test/train/pre2001 行零开仓:全部成交笔事件键 ∈ seg=='val' 键集。
    """
    RLH.set_stage("check-label: 成交单 vs 标签同机制对账(18 格)", 0, len(cells))
    lab_df = labels.copy()
    lab_df["date_int"] = (lab_df["date"].dt.year * 10000
                          + lab_df["date"].dt.month * 100
                          + lab_df["date"].dt.day).astype(np.int64)
    lab = lab_df.set_index(["ts_code", "date_int"], verify_integrity=True)

    def label_shares(px: float) -> tuple[int, float]:
        """标签制股数与买佣(BUDGET=10 万,build_labels_v6 逐字口径)独立重算。"""
        sh = int(100_000.0 / px / se.BOARD_LOT) * se.BOARD_LOT
        if sh < se.BOARD_LOT:
            sh = se.BOARD_LOT
        comm = se.buy_cost(sh, px)
        while sh > 0 and sh * px + comm > 100_000.0 + 1e-6:
            sh -= se.BOARD_LOT
            comm = se.buy_cost(sh, px) if sh > 0 else 0.0
        return int(sh), float(comm)

    per_cell: dict = {}
    all_ok = True
    for i, c in enumerate(cells):
        cid = c["cell_id"]
        tr = canon_trades(pd.read_parquet(RLH.RUNS_DIR / cid / "trades.parquet"))
        mism: list[str] = []
        n_subset = 0
        n_subset_same_shares = 0
        n_cash_exception = 0
        n_trunc_exhausted_traded = 0
        n_entry_checked = 0
        for r in tr.itertuples(index=False):
            key = (r.ts_code, int(r.event_date))
            if key not in val_keys:
                mism.append(f"{key} 事件不在 seg=='val' 键集(embargo/test 开仓)")
                continue
            lr = lab.loc[key]
            status = str(lr["status_60d"])
            # 标签 NaN 事件不得成交(§四.3,修订记录 2 分层)
            if status in ("dropped_limitup", "dropped_no_quote", "truncated_no_next"):
                mism.append(f"{key} 标签 status_60d={status}(入口侧同机制剔除)却成交")
            elif status == "dropped_cash":
                cost100 = 100 * r.entry_price + se.buy_cost(100, r.entry_price)
                if cost100 > 100_000.0 + 1e-6:
                    n_cash_exception += 1
                else:
                    mism.append(f"{key} 标签 dropped_cash 但 10 万本金可买一手"
                                f"(成本 {cost100:.4f})")
            elif status == "truncated_exhausted":
                if r.exit_reason in ("tp", "sl", "horizon"):
                    n_trunc_exhausted_traded += 1  # E1_A6 屏障提前离场,合法,披露
                else:
                    mism.append(f"{key} 标签 truncated_exhausted 且出场原因异常: "
                                f"{r.exit_reason}")
            # 入场侧逐位一致(全部成交笔)
            ed = lr["entry_date"]
            ed_int = int(ed.year * 10000 + ed.month * 100 + ed.day) if pd.notna(ed) else -1
            n_entry_checked += 1
            if ed_int != int(r.entry_date):
                mism.append(f"{key} entry_date: 标签 {ed_int} != 引擎 {int(r.entry_date)}")
                continue
            # 同机制子集(§四.2,修订记录 2 扩展:标签口径独立重算,全 K 覆盖)
            if (r.exit_reason == "horizon" and int(r.held_rows) == 60
                    and int(r.deferred_days) == 0 and status == "closed"):
                n_subset += 1
                xd = lr["exit_date_60d"]
                xd_int = int(xd.year * 10000 + xd.month * 100 + xd.day)
                if xd_int != int(r.exit_date):
                    mism.append(f"{key} exit_date: 标签 {xd_int} != 引擎 {int(r.exit_date)}")
                px = float(r.entry_price)
                xs = float(r.exit_exec_price)
                sh_l, comm_l = label_shares(px)
                xcomm_l, stamp_l = se.sell_costs(sh_l, xs,
                                                 pd.Timestamp(str(int(r.exit_date))))
                net_l = (sh_l * (xs - px) - comm_l - xcomm_l - stamp_l) / (sh_l * px + comm_l)
                if not abs(net_l - float(lr["net_ret_60d"])) <= 1e-9:
                    mism.append(f"{key} net_ret(标签口径重算): 标签 {lr['net_ret_60d']!r} "
                                f"!= 重算 {net_l!r}")
                if int(r.shares) == sh_l:
                    n_subset_same_shares += 1
                    if not abs(float(r.net_ret) - float(lr["net_ret_60d"])) <= 1e-9:
                        mism.append(f"{key} net_ret(同股数逐位): 标签 {lr['net_ret_60d']!r} "
                                    f"!= 引擎 {r.net_ret!r}")
        ok = not mism
        all_ok = all_ok and ok
        per_cell[cid] = dict(n_trades=len(tr), n_entry_checked=n_entry_checked,
                             n_subset=n_subset,
                             n_subset_same_shares=n_subset_same_shares,
                             n_cash_exception=n_cash_exception,
                             n_trunc_exhausted_traded=n_trunc_exhausted_traded,
                             n_mismatch=len(mism), mismatch_head=mism[:5], ok=bool(ok))
        log(f"[check-label] {cid}: 成交 {len(tr)} 笔,入场侧逐位 {n_entry_checked} 笔,"
            f"同机制子集 {n_subset} 笔(1e-9;其中同股数逐位 {n_subset_same_shares} 笔),"
            f"dropped_cash 本金例外 {n_cash_exception} 笔,"
            f"truncated_exhausted 屏障提前离场 {n_trunc_exhausted_traded} 笔,"
            f"不一致 {len(mism)} -> {'PASS' if ok else 'FAIL'}")
        for mline in mism[:5]:
            log(f"[check-label-mismatch] {mline}")
        RLH._HB["done"] = i + 1
    return dict(ok=bool(all_ok), cells=per_cell,
                note=("同机制子集 = horizon 出场且 held_rows==60 且 deferred_days==0 且标签 "
                      "closed:标签口径(10 万本金)独立重算净收益与标签容差 1e-9,全 K 覆盖;"
                      "其中引擎股数==标签制股数者引擎 net_ret 与标签容差 1e-9(预期逐位);"
                      "标签 NaN 分层:入口侧同机制状态零成交硬断言;dropped_cash 本金口径例外"
                      "与 truncated_exhausted 屏障提前离场逐格计数披露(README §四.2/§四.3,"
                      "修订记录 2)"))


# ---------------------------------------------------------------- 副门:precision@top10%(README §一.7)
def secondary_gate(scores_val: pd.DataFrame, labels: pd.DataFrame) -> dict:
    lab = labels[["event_id", "status_60d", "net_ret_60d"]]
    m = scores_val.merge(lab, on="event_id", how="left", validate="one_to_one")
    uni = m[m["status_60d"] == "closed"].copy()
    assert len(uni) == 24247, f"副门全集 {len(uni)} != 24247"
    uni = uni.sort_values(["score", "ts_code", "event_id"],
                          ascending=[False, True, True], kind="mergesort")
    n_top = int(np.ceil(0.10 * len(uni)))
    top = uni.head(n_top)
    precision = float((top["net_ret_60d"] > 0).mean())
    baseline = float((uni["net_ret_60d"] > 0).mean())
    log(f"[副门] precision@top10%(全段 pooled,前 {n_top}/{len(uni)})= {precision:.6f},"
        f"池基线 {baseline:.6f},提升 {(precision - baseline) * 100:+.2f}pp")
    return dict(universe=int(len(uni)), n_top=int(n_top),
                precision_top10=precision, pool_baseline=baseline,
                lift_pp=round((precision - baseline) * 100, 6),
                note=("口径(README §一.7):全段 pooled;排序全集 = val 段有效标签(closed)事件;"
                      "score 降序,平局 ts_code 升序、event_id 升序;前 ⌈10%⌉;"
                      "标签无效 158 起不参与(净收益无定义)"))


# ---------------------------------------------------------------- 主门宣判(README §一.6/§五)
def main_gate(metrics: dict) -> dict:
    per_k: dict = {}
    for K in K_LIST_M5:
        m_cid = f"{CID_PREFIX}ALL__H60__M__P7K{K}__E1_A6"
        m_avg = metrics[m_cid]["net_avg"]
        m_ct = metrics[m_cid]["cluster_t"]
        s_best_sel, s_best = None, -np.inf
        for s in ("S1", "S2", "S3", "S4", "S5"):
            v = metrics[f"{CID_PREFIX}ALL__H60__{s}__P7K{K}__E1_A6"]["net_avg"]
            if v > s_best:
                s_best, s_best_sel = v, s
        diff_pp = (m_avg - s_best) * 100.0
        passed = bool(np.isfinite(diff_pp) and diff_pp >= 2.0
                      and m_ct is not None and np.isfinite(m_ct) and m_ct >= 2.0)
        per_k[str(K)] = dict(
            model_cell=m_cid, model_net_avg=m_avg, model_cluster_t=m_ct,
            best_manual_sel=s_best_sel, best_manual_net_avg=s_best,
            diff_pp=round(diff_pp, 6), threshold_pp=2.0,
            cluster_t_ok=bool(m_ct is not None and np.isfinite(m_ct) and m_ct >= 2.0),
            diff_ok=bool(np.isfinite(diff_pp) and diff_pp >= 2.0), passed=passed)
        ct_txt = "—" if m_ct is None or not np.isfinite(m_ct) else f"{m_ct:.4f}"
        log(f"[主门] K={K}: M 净笔均 {m_avg * 100:+.4f}% vs 最好手工 "
            f"{s_best_sel} {s_best * 100:+.4f}%,差值 {diff_pp:+.4f}pp(阈值 ≥+2pp),"
            f"cluster_t {ct_txt}(阈值 ≥2) -> {'过线' if passed else '不过'}")
    overall = any(v["passed"] for v in per_k.values())
    log(f"[主门宣判] {'过线 —— 推翻 #52,模型路线复活' if overall else '三档全不过 —— v2 路线与 M 路线互证判死'}")
    return dict(per_k=per_k, passed=bool(overall),
                verdict=("V2_ROUTE_ALIVE" if overall else "V2_ROUTE_DEAD"),
                clause=("README §八原文:任一 K 档 M 净笔均 ≥ max(S1~S5 同 K)+2pp 且 M cluster_t ≥ 2 "
                        "→ 过线(推翻 #52 的 MODEL_ROUTE_DEAD,模型路线复活,承认此前管线裁决有误);"
                        "全不过 → v2 路线与 M 路线互证判死,「170 vs 10」归因于 v2 池污染与"
                        "宽松选择规则,回用户拍板"))


# ---------------------------------------------------------------- 汇总与报告
def build_metrics(cells: list[dict], ckpt: dict) -> dict:
    metrics: dict = {}
    for c in cells:
        cid = c["cell_id"]
        st = ckpt[cid]["stats"]
        tr = canon_trades(pd.read_parquet(RLH.RUNS_DIR / cid / "trades.parquet"))
        net = tr["net_ret"].to_numpy(dtype=float) if len(tr) else np.array([])
        metrics[cid] = dict(
            cell_id=cid, sel=c["sel"], K=int(c["k"]), n_trades=int(st["n_trades"]),
            entered=int(st["entered"]),
            net_avg=(float(net.mean()) if len(net) else None),
            win_rate=(float((net > 0).mean()) if len(net) else None),
            cluster_t=st["cluster_t"],
            capital_utilization=float(st["capital_utilization"]),
            annualized_cash=float(st["annualized_cash"]),
            excess_cash=float(st["excess_cash"]),
            sharpe_cash=st["sharpe_cash"],
            bench_annualized=float(st["bench_annualized"]),
            dropped_slot_full=int(st["dropped_slot_full"]),
            dropped_limitup=int(st["dropped_limitup"]),
            dropped_cash=int(st["dropped_cash"]),
            truncated_no_next=int(st["truncated_no_next"]),
            truncated_window=int(st["truncated_window"]),
            truncated_exhausted=int(st["truncated_exhausted"]),
            open_at_end=int(st["open_at_end"]),
            deferred_exits=int(st["deferred_exits"]),
            exits_tp=int(st["exits_tp"]), exits_sl=int(st["exits_sl"]),
            exits_horizon=int(st["exits_horizon"]),
            coverage=float(st["coverage"]),
            runtime_sec=float(st["runtime_sec"]))
    return metrics


def write_summary(metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame([metrics[cid] for cid in sorted(
        metrics, key=lambda c: (SELS6.index(metrics[c]["sel"]), metrics[c]["K"]))])
    df = df.drop(columns=["runtime_sec"])
    # 全精度落盘(默认 repr 最短往返,selfcheck 以字符串读入 + float() 精确解析逐位对拍)
    df.to_csv(GATE_DIR / "summary_gate_v2on6.csv", index=False)
    log(f"[dump] summary_gate_v2on6.csv {len(df)} 行(全精度)")
    return df


def write_report(metrics: dict, gate: dict, secondary: dict, checks: dict,
                 elapsed: float) -> None:
    L: list[str] = []
    L.append("# v2-on-v6 复刻线 生死门审判(issue #54)· 裁决报告")
    L.append("")
    L.append("范围:验证段(seg=='val',事件日 2020-02-21 ~ 2023-11-17,24,405 起事件,ALL 全池);"
             "骨架 = P7 精选加厚仓位 + E1_A6 出场(止盈 +0.25 / 止损 −0.18)+ H=60;"
             "网格 = 挑选规则 {M 模型分数排序, S1~S5} × K ∈ {3,5,10} 共 18 格。")
    L.append("预登记军令状 = 同目录 README.md §八(commit 49b9c5b,冻结);驱动 = M5 "
             "run_stage_gate_m5.py 克隆(改动面白名单:scores 路径/输出目录/格 ID 前缀/"
             "verdict 与报告文件名/README 引用,机制零改动);机器底座 = #40 "
             "run_long_hold.py 只读 import(本体零改动,md5 校验通过)。")
    L.append("本报告全部数字由脚本从产物重算生成;配置为行全出数。")
    L.append("")
    all_ok = all(v.get("ok", True) for v in checks.values())
    L.append(f"**自检总评:{'ALL PASS' if all_ok else 'HAS FAIL(如实交付)'}**")
    L.append("")
    # ---- 1. 自检 ----
    L.append("## 1. 自检结果(README §四)")
    L.append("")
    c40 = checks.get("check_r40", {})
    n_mis40 = sum(v.get("n_mismatch_full", 0) + v.get("n_mismatch_inter", 0)
                  for v in c40.get("cells", {}).values())
    n_ref40 = sum(v.get("n_ref_val", 0) for v in c40.get("cells", {}).values())
    n_ex40 = sum(v.get("n_cash_exempt", 0) for v in c40.get("cells", {}).values())
    L.append(f"1. 引擎口径回归(#40 机制一致性):退化格(P1×E1_A6×S1×H60,上限无穷+现金放开)"
             f"全覆盖对账 #40 五格(S1~S5 × K3)验证段全部成交单 {n_ref40} 笔,市场机制字段"
             f"(entry/exit 日期与价格、exit_reason、held_rows、deferred_days)浮点差 0,"
             f"不一致 {n_mis40} 笔(dropped_cash 预算口径豁免 {n_ex40} 笔,披露)"
             f" -> {'PASS' if c40.get('ok') else 'FAIL'};同配置网格交集对账与接入集合差"
             f"(窗口不同所致组合状态差,预期内)逐格披露于 verdict_gate_v2on6.json。")
    cl = checks.get("check_label", {})
    n_mis_l = sum(v["n_mismatch"] for v in cl.get("cells", {}).values())
    n_sub = sum(v["n_subset"] for v in cl.get("cells", {}).values())
    n_sub_same = sum(v.get("n_subset_same_shares", 0) for v in cl.get("cells", {}).values())
    n_exc = sum(v["n_cash_exception"] for v in cl.get("cells", {}).values())
    n_tex = sum(v.get("n_trunc_exhausted_traded", 0) for v in cl.get("cells", {}).values())
    L.append(f"2. 成交单对账(vs 标签 H=60 同机制,容差 1e-9):18 格同机制子集合计 {n_sub} 笔"
             f"(标签口径独立重算,全 K 覆盖;其中同股数逐位 {n_sub_same} 笔)逐笔对账,"
             f"不一致 {n_mis_l} 笔;入场侧 entry_date 全部成交笔逐位一致;"
             f"dropped_cash 本金口径例外合计 {n_exc} 笔、truncated_exhausted 屏障提前离场"
             f"合计 {n_tex} 笔(均披露,修订记录 2)"
             f" -> {'PASS' if cl.get('ok') else 'FAIL'}")
    L.append(f"3. 模型分数接入断言:scores 与事件键一对一、val 24,405 行 score 非 NaN 全覆盖、"
             f"标签 NaN 事件不得成交(例外仅 dropped_cash 本金口径,其余状态零成交硬断言)、"
             f"embargo/test/train/pre2001 行零开仓 -> "
             f"{'PASS' if checks.get('check_score', {}).get('ok') else 'FAIL'}")
    cd = checks.get("check_detcmp", {})
    L.append(f"4. 确定性:S4 格(K 三档)与全 18 格双跑逐位一致(summary+stats+trades+equity),"
             f"不一致 {cd.get('n_mismatch')} 格 -> {'PASS' if cd.get('ok') else 'FAIL'}"
             f"(落 detcmp_v2on6.log)")
    L.append(f"5. 底座完整性:run_long_hold.py md5 校验一致;引擎内置断言(资金守恒零容差、"
             f"因果、双口径对账)逐格生效 -> "
             f"{'PASS' if checks.get('check_base', {}).get('ok') else 'FAIL'}")
    L.append("")
    # ---- 2. 18 格全出数 ----
    L.append("## 2. 18 格全出数(格为行)")
    L.append("")
    L.append("| 格 | 挑选规则 | K | 成交笔数 | 净笔均 | cluster_t | 胜率 | 资金利用率 | "
             "净年化超额(纯现金) | sharpe_cash | 接入 | 槽满丢弃 | 现金丢弃 | 涨停丢弃 | "
             "截断持仓 | tp/sl/horizon 出场 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cid in sorted(metrics, key=lambda c: (SELS6.index(metrics[c]["sel"]),
                                              metrics[c]["K"])):
        m = metrics[cid]
        net_avg = "—" if m["net_avg"] is None else f"{m['net_avg'] * 100:+.4f}%"
        ct = "—" if m["cluster_t"] is None else f"{m['cluster_t']:+.4f}"
        wr = "—" if m["win_rate"] is None else f"{m['win_rate'] * 100:.2f}%笔净收益>0"
        sc = "—" if m["sharpe_cash"] is None or not np.isfinite(m["sharpe_cash"]) \
            else f"{m['sharpe_cash']:.3f}"
        L.append(f"| {cid} | {SEL_FULL_NAMES[m['sel']]} | {m['K']} | {m['n_trades']} | "
                 f"{net_avg} | {ct} | {wr} | {m['capital_utilization']:.4f} | "
                 f"{m['excess_cash'] * 100:+.4f}% | {sc} | {m['entered']} | "
                 f"{m['dropped_slot_full']} | {m['dropped_cash']} | {m['dropped_limitup']} | "
                 f"{m['truncated_window'] + m['truncated_exhausted']} | "
                 f"{m['exits_tp']}/{m['exits_sl']}/{m['exits_horizon']} |")
    L.append("")
    L.append("注:净笔均 = 全部成交笔 net_ret 算术均值;胜率按「覆盖率→门槛」方向渲染为"
             "「X%笔净收益>0」;净年化超额(纯现金)与 sharpe_cash 为信息性披露列"
             "(延伸日历全程口径,README §二),不入主门;截断持仓 = truncated_window(恒 0,"
             "硬断言)+ truncated_exhausted(个股数据先耗尽,披露)。")
    L.append("")
    # ---- 3. 主门宣判 ----
    L.append("## 3. 主门宣判(生死门,README §八原文执行)")
    L.append("")
    L.append(f"条款原文:「{gate['clause']}」。")
    L.append("")
    L.append("| K | M 模型分数排序净笔均 | 最好手工规则 | 最好手工净笔均 | 差值(pp) | "
             "≥+2pp | M 格 cluster_t | ≥2 | 宣判 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for K in K_LIST_M5:
        g = gate["per_k"][str(K)]
        mct = ("—" if g["model_cluster_t"] is None
               or not np.isfinite(g["model_cluster_t"])
               else f"{g['model_cluster_t']:+.4f}")
        L.append(f"| {K} | {g['model_net_avg'] * 100:+.4f}% | "
                 f"{SEL_FULL_NAMES[g['best_manual_sel']]} | "
                 f"{g['best_manual_net_avg'] * 100:+.4f}% | {g['diff_pp']:+.4f} | "
                 f"{'是' if g['diff_ok'] else '否'} | {mct} | "
                 f"{'是' if g['cluster_t_ok'] else '否'} | "
                 f"{'**过线**' if g['passed'] else '不过'} |")
    L.append("")
    if gate["passed"]:
        passed_ks = [k for k, v in gate["per_k"].items() if v["passed"]]
        L.append(f"**主门宣判:过线(K 档 {', '.join(passed_ks)})—— 推翻 #52 的 "
                 f"MODEL_ROUTE_DEAD,模型路线复活,承认此前管线裁决有误(README §八)。**")
    else:
        L.append("**主门宣判:三个 K 档全不过 —— v2 路线与 M 路线互证判死,「170 vs 10」归因于 "
                 "v2 池污染与宽松选择规则,如实落盘,回用户拍板(README §八判死条款)。**")
    L.append("")
    L.append("逐 K 档全对照(模型 vs S1~S5 五规则逐列,净笔均):")
    L.append("")
    L.append("| K | M 模型分数排序 | S1 先到先得 | S2 前20日跌幅最深 | S3 种子强度 | S4 随机 | "
             "S5 已弹最少+量比最冷 |")
    L.append("|---|---|---|---|---|---|---|")
    for K in K_LIST_M5:
        vals = [metrics[f"{CID_PREFIX}ALL__H60__{s}__P7K{K}__E1_A6"]["net_avg"] for s in SELS6]
        L.append(f"| {K} | " + " | ".join("—" if v is None else f"{v * 100:+.4f}%"
                                          for v in vals) + " |")
    L.append("")
    L.append("cluster_t 全对照(行 = 挑选规则,列 = K):")
    L.append("")
    L.append("| 挑选规则 | K=3 | K=5 | K=10 |")
    L.append("|---|---|---|---|")
    for s in SELS6:
        vals = [metrics[f"{CID_PREFIX}ALL__H60__{s}__P7K{K}__E1_A6"]["cluster_t"] for K in K_LIST_M5]
        L.append(f"| {SEL_FULL_NAMES[s]} | " + " | ".join(
            "—" if v is None else f"{v:+.4f}" for v in vals) + " |")
    L.append("")
    # ---- 4. 副门 ----
    L.append("## 4. 副门(信息性,不判死):precision@top10% 相对池基线")
    L.append("")
    L.append(f"- 口径:全段 pooled;排序全集 = val 段有效标签(closed)事件 "
             f"{secondary['universe']:,} 起,按模型分数降序取前 ⌈10%⌉ = {secondary['n_top']:,} 名。")
    L.append(f"- precision@top10%(头部 H=60 净收益>0 占比)= "
             f"{secondary['precision_top10'] * 100:.2f}%。")
    L.append(f"- 池基线(全集净收益>0 占比)= {secondary['pool_baseline'] * 100:.2f}%。")
    L.append(f"- 提升 = {secondary['lift_pp']:+.2f}pp。")
    L.append("")
    # ---- 5. 池价值与排序增益分开表述 ----
    L.append("## 5. 信号池价值与池内排序增益(分开表述,不混比)")
    L.append("")
    L.append(f"- 信号池价值(不涉排序):池基线精度 {secondary['pool_baseline'] * 100:.2f}%"
             f"(24,247 起有效标签事件 H=60 净收益>0 占比);事件级边缘见 #31(#40 实证只作机器来源,"
             f"不作本阶段输入)。")
    best_s_overall = max(
        (metrics[f"{CID_PREFIX}ALL__H60__{s}__P7K{K}__E1_A6"]["net_avg"], s, K)
        for s in ("S1", "S2", "S3", "S4", "S5") for K in K_LIST_M5
        if metrics[f"{CID_PREFIX}ALL__H60__{s}__P7K{K}__E1_A6"]["net_avg"] is not None)
    L.append(f"- 池内排序增益(本阶段主门主体):M 模型排序 vs 手工规则,逐 K 档差值见 §3;"
             f"手工规则全场最优 = {SEL_FULL_NAMES[best_s_overall[1]]} K={best_s_overall[2]} "
             f"净笔均 {best_s_overall[0] * 100:+.4f}%。")
    L.append("")
    # ---- 6. 披露 ----
    L.append("## 6. 披露(README §二/§四,出数后原样保留)")
    L.append("")
    L.append("1. 日历延伸:事件窗口 2020-01-01 ~ 2023-12-31(seg=='val' 子集),回测日历延伸至 "
             "2024-05-21(2023-12-31 后第 90 个上证指数交易日,H=60 视野 + 顺延缓冲);"
             "约 2023 年 11 月入场持仓的出场用了 2024-01 ~ 2024-05 行情,与 M3 标签构建口径一致;"
             "入场/排序决策只用 ≤ 信号日信息与 M4 一次性产出分数,入场侧零泄漏。")
    L.append("2. 资金利用率/净年化超额/sharpe_cash 按延伸日历全程计算,尾部约 90 个日历交易日"
             "为纯出场延伸段;三者为信息性披露列,不入主门(主门指标 = 净笔均与 cluster_t,"
             "均为成交笔口径,不受日历延伸影响)。")
    tot_trunc_ex = sum(m["truncated_exhausted"] for m in metrics.values())
    tot_trunc_win = sum(m["truncated_window"] for m in metrics.values())
    L.append(f"3. 截断:全部 18 格 truncated_window 合计 {tot_trunc_win}(硬断言恒 0);"
             f"truncated_exhausted 合计 {tot_trunc_ex}(个股数据先耗尽持仓,按末日已知价估值"
             f"留仓至日历终点,逐格计数,与 #40 窗口终点条款同型)。")
    L.append(f"4. dropped_cash 本金口径例外:标签制本金 10 万买不起一手(val 段仅 4 起候选)而 "
             f"P7K3/K5 名义本金(33.3/20 万)可买,引擎按 P7 口径成交;合计 "
             f"{sum(v['n_cash_exception'] for v in cl.get('cells', {}).values())} 笔,"
             f"逐格披露;truncated_exhausted(标签纯 H 机制数据耗尽走不到出场)事件下 E1_A6 "
             f"屏障提前离场成交合计 "
             f"{sum(v.get('n_trunc_exhausted_traded', 0) for v in cl.get('cells', {}).values())} 笔"
             f"(全部 tp/sl/horizon 出场,机制合法,修订记录 2),逐格披露;"
             f"入口侧同机制状态(dropped_limitup/dropped_no_quote/truncated_no_next)"
             f"零成交硬断言通过。")
    L.append("5. S4 随机规则种子 = 42(沿用 #32~#40 冻结值,README §一.4 预登记钉死)。")
    L.append("6. 副门标签无效事件 158 起(dropped_limitup 107 + truncated_exhausted 47 + "
             "dropped_cash 4,占 val 事件 0.65%)不参与 precision 计算(净收益无定义,README §一.7)。")
    L.append("7. #40 机制一致性对账的接入集合差(交集外两侧独有笔数)为窗口不同的预期结果,"
             "逐格披露于 verdict_gate_v2on6.json,不构成口径分歧。")
    L.append("")
    L.append(f"总耗时 {elapsed:.0f} 秒;产物 = summary_gate_v2on6.csv(18 格全精度)/ "
             f"verdict_gate_v2on6.json / detcmp_v2on6.log / runs/(每格 equity+trades+stats)。")
    with open(GATE_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log("[dump] report.md")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="v2-on-v6 复刻线生死门审判(预登记 README.md §八)")
    ap.add_argument("--mode", choices=["full", "emit"], default="full")
    ap.add_argument("--only", default="")      # emit 用:逗号分隔 cell_id
    ap.add_argument("--emit-dir", default="")  # emit 用:落盘目录
    args = ap.parse_args()

    t_all = time.time()
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    # ---- M5 适配 §三.1:模块常量覆盖(进程内,底座文件不动)----
    RLH.OUT_DIR = GATE_DIR
    RLH.RUNS_DIR = GATE_DIR / "runs_gate"
    RLH.LOG_PATH = GATE_DIR / "progress.log"
    RLH.DETCMP_PATH = GATE_DIR / "detcmp_v2on6.log"
    RLH.EVENT_START, RLH.EVENT_END = VAL_START, VAL_END
    RLH.D_START, RLH.D_END = D_START_M5, D_END_M5
    RLH.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 自检 §四.5:底座 md5 校验(先于一切)----
    h = hashlib.md5()
    with open(REPO / "experiments" / "v6_long_hold_trial" / "run_long_hold.py", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    checks: dict = {"check_base": dict(
        ok=bool(h.hexdigest() == BASE_MD5), md5=h.hexdigest(), expect=BASE_MD5,
        note="run_long_hold.py 本体零改动校验(README §四.5)")}
    if not checks["check_base"]["ok"]:
        log(f"CHECK-BASE FAIL: 底座 md5 {h.hexdigest()} != {BASE_MD5},停止")
        sys.exit(1)

    _HB = RLH._HB
    _HB["t0"] = time.time()
    import threading
    hb = threading.Thread(target=RLH._heartbeat, daemon=True)
    hb.start()
    log(f"V2ON6 GATE START | mode={args.mode} | 预登记=README.md §八(冻结) | "
        f"CPU {mp.cpu_count()} | 底座 md5 校验通过")

    # ---- M5 适配 §三.3:select_chosen 包装(S1~S5 调底座原函数)----
    global _ORIG_SELECT
    _ORIG_SELECT = RLH.select_chosen
    RLH.select_chosen = v2on6_select_chosen

    load_all()
    scores_val = pd.read_parquet(SCORES_PATH)
    scores_val = scores_val[scores_val["seg"] == "val"].reset_index(drop=True)
    labels = pd.read_parquet(LABELS_PATH)
    val_keys = set(zip(scores_val["ts_code"].to_numpy(),
                       (scores_val["date"].dt.year * 10000
                        + scores_val["date"].dt.month * 100
                        + scores_val["date"].dt.day).astype(int).to_numpy()))
    assert len(val_keys) == 24405
    checks["check_score"] = dict(
        ok=True, n_val_events=len(val_keys),
        n_val_score_non_nan=int(scores_val["score"].notna().sum()),
        note=("scores 与事件键一对一(加载合并 validate='one_to_one' 断言);val score 非 NaN "
              "全覆盖;embargo/test/train/pre2001 零开仓由 check-label 逐格断言(成交笔事件键 "
              "∈ seg=='val' 键集)"))

    cells = enumerate_grid_v2on6()

    if args.mode == "emit":
        only = [s for s in args.only.split(",") if s]
        want = [c for c in cells if c["cell_id"] in only]
        assert len(want) == len(only), f"emit 格名不对: {only}"
        edir = Path(args.emit_dir)
        edir.mkdir(parents=True, exist_ok=True)
        for c in want:
            res = RLH.run_cell({**c})
            tr = canon_trades(res["trades"])
            tr.to_parquet(edir / f"{c['cell_id']}__trades.parquet")
            log(f"[emit] {c['cell_id']} trades {len(tr)} 行 -> {edir}")
        _HB["stop"] = True
        log(f"V2ON6 EMIT DONE ({time.time() - t_all:.0f}s)")
        return

    # ---- 第一遍(断点续跑,逐格落盘)+ 第二遍(确定性重算)----
    ckpt = run_pass1(cells)
    pass2 = run_pass2(cells)
    checks["check_detcmp"] = detcmp_v2on6(cells, ckpt, pass2)
    if not checks["check_detcmp"]["ok"]:
        log("DETCMP FAIL —— 双跑不逐位一致,停工排查(README §四.4)")
        _HB["stop"] = True
        sys.exit(1)

    # ---- 截断硬断言(README §二:truncated_window 恒 0)----
    n_tw = {c["cell_id"]: int(ckpt[c["cell_id"]]["stats"]["truncated_window"])
            for c in cells}
    bad_tw = {k: v for k, v in n_tw.items() if v != 0}
    checks["check_truncation"] = dict(
        ok=not bad_tw, truncated_window_total=int(sum(n_tw.values())),
        truncated_exhausted_total=int(sum(
            int(ckpt[c["cell_id"]]["stats"]["truncated_exhausted"]) for c in cells)),
        bad_cells=bad_tw,
        note="README §二 硬断言:全 18 格 truncated_window == 0;违例则日历顺延至 +120 重跑并落修订记录")
    if bad_tw:
        log(f"TRUNCATION FAIL: {bad_tw} —— 按 README §二 应顺延日历重跑,停止待处置")
        _HB["stop"] = True
        sys.exit(1)

    # ---- 对账自检(§四.1/#40、§四.2/§四.3/标签与分数)----
    checks["check_r40"] = reconcile_vs_40(val_keys)
    checks["check_label"] = reconcile_vs_labels(cells, labels, val_keys)

    # ---- 指标、主门、副门 ----
    metrics = build_metrics(cells, ckpt)
    gate = main_gate(metrics)
    secondary = secondary_gate(scores_val, labels)

    summary = write_summary(metrics)
    all_ok = all(v.get("ok", True) for v in checks.values())
    verdict = dict(
        experiment="v2-on-v6 复刻线:生死门审判(v2 栈分数 vs 手工规则)",
        issue=54, part_of="v2 全链重做战役(#30 规格)",
        prereadme="README.md 先于任何跑数落盘(冻结)",
        scope=("验证段 seg=='val' 24,405 事件;ALL 池;P7K{3,5,10} + E1_A6 + H60;"
               "挑选 {M,S1~S5} 共 18 格"),
        main_gate=gate, secondary_gate=secondary,
        metrics=metrics, checks=checks, checks_all_pass=bool(all_ok),
        elapsed_sec=round(time.time() - t_all, 1),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(GATE_DIR / "verdict_gate_v2on6.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)
    log("[dump] verdict_gate_v2on6.json")
    write_report(metrics, gate, secondary, checks, time.time() - t_all)

    _HB["stop"] = True
    log(f"V2ON6 GATE DONE ({time.time() - t_all:.0f}s) "
        f"主门={'过线' if gate['passed'] else '判死'} 自检={'ALL PASS' if all_ok else 'HAS FAIL'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
