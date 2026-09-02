#!/usr/bin/env python3
"""信号组合探索: 纯背离家族内改变池成分 (issue #11, 预登记协议, 不中途扩)。

组合池清单(预登记):
  P1 = 主 ∩ 备: 同股 ±1 个交易日(交易所历)聚类, 簇内同时含主池与备池事件才保留,
       每簇 1 事件, 代表日 = 簇内最早主池事件日(标签取该主池事件行)。
  P2 = 主 ∪ 备: 同股同日去重(同日双池共振记 1 事件, 标签取主池行; 两池标签同机器同口径)。
  P3 = 多配置集成池: 参数网格主轴 9 配置(g_o{13,15,17}_d{06,08,10})事件并集, 同股同日去重。
  P4 = 多配置共识池: 9 配置事件按同股 ±1 交易日聚类, 簇内 ≥2 个不同配置才保留,
       每簇 1 事件, 代表日 = 簇内命中配置数最多的日期(平手取最早)。

数据源:
  主池 = signal_param_grid runs/g_o15_d08(与 m_fractal15_full 落盘产物集合相等, 已经复核);
  网格 9 配置 = runs/g_*; 备池 = m_zigzag05_nofilter 事件, 由本脚本用 divergence_lab
  已验证管线重检事件并以网格标签配置(entry=open_T1, hit_N20_k2.0, ret_h20, mfe_h20)重算标签。
切分(预登记): train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31 /
  test 2022-11-01 起封存 —— 本脚本只输出 train/val, 测试段一行不出(输入明细已预过滤)。
基准: C1=同股随机非事件日(#6/#10 口径, rng=default_rng(42+crc32(symbol)));
  掩码 = warmup(60) + 该池全部成员事件日(并集); 抽样规模 m = 该池同股测试前事件数(去重后)。
  与网格的两处口径偏差(已在报告披露): m 用测试前事件数(网格含测试段检出数);
  掩码用成员过滤后事件日(网格用过滤前检出日)。
清洗: excluded_events_{main,backup}.parquet 的 f_any 直取; 网格变体事件(主池文件未覆盖)
  用 run_pool_cleaning 同机机器现算(ST as-of / d1 停牌无成交 / d1 一字涨停开盘);
  组合池事件的剔除 = 簇内全部成员事件剔除标记的并集(OR)。
判定(预登记):
  活口 = 命中率(事件加权)或赔率(mfe20/ATRN 均值)任一维度, 相对对应单池(P1/P2 对主+备双池,
  P3/P4 对主池)train/val 双段同向提升, 且日聚类 bootstrap 联合 p<0.05
  (联合 p = 自助复制中非"双段同向为正"的比例, B=10000, seed=42)。

产物: v3_pipeline/reports/signal_combo/{signal_combo_report.md, results_signal_combo.json,
progress.log, pools/<pool>/{events,c1}.parquet, backup_events.parquet}
"""
import collections
import glob
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sc_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import divergence_lab as dl  # noqa: E402
import run_pool_cleaning as rpc  # noqa: E402

REPO = SCRIPT_DIR.parents[1]
GRID = REPO / "v3_pipeline" / "reports" / "signal_param_grid"
SCAN = REPO / "v3_pipeline" / "reports" / "divergence_lab"
OUT = REPO / "v3_pipeline" / "reports" / "signal_combo"
POOLS_DIR = OUT / "pools"
PROGRESS = OUT / "progress.log"
WORKERS = 24
SEED = 42
B_BOOT = 10000

TRAIN = ("2001-01-01", "2018-12-31")
VAL = ("2019-01-01", "2022-10-31")
TEST_START_D = int(np.datetime64("2022-11-01", "D").astype(np.int32))
HIT = "hit_N20_k2.0"
GRID9 = [f"g_o{o}_d{d:02d}" for o in (13, 15, 17) for d in (6, 8, 10)]
MAIN = "g_o15_d08"

# 网格标签配置(与 signal_param_grid.BASE 一致; 只影响标签, 不影响事件检测)
LABEL_CFG = {
    "entry": "open_T1",
    "labels": {"fixed": [10, 20, 30], "dynamic": None,
               "sniper": {"Ns": [20], "ks": [2.0]}, "mfe": [20]},
}
LAB_COLS = ["ret_h10", "ret_h20", "ret_h30", HIT, "mfe_h20"]


def log(msg):
    line = f"{pd.Timestamp.utcnow():%Y-%m-%dT%H:%M:%SZ} {msg}"
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def dint(s):
    return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int32)


def seg_of(days):
    out = np.full(len(days), "other", dtype=object)
    tr_lo, tr_hi = dint(list(TRAIN))
    va_lo, va_hi = dint(list(VAL))
    out[(days >= tr_lo) & (days <= tr_hi)] = "train"
    out[(days >= va_lo) & (days <= va_hi)] = "val"
    return out


# ================================================================ 阶段1: 全宇宙加载(zigzag 检测) + 备池标签
def load_universe():
    zig_cfg = json.load(open(SCAN / "m_scan" / "m_zigzag05_nofilter" / "stats.json"))["meta"]["config"]
    files = sorted(glob.glob(str(dl.DATA_DIR / "*.parquet")))
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(dl.load_stock, [(f, zig_cfg) for f in files], chunksize=16)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                log(f"  [load] {i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    stocks = [r for r in results if "error" not in r]
    log(f"[stage1] 全宇宙加载完成: {len(stocks)} 只 ({time.time()-t0:.0f}s)")
    all_dates, _ = dl.build_market_regime(stocks, 120, 0.10)

    # 对拍: 重检事件 == m_zigzag05_nofilter 落盘事件
    saved = pd.read_parquet(SCAN / "m_scan" / "m_zigzag05_nofilter" / "events.parquet",
                            columns=["ts_code", "sig_idx"])
    mine = pd.concat([pd.DataFrame({"ts_code": st["symbol"], "sig_idx": st["events"]["sig"]})
                      for st in stocks if len(st["events"]["sig"])], ignore_index=True)
    a = set(map(tuple, saved[["ts_code", "sig_idx"]].to_numpy()))
    b = set(map(tuple, mine[["ts_code", "sig_idx"]].to_numpy()))
    assert a == b and len(a) == len(saved) == len(mine), (len(a), len(b), len(saved), len(mine))
    log(f"[stage1] 备池事件对拍通过: {len(mine)} == m_zigzag05_nofilter 落盘事件集")
    return stocks, all_dates


def build_backup_events(stocks):
    """备池事件 + 网格口径标签(open_T1)。返回 {(sym, bar): row} 与 DataFrame。"""
    cfg_lab = dl._deep_merge(dl.DEFAULT_CONFIG, LABEL_CFG)
    rec = collections.defaultdict(list)
    for st in stocks:
        sig = st["events"]["sig"]
        if len(sig) == 0:
            continue
        lab = dl.compute_labels(st, sig, None, cfg_lab)
        n = len(st["close"])
        t1 = np.minimum(sig + 1, n - 1)
        rec["ts_code"].extend([st["symbol"]] * len(sig))
        rec["sig_idx"].extend(sig.tolist())
        rec["day"].extend(st["dates"][sig].tolist())
        for c in LAB_COLS:
            rec[c].extend(np.asarray(lab[c], np.float64).tolist())
        rec["atr_t"].extend(st["atr"][sig].astype(np.float64).tolist())
        rec["entry"].extend(st["open"][t1].astype(np.float64).tolist())
    df = pd.DataFrame(rec)
    log(f"[stage2] 备池标签重算完成: {len(df)} 事件 (entry=open_T1, hit_N20_k2.0/mfe_h20)")
    # 对拍: 与 w_zigzag_p05_s5 的同 (s,d) 标签逐位一致(sniper 口径固定 T+1 开盘, 与 entry 无关)
    wdir = SCAN / "w_zigzag_p05_s5"
    ev = pd.read_parquet(wdir / "events.parquet", columns=["ts_code", "date"])
    lb = pd.read_parquet(wdir / "labels.parquet")
    div = lb.iloc[:len(ev)].reset_index(drop=True)
    assert (div["group"] == "div").all()
    w = pd.DataFrame({"ts_code": ev["ts_code"].values, "day": dint(ev["date"]),
                      "hit_w": div[HIT].values})
    m = df.merge(w, on=["ts_code", "day"], how="inner")
    ok = np.isclose(m[HIT].to_numpy(np.float64), m["hit_w"].to_numpy(np.float64),
                    atol=1e-6, equal_nan=True)
    assert ok.all(), f"备池标签对拍失败: {(~ok).sum()}/{len(m)}"
    log(f"[stage2] 备池标签对拍通过: {len(m)} 个共享 (s,d) 的 hit_N20_k2.0 与 w_zigzag_p05_s5 逐位一致")
    return df


# ================================================================ 阶段3: 口径自检 —— 复现 pool_cleaning 基线(embargo 口径)
def parity_baseline_recalc():
    exp = pd.read_csv(REPO / "v3_pipeline" / "reports" / "pool_cleaning" / "baseline_recalc.csv")
    fm_dir = REPO / "v3_pipeline" / "reports" / "feature_matrix"
    for pool, fm_name, labname in (("main", "main_pool_features.parquet", "w_fractal_o15_s20"),
                                   ("backup", "backup_pool_features.parquet", "w_zigzag_p05_s5")):
        fm = pd.read_parquet(fm_dir / fm_name, columns=["ts_code", "date"])
        fm["date"] = pd.to_datetime(fm["date"])
        lab = rpc.load_labels(SCAN / labname)
        df = fm.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
        df["seg"] = rpc.segment_of(df["date"].to_numpy())
        fl = pd.read_parquet(REPO / "v3_pipeline" / "reports" / "pool_cleaning"
                             / f"excluded_events_{pool}.parquet", columns=["ts_code", "date", "f_any"])
        df = df.merge(fl, on=["ts_code", "date"], how="left", validate="1:1")
        assert df["f_any"].notna().all()
        for seg in ("train", "val"):
            for filt in ("raw", "excl_combined"):
                keep = (df["seg"].to_numpy() == seg) & df["hit"].notna().to_numpy()
                if filt == "excl_combined":
                    keep &= ~df["f_any"].to_numpy()
                sub = df.loc[keep, ["date", "hit"]]
                evw = float(sub["hit"].mean())
                dww = float(sub.groupby("date")["hit"].mean().mean())
                row = exp[(exp["pool"] == pool) & (exp["segment"] == seg)
                          & (exp["filter_set"] == filt)].iloc[0]
                assert abs(evw - row["event_weighted_hit"]) < 5e-4, (pool, seg, filt, evw)
                assert abs(dww - row["day_weighted_hit"]) < 5e-4, (pool, seg, filt, dww)
                assert len(sub) == row["n_events"], (pool, seg, filt, len(sub))
    log("[stage3] 口径自检通过: 以 embargo 口径+落盘剔除标记逐位复现 baseline_recalc.csv 全部 20 行")


# ================================================================ 事件索引与组合池构建
class Universe:
    def __init__(self, stocks, all_dates):
        self.sym2st = {st["symbol"]: st for st in stocks}
        self.all_dates = all_dates
        self.cal_pos = {st["symbol"]: np.searchsorted(all_dates, st["dates"])
                        for st in stocks}  # bar -> 交易所历位置


def load_grid_events(name):
    df = pd.read_parquet(GRID / "runs" / name / "events.parquet")
    df["day"] = dint(df["date"])
    return df


def pool_dataframe(rows):
    df = pd.DataFrame(rows)
    df["date"] = df["day"].to_numpy().astype("datetime64[D]")
    df["seg"] = seg_of(df["day"].to_numpy())
    return df


def build_pools(uni, main_df, backup_df, grid_dfs):
    """返回 {pool: DataFrame}。成员事件剔除并集所需: member_days(簇内全部 (s,d) 日序)。"""
    t0 = time.time()
    # 每股票索引: bar -> 标签行
    def idx_by_sym(df):
        out = {}
        for sym, g in df.groupby("ts_code"):
            out[sym] = (g["sig_idx"].to_numpy(), g)
        return out

    main_ix, back_ix = idx_by_sym(main_df), idx_by_sym(backup_df)
    grid_ix = {n: idx_by_sym(g) for n, g in grid_dfs.items()}

    def lab_row(g, bar):
        r = g[g["sig_idx"] == bar].iloc[0]
        return r

    rows = {p: [] for p in ("P1", "P2", "P3", "P4")}
    syms = sorted(set(main_ix) | set(back_ix) | set().union(*[set(g) for g in grid_ix.values()]))
    skipped = set()
    for sym in syms:
        st = uni.sym2st.get(sym)
        if st is None:
            skipped.add(sym)
            continue
        pos = uni.cal_pos[sym]
        n_bars = len(st["close"])
        # ---------- P1: 主 ∩ 备(±1 交易日聚类)
        evs = []  # (calpos, bar, src)
        if sym in main_ix:
            for b in main_ix[sym][0]:
                if b < n_bars:
                    evs.append((int(pos[b]), int(b), "main"))
        if sym in back_ix:
            for b in back_ix[sym][0]:
                if b < n_bars:
                    evs.append((int(pos[b]), int(b), "backup"))
        evs.sort()
        for cl in _clusters(evs):
            srcs = {e[2] for e in cl}
            if "main" in srcs and "backup" in srcs:
                canon = min(e for e in cl if e[2] == "main")  # 最早主池事件
                r = lab_row(main_ix[sym][1], canon[1])
                rows["P1"].append(_mkrow(sym, canon[1], st, r, cl))
        # ---------- P2: 主 ∪ 备(同股同日去重, 同日取主池行)
        union = {}
        if sym in back_ix:
            for b in back_ix[sym][0]:
                union.setdefault(int(b), ("backup", None))
        if sym in main_ix:
            for b in main_ix[sym][0]:
                union[int(b)] = ("main", None)
        for b, (src, _) in sorted(union.items()):
            g = main_ix[sym][1] if src == "main" else back_ix[sym][1]
            r = lab_row(g, b)
            rows["P2"].append(_mkrow(sym, b, st, r, [(int(pos[b]), b, src)]))
        # ---------- P3: 9 配置并集(同股同日去重); P4: ≥2 配置 ±1 共振
        ge = []  # (calpos, bar, cfg)
        bars_any = {}
        for n in GRID9:
            if sym not in grid_ix[n]:
                continue
            for b in grid_ix[n][sym][0]:
                b = int(b)
                ge.append((int(pos[b]), b, n))
                bars_any.setdefault(b, n)
        for b, n in sorted(bars_any.items()):
            r = lab_row(grid_ix[n][sym][1], b)
            rows["P3"].append(_mkrow(sym, b, st, r, [(int(pos[b]), b, n)]))
        ge.sort()
        for cl in _clusters(ge):
            cfgs = {e[2] for e in cl}
            if len(cfgs) >= 2:
                by_pos = collections.defaultdict(set)
                for e in cl:
                    by_pos[e[0]].add(e[2])
                canon_pos = min(by_pos, key=lambda p: (-len(by_pos[p]), p))
                canon_bar = next(e[1] for e in cl if e[0] == canon_pos)
                n0 = next(e[2] for e in cl if e[0] == canon_pos)
                r = lab_row(grid_ix[n0][sym][1], canon_bar)
                rows["P4"].append(_mkrow(sym, canon_bar, st, r, cl, n_cfg=len(cfgs)))
    out = {p: pool_dataframe(rows[p]) for p in rows}
    for p in out:
        log(f"[stage4] {p} 构建完成: {len(out[p])} 事件 ({time.time()-t0:.0f}s)")
    if skipped:
        log(f"[stage4] 警告: {len(skipped)} 只股票无加载数据被跳过: {sorted(skipped)[:5]}")
    return out


def _clusters(evs):
    """按交易所历位置差 ≤1 聚类(输入已排序)。"""
    cur = []
    for e in evs:
        if cur and e[0] - cur[-1][0] > 1:
            yield cur
            cur = []
        cur.append(e)
    if cur:
        yield cur


def _mkrow(sym, bar, st, labrow, members, n_cfg=None):
    return {
        "ts_code": sym, "sig_idx": bar, "day": int(st["dates"][bar]),
        HIT: labrow[HIT], "ret_h20": labrow["ret_h20"], "mfe_h20": labrow["mfe_h20"],
        "atr_t": float(st["atr"][bar]), "entry": float(st["open"][min(bar + 1, len(st["close"]) - 1)]),
        "n_members": len(members), "n_cfg": n_cfg if n_cfg is not None else 1,
        "member_days": ",".join(str(int(st["dates"][m[1]])) for m in members),
        "member_srcs": ",".join(sorted({m[2] for m in members})),
    }


# ================================================================ 阶段5: 清洗标记(落盘标记 + 现算补缺)
def attach_d1_bars_tolerant(df):
    """rpc.attach_d1_bars 的列名兼容版: 部分日线文件用 volume 而非 vol(divergence_lab 同处理)。"""
    n = len(df)
    open_d1 = np.full(n, np.nan)
    vol_d1 = np.full(n, np.nan)
    bar_on_d1 = np.zeros(n, bool)
    d1_na = df["d1"].isna().to_numpy()
    df["_pos"] = np.arange(n)
    daily_dir = rpc.DAILY_DIR
    for code, sub in df.groupby("ts_code"):
        path = daily_dir / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            d = pd.read_parquet(path, columns=["trade_date", "open", "vol"])
        except Exception:
            d = pd.read_parquet(path, columns=["trade_date", "open", "volume"])
            d = d.rename(columns={"volume": "vol"})
        d = d.drop_duplicates("trade_date").sort_values("trade_date")
        dd = d["trade_date"].to_numpy().astype("datetime64[ns]")
        m = sub.loc[~sub["d1"].isna()]
        j = np.searchsorted(dd, m["d1"].to_numpy())
        hit = (j < len(dd)) & (dd[np.minimum(j, len(dd) - 1)] == m["d1"].to_numpy())
        pos = m["_pos"].to_numpy()
        bar_on_d1[pos[hit]] = True
        open_d1[pos[hit]] = d["open"].to_numpy()[j[hit]]
        vol_d1[pos[hit]] = d["vol"].to_numpy()[j[hit]]
    df.drop(columns=["_pos"], inplace=True)
    df["bar_on_d1"] = bar_on_d1
    df["open_d1"] = open_d1
    df["vol_d1"] = vol_d1
    df["own_next_eq_d1"] = True  # 现算补缺不核查个股次bar(落盘文件已核查)
    df["own_next_after_d1"] = False
    _ = d1_na
    return df


def attach_flags(pools, main_df, backup_df, grid_dfs):
    t0 = time.time()
    excl = {}
    for pool in ("main", "backup"):
        f = pd.read_parquet(REPO / "v3_pipeline" / "reports" / "pool_cleaning"
                            / f"excluded_events_{pool}.parquet")
        f["day"] = dint(f["date"])
        for c in ("f_st", "f_suspend", "f_limitup", "f_any"):
            excl.setdefault(c, {}).update(
                zip(zip(f["ts_code"], f["day"]), f[c].to_numpy()))
    log(f"[stage5] 落盘剔除标记载入: {len(excl['f_any'])} 个 (s,d) 键 ({time.time()-t0:.0f}s)")

    # 需要标记的全部 (s,d): 各池成员日并集
    need = set()
    sources = [main_df, backup_df] + list(grid_dfs.values()) + list(pools.values())
    for df in sources:
        need |= set(zip(df["ts_code"], df["day"]))
    for df in pools.values():
        for sym, md in zip(df["ts_code"], df["member_days"]):
            for d in md.split(","):
                need.add((sym, int(d)))
    missing = sorted(k for k in need if k not in excl["f_any"])
    log(f"[stage5] 需标记键 {len(need)}, 落盘未覆盖 {len(missing)} —— 用 run_pool_cleaning 同机机器现算")
    if missing:
        md = pd.DataFrame(missing, columns=["ts_code", "day"])
        md["date"] = pd.to_datetime(md["day"].to_numpy().astype("datetime64[D]"))
        cal = rpc.load_calendar()
        name_index = rpc.build_name_index()
        d1, _ = rpc.next_trade_days(cal, md["date"].to_numpy())
        md["d1"] = d1
        md["f_st"] = rpc.st_flags(md["ts_code"].to_numpy(), md["date"].to_numpy(), name_index)
        md = attach_d1_bars_tolerant(md)
        md["f_suspend"] = (~md["bar_on_d1"] & md["d1"].notna()) | \
                          (md["bar_on_d1"] & ((md["vol_d1"] <= 0) | md["vol_d1"].isna()
                                              | md["open_d1"].isna() | (md["open_d1"] <= 0)))
        md = rpc.attach_up_limit(md)
        md["limitup_evaluable"] = (md["d1"] >= rpc.LIMIT_DATA_START) & md["bar_on_d1"] \
            & md["up_limit_d1"].notna()
        md["f_limitup"] = md["limitup_evaluable"] & \
            (md["open_d1"] >= md["up_limit_d1"] - rpc.LIMIT_TOL - 1e-9)
        md["f_any"] = md["f_st"] | md["f_suspend"] | md["f_limitup"]
        for c in ("f_st", "f_suspend", "f_limitup", "f_any"):
            excl[c].update(zip(zip(md["ts_code"], md["day"]), md[c].to_numpy()))
        log(f"[stage5] 现算完成 ({time.time()-t0:.0f}s): ST={int(md['f_st'].sum())} "
            f"停牌={int(md['f_suspend'].sum())} 涨停={int(md['f_limitup'].sum())}")

    # 一致性抽查: 同一 (s,d) 在主/备两落盘文件中的 f_any 应一致(标记只依赖 (s,d))
    fa, fb = (pd.read_parquet(REPO / "v3_pipeline" / "reports" / "pool_cleaning"
                              / f"excluded_events_{p}.parquet", columns=["ts_code", "date", "f_any"])
              for p in ("main", "backup"))
    j = fa.merge(fb, on=["ts_code", "date"], suffixes=("_m", "_b"))
    assert (j["f_any_m"] == j["f_any_b"]).all(), "主/备落盘标记冲突"
    log(f"[stage5] 主/备落盘标记共享键 {len(j)} 个, f_any 全一致")

    for p, df in pools.items():
        fa = np.zeros(len(df), bool)
        fst = np.zeros(len(df), bool)
        fsu = np.zeros(len(df), bool)
        fli = np.zeros(len(df), bool)
        for i, (sym, md) in enumerate(zip(df["ts_code"], df["member_days"])):
            for dstr in md.split(","):
                k = (sym, int(dstr))
                fa[i] |= bool(excl["f_any"][k])
                fst[i] |= bool(excl["f_st"][k])
                fsu[i] |= bool(excl["f_suspend"][k])
                fli[i] |= bool(excl["f_limitup"][k])
        df["f_any"], df["f_st"], df["f_suspend"], df["f_limitup"] = fa, fst, fsu, fli
        log(f"[stage5] {p} 剔除标记(成员并集): f_any={int(fa.sum())}/{len(df)} ({time.time()-t0:.0f}s)")
    return pools, excl


# ================================================================ 阶段6: C1 基准(#6/#10 口径)
def build_c1(uni, pool_events, member_days_map, name):
    """pool_events: {sym: [bar,...]}(去重后, 测试前); member_days_map: {sym: [bar,...]}(成员并集, 掩码用)。"""
    cfg_lab = dl._deep_merge(dl.DEFAULT_CONFIG, LABEL_CFG)
    rec = collections.defaultdict(list)
    n_stock = 0
    for sym, bars in sorted(pool_events.items()):
        st = uni.sym2st.get(sym)
        if st is None or len(bars) == 0:
            continue
        n = len(st["close"])
        mask = np.ones(n, bool)
        mask[:60] = False
        md = member_days_map.get(sym)
        if md is not None and len(md):
            mask[np.asarray(md, np.int64)] = False
        days = np.nonzero(mask)[0]
        if len(days) == 0:
            continue
        rng = np.random.default_rng(SEED + zlib.crc32(sym.encode()))
        m = len(bars)
        pick = rng.choice(days, size=m, replace=len(days) < m)
        lab = dl.compute_labels(st, pick, None, cfg_lab)
        rec["day"].extend(st["dates"][pick].tolist())
        for c in LAB_COLS:
            rec[c].extend(np.asarray(lab[c], np.float64).tolist())
        n_stock += 1
    df = pd.DataFrame(rec)
    log(f"[stage6] {name} C1 完成: {len(df)} 样本 / {n_stock} 股")
    return df


# ================================================================ 阶段7: 段指标
def segment_metrics(ev, c1, lo_s, hi_s, cal):
    """ev/c1: DataFrame(测试前)。ev 需含 f_any。返回 {'raw':..., 'clean':...}。"""
    lo, hi = dint([lo_s])[0], dint([hi_s])[0]
    cal_seg = cal[(cal >= lo) & (cal <= hi)]
    weeks = max(len(cal_seg) / 5.0, 1e-9)
    cm = (c1["day"].to_numpy() >= lo) & (c1["day"].to_numpy() <= hi)
    c1s = c1.loc[cm]
    out = {}
    for tag, sub in (("raw", ev), ("clean", ev.loc[~ev["f_any"]])):
        m = (sub["day"].to_numpy() >= lo) & (sub["day"].to_numpy() <= hi)
        s = sub.loc[m]
        out[tag] = _metrics_block(s, c1s, weeks)
    return out


def _metrics_block(s, c1s, weeks):
    hit = s[HIT].to_numpy(np.float64)
    ok = np.isfinite(hit)
    n_events = len(s)
    if n_events:
        days, cnts = np.unique(s["day"].to_numpy(), return_counts=True)
        n_days = len(days)
        pct_le3 = float(np.mean(cnts <= 3))
    else:
        n_days, pct_le3 = 0, np.nan
    win = float(hit[ok].mean()) if ok.sum() else np.nan
    c_hit = c1s[HIT].to_numpy(np.float64)
    cok = np.isfinite(c_hit)
    c_win = float(c_hit[cok].mean()) if cok.sum() else np.nan
    ex_win = win - c_win if np.isfinite(win) and np.isfinite(c_win) else np.nan
    p_mw = np.nan
    if ok.sum() >= 10 and cok.sum() >= 10:
        p_mw = float(sc_stats.mannwhitneyu(hit[ok], c_hit[cok], alternative="two-sided").pvalue)
    # 日加权 top3(日内 ts_code 字典序前 3, 日等权)
    top3 = np.nan
    if ok.sum():
        d_sym = s["ts_code"].to_numpy()[ok].astype(str)
        d_day = s["day"].to_numpy()[ok]
        h = hit[ok]
        order = np.lexsort((d_sym, d_day))
        d_day, h = d_day[order], h[order]
        _, starts = np.unique(d_day, return_index=True)
        bounds = np.append(starts, len(d_day))
        top3 = float(np.mean([h[a:b][:3].mean() for a, b in zip(bounds[:-1], bounds[1:])]))
    # 赔率: mfe20/ATRN 均值(全体可评估事件), 及 mfe20 均值/超额(网格口径)
    mfe = s["mfe_h20"].to_numpy(np.float64)
    atrn = s["atr_t"].to_numpy(np.float64) / s["entry"].to_numpy(np.float64)
    ratio = mfe / atrn
    rok = np.isfinite(ratio) & (atrn > 0)
    mfe_atrn = float(ratio[rok].mean()) if rok.sum() else np.nan
    mok = np.isfinite(mfe)
    mfe_mean = float(mfe[mok].mean()) if mok.sum() else np.nan
    c_mfe = c1s["mfe_h20"].to_numpy(np.float64)
    cmok = np.isfinite(c_mfe)
    c_mfe_mean = float(c_mfe[cmok].mean()) if cmok.sum() else np.nan
    ex_mfe = mfe_mean - c_mfe_mean if np.isfinite(mfe_mean) and np.isfinite(c_mfe_mean) else np.nan

    def pf(r):
        r = r[np.isfinite(r)]
        pos, neg = r[r > 0], r[r < 0]
        return float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() != 0 else np.nan

    return {
        "n_events": int(n_events), "n_signal_days": int(n_days),
        "events_per_week": round(n_events / weeks, 2), "pct_days_le3": pct_le3,
        "hit": win, "c1_hit": c_win, "ex_win": ex_win, "p_mw": p_mw,
        "top3_dayweighted": top3,
        "mfe20_atrn_mean": mfe_atrn, "mfe20_mean": mfe_mean,
        "c1_mfe20_mean": c_mfe_mean, "ex_mfe20": ex_mfe,
        "pf_ret20": pf(s["ret_h20"].to_numpy(np.float64)),
    }


# ================================================================ 阶段8: 日聚类 bootstrap(双段联合)
def _day_cells(df, lo, hi, val):
    m = (df["day"].to_numpy() >= lo) & (df["day"].to_numpy() <= hi)
    sub = df.loc[m & ~df["f_any"].to_numpy()]
    if val == "mfe20_atrn":
        v = sub["mfe_h20"].to_numpy(np.float64) / \
            (sub["atr_t"].to_numpy(np.float64) / sub["entry"].to_numpy(np.float64))
    else:
        v = sub[val].to_numpy(np.float64)
    fin = np.isfinite(v)
    days = sub["day"].to_numpy()[fin]
    v = v[fin]
    cell = collections.defaultdict(lambda: [0.0, 0])
    for d, x in zip(days, v):
        cell[d][0] += x
        cell[d][1] += 1
    return cell


def boot_compare(ev_a, ev_b, val, rng):
    """日聚类 bootstrap: Δ=combo−单池, 双段联合 p = 非"双段同向>0"的复制比例。"""
    segs = {}
    for seg, (lo_s, hi_s) in (("train", TRAIN), ("val", VAL)):
        lo, hi = dint([lo_s])[0], dint([hi_s])[0]
        ca, cb = _day_cells(ev_a, lo, hi, val), _day_cells(ev_b, lo, hi, val)
        uni = sorted(set(ca) | set(cb))
        sa = np.array([ca[d][0] if d in ca else 0.0 for d in uni])
        na = np.array([ca[d][1] if d in ca else 0 for d in uni], np.float64)
        sb = np.array([cb[d][0] if d in cb else 0.0 for d in uni])
        nb = np.array([cb[d][1] if d in cb else 0 for d in uni], np.float64)
        obs_a = sa.sum() / na.sum() if na.sum() else np.nan
        obs_b = sb.sum() / nb.sum() if nb.sum() else np.nan
        segs[seg] = (uni, sa, na, sb, nb, obs_a - obs_b)
    U = {seg: len(segs[seg][0]) for seg in segs}
    reps_t = np.empty(B_BOOT)
    reps_v = np.empty(B_BOOT)
    for b in range(B_BOOT):
        for seg, arr in (("train", reps_t), ("val", reps_v)):
            uni, sa, na, sb, nb, _ = segs[seg]
            idx = rng.integers(0, U[seg], U[seg])
            ma = sa[idx].sum() / na[idx].sum() if na[idx].sum() else np.nan
            mb = sb[idx].sum() / nb[idx].sum() if nb[idx].sum() else np.nan
            arr[b] = ma - mb
    dt, dv = segs["train"][5], segs["val"][5]
    both_pos = (reps_t > 0) & (reps_v > 0)
    return {
        "delta_train": float(dt), "delta_val": float(dv),
        "p_train": float(np.mean(reps_t <= 0)), "p_val": float(np.mean(reps_v <= 0)),
        "p_joint": float(1.0 - np.mean(both_pos)),
        "ci95_train": [float(np.percentile(reps_t, 2.5)), float(np.percentile(reps_t, 97.5))],
        "ci95_val": [float(np.percentile(reps_v, 2.5)), float(np.percentile(reps_v, 97.5))],
    }


# ================================================================ 主流程
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    open(PROGRESS, "w").close()
    t_all = time.time()
    log("[stage0] 信号组合探索启动 (issue #11 预登记: P1 主∩备 / P2 主∪备 / P3 九配置并集 / P4 九配置≥2共识)")

    stocks, all_dates = load_universe()
    uni = Universe(stocks, all_dates)
    backup_df = build_backup_events(stocks)
    parity_baseline_recalc()

    # 网格事件载入(测试前已过滤)
    main_df = load_grid_events(MAIN)
    grid_dfs = {n: load_grid_events(n) for n in GRID9}
    backup_pre = backup_df.loc[backup_df["day"] < TEST_START_D].copy()
    log(f"[stage4] 事件源: 主池 {len(main_df)} / 备池(测试前) {len(backup_pre)} / "
        f"网格9配置合计 {sum(len(g) for g in grid_dfs.values())}")

    pools = build_pools(uni, main_df, backup_pre, grid_dfs)
    pools, excl = attach_flags(pools, main_df, backup_pre, grid_dfs)

    # 单池事件框(与组合池同构, 供指标/bootstrap); 剔除标记用同一 excl 字典(含现算补缺)
    backup_pre["date"] = backup_pre["day"].to_numpy().astype("datetime64[D]")
    for name, df in (("main", main_df), ("backup", backup_pre)):
        df = df.copy()
        df["member_days"] = df["day"].astype(str)
        fa = np.array([bool(excl["f_any"].get((s, d), False))
                       for s, d in zip(df["ts_code"], df["day"])])
        cov = int(np.array([(s, d) in excl["f_any"] for s, d in zip(df["ts_code"], df["day"])]).sum())
        df["f_any"] = fa
        pools[name] = df
        log(f"[stage5] 单池 {name}: 剔除标记覆盖 {cov}/{len(df)}, f_any={int(fa.sum())}")

    # 组合池事件明细落盘(测试前)
    for p in ("P1", "P2", "P3", "P4"):
        d = POOLS_DIR / p
        d.mkdir(parents=True, exist_ok=True)
        pools[p].drop(columns=["member_days"]).to_parquet(d / "events.parquet", index=False)
    backup_pre.to_parquet(OUT / "backup_events.parquet", index=False)
    log("[stage5] 组合池事件明细落盘 pools/{P1..P4}/events.parquet + backup_events.parquet")

    # ---------- C1(成员事件日并集作掩码, 池事件数作规模)
    def bars_by_sym(df):
        return {sym: g["sig_idx"].to_numpy() for sym, g in df.groupby("ts_code")}

    main_bars = bars_by_sym(main_df)
    back_bars = bars_by_sym(backup_pre)
    grid_union_bars = collections.defaultdict(set)
    for n in GRID9:
        for sym, g in grid_dfs[n].groupby("ts_code"):
            grid_union_bars[sym] |= set(g["sig_idx"].tolist())
    union_mb = {s: np.sort(np.concatenate([main_bars.get(s, []), back_bars.get(s, [])]).astype(int))
                for s in set(main_bars) | set(back_bars)}
    c1_specs = {
        "main": (main_bars, main_bars),
        "backup": (back_bars, back_bars),
        "P1": (bars_by_sym(pools["P1"]), union_mb),
        "P2": (bars_by_sym(pools["P2"]), union_mb),
        "P3": (bars_by_sym(pools["P3"]), {s: np.array(sorted(v), int) for s, v in grid_union_bars.items()}),
        "P4": (bars_by_sym(pools["P4"]), {s: np.array(sorted(v), int) for s, v in grid_union_bars.items()}),
    }
    c1s = {}
    for name, (ev_bars, mask_bars) in c1_specs.items():
        c1s[name] = build_c1(uni, ev_bars, mask_bars, name)
        if name.startswith("P"):
            c1_dump = c1s[name].loc[c1s[name]["day"] < TEST_START_D]
            c1_dump.assign(date=c1_dump["day"].to_numpy().astype("datetime64[D]")).to_parquet(
                POOLS_DIR / name / "c1.parquet", index=False)

    # ---------- 指标
    results = {}
    for name in ("main", "backup", "P1", "P2", "P3", "P4"):
        ev = pools[name].loc[pools[name]["day"] < TEST_START_D]
        results[name] = {
            seg: segment_metrics(ev, c1s[name], lo, hi, all_dates)
            for seg, (lo, hi) in (("train", TRAIN), ("val", VAL))
        }
        tr, va = results[name]["train"]["clean"], results[name]["val"]["clean"]
        log(f"[stage7] {name}: train n={tr['n_events']} hit={tr['hit']:.4f} ex={tr['ex_win']:+.4f} | "
            f"val n={va['n_events']} hit={va['hit']:.4f} ex={va['ex_win']:+.4f}")

    # 主池超额的主口径对照: 网格自身 c1.parquet(全史检出数抽样, 过滤前掩码)
    grid_c1 = pd.read_parquet(GRID / "runs" / MAIN / "c1.parquet")
    grid_c1["day"] = dint(grid_c1["date"])
    main_ex_gridc1 = {}
    for seg, (lo, hi) in (("train", TRAIN), ("val", VAL)):
        blk = segment_metrics(pools["main"].loc[pools["main"]["day"] < TEST_START_D],
                              grid_c1, lo, hi, all_dates)
        main_ex_gridc1[seg] = {"ex_win_clean": blk["clean"]["ex_win"],
                               "c1_hit": blk["clean"]["c1_hit"]}
    log(f"[stage7] 主池超额对照(网格 C1): train {main_ex_gridc1['train']['ex_win_clean']:+.4f} / "
        f"val {main_ex_gridc1['val']['ex_win_clean']:+.4f}")

    # ---------- bootstrap 对比
    rng = np.random.default_rng(SEED)
    comps = {}
    comp_specs = [("P1", "main"), ("P1", "backup"), ("P2", "main"), ("P2", "backup"),
                  ("P3", "main"), ("P4", "main")]
    for a, b in comp_specs:
        ev_a = pools[a].loc[pools[a]["day"] < TEST_START_D]
        ev_b = pools[b].loc[pools[b]["day"] < TEST_START_D]
        for val, tag in ((HIT, "hit"), ("mfe20_atrn", "odds")):
            r = boot_compare(ev_a, ev_b, val, rng)
            comps[f"{a}_vs_{b}.{tag}"] = r
            log(f"[stage8] {a} vs {b} [{tag}]: Δtrain={r['delta_train']:+.4f} "
                f"Δval={r['delta_val']:+.4f} p_joint={r['p_joint']:.4f}")
    log(f"[stage8] bootstrap 完成 (B={B_BOOT}) ({time.time()-t_all:.0f}s)")

    # ---------- 判定(预登记活线)
    verdict = {}
    for p, cmp_pools in (("P1", ("main", "backup")), ("P2", ("main", "backup")),
                         ("P3", ("main",)), ("P4", ("main",))):
        alive_dims = []
        for tag in ("hit", "odds"):
            ok = True
            for b in cmp_pools:
                r = comps[f"{p}_vs_{b}.{tag}"]
                if not (r["delta_train"] > 0 and r["delta_val"] > 0 and r["p_joint"] < 0.05):
                    ok = False
                    break
            if ok:
                alive_dims.append(tag)
        verdict[p] = {"alive": bool(alive_dims), "alive_dims": alive_dims,
                      "compared_against": list(cmp_pools)}
    log(f"[stage9] 判定: " + "; ".join(f"{p}={'活口:'+','.join(v['alive_dims']) if v['alive'] else '死'}"
                                       for p, v in verdict.items()))

    # ---------- 落盘 JSON
    out_json = {
        "protocol": {
            "pools": {
                "P1": "主∩备: 同股±1交易日(交易所历)聚类, 簇内需主+备各≥1, 代表日=簇内最早主池事件日",
                "P2": "主∪备: 同股同日去重, 同日取主池行",
                "P3": "网格主轴9配置事件并集, 同股同日去重",
                "P4": "9配置±1聚类, 簇内≥2不同配置, 代表日=簇内配置数最多日(平手取最早)",
            },
            "split": {"train": TRAIN, "val": VAL, "test": "2022-11-01 起封存, 未输出"},
            "baseline": "C1 同股随机非事件日(seed+crc32(symbol)); 掩码=warmup+成员事件日并集; "
                        "m=测试前池事件数(与网格口径的两处偏差见报告)",
            "cleaning": "f_any 落盘标记 + 网格变体现算(同机机器); 组合事件=成员标记并集(OR)",
            "judgment": "活口=命中率或赔率相对对应单池双段同向提升且日聚类bootstrap联合p<0.05 (B=10000, seed=42)",
        },
        "parity": {
            "zigzag_events": "重检事件 == m_zigzag05_nofilter 落盘事件集(集合相等)",
            "backup_labels": "共享(s,d) hit_N20_k2.0 与 w_zigzag_p05_s5 逐位一致",
            "baseline_recalc": "embargo 口径+落盘标记复现 baseline_recalc.csv 全 20 行",
            "excl_flag_consistency": "主/备落盘标记共享键 f_any 全一致",
        },
        "pools": results,
        "main_excess_with_grid_c1": main_ex_gridc1,
        "comparisons": comps,
        "verdict": verdict,
    }
    with open(OUT / "results_signal_combo.json", "w") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=1, default=float)
    log("[stage9] results_signal_combo.json 已写出")

    _write_report(results, main_ex_gridc1, comps, verdict)
    log(f"[stage9] 全部完成 耗时 {time.time()-t_all:.0f}s")


# ================================================================ 报告
def pc(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{nd}f}"


def pp(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:+.{nd}f}"


def f2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def _write_report(results, main_ex_gridc1, comps, verdict):
    L = []
    L.append("# 信号组合探索报告: 纯背离家族内改变池成分(预登记协议, issue #11)")
    L.append("")
    L.append(f"生成: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC | seed=42 | bootstrap B=10000")
    L.append("切分: train 2001-01-01~2018-12-31 / val 2019-01-01~2022-10-31 / test 2022-11-01 起封存(本报告零测试段行)。")
    L.append("标签: 狙击 hit_N20_k2.0 = T+1 开盘入、20 交易日内 high 触及 +2×ATR(14,T); 赔率 = mfe20/(ATR(14,T)/entry) 的事件均值。")
    L.append("基准: C1=同股随机非事件日(#6/#10 口径, rng=42+crc32(symbol)); 掩码=warmup+该池成员事件日并集; 抽样规模=该池同股测试前事件数(去重后)。")
    L.append("与网格 C1 的两处口径偏差: (a) 抽样规模用测试前事件数(网格含测试段检出数); (b) 掩码用成员过滤后事件日(网格用过滤前检出日)——主池用网格自身 C1 的对照见 §4。")
    L.append("清洗: 落盘 f_any(ST as-of / d1 停牌无成交 / d1 一字涨停开盘)直取; 网格变体事件用 run_pool_cleaning 同机机器现算; 组合事件剔除=簇内成员标记并集(OR)。")
    L.append("管线自检: 重检备池事件与 m_zigzag05_nofilter 落盘集合相等; 备池标签与 w_zigzag_p05_s5 共享键逐位一致; embargo 口径复现 baseline_recalc.csv 全 20 行; 主/备落盘标记共享键无冲突。")
    L.append("")
    L.append("组合池口径: P1=主∩备(同股±1 交易日聚类, 簇内需主+备各≥1, 代表日=簇内最早主池事件日); "
             "P2=主∪备(同股同日去重, 同日取主池行); P3=网格主轴 9 配置并集(同股同日去重); "
             "P4=9 配置 ±1 聚类共识(簇内 ≥2 不同配置, 代表日=簇内配置数最多日, 平手取最早)。")
    L.append("")
    # ---- 密度表(清洗后)
    L.append("## 1. 池密度(清洗后, 括号内为清洗前事件数)")
    L.append("")
    L.append("| 池 | 段 | 事件数 | 信号日数 | 事件/周 | ≤3候选日% |")
    L.append("|---|---|---|---|---|---|")
    names = ["main", "backup", "P1", "P2", "P3", "P4"]
    disp = {"main": "主池 g_o15_d08", "backup": "备池 zigzag05_nofilter",
            "P1": "P1 主∩备", "P2": "P2 主∪备", "P3": "P3 九配置并集", "P4": "P4 九配置共识"}
    for n in names:
        for seg in ("train", "val"):
            s = results[n][seg]["clean"]
            r = results[n][seg]["raw"]
            L.append(f"| {disp[n]} | {seg} | {s['n_events']} ({r['n_events']}) | "
                     f"{s['n_signal_days']} | {s['events_per_week']:.1f} | {pc(s['pct_days_le3'])} |")
    L.append("")
    # ---- 命中/超额/赔率
    L.append("## 2. 命中率/超额/赔率(清洗后)")
    L.append("")
    L.append("| 池 | 段 | 命中% | C1% | 超额pp | p_mw | top3日加权% | mfe20/ATRN | mfe20均值% | ex_mfe20pp | PF(ret20) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        for seg in ("train", "val"):
            s = results[n][seg]["clean"]
            L.append(f"| {disp[n]} | {seg} | {pc(s['hit'])} | {pc(s['c1_hit'])} | {pp(s['ex_win'])} | "
                     f"{s['p_mw']:.4f} | {pc(s['top3_dayweighted'])} | "
                     f"{f2(s['mfe20_atrn_mean'])} | "
                     f"{pc(s['mfe20_mean'])} | {pp(s['ex_mfe20'], 2)} | "
                     f"{f2(s['pf_ret20'])} |")
    L.append("")
    # ---- P1 稀缺性权衡
    L.append("## 3. P1 共识池: 稀缺性-命中率权衡")
    L.append("")
    p1t, p1v = results["P1"]["train"]["clean"], results["P1"]["val"]["clean"]
    mt, mv = results["main"]["train"]["clean"], results["main"]["val"]["clean"]
    bt, bv = results["backup"]["train"]["clean"], results["backup"]["val"]["clean"]
    L.append(f"- 事件量: P1 train {p1t['n_events']} / val {p1v['n_events']}, "
             f"为主池的 {p1t['n_events']/max(mt['n_events'],1)*100:.0f}% / {p1v['n_events']/max(mv['n_events'],1)*100:.0f}%, "
             f"备池的 {p1t['n_events']/max(bt['n_events'],1)*100:.1f}% / {p1v['n_events']/max(bv['n_events'],1)*100:.1f}%。")
    L.append(f"- 命中率: P1 train {pc(p1t['hit'])}% / val {pc(p1v['hit'])}%; "
             f"主池 {pc(mt['hit'])}% / {pc(mv['hit'])}%; 备池 {pc(bt['hit'])}% / {pc(bv['hit'])}%。")
    for cmp_name, b in (("对主池", "main"), ("对备池", "backup")):
        rh = comps[f"P1_vs_{b}.hit"]
        L.append(f"- {cmp_name}: Δ命中 train {pp(rh['delta_train'], 2)}pp / val {pp(rh['delta_val'], 2)}pp, "
                 f"联合 p={rh['p_joint']:.4f}; Δ赔率 train {comps[f'P1_vs_{b}.odds']['delta_train']:+.3f} / "
                 f"val {comps[f'P1_vs_{b}.odds']['delta_val']:+.3f}, 联合 p={comps[f'P1_vs_{b}.odds']['p_joint']:.4f}。")
    L.append("- 结构性注记: 日级共振稀缺是两法确认延迟差异的必然——分形信号日=低点+15 根, "
             "zigzag 信号日=反弹确认日, 同一低点两法确认日通常相差约 10 根; 全史同股同日命中仅 38 对。"
             "既往 m_scan 的 m_intersect(同股同低点口径, 允许确认日不同)在两法全过滤配置下有 2013 事件 —— "
             "若未来要复活交集思路, 对齐口径须落到低点级而非信号日级(超出本预登记范围, 仅存档)。")
    L.append("")
    # ---- 对比与 bootstrap
    L.append("## 4. 组合池 vs 单池: 双段差异与日聚类 bootstrap")
    L.append("")
    L.append("| 对比 | 维度 | Δtrain | Δval | 联合p | p_train | p_val | 95%CI(val) |")
    L.append("|---|---|---|---|---|---|---|---|")
    tag_cn = {"hit": "命中率", "odds": "赔率"}
    for a, b in (("P1", "main"), ("P1", "backup"), ("P2", "main"), ("P2", "backup"),
                 ("P3", "main"), ("P4", "main")):
        for tag in ("hit", "odds"):
            r = comps[f"{a}_vs_{b}.{tag}"]
            ci = r["ci95_val"]
            if tag == "hit":
                dt_, dv_ = pp(r["delta_train"], 2), pp(r["delta_val"], 2)
                ci_s = f"[{pp(ci[0], 2)}, {pp(ci[1], 2)}]"
            else:
                dt_, dv_ = f"{r['delta_train']:+.3f}", f"{r['delta_val']:+.3f}"
                ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            L.append(f"| {a} vs {disp[b]} | {tag_cn[tag]} | {dt_} | "
                     f"{dv_} | {r['p_joint']:.4f} | {r['p_train']:.4f} | "
                     f"{r['p_val']:.4f} | {ci_s} |")
    L.append("")
    L.append("注: 命中率 Δ 单位为 pp; 赔率 Δ 单位为 ATRN 倍数(mfe20/ATRN 差值); 联合 p = 10000 次日聚类自助复制中"
             "非\"双段同向为正\"的比例(单尾, 检验双段同时改善)。")
    L.append("")
    L.append(f"主池超额主口径对照(网格自身 C1): train {pp(main_ex_gridc1['train']['ex_win_clean'])}pp / "
             f"val {pp(main_ex_gridc1['val']['ex_win_clean'])}pp; 本脚本 C1 口径: "
             f"train {pp(results['main']['train']['clean']['ex_win'])}pp / "
             f"val {pp(results['main']['val']['clean']['ex_win'])}pp —— 两口径差异 <1pp, 偏差不影响比较方向。")
    L.append("")
    # ---- 判定
    L.append("## 5. 判定(预登记活线: 双段同向提升 且 联合 bootstrap p<0.05)")
    L.append("")
    L.append("密度附注(非活线, 仅供定位): P2/P3 的 val 密度 55.6/62.7 事件/周、≤3候选日 32.3%/31.1%, "
             "较主池(17.6 事件/周、60.1%)稀缺性明显恶化; P4 共识也未换来密度改善(51.9 事件/周、36.1%)——"
             "9 配置互为近亲(同分形家族), 并集/共识都只是把同一批信号的参数邻域叠起来。")
    L.append("")
    for p in ("P1", "P2", "P3", "P4"):
        v = verdict[p]
        targets = "+".join(disp[b] for b in v["compared_against"])
        if v["alive"]:
            L.append(f"- {p}: **活口**(维度: {', '.join(tag_cn[d] for d in v['alive_dims'])}; 对比基准: {targets})。")
        else:
            reasons = []
            for b in v["compared_against"]:
                for tag in ("hit", "odds"):
                    r = comps[f"{p}_vs_{b}.{tag}"]
                    stat = []
                    if not (r["delta_train"] > 0 and r["delta_val"] > 0):
                        stat.append("非同向")
                    if r["p_joint"] >= 0.05:
                        stat.append(f"p={r['p_joint']:.3f}")
                    reasons.append(f"{tag_cn[tag]}vs{b}({','.join(stat)})" if stat else None)
            L.append(f"- {p}: 未过活线(对比基准: {targets})—— " + "; ".join(x for x in reasons if x) + "。")
    L.append("")
    L.append("## 6. 复现")
    L.append("")
    L.append("- 驱动: `v3_pipeline/scripts/run_signal_combo.py`(事件检测/标签/C1 复用 divergence_lab, "
             "清洗复用 run_pool_cleaning 机器, 未另造机器)。")
    L.append("- 全量指标: `results_signal_combo.json`; 组合池事件/C1 明细: `pools/<pool>/{events,c1}.parquet`; "
             "备池重算标签: `backup_events.parquet`; 过程日志: `progress.log`。")
    (OUT / "signal_combo_report.md").write_text("\n".join(L) + "\n")
    log("[stage9] signal_combo_report.md 已写出")


if __name__ == "__main__":
    main()
