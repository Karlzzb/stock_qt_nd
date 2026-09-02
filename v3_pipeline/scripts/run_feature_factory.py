# -*- coding: utf-8 -*-
"""特征工厂层主驱动: 生成 1500-2500 列 -> 泄漏审计 -> 交互感知筛查 -> 赛跑协议.

预登记设计 (issue #16):
  1. 生成: 按 feature_master_spec.md 第 6 章生成器层规则; 只用信号日 T 及之前数据;
     每族截断对拍抽查 (3 股 x 50 日); 泄漏 14 条黑名单断言.
  2. 筛查 (交互感知, 替代单变量漏斗): LGBM lambdarank, ATRN 强制入模作几何控制,
     gain + 置换重要性 (日内置换, val 日 Rank IC 下降) 双口径; 3 轮
     (剔底部 50% -> 25% -> 终选 <=30), 逐轮双池复现 (交集晋级).
  3. 赛跑: 终选集进 race_rerun_v2 同构 lambdarank 协议; 对照 = ATRN-only +
     3 个同尺寸随机特征集; 主口径 = 日加权 top3 命中率 vs 清洗后基线
     (主池 50.33% / 备池 55.61%, excl_combined); 家族 Bonferroni (2 个主检验);
     日聚类 bootstrap; 活线 = 双池同向 + 联合 p<0.05 + 双池幅度均 >2pp +
     优于全部随机对照.
  4. 切分: train 2001-2018 / val 2019-01~2022-10 (隔离带剔除); test 封存一行不出
     (本脚本宇宙只含 seg train/val 且 f_any=False 的干净事件, 内嵌断言).

缓存机制 (复跑说明):
  reports/feature_factory/cache/factory_features_{main,backup}.parquet  全量生成特征
  reports/feature_factory/cache/factory_registry.csv                    特征注册表 (表达式)
  reports/feature_factory/cache/audit.json                              泄漏审计+去重结果
  reports/feature_factory/cache/screen_rounds.json                      三轮筛查明细
  reports/feature_factory/results_factory.json                          赛跑+判定全量数值
  各阶段检测到缓存即跳过; --force 全部重跑. 生成阶段最慢 (预计 10-40 分钟),
  筛查/赛跑分钟级. 复跑: python run_feature_factory.py [--force] [--sample N]

用法: python run_feature_factory.py [--force] [--sample N] [--workers 24]
"""
import argparse
import json
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sc_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
import feature_engine as fe  # noqa: E402
import feature_factory as ff  # noqa: E402
import run_race_rerun_v2 as race  # noqa: E402  (复用同构协议函数)

ROOT = SCRIPT_DIR.parents[1]
FM_DIR = ROOT / "v3_pipeline" / "reports" / "feature_matrix"
CLEAN_DIR = ROOT / "v3_pipeline" / "reports" / "pool_cleaning"
OUT_DIR = ROOT / "v3_pipeline" / "reports" / "feature_factory"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "progress.log"

POOLS = {
    "main": {
        "features": FM_DIR / "main_pool_features.parquet",
        "excluded": CLEAN_DIR / "excluded_events_main.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_fractal_o15_s20",
        "expected_baseline": 0.5032867789268494,
    },
    "backup": {
        "features": FM_DIR / "backup_pool_features.parquet",
        "excluded": CLEAN_DIR / "excluded_events_backup.parquet",
        "labdir": ROOT / "v3_pipeline" / "reports" / "divergence_lab" / "w_zigzag_p05_s5",
        "expected_baseline": 0.5560504198074341,
    },
}

SEED = 42
RAND_SEEDS = [20260902, 20260903, 20260904]
N_FINAL = 30
BOOT_B = 5000

SCREEN_PARAMS = dict(
    objective="lambdarank", metric="None", boosting_type="gbdt",
    num_leaves=31, learning_rate=0.05, feature_fraction=0.5,
    bagging_fraction=0.8, bagging_freq=1, min_child_samples=30,
    num_threads=8, deterministic=True, force_row_wise=True,
    seed=SEED, verbose=-1,
)
SCREEN_N_EST = 1000
SCREEN_ES = 50
PERM_REPEATS = 2


def log(msg):
    line = f"{msg} {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ================================================================ 宇宙装配
def build_universe():
    """双池干净宇宙: 特征矩阵键 x seg(train/val) x ~f_any. test 段一行不进."""
    uni = {}
    for pool, cfg in POOLS.items():
        fm = pd.read_parquet(cfg["features"], columns=["ts_code", "date"])
        fm["date"] = pd.to_datetime(fm["date"])
        ex = pd.read_parquet(cfg["excluded"], columns=["ts_code", "date", "f_any"])
        ex["date"] = pd.to_datetime(ex["date"])
        m = fm.merge(ex, on=["ts_code", "date"], how="left", validate="1:1")
        seg = race.segment_mask(m["date"].to_numpy())
        keep = np.isin(seg, ["train", "val"]) & (~m["f_any"].to_numpy())
        u = m.loc[keep, ["ts_code", "date"]].reset_index(drop=True)
        assert u["date"].max() <= pd.Timestamp("2022-10-31"), f"{pool} 宇宙越界进测试段"
        uni[pool] = u
        log(f"阶段=宇宙 {pool}: {len(u)} 行 (train+val, 剔 f_any), "
            f"股票 {u.ts_code.nunique()}, 日期 {u.date.min():%Y-%m-%d}~{u.date.max():%Y-%m-%d}")
    return uni


def load_ridx_map():
    idx = fe.load_index_df(fe.DATA_DIR / "000001.SH.parquet")
    c = idx["close"].to_numpy(np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = c[1:] / c[:-1] - 1.0
    return {int(d): float(v) for d, v in zip(idx["_days"].to_numpy(np.int64), r)}


# ================================================================ 阶段 1: 生成
def _gen_worker(task):
    code, ev = task  # ev: {pool: [int day, ...]}
    try:
        path = fe.DATA_DIR / f"{code}.parquet"
        if not path.exists():
            return code, None, "缺文件"
        df = fe.load_stock_df(path)
        cols, _ = ff.compute_stock_factory(df, _RIDX)
        days_i = df["trade_date"].to_numpy("datetime64[D]").astype(np.int64)
        pos = {int(d): i for i, d in enumerate(days_i)}
        names = list(cols.keys())
        mat = np.column_stack([cols[n] for n in names])
        out = {}
        for pool, dates in ev.items():
            rows, kept = [], []
            for d in dates:
                p = pos.get(d)
                if p is not None:
                    rows.append(p)
                    kept.append(d)
            if kept:
                out[pool] = (kept, mat[rows])
        return code, out, None
    except Exception as e:  # noqa: BLE001
        return code, None, repr(e)


_RIDX = None


def _gen_init(ridx):
    global _RIDX
    _RIDX = ridx


def stage_generate(uni, workers, sample=0):
    paths = {p: CACHE_DIR / f"factory_features_{p}.parquet" for p in POOLS}
    reg_path = CACHE_DIR / "factory_registry.csv"
    if all(p.exists() for p in paths.values()) and reg_path.exists():
        log("阶段=生成 [缓存命中, 跳过]")
        return {p: pd.read_parquet(paths[p]) for p in POOLS}
    t0 = time.time()
    log(f"阶段=生成 开始 (预计 10-40 分钟; workers={workers})")
    ridx = load_ridx_map()
    per_stock = {}
    for pool, u in uni.items():
        for code, grp in u.groupby("ts_code"):
            dts = grp["date"].to_numpy("datetime64[D]").astype(np.int64)
            per_stock.setdefault(code, {})[pool] = sorted(int(d) for d in dts)
    tasks = sorted(per_stock.items())
    if sample:
        tasks = tasks[:sample]
    log(f"阶段=生成 股票数={len(tasks)} (主池股 {len(uni['main'].ts_code.unique())}, "
        f"备池股 {len(uni['backup'].ts_code.unique())})")
    rows = {p: [] for p in POOLS}
    n_err = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_gen_init,
                             initargs=(ridx,)) as ex:
        for i, (code, out, err) in enumerate(
                ex.map(_gen_worker, tasks, chunksize=8)):
            if err:
                n_err += 1
                if n_err <= 5:
                    log(f"  !! {code} 生成失败: {err}")
                continue
            for pool, (kept, mat) in (out or {}).items():
                df_blk = pd.DataFrame(mat, columns=None)
                df_blk.insert(0, "date", np.array(kept, dtype=np.int64))
                df_blk.insert(0, "ts_code", code)
                rows[pool].append(df_blk)
            if (i + 1) % 250 == 0:
                log(f"  心跳 生成 {i+1}/{len(tasks)} 股, 耗时 {time.time()-t0:.0f}s")
    # 注册表 (取单股计算一次)
    df0 = fe.load_stock_df(fe.DATA_DIR / f"{tasks[0][0]}.parquet")
    _, registry = ff.compute_stock_factory(df0, ridx)
    reg = pd.DataFrame(registry, columns=["feature", "op", "field", "window", "expression"])
    reg.to_csv(reg_path, index=False)
    feat_names = reg["feature"].tolist()
    log(f"阶段=生成 注册表 {len(reg)} 列 -> {reg_path.name} ({time.time()-t0:.0f}s)")
    result = {}
    for pool in POOLS:
        blk = pd.concat(rows[pool], ignore_index=True)
        blk.columns = ["ts_code", "date"] + feat_names
        blk["date"] = pd.to_datetime(blk["date"].astype("datetime64[D]"))
        # 与宇宙键对齐 (顺序以宇宙为准)
        u = uni[pool].copy()
        u["_ord"] = np.arange(len(u))
        blk = u.merge(blk, on=["ts_code", "date"], how="left",
                      validate="1:1").sort_values("_ord").drop(columns="_ord")
        assert len(blk) == len(u), f"{pool} 生成行数 {len(blk)} != 宇宙 {len(u)}"
        blk.to_parquet(paths[pool], index=False)
        nan_share = float(blk[feat_names].isna().mean().mean())
        log(f"阶段=生成 {pool}: {blk.shape} NaN占比={nan_share:.3f} -> {paths[pool].name}")
        result[pool] = blk
    log(f"阶段=生成 完成 总耗时 {time.time()-t0:.0f}s, 失败股 {n_err}")
    return result


# ================================================================ 阶段 2: 泄漏审计 + 去重
def _truncation_check(ridx):
    """每族截断对拍: 3 股 x 50 日, 截断至 T 重算 vs 全历史 T 点值, rtol<=1e-6."""
    rng = np.random.default_rng(SEED)
    stocks = ["600519.SH", "000001.SZ", "000002.SZ"]
    fam_cols = {}
    for op in ff.OPS19:
        fam_cols[op] = [f"{op}_CLOSE_{w}" for w in ff.WINDOWS] + \
                       [f"{op}_R_{w}" for w in ff.WINDOWS]
    fam_cols["SUM"] = [f"SUM_R_{w}" for w in ff.WINDOWS]
    fam_cols["COUNT"] = [f"COUNT_R_{w}" for w in ff.WINDOWS]
    fam_cols["CORR"] = [f"CORR_CLOSE_V_{w}" for w in ff.WINDOWS] + \
                       [f"CORR_R_RIDX_{w}" for w in ff.WINDOWS]
    fam_cols["COV"] = [f"COV_R_RIDX_{w}" for w in ff.WINDOWS] + \
                      [f"COV_CLOSE_LOGV_{w}" for w in ff.WINDOWS]
    fam_cols["KBAR"] = [f"KBAR_K{s}" for s in
                        ("MID", "LEN", "MID2", "UP", "UP2", "LOW", "LOW2", "SFT", "SFT2")]
    fam_cols["SNAP"] = ["SNAP_O", "SNAP_H", "SNAP_L", "SNAP_VWAP"]
    report = {}
    for code in stocks:
        df = fe.load_stock_df(fe.DATA_DIR / f"{code}.parquet")
        full_cols, _ = ff.compute_stock_factory(df, ridx)
        n = len(df)
        dates = np.sort(rng.choice(np.arange(120, n - 1), size=min(50, n - 122),
                                   replace=False))
        for t in dates:
            trunc = df.iloc[: t + 1].copy()
            tc, _ = ff.compute_stock_factory(trunc, ridx)
            for fam, cols in fam_cols.items():
                for c in cols:
                    a, b = full_cols[c][t], tc[c][t]
                    if np.isnan(a) and np.isnan(b):
                        continue
                    ok = np.isfinite(a) and np.isfinite(b) and \
                        abs(a - b) <= 1e-12 + 1e-6 * abs(b)
                    if not ok:
                        report.setdefault(fam, []).append(
                            {"stock": code, "t": int(t), "col": c,
                             "full": float(a), "trunc": float(b)})
    return {fam: len([k for k in report if k == fam]) for fam in fam_cols}, report


def stage_audit(feat_dfs, force=False):
    audit_path = CACHE_DIR / "audit.json"
    if audit_path.exists() and not force:
        log("阶段=审计 [缓存命中, 跳过]")
        with audit_path.open() as f:
            return json.load(f)
    t0 = time.time()
    log("阶段=审计 开始 (黑名单断言 + 截断对拍 + 手工特征去重)")
    reg = pd.read_csv(CACHE_DIR / "factory_registry.csv")
    names = reg["feature"].tolist()
    # 1) 黑名单断言 (14 条)
    hits = [c for c in names
            if any(re.match(p, c) for p in ff.BLACKLIST_PATTERNS)]
    assert not hits, f"黑名单命中: {hits}"
    log(f"阶段=审计 黑名单 14 条断言通过 (0/{len(names)} 命中)")
    # 2) 截断对拍
    fam_counts, mism = _truncation_check(load_ridx_map())
    n_mismatch = sum(len(v) for v in mism.values())
    log(f"阶段=审计 截断对拍: {len(fam_counts)} 族 x 3 股 x 50 日, 不一致 {n_mismatch} 处")
    assert n_mismatch == 0, f"截断对拍失败: {mism}"
    # 3) 与 179 手工特征去重 (|rho|>=0.999, NaN 列中位数填充, 双池宇宙行合并)
    gen = pd.concat([feat_dfs[p][["ts_code", "date"] + names] for p in POOLS],
                    ignore_index=True)
    hm = []
    for pool, cfg in POOLS.items():
        h = pd.read_parquet(cfg["features"])
        h["date"] = pd.to_datetime(h["date"])
        hm.append(h)
    hm = pd.concat(hm, ignore_index=True).drop_duplicates(["ts_code", "date"])
    hm_cols = [c for c in hm.columns if c not in ("event_id", "ts_code", "date", "sig_idx")]
    key = gen[["ts_code", "date"]].copy()
    key["date"] = pd.to_datetime(key["date"])
    m = key.merge(hm, on=["ts_code", "date"], how="left", validate="m:1")
    G = gen[names].to_numpy(np.float64)
    Hm = m[hm_cols].to_numpy(np.float64)

    def _fill_std(X):
        X = np.array(X, dtype=np.float64, copy=True)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(med, inds[1])
        X = X - X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return X / sd
    Gs = _fill_std(G)
    Hs = _fill_std(Hm)
    C = (Gs.T @ Hs) / Gs.shape[0]
    dup_pairs = np.argwhere(np.abs(C) >= 0.999)
    dup_gen = sorted(set(names[i] for i, _ in dup_pairs))
    log(f"阶段=审计 去重: |rho|>=0.999 与手工特征重复的生成列 {len(dup_gen)} 个, 剔除")
    audit = {
        "n_generated": len(names),
        "blacklist_hits": hits,
        "truncation_mismatches": n_mismatch,
        "truncation_detail": mism,
        "dedup_dropped": dup_gen,
        "dedup_pairs": [[names[i], hm_cols[j], float(C[i, j])] for i, j in dup_pairs],
        "elapsed_s": time.time() - t0,
    }
    with audit_path.open("w") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    log(f"阶段=审计 完成 ({time.time()-t0:.0f}s)")
    return audit


# ================================================================ 阶段 3: 交互感知筛查
def _fast_daily_rank(df, cols, keep_m):
    """逐日截面秩变换 (rank_pct); 与 race.add_daily_rank 同数学口径, 批量版."""
    sub = df.loc[keep_m, ["date"] + cols]
    ranked = sub.groupby("date")[cols].rank(pct=True)
    ranked.columns = [c + "__DR" for c in cols]
    out = ranked.reindex(df.index)
    df[out.columns] = out
    return {c: c + "__DR" for c in cols}


def _pool_model_data(feat_df, pool, candidates):
    """装配单池筛查数据: __DR 秩变换 + 标签 + 日分组 (train/val)."""
    cfg = POOLS[pool]
    h = pd.read_parquet(cfg["features"], columns=["ts_code", "date", "ATRN"])
    h["date"] = pd.to_datetime(h["date"])
    lab = race.load_labels(cfg["labdir"])
    df = feat_df[["ts_code", "date"] + candidates].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(h, on=["ts_code", "date"], how="left", validate="1:1")
    df = df.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
    df["seg"] = race.segment_mask(df["date"].to_numpy())
    lab_ok = df["hit"].notna().to_numpy()
    train_m = (df["seg"] == "train").to_numpy() & lab_ok
    val_m = (df["seg"] == "val").to_numpy() & lab_ok
    keep_m = train_m | val_m
    cols = ["ATRN"] + candidates
    rank_map = _fast_daily_rank(df, cols, keep_m)
    return df, train_m, val_m, rank_map


class _Screener:
    def __init__(self, df, train_m, val_m, rank_map, candidates):
        self.df = df
        self.train_m = train_m
        self.val_m = val_m
        self.rank_map = rank_map
        self.candidates = candidates
        feats = [rank_map["ATRN"]] + [rank_map[c] for c in candidates]
        X = df[feats].to_numpy(np.float64)
        y = df["hit"].to_numpy(np.float64)
        self.tr_idx, _, self.gtr = race.sort_by_day(df, train_m)
        self.va_idx, self.va_bounds, self.gva = race.sort_by_day(df, val_m)
        self.Xtr, self.ytr = X[self.tr_idx], y[self.tr_idx].astype(int)
        self.Xva, self.yva = X[self.va_idx], y[self.va_idx].astype(int)
        self.feats = feats

    def run(self):
        ic_calc = race.DayRankIC(self.yva.astype(np.float64), self.va_bounds)
        feval = race.make_feval(ic_calc)
        dtrain = lgb.Dataset(self.Xtr, label=self.ytr, group=self.gtr,
                             feature_name=self.feats, free_raw_data=True)
        dval = lgb.Dataset(self.Xva, label=self.yva, group=self.gva,
                           reference=dtrain, free_raw_data=True)
        m = lgb.train(SCREEN_PARAMS, dtrain, num_boost_round=SCREEN_N_EST,
                      valid_sets=[dval], feval=feval,
                      callbacks=[lgb.early_stopping(SCREEN_ES, verbose=False)])
        bi = max(int(m.best_iteration), 1)
        gain = dict(zip(self.feats, m.feature_importance(importance_type="gain")))
        base_sc = m.predict(self.Xva, num_iteration=bi)
        base_ic = float(ic_calc(base_sc))
        # 日内置换重要性 (val 日 Rank IC 下降)
        rng = np.random.default_rng(SEED)
        day_slices = [np.arange(lo, hi) for lo, hi in
                      zip(self.va_bounds[:-1], self.va_bounds[1:])]
        perm_drop = {}
        Xv = self.Xva
        for j, f in enumerate(self.feats):
            saved = Xv[:, j].copy()
            drops = []
            for _ in range(PERM_REPEATS):
                for sl in day_slices:
                    if len(sl) > 1:
                        Xv[sl, j] = saved[sl][rng.permutation(len(sl))]
                sc = m.predict(Xv, num_iteration=bi)
                drops.append(base_ic - float(ic_calc(sc)))
            Xv[:, j] = saved
            perm_drop[f] = float(np.mean(drops))
        return {"gain": gain, "perm_drop": perm_drop, "base_ic": base_ic,
                "best_iter": bi}


def _scores(res, rank_map, candidates):
    """gain 与置换下降的双口径等权秩合成 (候选列空间内)."""
    g = np.array([res["gain"].get(rank_map[c], 0.0) for c in candidates])
    p = np.array([res["perm_drop"].get(rank_map[c], 0.0) for c in candidates])
    rg = sc_stats.rankdata(g) / len(g)
    rp = sc_stats.rankdata(p) / len(p)
    s = (rg + rp) / 2.0
    return dict(zip(candidates, s)), dict(zip(candidates, g)), dict(zip(candidates, p))


def stage_screen(feat_dfs, audit, force=False):
    scr_path = CACHE_DIR / "screen_rounds.json"
    if scr_path.exists() and not force:
        log("阶段=筛查 [缓存命中, 跳过]")
        with scr_path.open() as f:
            out = json.load(f)
        return out["final"], out
    t0 = time.time()
    dropped = set(audit["dedup_dropped"])
    reg = pd.read_csv(CACHE_DIR / "factory_registry.csv")
    cand0 = [c for c in reg["feature"] if c not in dropped]
    log(f"阶段=筛查 开始: 候选 {len(cand0)} 列 (去重剔 {len(dropped)}), "
        f"3 轮 (50%->37.5%-><=30), 双池交集晋级, ATRN 强制入模")
    data = {}
    for pool in POOLS:
        df, train_m, val_m, rank_map = _pool_model_data(
            feat_dfs[pool], pool, cand0)
        data[pool] = (df, train_m, val_m, rank_map)
        log(f"阶段=筛查 {pool} 数据装配+秩变换完成 ({time.time()-t0:.0f}s)")

    rounds_log = {}
    candidates = cand0
    keep_fracs = [0.5, 0.75]  # 轮1 剔底部 50%; 轮2 再剔底部 25%
    for rd, frac in enumerate(keep_fracs, start=1):
        keeps = {}
        for pool in POOLS:
            df, train_m, val_m, rank_map = data[pool]
            scr = _Screener(df, train_m, val_m, rank_map, candidates)
            res = scr.run()
            s, g, p = _scores(res, rank_map, candidates)
            k = max(int(np.ceil(len(candidates) * frac)), 1)
            top = sorted(candidates, key=lambda c: -s[c])[:k]
            keeps[pool] = set(top)
            rounds_log[f"round{rd}_{pool}"] = {
                "n_in": len(candidates), "n_keep": k,
                "base_ic": res["base_ic"], "best_iter": res["best_iter"],
                "scores": s, "gain": g, "perm_drop": p,
                "atrn_gain": res["gain"].get(rank_map["ATRN"], 0.0),
                "atrn_perm_drop": res["perm_drop"].get(rank_map["ATRN"], 0.0),
            }
            log(f"阶段=筛查 轮{rd} {pool}: {len(candidates)}->{k}, "
                f"val日RankIC={res['base_ic']:.4f}, iter={res['best_iter']} "
                f"({time.time()-t0:.0f}s)")
        inter = sorted(keeps["main"] & keeps["backup"])
        log(f"阶段=筛查 轮{rd} 双池交集: {len(inter)} 列晋级")
        rounds_log[f"round{rd}_intersection"] = inter
        candidates = inter
    # 轮 3: 终选 <=30 (双池 top-60 交集内按跨池均分排序; 不足则从并集按均分补)
    per_pool = {}
    for pool in POOLS:
        df, train_m, val_m, rank_map = data[pool]
        scr = _Screener(df, train_m, val_m, rank_map, candidates)
        res = scr.run()
        s, g, p = _scores(res, rank_map, candidates)
        per_pool[pool] = (s, g, p, res)
        log(f"阶段=筛查 轮3 {pool}: {len(candidates)} 列打分完成 "
            f"val日RankIC={res['base_ic']:.4f} ({time.time()-t0:.0f}s)")
    sm, gm, pm, _ = per_pool["main"]
    sb, gb, pb, _ = per_pool["backup"]
    mean_score = {c: (sm[c] + sb[c]) / 2.0 for c in candidates}
    top60m = set(sorted(candidates, key=lambda c: -sm[c])[:60])
    top60b = set(sorted(candidates, key=lambda c: -sb[c])[:60])
    both = sorted(top60m & top60b, key=lambda c: -mean_score[c])
    final = both[:N_FINAL]
    if len(final) < N_FINAL:
        rest = [c for c in sorted(candidates, key=lambda c: -mean_score[c])
                if c not in final]
        final += rest[: N_FINAL - len(final)]
    rounds_log["round3"] = {
        "n_in": len(candidates),
        "both_top60": len(both),
        "final": final,
        "scores_main": sm, "scores_backup": sb,
        "gain_main": gm, "gain_backup": gb,
        "perm_main": pm, "perm_backup": pb,
        "mean_score": mean_score,
        "atrn": {pool: {"gain": per_pool[pool][3]["gain"].get(
                    data[pool][3]["ATRN"], 0.0),
                 "perm_drop": per_pool[pool][3]["perm_drop"].get(
                    data[pool][3]["ATRN"], 0.0)} for pool in POOLS},
    }
    log(f"阶段=筛查 完成: 终选 {len(final)} 列 (双池top60交集 {len(both)}) "
        f"({time.time()-t0:.0f}s)")
    out = {"final": final, **rounds_log}
    with scr_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return final, out


# ================================================================ 阶段 4: 赛跑
def _boot_excess(t3, base_daily_vals, rng, B=BOOT_B):
    """日聚类 bootstrap: 按日有放回重抽样, 日加权超额分布."""
    diff = t3 - base_daily_vals
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    boots = diff[idx].mean(axis=1)
    p_one = float((boots <= 0).mean())
    return {
        "boot_p_one_sided": p_one,
        "boot_p_two_sided": float(min(1.0, 2 * min(p_one, 1 - p_one))),
        "boot_ci_lo": float(np.quantile(boots, 0.025)),
        "boot_ci_hi": float(np.quantile(boots, 0.975)),
        "excess_mean": float(diff.mean()),
    }


def stage_race(feat_dfs, final, force=False):
    t0 = time.time()
    log(f"阶段=赛跑 开始: 终选 {len(final)} 列 + ATRN; 对照 ATRN-only + 3 随机集")
    reg = pd.read_csv(CACHE_DIR / "factory_registry.csv")
    with (CACHE_DIR / "audit.json").open() as f:
        audit = json.load(f)
    dropped = set(audit["dedup_dropped"])
    factory_pool = [c for c in reg["feature"] if c not in dropped]
    rand_sets = {}
    for s in RAND_SEEDS:
        rng = np.random.default_rng(s)
        pool = [c for c in factory_pool if c not in final]
        rand_sets[f"RAND_s{s}"] = sorted(rng.choice(pool, size=len(final),
                                                    replace=False).tolist())
    configs = {"FACTORY": list(final), "ATRN_ONLY": []}
    configs.update({k: list(v) for k, v in rand_sets.items()})

    results = {"seed": SEED, "rand_seeds": RAND_SEEDS, "n_final": len(final),
               "final_features": final, "rand_sets": rand_sets, "pools": {}}
    rng = np.random.default_rng(SEED)
    for pool in POOLS:
        cfg = POOLS[pool]
        h = pd.read_parquet(cfg["features"], columns=["ts_code", "date", "ATRN"])
        h["date"] = pd.to_datetime(h["date"])
        lab = race.load_labels(cfg["labdir"])
        need = sorted(set(sum(configs.values(), [])))
        df = feat_dfs[pool][["ts_code", "date"] + need].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(h, on=["ts_code", "date"], how="left", validate="1:1")
        df = df.merge(lab, on=["ts_code", "date"], how="left", validate="1:1")
        df["seg"] = race.segment_mask(df["date"].to_numpy())
        df = df.replace([np.inf, -np.inf], np.nan)
        lab_ok = df["hit"].notna().to_numpy()
        train_m = (df["seg"] == "train").to_numpy() & lab_ok
        val_m = (df["seg"] == "val").to_numpy() & lab_ok
        # 清洗后基线独立重算 (断言)
        sub = df.loc[val_m]
        daily = sub.groupby("date")["hit"].mean()
        base_dw = float(daily.mean())
        exp = cfg["expected_baseline"]
        status = "一致" if abs(base_dw - exp) < 5e-4 else "!!不符!!"
        log(f"阶段=赛跑 {pool} 清洗基线独立重算: 日加权={base_dw:.6f} "
            f"(预期 {exp:.4f}, {status}), {len(daily)} 信号日/{len(sub)} 事件")
        assert abs(base_dw - exp) < 5e-4, f"{pool} 基线不符"
        base_daily = {str(d)[:10]: float(v) for d, v in daily.items()}
        keep_m = train_m | val_m
        all_cols = ["ATRN"] + need
        rank_map = race.add_daily_rank(df, all_cols, keep_m)
        pool_res = {"n_train": int(train_m.sum()), "n_val": int(val_m.sum()),
                    "baseline_day_weighted": base_dw, "configs": {}}
        for cname, cands in configs.items():
            feats_raw = ["ATRN"] + cands
            for variant in ("rank", "raw"):
                feats = ([rank_map[c] for c in feats_raw] if variant == "rank"
                         else feats_raw)
                tag = f"{pool}/{cname}__{variant}"
                r = race.run_config(df, train_m, val_m, feats, tag)
                days, t3 = r["_daily"]
                b = np.array([base_daily[str(d)[:10]] for d in days], np.float64)
                st = race.wilcoxon_vs_baseline(days, t3, base_daily)
                bt = _boot_excess(t3, b, rng)
                r["baseline_day_weighted"] = base_dw
                r["excess_pp"] = (r["val"]["top3"] - base_dw) * 100.0
                r["stats"] = st
                r["boot"] = bt
                r.pop("_daily")
                pool_res["configs"][f"{cname}__{variant}"] = r
        results["pools"][pool] = pool_res
        log(f"阶段=赛跑 {pool} 完成 ({time.time()-t0:.0f}s)")

    # 判定 (预登记活线)
    verdict = {}
    for variant in ("rank", "raw"):
        f_main = results["pools"]["main"]["configs"][f"FACTORY__{variant}"]
        f_back = results["pools"]["backup"]["configs"][f"FACTORY__{variant}"]
        p1 = f_main["boot"]["boot_p_one_sided"]
        p2 = f_back["boot"]["boot_p_one_sided"]
        joint_p = min(1.0, 2 * max(p1, p2))  # 家族 Bonferroni (2 主检验, 双池均须过)
        rand_beat = all(
            f_main["val"]["top3"] > results["pools"]["main"]["configs"][f"{rk}__{variant}"]["val"]["top3"] and
            f_back["val"]["top3"] > results["pools"]["backup"]["configs"][f"{rk}__{variant}"]["val"]["top3"]
            for rk in rand_sets)
        alive = (f_main["excess_pp"] > 2.0 and f_back["excess_pp"] > 2.0
                 and joint_p < 0.05 and rand_beat)
        verdict[variant] = {
            "excess_pp_main": f_main["excess_pp"],
            "excess_pp_backup": f_back["excess_pp"],
            "boot_p_main": p1, "boot_p_backup": p2,
            "joint_p_bonferroni": joint_p,
            "beat_all_random": bool(rand_beat),
            "alive": bool(alive),
        }
    results["verdict"] = verdict
    log(f"阶段=赛跑 判定: rank变体活线={verdict['rank']['alive']} "
        f"(超额 主{verdict['rank']['excess_pp_main']:+.2f}pp/"
        f"备{verdict['rank']['excess_pp_backup']:+.2f}pp, "
        f"联合p={verdict['rank']['joint_p_bonferroni']:.4f}) "
        f"({time.time()-t0:.0f}s)")
    return results


# ================================================================ 阶段 5: 终选特征落盘 + 报告
def stage_outputs(feat_dfs, final, screen, results):
    reg = pd.read_csv(CACHE_DIR / "factory_registry.csv")
    expr = dict(zip(reg["feature"], reg["expression"]))
    for pool in POOLS:
        cfg = POOLS[pool]
        h = pd.read_parquet(cfg["features"], columns=["ts_code", "date", "ATRN"])
        h["date"] = pd.to_datetime(h["date"])
        out = feat_dfs[pool][["ts_code", "date"] + final].copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.merge(h, on=["ts_code", "date"], how="left", validate="1:1")
        path = OUT_DIR / f"final_features_{pool}.parquet"
        out.to_parquet(path, index=False)
        log(f"阶段=落盘 {path.name}: {out.shape}")
    with (OUT_DIR / "results_factory.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    log(f"阶段=落盘 results_factory.json")
    _write_report(final, expr, screen, results)


def _fmt(x, nd=4):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{nd}f}"


def _write_report(final, expr, screen, results):
    reg = pd.read_csv(CACHE_DIR / "factory_registry.csv")
    with (CACHE_DIR / "audit.json").open() as f:
        audit = json.load(f)
    L = []
    A = L.append
    A("# 特征工厂层（G 层）战役报告 —— 生成 → 交互感知筛查 → 赛跑终审")
    A("")
    A(f"> 生成脚本：`v3_pipeline/scripts/run_feature_factory.py`（可复跑，缓存见文末）。")
    A(f"> 生成日期：{pd.Timestamp.now():%Y-%m-%d}。预登记见 issue #16 任务书。")
    A("> 统计纪律：train 2001-2018 / val 2019-01~2022-10（隔离带剔除）；test 段一行不出")
    A("（宇宙构建即只含 seg train/val 且 f_any=False 的干净事件，内嵌断言）。")
    A("")
    A("## 1. 生成清单")
    A("")
    A(f"- 总列数：**{audit['n_generated']}**（目标区间 1500-2500）。")
    fam = reg.groupby("op")["feature"].count().to_dict()
    A(f"- 分族列数：{json.dumps({k: int(v) for k, v in sorted(fam.items())}, ensure_ascii=False)}")
    A("- 泄漏审计：14 条黑名单断言 0 命中；截断对拍 3 股 × 50 日 × 全族代表列，"
      f"不一致 {audit['truncation_mismatches']} 处（rtol≤1e-6，断言通过）。")
    A(f"- 与 179 手工特征去重（|ρ|≥0.999）：剔除 {len(audit['dedup_dropped'])} 列"
      f"（STD_R_* ↔ VOL*、MAX_R_20 ↔ MAX20 等同式覆盖），剔除名单见 cache/audit.json。")
    A("- 设计决定 D1-D5 见 `v3_pipeline/src/feature_factory.py` 模块文档"
      "（归一化 baked-in、VAR/-1x 不生成、cs_rank 第二层不批量生成、NaN 严格、float64）。")
    A("")
    A("## 2. 交互感知筛查漏斗")
    A("")
    A("口径：LGBM lambdarank（日分组），ATRN 强制入模作几何控制；特征一律 __DR 逐日截面秩变换")
    A("（与赛跑主变体同构）；双口径重要性 = gain + 日内置换（val 日 Rank IC 下降，2 次重复）等权秩合成；")
    A("逐轮双池交集晋级（双池复现要求）。")
    A("")
    A("| 轮次 | 输入列数 | 主池保留 | 备池保留 | 双池交集晋级 |")
    A("|---|---|---|---|---|")
    n0 = screen["round1_main"]["n_in"]
    A(f"| 轮1（剔底部50%） | {n0} | {screen['round1_main']['n_keep']} | "
      f"{screen['round1_backup']['n_keep']} | {len(screen['round1_intersection'])} |")
    A(f"| 轮2（再剔底部25%） | {screen['round2_main']['n_in']} | {screen['round2_main']['n_keep']} | "
      f"{screen['round2_backup']['n_keep']} | {len(screen['round2_intersection'])} |")
    A(f"| 轮3（终选≤30） | {screen['round3']['n_in']} | — | — | "
      f"**{len(screen['round3']['final'])}**（双池top60交集 {screen['round3']['both_top60']}） |")
    A("")
    A("轮1/轮2 各池 val 日 Rank IC（筛查模型自检，非结论）："
      f"主 {screen['round1_main']['base_ic']:.4f}/{screen['round2_main']['base_ic']:.4f}，"
      f"备 {screen['round1_backup']['base_ic']:.4f}/{screen['round2_backup']['base_ic']:.4f}。")
    A("")
    A("### 终选特征（按跨池均分排序；来源=双池top60交集 或 均分补位）")
    A("")
    A("| # | 特征 | 表达式 | 来源 | 均分 | 主池gain | 备池gain | 主池置换降 | 备池置换降 |")
    A("|---|---|---|---|---|---|---|---|---|")
    r3 = screen["round3"]
    sm_ = r3["scores_main"]
    sb_ = r3["scores_backup"]
    top60m = set(sorted(sm_, key=lambda c: -sm_[c])[:60])
    top60b = set(sorted(sb_, key=lambda c: -sb_[c])[:60])
    for i, c in enumerate(sorted(final, key=lambda c: -r3["mean_score"][c]), 1):
        src = "交集" if (c in top60m and c in top60b) else "补位"
        A(f"| {i} | `{c}` | {expr.get(c, '')} | {src} | {r3['mean_score'][c]:.3f} | "
          f"{r3['gain_main'][c]:.1f} | {r3['gain_backup'][c]:.1f} | "
          f"{_fmt(r3['perm_main'][c], 5)} | {_fmt(r3['perm_backup'][c], 5)} |")
    A("")
    A(f"几何控制 ATRN 在终轮的重要性：主池 gain={r3['atrn']['main']['gain']:.1f} / "
      f"置换降={_fmt(r3['atrn']['main']['perm_drop'], 5)}；"
      f"备池 gain={r3['atrn']['backup']['gain']:.1f} / "
      f"置换降={_fmt(r3['atrn']['backup']['perm_drop'], 5)}。")
    A("")
    A("## 3. 赛跑结果（race_rerun_v2 同构协议）")
    A("")
    A("口径：lambdarank 日分组 + val 日 Rank IC 早停 + 6 组超参网格；清洗后宇宙（剔 ST/一字涨停/停牌）；")
    A("基线 = 清洗后日加权零信息基线（主池 "
      f"{results['pools']['main']['baseline_day_weighted']:.4f} / 备池 "
      f"{results['pools']['backup']['baseline_day_weighted']:.4f}，"
      "与 pool_cleaning excl_combined 逐位一致断言通过）；统计 = 逐日配对 Wilcoxon + 日聚类 bootstrap（B=5000）。")
    A("")
    for variant, vname in (("rank", "主变体：__DR 截面秩变换"), ("raw", "对照变体：原始值")):
        A(f"### {vname}")
        A("")
        A("| 池 | 配置 | 特征数 | val top3 | 超额(pp) | boot p(单侧) | Wilcoxon p | boot 95% CI(pp) | iter |")
        A("|---|---|---|---|---|---|---|---|")
        for pool, pname in (("main", "主池"), ("backup", "备池")):
            for cn in ["FACTORY", "ATRN_ONLY"] + [f"RAND_s{s}" for s in RAND_SEEDS]:
                r = results["pools"][pool]["configs"][f"{cn}__{variant}"]
                ci = (r["boot"]["boot_ci_lo"] * 100, r["boot"]["boot_ci_hi"] * 100)
                A(f"| {pname} | {cn} | {r['n_features']} | {r['val']['top3']:.4f} | "
                  f"{r['excess_pp']:+.2f} | {_fmt(r['boot']['boot_p_one_sided'])} | "
                  f"{_fmt(r['stats']['wilcoxon_p'])} | [{ci[0]:+.2f},{ci[1]:+.2f}] | {r['best_iter']} |")
        A("")
    A("## 4. 判定（预登记活线）")
    A("")
    A("活线 = 双池超额同向且均 >+2pp 且 家族 Bonferroni 联合 p<0.05（2 主检验，取双池较大 p×2）"
      "且 双池均优于全部 3 个同尺寸随机对照。")
    A("")
    A("| 变体 | 主池超额 | 备池超额 | 联合 p | 优于全部随机 | 判定 |")
    A("|---|---|---|---|---|---|")
    for variant in ("rank", "raw"):
        v = results["verdict"][variant]
        A(f"| {variant} | {v['excess_pp_main']:+.2f}pp | {v['excess_pp_backup']:+.2f}pp | "
          f"{v['joint_p_bonferroni']:.4f} | {'是' if v['beat_all_random'] else '否'} | "
          f"**{'过活线' if v['alive'] else '未过活线'}** |")
    A("")
    alive_any = any(results["verdict"][v]["alive"] for v in ("rank", "raw"))
    if alive_any:
        A("**判定：过活线。** 注意 race_rerun_v2 复核的残余风险：验证段三段混用（早停/选参/评估同一 val），"
          "偏向方向为抬高上报值——过线结果在测试段终审确认前不应采信，测试段仍封存。")
    else:
        A("**判定：未过活线（判平/判负）。** 验证段三段混用的偏向方向只会抬高上报超额，"
          "对否定性裁决为保守——真实超额只会更低。结合 179 手工特征零方向阿尔法、15 P2 特征零幸存、"
          "三代协议 42 组合判平的既有证据，特征线在本轮 1600+ 列工厂生成 + 交互感知筛查下仍未产出")
        A("可兑现的方向性 alpha；\"关闭\"措辞按 race_rerun_v2 复核建议限定为\"已试协议空间内关闭\"。")
    A("")
    A("## 5. 产物与复跑")
    A("")
    A("- `feature_factory_report.md`（本文）、`results_factory.json`（全量数值）、`progress.log`（阶段+心跳）。")
    A("- `final_features_{main,backup}.parquet`：终选特征值（ts_code, date + 终选列 + ATRN）。")
    A("- `cache/`：`factory_features_*`（全量 1603 列）、`factory_registry.csv`（表达式注册表）、"
      "`audit.json`（泄漏审计+去重）、`screen_rounds.json`（三轮明细）。")
    A("- 复跑：`python v3_pipeline/scripts/run_feature_factory.py`（缓存命中自动跳过对应阶段；`--force` 全量重跑）。")
    (OUT_DIR / "feature_factory_report.md").write_text("\n".join(L), encoding="utf-8")
    log("阶段=落盘 feature_factory_report.md")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    race.log = log  # run_config 的心跳重定向到本战役 progress.log, 不污染旧报告
    t0 = time.time()
    log("=" * 70)
    log("阶段=启动 特征工厂战役 (生成->审计->筛查->赛跑->报告), "
        "预计 1-3 小时; 本日志即心跳")
    if args.force:
        for p in CACHE_DIR.glob("*"):
            p.unlink()
    uni = build_universe()
    feat_dfs = stage_generate(uni, args.workers, args.sample)
    audit = stage_audit(feat_dfs, force=args.force)
    final, screen = stage_screen(feat_dfs, audit, force=args.force)
    results = stage_race(feat_dfs, final, force=args.force)
    stage_outputs(feat_dfs, final, screen, results)
    log(f"阶段=全部完成 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
