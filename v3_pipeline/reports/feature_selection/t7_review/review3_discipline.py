#!/usr/bin/env python3
"""T7 复核项3: 纪律核查（test 零触碰 / 预登记时戳 / 段计数 / 特征泄漏）+ 项4 复现产物。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/karl/repos/personal/stock_qt_nd")
sys.path.insert(0, str(REPO / "v3_pipeline" / "src"))
import feature_master as fm  # noqa: E402
import train_eval_pipeline as tep  # noqa: E402

RACE = REPO / "v3_pipeline" / "reports" / "label_race"
OUT = REPO / "v3_pipeline" / "reports" / "feature_selection"
PREREG_TS = pd.Timestamp("2026-09-03T04:07:40Z")


def main():
    # (a) test 零触碰: 产物中不得出现 test 段指标; scores test y 全 NaN
    hits = []
    for p in sorted(OUT.glob("*")):
        if p.suffix in (".csv", ".json", ".log") and p.name != "selection_results.json":
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(txt.splitlines(), 1):
                if "test" in line.lower():
                    hits.append(f"{p.name}:{i}: {line[:120]}")
    print("(a) 产物中 test 字样命中（逐条甄别）:")
    for h in hits:
        print("   ", h)
    scores = pd.read_parquet(OUT / "scores_final.parquet")
    assert scores.loc[scores["seg"] == "test", "y"].isna().all()
    print("    scores_final.parquet test 段 y 全 NaN: 确认")

    # (b) 预登记时戳早于全部结果文件
    mtimes = {p.name: pd.Timestamp(os.path.getmtime(p), unit="s", tz="UTC")
              for p in OUT.glob("*") if p.name != "progress.log"}
    earliest = min(mtimes.values())
    print(f"(b) 预登记 {PREREG_TS} vs 最早结果文件 "
          f"{min(mtimes, key=mtimes.get)} {earliest} "
          f"(预登记早 {(earliest - PREREG_TS).total_seconds():.0f}s)")
    assert earliest > PREREG_TS, "存在早于预登记的结果文件!"

    # (c) 段计数与段界独立断言
    master = pd.read_parquet(RACE / "master_merged.parquet",
                             columns=["date", "seg"])
    cal_d = pd.read_parquet(REPO / "stock_data" / "daily" / "000001.SH.parquet",
                            columns=["trade_date"])
    cal = np.sort(pd.to_datetime(cal_d["trade_date"].astype(str)).unique())
    df_seg = master[master["seg"].isin(["train", "val", "test"])]
    tep.assert_segment_integrity(df_seg, cal)
    seg_counts = {k: int(v) for k, v in df_seg["seg"].value_counts().items()}
    with (OUT / "selection_results.json").open(encoding="utf-8") as f:
        results = json.load(f)
    assert seg_counts == results["seg_counts"], f"段计数不一致: {seg_counts}"
    print(f"(c) 段界/隔离带独立断言通过, 段计数与台账一致: {seg_counts}")

    # (d) 特征泄漏与 pool 列
    full = pd.read_parquet(RACE / "master_merged.parquet")
    feats = tep.model_feature_columns(full)
    assert len(feats) == 2060 and "pool" not in feats
    fm.assert_no_leakage(feats)
    excl = fm.excluded_columns(full.columns)
    assert not excl, f"排除模式命中: {excl}"
    with (OUT / "final_features.json").open(encoding="utf-8") as f:
        final = json.load(f)
    feat_final = [r["feature"] for r in final["features"]]
    assert set(feat_final) <= set(feats), "终选特征越出权威特征列"
    assert len(set(feat_final)) == len(feat_final)
    print(f"(d) 特征列 2060 泄漏断言零命中, 终选 {len(feat_final)} 全部在权威列内")

    # (e) 当选配置链: 裁决 winner 与 summary 行自写复核
    with (RACE / "adjudication_merged.json").open(encoding="utf-8") as f:
        adj = json.load(f)
    summary = pd.read_csv(RACE / "summary_merged.csv", dtype=str)
    row = summary[summary["candidate"] == adj["winner"]].iloc[0]
    assert adj["winner"] == "hit_N20_k2.0" and int(row["config_id"]) == 13
    grid_expect = {"num_leaves": "31", "min_data_in_leaf": "50",
                   "learning_rate": "0.05", "feature_fraction": "0.8"}
    for k, v in grid_expect.items():
        assert float(row[k]) == float(v), f"summary 超参 {k} 与预登记网格不符"
    assert final["config_id"] == 13 and final["label"] == adj["winner"]
    print("(e) 当选配置链复核: winner=hit_N20_k2.0 config_id=13 超参与网格逐位一致")

    # 项4: 复现产物核查 —— 全新进程再跑一次 --repro-check
    r = subprocess.run([sys.executable,
                        str(REPO / "v3_pipeline" / "scripts" /
                            "run_feature_selection.py"), "--repro-check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"复现断言 FAIL:\n{r.stdout}\n{r.stderr}"
    print(f"项4 复现产物核查: {r.stdout.strip()}")
    print("项3/项4 总判: PASS")


if __name__ == "__main__":
    main()
