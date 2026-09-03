#!/usr/bin/env python3
"""T9 复核项 5+6：测试段面板核查与预登记时序。

5. 面板断言与抽查（自写）：
   - results json 四断言全 PASS；面板行数实数 == 3911166；窗口内股票数 == 4375；
   - prob_anchor 独立抽验：随机抽 500 个测试段事件，面板 (ts_code, event_date) 的 prob
     与 scores_final.parquet 原始事件行逐位一致（容差 1e-12）；
   - 面板 prob 值域 [0,1]，(ts_code,date) 唯一。
6. 预登记时序（事实打印 + 断言）：
   - gh 预登记评论 created_at（+0800）先于面板 mtime、面板 mtime 先于终审首跑日志时间。
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

REPO = "/home/karl/repos/personal/stock_qt_nd"
PANEL = os.path.join(REPO, "v3_pipeline/reports/strategy_tuning",
                     "daily_score_panel_20221101_20260831.parquet")
RESULTS = PANEL.replace(".parquet", "_results.json")
SCORES = os.path.join(REPO, "v3_pipeline/reports/feature_selection/scores_final.parquet")

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")


res = json.load(open(RESULTS))
check("四断言全 PASS", all(a["pass"] for a in res["assertions"]),
      str({a["name"]: a["pass"] for a in res["assertions"]}))
check("窗口=测试段", res["window"] == ["2022-11-01", "2026-08-31"])

panel = pd.read_parquet(PANEL, columns=["ts_code", "date", "prob"])
panel["date"] = pd.to_datetime(panel["date"])
check("面板行数==3911166", len(panel) == 3911166, f"(实际 {len(panel)})")
check("股票数==4375", panel["ts_code"].nunique() == 4375,
      f"(实际 {panel['ts_code'].nunique()})")
check("(ts_code,date) 唯一", not panel.duplicated(["ts_code", "date"]).any())
check("prob 值域 [0,1]",
      bool((panel["prob"] >= 0).all() and (panel["prob"] <= 1).all()),
      f"[{panel['prob'].min():.6f},{panel['prob'].max():.6f}]")
check("面板日期全在窗口内",
      bool((panel["date"] >= "2022-11-01").all() and (panel["date"] <= "2026-08-31").all()))

# prob_anchor 独立抽验（500 事件随机抽样，含全部 13 个跨池重复对）
sf = pd.read_parquet(SCORES)
t = sf[sf["seg"] == "test"].copy()
t["date"] = pd.to_datetime(t["date"])
dup_pairs = t[t.duplicated(["ts_code", "date"], keep=False)]
sample = pd.concat([t.sample(500, random_state=29), dup_pairs]).drop_duplicates()
sp = panel.set_index(["ts_code", "date"])["prob"]
mismatch = 0
missing = 0
for r in sample.itertuples():
    v = sp.get((r.ts_code, r.date), np.nan)
    if not np.isfinite(v):
        missing += 1
    elif abs(v - r.prob) > 1e-12:
        mismatch += 1
check(f"事件行 prob 抽验 {len(sample)} 条：缺失 {missing} 不一致 {mismatch}",
      missing == 0 and mismatch == 0)

# ---- 6. 预登记时序 ----
out = subprocess.run(
    ["gh", "issue", "view", "29", "--comments", "--json", "comments"],
    capture_output=True, text=True, check=True)
comments = json.loads(out.stdout)["comments"]
prereg = [c for c in comments if c["id"] == "IC_kwDOUKbGoM8AAAABSR3GfQ"]
check("预登记评论 id 5521655421 存在", len(prereg) == 1)
if prereg:
    created = pd.Timestamp(prereg[0]["createdAt"]).tz_convert("Asia/Shanghai")
    panel_mtime = pd.Timestamp(os.stat(PANEL).st_mtime, unit="s", tz="Asia/Shanghai")
    res_mtime = pd.Timestamp(os.stat(RESULTS).st_mtime, unit="s", tz="Asia/Shanghai")
    log_mtime_first = pd.Timestamp("2026-09-03 14:57:47+0800")  # t9_progress.log 首行
    print(f"  预登记 created_at = {created}")
    print(f"  面板 parquet mtime = {panel_mtime}；results mtime = {res_mtime}")
    print(f"  终审首跑开始 = {log_mtime_first}")
    check("预登记先于面板构建", created < panel_mtime,
          f"(早 {(panel_mtime - created).total_seconds():.0f}s)")
    check("面板先于终审首跑", panel_mtime < log_mtime_first)
    # 预登记文本自检：三配置参数与 adjudication.json spec 一致
    adj = json.load(open(os.path.join(
        REPO, "v3_pipeline/reports/test_adjudication/adjudication.json")))
    body = prereg[0]["body"]
    a, b, c = (adj["results"][k]["spec"] for k in ("A13", "B15", "C15"))
    check("A13 spec 与预登记一致",
          a["tp"] == 0.25 and a["sl"] == -0.14 and a["horizon"] == 12
          and "tp +25% / sl -14% / H12 / N3" in body)
    check("B15 spec 与预登记一致",
          b["horizon"] == 12 and b["vol_lookback"] == 21 and b["vol_high_thresh"] == 1.8
          and b["vol_low_thresh"] == 0.6 and b["vol_profit_mult"] == 1.5
          and b["vol_stop_mult"] == 1.1 and b["low_vol_profit_mult"] == 1.0
          and "LB21" in body and "1.8" in body)
    check("C15 spec 与预登记一致",
          c["horizon"] == 15 and c["top_k"] == 3 and c["score_margin"] == 0.0
          and "H15 / N3 / top_k=3 / margin=0" in body)
    check("预登记窗口与裁决窗口一致",
          "2022-11-01..2026-08-31" in body and adj["window"] == ["2022-11-01", "2026-08-31"])
    check("预登记过线标准与裁决阈值一致",
          "> 基准 +15pp" in body.replace("+15pp", "+15pp")
          and adj["thresholds"] == {"excess_ann_gt": 0.15, "sharpe_gt": 0.5})

print("REVIEW56:", "PASS" if ok else "FAIL")
