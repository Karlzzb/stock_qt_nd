#!/usr/bin/env python3
"""T9 复核项 3：僵尸仓修复正确性 + 类 C 全量独立重放（自写，不调被复核脚本任何函数）。

A. 僵尸仓核验：
   1. 600781.SH 原始日线最后行情日 == 2023-06-19（入场日）；
   2. 三只 open_positions 均不在 trades 表；逐只解释"为何永不平仓"
      （入场后无行情 / 有行情日的面板分与到期/跌停条件）；
   3. 面板对三无行情日是否有分（若有则重放断言的候选集会与引擎分歧——引擎要求当日有 bar）。

B. 独立重放 C15 全部 224 笔（自写候选集重建，口径按引擎语义独立推导）：
   - 在持集合 = trades ∪ open_positions；
   - 已平仓 q 在决策日 D 评估时在持 ⟺ entry<=D 且 (exit>D 或 (exit==D 且 reason=='horizon'))；
   - 引擎口径：无 D 日 bar 的持仓不入候选集——本脚本同时算"含/不含 bar 过滤"两版，
     证明在该数据上两版等价（否则被复核的重放断言与引擎存在口径差）；
   - rank_out：本股不在 top_k；score_drop：s_D < buy_prob×(1-margin)；
   - horizon：exit_date 为 entry 起首个 held_days>=H 且有行情日，exit_raw=当日收盘；
   - rank_out/score_drop 的 exit_raw_price = exit_date 开盘价（独立读原始日线）；
   - 决策日 D = exit_date 前该股最后一个有行情日。
C. 688271.SH 2023-06-20 rank_out 专项复算（首跑唯一不一致笔）。
D. 入场侧抽查：entry_date = event_date 后首个交易日、entry_price 与原始开盘关系。
"""
import os

import numpy as np
import pandas as pd

REPO = "/home/karl/repos/personal/stock_qt_nd"
RUN = os.path.join(REPO, "v3_pipeline/reports/test_adjudication/runs/test_C15")
PANEL = os.path.join(REPO, "v3_pipeline/reports/strategy_tuning",
                     "daily_score_panel_20221101_20260831.parquet")
SCORES = os.path.join(REPO, "v3_pipeline/reports/feature_selection/scores_final.parquet")
START, END = pd.Timestamp("2022-11-01"), pd.Timestamp("2026-08-31")
TOP_K, MARGIN, H = 3, 0.0, 15

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'MISMATCH'}] {name} {detail}")


# ---- 事件重建（自写去重：prob 高者；平局 main 优先；再平局 event_id 升序）----
sf = pd.read_parquet(SCORES)
t = sf[sf["seg"] == "test"].copy()
t["date"] = pd.to_datetime(t["date"])
t["pool_rank"] = (t["pool"] != "main").astype(int)  # main=0 优先
t = t.sort_values(["ts_code", "date", "prob", "pool_rank", "event_id"],
                  ascending=[True, True, False, True, True])
events = t.drop_duplicates(["ts_code", "date"], keep="first").reset_index(drop=True)
check("事件去重 10643->10630", len(t) == 10643 and len(events) == 10630,
      f"(实际 {len(t)}->{len(events)})")

cal = pd.read_parquet(os.path.join(REPO, "stock_data/index/000905.SH.parquet"),
                      columns=["trade_date"])
cal["trade_date"] = pd.to_datetime(cal["trade_date"])
calendar = sorted(d for d in cal["trade_date"].unique() if START <= d <= END)
cal_index = {d: i for i, d in enumerate(calendar)}
check("日历 931 天", len(calendar) == 931, f"(实际 {len(calendar)})")
events = events[events["date"].isin(set(calendar))].reset_index(drop=True)
check("日历过滤后事件 10630", len(events) == 10630, f"(实际 {len(events)})")
by_date = {d: g for d, g in events.groupby("date")}

panel = pd.read_parquet(PANEL, columns=["ts_code", "date", "prob"])
panel["date"] = pd.to_datetime(panel["date"])
sp = panel.set_index(["ts_code", "date"])["prob"]

trades = pd.read_parquet(os.path.join(RUN, "trades.parquet"))
opens = pd.read_parquet(os.path.join(RUN, "open_positions.parquet"))
for c in ("event_date", "entry_date", "exit_date"):
    trades[c] = pd.to_datetime(trades[c])
opens["entry_date"] = pd.to_datetime(opens["entry_date"])

daily_cache: dict[str, pd.DataFrame] = {}


def daily(code):
    if code not in daily_cache:
        p = os.path.join(REPO, "stock_data/daily", f"{code}.parquet")
        df = pd.read_parquet(p, columns=["trade_date", "open", "close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        daily_cache[code] = df.set_index("trade_date").sort_index()
    return daily_cache[code]


# ================= A. 僵尸仓核验 =================
print("== A. 僵尸仓核验 ==")
d781 = daily("600781.SH")
check("600781.SH 最后行情日=2023-06-19", str(d781.index[-1].date()) == "2023-06-19",
      f"(实际 {d781.index[-1].date()})")
# 僵尸仓判定口径=(ts_code, entry_date) 持仓对：同股其他事件的正常交易不构成平仓
z781 = ((trades["ts_code"] == "600781.SH")
        & (trades["entry_date"] == pd.Timestamp("2023-06-19")))
check("600781.SH 僵尸仓(entry=2023-06-19)不在 trades", not z781.any(),
      f"(同股其他事件交易 {int((trades['ts_code'] == '600781.SH').sum())} 笔，属正常)")
check("open_positions 含 600781.SH 且 entry=2023-06-19",
      ((opens["ts_code"] == "600781.SH")
       & (opens["entry_date"] == pd.Timestamp("2023-06-19"))).any())

for _, r in opens.iterrows():
    code = r.ts_code
    dd = daily(code)
    after = dd.index[dd.index > r.entry_date]
    # 入场后行情日的面板分覆盖
    n_scored = sum(1 for d0 in after if np.isfinite(sp.get((code, d0), np.nan)))
    # 到期退出条件：入场后第 H 个交易日及之后是否有行情
    h_days = [d0 for d0 in after
              if d0 in cal_index and cal_index[d0] - cal_index[r.entry_date] + 1 >= H]
    z = ((trades["ts_code"] == code) & (trades["entry_date"] == r.entry_date))
    check(f"僵尸仓 {code} entry={r.entry_date.date()} 不在 trades", not z.any(),
          f"(同股其他事件交易 {int((trades['ts_code'] == code).sum())} 笔)")
    print(f"    {code}: 入场后行情日数={len(after)} 其中有面板分={n_scored} "
          f"满足到期(held>={H})的行情日数={len(h_days)}"
          + (f" 首个到期行情日={h_days[0].date()}" if h_days else " （无到期行情日->永不平仓）"))
    # 若存在到期行情日却未平仓，唯一合法解释=该日收盘跌停顺延；打印该日事实供人工判断
    if h_days:
        d0 = h_days[0]
        row = dd.loc[d0]
        print(f"    {code} {d0.date()}: open={row['open']} close={row['close']} "
              f"score={sp.get((code, d0), np.nan)}")
# 面板对三无行情日是否有分（口径分歧检验）
for code, entry in (("600781.SH", pd.Timestamp("2023-06-19")),):
    p_dates = panel.loc[panel["ts_code"] == code, "date"]
    extra = p_dates[~p_dates.isin(daily(code).index)]
    check(f"{code} 面板日期全部有对应 bar", len(extra) == 0,
          f"(无 bar 却有分的日期数={len(extra)})")

# ================= B. 独立重放 224 笔 =================
print("== B. 全量独立重放（含/不含 bar 过滤双版） ==")
ev_prob = events.set_index(["ts_code", "date"])["prob"]
mismatch = []
reasons = {"rank_out": 0, "score_drop": 0, "horizon": 0}
diverge_bar_filter = 0

for tr in trades.itertuples():
    code = tr.ts_code
    dd = daily(code)
    # buy_prob 自 scores_final 原始事件行（去重后）
    bp = ev_prob.get((code, tr.event_date), np.nan)
    if not np.isfinite(bp):
        mismatch.append((code, str(tr.event_date.date()), "buy_prob 缺失"))
        continue

    if tr.exit_reason == "horizon":
        reasons["horizon"] += 1
        hd = cal_index[tr.exit_date] - cal_index[tr.entry_date] + 1
        prev_days = dd.index[(dd.index >= tr.entry_date) & (dd.index < tr.exit_date)]
        first_h = all(cal_index[d0] - cal_index[tr.entry_date] + 1 < H
                      for d0 in prev_days if d0 in cal_index)
        px = float(dd.loc[tr.exit_date, "close"])
        if not (hd >= H and first_h and abs(px - tr.exit_raw_price) < 1e-9):
            mismatch.append((code, str(tr.exit_date.date()),
                             f"horizon hd={hd} first_h={first_h} px={px} vs {tr.exit_raw_price}"))
        continue

    reasons[tr.exit_reason] += 1
    bar_days = dd.index[dd.index < tr.exit_date]
    if len(bar_days) == 0:
        mismatch.append((code, str(tr.exit_date.date()), "exit 前无行情日"))
        continue
    D = bar_days[-1]
    s_t = sp.get((code, D), np.nan)
    if not np.isfinite(s_t):
        mismatch.append((code, str(tr.exit_date.date()), "决策日无面板分"))
        continue

    def build_cand(use_bar_filter: bool) -> dict:
        cand: dict[str, float] = {}
        if D in by_date:
            for r in by_date[D].itertuples():
                cand[r.ts_code] = max(cand.get(r.ts_code, -np.inf), float(r.prob))
        for q in trades.itertuples():
            if q.entry_date <= D and (q.exit_date > D or
                                      (q.exit_date == D and q.exit_reason == "horizon")):
                if use_bar_filter and D not in daily(q.ts_code).index:
                    continue
                v = sp.get((q.ts_code, D), np.nan)
                if np.isfinite(v):
                    cand[q.ts_code] = max(cand.get(q.ts_code, -np.inf), float(v))
        for q in opens.itertuples():
            if q.entry_date <= D:
                if use_bar_filter and D not in daily(q.ts_code).index:
                    continue
                v = sp.get((q.ts_code, D), np.nan)
                if np.isfinite(v):
                    cand[q.ts_code] = max(cand.get(q.ts_code, -np.inf), float(v))
        return cand

    c_no_filt = build_cand(False)
    c_filt = build_cand(True)

    def judge(cand):
        ranked = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [c0 for c0, _ in ranked[:TOP_K]]
        if code not in top:
            return "rank_out"
        if s_t < bp * (1.0 - MARGIN):
            return "score_drop"
        return None

    j1, j2 = judge(c_no_filt), judge(c_filt)
    if j1 != j2:
        diverge_bar_filter += 1
    if j1 != tr.exit_reason:
        mismatch.append((code, str(tr.exit_date.date()),
                         f"reason={tr.exit_reason} replay={j1} (filt={j2})"))
        continue
    px = float(dd.loc[tr.exit_date, "open"])
    if abs(px - tr.exit_raw_price) > 1e-9:
        mismatch.append((code, str(tr.exit_date.date()),
                         f"exit_raw {tr.exit_raw_price} != open {px}"))

print(f"  重放 {len(trades)} 笔：不一致 {len(mismatch)}；出场分布 {reasons}；"
      f"bar 过滤两版判定分歧笔数={diverge_bar_filter}")
check("全量重放零不一致", len(mismatch) == 0, str(mismatch[:5]))
check("出场分布 107/111/6",
      reasons == {"rank_out": 107, "score_drop": 111, "horizon": 6}, str(reasons))
check("bar 过滤两版等价", diverge_bar_filter == 0, f"(分歧 {diverge_bar_filter})")

# ================= C. 688271.SH 专项 =================
print("== C. 688271.SH 专项复算 ==")
row = trades[(trades["ts_code"] == "688271.SH")
             & (trades["exit_date"] == pd.Timestamp("2023-06-20"))]
check("688271.SH 2023-06-20 交易存在且 rank_out",
      len(row) == 1 and row.iloc[0]["exit_reason"] == "rank_out")
if len(row) == 1:
    D = pd.Timestamp("2023-06-19")
    cand = {}
    if D in by_date:
        for r in by_date[D].itertuples():
            cand[r.ts_code] = max(cand.get(r.ts_code, -np.inf), float(r.prob))
    for q in trades.itertuples():
        if q.entry_date <= D and (q.exit_date > D or
                                  (q.exit_date == D and q.exit_reason == "horizon")):
            v = sp.get((q.ts_code, D), np.nan)
            if np.isfinite(v):
                cand[q.ts_code] = max(cand.get(q.ts_code, -np.inf), float(v))
    for q in opens.itertuples():
        if q.entry_date <= D:
            v = sp.get((q.ts_code, D), np.nan)
            if np.isfinite(v):
                cand[q.ts_code] = max(cand.get(q.ts_code, -np.inf), float(v))
    ranked = sorted(cand.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"  D=2023-06-19 候选集 top5: {[(c0, round(s, 5)) for c0, s in ranked[:5]]}")
    print(f"  688271.SH 分={cand.get('688271.SH')} 600781.SH 分={cand.get('600781.SH')}")
    check("600781.SH 在候选集且排在 688271.SH 之前",
          cand.get("600781.SH", -1) > cand.get("688271.SH", -2))
    # 反事实：剔除僵尸仓后 688271.SH 是否进前三（证明僵尸仓确实改变了排名）
    cand_cf = {k: v for k, v in cand.items() if k != "600781.SH"}
    top_cf = [c0 for c0, _ in sorted(cand_cf.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]]
    print(f"  反事实（剔除 600781.SH）top3={top_cf} 688271.SH 在内: {'688271.SH' in top_cf}")

# ================= D. 入场侧抽查 =================
print("== D. 入场侧抽查（20 笔） ==")
sample = trades.sample(20, random_state=29)
entry_ok = True
for tr in sample.itertuples():
    nxt = [d0 for d0 in calendar if d0 > tr.event_date]
    dd = daily(tr.ts_code)
    exp_entry = next((d0 for d0 in nxt if d0 in dd.index), None)
    raw_open = float(dd.loc[tr.entry_date, "open"])
    slip = tr.entry_price / raw_open - 1.0
    if not (exp_entry == tr.entry_date and 0 <= slip < 0.01):
        entry_ok = False
        print(f"  MISMATCH {tr.ts_code} {tr.event_date.date()} entry={tr.entry_date.date()} "
              f"exp={exp_entry} slip={slip:.5f}")
check("入场日=事件后首个有行情交易日 且 滑点在 [0,1%)", entry_ok)

print("REVIEW3:", "PASS" if ok else "FAIL")
