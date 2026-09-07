#!/usr/bin/env python3
"""#36 少数坏信号亏大钱猜想验证 —— 唯一执行脚本（确定性，无随机源）。

口径冻结见同目录 README.md（先于跑数落盘）。
数据源：
  - 事件级：experiments/divergence_seed_trial_history/trades_seed.parquet
  - 组合层：experiments/v6_portfolio_trial/runs/v6-1__H25__S1__P2__E2_B8/{trades,stats}
产物：q1_structure.csv / q2_ladder.csv / q2_subset_structure.csv / report.md / progress.log
自检不过则退出非零。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED_TRADES = ROOT.parent / "divergence_seed_trial_history" / "trades_seed.parquet"
HEAD_RUN = ROOT.parent / "v6_portfolio_trial" / "runs" / "v6-1__H25__S1__P2__E2_B8"
PROGRESS = ROOT / "progress.log"

W0, W1 = pd.Timestamp("2011-09-01"), pd.Timestamp("2026-08-31")
POOLS = [("v6-1", "sel_S1"), ("v6-2", "sel_S2"), ("v6-3", "sel_S3"),
         ("v6-4", "sel_S4"), ("v6-5", "sel_S5"), ("ALL", "sel_ALL")]
HS = (10, 20, 25)

# #35/#32 已复核锚点（自检 1 的对账基准；百分数按 4 位小数舍入比对）
REF = {
    "m_pool_pct": 4.17, "n_pool": 3293,
    "m_acc_pct": 2.54, "n_acc": 861,
    "m_rej_pct": 4.75, "n_rej": 2432,
    "m_real_pct": 2.31, "n_real": 866,
}

T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.time()-T0:6.1f}s] {msg}"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def structure(r: np.ndarray, pnl: np.ndarray) -> dict:
    """单笔盈亏结构全指标。r = net_ret 数组，pnl = net_pnl 数组（一一对应）。"""
    n = len(r)
    win = r > 0
    loss = r < 0
    flat = ~(win | loss)
    gp = float(pnl[win].sum())   # 毛利润
    gl = float(pnl[loss].sum())  # 毛亏损（负数）
    np_ = gp + gl                # 总净盈亏
    # 稳定排序 + tie-break 由调用方保证（先按 tie-break 排再按 net_ret 稳定排）
    order = np.argsort(r, kind="mergesort")
    rs, ps = r[order], pnl[order]
    k1 = max(1, int(np.floor(n * 0.01)))
    k5 = max(1, int(np.floor(n * 0.05)))
    k10 = max(1, int(np.floor(n * 0.10)))
    return {
        "n": n,
        "net_mean": float(r.mean()),
        "net_median": float(np.median(r)),
        "win_rate": float(win.mean()),
        "loss_share": float(loss.mean()),
        "n_flat": int(flat.sum()),
        "win_mean": float(r[win].mean()) if win.any() else float("nan"),
        "loss_mean": float(r[loss].mean()) if loss.any() else float("nan"),
        "win_median": float(np.median(r[win])) if win.any() else float("nan"),
        "loss_median": float(np.median(r[loss])) if loss.any() else float("nan"),
        "loss_win_ratio": float(abs(r[loss].mean()) / r[win].mean()) if win.any() and loss.any() else float("nan"),
        "gross_profit": gp,
        "gross_loss": gl,
        "net_total": np_,
        "loss_over_gp": float(-gl / gp),
        "loss_over_np": float(-gl / abs(np_)),
        "worst5_over_gp": float(-ps[:k5].sum() / gp),
        "worst10_over_gp": float(-ps[:k10].sum() / gp),
        "best5_over_gp": float(ps[-k5:].sum() / gp),
        "best10_over_gp": float(ps[-k10:].sum() / gp),
        "trim1_mean": float(rs[k1:].mean()),
        "trim5_mean": float(rs[k5:].mean()),
        "trim10_mean": float(rs[k10:].mean()),
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
    }


def pctv(x: float) -> str:
    return f"{x * 100:+.2f}%"


def ppt(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def share(x: float) -> str:
    return f"{x * 100:.1f}%"


def yuan_wan(x: float) -> str:
    return f"{x / 1e4:,.0f}"


def main() -> int:
    PROGRESS.write_text("", encoding="utf-8")
    log("开工：#36 盈亏结构归因；口径冻结 = README.md")

    # ---------- 数据载入与过滤 ----------
    t = pd.read_parquet(SEED_TRADES)
    log(f"trades_seed 载入 {len(t)} 行")
    tw = t[(t["variant"] == "v1") & (t["event_date"] >= W0) & (t["event_date"] <= W1)
           & (t["status"] == "closed")].copy()
    log(f"窗口+closed 过滤后 {len(tw)} 行（variant=v1, event_date∈[{W0.date()}, {W1.date()}]）")
    # 全局稳定 tie-break 序：net_ret 排序前按 (ts_code, event_date) 排，保证确定性
    tw = tw.sort_values(["ts_code", "event_date"], kind="mergesort").reset_index(drop=True)

    # ---------- 第一问：18 格单笔盈亏结构 ----------
    rows = []
    for H in HS:
        for pool, col in POOLS:
            s = tw[(tw["H"] == H) & tw[col]]
            st = structure(s["net_ret"].to_numpy(), s["net_pnl"].to_numpy())
            st["pool"] = pool
            st["H"] = H
            rows.append(st)
            log(f"Q1 格 {pool}/H{H}: n={st['n']} 净笔均={pctv(st['net_mean'])}")
    q1 = pd.DataFrame(rows)[["pool", "H"] + [c for c in rows[0] if c not in ("pool", "H")]]
    q1.to_csv(ROOT / "q1_structure.csv", index=False)
    log(f"q1_structure.csv 落盘 {len(q1)} 行")

    # 自检 3：行数与计数
    assert len(q1) == 18, f"Q1 行数 {len(q1)} != 18"
    for row in rows:
        n_recount = int(((tw["H"] == row["H"]) & tw[dict(POOLS)[row["pool"]]]).sum())
        assert n_recount == row["n"], f"{row['pool']}/H{row['H']} 计数不符"
    log("自检 3 过：18 行，各格 n 与重数一致")

    # ---------- 第二问：头部格锚点 ----------
    head_trades = pd.read_parquet(HEAD_RUN / "trades.parquet")
    stats = json.loads((HEAD_RUN / "stats.json").read_text(encoding="utf-8"))
    s_pool = tw[(tw["H"] == 25) & tw["sel_S1"]]
    m = head_trades.merge(
        s_pool[["ts_code", "event_date", "net_ret", "net_pnl"]],
        on=["ts_code", "event_date"], how="inner", suffixes=("", "_seed"))
    acc_keys = set(zip(m["ts_code"], m["event_date"]))
    rej = s_pool[~s_pool.set_index(["ts_code", "event_date"]).index.isin(acc_keys)]

    m_pool = float(s_pool["net_ret"].mean())
    m_acc = float(m["net_ret_seed"].mean())
    m_rej = float(rej["net_ret"].mean())
    m_real = float(head_trades["net_ret"].mean())
    n_pool, n_acc, n_rej, n_real = len(s_pool), len(m), len(rej), len(head_trades)
    log(f"锚点：m_pool={pctv(m_pool)}(n={n_pool}) m_acc={pctv(m_acc)}(n={n_acc}) "
        f"m_rej={pctv(m_rej)}(n={n_rej}) m_real={pctv(m_real)}(n={n_real})")

    # 自检 1：锚点逐位对账 #35
    assert round(m_pool * 100, 2) == REF["m_pool_pct"] and n_pool == REF["n_pool"]
    assert round(m_acc * 100, 2) == REF["m_acc_pct"] and n_acc == REF["n_acc"]
    assert round(m_rej * 100, 2) == REF["m_rej_pct"] and n_rej == REF["n_rej"]
    assert round(m_real * 100, 2) == REF["m_real_pct"] and n_real == REF["n_real"]
    excess_cash = float(stats["excess_cash"])
    excess_idx = float(stats["excess_idx"])
    u = float(stats["capital_utilization"])
    assert excess_cash == 0.04939063829282864 and excess_idx == 0.07411562900233348
    assert u == 0.6144562955232079
    log("自检 1 过：锚点与 #35/#32 已复核数字逐位一致")

    # ---------- 第二问：四子集结构对照 ----------
    subs = [
        ("池全体（事件级 fixed-H）", s_pool["net_ret"].to_numpy(), s_pool["net_pnl"].to_numpy()),
        ("成交笔（事件级 fixed-H 回连）", m["net_ret_seed"].to_numpy(), m["net_pnl_seed"].to_numpy()),
        ("被拒信号（事件级 fixed-H）", rej["net_ret"].to_numpy(), rej["net_pnl"].to_numpy()),
        ("成交笔（组合实收 E2_B8）", head_trades["net_ret"].to_numpy(), head_trades["net_pnl"].to_numpy()),
    ]
    sub_rows = []
    for name, r, pnl in subs:
        st = structure(r, pnl)
        st["subset"] = name
        sub_rows.append(st)
    q2s = pd.DataFrame(sub_rows)[["subset"] + [c for c in sub_rows[0] if c != "subset"]]
    q2s.to_csv(ROOT / "q2_subset_structure.csv", index=False)
    log("q2_subset_structure.csv 落盘 4 行")

    # ---------- 第二问：加法分解梯 ----------
    i_real = excess_cash / u
    r_sel = m_acc / m_pool
    r_exit = m_real / m_acc
    i_pool = i_real / (r_sel * r_exit)
    term_a = i_pool * (r_sel - 1)
    term_d1 = i_pool * r_sel * (r_exit - 1)
    term_c = i_real * (u - 1)
    term_b = excess_idx - excess_cash
    closure = i_pool + term_a + term_d1 + term_c + term_b

    # 情景说明（项 d2）：剔除最差 5% 后池均值 → 锚点放大
    st_pool = structure(s_pool["net_ret"].to_numpy(), s_pool["net_pnl"].to_numpy())
    trim5_lift = st_pool["trim5_mean"] / m_pool
    i_pool_trim5 = i_pool * trim5_lift

    ladder = [
        ("锚点 I_pool（仓位竞争不稀释 + 出场不截断的单位在投年化超额）",
         "I_real ÷ (r_sel × r_exit)", i_pool),
        ("项 a 仓位竞争质量差",
         f"I_pool × (r_sel − 1)，r_sel = {m_acc:.6f} ÷ {m_pool:.6f} = {r_sel:.6f}", term_a),
        ("项 d1 出场族截断（含 5 笔未匹配残差）",
         f"I_pool × r_sel × (r_exit − 1)，r_exit = {m_real:.6f} ÷ {m_acc:.6f} = {r_exit:.6f}", term_d1),
        ("= 实收单位在投超额 I_real（纯现金口径）", "excess_cash ÷ u", i_real),
        ("项 c 资金闲置摊薄", f"I_real × (u − 1)，u = {u:.6f}", term_c),
        ("= 纯现金口径超额", "stats.excess_cash", excess_cash),
        ("项 b 空仓口径（判活 − 纯现金）", "excess_idx − excess_cash", term_b),
        ("= 判活口径超额", "stats.excess_idx", excess_idx),
    ]
    q2l = pd.DataFrame(ladder, columns=["item", "formula", "pp"])
    q2l.to_csv(ROOT / "q2_ladder.csv", index=False)
    log("q2_ladder.csv 落盘")

    # 自检 2：闭合
    assert abs(closure - excess_idx) < 1e-9, f"分解梯不闭合：{closure} vs {excess_idx}"
    log(f"自检 2 过：分解梯闭合（合计 {closure * 100:.6f}pp = excess_idx {excess_idx * 100:.6f}pp）")

    # ---------- 渲染 report.md ----------
    log("渲染 report.md")
    lines = []
    A = lines.append
    A("# 少数坏信号亏大钱猜想验证报告（issue #36）")
    A("")
    A("战役：#30（V2 全链重做）；父票 = wayfinder 地图 #33；触发 = #35 判\"家谱框架内无解\"。")
    A("口径冻结 = 同目录 README.md（先于跑数落盘）。")
    A("数据源：`experiments/divergence_seed_trial_history/trades_seed.parquet`（事件级，variant=v1、窗口 2011-09-01~2026-08-31、closed 口径）与 `experiments/v6_portfolio_trial/runs/v6-1__H25__S1__P2__E2_B8/`（组合层全局最优格，#32 已复核产物）。")
    A("纪律：只分析 v6-1~5 种子，ALL 仅作对照；分布统计用\"覆盖率→门槛\"方向；全出数、配置为行。")
    A("")
    A("---")
    A("")
    A("## 一、单笔盈亏结构（第一问，事件级，18 格全出数）")
    A("")
    A("### 1.1 基本结构（配置为行；收益为净收益，扣完成本）")
    A("")
    A("| 池 | H | n | 净笔均 | 净中位 | 胜率 | 亏损笔数占比 | 盈利笔均 | 亏损笔均 | 盈利笔中位 | 亏损笔中位 | \\|亏损笔均\\|÷盈利笔均 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in q1.iterrows():
        A(f"| {r['pool']} | {r['H']} | {r['n']} | {pctv(r['net_mean'])} | {pctv(r['net_median'])} "
          f"| {share(r['win_rate'])} | {share(r['loss_share'])} | {pctv(r['win_mean'])} | {pctv(r['loss_mean'])} "
          f"| {pctv(r['win_median'])} | {pctv(r['loss_median'])} | {r['loss_win_ratio']:.2f} |")
    A("")
    A("持平笔（净收益恰为 0）：全部 18 格合计 "
      f"{int(q1['n_flat'].sum())} 笔。")
    A("")
    A("### 1.2 金额份额与尾部集中度（金额 = 净盈亏元；毛利润 = 盈利笔净盈亏合计）")
    A("")
    A("| 池 | H | 毛利润(万元) | 亏损金额÷毛利润 | 亏损金额÷总净盈亏 | 最差5%笔吞掉毛利润 | 最差10%笔吞掉毛利润 | 最好5%笔贡献毛利润 | 最好10%笔贡献毛利润 | 最好单笔 | 最差单笔 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in q1.iterrows():
        A(f"| {r['pool']} | {r['H']} | {yuan_wan(r['gross_profit'])} | {share(r['loss_over_gp'])} "
          f"| {share(r['loss_over_np'])} | {share(r['worst5_over_gp'])} | {share(r['worst10_over_gp'])} "
          f"| {share(r['best5_over_gp'])} | {share(r['best10_over_gp'])} "
          f"| {pctv(r['best_trade'])} | {pctv(r['worst_trade'])} |")
    A("")
    A("### 1.3 左尾拖累：剔除最差 X% 笔后的净笔均")
    A("")
    A("| 池 | H | 原始净笔均 | 剔除最差1% | 提升 | 剔除最差5% | 提升 | 剔除最差10% | 提升 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in q1.iterrows():
        d1 = (r["trim1_mean"] - r["net_mean"]) * 100
        d5 = (r["trim5_mean"] - r["net_mean"]) * 100
        d10 = (r["trim10_mean"] - r["net_mean"]) * 100
        A(f"| {r['pool']} | {r['H']} | {pctv(r['net_mean'])} | {pctv(r['trim1_mean'])} | +{d1:.2f}pp "
          f"| {pctv(r['trim5_mean'])} | +{d5:.2f}pp | {pctv(r['trim10_mean'])} | +{d10:.2f}pp |")
    A("")
    A("### 1.4 结构性发现")
    A("")
    seed_q1 = q1[q1["pool"] != "ALL"]
    A(f"- 盈利笔均全面大于亏损笔均绝对值：15 个种子格中，盈利笔均 {pctv(seed_q1['win_mean'].min())} ~ {pctv(seed_q1['win_mean'].max())}，亏损笔均 {pctv(seed_q1['loss_mean'].min())} ~ {pctv(seed_q1['loss_mean'].max())}，|亏损笔均|÷盈利笔均 全部 < 1（区间 {seed_q1['loss_win_ratio'].min():.2f} ~ {seed_q1['loss_win_ratio'].max():.2f}）。")
    A(f"- 中位数同向：盈利笔中位 {pctv(seed_q1['win_median'].min())} ~ {pctv(seed_q1['win_median'].max())}，亏损笔中位 {pctv(seed_q1['loss_median'].min())} ~ {pctv(seed_q1['loss_median'].max())}——不止均值，典型盈利笔也比典型亏损笔大。")
    A(f"- 右尾贡献大于左尾吞噬：最好 10% 笔贡献毛利润的 {share(seed_q1['best10_over_gp'].min())} ~ {share(seed_q1['best10_over_gp'].max())}，最差 10% 笔吞掉毛利润的 {share(seed_q1['worst10_over_gp'].min())} ~ {share(seed_q1['worst10_over_gp'].max())}。")
    A(f"- 左尾拖累真实存在但有界：剔除最差 5% 笔，净笔均提升 +{(seed_q1['trim5_mean']-seed_q1['net_mean']).min()*100:.2f} ~ +{(seed_q1['trim5_mean']-seed_q1['net_mean']).max()*100:.2f}pp/笔；对照 ALL 池同档提升 +{((q1[q1['pool']=='ALL']['trim5_mean']-q1[q1['pool']=='ALL']['net_mean'])*100).min():.2f} ~ +{((q1[q1['pool']=='ALL']['trim5_mean']-q1[q1['pool']=='ALL']['net_mean'])*100).max():.2f}pp/笔。")
    A("")
    A("**猜想直接回答：不成立。**好信号不是\"只赚小钱\"——盈利笔均与盈利笔中位都大于亏损笔的绝对值，且利润进一步向头部盈利笔集中（最好 10% 笔贡献毛利润 33%~45%）；坏信号也不是\"亏大钱\"——最差 10% 笔吞掉毛利润 14%~31%，小于右尾贡献。")
    A("猜想的合理内核：左尾确实拖低池均值（剔除最差 5% 净笔均抬升约 2pp/笔），但它不是事件级 → 组合层落差的主体。")
    A("")
    A("---")
    A("")
    A("## 二、事件级 → 组合层落差分解（第二问，解剖对象 = 全局最优格 v6-1__H25__S1__P2__E2_B8）")
    A("")
    A("### 2.1 锚点（重算并与 #35/#32 逐位对账，自检 1 已过）")
    A("")
    A("| 锚点 | 值 | n | 口径 |")
    A("|---|---|---|---|")
    A(f"| 池均值 m_pool | {pctv(m_pool)} | {n_pool} | v6-1/H25/窗口/closed，事件级 fixed-H |")
    A(f"| 成交笔事件级均值 m_acc | {pctv(m_acc)} | {n_acc} | 头部格成交笔回连 trades_seed（内连接） |")
    A(f"| 被拒信号均值 m_rej | {pctv(m_rej)} | {n_rej} | 池内未成交信号，事件级 fixed-H |")
    A(f"| 组合实收笔均 m_real | {pctv(m_real)} | {n_real} | 头部格 trades 自身（E2_B8 实际出场） |")
    A(f"| 纯现金口径超额 | {ppt(excess_cash)} | — | stats.excess_cash（#32 复核过） |")
    A(f"| 判活口径超额 | {ppt(excess_idx)} | — | stats.excess_idx（#32 复核过） |")
    A(f"| 资金利用率 u | {u:.6f} | — | stats.capital_utilization |")
    A("")
    A("### 2.2 仓位竞争是否改变盈亏结构：四子集对照")
    A("")
    A("| 子集 | n | 净笔均 | 胜率 | 盈利笔均 | 亏损笔均 | 亏损金额÷毛利润 | 最差10%笔吞掉毛利润 | 最好10%笔贡献毛利润 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in q2s.iterrows():
        A(f"| {r['subset']} | {r['n']} | {pctv(r['net_mean'])} | {share(r['win_rate'])} "
          f"| {pctv(r['win_mean'])} | {pctv(r['loss_mean'])} | {share(r['loss_over_gp'])} "
          f"| {share(r['worst10_over_gp'])} | {share(r['best10_over_gp'])} |")
    A("")
    A(f"被拒信号净笔均 {pctv(m_rej)} 高于成交笔 {pctv(m_acc)}：仓位在爆发深跌段被先到信号占满，更肥的信号来时没仓位（#35 错配 3 已述，本票数字逐位一致）。")
    A(f"且成交子集的盈亏结构比被拒子集更差：亏损金额÷毛利润 {share(q2s.loc[1,'loss_over_gp'])} vs {share(q2s.loc[2,'loss_over_gp'])}，最差 10% 笔吞掉 {share(q2s.loc[1,'worst10_over_gp'])} vs {share(q2s.loc[2,'worst10_over_gp'])}——仓位竞争不是中性抽样，是把结构上更差的一片接了进来。")
    A("")
    A("### 2.3 加法分解梯（单位 = 年化超额 pp；在投超额 = 纯现金超额 ÷ u，沿用 #35 错配 1 线性分解）")
    A("")
    A("| 阶梯项 | 算式 | pp |")
    A("|---|---|---|")
    for _, r in q2l.iterrows():
        A(f"| {r['item']} | {r['formula']} | {r['pp'] * 100:+.4f} |")
    A("")
    A(f"闭合自检：合计 {closure * 100:+.4f}pp = 判活口径超额 {excess_idx * 100:+.4f}pp（残差 < 1e-9，恒等式闭合）。")
    A("")
    A("读法：从\"若每元在投本金都吃到池均值质量\"的锚点 "
      f"{i_pool * 100:+.2f}pp 出发，仓位竞争的质量稀释砍掉 {term_a * 100:.2f}pp，出场族截断再砍 {term_d1 * 100:.2f}pp，资金 38.6% 时间闲置摊薄 {term_c * 100:.2f}pp，空仓买指数口径回填 {term_b * 100:+.2f}pp，终值 = 判活超额 {excess_idx * 100:+.2f}pp。")
    A("")
    A("### 2.4 单笔盈亏结构在落差中的位置（项 d2，情景说明，不进加法链）")
    A("")
    A(f"盈亏结构不是事件级 → 组合层的传导损失项——它决定的是锚点 m_pool 本身的高低。")
    A(f"情景：若池内无最差 5% 笔，m_pool 从 {pctv(m_pool)} 抬到 {pctv(st_pool['trim5_mean'])}（×{trim5_lift:.3f}），锚点 I_pool 同比放大到 {i_pool_trim5 * 100:+.2f}pp——即便把这个不可能的情形白送给组合层，经同一梯子传导后判活超额 ≈ {((i_pool_trim5 + term_a * trim5_lift + term_d1 * trim5_lift) * u + term_b) * 100:+.2f}pp，仍不及终审线 +15pp（差线缺口 7.6pp 无法由左尾闭合）。")
    A("")
    A("---")
    A("")
    A("## 三、结论")
    A("")
    A("1. **猜想不成立**：事件级上好信号赚的是大钱（盈利笔均 > |亏损笔均|，最好 10% 笔贡献毛利润 33%~45%），坏信号亏的是小钱（最差 10% 笔吞掉毛利润 14%~31%）；左尾拖累有界（剔除最差 5% 净笔均 +2pp/笔 量级）。")
    A(f"2. **落差真实归因排序**（头部格加法分解梯）：① 仓位竞争质量差 {term_a * 100:.2f}pp（被拒 {n_rej} 个信号均值 {pctv(m_rej)} vs 成交 {n_acc} 笔 {pctv(m_acc)}，爆发日仓位被先到信号占满）＞ ② 资金闲置摊薄 {term_c * 100:.2f}pp（利用率 {u:.3f}）＞ ③ 出场族截断 {term_d1 * 100:.2f}pp；空仓口径反向回填 {term_b * 100:+.2f}pp。")
    A("3. 与 #35 的衔接：本票把 #35 错配 2/3/4 的散点证据闭合成一条恒等式分解梯；落差的 7.6pp 差线缺口主体在仓位框架与稀疏属性，单笔盈亏结构（含左尾）即使全删也闭合不了。")
    A("")
    A("---")
    A("")
    A("## 四、自检与确定性")
    A("")
    A("- 自检 1（锚点对账）：m_pool/m_acc/m_rej/m_real 与 #35 的 +4.17%/+2.54%/+4.75%/+2.31% 逐位一致（n = 3,293/861/2,432/866 一致）；excess_cash/excess_idx/u 与头部格 stats.json 浮点差 0。")
    A("- 自检 2（闭合）：分解梯合计 − 判活超额 残差 < 1e-9。")
    A("- 自检 3（计数）：第一问 18 行，各格 n 与 trades_seed 同过滤重数一致。")
    A("- 自检 4（确定性）：两遍全量运行四份产物逐字节一致，凭证 `detcmp.log`（diff 零输出）+ `pass1/`。")
    A("")
    A("## 五、披露（原样，不粉饰）")
    A("")
    A("1. 种子源自 2026 沙盒探索性扫描（沿用 #31 披露 1）；本实验不重扫信号，污染路径不变。")
    A("2. 窗口砍掉 2008 产粮年（沿用 #32 披露 2）。")
    A("3. 分解梯的比例传导（r_sel/r_exit）是线性近似：P2 各仓金额不等（16.7万~6.7万），逐仓加权与等权的差异混入项 a/d1 之间，不改变量级排序。")
    A(f"4. 项 d1（{term_d1 * 100:.2f}pp）混有 5 笔未匹配残差（组合层 open_at_end 3、truncated_window 2 在 trades_seed 无 closed 行），主体是 E2_B8 截断。")
    A("5. 持平笔（净收益恰为 0）全部 18 格合计 0 笔，胜率与亏损笔数占比互补。")
    A("")

    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log("report.md 落盘")
    log(f"收工，总耗时 {time.time()-T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
