#!/usr/bin/env python3
"""T9 复核项 2：一次性纪律全程审计（自写日志解析，不复用被复核代码）。

核查点：
  a. t9_progress.log 的 RUN START 块数与每块三配置结论行；
  b. attempt#1 无 RUN DONE、止于类 C 重放 1/224 FAIL（断言缺陷，未落盘 adjudication.json）；
  c. attempt#2 完整 DONE；attempt#3 带 --allow-rerun 登记行并 DONE；
  d. 三次运行三配置市场结果（final_equity/ann/excess/sharpe/trades）逐位一致；
  e. adjudication.json one_shot_guard 字段与日志事实一致
     （attempt=3、prior_run_starts=2、rerun_of=attempt#2 run_ts、理由已登记）；
  f. 每次重跑原因均为工程/台账缺陷登记，无"结果不理想"型重跑。
"""
import json
import os
import re

REPO = "/home/karl/repos/personal/stock_qt_nd"
LOG = os.path.join(REPO, "v3_pipeline/reports/test_adjudication/t9_progress.log")
ADJ = os.path.join(REPO, "v3_pipeline/reports/test_adjudication/adjudication.json")


def main():
    lines = open(LOG, encoding="utf-8").read().splitlines()
    starts = [i for i, l in enumerate(lines) if "T9 RUN START" in l]
    print(f"RUN START 块数={len(starts)} (行号 {starts})")
    ok = True

    bounds = starts + [len(lines)]
    blocks = [lines[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    pat_cfg = re.compile(r"(A13|B15|C15) done .*ann=([+-]\d+\.\d+) excess=([+-]\d+\.\d+) "
                         r"sharpe=([+-]?\d+\.\d+) trades=(\d+)")
    pat_eq = re.compile(r"run_backtest_v3 done: class=(\w+) .*trades=(\d+) "
                        r"final_equity=([\d.]+)")
    res = []
    for bi, blk in enumerate(blocks):
        cfg = {m.group(1): m.groups()[1:] for l in blk for m in [pat_cfg.search(l)] if m}
        fin = {m.group(1): (m.group(2), m.group(3)) for l in blk for m in [pat_eq.search(l)] if m}
        replay = [l for l in blk if "逐笔重放" in l]
        done = any("RUN DONE" in l for l in blk)
        res.append(dict(cfg=cfg, fin=fin, replay=replay, done=done))
    # 护栏旁路行在 runner 中先于对应 RUN START 打印：定位为"紧邻某 RUN START 之前"
    bypass_idx = [i for i, l in enumerate(lines) if "护栏旁路" in l]
    bypass_before = {next(s for s in starts if s > i) for i in bypass_idx}
    for bi, s in enumerate(starts):
        res[bi]["bypass"] = 1 if s in bypass_before else 0
    for bi, r in enumerate(res):
        print(f"attempt#{bi+1}: DONE={r['done']} bypass={r['bypass']} 配置行={sorted(r['cfg'])} "
              f"重放={[' '.join(x.split()[2:]) for x in r['replay']]}")
        for name in sorted(r['cfg']):
            print(f"    {name}: {r['cfg'][name]} final_eq({[k for k in r['fin']]})")

    # b/c. 结构断言
    struct_ok = (len(blocks) == 3
                 and not res[0]["done"] and res[1]["done"] and res[2]["done"]
                 and res[0]["bypass"] == 0 and res[1]["bypass"] == 0
                 and res[2]["bypass"] == 1)
    ok &= struct_ok
    print(f"结构（#1 FAIL 无 DONE 无旁路 / #2 DONE 无旁路 / #3 DONE 有旁路登记）: {struct_ok}")

    # f. 重放轨迹：#1 FAIL(1/224)，#2/#3 PASS(0)
    replay_ok = ("不一致 1" in res[0]["replay"][-1] and "FAIL" in res[0]["replay"][-1]
                 and "不一致 0" in res[1]["replay"][-1] and "PASS" in res[1]["replay"][-1]
                 and "不一致 0" in res[2]["replay"][-1] and "PASS" in res[2]["replay"][-1])
    ok &= replay_ok
    print(f"重放轨迹 FAIL(1) -> PASS(0) -> PASS(0): {replay_ok}")

    # d. 三次运行市场结果逐位一致
    for name in ("A13", "B15", "C15"):
        vals = [r["cfg"].get(name) for r in res]
        same = vals[0] == vals[1] == vals[2] and vals[0] is not None
        ok &= same
        print(f"  {name} 三次运行结论行逐位一致: {same}")
    for cls in ("fixed_tp_sl", "vol_adaptive", "score_decay"):
        vals = [r["fin"].get(cls) for r in res]
        same = vals[0] == vals[1] == vals[2] and vals[0] is not None
        ok &= same
        print(f"  {cls} final_equity 三次逐位一致: {same} {vals[0]}")

    # e. one_shot_guard 字段
    adj = json.load(open(ADJ))
    g = adj["one_shot_guard"]
    g_ok = (g.get("attempt") == 3 and g.get("prior_run_starts") == 2
            and g.get("completed_rerun") is True
            and isinstance(g.get("rerun_reason"), str) and len(g["rerun_reason"]) > 0
            and g.get("rerun_of") == "2026-09-03T15:08:24+0800"
            and adj["run_ts"] == "2026-09-03T15:17:42+0800")
    ok &= g_ok
    print(f"  one_shot_guard={json.dumps(g, ensure_ascii=False)}")
    print(f"  护栏字段与日志事实一致: {g_ok}")

    print("REVIEW2:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
