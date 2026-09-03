# T9 测试段终审独立复核记录（issue #29）

复核员：独立复核（证伪式，默认立场"结果有问题"，逐项尝试推翻）。
复核对象：`v3_pipeline/reports/test_adjudication/` 全部落盘产物、`run_test_adjudication.py`、本次三处代码修改（strategy_engine_v3.py / run_strategy_tuning.py / build_daily_score_panel.py）。
复核代码：本目录 t9_review/review{1_metrics,2_discipline,3_zombie_replay,4_engine_regression,5_panel_prereg,7_report_consistency}.py（自写，不调被复核脚本的业务函数；已随产物归档，任何人可重跑）。
允许复用的基础设施：pandas/numpy、原始日线与指数 parquet、scores_final.parquet（#27 已复核产物）、T8 归档（#28 已复核产物）。
复核日期：2026-09-03。

## 结论总览

| 复核项 | 结果 |
|---|---|
| 1 裁决数字独立重算 | PASS |
| 2 一次性纪律全程审计 | PASS |
| 3 僵尸仓修复正确性 + 全量独立重放 | PASS |
| 4 引擎修改行为保持 | PASS |
| 5 面板断言与抽查 | PASS |
| 6 预登记时序 | PASS |
| 7 报告一致性 | PASS（两处措辞瑕疵记录备查） |
| **总判** | **放行** |

## 项 1：裁决数字独立重算 —— PASS

方法：仅从 runs/test_*/equity_curve.parquet + trades.parquet 与 stock_data/index/000905.SH.parquet 自写重算。
年化=(末权益/1e6)^(365.25/自然日)-1；Sharpe=日收益(首日对本金) mean/std(ddof=1)×√252；回撤=equity/cummax-1 最小值；逐年=年界权益链式；基准同窗口 close 年化；超额=算术差；过线判定按预登记阈值自判。
结果：基准 ann/mdd 与三配置 ann/excess/max_dd/sharpe/calmar/n_trades/逐年收益/pass_threshold 全部与 adjudication.json 逐位一致（容差 1e-9，实际差全为 0）；0/3 过线判定独立成立。

## 项 2：一次性纪律 —— PASS

方法：自写解析 t9_progress.log 全部 87 行；功能实测护栏（不带 --allow-rerun 直接运行 runner，exit=1 拒绝且日志零追加）。
结果：
- 三次 RUN START：attempt#1（14:57）止于类 C 重放 1/224 FAIL、无 DONE、未落盘 adjudication.json；attempt#2（15:08）DONE；attempt#3（15:17，两轴评审后追加）带显式 --allow-rerun 登记行并 DONE。
- 三次运行三配置结论行与 final_equity 逐位一致（A13 973511.68 / B15 1228427.30 / C15 754838.24）。
- one_shot_guard（attempt=3、prior_run_starts=2、completed_rerun=true、rerun_of=attempt#2 时间戳、理由非空）与日志事实一致。
- 两次重跑理由均为工程/台账缺陷登记（重放断言缺陷、护栏口径台账补记），无"结果不理想"型重跑；t9_report.md 第 49-50 行对两次重跑均有显式登记。

## 项 3：僵尸仓修复正确性 + 全量独立重放 —— PASS

方法：僵尸仓判定口径=(ts_code, entry_date) 持仓对；独立重放为全自写实现（事件去重、在持集合重建、候选集、排名、阈值判定全部自写代码，不调 replay_score_decay_assertions）。
结果：
- 600781.SH 原始日线最后行情日=2023-06-19=入场日，入场后 0 个行情日，永不平仓成立；600306.SH（entry 2024-06-18）与 000627.SZ（entry 2025-08-12）入场后各仅 1 个行情日且无满足到期(held>=15)的行情日，永不平仓成立。三只僵尸仓均不以各自入场日出现于 trades（600781/600306 在 trades 中的记录为其他事件的正常交易，已甄别）。
- 自写独立重放 C15 全部 224 笔：零不一致；出场分布 rank_out 107 / score_drop 111 / horizon 6 与 stats 一致。
- 口径分歧检验（重要）：引擎组建候选集时跳过当日无 bar 的持仓，被复核的重放断言不做此过滤——本复核同时实现两版，224 笔判定分歧为 0（面板对无 bar 日无分数，两版在该数据上等价）。修复正确，但此隐含依赖记录备查。
- 688271.SH 专项：D=2023-06-19 候选集 top5 自算为 002816.SZ(0.54747) / 000638.SZ(0.53453) / 600781.SH(0.52837) / 688271.SH(0.52712) / 002073.SZ(0.492)，与报告数字一致；反事实剔除僵尸仓后 688271.SH 进入前三——僵尸仓挤出效应实锤。
- 入场侧抽查 20 笔：entry_date=事件后首个有行情交易日、入场滑点均在 [0,1%)。

## 项 4：引擎修改行为保持 —— PASS

方法：git diff 逐行审阅 + pytest + 独立回归对拍（用当前引擎重跑 T8 归档 val_C15 与归档产物全列对拍）。
结果：
- strategy_engine_v3.py diff 纯附加（open_positions 组装 + stats.open_at_end + parquet 落盘），无既有逻辑改动；strategy_engine.py（v1/v2）零改动（git diff 全仓仅三个文件）。
- pytest v3_pipeline/tests/test_strategy_engine_v3.py：17 passed。
- 当前引擎重跑 val_C15：trades 534/534 全 19 列逐位一致、equity 928 行数值列逐位一致、验证段 open_positions=0——附加输出不改变任何交易行为实锤。
- run_strategy_tuning.py 改动仅重放断言（项 3 已验）；build_daily_score_panel.py 改动仅缓存按窗口打标（panel_cache_{tag}），无计算逻辑改动。

## 项 5：面板断言与抽查 —— PASS

方法：读 results json；自写抽查面板。
结果：四断言（coverage/feature_anchor/prefix_spot/prob_anchor）全 PASS；面板实测 3911166 行、4375 股、(ts_code,date) 唯一、prob 值域 [0.334695,0.757452]、日期全在窗口内；事件行 prob 独立抽验 524 条（随机 500 + 全部 13 个跨池重复对）零缺失零不一致（容差 1e-12）。

## 项 6：预登记时序 —— PASS

方法：gh 查评论 created_at 与文件 mtime、日志时间戳对拍；预登记文本与 adjudication.json spec 逐字段核对。
结果：
- 预登记评论（id 5521655421）created_at=2026-09-03T14:40:57+0800；测试段面板构建日志始于 14:41:37、落盘 14:55:40；终审首跑 14:57:47。预登记先于一切测试段产物。
- 预登记文本的三配置参数（A13 tp.25/sl-.14/H12/N3；B15 v12 中心+H12 全参；C15 H15/N3/top_k3/margin0）、窗口、过线标准（>+15pp 且 Sharpe>0.5）与 adjudication.json 逐项一致。
- 旁证：panel_run.out（13:49，预登记前）为验证段面板构建日志，不构成测试段预泄。

## 项 7：报告一致性 —— PASS（两处措辞瑕疵记录备查）

方法：t9_report.md 全部数字声明（基准、裁决表 6 列×3 配置、逐年表 15 格、事件数、面板行数、出场分布、僵尸仓数、利用率区间、换手、T8 对照、56 配置声明、僵尸仓分数）与落盘产物逐项对拍。
结果：全部数字一致；T8"56 配置仅 10 个超额为正"经 tuning_class_*.csv 独立清点证实（A1+B1+C8=10）。
两处措辞瑕疵（不影响任何数字与结论，记录备查）：
1. 第 55 行括注"4375 股 × 931 交易日"=4073125 与实际行数 3911166 不等（面板非全交叉，新股/停牌日无行）；主数字"391 万行"正确。
2. 第 32 行"仓位利用率 0.80-0.85"：A13 精确值 0.85153 严格大于 0.85，两位舍入后为 0.85，擦边。

## 总判：放行

七项复核全部通过。0/3 过线结论在独立重算下逐位复现；一次性纪律、预登记时序、僵尸仓修复、引擎行为保持均经独立实证；报告数字与产物一致。
唯二记录备查项为报告措辞瑕疵（面板行数括注、利用率区间擦边），建议后续修订时顺手修正，不构成打回理由。
