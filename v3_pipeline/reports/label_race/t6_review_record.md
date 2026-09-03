# T6 标签赛独立复核记录（issue #26）

复核员：独立复核（证伪式，默认立场"结果有问题"，逐项尝试推翻）。
复核对象：`v3_pipeline/reports/label_race/` 全部落盘产物。
复核代码：/tmp/t6_review/review{1_adjudication,2_rerun,3_labels,4_discipline}.py（自写，不调 label_race.adjudicate / select_best_config / build_merged_master）。
复核日期：2026-09-03。

## 结论总览

| 复核项 | 结果 |
|---|---|
| 1 裁决重算 | PASS |
| 2 独立重跑抽验（正赛当选配置） | PASS |
| 3 标签抽验 | PASS |
| 4 纪律核查（a~e） | PASS |
| 5 复现断言产物核查 | PASS |
| **总判** | **放行** |

## 项 1：裁决重算 —— PASS

方法：自写代码读 38 个 metrics_{pool}_{candidate}.csv（merged/main 各 19 候选），逐文件断言恰 36 行、config_id 0..35、每行四项超参与自写预登记网格（num_leaves{15,31,63}×min_data_in_leaf{50,100,200}×learning_rate{0.05,0.10}×feature_fraction{0.6,0.8}）逐位一致，无重复无缺失。
自写选配置规则（val_precision_at_5_dayavg 降 → val_average_precision 降 → config_id 升，mergesort 稳定序）与自写总裁决规则（候选序 hit→cur 九视野→open_exec 九视野，中位数约束），与 summary_*.csv、adjudication_*.json 全字段对比。

结果：
- 38 文件 × 36 行 × 4 超参逐行核对全部通过。
- 38 个候选当选 config_id 与两项 val 指标与 summary 完全一致。
- 平局裁决被真实触发且规则适用正确：merged 池 cur_pos_45d 头部 2 行 dayavg 并列（AP 分出胜负）；main 池 7 个候选头部 3 行 dayavg 并列（其中 6 个 AP 亦并列，落到 config_id 小者）。
- 正赛（merged）裁决重算：winner=hit_N20_k2.0，val_precision_at_5_dayavg=0.5750214224507284，val_average_precision=0.622474953133874，19 候选 val AP 中位数=0.5794460048968032，0.6225≥0.5794 约束通过——与 adjudication_merged.json 逐字段一致。
- 对照（main）裁决重算：winner=hit_N20_k2.0（config_id=0），dayavg=0.5164556962025316，AP=0.6282588395379161≥中位数 0.5599044253188422——与 adjudication_main.json 一致，role 标注"对照（不参与正式裁决）"无误。
- 正赛 winner 优势无平局歧义：次优候选 open_exec_pos_10d dayavg=0.4906，差距 8.4pp。

## 项 2：独立重跑抽验（hit_N20_k2.0 merged config_id=13）—— PASS

方法：自写代码从 master_main.parquet + master_backup.parquet 重建合并池（行并集不去重、event_id 加 main_/backup_ 前缀、downtrend/hammer_signal 归一可空 Int8、pool 溯源列），与落盘 master_merged.parquet 对照行数/列集/event_id 集合一致；标签表自写拼接后按 [pool,ts_code,date] 左连（validate one_to_one）；NaN 标签剔除后 train/val 切分，train 按 (date,ts_code,event_id) mergesort。
走 train_eval_pipeline 公开函数：五折 OOF → SquaredLogitCalibrator[p,p²] → 终模（轮数=五折 best_iteration 均值取整）→ evaluate_segment。
对比口径：指标行用 csv 模块 + float() 精确解析后逐位 ==（容差 0）。
注：曾发现 pandas C 解析器对个别 double 存在 1 ulp 解析偏差（calib_intercept 原文 0.00467052561134235 被 read_csv 解析成相邻 double），改用 Python float() 解析后消除——属解析器伪差异，非计算差异。

结果：
- 合并池 45140 行 × 2072 列，特征 2060 列。
- config_id=13（nl=31, md=50, lr=0.05, ff=0.8）重算 best_iters=[37,60,54,89,34]、终模 55 轮、校准系数/截距、train_oof 与 val 全部 19 个指标字段与 metrics_merged_hit_N20_k2.0.csv 第 13 行逐位一致。
- 关键数字复算值：val_average_precision=0.622474953133874，val_precision_at_5_dayavg=0.5750214224507284，train_oof_n_events=20795，val_n_events=10219。

## 项 3：标签抽验 —— PASS

方法：自写口径（不调用 label_candidates）：cur_pos_10d=1{close[t+11]/close[t]−1>0}，open_exec_pos_10d=1{close[t+11]/open[t+1]−1>0}，t+k 为该票日线升序第 k 个后续 bar，越界为 NaN。
抽验 labels_race_main 3 随机事件 + 1 尾部事件、labels_race_backup 3 随机事件 + 1 尾部事件，共 8 事件逐位对比；并系统性核对全部 NaN 行成因与非 NaN 抽样反查。

结果：
- 8 事件手工重算与 labels_race_*.parquet 逐位一致（含 2 个尾部事件双双 NaN 对 NaN）。
- NaN 未被吞成 0：main 池 cur_pos_10d 全部 1 个 NaN 行、backup 池全部 251 个 NaN 行逐一核对，均为事件后可用 bar ≤11（尾部截断）所致；反向抽 500 非 NaN 行（每池）确认 bar 充足。
- 附带发现（良性，已解释）：标签表各池比主表多 4/26 行（共 30 行），全部为 000001.SH/399001.SZ 指数事件——主表构建期已剔指数伪股，标签表按原始事件表构建，左连后这 30 行不进模型，主表全部行标签零缺失。

## 项 4：纪律核查 —— PASS（a~e 全过）

(a) test 段零触碰：grep 全部 csv/json/md/log/out 产物，"test" 字样共 10 处命中，逐条甄别全部为段计数台账（train/val/test 行数）与"零触碰"声明；无任何 test 段指标（average_precision/precision_at/base_rate/n_events/oof）与逐行统计；38 个 metrics CSV 与 2 个 summary CSV 表头均无 test 字样。
(b) 预登记时戳：gh api 查得置顶评论 created_at=2026-09-02T21:10:15Z；最早 metrics 文件 metrics_merged_hit_N20_k2.0.csv mtime=2026-09-02T21:18:58Z，预登记早 523 秒，全部 38 个指标文件均在预登记之后。
另注：labels_race_*.parquet（21:09Z）早于预登记约 1 分钟，与预登记正文"标签表构建已过因果性断言"的事后追述表述一致，不违反"先于本赛任何结果"（指标结果）。
(c) 段计数：master_merged.parquet 实测 train 22963 / val 10231 / test 10643（另 pre2001 504 / embargo 799），与 race_results_merged.json 台账逐项一致；主池 2838/3144/1792 与台账一致。
补充：独立调用 tep.assert_segment_integrity（日历=000001.SH 交易日）对合并池与主表双双通过（段界、隔离带 ≥30 交易日、无串段）。
(d) 特征列：fm.feature_columns(合并池)=2060 列；fm.assert_no_leakage 零命中；fm.excluded_columns 全表零命中；pool 溯源列未混入特征。
(e) 兜底触发条件实测：master_main train 事件 2838<3000，兜底规则真实触发，合并池升正赛合法。

## 项 5：复现断言产物核查 —— PASS

- oof_{pool}_{candidate}.parquet 38 个文件全部在场。
- 结构核实：每文件恰 2 列（event_id + config_{当选config_id}），列名与 summary 当选配置 38/38 对应一致。
- 落盘口径登记：38 文件均为 self_rerun_x2 模式产物（首跑未落盘全 36 配置折外概率，复现断言阶段全程自复跑两遍逐位一致后补档单配置列），race_results_*.json 已如实登记模式，强度等价于 #25 口径。
- 抽验重算（与项 2 同脚本、同机同库版本）：
  - 正赛 hit_N20_k2.0 config_13：重算 OOF 数组与 parquet config_13 列 np.array_equal(equal_nan=True) 逐位一致，event_id 序逐位一致。
  - 收益族 cur_pos_10d（merged）config_3：重算 OOF 与 parquet config_3 列逐位一致，event_id 序逐位一致；指标行 19 字段逐位一致（best_iters=[5,4,105,31,12]、终模 31 轮、val AP=0.5861542868235563、dayavg=0.4877677806341046）。

## 存疑与边界说明（不影响放行）

1. 首跑（run_merged.out/run_main.out 的 06:40 前后日志）使用旧版驱动的"复跑"口径，07:05 续跑用现版驱动 self_rerun_x2 补档；两次断言均逐位一致且本复核第三次独立重算再次逐位一致，链条完整。
2. pandas read_csv 快解析器存在个别 double 的 1 ulp 解析偏差；裁决重算使用 <1e-12 容差、指标逐位对比使用 float() 精确解析，均已规避。
3. 预登记评论与最早指标文件间隔仅 523 秒，时序合规但余量小；标签表早于预登记落盘，属预登记正文明确追述的既成产物，不构成后见之明风险（标签定义在预登记前已由 #25 口径锁定）。

## 最终结论

全部五项复核 PASS，无一 FAIL。
裁决链（38×36 网格 → 选配置 → 总裁决 → 中位数约束）重算逐位一致；正赛当选配置独立重跑逐位一致；标签手工抽验与 NaN 纪律过关；test 段零触碰、预登记时序、段计数、特征泄漏断言全部合规。
**复核意见：放行。**
当选标签 hit_N20_k2.0（合并池正赛，config_id=13，val_precision_at_5_dayavg=0.5750，val_average_precision=0.6225 ≥ 中位数 0.5794）可落 #20 锁定。
