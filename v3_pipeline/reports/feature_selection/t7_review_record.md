# T7 特征精选独立复核记录（issue #27）

复核员：独立复核（证伪式，默认立场"结果有问题"，逐项尝试推翻）。
复核对象：`v3_pipeline/reports/feature_selection/` 全部落盘产物。
复核代码：本目录 t7_review/review{1_layers,2_curve_final,3_discipline}.py（自写，不调 feature_selection 的 layer1_rank / layer2_yearly_signs / layer3_clusters / k_ladder / find_elbow / resolve_winner / assemble_scores；已随产物归档入库，任何人可重跑）。
允许复用的基础设施：feature_master（主表口径/相关矩阵）与 train_eval_pipeline / label_race（#25/#26 已复核管线），与 #26 复核同口径。
复核日期：2026-09-03。

## 结论总览

| 复核项 | 结果 |
|---|---|
| 1 层1~层3 自写重算 | PASS |
| 2 层4 拐点重算 + 独立重跑抽验 + 定版产物核查 | PASS |
| 3 纪律核查（a~e） | PASS |
| 4 复现产物核查 | PASS |
| 5 终选清单链式核对 | PASS（并入项 2） |
| **总判** | **放行** |

## 项 1：层1~层3 自写重算 —— PASS

方法：自写代码重建合并池帧（master_merged ⋈ 两池标签表，validate one_to_one），自走 tep 公开函数重建基模（五折 OOF best_iters=[37,60,54,89,34]、终模 55 轮），SHAP 用 booster.predict(pred_contrib=True) 自算。
层1 排序（importance 降序、平局特征名升序、kept=importance>0）自写实现；层2 分年度 Pearson 符号（成对去 NaN、<30 对或方差为 0 记 0、四年全同号非零才幸存）自写实现；层3 贪婪聚类（按层1 名次序、|corr|≥0.9 入名次最靠前匹配簇）自写实现（相关矩阵用 fm.pairwise_corr 基础设施）。

结果：
- 层1：2060 -> 516（零重要性剔 1544），feature 序、importance（float() 精确解析后容差 0）、rank、kept 四列与 layer1_shap_importance.csv 逐位一致。
- 层2：516 -> 473（漂移剔 43），四年符号列与 consistent 列与 layer2_yearly_signs.csv 逐条一致。
- 层3：473 -> 385 代表（簇吸收 88），representative / is_representative / rank / corr_with_rep 与 layer3_clusters.csv 逐条一致。
- 层1 第一名 cs_n（importance 0.194563）层2 符号序列 [+1,-1,-1,-1] 被漂移剔除——剔除规则真实生效的实证。

## 项 2：层4 拐点重算 + 独立重跑抽验 + 定版产物核查 —— PASS

方法：拐点几何自写（x=log2(K)、首末连线垂距、相对容差 1e-12 内并列取小 K），输入为 layer4_curve.csv 字符串读入 + float() 精确解析。
K=25 与 K=10 两个规模用 label_race.run_single_config 独立重跑全程（五折 OOF -> 校准层 -> 终模），与曲线行逐位对比。
定版产物：lgb.Booster 载 model.txt 核对轮数与特征名；校准层用重跑 OOF 重新拟合对比 calibrator.json；scores_final.parquet 结构核查 + val 段前 500 行用落盘模型重算逐位对比。

结果：
- 拐点重算 K*=25，与 layer4_elbow.json 一致；阶梯落盘与曲线行集合一致。
- K=25 重跑：dayavg=0.5737360754070265、轮数 60，六项指标字段逐位一致；K=10 重跑：dayavg=0.5629391602399315、轮数 38，逐位一致。
- model.txt 60 轮、特征名与 final_features.json 序一致；校准系数 [1.0074593655590922, 1.474403348584496]、截距 -0.8603425717784164 重拟合逐位一致。
- scores_final.parquet 43837 行（train 22963 / val 10231 / test 10643），列结构 (pool, ts_code, date, event_id, seg, prob, y)，event_id 唯一，prob 全在 [0,1]，test 段 y 全 NaN，val 前 500 行重算逐位一致。

## 项 3：纪律核查 —— PASS（a~e 全过）

(a) test 零触碰：grep 全部 csv/json/log 产物，"test" 命中仅 3 处，逐条甄别全部为段计数台账与"test 段 y 全 NaN"声明；无任何 test 段指标与逐行统计；scores_final.parquet test 段 y 全 NaN 实测确认。
(b) 预登记时戳：gh 查得置顶评论 created_at=2026-09-03T04:07:40Z；最早结果文件 layer1_shap_importance.csv mtime=2026-09-03T04:20:48Z，预登记早 788 秒，全部结果文件均在预登记之后。
(c) 段计数：独立调用 tep.assert_segment_integrity（日历=000001.SH 交易日）通过；实测 train 22963 / val 10231 / test 10643 与 selection_results.json 台账逐项一致。
(d) 特征列：tep.model_feature_columns=2060 列，fm.assert_no_leakage 零命中，fm.excluded_columns 全表零命中，pool 溯源列未混入；终选 25 特征全部在权威列内且无重复。
(e) 当选配置链：adjudication_merged.json winner=hit_N20_k2.0 且 ap_constraint_passed=true；summary 行 config_id=13 四项超参与预登记网格 GRID[13] 逐位一致；final_features.json 的 label/config_id 与裁决一致——非手抄确认。

## 项 4：复现产物核查 —— PASS

全新进程独立执行 `run_feature_selection.py --repro-check`（两轴评审修复后为加强版）：全链重导层1~层4 并与落盘层表/拐点逐位对比，再比定版模型字节、校准系数、三段分数（prob 与 y 两列含 NaN 位、event_id 序、列结构全部核验），退出码 0。
加上主流程阶段 4 内置的同款断言，复现链共两次独立进程验证通过。

## 两轴评审修复登记（复核后追加，修复后全链重跑并复核通过）

两轴评审（Standards + Spec）后做了以下修复，全部不改变任何数值结果（重跑产物与首轮逐位一致，本复核三项脚本对重跑产物再次全部 PASS）：

1. 复现断言升级为全链重导：--repro-check 与主跑共用同一 run_chain 实现，全新进程重导层1~层4 并与全部落盘层表/拐点逐位对比，再比定版模型字节、校准系数、三段分数——消除"只复跑 K* 打分段"的旁路面。
2. 复核脚本由 /tmp 归档入库（t7_review/），独立复核成为可重跑产物。
3. 层2 记 0 口径如实写明：除预登记的"对不足/特征方差为 0"外，SHAP 方差为 0 与相关恰为 0 两种"相关无法定义/无方向"形态同样记 0——只可能触发剔除、不可能促成幸存（模块 docstring 已写明）。
4. 拐点"平局取小 K"的浮点化：相对容差 1e-12 内视同并列（共线浮点尘不判最远）；实测 K*=25 为严格唯一最远点（次远点距为其 89.6%），容差对结果零影响。
5. 层2 年份覆盖断言（观测年份集必须恰等于预登记四年）、折外键唯一性断言、repro 侧 config_id/超参与落盘清单一致性断言补齐；层2 后的提前容量下限断言移除（预登记的 <5 不开路只在层4 k_ladder 触发）。
6. 运行日志增列（val_precision_at_5_eventavg、train_oof_* 两项）与 T6 锚定断言属预登记之外的加强护栏，不参与任何选择动作，在此如实登记。

## 存疑与边界说明（不影响放行）

1. pandas read_csv 快解析器存在个别 double 的 1 ulp 解析偏差（#26 已识）；本复核凡涉浮点逐位对比一律字符串读入 + float() 精确解析，已规避。
2. 预登记与最早结果文件间隔 788 秒，时序合规。
3. 层4 曲线在 K>=40 区间呈震荡（0.560~0.570），拐点落于全局最高点 K=25；规则为预登记垂距法，无自由裁量，震荡形态如实落盘备查。
4. 终选 25 特征 val 头部五名精确率日加权 0.5737 略低于全特征基模 0.5750（-0.13pp）；预登记拐点规则不以"超过基模"为约束，属正常容量裁减代价，如实登记。

## 最终结论

全部复核项 PASS，无一 FAIL。
五层剔除链（2060 -> 516 -> 473 -> 385 -> 25）逐层重算一致；层4 拐点与独立重跑逐位一致；定版模型、校准层、分数序列核查通过；test 零触碰、预登记时序、段界、泄漏断言全部合规；复现断言双进程通过。
**复核意见：放行。**
终选 25 特征集与定版模型可落 #20 锁定，scores_final.parquet 移交 T8 策略终审。
