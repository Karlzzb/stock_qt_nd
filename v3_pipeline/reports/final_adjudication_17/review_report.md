# issue #17 测试段终审产物独立复核报告

复核人：独立复核代理（未参与执行，怀疑立场，逐条验证而非转述）。
复核日期：2026-09-02。
复核对象：v3_pipeline/scripts/run_final_adjudication_17.py 与 v3_pipeline/reports/final_adjudication_17/ 全部产物。
协议基准：issue #17 预登记正文 + 2026-09-02 执行前补充冻结 comment（经 gh api 读取原文）。

## 逐项结论

### 1. 协议一致性：确认

脚本逐条核对了补充协议 7 款，全部落实。
模型冻结：run_variant 逐行复刻 rt.run_config（label='hit' 分支），训练段=train 掩码（2001-2018，经 segment_mask+EMBARGO），6 组网格 rv2.LGBM_GRID 由验证段日 RankIC 早停选参，最终模型仅在 dtrain 上按 best_iter 重训；测试段只做 final.predict 评分，无任何调参/早停/重训入口。
测试掩码：date>=2022-11-01 且 hit 非 NaN 且 ~f_any，双池独立计算，与补充协议第 3 款一致。
基线：test_baseline 直读 labels.parquet、仅借特征矩阵键集合与清洗标记，逐日池命中率后对信号日日加权，与补充协议第 4 款一致。
bootstrap：直接调用 rt.bootstrap_daycluster（B=20000、seed=42、双侧，min(1, 2*min(p_lo,p_hi))），与第 5 款一致。
家族=4：VARIANTS 字典与预登记 4 变体（主/对照/锚点/几何基线，trunc 与特征名单逐项相符）一致，label 全为 hit。
判定逻辑：passed = (超额>0) and (boot_p<0.0125)，α=0.05/4，机械执行无裁量。

### 2. 断言门真实性：确认

GATE 在 main() 中以硬编码预期值 GATE_EXPECT={val_top3:0.533105, best_iter:24, tol:1e-3} 实现，gate_ok 失败时写出 GATE 失败 JSON 并 sys.exit(1)，结构上不可能触碰测试段。
GATE 运行本身 score_test=False，run_variant 在该分支不调用任何测试段评估。
时序证据：progress.log 与两份 run_stdout 日志显示 05:34:35 装配 → 05:34:37 GATE 通过 → 05:34:37 测试基线 → 05:34:38 起变体终审，断言先于一切测试段评分。
数值证据：GATE 复现 val_top3=0.5331050228310501、best_iter=24、best_params={num_leaves:31, lr:0.08}，与 #15 归档 v3_pipeline/reports/revival_targets/results_revival.json 中 trunc1_hit__withATRN 条目逐位一致。

### 3. 泄漏审计：确认无泄漏

训练/选参/早停路径：dtrain 仅 train_ok 行、dval 仅 val_ok 行，lambdarank group 由 rv2.sort_by_day 在各自段内按日构造，早停 feval 只用验证段 DayRankIC；测试行不进入任何 Dataset。
特征：直接使用特征矩阵原始列（withATRN=ATRN+b28+band 共 60 列，noATRN=59 列，ATRN_only=1 列），与 #15 同码路；本脚本未做任何秩变换或重新特征工程，rv2.add_daily_rank 未被调用。
标签：build_labels('hit') 直读 df['hit']，train_ok/val_ok 掩码把测试行排除在拟合与早停之外。
基线与统计：test_baseline 与 bootstrap/Wilcoxon 的配对序列只用测试段自身的 hit 与评分，无跨段信息。
评分：final.predict(X) 对全表打分，但模型参数已在训练段固定，测试行仅被前向传播一次，属评分非学习。
特征矩阵与 labels.parquet 本身是前序阶段产物（build_feature_matrix.py mtime 2026-09-01 19:49，早于终审会话），其时点性已由 #15 复核链路覆盖，不在本次执行代理的新增风险面内。

### 4. 数值抽验：确认

独立重算全部对账成功（不复用执行脚本的结果文件之外的任何中间态）：
主池主变体测试 top3 由 daily_curves.json 逐日均值重算=0.5326340326，与 results_final.json 一致；基线日加权重算=0.5260727（差异 2e-9 为浮点求和顺序），超额=+0.6561pp，与 0.6561351683867733 一致。
配对差序列：剔零差日后 n_pairs=52、均值=0.0180437704，与 boot_mean_diff 逐位一致。
用同一 rt.bootstrap_daycluster（seed=42）从 daily_curves 重跑：boot_p=0.5518724064，与归档 0.551872406379681 逐位一致；CI 一致。
Wilcoxon 用 scipy 独立重算 p=0.5061135415，与归档一致。
备池主变体：top3 重算=0.5641217565，超额=+0.9821pp，boot_p 重跑=0.0769961502，均与归档逐位一致。
源头复核：直接从 feature_matrix/labels/pool_cleaning 原始 parquet 重建测试掩码，主池 1792 键剔 1 个 hitNaN 剔 81 个 f_any=1710 事件/143 信号日/基线 0.5260726810，备池 8851 键剔 274 剔 438=8139/668/0.5543002486，与归档完全一致；预登记估计的事件数（1792/8851）为未剔 hitNaN 与清洗项的键集合口径，差异全部可解释。

### 5. 判定逻辑：确认

verdict.passed 由 exc>0 and isfinite(bp) and bp<ALPHA 机械计算：+0.6561pp>0 成立，但 boot_p=0.55187 不<0.0125，故 passed=false，输出"否决"。
无任何人为裁量入口；备池同号（+0.98pp）仅作加分项记录，未参与判定，符合预登记。

### 6. 附属组合模拟：确认（附两点观察）

方法论与报告所述一致：T+1 开盘入场（close[t1]/open[t1]*cf[e]/cf[t1]-1）、exit=min(sig+20, 末端) 即持有至多 20 交易日、总收益按持有天数逐日分摊、当日在持仓位等权平均、无持仓日不参与复利（与记 0 对权益曲线等价，仅 n_days_invested 口径不同，对年化无影响）、未计成本与滑点、复权经 pct_chg 链式 cf 与 target_switch 同式。
"双池模型组合均跑输零信息基线"与数值一致：主池模型年化 -4.59% vs 基线 +5.90%（超额 -10.49pp），备池 +21.58% vs +34.58%（超额 -13.00pp），逐年超额亦多数为负；本人从 daily_curves 的组合日收益独立重算年化/回撤/总收益，与归档逐位一致。
首轮崩溃 bug：run_stdout.log 的 Traceback 显示崩溃点在 simulate_portfolio 第 228 行 n_years 计算（numpy.timedelta64 无 .days），位于判定（05:35:34 已落日志）之后的附属段；修复后当前脚本同位置改为 pd.Timestamp 差值，属最小定点修复。
对照两次 run_stdout：判定行逐字相同（测试超额=+0.66pp boot_p=0.55187 备池同号=True -> 否决），且 8 个变体的 val_top3/best_iter/基线在两次运行间全部一致，证明修复未触及判定链路。
观察一：严格意义上测试段被评分了两次（首轮崩溃前与修复后重跑），但首轮判定在崩溃前已完整落盘、修复为确定性机械修复、重跑所有判定数值逐位相同，"一次性、不回炉"纪律在实质上成立。
观察二：终审报告第 25 行只对备池事件数（8851→8139）做了对账说明，主池 1792→1710（1 hitNaN+81 f_any）未在报告中解释，属报告完备性小瑕疵，数值本身经本人独立重建验证无误。

### 7. 纪律：确认

git status 核对：本次终审新增且仅新增 run_final_adjudication_17.py 与 reports/final_adjudication_17/ 目录（均为 untracked）。
三个已修改的已跟踪文件（.gitignore、src/tinyshare_auth.py、v3_pipeline/scripts/train_ranking.py）mtime 均为 2026-09-01 上午至晚间，早于终审会话（09-02 05:28 起），且 train_ranking.py 的 diff 为 V5 label_prefix 配置化，不在终审 import 链上。
共享模块 run_race_rerun_v2.py（mtime 09-01 23:01）、run_revival_targets.py（09-02 02:39）、run_target_switch.py（09-01 23:26）在终审会话期间未被修改，执行代理未为通过 GATE 而改动上游码路。

## 补充观察（不影响结论）

issue #17 目前仅有 1 条 comment（执行前补充冻结），终局结果尚未按补充协议第 7 款落 issue comment 归档；按记忆"AFK 战役终态"，此举在等待用户拍板，属流程待办而非产物缺陷。

## 总体意见

**支持归档否决。**
7 项复核全部确认，无返工项。
判定（主池主变体测试超额 +0.66pp>0 但日聚类 bootstrap p=0.55187 不<0.0125 → 否决）由预登记标准机械推出，全部关键数值经独立重算逐位对账，泄漏审计干净，GATE 真实且先于测试段，纪律干净。
两条小观察（测试段因附属 bug 被确定性重评一次、主池事件数对账未写入报告）不改变产物可信度。
