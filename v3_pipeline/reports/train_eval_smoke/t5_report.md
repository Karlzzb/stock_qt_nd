# T5 训练评测管线冒烟报告（issue #25）

日期：2026-09-02
产出：v3_pipeline/reports/train_eval_smoke/（指标表、折外概率产物、val 预测、smoke_results.json 台账）
管线库：v3_pipeline/src/train_eval_pipeline.py；驱动：v3_pipeline/scripts/run_train_eval_smoke.py
单元测试：v3_pipeline/tests/test_train_eval_pipeline.py（21 例，合成数据已知值断言 + tmp_path 标签装载守卫，无外部磁盘依赖）。

## 结论

**训练评测管线已打通：主池 + 狙击标签 hit_N20_k2.0 端到端冒烟通过，三条验收标准全部 PASS。**
管线构成：LightGBM 二分类 + 逻辑回归校准层（输入 [p, p²]）、训练段内五折时间序列折外概率、段界与 30 个交易日隔离带断言、平均精确率与头部五名精确率指标表。
本票只验证管线正确性，不出任何"标签好坏"结论；冒烟指标不构成标签赛证据（标签赛为 #26，全量特征 × 十九候选）。

## 验收断言（issue #25 AC）

1. **单标签端到端跑通并出指标表**：PASS。
   主池 master_main（8154 行 × 2071 列，2060 特征）合并狙击标签（匹配率 99.9755%，缺 2 事件：train 0 / val 1 / test 1，为标签截断/停牌顺延，已登记剔除），train 2838 / val 3143 行建模，test 1792 行在场但不出任何数字。
   指标表见 metrics_main_hit_N20_k2.0.{csv,md}。
2. **切分段界与隔离带无串段断言通过**：PASS。
   assert_segment_integrity 四项硬断言：主表 seg 列与权威 segment_of 逐行一致（0 行错位）；train/val/test 行无一落入两段隔离带日期区间；两段隔离带在上证指数交易日历上各恰 30 个交易日（2018-11-19~2018-12-28、2022-09-13~2022-10-31）；数据实测 train 末行~val 首行、val 末行~test 首行间隔均 ≥30 个交易日。
3. **校准层五折折外概率产出可复现**：PASS。
   LightGBM deterministic=true + 全种子固定（seed/feature_fraction_seed/bagging_seed/data_random_seed=20260902）+ num_threads 固定 8；
   冒烟脚本内置复跑断言：五折 OOF 全程重算一遍，np.array_equal(equal_nan=True) 逐位一致、best_iters 序列一致。
   折外概率落盘 oof_main_hit_N20_k2.0.parquet（md5 记入 smoke_results.json）作复现凭证。

## 口径登记（全部预登记，非调参结果）

- 模型：LightGBM binary，固定冒烟参数（num_leaves=31、min_data_in_leaf=100、learning_rate=0.05、feature_fraction=0.8、bagging 0.8/5），num_boost_round 上限 1000、早停 50 轮（折内验证块）；终模轮数 = 五折 best_iteration 均值取整（本次 = 15）。
  超参小网格（≤50 组）属后续模型票据，本票不动。
- 校准层：sklearn LogisticRegression（C=1.0，max_iter=1000），设计矩阵 [p, p²]，拟合于 (OOF 概率, 训练段标签)；本次系数 [1.108, 1.737]、截距 −0.940。
- 五折切分：按事件日分块（TimeSeriesSplit 作用于唯一日期再映射回行），折界不落日内、同日事件不跨折，折内 max(train 日期) < min(val 日期) 有断言。
  首块（最早的 1/6 事件日，183 行）无折外概率，校准样本为其余 2655 行——时间序列 CV 固有口径，已登记。
- 指标：平均精确率（sklearn average_precision_score）；头部五名精确率 = 按事件日截面取概率 top-5（不足 5 取当日全部），平局按 (prob 降序, ts_code, event_id) 确定性裁决；日加权为主口径（与 pool_cleaning 基线同口径），事件加权为辅。
- 泄漏守卫：特征列 = 主表 2071 列减 11 元数据列，过 feature_master 既定排除模式断言（零命中）；evaluate_segment 硬拒绝 test 段输入。

## 冒烟指标（仅证明管线会算数，不构成标签证据）

| segment | n_events | base_rate | average_precision | n_days | n_selected | precision_at_5_dayavg | precision_at_5_eventavg |
|---|---|---|---|---|---|---|---|
| train_oof | 2655 | 0.611676 | 0.672863 | 335 | 801 | 0.529851 | 0.570537 |
| val | 3143 | 0.580019 | 0.621137 | 316 | 932 | 0.508861 | 0.531116 |

解读边界：两段 AP 均高于各自零信息线（= base_rate），方向合理；头部五名日加权与 pool_cleaning 的 val 零信息基线（日加权 0.4997，top3 口径）同量级，属未调参冒烟参数下的正常表现。
best_iters=[1, 4, 48, 5, 15] 显示早停很早（min_data_in_leaf=100 下小树即饱和），终模仅 15 轮——这是"管线能跑通"的代价，标签赛前的调参票据再议。

## 评审登记（两轴代码评审后）

- 校准层输入分布错位（已登记，不改）：校准层拟合于五折早停模型的折外概率，再施于全训练段 15 轮终模的概率，两者分布不完全同源；此为 #20 既定 OOF 校准架构的固有限度。
- train_oof 指标行的证据力弱于 val 行：校准层在自身拟合数据上打分，in-sample 偏乐观；标签赛裁决只认 val 口径。
- 段界理论边缘（已登记，不改）：segment_of 口径下 2018-12-31 仍属 train（隔离带 1 终于 12-28），若有该日事件将距 val 仅 1 个交易日；现行两池 train 段最末事件为 2017-11-15，零事件触及该边缘，且数据实测间隔断言（≥30 交易日）兜底。
- 评审修复：特征列口径收编为 feature_master 权威 feature_columns（数值 dtype 过滤）+ 泄漏断言单一入口 model_feature_columns；冒烟脚本移除 --skip-repro-check（AC3 复跑断言不可旁路）与多余 sys.path 项；test 段标签 NaN 计数移出台账（终审前 test 零逐行统计）；标签装载 load_div_labels 补 3 例 tmp_path 单测（div 组对齐、非 div 拒绝、重复键拒绝）。
- 顺带修复（与 #25 无直接关系）：tests/test_issue12_market_features_dataframe.py 测试函数遗留 return True 触发 pytest 警告，按全局质量纪律删除。
- 历史遗留 RuntimeWarning（pandas sqrt invalid value，test_issue10_pipeline_fixes 三例）为合成 fixture 数据伪影，登记不修。

## 单测覆盖

test_train_eval_pipeline.py 21 例：段完整性正反例（段标签篡改、隔离带混入、隔离带不足 30 交易日、数据实测间隔被穿透）；五折折界不落日内与覆盖口径；OOF 逐位可复现与未排序拒绝；终模轮数均值与确定性；校准层设计矩阵确为 [p, p²]、方向可恢复、确定性、越界拒绝；头部五名精确率已知值与平局确定性；泄漏列、非数值列与 test 段评测守卫；标签装载（div 组逐位对齐、非 div 拒绝、重复键拒绝）。
踩坑登记：np.array_equal 默认 NaN≠NaN，折外概率首块为 NaN，复现断言须 equal_nan=True（单测已钉死）。

## 对后续票据的约束（#26 标签赛）

- 标签赛可直接复用本管线：换标签列即可，切分/断言/校准层/指标口径不变。
- test 段在终审前保持零触碰（本次冒烟也未出 test 数字）。
- 校准层输入形态 [p, p²] 已由系数维度单测钉死，与 #20 第 11 条既定形态一致。
