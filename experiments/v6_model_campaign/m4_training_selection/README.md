# M4 训练管线与 SHAP 五层精选 —— 预登记军令状(先于跑数落盘,冻结勿改)

- 票:`Karlzzb/stock_qt_nd` issue #51(Part of #47)。
- 分支:`v6-portfolio`。
- 日期:2026-09-09。
- 战役规格:issue #47 M4 节;子票 #51。
- 复用机器:v5 SHAP 五层精选纯函数层 `v3_pipeline/src/feature_selection.py`(模块 docstring 全口径)、v5 驱动范式 `v3_pipeline/scripts/run_feature_selection.py`、训练原语 `v3_pipeline/src/train_eval_pipeline.py`(time_series_oof / fit_final_model / final_num_boost_round / DEFAULT_LGBM_PARAMS 复现性四件套)、36 组预登记网格 `v3_pipeline/src/label_race.py`(GRID 原序)、切分常量 `v3_pipeline/src/feature_master.py`(import 使用,本阶段对 v3_pipeline 零改动;发现 bug 停下来记录报告,不私改)。
- M3 裁决链:`experiments/v6_model_campaign/m3_label_race/`(回归头管线:无校准层,排序直接用预测原值——M4 定版模型同属回归头,沿用此口径,本条即预登记写明)。
- 本阶段只做基模冒烟、五层精选、定版模型与三段分数序列落盘;不做策略层审判(归 M5)、测试段只出分数不出任何指标。

## 一、施工范围(钉死)

1. 基模冒烟:当选配置(config_id=24,断言读出,禁止手抄)+ 回归头管线(五折时间序列折外 → 终模均值轮数)在全 1,997 特征上跑通,折外分数与 M3 落盘 `oof_v6_net_ret_60d.parquet` 的 config_24 列逐位一致断言(event_id 行序对齐 + np.array_equal equal_nan=True),val 头部五名净笔均(日加权)与 M3 metrics 行精确相等断言。
2. SHAP 五层精选(不抽样,96,577 < 10 万未触发 #30 US20;分层抽样预案 30,000 行封顶、种子 20260909 备而不用,本条预登记备而不启用)。
3. 定版模型:终选特征集上按当选配置重训(五折折外 → 终模均值轮数),三段分数序列落盘;embargo/pre2001 行同样出终模预测原值分数、seg 列照标(不作建模型行的纪律由下游执行)。
4. 精选各层产物、层4 曲线(K 为行全出数)、终选特征集清单(含中文全称)落盘;独立自检(层5 口径)通过后锁定。

## 二、输入(全部只读)

- 特征主表:`experiments/v6_model_campaign/m2_feature_master/master_v6.parquet`(96,577 × 2,002;元数据列 event_id/ts_code/date/event_row/seg,特征 1,997 列)。
- 标签表:`experiments/v6_model_campaign/m3_label_race/labels_v6.parquet`(用 net_ret_60d 列;join 键 event_id,与 (ts_code, date) 一对一互证)。
- 当选裁决:`experiments/v6_model_campaign/m3_label_race/adjudication_v6.json` + `summary_v6.csv` + `metrics_v6_net_ret_60d.csv` + `oof_v6_net_ret_60d.parquet`(折外对账基准)。
- 特征词典:`experiments/v6_model_campaign/m2_feature_master/master_dictionary_v6.csv`(column/cn_name,中文全称唯一来源;status=='kept' 恰 1,997 行)。
- 交易日历:`stock_data/daily/000001.SH.parquet`(段界断言用)。

## 三、当选标签与配置解析(v6 适配,断言守护,不接受手抄)

从 adjudication_v6.json + summary_v6.csv + metrics_v6_net_ret_60d.csv 三重读出并互证:
1. adjudication_v6.json: winner == "net_ret_60d"(次日开盘入场净收益幅度回归标签,60 个交易日视野),double_constraint_passed == true。
2. summary_v6.csv: winner 行恰一行,其 config_id 与网格原序断言相等(config_id=24 ⇔ num_leaves=63 / min_data_in_leaf=50 / learning_rate=0.05 / feature_fraction=0.6,与 `label_race.GRID[24]` 逐项 float 相等断言)。
3. metrics_v6_net_ret_60d.csv: config_id=24 行的 val_top5_net_dayavg / val_top5_net_cluster_t 与 adjudication_v6.json 的 winner 值精确相等(容差 0,float 解析)。
4. 完整 LightGBM 参数 = `train_eval_pipeline.DEFAULT_LGBM_PARAMS` ∪ GRID[24] ∪ {objective: regression, metric: rmse}(M3 预登记 §五.2 钉死,回归头无校准层)。
5. 头型断言:winner 属回归头 → 全程无 SquaredLogitCalibrator,排序与指标一律用预测原值。

## 四、SHAP 五层精选口径(v6 适配点逐条预登记)

基模 = 当选配置 + 全 1,997 特征;SHAP 计算用基模终模(全训练段重训,轮数 = 五折 best_iteration 均值取整,下限 1)。

- SHAP 值:LightGBM `pred_contrib=True`(TreeSHAP 精确值,末列偏置项剔除),复用 `feature_selection.shap_values` 原样;计算行集合 = val 段且标签 net_ret_60d 非 NaN 的事件(v5 同口径,v6 实测 24,247 行;主表 val 段 24,405 行中标签 NaN 的 158 行不入 SHAP 行集)。
- 层1 重要性排序:规则原样(`feature_selection.layer1_rank`):importance = mean(|SHAP|) 按 SHAP 行集;importance 恰为 0 剔除;名次 = importance 降序、平局特征名升序(mergesort 确定性)。
- 层2 分年度符号一致性:规则原样(`feature_selection.layer2_yearly_signs`),**v6 适配点 = 年份清单**:函数签名自带 year_list 形参,v6 传 (2020, 2021, 2022, 2023)(v6 val 段覆盖年份;v5 默认值 (2019,2020,2021,2022) 为 v5 历史口径,本阶段显式传参覆盖,不改 v3_pipeline 源码)。
  规则不变:val 各年份特征值与其 SHAP 值的 Pearson 相关定号(成对去 NaN;年内有效对 < 30 或年内方差为 0 记 0;SHAP 方差为 0 或相关恰为 0.0 同记 0);四年非全同号且非零 → 漂移剔除。
  年份覆盖断言原样:SHAP 行集实测年份集合必须恰等于 {2020, 2021, 2022, 2023}。
- 层3 相关簇去重:规则原样(`feature_selection.layer3_clusters` + `feature_master.pairwise_corr`):train+val 段标签非 NaN 行上特征值 Pearson 相关;按层1 名次贪婪聚类,与既有簇代表 |corr| ≥ 0.9 入名次最靠前匹配簇,否则自立为代表;只留代表;NaN 相关不判同簇。
- 层4 拐点定容:**v6 适配点 = 评估指标**:从 v5 的头部五名精确率(日加权)改为 M3 主指标 —— 验证段头部五名净笔均(日加权)(每事件日按预测原值降序、平局 (ts_code 升序, event_id 升序) 确定性裁决取 top-min(5, 当日事件数),日截面净收益均值再按日等权平均;净收益 = net_ret_60d 真实值;回归头无校准、预测原值排序),与战役裁决链同型(M3 §五.4),本条即该变更的预登记。
  K 阶梯沿用 `feature_selection.K_LADDER` 原序经 `k_ladder` 过滤(N < 5 不开路断言原样);每 K 取层3 幸存代表按层1 名次前 K 走完整基模管线(训练段五折折外 → 终模均值轮数 → val 终模预测原值),记 val 头部五名净笔均(日加权)附 cluster_t(按 entry_date 聚类的 Liang-Zeger 稳健 t,#31 §2.4 口径,复用 M3 驱动实现);同记 train_oof 口径与终模轮数,K 为行全出数。
  拐点判定复用 `feature_selection.find_elbow` 原样(x = log2(K),首末连线垂距最远,平局取小 K,相对容差 1e-12),传入时将本阶段指标列改名挂载到该函数约定的列名接口(纯列名适配,算法零改动);末点最远则取末点。
- 层5 独立复核:证伪式自写代码重算精选结果(selfcheck_m4.py,不调 feature_selection 的选择函数 layer1_rank / layer2_yearly_signs / layer3_clusters / find_elbow / k_ladder,SHAP 值用 LightGBM pred_contrib 独立重算,相关系数用 numpy 独立实现),记录落 selfcheck_results.json 与 report.md。
- 选择动作只在 train/val 段;test 段零指标零逐行统计(分数表无任何标签列,test 零触碰由构造保证并机检断言)。

## 五、定版模型与分数序列口径

- 定版模型 = 层4 拐点 K* 那一跑管线产物:终选特征集 = 层3 代表按层1 名次前 K*;训练 = 训练段(seg=='train' 且标签非 NaN,按 (date, ts_code, event_id) mergesort)五折折外 → 终模(轮数 = 五折均值取整,下限 1)。
- 分数表 `scores_v6.parquet`:覆盖主表全 96,577 行,列 = event_id / ts_code / date / seg / score 恰五列(无标签列)。
  - train 行 = 五折折外预测原值;首块无折外与标签 NaN 行 score = NaN。
  - val / test / embargo / pre2001 行 = 定版终模预测原值;seg 列照标主表原值。
  - 行序 = 主表行序(event_date, ts_code 升序);event_id 唯一断言。
- 复现断言(不可旁路):主跑完成后以全新进程 `--repro-check` 全链重导(基模 → 层1~层4 → 定版打分),层表/拐点/终选清单/model.txt 逐字节/分数数组逐位一致才生效;同进程内将重算分数另落临时 parquet 与 scores_v6.parquet 双跑 md5 比对一致,双 md5 记入台账。

## 六、验收断言清单(执行版,全过才算完)

1. 当选解析:§三 五重断言全过(winner/双约束/config_id/网格超参/metrics 值精确相等/回归头)。
2. 基模对账:折外分数与 M3 `oof_v6_net_ret_60d.parquet` config_24 列逐位一致(event_id 序 + equal_nan);基模 val 头部五名净笔均(日加权)与 metrics_v6_net_ret_60d.csv config_id=24 行精确相等。
3. 段界硬断言:`train_eval_pipeline.assert_segment_integrity` 在主表 (date, seg) 全量执行;训练行 seg 全 'train',SHAP/层2 行集 seg 全 'val',层3 行集 seg ∈ {train, val}。
4. 泄漏列:1,997 特征列对 `feature_master.EXCLUDE_PATTERNS` 零命中(selfcheck 独立重扫);特征集无 label_ 混入。
5. 精选完整性:层1 表恰 1,997 行;层2 表行数 == 层1 幸存数且年份列恰 2020/2021/2022/2023;层3 代表数 ≥ 5(否则层4 不开路报错即停);层4 曲线表恰 len(ladder) 行按 K 升序全出数;终选特征数 == K*。
6. 中文全称:层1/层2/层3 表与终选清单的每个特征在词典中有非空 cn_name(断言守护,命名全称纪律)。
7. 确定性:分数序列双跑 md5 逐位一致;`--repro-check` 全新进程全链重导逐位一致(层表/拐点/终选/model.txt 字节/分数)。
8. test 零触碰:全程不计算任何 test 指标、不做 test 逐行统计;分数表无标签列;台账与曲线表无 test 字样列;test 行数在场断言(> 0)。

## 七、机时台账预登记

- 基线:M3 实测 288 配置 3,248.4 秒(workers=3,单配置墙钟约 30~40 秒,含折外五折 + 终模 + 指标)。
- 本阶段单进程串行预估:基模(全 1,997 特征,config 24)约 1~2 分钟;SHAP(24,247 行 × 1,997 特征,终模仅 15 轮)预估 2~10 分钟;层1~层3(纯表算 + 层3 相关矩阵 ≤ 1,997 × 49,806+24,247=74,053 行)预估 1~5 分钟;层4 阶梯(K ≤ 17 档,特征数 ≤ N_reps,单档 5~40 秒)预估 3~15 分钟;定版打分 < 1 分钟;--repro-check 全链重导约等于主跑。
- 合计预估 20~60 分钟;冒烟纪律:全量开跑前先做基模冒烟(§一.1)计时,实测写入修订记录后再开全量。

## 八、产物清单与路径(全部在 experiments/v6_model_campaign/m4_training_selection/)

| 产物 | 入库 | 说明 |
|---|---|---|
| README.md | 是 | 本预登记(冻结,变更只追加修订记录) |
| run_training_selection_v6.py | 是 | 驱动(基模冒烟/五层精选/定版打分合一;--smoke 与 --repro-check 子模式) |
| selfcheck_m4.py | 是 | 独立自检(层5 口径,不复用 feature_selection 选择函数与驱动函数) |
| smoke_results.json | 是 | 冒烟计时与基模对账结果 |
| layer1_shap_importance.csv | 是 | 层1 重要性排序表(feature/cn_name/importance/rank/kept,1,997 行) |
| layer2_yearly_signs.csv | 是 | 层2 分年度符号表(feature/cn_name/sign_2020..2023/consistent) |
| layer3_clusters.csv | 是 | 层3 簇明细(feature/cn_name/representative/is_representative/corr_with_rep/rank) |
| layer4_curve.csv | 是 | 层4 曲线(K 为行全出数:轮数/val 与 train_oof 主指标与 cluster_t) |
| layer4_elbow.json | 是 | 拐点(k_star/ladder/distances/规则文本) |
| final_features.json | 是 | 终选特征集清单(含中文全称与层1 名次) |
| selection_results_v6.json | 是 | 口径/断言/各层计数/双跑 md5/台账 |
| model.txt | 是 | 定版 LightGBM 模型(终选特征,终模均值轮数) |
| base_model.txt | 是 | 基模终模(SHAP 计算用,自检独立重算层1 的对照锚) |
| scores_v6.parquet | 否(*.parquet 按 .gitignore) | 定版三段分数序列(96,577 行 × 5 列) |
| selfcheck_results.json | 是 | 自检结果(全绿 all_pass=true) |
| report.md | 是 | 层4 曲线全表 + 各层幸存/剔除清单要点 + 终选特征集 |
| progress.log | 否(*.log) | 长任务心跳日志(驱动名标签+时间戳) |

## 九、执行纪律

- 长任务落 progress.log(驱动名标签+心跳);派活即报日志路径与预计时长。
- 本阶段对 v3_pipeline 零改动;发现 bug 停下来在 progress.log 记录并向监工报告,不私改。
- 汇报口径:配置为行全出数;分布统计用「覆盖率→门槛」方向;特征/标签命名用中文全称;v5 实证结论不作任何输入(v5 只提供机器与范式)。
- 选择动作只在 train/val 段;test 段只出分数不出指标。
- commit 到 v6-portfolio,不加 co-author 行,不 push;不关 #51(监工独立证伪式复核后处置)。

## 修订记录(预登记正文逐字未动,仅追加)

1. 2026-09-09 冒烟计时(先于全量开跑):基模(当选配置 config_id=24 × 全 1,997 特征,五折折外 + 终模均值轮数 15 + val 打分与指标)单跑实测 22.3 秒;
   折外分数对 M3 落盘 oof_v6_net_ret_60d.parquet 的 config_24 列逐位一致(event_id 序 + np.array_equal equal_nan=True);
   基模 val 头部五名净笔均(日加权)= +0.016313 与 M3 metrics_v6_net_ret_60d.csv config_id=24 行精确相等,cluster_t = +3.289。
   全量预估:基模 22.3s + SHAP(24,247 行 × 1,997 特征,15 轮终模)+ 层1~层3 表算 + 层4 阶梯(≤17 档,单档按特征规模 5~40s)+ --repro-check 全链重导(≈主跑),合计预估 15~45 分钟,在 §七 预估区间(20~60 分钟)内偏下。
2. 2026-09-09 施工实现修正(读取口径一处,非数据/模型口径变更):M3 落盘 CSV 一律改以字符串读入 + float() 精确解析(容差 0 口径,v5 #26 复核记录同款——pandas 快解析器存在 double 解析偏差);
   实测 adjudication_v6.json 的 winner_val_top5_net_dayavg 系 M3 汇总时经快解析器回读落盘,与 metrics CSV 精确解析值差 1 ulp(0.0163133682364511 vs 0.016313368236451103);
   §三.3 的 metrics↔adjudication 交叉断言相应以 1e-15(1 ulp 量级)容差守护非系统性偏差,§六.2 的基模 val 指标锚定与复现断言仍以 metrics CSV 精确解析值为锚(与重算逐位相等,冒烟已实证);
   裁决结论(winner/config_id/双约束)不受影响。
