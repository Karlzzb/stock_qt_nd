# #44 特征主表与训练链重建工程勘察（v5 机器 → v6 池适配面）

勘察日期：2026-09-09。
性质：只读勘察与草案，不改代码、不跑重建、不动既有产物。
目标事件表：`experiments/divergence_seed_trial_history/events_history_v1.parquet`（96,577 事件，变体1 主判口径）。

## 0. 输入事实核对

v6 事件表实测 schema（pyarrow 直读）：9 列 = `ts_code`(large_string) / `event_date`(timestamp) / `anchor_date` / `anchor_close` / `cross_prev_date` / `cross_prev_dif` / `cross_date` / `cross_dif` / `dif_lift`（后五列为 double/timestamp 几何量）。
实测 96,577 行、5,748 只股票、事件日并集 5,696 天（1992-12-25 ~ 2026-08-31）、`(ts_code, event_date)` 无重复。
注意：**无 `event_id` 列、无 `date` 列名**，与 v5 机器全部输入假设不同（详见第 2 节）。
按 #30 预登记切分原始归属（未扣隔离带）：训练段 50,619 / 验证段 26,018 / 测试段 18,898 / 1992~2000 段 1,042。
段界双侧各 30 交易日隔离带（按上证指数交易日历派生）：2019-11-19 ~ 2020-02-20 与 2023-11-20 ~ 2024-02-21，带内事件分别为 1,374 与 1,768。
扣带后有效建模型行：训练 50,104 / 验证 24,405 / 测试 17,884。

## 1. 重大事实澄清：v4daily_parts 是旧池事件日切片，Pass A 必须重跑

`v3_pipeline/reports/feature_master/cache/v4daily_parts/` 5,820 个逐股 parquet（合计 15 GiB，12,361,151 行）**不是**全历史逐股产物。
证据链：
- `rebuild_v4_daily_snapshot.py:216` 把**全局事件日并集**（旧两池 4,204 天）发给每只股票，`v4_daily_snapshot.snapshot_rows` 只保留落在并集内的行。
- 实测 `000001.SZ.parquet` 3,866 行，而该股在旧两池的自有事件日仅 15 天、原始日线 6,000 行——3,866 ≈ 该股历史与 4,204 天全局并集的交集，恰证"切片"性质。
- 驱动日志（`cache/v4daily_run.out`，2026-09-02）：Pass A 5,889 只 / 24  workers / 141 秒新算 12,361,151 行——量级只可能是"历史 ∩ 全局事件日并集"，不可能是 45,140 事件行。

两条直接后果：
1. v6 事件日并集（5,696 天）与旧并集（4,204 天）不同，parts 缺 v6 新增事件日的行，**Pass A 必须重跑**。
2. **缓存命中陷阱**：`_worker`（脚本第 117-118 行）以 `out_path.exists()` 为命中条件直接跳过， naive 复跑会静默产出缺行的快照。改造时必须清空/改名 parts 目录（或把缓存键改为"事件日并集哈希"）。
规格建议（供战役规格参考，非本次改造）：parts 改为落**全历史行**（约 21 GiB，按 6,000/4,204 × 15 GiB 外推），parts 即池无关，未来任何换池只重跑 Pass B。

## 2. 逐脚本适配清单

分级口径：小 = 换事件键来源 + 单池化，~20 行内；中 = 另有口径/断言改写；大 = 涉及新语义或缺失资产重建。

### 2.1 `rebuild_v4_daily_snapshot.py`（+ `src/v4_daily_snapshot.py`）—— 中

- 现行输入假设：`v3_pipeline/reports/feature_matrix/{main,backup}_pool_features.parquet` 的 `(ts_code, date)` 键（`load_pool_keys`，第 64-75 行），双池结构 `POOLS=("main","backup")`。
- 换 v6：事件键改读 `events_history_v1.parquet` 的 `(ts_code, event_date)`（重命名 `event_date→date`）；单池化（POOLS 改一元组，Pass B 装配循环与落盘路径随之简化）；剔除指数伪股（`INDEX_CODES=("000001.SH","399001.SZ")`）断言保留。
- parts 目录必须作废（第 1 节缓存陷阱）。
- 段界/隔离带断言落点：**本脚本无段界断言**（快照口径与切分无关），无需改。
- 既有硬断言（保留原样）：前缀稳定性抽检 `prefix_stability_assert`（12 股 × 3 日，seed 42，全历史值与截断重算逐位一致，rtol=1e-9）；泄漏列模式 `FORBIDDEN_PATTERNS`（future_/stop_loss_/label_ 族）为模块级防御。

### 2.2 `regen_factory_full.py` —— 小

- 现行输入假设：同上的 feature_matrix 双池键（第 102-109 行）。
- 换 v6：改事件键来源 + 单池化即可；`_gen_worker` 的 `compute_stock_factory` 为逐股全历史纯因果滚动算子，池无关，计算本体零改动。
- 段界断言：无。
- 注意：`--force` 之外的缓存短路（第 114 行）以产物存在为准，换池必须删旧产物或 `--force`。
- 产物体积外推：旧主池 8,154 行 × 1,605 列 = 100 MiB → v6 96,577 行 ≈ 1.2 GiB。

### 2.3 `build_t3_features.py`（+ `src/t3_features.py`）—— 小

- 现行输入假设：同上的 feature_matrix 双池键（`load_pool_keys`，第 52-61 行）。
- 换 v6：改事件键来源 + 单池化；`build_panel`（全市场 17,321,333 行 × 18 列面板）与 `compute_all` 完全池无关，零改动。
- 段界断言：无。
- 既有硬断言（保留）：前缀稳定性抽检（截断 [T-1600 自然日, T] 重算逐位一致，seed 20260902）；快照键覆盖计数落盘。
- `coverage_report` 的评审遗留 callouts（#14/#10/#28/#25/#29 口径）为 v5 遗产，可保留也可随规格裁剪，不影响正确性。

### 2.4 `build_feature_master.py`（+ `src/feature_master.py`）—— 大

四处实质改动：
1. **事件表加载**（`load_events`，脚本第 123-130 行）：现读 `divergence_lab/m_scan/{name}/events.parquet`（v5 检测器事件，含 `event_id/sig_idx/low_date/prev_low_date/compare_rank/formation/regime/above_ma200`）。
   v6 事件表无 `event_id`（需确定性铸造，建议按 `(ts_code, event_date)` 字典序序号）、几何列语义全然不同（`anchor_date/cross_prev_date/cross_date/dif_lift` 对 `low_date/prev_low_date/formation`）。
   `EVENT_META_COLS`（feature_master.py 第 49-51 行）须按 v6 列重定义；种子成员资格五条布尔（v6-1~v6-5，#41 二轮裁决 Q1）在此并入为特征列。
2. **来源 1（事件级特征词典）资产缺失**：其生产器 `build_feature_matrix.py` 已在 commit 4c7717f 删除（提交说明"之前#20的实现代码，废物代码"），可从 `4c7717f^` 恢复。
   但其 DIV_* 事件几何族（`compute_event_features`，feature_engine.py 第 838 行起）语义绑定 V1 检测器低点几何，对 v6 金叉背离事件无定义——按 #41 二轮裁决 Q2，v6 专属几何特征族清单由 #42 定稿，来源 1 是"恢复 + 几何族重定义"，不是纯换输入。
3. **切分常量改写**（feature_master.py 第 56-59 行）：现为旧 v5 切分（训练 2001-01~2018-12 / 验证 2019-01~2022-10 / 测试 2022-11 起，`EMBARGO` 两条硬编码日历带、且只在段界**左侧** 30 交易日）。
   须改为 #30 切分（训练 2001-01~2019-12 / 验证 2020-01~2023-12 / 测试 2024-01~2026-08），隔离带按段界**两侧**各 30 交易日且由交易日历派生（见第 4 节矛盾记录①）。
4. **双池 → 单池**：`POOLS={"main":..., "backup":...}` 贯穿合并、去重（两池 train+val 合并算相关）、抽检、落盘；v6 单池后去重输入退化为单池训练+验证段，逻辑不变但代码路径全部要梳一遍。
- 泄漏排除模式（`EXCLUDE_PATTERNS` 17 条 + `RANK_WHITELIST` 豁免）与 `assert_no_leakage` 保留；v6 事件表自带列（`cross_dif/dif_lift/anchor_close` 等）的事件日收盘可知性应列入预登记因果审查，确认无需追加排除模式。
- 既有硬断言落点（保留）：行数=事件数（脚本第 222-223 行，键源改为 v6 事件表）、四来源在场、碰撞值异报错（feature_master.py `merge_sources` 第 130-131 行）、`assert_dedup_clean`（保留列两两 |ρ|<0.999）、末端新鲜抽检（5 格，seed 20260902）与 T3 末端抽检（4 格，每格约 40 秒）。

### 2.5 `build_race_labels.py`（+ `src/label_candidates.py` / `src/label_race.py`）—— 大

- 现行输入假设：`m_scan/{name}/events.parquet` 双池键 + 各池 `labels.parquet` 的 div 组狙击标签 `hit_N20_k2.0`（`tep.load_div_labels` 逐位对齐装载）。
- 换 v6 的实质改动：
  1. 狙击标签装载路径废弃（v6 无 div 组 labels.parquet）；#41 二轮裁决 Q5② 标签 = 次日开盘 → 第 H 个交易日收盘、**扣成本常量**（滑点双边 0.1% + 佣金万 2.5 最低 5 元按 10 万本金折算 + 印花税分时段）——现有 `label_candidates.py` 三族（cur/open_exec/mfr，9 视野）全部毛收益口径，无一扣成本，须新写净收益算子。
  2. 涨停买不进（开盘 ≥ 涨停价）与退市/数据耗尽截断 → 标签剔除 + 占比披露（Q5③）：需引入 `stock_data/stk_limit` 逐日涨停价（2007-01-04 起，缺文件日口径按 #31 预登记"视为无约束"），现有标签机器无此Mask。
  3. 视野集 H∈{10,20,25,60}（#41 一轮裁决 6），二分类（净收益>0）+ 幅度回归**双头**（Q5④）——回归头标签为连续净收益，标签表结构与后续管线口径都要扩。
- 保留资产：逐股并行全序列计算 + 按事件键合并的骨架、头部截断 200 行重算逐位一致的因果性断言（`_truncation_check`，seed 20260903）、键唯一与键缺失台账。
- 段界断言：无（标签与切分无关）。

### 2.6 `run_train_eval_smoke.py`（+ `src/train_eval_pipeline.py`）—— 中

- 现行输入假设：`master_{pool}.parquet`（POOL_LAB 双池映射）+ 狙击标签 `hit_N20_k2.0`。
- 换 v6：主表路径单池化；标签列改新标签赛产物；排序键 `(date, ts_code, event_id)` 依赖 event_id 铸造（见 2.4）。
- **段界断言落点**：第 98 行 `tep.assert_segment_integrity(master[["date","seg"]], cal)`；实现在 train_eval_pipeline.py 第 69-107 行，读 `feature_master.EMBARGO` 与 `MIN_EMBARGO_TRADING_DAYS=30`——段界常量改对后此处自动生效，无需改断言本体。
- 回归头注意：管线本体（`time_series_oof` + `SquaredLogitCalibrator` + `evaluate_segment`）为二分类专用（校准层输入 [p, p²]、指标为平均精确率/头部五名精确率）；若冒烟要覆盖回归头，需预登记回归评测口径（建议冒烟仍只用二分类头，回归头口径放标签赛票据）。
- 既有硬断言（保留）：OOF 全程复跑逐位一致（`np.array_equal(equal_nan=True)` + best_iters 一致）、oof parquet md5 台账、test 段只在场不出数。

### 2.7 `run_feature_selection.py`（+ `src/feature_selection.py`）—— 中

- 现行输入假设：`label_race/master_merged.parquet`（双池合并主表）+ `labels_race_{main,backup}.parquet` 拼接 + `summary_merged.csv`/`adjudication_merged.json` 裁决锚。
- 换 v6：单池化（去掉 merged 装载与 pool 列）；锚定断言（基模 OOF 与标签赛落盘折外逐位一致、基模 val 指标与汇总一致）改锚到 v6 新标签赛产物；层 1~层 4 算法本体与复现断言结构（`--repro-check` 全新进程全链重导逐位对比）原样保留。
- SHAP 层 1 现状：对**验证段全量**行算 `pred_contrib`（feature_selection.py 第 67-74 行），无抽样——v6 验证段 24,405 行 × ~2,000 列仍在分钟级（v5 合并池 10,231 行 × 2,060 列时五层全链总耗时仅 86 秒），见第 5 节结论：不抽样。
- 段界断言落点：第 312 行 `tep.assert_segment_integrity`。

### 2.8 相邻机器（不在票据清单但换池必动，登记于此）

- `run_label_race.py`：候选标签集改 H∈{10,20,25,60} × {二分类, 回归} 八头；`MIN_TRAIN_EVENTS` 主备合并兜底结构（#20 遗产）在单池 ALL 口径下废弃；裁决规则按单池重新预登记（#30 US18 已要求）。
- 事件流下游（策略引擎 v3 / 终审 harness）：本次未勘察，留战役规格工程章节另立勘察点。

## 3. 工程量估算（全部有日志/体积依据）

单机基线 = v5 全量实跑日志（2026-09-02，24 workers 机器）。

| 环节 | v5 实跑 | v6 估算 | 依据 |
|---|---|---|---|
| 日频快照 Pass A（逐股全历史特征链） | 141 s（5,889 只） | 3~6 min | 计算量只与股票数×历史长度有关，与事件日数量无关；v6 涉股 5,748 只 ≈ 同量级；写出量 12.36M→约 16.8M 行（5,696/4,204 并集比 1.355 外推），parts 体积 15→约 20 GiB |
| 日频快照全市场 rank pass | 34 s（4,204 事件日） | 约 1 min | 按事件日线性 |
| 日频快照 Pass B（按日装配横截面） | 986 s（17 块×250 日） | 25~35 min | 每块读全部 parts：块数 17→23、单块读出体积 ×1.36，I/O 主导约 ×1.8；单块面板行数两者相近（峰值均约 110 万行） |
| 日频快照合计 | 1,132 s（19 min） | **30~45 min** | 上三项之和 |
| 特征工厂全池重生成 | 197 s | **4~6 min** | 逐股全历史计算池无关，股票数近似；仅写出体积 ×11.8（96,577 vs 8,154 行） |
| T3 新特征快照 | 387 s | **7~10 min** | 面板装配（38 s）与 compute_all 池无关；快照步骤随事件数线性但占比小 |
| 特征主表合并+去重+抽检 | 239 s | **8~12 min** | 合并行数 ×11.8；去重相关矩阵耗时按 train+val 行数 31,253→74,509（×2.4）外推 14 s→约 35 s；T3 末端抽检每格约 40 s 不变 |
| 标签表构建（新净收益族） | 23.6 s（19 列毛收益） | **分钟级**（算力）；**主要成本在算子新写**（净成本+涨停 Mask+双头） | 逐股向量化计算，股票数近似 |
| 标签赛（相邻机器） | 5,369 s @3 workers（19 候选 × 36 配置，训练 22,963 行） | **1~2 h**（八头 × 36 配置 = 288 任务，训练行 ×2.2，单任务约 23.5→约 52 worker-s） | 任务粒度与实测单任务耗时外推 |
| 训练冒烟 | 2.8 s（主池 2,838 行） | **1~2 min** | 训练行 ×17.7，绝对量仍小 |
| SHAP 五层精选 | 86 s（含层 4 十四点阶梯全管线重训与 repro-check） | **10~20 min** | 行数 ×2.4；层 4 阶梯每点一次五折 OOF，约 14×52 s ≈ 12 min 主导 |

**总工期量级**：纯机时约 **半天以内**（特征链 1~1.5 h + 标签与训练链 2~3 h，可串行一夜跑完）；瓶颈不在机时而在适配改造与预登记/断言撰写，**人/代理工作量 2~4 个工作日**（大头：来源 1 几何族重定义依赖 #42 清单、净收益标签族新写、段界常量与单池化改造及其复核）。
存储增量：parts 约 20 GiB + 主表约 1.5 GiB（96,577 行 × 约 2,000 列，按 master_main 125 MiB/8,154 行外推）+ 工厂/T3/标签约 2 GiB，合计 **约 25 GiB**（现有 parts 15 GiB 作废后可回收）。

## 4. 切分断言落点清单（#30 切分）

权威常量唯一落点：`v3_pipeline/src/feature_master.py` 第 56-59 行（`TRAIN_LO/TRAIN_HI/VAL_LO/VAL_HI/EMBARGO`）+ `segment_of`（第 65-74 行）。
断言执行器唯一落点：`v3_pipeline/src/train_eval_pipeline.py` 的 `assert_segment_integrity`（第 69-107 行，四项硬断言：seg 与 segment_of 逐行一致 / 隔离带零建模型行 / 每带交易日历 ≥30 日 / train-val 与 val-test 实测间隔 ≥30 交易日）与 `MIN_EMBARGO_TRADING_DAYS=30`（第 34 行）。
断言调用点（换池后全部自动继承，无需逐处改）：
- `run_train_eval_smoke.py:98`
- `run_label_race.py:186`
- `run_feature_selection.py:312`
- `feature_master.segment_of` 另被 `build_feature_master.py:128`（load_events 打 seg 列）与去重行掩码（脚本第 246 行，train+val 合并算相关）使用。
改造要点：`EMBARGO` 由硬编码日历带改为"段界两侧各 30 交易日、按交易日历派生"（现码仅段界左侧单带，语义不符 #30）；派生结果应落盘进 results.json 台账（带界日期 + 带内交易日数 + 带内剔除事件数）。

## 5. SHAP 抽样口径（#30 US20 勘察结论）

US20 触发条件："池规模上到十万级以上"。
本池 96,577 事件 < 100,000，且层 1 SHAP 实际只在**验证段**（24,405 行）上计算，`pred_contrib` 为确定性算法（无随机性），v5 实跑 10,231 行 × 2,060 列时五层全链总共 86 秒。
**结论：不需要抽样**，工程量无受控需求。
口径草案（作为规模预案写入规格，防御未来池/特征膨胀）：若验证段建模型行 > 50,000 或特征列 > 4,000，则层 1 改为按（事件日年份 × 标签符号）分层等概率抽样至 30,000 行封顶，`numpy.random.default_rng(20260909)` 种子固定，抽样框、各层命中率与样本清单落盘；层 2（分年度符号一致性）与层 4（拐点定容）始终用全量验证段，不抽样。

## 6. 验收断言清单草案（逐条可机器检查）

1. **泄漏特征物理剔除**：主表列对 `feature_master.EXCLUDE_PATTERNS`（17 条）∪ `feature_engine.BLACKLIST_PATTERNS`（14 条）∪ `v4_daily_snapshot.FORBIDDEN_PATTERNS`（5 条）零命中；断言函数 = `assert_no_leakage` + `assert_no_blacklisted`，在 build_feature_master 与 run_train_eval_smoke 双侧各跑一遍；剔除清单落 results.json。
2. **同式去重**：去重后保留列两两 |ρ| < 0.999（pairwise-complete，min_pairs=30），断言 = `assert_dedup_clean`；去重台账（被剔列 / 锚列 / ρ）逐行落盘。
3. **折外概率逐位可复现**：同进程 OOF 复跑 `np.array_equal(equal_nan=True)` 且 best_iters 一致；oof parquet 落盘 + md5 入台账；精选阶段另加 `--repro-check` 全新进程全链重导逐位对比（层表 / 拐点 / model.txt 逐字节 / 校准层系数 / 三段分数）。
4. **段界硬断言**：`assert_segment_integrity` 四项（见第 4 节）在冒烟、标签赛、精选三个入口各跑一遍；隔离带派生结果（带界 / 交易日数 ≥30 / 带内事件数）落台账。
5. **事件键守恒**：`(ts_code, event_date)` 无重复；主表行数 == v6 事件表剔指数伪股后行数；各来源键覆盖率与缺快照/缺标签事件计数落盘（NaN 保留、剔除时机按段登记）。
6. **时点一致性**：v4daily 前缀稳定性抽检（12 股 × 3 日，seed 42，rtol=1e-9 零不一致）+ 主表末端新鲜抽检（5 格，seed 20260902）+ T3 末端抽检（4 格）+ 标签头部截断 200 行重算逐位一致（20 股，seed 20260903）——四条全部保留，仅事件键来源换 v6。
7. **列名碰撞**：跨来源同名列值异即 ValueError 浮出，禁止静默改名（`merge_sources` 现成语义）。
8. **测试段零触碰**：测试段只出分数不出任何指标，分数表 y 列恒 NaN；每候选一次触碰台账从零起立（v5 旧台账封存）。

## 7. v5 产物与文档/代码矛盾记录（只记录，不修复）

1. **切分常量与 #30 预登记不符**：`feature_master.py` 第 56-59 行仍是旧 v5 切分（训练止 2018-12 / 验证止 2022-10 / 测试 2022-11 起），`EMBARGO` 为两条硬编码日历带且只在段界左侧；#30 US12 与 Implementation Decisions 预登记的新切分（训练止 2019-12 / 验证 2020-01~2023-12 / 测试 2024-01~2026-08，段界两侧各 30 交易日）从未在 v3_pipeline 落码——#30 战役的特征主表阶段（T2'）实际未执行，v6 重建是首次真正应用该切分。
2. **parts 缓存语义与外观不符**：脚本 docstring 称"抽取事件日并集行落盘"，实为**全局**事件日并集切片（非该股自身事件日、非全历史），且缓存命中只看文件存在，换池 naive 复跑静默出错（第 1 节）。
3. **来源 1 生产器已删除**：#30 复用资产清单列"特征三来源构建器"，但 `build_feature_matrix.py` 已在 4c7717f 删除，需从 git 历史恢复且几何族语义要按 v6 重定义（依赖 #42）。
4. **事件数口径漂移**：m_scan 事件表 8,158 行（含 4 条指数伪股事件）→ 主表 8,154 行（剔除）；而 `build_race_labels.py` 用未剔除的事件表落标签表 8,158 行（键合不受影响，指数伪股无日线）。行数口径跨产物不统一，v6 重建建议在规格中钉死唯一口径（事件表即剔指数伪股）。
5. **隔离带语义两版并存**：v5 码 = 段界左侧 30 交易日单带；#30/#41 = 段界两侧各 30 交易日。按 #41 二轮裁决 Q3"原样采用 #30"，以两侧为准。
6. master_main.parquet 实测 8,154 行 × 2,071 列，与 #44 所述一致（无矛盾，登记核对）。

## 附：关键证据文件指针

- v6 事件表：`experiments/divergence_seed_trial_history/events_history_v1.parquet`（schema 见第 0 节）
- v5 运行日志：`v3_pipeline/reports/feature_master/cache/v4daily_run.out`（Pass A/B 分块耗时）、`factory_full_progress.log`、`t3_progress.log`、`master_progress.log`、`label_race/run_merged.out`、`train_eval_smoke/smoke_results.json`、`feature_selection/selection_results.json`
- 已删来源 1 生产器：`git show 4c7717f^:v3_pipeline/scripts/build_feature_matrix.py`
- 段界常量：`v3_pipeline/src/feature_master.py:56-74`；断言执行器：`v3_pipeline/src/train_eval_pipeline.py:34,69-107`
