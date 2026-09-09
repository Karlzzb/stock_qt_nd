# M2 特征主表重建 —— 预登记军令状(先于跑数落盘,冻结勿改)

- 票:`Karlzzb/stock_qt_nd` issue #49(Part of #47)。
- 分支:`v6-portfolio`。
- 日期:2026-09-09。
- 施工图:`reports/wayfinder/44-v6-rebuild-survey.md`(适配分级逐脚本执行)。
- 几何族定义:`reports/wayfinder/42-v6-geometric-features.md`(六族 43 条,入层 40 条)。
- 输入事件表:`experiments/v6_model_campaign/m1_event_table/events_ext_v1.parquet`(96,577 行 × 25 列,M1 复核锁定,只读)。
- 本阶段只建特征主表,不建标签、不训练(标签归 M3,训练归 M4)。

## 一、施工范围(钉死)

v5 三来源全量重建 + v6 几何族 40 条 + 种子资格布尔五条,合并为 96,577 行事件×特征主表。
来源构成与键:

| 来源 | 内容 | 键 | 生产器 |
|---|---|---|---|
| s0 | 事件表直透列 13 条(因果审查见 §六) | 随事件表 | 主表驱动内并入 |
| s1 | 事件级特征词典:P0/P1 股内特征 + 市场/宽度/日历特征 + RET20_CSR + v6 几何族 40 条(含种子布尔 F6~F10) | event_id | `build_event_dictionary.py`(自 git 历史 4c7717f^ 恢复 `build_feature_matrix.py` 机器,几何族按 #42 重定义) |
| s2 | 特征工厂全量重生成(G 层生成式,约 1,603 列) | (ts_code, date) | `run_factory_v6.py`(适配 `regen_factory_full.py`:换键 + 单池化) |
| s3 | 日频特征快照(V2 日频族 + 横截面 + 大盘 + 日历) | (ts_code, date) | `run_v4daily_v6.py`(适配 `rebuild_v4_daily_snapshot.py`:Pass A 改落全历史行,池无关) |
| s4 | T3 新特征事件日快照(38 列) | (ts_code, date) | `run_t3_v6.py`(适配 `build_t3_features.py`:换键 + 单池化) |

v5 的 17 条 V1 事件结构特征(DIV_* 族,低点对低点语义)在 v6 金叉对金叉事件上无定义,**不重建、不带入**;其在来源 1 的位置由 #42 几何族 40 条顶替。
#42 族E 的 E2/E3/E4(事件日二十日收益波动率/事件日量比/事件前波动率变化)按族E 立规由 s3 日频快照覆盖(VOL20/VMA5/VOL20÷VOL60 组合),不进 v6 专属层。

## 二、切分常量(#30 预登记首次落码)

- 训练段 2001-01-01 ~ 2019-12-31;验证段 2020-01-01 ~ 2023-12-31;测试段 2024-01-01 ~ 2026-08-31(数据末日为界)。
- 1992~2000 段标 pre2001,不入训练。
- 隔离带派生规则(钉死):对每个段界(训练|验证界、验证|测试界),取上证指数交易日历(stock_data/daily/000001.SH.parquet 排序去重)上**早段内侧最后 30 个交易日**与**晚段内侧最前 30 个交易日**,带 = 两侧并集区间 [左带首日, 右带末日],带内事件标 embargo、不作建模型行。
- 派生结果(落码常量,构建时以 `derive_embargo_bands(交易日历)` 重算核验一致):
  - 带1 = 2019-11-20 ~ 2020-02-20(左 30 日 2019-11-20~2019-12-31,右 30 日 2020-01-02~2020-02-20)。
  - 带2 = 2023-11-20 ~ 2024-02-20(左 30 日 2023-11-20~2023-12-29,右 30 日 2024-01-02~2024-02-20)。
- 扣带后段内事件数(预登记推算值,跑数后核验):训练 50,165 / 验证 24,405 / 测试 17,902 / pre2001 1,042 / 带1 内 1,313 / 带2 内 1,750。
- 与 #44 勘察表 §0 的偏差披露(先于跑数登记):#44 所述带界 2019-11-19~2020-02-20 与 2023-11-20~2024-02-21 两侧不对称(61/60 交易日),无任何单一派生规则可同时复现;本军令状采用对称规则(每侧恰 30 交易日,唯一满足「两侧各 30 个交易日」的无歧义读法),与 #44 表相差带1 左侧一日、带2 右侧一日,对应训练段 +61 行、测试段 +18 行。
- 落码位置:`v3_pipeline/src/feature_master.py` 的 TRAIN_LO/TRAIN_HI/VAL_LO/VAL_HI/EMBARGO(#44 §4 指定的唯一权威落点),`assert_segment_integrity` 三调用点(M4 冒烟/标签赛/精选)自动继承。

## 三、泄漏物理剔除与同式去重规则(原样沿用 v5 口径)

- 泄漏排除模式:`feature_master.EXCLUDE_PATTERNS` 17 条(^rank_ 对白名单 {rank_return, rank_volume} 豁免)∪ `feature_engine.BLACKLIST_PATTERNS` 14 条 ∪ `v4_daily_snapshot.FORBIDDEN_PATTERNS` 5 条。
- 三集合在 ^rank_ 上语义重叠:以 feature_master 的白名单口径为准(v5 既有口径),即 rank_return/rank_volume 两列豁免,其余 ^rank_ 一律剔除。
- 命中列**物理剔除**(从主表删列),剔除清单逐列落 master_results_v6.json。
- 同式去重:训练+验证段(扣带)合并行上精确成对完全观测相关(pairwise-complete, min_pairs=30),|ρ| ≥ 0.999 贪心去重;保留优先级 s1 > s2 > s3 > s4,s0 直透列最低(同优先级按列名字典序);去重台账(被剔列/锚列/ρ)逐行落盘;去重后断言保留列两两 |ρ| < 0.999。
- 列名跨源碰撞:值同去重、值异 ValueError 浮出,禁止静默改名(merge_sources 现成语义)。

## 四、种子资格布尔(口径 = #31 §2.2 逐字)

- dd20 = close[j] / max(close[max(0, j-20) .. j]) − 1(含事件日共至多 21 根,行号 j 为事件日行号)。
- bounce = close[j] / anchor_close − 1。
- 五条布尔:v6-1 `dd20≤−0.15 且 0.02<bounce≤0.08`;v6-2 `dd20≤−0.20` 同 bounce 带;v6-3 `dd20≤−0.25` 同 bounce 带;v6-4 `dd20≤−0.15`;v6-5 `dd20≤−0.25`。
- 精度守卫:布尔在 float64 的 dd20/bounce 上判定后再存 float32(0/1 精确表示),杜绝阈值邻域浮点翻转。
- 全表硬断言:计数 == frozen 值 (5,580 / 2,419 / 1,159 / 11,888 / 3,271),嵌套链 v6-3⊂v6-2⊂v6-1⊂v6-4、v6-5⊂v6-4 零违反。

## 五、工程量与产物预算(vs #44 §3 估算)

| 环节 | #44 估算 | 本阶段预算 |
|---|---|---|
| 日频快照 Pass A(全历史行) | 3~6 min | ≤15 min ×2 次(确定性双跑) |
| 日频快照 rank pass + Pass B | 26~36 min | ≤45 min ×2 次 |
| 特征工厂全池重生成 | 4~6 min | ≤10 min ×2 次 |
| T3 快照 | 7~10 min | ≤15 min ×2 次 |
| 来源 1(#44 未估,生产器已删) | — | ≤30 min ×2 次 |
| 主表合并+去重+抽检 | 8~12 min | ≤20 min ×2 次 |
| 存储增量 | 约 25 GiB | parts 约 21 GiB ×2(双跑,核后删副本)+ 主表约 1.5 GiB ×2 + 其余约 4 GiB |

产物路径(全部在 `experiments/v6_model_campaign/m2_feature_master/`):
- `cache/s1_event_dictionary.parquet`、`cache/s2_factory_full.parquet`、`cache/s3_v4daily_snapshot.parquet`、`cache/s4_t3_snapshot.parquet`。
- `cache/v4daily_parts_fullhist/{code}.parquet`(Pass A 全历史逐股,池无关)。
- `master_v6.parquet`(96,577 行)、`master_dictionary_v6.csv`(含中文全名列)、`master_results_v6.json`(全部台账)。
- `rebuild/`(确定性第二次构建副本,核验 md5 后删除,哈希值留台账)。
- parquet 与 progress.log 按 .gitignore 不入库;代码、README、report.md、自检 JSON 入库。

## 六、s0 事件表直透列因果审查(#44 §2.4 要求的预登记)

事件表 25 列中:键 2 列(ts_code/event_date)与 event_row 为元数据;日期列 5 个(anchor_date/cross_prev_date/cross_prev2_date/cross_date/min_prev_date,cross_date 恒等于 event_date)与行号列 3 个(cross_prev_row/cross_prev2_row/min_prev_row,股内行号跨股不可比)**不入主表**;其余 13 列全部经 M1 断言 1/3/5/6 证实最晚可知时点 ≤ 事件日收盘,准入为 s0 直透特征列:anchor_close、cross_prev_dif、cross_dif、dif_lift、cross_prev2_dif、cross_prev2_dea、cross_prev_dea、cross_dea、min_prev_close、event_close、event_vol、event_amount、anchor_bars。
无需为 v6 追加泄漏排除模式;anchor_bars 与几何族 D1 恒等,去重阶段自然二留一(保 D1,优先级 s1 > s0)。

## 七、验收断言清单(执行版,全过才算完;#44 §6 草案落地)

1. **泄漏物理剔除**:主表列对三套排除模式(§三口径)零命中;剔除清单落台账;selfcheck 独立重扫主表列名复核。
2. **同式去重**:去重台账逐行落盘;断言保留列两两 |ρ| < 0.999;selfcheck 对抽样被剔对(自来源 parquet 重取列重算 ρ)与抽样保留对独立复核。
3. **段界与隔离带硬断言**:`assert_segment_integrity` 四项(seg 与 segment_of 逐行一致 / 带内零建模型行 / 每带交易日历 ≥30 日 / 段间实测间隔 ≥30 交易日)在主表上执行;EMBARGO 落码值与 `derive_embargo_bands(交易日历)` 重算逐日一致;带界/带内交易日数/带内事件数/各段行数落台账。
4. **行数与键守恒**:主表行数 == 96,577 == 事件表行数;(ts_code, date) 与 event_id 双侧唯一;event_id 按 (ts_code, event_date) 字典序重铸与 s1 键逐位一致;各来源键覆盖率与缺失计数落盘(NaN 保留)。
5. **来源在场与碰撞纪律**:s1/s2/s3/s4 四来源各有列在表;碰撞记录全为值同去重,零值异。
6. **时点一致性**:s3 前缀稳定性抽检(12 股 × 3 日,seed 42,rtol=1e-9 零不一致)+ 主表末端新鲜抽检(5 格,seed 20260902)+ s4 末端抽检(4 格)+ s1 构建内全表事件列对账(重算 DIF/DEA/区间最低/锚点与事件表逐位相等,不等即停线)。
7. **几何族因果(#42 §7 M2 落点)**:selfcheck 独立实现抽样重算 40 条(500 事件,seed 20260909);MACD 截尾无前视抽样(100 事件,j≥100,容差 1e-8);dd20 ≤ 1e-12 全表;种子布尔全表独立重算与主表 SEED_V6_* 列逐行一致(计数对 frozen 值);含守卫 NaN 分支特征(B3/B4/B6/C4/E1)NaN 占比落盘。
8. **确定性双跑**:s1/s2/s3/s4/主表各构建两次,逐产物 md5 逐位一致;s3 Pass A 双份 parts 逐文件 md5 一致;哈希全部落台账。
9. **特征-标签隔离**:主表列名无 label_ 前缀;本阶段不产出任何标签表;join 键 (ts_code, event_date) 移交 M3。
10. **测试段零触碰**:本阶段不计算任何标签与指标,测试段行仅作为特征行存在于主表。

## 八、对 v3_pipeline 既有代码的改动预登记(能不改就不改;必须改处全列于此)

1. `v3_pipeline/src/feature_master.py`:切分常量改 §二 新值(TRAIN_HI/VAL_LO/VAL_HI/EMBARGO);新增纯函数 `derive_embargo_bands(calendar)`(派生规则落码);EVENT_META_COLS 改 v6 元数据集 [event_id, ts_code, date, event_row, seg];模块 docstring 相应行更新。其余(排除模式/合并/去重/断言函数)一行不动。
2. `v3_pipeline/src/train_eval_pipeline.py`:仅 docstring 中旧切分描述行更新(断言本体不动)。
3. `v3_pipeline/tests/test_feature_master.py`、`v3_pipeline/tests/test_train_eval_pipeline.py`:段标签用例日期改新切分等价用例(合成数据、无磁盘依赖纪律不变)。
4. 其余 v3_pipeline 代码零改动;四台生产器均为 m2 目录下新驱动,import v3_pipeline/src 复用计算本体。
5. v5 既有产物(master_main/backup、cache/v4daily_parts 等)一律只读不动;旧 parts 目录的缓存命中陷阱(#44 §1)以全新 parts 目录 `v4daily_parts_fullhist/` 规避,不删旧产物(归档义务),作废声明落本文件与 report.md。

## 九、执行纪律

- 长任务落 `progress.log`(各驱动追加,带驱动名标签与心跳);预计总机时 3~4.5 小时(含确定性双跑)。
- 背离v6 信号定义不动;事件表只读;experiments/ 下冻结目录只读。
- commit 到 v6-portfolio,不加 co-author 行,不 push;不关 #49(监工独立证伪式复核后处置)。
