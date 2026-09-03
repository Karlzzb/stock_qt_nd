# v3_pipeline —— 当前生产管线

> 目录名 `v3_pipeline` 是历史遗留。
> 现行方法论是 2026-09 的 v5 重建（GitHub issues #20–#29），旧的 V1–V4 代码与文档已在本目录瘦身时删除。
> 术语以仓库根目录 `CONTEXT.md` 为唯一权威，先读它再读本文。

## 这条管线做什么

以**背离事件池**（而非全市场日频横截面）为唯一样本空间，走完一条可复现的链路：
背离事件扫描 → 事件日快照特征主表 → 标签赛 → 特征精选与定版模型 → 三类策略回测 → 测试段终审。
每一步都有预登记、硬断言和独立复核，证据落盘在 `reports/` 并与 GitHub issues 互证。

## 数据流总览

```
stock_data/                     原始数据：daily/ 个股日线 parquet、index/ 指数、stk_limit/ 涨跌停
   │
   ├─ T1  divergence_stability_check.py   背离事件池验收（时点一致性对拍 + 参数敏感性）
   │        配置：configs/divergence_lab/m_scan/{m_fractal15_full, m_zigzag05_nofilter}.json
   │        产出：reports/divergence_stability/（主池≈8154 事件，备池≈36986 事件，池定义锁定）
   │
   ├─ T2  build_feature_master.py         三来源合并成"事件×特征"主表（事件日快照对齐、泄漏列物理剔除、同式去重）
   │        ├─ regen_factory_full.py       来源2：特征工厂全池重生成
   │        └─ rebuild_v4_daily_snapshot.py 来源3：日频特征按事件日快照重建
   │     T4  build_t3_features.py          30 条预登记新特征实现并入主表
   │        产出：reports/feature_master/master_{main,backup}.parquet（2071 列 = 2060 特征 + 11 元数据）
   │
   ├─ T5  run_train_eval_smoke.py         训练评测管线冒烟：LightGBM + 逻辑回归校准层[p,p²]、
   │        五折时间序列折外概率、段界与 30 交易日隔离带断言
   │
   ├─ T6  build_race_labels.py + run_label_race.py   19 候选标签 × 预登记小网格全量标签赛
   │        产出：当选标签 hit_N20_k2.0（狙击标签，两池双胜锁定）
   │
   ├─ T7  run_feature_selection.py        SHAP 五层精选 2060 → 25 特征，重训定版模型
   │        产出：reports/feature_selection/{final_features.json, model.txt, scores_final.parquet}
   │
   ├─ T8  run_strategy_tuning.py          三类策略验证段调优（56 个预登记配置，配置为行出表）
   │        引擎：strategy_engine_v3.py（冻结 v1 引擎之上叠加三类出场）
   │        类 C 依赖：build_daily_score_panel.py 构建的日频打分面板
   │        产出：每类最优 A13 / B15 / C15
   │
   └─ T9  run_test_adjudication.py        测试段一次性终审回测（过线=净年化超额>基准+15pp 且夏普>0.5）
            结果：0/3 过线（超额 -8.6 / -2.4 / -15.0pp），测试段已消耗并记台账
            台账：reports/test_adjudication/test_segment_ledger.md
```

## 目录结构（当前）

```
v3_pipeline/
├── src/        # 管线库：feature_master / t3_features / train_eval_pipeline / label_race /
│               # feature_selection / v4_daily_snapshot / feature_engine / feature_factory / label_candidates
├── scripts/    # 入口脚本（上表）+ divergence_lab（池扫描库）+ strategy_engine{,_v3}（回测引擎）
│               # + run_strategy_family.py（指标计算，被 T8/T9 复用）
├── tests/      # 单元测试，合成数据已知值断言，无外部磁盘依赖
├── configs/    # divergence_lab/m_scan/ 两个池配置（其余旧配置已删）
└── reports/    # 全部战役证据归档（含 v5 之前的封存战役报告，与 issues 互证，勿删）
```

## 当前定版资产（后续任何重跑的起点）

| 资产 | 位置 |
|---|---|
| 锁定事件池 | `reports/divergence_lab/m_scan/`（T1 验收底座） |
| 特征主表 2022 列 | `reports/feature_master/master_{main,backup}.parquet`，词典 `master_dictionary.csv` |
| 当选标签 | `hit_N20_k2.0`（狙击标签：次日开盘入场，20 日内最高价触及开盘价+2 倍 ATR） |
| 终选 25 特征 | `reports/feature_selection/final_features.json`（含中文全名） |
| 定版模型与三段分数序列 | `reports/feature_selection/model.txt`、`scores_final.parquet` |
| 三类最优策略配置 | `reports/strategy_tuning/tuning_summary.json`（A13 固定止盈止损 / B15 波动率自适应 / C15 分数衰减） |
| 终审结论 | `reports/test_adjudication/t9_report.md`：0/3 过线 |

## 测试

```bash
python -m pytest v3_pipeline/tests tests -q
```

全部单元测试用合成数据已知值断言，不依赖 `stock_data/` 真实数据。

## 纪律指针

- 术语与切分定义：`../CONTEXT.md`（训练段 2001-01~2018-12 / 验证段 2019-01~2022-10 / 测试段 2022-11~2026-08，段界各 30 交易日隔离带）。
- 测试段已按"每个候选只碰一次"纪律消耗，重跑终审需要新数据或新协议，先看台账。
- 实验汇报与证据纪律见 GitHub issues #20–#29 及各 `reports/**/t*_report.md`。
