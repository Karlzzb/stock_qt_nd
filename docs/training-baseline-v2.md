# 训练基线 v2（协议 v2 首个干净基线，2026-08-22/23）

本文档固化当前训练管线的完整配方，作为后续一切优化的出发点。
任何改动（特征、label、超参数、训练流程）都应相对本基线做对照实验，并在验证集上报告完整指标块。

## 1. 数据快照

| 项 | 值 |
|---|---|
| 特征缓存 | `feature_cache_all.parquet`，1,538,790 行 × 750 列，2.3GB |
| 缓存指纹 | `feature_cache_all.parquet.fp.json`：pipeline_version=2.3.0，daily_fingerprint=be17c7d3f40195a1，csv_count=6211 |
| 底层数据 | `real_feature_data_daily/realistic_features_YYYYMMDD.csv` × 6211（2001-01-02 → 2026-08-14） |
| 训练集 | 2001-01 → 2021-12-31，1,101,378 行 → 丢 label 缺失后 **1,019,416 行**（15d 标签口径） |
| 验证集 | 2022-01-01 → 2025-07-31，337,185 行 → 丢 label 缺失后 **300,784 行** |
| 测试集 | 2025-08-01 → 数据末日，**封存**，每候选只评估一次 |
| 无菌检查 | 打分窗口触及测试集边界前 2 个月即拒绝执行（`--sterile-months`） |

切分参数：`--train-end 2021-12-31 --score-start 2022-01-01 --score-end 2025-07-31`。
缓存命中前强制校验指纹，指纹缺失或不一致直接拒绝训练（防止拿旧缓存训出"新"模型）。

## 2. 特征集

- 池构造（`build_full_feature_pool`）：缓存中全部数值列，剔除 `timestamp`/`symbol`/`sell_date`、原始行情五列（open/high/low/close/volume）、标签族（`future_*`、`stop_loss_*`）→ **709 列**。
- 死列过滤（训练时执行，`train_one_split` 内）：先剔训练段全 NaN 列，再剔训练段缺失率 >30% 的列 → **154 列被剔，最终 552 列**。
- 最终特征清单：`models/single_2021-12-31_2022-01-01_2025-07-31_poolfull/lgbm_features.json`（552 个，逐列列出）。
- 缺失值处理：训练段逐列精确中位数填充（`SimpleImputer(strategy="median")`，统计量为全量精确中位数）；inf/-inf 先转 NaN 再填充。
- 数据类型：float32（LightGBM 按 ≤255 bin 离散化，精度损失可忽略，内存减半）。
- LR 层稳态特征：**空列表**（消融证实加稳态原始特征有害，旧"LR 层 150 特征"设计已废弃）。

## 3. 标签定义

- 列：`future_return_{N}d`，N ∈ {3,5,10,15,20,25,30}，当前基准 N=15（`Config.SETTLEMENT_DAYS`）。
- 生成：`label = 1{ future_return_Nd > 0.01 }`（`Config.RETURN_THRESHOLD=0.01`，`get_return_threshold` 固定返回 0.01）。
- 标签列本身是止盈止损路径收益：止盈上限 1.15（+15% 封顶）、止损 0.65（`Config.EXPECTED_PROFIT/EXPECTED_LOSS`）。
- label 生成前先 dropna（标签列缺失行不参与训练/评估）。

## 4. 训练流程（`train_one_split` → `train_lgb_models` → `train_meta_model_reduce`）

1. 训练帧：特征 float32 → 剔全 NaN 列 → 剔缺失率 >30% 列 → 中位数填充。
2. **LightGBM 5 折 TimeSeriesSplit**（`Config.N_SPLITS=5`）：每折用后续折做验证，`early_stopping(stopping_rounds=150)`，eval_metric=[auc, binary_logloss]。
3. 每折产 OOF 预测；折得分 = 验证折 PR-AUC；**集成权重 = 折 PR-AUC 归一化**。
4. 无样本权重（`FP_PENALTY_WEIGHT=1.0` 且 sample_weight 调用已注释；scale_pos_weight=1.0）。
5. **LR meta 层**：输入 = [pred_lgb, pred_lgb²]（STABLE=[]，无其他特征），StandardScaler 标准化后 LogisticRegression。
6. 打分：验证帧同口径填充 → 5 模型加权平均 → meta 变换 → LR 概率。
7. 产物按切分目录落盘（见 §6），OOS 预测即时保存支持断点续跑。

## 5. 超参数（`src/comm_fun.py: Config`，未做任何新调参）

### LightGBM（`Config.LGB_PARAMS`）

```python
{
    'n_estimators': 1000,        # 实际由 early stopping 决定
    'learning_rate': 0.02,
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'num_leaves': 63,
    'max_depth': 4,
    'min_child_samples': 58,     # 折样本 <10000 时动态调小（本基线未触发）
    'min_split_gain': 0.6,
    'min_child_weight': 0.03,
    'scale_pos_weight': 1.0,
    'reg_alpha': 2,
    'reg_lambda': 2,
    'bagging_fraction': 0.875,
    'bagging_freq': 7,
    'feature_fraction': 0.85,
    'n_jobs': 4,
    'random_state': 42,
    'verbosity': -1,
}
```

### LR meta 层（`train_meta_model_reduce`）

```python
LogisticRegression(
    C=0.001,              # 强 L1 正则
    penalty='l1',
    solver='saga',
    max_iter=1000,
    fit_intercept=True,
    random_state=42,
)
```

### 其他开关

`USE_LGBM_LEAF=False`、`USE_PCA=False`、`PROBA_THRESHOLD=0.7`（仅用于指标块口径；干净模型 proba 最大 0.6078，该阈值当前无实际选中能力，选股须用 top-k 或重校准阈值）。

## 6. 模型产物

| 目录 | 内容 |
|---|---|
| `models/single_2021-12-31_2022-01-01_2025-07-31_poolfull/` | 全量池 552 特征基线（label=15d），含 lgb_models/imputer/lr_meta/stack_scaler/fold_scores/lgbm_features.json/oos_predictions.parquet/lgb_models_null.pkl（标签打乱对照） |
| `models/single_2021-12-31_2022-01-01_2025-07-31_label{3,5,10,15,20,25,30}d/` | label 网格 7 格，同构产物 |
| `experiments/label_grid_2021-12-31_2022-01-01_2025-07-31.json` | 7 格完整指标 + 分年度 + 逐日 top-k |
| `experiments/ablation_2021-12-31_2022-01-01_2025-07-31.json` | 两层消融三配置指标 |
| `experiments/screening_importance_2021-12-31_2022-01-01_2025-07-31.csv` | 552 特征分年度置换重要性 + 零分布卡线 |

基线指标（全量池 15d）：验证集 PR-AUC 0.4944、precision_top1% 0.4549、top5% 0.5695、top100 0.9700。
各 label 完整对照见 issue #16 评论（2026-08-23）。

## 7. 复现命令

```bash
# 全量池基线训练（~80s）
MALLOC_ARENA_MAX=4 uv run python scripts/run_walk_forward_train.py --single-split --full-pool
# 两层消融（~6min）
MALLOC_ARENA_MAX=4 uv run python scripts/run_two_layer_ablation.py \
    --split-dir models/single_2021-12-31_2022-01-01_2025-07-31_poolfull
# Label 网格（~18min）
MALLOC_ARENA_MAX=4 uv run python scripts/run_label_grid.py --horizons 3,5,10,15,20,25,30
```

## 8. 评估口径（2026-08-23 修订）

**排名口径为唯一优化基准**（`src/rank_metrics.py`，指标通俗说明见 #16 评论 2026-08-23）：
daily_rank_ic / icir / ic_pos_ratio、daily_top{1,3,5,10}_hit、daily_top5_excess_ret、top5_turnover、quantile spread。
threshold 类指标（precision/recall/f1@0.3/0.5/0.7）已退役——概率尺度随训练漂移，阈值口径不可比。
PR-AUC 与二元命中率降级为参考：实证显示二元命中与真实收益方向可以相反（15d：top1 命中 0.451 但 top5 超额 -1.32%）。

## 9. 后续优化的开放旋钮（按优先级）

1. **label 形式修正**（最高优先级）：排名口径下 7 个二元 label 全部不达可交易线（IC≤0.034，ICIR≤0.162），二元分类与赚钱方向错位，候选须扩展到回归/排序目标。
2. **聚类级特征筛选**（#16 US 11）：待 label 口径修正后进行，否则在错误方向上筛特征。
3. **超参数**：本基线沿用历史 LGB_PARAMS，从未在干净数据上调过。
4. **换手与持有期设计**：日换手 85~95%，须在策略层重新定义调仓频率。
5. **10d 标签列异常**：二元口径四年度低于先验但 Rank IC 最高（+0.034），生成链路待查。
