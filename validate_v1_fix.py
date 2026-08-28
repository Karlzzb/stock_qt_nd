#!/usr/bin/env python3
"""
验证 V1 背离检测器修复效果
用 2022 年 7-9 月的新特征快速训练模型，看 PR-AUC 能否恢复
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, average_precision_score
import lightgbm as lgb
from config.settings import DAILY_FEATURE_DIR
from src.comm_fun import model_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("验证 V1 背离检测器修复效果")
print("="*80)

# 1. 加载 2022 年 7-9 月的特征数据
logger.info("加载 2022 年 7-9 月的特征数据...")

feature_files = []
for year_month in ['202207', '202208', '202209']:
    files = list(DAILY_FEATURE_DIR.glob(f"realistic_features_{year_month}*.csv"))
    feature_files.extend(files)

logger.info(f"找到 {len(feature_files)} 个特征文件")

all_data = []
for file in sorted(feature_files):
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except Exception as e:
        logger.warning(f"跳过文件 {file.name}: {e}")

if not all_data:
    logger.error("没有找到任何特征数据！")
    sys.exit(1)

df_all = pd.concat(all_data, ignore_index=True)
logger.info(f"总数据量: {len(df_all)} 行")

# 2. 准备特征和标签
label_col = model_config.LABEL_COL
if label_col not in df_all.columns:
    logger.error(f"标签列 {label_col} 不存在！")
    logger.info(f"可用列: {df_all.columns.tolist()[:20]}")
    sys.exit(1)

# 删除不需要的列
meta_cols = ['symbol', 'timestamp', 'detection_date', 'divergence_date', 'prev_time']
drop_cols = [c for c in meta_cols if c in df_all.columns]
drop_cols.append(label_col)

# 删除所有包含 future/forward/stop_loss 的列（数据泄露）
leaky_patterns = ['future_', 'forward_', 'stop_loss_', '_sell_date']
for col in df_all.columns:
    if any(pattern in col for pattern in leaky_patterns):
        if col not in drop_cols:
            drop_cols.append(col)

logger.info(f"删除 {len(drop_cols)} 列（包括 meta 和泄露特征）")

# 只保留数值列
X = df_all.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=[np.number])

y = df_all[label_col]

# 检查标签是否合法（二分类：0 或 1）
logger.info(f"标签唯一值: {y.unique()[:10]}")
logger.info(f"标签类型: {y.dtype}")

# 如果 label 不是 0/1，可能是 future_return，需要转换
if y.min() < 0 or y.max() > 1:
    logger.warning(f"标签不是二分类！范围: [{y.min():.4f}, {y.max():.4f}]")
    logger.info("尝试用阈值转换为二分类...")
    threshold = 0.05  # 涨幅 > 5% 为正例
    y = (y > threshold).astype(int)
    logger.info(f"转换后标签分布: 正例 {y.sum()} / {len(y)} = {y.mean():.4f}")

logger.info(f"特征维度: {X.shape}")
logger.info(f"标签分布: 正例 {y.sum()} / {len(y)} = {y.mean():.4f}")

# 3. 切分训练/测试集（简单按时间切分）
# 前 70% 训练，后 30% 测试
split_idx = int(len(df_all) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

logger.info(f"训练集: {len(X_train)} 行 (正例率 {y_train.mean():.4f})")
logger.info(f"测试集: {len(X_test)} 行 (正例率 {y_test.mean():.4f})")

# 4. 缺失值填充
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 获取保留的列名（imputer 会删除全 NaN 的列）
valid_cols = X_train.columns[~pd.DataFrame(X_train).isna().all()]
logger.info(f"填充后特征数: {X_train_imputed.shape[1]} (删除了 {len(X_train.columns) - X_train_imputed.shape[1]} 个全 NaN 列)")

# 5. 训练 LightGBM
logger.info("训练 LightGBM 模型...")

# 计算类别权重
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
logger.info(f"类别权重 scale_pos_weight: {pos_weight:.2f}")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': pos_weight,
    'verbose': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train_imputed, label=y_train)
model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data],
    callbacks=[lgb.log_evaluation(period=20)]
)

# 6. 预测和评估
logger.info("在测试集上评估...")

y_pred_proba = model.predict(X_test_imputed)
y_pred = (y_pred_proba >= 0.5).astype(int)

# 计算指标
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, y_pred_proba)

# 先验概率（随机基线）
prior = y_test.mean()

print("\n" + "="*80)
print("评估结果")
print("="*80)
print(f"测试集样本数: {len(y_test)}")
print(f"正例先验概率: {prior:.4f}")
print(f"")
print(f"Precision (阈值 0.5): {precision:.4f}")
print(f"Recall (阈值 0.5):    {recall:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"")

# 判断修复效果
if pr_auc > 0.55:
    print("✅ 修复成功！PR-AUC > 0.55，样本分布恢复正常")
    print("   建议：补齐 15 个大盘情绪特征，然后全量重跑")
elif pr_auc > 0.50:
    print("⚠️  有改善但不够。PR-AUC > 0.50 但 < 0.55")
    print("   建议：先补齐大盘情绪特征再测试")
else:
    print("❌ 问题不只是背离算法。PR-AUC < 0.50")
    print("   建议：深入诊断（label 定义？特征质量？）")

print("="*80)
