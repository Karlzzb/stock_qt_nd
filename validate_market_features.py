#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证补充大盘特征后的模型效果
"""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.impute import SimpleImputer

print("=" * 60)
print("加载 2022-07 到 2022-09 特征数据")
print("=" * 60)

# 加载特征文件
feature_files = []
for month in ['202207', '202208', '202209']:
    files = sorted(glob.glob(f'real_feature_data_daily/realistic_features_{month}*.csv'))
    feature_files.extend(files)

print(f"找到 {len(feature_files)} 个特征文件")

# 合并所有数据
dfs = []
for f in feature_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"总样本数: {len(data)}")
print(f"总特征数: {len(data.columns)}")

# 检查大盘特征
market_features = [col for col in data.columns if col.startswith('sh_') or col.startswith('sz_')]
print(f"\n大盘特征数量: {len(market_features)}")

# 准备特征和标签
label_col = 'future_return_3d'
if label_col not in data.columns:
    print(f"错误: 未找到标签列 {label_col}")
    exit(1)

# 排除非特征列
exclude_cols = ['symbol', 'timestamp', 'Unnamed: 0'] + \
               [c for c in data.columns if c.startswith('future_return_') or
                c.startswith('stop_loss_') or c.startswith('future_sell_date_')]

feature_cols = [c for c in data.columns if c not in exclude_cols and c != label_col]
print(f"\n可用特征数: {len(feature_cols)}")

# 过滤有效样本
data = data.dropna(subset=[label_col])
print(f"有标签样本数: {len(data)}")

X = data[feature_cols].select_dtypes(include=[np.number])
y = data[label_col]

# 转换为二分类标签
threshold = 0.05
y_binary = (y > threshold).astype(int)
print(f"\n标签分布:")
print(f"  正样本 (收益>{threshold}): {y_binary.sum()} ({100*y_binary.mean():.2f}%)")
print(f"  负样本: {len(y_binary) - y_binary.sum()} ({100*(1-y_binary.mean()):.2f}%)")

# 填充缺失值
print(f"\n原始特征shape: {X.shape}")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
print(f"填充后shape: {X_imputed.shape}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\n训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 训练LightGBM
print("\n" + "=" * 60)
print("训练 LightGBM 模型")
print("=" * 60)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# 预测
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

# 计算 PR-AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)
print(f"PR-AUC: {pr_auc:.4f}")
print(f"最佳迭代: {model.best_iteration}")

# 特征重要性
feature_importance = model.feature_importance(importance_type='gain')
# 注意：imputer可能会删除全NaN的列，所以特征数可能不匹配
actual_feature_names = [X.columns[i] for i in range(len(feature_importance))]
importance_df = pd.DataFrame({
    'feature': actual_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 最重要特征:")
for idx, row in importance_df.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.1f}")

# 统计大盘特征的重要性
market_feature_names = [f for f in X.columns if f.startswith('sh_') or f.startswith('sz_')]
market_importance = importance_df[importance_df['feature'].isin(market_feature_names)]
print(f"\n大盘特征在Top 50中: {len(market_importance.head(50))}")
print(f"大盘特征总重要性占比: {market_importance['importance'].sum() / importance_df['importance'].sum():.2%}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
