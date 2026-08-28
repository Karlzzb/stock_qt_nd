#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用2022年其他月份验证大盘特征效果
"""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.impute import SimpleImputer

print("=" * 70)
print("验证大盘特征效果（使用2022年1-6月、10-12月数据）")
print("=" * 70)

# 加载特征文件
feature_files = sorted(glob.glob('real_feature_data_daily/realistic_features_2022*.csv'))
print(f"\n找到 {len(feature_files)} 个2022年特征文件")

if len(feature_files) == 0:
    print("错误：未找到特征文件")
    exit(1)

# 合并所有数据
dfs = []
for f in feature_files[:100]:  # 先加载前100个文件测试
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
    print(f"\n错误: 未找到标签列 {label_col}")
    exit(1)

exclude_cols = ['symbol', 'timestamp', 'Unnamed: 0'] + \
               [c for c in data.columns if c.startswith('future_return_') or
                c.startswith('stop_loss_') or c.startswith('future_sell_date_')]

feature_cols = [c for c in data.columns if c not in exclude_cols and c != label_col]
print(f"\n可用特征数: {len(feature_cols)}")

data = data.dropna(subset=[label_col])
print(f"有标签样本数: {len(data)}")

X = data[feature_cols].select_dtypes(include=[np.number])
y = data[label_col]

threshold = 0.05
y_binary = (y > threshold).astype(int)
print(f"\n标签分布:")
print(f"  正样本: {y_binary.sum()} ({100*y_binary.mean():.2f}%)")
print(f"  负样本: {len(y_binary) - y_binary.sum()} ({100*(1-y_binary.mean()):.2f}%)")

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\n训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

print("\n" + "=" * 70)
print("训练 LightGBM 模型")
print("=" * 70)

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

y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n" + "=" * 70)
print("评估结果")
print("=" * 70)
print(f"PR-AUC: {pr_auc:.4f}")
print(f"最佳迭代: {model.best_iteration}")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
