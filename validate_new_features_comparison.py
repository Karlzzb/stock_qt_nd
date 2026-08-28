#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比新增特征前后的效果
"""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.impute import SimpleImputer

print("=" * 70)
print("验证新增流动性微观结构特征的效果")
print("=" * 70)

# 加载特征文件
feature_files = []
for month in ['202207', '202208', '202209']:
    files = sorted(glob.glob(f'real_feature_data_daily/realistic_features_{month}*.csv'))
    feature_files.extend(files)

print(f"\n找到 {len(feature_files)} 个特征文件")

if len(feature_files) == 0:
    print("错误：未找到特征文件，请先运行特征生成")
    exit(1)

# 合并所有数据
dfs = []
for f in feature_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"总样本数: {len(data)}")
print(f"总特征数: {len(data.columns)}")

# 检查新特征
new_features = [
    'amihud_illiq_intraday',
    'roll_spread',
    'hl_spread',
    'alpha12',
    'price_high_volume_low'
]

present_new_features = [f for f in new_features if f in data.columns]
print(f"\n新增特征:")
for feat in new_features:
    if feat in data.columns:
        non_null = data[feat].notna().sum()
        print(f"  ✓ {feat}: {non_null}/{len(data)} 非空 ({100*non_null/len(data):.1f}%)")
    else:
        print(f"  ✗ {feat}: 未找到")

# 准备特征和标签
label_col = 'future_return_3d'
if label_col not in data.columns:
    print(f"\n错误: 未找到标签列 {label_col}")
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
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\n训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 训练LightGBM
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

# 预测
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

# 计算 PR-AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print("\n" + "=" * 70)
print("评估结果")
print("=" * 70)
print(f"PR-AUC: {pr_auc:.4f}")
print(f"最佳迭代: {model.best_iteration}")

# 特征重要性
feature_importance = model.feature_importance(importance_type='gain')
actual_feature_names = [X.columns[i] for i in range(len(feature_importance))]
importance_df = pd.DataFrame({
    'feature': actual_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 30 最重要特征:")
for idx, row in importance_df.head(30).iterrows():
    marker = "🆕" if row['feature'] in new_features else "  "
    print(f"{marker} {row['feature']}: {row['importance']:.1f}")

# 统计新特征的重要性
if present_new_features:
    new_feat_importance = importance_df[importance_df['feature'].isin(present_new_features)]
    print(f"\n" + "=" * 70)
    print("新增特征分析")
    print("=" * 70)
    print(f"新特征在 Top 50 中的数量: {len(new_feat_importance.head(50))}")
    print(f"新特征总重要性占比: {new_feat_importance['importance'].sum() / importance_df['importance'].sum():.2%}")

    if len(new_feat_importance) > 0:
        print(f"\n新特征重要性排名:")
        for idx, row in new_feat_importance.iterrows():
            rank = importance_df.index.get_loc(idx) + 1
            print(f"  {row['feature']}: 排名 #{rank}, 重要性 {row['importance']:.1f}")

print("\n" + "=" * 70)
print("对比基线")
print("=" * 70)
print("基线 PR-AUC (仅大盘特征): 0.5871")
print(f"当前 PR-AUC (大盘+新特征): {pr_auc:.4f}")
if pr_auc > 0.5871:
    improvement = (pr_auc - 0.5871) / 0.5871 * 100
    print(f"提升: +{improvement:.2f}%")
else:
    decline = (0.5871 - pr_auc) / 0.5871 * 100
    print(f"下降: -{decline:.2f}%")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
