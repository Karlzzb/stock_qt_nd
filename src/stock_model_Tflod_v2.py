import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
import lightgbm as lgb
import joblib
import warnings
from glob import glob
import os
import matplotlib.pyplot as plt
from comm_fun import model_config, get_return_threshold
import shap
import json
import traceback
from feature_scaler import FeatureScaler
from sklearn.utils.class_weight import  compute_class_weight, compute_sample_weight
from sklearn.decomposition import PCA  # 主成分分析 降维操作
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from config.settings import MODEL_DIR, DATASET_DIR

def find_optimal_threshold_beta(y_true, y_proba, beta=0.5):
    """
    beta < 1: 更加重视 Precision (适合对误报敏感的场景，如选股)
    beta > 1: 更加重视 Recall (适合对漏报敏感的场景，如癌症筛查)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # 加上 1e-8 防止除零
    numerator = (1 + beta ** 2) * (precisions[:-1] * recalls[:-1])
    denominator = ((beta ** 2 * precisions[:-1]) + recalls[:-1]) + 1e-8

    fbeta_scores = numerator / denominator

    optimal_idx = np.argmax(fbeta_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = fbeta_scores[optimal_idx]

    print(f"Beta={beta} 下的最优阈值: {optimal_threshold:.4f}, Score: {optimal_score:.4f}")
    return optimal_threshold

def find_optimal_threshold(y_true, y_proba):
    """寻找最优分类阈值"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    # 输出阈值搜索信息
    print(f"阈值搜索范围: {thresholds[0]:.3f} ~ {thresholds[-1]:.3f}")
    print(f"找到最优阈值: {optimal_threshold:.4f}, F1-score: {optimal_f1:.4f}")

    return optimal_threshold, optimal_f1

def load_and_preprocess_data(config):
    """加载并预处理数据 - 增强版本"""
    csv_files = glob(os.path.join(DATASET_DIR, "t*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {config.DATA_PATH}")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).sort_values(by=config.TIME_COLS)
    print(f"原始数据量: {len(data)}")

    # 创建标签
    data["label"] = (data[config.LABEL_COL] > get_return_threshold(data)).astype(int)

    # 2. 检查特征质量
    print("\n特征质量分析:")
    for col in config.OPTIMIZED_FEATURE_COLS:
        if data[col].dtype in ['float64', 'int64']:
            corr = data[col].corr(data['label'])
            print(f"  {col}: 与标签相关性={corr:.4f}")

    # 添加高级特征
    # data = create_advanced_features(data, config)

    print(f"最终数据量: {len(data)}")
    print(f"正样本比例: {data['label'].mean():.4f}")
    return data

def train_lgb_models(X_train, y_train, config):
    """使用优化参数的LightGBM训练"""
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    oof = np.zeros(len(X_train))
    models = []
    fold_scores = []

    # 计算全局类别权重
    positive_count = np.sum(y_train)
    negative_count = len(y_train) - positive_count
    # scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1
    print(f"正样本: {positive_count}, 负样本: {negative_count}")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"\nTimeSeries Fold {fold + 1}/{config.N_SPLITS}")
        print(f"  训练集: {len(tr_idx)}, 验证集: {len(val_idx)}")
        print(f"  训练集正样本比例: {y_train.iloc[tr_idx].mean():.4f}")

        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]


        # 方法1: 调整样本权重
        sample_weights = np.ones(len(y_tr))
        sample_weights[y_tr == 0] = config.FP_PENALTY_WEIGHT  # 负样本更高权重
        # 动态调整参数
        lgb_params = config.LGB_PARAMS.copy()

        # 调试打印
        class_weight = compute_class_weight('balanced', classes=np.unique(y_tr),y=y_tr)
        print(f"Fold {fold + 1} class_weight {class_weight}")
        auto_sample_weight = compute_sample_weight(class_weight='balanced',y=y_tr)
        print(f"Fold {fold + 1} Label [0] weight {auto_sample_weight[y_tr == 0]}")
        print(f"Fold {fold + 1} Label [1] weight {auto_sample_weight[y_tr == 1]}")


        # 根据数据量调整min_child_samples
        if len(tr_idx) < 10000:
            lgb_params['min_child_samples'] = max(5, int(len(tr_idx) * 0.01))
            print(f"  调整 min_child_samples 为: {lgb_params['min_child_samples']}")

        clf = lgb.LGBMClassifier(**lgb_params)

        clf.fit(
            X_tr, y_tr,
            # sample_weight=sample_weights,  # 应用样本权重
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'binary_logloss'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )

        oof[val_idx] = clf.predict_proba(X_val)[:, 1]

        models.append(clf)

        # 计算fold性能
        fold_score = average_precision_score(y_val, oof[val_idx])
        # fold_score, fp_rate = evaluate_with_fp_penalty(y_val, oof[val_idx], model_config.FP_PENALTY_WEIGHT)
        fold_scores.append(fold_score)

        # 计算该fold的最优阈值
        fold_optimal_threshold, fold_optimal_f1 = find_optimal_threshold(y_val, oof[val_idx])
        y_pred_optimal = (oof[val_idx] >= fold_optimal_threshold).astype(int)
        precision_optimal = precision_score(y_val, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_val, y_pred_optimal, zero_division=0)

        print(f"  Fold {fold + 1} PR-AUC: {fold_score:.4f}")
        # print(f"  Fold {fold + 1} PR-AUC: {fold_score:.4f}, FP率: {fp_rate:.4f}")
        print(f"  Fold {fold + 1} 最优阈值: {fold_optimal_threshold:.4f}, F1: {fold_optimal_f1:.4f}")
        print(f"  Fold {fold + 1} Precision: {precision_optimal:.4f}, Recall: {recall_optimal:.4f}")

    # 寻找全局最优阈值
    # optimal_threshold, optimal_f1 = find_optimal_threshold(y_train, oof)
    optimal_threshold = find_optimal_threshold_beta(y_train, oof)

    # 输出整体OOF性能
    oof_score = average_precision_score(y_train, oof)
    print(f"\n{'=' * 50}")
    print("整体OOF性能总结:")
    print(f"{'=' * 50}")
    print(f"整体OOF PR-AUC: {oof_score:.4f}")
    print(f"各Fold PR-AUC: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"平均Fold PR-AUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    # print(f"全局最优阈值: {optimal_threshold:.4f}, 对应F1: {optimal_f1:.4f}")
    print(f"全局最优阈值: {optimal_threshold:.4f}")

    # 对比不同阈值效果
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (oof >= threshold).astype(int)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        print(f"阈值 {threshold}: F1 = {f1:.4f}")

    return oof, models, optimal_threshold, fold_scores

def evaluate_with_fp_penalty(y_true, y_pred_proba, fp_penalty=1.0):
    """考虑FP惩罚的评估"""
    from sklearn.metrics import precision_recall_curve, confusion_matrix

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # 对每个阈值计算包含FP惩罚的分数
    penalized_scores = []
    for i, threshold in enumerate(thresholds):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 包含FP惩罚的分数
        penalized_f1 = (2 * tp) / (2 * tp + fp_penalty * fp + fn)
        penalized_scores.append(penalized_f1)

    best_idx = np.argmax(penalized_scores)
    best_threshold = thresholds[best_idx]
    best_score = penalized_scores[best_idx]

    # 计算最终预测的FP率
    final_pred = (y_pred_proba >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, final_pred).ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return best_score, fp_rate

def reduce_dimension(X, config, dim_reducer = None):
    """
    特征降维
    :param X: 训练数据
    :param config: 模型配置
    :param dim_reducer: PCA
    :return:
    """
    if dim_reducer is None:
        pca = PCA()
        pca.fit(X)
        explained_variance_radio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance_radio >= config.PCA_VARIANCE_THRESHOLD) + 1
        print(
            f"PCA降维: 保留{config.PCA_VARIANCE_THRESHOLD * 100}%方差需要{n_components}个主成分"
        )
        print(f"各主成分方差贡献率: {pca.explained_variance_ratio_}")
        dim_reducer = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        X = dim_reducer.fit_transform(X)
        print(f"PCA降维后特征数: {X.shape[1]}")
    else:
        X = dim_reducer.transform(X)

    return X, dim_reducer

def meta_features_process(X, oof_preds, lgb_models, config, lgbm_feature_names,
                          scaler = None,
                          pca = None):
    X_stack = pd.DataFrame(index=X.index)
    feature_names = []  # 用于存储所有特征名称

    # A 类：预测信号
    X_stack["pred_lgb"] = oof_preds
    X_stack["pred_lgb_squared"] = oof_preds ** 2
    feature_names.extend(["pred_lgb", "pred_lgb_squared"])  # 记录特征名

    # B 类：Leaf PCA
    if config.USE_LGBM_LEAF:
        leaf_all = []
        for model in lgb_models:
            leaf = model.predict(X[lgbm_feature_names], pred_leaf=True)
            leaf_all.append(leaf)
        leaf_all = np.concatenate(leaf_all, axis=1)
        selected = config.KEEP_LGBM_LEAF_TOP
        if model_config.USE_PCA:
            leaf_pca, pca = reduce_dimension(leaf_all, config, pca)
        else:
            leaf_pca = leaf_all
        # 动态生成leaf特征名称
        leaf_feature_names = []
        for i in range(selected):
            feature_name = f"leaf_pca_{i}" if model_config.USE_PCA else f"leaf_{i}"
            X_stack[feature_name] = leaf_pca[:, i]
            leaf_feature_names.append(feature_name)
        feature_names.extend(leaf_feature_names)  # 记录特征名

    # C 类：少量稳态特征
    stable_feats = config.STABLE_FEATURES  # 你自己定义 5–10 个
    X_stack[stable_feats] = X[stable_feats]
    feature_names.extend(stable_feats)  # 记录特征名

    # 标准化 + LR
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_stack)
    else:
        X_scaled = scaler.transform(X_stack)

    print(f"最终特征总数: {len(feature_names)}")
    print(f"X_stack 列数: {X_stack.shape[1]}")

    return X_scaled, scaler, pca, feature_names

def train_meta_model_reduce(X_train, y_train, oof_preds, lgb_models, config, lgbm_feature_names):

    X_scaled, scaler, pca, feature_names = meta_features_process(X_train, oof_preds, lgb_models, config, lgbm_feature_names)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        # class_weight = {0: 0.5, 1: 0.5}, # 大幅惩罚假阳性（错误预测为买入）
        random_state=config.RANDOM_STATE,
        C=0.01,  # 更强的正则化，防止过拟合
        solver='saga',
        penalty='l1',
        fit_intercept=True,  # 决定了：模型能不能“整体平移决策边界”， FALSE强行规定：0 分就是生死线
        # tol = 1e-4  # 添加容差
    )
    lr.fit(X_scaled, y_train)

    return lr, scaler, pca, feature_names, X_scaled


def enhanced_evaluate(y_true, proba, title="Evaluation", save_path=None):
        """增强的评估函数，支持结果保存"""
        results = {}

        # 基础指标
        y_pred = (proba >= model_config.PROBA_THRESHOLD).astype(int)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        results['pr_auc'] = average_precision_score(y_true, proba) if len(set(y_true)) > 1 else np.nan

        # 多阈值评估
        thresholds = [0.3, 0.5, 0.7]
        for thresh in thresholds:
            y_pred_thresh = (proba >= thresh).astype(int)
            results[f'f1_thresh_{thresh}'] = f1_score(y_true, y_pred_thresh, zero_division=0)

        # Top-K Precision
        n = len(y_true)
        top_ks = [max(1, int(0.01 * n)), max(1, int(0.05 * n)), min(100, n)]

        for k in top_ks:
            idx_top = np.argsort(proba)[-k:][::-1]
            prec_topk = y_true.iloc[idx_top].mean() if len(idx_top) > 0 else 0
            results[f'precision_top{k}'] = prec_topk

        # 输出结果
        print(f"\n{title}")
        print("=" * 50)
        for metric, value in results.items():
            print(f"{metric:20}: {value:.4f}")

        # 保存结果到文件
        if save_path:
            save_evaluation_results(results, title, save_path)

        return results

def main():
    """优化的主训练流程"""
    print("开始优化版模型训练...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 数据加载与预处理（增强版）
    data = load_and_preprocess_data(model_config)
    X, y = prepare_features(data, model_config)

    # 2. 数据集划分
    cut = int(len(X) * (1 - model_config.TEST_RATIO))
    X_train_raw = X.iloc[:cut].reset_index(drop=True)
    y_train = y.iloc[:cut].reset_index(drop=True)
    X_test_raw = X.iloc[cut:].reset_index(drop=True)
    y_test = y.iloc[cut:].reset_index(drop=True)

    # 3. 缺失值处理
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns)

    # 4. 标准化处理
    feature_scaler = FeatureScaler(method='standard')
    # X = feature_scaler.fit_transform(X_train_imputer)
    # X_test = feature_scaler.transform(X_test_imputer)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"  训练集正样本比例: {y_train.mean():.4f}")
    print(f"  测试集正样本比例: {y_test.mean():.4f}")

    # 5. 训练优化的LightGBM模型
    print(f"\n开始训练LightGBM模型 (TimeSeriesSplit, n_splits={model_config.N_SPLITS})...")
    lgbm_feature_names = model_config.OPTIMIZED_FEATURE_COLS
    oof_preds, lgb_models, optimal_threshold, fold_scores = train_lgb_models(X_train[lgbm_feature_names], y_train, model_config)
    # 保存特征列表
    with open(MODEL_DIR / "lgbm_features.json", "w", encoding='utf-8') as f:
        json.dump(lgbm_feature_names, f, ensure_ascii=False, indent=2)

    # 6. 训练元模型
    print("\n训练元模型...")
    lr_meta, scaler, dim_reducer, lr_feature_names, X_train_lr = train_meta_model_reduce(X_train, y_train, oof_preds, lgb_models, model_config, lgbm_feature_names)
    # 保存特征列表
    with open(MODEL_DIR / "lr_features.json", "w", encoding='utf-8') as f:
        json.dump(lr_feature_names, f, ensure_ascii=False, indent=2)

    # 7. 测试集预测
    print("\n测试集预测...")
    weights = np.array(fold_scores)
    weights = weights / weights.sum()
    print(f"各fold权重: {np.round(weights, 3)}")

    # 按权重融合fold预测
    test_preds_lgb = np.average(
        [model.predict_proba(X_test[lgbm_feature_names])[:, 1] for model in lgb_models],
        axis=0,
        weights=weights
    )

    # 准备测试集stacking特
    X_stack_test, *_, lr_feature_names = meta_features_process(X_test, test_preds_lgb, lgb_models, model_config, lgbm_feature_names, scaler, dim_reducer)

    final_proba = lr_meta.predict_proba(X_stack_test)[:, 1]

    # 8. 使用最优阈值评估
    print("\n最终模型评估:")
    enhanced_evaluate(
        y_test, final_proba,
        "优化参数最终模型测试集表现",
        MODEL_DIR / "evaluate_report.txt"
    )

    # 9. 模型保存
    joblib.dump(feature_scaler, MODEL_DIR / "feature_scaler.pkl")
    joblib.dump(lgb_models, MODEL_DIR / "lgb_models.pkl")
    joblib.dump(lr_meta, MODEL_DIR / "lr_meta.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(scaler, MODEL_DIR / "stack_scaler.pkl")
    joblib.dump(optimal_threshold, MODEL_DIR / "optimal_threshold.pkl")
    joblib.dump(fold_scores, MODEL_DIR / "fold_scores.pkl")
    if model_config.USE_PCA:
        joblib.dump(dim_reducer, MODEL_DIR / "dim_reducer.joblib")

    print(f"\n模型保存完成!")
    print(f"推荐使用阈值: {optimal_threshold:.4f} 进行预测")

    # 10. 特征重要性分析
    print("\n特征重要性分析...")
    lgb_importance = analyze_feature_importance(lgb_models, lgbm_feature_names)
    #方法2: SHAP分析
    shap_lgbm_importance = pd.DataFrame()
    shap_lr_importance = pd.DataFrame()
    try:
        shap_lgbm_importance = analyze_lgbm_shap_importance(lgb_models, X_train[lgbm_feature_names], lgbm_feature_names)
        shap_lr_importance = analyze_lr_shap_importance(lr_meta, X_train_lr, lr_feature_names)
    except Exception as e:
        print(f"SHAP分析失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()  # 打印完整的堆栈跟踪
    # 方法3: 元模型特征重要性
    meta_importance = analyze_meta_feature_importance(lr_meta, lr_feature_names)
    # 方法4: 相关性分析
    corr_analysis = analyze_feature_correlation(X_train, y_train, lr_feature_names)
    # 按特征名合并两个重要性结果
    merged_importance = meta_importance
    if not shap_lr_importance.empty:
        merged_importance = pd.merge(
            shap_lr_importance,
            meta_importance[['feature', 'importance', 'coefficient']],
            on='feature',
            how='outer',  # 外连接，保留所有特征
            suffixes = ('_shap', '_meta')
        )
    # 保存重要性结果
    with pd.ExcelWriter(MODEL_DIR / "feature_importance_analysis.xlsx") as writer:
        lgb_importance.to_excel(writer, sheet_name='LGB重要性')
        shap_lgbm_importance.to_excel(writer, sheet_name='LGBM SHAP重要性')
        shap_lr_importance.to_excel(writer, sheet_name='LR SHAP重要性')
        meta_importance.to_excel(writer, sheet_name='元模型重要性')
        corr_analysis.to_excel(writer, sheet_name='相关性分析')
        merged_importance.to_excel(writer, sheet_name='SHAP+元模型重要性')


    # 11. 在最终评估后添加过拟合检查
    print("\n=== 过拟合检查 ===")
    # 1. 训练集评估（使用训练时的OOF预测）
    train_oof_pr_auc = average_precision_score(y_train, oof_preds)
    print(f"训练集OOF PR-AUC: {train_oof_pr_auc:.4f}")

    # 2. 测试集评估（已有final_proba）
    test_pr_auc = average_precision_score(y_test, final_proba)
    print(f"测试集 PR-AUC: {test_pr_auc:.4f}")

    # 3. 计算过拟合度
    overfitting_ratio = (train_oof_pr_auc - test_pr_auc) / test_pr_auc * 100
    print(f"过拟合度: {overfitting_ratio:.2f}% (正值表示过拟合)")

    # 4. 如果过拟合严重，给出警告
    if overfitting_ratio > 20:  # 超过20%认为严重过拟合
        print("⚠️ 警告: 模型可能存在严重过拟合!")
        print("建议:")
        print("  1. 降低PCA_VARIANCE_THRESHOLD")
        print("  2. 增加逻辑回归正则化(C参数降低)")
        print("  3. 减少特征数量")

    print("\n🎯 优化训练完成!")


def analyze_lgbm_shap_importance(lgb_models, X_train, feature_names, top_k=20):
    """使用SHAP分析特征重要性"""
    print("\n" + "=" * 50)
    print("SHAP 特征重要性分析")
    print("=" * 50)

    # 合并所有模型的预测
    explainer = shap.TreeExplainer(lgb_models[0])
    shap_values_list = []

    for model in lgb_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap_values_list.append(shap_values[1] if isinstance(shap_values, list) else shap_values)

    # 平均SHAP值
    mean_shap_values = np.mean([np.abs(values).mean(0) for values in shap_values_list], axis=0)

    # 创建DataFrame
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_shap_values
    }).sort_values('shap_importance', ascending=False)

    # 输出结果
    print(f"Top-{top_k} SHAP重要性:")
    print("-" * 50)
    for i, row in shap_importance.head(top_k).iterrows():
        print(f"{i + 1:2d}. {row['feature']:30s}: {row['shap_importance']:.4f}")

    return shap_importance

def analyze_lr_shap_importance(lr_model, X_train, feature_names, top_k=20):
    """使用SHAP分析逻辑回归特征重要性"""
    print("\n" + "=" * 50)
    print("逻辑回归层 SHAP 特征重要性分析")
    print("=" * 50)

    # 使用 LinearExplainer
    explainer = shap.LinearExplainer(lr_model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)  # 返回每个样本每个特征的SHAP值

    # 平均SHAP值
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # 创建DataFrame
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_shap_values
    }).sort_values('shap_importance', ascending=False)

    # 输出结果
    print(f"Top-{top_k} SHAP重要性:")
    print("-" * 50)
    for i, row in enumerate(shap_importance.head(top_k).itertuples()):
        print(f"{i + 1:2d}. {row.feature:30s}: {row.shap_importance:.4f}")

    return shap_importance


def analyze_meta_feature_importance(lr_meta, meta_feature_names, top_k=20):
    """分析元模型特征重要性"""
    print("\n" + "=" * 50)
    print("元模型特征重要性")
    print("=" * 50)

    # 检查维度是否匹配
    if len(meta_feature_names) != len(lr_meta.coef_[0]):
        print(f"警告: 特征数量不匹配! 特征名: {len(meta_feature_names)}, 系数: {len(lr_meta.coef_[0])}")
        # 如果维度不匹配，使用通用名称
        meta_feature_names = [f'feature_{i}' for i in range(len(lr_meta.coef_[0]))]

    # 逻辑回归的系数绝对值作为重要性
    coef_importance = pd.DataFrame({
        'feature': meta_feature_names,
        'coefficient': lr_meta.coef_[0],
        'importance': np.abs(lr_meta.coef_[0])
    }).sort_values('importance', ascending=False)

    print(f"Top-{top_k} 元模型特征重要性:")
    print("-" * 50)
    for i, (_, row) in enumerate(coef_importance.head(top_k).iterrows()):
        print(f"{i + 1:2d}. {row['feature']:30s}: {row['importance']:.4f} (coef: {row['coefficient']:.4f})")

    # 新增：按特征类型分组统计重要性
    print("\n按特征类型统计重要性:")
    print("-" * 50)

    # 定义特征类型
    feature_categories = {
        "pred_features": ["pred_lgb", "pred_lgb_squared"],
        "leaf_features": [f for f in meta_feature_names if f.startswith("leaf_")],
        "stable_features": [f for f in meta_feature_names if f not in ["pred_lgb", "pred_lgb_squared"]
                            and not f.startswith("leaf_")]
    }

    for category, features in feature_categories.items():
        if features:
            cat_importance = coef_importance[coef_importance['feature'].isin(features)]['importance'].sum()
            cat_mean_coef = coef_importance[coef_importance['feature'].isin(features)]['coefficient'].mean()
            print(
                f"{category:15s}: 总重要性={cat_importance:.4f}, 平均系数={cat_mean_coef:.4f}, 特征数={len(features)}")

    return coef_importance

def analyze_feature_correlation(X, y, feature_names, top_k=20):
    """分析特征与目标的相关性"""
    print("\n" + "=" * 50)
    print("特征-目标相关性分析")
    print("=" * 50)

    correlations = []
    for col in X.columns:
        if len(X[col].unique()) > 1:  # 避免常数特征
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations.append((col, corr))

    corr_df = pd.DataFrame(correlations, columns=['feature', 'correlation'])
    corr_df['abs_correlation'] = np.abs(corr_df['correlation'])
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    print(f"Top-{top_k} 特征-目标相关性:")
    print("-" * 50)
    for i, row in corr_df.head(top_k).iterrows():
        print(f"{i + 1:2d}. {row['feature']:30s}: {row['correlation']:.4f}")

    return corr_df

def analyze_feature_importance(lgb_models, feature_names, top_n=250, threshold=0.6):
    """分析特征重要性"""
    print(f"\n{'=' * 50}")
    print("特征重要性分析")
    print(f"{'=' * 50}")

    importance_df = pd.DataFrame(index=feature_names)
    n_folds = len(lgb_models)

    for i, model in enumerate(lgb_models):
        importance_df[f'fold_{i}'] = model.feature_importances_

    importance_df['importance_mean'] = importance_df.mean(axis=1)
    importance_df['importance_std'] = importance_df.std(axis=1)
    importance_df = importance_df.sort_values('importance_mean', ascending=False)

    # 2. 分析在多数折叠层中都重要的特征
    print(f"\n在≥{threshold * 100:.0f}%折叠层中都进入Top-{top_n}的特征:")
    print("-" * 60)

    # 对于每个特征，计算在多少个fold中进入前top_n
    feature_top_counts = []
    fold_importance_ranks = []

    for i in range(n_folds):
        # 获取当前fold的特征重要性并排序
        fold_importance = importance_df[f'fold_{i}']
        fold_rank = fold_importance.rank(ascending=False)  # 重要性越高，排名值越小
        fold_importance_ranks.append(fold_rank)

        # 找出当前fold中进入前top_n的特征
        top_features_in_fold = fold_importance.nlargest(top_n).index.tolist()

        for feature in feature_names:
            if feature in top_features_in_fold:
                # 如果是第一个fold，初始化计数
                if i == 0:
                    feature_top_counts.append({'feature': feature, 'count': 1})
                else:
                    # 找到对应的特征并增加计数
                    for item in feature_top_counts:
                        if item['feature'] == feature:
                            item['count'] += 1
                            break
                    else:
                        # 如果特征不在列表中，添加它
                        feature_top_counts.append({'feature': feature, 'count': 1})

    # 转换为DataFrame以便分析
    stability_df = pd.DataFrame(feature_top_counts)
    if not stability_df.empty:
        stability_df['percentage'] = stability_df['count'] / n_folds

        # 筛选出达到阈值的特征
        stable_features = stability_df[stability_df['percentage'] >= threshold].copy()
        stable_features = stable_features.sort_values('percentage', ascending=False)

        if not stable_features.empty:
            print(f"找到 {len(stable_features)} 个稳定重要特征:")
            # 保存稳定特征列表
            stable_features_list = stable_features['feature'].tolist()
            stable_features_summary = pd.DataFrame({
                'feature': stable_features['feature'],
                'fold_count': stable_features['count'],
                'fold_percentage': stable_features['percentage'],
                'overall_rank': [importance_df.index.get_loc(f) + 1 for f in stable_features['feature']],
                'importance_mean': [importance_df.loc[f, 'importance_mean'] for f in stable_features['feature']],
                'importance_std': [importance_df.loc[f, 'importance_std'] for f in stable_features['feature']]
            })

            # 保存到文件
            stable_features_summary.to_csv(MODEL_DIR / 'lgb_stable_features.csv', index=False)
            print(f"\n稳定特征列表已保存到: {MODEL_DIR / 'lgb_stable_features.csv'}")
        else:
            print(f"未找到在≥{threshold * 100:.0f}%折叠层中都进入前{top_n}的特征")
    else:
        print("未找到任何重要特征")

    return importance_df


def save_evaluation_results(results, title, save_path):
    """保存评估结果到文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        for metric, value in results.items():
            f.write(f"{metric:25}: {value:.4f}\n")
        f.write(f"\n保存时间: {pd.Timestamp.now()}\n")
    print(f"结果已保存到: {save_path}")



def prepare_features(data, config):
    """准备特征"""
    feature_cols = config.FULL_FEATURE_COLS
    # 将特征列转换为低精度浮点数（float32）
    features = data[feature_cols].astype(np.float32)

    # 处理无穷大和NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # 填充NaN（选择适合你业务逻辑的方式）
    # 方法A：用0填充
    features = features.fillna(0)

    # 方法B：用中位数填充
    # for col in features.columns:
    #     median_val = features[col].median()
    #     features[col] = features[col].fillna(median_val)

    print(f"特征数量: {len(feature_cols)}")
    print(f"正样本比例: {data['label'].mean():.4f}")

    return features, data["label"]

if __name__ == "__main__":
    main()