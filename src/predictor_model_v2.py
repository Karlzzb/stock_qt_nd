import pandas as pd
import numpy as np
import joblib
from comm_fun import model_config, get_return_threshold
import warnings
import os
import json
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
warnings.filterwarnings("ignore")
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from stock_model_Tflod_v2 import meta_features_process
from config.settings import MODEL_DIR, DATASET_DIR

class PriceChangePredictor:
    def __init__(self, model_dir=MODEL_DIR):
        """初始化预测器"""
        self.model_dir = model_dir
        self.models_loaded = False
        self.load_models()

    def load_models(self):
        """加载所有训练好的模型"""
        try:
            self.lgb_models = joblib.load(f"{self.model_dir}/lgb_models.pkl")
            self.lr_meta = joblib.load(f"{self.model_dir}/lr_meta.pkl")
            self.imputer = joblib.load(f"{self.model_dir}/imputer.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/stack_scaler.pkl")
            self.dim_reducer =  None
            if model_config.USE_PCA:
                self.dim_reducer = joblib.load(f"{self.model_dir}/dim_reducer.joblib")
            self.feature_scaler = joblib.load(f"{self.model_dir}/feature_scaler.pkl")
            self.fold_scores = joblib.load(f"{self.model_dir}/fold_scores.pkl")
            self.optimal_threshold = model_config.PROBA_THRESHOLD
            self.models_loaded = True
            logger.info(f"✅预测模型加载成功!")
            self.model_config = model_config
            # 加载特征列表
            with open(f"{self.model_dir}/lgbm_features.json", "r", encoding='utf-8') as f:
                self.lgbm_feature_names = json.load(f)
            with open(f"{self.model_dir}/lr_features.json", "r", encoding='utf-8') as f:
                self.lr_feature_names = json.load(f)

        except Exception as e:
            logger.error(f"❌模型加载失败: {e}")
            self.models_loaded = False

    def predict_proba(self, raw_features):
        """
        预测概率
        raw_features: DataFrame，包含所有特征列
        """
        if not self.models_loaded:
            raise ValueError("模型未加载")

        X = self._prepare_features(raw_features)


        # 缺失值填充
        X_feature = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )

        # === 加权融合 LightGBM 模型预测 ===
        weights = np.array(self.fold_scores)
        weights = weights / weights.sum()
        print(f"各fold权重: {np.round(weights, 3)}")

        # 按权重融合fold预测
        test_preds_lgb = np.average(
            [model.predict_proba(X_feature[self.lgbm_feature_names])[:, 1] for model in self.lgb_models],
            axis=0,
            weights=weights
        )

        # === 构造 stacking 特征 ===
        X_test_scaled, *_, lr_feature_names = meta_features_process(X_feature, test_preds_lgb, self.lgb_models, model_config,
                                                                   self.lgbm_feature_names, self.scaler, self.dim_reducer)

        final_proba = self.lr_meta.predict_proba(X_test_scaled)[:, 1]

        return final_proba

    def enhanced_evaluate(self,y_true, proba, title="Evaluation", save_path=None):
        """增强的评估函数，支持结果保存"""
        results = {}

        # 基础指标
        y_pred = (proba >= self.optimal_threshold).astype(int)
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
            self.save_evaluation_results(results, title, save_path)

        return results

    def save_evaluation_results(self,results, title, save_path):
        """保存评估结果到文件"""
        # 保存为文本文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")
            f.write("=" * 50 + "\n")
            for metric, value in results.items():
                f.write(f"{metric:20}: {value:.4f}\n")
            f.write(f"\n保存时间: {pd.Timestamp.now()}\n")
        print(f"\n结果已保存到: {save_path} (文本格式)")

    def evaluate(self, X_test, y_test, title="模型评估结果", save_path=None):
        """
        评估模型性能
        X_test: 测试集特征
        y_test: 测试集真实标签
        title: 评估标题
        save_path: 结果保存路径
        """
        if not self.models_loaded:
            raise ValueError("模型未加载")

        logger.info("开始模型评估...")

        # 获取预测概率
        proba = self.predict_proba(X_test)

        # 使用增强评估函数
        results = self.enhanced_evaluate(y_test, proba, title, save_path)

        logger.info("模型评估完成!")
        return results

    def _prepare_features(self, data):
        """准备特征"""
        feature_cols = model_config.FULL_FEATURE_COLS
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

        return features



# 初始化预测器
def main():
    predictor = PriceChangePredictor(MODEL_DIR)
    config = model_config

    # 文件加载
    csv_files = glob(os.path.join(DATASET_DIR, "validation_set.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {config.DATA_PATH}")

    dfs = []
    for file in csv_files:
        df1 = pd.read_csv(file)
        dfs.append(df1)

    # 合并数据
    data = pd.concat(dfs, ignore_index=True).sort_values(by=config.TIME_COLS)
    print(f"原始数据量: {len(data)}")


    # NOTE: 对ST和不能交易的进行过滤
    full_len = len(data)
    st_df = pd.read_csv(DATASET_DIR / 'st_stocks_list.csv')
    data = data[data['symbol'].str.match(r'^[60]')]
    data = data[~data['symbol'].isin(st_df['ts_code'])]
    print(f"原始数据量: {full_len} 过滤后数据量: {len(data)} 过滤后%: {(len(data)/full_len)* 100:.2f}%")
    print(f"正样本比例: {data['label'].mean():.4f}")

    # 直接评估模型
    X = data[model_config.FULL_FEATURE_COLS]
    y = (data[predictor.model_config.LABEL_COL] > get_return_threshold(data)).astype(int)
    predictor.evaluate(
        X_test=X,
        y_test=y,
        title="模型验证集表现",
        save_path= MODEL_DIR / "validation_report.txt"
)

if __name__ == "__main__":
    main()