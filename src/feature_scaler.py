import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class FeatureScaler:
    """特征标准化管理器"""

    def __init__(self, method='standard'):
        """
        Args:
            method: 标准化方法
                'standard': 标准正态分布 (均值为0，方差为1)
                'minmax': 最小最大归一化 (范围0-1)
                'robust': 稳健标准化 (使用中位数和四分位数)
                'quantile': 分位数归一化 (转换为均匀分布)
                'power': 幂变换 (Yeo-Johnson)
                'log': 对数变换
        """
        self.method = method
        self.scaler = None
        self.original_stats = None

    def fit_transform(self, X):
        """拟合并转换训练数据"""
        if self.method == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)

        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)

        elif self.method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(
                output_distribution='normal',
                random_state=42,
                n_quantiles=min(len(X), 1000)
            )
            X_scaled = self.scaler.fit_transform(X)

        elif self.method == 'power':
            from sklearn.preprocessing import PowerTransformer
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            X_scaled = self.scaler.fit_transform(X)

        elif self.method == 'log':
            # 对数变换 + 标准化
            X_positive = X.copy()
            # 确保所有值大于0
            min_val = X_positive.min().min()
            if min_val <= 0:
                shift = abs(min_val) + 0.001
                X_positive = X_positive + shift

            # 对数变换
            X_log = np.log1p(X_positive)

            # 再标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_log)

        else:
            X_scaled = X.values
            self.scaler = None

        # 保存原始数据统计信息
        if isinstance(X, pd.DataFrame):
            self.original_stats = {
                'mean': X.mean(),
                'std': X.std(),
                'min': X.min(),
                'max': X.max(),
                'median': X.median(),
                'skew': X.skew()
            }

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def transform(self, X):
        """转换新数据"""
        if self.scaler is None:
            return X

        if self.method == 'log':
            # 对数变换需要特殊处理
            X_positive = X.copy()
            min_val = X_positive.min().min()
            if min_val <= 0:
                shift = abs(min_val) + 0.001
                X_positive = X_positive + shift

            X_log = np.log1p(X_positive)
            X_scaled = self.scaler.transform(X_log)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def inverse_transform(self, X_scaled):
        """逆转换"""
        if self.scaler is None:
            return X_scaled

        if self.method == 'log':
            X_log = self.scaler.inverse_transform(X_scaled)
            X = np.expm1(X_log)
            # 如果原始数据有平移，需要反平移
            if hasattr(self, 'shift_value'):
                X = X - self.shift_value
        else:
            X = self.scaler.inverse_transform(X_scaled)

        return pd.DataFrame(X, columns=X_scaled.columns, index=X_scaled.index)

    def save(self, path):
        """保存标准化器"""
        joblib.dump({
            'method': self.method,
            'scaler': self.scaler,
            'original_stats': self.original_stats
        }, path)

    @classmethod
    def load(cls, path):
        """加载标准化器"""
        data = joblib.load(path)
        scaler_obj = cls(method=data['method'])
        scaler_obj.scaler = data['scaler']
        scaler_obj.original_stats = data['original_stats']
        return scaler_obj