import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import tushare as ts
from sklearn.preprocessing import LabelEncoder
from config.settings import STOCK_ND_CSV_DIR, DATASET_DIR
import numpy as np
def load_price_data_for_symbol(symbol, start_date=None, end_date=None, data_folder_path=str(STOCK_ND_CSV_DIR)):
    """
    根据股票代码和日期范围加载价格数据

    参数:
    symbol: str, 股票代码
    start_date: datetime, 开始日期
    end_date: datetime, 结束日期
    data_folder_path: str, 数据文件夹路径

    返回:
    pandas.DataFrame: 指定日期范围内的价格数据
    """

    # 构建文件名
    file_path = os.path.join(data_folder_path, f"{symbol}_price_data.pkl")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return None

    try:
        # 加载pickle文件
        price_df = pd.read_pickle(file_path)

        if start_date is None and end_date is None:
            return price_df

        # 将索引转换为datetime（假设索引就是timestamp字符串）
        price_df.index = pd.to_datetime(price_df.index)



        # 按日期范围筛选数据
        mask = (price_df.index >= start_date) & (price_df.index <= end_date)
        filtered_data = price_df[mask].sort_index()

        # print(f"成功加载 {symbol} 从 {start_date} 到 {end_date} 的数据，共 {len(filtered_data)} 行")

        return filtered_data

    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def init_tushare(token):
    """初始化tushare"""
    ts.set_token(token)
    return ts.pro_api()

def process_cvs_data(file_path, n_rows=None, days_ago=[1, 180]):
    """
    处理CVS文件，提取指定字段并计算前N天的日期

    参数:
    file_path: str, CSV文件路径
    n_rows: int or None, 读取的行数，None表示读取所有行
    days_ago: list, 需要计算的前几天，默认[1, 150]表示前1天和前150天

    返回:
    pandas.DataFrame: 处理后的数据框
    """

    # 读取CSV文件
    df = pd.read_csv(file_path, nrows=n_rows)

    print(f"成功读取 {len(df)} 行数据")

    # 转换日期列为datetime格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prev_time'] = pd.to_datetime(df['prev_time'])

    # 提取所需的字段
    result_df = df[['symbol', 'timestamp', 'prev_time', 'close_previous', 'close_current', 'macd_increase_pct']].copy()

    # 为每个指定的天数计算前N天的日期
    for days in days_ago:
        # 计算前N天的日期
        date_col_name = f'date_{days}d_ago'
        result_df[date_col_name] = result_df['timestamp'] - pd.Timedelta(days=days)

    # 显示处理结果信息
    # print(f"处理后的数据行数: {len(result_df)}")
    # print(f"包含的股票代码数量: {result_df['symbol'].nunique()}")
    # print(f"时间范围: {result_df['timestamp'].min()} 到 {result_df['timestamp'].max()}")

    return result_df

def calculate_total_fees(amount, broker_rate=0.00025):
    """
    计算一次完整买卖交易的总手续费（买入和卖出金额相同）

    Parameters:
    amount: 交易金额（元）
    broker_rate: 券商佣金费率，默认万2.5(0.025%)

    Returns:
    dict: 包含各项费用的字典
    """

    # 计算买入费用
    buy_broker_fee = max(amount * broker_rate, 5)  # 佣金，最低5元
    buy_transfer_fee = amount * 0.00001  # 过户费
    buy_total_fees = buy_broker_fee + buy_transfer_fee

    # 计算卖出费用
    sell_broker_fee = max(amount * broker_rate, 5)  # 佣金，最低5元
    sell_stamp_duty = amount * 0.0005  # 印花税
    sell_transfer_fee = amount * 0.00001  # 过户费
    sell_total_fees = sell_broker_fee + sell_stamp_duty + sell_transfer_fee

    # 计算总费用
    total_fees = buy_total_fees + sell_total_fees

    # return {
    #     '交易金额': amount,
    #     '买入费用': {
    #         '佣金': buy_broker_fee,
    #         '过户费': buy_transfer_fee,
    #         '小计': buy_total_fees
    #     },
    #     '卖出费用': {
    #         '佣金': sell_broker_fee,
    #         '印花税': sell_stamp_duty,
    #         '过户费': sell_transfer_fee,
    #         '小计': sell_total_fees
    #     },
    #     '总手续费': total_fees,
    #     '费用占比': (total_fees / amount) * 100
    # }
    return total_fees


def label_encoding(df, categorical_cols=None):
    """
    将分类标签（字符串或非数值型数据）转换为数值标签
    自动检查列是否为数值类型，非数值列才进行转换
    """
    if categorical_cols is None:
        categorical_cols = ["volume_signal", "hammer_signal"]

    label_encoders = {}

    for col in categorical_cols:
        # 检查列是否存在
        if col not in df.columns:
            print(f"警告ERROR: 列 '{col}' 不存在于DataFrame中，已跳过")
            continue

        # 检查列是否已经是数值类型
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"提示WARNING: 列 '{col}' 已经是数值类型，无需转换")
            continue

        # 对于非数值列进行标签编码
        le = LabelEncoder()
        df[col] = df[col].fillna("missing")
        df[col] = le.fit_transform(df[col].astype(str))  # 确保转换为字符串
        label_encoders[col] = le
        print(f"已转换列: {col}")

    return df, label_encoders

def get_return_threshold(data, quantile=0.62):
    """动态计算阈值"""
    # return np.quantile(data[model_config.LABEL_COL].dropna(), quantile)
    return 0.01

class Config:
    DATA_PATH = str(DATASET_DIR)
    TEST_RATIO = 0.15
    TIME_COLS = ["timestamp"]
    N_SPLITS = 5
    RANDOM_STATE = 42
    SETTLEMENT_DAYS = 15
    LABEL_COL = f"future_return_{SETTLEMENT_DAYS}d"
    EXPECTED_PROFIT = 1.15 # 特征中的止盈
    EXPECTED_LOSS = 0.65 # 特征中的止损
    RETURN_THRESHOLD = 0.01  # 分类阈值
    PROBA_THRESHOLD = 0.7 #阈值就是判断"是"与"否"的分界线。
    FP_PENALTY_WEIGHT = 1.0 # 暂时没用到

    USE_LGBM_LEAF = False # 是否使用LGBM的叶子节点特征作为LR层的输入
    USE_PCA = False  # 是否将LGBM的叶子节点特征降维
    KEEP_LGBM_LEAF_TOP = 15 # 在LGBM的叶子节点特征中选择几个

    PCA_VARIANCE_THRESHOLD = 0.95 # PCA降维特征
    AFFORDABLE_PRICE = 100 # 最大买卖股价
    RETURN_PERIODS = [3, 5, 10, 15, 20, 25, 30]
    WINDOWS_VOLATILITY = [3, 5, 10, 15, 20, 25, 30]  # 波动率计算窗口（与RETURN_PERIODS匹配）
    LAG_PERIODS = [3, 5, 10, 15, 20, 25, 30]
    FEATURE_NEED_MAX_DAYS =  100 # 用于特征计算的最大天数


    LGB_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'metric': ['auc','binary_logloss'],
        'num_leaves': 63,#num_leaves 从 31 改成 63（让模型能学更细的规律）
        'objective': 'binary',
        'n_jobs': 4,
        'random_state': RANDOM_STATE,
        'max_depth': 4, #避免学杂了
        'min_child_samples': 58,
        'min_split_gain': 0.6, # 它控制的是：模型是否愿意为了区分少数样本而“冒险”
        'scale_pos_weight': 1.0,
        'verbosity': -1,  # 屏蔽 LightGBM 警告输出
        "reg_alpha": 2,
        "reg_lambda": 2,
        "min_child_weight": 0.03,
        # 重复的组合，用其一
        'bagging_fraction': 0.875,
        # "subsample": 0.8,
        'bagging_freq': 7,  # 启用 bagging
        # "subsample_freq": 10,
        'feature_fraction': 0.85,
        # "colsample_bytree": 0.66,
    }

    # 超级扩展
    FULL_FEATURE_COLS = [
        'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi_6', 'rsi_14', 'rsi_24', 'ma_5', 'ma_20', 'ma_60', 'bb_upper',
        'bb_middle', 'bb_lower', 'volume_ma_20', 'obv', 'atr', 'slowk', 'slowd', 'price_vs_ma5', 'price_vs_ma20',
        'price_vs_ma60', 'ma_arrangement', 'bb_position', 'bb_squeeze', 'rsi_oversold_6', 'rsi_oversold_14',
        'rsi_oversold_24', 'rsi_overbought_6', 'rsi_overbought_14', 'rsi_overbought_24', 'macd_signal_distance',
        'macd_golden_cross', 'volume_ma20', 'volume_ratio', 'volume_spike', 'volume_dryup', 'atr_ratio',
        'hammer_pattern', 'downtrend', 'hammer_signal', 'doji_pattern', 'distance_to_support', 'distance_to_resistance',
        'stoch_oversold', 'stoch_overbought', 'macd_percentile', 'obv_trend', 'volatility_3d', 'volatility_5d',
        'volatility_10d', 'volatility_15d', 'volatility_20d', 'volatility_25d', 'volatility_30d', 'engulfing_pattern',
        'signed_volume_strength', 'close_vs_high', 'volume_ma_ratio', 'rsi_momentum', 'rsi_turning_simple',
        'rsi_turning', 'volume_trend_5', 'volume_trend_10', 'price_trend_5', 'price_volume_divergence',
        'volume_consistency', 'macd_hist_trend_5', 'macd_death_cross', 'macd_signal_cross', 'macd_zero_cross_up',
        'macd_zero_cross_down', 'macd_zero_cross', 'macd_hist_amplitude', 'macd_hist_direction',
        'macd_hist_acceleration', 'macd_signal_convergence', 'macd_signal_convergence_trend', 'pct_change', 'clv',
        'upper_shadow_ratio', 'body_strength', 'rank_return', 'rank_volume', 'signed_vol_strength', 'pv_corr_10',
        'dist_to_high_60', 'vol_divergence', 'vol_gk', 'vol_gk_ratio', 'illiq', 'efficiency_ratio', 'intraday_pos',
        'ret_overnight', 'ret_intraday', 'smart_money_diff', 'high_mean_20', 'low_mean_20', 'support_resistance_ratio',
        'log_volume', 'boxcox_atr', 'rsi_robust', 'macd_robust', 'close_lag_3',
        'open_lag_3', 'high_lag_3', 'low_lag_3', 'close_lag_5', 'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_10',
        'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_15', 'open_lag_15', 'high_lag_15', 'low_lag_15',
        'close_lag_20', 'open_lag_20', 'high_lag_20', 'low_lag_20', 'close_lag_25', 'open_lag_25', 'high_lag_25',
        'low_lag_25', 'close_lag_30', 'open_lag_30', 'high_lag_30', 'low_lag_30', 'volume_lag_3', 'volume_lag_5',
        'volume_lag_10', 'volume_lag_15', 'volume_lag_20', 'volume_lag_25', 'volume_lag_30', 'daily_return',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20', 'amplitude',
        'amplitude_lag_1', 'amplitude_lag_3', 'amplitude_lag_5', 'vol_gk_lag_1', 'vol_gk_ratio_lag_1', 'vol_gk_lag_3',
        'vol_gk_ratio_lag_3', 'vol_gk_lag_5', 'vol_gk_ratio_lag_5', 'vol_gk_lag_10', 'vol_gk_ratio_lag_10',
        'illiq_lag_1', 'illiq_lag_3', 'illiq_lag_5', 'efficiency_ratio_lag_1', 'efficiency_ratio_lag_3',
        'efficiency_ratio_lag_5', 'intraday_pos_lag_1', 'intraday_pos_lag_2', 'intraday_pos_lag_3',
        'smart_money_diff_lag_1', 'ret_overnight_lag_1', 'ret_intraday_lag_1', 'smart_money_diff_lag_3',
        'ret_overnight_lag_3', 'ret_intraday_lag_3', 'smart_money_diff_lag_5', 'ret_overnight_lag_5',
        'ret_intraday_lag_5', 'support_resistance_ratio_lag_1', 'support_resistance_ratio_lag_3',
        'support_resistance_ratio_lag_5', 'sh_price_change', 'sh_amplitude', 'sh_volume_ratio', 'sh_price_change_abs',
        'sh_price_wave_abs', 'sh_sentiment', 'sh_volume_signal', 'sz_price_change', 'sz_amplitude', 'sz_volume_ratio',
        'sz_price_change_abs', 'sz_price_wave_abs', 'sz_sentiment', 'sz_volume_signal', 'sh_sz_sync_direction',
        'sh_sz_sync_strength', 'market_sentiment', 'market_avg_change', 'market_avg_amplitude', 'market_sync_score',
        'cs_n', 'volume_rankpct', 'volume_z', 'macd_rankpct', 'macd_z', 'macd_signal_rankpct', 'macd_signal_z',
        'macd_hist_rankpct', 'macd_hist_z', 'rsi_6_rankpct', 'rsi_6_z', 'rsi_14_rankpct', 'rsi_14_z', 'rsi_24_rankpct',
        'rsi_24_z', 'ma_5_rankpct', 'ma_5_z', 'ma_20_rankpct', 'ma_20_z', 'ma_60_rankpct', 'ma_60_z',
        'bb_upper_rankpct', 'bb_upper_z', 'bb_middle_rankpct', 'bb_middle_z', 'bb_lower_rankpct', 'bb_lower_z',
        'volume_ma_20_rankpct', 'volume_ma_20_z', 'obv_rankpct', 'obv_z', 'atr_rankpct', 'atr_z', 'slowk_rankpct',
        'slowk_z', 'slowd_rankpct', 'slowd_z', 'price_vs_ma5_rankpct', 'price_vs_ma5_z', 'price_vs_ma20_rankpct',
        'price_vs_ma20_z', 'price_vs_ma60_rankpct', 'price_vs_ma60_z', 'ma_arrangement_rankpct', 'ma_arrangement_z',
        'bb_position_rankpct', 'bb_position_z', 'bb_squeeze_rankpct', 'bb_squeeze_z', 'rsi_oversold_6_rankpct',
        'rsi_oversold_6_z', 'rsi_oversold_14_rankpct', 'rsi_oversold_14_z', 'rsi_oversold_24_rankpct',
        'rsi_oversold_24_z', 'rsi_overbought_6_rankpct', 'rsi_overbought_6_z', 'rsi_overbought_14_rankpct',
        'rsi_overbought_14_z', 'rsi_overbought_24_rankpct', 'rsi_overbought_24_z', 'macd_signal_distance_rankpct',
        'macd_signal_distance_z', 'macd_golden_cross_rankpct', 'macd_golden_cross_z', 'volume_ma20_rankpct',
        'volume_ma20_z', 'volume_ratio_rankpct', 'volume_ratio_z', 'volume_spike_rankpct', 'volume_spike_z',
        'volume_dryup_rankpct', 'volume_dryup_z', 'atr_ratio_rankpct', 'atr_ratio_z', 'hammer_pattern_rankpct',
        'hammer_pattern_z', 'doji_pattern_rankpct', 'doji_pattern_z', 'distance_to_support_rankpct',
        'distance_to_support_z', 'distance_to_resistance_rankpct', 'distance_to_resistance_z', 'stoch_oversold_rankpct',
        'stoch_oversold_z', 'stoch_overbought_rankpct', 'stoch_overbought_z', 'macd_percentile_rankpct',
        'macd_percentile_z', 'obv_trend_rankpct', 'obv_trend_z', 'volatility_3d_rankpct', 'volatility_3d_z',
        'volatility_5d_rankpct', 'volatility_5d_z', 'volatility_10d_rankpct', 'volatility_10d_z',
        'volatility_15d_rankpct', 'volatility_15d_z', 'volatility_20d_rankpct', 'volatility_20d_z',
        'volatility_25d_rankpct', 'volatility_25d_z', 'volatility_30d_rankpct', 'volatility_30d_z',
        'engulfing_pattern_rankpct', 'engulfing_pattern_z', 'signed_volume_strength_rankpct',
        'signed_volume_strength_z', 'close_vs_high_rankpct', 'close_vs_high_z', 'volume_ma_ratio_rankpct',
        'volume_ma_ratio_z', 'rsi_momentum_rankpct', 'rsi_momentum_z', 'rsi_turning_simple_rankpct',
        'rsi_turning_simple_z', 'rsi_turning_rankpct', 'rsi_turning_z', 'volume_trend_5_rankpct', 'volume_trend_5_z',
        'volume_trend_10_rankpct', 'volume_trend_10_z', 'price_trend_5_rankpct', 'price_trend_5_z',
        'price_volume_divergence_rankpct', 'price_volume_divergence_z', 'volume_consistency_rankpct',
        'volume_consistency_z', 'macd_hist_trend_5_rankpct', 'macd_hist_trend_5_z', 'macd_death_cross_rankpct',
        'macd_death_cross_z', 'macd_signal_cross_rankpct', 'macd_signal_cross_z', 'macd_zero_cross_up_rankpct',
        'macd_zero_cross_up_z', 'macd_zero_cross_down_rankpct', 'macd_zero_cross_down_z', 'macd_zero_cross_rankpct',
        'macd_zero_cross_z', 'macd_hist_amplitude_rankpct', 'macd_hist_amplitude_z', 'macd_hist_direction_rankpct',
        'macd_hist_direction_z', 'macd_hist_acceleration_rankpct', 'macd_hist_acceleration_z',
        'macd_signal_convergence_rankpct', 'macd_signal_convergence_z', 'macd_signal_convergence_trend_rankpct',
        'macd_signal_convergence_trend_z', 'pct_change_rankpct', 'pct_change_z', 'clv_rankpct', 'clv_z',
        'upper_shadow_ratio_rankpct', 'upper_shadow_ratio_z', 'body_strength_rankpct', 'body_strength_z',
        'rank_return_rankpct', 'rank_return_z', 'rank_volume_rankpct', 'rank_volume_z', 'signed_vol_strength_rankpct',
        'signed_vol_strength_z', 'pv_corr_10_rankpct', 'pv_corr_10_z', 'dist_to_high_60_rankpct', 'dist_to_high_60_z',
        'vol_divergence_rankpct', 'vol_divergence_z', 'vol_gk_rankpct', 'vol_gk_z', 'vol_gk_ratio_rankpct',
        'vol_gk_ratio_z', 'illiq_rankpct', 'illiq_z', 'efficiency_ratio_rankpct', 'efficiency_ratio_z',
        'intraday_pos_rankpct', 'intraday_pos_z', 'ret_overnight_rankpct', 'ret_overnight_z', 'ret_intraday_rankpct',
        'ret_intraday_z', 'smart_money_diff_rankpct', 'smart_money_diff_z', 'high_mean_20_rankpct', 'high_mean_20_z',
        'low_mean_20_rankpct', 'low_mean_20_z', 'support_resistance_ratio_rankpct', 'support_resistance_ratio_z',
        'log_volume_rankpct', 'log_volume_z', 'boxcox_atr_rankpct', 'boxcox_atr_z', 'rsi_robust_rankpct',
        'rsi_robust_z', 'macd_robust_rankpct', 'macd_robust_z', 'close_lag_3_rankpct', 'close_lag_3_z', 'open_lag_3_rankpct',
        'open_lag_3_z', 'high_lag_3_rankpct', 'high_lag_3_z', 'low_lag_3_rankpct', 'low_lag_3_z', 'close_lag_5_rankpct',
        'close_lag_5_z', 'open_lag_5_rankpct', 'open_lag_5_z', 'high_lag_5_rankpct', 'high_lag_5_z',
        'low_lag_5_rankpct', 'low_lag_5_z', 'close_lag_10_rankpct', 'close_lag_10_z', 'open_lag_10_rankpct',
        'open_lag_10_z', 'high_lag_10_rankpct', 'high_lag_10_z', 'low_lag_10_rankpct', 'low_lag_10_z',
        'close_lag_15_rankpct', 'close_lag_15_z', 'open_lag_15_rankpct', 'open_lag_15_z', 'high_lag_15_rankpct',
        'high_lag_15_z', 'low_lag_15_rankpct', 'low_lag_15_z', 'close_lag_20_rankpct', 'close_lag_20_z',
        'open_lag_20_rankpct', 'open_lag_20_z', 'high_lag_20_rankpct', 'high_lag_20_z', 'low_lag_20_rankpct',
        'low_lag_20_z', 'close_lag_25_rankpct', 'close_lag_25_z', 'open_lag_25_rankpct', 'open_lag_25_z',
        'high_lag_25_rankpct', 'high_lag_25_z', 'low_lag_25_rankpct', 'low_lag_25_z', 'close_lag_30_rankpct',
        'close_lag_30_z', 'open_lag_30_rankpct', 'open_lag_30_z', 'high_lag_30_rankpct', 'high_lag_30_z',
        'low_lag_30_rankpct', 'low_lag_30_z', 'volume_lag_3_rankpct', 'volume_lag_3_z', 'volume_lag_5_rankpct',
        'volume_lag_5_z', 'volume_lag_10_rankpct', 'volume_lag_10_z', 'volume_lag_15_rankpct', 'volume_lag_15_z',
        'volume_lag_20_rankpct', 'volume_lag_20_z', 'volume_lag_25_rankpct', 'volume_lag_25_z', 'volume_lag_30_rankpct',
        'volume_lag_30_z', 'daily_return_rankpct', 'daily_return_z', 'return_lag_1_rankpct', 'return_lag_1_z',
        'return_lag_2_rankpct', 'return_lag_2_z', 'return_lag_3_rankpct', 'return_lag_3_z', 'return_lag_5_rankpct',
        'return_lag_5_z', 'return_lag_10_rankpct', 'return_lag_10_z', 'return_lag_20_rankpct', 'return_lag_20_z',
        'amplitude_rankpct', 'amplitude_z', 'amplitude_lag_1_rankpct', 'amplitude_lag_1_z', 'amplitude_lag_3_rankpct',
        'amplitude_lag_3_z', 'amplitude_lag_5_rankpct', 'amplitude_lag_5_z', 'vol_gk_lag_1_rankpct', 'vol_gk_lag_1_z',
        'vol_gk_ratio_lag_1_rankpct', 'vol_gk_ratio_lag_1_z', 'vol_gk_lag_3_rankpct', 'vol_gk_lag_3_z',
        'vol_gk_ratio_lag_3_rankpct', 'vol_gk_ratio_lag_3_z', 'vol_gk_lag_5_rankpct', 'vol_gk_lag_5_z',
        'vol_gk_ratio_lag_5_rankpct', 'vol_gk_ratio_lag_5_z', 'vol_gk_lag_10_rankpct', 'vol_gk_lag_10_z',
        'vol_gk_ratio_lag_10_rankpct', 'vol_gk_ratio_lag_10_z', 'illiq_lag_1_rankpct', 'illiq_lag_1_z',
        'illiq_lag_3_rankpct', 'illiq_lag_3_z', 'illiq_lag_5_rankpct', 'illiq_lag_5_z',
        'efficiency_ratio_lag_1_rankpct', 'efficiency_ratio_lag_1_z', 'efficiency_ratio_lag_3_rankpct',
        'efficiency_ratio_lag_3_z', 'efficiency_ratio_lag_5_rankpct', 'efficiency_ratio_lag_5_z',
        'intraday_pos_lag_1_rankpct', 'intraday_pos_lag_1_z', 'intraday_pos_lag_2_rankpct', 'intraday_pos_lag_2_z',
        'intraday_pos_lag_3_rankpct', 'intraday_pos_lag_3_z', 'smart_money_diff_lag_1_rankpct',
        'smart_money_diff_lag_1_z', 'ret_overnight_lag_1_rankpct', 'ret_overnight_lag_1_z',
        'ret_intraday_lag_1_rankpct', 'ret_intraday_lag_1_z', 'smart_money_diff_lag_3_rankpct',
        'smart_money_diff_lag_3_z', 'ret_overnight_lag_3_rankpct', 'ret_overnight_lag_3_z',
        'ret_intraday_lag_3_rankpct', 'ret_intraday_lag_3_z', 'smart_money_diff_lag_5_rankpct',
        'smart_money_diff_lag_5_z', 'ret_overnight_lag_5_rankpct', 'ret_overnight_lag_5_z',
        'ret_intraday_lag_5_rankpct', 'ret_intraday_lag_5_z', 'support_resistance_ratio_lag_1_rankpct',
        'support_resistance_ratio_lag_1_z', 'support_resistance_ratio_lag_3_rankpct',
        'support_resistance_ratio_lag_3_z', 'support_resistance_ratio_lag_5_rankpct',
        'support_resistance_ratio_lag_5_z', 'sh_price_change_rankpct', 'sh_price_change_z', 'sh_amplitude_rankpct',
        'sh_amplitude_z', 'sh_volume_ratio_rankpct', 'sh_volume_ratio_z', 'sh_price_change_abs_rankpct',
        'sh_price_change_abs_z', 'sh_price_wave_abs_rankpct', 'sh_price_wave_abs_z', 'sh_sentiment_rankpct',
        'sh_sentiment_z', 'sh_volume_signal_rankpct', 'sh_volume_signal_z', 'sz_price_change_rankpct',
        'sz_price_change_z', 'sz_amplitude_rankpct', 'sz_amplitude_z', 'sz_volume_ratio_rankpct', 'sz_volume_ratio_z',
        'sz_price_change_abs_rankpct', 'sz_price_change_abs_z', 'sz_price_wave_abs_rankpct', 'sz_price_wave_abs_z',
        'sz_sentiment_rankpct', 'sz_sentiment_z', 'sz_volume_signal_rankpct', 'sz_volume_signal_z',
        'sh_sz_sync_direction_rankpct', 'sh_sz_sync_direction_z', 'sh_sz_sync_strength_rankpct',
        'sh_sz_sync_strength_z', 'market_sentiment_rankpct', 'market_sentiment_z', 'market_avg_change_rankpct',
        'market_avg_change_z', 'market_avg_amplitude_rankpct', 'market_avg_amplitude_z', 'market_sync_score_rankpct',
        'market_sync_score_z', 'is_quick_divergence_x', 'compare_rank', 'close_current', 'close_previous',
        'macd_current', 'macd_previous', 'price_decline_pct', 'macd_increase_pct', 'formation_period',
        'is_quick_divergence_y', 'divergence_strength', 'volume_signal', 'price_macd_ratio', 'divergence_magnitude',
        'confirmation_score', 'divergence_amount',
    ]

    # 最优
    OPTIMIZED_FEATURE_COLS = [
        'bb_upper_rankpct', 'stoch_oversold_rankpct', 'illiq_z', 'price_decline_pct', 'stoch_overbought_rankpct',
        'market_sync_score', 'sh_volume_signal', 'support_resistance_ratio_lag_5_z', 'doji_pattern_rankpct',
        'amplitude_lag_1', 'close_current', 'atr_ratio_z', 'rsi_overbought_24_rankpct', 'rsi_overbought_6_rankpct',
        'vol_divergence_z', 'sh_volume_ratio', 'vol_gk_ratio_rankpct', 'bb_squeeze_rankpct', 'divergence_magnitude',
        'volume_dryup_rankpct', 'volume_lag_5', 'high_lag_15_z', 'amplitude_z', 'volume_lag_5_z', 'high_lag_10',
        'market_avg_change', 'smart_money_diff_lag_1', 'ma_arrangement_rankpct', 'rsi_oversold_14_rankpct',
        'engulfing_pattern_rankpct', 'clv', 'divergence_amount', 'pct_change', 'illiq_lag_5_z', 'ret_intraday_rankpct',
        'vol_gk_ratio', 'sh_amplitude', 'volume_lag_20_rankpct', 'vol_gk_lag_1', 'pv_corr_10_z', 'ret_overnight',
        'return_lag_5_z', 'vol_gk_rankpct', 'rsi_oversold_6_rankpct', 'ret_intraday_z', 'efficiency_ratio_rankpct',
        'rank_return_rankpct', 'ret_intraday_lag_1_rankpct', 'macd_zero_cross_rankpct', 'return_lag_3',
        'vol_gk_ratio_z', 'clv_rankpct', 'ret_overnight_lag_1_z', 'close_lag_15_rankpct', 'low_lag_5', 'cs_n',
        'support_resistance_ratio_lag_5_rankpct', 'illiq_lag_1', 'support_resistance_ratio', 'dist_to_high_60_rankpct',
        'close_lag_25', 'sz_volume_ratio', 'close_lag_15_z', 'illiq_lag_3', 'amplitude_rankpct',
        'dist_to_high_60', 'ma_20', 'volume_spike_rankpct', 'high_lag_30', 'ma_60', 'ma_20_z',
        'macd_percentile_rankpct', 'macd_golden_cross_rankpct', 'sz_price_change', 'sz_amplitude',
        'ret_intraday_lag_5_rankpct', 'ma_60_rankpct', 'rsi_turning_simple_rankpct', 'vol_divergence',
        'ret_intraday_lag_1', 'open_lag_20_z', 'close_lag_10', 'macd_death_cross_rankpct', 'illiq_lag_3_z',
        'sz_sentiment', 'rank_return_z', 'price_volume_divergence_rankpct', 'distance_to_resistance',
        'macd_zero_cross_up_rankpct', 'pct_change_z', 'close_vs_high', 'intraday_pos_lag_2_z', 'sz_price_wave_abs',
        'illiq_lag_5', 'return_lag_20', 'open_lag_5', 'smart_money_diff', 'smart_money_diff_z',
        'rsi_overbought_14_rankpct', 'market_avg_amplitude', 'sh_sz_sync_direction', 'illiq',
        'support_resistance_ratio_lag_5', 'sz_price_change_abs', 'price_volume_divergence_z', 'obv',
        'ma_5', 'low_lag_30', 'sh_price_change', 'support_resistance_ratio_lag_1_z', 'smart_money_diff_rankpct',
        'sh_sz_sync_strength', 'distance_to_support_rankpct', 'support_resistance_ratio_lag_3_z',
        'support_resistance_ratio_lag_1', 'macd_hist_direction_rankpct', 'ret_overnight_lag_1_rankpct', 'open_lag_25',
        'return_lag_1_rankpct', 'low_lag_3_rankpct', 'volume_dryup_z', 'return_lag_3_z', 'volume_lag_3_z',
        'open_lag_5_z', 'close_previous', 'close_lag_30_rankpct', 'open_lag_15', 'ret_overnight_z',
        'sh_price_change_abs', 'ma_60_z', 'volume_ratio_z', 'body_strength_rankpct',
        'volume_lag_10_rankpct', 'bb_upper', 'return_lag_1', 'sh_price_wave_abs', 'ret_overnight_rankpct',
        'distance_to_support', 'return_lag_5', 'efficiency_ratio_lag_5_rankpct', 'high_lag_5_z', 'vol_gk',
        'macd_zero_cross_down_rankpct', 'vol_gk_z', 'open_lag_3_rankpct', 'close_lag_15', 'pv_corr_10',
        'dist_to_high_60_z', 'macd_signal_cross_rankpct', 'obv_rankpct', 'amplitude', 'body_strength_z',
        'rsi_oversold_24_rankpct', 'volume_lag_5_rankpct',
    ]


    STABLE_FEATURES = [
        'boxcox_atr', 'boxcox_atr_z', 'distance_to_support_z', 'distance_to_support', 'volatility_30d',
        'close_vs_high_z', 'sh_amplitude', 'obv', 'volatility_30d_z', 'volatility_25d', 'close_vs_high',
        'volatility_20d', 'return_lag_20', 'sh_sz_sync_strength', 'price_volume_divergence_rankpct', 'volatility_25d_z',
        'sz_amplitude', 'volatility_20d_z', 'dist_to_high_60', 'pv_corr_10', 'pv_corr_10_z', 'market_sync_score',
        'volume_dryup', 'volatility_15d', 'dist_to_high_60_z', 'price_volume_divergence', 'support_resistance_ratio',
        'clv_z', 'intraday_pos_z', 'volatility_15d_z', 'ret_intraday_z', 'volume_consistency_z', 'volume_dryup_z',
        'ret_intraday', 'clv', 'intraday_pos', 'volume_consistency', 'vol_gk_lag_3', 'upper_shadow_ratio_z', 'cs_n',
        'price_vs_ma5', 'support_resistance_ratio_z', 'macd_hist_amplitude', 'price_vs_ma5_z', 'volatility_10d',
        'price_volume_divergence_z', 'efficiency_ratio_lag_1', 'atr_ratio_rankpct', 'volatility_10d_z',
        'vol_gk_lag_3_z', 'ma_20', 'sh_volume_ratio', 'efficiency_ratio_lag_1_z', 'return_lag_1', 'pv_corr_10_rankpct',
        'ret_overnight_lag_1', 'smart_money_diff', 'vol_gk_ratio_lag_3', 'body_strength_z', 'intraday_pos_lag_3',
        'low_lag_30_rankpct', 'sz_price_change', 'vol_gk_z', 'bb_squeeze_rankpct', 'ret_intraday_lag_1_z',
        'efficiency_ratio_lag_5_z', 'upper_shadow_ratio', 'vol_gk_ratio_lag_3_z', 'return_lag_20_z',
        'efficiency_ratio_lag_3', 'sh_volume_signal_rankpct', 'sh_volume_ratio_rankpct', 'hammer_pattern_rankpct',
        'market_sentiment_rankpct', 'rsi_oversold_24_rankpct', 'sh_sz_sync_direction_rankpct',
        'market_sync_score_rankpct', 'sz_volume_signal_rankpct', 'market_avg_amplitude_rankpct',
        'market_avg_change_rankpct', 'sh_sentiment_rankpct', 'sh_price_change_rankpct', 'sz_amplitude_rankpct',
        'doji_pattern_rankpct', 'sz_price_change_rankpct', 'sz_sentiment_rankpct', 'sh_sz_sync_strength_rankpct',
        'rsi_oversold_14_rankpct', 'sh_amplitude_rankpct', 'sz_volume_ratio_rankpct',
        'efficiency_ratio_lag_5', 'vol_divergence', 'market_sentiment', 'vol_divergence_z',
        'support_resistance_ratio_lag_5_rankpct', 'sh_sentiment', 'efficiency_ratio_z', 'efficiency_ratio_lag_3_z',
        'intraday_pos_lag_3_z', 'upper_shadow_ratio_rankpct', 'body_strength_rankpct', 'log_volume',
        'intraday_pos_lag_2',
    ]


    # 15d
    STRATEGY_PARAMS_CANDIDATES_V8 = {
     "param1": {  # 回报率:2.8701 | 最大回撤:-0.2839 | 胜率:0.5692 | 夏普:1.4527 | 交易数:130
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param10": {  # 回报率:2.5644 | 最大回撤:-0.2761 | 胜率:0.5692 | 夏普:1.4346 | 交易数:130
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param11": {  # 回报率:2.5573 | 最大回撤:-0.2748 | 胜率:0.6031 | 夏普:1.4039 | 交易数:131
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param12": {  # 回报率:2.5366 | 最大回撤:-0.2865 | 胜率:0.5270 | 夏普:1.4886 | 交易数:241
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.1,
         "max_hold_days": 15,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param13": {  # 回报率:2.5178 | 最大回撤:-0.2749 | 胜率:0.5643 | 夏普:1.4666 | 交易数:140
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param14": {  # 回报率:2.4932 | 最大回撤:-0.2882 | 胜率:0.5259 | 夏普:1.4091 | 交易数:232
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.1,
         "max_hold_days": 16,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param15": {  # 回报率:2.4931 | 最大回撤:-0.2737 | 胜率:0.5814 | 夏普:1.3805 | 交易数:129
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param16": {  # 回报率:2.4173 | 最大回撤:-0.2748 | 胜率:0.5672 | 夏普:1.4027 | 交易数:134
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 17,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param17": {  # 回报率:2.4030 | 最大回撤:-0.2997 | 胜率:0.5190 | 夏普:1.3755 | 交易数:237
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.1,
         "max_hold_days": 16,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param18": {  # 回报率:2.4019 | 最大回撤:-0.2882 | 胜率:0.5190 | 夏普:1.4084 | 交易数:237
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.1,
         "max_hold_days": 16,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param19": {  # 回报率:2.3448 | 最大回撤:-0.2826 | 胜率:0.5474 | 夏普:1.3339 | 交易数:137
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param2": {  # 回报率:2.8351 | 最大回撤:-0.2833 | 胜率:0.6031 | 夏普:1.4136 | 交易数:131
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param20": {  # 回报率:2.3203 | 最大回撤:-0.2835 | 胜率:0.5481 | 夏普:1.2953 | 交易数:135
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param21": {  # 回报率:2.3117 | 最大回撤:-0.2815 | 胜率:0.5323 | 夏普:1.2328 | 交易数:124
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param22": {  # 回报率:2.3051 | 最大回撤:-0.2896 | 胜率:0.5446 | 夏普:1.1577 | 交易数:101
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 30,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param23": {  # 回报率:2.2731 | 最大回撤:-0.2746 | 胜率:0.5726 | 夏普:1.2734 | 交易数:124
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param24": {  # 回报率:2.2385 | 最大回撤:-0.2826 | 胜率:0.5242 | 夏普:1.2078 | 交易数:124
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param25": {  # 回报率:2.2132 | 最大回撤:-0.2973 | 胜率:0.5359 | 夏普:1.3375 | 交易数:237
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.1,
         "max_hold_days": 15,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param26": {  # 回报率:2.2023 | 最大回撤:-0.2923 | 胜率:0.5227 | 夏普:1.2111 | 交易数:132
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param27": {  # 回报率:2.2014 | 最大回撤:-0.2762 | 胜率:0.5680 | 夏普:1.2579 | 交易数:125
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param28": {  # 回报率:2.1805 | 最大回撤:-0.2954 | 胜率:0.5455 | 夏普:1.1442 | 交易数:110
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 25,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param29": {  # 回报率:2.1780 | 最大回撤:-0.2855 | 胜率:0.5481 | 夏普:1.2605 | 交易数:135
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param3": {  # 回报率:2.7953 | 最大回撤:-0.2827 | 胜率:0.5814 | 夏普:1.3987 | 交易数:129
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param30": {  # 回报率:2.1582 | 最大回撤:-0.2754 | 胜率:0.5474 | 夏普:1.3340 | 交易数:137
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param31": {  # 回报率:2.1404 | 最大回撤:-0.2916 | 胜率:0.5227 | 夏普:1.1991 | 交易数:132
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param32": {  # 回报率:2.1403 | 最大回撤:-0.2826 | 胜率:0.6036 | 夏普:1.1942 | 交易数:111
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 30,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param33": {  # 回报率:2.1370 | 最大回撤:-0.2871 | 胜率:0.5359 | 夏普:1.3466 | 交易数:237
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.1,
         "max_hold_days": 15,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param34": {  # 回报率:2.1192 | 最大回撤:-0.2821 | 胜率:0.5403 | 夏普:1.2083 | 交易数:124
         "base_ratio": 1.0,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param35": {  # 回报率:2.1148 | 最大回撤:-0.2924 | 胜率:0.5414 | 夏普:1.2266 | 交易数:133
         "base_ratio": 1.0,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param36": {  # 回报率:2.1143 | 最大回撤:-0.2811 | 胜率:0.5446 | 夏普:1.1556 | 交易数:101
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 30,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param37": {  # 回报率:2.1049 | 最大回撤:-0.2880 | 胜率:0.5397 | 夏普:1.2711 | 交易数:126
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 17,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param38": {  # 回报率:2.0704 | 最大回撤:-0.2827 | 胜率:0.5714 | 夏普:1.1811 | 交易数:126
         "base_ratio": 1.0,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param4": {  # 回报率:2.7700 | 最大回撤:-0.2831 | 胜率:0.5643 | 夏普:1.4698 | 交易数:140
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param40": {  # 回报率:2.0475 | 最大回撤:-0.2835 | 胜率:0.5285 | 夏普:1.1405 | 交易数:123
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param41": {  # 回报率:2.0296 | 最大回撤:-0.2757 | 胜率:0.5481 | 夏普:1.2676 | 交易数:135
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param42": {  # 回报率:2.0249 | 最大回撤:-0.2841 | 胜率:0.5481 | 夏普:1.2292 | 交易数:135
         "base_ratio": 1.0,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param46": {  # 回报率:1.9868 | 最大回撤:-0.2822 | 胜率:0.5203 | 夏普:1.1206 | 交易数:123
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param47": {  # 回报率:1.9785 | 最大回撤:-0.2808 | 胜率:0.5227 | 夏普:1.1999 | 交易数:132
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param5": {  # 回报率:2.6911 | 最大回撤:-0.2825 | 胜率:0.5726 | 夏普:1.3235 | 交易数:124
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param50": {  # 回报率:1.9477 | 最大回撤:-0.2758 | 胜率:0.5323 | 夏普:1.1806 | 交易数:124
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param52": {  # 回报率:1.9268 | 最大回撤:-0.2746 | 胜率:0.6036 | 夏普:1.1805 | 交易数:111
         "base_ratio": 0.86,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 30,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param53": {  # 回报率:1.9258 | 最大回撤:-0.2881 | 胜率:0.5360 | 夏普:1.2041 | 交易数:125
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 17,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param54": {  # 回报率:1.9168 | 最大回撤:-0.2748 | 胜率:0.5481 | 夏普:1.2364 | 交易数:135
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param55": {  # 回报率:1.9155 | 最大回撤:-0.2819 | 胜率:0.5227 | 夏普:1.1853 | 交易数:132
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param58": {  # 回报率:1.8946 | 最大回撤:-0.2753 | 胜率:0.5242 | 夏普:1.1579 | 交易数:124
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 19,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param59": {  # 回报率:1.8648 | 最大回撤:-0.2819 | 胜率:0.5414 | 夏普:1.2046 | 交易数:133
         "base_ratio": 0.86,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 16,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param6": {  # 回报率:2.6724 | 最大回撤:-0.2828 | 胜率:0.5672 | 夏普:1.4094 | 交易数:134
         "base_ratio": 1.0,
         "target_profit": 0.2,
         "hard_stop_loss": -0.14,
         "max_hold_days": 17,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param69": {  # 回报率:1.7475 | 最大回撤:-0.2770 | 胜率:0.5481 | 夏普:1.1910 | 交易数:135
         "base_ratio": 0.86,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 15,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param7": {  # 回报率:2.6155 | 最大回撤:-0.2831 | 胜率:0.5680 | 夏普:1.3101 | 交易数:125
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param71": {  # 回报率:1.7374 | 最大回撤:-0.2748 | 胜率:0.5714 | 夏普:1.1254 | 交易数:126
         "base_ratio": 0.86,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 18,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param72": {  # 回报率:1.7286 | 最大回撤:-0.2748 | 胜率:0.5285 | 夏普:1.0900 | 交易数:123
         "base_ratio": 0.86,
         "target_profit": 0.3,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param74": {  # 回报率:1.6879 | 最大回撤:-0.2750 | 胜率:0.5203 | 夏普:1.0730 | 交易数:123
         "base_ratio": 0.86,
         "target_profit": 0.35,
         "hard_stop_loss": -0.14,
         "max_hold_days": 20,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param77": {  # 回报率:1.6340 | 最大回撤:-0.2882 | 胜率:0.5312 | 夏普:1.1227 | 交易数:128
         "base_ratio": 0.86,
         "target_profit": 0.25,
         "hard_stop_loss": -0.14,
         "max_hold_days": 17,
         "max_positions": 3,
         "min_probability": 0.5,
     },
     "param8": {  # 回报率:2.5863 | 最大回撤:-0.2974 | 胜率:0.5270 | 夏普:1.4652 | 交易数:241
         "base_ratio": 1.0,
         "target_profit": 0.3,
         "hard_stop_loss": -0.1,
         "max_hold_days": 15,
         "max_positions": 5,
         "min_probability": 0.5,
     },
     "param9": {  # 回报率:2.5672 | 最大回撤:-0.2999 | 胜率:0.5259 | 夏普:1.3956 | 交易数:232
         "base_ratio": 1.0,
         "target_profit": 0.35,
         "hard_stop_loss": -0.1,
         "max_hold_days": 16,
         "max_positions": 5,
         "min_probability": 0.5,
     },
    }
    STRATEGY_PARAMS_V8 = STRATEGY_PARAMS_CANDIDATES_V8["param35"]

    STRATEGY_PARAMS_CANDIDATES_V12 = {
        "param1": {  # v8_param29 | 回报率:4.40137 | 最大回撤:-0.283 | 胜率:0.5873 | 夏普:2.09 | 交易数:126
            "base_ratio": 1.0,
            "target_profit": 0.3,
            "hard_stop_loss": -0.14,
            "max_hold_days": 15,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.4,
            "vol_profit_mult": 1.5,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param2": {  # v8_param35 | 回报率:4.0518 | 最大回撤:-0.2918 | 胜率:0.5827 | 夏普:1.9795 | 交易数:127
            "base_ratio": 1.0,
            "target_profit": 0.25,
            "hard_stop_loss": -0.14,
            "max_hold_days": 16,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.6,
            "vol_profit_mult": 1.5,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param3": {  # v8_param20 | 回报率:4.3396 | 最大回撤:-0.2782 | 胜率:0.5873 | 夏普:2.06931 | 交易数:126
            "base_ratio": 1.0,
            "target_profit": 0.35,
            "hard_stop_loss": -0.14,
            "max_hold_days": 15,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 7.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.8,
            "vol_profit_mult": 1.2,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 0.8,
            "use_volatility_adaptive": True,
        },
        "param4": {  # v8_param21 | 回报率:3.9889 | 最大回撤:-0.2814 | 胜率:0.61017 | 夏普:1.874 | 交易数:118
            "base_ratio": 1.0,
            "target_profit": 0.3,
            "hard_stop_loss": -0.14,
            "max_hold_days": 19,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.8,
            "vol_profit_mult": 1.5,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param5": {  # v8_param42 | 回报率:3.744 | 最大回撤:-0.2844 | 胜率:0.5781 | 夏普:1.9239 | 交易数:128
            "base_ratio": 1.0,
            "target_profit": 0.25,
            "hard_stop_loss": -0.14,
            "max_hold_days": 15,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.6,
            "vol_profit_mult": 1.5,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param6": {  # v8_param31 | 回报率:4.016 | 最大回撤:-0.291 | 胜率:0.5806 | 夏普:1.9357 | 交易数:124
            "base_ratio": 1.0,
            "target_profit": 0.3,
            "hard_stop_loss": -0.14,
            "max_hold_days": 16,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.8,
            "vol_profit_mult": 1.5,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param7": {  # v8_param24 | 回报率:3.9991 | 最大回撤:-0.2833 | 胜率:0.61538 | 夏普:1.9013 | 交易数:117
            "base_ratio": 1.0,
            "target_profit": 0.35,
            "hard_stop_loss": -0.14,
            "max_hold_days": 19,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 2.5,
            "vol_low_thresh": 0.4,
            "vol_profit_mult": 2.0,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 0.8,
            "use_volatility_adaptive": True,
        },
        "param8": {  # v8_param26 | 回报率:3.792 | 最大回撤:-0.2920 | 胜率:0.5806 | 夏普:1.869 | 交易数:124
            "base_ratio": 1.0,
            "target_profit": 0.35,
            "hard_stop_loss": -0.14,
            "max_hold_days": 16,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 2.5,
            "vol_low_thresh": 0.8,
            "vol_profit_mult": 1.2,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 0.8,
            "use_volatility_adaptive": True,
        },
        "param9": {  # v8_param1 | 回报率:3.0808 | 最大回撤:-0.2428 | 胜率:0.5950 | 夏普:1.61 | 交易数:121
            "base_ratio": 1.0,
            "target_profit": 0.2,
            "hard_stop_loss": -0.14,
            "max_hold_days": 19,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 21.0,
            "vol_high_thresh": 1.8,
            "vol_low_thresh": 0.6,
            "vol_profit_mult": 2.0,
            "vol_stop_mult": 1.5,
            "low_vol_profit_mult": 1.0,
            "use_volatility_adaptive": True,
        },
        "param10": {  # v8_param5 | 回报率:3.321 | 最大回撤:-0.2825 | 胜率:0.61334 | 夏普:1.679 | 交易数:119
            "base_ratio": 1.0,
            "target_profit": 0.35,
            "hard_stop_loss": -0.14,
            "max_hold_days": 18,
            "max_positions": 3,
            "min_probability": 0.5,
            "vol_lookback": 10.0,
            "vol_high_thresh": 2.0,
            "vol_low_thresh": 0.4,
            "vol_profit_mult": 1.2,
            "vol_stop_mult": 1.1,
            "low_vol_profit_mult": 0.8,
            "use_volatility_adaptive": True,
        },
    }
    STRATEGY_PARAMS_V12 = STRATEGY_PARAMS_CANDIDATES_V12["param2"]


    STRATEGY_PARAMS_CANDIDATES_V6 = {
         "参数1":   {  # 测试&验证集表现： 回报率:2.047596549254404 倍  |  最大回撤： -0.3770660175339056 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.805274568297548
                'max_positions': 5,
                'base_ratio': 0.86,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.08,
            },
         "参数2":   {  # 测试&验证集表现： 回报率:2.047596549254404 倍  |  最大回撤： -0.3770660175339056 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.805274568297548
                'max_positions': 5,
                'base_ratio': 0.86,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.16,
            },
         "参数3":   {  # 测试&验证集表现： 回报率:2.047596549254404 倍  |  最大回撤： -0.3770660175339056 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.805274568297548
                'max_positions': 5,
                'base_ratio': 0.86,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.12,
            },
         "参数4":   {  # 测试&验证集表现： 回报率:2.047596549254404 倍  |  最大回撤： -0.3770660175339056 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.805274568297548
                'max_positions': 5,
                'base_ratio': 0.86,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.18,
            },
         "参数5":   {  # 测试&验证集表现： 回报率:2.047596549254404 倍  |  最大回撤： -0.3770660175339056 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.805274568297548
                'max_positions': 5,
                'base_ratio': 0.86,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.2,
            },
         "参数6":   {  # 测试&验证集表现： 回报率:2.0539561253148566 倍  |  最大回撤： -0.3838741868181223 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.7685638993280577
                'max_positions': 5,
                'base_ratio': 1.0,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.2,
            },
         "参数7":   {  # 测试&验证集表现： 回报率:2.0539561253148566 倍  |  最大回撤： -0.3838741868181223 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.7685638993280577
                'max_positions': 5,
                'base_ratio': 1.0,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.12,
            },
         "参数8":   {  # 测试&验证集表现： 回报率:2.0539561253148566 倍  |  最大回撤： -0.3838741868181223 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.7685638993280577
                'max_positions': 5,
                'base_ratio': 1.0,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.08,
            },
         "参数9":   {  # 测试&验证集表现： 回报率:2.0539561253148566 倍  |  最大回撤： -0.3838741868181223 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.7685638993280577
                'max_positions': 5,
                'base_ratio': 1.0,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.18,
            },
         "参数10":   {  # 测试&验证集表现： 回报率:2.0539561253148566 倍  |  最大回撤： -0.3838741868181223 | 胜率：0.5326633165829145 | 总交易数：199.0 | sharpe_ratio：1.7685638993280577
                'max_positions': 5,
                'base_ratio': 1.0,
                'target_profit': 0.35,
                'max_hold_days': 20,
                'hard_stop_loss': -0.14,
                'top_k': 0.16,
            },
    }
    TOP_K_STRATEGY_PARAMS_V6 = STRATEGY_PARAMS_CANDIDATES_V6['参数1']

    STRATEGY_PARAMS_CANDIDATES_V7 = {
        "参数1": {     # 测试&验证集表现： 回报率:1.423350472 倍  |  最大回撤： -0.430780653 | 胜率：0.556886228 | 总交易数：167 | sharpe_ratio：1.433298662
            'base_ratio': 1,
            'target_profit': 0.2,
            'hard_stop_loss': -0.12,
            'max_hold_days': 30,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.2,
            'min_probability': 0.4
        },
        "参数2": {     # 测试&验证集表现： 回报率:1.358347133 倍  |  最大回撤： -0.322612726 | 胜率：0.442982456 | 总交易数：228 | sharpe_ratio：1.728955751
            'base_ratio': 1,
            'target_profit': 0.2,
            'hard_stop_loss': -0.03,
            'max_hold_days': 100,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
        "参数3": {     # 测试&验证集表现： 回报率:1.250257159 倍  |  最大回撤： -0.322612726 | 胜率：0.441048035 | 总交易数：229 | sharpe_ratio：1.657032405
            'base_ratio': 1,
            'target_profit': 0.2,
            'hard_stop_loss': -0.03,
            'max_hold_days': 30,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
        "参数4": {     # 测试&验证集表现： 回报率:1.193258503 倍  |  最大回撤： -0.41853331 | 胜率：0.538461538 | 总交易数：156 | sharpe_ratio：1.257038397
            'base_ratio': 1,
            'target_profit': 0.3,
            'hard_stop_loss': -0.12,
            'max_hold_days': 30,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.2,
            'min_probability': 0.4
        },
        "参数5": {     # 测试&验证集表现： 回报率:1.178915603 倍  |  最大回撤： -0.339598002 | 胜率：0.449074074 | 总交易数：216 | sharpe_ratio：1.493038447
            'base_ratio': 1,
            'target_profit': 0.6,
            'hard_stop_loss': -0.03,
            'max_hold_days': 100,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
        "参数6": {     # 测试&验证集表现： 回报率:1.178915603 倍  |  最大回撤： -0.339598002 | 胜率：0.449074074 | 总交易数：216 | sharpe_ratio：1.493038447
            'base_ratio': 1,
            'target_profit': 0.8,
            'hard_stop_loss': -0.03,
            'max_hold_days': 100,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
        "参数7": {     # 测试&验证集表现： 回报率:1.178915603 倍  |  最大回撤： -0.339598002 | 胜率：0.449074074 | 总交易数：216 | sharpe_ratio：1.493038447
            'base_ratio': 1,
            'target_profit': 10,
            'hard_stop_loss': -0.03,
            'max_hold_days': 100,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
        "参数8": {     # 测试&验证集表现： 回报率:1.173580274 倍  |  最大回撤： -0.339598002 | 胜率：0.444954128 | 总交易数：218 | sharpe_ratio：1.501673344
            'base_ratio': 1,
            'target_profit': 0.6,
            'hard_stop_loss': -0.03,
            'max_hold_days': 30,
            'max_positions': 3,
            'top_k_buy': 0.08,
            'top_k_hold': 0.3,
            'min_probability': 0.5
        },
    }
    TOP_K_STRATEGY_PARAMS_V7 = STRATEGY_PARAMS_CANDIDATES_V7['参数8']

EPS = 1e-9 # 极小值
model_config = Config()

# 模式选择: 'z_score' (推荐), 'tiered' (分档)
ALLOCATION_STRATEGY = 'z_score'

# 统计参数 (来自您的报告数据)
PROBA_MEAN = 0.6424  # 均值
PROBA_STD  = 0.0848  # 标准差
PROBA_MIN  = 0.5502  # 最小值
PROBA_MAX  = 0.9392  # 最大值


