import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import tushare as ts
from sklearn.preprocessing import LabelEncoder
from config.settings import STOCK_DATA_DIR, DATASET_DIR
import numpy as np
def load_price_data_for_symbol(symbol, start_date=None, end_date=None, data_folder_path=str(STOCK_DATA_DIR)):
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

def get_return_threshold(data, quantile=0.58):
    """动态计算阈值"""
    return np.quantile(data[model_config.LABEL_COL].dropna(), quantile)
    # return 0.01

class Config:
    DATA_PATH = str(DATASET_DIR)
    TEST_RATIO = 0.15
    TIME_COLS = ["timestamp"]
    N_SPLITS = 5
    RANDOM_STATE = 42
    SETTLEMENT_DAYS = 10
    LABEL_COL = f"future_return_{SETTLEMENT_DAYS}d"
    EXPECTED_PROFIT = 1.15 # 特征中的止盈
    EXPECTED_LOSS = 0.65 # 特征中的止损
    RETURN_THRESHOLD = 0.01  # 分类阈值
    PROBA_THRESHOLD = 0.7 #阈值就是判断"是"与"否"的分界线。
    FP_PENALTY_WEIGHT = 1.0 # 暂时没用到
    USE_PCA = True
    KEEP_LGBM_LEAF_AMOUNT = 15
    PCA_VARIANCE_THRESHOLD = 0.95 # PCA降维特征
    AFFORDABLE_PRICE = 100 # 最大买卖股价
    RETURN_PERIODS = [3, 5, 10, 15, 20, 25, 30]
    WINDOWS_VOLATILITY = [3, 5, 10, 15, 20, 25, 30]  # 波动率计算窗口（与RETURN_PERIODS匹配）
    LAG_PERIODS = [3, 5, 10, 15, 20, 25, 30]
    FEATURE_NEED_MAX_DAYS =  100 # 用于特征计算的最大天数

    ## 参数组1：
    # LGB_PARAMS = {
    #     'n_estimators': 1000,
    #     'learning_rate': 0.02,
    #     'metric': ['auc', 'binary_logloss'],
    #     'num_leaves': 63,  # num_leaves 从 31 改成 63（让模型能学更细的规律）
    #     'objective': 'binary',
    #     'feature_fraction': 0.85,
    #     'bagging_fraction': 0.875,
    #     'n_jobs': 4,
    #     'random_state': RANDOM_STATE,
    #     'max_depth': 3,  # 避免学杂了
    #     'bagging_freq': 7,  # 启用 bagging
    #     'min_child_samples': 58,
    #     # 'lambda_l1': 3.0,  # L1 正则
    #     # 'lambda_l2': 1.0,  # L2 正则
    #     'min_split_gain': 0.4,
    #     'scale_pos_weight': 1.0,
    #     'verbosity': -1,  # 屏蔽 LightGBM 警告输出
    #     "subsample": 0.6281355753373223,
    #     "colsample_bytree": 0.6603882248092653,
    #     "reg_alpha": 0.009814834188041289,
    #     "reg_lambda": 1.3746978723023734e-08,
    #     "subsample_freq": 10,
    #     "min_child_weight": 0.03051557504873871
    # }

    ## 参数组2：
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
        'volume', 'amount', 'macd', 'macd_signal', 'macd_hist', 'rsi_6', 'rsi_14', 'rsi_24', 'ma_5', 'ma_20', 'ma_60',
        'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma_20', 'obv', 'atr', 'slowk', 'slowd', 'price_vs_ma5',
        'price_vs_ma20', 'price_vs_ma60', 'ma_arrangement', 'bb_position', 'bb_squeeze', 'rsi_oversold_6',
        'rsi_oversold_14', 'rsi_oversold_24', 'rsi_overbought_6', 'rsi_overbought_14', 'rsi_overbought_24',
        'macd_signal_distance', 'macd_golden_cross', 'volume_ma20', 'volume_ratio', 'volume_spike', 'volume_dryup',
        'atr_ratio', 'hammer_pattern', 'doji_pattern', 'distance_to_support', 'distance_to_resistance',
        'stoch_oversold', 'stoch_overbought', 'macd_percentile', 'obv_trend', 'volatility_3d', 'volatility_5d',
        'volatility_10d', 'volatility_15d', 'volatility_20d', 'volatility_25d', 'volatility_30d', 'engulfing_pattern',
        'signed_volume_strength', 'close_vs_high', 'volume_ma_ratio', 'rsi_momentum', 'rsi_turning_simple',
        'rsi_turning', 'volume_trend_5', 'volume_trend_10', 'price_trend_5', 'price_volume_divergence',
        'volume_consistency', 'macd_hist_trend_5', 'macd_death_cross', 'macd_signal_cross', 'macd_zero_cross_up',
        'macd_zero_cross_down', 'macd_zero_cross', 'macd_hist_amplitude', 'macd_hist_direction',
        'macd_hist_acceleration', 'macd_signal_convergence', 'macd_signal_convergence_trend', 'pct_change', 'clv',
        'upper_shadow_ratio', 'body_strength', 'rank_return', 'rank_volume', 'signed_vol_strength', 'pv_corr_10',
        'dist_to_high_60', 'vol_divergence', 'illiq', 'efficiency_ratio', 'intraday_pos', 'ret_overnight',
        'ret_intraday', 'smart_money_diff', 'high_mean_20', 'low_mean_20', 'support_resistance_ratio', 'log_volume',
        'boxcox_atr', 'rsi_robust', 'macd_robust',  
        'high_lag_3', 'low_lag_3', 'close_lag_5', 'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_10',
        'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_15', 'open_lag_15', 'high_lag_15', 'low_lag_15',
        'close_lag_20', 'open_lag_20', 'high_lag_20', 'low_lag_20', 'close_lag_25', 'open_lag_25', 'high_lag_25',
        'low_lag_25', 'close_lag_30', 'open_lag_30', 'high_lag_30', 'low_lag_30', 'volume_lag_3', 'volume_lag_5',
        'volume_lag_10', 'volume_lag_15', 'volume_lag_20', 'volume_lag_25', 'volume_lag_30', 'daily_return',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20', 'amplitude',
           'illiq_lag_1', 'illiq_lag_3', 'illiq_lag_5',
        'efficiency_ratio_lag_1', 'efficiency_ratio_lag_3', 'efficiency_ratio_lag_5', 'intraday_pos_lag_1',
        'intraday_pos_lag_2', 'intraday_pos_lag_3', 'smart_money_diff_lag_1', 'ret_overnight_lag_1',
        'smart_money_diff_lag_3', 'ret_overnight_lag_3', 'ret_intraday_lag_3', 'smart_money_diff_lag_5',
        'ret_overnight_lag_5', 'ret_intraday_lag_5', 'support_resistance_ratio_lag_1', 'support_resistance_ratio_lag_3',
        'support_resistance_ratio_lag_5', 'hs_price_change', 'hs_amplitude', 'hs_volume_ratio', 'hs_price_change_abs',
        'hs_price_wave_abs', 'hs_sentiment', 'hs_volume_signal', 'gq_price_change', 'gq_amplitude',
        'gq_price_change_abs', 'gq_price_wave_abs', 'gq_sentiment', 'gq_volume_signal', 'hc_price_change',
        'hc_amplitude', 'hc_volume_ratio', 'hc_price_change_abs', 'hc_price_wave_abs', 'hc_sentiment',
        'hc_volume_signal', 'hs_gq_sync_direction', 'hs_gq_sync_strength', 'hs_hc_sync_direction',
        'hs_hc_sync_strength', 'gq_hc_sync_direction', 'gq_hc_sync_strength', 'market_sentiment', 'market_avg_change',
        'market_avg_amplitude', 'market_sync_score', 'cs_n', 'volume_rankpct', 'volume_z', 'amount_rankpct', 'amount_z',
        'macd_rankpct', 'macd_z', 'macd_signal_rankpct', 'macd_signal_z', 'macd_hist_rankpct', 'macd_hist_z',
        'rsi_6_rankpct', 'rsi_6_z', 'rsi_14_rankpct', 'rsi_14_z', 'rsi_24_rankpct', 'rsi_24_z', 'ma_5_rankpct',
        'ma_5_z', 'ma_20_rankpct', 'ma_20_z', 'ma_60_rankpct', 'ma_60_z', 'bb_upper_rankpct', 'bb_upper_z',
        'bb_middle_rankpct', 'bb_middle_z', 'bb_lower_rankpct', 'bb_lower_z', 'volume_ma_20_rankpct', 'volume_ma_20_z',
        'obv_rankpct', 'obv_z', 'atr_rankpct', 'atr_z', 'slowk_rankpct', 'slowk_z', 'slowd_rankpct', 'slowd_z',
        'price_vs_ma5_rankpct', 'price_vs_ma5_z', 'price_vs_ma20_rankpct', 'price_vs_ma20_z', 'price_vs_ma60_rankpct',
        'price_vs_ma60_z', 'ma_arrangement_rankpct', 'ma_arrangement_z', 'bb_position_rankpct', 'bb_position_z',
        'bb_squeeze_rankpct', 'bb_squeeze_z', 'rsi_oversold_6_rankpct', 'rsi_oversold_6_z', 'rsi_oversold_14_rankpct',
        'rsi_oversold_14_z', 'rsi_oversold_24_rankpct', 'rsi_oversold_24_z', 'rsi_overbought_6_rankpct',
        'rsi_overbought_6_z', 'rsi_overbought_14_rankpct', 'rsi_overbought_14_z', 'rsi_overbought_24_rankpct',
        'rsi_overbought_24_z', 'macd_signal_distance_rankpct', 'macd_signal_distance_z', 'macd_golden_cross_rankpct',
        'macd_golden_cross_z', 'volume_ma20_rankpct', 'volume_ma20_z', 'volume_ratio_rankpct', 'volume_ratio_z',
        'volume_spike_rankpct', 'volume_spike_z', 'volume_dryup_rankpct', 'volume_dryup_z', 'atr_ratio_rankpct',
        'atr_ratio_z', 'hammer_pattern_rankpct', 'hammer_pattern_z', 'doji_pattern_rankpct', 'doji_pattern_z',
        'distance_to_support_rankpct', 'distance_to_support_z', 'distance_to_resistance_rankpct',
        'distance_to_resistance_z', 'stoch_oversold_rankpct', 'stoch_oversold_z', 'stoch_overbought_rankpct',
        'stoch_overbought_z', 'macd_percentile_rankpct', 'macd_percentile_z', 'obv_trend_rankpct', 'obv_trend_z',
        'volatility_3d_rankpct', 'volatility_3d_z', 'volatility_5d_rankpct', 'volatility_5d_z',
        'volatility_10d_rankpct', 'volatility_10d_z', 'volatility_15d_rankpct', 'volatility_15d_z',
        'volatility_20d_rankpct', 'volatility_20d_z', 'volatility_25d_rankpct', 'volatility_25d_z',
        'volatility_30d_rankpct', 'volatility_30d_z', 'engulfing_pattern_rankpct', 'engulfing_pattern_z',
        'signed_volume_strength_rankpct', 'close_vs_high_rankpct', 'volume_ma_ratio_rankpct', 'volume_ma_ratio_z',
        'rsi_momentum_rankpct', 'rsi_momentum_z', 'rsi_turning_simple_rankpct', 'rsi_turning_simple_z',
        'rsi_turning_rankpct', 'rsi_turning_z', 'volume_trend_5_rankpct', 'volume_trend_5_z', 'volume_trend_10_rankpct',
        'volume_trend_10_z', 'price_trend_5_rankpct', 'price_trend_5_z', 'price_volume_divergence_rankpct',
        'price_volume_divergence_z', 'volume_consistency_rankpct', 'volume_consistency_z', 'macd_hist_trend_5_rankpct',
        'macd_hist_trend_5_z', 'macd_death_cross_rankpct', 'macd_death_cross_z', 'macd_signal_cross_rankpct',
        'macd_signal_cross_z', 'macd_zero_cross_up_rankpct', 'macd_zero_cross_up_z', 'macd_zero_cross_down_rankpct',
        'macd_zero_cross_down_z', 'macd_zero_cross_rankpct', 'macd_zero_cross_z', 'macd_hist_amplitude_rankpct',
        'macd_hist_amplitude_z', 'macd_hist_direction_rankpct', 'macd_hist_direction_z',
        'macd_hist_acceleration_rankpct', 'macd_hist_acceleration_z', 'macd_signal_convergence_rankpct',
        'macd_signal_convergence_z', 'macd_signal_convergence_trend_rankpct', 'macd_signal_convergence_trend_z',
        'pct_change_rankpct', 'clv_rankpct', 'clv_z', 'upper_shadow_ratio_rankpct', 'upper_shadow_ratio_z',
        'body_strength_rankpct', 'body_strength_z', 'rank_return_rankpct', 'rank_return_z', 'rank_volume_rankpct',
        'rank_volume_z', 'signed_vol_strength_rankpct', 'signed_vol_strength_z', 'pv_corr_10_rankpct',
        'dist_to_high_60_rankpct', 'dist_to_high_60_z', 'vol_divergence_rankpct', 'vol_divergence_z', 'illiq_rankpct',
        'efficiency_ratio_rankpct', 'efficiency_ratio_z', 'intraday_pos_rankpct', 'intraday_pos_z',
        'ret_overnight_rankpct', 'ret_intraday_rankpct', 'smart_money_diff_rankpct', 'high_mean_20_rankpct',
        'high_mean_20_z', 'low_mean_20_rankpct', 'low_mean_20_z', 'support_resistance_ratio_rankpct',
        'support_resistance_ratio_z', 'log_volume_rankpct', 'log_volume_z', 'boxcox_atr_rankpct', 'boxcox_atr_z',
        'rsi_robust_rankpct', 'rsi_robust_z', 'macd_robust_rankpct', 'macd_robust_z', 
        'open_lag_3_rankpct', 'open_lag_3_z', 'high_lag_3_rankpct', 'high_lag_3_z', 'low_lag_3_rankpct', 'low_lag_3_z',
        'close_lag_5_rankpct', 'close_lag_5_z', 'open_lag_5_rankpct', 'open_lag_5_z', 'high_lag_5_rankpct',
        'high_lag_5_z', 'low_lag_5_rankpct', 'low_lag_5_z', 'close_lag_10_rankpct', 'close_lag_10_z',
        'open_lag_10_rankpct', 'open_lag_10_z', 'high_lag_10_rankpct', 'high_lag_10_z', 'low_lag_10_rankpct',
        'low_lag_10_z', 'close_lag_15_rankpct', 'close_lag_15_z', 'open_lag_15_rankpct', 'open_lag_15_z',
        'high_lag_15_rankpct', 'high_lag_15_z', 'low_lag_15_rankpct', 'low_lag_15_z', 'close_lag_20_rankpct',
        'close_lag_20_z', 'open_lag_20_rankpct', 'open_lag_20_z', 'high_lag_20_rankpct', 'high_lag_20_z',
        'low_lag_20_rankpct', 'low_lag_20_z', 'close_lag_25_rankpct', 'close_lag_25_z', 'open_lag_25_rankpct',
        'open_lag_25_z', 'high_lag_25_rankpct', 'high_lag_25_z', 'low_lag_25_rankpct', 'low_lag_25_z',
        'close_lag_30_rankpct', 'close_lag_30_z', 'open_lag_30_rankpct', 'open_lag_30_z', 'high_lag_30_rankpct',
        'high_lag_30_z', 'low_lag_30_rankpct', 'low_lag_30_z', 'volume_lag_3_rankpct', 'volume_lag_3_z',
        'volume_lag_5_rankpct', 'volume_lag_5_z', 'volume_lag_10_rankpct', 'volume_lag_10_z', 'volume_lag_15_rankpct',
        'volume_lag_15_z', 'volume_lag_20_rankpct', 'volume_lag_20_z', 'volume_lag_25_rankpct', 'volume_lag_25_z',
        'volume_lag_30_rankpct', 'volume_lag_30_z', 'daily_return_rankpct', 'return_lag_1_rankpct',
        'return_lag_2_rankpct', 'return_lag_3_rankpct', 'return_lag_5_rankpct', 'return_lag_10_rankpct',
        'return_lag_20_rankpct', 'amplitude_rankpct',  
         'illiq_lag_1_rankpct', 'illiq_lag_3_rankpct', 'illiq_lag_5_rankpct',
        'efficiency_ratio_lag_1_rankpct', 'efficiency_ratio_lag_1_z', 'efficiency_ratio_lag_3_rankpct',
        'efficiency_ratio_lag_3_z', 'efficiency_ratio_lag_5_rankpct', 'efficiency_ratio_lag_5_z',
        'intraday_pos_lag_1_rankpct', 'intraday_pos_lag_1_z', 'intraday_pos_lag_2_rankpct', 'intraday_pos_lag_2_z',
        'intraday_pos_lag_3_rankpct', 'intraday_pos_lag_3_z', 'smart_money_diff_lag_1_rankpct',
        'ret_overnight_lag_1_rankpct', 'ret_intraday_lag_1_rankpct', 
        'smart_money_diff_lag_3_rankpct', 'ret_overnight_lag_3_rankpct', 'ret_intraday_lag_3_rankpct',
        'smart_money_diff_lag_5_rankpct', 'ret_overnight_lag_5_rankpct', 'ret_intraday_lag_5_rankpct',
        'support_resistance_ratio_lag_1_rankpct', 'support_resistance_ratio_lag_1_z',
        'support_resistance_ratio_lag_3_rankpct', 'support_resistance_ratio_lag_3_z',
        'support_resistance_ratio_lag_5_rankpct', 'support_resistance_ratio_lag_5_z', 'hs_price_change_rankpct',
        'hs_price_change_z', 'hs_amplitude_rankpct', 'hs_amplitude_z', 'hs_volume_ratio_rankpct', 'hs_volume_ratio_z',
        'hs_price_change_abs_rankpct', 'hs_price_change_abs_z', 'hs_price_wave_abs_rankpct', 'hs_price_wave_abs_z',
        'hs_sentiment_rankpct', 'hs_sentiment_z', 'hs_volume_signal_rankpct', 'hs_volume_signal_z',
        'gq_price_change_rankpct', 'gq_price_change_z', 'gq_amplitude_rankpct', 'gq_amplitude_z',
        'gq_price_change_abs_rankpct', 'gq_price_change_abs_z', 'gq_price_wave_abs_rankpct', 'gq_price_wave_abs_z',
        'gq_sentiment_rankpct', 'gq_sentiment_z', 'gq_volume_signal_rankpct', 'gq_volume_signal_z',
        'hc_price_change_rankpct', 'hc_price_change_z', 'hc_amplitude_rankpct', 'hc_amplitude_z',
        'hc_volume_ratio_rankpct', 'hc_volume_ratio_z', 'hc_price_change_abs_rankpct', 'hc_price_change_abs_z',
        'hc_price_wave_abs_rankpct', 'hc_price_wave_abs_z', 'hc_sentiment_rankpct', 'hc_sentiment_z',
        'hc_volume_signal_rankpct', 'hc_volume_signal_z', 'hs_gq_sync_direction_rankpct', 'hs_gq_sync_direction_z',
        'hs_gq_sync_strength_rankpct', 'hs_gq_sync_strength_z', 'hs_hc_sync_direction_rankpct',
        'hs_hc_sync_direction_z', 'hs_hc_sync_strength_rankpct', 'hs_hc_sync_strength_z',
        'gq_hc_sync_direction_rankpct', 'gq_hc_sync_direction_z', 'gq_hc_sync_strength_rankpct',
        'gq_hc_sync_strength_z', 'market_sentiment_rankpct', 'market_sentiment_z', 'market_avg_change_rankpct',
        'market_avg_change_z', 'market_avg_amplitude_rankpct', 'market_avg_amplitude_z', 'market_sync_score_rankpct',
        'market_sync_score_z', 'is_quick_divergence_x', 'current_idx', 'prev_idx', 'compare_rank', 'close_current',
        'close_previous', 'macd_current', 'macd_previous', 'price_decline_pct', 'macd_increase_pct', 'formation_period',
        'is_quick_divergence_y', 'divergence_strength', 'price_macd_ratio', 'divergence_magnitude',
        'confirmation_score', 'day_of_year', 'week_of_year', 'quarter', 'day_of_month', 'is_month_end',
        'is_month_start', 'is_quarter_end', 'is_quarter_start', 'volume_signal', 'hammer_signal',
    ]

    # 全量
    FEATURE_COLS = [
        'volume', 'amount', 'macd', 'macd_signal', 'macd_hist', 'rsi_6', 'rsi_14', 'rsi_24', 'ma_5', 'ma_20', 'ma_60',
        'bb_upper', 'bb_middle', 'bb_lower', 'volume_ma_20', 'obv', 'atr', 'slowk', 'slowd', 'price_vs_ma5',
        'price_vs_ma20', 'price_vs_ma60', 'ma_arrangement', 'bb_position', 'bb_squeeze', 'rsi_oversold_6',
        'rsi_oversold_14', 'rsi_oversold_24', 'rsi_overbought_6', 'rsi_overbought_14', 'rsi_overbought_24',
        'macd_signal_distance', 'macd_golden_cross', 'volume_ma20', 'volume_ratio', 'volume_spike', 'volume_dryup',
        'atr_ratio', 'hammer_pattern', 'doji_pattern', 'distance_to_support', 'distance_to_resistance',
        'stoch_oversold', 'stoch_overbought', 'macd_percentile', 'obv_trend', 'volatility_3d', 'volatility_5d',
        'volatility_10d', 'volatility_15d', 'volatility_20d', 'volatility_25d', 'volatility_30d', 'engulfing_pattern',
        'signed_volume_strength', 'close_vs_high', 'volume_ma_ratio', 'rsi_momentum', 'rsi_turning_simple',
        'rsi_turning', 'volume_trend_5', 'volume_trend_10', 'price_trend_5', 'price_volume_divergence',
        'volume_consistency', 'macd_hist_trend_5', 'macd_death_cross', 'macd_signal_cross', 'macd_zero_cross_up',
        'macd_zero_cross_down', 'macd_zero_cross', 'macd_hist_amplitude', 'macd_hist_direction',
        'macd_hist_acceleration', 'macd_signal_convergence', 'macd_signal_convergence_trend', 'pct_change', 'clv',
        'upper_shadow_ratio', 'body_strength', 'rank_return', 'rank_volume', 'signed_vol_strength', 'pv_corr_10',
        'dist_to_high_60', 'vol_divergence', 'illiq', 'efficiency_ratio', 'intraday_pos', 'ret_overnight',
        'ret_intraday', 'smart_money_diff', 'high_mean_20', 'low_mean_20', 'support_resistance_ratio', 'log_volume',
        'boxcox_atr', 'rsi_robust', 'macd_robust',  
        'high_lag_3', 'low_lag_3', 'close_lag_5', 'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_10',
        'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_15', 'open_lag_15', 'high_lag_15', 'low_lag_15',
        'close_lag_20', 'open_lag_20', 'high_lag_20', 'low_lag_20', 'close_lag_25', 'open_lag_25', 'high_lag_25',
        'low_lag_25', 'close_lag_30', 'open_lag_30', 'high_lag_30', 'low_lag_30', 'volume_lag_3', 'volume_lag_5',
        'volume_lag_10', 'volume_lag_15', 'volume_lag_20', 'volume_lag_25', 'volume_lag_30', 'daily_return',
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20', 'amplitude',
           'illiq_lag_1', 'illiq_lag_3', 'illiq_lag_5',
        'efficiency_ratio_lag_1', 'efficiency_ratio_lag_3', 'efficiency_ratio_lag_5', 'intraday_pos_lag_1',
        'intraday_pos_lag_2', 'intraday_pos_lag_3', 'smart_money_diff_lag_1', 'ret_overnight_lag_1',
        'smart_money_diff_lag_3', 'ret_overnight_lag_3', 'ret_intraday_lag_3', 'smart_money_diff_lag_5',
        'ret_overnight_lag_5', 'ret_intraday_lag_5', 'support_resistance_ratio_lag_1', 'support_resistance_ratio_lag_3',
        'support_resistance_ratio_lag_5', 'hs_price_change', 'hs_amplitude', 'hs_volume_ratio', 'hs_price_change_abs',
        'hs_price_wave_abs', 'hs_sentiment', 'hs_volume_signal', 'gq_price_change', 'gq_amplitude',
        'gq_price_change_abs', 'gq_price_wave_abs', 'gq_sentiment', 'gq_volume_signal', 'hc_price_change',
        'hc_amplitude', 'hc_volume_ratio', 'hc_price_change_abs', 'hc_price_wave_abs', 'hc_sentiment',
        'hc_volume_signal', 'hs_gq_sync_direction', 'hs_gq_sync_strength', 'hs_hc_sync_direction',
        'hs_hc_sync_strength', 'gq_hc_sync_direction', 'gq_hc_sync_strength', 'market_sentiment', 'market_avg_change',
        'market_avg_amplitude', 'market_sync_score', 'cs_n', 'volume_rankpct', 'volume_z', 'amount_rankpct', 'amount_z',
        'macd_rankpct', 'macd_z', 'macd_signal_rankpct', 'macd_signal_z', 'macd_hist_rankpct', 'macd_hist_z',
        'rsi_6_rankpct', 'rsi_6_z', 'rsi_14_rankpct', 'rsi_14_z', 'rsi_24_rankpct', 'rsi_24_z', 'ma_5_rankpct',
        'ma_5_z', 'ma_20_rankpct', 'ma_20_z', 'ma_60_rankpct', 'ma_60_z', 'bb_upper_rankpct', 'bb_upper_z',
        'bb_middle_rankpct', 'bb_middle_z', 'bb_lower_rankpct', 'bb_lower_z', 'volume_ma_20_rankpct', 'volume_ma_20_z',
        'obv_rankpct', 'obv_z', 'atr_rankpct', 'atr_z', 'slowk_rankpct', 'slowk_z', 'slowd_rankpct', 'slowd_z',
        'price_vs_ma5_rankpct', 'price_vs_ma5_z', 'price_vs_ma20_rankpct', 'price_vs_ma20_z', 'price_vs_ma60_rankpct',
        'price_vs_ma60_z', 'ma_arrangement_rankpct', 'ma_arrangement_z', 'bb_position_rankpct', 'bb_position_z',
        'bb_squeeze_rankpct', 'bb_squeeze_z', 'rsi_oversold_6_rankpct', 'rsi_oversold_6_z', 'rsi_oversold_14_rankpct',
        'rsi_oversold_14_z', 'rsi_oversold_24_rankpct', 'rsi_oversold_24_z', 'rsi_overbought_6_rankpct',
        'rsi_overbought_6_z', 'rsi_overbought_14_rankpct', 'rsi_overbought_14_z', 'rsi_overbought_24_rankpct',
        'rsi_overbought_24_z', 'macd_signal_distance_rankpct', 'macd_signal_distance_z', 'macd_golden_cross_rankpct',
        'macd_golden_cross_z', 'volume_ma20_rankpct', 'volume_ma20_z', 'volume_ratio_rankpct', 'volume_ratio_z',
        'volume_spike_rankpct', 'volume_spike_z', 'volume_dryup_rankpct', 'volume_dryup_z', 'atr_ratio_rankpct',
        'atr_ratio_z', 'hammer_pattern_rankpct', 'hammer_pattern_z', 'doji_pattern_rankpct', 'doji_pattern_z',
        'distance_to_support_rankpct', 'distance_to_support_z', 'distance_to_resistance_rankpct',
        'distance_to_resistance_z', 'stoch_oversold_rankpct', 'stoch_oversold_z', 'stoch_overbought_rankpct',
        'stoch_overbought_z', 'macd_percentile_rankpct', 'macd_percentile_z', 'obv_trend_rankpct', 'obv_trend_z',
        'volatility_3d_rankpct', 'volatility_3d_z', 'volatility_5d_rankpct', 'volatility_5d_z',
        'volatility_10d_rankpct', 'volatility_10d_z', 'volatility_15d_rankpct', 'volatility_15d_z',
        'volatility_20d_rankpct', 'volatility_20d_z', 'volatility_25d_rankpct', 'volatility_25d_z',
        'volatility_30d_rankpct', 'volatility_30d_z', 'engulfing_pattern_rankpct', 'engulfing_pattern_z',
        'signed_volume_strength_rankpct', 'close_vs_high_rankpct', 'volume_ma_ratio_rankpct', 'volume_ma_ratio_z',
        'rsi_momentum_rankpct', 'rsi_momentum_z', 'rsi_turning_simple_rankpct', 'rsi_turning_simple_z',
        'rsi_turning_rankpct', 'rsi_turning_z', 'volume_trend_5_rankpct', 'volume_trend_5_z', 'volume_trend_10_rankpct',
        'volume_trend_10_z', 'price_trend_5_rankpct', 'price_trend_5_z', 'price_volume_divergence_rankpct',
        'price_volume_divergence_z', 'volume_consistency_rankpct', 'volume_consistency_z', 'macd_hist_trend_5_rankpct',
        'macd_hist_trend_5_z', 'macd_death_cross_rankpct', 'macd_death_cross_z', 'macd_signal_cross_rankpct',
        'macd_signal_cross_z', 'macd_zero_cross_up_rankpct', 'macd_zero_cross_up_z', 'macd_zero_cross_down_rankpct',
        'macd_zero_cross_down_z', 'macd_zero_cross_rankpct', 'macd_zero_cross_z', 'macd_hist_amplitude_rankpct',
        'macd_hist_amplitude_z', 'macd_hist_direction_rankpct', 'macd_hist_direction_z',
        'macd_hist_acceleration_rankpct', 'macd_hist_acceleration_z', 'macd_signal_convergence_rankpct',
        'macd_signal_convergence_z', 'macd_signal_convergence_trend_rankpct', 'macd_signal_convergence_trend_z',
        'pct_change_rankpct', 'clv_rankpct', 'clv_z', 'upper_shadow_ratio_rankpct', 'upper_shadow_ratio_z',
        'body_strength_rankpct', 'body_strength_z', 'rank_return_rankpct', 'rank_return_z', 'rank_volume_rankpct',
        'rank_volume_z', 'signed_vol_strength_rankpct', 'signed_vol_strength_z', 'pv_corr_10_rankpct',
        'dist_to_high_60_rankpct', 'dist_to_high_60_z', 'vol_divergence_rankpct', 'vol_divergence_z', 'illiq_rankpct',
        'efficiency_ratio_rankpct', 'efficiency_ratio_z', 'intraday_pos_rankpct', 'intraday_pos_z',
        'ret_overnight_rankpct', 'ret_intraday_rankpct', 'smart_money_diff_rankpct', 'high_mean_20_rankpct',
        'high_mean_20_z', 'low_mean_20_rankpct', 'low_mean_20_z', 'support_resistance_ratio_rankpct',
        'support_resistance_ratio_z', 'log_volume_rankpct', 'log_volume_z', 'boxcox_atr_rankpct', 'boxcox_atr_z',
        'rsi_robust_rankpct', 'rsi_robust_z', 'macd_robust_rankpct', 'macd_robust_z',
        'open_lag_3_rankpct', 'open_lag_3_z', 'high_lag_3_rankpct', 'high_lag_3_z', 'low_lag_3_rankpct', 'low_lag_3_z',
        'close_lag_5_rankpct', 'close_lag_5_z', 'open_lag_5_rankpct', 'open_lag_5_z', 'high_lag_5_rankpct',
        'high_lag_5_z', 'low_lag_5_rankpct', 'low_lag_5_z', 'close_lag_10_rankpct', 'close_lag_10_z',
        'open_lag_10_rankpct', 'open_lag_10_z', 'high_lag_10_rankpct', 'high_lag_10_z', 'low_lag_10_rankpct',
        'low_lag_10_z', 'close_lag_15_rankpct', 'close_lag_15_z', 'open_lag_15_rankpct', 'open_lag_15_z',
        'high_lag_15_rankpct', 'high_lag_15_z', 'low_lag_15_rankpct', 'low_lag_15_z', 'close_lag_20_rankpct',
        'close_lag_20_z', 'open_lag_20_rankpct', 'open_lag_20_z', 'high_lag_20_rankpct', 'high_lag_20_z',
        'low_lag_20_rankpct', 'low_lag_20_z', 'close_lag_25_rankpct', 'close_lag_25_z', 'open_lag_25_rankpct',
        'open_lag_25_z', 'high_lag_25_rankpct', 'high_lag_25_z', 'low_lag_25_rankpct', 'low_lag_25_z',
        'close_lag_30_rankpct', 'close_lag_30_z', 'open_lag_30_rankpct', 'open_lag_30_z', 'high_lag_30_rankpct',
        'high_lag_30_z', 'low_lag_30_rankpct', 'low_lag_30_z', 'volume_lag_3_rankpct', 'volume_lag_3_z',
        'volume_lag_5_rankpct', 'volume_lag_5_z', 'volume_lag_10_rankpct', 'volume_lag_10_z', 'volume_lag_15_rankpct',
        'volume_lag_15_z', 'volume_lag_20_rankpct', 'volume_lag_20_z', 'volume_lag_25_rankpct', 'volume_lag_25_z',
        'volume_lag_30_rankpct', 'volume_lag_30_z', 'daily_return_rankpct', 'return_lag_1_rankpct',
        'return_lag_2_rankpct', 'return_lag_3_rankpct', 'return_lag_5_rankpct', 'return_lag_10_rankpct',
        'return_lag_20_rankpct', 'amplitude_rankpct',  
         'illiq_lag_1_rankpct', 'illiq_lag_3_rankpct', 'illiq_lag_5_rankpct',
        'efficiency_ratio_lag_1_rankpct', 'efficiency_ratio_lag_1_z', 'efficiency_ratio_lag_3_rankpct',
        'efficiency_ratio_lag_3_z', 'efficiency_ratio_lag_5_rankpct', 'efficiency_ratio_lag_5_z',
        'intraday_pos_lag_1_rankpct', 'intraday_pos_lag_1_z', 'intraday_pos_lag_2_rankpct', 'intraday_pos_lag_2_z',
        'intraday_pos_lag_3_rankpct', 'intraday_pos_lag_3_z', 'smart_money_diff_lag_1_rankpct',
        'ret_overnight_lag_1_rankpct', 'ret_intraday_lag_1_rankpct', 
        'smart_money_diff_lag_3_rankpct', 'ret_overnight_lag_3_rankpct', 'ret_intraday_lag_3_rankpct',
        'smart_money_diff_lag_5_rankpct', 'ret_overnight_lag_5_rankpct', 'ret_intraday_lag_5_rankpct',
        'support_resistance_ratio_lag_1_rankpct', 'support_resistance_ratio_lag_1_z',
        'support_resistance_ratio_lag_3_rankpct', 'support_resistance_ratio_lag_3_z',
        'support_resistance_ratio_lag_5_rankpct', 'support_resistance_ratio_lag_5_z', 'hs_price_change_rankpct',
        'hs_price_change_z', 'hs_amplitude_rankpct', 'hs_amplitude_z', 'hs_volume_ratio_rankpct', 'hs_volume_ratio_z',
        'hs_price_change_abs_rankpct', 'hs_price_change_abs_z', 'hs_price_wave_abs_rankpct', 'hs_price_wave_abs_z',
        'hs_sentiment_rankpct', 'hs_sentiment_z', 'hs_volume_signal_rankpct', 'hs_volume_signal_z',
        'gq_price_change_rankpct', 'gq_price_change_z', 'gq_amplitude_rankpct', 'gq_amplitude_z',
        'gq_price_change_abs_rankpct', 'gq_price_change_abs_z', 'gq_price_wave_abs_rankpct', 'gq_price_wave_abs_z',
        'gq_sentiment_rankpct', 'gq_sentiment_z', 'gq_volume_signal_rankpct', 'gq_volume_signal_z',
        'hc_price_change_rankpct', 'hc_price_change_z', 'hc_amplitude_rankpct', 'hc_amplitude_z',
        'hc_volume_ratio_rankpct', 'hc_volume_ratio_z', 'hc_price_change_abs_rankpct', 'hc_price_change_abs_z',
        'hc_price_wave_abs_rankpct', 'hc_price_wave_abs_z', 'hc_sentiment_rankpct', 'hc_sentiment_z',
        'hc_volume_signal_rankpct', 'hc_volume_signal_z', 'hs_gq_sync_direction_rankpct', 'hs_gq_sync_direction_z',
        'hs_gq_sync_strength_rankpct', 'hs_gq_sync_strength_z', 'hs_hc_sync_direction_rankpct',
        'hs_hc_sync_direction_z', 'hs_hc_sync_strength_rankpct', 'hs_hc_sync_strength_z',
        'gq_hc_sync_direction_rankpct', 'gq_hc_sync_direction_z', 'gq_hc_sync_strength_rankpct',
        'gq_hc_sync_strength_z', 'market_sentiment_rankpct', 'market_sentiment_z', 'market_avg_change_rankpct',
        'market_avg_change_z', 'market_avg_amplitude_rankpct', 'market_avg_amplitude_z', 'market_sync_score_rankpct',
        'market_sync_score_z', 'is_quick_divergence_x', 'current_idx', 'prev_idx', 'compare_rank', 'close_current',
        'close_previous', 'macd_current', 'macd_previous', 'price_decline_pct', 'macd_increase_pct', 'formation_period',
        'is_quick_divergence_y', 'divergence_strength', 'price_macd_ratio', 'divergence_magnitude',
        'confirmation_score', 'day_of_year', 'week_of_year', 'quarter', 'day_of_month', 'is_month_end',
        'is_month_start', 'is_quarter_end', 'is_quarter_start', 'volume_signal', 'hammer_signal',
    ]

    # 最优
    OPTIMIZED_FEATURE_COLS = [
        "support_resistance_ratio_lag_3", "low_mean_20", "macd_hist_direction_rankpct", "close_lag_30_z",
        "ma_arrangement_z", "support_resistance_ratio_lag_5_z", "rank_return_rankpct", "atr_ratio", "amount",
        "close_lag_5_z", "high_mean_20_rankpct", "volatility_30d_z", "intraday_pos_lag_1_z", "return_lag_3",
        "rsi_overbought_24_rankpct", "rsi_overbought_6_rankpct", "efficiency_ratio_lag_5_z", "efficiency_ratio_lag_1_z",
        "macd_zero_cross_rankpct", "hs_volume_ratio", "distance_to_resistance", "support_resistance_ratio_lag_3_z",
        "vol_divergence", "return_lag_10", "low_lag_30", "illiq", "volume_spike_rankpct", "current_idx",
        "market_avg_change", "slowk", "return_lag_1", "illiq_lag_5", "price_decline_pct", "macd_current", "rsi_24",
        "daily_return", "intraday_pos_lag_3_z", "ma_arrangement_rankpct", "close_lag_25_rankpct",
        "macd_hist_direction_z", "intraday_pos_lag_2_rankpct", "rank_return", "close_lag_15", "smart_money_diff_lag_5",
        "clv", "macd_hist_amplitude_z", "volume_lag_20_rankpct", "stoch_overbought_rankpct", "atr_ratio_z",
        "rsi_turning_simple_rankpct", "dist_to_high_60", "gq_price_wave_abs", "support_resistance_ratio_rankpct",
        "illiq_lag_3_rankpct", "amount_rankpct", "close_previous", "market_avg_amplitude", "high_lag_30",
        "body_strength", "hs_amplitude", "intraday_pos_lag_2_z", "log_volume_z", "gq_price_change", "atr_rankpct",
        "close_lag_10_z", "volatility_10d", "ret_overnight_lag_1_rankpct", "price_vs_ma20_rankpct", "day_of_month",
        "distance_to_support_rankpct", "rsi_oversold_6_rankpct", "clv_z", "low_mean_20_z", "return_lag_1_rankpct",
        "macd_signal_cross_rankpct", "clv_rankpct", "efficiency_ratio_lag_1", "ret_overnight_lag_1", "ret_intraday",
        "rsi_24_rankpct", "volume_lag_5_rankpct", "illiq_lag_5_rankpct", "low_lag_25", "high_lag_30_z",
        "return_lag_2_rankpct", "volume_lag_15", "price_volume_divergence_z", "pv_corr_10", "ret_overnight",
        "close_lag_30_rankpct", "atr_ratio_rankpct", "volume_lag_10", "volume_lag_3_z",
        "support_resistance_ratio_lag_1", "return_lag_2", "amplitude", "illiq_lag_1_rankpct", "volume_ma_20",
        "support_resistance_ratio_lag_1_rankpct", "support_resistance_ratio_z", "gq_price_change_abs",
        "macd_zero_cross_up_rankpct", "support_resistance_ratio_lag_1_z", "close_lag_15_z", "volume", "high_mean_20_z",
        "stoch_oversold_rankpct", "gq_amplitude", "open_lag_30", "dist_to_high_60_rankpct", "distance_to_resistance_z",
        "pct_change", "smart_money_diff_lag_1", "volume_lag_10_rankpct", "obv", "dist_to_high_60_z",
        "macd_golden_cross_rankpct", "bb_squeeze_rankpct", "efficiency_ratio", "engulfing_pattern_rankpct",
        "price_volume_divergence_rankpct", "volume_dryup_rankpct", "bb_upper_rankpct", "ma_5_rankpct",
        "low_mean_20_rankpct", "close_lag_30", "atr", "boxcox_atr", "volume_lag_3", "close_lag_20",
        "efficiency_ratio_lag_3", "intraday_pos_z", "ret_intraday_rankpct", "week_of_year", "volume_lag_30",
        "support_resistance_ratio_lag_3_rankpct", "support_resistance_ratio", "illiq_rankpct", "pv_corr_10_rankpct",
        "illiq_lag_1", "support_resistance_ratio_lag_5_rankpct", "high_mean_20", "amount_z", "close_lag_20_z",
        "pct_change_rankpct", "volume_ma_20_rankpct", "volume_lag_25", "slowd", "bb_upper_z", "prev_idx",
        "volume_lag_15_z", "hc_volume_ratio", "smart_money_diff", "macd_death_cross_rankpct", "volume_lag_20_z",
        "volume_lag_5_z", "volatility_25d", "return_lag_5", "macd_zero_cross_down_rankpct", "macd_previous",
        "open_lag_5_rankpct", "intraday_pos_lag_3", "cs_n", "support_resistance_ratio_lag_5", "hs_price_change",
        "close_lag_25", "efficiency_ratio_lag_5", "day_of_year", "close_current", "intraday_pos_lag_1_rankpct", "ma_5",
        "rsi_turning_rankpct", "open_lag_25", "obv_rankpct", "hs_gq_sync_strength", "volume_dryup_z",
        "close_lag_5_rankpct", "daily_return_rankpct", "rsi_overbought_14_rankpct", "intraday_pos_rankpct",
        "rsi_oversold_14_rankpct"
    ]

    # 用于 逻辑回归 层的稳定参数
    STABLE_FEATURES = [
        "ma_arrangement_rankpct", "ma_arrangement_z", "dist_to_high_60_z", "dist_to_high_60", "cs_n",
        "support_resistance_ratio_z", "price_decline_pct", "support_resistance_ratio", "day_of_year",
        "dist_to_high_60_rankpct", "day_of_month", "illiq_rankpct", "gq_amplitude", "macd_hist_direction_z",
        "gq_price_change", "macd_hist_direction_rankpct", "close_lag_30_rankpct", "support_resistance_ratio_lag_1_z",
        "illiq", "return_lag_2", "return_lag_1_rankpct", "support_resistance_ratio_rankpct", "volume_spike_rankpct",
        "low_mean_20_rankpct", "current_idx", "intraday_pos_lag_1_z", "rank_return", "daily_return_rankpct",
        "clv_rankpct", "intraday_pos_lag_3_z", "gq_price_change_abs", "volume_lag_25", "pct_change_rankpct",
        "support_resistance_ratio_lag_5_rankpct", "illiq_lag_1_rankpct", "hs_price_change", "amplitude",
        "support_resistance_ratio_lag_5_z", "return_lag_1", "volume_dryup_rankpct", "macd_zero_cross_down_rankpct",
        "intraday_pos_lag_2_rankpct", "prev_idx", "macd_death_cross_rankpct", "atr_rankpct", "bb_upper_z",
        "efficiency_ratio_lag_5", "atr", "amount_rankpct", "volume_ma_20", "intraday_pos_z", "clv_z", "illiq_lag_1",
        "support_resistance_ratio_lag_1", "stoch_overbought_rankpct", "rsi_turning_simple_rankpct", "obv",
        "obv_rankpct", "support_resistance_ratio_lag_5", "amount_z", "return_lag_5", "pct_change", "daily_return",
        "close_lag_25", "macd_golden_cross_rankpct", "boxcox_atr", "atr_ratio", "market_avg_amplitude",
        "illiq_lag_5_rankpct", "engulfing_pattern_rankpct", "rsi_overbought_24_rankpct", "atr_ratio_rankpct",
        "rsi_overbought_14_rankpct", "efficiency_ratio_lag_1", "macd_zero_cross_up_rankpct", "volume_ma_20_rankpct",
    ]

    STRATEGY_PARAMS_CANDIDATES = {
        "参数1": {
            # 测试&验证集表现： 回报率:4.4155895962595455 倍  |  最大回撤： -0.30762444997644 | 胜率：0.5988700564971752 | 总交易数：177.0 | sharpe_ratio：1.893749281487686

            'base_ratio': 1.0,
            'target_profit': 0.35,
            'hard_stop_loss': -0.14,
            'max_hold_days': 25,
            'max_positions': 5,
            'min_probability': 0.55,
        },
        "参数2": {
            # 测试&验证集表现： 回报率:4.343556046449869 倍  |  最大回撤： -0.2689066407696095 | 胜率：0.5892857142857143 | 总交易数：112.0 | sharpe_ratio：1.6934532200426269

            'base_ratio': 1.0,
            'target_profit': 0.35,
            'hard_stop_loss': -0.14,
            'max_hold_days': 25,
            'max_positions': 3,
            'min_probability': 0.55,
        },
        "参数4": {
            # 测试&验证集表现： 回报率:4.054925480633818 倍  |  最大回撤： -0.2607550200003021 | 胜率：0.5892857142857143 | 总交易数：112.0 | sharpe_ratio：1.7269705238902664

            'base_ratio': 0.86,
            'target_profit': 0.35,
            'hard_stop_loss': -0.14,
            'max_hold_days': 25,
            'max_positions': 3,
            'min_probability': 0.55,
        },
        "参数5": {
            # 测试&验证集表现： 回报率:3.993467403812881 倍  |  最大回撤： -0.3039647395511332 | 胜率：0.5988700564971752 | 总交易数：177.0 | sharpe_ratio：1.8829951317387763

            'base_ratio': 0.86,
            'target_profit': 0.35,
            'hard_stop_loss': -0.14,
            'max_hold_days': 25,
            'max_positions': 5,
            'min_probability': 0.55,
        },
        "参数7": {
            # 测试&验证集表现： 回报率:3.927396771363962 倍  |  最大回撤： -0.2486922880000421 | 胜率：0.5706521739130435 | 总交易数：184.0 | sharpe_ratio：1.8441101108037592

            'base_ratio': 1.0,
            'target_profit': 0.35,
            'hard_stop_loss': -0.12,
            'max_hold_days': 25,
            'max_positions': 5,
            'min_probability': 0.55,
        },
        "参数12": {
            # 测试&验证集表现： 回报率:3.773247064693432 倍  |  最大回撤： -0.2743779367396639 | 胜率：0.5518453427065027 | 总交易数：569.0 | sharpe_ratio：1.820856926118496

            'base_ratio': 0.86,
            'target_profit': 0.2,
            'hard_stop_loss': -0.12,
            'max_hold_days': 25,
            'max_positions': 10,
            'min_probability': 0.5,
        },
        "参数13": {
            # 测试&验证集表现： 回报率:3.744923589483595 倍  |  最大回撤： -0.2377550560989665 | 胜率：0.5702479338842975 | 总交易数：121.0 | sharpe_ratio：1.6406033611213324

            'base_ratio': 1.0,
            'target_profit': 0.25,
            'hard_stop_loss': -0.12,
            'max_hold_days': 25,
            'max_positions': 3,
            'min_probability': 0.55,
        },
        "参数14": {
            # 测试&验证集表现： 回报率:3.736008264728841 倍  |  最大回撤： -0.2815895291172428 | 胜率：0.5288461538461539 | 总交易数：728.0 | sharpe_ratio：1.678440363333714

            'base_ratio': 1.0,
            'target_profit': 0.25,
            'hard_stop_loss': -0.12,
            'max_hold_days': 15,
            'max_positions': 10,
            'min_probability': 0.5,
        },
        "参数30": {
            # 测试&验证集表现： 回报率:3.2777853423786643 倍  |  最大回撤： -0.298554911997731 | 胜率：0.5792079207920792 | 总交易数：202.0 | sharpe_ratio：1.7924028657508888

            'base_ratio': 1.0,
            'target_profit': 0.2,
            'hard_stop_loss': -0.12,
            'max_hold_days': 25,
            'max_positions': 5,
            'min_probability': 0.55,
        },
        "参数51": {
            # 测试&验证集表现： 回报率:3.118385199134094 倍  |  最大回撤： -0.2745721627666539 | 胜率：0.5792079207920792 | 总交易数：202.0 | sharpe_ratio：1.8092983086513577

            'base_ratio': 0.86,
            'target_profit': 0.2,
            'hard_stop_loss': -0.12,
            'max_hold_days': 25,
            'max_positions': 5,
            'min_probability': 0.55,
        },
    }
    STRATEGY_PARAMS = STRATEGY_PARAMS_CANDIDATES["参数14"]

EPS = 1e-9 # 极小值
model_config = Config()

# 模式选择: 'z_score' (推荐), 'tiered' (分档)
ALLOCATION_STRATEGY = 'z_score'

# 统计参数 (来自您的报告数据)
PROBA_MEAN = 0.6424  # 均值
PROBA_STD  = 0.0848  # 标准差
PROBA_MIN  = 0.5502  # 最小值
PROBA_MAX  = 0.9392  # 最大值


