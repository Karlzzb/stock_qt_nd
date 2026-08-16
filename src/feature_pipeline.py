import warnings
import glob
import talib
import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import percentileofscore, linregress
from src.comm_fun import model_config, EPS
from scipy import stats
import re
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config.settings import DAILY_FEATURE_DIR,  STOCK_ND_CSV_DIR
from pathlib import Path
DEFAULT_STOCK_DATA_DIR = Path(__file__).parent.parent / 'stock_data'
STOCK_DATA_DIR = Path(os.environ.get('STOCK_DATA_DIR', DEFAULT_STOCK_DATA_DIR))

# 大盘ID
df_sh_symbol = "000001.SH"
df_sz_symbol = "399001.SZ"


# ========================================
# 辅助函数
# ========================================
def macd_features_rolling(df):
    """使用rolling窗口计算MACD特征"""

    # 0. 先清理数据，确保没有inf和NaN
    # 处理macd_hist列的非有限值
    df['macd_hist_clean'] = df['macd_hist'].replace([np.inf, -np.inf], np.nan).fillna(0)
    # 可选：同时清理macd和macd_signal列
    df['macd_clean'] = df['macd'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['macd_signal_clean'] = df['macd_signal'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 1. MACD柱状图趋势（5日）
    def linear_slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        x_mean = np.mean(x)
        y_mean = np.mean(arr)
        numerator = np.sum((x - x_mean) * (arr - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        return numerator / denominator if denominator != 0 else 0.0

    df['macd_hist_trend_5'] = df['macd_hist_clean'].rolling(window=6, min_periods=2).apply(
        linear_slope, raw=True
    ).fillna(0)

    # 2. MACD信号线交叉
    # 注意：原代码只检测金叉，这里我们检测两种交叉
    df['macd_golden_cross'] = (
            (df['macd_clean'].shift(1) < df['macd_signal_clean'].shift(1)) &
            (df['macd_clean'] > df['macd_signal_clean'])
    ).astype(int)

    df['macd_death_cross'] = (
            (df['macd_clean'].shift(1) > df['macd_signal_clean'].shift(1)) &
            (df['macd_clean'] < df['macd_signal_clean'])
    ).astype(int)

    df['macd_signal_cross'] = df['macd_golden_cross']  # 与原代码一致

    # 3. MACD零轴穿越
    df['macd_zero_cross_up'] = (
            (df['macd_clean'].shift(1) <= 0) & (df['macd_clean'] > 0)
    ).astype(int)

    df['macd_zero_cross_down'] = (
            (df['macd_clean'].shift(1) >= 0) & (df['macd_clean'] < 0)
    ).astype(int)

    df['macd_zero_cross'] = (df['macd_zero_cross_up'] | df['macd_zero_cross_down']).astype(int)

    # 4. MACD柱状图振幅
    df['macd_hist_amplitude'] = df['macd_hist_clean'].rolling(window=20, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True
    ).fillna(0)

    # 5. 额外特征：MACD柱状图的变化方向
    # 使用清理后的数据计算方向
    df['macd_hist_direction'] = np.sign(df['macd_hist_clean']).astype(int)

    # 6. MACD柱状图加速/减速
    df['macd_hist_acceleration'] = df['macd_hist_clean'].diff().diff()

    # 7. MACD双线收敛/发散
    df['macd_signal_convergence'] = df['macd_clean'] - df['macd_signal_clean']
    df['macd_signal_convergence_trend'] = df['macd_signal_convergence'].rolling(
        window=5, min_periods=2
    ).apply(linear_slope, raw=True).fillna(0)

    # 8. 可选：清理临时列
    # 删除临时清理列，如果需要保留原始列
    df.drop(['macd_hist_clean', 'macd_clean', 'macd_signal_clean'], axis=1, inplace=True, errors='ignore')

    return df

def calculate_volume_features_vectorized(df):
    """
    向量化计算成交量相关特征
    """
    # 初始化结果列
    result = pd.DataFrame(index=df.index)

    # 辅助函数：计算滑动窗口的线性回归斜率
    def rolling_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            x = np.arange(window)
            y = series.iloc[i - window + 1:i + 1].values
            if len(y) == window:
                slope, _, _, _, _ = linregress(x, y)
                slopes[i] = slope
        return slopes

    # 计算成交量趋势（5日和10日）
    # 注意：窗口大小是6和11（包括当前行）
    result['volume_trend_5'] = rolling_slope(df['volume'], 6)
    result['volume_trend_10'] = rolling_slope(df['volume'], 11)

    # 计算价格趋势（5日）
    result['price_trend_5'] = rolling_slope(df['close'], 6)

    # 价量背离：价格和成交量趋势方向相反
    result['price_volume_divergence'] = (
            (result['price_trend_5'] * result['volume_trend_5']) < 0
    ).astype(int)

    # 成交量一致性（5日变异系数）
    def rolling_cv(series, window):
        cv = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            window_data = series.iloc[i - window + 1:i + 1]
            mean_val = window_data.mean()
            if mean_val > 0:
                cv[i] = window_data.std() / mean_val
        return cv

    result['volume_consistency'] = rolling_cv(df['volume'], 6)

    # 填充NaN值
    result = result.fillna(0)

    return result

def optimized_rsi_features(df):
    """优化的RSI特征计算"""
    # RSI动量
    df['rsi_momentum'] = df['rsi_14'].diff()

    # 计算一阶和二阶差分
    df['rsi_diff1'] = df['rsi_14'].diff(1)  # 当前 - 前1
    df['rsi_diff2'] = df['rsi_14'].diff(2)  # 当前 - 前2

    # 更复杂的转折点检测
    # 方法1：简单转折（符号变化）
    df['rsi_turning_simple'] = (df['rsi_diff1'].shift(1) * df['rsi_diff1'] < 0).astype(int)

    # 方法2：原代码的转折点
    # 顶部转折：前一期高于前两期，且当前低于前一期
    top = (df['rsi_14'].shift(1) > df['rsi_14'].shift(2)) & (df['rsi_14'] < df['rsi_14'].shift(1))

    # 底部转折：前一期低于前两期，且当前高于前一期
    bottom = (df['rsi_14'].shift(1) < df['rsi_14'].shift(2)) & (df['rsi_14'] > df['rsi_14'].shift(1))

    df['rsi_turning'] = (top | bottom).astype(int)

    # 处理NaN
    df['rsi_momentum'] = df['rsi_momentum'].fillna(0)
    df['rsi_turning'] = df['rsi_turning'].fillna(0)
    df['rsi_turning_simple'] = df['rsi_turning_simple'].fillna(0)

    # 删除临时列
    df = df.drop(['rsi_diff1', 'rsi_diff2'], axis=1)

    return df

def calculate_volume_ma_ratio_vectorized(df, short_window=5, long_window=10):
    """
    向量化计算成交量MA比率

    参数:
    df: 包含'volume'列的DataFrame
    short_window: 短期窗口，默认5天
    long_window: 长期窗口，默认10天
    """
    # 计算移动平均（不包括当天）
    # 使用shift(1)排除当前行，与原逻辑一致
    volume_ma_short = df['volume'].rolling(window=short_window, min_periods=short_window).mean().shift(1)
    volume_ma_long = df['volume'].rolling(window=long_window, min_periods=long_window).mean().shift(1)

    # 计算比率
    ratio = volume_ma_short / volume_ma_long

    # 处理分母为0的情况
    ratio = np.where(volume_ma_long > 0, ratio, 1.0)

    # 填充NaN值（数据不足时）
    # 前long_window行设为1（因为需要long_window个数据）
    ratio = np.where(pd.isna(ratio), 1.0, ratio)

    return ratio

def multi_window_volatility(df, windows=model_config.WINDOWS_VOLATILITY):
    """
    计算多个窗口的波动率
    """
    # 计算对数收益率
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 计算不同窗口的波动率
    for window in windows:
        vol_col = f'volatility_{window}d'

        # 日波动率
        daily_vol = df['log_return'].rolling(window=window, min_periods=window).std()

        # 年化
        df[vol_col] = daily_vol * np.sqrt(252)

        # 填充NaN
        df[vol_col] = df[vol_col].fillna(0)

    # 删除临时列
    if 'log_return' in df.columns:
        result = df.drop('log_return', axis=1)

    return result

def fast_obv_trend(df, window=5):
    """
    高效向量化计算OBV趋势

    使用公式：斜率 = [n*Σ(xy) - Σx*Σy] / [n*Σ(x²) - (Σx)²]
    其中x=0,1,2,...,window
    """
    # 预计算x的值
    n = window + 1  # 窗口大小（包括当前行）
    x = np.arange(n)

    # 预计算常数
    sum_x = np.sum(x)
    sum_x2 = np.sum(x ** 2)
    denom = n * sum_x2 - sum_x ** 2

    if denom == 0:
        df['obv_trend'] = 0.0
        return df

    # 计算每个窗口的斜率
    obv_values = df['obv'].values
    trend = np.zeros(len(df))

    for i in range(n - 1, len(df)):
        # 取窗口数据
        window_obv = obv_values[i - n + 1:i + 1]

        # 计算Σy和Σxy
        sum_y = np.sum(window_obv)
        sum_xy = np.sum(x * window_obv)

        # 计算斜率
        slope = (n * sum_xy - sum_x * sum_y) / denom
        trend[i] = slope

    df['obv_trend'] = trend
    return df

def calculate_macd_percentile_vectorized(df, window=100):
    """
    向量化计算MACD百分位数特征

    参数:
    df: 包含'macd'列的DataFrame
    window: 窗口大小，默认为100天
    """
    df = df.copy()

    # 初始化结果列
    df['macd_percentile'] = 50.0  # 默认值50

    # 计算每个位置的百分位数
    for i in range(window, len(df)):
        historical_macd = df['macd'].iloc[i - window:i]  # 不包括当前行
        current_macd = df['macd'].iloc[i]

        # 计算百分位数
        percentile = percentileofscore(historical_macd, current_macd, kind='rank')
        df.loc[df.index[i], 'macd_percentile'] = percentile

    return df['macd_percentile']

def vectorized_support_resistance(df, window=20):
    """
    向量化计算支撑阻力特征

    参数:
    df: 包含 'low', 'high', 'close' 的 DataFrame
    window: 窗口大小，默认为20天
    """
    # 初始化结果列
    df = df.copy()

    # 计算过去N天的低点和高点（包括当天）
    support = df['low'].rolling(window=window, min_periods=1).min()
    resistance = df['high'].rolling(window=window, min_periods=1).max()

    # 计算距离（百分比形式）
    # 使用 np.where 处理 close <= 0 的情况
    df['distance_to_support'] = np.where(
        df['close'] > 0,
        (df['close'] - support) / df['close'],
        0
    )

    df['distance_to_resistance'] = np.where(
        df['close'] > 0,
        (resistance - df['close']) / df['close'],
        0
    )

    # 将前window行设为0（与原逻辑一致）
    df.iloc[:window, df.columns.get_loc('distance_to_support')] = 0
    df.iloc[:window, df.columns.get_loc('distance_to_resistance')] = 0

    return df
# 检测锤子线形态
def detect_hammer_pattern_single(kline):
    """检测锤子线形态（单根K线）"""
    try:
        body_size = abs(kline['close'] - kline['open'])
        lower_shadow = min(kline['open'], kline['close']) - kline['low']
        upper_shadow = kline['high'] - max(kline['open'], kline['close'])
        total_range = kline['high'] - kline['low']

        if total_range == 0:
            return 0

        # 锤子线形态条件
        is_hammer_shape = (lower_shadow >= 2 * body_size and
                          upper_shadow <= body_size * 0.5 and
                          lower_shadow >= total_range * 0.6)

        # 还需要检查是否处于下跌趋势（这里只是形态检测，趋势应该在外部判断）
        return 1 if is_hammer_shape else 0
    except Exception as e:
        logger.debug(f"检测锤子线形态时出错: {e}")
        return 0
#  检测十字星形态
def detect_doji_pattern_single(kline):
    """检测十字星形态（单根K线）"""
    try:
        body_size = abs(kline['close'] - kline['open'])
        total_range = kline['high'] - kline['low']

        if total_range == 0:
            return 0

        # 十字星条件：实体很小
        is_doji = body_size < total_range * 0.1
        return 1 if is_doji else 0
    except Exception as e:
        logger.debug(f"检测十字星形态时出错: {e}")
        return 0

def fast_engulfing_detection(df):
    """快速向量化吞噬形态检测"""
    # 获取前一根K线数据
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)

    # 判断阴阳线
    prev_is_bearish = prev_close < prev_open
    prev_is_bullish = prev_close > prev_open
    curr_is_bullish = df['close'] > df['open']
    curr_is_bearish = df['close'] < df['open']

    # 看涨吞噬
    bullish_engulfing = (
            prev_is_bearish &
            curr_is_bullish &
            (df['open'] <= prev_close) &
            (df['close'] >= prev_open)
    )

    # 看跌吞噬
    bearish_engulfing = (
            prev_is_bullish &
            curr_is_bearish &
            (df['open'] >= prev_close) &
            (df['close'] <= prev_open)
    )

    # 合并结果
    engulfing = (bullish_engulfing | bearish_engulfing).astype(int)

    # 第一行设为0
    engulfing.iloc[0] = 0

    return engulfing

class FeaturePipeline:
    """
    单类版本 Feature Pipeline
    - 不插件化
    - 不反射
    - 不动态注册
    - 但结构清晰、可维护
    """

    def __init__(
        self,divergence_detector, full_stocks_inner
    ):
        self.divergence_detector = divergence_detector
        self.full_stocks_data =full_stocks_inner

    def _calculate_future_return(self, full_data, signal_date, periods=model_config.RETURN_PERIODS):
        """
        计算未来收益（用于制作标签）
        包含三种收益计算方式：
        1. 原版：期间内超过5%用最高价，否则用期末收盘价
        2. 止损版：期间内先达到5%止盈或3%止损（时间优先）
        3. 两种收益的真实卖出日期
        """
        returns = {}

        try:
            # 确保signal_date是datetime类型
            if not isinstance(signal_date, pd.Timestamp):
                signal_date = pd.to_datetime(signal_date)

            # 确保是按日期排序
            full_data.sort_index(ascending=True)

            # 找到信号日期的位置
            signal_idx = full_data.index.get_loc(signal_date)
            signal_price = full_data['close'].iat[signal_idx]  # 信号日的收盘价作为目标买入价

            # 检查第二天是否能以信号价格买入
            if signal_idx + 1 >= len(full_data):
                # 没有第二天数据，返回NaN
                for period in periods:
                    returns[f'future_return_{period}d'] = np.nan
                    returns[f'future_sell_date_{period}d'] = np.nan
                    returns[f'stop_loss_return_{period}d'] = np.nan
                    returns[f'stop_loss_sell_date_{period}d'] = np.nan
                return returns

            next_day_low = full_data['low'].iat[signal_idx + 1]

            # 如果第二天最低价高于信号价格，无法买入，返回NaN
            if next_day_low > signal_price:
                for period in periods:
                    returns[f'future_return_{period}d'] = np.nan
                    returns[f'future_sell_date_{period}d'] = np.nan
                    returns[f'stop_loss_return_{period}d'] = np.nan
                    returns[f'stop_loss_sell_date_{period}d'] = np.nan
                return returns

            # 能够买入，使用信号价格作为买入价格
            buy_price = signal_price

            for period in periods:
                # 检查是否有足够的数据（到买入后第period天）
                if signal_idx + 1 + period < len(full_data):
                    # 获取未来period天的数据（从买入后第一天到第period天）
                    future_data = full_data.iloc[signal_idx + 2: signal_idx + 1 + period + 1]

                    if len(future_data) > 0:
                        # 原版收益计算和卖出日期
                        future_return, future_sell_date = self._calculate_original_return(
                            full_data, signal_idx, buy_price, period
                        )
                        returns[f'future_return_{period}d'] = future_return
                        returns[f'future_sell_date_{period}d'] = future_sell_date

                        # 止损版收益计算和卖出日期
                        stop_loss_return, stop_loss_sell_date = self._calculate_stop_loss_return(
                            full_data, signal_idx, buy_price, period
                        )
                        returns[f'stop_loss_return_{period}d'] = stop_loss_return
                        returns[f'stop_loss_sell_date_{period}d'] = stop_loss_sell_date

                    else:
                        returns[f'future_return_{period}d'] = np.nan
                        returns[f'future_sell_date_{period}d'] = np.nan
                        returns[f'stop_loss_return_{period}d'] = np.nan
                        returns[f'stop_loss_sell_date_{period}d'] = np.nan
                else:
                    returns[f'future_return_{period}d'] = np.nan
                    returns[f'future_sell_date_{period}d'] = np.nan
                    returns[f'stop_loss_return_{period}d'] = np.nan
                    returns[f'stop_loss_sell_date_{period}d'] = np.nan
                    logger.debug(f"数据不足，无法计算 {period} 天后的收益")

        except Exception as e:
            logger.error(f"计算未来收益时出错: {e}")
            for period in periods:
                returns[f'future_return_{period}d'] = np.nan
                returns[f'future_sell_date_{period}d'] = np.nan
                returns[f'stop_loss_return_{period}d'] = np.nan
                returns[f'stop_loss_sell_date_{period}d'] = np.nan

        return returns

    def _calculate_original_return(self, full_data, signal_idx, buy_price, period):
        """
        计算原版收益和卖出日期 - 修正版
        规则：期间内如果价格达到5%止盈点，就以5%收益卖出；否则用期末收盘价
        """
        try:
            buy_day_idx = signal_idx + 1  # 买入日
            start_idx = buy_day_idx + 1  # 买入后第一天
            end_idx = buy_day_idx + period  # 买入后第period天

            if end_idx >= len(full_data):
                return np.nan, np.nan

            future_data = full_data.iloc[start_idx:end_idx + 1]

            if len(future_data) == 0:
                return np.nan, np.nan

            target_price = buy_price * model_config.EXPECTED_PROFIT

            # 检查是否在期间内达到目标价 - 找到第一次达到的日期
            target_hit_mask = future_data['high'] >= target_price
            if target_hit_mask.any():
                # 找到第一次达到目标价的日期
                first_target_date = future_data[target_hit_mask].index[0]
                # 以目标价卖出，收益为5%
                future_return = (target_price - buy_price) / buy_price
                return future_return, first_target_date
            else:
                # 使用期末收盘价计算收益
                future_price = full_data['close'].iat[end_idx]
                future_return = (future_price - buy_price) / buy_price
                future_sell_date = full_data.index[end_idx]
                return future_return, future_sell_date

        except Exception as e:
            logger.error(f"计算原版收益时出错: {e}")
            return np.nan, np.nan

    def _calculate_stop_loss_return(self, full_data, signal_idx, buy_price, period):
        """
        计算止损方式的收益和真实卖出日期 - 修正版
        """
        try:
            buy_day_idx = signal_idx + 1  # 买入日
            end_idx = buy_day_idx + period  # 买入后第period天

            if end_idx >= len(full_data):
                return np.nan, np.nan

            # 遍历每一天，检查是否触发止损或止盈
            for days_after_buy in range(1, period + 1):
                current_idx = buy_day_idx + days_after_buy
                if current_idx >= len(full_data):
                    break

                current_data = full_data.iloc[current_idx]
                current_date = full_data.index[current_idx]
                current_open = current_data['open']
                current_low = current_data['low']
                current_high = current_data['high']

                # 1. 检查开盘止损
                if current_open <= buy_price * model_config.EXPECTED_LOSS:
                    sell_price = current_open
                    stop_loss_return = (sell_price - buy_price) / buy_price
                    return stop_loss_return, current_date

                # 2. 检查盘中止损
                if current_low <= buy_price * model_config.EXPECTED_LOSS:
                    sell_price = buy_price * model_config.EXPECTED_LOSS
                    stop_loss_return = (sell_price - buy_price) / buy_price
                    return stop_loss_return, current_date

                # 3. 检查止盈
                if current_high >= buy_price * model_config.EXPECTED_PROFIT:
                    sell_price = buy_price * model_config.EXPECTED_PROFIT
                    stop_loss_return = (sell_price - buy_price) / buy_price
                    return stop_loss_return, current_date

            # 如果没有触发止损或止盈，则在期末卖出
            final_price = full_data['close'].iloc[end_idx]
            final_date = full_data.index[end_idx]
            stop_loss_return = (final_price - buy_price) / buy_price
            return stop_loss_return, final_date

        except Exception as e:
            logger.error(f"计算止损收益时出错: {e}")
            return np.nan, np.nan

    def enrich(self, all_2_target_day_df, df_sh, df_sz):
        try:

            # 1. 全量数据的特征计算
            logger.info(f"开始全量数据的特征计算....")
            # 这里使用feature_used_df（最后100行），避免计算太久特征
            df_sorted = all_2_target_day_df.sort_values('timestamp')
            unique_dates = sorted(df_sorted['timestamp'].unique())
            # 取最后100个不重复的日期
            if len(unique_dates) >= model_config.FEATURE_NEED_MAX_DAYS:
                last_dates = unique_dates[-model_config.FEATURE_NEED_MAX_DAYS:]
            else:
                last_dates = unique_dates  # 如果总天数不足100天，则取全部
            feature_used_df = df_sorted[df_sorted['timestamp'].isin(last_dates)]
            feature_used_df = self._calculate_basic_technical_features(feature_used_df)
            feature_used_df = self._calculate_advance_technical_features(feature_used_df)
            feature_used_df = self._generate_alpha_features(feature_used_df)
            feature_used_df = self.generate_structure_features(feature_used_df)
            feature_used_df = self.generate_lag_features(feature_used_df)
            all_2_target_day_df = self._calculate_basic_technical_features(all_2_target_day_df) # 为计算背离所用的MACD等相关指标
            logger.info(f"全量数据的特征计算完成")

            # 2. 取出预测日期的所有数据
            target_date = feature_used_df['timestamp'].dt.date.max()
            target_df = feature_used_df[feature_used_df['timestamp'].dt.date == target_date]

            # 3. 将大盘特征合并到目标数据（向量化操作，不需要循环）
            logger.info(f"{target_date} 开始大盘特征计算....")
            market_features_df = self._calculate_market_features(
                target_date,
                df_sh=df_sh,
                df_sz=df_sz,
            )
            target_df = target_df.merge(
                market_features_df,
                left_on='timestamp',
                right_index=True,
                how='left'
            )
            logger.info(f"{target_date} 大盘特征计算完成")

            # 4、丰富目标日期的DF特征和背离点提取
            logger.info(f"{target_date} 丰富目标日期的DF特征和背离点提取......")
            all_divergence_points = pd.DataFrame()
            enriched_points = pd.DataFrame()
            process_id = 0
            report_interval = 10
            for _, row in target_df.iterrows():
                symbol = row['symbol']
                enriched_point = row.to_dict()
                data_with_indicators = all_2_target_day_df[all_2_target_day_df['symbol'] == symbol].set_index('timestamp')
                enriched_points = pd.concat([enriched_points, pd.DataFrame(enriched_point, index=[0])], ignore_index=True)
                divergence_points_df = self.divergence_detector.detect_daily_divergence(data_with_indicators,symbol,target_date)
                if len(divergence_points_df) > 0:
                    all_divergence_points = pd.concat([all_divergence_points, divergence_points_df],ignore_index=True)
                process_id += 1
                # 阶段性报告进度
                if (process_id + 1) % report_interval == 0:
                    logger.info(f"{target_date.strftime('%Y%m%d')}丰富特征与计算背离:: {process_id + 1}/{len(target_df)}")

            if all_divergence_points.empty:
                logger.warning(f"{target_date.strftime('%Y%m%d')}无背离数据")
                return None
            logger.info(f"{target_date} 丰富目标日期的DF特征和背离点提取完成")

            # 5. 交叉特征添加
            logger.info(f"{target_date} 交叉特征添加......")
            enriched_points = self._calculate_cross_features(enriched_points)
            enriched_points['is_quick_divergence'] = 1 if enriched_points.get('formation_period', 10) <= 3 else 0
            logger.info(f"{target_date} 交叉特征添加完成")

            # 6. 背离点与特征INNER JOIN
            target_divergence_df = enriched_points.merge(
                all_divergence_points,
                on=['symbol', 'timestamp'],  # 两边都有 symbol timestamp 列
                how='inner'
            )

            # 7. 剔除第二天无法买入的背离点
            target_divergence_df = self._filter_valid_rows_apply(target_divergence_df)

            if target_divergence_df.empty:
                logger.warning(f"{target_date.strftime('%Y%m%d')}无有效背离数据")
                return

            # 8.计算未来价格变化（用于标签）- 这里使用完整数据是允许的，因为只是制作标签
            target_divergence_df = target_divergence_df.apply(
                lambda row: self._calculate_and_assign(row, row['symbol']),
                axis=1
            )

            target_divergence_df['divergence_amount']  = len(target_divergence_df)
            self.save_to_csv(target_divergence_df, DAILY_FEATURE_DIR / f"realistic_features_{target_date.strftime('%Y%m%d')}.csv")

            return target_divergence_df
        except Exception as e:
            logger.exception(f"特征生成失败: {e}")
            return None


    def _filter_valid_rows_apply(self, target_divergence_df):
        """
        剔除第二天无法买入的背离点
        """

        def is_valid_row(row):
            symbol = row['symbol']
            timestamp = row['timestamp']
            price = row['close_current']

            if symbol not in self.full_stocks_data:
                logger.error(f"{symbol} 不在全量股票列表里")
                return False

            df = self.full_stocks_data[symbol]

            if timestamp not in df.index:
                logger.error(f"{symbol}在{timestamp} 没数据")
                return False

            pos = df.index.get_loc(timestamp)

            # 如果是最后一天，仍然有效（可用于当天预测）
            if pos + 1 >= len(df):
                return True

            next_day_low = df['low'].iat[pos + 1]
            return next_day_low <= price

        mask = target_divergence_df.apply(is_valid_row, axis=1)
        return target_divergence_df[mask].reset_index(drop=True)

    def _calculate_and_assign(self, row, symbol):
        """
        计算未来价格变化（用于标签）- 这里使用完整数据是允许的，因为只是制作标签
        """
        return_periods = model_config.RETURN_PERIODS
        future_price_change = self._calculate_future_return(
            self.full_stocks_data.get(symbol),
            row['timestamp'],
            return_periods
        )

        for period in return_periods:
            row[f'future_return_{period}d'] = future_price_change.get(
                f'future_return_{period}d', np.nan
            )
            row[f'future_sell_date_{period}d'] = future_price_change.get(
                f'future_sell_date_{period}d', np.nan
            )
            row[f'stop_loss_return_{period}d'] = future_price_change.get(
                f'stop_loss_return_{period}d', np.nan
            )
            row[f'stop_loss_sell_date_{period}d'] = future_price_change.get(
                f'stop_loss_sell_date_{period}d', np.nan
            )
        return row

    def save_to_csv(self, df, file_path):
        """
        保存DataFrame为CSV文件

        参数:
            df: 要保存的DataFrame
            file_path: 保存路径，可以是文件名或完整路径
        """
        try:
            # 创建目录（如果不存在）
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"创建目录: {directory}")

            # 保存DataFrame
            df.to_csv(file_path, index=True if df.index.name else False)
            logger.info(f"数据已保存到: {file_path}")
            logger.info(f"数据形状: {df.shape}, 文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")

        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            raise

    # =========================
    # 基础特征
    # =========================
    def _generate_alpha_features(self, df):
        """
        输入: 包含基础行情数据的 DataFrame
        输出: 增加了高阶特征的 DataFrame
        """

        # 确保数据足够长
        if len(df) < 100:
            logger.warning(f"数据长度不足100，无法计算技术指标")
            return None


        # ------------------------------------------------------
        # 0. 准备工作：防止除零报错的安全气囊
        # ------------------------------------------------------
        EPSILON = 1e-9

        # 确保数据按 股票+时间 排序 (防止 rolling 计算错误)
        if 'symbol' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

        logger.debug("开始计算特征...")

        # 预计算基础变量
        high_low_range = df['high'] - df['low']
        close_open_range = df['close'] - df['open']

        # 计算涨跌幅 (如果没有现成的)
        # 这里的 'symbol' 代表股票代码列名，如果是单只股票数据则不需要 groupby
        df['pct_change'] = df.groupby('symbol')['close'].pct_change()

        # ------------------------------------------------------
        # 1. K线微观结构特征 (解决“方向感”问题)
        # ------------------------------------------------------
        logger.debug("1. 计算 K线微观结构...")

        # [CLV] 收盘位置系数: 1代表收在最高，0代表收在最低
        # 强多头信号
        df['clv'] = (df['close'] - df['low']) / (high_low_range + EPSILON)

        # [Upper Shadow] 上影线比例: 越接近1，抛压越重
        # 过滤假突破
        # 逻辑: (最高价 - 实体顶部) / 总波动
        entity_top = df[['open', 'close']].max(axis=1)
        df['upper_shadow_ratio'] = (df['high'] - entity_top) / (high_low_range + EPSILON)

        # [Body Strength] 实体力度: 绝对值越大，趋势越强
        df['body_strength'] = close_open_range / (high_low_range + EPSILON)

        # ------------------------------------------------------
        # 2. 横截面博弈特征 (解决“相对强弱”问题) - 【最重要】
        # ------------------------------------------------------
        logger.debug("2. 计算横截面排名 (可能需要几秒钟)...")

        # 只有当数据包含多只股票时，这部分才有效
        if 'timestamp' in df.columns and df['timestamp'].nunique() > 1:
            # [Rank Return] 当日涨幅排名 (0.0 ~ 1.0)
            # 0.99 意味着你是当天涨幅前 1% 的靓仔
            df['rank_return'] = df.groupby('timestamp')['pct_change'].rank(pct=True)

            # [Rank Volume] 当日成交量排名 (寻找焦点股)
            # 如果你有 volume_ratio (量比)，用 volume_ratio 排名效果更好
            target_vol_col = 'hs_volume_ratio' if 'hs_volume_ratio' in df.columns else 'volume'
            df['rank_volume'] = df.groupby('timestamp')[target_vol_col].rank(pct=True)
        else:
            logger.warning("警告: 未检测到日期列或只有单只股票，跳过横截面排名计算。")

        # ------------------------------------------------------
        # 3. 量价配合特征 (解决“有量无价”问题)
        # ------------------------------------------------------
        logger.debug("3. 计算量价互动...")

        # [Signed Volume] 带符号的成交量强度
        # 区分“放量涨”和“放量跌”
        # 如果你有 volume_ratio，把下面的 'volume' 换成 'volume_ratio'
        vol_col = 'hs_volume_ratio' if 'hs_volume_ratio' in df.columns else 'volume'
        df['signed_vol_strength'] = df[vol_col] * np.sign(close_open_range)

        # [Price-Volume Correlation] 量价相关性 (过去10天)
        # 正值高 = 越涨越放量 (健康)
        # 负值高 = 越跌越放量 (恐慌)
        # 注意：计算量大，如果太慢可以调小窗口
        # df['pv_corr_10'] = df.groupby('symbol').apply(
        #     lambda x: x['close'].rolling(10).corr(x[vol_col])
        # ).reset_index(0, drop=True)
        df['pv_corr_10'] = ( #性能优化
            df.groupby('symbol')
            .apply(lambda x: x['close'].rolling(10).corr(x[vol_col]))
            .reset_index(level=0, drop=True)
        )

        # ------------------------------------------------------
        # 4. 趋势与波动特征 (解决“爆发点”问题)
        # ------------------------------------------------------
        logger.debug("4. 计算趋势与波动...")

        # [Distance to High] 距离60日新高的距离
        # 接近 1.0 说明即将突破
        rolling_max_60 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(60).max())
        df['dist_to_high_60'] = df['close'] / (rolling_max_60 + EPSILON)

        # [Volatility Divergence] 波动率乖离
        # 今天的波动率 / 过去60天平均波动率
        # 大于 1.5 说明今天波动异常放大，可能有大事发生
        # 使用 5 日标准差作为短期波动率
        volatility_short = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(5).std()
        )
        volatility_long = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(60).std()
        )
        volatility_long_mean = volatility_long.mean()  # 这里的rolling是对 series 做
        # 修正: groupby transform 出来的是 Series，直接 rolling 可能会错位，建议再次 groupby 或者简化处理
        # 为了代码稳定性，这里简化处理：
        df['vol_divergence'] = volatility_short / (volatility_long_mean + EPSILON)

        # ------------------------------------------------------
        # 5. 清洗与收尾
        # ------------------------------------------------------
        # 刚才的 rolling 计算会产生 NaN (前几行)，需要处理
        # 这里的 fillna(0) 是为了跑通，建议实际训练时 dropna() 掉最前面几十行
        df.fillna(0, inplace=True)
        if 'macd_percentile' in df.columns:
            # 或者对无效数据进行特殊处理
            mask = df['macd_percentile'] == 50
            df.loc[mask, 'macd_percentile'] = np.nan
        logger.debug("特征计算完成！")
        return df

    # =========================
    # 基础技术指标
    # =========================
    def _calculate_basic_technical_features(self, df):
        """
        使用历史数据计算技术指标（无未来函数）
        """
        try:
            # 确保数据足够长
            if len(df) < 100:
                logger.warning(f"数据长度不足100，无法计算技术指标")
                return None

            # MACD指标
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

            # RSI指标
            df['rsi_6'] = talib.RSI(df['close'], timeperiod=6)
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_24'] = talib.RSI(df['close'], timeperiod=24)

            # 移动平均线
            df['ma_5'] = talib.MA(df['close'], timeperiod=5)
            df['ma_20'] = talib.MA(df['close'], timeperiod=20)
            df['ma_60'] = talib.MA(df['close'], timeperiod=60)

            # 布林带
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])

            # 成交量指标
            df['volume_ma_20'] = talib.MA(df['volume'], timeperiod=20)
            df['obv'] = talib.OBV(df['close'], df['volume'])

            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])

            # 随机指标
            df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])

            logger.debug(f"技术指标计算完成，数据形状: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return None

    # =========================
    # 市场技术指标
    # =========================
    def _calculate_market_features(self, timestamp, df_sh=None, df_sz=None):
        """
        计算大盘特征 - 计算恒生指数、国企指数、红筹指数特征，并合并
        """
        all_features = {}

        try:
            # 确保 timestamp 是 datetime 类型
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)

            # 计算各个指数的特征
            features_list = []

            # 计算恒生指数特征
            if df_sh is not None:
                sh_features = self._calculate_single_index_features(timestamp, df_sh, 'sh')
                features_list.append(sh_features)

            # 计算国企指数特征
            if df_sz is not None:
                sz_features = self._calculate_single_index_features(timestamp, df_sz, 'sz')
                features_list.append(sz_features)

            # 如果没有传入任何数据，返回默认值
            if len(features_list) == 0:
                logger.error("未传入任何指数数据")
                return self._get_default_combined_features(timestamp)

            # 合并所有特征
            for feature_dict in features_list:
                all_features.update(feature_dict)

            # 计算指数间的协同特征
            if df_sh is not None and df_sz is not None:
                # 计算恒生指数和国企指数的协同特征
                if 'sh_price_change' in all_features and 'sz_price_change' in all_features:
                    all_features['sh_sz_sync_direction'] = 1 if all_features['sh_price_change'] * all_features[
                        'sz_price_change'] > 0 else 0
                    all_features['sh_sz_sync_strength'] = abs(
                        all_features['sh_price_change'] - all_features['sz_price_change'])

            # 计算整体市场情绪（基于所有可用指数）
            price_changes = []
            amplitudes = []

            for prefix in ['sh', 'sz']:
                price_key = f'{prefix}_price_change'
                amplitude_key = f'{prefix}_amplitude'

                if price_key in all_features:
                    price_changes.append(all_features[price_key])
                if amplitude_key in all_features:
                    amplitudes.append(all_features[amplitude_key])

            if price_changes and amplitudes:
                avg_price_change = sum(price_changes) / len(price_changes)
                avg_amplitude = sum(amplitudes) / len(amplitudes)

                # 市场情绪特征
                conditions = [
                    (avg_price_change > 0.01) & (avg_amplitude < 0.02),
                    (avg_price_change < -0.01) & (avg_amplitude > 0.03),
                ]
                choices = [2, 0]  # 2: 强势, 0: 弱势
                all_features['market_sentiment'] = int(np.select(conditions, choices, default=1))  # 1: 中性
                all_features['market_avg_change'] = avg_price_change
                all_features['market_avg_amplitude'] = avg_amplitude

                # 市场同步评分
                if len(price_changes) > 1:
                    max_diff = max(price_changes) - min(price_changes)
                    all_features['market_sync_score'] = 1 - min(1.0, max_diff / 0.02)
                else:
                    all_features['market_sync_score'] = 0.5

            logger.debug(f"大盘特征计算完成，共 {len(all_features)} 个特征")

        except Exception as e:
            logger.error(f"计算大盘特征时出错: {e}")
            # 设置默认值
            all_features = self._get_default_combined_features(timestamp)

        # 转换为DataFrame
        features_list = [all_features]
        market_features_df = pd.DataFrame(features_list)
        market_features_df['timestamp'] = timestamp

        # 设置索引为timestamp
        if 'timestamp' in market_features_df.columns:
            market_features_df.set_index('timestamp', inplace=True)

        return market_features_df

    def _calculate_single_index_features(self, timestamp, df_index, prefix):
        """
        计算单个指数的特征
        """
        features = {}

        try:
            # 保证大盘特征只使用信号日及之前的数据
            index_hist = df_index[df_index.index <= timestamp]

            # 找到对应日期的大盘数据
            index_data = index_hist[index_hist.index == timestamp]

            if len(index_data) > 0:
                index_row = index_data.iloc[0]

                # 1. 价格变化特征
                features[f'{prefix}_price_change'] = (index_row['close'] - index_row['open']) / index_row['open']
                features[f'{prefix}_amplitude'] = (index_row['high'] - index_row['low']) / index_row['low']

                # 2. 成交量特征
                if len(index_hist) >= 20:
                    avg_volume = index_hist['volume'].rolling(20).mean().iloc[-1]
                    features[f'{prefix}_volume_ratio'] = index_row['volume'] / avg_volume if avg_volume != 0 else 1
                else:
                    features[f'{prefix}_volume_ratio'] = 1

                # 3. 绝对价格变化
                features[f'{prefix}_price_change_abs'] = abs(features[f'{prefix}_price_change'])
                features[f'{prefix}_price_wave_abs'] = features[f'{prefix}_amplitude']

                # 4. 单个指数情绪特征
                conditions = [
                    (features[f'{prefix}_price_change'] > 0.01) & (features[f'{prefix}_amplitude'] < 0.02),
                    (features[f'{prefix}_price_change'] < -0.01) & (features[f'{prefix}_amplitude'] > 0.03),
                ]
                choices = [2, 0]  # 2: 强势, 0: 弱势
                features[f'{prefix}_sentiment'] = int(np.select(conditions, choices, default=1))  # 1: 中性

                # 5. 成交量信号
                features[f'{prefix}_volume_signal'] = 1 if features[f'{prefix}_volume_ratio'] > 1.2 else 0

            else:
                # 如果找不到对应日期数据，设置为默认值
                default_features = {
                    f'{prefix}_price_change': 0,
                    f'{prefix}_amplitude': 0.02,
                    f'{prefix}_volume_ratio': 1,
                    f'{prefix}_price_change_abs': 0,
                    f'{prefix}_price_wave_abs': 0.02,
                    f'{prefix}_sentiment': 1,
                    f'{prefix}_volume_signal': 0
                }
                features.update(default_features)

        except Exception as e:
            logger.error(f"计算{prefix}指数特征时出错: {e}")
            # 设置默认值
            default_features = {
                f'{prefix}_price_change': 0,
                f'{prefix}_amplitude': 0.02,
                f'{prefix}_volume_ratio': 1,
                f'{prefix}_price_change_abs': 0,
                f'{prefix}_price_wave_abs': 0.02,
                f'{prefix}_sentiment': 1,
                f'{prefix}_volume_signal': 0
            }
            features.update(default_features)

        return features

    def _get_default_combined_features(self, timestamp):
        """
        返回合并特征的默认值
        """
        default_features = {}

        # 为每个可能的指数设置默认特征
        for prefix in ['hs', 'gq', 'hc']:
            default_features.update({
                f'{prefix}_price_change': 0,
                f'{prefix}_amplitude': 0.02,
                f'{prefix}_volume_ratio': 1,
                f'{prefix}_price_change_abs': 0,
                f'{prefix}_price_wave_abs': 0.02,
                f'{prefix}_sentiment': 1,
                f'{prefix}_volume_signal': 0
            })

        # 协同特征的默认值
        default_features.update({
            'hs_gq_sync_direction': 1,
            'hs_gq_sync_strength': 0,
            'hs_hc_sync_direction': 1,
            'hs_hc_sync_strength': 0,
            'gq_hc_sync_direction': 1,
            'gq_hc_sync_strength': 0,
            'market_sentiment': 1,
            'market_avg_change': 0,
            'market_avg_amplitude': 0.02,
            'market_sync_score': 0.5
        })

        return default_features


    # =========================
    # V2版本特征升级
    # =========================
    def robust_zscore(self, series):
        """
        稳健 Z-Score 计算
        公式: (x - median) / (IQR / 1.34896)
        1.34896 是正态分布下 IQR 与标准差的换算系数
        """
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)

        # 防止 IQR 为 0 (例如长时间停牌或一字板)
        if iqr == 0:
            return series - median

        return (series - median) / (iqr / 1.34896)

    def wavelet_denoising(self, series, wavelet='db4', level=1):
        """
        小波去噪改进版本

        参数:
        ----------
        series : array-like
            输入时间序列
        wavelet : str, 默认'db4'
            小波基函数，'db4'在金融时间序列中常用
        level : int, 默认1
            分解层数，通常设为log2(len(series))或经验值

        返回:
        -------
        denoised : ndarray
            去噪后的序列，长度与输入相同
        """
        import numpy as np
        import pywt

        # 1. 输入验证
        series = np.array(series, copy=True)
        if len(series) < 2 ** level:
            raise ValueError(f"序列长度({len(series)})不足以进行{level}层小波分解")

        # 2. 处理NaN值（可选策略）
        if np.any(np.isnan(series)):
            # 方法1：使用线性插值填充NaN
            nan_mask = np.isnan(series)
            if np.all(nan_mask):
                return series.copy()

            # 简单前向填充处理（根据实际需求选择）
            series_filled = series.copy()
            series_filled[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                np.where(~nan_mask)[0],
                series_filled[~nan_mask]
            )
            series = series_filled

        # 3. 小波分解
        coeff = pywt.wavedec(series, wavelet, mode='per', level=level)

        # 4. 阈值计算（改进版）
        # 使用最后一级细节系数的中位数绝对偏差估计噪声标准差
        if len(coeff) > 1 and len(coeff[-1]) > 0:
            detail_coeff = coeff[-1]
            sigma = np.median(np.abs(detail_coeff)) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(series)))
        else:
            # 如果无法计算阈值，返回原始序列
            return series.copy()

        # 5. 阈值处理（可以尝试不同模式）
        threshold_mode = 'soft'  # 'soft', 'hard', 'garrote'

        # 只对细节系数（高频部分）进行阈值处理，保留近似系数（低频部分）
        coeff_thresh = []
        coeff_thresh.append(coeff[0])  # 保留近似系数

        for i in range(1, len(coeff)):
            coeff_i = coeff[i]
            # 应用阈值
            coeff_thresh.append(pywt.threshold(coeff_i, value=uthresh, mode=threshold_mode))

        # 6. 信号重构
        try:
            denoised = pywt.waverec(coeff_thresh, wavelet, mode='per')
        except Exception as e:
            logger.error(f"小波重构失败: {e}")
            return series.copy()

        # 7. 确保输出长度与输入一致
        if len(denoised) != len(series):
            # 裁剪或填充以匹配原始长度
            min_len = min(len(denoised), len(series))
            result = np.full_like(series, np.nan)
            result[:min_len] = denoised[:min_len]
        else:
            result = denoised

        return result

    def get_weights_ffd(self, d, thres, lim):
        """生成分数阶差分的权重系数"""
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1
            if k >= lim:
                break
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series, d, thres=1e-5, lim=10000):
        """
        固定窗口分数阶差分
        :param d: 差分阶数，如 0.4
        :param thres: 权重截断阈值
        """
        # 1. 配置权重
        w = self.get_weights_ffd(d, thres, lim)
        width = len(w) - 1

        # 2. 向量化应用权重
        output = []
        # 注意：这里为了演示简单用了循环，生产环境建议用 stride_tricks 或卷积
        series_val = series.values
        for i in range(width, len(series_val)):
            # 窗口内的数据 * 权重
            window0 = series_val[i - width: i + 1]
            output.append(np.dot(window0, w)[0])

        return pd.Series(output, index=series.index[width:])

    def generate_structure_features(self, df):
        """
        结构化高阶特征 (Volatility, Liquidity, Structure)
        """
        # 确保数据足够长
        if len(df) < 100:
            logger.warning(f"数据长度不足100，无法计算技术指标")
            return None

        eps = EPS

        # 1. Garman-Klass 波动率 (相比普通std更精准的波动率)
        # 捕捉日内震荡幅度，往往在大变盘前 GK Vol 会先缩小后放大
        log_hl = np.log(df['high'] / (df['low'] + eps))
        log_co = np.log(df['close'] / (df['open'] + eps))
        df['vol_gk'] = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)
        # 归一化：GK波动率 / 历史平均GK波动率
        df['vol_gk_ratio'] = df['vol_gk'] /df['vol_gk'].rolling(20).mean()

        # 2. Amihud 非流动性因子 (Abs(Ret) / Vol)
        # 衡量单位成交量带来的价格变化。数值大代表流动性枯竭(顶部或底部特征)
        # 这里用了对数处理防止数值过大
        df['illiq'] = np.log((df['close'].pct_change().abs() / (df['close'] * df['volume'] + eps)) + 1)

        # 3. 考夫曼效率系数 (Efficiency Ratio)
        # 1.0 代表单边趋势，0.0 代表无序震荡
        period = 10
        change = (df['close'] - df['close'].shift(period)).abs()
        volatility = df['close'].diff().abs().rolling(period).sum()
        df['efficiency_ratio'] = change / (volatility + eps)

        # 4. 价格相对强度 (Intraday Strength)
        # 收盘价相对于全天波动的分位。
        # 接近1代表全天强势收高，接近0代表冲高回落或低开低走
        df['intraday_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)

        # 5. 隔夜 vs 日内 动量差异 (Smart Money Proxy)
        # 假设：隔夜代表散户情绪，日内代表主力意图
        # 如果隔夜大涨但日内下跌（Gap Up & Sell），往往是出货
        df['ret_overnight'] = (df['open'] / df['close'].shift(1)) - 1 #隔夜
        df['ret_intraday'] = (df['close'] / df['open']) - 1 #日内
        df['smart_money_diff'] = df['ret_intraday'] - df['ret_overnight']

        # 6 相对强弱（RSRS- Resistance Support Relative Strength)
        df['high_mean_20'] = df['high'].rolling(20).mean()
        df['low_mean_20'] =  df['low'].rolling(20).mean()
        df['support_resistance_ratio'] = df['high_mean_20']  / df['low_mean_20']

        # 基础对数变换，适用于成交量和持仓量
        # 效果：将指数级增长的量级拉平
        df['log_volume'] = np.log1p(df['volume'])


        # Box-cox变换
        # 适用于：波动率换手率等必须为正的指标
        atr_positive = df['atr'].copy()
        atr_positive[(atr_positive <= 0) | atr_positive.isna()] = eps
        df['boxcox_atr'], lmbda = stats.boxcox(atr_positive)

        # 稳健标准化
        df['rsi_robust'] = df.groupby('symbol')['rsi_14'].transform(self.robust_zscore)
        df['macd_robust'] = df.groupby('symbol')['macd'].transform(self.robust_zscore)

        # 小波变换去噪 NOTED 这个特征可能有不一致性问题，导致预测时候的数据泄露（暂时不用）
        df['close_wavelet'] = self.wavelet_denoising(df['close'].values)

        # 分数阶差分 NOTED 这个特征可能有不一致性问题，导致预测时候的数据泄露（暂时不用）
        df['close_d0.4'] = self.frac_diff_ffd(df['close'], d=0.4)

        return df

    def generate_lag_features(self, df):
        """
        为指定特征生成滞后特征

        Parameters:
        -----------
        df : DataFrame
            输入数据
        feature_columns : list, optional
            需要生成滞后特征的特征列表，如果为None则使用默认列表
        """

        # 确保数据足够长
        if len(df) < 100:
            logger.warning(f"数据长度不足100，无法计算技术指标")
            return None

        # ==============================================
        # 新增：基础滞后特征
        # ==============================================
        logger.debug("添加滞后特征... （3, 5, 10, 15, 20, 30天）")
        lag_periods = model_config.LAG_PERIODS

        # 基础价格滞后
        price_lags = lag_periods
        for lag in price_lags:
            df[f'close_lag_{lag}'] = df.groupby('symbol')['close'].shift(lag)
            df[f'open_lag_{lag}'] = df.groupby('symbol')['open'].shift(lag)
            df[f'high_lag_{lag}'] = df.groupby('symbol')['high'].shift(lag)
            df[f'low_lag_{lag}'] = df.groupby('symbol')['low'].shift(lag)

        # 成交量滞后
        volume_lags = lag_periods
        for lag in volume_lags:
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['volume'].shift(lag)

        # 收益率滞后
        df['daily_return'] = df.groupby('symbol')['close'].pct_change()
        return_lags = [1, 2, 3, 5, 10, 20]
        for lag in return_lags:
            df[f'return_lag_{lag}'] = df.groupby('symbol')['daily_return'].shift(lag)

        # 振幅滞后
        df['amplitude'] = (df['high'] - df['low']) / df['low']
        for lag in [1, 3, 5]:
            df[f'amplitude_lag_{lag}'] = df.groupby('symbol')['amplitude'].shift(lag)

        # ==============================================
        # 新增：结构化特征滞后
        # ==============================================

        # Garman-Klass波动率滞后
        for lag in [1, 3, 5, 10]:
            df[f'vol_gk_lag_{lag}'] = df.groupby('symbol')['vol_gk'].shift(lag)
            df[f'vol_gk_ratio_lag_{lag}'] = df.groupby('symbol')['vol_gk_ratio'].shift(lag)

        # Amihud非流动性因子滞后
        for lag in [1, 3, 5]:
            df[f'illiq_lag_{lag}'] = df.groupby('symbol')['illiq'].shift(lag)

        # 效率系数滞后
        for lag in [1, 3, 5]:
            df[f'efficiency_ratio_lag_{lag}'] = df.groupby('symbol')['efficiency_ratio'].shift(lag)

        # 日内强度滞后
        for lag in [1, 2, 3]:
            df[f'intraday_pos_lag_{lag}'] = df.groupby('symbol')['intraday_pos'].shift(lag)

        # 隔夜/日内动量滞后
        for lag in [1, 3, 5]:
            df[f'smart_money_diff_lag_{lag}'] = df.groupby('symbol')['smart_money_diff'].shift(lag)
            df[f'ret_overnight_lag_{lag}'] = df.groupby('symbol')['ret_overnight'].shift(lag)
            df[f'ret_intraday_lag_{lag}'] = df.groupby('symbol')['ret_intraday'].shift(lag)

        # 相对强弱滞后
        for lag in [1, 3, 5]:
            df[f'support_resistance_ratio_lag_{lag}'] = df.groupby('symbol')['support_resistance_ratio'].shift(lag)

        return df

    # =========================
    # 交叉指标
    # =========================
    def _calculate_cross_features(self, final_features):
        ts_col = 'timestamp'
        # 所有数据合并后再添加截面计算逻辑
        return_periods = model_config.RETURN_PERIODS
        label_columns = []
        for period in return_periods:
            label_columns.extend([
                f'future_return_{period}d',
                f'future_sell_date_{period}d',
                f'stop_loss_return_{period}d',
                f'stop_loss_sell_date_{period}d'
            ])
        exclude = {'timestamp', 'symbol', 'label','open','close','high','low'}
        exclude.update(label_columns)
        features = [c for c in final_features.columns if
                c not in exclude and final_features[c].dtype in (np.float64, np.int64)]
        logger.debug(f"添加全局截面计算逻辑的特征 = {list(features)}")
        final_features['cs_n'] = final_features.groupby(ts_col)[ts_col].transform('count')
        for f in features:
            final_features[f + '_rankpct'] = final_features.groupby(ts_col)[f].rank(pct=True)
            final_features[f + '_z'] = final_features.groupby(ts_col)[f].transform(lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-9))
        return final_features

    # ============辅助计算函数=============
    def _detect_hammer_pattern_batch(self, df):
        """批量检测锤子线形态"""
        # 简化版本，您可以根据实际逻辑实现
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

        # 锤子线条件：下影线至少是实体的2倍，上影线很短，实体较小
        is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body * 0.3) & (body / df['close'] < 0.03)
        return is_hammer.astype(int)

    def _detect_doji_pattern_batch(self, df):
        """批量检测十字星形态"""
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']

        # 十字星条件：实体非常小（小于总范围的10%）
        is_doji = (body / total_range < 0.1) & (total_range > 0)
        return is_doji.astype(int)

    def _detect_engulfing_pattern_single(self, prev_row, curr_row):
        """
        检测吞没形态（单个K线对）
        """
        try:
            # 前一日的实体大小和颜色
            prev_body = abs(prev_row['close'] - prev_row['open'])
            prev_is_bullish = prev_row['close'] > prev_row['open']

            # 当日的实体大小和颜色
            curr_body = abs(curr_row['close'] - curr_row['open'])
            curr_is_bullish = curr_row['close'] > curr_row['open']

            # 吞没形态条件：
            # 1. 颜色相反（前一阴，后一阳或前一阳，后一阴）
            # 2. 后一根K线的实体完全包含前一根K线的实体
            if prev_is_bullish != curr_is_bullish:
                if curr_is_bullish:  # 看涨吞没
                    if (curr_row['open'] < prev_row['close'] and
                            curr_row['close'] > prev_row['open']):
                        return 1
                else:  # 看跌吞没
                    if (curr_row['open'] > prev_row['close'] and
                            curr_row['close'] < prev_row['open']):
                        return 1

            return 0
        except Exception as e:
            logger.warning(f"检测吞没形态时出错: {e}")
            return 0

    def _calculate_advance_technical_features(self, df):
        """
        替换原有的_calculate_advance_technical_features_batch
        """
        # 1. 均线系统特征
        df['price_vs_ma5'] = df['close'] / df['ma_5'] - 1
        df['price_vs_ma20'] = df['close'] / df['ma_20'] - 1
        df['price_vs_ma60'] = df['close'] / df['ma_60'] - 1
        df['ma_arrangement'] = np.where(
            (df['ma_5'] > df['ma_20']) & (df['ma_20'] > df['ma_60']),
            1,
            np.where(
                (df['ma_5'] < df['ma_20']) & (df['ma_20'] < df['ma_60']),
                -1,
                0
            )
        ) #均线排列

        # 2. 布林带特征
        bb_width = df['bb_upper'] - df['bb_lower']# 计算布林带宽度
        # 计算布林带位置
        df['bb_position'] = np.where(
            bb_width > 0,
            (df['close'] - df['bb_lower']) / bb_width,
            0.5
        )
        # 计算布林带收缩指标（使用向量化操作）
        df['bb_squeeze'] = np.where(
            bb_width / df['close'] < 0.05,  # 布林带宽度小于收盘价的5%
            1,
            0
        )

        # 3. RSI补充
        # RSI超卖信号 (低于30)
        df['rsi_oversold_6'] = np.where(df['rsi_6'] < 30, 1, 0)
        df['rsi_oversold_14'] = np.where(df['rsi_14'] < 30, 1, 0)
        df['rsi_oversold_24'] = np.where(df['rsi_24'] < 30, 1, 0)
        # RSI超买信号 (高于70)
        df['rsi_overbought_6'] = np.where(df['rsi_6'] > 70, 1, 0)
        df['rsi_overbought_14'] = np.where(df['rsi_14'] > 70, 1, 0)
        df['rsi_overbought_24'] = np.where(df['rsi_24'] > 70, 1, 0)

        # 4. MACD 补充
        df['macd_signal_distance'] = df['macd'] - df['macd_signal']
        df['macd_golden_cross'] = np.where(df['macd_signal_distance'] > 0, 1, 0)

        # 5. 成交量特征补充
        df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = np.where(
            df['volume_ma20'] > 0,  # 条件：成交量均线大于0
            df['volume'] / df['volume_ma20'],  # 条件为真时的值
            1  # 条件为假时的值（均量为0时设为1，表示正常
        )
        # 成交量放量信号（成交量比率 > 2.0）
        df['volume_spike'] = np.where(df['volume_ratio'] > 2.0, 1, 0)
        # 成交量萎缩信号（成交量比率 < 0.5）
        df['volume_dryup'] = np.where(df['volume_ratio'] < 0.5, 1, 0)

        # 6. 波动率特征
        df['atr_ratio'] = np.where(
            df['close'] > 0,  # 条件：收盘价大于0
            df['atr'] / df['close'],  # 条件为真时的值
            0  # 条件为假时的值（收盘价<=0时设为0）
        )

        # 7. K线形态特征
        df['hammer_pattern'] = df.apply(detect_hammer_pattern_single) #检测锤子线形态
        df['downtrend'] = df['close'].rolling(5).mean() < df['close'].rolling(10).mean() # 检测下跌趋势（5日均线 < 10日均线）
        df['hammer_signal'] = df['hammer_pattern'] & df['downtrend'] # 锤子线信号：既是锤子形态，又处于下跌趋势
        df['doji_pattern'] = df.apply(detect_doji_pattern_single) # 检测十字星形态

        # 8. 支撑阻力特征
        df = vectorized_support_resistance(df, window=20)

        # 9. 随机指标特征补充
        # 随机指标超卖信号（慢K线 < 20）
        df['stoch_oversold'] = np.where(df['slowk'] < 20, 1, 0)
        # 随机指标超买信号（慢D线 > 80）
        df['stoch_overbought'] = np.where(df['slowd'] > 80, 1, 0)

        # 10. 计算MACD指标在历史分布中的百分位数
        df['macd_percentile'] = calculate_macd_percentile_vectorized(df, window=100)

        # 11. OBV趋势
        df = fast_obv_trend(df, window=5)

        # 12. 历史波动率
        df = multi_window_volatility(df, windows=model_config.WINDOWS_VOLATILITY)

        # 13. 吞噬形态检测
        df['engulfing_pattern'] = fast_engulfing_detection(df)

        # 14. 量级横截面
        df['signed_volume_strength'] = df['volume_ratio'] * (
                (df['close'] - df['open']) / df['open'])
        df['close_vs_high'] = (df['high'] - df['close']) / (
                df['high'] - df['low'])

        # 15. 成交量MA比率
        df['volume_ma_ratio'] = calculate_volume_ma_ratio_vectorized(df, short_window=5, long_window=10)

        # 16. RSI动量和RSI转折点
        df = optimized_rsi_features(df)

        # 17. 成交量趋势指标
        volume_features = calculate_volume_features_vectorized(df)
        df = pd.concat([df, volume_features], axis=1)

        # 18. MACD深度特征增强
        df = macd_features_rolling(df)

        return df


# 降低精度
def optimize_dtypes(df):
    """
    简化版数据类型优化：只降低浮点数精度，保留4位小数
    """
    if df is None or len(df) == 0:
        return df

    # 记录原始内存
    # memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB

    # 只处理浮点数列
    float_cols = df.select_dtypes(include=['float64']).columns

    for col in float_cols:
        try:
            # 保留4位小数后转换为float32
            df[col] = df[col].round(4).astype(np.float32)
        except Exception as e:
            logger.warning(f"无法转换列 {col} 到 float32: {e}")
            # 保持原样

    # memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    # reduction = (memory_before - memory_after) / memory_before * 100

    # logger.debug(f"数据类型优化完成: {memory_before:.2f}MB → {memory_after:.2f}MB, 节省 {reduction:.1f}%")

    return df

#加载目标日期段的股票数据
def load_price_data(directory_path, start_date='2009-01-01', end_date=None):
    """
    加载目录下所有 {symbol}_price_data.pkl 文件到内存中，并按时间范围过滤

    Parameters:
    -----------
    directory_path : str
        数据文件所在目录
    start_date : str or datetime, optional
        开始日期，格式如 '2023-01-01'
    end_date : str or datetime, optional
        结束日期，格式如 '2023-12-31'

    Returns:
    --------
    dict
        以股票代码为键，DataFrame为值的字典
    """
    # 构建文件匹配模式
    pattern = os.path.join(directory_path, "*_price_data.csv")

    # 查找所有匹配的文件
    file_paths = glob.glob(pattern)

    # 存储所有DataFrame的字典
    dataframes = {}

    for file_path in file_paths:
        try:
            # 从文件名提取symbol
            pattern = r'(\d{6}\.[A-Z]{2})_price_data\.csv'
            filename = os.path.basename(file_path)
            match = re.match(pattern, filename)
            if match:
                symbol = match.group(1)
            else:
                logger.error(f"提取symbol出错 filename={filename}")
                continue

            # 加载pkl文件
            df = pd.read_csv(file_path, encoding='gb2312', index_col=0, dtype={'symbol': str})
            # 列名字转换
            df.columns = [ 'open','high','low','close','volume']
            # 只保留需要的列
            columns_to_keep = ['open','high','low','close','volume']  # 指定要保留的列
            df = df[columns_to_keep]

            # 确保值为字符串 非常重要
            df['symbol'] = symbol

            df = optimize_dtypes(df)

            # 确保索引是datetime类型&排序（不排序后续数据全乱 BUGFIXES）
            df = ensure_datetime_index(df)
            df = df.sort_index(ascending=True)

            # 按时间范围过滤数据
            if start_date is not None or end_date is not None:
                # 转换日期参数为datetime格式
                if start_date is not None:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                if end_date is not None:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)

                # 构建时间范围过滤条件
                mask = pd.Series(True, index=df.index)
                if start_date is not None:
                    mask = mask & (df.index >= start_date)
                if end_date is not None:
                    mask = mask & (df.index <= end_date)

                # 应用过滤
                df_filtered = df[mask]

                logger.debug(f"时间范围过滤：{start_date} ~ {end_date if end_date is not None else '最新'}，过滤后行数：{len(df_filtered)}，过滤前行数：{len(df)}")

                # 检查过滤后是否有数据
                if len(df_filtered) == 0:
                    logger.debug(f"股票 {symbol} 在时间范围 {start_date} 到 {end_date} 内没有数据")
                    continue

                df = df_filtered
                logger.debug(f"股票 {symbol}: 时间范围过滤后数据形状: {df.shape}")

            # 不在这里计算技术指标，避免数据泄露
            # 技术指标将在按日期处理时实时计算
            dataframes[symbol] = df
            logger.debug(f"成功加载: {filename}, 数据形状: {df.shape}")

        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")

    logger.debug(f"共加载 {len(dataframes)} 个股票数据文件")

    # 如果没有加载到任何数据，给出警告
    if len(dataframes) == 0:
        logger.warning("没有加载到任何股票数据！请检查时间范围或数据文件")

    return dataframes

def ensure_datetime_index(data):
    """确保数据索引是datetime类型"""
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        try:
            data.index = pd.to_datetime(data.index)
            logger.debug("成功转换索引为datetime类型")
        except Exception as e:
            logger.error(f"转换索引失败: {e}")
            raise
    return data
# 股票字典数据转化为dataframe
def convert_dict_to_dataframe_from_index(stock_dict):
    logger.debug(f"正在合并 {len(stock_dict)} 只股票的数据 (时间在Index)...")
    all_dfs = []

    for symbol, sub_df in stock_dict.items():
        # 1. 复制一份，以免修改原始数据
        temp_df = sub_df.copy()

        # 2. 【关键】把时间索引变成普通列
        # reset_index() 会把原来的 index 变成一列，通常默认列名叫 'index'
        temp_df = temp_df.reset_index()

        # 3. 重命名该列为 'timestamp'，方便后续统一计算
        # 你的 index 名字可能是 None，也可能是 'timestamp' 或 'trade_date'
        # 我们统一把第一列（也就是刚刚 reset 出来的索引列）改名为 'timestamp'
        temp_df.rename(columns={temp_df.columns[0]: 'timestamp'}, inplace=True)

        # 4. 加上股票代码列
        temp_df['symbol'] = symbol

        all_dfs.append(temp_df)

    # 5. 合并
    big_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 确保是时间格式
    big_df['timestamp'] = pd.to_datetime(big_df['timestamp'])
    big_df.sort_values(by=['timestamp', 'symbol'], inplace=True)

    logger.debug(f"合并完成！数据形状: {big_df.shape}")
    return big_df

from src.divergence_detector_v2 import DivergenceDetectorV2


def feature_generator(target_date, divergence_detector=DivergenceDetectorV2()):
    """
    特征生成
    """
    logger.info(f"开始处理{target_date}的特征数据...")

    # 抓取全量数据
    full_stocks_once = get_cached_dataset(FULL_STOCK_DATA_KEY)

    # 过滤预测日期之前的所有数据
    end_date = pd.to_datetime(target_date)
    filtered_stocks_data = {}
    for symbol, df in full_stocks_once.items():
        if df is not None and len(df) > 0:
            # 先过滤
            filtered_df = df[df.index <= end_date]
            # 再判断过滤后的数据是否为空
            if len(filtered_df) > 0:
                filtered_stocks_data[symbol] = filtered_df.copy()
    filtered_stock_data_df = convert_dict_to_dataframe_from_index(filtered_stocks_data)

    # 获取大盘数据指标
    df_sh = filtered_stocks_data.get(df_sh_symbol)
    df_sz = filtered_stocks_data.get(df_sz_symbol)

    # 构建特征工程
    features_pipeline = FeaturePipeline(divergence_detector,full_stocks_once)
    feature_df  = features_pipeline.enrich(filtered_stock_data_df, df_sh, df_sz)
    return feature_df


import concurrent.futures
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor


from datetime import datetime
def process_stocks_batch_parallel_optimized(full_stocks, batch_size=100, max_workers=8, start_date=None, end_date=None):
    """
    优化的多进程处理，避免内存复制

    Args:
        full_stocks: 完整的股票数据
        batch_size: 每批处理的日期数量
        max_workers: 最大进程数（不是线程数）
        start_date: 开始处理的日期 (YYYY-MM-DD)
        end_date: 结束处理的日期 (YYYY-MM-DD)，不指定则处理到最后
    """
    stock_data_df = convert_dict_to_dataframe_from_index(full_stocks)
    unique_timestamps = stock_data_df['timestamp'].sort_values().unique()

    # 1. 根据开始和结束日期过滤
    # 准备日期列表
    target_dates = [date.strftime('%Y-%m-%d') for date in unique_timestamps]

    filtered_dates = []
    for date_str in target_dates:
        try:
            current_date = datetime.strptime(date_str, '%Y-%m-%d')

            # 检查是否在开始日期之后
            if start_date:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                if current_date < start_date_obj:
                    continue

            # 检查是否在结束日期之前
            if end_date:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                if current_date > end_date_obj:
                    continue

            filtered_dates.append(date_str)

        except ValueError:
            logger.warning(f"跳过无效日期格式: {date_str}")

    if not filtered_dates:
        logger.warning("没有符合条件的日期需要处理")
        return

    # 2. 过滤已存在的文件
    dates_to_process = []
    for target_date in filtered_dates:
        # 生成文件名
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        filename = str(DAILY_FEATURE_DIR / f"realistic_features_{date_obj.strftime('%Y%m%d')}.csv")

        # 如果文件不存在，才需要处理
        if not os.path.exists(filename):
            dates_to_process.append(target_date)
        else:
            logger.debug(f"文件已存在，跳过日期: {target_date}")

    if not dates_to_process:
        logger.info("所有日期都已处理完成！")
        return
    logger.info(f"总日期数: {len(dates_to_process)}")
    logger.info (f"批次大小: {batch_size}, 总批次数: {math.ceil(len(dates_to_process) / batch_size)}")


    # 分批处理
    all_errors = []
    total_processed = 0
    for batch_start in range(0, len(dates_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(dates_to_process))
        batch_dates = dates_to_process[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        logger.debug(f"\n处理批次 {batch_num} ({batch_start + 1}-{batch_end})...")

        # 准备批处理参数
        params = []
        for target_date in batch_dates:
            # 只传递symbol列表和日期，让子进程自己加载需要的数据
            params.append(target_date)

        batch_errors = []

        # 使用进程池
        with ProcessPoolExecutor(max_workers=min(max_workers, len(batch_dates))) as executor:
            # 提交批处理任务
            futures = []
            for target_date in params:
                future = executor.submit(
                    process_date_standalone,  # 独立的处理函数
                    target_date
                )
                futures.append(future)

            # 跟踪进度
            with tqdm(total=len(batch_dates), desc=f"批次{batch_num}", unit="日期") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result and isinstance(result, str):
                            batch_errors.append(result)
                    except Exception as e:
                        batch_errors.append(str(e)[:100])

        # 更新统计
        total_processed += len(batch_dates)
        all_errors.extend(batch_errors)

        success_in_batch = len(batch_dates) - len(batch_errors)
        logger.debug(f"  批次完成: {success_in_batch}/{len(batch_dates)} 成功")
        if batch_errors:
            logger.error(f"  本批次错误: {len(batch_errors)} 个")
            # 打印前几个错误
            for err in batch_errors[:3]:
                logger.error(f"    - {err}")

    # 打印最终结果
    logger.debug(f"\n{'=' * 60}")
    logger.debug(f"所有批次处理完成！")
    logger.debug(f"总处理日期: {total_processed}")
    logger.debug(f"成功: {total_processed - len(all_errors)}")
    logger.debug(f"失败: {len(all_errors)}")
    logger.debug('=' * 60)


def process_date_standalone(target_date):
    """
    独立的日期处理函数，每个进程自己加载需要的数据
    这个函数必须是顶级函数（不在类中），以便pickle
    """
    import os
    import sys

    # 确保导入路径正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # 重新导入必要的模块
    try:
        from feature_pipeline import load_price_data, feature_generator
    except ImportError:
        # 如果无法导入，尝试相对导入
        from . import load_price_data, feature_generator

    try:
        # 生成特征
        feature_generator(target_date)
        return True
    except Exception as e:
        return f"{target_date}: {str(e)[:100]}"

import threading

# 创建全局缓存
CACHE = {}
CACHE_LOCK = threading.Lock()
FULL_STOCK_DATA_KEY = "full_stocks"
# 使用缓存
def get_cached_dataset(dataset_name):
    """获取或创建数据集缓存"""
    with CACHE_LOCK:
        if dataset_name not in CACHE:
            CACHE[dataset_name] = load_price_data(str(STOCK_ND_CSV_DIR))

        logger.info(f"成功从 {STOCK_ND_CSV_DIR} 加载并处理 {len(CACHE[dataset_name])} 只股票数据")
        return CACHE[dataset_name]


if __name__ == "__main__":
    full_stocks_data = get_cached_dataset(FULL_STOCK_DATA_KEY)

    # process_stocks_batch_parallel_optimized(full_stocks_data, 430, 10, start_date='2026-1-12')


    # 生成莫一天的特征
    feature_generator('2025-10-10')
