import glob
import hashlib
import json
import logging
import os
import re
import warnings
import traceback
import talib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import percentileofscore, linregress

from src.comm_fun import model_config, EPS

# ============================================================
# 缓存指纹版本号
# 修改特征逻辑、关键参数含义或数据格式时，手动递增此值，
# 以确保所有旧缓存自动失效重算。
# ============================================================
FEATURE_PIPELINE_VERSION = "2.1.0"  # issue #10: 删泄露、修 3 bug、去前视


def _build_fingerprint_params() -> dict:
    """返回影响特征输出的关键参数字典，用于构造缓存指纹。"""
    return {
        "version": FEATURE_PIPELINE_VERSION,
        "return_periods": list(model_config.RETURN_PERIODS),
        "windows_volatility": list(model_config.WINDOWS_VOLATILITY),
        "lag_periods": list(model_config.LAG_PERIODS),
        "feature_need_max_days": model_config.FEATURE_NEED_MAX_DAYS,
        "expected_profit": model_config.EXPECTED_PROFIT,
        "expected_loss": model_config.EXPECTED_LOSS,
    }


def compute_cache_fingerprint() -> str:
    """计算当前特征管线的缓存指纹（SHA-256 hex 取前 16 位）。"""
    serialized = json.dumps(_build_fingerprint_params(), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _fp_path(csv_path) -> Path:
    """返回 CSV 文件对应的指纹 sidecar 文件路径（.fp 后缀）。"""
    return Path(str(csv_path) + ".fp")


def is_cache_valid(csv_path, fingerprint: str) -> bool:
    """
    校验缓存是否有效。
    有效条件：CSV 存在 + 同名 .fp 文件存在 + 指纹完全匹配。
    任一条件不满足均返回 False，触发重算。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    fp_file = _fp_path(csv_path)
    if not fp_file.exists():
        return False
    return fp_file.read_text().strip() == fingerprint


def write_cache_fingerprint(csv_path, fingerprint: str) -> None:
    """在 CSV 同目录写入 .fp 指纹 sidecar 文件。"""
    _fp_path(csv_path).write_text(fingerprint)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config.settings import DAILY_FEATURE_DIR, STOCK_ND_CSV_DIR
from pathlib import Path

DEFAULT_STOCK_DATA_DIR = Path(__file__).parent.parent / 'stock_data'
STOCK_DATA_DIR = Path(os.environ.get('STOCK_DATA_DIR', DEFAULT_STOCK_DATA_DIR))

# 大盘ID
df_sh_symbol = "000001.SH"
df_sz_symbol = "399001.SZ"


# ========================================
# 辅助函数 (全面增加 groupby 隔离)
# ========================================
def macd_features_rolling(df):
    """使用rolling窗口计算MACD特征"""
    df['macd_hist_clean'] = df['macd_hist'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['macd_clean'] = df['macd'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['macd_signal_clean'] = df['macd_signal'].replace([np.inf, -np.inf], np.nan).fillna(0)

    def linear_slope(arr):
        if len(arr) < 2: return 0.0
        x = np.arange(len(arr))
        x_mean = np.mean(x)
        y_mean = np.mean(arr)
        denominator = np.sum((x - x_mean) ** 2)
        return np.sum((x - x_mean) * (arr - y_mean)) / denominator if denominator != 0 else 0.0

    # 【修复】限制在单只股票内部rolling
    df['macd_hist_trend_5'] = df.groupby('symbol')['macd_hist_clean'].transform(
        lambda x: x.rolling(window=6, min_periods=2).apply(linear_slope, raw=True)
    ).fillna(0)

    macd_shift = df.groupby('symbol')['macd_clean'].shift(1)
    signal_shift = df.groupby('symbol')['macd_signal_clean'].shift(1)

    df['macd_golden_cross'] = ((macd_shift < signal_shift) & (df['macd_clean'] > df['macd_signal_clean'])).astype(int)
    df['macd_death_cross'] = ((macd_shift > signal_shift) & (df['macd_clean'] < df['macd_signal_clean'])).astype(int)
    df['macd_signal_cross'] = df['macd_golden_cross']

    df['macd_zero_cross_up'] = ((macd_shift <= 0) & (df['macd_clean'] > 0)).astype(int)
    df['macd_zero_cross_down'] = ((macd_shift >= 0) & (df['macd_clean'] < 0)).astype(int)
    df['macd_zero_cross'] = (df['macd_zero_cross_up'] | df['macd_zero_cross_down']).astype(int)

    df['macd_hist_amplitude'] = df.groupby('symbol')['macd_hist_clean'].transform(
        lambda x: x.rolling(window=20, min_periods=1).apply(lambda y: y.max() - y.min(), raw=True)
    ).fillna(0)

    df['macd_hist_direction'] = np.sign(df['macd_hist_clean']).astype(int)
    df['macd_hist_acceleration'] = df.groupby('symbol')['macd_hist_clean'].transform(lambda x: x.diff().diff())

    df['macd_signal_convergence'] = df['macd_clean'] - df['macd_signal_clean']
    df['macd_signal_convergence_trend'] = df.groupby('symbol')['macd_signal_convergence'].transform(
        lambda x: x.rolling(window=5, min_periods=2).apply(linear_slope, raw=True)
    ).fillna(0)

    df.drop(['macd_hist_clean', 'macd_clean', 'macd_signal_clean'], axis=1, inplace=True, errors='ignore')
    return df


def calculate_volume_features_vectorized(df):
    """向量化计算成交量相关特征"""
    result = pd.DataFrame(index=df.index)

    def rolling_slope_1d(series, window):
        slopes = np.zeros(len(series))
        vals = series.values
        for i in range(window - 1, len(vals)):
            x = np.arange(window)
            y = vals[i - window + 1:i + 1]
            slope, _, _, _, _ = linregress(x, y)
            slopes[i] = slope
        return slopes

    def rolling_cv_1d(series, window):
        cv = np.zeros(len(series))
        vals = series.values
        for i in range(window - 1, len(vals)):
            window_data = vals[i - window + 1:i + 1]
            mean_val = np.mean(window_data)
            if mean_val > 0:
                cv[i] = np.std(window_data, ddof=1) / mean_val
        return cv

    result['volume_trend_5'] = df.groupby('symbol')['volume'].transform(lambda x: rolling_slope_1d(x, 6))
    result['volume_trend_10'] = df.groupby('symbol')['volume'].transform(lambda x: rolling_slope_1d(x, 11))
    result['price_trend_5'] = df.groupby('symbol')['close'].transform(lambda x: rolling_slope_1d(x, 6))
    result['price_volume_divergence'] = ((result['price_trend_5'] * result['volume_trend_5']) < 0).astype(int)
    result['volume_consistency'] = df.groupby('symbol')['volume'].transform(lambda x: rolling_cv_1d(x, 6))
    return result.fillna(0)


def optimized_rsi_features(df):
    """优化的RSI特征计算"""
    # 【修复】限制在组内 diff/shift
    df['rsi_momentum'] = df.groupby('symbol')['rsi_14'].diff()
    df['rsi_diff1'] = df.groupby('symbol')['rsi_14'].diff(1)

    rsi_diff1_shift1 = df.groupby('symbol')['rsi_diff1'].shift(1)
    df['rsi_turning_simple'] = (rsi_diff1_shift1 * df['rsi_diff1'] < 0).astype(int)

    rsi_shift1 = df.groupby('symbol')['rsi_14'].shift(1)
    rsi_shift2 = df.groupby('symbol')['rsi_14'].shift(2)

    top = (rsi_shift1 > rsi_shift2) & (df['rsi_14'] < rsi_shift1)
    bottom = (rsi_shift1 < rsi_shift2) & (df['rsi_14'] > rsi_shift1)

    df['rsi_turning'] = (top | bottom).astype(int)
    df['rsi_momentum'] = df['rsi_momentum'].fillna(0)
    df['rsi_turning'] = df['rsi_turning'].fillna(0)
    df['rsi_turning_simple'] = df['rsi_turning_simple'].fillna(0)
    df = df.drop(['rsi_diff1'], axis=1, errors='ignore')
    return df


def calculate_volume_ma_ratio_vectorized(df, short_window=5, long_window=10):
    """向量化计算成交量MA比率"""
    volume_ma_short = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window=short_window, min_periods=short_window).mean().shift(1))
    volume_ma_long = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window=long_window, min_periods=long_window).mean().shift(1))
    ratio = volume_ma_short / volume_ma_long
    ratio = np.where(volume_ma_long > 0, ratio, 1.0)
    ratio = np.where(pd.isna(ratio), 1.0, ratio)
    return ratio


def multi_window_volatility(df, windows=model_config.WINDOWS_VOLATILITY):
    """计算多个窗口的波动率"""
    df['log_return'] = np.log(df['close'] / df.groupby('symbol')['close'].shift(1))
    for window in windows:
        vol_col = f'volatility_{window}d'
        daily_vol = df.groupby('symbol')['log_return'].transform(
            lambda x: x.rolling(window=window, min_periods=window).std())
        df[vol_col] = daily_vol * np.sqrt(252)
        df[vol_col] = df[vol_col].fillna(0)

    if 'log_return' in df.columns:
        df = df.drop('log_return', axis=1)
    return df


def fast_obv_trend(df, window=5):
    """高效向量化计算OBV趋势"""

    def obv_trend_1d(series):
        n = window + 1
        x = np.arange(n)
        sum_x = np.sum(x)
        sum_x2 = np.sum(x ** 2)
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0: return np.zeros(len(series))
        trend = np.zeros(len(series))
        obv_values = series.values
        for i in range(n - 1, len(series)):
            window_obv = obv_values[i - n + 1:i + 1]
            sum_y = np.sum(window_obv)
            sum_xy = np.sum(x * window_obv)
            trend[i] = (n * sum_xy - sum_x * sum_y) / denom
        return trend

    # 【修复】限制在组内 transform
    df['obv_trend'] = df.groupby('symbol')['obv'].transform(obv_trend_1d)
    return df


def calculate_macd_percentile_vectorized(df, window=100):
    """向量化计算MACD百分位数特征"""

    def macd_pct_1d(series):
        res = np.full(len(series), 50.0)
        vals = series.values
        for i in range(window, len(vals)):
            historical = vals[i - window:i]
            res[i] = percentileofscore(historical, vals[i], kind='rank')
        return res

    return df.groupby('symbol')['macd'].transform(macd_pct_1d)


def vectorized_support_resistance(df, window=20):
    """向量化计算支撑阻力特征"""
    df = df.copy()
    support = df.groupby('symbol')['low'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
    resistance = df.groupby('symbol')['high'].transform(lambda x: x.rolling(window=window, min_periods=1).max())

    df['distance_to_support'] = np.where(df['close'] > 0, (df['close'] - support) / df['close'], 0)
    df['distance_to_resistance'] = np.where(df['close'] > 0, (resistance - df['close']) / df['close'], 0)
    return df


def detect_hammer_pattern_single(kline):
    """检测锤子线形态（单根K线）"""
    try:
        body_size = abs(kline['close'] - kline['open'])
        lower_shadow = min(kline['open'], kline['close']) - kline['low']
        upper_shadow = kline['high'] - max(kline['open'], kline['close'])
        total_range = kline['high'] - kline['low']
        if total_range == 0: return 0
        is_hammer_shape = (
                    lower_shadow >= 2 * body_size and upper_shadow <= body_size * 0.5 and lower_shadow >= total_range * 0.6)
        return 1 if is_hammer_shape else 0
    except:
        return 0


def detect_doji_pattern_single(kline):
    """检测十字星形态（单根K线）"""
    try:
        body_size = abs(kline['close'] - kline['open'])
        total_range = kline['high'] - kline['low']
        if total_range == 0: return 0
        is_doji = body_size < total_range * 0.1
        return 1 if is_doji else 0
    except:
        return 0


def fast_engulfing_detection(df):
    """快速向量化吞噬形态检测"""
    prev_open = df.groupby('symbol')['open'].shift(1)
    prev_close = df.groupby('symbol')['close'].shift(1)

    prev_is_bearish = prev_close < prev_open
    prev_is_bullish = prev_close > prev_open
    curr_is_bullish = df['close'] > df['open']
    curr_is_bearish = df['close'] < df['open']

    bullish_engulfing = (prev_is_bearish & curr_is_bullish & (df['open'] <= prev_close) & (df['close'] >= prev_open))
    bearish_engulfing = (prev_is_bullish & curr_is_bearish & (df['open'] >= prev_close) & (df['close'] <= prev_open))

    engulfing = (bullish_engulfing | bearish_engulfing).astype(int)
    return engulfing


class FeaturePipeline:
    """单类版本 Feature Pipeline"""

    def __init__(self, divergence_detector, full_stocks_inner):
        self.divergence_detector = divergence_detector
        self.full_stocks_data = full_stocks_inner

    def _calculate_future_return(self, full_data, signal_date, periods=model_config.RETURN_PERIODS):
        returns = {}
        try:
            if not isinstance(signal_date, pd.Timestamp):
                signal_date = pd.to_datetime(signal_date)

            full_data.sort_index(ascending=True)
            signal_idx = full_data.index.get_loc(signal_date)
            signal_price = full_data['close'].iat[signal_idx]

            if signal_idx + 1 >= len(full_data):
                for period in periods:
                    returns[f'future_return_{period}d'] = np.nan
                    returns[f'future_sell_date_{period}d'] = np.nan
                    returns[f'stop_loss_return_{period}d'] = np.nan
                    returns[f'stop_loss_sell_date_{period}d'] = np.nan
                return returns

            next_day_low = full_data['low'].iat[signal_idx + 1]
            if next_day_low > signal_price:
                for period in periods:
                    returns[f'future_return_{period}d'] = np.nan
                    returns[f'future_sell_date_{period}d'] = np.nan
                    returns[f'stop_loss_return_{period}d'] = np.nan
                    returns[f'stop_loss_sell_date_{period}d'] = np.nan
                return returns

            buy_price = signal_price
            for period in periods:
                if signal_idx + 1 + period < len(full_data):
                    future_data = full_data.iloc[signal_idx + 2: signal_idx + 1 + period + 1]
                    if len(future_data) > 0:
                        future_return, future_sell_date = self._calculate_original_return(full_data, signal_idx,
                                                                                          buy_price, period)
                        returns[f'future_return_{period}d'] = future_return
                        returns[f'future_sell_date_{period}d'] = future_sell_date

                        stop_loss_return, stop_loss_sell_date = self._calculate_stop_loss_return(full_data, signal_idx,
                                                                                                 buy_price, period)
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
        except Exception as e:
            for period in periods:
                returns[f'future_return_{period}d'] = np.nan
                returns[f'future_sell_date_{period}d'] = np.nan
                returns[f'stop_loss_return_{period}d'] = np.nan
                returns[f'stop_loss_sell_date_{period}d'] = np.nan
        return returns

    def _calculate_original_return(self, full_data, signal_idx, buy_price, period):
        try:
            buy_day_idx = signal_idx + 1
            start_idx = buy_day_idx + 1
            end_idx = buy_day_idx + period
            if end_idx >= len(full_data): return np.nan, np.nan
            future_data = full_data.iloc[start_idx:end_idx + 1]
            if len(future_data) == 0: return np.nan, np.nan

            target_price = buy_price * model_config.EXPECTED_PROFIT
            target_hit_mask = future_data['high'] >= target_price
            if target_hit_mask.any():
                first_target_date = future_data[target_hit_mask].index[0]
                future_return = (target_price - buy_price) / buy_price
                return future_return, first_target_date
            else:
                future_price = full_data['close'].iat[end_idx]
                future_return = (future_price - buy_price) / buy_price
                future_sell_date = full_data.index[end_idx]
                return future_return, future_sell_date
        except Exception:
            return np.nan, np.nan

    def _calculate_stop_loss_return(self, full_data, signal_idx, buy_price, period):
        try:
            buy_day_idx = signal_idx + 1
            end_idx = buy_day_idx + period
            if end_idx >= len(full_data): return np.nan, np.nan

            for days_after_buy in range(1, period + 1):
                current_idx = buy_day_idx + days_after_buy
                if current_idx >= len(full_data): break
                current_data = full_data.iloc[current_idx]
                current_date = full_data.index[current_idx]

                if current_data['open'] <= buy_price * model_config.EXPECTED_LOSS:
                    return (current_data['open'] - buy_price) / buy_price, current_date
                if current_data['low'] <= buy_price * model_config.EXPECTED_LOSS:
                    return model_config.EXPECTED_LOSS - 1, current_date
                if current_data['high'] >= buy_price * model_config.EXPECTED_PROFIT:
                    return model_config.EXPECTED_PROFIT - 1, current_date

            final_price = full_data['close'].iloc[end_idx]
            final_date = full_data.index[end_idx]
            return (final_price - buy_price) / buy_price, final_date
        except Exception:
            return np.nan, np.nan

    def enrich(self, all_2_target_day_df, df_sh, df_sz):
        try:
            logger.debug(f"开始全量数据的特征计算....")
            df_sorted = all_2_target_day_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            unique_dates = sorted(df_sorted['timestamp'].unique())

            if len(unique_dates) >= model_config.FEATURE_NEED_MAX_DAYS:
                last_dates = unique_dates[-model_config.FEATURE_NEED_MAX_DAYS:]
            else:
                last_dates = unique_dates

            feature_used_df = df_sorted[df_sorted['timestamp'].isin(last_dates)].copy()

            # 【修复】特征计算流水线
            feature_used_df = self._calculate_basic_technical_features(feature_used_df)
            feature_used_df = self._calculate_advance_technical_features(feature_used_df)
            feature_used_df = self._generate_alpha_features(feature_used_df)
            feature_used_df = self.generate_structure_features(feature_used_df)
            feature_used_df = self.generate_lag_features(feature_used_df)
            all_2_target_day_df = self._calculate_basic_technical_features(all_2_target_day_df)

            logger.debug(f"全量数据的特征计算完成")

            target_date = feature_used_df['timestamp'].dt.date.max()
            target_df = feature_used_df[feature_used_df['timestamp'].dt.date == target_date]

            logger.debug(f"{target_date} 开始大盘特征计算....")
            market_features_df = self._calculate_market_features(target_date, df_sh=df_sh, df_sz=df_sz)
            target_df = target_df.merge(market_features_df, left_on='timestamp', right_index=True, how='left')
            logger.debug(f"{target_date} 大盘特征计算完成")

            logger.debug(f"{target_date} 丰富目标日期的DF特征和背离点提取......")
            all_divergence_points = pd.DataFrame()
            enriched_points = pd.DataFrame()
            process_id = 0

            for _, row in target_df.iterrows():
                symbol = row['symbol']
                enriched_point = row.to_dict()
                data_with_indicators = all_2_target_day_df[all_2_target_day_df['symbol'] == symbol].set_index(
                    'timestamp')
                enriched_points = pd.concat([enriched_points, pd.DataFrame(enriched_point, index=[0])],
                                            ignore_index=True)
                divergence_points_df = self.divergence_detector.detect_daily_divergence(data_with_indicators, symbol,
                                                                                        target_date)
                if len(divergence_points_df) > 0:
                    all_divergence_points = pd.concat([all_divergence_points, divergence_points_df], ignore_index=True)
                process_id += 1

            if all_divergence_points.empty:
                logger.warning(f"{target_date.strftime('%Y%m%d')}无背离数据")
                return None

            logger.debug(f"{target_date} 交叉特征添加......")
            enriched_points = self._calculate_cross_features(enriched_points)
            # 【修复】DataFrame.get() 返回 Series，if Series <= 3 永远走 else 分支导致恒为 0
            if 'formation_period' in enriched_points.columns:
                enriched_points['is_quick_divergence'] = (enriched_points['formation_period'] <= 3).astype(int)
            else:
                enriched_points['is_quick_divergence'] = 0

            # ====== 【核心修复】强行对齐时间列的数据类型并防止列名冲突 ======
            enriched_points['timestamp'] = pd.to_datetime(enriched_points['timestamp'])

            # 兼容检测：判断背离表里是用 detection_date 还是 timestamp 记录的信号日
            if 'detection_date' in all_divergence_points.columns:
                all_divergence_points['detection_date'] = pd.to_datetime(all_divergence_points['detection_date'])

                # 【新增修复】：防止左右表都有 timestamp 导致合并后变成 timestamp_x
                if 'timestamp' in all_divergence_points.columns:
                    all_divergence_points.rename(columns={'timestamp': 'divergence_date'}, inplace=True)

                target_divergence_df = enriched_points.merge(
                    all_divergence_points,
                    left_on=['symbol', 'timestamp'],
                    right_on=['symbol', 'detection_date'],
                    how='inner'
                )
            else:
                all_divergence_points['timestamp'] = pd.to_datetime(all_divergence_points['timestamp'])
                target_divergence_df = enriched_points.merge(
                    all_divergence_points, on=['symbol', 'timestamp'], how='inner'
                )
            # ==================================================

            target_divergence_df = self._filter_valid_rows_apply(target_divergence_df)

            if target_divergence_df.empty:
                logger.warning(f"{target_date.strftime('%Y%m%d')}无有效背离数据")
                return

            target_divergence_df = target_divergence_df.apply(
                lambda row: self._calculate_and_assign(row, row['symbol']), axis=1)
            target_divergence_df['divergence_amount'] = len(target_divergence_df)

            self.save_to_csv(target_divergence_df,
                             DAILY_FEATURE_DIR / f"realistic_features_{target_date.strftime('%Y%m%d')}.csv")
            return target_divergence_df

        except Exception as e:
            logger.exception(f"特征生成失败: {e}")
            return None

    def _filter_valid_rows_apply(self, target_divergence_df):
        """过滤掉无法查到历史行情数据的背离信号行。

        只使用信号日当日及之前的已知信息（symbol 存在、timestamp 在数据范围内）。
        原先"偷看次日最低价以判断买单是否成交"的前视过滤已移除——
        可成交性由 _calculate_future_return 通过标签 NaN 处理，训练时再过滤 NaN 行。
        """
        def is_valid_row(row):
            symbol = row['symbol']
            timestamp = row['timestamp']

            if symbol not in self.full_stocks_data:
                return False
            df = self.full_stocks_data[symbol]
            return timestamp in df.index

        mask = target_divergence_df.apply(is_valid_row, axis=1)
        return target_divergence_df[mask].reset_index(drop=True)

    def _calculate_and_assign(self, row, symbol):
        return_periods = model_config.RETURN_PERIODS
        future_price_change = self._calculate_future_return(self.full_stocks_data.get(symbol), row['timestamp'],
                                                            return_periods)

        for period in return_periods:
            row[f'future_return_{period}d'] = future_price_change.get(f'future_return_{period}d', np.nan)
            row[f'future_sell_date_{period}d'] = future_price_change.get(f'future_sell_date_{period}d', np.nan)
            row[f'stop_loss_return_{period}d'] = future_price_change.get(f'stop_loss_return_{period}d', np.nan)
            row[f'stop_loss_sell_date_{period}d'] = future_price_change.get(f'stop_loss_sell_date_{period}d', np.nan)
        return row

    def save_to_csv(self, df, file_path):
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            # 落盘时降精度（float64 → float32），不影响内存中的计算精度
            disk_df = optimize_dtypes(df)
            disk_df.to_csv(file_path, index=True if disk_df.index.name else False)
            # 写入指纹 sidecar，使后续缓存校验有效
            write_cache_fingerprint(file_path, compute_cache_fingerprint())
            logger.debug(f"数据已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            raise

    def _generate_alpha_features(self, df):
        if len(df) < 100: return None
        EPSILON = 1e-9

        df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
        high_low_range = df['high'] - df['low']
        close_open_range = df['close'] - df['open']

        df['pct_change'] = df.groupby('symbol')['close'].pct_change()
        df['clv'] = (df['close'] - df['low']) / (high_low_range + EPSILON)

        is_limit_up_down = df['high'] == df['low']
        entity_top = df[['open', 'close']].max(axis=1)
        df['upper_shadow_ratio'] = np.where(is_limit_up_down, 0.0,
                                            (df['high'] - entity_top) / (high_low_range + EPSILON))
        df['body_strength'] = close_open_range / (high_low_range + EPSILON)

        if 'timestamp' in df.columns and df['timestamp'].nunique() > 1:
            df['rank_return'] = df.groupby('timestamp')['pct_change'].rank(pct=True)
            target_vol_col = 'hs_volume_ratio' if 'hs_volume_ratio' in df.columns else 'volume'
            df['rank_volume'] = df.groupby('timestamp')[target_vol_col].rank(pct=True)

        vol_col = 'hs_volume_ratio' if 'hs_volume_ratio' in df.columns else 'volume'
        df['signed_vol_strength'] = df[vol_col] * np.sign(close_open_range)
        # 因为数据已经严格按 symbol, timestamp 排序
        df['pv_corr_10'] = df['close'].rolling(10).corr(df[vol_col])
        # 使用 cumcount 掐断每只股票的前9天（跨股票串联产生的废数据）
        df.loc[df.groupby('symbol').cumcount() < 10, 'pv_corr_10'] = np.nan

        rolling_max_60 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(60).max())
        df['dist_to_high_60'] = df['close'] / (rolling_max_60 + EPSILON)

        volatility_short = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).std())
        volatility_long_mean = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(60).std().rolling(60, min_periods=1).mean())
        df['vol_divergence'] = volatility_short / (volatility_long_mean + EPSILON)

        df.fillna(0, inplace=True)
        if 'macd_percentile' in df.columns:
            df.loc[df['macd_percentile'] == 50, 'macd_percentile'] = np.nan
        return df

    def _calculate_basic_technical_features(self, df):
        """【修复】严格按股票隔离的底层指标计算，并防止次新股导致列名丢失"""
        try:
            if len(df) < 100: return None
            df = df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

            # ====== 【核心修复】提前初始化所有指标列的坑位 ======
            # 强行锁定表头，防止某只股票数据不足 35 天时触发 return g，导致 Pandas 推断丢失全表列名
            indicator_cols = [
                'macd', 'macd_signal', 'macd_hist', 'rsi_6', 'rsi_14', 'rsi_24',
                'ma_5', 'ma_20', 'ma_60', 'bb_upper', 'bb_middle', 'bb_lower',
                'volume_ma_20', 'obv', 'atr', 'slowk', 'slowd'
            ]
            for col in indicator_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # ====================================================

            def calc_talib(g):
                if len(g) < 35: return g  # 现在直接返回也很安全，因为表头(NaN)已经存在了
                g = g.copy()
                g['macd'], g['macd_signal'], g['macd_hist'] = talib.MACD(g['close'])
                g['rsi_6'] = talib.RSI(g['close'], timeperiod=6)
                g['rsi_14'] = talib.RSI(g['close'], timeperiod=14)
                g['rsi_24'] = talib.RSI(g['close'], timeperiod=24)
                g['ma_5'] = talib.MA(g['close'], timeperiod=5)
                g['ma_20'] = talib.MA(g['close'], timeperiod=20)
                g['ma_60'] = talib.MA(g['close'], timeperiod=60)
                g['bb_upper'], g['bb_middle'], g['bb_lower'] = talib.BBANDS(g['close'])
                g['volume_ma_20'] = talib.MA(g['volume'], timeperiod=20)
                g['obv'] = talib.OBV(g['close'], g['volume'])
                g['atr'] = talib.ATR(g['high'], g['low'], g['close'])
                g['slowk'], g['slowd'] = talib.STOCH(g['high'], g['low'], g['close'])
                return g

            logger.debug("正在严格按股票分组计算 TA-Lib 指标...")
            symbol_series = df['symbol'].copy()
            result = df.groupby('symbol', group_keys=False).apply(calc_talib)
            # pandas 3.x groupby.apply 会丢弃分组键列，需要手动恢复
            if 'symbol' not in result.columns:
                result['symbol'] = symbol_series.values
            return result
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            return None

    def _calculate_market_features(self, timestamp, df_sh=None, df_sz=None):
        all_features = {}
        try:
            if not isinstance(timestamp, pd.Timestamp): timestamp = pd.to_datetime(timestamp)
            features_list = []
            if df_sh is not None: features_list.append(self._calculate_single_index_features(timestamp, df_sh, 'sh'))
            if df_sz is not None: features_list.append(self._calculate_single_index_features(timestamp, df_sz, 'sz'))
            if len(features_list) == 0: return self._get_default_combined_features(timestamp)

            for feature_dict in features_list: all_features.update(feature_dict)

            if df_sh is not None and df_sz is not None:
                if 'sh_price_change' in all_features and 'sz_price_change' in all_features:
                    all_features['sh_sz_sync_direction'] = 1 if all_features['sh_price_change'] * all_features[
                        'sz_price_change'] > 0 else 0
                    all_features['sh_sz_sync_strength'] = abs(
                        all_features['sh_price_change'] - all_features['sz_price_change'])

            price_changes, amplitudes = [], []
            for prefix in ['sh', 'sz']:
                if f'{prefix}_price_change' in all_features: price_changes.append(
                    all_features[f'{prefix}_price_change'])
                if f'{prefix}_amplitude' in all_features: amplitudes.append(all_features[f'{prefix}_amplitude'])

            if price_changes and amplitudes:
                avg_price_change = sum(price_changes) / len(price_changes)
                avg_amplitude = sum(amplitudes) / len(amplitudes)
                conditions = [(avg_price_change > 0.01) & (avg_amplitude < 0.02),
                              (avg_price_change < -0.01) & (avg_amplitude > 0.03)]
                all_features['market_sentiment'] = int(np.select(conditions, [2, 0], default=1))
                all_features['market_avg_change'] = avg_price_change
                all_features['market_avg_amplitude'] = avg_amplitude
                if len(price_changes) > 1:
                    max_diff = max(price_changes) - min(price_changes)
                    all_features['market_sync_score'] = 1 - min(1.0, max_diff / 0.02)
                else:
                    all_features['market_sync_score'] = 0.5
        except:
            all_features = self._get_default_combined_features(timestamp)

        market_features_df = pd.DataFrame([all_features])
        market_features_df['timestamp'] = timestamp
        market_features_df.set_index('timestamp', inplace=True)
        return market_features_df

    def _calculate_single_index_features(self, timestamp, df_index, prefix):
        features = {}
        try:
            index_hist = df_index[df_index.index <= timestamp]
            index_data = index_hist[index_hist.index == timestamp]
            if len(index_data) > 0:
                index_row = index_data.iloc[0]
                features[f'{prefix}_price_change'] = (index_row['close'] - index_row['open']) / index_row['open']
                features[f'{prefix}_amplitude'] = (index_row['high'] - index_row['low']) / index_row['low']
                if len(index_hist) >= 20:
                    avg_volume = index_hist['volume'].rolling(20).mean().iloc[-1]
                    features[f'{prefix}_volume_ratio'] = index_row['volume'] / avg_volume if avg_volume != 0 else 1
                else:
                    features[f'{prefix}_volume_ratio'] = 1
                features[f'{prefix}_price_change_abs'] = abs(features[f'{prefix}_price_change'])
                features[f'{prefix}_price_wave_abs'] = features[f'{prefix}_amplitude']
                conditions = [(features[f'{prefix}_price_change'] > 0.01) & (features[f'{prefix}_amplitude'] < 0.02),
                              (features[f'{prefix}_price_change'] < -0.01) & (features[f'{prefix}_amplitude'] > 0.03)]
                features[f'{prefix}_sentiment'] = int(np.select(conditions, [2, 0], default=1))
                features[f'{prefix}_volume_signal'] = 1 if features[f'{prefix}_volume_ratio'] > 1.2 else 0
            else:
                raise Exception("No Data")
        except:
            features.update({f'{prefix}_price_change': 0, f'{prefix}_amplitude': 0.02, f'{prefix}_volume_ratio': 1,
                             f'{prefix}_price_change_abs': 0, f'{prefix}_price_wave_abs': 0.02,
                             f'{prefix}_sentiment': 1, f'{prefix}_volume_signal': 0})
        return features

    def _get_default_combined_features(self, timestamp):
        default_features = {}
        for prefix in ['hs', 'gq', 'hc']:
            default_features.update(
                {f'{prefix}_price_change': 0, f'{prefix}_amplitude': 0.02, f'{prefix}_volume_ratio': 1,
                 f'{prefix}_price_change_abs': 0, f'{prefix}_price_wave_abs': 0.02, f'{prefix}_sentiment': 1,
                 f'{prefix}_volume_signal': 0})
        default_features.update(
            {'hs_gq_sync_direction': 1, 'hs_gq_sync_strength': 0, 'hs_hc_sync_direction': 1, 'hs_hc_sync_strength': 0,
             'gq_hc_sync_direction': 1, 'gq_hc_sync_strength': 0, 'market_sentiment': 1, 'market_avg_change': 0,
             'market_avg_amplitude': 0.02, 'market_sync_score': 0.5})
        return default_features

    def get_weights_ffd(self, d, thres, lim):
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres: break
            w.append(w_k)
            k += 1
            if k >= lim: break
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series, d, thres=1e-5, lim=10000):
        w = self.get_weights_ffd(d, thres, lim)
        width = len(w) - 1
        output = np.full(len(series), np.nan)
        series_val = series.values if hasattr(series, 'values') else np.array(series)
        for i in range(width, len(series_val)):
            window0 = series_val[i - width: i + 1]
            output[i] = np.dot(window0, w)[0]
        return output

    def robust_zscore_rolling(self, series, window=100):
        rolling = series.rolling(window=window, min_periods=max(10, window // 2))
        median = rolling.median()
        iqr = rolling.quantile(0.75) - rolling.quantile(0.25)
        iqr = np.where(iqr == 0, 1e-9, iqr)
        return (series - median) / (iqr / 1.34896)

    def generate_structure_features(self, df):
        if len(df) < 100: return None
        eps = EPS
        log_hl = np.log(df['high'] / (df['low'] + eps))
        log_co = np.log(df['close'] / (df['open'] + eps))
        df['vol_gk'] = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)

        # 组内滚动均值
        df['vol_gk_ratio'] = df['vol_gk'] / df.groupby('symbol')['vol_gk'].transform(lambda x: x.rolling(20).mean())
        df['illiq'] = np.log(
            (df.groupby('symbol')['close'].pct_change().abs() / (df['close'] * df['volume'] + eps)) + 1)

        change = (df['close'] - df.groupby('symbol')['close'].shift(10)).abs()
        volatility = df.groupby('symbol')['close'].transform(lambda x: x.diff().abs().rolling(10).sum())
        df['efficiency_ratio'] = change / (volatility + eps)

        is_limit_up_down = df['high'] == df['low']
        is_bullish = df['close'] >= df['open']
        df['intraday_pos'] = np.where(is_limit_up_down, np.where(is_bullish, 1.0, 0.0),
                                      (df['close'] - df['low']) / (df['high'] - df['low'] + eps))

        df['ret_overnight'] = (df['open'] / df.groupby('symbol')['close'].shift(1)) - 1
        df['ret_intraday'] = (df['close'] / df['open']) - 1
        df['smart_money_diff'] = df['ret_intraday'] - df['ret_overnight']

        df['high_mean_20'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).mean())
        df['low_mean_20'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20).mean())
        df['support_resistance_ratio'] = df['high_mean_20'] / df['low_mean_20']
        df['log_volume'] = np.log1p(df['volume'])

        # ✅ 【安全平替1】使用 log1p 替代 Box-Cox
        df['boxcox_atr'] = np.log1p(df['atr'].clip(lower=eps))

        df['rsi_robust'] = df.groupby('symbol')['rsi_14'].transform(lambda x: self.robust_zscore_rolling(x, window=100))
        df['macd_robust'] = df.groupby('symbol')['macd'].transform(lambda x: self.robust_zscore_rolling(x, window=100))

        # ✅ 【安全平替2】使用 EMA 平滑替代全局小波去噪
        df['close_smooth_10'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())

        return df

    def generate_lag_features(self, df):
        if len(df) < 100: return None
        for lag in model_config.LAG_PERIODS:
            df[f'close_lag_{lag}'] = df.groupby('symbol')['close'].shift(lag)
            df[f'open_lag_{lag}'] = df.groupby('symbol')['open'].shift(lag)
            df[f'high_lag_{lag}'] = df.groupby('symbol')['high'].shift(lag)
            df[f'low_lag_{lag}'] = df.groupby('symbol')['low'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['volume'].shift(lag)

        df['daily_return'] = df.groupby('symbol')['close'].pct_change()
        for lag in [1, 2, 3, 5, 10, 20]: df[f'return_lag_{lag}'] = df.groupby('symbol')['daily_return'].shift(lag)

        df['amplitude'] = (df['high'] - df['low']) / df['low']
        for lag in [1, 3, 5]: df[f'amplitude_lag_{lag}'] = df.groupby('symbol')['amplitude'].shift(lag)

        for lag in [1, 3, 5, 10]:
            df[f'vol_gk_lag_{lag}'] = df.groupby('symbol')['vol_gk'].shift(lag)
            df[f'vol_gk_ratio_lag_{lag}'] = df.groupby('symbol')['vol_gk_ratio'].shift(lag)

        for lag in [1, 3, 5]:
            df[f'illiq_lag_{lag}'] = df.groupby('symbol')['illiq'].shift(lag)
            df[f'efficiency_ratio_lag_{lag}'] = df.groupby('symbol')['efficiency_ratio'].shift(lag)
            df[f'smart_money_diff_lag_{lag}'] = df.groupby('symbol')['smart_money_diff'].shift(lag)
            df[f'ret_overnight_lag_{lag}'] = df.groupby('symbol')['ret_overnight'].shift(lag)
            df[f'ret_intraday_lag_{lag}'] = df.groupby('symbol')['ret_intraday'].shift(lag)
            df[f'support_resistance_ratio_lag_{lag}'] = df.groupby('symbol')['support_resistance_ratio'].shift(lag)

        for lag in [1, 2, 3]: df[f'intraday_pos_lag_{lag}'] = df.groupby('symbol')['intraday_pos'].shift(lag)
        return df

    def _calculate_cross_features(self, final_features):
        ts_col = 'timestamp'
        return_periods = model_config.RETURN_PERIODS
        label_columns = []
        for period in return_periods:
            label_columns.extend(
                [f'future_return_{period}d', f'future_sell_date_{period}d', f'stop_loss_return_{period}d',
                 f'stop_loss_sell_date_{period}d'])
        exclude = {'timestamp', 'symbol', 'label', 'open', 'close', 'high', 'low'}
        exclude.update(label_columns)
        features = [c for c in final_features.columns if
                    c not in exclude and final_features[c].dtype in (np.float64, np.int64)]
        final_features['cs_n'] = final_features.groupby(ts_col)[ts_col].transform('count')
        for f in features:
            final_features[f + '_rankpct'] = final_features.groupby(ts_col)[f].rank(pct=True)
            final_features[f + '_z'] = final_features.groupby(ts_col)[f].transform(
                lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-9))
        return final_features

    def _calculate_advance_technical_features(self, df):
        # 1. 均线系统特征
        df['price_vs_ma5'] = df['close'] / df['ma_5'] - 1
        df['price_vs_ma20'] = df['close'] / df['ma_20'] - 1
        df['price_vs_ma60'] = df['close'] / df['ma_60'] - 1
        df['ma_arrangement'] = np.where((df['ma_5'] > df['ma_20']) & (df['ma_20'] > df['ma_60']), 1,
                                        np.where((df['ma_5'] < df['ma_20']) & (df['ma_20'] < df['ma_60']), -1, 0))

        # 2. 布林带特征
        bb_width = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = np.where(bb_width > 0, (df['close'] - df['bb_lower']) / bb_width, 0.5)
        df['bb_squeeze'] = np.where(bb_width / df['close'] < 0.05, 1, 0)

        # 3. RSI补充
        df['rsi_oversold_6'] = np.where(df['rsi_6'] < 30, 1, 0)
        df['rsi_oversold_14'] = np.where(df['rsi_14'] < 30, 1, 0)
        df['rsi_oversold_24'] = np.where(df['rsi_24'] < 30, 1, 0)
        df['rsi_overbought_6'] = np.where(df['rsi_6'] > 70, 1, 0)
        df['rsi_overbought_14'] = np.where(df['rsi_14'] > 70, 1, 0)
        df['rsi_overbought_24'] = np.where(df['rsi_24'] > 70, 1, 0)

        # 4. MACD 补充
        df['macd_signal_distance'] = df['macd'] - df['macd_signal']
        # 【修复】原代码用 macd_golden_cross 覆盖了 macd_features_rolling() 里的真实金叉检测信号。
        # 此处语义是"MACD 线在信号线之上"，改名为 macd_above_signal，避免覆盖。
        df['macd_above_signal'] = np.where(df['macd_signal_distance'] > 0, 1, 0)

        # 5. 成交量特征补充 【修复】
        df['volume_ma20'] = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean())
        df['volume_ratio'] = np.where(df['volume_ma20'] > 0, df['volume'] / df['volume_ma20'], 1)
        df['volume_spike'] = np.where(df['volume_ratio'] > 2.0, 1, 0)
        df['volume_dryup'] = np.where(df['volume_ratio'] < 0.5, 1, 0)

        # 6. 波动率特征
        df['atr_ratio'] = np.where(df['close'] > 0, df['atr'] / df['close'], 0)

        # 7. K线形态特征
        df['hammer_pattern'] = df.apply(detect_hammer_pattern_single, axis=1)
        # 【修复】
        df['downtrend'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean() < x.rolling(10).mean())
        df['hammer_signal'] = df['hammer_pattern'] & df['downtrend']
        df['doji_pattern'] = df.apply(detect_doji_pattern_single, axis=1)

        df = vectorized_support_resistance(df, window=20)
        df['stoch_oversold'] = np.where(df['slowk'] < 20, 1, 0)
        df['stoch_overbought'] = np.where(df['slowd'] > 80, 1, 0)
        df['macd_percentile'] = calculate_macd_percentile_vectorized(df, window=100)
        df = fast_obv_trend(df, window=5)
        df = multi_window_volatility(df, windows=model_config.WINDOWS_VOLATILITY)
        df['engulfing_pattern'] = fast_engulfing_detection(df)
        df['signed_volume_strength'] = df['volume_ratio'] * ((df['close'] - df['open']) / df['open'])
        df['close_vs_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['volume_ma_ratio'] = calculate_volume_ma_ratio_vectorized(df, short_window=5, long_window=10)

        df = optimized_rsi_features(df)
        volume_features = calculate_volume_features_vectorized(df)
        df = pd.concat([df, volume_features], axis=1)
        df = macd_features_rolling(df)
        return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """落盘专用：将 float64 列降精度为 float32（保留 4 位小数）。

    返回新 DataFrame，不修改传入的对象。
    仅在写入磁盘前调用；特征计算全程应使用 float64 原始精度。
    """
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        try:
            df[col] = df[col].round(4).astype(np.float32)
        except Exception:
            pass
    return df


def load_price_data(directory_path, start_date='2009-01-01', end_date=None):
    pattern = os.path.join(directory_path, "*_price_data.csv")
    file_paths = glob.glob(pattern)
    dataframes = {}

    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            match = re.match(r'(\d{6}\.[A-Z]{2})_price_data\.csv', filename)
            if match:
                symbol = match.group(1)
            else:
                continue

            df = pd.read_csv(file_path, encoding='gb2312', index_col=0, dtype={'symbol': str})
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df['symbol'] = symbol
            # 原始行情保持 float64，全程参与特征计算；降精度仅在落盘时执行
            df = ensure_datetime_index(df)
            df = df.sort_index(ascending=True)

            if start_date or end_date:
                mask = pd.Series(True, index=df.index)
                if start_date: mask &= (df.index >= pd.to_datetime(start_date))
                if end_date: mask &= (df.index <= pd.to_datetime(end_date))
                df = df[mask]

            if len(df) > 0: dataframes[symbol] = df
        except Exception as e:
            logger.error(f"加载文件出错: {e}")
    return dataframes


def ensure_datetime_index(data):
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    return data


def convert_dict_to_dataframe_from_index(stock_dict):
    all_dfs = []
    for symbol, sub_df in stock_dict.items():
        temp_df = sub_df.copy().reset_index()
        temp_df.rename(columns={temp_df.columns[0]: 'timestamp'}, inplace=True)
        temp_df['symbol'] = symbol
        all_dfs.append(temp_df)
    if not all_dfs: return pd.DataFrame()
    big_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    big_df['timestamp'] = pd.to_datetime(big_df['timestamp'])
    big_df.sort_values(by=['symbol', 'timestamp'], inplace=True)
    return big_df.reset_index(drop=True)


import concurrent.futures
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

global_worker_stocks = None


def worker_initializer(full_stocks_data):
    global global_worker_stocks
    global_worker_stocks = full_stocks_data


def get_sliced_stock_context(full_stocks, target_date_str, history_days=150, future_days=30):
    target_dt = pd.to_datetime(target_date_str)
    sliced_stocks = {}
    for symbol, df in full_stocks.items():
        if df is None or df.empty: continue
        mask_past = (df.index <= target_dt)
        mask_future = (df.index > target_dt)
        past_data = df[mask_past].tail(history_days)
        future_data = df[mask_future].head(future_days)
        if not past_data.empty: sliced_stocks[symbol] = pd.concat([past_data, future_data])
    return sliced_stocks


def process_date_standalone_optimized(target_date):
    global global_worker_stocks
    try:
        max_history_days = model_config.FEATURE_NEED_MAX_DAYS + 250
        max_future_days = max(model_config.RETURN_PERIODS) + 5
        small_context_data = get_sliced_stock_context(global_worker_stocks, target_date, max_history_days,
                                                      max_future_days)
        if not small_context_data: return f"{target_date}: 无切片数据"

        target_dt = pd.to_datetime(target_date)
        feature_calc_data = {}
        for symbol, df in small_context_data.items():
            past_df = df[df.index <= target_dt]
            if not past_df.empty: feature_calc_data[symbol] = past_df

        if not feature_calc_data: return f"{target_date}: 截断后无历史数据"

        filtered_stock_data_df = convert_dict_to_dataframe_from_index(feature_calc_data)
        df_sh = feature_calc_data.get(df_sh_symbol)
        df_sz = feature_calc_data.get(df_sz_symbol)

        from src.divergence_detector_v2 import DivergenceDetectorV2
        detector = DivergenceDetectorV2()
        features_pipeline = FeaturePipeline(detector, small_context_data)
        feature_df = features_pipeline.enrich(filtered_stock_data_df, df_sh, df_sz)

        if feature_df is None or feature_df.empty: return f"{target_date}: 无有效背离数据或生成失败"
        return True
    except Exception as e:
        error_msg = f"\n❌ [{target_date}] 严重崩溃: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def process_stocks_batch_parallel_optimized(full_stocks, batch_size=100, max_workers=8, start_date=None, end_date=None):
    unique_timestamps_set = set()
    for df in full_stocks.values():
        if df is not None and not df.empty: unique_timestamps_set.update(df.index)
    unique_timestamps = sorted(list(unique_timestamps_set))
    target_dates = [ts.strftime('%Y-%m-%d') for ts in unique_timestamps]
    filtered_dates = []

    for date_str in target_dates:
        try:
            current_date = datetime.strptime(date_str, '%Y-%m-%d')
            if start_date and current_date < datetime.strptime(start_date, '%Y-%m-%d'): continue
            if end_date and current_date > datetime.strptime(end_date, '%Y-%m-%d'): continue
            filtered_dates.append(date_str)
        except ValueError:
            pass

    current_fp = compute_cache_fingerprint()
    dates_to_process = []
    for target_date in filtered_dates:
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        filename = str(DAILY_FEATURE_DIR / f"realistic_features_{date_obj.strftime('%Y%m%d')}.csv")
        if not is_cache_valid(filename, current_fp):
            dates_to_process.append(target_date)

    if not dates_to_process:
        logger.info("所有日期缓存均有效（指纹匹配），无需重算。")
        return

    logger.info(f"总待处理天数: {len(dates_to_process)} | 进程数: {max_workers}")
    total_processed, all_errors = 0, []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_initializer,
                             initargs=(full_stocks,)) as executor:
        for batch_start in range(0, len(dates_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(dates_to_process))
            batch_dates = dates_to_process[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            logger.info(f"\n开始并行计算批次 {batch_num} (包含 {len(batch_dates)} 天)...")
            batch_errors = []

            futures = {executor.submit(process_date_standalone_optimized, t_date): t_date for t_date in batch_dates}
            with tqdm(total=len(batch_dates), desc=f"批次{batch_num}", unit="天") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not True: batch_errors.append(result)
                    except Exception as e:
                        batch_errors.append(f"进程级崩溃: {str(e)[:100]}")

            total_processed += len(batch_dates)
            all_errors.extend(batch_errors)
            if batch_errors: logger.warning(f"  ⚠️ 本批次有 {len(batch_errors)} 天未生成文件。例: {batch_errors[0]}")

    logger.info(
        f"\n{'=' * 60}\n处理完成！成功生成: {total_processed - len(all_errors)} 天 | 未生成/失败: {len(all_errors)} 天\n{'=' * 60}")


import threading

CACHE = {}
CACHE_LOCK = threading.Lock()
FULL_STOCK_DATA_KEY = "full_stocks"


def get_cached_dataset(dataset_name):
    with CACHE_LOCK:
        if dataset_name not in CACHE:
            CACHE[dataset_name] = load_price_data(str(STOCK_ND_CSV_DIR))
        return CACHE[dataset_name]


def generate_single_day_features(full_stocks, target_date_str):
    logger.info(f"[{target_date_str}] 开始生成单日特征...")
    max_history_days = model_config.FEATURE_NEED_MAX_DAYS + 250
    max_future_days = max(model_config.RETURN_PERIODS) + 5

    small_context = get_sliced_stock_context(full_stocks, target_date_str, history_days=max_history_days,
                                             future_days=max_future_days)
    if not small_context:
        logger.warning(f"[{target_date_str}] 未找到任何有效股票数据切片，请检查数据源日期。")
        return False

    # 手动挂载全局上下文（单线程跑时需要）
    global global_worker_stocks
    global_worker_stocks = full_stocks

    result = process_date_standalone_optimized(target_date_str)
    if result is True:
        logger.info(f"[{target_date_str}] 🎉 单日特征生成成功！文件已保存。")
        return True
    else:
        logger.error(f"[{target_date_str}] ❌ 单日特征生成失败: {result}")
        return False


if __name__ == "__main__":
    logger.info("正在加载全量股票数据到主进程内存...")
    full_stocks_data = get_cached_dataset(FULL_STOCK_DATA_KEY)
    if not full_stocks_data:
        logger.error("全量数据加载失败，请检查数据源！")
        exit(1)

    # ==========================================
    # 模式 A: 跑历史回测（批处理模式）
    # ==========================================
    process_stocks_batch_parallel_optimized(
        full_stocks=full_stocks_data,

        # 【参数调优建议】
        # batch_size: 每次打包分发的天数。设太大主进程切片会卡顿，设太小进度条刷新太快。100~200 是甜点区间。
        batch_size=200,

        # max_workers: 进程数。
        # 强烈建议设置为：你的 CPU 物理核心数 - 1（留一个核心给操作系统和其他软件，防止电脑彻底卡死）。
        # 如果你是 16 核机器，建议设为 12-14。
        max_workers=24,

        # start_date / end_date: 控制回测区间。格式必须是 'YYYY-MM-DD'
        # 如果你要跑全量，两个都设为 None 即可。
        start_date='2010-02-10',
    )

    # ==========================================
    # 模式 B: 跑指定单日（实盘/测试模式）
    # ==========================================
    # target_today = '2025-10-10'  # 或者使用 datetime.now().strftime('%Y-%m-%d') 获取今天日期
    #
    # generate_single_day_features(
    #     full_stocks=full_stocks_data,
    #     target_date_str=target_today
    # )
