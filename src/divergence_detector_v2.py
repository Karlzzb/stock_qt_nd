from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from comm_fun import model_config
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DivergenceDetectorV2:
    def detect_daily_divergence(self, data_with_indicators, symbol, current_date):
        """
        只检测指定日期的背离点
        使用截至当前日期的数据进行计算
        """
        try:
            # 确保数据按时间排序
            data_with_indicators = data_with_indicators.sort_index()


            if data_with_indicators is None or len(data_with_indicators) < 100:
                return pd.DataFrame()

            # 检测所有背离点
            all_divergence = self._detect_divergence_by_close_historical(data_with_indicators)

            if len(all_divergence) == 0:
                return pd.DataFrame()

            # 只保留当前日期出现的背离点
            # 将 DataFrame 的 timestamp 转换为 date 进行比较
            all_divergence['date_part'] = all_divergence['timestamp'].dt.date
            daily_divergence = all_divergence[all_divergence['date_part'] == current_date]

            # 删除临时列
            daily_divergence = daily_divergence.drop('date_part', axis=1)

            if len(daily_divergence) == 0:
                return pd.DataFrame()

            # 添加基础信息
            divergence_points = []
            for _, row in daily_divergence.iterrows():
                point = row.to_dict()
                point['symbol'] = symbol
                point['detection_date'] = current_date

                # 添加基础特征（只用历史数据计算）
                point.update(self._calculate_basic_features_historical(point))

                divergence_points.append(point)

            logger.debug(f"股票 {symbol} 在 {current_date} 发现 {len(divergence_points)} 个背离点")
            return pd.DataFrame(divergence_points) if divergence_points else pd.DataFrame()

        except Exception as e:
            logger.error(f"检测 {symbol} 在 {current_date} 的背离时出错: {e}")
            return pd.DataFrame()

    def _calculate_divergence_strength(self, price_decline_pct, macd_increase_pct):
        """计算背离强度"""
        price_strength = min(1.0, abs(price_decline_pct) / 0.1)
        macd_strength = min(1.0, macd_increase_pct / 0.1)
        strength = (price_strength * 0.4 + macd_strength * 0.6)
        return strength

    def _check_volume_signal(self, df, current_idx, prev_idx):
        """检查成交量信号"""
        current_volume = df['volume'].iloc[current_idx]
        prev_volume = df['volume'].iloc[prev_idx]

        if current_volume < prev_volume:
            return 'bullish'
        elif current_volume > prev_volume * 1.5:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_basic_features_historical(self, divergence_point):
        """
        使用历史数据计算基础特征
        """
        features = {}

        try:

            # 价格-MACD变化比率
            denom = divergence_point['price_decline_pct'] if abs(divergence_point['price_decline_pct']) > 1e-6 else 1e-6
            features['price_macd_ratio'] = abs(divergence_point['macd_increase_pct'] / denom)

            # 背离幅度评分
            price_change = abs(divergence_point['price_decline_pct'])
            macd_change = abs(divergence_point['macd_increase_pct'])
            features['divergence_magnitude'] = (price_change + macd_change) / 2

            # 背离确认度
            features['confirmation_score'] = 1 if (
                    divergence_point['price_decline_pct'] < -0.02 and
                    divergence_point['macd_increase_pct'] > 0.01
            ) else 0

            logger.debug(f"基础特征计算完成，共 {len(features)} 个特征")

        except Exception as e:
            logger.error(f"计算基础特征时出错: {e}")

        return features

    def _detect_divergence_by_close_historical(self, df):
        """
        基于历史数据的背离检测
        """
        try:
            # 找到基于收盘价的低点
            close_lows = self._find_close_lows(df)

            if len(close_lows) <= 2:  # lookback_lows=2
                logger.debug("低点数量不足，无法检测背离")
                return pd.DataFrame()

            divergence_points = []
            lookback_lows = 2
            min_macd_change = 0.001

            # 从第lookback_lows+1个低点开始检测
            for i in range(lookback_lows, len(close_lows)):
                current_idx = close_lows[i]
                current_close = df['close'].iloc[current_idx]
                current_macd = df['macd'].iloc[current_idx]

                # 与之前的N个低点逐一对比
                for j in range(1, lookback_lows + 1):
                    prev_idx = close_lows[i - j]

                    # 确保时间间隔合理
                    if current_idx - prev_idx < 5:
                        continue

                    prev_close = df['close'].iloc[prev_idx]
                    prev_macd = df['macd'].iloc[prev_idx]

                    # 过滤 NaN 的 MACD 等值
                    if np.isnan(current_macd) or np.isnan(prev_macd):
                        continue

                    # 背离条件
                    price_new_low = current_close < prev_close
                    macd_higher = current_macd > prev_macd + min_macd_change

                    if price_new_low and macd_higher:
                        price_decline_pct = (current_close - prev_close) / prev_close
                        macd_increase_pct = (current_macd - prev_macd) / abs(prev_macd) if prev_macd != 0 else 0

                        # 计算背离强度
                        divergence_strength = self._calculate_divergence_strength(price_decline_pct, macd_increase_pct)

                        # 检查成交量特征
                        volume_signal = self._check_volume_signal(df, current_idx, prev_idx)

                        divergence_points.append({
                            'timestamp': df.index[current_idx],
                            'prev_time': df.index[prev_idx],
                            'current_idx': current_idx,
                            'prev_idx': prev_idx,
                            'compare_rank': j,
                            'close_current': current_close,
                            'close_previous': prev_close,
                            'macd_current': current_macd,
                            'macd_previous': prev_macd,
                            'price_decline_pct': price_decline_pct,
                            'macd_increase_pct': macd_increase_pct,
                            'formation_period': current_idx - prev_idx,
                            'is_quick_divergence': 1 if (current_idx - prev_idx) < 3 else 0,
                            'divergence_strength': divergence_strength,
                            'volume_signal': volume_signal,
                        })
            logger.debug(f"背离检测完成，找到 {len(divergence_points)} 个背离点")
            return pd.DataFrame(divergence_points)

        except Exception as e:
            logger.error(f"背离检测时出错: {e}")
            return pd.DataFrame()

    def _find_close_lows(self, df, left_window=3, right_window=2):
        """
        【绝对稳定版】局部波谷检测

        参数:
        left_window: 左侧观察窗口（默认3，表示该点必须是过去3天内的最低点，代表前期下跌）
        right_window: 右侧观察窗口（默认2，表示该点必须比随后的2天都低，代表触底反弹被确认）

        注意：
        这里的“右侧”数据完全限制在传入的 df 内部。
        因为传入的 df 已经被 `df.index <= target_date` 严格切断，
        所以这里的运算 100% 免疫未来函数，且生成的低点坐标永远不会随时间推移而漂移。
        """
        if len(df) < left_window + right_window + 1:
            return []

        # 获取收盘价序列的 numpy 数组，计算速度更快
        closes = df['close'].values
        n = len(closes)

        lows = []

        # 遍历所有可能的候选点 (必须留出左侧和右侧的窗口)
        for i in range(left_window, n - right_window):
            current_price = closes[i]

            # 1. 检查是否比左侧所有点都低 (下跌趋势)
            left_prices = closes[i - left_window: i]
            if current_price > np.min(left_prices):
                continue

            # 2. 检查是否比右侧所有点都低 (反弹确认)
            # 这里的 i+right_window+1 绝对不会超过 target_date，因为 n = len(df)
            right_prices = closes[i + 1: i + right_window + 1]
            if current_price >= np.min(right_prices):
                continue

            # 如果同时满足，这就是一个严谨的波谷（V型底）
            lows.append(i)

        # 过滤：防止连续多天价格一模一样导致密集触发，要求低点之间至少相隔一定天数
        final_lows = []
        min_distance = max(left_window, right_window)
        for idx in lows:
            if not final_lows or idx - final_lows[-1] >= min_distance:
                final_lows.append(idx)

        logger.debug(f"找到 {len(final_lows)} 个局部稳定的波谷锚点")
        return final_lows

