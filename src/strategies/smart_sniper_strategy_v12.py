import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from src.comm_fun import model_config, ALLOCATION_STRATEGY, PROBA_MEAN, PROBA_STD

# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from src.stock_eligibility_filter import StockEligibilityFilter

class SmartSniperStrategyV12:
    """
    V12 策略：波动率自适应止盈止损

    核心改进：
    - 用个股历史波动率（基于真实波幅ATR）动态调整止盈/止损目标
    - 高波动环境：放宽止盈和止损容忍度，让利润奔跑
    - 低波动环境：收缩止盈目标，更快锁定利润

    基于 V8 基础策略构建（参照 V8 的核心逻辑）
    """

    def __init__(self, initial_capital=1000000, max_positions=10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions

        # --- 基础参数（继承 V8 参数55 的优秀配置）---
        self.base_ratio = 1.0            # 仓位比例
        self.target_profit = 0.30        # 基础止盈（V8参数55原值）
        self.hard_stop_loss = -0.10      # 基础止损（V8参数55原值）
        self.max_hold_days = 18          # 最大持仓天数
        self.min_probability = 0.50      # 开仓最低概率

        # --- V12 波动率自适应参数 ---
        self.use_volatility_adaptive = True   # 启用波动率自适应
        self.vol_lookback = 14                 # 波动率计算窗口（N日）
        self.vol_high_thresh = 2.5             # 高波动倍数阈值（相对市场均值）
        self.vol_low_thresh = 0.6              # 低波动倍数阈值（相对市场均值）
        self.vol_profit_mult = 1.5             # 高波动时止盈放大系数
        self.vol_stop_mult = 1.3               # 高波动时止损放宽系数
        self.low_vol_profit_mult = 0.80       # 低波动时止盈收缩系数
        self.use_market_vol = False           # True=用市场整体ATR，False=用个股ATR

        # V10 增强功能参数（默认关闭，可通过策略参数开启）
        self.trailing_offset = 0.08
        self.trailing_pct = 0.20
        self.use_partial_take_profit = False
        self.partial_profit_1 = 0.12
        self.partial_profit_2 = 0.25
        self.partial_ratio_1 = 0.5
        self.use_trailing_stop = False
        self.trailing_stop_pct = 0.20

        self.positions = {}
        self.history = []
        self.daily_assets = []
        self.market_avg_atr = None  # 每日市场ATR均值（运行时会更新）
        self._market_atr_cache: dict = {}  # 预计算的每日市场ATR均值缓存

        # 统一股票过滤器
        self.stock_filter = StockEligibilityFilter()

    def _compute_atr(self, prices_df, symbol, current_date, window=14):
        """计算个股ATR（Average True Range）

        ATR = (Prev_ATR * (N-1) + TR) / N
        TR = max(H-L, |H-PC|, |L-PC|)
        """
        if symbol not in prices_df:
            return None

        df = prices_df[symbol].copy()
        df = df[df.index < current_date]

        if len(df) < window + 1:
            return None

        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr.iloc[-1] if not atr.empty else None

    def _precompute_market_atr(self, today, prices_df):
        """每日预计算市场ATR均值（使用缓存，避免重复计算）"""
        today_str = today.strftime('%Y%m%d') if hasattr(today, 'strftime') else str(today)

        # 检查缓存
        if today_str in self._market_atr_cache:
            return self._market_atr_cache[today_str]

        market_atrs = []
        for code in prices_df:
            atr = self._compute_atr(prices_df, code, today, self.vol_lookback)
            if atr is not None and atr > 0:
                market_atrs.append(atr)
        result = np.mean(market_atrs) if market_atrs else None
        self._market_atr_cache[today_str] = result
        return result

    def _get_volatility_multiplier(self, symbol, current_date, prices_df):
        """
        获取个股波动率相对于市场均值的倍数。
        返回 (>1=高波动, <1=低波动) 的倍数因子。
        """
        if not self.use_volatility_adaptive:
            return 1.0

        # 步骤1：计算个股ATR
        individual_atr = self._compute_atr(prices_df, symbol, current_date, self.vol_lookback)
        if individual_atr is None:
            logger.debug(f"{symbol}无法计算个股ATR, 返回默认值1")
            return 1.0

        # 步骤2：使用预计算的当日市场ATR均值（所有股票，不依赖持仓）
        market_avg_atr = self.market_avg_atr
        if market_avg_atr is None or market_avg_atr == 0:
            logger.error(f"全市场无法计算ATR, 返回默认值1")
            return 1.0

        vol_ratio = individual_atr / market_avg_atr
        return vol_ratio

    def _get_adaptive_targets(self, pos):
        """根据持仓股票的波动率，动态计算止盈和止损目标价"""
        base_cost = pos['avg_cost']

        if not self.use_volatility_adaptive:
            return (
                base_cost * (1 + self.target_profit),
                base_cost * (1 + self.hard_stop_loss)
            )

        # 从持仓中获取波动率倍数
        vol_mult = pos.get('volatility_mult', 1.0)

        # 判断波动环境
        if vol_mult >= self.vol_high_thresh:
            # 高波动：放宽止盈和止损
            effective_profit = self.target_profit * self.vol_profit_mult
            effective_stop = self.hard_stop_loss * self.vol_stop_mult
        elif vol_mult <= self.vol_low_thresh:
            # 低波动：收缩止盈，保持止损不变
            effective_profit = self.target_profit * self.low_vol_profit_mult
            effective_stop = self.hard_stop_loss
        else:
            # 正常波动：使用基础参数
            effective_profit = self.target_profit
            effective_stop = self.hard_stop_loss

        return (
            base_cost * (1 + effective_profit),
            base_cost * (1 + effective_stop)
        )

    def _current_budget(self):
        return self.cash / max(1, (self.max_positions - len(self.positions)))

    def run(self, df, prices_df=None):
        """
        运行回测

        参数:
            df: 主数据DataFrame，包含 date, code, open, high, low, close, y_pred_proba 等列
            prices_df: 价格数据字典 {symbol: DataFrame}，用于计算ATR。如果为None，使用df中的数据计算。
        """
        random.seed(42)
        np.random.seed(42)

        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())

        # 构建 prices_df（如果未提供）：从 df 重建每只股票的价格序列
        if prices_df is None:
            prices_df = {}
            if 'code' in df.columns and 'date' in df.columns:
                for code, group in df.groupby('code'):
                    g = group.sort_values('date')
                    g = g.set_index('date')
                    prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]

        logger.debug(f"--- 启动 V12 狙击回测（波动率自适应）---")
        logger.debug(f"基础止盈: {self.target_profit:.0%}, 基础止损: {self.hard_stop_loss:.0%}")
        logger.debug(f"波动率自适应: {self.use_volatility_adaptive}, 窗口: {self.vol_lookback}日")
        logger.debug(f"高波动倍数阈值: {self.vol_high_thresh}, 止盈放大系数: {self.vol_profit_mult}, 止损放宽系数: {self.vol_stop_mult}")

        for i, today in enumerate(tqdm(dates)):
            if i + 1 < len(dates):
                next_day = dates[i + 1]
            else:
                next_day = None

            # 获取当日切片
            daily_data = df[df['date'] == today].set_index('code')

            # 0. 预计算当日全市场ATR均值（作为波动率比较基准）
            self.market_avg_atr = self._precompute_market_atr(today, prices_df)

            # 1. 更新持仓的波动率倍数（每日更新一次）
            for code, pos in self.positions.items():
                vol_mult = self._get_volatility_multiplier(code, today, prices_df)
                pos['volatility_mult'] = vol_mult

            # 1. 卖出/管理持仓
            self._manage_positions(daily_data, today)

            # 2. 买入新仓
            self._open_new_positions(daily_data, today, next_day, prices_df)

            # 3. 结算当日资产
            self._record_daily_asset(today, daily_data)

        return pd.DataFrame(self.history), pd.DataFrame(self.daily_assets)

    def _manage_positions(self, daily_data, today):
        codes_to_remove = []

        for code, pos in self.positions.items():
            if code not in daily_data.index:
                continue

            row = daily_data.loc[code]
            curr_h = row['high']
            curr_l = row['low']
            curr_o = row['open']
            curr_c = row['close']

            days_held = (today - pos['entry_date']).days

            # T+1 规则：如果是今天买的(days_held=0)，不能卖
            if days_held < 1:
                continue

            # 获取动态止盈/止损目标
            target_price, stop_price = self._get_adaptive_targets(pos)

            # --- 1. 优先检查硬止损 ---
            if curr_o <= stop_price or curr_l <= stop_price:
                sell_price = curr_o if curr_o <= stop_price else stop_price
                sell_price = min(sell_price, curr_h)
                self._execute_trade(today, code, 'INTRADAY_STOP_LOSS', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            # --- 2. 检查止盈 ---
            if curr_h >= target_price:
                sell_price = max(curr_o, target_price)
                sell_price = min(sell_price, curr_h)
                if sell_price > curr_h:
                    sell_price = curr_l
                self._execute_trade(today, code, 'TAKE_PROFIT', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            # --- 3. 检查时间过期 ---
            if days_held >= self.max_hold_days:
                self._execute_trade(today, code, 'TIME_EXIT', curr_c, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            # ===== V10 增强逻辑 =====

            # 1. 分批止盈
            if self.use_partial_take_profit:
                partial_target_1 = pos['avg_cost'] * (1 + self.partial_profit_1)
                partial_target_2 = pos['avg_cost'] * (1 + self.partial_profit_2)

                if curr_h >= partial_target_1 and not pos.get('partial_1_done'):
                    shares_to_sell = int(pos['shares'] * self.partial_ratio_1)
                    if shares_to_sell > 0:
                        self._execute_partial_trade(today, code, 'PARTIAL_TAKE_PROFIT_1',
                                                     curr_o, shares_to_sell, pos, days_held)
                    pos['partial_1_done'] = True

                if curr_h >= partial_target_2:
                    remaining_shares = pos['shares'] - int(pos['shares'] * self.partial_ratio_1)
                    if remaining_shares > 0:
                        self._execute_partial_trade(today, code, 'PARTIAL_TAKE_PROFIT_2',
                                                     curr_o, remaining_shares, pos, days_held)
                    codes_to_remove.append(code)
                    continue

            # 2. 移动止盈
            if curr_h > pos['avg_cost'] * (1 + self.trailing_offset):
                new_trailing_stop = curr_h * (1 - self.trailing_pct)
                pos['trailing_stop'] = max(pos.get('trailing_stop', 0), new_trailing_stop)

            if 'trailing_stop' in pos:
                if curr_c <= pos['trailing_stop']:
                    self._execute_trade(today, code, 'TRAILING_TAKE_PROFIT', curr_c,
                                       -pos['shares'], pos, days_held)
                    codes_to_remove.append(code)
                    continue

            # 3. 跟踪止损
            if self.use_trailing_stop:
                highest_price = pos.get('highest_price', pos['avg_cost'])
                highest_price = max(highest_price, curr_h)
                pos['highest_price'] = highest_price

                if curr_c < highest_price * (1 - self.trailing_stop_pct):
                    self._execute_trade(today, code, 'TRAILING_STOP_LOSS', curr_c,
                                       -pos['shares'], pos, days_held)
                    codes_to_remove.append(code)
                    continue

        for c in codes_to_remove:
            del self.positions[c]

    def _open_new_positions(self, daily_data, today, next_day, prices_df=None):
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return

        candidates = daily_data[
            ~daily_data.index.isin(self.positions.keys()) &
            daily_data['y_pred_proba'].notna() &
            (daily_data['y_pred_proba'] >= self.min_probability) &
            (daily_data['close'] <= model_config.AFFORDABLE_PRICE)
            ].copy()

        # 统一过滤：主板 + ST + 次新股
        trade_date = today.strftime('%Y%m%d')
        candidates = self.stock_filter.filter(candidates, trade_date)

        if candidates.empty:
            return

        top_candidates = candidates.sort_values(
            ['y_pred_proba', 'code'],
            ascending=[False, True]
        ).head(int(available_slots * 2))

        orders = []
        total_frozen = 0.0

        for code, row in top_candidates.iterrows():
            remaining_slots = available_slots - len(orders)
            if remaining_slots <= 0:
                break

            available_cash = self.cash - total_frozen
            if available_cash < 100:
                break

            signal_price = row['close']
            entry_date = row.get('entry_date')
            next_open = row.get('next_open')
            next_low = row.get('next_low')
            proba = row['y_pred_proba']

            # Z-Score 资金分配权重
            weight_factor = 1.0
            if ALLOCATION_STRATEGY == 'z_score':
                z_score = (proba - PROBA_MEAN) / PROBA_STD
                weight_factor = 1.0 + (0.2 * z_score)
                weight_factor = max(0.8, min(1.4, weight_factor))
            elif ALLOCATION_STRATEGY == 'tiered':
                if proba >= 0.72:
                    weight_factor = 1.3
                elif proba >= 0.60:
                    weight_factor = 1.0
                else:
                    weight_factor = 0.8

            base_budget_per_slot = available_cash / max(1, remaining_slots)
            weighted_budget = base_budget_per_slot * weight_factor
            budget = weighted_budget * self.base_ratio

            planned_shares = int(budget / signal_price / 100) * 100
            if planned_shares <= 100:
                if remaining_slots == 1:
                    planned_shares = int(available_cash / signal_price / 100) * 100
                    if planned_shares <= 100:
                        continue
                else:
                    continue
            required_cash = planned_shares * signal_price
            if required_cash > available_cash:
                max_shares = int(available_cash / signal_price / 100) * 100
                if max_shares < 100:
                    continue
                planned_shares = max_shares

            required_cash = planned_shares * signal_price
            total_frozen += required_cash

            # === V12: 为新开仓股计算波动率倍数 ===
            vol_mult = 1.0
            if self.use_volatility_adaptive and prices_df is not None:
                vol_mult = self._get_volatility_multiplier(code, today, prices_df)

            orders.append({
                'code': code,
                'entry_date': entry_date,
                'signal_price': signal_price,
                'planned_shares': planned_shares,
                'frozen_cash': required_cash,
                'next_open': next_open,
                'next_low': next_low,
                'proba': row['y_pred_proba'],
                'volatility_mult': vol_mult,
            })

            logger.debug(f"挂单: {code} 信号价{signal_price:.2f} 计划{planned_shares}股 冻结{required_cash:.2f} 波动率倍数={vol_mult:.2f}")

        if not orders:
            return

        actual_cost = 0.0
        filled_count = 0

        for order in orders:
            code = order['code']
            signal_price = order['signal_price']
            planned_shares = order['planned_shares']
            next_open = order.get('next_open')
            next_low = order.get('next_low')

            if next_open is None or next_low is None:
                logger.debug(f"{code} 无开盘/最低价数据，视为未成交")
                continue

            if next_low > signal_price:
                logger.debug(f"{code} 未成交: next_low {next_low:.2f} > signal_price {signal_price:.2f}")
                continue

            open_pct = (next_open - signal_price) / signal_price
            if open_pct < -0.09:
                logger.debug(f"{code} 未成交(仅仅提示，照常买入): 开盘跌停, next_open {next_open:.2f}")

            if next_day is None:
                logger.warning(f"在{today}，无法买入{code},没有下一天数据了")
                continue
            if (order.get('entry_date') - next_day).days > 0:
                logger.warning(f"在{today}，无法买入{code}，最近开盘日期{next_day}")
                continue

            actual_price = signal_price
            order_cost = actual_price * planned_shares

            self.positions[code] = {
                'avg_cost': actual_price,
                'shares': planned_shares,
                'last_buy_price': actual_price,
                'entry_date': order.get('entry_date'),
                'proba': order['proba'],
                'highest_price': actual_price,
                'trailing_stop': 0,
                'volatility_mult': order['volatility_mult'],  # V12: 记录波动率倍数
            }
            self._log(order.get('entry_date'), code, 'OPEN_BUY', actual_price, planned_shares, 0, 0.0, 0, order['proba'])

            actual_cost += order_cost
            filled_count += 1
            logger.debug(f"成交: {code} 成交价{actual_price:.2f} 数量{planned_shares} 波动率倍数={order['volatility_mult']:.2f}")

        self.cash -= actual_cost
        returned_cash = total_frozen - actual_cost
        logger.debug(f"资金结算 总冻结{total_frozen:.2f} 实际成交{actual_cost:.2f} 返还{returned_cash:.2f}")

    def _execute_trade(self, today, code, action, price, shares, pos, days_held):
        revenue = price * abs(shares)
        profit = revenue - (pos['avg_cost'] * abs(shares))
        profit_pct = (price - pos['avg_cost']) / pos['avg_cost']
        self.cash += revenue
        self._log(today, code, action, price, shares, profit, profit_pct, days_held, pos['proba'])

    def _execute_partial_trade(self, today, code, action, price, shares, pos, days_held):
        revenue = price * abs(shares)
        profit = revenue - (pos['avg_cost'] * abs(shares))
        profit_pct = (price - pos['avg_cost']) / pos['avg_cost']
        self.cash += revenue
        pos['shares'] -= abs(shares)
        self._log(today, code, action, price, shares, profit, profit_pct, days_held, pos['proba'])

    def _log(self, date, code, action, price, shares, profit, profit_pct, days_held, proba):
        self.history.append({
            'date': date, 'code': code, 'action': action,
            'price': price, 'shares': shares,
            'profit': profit, 'profit_pct': profit_pct, 'days_held': days_held,
            'proba': proba
        })

    def _record_daily_asset(self, today, daily_data):
        mkt_val = 0
        for code, pos in self.positions.items():
            price = daily_data.loc[code, 'close'] if code in daily_data.index else pos['avg_cost']
            mkt_val += price * pos['shares']
        self.daily_assets.append({'date': today, 'total': self.cash + mkt_val})
