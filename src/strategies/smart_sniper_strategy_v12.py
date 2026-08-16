import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from src.comm_fun import model_config, ALLOCATION_STRATEGY, PROBA_MEAN, PROBA_STD
from src.stock_eligibility_filter import StockEligibilityFilter


class PrecomputedATR:
    def __init__(self, prices_df: dict, dates: list, lookbacks: list):
        t0 = time.time()
        self.lookbacks = lookbacks
        self._dates = [pd.to_datetime(d) for d in dates]
        self._date_to_pos = {d: i for i, d in enumerate(self._dates)}

        self._data = {}
        codes = list(prices_df.keys())
        n = len(codes)

        for i, code in enumerate(codes):
            df = prices_df[code].sort_index()
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)

            prev_close = np.empty_like(close)
            prev_close[0] = close[0]
            prev_close[1:] = close[:-1]
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            self._data[code] = {}
            stock_dates = [pd.to_datetime(d) for d in df.index]

            for lb in lookbacks:
                cs = np.cumsum(tr)
                cs_shifted = np.concatenate([[0.0], cs[:-1]])
                atr_arr = np.full(len(self._dates), np.nan, dtype=np.float64)

                for idx, s_date in enumerate(stock_dates):
                    global_pos = self._date_to_pos.get(s_date)
                    if global_pos is not None and idx >= lb - 1:
                        window_sum = cs[idx] - cs_shifted[idx - lb + 1]
                        atr_arr[global_pos] = window_sum / lb

                self._data[code][lb] = atr_arr

        logger.info(f"ATR预计算完成: {n} 只股票 × {len(lookbacks)} 个窗口, 耗时 {time.time() - t0:.1f}s")

    def get_atr(self, code: str, lookback: int, date, default=None):
        pos = self._date_to_pos.get(pd.to_datetime(date))
        if pos is None or pos == 0:
            return default

        atr_arr = self._data.get(code, {}).get(lookback)
        if atr_arr is None:
            return default

        val = atr_arr[pos - 1]
        return float(val) if not np.isnan(val) else default

    def market_avg_atr(self, date, lookback: int, codes: list) -> float | None:
        pos = self._date_to_pos.get(pd.to_datetime(date))
        if pos is None or pos == 0:
            return None

        target_pos = pos - 1
        atr_vals = []
        for code in codes:
            atr_arr = self._data.get(code, {}).get(lookback)
            if atr_arr is not None:
                v = atr_arr[target_pos]
                if not np.isnan(v) and v > 0:
                    atr_vals.append(v)
        return np.mean(atr_vals) if atr_vals else None


class SmartSniperStrategyV12:
    def __init__(self, initial_capital=1000000, max_positions=10, st_preloaded=None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions

        self.base_ratio = 1.0
        self.target_profit = 0.30
        self.hard_stop_loss = -0.10
        self.max_hold_days = 18
        self.min_probability = 0.50

        self.use_volatility_adaptive = True
        self.vol_lookback = 14
        self.vol_high_thresh = 2.5
        self.vol_low_thresh = 0.6
        self.vol_profit_mult = 1.5
        self.vol_stop_mult = 1.3
        self.low_vol_profit_mult = 0.80
        self.use_market_vol = False

        self.trailing_offset = 0.08
        self.trailing_pct = 0.20
        self.use_partial_take_profit = False
        self.partial_profit_1 = 0.12
        self.partial_profit_2 = 0.25
        self.partial_ratio_1 = 0.5
        self.use_trailing_stop = False
        self.trailing_stop_pct = 0.20

        self.atr_cache = None
        self.positions = {}
        self.history = []
        self.daily_assets = []
        self.market_avg_atr = None
        self._market_atr_cache: dict = {}

        self.stock_filter = StockEligibilityFilter(
            filter_main_board=True,
            filter_st=True,
            filter_new_stock=True,
            st_preloaded=st_preloaded)

    def _compute_atr(self, prices_df, symbol, current_date, window=14):
        if self.atr_cache:
            return self.atr_cache.get_atr(symbol, int(window), current_date)

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
        atr = tr.rolling(window=int(window)).mean()

        return atr.iloc[-1] if not atr.empty else None

    def _precompute_market_atr(self, today, prices_df):
        today_str = today.strftime('%Y%m%d') if hasattr(today, 'strftime') else str(today)

        if today_str in self._market_atr_cache:
            return self._market_atr_cache[today_str]

        if self.atr_cache:
            codes = list(prices_df.keys())
            result = self.atr_cache.market_avg_atr(today, self.vol_lookback, codes)
        else:
            market_atrs = []
            for code in prices_df:
                atr = self._compute_atr(prices_df, code, today, self.vol_lookback)
                if atr is not None and atr > 0:
                    market_atrs.append(atr)
            result = np.mean(market_atrs) if market_atrs else None

        self._market_atr_cache[today_str] = result
        return result

    def _get_volatility_multiplier(self, symbol, current_date, prices_df):
        if not self.use_volatility_adaptive:
            return 1.0
        individual_atr = self._compute_atr(prices_df, symbol, current_date, self.vol_lookback)
        if individual_atr is None:
            return 1.0
        market_avg_atr = self.market_avg_atr
        if market_avg_atr is None or market_avg_atr == 0:
            return 1.0
        return individual_atr / market_avg_atr

    def _get_adaptive_targets(self, pos):
        base_cost = pos['avg_cost']
        if not self.use_volatility_adaptive:
            return (base_cost * (1 + self.target_profit), base_cost * (1 + self.hard_stop_loss))

        vol_mult = pos.get('volatility_mult', 1.0)
        if vol_mult >= self.vol_high_thresh:
            effective_profit = self.target_profit * self.vol_profit_mult
            effective_stop = self.hard_stop_loss * self.vol_stop_mult
        elif vol_mult <= self.vol_low_thresh:
            effective_profit = self.target_profit * self.low_vol_profit_mult
            effective_stop = self.hard_stop_loss
        else:
            effective_profit = self.target_profit
            effective_stop = self.hard_stop_loss

        return (base_cost * (1 + effective_profit), base_cost * (1 + effective_stop))

    def run(self, df, prices_df=None, show_progress=False, _timing_hook=None):
        random.seed(42)
        np.random.seed(42)

        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())

        if prices_df is None:
            prices_df = {}
            if 'code' in df.columns and 'date' in df.columns:
                for code, group in df.groupby('code'):
                    g = group.sort_values('date')
                    g = g.set_index('date')
                    prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]

        # 预构建日期索引：O(N) 一次性代价，每日 O(1) 查找，替代每日全表 boolean 扫描
        daily_index: dict = {d: grp.set_index('code') for d, grp in df.groupby('date')}

        logger.debug(f"--- 启动 V12 狙击回测（波动率自适应）---")

        # 恢复进度条支持，根据 show_progress 开关决定是否显示
        iterator = dates
        if show_progress:
            iterator = tqdm(dates, desc="策略每日回测中", leave=True)

        for i, today in enumerate(iterator):

            if i + 1 < len(dates):
                next_day = dates[i + 1]
            else:
                next_day = None

            # O(1) 字典查找，替代 O(N) boolean 扫描
            daily_data = daily_index[today]
            self.market_avg_atr = self._precompute_market_atr(today, prices_df)

            for code, pos in self.positions.items():
                vol_mult = self._get_volatility_multiplier(code, today, prices_df)
                pos['volatility_mult'] = vol_mult

            self._manage_positions(daily_data, today)
            self._open_new_positions(daily_data, today, next_day, prices_df)
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

            if days_held < 1:
                continue

            target_price, stop_price = self._get_adaptive_targets(pos)

            if curr_o <= stop_price or curr_l <= stop_price:
                sell_price = curr_o if curr_o <= stop_price else stop_price
                sell_price = min(sell_price, curr_h)
                self._execute_trade(today, code, 'INTRADAY_STOP_LOSS', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            if curr_h >= target_price:
                sell_price = max(curr_o, target_price)
                sell_price = min(sell_price, curr_h)
                if sell_price > curr_h:
                    sell_price = curr_l
                self._execute_trade(today, code, 'TAKE_PROFIT', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            if days_held >= self.max_hold_days:
                self._execute_trade(today, code, 'TIME_EXIT', curr_c, -pos['shares'], pos, days_held)
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

        for row in top_candidates.itertuples(name='Row'):
            code = row.Index
            remaining_slots = available_slots - len(orders)
            if remaining_slots <= 0:
                break

            available_cash = self.cash - total_frozen
            if available_cash < 100:
                break

            signal_price = row.close
            entry_date = getattr(row, 'entry_date', None)
            next_open = getattr(row, 'next_open', None)
            next_low = getattr(row, 'next_low', None)
            proba = row.y_pred_proba

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
                'proba': row.y_pred_proba,
                'volatility_mult': vol_mult,
            })

        if not orders:
            return

        actual_cost = 0.0
        for order in orders:
            code = order['code']
            signal_price = order['signal_price']
            planned_shares = order['planned_shares']
            next_open = order.get('next_open')
            next_low = order.get('next_low')

            if next_open is None or next_low is None or next_low > signal_price:
                continue
            if next_day is None or (order.get('entry_date') - next_day).days > 0:
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
                'volatility_mult': order['volatility_mult'],
            }
            self._log(order.get('entry_date'), code, 'OPEN_BUY', actual_price, planned_shares, 0, 0.0, 0,
                      order['proba'])
            actual_cost += order_cost

        self.cash -= actual_cost

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