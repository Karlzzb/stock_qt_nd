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

class SmartSniperStrategy:
    def __init__(self, initial_capital=1000000, max_positions=10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions

        # --- 核心优化参数 ---
        # 1. 激进底仓：模型准，就敢重仓。建议 0.8 或 1.0
        self.base_ratio = 0.8

        # 3. 快速止盈
        self.target_profit = 0.08  # 止盈

        # 4. 风控
        self.hard_stop_loss = -0.05  # 止损
        self.max_hold_days = 10  # 持仓时长

        # 5. 概率阈值（新增）
        self.min_probability = 0.60  # 开仓最低概率

        self.positions = {}
        self.history = []
        self.daily_assets = []

    def _current_budget(self):
        return self.cash / max(1, (self.max_positions - len(self.positions)))

    def run(self, df):

        # 上游模型预测带随机性
        # 参数搜索顺序使用随机采样
        # 结果都会漂移。
        random.seed(42)
        np.random.seed(42)


        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())

        logger.debug(f"--- 启动狙击回测 ---")
        logger.debug(f"底仓比例: {self.base_ratio:.0%}")

        for i, today in enumerate(tqdm(dates)):
            if i + 1 < len(dates):
                next_day = dates[i + 1]
            else:
                next_day = None  # 或者处理最后一个元素的情况

            # 获取当日切片
            daily_data = df[df['date'] == today].set_index('code')

            # 1. 卖出/管理持仓
            self._manage_positions(daily_data, today)

            # 2. 买入新仓（下一个交易日的开仓，所以先卖出/管理持仓）
            self._open_new_positions(daily_data, today, next_day)

            # 3. 结算当日资产
            self._record_daily_asset(today, daily_data)

        return pd.DataFrame(self.history), pd.DataFrame(self.daily_assets)

    def _manage_positions(self, daily_data, today):
        codes_to_remove = []

        for code, pos in self.positions.items():
            if code not in daily_data.index:
                continue

            row = daily_data.loc[code]
            # 数据容错
            curr_h = row['high']
            curr_l = row['low']
            curr_o = row['open']
            curr_c = row['close']

            days_held = (today - pos['entry_date']).days

            # T+1 规则：如果是今天买的(days_held=0)，不能卖
            if days_held < 1:
                continue


            # --- 1. 优先检查硬止损 ---
            # 如果开盘或盘中触及止损
            stop_price = pos['avg_cost'] * (1 + self.hard_stop_loss)
            if curr_o <= stop_price or curr_l <= stop_price:
                # 开盘就跌破：按开盘价止损
                # 盘中跌破：按止损价止损
                sell_price = curr_o if curr_o <= stop_price else stop_price
                sell_price = min(sell_price, curr_h)  # 安全限制
                self._execute_trade(today, code, 'INTRADAY_STOP_LOSS', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)  # 重要：添加这行！
                continue

            # --- 2. 检查止盈 (Sell Logic) ---
            # 逻辑：只要 High 碰到了目标价，就算成交
            # 实际上限：如果开盘价就直接超过了目标价，那就按开盘价止盈（赚更多）
            target_price = pos['avg_cost'] * (1 + self.target_profit)

            if curr_h >= target_price:
                # 确定卖出价格：取 (开盘价, 目标价) 的较大值，但不能超过最高价
                # 这种逻辑最符合实盘挂单
                sell_price = max(curr_o, target_price)
                sell_price = min(sell_price, curr_h)  # 修正：不能超过当日最高
                # 防止止损价高于今日最高价（极低概率，但逻辑要严密）
                if sell_price > curr_h: sell_price = curr_l
                self._execute_trade(today, code, 'TAKE_PROFIT', sell_price, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

            # --- 3. 检查时间过期 ---
            if days_held >= self.max_hold_days:
                self._execute_trade(today, code, 'TIME_EXIT', curr_c, -pos['shares'], pos, days_held)
                codes_to_remove.append(code)
                continue

        for c in codes_to_remove:
            del self.positions[c]

    def _open_new_positions(self, daily_data, today, next_day):
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return

        # 1. 选股过滤
        candidates = daily_data[
            ~daily_data.index.isin(self.positions.keys()) &
            daily_data['y_pred_proba'].notna() &
            (daily_data['y_pred_proba'] >= self.min_probability) &
            # FIXED BUG  这里过滤，而不在数据预处理过滤，避免simulation与 real_world 执行不一致的问题。
            (daily_data['close'] <= model_config.AFFORDABLE_PRICE)
            ].copy()

        if candidates.empty:
            return

        top_candidates = candidates.sort_values(
            ['y_pred_proba', 'code'],
            ascending=[False, True]
        ).head(int(available_slots * 2))

        # 2. 挂单阶段: 不扣 self.cash, 只冻结资金（total_frozen）
        orders = []
        total_frozen = 0.0

        for code, row in top_candidates.iterrows():
            remaining_slots = available_slots - len(orders)
            if remaining_slots <= 0:
                break

            # 可用于分配的真实可用现金 = 当前现金 - 已冻结资金
            available_cash = self.cash - total_frozen
            if available_cash < 100:  # 现金太少，跳出
                break

            signal_price = row['close']
            entry_date = row.get('entry_date')


            # 不在挂单阶段使用 next_open/next_low 决策。但保存以便开盘结算使用
            next_open = row.get('next_open')
            next_low = row.get('next_low')

            # 获取预测概率 (确保列名正确)
            proba = row['y_pred_proba']

            # --------------------------------------------------------------------------
            # [新增核心逻辑] 计算资金分配权重系数
            # --------------------------------------------------------------------------
            weight_factor = 1.0  # 默认为1.0 (即平均分配)

            if ALLOCATION_STRATEGY == 'z_score':
                # 方案一：Z-Score 统计加权 (推荐)
                # 逻辑：利用均值和标准差判断稀缺性。每增加1个标准差，仓位增加20%
                z_score = (proba - PROBA_MEAN) / PROBA_STD
                weight_factor = 1.0 + (0.2 * z_score)
                # 风险控制：限制系数在 [0.8, 1.4] 之间，防止过度偏离
                weight_factor = max(0.8, min(1.4, weight_factor))
            elif ALLOCATION_STRATEGY == 'tiered':
                # 方案二：分箱阶梯法
                if proba >= 0.72:  # > mean + 1std
                    weight_factor = 1.3
                elif proba >= 0.60:  # 核心区
                    weight_factor = 1.0
                else:  # < 0.60 基础区
                    weight_factor = 0.8

            # --------------------------------------------------------------------------
            # [修改资金计算] 应用权重系数
            # --------------------------------------------------------------------------
            # 基础平均预算
            base_budget_per_slot = available_cash / max(1, remaining_slots)

            # 应用权重系数 (核心修改点)
            weighted_budget = base_budget_per_slot * weight_factor

            # 最终预算 (乘用户策略比率)
            budget = weighted_budget * self.base_ratio

            # --------------------------------------------------------------------------
            # 后续原有逻辑保持不变
            # --------------------------------------------------------------------------

            # 以 signal_price 作为限价挂单价格（挂单阶段不知道 next_open）
            planned_shares = int(budget / signal_price / 100) * 100
            if planned_shares <= 100:
                # 如果本slot预算买不到100股，尝试把整个剩余可用现金用在这个slot（若只剩1个slot）
                if remaining_slots == 1:
                    planned_shares = int(available_cash / signal_price / 100) * 100
                    if planned_shares <= 100:
                        continue
                else:
                    continue
            required_cash = planned_shares * signal_price
            # 再次安全检查：如果剩余可用现金不够，降级planned_shares
            if required_cash > available_cash:
                max_shares = int(available_cash / signal_price / 100) * 100
                if max_shares < 100:
                    continue
                planned_shares = max_shares

            required_cash = planned_shares * signal_price
            # 冻结资金（不从 self.cash 扣除）
            total_frozen += required_cash

            orders.append({
                'code': code,
                'entry_date': entry_date,
                'signal_price': signal_price,
                'planned_shares': planned_shares,
                'frozen_cash': required_cash,
                'next_open': next_open,
                'next_low': next_low,
                'proba': row['y_pred_proba'],
            })

            logger.debug(f"挂单: {code} 信号价{signal_price:.2f} 计划{planned_shares}股 冻结{required_cash:.2f}")

        if not orders:
            return

        # 3. 成交阶段：开盘后根据 next_open / next_low 判断成交与否
        actual_cost = 0.0
        filled_count = 0

        for order in orders:
            code = order['code']
            signal_price = order['signal_price']
            planned_shares = order['planned_shares']
            next_open = order.get('next_open')
            next_low = order.get('next_low')

            # 若缺少开盘或最低价数据，认为无法成交（或按策略决定）
            if next_open is None or next_low is None:
                logger.debug(f"{code} 无开盘/最低价数据，视为未成交")
                continue

            # 条件1: 次日最低价必须 <= signal_price 才可能成交（这种应该不会发生）
            if next_low > signal_price:
                logger.debug(f"{code} 未成交: next_low {next_low:.2f} > signal_price {signal_price:.2f}")
                continue

            # 条件2: 如果开盘跌停（跌幅超过或等于 9%），不能成交
            open_pct = (next_open - signal_price) / signal_price
            if open_pct < -0.09:
                logger.debug(f"{code} 未成交(仅仅提示，照常买入): 开盘跌停, next_open {next_open:.2f}")
                # 跌停也买入（开盘前挂单，无法控制）
                # continue

            # 条件3: 如果第二天没开盘也不买入，匹配真实场景TODO
            if next_day is None:
                logger.warning(f"在{today}，无法买入{code},没有下一天数据了")
                continue
            if (order.get('entry_date') - next_day).days > 0:
                logger.warning(f"在{today}，无法买入{code}，最近开盘日期{next_day}")
                continue

            # NODE 成交价格规则（这里不知道，挂高后按照什么成交，所以按照高价成交逻辑模拟）
            # actual_price = next_open if next_open < signal_price else signal_price
            actual_price = signal_price
            order_cost = actual_price * planned_shares

            # 更新仓位与日志
            self.positions[code] = {
                'avg_cost': actual_price,
                'shares': planned_shares,
                'last_buy_price': actual_price,
                'entry_date': order.get('entry_date'),
                'proba': order['proba'],
            }
            self._log(order.get('entry_date'), code, 'OPEN_BUY', actual_price, planned_shares, 0, 0.0, 0, order['proba'])

            actual_cost += order_cost
            filled_count += 1

            logger.debug(f"成交: {code} 成交价{actual_price:.2f} 数量{planned_shares} 成交额{order_cost:.2f}")

        # 4. 结算资金: 扣除实际成交金额
        self.cash -= actual_cost

        returned_cash = total_frozen - actual_cost
        logger.debug(f"资金结算 总冻结{total_frozen:.2f} 实际成交{actual_cost:.2f} 返还{returned_cash:.2f}")

        # 5. 订单统计: 用 orders 的字段来统计真实成交数
        total_orders = len(orders)
        # filled_count 已计算
        logger.debug(f"订单统计 总下单{total_orders}笔 成功成交{filled_count}笔")

    def _execute_trade(self, today, code, action, price, shares, pos, days_held):
        revenue = price * abs(shares)
        profit = revenue - (pos['avg_cost'] * abs(shares))
        profit_pct = (price - pos['avg_cost']) / pos['avg_cost']
        self.cash += revenue
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
