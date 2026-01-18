import pandas as pd
import numpy as np
from tqdm import tqdm
import random

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

        # 5. 概率阈值 （下面两个值互相替代）
        self.top_k = 0.08 # 取预测股中的头部

        # 6. 结果变量
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
                self._execute_trade(today, code, 'INTRADAY_STOP_LOSS', sell_price, -pos['shares'], pos)
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
                self._execute_trade(today, code, 'TAKE_PROFIT', sell_price, -pos['shares'], pos)
                codes_to_remove.append(code)
                continue

            # --- 3. 检查时间过期 ---
            if days_held >= self.max_hold_days:
                self._execute_trade(today, code, 'TIME_EXIT', curr_c, -pos['shares'], pos)
                codes_to_remove.append(code)
                continue

        for c in codes_to_remove:
            del self.positions[c]

    def _open_new_positions(self, daily_data, today, next_day):
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return

        # 1. 使用 TOP方式选股
        # 过滤已经持股的和没有预测值的
        candidates = daily_data[
            ~daily_data.index.isin(self.positions.keys()) &
            daily_data['y_pred_proba'].notna()
            ].copy()
        if candidates.empty:
            return
        # 按 y_pred_proba 降序排序
        candidates = candidates.sort_values('y_pred_proba', ascending=False)
        k_value = max(1, int(len(candidates) * self.top_k))
        # 取前k_value个（已经是排序好的）
        top_candidates = candidates.head(k_value)

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
            entry_date = row['entry_date']


            # 不在挂单阶段使用 next_open/next_low 决策。但保存以便开盘结算使用
            next_open = row['next_open']
            next_low = row['next_low']

            # 动态计算本次分配的预算
            # 每次把剩余可用现金按剩余仓位均分，再乘 base_ratio（用户策略比率）
            # NOTE: 经过测试，平均分配仓位的逻辑比按y_pred_proba在top_candidates的比例来分配的收益率高
            budget_per_slot = available_cash / max(1, remaining_slots)
            budget = budget_per_slot * self.base_ratio

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
                'y_pred_proba': row['y_pred_proba'],
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
                'y_pred_proba': order.get('y_pred_proba'),
            }
            self._log(order.get('entry_date'), code, 'OPEN_BUY', actual_price, planned_shares, 0, order.get('y_pred_proba'))

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

    def _execute_trade(self, today, code, action, price, shares, pos):
        revenue = price * abs(shares)
        profit = revenue - (pos['avg_cost'] * abs(shares))
        self.cash += revenue
        self._log(today, code, action, price, shares, profit, pos['y_pred_proba'])

    def _log(self, date, code, action, price, shares, profit, y_pred_proba):
        self.history.append({
            'date': date, 'code': code, 'action': action,
            'price': price, 'shares': shares, 'profit': profit,
            'y_pred_proba': y_pred_proba,
        })

    def _record_daily_asset(self, today, daily_data):
        mkt_val = 0
        for code, pos in self.positions.items():
            price = daily_data.loc[code, 'close'] if code in daily_data.index else pos['avg_cost']
            mkt_val += price * pos['shares']
        self.daily_assets.append({'date': today, 'total': self.cash + mkt_val})
from data_process import data_clean
from config.settings import MODEL_DIR, DATASET_DIR, RESULT_DIR,STOCK_DATA_DIR
from comm_fun import model_config, label_encoding
OUTPUT_DIR = RESULT_DIR / 'simple_run_log'

import os
import predictor_model
import glob
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data(dataset_dir = DATASET_DIR, required_files=None, start_date=None, end_date=None):
    """加载所有文件数据并按时间戳排序"""
    if required_files is None:
        required_files = ["test_set.csv", "validation_set.csv"]
    file_list = []
    for filename in required_files:
        file_path = dataset_dir / filename
        if file_path.exists():
            file_list.append(str(file_path))
        else:
            print(f"警告：文件 {filename} 不存在")

    all_data = []

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
            logger.debug(f"✅ 成功加载: {file_path} ({len(df)} 行)")
        except Exception as e:
            logger.error(f"❌ 加载文件 {file_path} 时出错: {e}")

    if not all_data:
        logger.error(f"❌没有找到有效数据")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    logger.debug(f"✅ 合并后总数据量: {len(combined_df)} 行")

    # 检查必要列是否存在
    required_columns = ['timestamp', model_config.LABEL_COL]
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        logger.error(f"❌缺少必要列: {missing_columns}")
        return None

    # 转换时间戳
    try:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    except Exception as e:
        logger.error(f"❌时间戳转换错误: {e}")
        return None
    # 按时间戳排序
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    logger.debug(f"✅数据时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")


    # BUGFIXED 这里没有对数据做预处理，导致模拟和真实场景不一致
    combined_df = data_clean(combined_df)

    # NOTE: 对ST、次新股和不能交易的进行过滤
    combined_df, _ = label_encoding(combined_df)
    st_df = pd.read_csv(DATASET_DIR / 'st_stocks_list.csv')
    new_df = pd.read_csv(DATASET_DIR / 'new_stocks_list.csv')
    combined_df = combined_df[combined_df['symbol'].str.match(r'^[60]')]
    combined_df = combined_df[~combined_df['symbol'].isin(st_df['ts_code'])]
    combined_df = combined_df[~combined_df['symbol'].isin(new_df['ts_code'])]

    # 时间过滤
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['timestamp'] >= start_date]
        filtered_count = original_count - len(combined_df)
        logger.info(f"过滤掉 {filtered_count} 个开始时间之前的数据点")

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['timestamp'] <= end_date]
        filtered_count = original_count - len(combined_df)
        logger.info(f"过滤掉 {filtered_count} 个结束时间之后的数据点")

    if len(combined_df) > 0:
        logger.debug(f"过滤后数据时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")
        logger.debug(f"最终数据点数: {len(combined_df)} 个")
    else:
        logger.warning(f"警告: 过滤后没有剩余数据")

    return combined_df

def load_price_data(directory_path):
    """
    加载目录下所有 {symbol}_price_data.pkl 文件到内存中
    """
    # 构建文件匹配模式
    pattern = os.path.join(directory_path, "*_price_data.pkl")

    # 查找所有匹配的文件
    file_paths = glob.glob(pattern)

    # 存储所有DataFrame的字典
    dataframes = {}

    for file_path in file_paths:
        try:
            # 从文件名提取symbol
            filename = os.path.basename(file_path)
            symbol = filename.replace("_price_data.pkl", "")

            # 加载pkl文件
            df = pd.read_pickle(file_path)

            # 不在这里计算技术指标，避免数据泄露
            # 技术指标将在按日期处理时实时计算
            dataframes[symbol] = df
            logger.debug(f"成功加载: {filename}, 数据形状: {df.shape}")

        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")

    logger.debug(f"共加载 {len(dataframes)} 个股票数据文件")
    return dataframes

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

def data_process(dataset_dir = DATASET_DIR, required_files=None):
    if required_files is None:
        required_files = ["test_set.csv", "validation_set.csv"]
    logger.info("正在加载数据...")
    raw_df = load_and_prepare_data(dataset_dir = dataset_dir, required_files = required_files)
    # 计算时间范围
    start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
    end_date = (raw_df['timestamp'].max() + pd.Timedelta(days=11)).strftime("%Y%m%d")

    # 检查 full_data_df 的重复情况
    df_duplicates = raw_df.duplicated(subset=['symbol', 'timestamp']).sum()
    logger.debug(f"DF 中 (symbol, timestamp) 重复数量: {df_duplicates}")

    logger.debug("正在去除 raw_df 中的重复数据...")
    before_count_full = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
    after_count_full = len(raw_df)
    logger.debug(f"去重前: {before_count_full} 条, 去重后: {after_count_full} 条, 移除: {before_count_full - after_count_full} 条重复数据")

    # 初始化预测器
    predictor = predictor_model.PriceChangePredictor(model_dir=str(MODEL_DIR))
    # 保存symbol列
    symbol_info = raw_df[
        'symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(
        ['N/A'] * len(raw_df))
    # 使用模型预测概率（排除symbol列）
    pred_probabilities = predictor.predict_proba(raw_df)
    # 直接赋值（顺序不会变）
    df_proba = raw_df.copy()
    df_proba['y_pred_proba'] = pred_probabilities
    # 恢复symbol信息
    df_proba['symbol'] = symbol_info

    full_data_dict = load_price_data(str(STOCK_DATA_DIR))

    full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)

    full_data_df = full_data_df[
        (full_data_df['timestamp'] >= pd.to_datetime(start_date))
        # & (full_data_df['timestamp'] <= pd.to_datetime(end_date))
        ]

    # 检查 full_data_df 的重复情况
    full_duplicates = full_data_df.duplicated(subset=['symbol', 'timestamp']).sum()
    logger.debug(f"full_data_df 中 (symbol, timestamp) 重复数量: {full_duplicates}")

    # 检查 df_proba 的重复情况
    proba_duplicates = df_proba.duplicated(subset=['symbol', 'timestamp']).sum()
    logger.debug(f"df_proba 中 (symbol, timestamp) 重复数量: {proba_duplicates}")

    full_data_df = full_data_df.merge(
        df_proba[['symbol', 'timestamp', 'y_pred_proba']],
        on=['symbol', 'timestamp'],
        how='left'
    )

    # 检查合并后的数据质量
    logger.debug(f"合并后数据量: {len(full_data_df)}")
    logger.debug(f"预测值为NaN的数量: {full_data_df['y_pred_proba'].isna().sum()}")
    logger.debug(f"预测值覆盖率: {1 - full_data_df['y_pred_proba'].isna().mean():.2%}")

    df_result = full_data_df.rename(columns={
        'timestamp': 'date',
        'symbol': 'code',
    })

    df_result['next_open'] = df_result.groupby('code')['open'].shift(-1)
    df_result['next_high'] = df_result.groupby('code')['high'].shift(-1)
    df_result['next_low'] = df_result.groupby('code')['low'].shift(-1)
    df_result['next_close'] = df_result.groupby('code')['close'].shift(-1)
    df_result['entry_date'] = df_result.groupby('code')['date'].shift(-1)

    # 删除没有次日数据的行（最后一天）
    # df_result = df_result.dropna(subset=['next_open'])

    return df_result


# 使用提示：
# 请确保你的 df 里有 'next_open' 列（第二天开盘价格）。
# 如果没有，请在传入 run 之前用 df_result['next_open'] = df_result.groupby('code')['open'].shift(-1) 生成

def simple_run(initial_capital, strategy_name, strategy_params, full_data):
    # 假设你的数据叫 df_result，包含 date, code, open, high, low, close, y_pred_proba


    # ==========================================
    # 第二步：调用策略
    # ==========================================

    # 初始化策略
    # {initial_capital}万本金，最多持仓 {max_positions} 只 (这意味着每只股票重仓 20万，极度聚焦)
    strategy = SmartSniperStrategy(initial_capital=initial_capital, max_positions=5)

    # 这里的参数可以根据你的风险偏好微调：
    strategy.max_positions = strategy_params['max_positions'] #最大持仓
    strategy.base_ratio = strategy_params['base_ratio']  # 开仓比例
    strategy.target_profit = strategy_params['target_profit']  # 止盈
    strategy.max_hold_days = strategy_params['max_hold_days']  # 最长持股
    strategy.hard_stop_loss = strategy_params['hard_stop_loss'] # 止损
    strategy.top_k = strategy_params['top_k']

    # 开始运行
    logger.info("开始回测...")
    trade_log, asset_curve = strategy.run(full_data.copy())

    logger.info("回测结束！")

    # ==========================================
    # 第三步：分析结果
    # ==========================================
    analysis_txt_content = [f"{strategy_name} 回测数据："]

    # 1. 计算最终收益
    final_asset = asset_curve.iloc[-1]['total']
    return_rate = (final_asset - initial_capital) / initial_capital
    logger.info(f"最终资产: {final_asset:,.2f}")
    logger.info(f"最终收益率: {return_rate:.2%}")
    analysis_txt_content.append(f"最终资产: {final_asset:,.2f}")
    analysis_txt_content.append(f"最终收益率: {return_rate:.2%}")

    # 2. 计算最大回撤
    if not asset_curve.empty:
        # 计算历史最高资产
        asset_curve['peak'] = asset_curve['total'].cummax()
        # 计算回撤
        asset_curve['drawdown'] = (asset_curve['total'] - asset_curve['peak']) / asset_curve['peak']
        # 计算最大回撤
        max_drawdown = asset_curve['drawdown'].min()
        logger.info(f"最大回撤: {max_drawdown:.2%}")
        analysis_txt_content.append(f"最大回撤: {max_drawdown:.2%}")

    # 3. 计算夏普比率
    if not asset_curve.empty and len(asset_curve) > 1:
        # 计算每日收益率
        asset_curve['daily_return'] = asset_curve['total'].pct_change().fillna(0)

        # 年化收益率
        total_days = len(asset_curve)
        annual_return = (1 + return_rate) ** (252 / total_days) - 1 if total_days > 0 else 0

        # 计算年化波动率
        daily_std = asset_curve['daily_return'].std()
        annual_volatility = daily_std * np.sqrt(252)

        # 夏普比率（假设无风险利率为0）
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"年化收益率: {annual_return:.2%}")
        analysis_txt_content.append(f"夏普比率: {sharpe_ratio:.4f}")
        analysis_txt_content.append(f"年化收益率: {annual_return:.2%}")

    # 4. 详细交易统计
    if not trade_log.empty:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        asset_curve.to_csv(OUTPUT_DIR / f'simple_run_grid_v6_asset_log_{strategy_name}.csv', index=False)
        trade_log.to_csv(OUTPUT_DIR / f'simple_run_grid_v6_trade_log_{strategy_name}.csv', index=False)

        # 所有开仓动作
        open_actions = ['OPEN_BUY', 'GRID_ADD']
        # 所有平仓动作
        close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT']

        # 统计开仓次数
        open_trades = trade_log[trade_log['action'].isin(open_actions)]
        logger.info(f"总开仓次数: {len(open_trades)}")
        analysis_txt_content.append(f"总开仓次数: {len(open_trades)}")

        # 统计平仓交易
        closed_trades = trade_log[trade_log['action'].isin(close_actions)]

        if len(closed_trades) > 0:
            win_count = len(closed_trades[closed_trades['profit'] > 0])
            total_count = len(closed_trades)
            win_rate = win_count / total_count

            logger.info(f"\n=== 平仓交易统计 ===")
            logger.info(f"总平仓次数: {total_count}")
            logger.info(f"盈利次数: {win_count}")
            logger.info(f"亏损次数: {total_count - win_count}")
            logger.info(f"胜率: {win_rate:.2%}")

            analysis_txt_content.append(f"\n=== 平仓交易统计 ===")
            analysis_txt_content.append(f"总平仓次数: {total_count}")
            analysis_txt_content.append(f"盈利次数: {win_count}")
            analysis_txt_content.append(f"亏损次数: {total_count - win_count}")
            analysis_txt_content.append(f"胜率: {win_rate:.2%}")

            # 按交易类型分析
            logger.info(f"\n=== 按交易类型分析 ===")
            analysis_txt_content.append(f"\n=== 按交易类型分析 ===")
            for action in close_actions:
                action_trades = closed_trades[closed_trades['action'] == action]
                if len(action_trades) > 0:
                    action_win_count = len(action_trades[action_trades['profit'] > 0])
                    action_win_rate = action_win_count / len(action_trades)
                    logger.info(f"  {action}: {len(action_trades)}次, 胜率{action_win_rate:.2%}")
                    analysis_txt_content.append(f"  {action}: {len(action_trades)}次, 胜率{action_win_rate:.2%}")
        else:
            logger.warning("没有完成的平仓交易")

        # analysis_dir = RESULT_DIR / f"portfolio_analysis_charts_v6_{strategy_name}"
        # analysis_txt = analysis_dir / f'simple_run_grid_v6_summary_{strategy_name}.txt'
        # os.makedirs(analysis_dir, exist_ok=True)
        # with open(analysis_txt, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(analysis_txt_content))
    return analysis_txt_content


import concurrent.futures


def run_strategy_with_analysis(name, params, capital, full_data):
    """执行单个策略的完整流程"""
    print(f"正在测试参数组: {name}")
    result = simple_run(initial_capital=capital, strategy_name=name,
               strategy_params=params, full_data=full_data)
    return name, result


def run_concurrent(init_capital, data):
    # 使用线程池并发执行
    max_workers = 12  # 根据CPU核心数调整
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备任务
        futures = []
        for name, params in model_config.STRATEGY_PARAMS_CANDIDATES_V6.items():
            future = executor.submit(
                run_strategy_with_analysis,
                name, params, init_capital, data.copy()
            )
            futures.append(future)

        all_analysis_result = []
        # 等待所有任务完成并处理结果
        for future in concurrent.futures.as_completed(futures):
            try:
                strategy_name, analysis_result = future.result()
                # hold_analyzer(version="v6", param_suffix=strategy_name)
                # profit_analyzer(version="v6", param_suffix=strategy_name)
                # return_analyzer(version="v6", param_suffix=strategy_name)
                # trades_analyzer(version="v6", param_suffix=strategy_name)
                # correlation_analyzer(version="v6", param_suffix=strategy_name)
                analysis_result.append("*" * 60)
                all_analysis_result.extend(analysis_result)
                print(f"参数组 {strategy_name} 测试完成")
            except Exception as e:
                print(f"参数组执行出错: {e}")
        if len(all_analysis_result) > 0:
            analysis_txt = RESULT_DIR / f'simple_run_grid_v6_full_params_analysis.txt'
            os.makedirs(RESULT_DIR, exist_ok=True)
            with open(analysis_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_analysis_result))


if __name__ == "__main__":
    processed_data = data_process(dataset_dir=DATASET_DIR,
                             required_files=[
                                 "test_set.csv",
                                 "validation_set.csv",
                             ])
    run_concurrent(248526, processed_data)