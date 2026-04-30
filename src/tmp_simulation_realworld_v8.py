from grid_trading_realworld_v8 import run as real_run
from datetime import datetime
# 设置日志
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import os
from config.settings import STOCK_DATA_DIR, REAL_TRADING_DIR_SIMULATION

# 加载所有股票的行情数据
from grid_trading_simulation_v8 import load_price_data

full_data_dict = load_price_data(str(STOCK_DATA_DIR))
from grid_trading_simulation_v8 import convert_dict_to_dataframe_from_index

full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)
# 检查 full_data_df 的重复情况
full_duplicates = full_data_df.duplicated(subset=['symbol', 'timestamp']).sum()
logger.debug(f"full_data_df 中 (symbol, timestamp) 重复数量: {full_duplicates}")

# 获取所有交易日
trading_dates = sorted(full_data_df['timestamp'].unique())

# 找到起始日期范围内的交易日
start_date_str = "20251013"
end_date_str = "20260408"
start_date = datetime.strptime(start_date_str, "%Y%m%d")
end_date = datetime.strptime(end_date_str, "%Y%m%d")
# 筛选在时间范围内的交易日
filtered_dates = [date for date in trading_dates if start_date <= date <= end_date]
logger.info(f"共有 {len(filtered_dates)} 个交易日需要处理")


data_dir = str( REAL_TRADING_DIR_SIMULATION / 'investment_data')
# 操盘数据中的建议文件
suggestions_file_prefix = os.path.join(data_dir, 'trade_suggestions')
# 操盘数据中的仓位状态
positions_file = os.path.join(data_dir, 'portfolio_positions.csv')
# 操盘数据中的现金状态
cash_file = os.path.join(data_dir, 'portfolio_cash.csv')

for i, predict_date in enumerate(filtered_dates):
    logger.info(f"📅 处理日期: {predict_date.strftime('%Y-%m-%d')}")

    # 找到上一交易日
    if i > 0:
        previous_date = filtered_dates[i-1]
    else:
        # 如果是第一个交易日，找上一个最近的交易日
        previous_dates = [date for date in trading_dates if date < predict_date]
        previous_date = previous_dates[-1]

    # 预测日期的操盘数据生成
    real_run(predict_date, previous_date, REAL_TRADING_DIR_SIMULATION)

    # 读取所需信息（建议、仓位状态和现金状态）
    suggestions_file = f"{suggestions_file_prefix}_{predict_date.strftime('%Y%m%d')}.csv"

    if not os.path.exists(suggestions_file):
        logger.warning(f"无交易建议文件，日期: {predict_date.strftime('%Y-%m-%d')}")
        suggestions_df = pd.DataFrame()
    else:
        # 读取交易建议
        suggestions_df = pd.read_csv(suggestions_file)

    # 读取当天的实际行情数据
    current_day_data = full_data_df[
        full_data_df['timestamp'] == pd.to_datetime(predict_date)].copy()

    # 读取当前仓位状态
    if os.path.exists(positions_file):
        positions_df = pd.read_csv(positions_file)
        # 转换日期列
        date_columns = ['entry_date', 'actual_sell_date']
        for col in date_columns:
            if col in positions_df.columns:
                positions_df[col] = pd.to_datetime(positions_df[col], errors='coerce')
        positions_dict = positions_df.set_index('code').to_dict('index')
    else:
        positions_dict = {}

    # 读取当前现金状态
    cash_df = pd.read_csv(cash_file)
    current_cash = cash_df.iloc[0]['cash']

    # 1. 先处理持仓中的止盈止损检查
    stop_return_amount = 0
    for code, pos in positions_dict.items():
        # 只处理未卖出的持仓
        if pos.get('is_sold') == 'NO':
            # 检查是否有止盈止损价格
            stop_loss_price = pos.get('stop_loss_price')
            take_profit_price = pos.get('take_profit_price')

            if stop_loss_price is not None and take_profit_price is not None:
                # 获取当天的股票数据
                if not current_day_data.empty and code in current_day_data['symbol'].values:
                    stock_data = current_day_data[current_day_data['symbol'] == code].iloc[0]

                    # 检查是否触发止盈止损
                    should_sell = False
                    sell_reason = ""

                    # 检查止损：如果当天最低价触及止损价，触发卖出
                    if stock_data['low'] <= stop_loss_price:
                        should_sell = True
                        sell_reason = "STOP_LOSS"
                        sell_price = min(stop_loss_price, stock_data['open'])  # 用止损价或开盘价中较低者

                    # 检查止盈：如果当天最高价触及止盈价，触发卖出
                    elif stock_data['high'] >= take_profit_price:
                        should_sell = True
                        sell_reason = "TAKE_PROFIT"
                        sell_price = take_profit_price
                        sell_price = max(take_profit_price, stock_data['open'])  # 用止盈价或开盘价中较高者

                    if should_sell:
                        # 计算卖出金额
                        shares = pos['shares']
                        cost = pos['avg_cost']
                        sell_amount = sell_price * shares
                        profit = (sell_price - cost) * shares

                        # 更新现金
                        stop_return_amount += sell_amount

                        # 更新持仓状态
                        positions_dict[code]['actual_sell_price'] = sell_price
                        positions_dict[code]['actual_sell_date'] = predict_date.strftime('%Y-%m-%d %H:%M:%S')
                        positions_dict[code]['is_sold'] = 'YES'

                        logger.info(
                            f"止盈止损触发 {sell_reason} {code}: {shares}股 @ ¥{sell_price:.2f}, 收益: ¥{profit:,.2f}")

    # 2、分离买入和卖出建议
    sell_return_amount = 0
    if suggestions_df is not None and not suggestions_df.empty:
        buy_suggestions = suggestions_df[suggestions_df['action'] == 'BUY']
        sell_suggestions = suggestions_df[suggestions_df['action'] == 'SELL']

        # 2.1处理卖出建议
        for _, sell_row in sell_suggestions.iterrows():
            code = sell_row['code']

            if code not in positions_dict:
                logger.error(f"❌  建议卖出的股票{code}不在持仓中")
                continue

            pos = positions_dict[code]

            # 检查是否已卖出
            if pos.get('is_sold') == 'NO':

                # 先筛选数据
                filtered_data = current_day_data[current_day_data['symbol'] == code]

                # 检查是否有数据
                if filtered_data.empty:
                    logger.warning(f"❌: 没有找到股票代码 {code} 的{predict_date.strftime('%Y-%m-%d')}数据, 今天无法交割")
                else:
                    stock_data = filtered_data.iloc[0]
                    # 获取当前价格（使用收盘价模拟实际卖出）
                    sell_price = stock_data['close']

                    # 计算卖出收益
                    shares = pos['shares']
                    cost = pos['avg_cost']
                    sell_amount = sell_price * shares
                    profit = (sell_price - cost) * shares

                    # 更新现金
                    sell_return_amount += sell_amount

                    # 更新持仓状态
                    positions_dict[code]['actual_sell_price'] = sell_price
                    positions_dict[code]['actual_sell_date'] = predict_date.strftime('%Y-%m-%d %H:%M:%S')
                    positions_dict[code]['is_sold'] = 'YES'

                    logger.info(f"卖出 {code}: {shares}股 @ ¥{sell_price:.2f}, 收益: ¥{profit:,.2f}")

        # 2.2处理买入建议
        for _, buy_row in buy_suggestions.iterrows():
            code = buy_row['code']
            suggested_price = buy_row['suggested_price']
            shares = buy_row['shares']

            # 获取实际买入价格（使用开盘价）
            filtered_data = current_day_data[current_day_data['symbol'] == code]

            # 检查是否有数据
            if filtered_data.empty:
                # TODO 这里就是真实模拟和grid_simulation脚本不一样的地方
                # 如果判定这个股票第二天可以买，而它又正好在后面几天都停盘，这里的模拟逻辑会过滤掉这个股票，而grid_simulation不会。
                logger.error(f"未找到股票 {code} 的数据")
                continue

            stock_data = current_day_data[current_day_data['symbol'] == code].iloc[0]
            if suggested_price < stock_data['low']:
                logger.warning(f"无法买入 {code}: {shares}股 @ suggested_price: ¥{suggested_price:.2f} low: ¥{stock_data['low']:.2f}")
                continue

            # buy_price = min(suggested_price,stock_data['open'])
            buy_price = suggested_price

            # 计算实际成本
            actual_cost = buy_price * shares

            # 更新现金
            current_cash -= actual_cost

            # 添加或更新持仓
            if code in positions_dict:
                # 如果已卖出过，覆盖为新持仓
                positions_dict[code] = {
                    'avg_cost': buy_price,
                    'shares': shares,
                    'entry_date': predict_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'stop_loss_price': None,
                    'take_profit_price': None,
                    'actual_sell_price': None,
                    'actual_sell_date': None,
                    'is_sold' : 'NO',
                }
                logger.warning(f"第二次买入 {code}: {shares}股 @ ¥{buy_price:.2f}, 花费: ¥{actual_cost:,.2f}")
            else:
                # 新持仓
                positions_dict[code] = {
                    'avg_cost': buy_price,
                    'shares': shares,
                    'entry_date': predict_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'stop_loss_price': None,
                    'take_profit_price': None,
                    'actual_sell_price': None,
                    'actual_sell_date': None,
                    'is_sold' : 'NO',
                }

            logger.info(f"买入 {code}: {shares}股 @ ¥{buy_price:.2f}, 花费: ¥{actual_cost:,.2f}")


    # 3、更新保存操作仓位状态文件(包括历史文件)
    if positions_dict:
        positions_list = []
        for code, pos in positions_dict.items():
            if pd.isna(pos.get('entry_date')) or pos.get('entry_date') == '':
                logger.error(f"发现 entry_date 为空的持仓: {pos.get('code')}")
            pos_data = {
                'code': code,
                'avg_cost': pos['avg_cost'],
                'shares': pos['shares'],
                'entry_date': pos.get('entry_date'),
                'stop_loss_price': pos.get('stop_loss_price'),
                'take_profit_price': pos.get('take_profit_price'),
                'actual_sell_price': pos.get('actual_sell_price'),
                'actual_sell_date': pos.get('actual_sell_date'),
                'is_sold': pos.get('is_sold')
            }
            positions_list.append(pos_data)

        positions_df = pd.DataFrame(positions_list)
        active_positions_df = positions_df[positions_df['is_sold'] == 'NO']
        active_positions_df.to_csv(positions_file, index=False)
        sold_positions = positions_df[positions_df['is_sold'] == 'YES']
        if not sold_positions.empty:
            try:
                position_history_file = os.path.join(str(REAL_TRADING_DIR_SIMULATION / 'investment_data'), 'portfolio_positions_history.csv')

                # 如果历史文件存在，追加数据
                if os.path.exists(position_history_file):
                    history_df = pd.read_csv(position_history_file)

                    # 合并新旧数据，避免重复
                    combined_df = pd.concat([history_df, sold_positions], ignore_index=True)

                    # 按卖出日期降序排列，最新的在前
                    if 'actual_sell_date' in combined_df.columns:
                        # 将日期列统一转换为 datetime 类型
                        combined_df['actual_sell_date'] = pd.to_datetime(combined_df['actual_sell_date'], errors='coerce')
                        combined_df = combined_df.sort_values('actual_sell_date', ascending=False)
                        # 将 datetime 转换回字符串以便保存到 CSV
                        combined_df['actual_sell_date'] = combined_df['actual_sell_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    combined_df.to_csv(position_history_file, index=False)
                    logger.info(f"📚 归档 {len(sold_positions)} 条已卖出持仓到历史文件，历史记录总数: {len(combined_df)}")
                else:
                    # 如果 sold_positions 中有 datetime 类型，先转换为字符串
                    if not sold_positions.empty and 'actual_sell_date' in sold_positions.columns:
                        sold_positions = sold_positions.copy()
                        sold_positions['actual_sell_date'] = pd.to_datetime(sold_positions['actual_sell_date']).dt.strftime(
                            '%Y-%m-%d %H:%M:%S')

                    sold_positions.to_csv(position_history_file, index=False)
                    logger.info(f"📚 创建历史文件并归档 {len(sold_positions)} 条已卖出持仓")
            except Exception as e:
                logger.error(f"❌ 归档已卖出持仓失败: {e}")
    else:
        # 清空持仓文件
        pd.DataFrame(columns=['code', 'avg_cost', 'shares', 'entry_date',
                              'stop_loss_price', 'take_profit_price',
                              'actual_sell_price', 'actual_sell_date', 'is_sold']).to_csv(positions_file, index=False)



    # 4、 更新保存操作现金状态文件
    current_cash += stop_return_amount
    current_cash += sell_return_amount
    cash_df = pd.DataFrame([{
        'cash': current_cash,
        'last_run_date': predict_date.strftime('%Y-%m-%d'),
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    cash_df.to_csv(cash_file, index=False)

    logger.info(
        f"更新完成: 现金 ¥{current_cash:,.2f}, 持仓数 {len([p for p in positions_dict.values() if p.get('is_sold') == 'NO'])}")

logger.info("✅ 所有日期处理完成")