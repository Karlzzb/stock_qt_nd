from grid_trading_realworld_v12 import run as real_run
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import os
from config.settings import STOCK_ND_CSV_DIR, REAL_TRADING_DIR_SIMULATION
from grid_trading_simulation_v12 import load_price_data, convert_dict_to_dataframe_from_index

# 核心优化：只在最外层加载一次庞大的价格库，传入给实盘模拟跑循环，拒绝每天重复卡顿
logger.info("⏳ 正在加载全局基础行情数据...")
full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)

prices_df_dict = {}
logger.info("⏳ 正在构建历史 K 线字典用于极速 ATR 计算...")
for code, group in full_data_df.groupby('symbol'):
    g = group.sort_values('timestamp').set_index('timestamp')
    prices_df_dict[code] = g[['open', 'high', 'low', 'close', 'volume']]
logger.info("✅ 数据加载完成！开始极速沙盒模拟。")

trading_dates = sorted(full_data_df['timestamp'].unique())
start_date = datetime.strptime("20251001", "%Y%m%d")
end_date = datetime.strptime("20260407", "%Y%m%d")
filtered_dates = [date for date in trading_dates if start_date <= date <= end_date]

data_dir = str(REAL_TRADING_DIR_SIMULATION / 'investment_data')
suggestions_file_prefix = os.path.join(data_dir, 'trade_suggestions')
positions_file = os.path.join(data_dir, 'portfolio_positions.csv')
cash_file = os.path.join(data_dir, 'portfolio_cash.csv')

for i, predict_date in enumerate(filtered_dates):
    logger.info(f"========== 📅 模拟人工操作日期: {predict_date.strftime('%Y-%m-%d')} ==========")

    if i > 0:
        previous_date = filtered_dates[i - 1]
    else:
        previous_dates = [date for date in trading_dates if date < predict_date]
        previous_date = previous_dates[-1]

    # 将全局行情字典透传给实盘模块，让其飞速计算 V12 波动率
    real_run(predict_date, previous_date, REAL_TRADING_DIR_SIMULATION, prices_df_dict)

    suggestions_file = f"{suggestions_file_prefix}_{predict_date.strftime('%Y%m%d')}.csv"
    if not os.path.exists(suggestions_file):
        suggestions_df = pd.DataFrame()
    else:
        suggestions_df = pd.read_csv(suggestions_file)

    current_day_data = full_data_df[full_data_df['timestamp'] == pd.to_datetime(predict_date)].copy()

    if os.path.exists(positions_file):
        positions_df = pd.read_csv(positions_file)
        date_columns = ['entry_date', 'actual_sell_date']
        for col in date_columns:
            if col in positions_df.columns:
                positions_df[col] = pd.to_datetime(positions_df[col], errors='coerce')
        positions_dict = positions_df.set_index('code').to_dict('index')
    else:
        positions_dict = {}

    cash_df = pd.read_csv(cash_file)
    current_cash = cash_df.iloc[0]['cash']

    stop_return_amount = 0
    sell_return_amount = 0

    # 1. 人工盯盘：检查持仓是否触及止盈止损（严格遵守 T+1 规则）
    for code, pos in positions_dict.items():
        if pos.get('is_sold') == 'NO':

            # 【重要修复】检查 T+1 规则：如果是今天或以后买的，今天绝对不能卖
            entry_date_str = pos.get('entry_date')
            if pd.notna(entry_date_str):
                entry_date_ts = pd.to_datetime(entry_date_str)
                if (predict_date.date() - entry_date_ts.date()).days < 1:
                    continue

            stop_loss_price = pos.get('stop_loss_price')
            take_profit_price = pos.get('take_profit_price')

            if pd.notna(stop_loss_price) and pd.notna(take_profit_price):
                if not current_day_data.empty and code in current_day_data['symbol'].values:
                    stock_data = current_day_data[current_day_data['symbol'] == code].iloc[0]
                    should_sell = False
                    sell_reason = ""
                    sell_price = 0

                    # 逻辑与 V12 严格对齐：先查硬止损，再查止盈
                    if stock_data['low'] <= stop_loss_price:
                        should_sell = True
                        sell_reason = "STOP_LOSS (自适应)"
                        sell_price = min(stop_loss_price, stock_data['open'])
                    elif stock_data['high'] >= take_profit_price:
                        should_sell = True
                        sell_reason = "TAKE_PROFIT (自适应)"
                        sell_price = max(take_profit_price, stock_data['open'])

                    if should_sell:
                        shares = pos['shares']
                        profit = (sell_price - pos['avg_cost']) * shares
                        stop_return_amount += sell_price * shares

                        positions_dict[code]['actual_sell_price'] = sell_price
                        positions_dict[code]['actual_sell_date'] = predict_date.strftime('%Y-%m-%d %H:%M:%S')
                        positions_dict[code]['is_sold'] = 'YES'
                        logger.info(
                            f"⚡ 盘中触发 {sell_reason} {code}: {shares}股 @ ¥{sell_price:.2f}, 收益: ¥{profit:,.2f}")

    # 2. 人工执行：早盘接收软件建议，挂单买卖
    if not suggestions_df.empty:
        buy_suggestions = suggestions_df[suggestions_df['action'] == 'BUY']
        sell_suggestions = suggestions_df[suggestions_df['action'] == 'SELL']

        # 2.1 执行时间到期强制卖出建议
        for _, sell_row in sell_suggestions.iterrows():
            code = sell_row['code']
            if code not in positions_dict: continue
            pos = positions_dict[code]

            if pos.get('is_sold') == 'NO':
                filtered_data = current_day_data[current_day_data['symbol'] == code]
                if not filtered_data.empty:
                    stock_data = filtered_data.iloc[0]
                    sell_price = stock_data['close']  # 模拟尾盘出清
                    shares = pos['shares']

                    sell_return_amount += sell_price * shares
                    positions_dict[code]['actual_sell_price'] = sell_price
                    positions_dict[code]['actual_sell_date'] = predict_date.strftime('%Y-%m-%d %H:%M:%S')
                    positions_dict[code]['is_sold'] = 'YES'
                    logger.info(
                        f"⏳ 时间到期清仓 {code}: {shares}股 @ ¥{sell_price:.2f}, 收益: ¥{(sell_price - pos['avg_cost']) * shares:,.2f}")

        # 2.2 执行买入建仓建议
        for _, buy_row in buy_suggestions.iterrows():
            code = buy_row['code']
            suggested_price = buy_row['suggested_price']
            shares = buy_row['shares']
            vol_mult = buy_row.get('volatility_mult', 1.0)  # 获取 V12 传来的波动力倍数

            filtered_data = current_day_data[current_day_data['symbol'] == code]
            if filtered_data.empty: continue

            stock_data = filtered_data.iloc[0]
            if suggested_price < stock_data['low']:
                logger.warning(
                    f"⏩ 无法买入 {code}: 最低价 ¥{stock_data['low']:.2f} 够不到限价单 ¥{suggested_price:.2f}")
                continue

            buy_price = suggested_price
            actual_cost = buy_price * shares
            current_cash -= actual_cost

            positions_dict[code] = {
                'avg_cost': buy_price,
                'shares': shares,
                'entry_date': predict_date.strftime('%Y-%m-%d %H:%M:%S'),
                'stop_loss_price': None,
                'take_profit_price': None,
                'actual_sell_price': None,
                'actual_sell_date': None,
                'is_sold': 'NO',
                'volatility_mult': vol_mult  # 忠实记录，传给第二天计算使用
            }
            logger.info(f"🛒 成功买入 {code}: {shares}股 @ ¥{buy_price:.2f}, 波动倍数: {vol_mult:.2f}x")

    # 3. 盘后清算与存盘
    if positions_dict:
        positions_list = []
        for code, pos in positions_dict.items():
            pos_data = {
                'code': code, 'avg_cost': pos['avg_cost'], 'shares': pos['shares'],
                'entry_date': pos.get('entry_date'), 'stop_loss_price': pos.get('stop_loss_price'),
                'take_profit_price': pos.get('take_profit_price'), 'actual_sell_price': pos.get('actual_sell_price'),
                'actual_sell_date': pos.get('actual_sell_date'), 'is_sold': pos.get('is_sold'),
                'volatility_mult': pos.get('volatility_mult', 1.0)
            }
            positions_list.append(pos_data)

        positions_df = pd.DataFrame(positions_list)
        active_positions_df = positions_df[positions_df['is_sold'] == 'NO']
        active_positions_df.to_csv(positions_file, index=False)

        sold_positions = positions_df[positions_df['is_sold'] == 'YES']
        if not sold_positions.empty:
            try:
                position_history_file = os.path.join(data_dir, 'portfolio_positions_history.csv')
                if os.path.exists(position_history_file):
                    history_df = pd.read_csv(position_history_file)
                    combined_df = pd.concat([history_df, sold_positions], ignore_index=True)
                    if 'actual_sell_date' in combined_df.columns:
                        combined_df['actual_sell_date'] = pd.to_datetime(combined_df['actual_sell_date'],
                                                                         errors='coerce')
                        combined_df = combined_df.sort_values('actual_sell_date', ascending=False)
                        combined_df['actual_sell_date'] = combined_df['actual_sell_date'].dt.strftime(
                            '%Y-%m-%d %H:%M:%S')
                    combined_df.to_csv(position_history_file, index=False)
                else:
                    if 'actual_sell_date' in sold_positions.columns:
                        sold_positions = sold_positions.copy()
                        sold_positions['actual_sell_date'] = pd.to_datetime(
                            sold_positions['actual_sell_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    sold_positions.to_csv(position_history_file, index=False)
            except Exception as e:
                logger.error(f"❌ 归档历史失败: {e}")
    else:
        pd.DataFrame(columns=['code', 'avg_cost', 'shares', 'entry_date', 'stop_loss_price', 'take_profit_price',
                              'actual_sell_price', 'actual_sell_date', 'is_sold', 'volatility_mult']).to_csv(
            positions_file, index=False)

    current_cash += (stop_return_amount + sell_return_amount)
    pd.DataFrame([{'cash': current_cash, 'last_run_date': predict_date.strftime('%Y-%m-%d'),
                   'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]).to_csv(cash_file, index=False)

logger.info("✅ 实盘全周期模拟验证结束")