import pandas as pd
import numpy as np
from strategies.smart_sniper_strategy import SmartSniperStrategy
from comm_fun import model_config
from config.settings import STOCK_DATA_DIR,MODEL_DIR, DATASET_DIR, RESULT_DIR
import os
from data_process import prepare_real_daily_features
from predictor_model_v2 import PriceChangePredictor
from feature_pipeline import load_price_data, convert_dict_to_dataframe_from_index
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = RESULT_DIR / 'simple_run_log'
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

    # BUGFIXED 这里没有对数据做预处理，导致模拟和真实场景不一致
    combined_df = prepare_real_daily_features(combined_df)

    # 创业版没有办法交易
    # combined_df = combined_df[combined_df['symbol'].str.match(r'^[60]')]

    # NOTE: 对ST、次新股和不能交易的进行过滤（模拟过程中这些信息无法实时获取）
    # st_df = pd.read_csv(DATASET_DIR / 'st_stocks_list.csv')
    # new_df = pd.read_csv(DATASET_DIR / 'new_stocks_list.csv')
    # combined_df = combined_df[~combined_df['symbol'].isin(st_df['ts_code'])]
    # combined_df = combined_df[~combined_df['symbol'].isin(new_df['ts_code'])]

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


def data_process(dataset_dir = DATASET_DIR, required_files=None, start_date=None, end_date=None):
    if required_files is None:
        required_files = ["test_set.csv", "validation_set.csv"]
    logger.info("正在加载数据...")
    raw_df = load_and_prepare_data(dataset_dir = dataset_dir, required_files = required_files, start_date = start_date)
    # 计算时间范围
    if start_date is None:
        start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
    if end_date is None:
        end_date = (raw_df['timestamp'].max() + pd.Timedelta(days=60)).strftime("%Y%m%d")

    # 检查 full_data_df 的重复情况
    df_duplicates = raw_df.duplicated(subset=['symbol', 'timestamp']).sum()
    logger.debug(f"DF 中 (symbol, timestamp) 重复数量: {df_duplicates}")

    logger.debug("正在去除 raw_df 中的重复数据...")
    before_count_full = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
    after_count_full = len(raw_df)
    logger.debug(f"去重前: {before_count_full} 条, 去重后: {after_count_full} 条, 移除: {before_count_full - after_count_full} 条重复数据")

    # 初始化预测器
    predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))
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
        & (full_data_df['timestamp'] <= pd.to_datetime(end_date))
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
    strategy.min_probability = strategy_params['min_probability'] # 预测阈值

    # 开始运行
    logger.info("开始回测...")
    trade_log, asset_curve = strategy.run(full_data.copy())

    logger.info("回测结束！")

    # ==========================================
    # 第三步：分析结果
    # ==========================================
    analysis_txt_content = [f"{strategy_name} 回测数据："]
    analysis_df_content = pd.DataFrame({'strategy_name': [strategy_name]})

    # 1. 计算最终收益
    final_asset = asset_curve.iloc[-1]['total']
    return_rate = (final_asset - initial_capital) / initial_capital
    logger.info(f"最终资产: {final_asset:,.2f}")
    logger.info(f"最终收益率: {return_rate:.2%}")
    analysis_txt_content.append(f"最终资产: {final_asset:,.2f}")
    analysis_txt_content.append(f"最终收益率: {return_rate:.2%}")
    analysis_df_content['final_asset'] =  [final_asset]
    analysis_df_content['return_rate'] = [return_rate]

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
        analysis_df_content['max_drawdown'] = [max_drawdown]

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
        analysis_df_content['sharpe_ratio'] = [sharpe_ratio]
        analysis_df_content['annual_return'] = [annual_return]

    # 4. 详细交易统计
    if not trade_log.empty:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        asset_curve.to_csv(OUTPUT_DIR / f'simple_run_grid_v8_asset_log_{strategy_name}.csv', index=False)
        trade_log.to_csv(OUTPUT_DIR / f'simple_run_grid_v8_trade_log_{strategy_name}.csv', index=False)

        # 所有开仓动作
        open_actions = ['OPEN_BUY', 'GRID_ADD']
        # 所有平仓动作
        close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT']

        # 统计开仓次数
        open_trades = trade_log[trade_log['action'].isin(open_actions)]
        logger.info(f"总开仓次数: {len(open_trades)}")
        analysis_txt_content.append(f"总开仓次数: {len(open_trades)}")
        analysis_df_content['open_trades'] = [len(open_trades)]

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

            analysis_df_content['total_count'] = [total_count]
            analysis_df_content['win_count'] =  [win_count]
            analysis_df_content['lose_count'] = [total_count - win_count]
            analysis_df_content['win_rate'] =  [win_rate]

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
                    analysis_df_content[f'{action}_trades'] =  [len(action_trades)]
                    analysis_df_content[f'{action}_win_rate'] = [action_win_rate]
        else:
            logger.warning("没有完成的平仓交易")

        # analysis_dir = RESULT_DIR / f"portfolio_analysis_charts_v8_{strategy_name}"
        # analysis_txt = analysis_dir / f'simple_run_grid_v8_summary_{strategy_name}.txt'
        # os.makedirs(analysis_dir, exist_ok=True)
        # with open(analysis_txt, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(analysis_txt_content))
    return analysis_txt_content, analysis_df_content


from tools.hold_analysis import hold_analyzer
from tools.profit_analysis import profit_analyzer
from tools.return_anlaysis import return_analyzer
from tools.trades_analysis import trades_analyzer
from tools.return_prob_correlation_anlaysis import correlation_analyzer

import concurrent.futures


def run_strategy_with_analysis(name, params, capital, full_data):
    """执行单个策略的完整流程"""
    print(f"正在测试参数组: {name}")
    result_txt, result_df = simple_run(initial_capital=capital, strategy_name=name,
               strategy_params=params, full_data=full_data)
    return name, result_txt, result_df


def run_concurrent(init_capital, data):
    # 使用线程池并发执行
    max_workers = 8  # 根据CPU核心数调整
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备任务
        futures = []
        for name, params in model_config.STRATEGY_PARAMS_CANDIDATES_V8.items():
            future = executor.submit(
                run_strategy_with_analysis,
                name, params, init_capital, data.copy()
            )
            futures.append(future)

        all_analysis_result = []
        analysis_df_list = []
        # 等待所有任务完成并处理结果
        for future in concurrent.futures.as_completed(futures):
            try:
                strategy_name, analysis_result_txt, analysis_result_df = future.result()
                # hold_analyzer(version="v8", param_suffix=strategy_name)
                # profit_analyzer(version="v8", param_suffix=strategy_name)
                # return_analyzer(version="v8", param_suffix=strategy_name)
                # trades_analyzer(version="v8", param_suffix=strategy_name)
                # correlation_analyzer(version="v8", param_suffix=strategy_name)
                analysis_result_txt.append("*" * 60)
                all_analysis_result.extend(analysis_result_txt)
                analysis_df_list.append(analysis_result_df)
                print(f"参数组 {strategy_name} 测试完成")
            except Exception as e:
                print(f"参数组执行出错: {e}")
        if len(analysis_df_list) > 0:
            final_df = pd.concat(analysis_df_list, ignore_index=True)
            analysis_df = RESULT_DIR / f'strategy_compare_v8_full_analysis.csv'
            os.makedirs(RESULT_DIR, exist_ok=True)
            final_df.to_csv(analysis_df, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    processed_data = data_process(dataset_dir=DATASET_DIR,
                             required_files=[
                                 # "train_set.csv",
                                 "test_set.csv",
                                 "validation_set.csv",
                             ],

                                  # start_date="20260106"
                                  )
    run_concurrent(248526, processed_data)
    # simple_run(initial_capital=248526,strategy_name ="参数55", strategy_params=model_config.STRATEGY_PARAMS_CANDIDATES_V8["参数55"],full_data=processed_data)