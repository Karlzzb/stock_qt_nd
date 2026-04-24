import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，使 src.comm_fun 等导入正常工作
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
from strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12
from comm_fun import model_config
from config.settings import STOCK_DATA_DIR, MODEL_DIR, DATASET_DIR, RESULT_DIR
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

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v12'

def load_and_prepare_data(dataset_dir=DATASET_DIR, required_files=None, start_date=None, end_date=None):
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

    required_columns = ['timestamp', model_config.LABEL_COL]
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        logger.error(f"❌缺少必要列: {missing_columns}")
        return None

    try:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    except Exception as e:
        logger.error(f"❌时间戳转换错误: {e}")
        return None

    combined_df = prepare_real_daily_features(combined_df)

    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['timestamp'] >= start_date]
        logger.info(f"过滤掉 {original_count - len(combined_df)} 个开始时间之前的数据点")

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['timestamp'] <= end_date]
        logger.info(f"过滤掉 {original_count - len(combined_df)} 个结束时间之后的数据点")

    return combined_df


def data_process(dataset_dir=DATASET_DIR, required_files=None, start_date=None, end_date=None):
    if required_files is None:
        required_files = ["test_set.csv", "validation_set.csv"]
    logger.info("正在加载数据...")
    raw_df = load_and_prepare_data(dataset_dir=dataset_dir, required_files=required_files, start_date=start_date)

    if start_date is None:
        start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
    if end_date is None:
        end_date = (raw_df['timestamp'].max() + pd.Timedelta(days=60)).strftime("%Y%m%d")

    logger.debug("正在去除 raw_df 中的重复数据...")
    before_count_full = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
    logger.debug(f"去重前: {before_count_full} 条, 去重后: {len(raw_df)} 条")

    # 模型预测
    predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))
    symbol_info = raw_df['symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(['N/A'] * len(raw_df))
    pred_probabilities = predictor.predict_proba(raw_df)
    df_proba = raw_df.copy()
    df_proba['y_pred_proba'] = pred_probabilities
    df_proba['symbol'] = symbol_info

    # 加载完整价格数据
    full_data_dict = load_price_data(str(STOCK_DATA_DIR))
    full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)
    full_data_df = full_data_df[
        (full_data_df['timestamp'] >= pd.to_datetime(start_date))
        & (full_data_df['timestamp'] <= pd.to_datetime(end_date))
    ]

    # 合并
    full_data_df = full_data_df.merge(
        df_proba[['symbol', 'timestamp', 'y_pred_proba']],
        on=['symbol', 'timestamp'],
        how='left'
    )

    logger.debug(f"合并后数据量: {len(full_data_df)}, 预测值覆盖率: {1 - full_data_df['y_pred_proba'].isna().mean():.2%}")

    df_result = full_data_df.rename(columns={
        'timestamp': 'date',
        'symbol': 'code',
    })

    # 构建次日数据列（用于模拟T+1挂单成交）
    df_result['next_open'] = df_result.groupby('code')['open'].shift(-1)
    df_result['next_high'] = df_result.groupby('code')['high'].shift(-1)
    df_result['next_low'] = df_result.groupby('code')['low'].shift(-1)
    df_result['next_close'] = df_result.groupby('code')['close'].shift(-1)
    df_result['entry_date'] = df_result.groupby('code')['date'].shift(-1)

    return df_result


def simple_run(initial_capital, strategy_name, strategy_params, full_data, prices_df=None):
    """单组参数回测（返回指标DataFrame）"""
    strategy = SmartSniperStrategyV12(initial_capital=initial_capital, max_positions=10)

    # 基础参数
    strategy.max_positions = strategy_params.get('max_positions', 3)
    strategy.base_ratio = strategy_params.get('base_ratio', 1.0)
    strategy.target_profit = strategy_params.get('target_profit', 0.30)
    strategy.max_hold_days = strategy_params.get('max_hold_days', 18)
    strategy.hard_stop_loss = strategy_params.get('hard_stop_loss', -0.10)
    strategy.min_probability = strategy_params.get('min_probability', 0.50)

    # V12 波动率自适应参数
    strategy.use_volatility_adaptive = strategy_params.get('use_volatility_adaptive', True)
    strategy.vol_lookback = strategy_params.get('vol_lookback', 14)
    strategy.vol_high_thresh = strategy_params.get('vol_high_thresh', 2.5)
    strategy.vol_low_thresh = strategy_params.get('vol_low_thresh', 0.6)
    strategy.vol_profit_mult = strategy_params.get('vol_profit_mult', 1.5)
    strategy.vol_stop_mult = strategy_params.get('vol_stop_mult', 1.3)
    strategy.low_vol_profit_mult = strategy_params.get('low_vol_profit_mult', 0.80)
    strategy.use_market_vol = strategy_params.get('use_market_vol', False)

    logger.info(f"[{strategy_name}] V12回测开始: 基础止盈={strategy.target_profit:.0%}, 止损={strategy.hard_stop_loss:.0%}, "
                f"波动率自适应={strategy.use_volatility_adaptive}, vol_lookback={strategy.vol_lookback}")

    trade_log, asset_curve = strategy.run(full_data.copy(), prices_df=prices_df)

    logger.info("回测结束！")

    # ===== 计算回测指标 =====
    result = {'strategy_name': strategy_name}

    final_asset = asset_curve.iloc[-1]['total']
    return_rate = (final_asset - initial_capital) / initial_capital
    result['final_asset'] = final_asset
    result['return_rate'] = return_rate

    if not asset_curve.empty:
        asset_curve['peak'] = asset_curve['total'].cummax()
        asset_curve['drawdown'] = (asset_curve['total'] - asset_curve['peak']) / asset_curve['peak']
        max_drawdown = asset_curve['drawdown'].min()
        result['max_drawdown'] = max_drawdown

        asset_curve['daily_return'] = asset_curve['total'].pct_change().fillna(0)
        total_days = len(asset_curve)
        annual_return = (1 + return_rate) ** (252 / total_days) - 1 if total_days > 0 else 0
        daily_std = asset_curve['daily_return'].std()
        annual_volatility = daily_std * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        result['sharpe_ratio'] = sharpe_ratio
        result['annual_return'] = annual_return

    if not trade_log.empty:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        asset_curve.to_csv(OUTPUT_DIR / f'simple_run_grid_v12_asset_log_{strategy_name}.csv', index=False)
        trade_log.to_csv(OUTPUT_DIR / f'simple_run_grid_v12_trade_log_{strategy_name}.csv', index=False)

        close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT']
        closed_trades = trade_log[trade_log['action'].isin(close_actions)]

        if len(closed_trades) > 0:
            win_count = len(closed_trades[closed_trades['profit'] > 0])
            total_count = len(closed_trades)
            win_rate = win_count / total_count
            result['total_trades'] = total_count
            result['win_count'] = win_count
            result['lose_count'] = total_count - win_count
            result['win_rate'] = win_rate

            for action in close_actions:
                action_trades = closed_trades[closed_trades['action'] == action]
                if len(action_trades) > 0:
                    action_win = len(action_trades[action_trades['profit'] > 0])
                    result[f'{action}_trades'] = len(action_trades)
                    result[f'{action}_win_rate'] = action_win / len(action_trades)

    result_df = pd.DataFrame([result])
    return result_df


from tools.hold_analysis import hold_analyzer
from tools.profit_analysis import profit_analyzer
from tools.return_anlaysis import return_analyzer
from tools.trades_analysis import trades_analyzer
from tools.return_prob_correlation_anlaysis import correlation_analyzer

import concurrent.futures


def run_strategy_with_analysis(name, params, capital, full_data, prices_df=None):
    """执行单个策略的完整流程"""
    print(f"正在测试参数组: {name}")
    result_df = simple_run(initial_capital=capital, strategy_name=name,
                           strategy_params=params, full_data=full_data, prices_df=prices_df)
    return name, result_df


def run_concurrent(init_capital, data, prices_df=None):
    """并发执行所有V12参数组"""
    max_workers = 16 #min(16, len(model_config.STRATEGY_PARAMS_CANDIDATES_V12))
    # 使用进程池替代线程池，绕过Python GIL限制，实现真正并行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for name, params in model_config.STRATEGY_PARAMS_CANDIDATES_V12.items():
            # 每个进程有独立内存空间，直接传递data引用即可，无需copy
            future = executor.submit(
                run_strategy_with_analysis,
                name, params, init_capital, data, prices_df
            )
            futures.append(future)

        all_result_dfs = []
        for future in concurrent.futures.as_completed(futures):
            try:
                strategy_name, result_df = future.result()
                # hold_analyzer(version="v12", param_suffix=strategy_name)
                profit_analyzer(version="v12", param_suffix=strategy_name)
                return_analyzer(version="v12", param_suffix=strategy_name)
                trades_analyzer(version="v12", param_suffix=strategy_name)
                # correlation_analyzer(version="v12", param_suffix=strategy_name)
                all_result_dfs.append(result_df)
                print(f"参数组 {strategy_name} 测试完成")
            except Exception as e:
                print(f"参数组执行出错: {e}")

    if len(all_result_dfs) > 0:
        final_df = pd.concat(all_result_dfs, ignore_index=True)
        analysis_df = RESULT_DIR / f'strategy_compare_v12_full_analysis.csv'
        os.makedirs(RESULT_DIR, exist_ok=True)
        final_df.to_csv(analysis_df, index=False, encoding='utf-8-sig')
        print(f"\n所有参数组测试完成，汇总结果已保存到: {analysis_df}")


if __name__ == "__main__":
    processed_data = data_process(dataset_dir=DATASET_DIR,
                             required_files=[
                                 # "train_set.csv",
                                 "test_set.csv",
                                 "validation_set.csv",
                             ],
                                  # start_date="20251001"
                                  )
    selected_param = "param1"
    result_df = simple_run(initial_capital=248526,strategy_name =selected_param, strategy_params=model_config.STRATEGY_PARAMS_CANDIDATES_V12[selected_param],full_data=processed_data)
    analysis_df = RESULT_DIR / f'strategy_{selected_param}_v12_full_analysis.csv'
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_df.to_csv(analysis_df, index=False, encoding='utf-8-sig')

    # run_concurrent(init_capital=248526, data=processed_data)
