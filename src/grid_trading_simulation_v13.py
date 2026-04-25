import sys
from pathlib import Path
import time
import os

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np
import logging

# 导入 V13
from strategies.smart_sniper_strategy_v13 import SmartSniperStrategyV13, PrecomputedATR
from comm_fun import model_config
from config.settings import STOCK_DATA_DIR, MODEL_DIR, DATASET_DIR, RESULT_DIR
from data_process import prepare_real_daily_features
from predictor_model_v2 import PriceChangePredictor
from feature_pipeline import load_price_data, convert_dict_to_dataframe_from_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v13'


def _preload_st_cache(trade_dates: list[str]) -> dict[str, set[str]]:
    import tinyshare as ts
    logger.info(">>> [ST预加载] 开始获取 ST 数据...")
    token = "3Q4RY56w8deQac5uQkcba5wzoaUf8XBdiLvBti22gv5jTstJ4d0ywZKU247ade48"
    ts.set_token(token)
    pro = ts.pro_api()

    st_cache: dict[str, set[str]] = {}
    for date in trade_dates:
        try:
            df = pro.stock_st(trade_date=date)
            st_cache[date] = set(df['ts_code']) if df is not None and len(df) > 0 else set()
        except Exception:
            st_cache[date] = set()
        time.sleep(0.05)
    return st_cache


def load_and_prepare_data(dataset_dir=DATASET_DIR, required_files=None, start_date=None, end_date=None):
    if required_files is None: required_files = ["test_set.csv", "validation_set.csv"]
    file_list = [str(dataset_dir / f) for f in required_files if (dataset_dir / f).exists()]

    all_data = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = os.path.basename(file_path)
            all_data.append(df)
        except Exception as e:
            logger.error(f"❌ 加载出错: {e}")

    if not all_data: return None
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = prepare_real_daily_features(combined_df)

    if start_date is not None: combined_df = combined_df[combined_df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date is not None: combined_df = combined_df[combined_df['timestamp'] <= pd.to_datetime(end_date)]
    return combined_df


def data_process(dataset_dir=DATASET_DIR, required_files=None, start_date=None, end_date=None):
    if required_files is None: required_files = ["test_set.csv", "validation_set.csv"]
    raw_df = load_and_prepare_data(dataset_dir=dataset_dir, required_files=required_files, start_date=start_date)

    if start_date is None: start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
    if end_date is None: end_date = (raw_df['timestamp'].max() + pd.Timedelta(days=60)).strftime("%Y%m%d")

    raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
    predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))
    symbol_info = raw_df['symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(['N/A'] * len(raw_df))
    df_proba = raw_df.copy()
    df_proba['y_pred_proba'] = predictor.predict_proba(raw_df)
    df_proba['symbol'] = symbol_info

    full_data_dict = load_price_data(str(STOCK_DATA_DIR))
    full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)
    full_data_df = full_data_df[(full_data_df['timestamp'] >= pd.to_datetime(start_date)) & (full_data_df['timestamp'] <= pd.to_datetime(end_date))]

    full_data_df = full_data_df.merge(df_proba[['symbol', 'timestamp', 'y_pred_proba']], on=['symbol', 'timestamp'], how='left')

    df_result = full_data_df.rename(columns={'timestamp': 'date', 'symbol': 'code'})
    df_result['next_open'] = df_result.groupby('code')['open'].shift(-1)
    df_result['next_high'] = df_result.groupby('code')['high'].shift(-1)
    df_result['next_low'] = df_result.groupby('code')['low'].shift(-1)
    df_result['next_close'] = df_result.groupby('code')['close'].shift(-1)
    df_result['entry_date'] = df_result.groupby('code')['date'].shift(-1)

    return df_result


def simple_run(initial_capital, strategy_name, strategy_params, full_data, prices_df=None, atr_cache=None, st_preloaded=None):
    strategy = SmartSniperStrategyV13(
        initial_capital=initial_capital, 
        max_positions=strategy_params.get('max_positions', 5), # 默认缩紧到5只
        st_preloaded=st_preloaded
    )
    
    strategy.atr_cache = atr_cache
    strategy.base_ratio = strategy_params.get('base_ratio', 1.0)
    strategy.target_profit = strategy_params.get('target_profit', 0.50)
    strategy.max_hold_days = strategy_params.get('max_hold_days', 20)
    strategy.hard_stop_loss = strategy_params.get('hard_stop_loss', -0.06)
    strategy.min_probability = strategy_params.get('min_probability', 0.35)

    strategy.use_volatility_adaptive = strategy_params.get('use_volatility_adaptive', True)
    strategy.vol_lookback = strategy_params.get('vol_lookback', 14)
    strategy.vol_high_thresh = strategy_params.get('vol_high_thresh', 2.5)
    strategy.vol_low_thresh = strategy_params.get('vol_low_thresh', 0.6)
    strategy.vol_profit_mult = strategy_params.get('vol_profit_mult', 1.5)
    strategy.vol_stop_mult = strategy_params.get('vol_stop_mult', 1.3)
    strategy.low_vol_profit_mult = strategy_params.get('low_vol_profit_mult', 0.80)

    trade_log, asset_curve = strategy.run(full_data.copy(), prices_df=prices_df, show_progress=True)

    result = {'strategy_name': strategy_name}
    final_asset = asset_curve.iloc[-1]['total']
    return_rate = (final_asset - initial_capital) / initial_capital
    result['final_asset'] = final_asset
    result['return_rate'] = return_rate

    if not asset_curve.empty:
        asset_curve['peak'] = asset_curve['total'].cummax()
        asset_curve['drawdown'] = (asset_curve['total'] - asset_curve['peak']) / asset_curve['peak']
        result['max_drawdown'] = asset_curve['drawdown'].min()

        asset_curve['daily_return'] = asset_curve['total'].pct_change().fillna(0)
        total_days = len(asset_curve)
        annual_return = (1 + return_rate) ** (252 / total_days) - 1 if total_days > 0 else 0
        annual_volatility = asset_curve['daily_return'].std() * np.sqrt(252)
        result['sharpe_ratio'] = annual_return / annual_volatility if annual_volatility != 0 else 0
        result['annual_return'] = annual_return

    if not trade_log.empty:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        asset_curve.to_csv(OUTPUT_DIR / f'simple_run_grid_v13_asset_log_{strategy_name}.csv', index=False)
        trade_log.to_csv(OUTPUT_DIR / f'simple_run_grid_v13_trade_log_{strategy_name}.csv', index=False)

        close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT', 'TRAILING_TAKE_PROFIT']
        closed_trades = trade_log[trade_log['action'].isin(close_actions)]

        if len(closed_trades) > 0:
            win_count = len(closed_trades[closed_trades['profit'] > 0])
            result['total_trades'] = len(closed_trades)
            result['win_count'] = win_count
            result['win_rate'] = win_count / len(closed_trades)

            # 按交易类型分析
            for action in close_actions:
                action_trades = closed_trades[closed_trades['action'] == action]
                if len(action_trades) > 0:
                    action_win_count = len(action_trades[action_trades['profit'] > 0])
                    action_win_rate = action_win_count / len(action_trades)
                    result[f'{action}_trades'] =  len(action_trades)
                    result[f'{action}_win_rate'] = action_win_rate

    return pd.DataFrame([result])

# ==========================================
# 新增的并发运行和分析逻辑 (模仿 V8)
# ==========================================

def run_strategy_with_analysis_v12(name, params, capital, full_data, prices_df, atr_cache, st_preloaded):
    """执行单个策略的完整流程"""
    logger.info(f"正在测试参数组: {name}")
    try:
        # 严格调用 V12 原有的 simple_run 逻辑
        result_df = simple_run(
            initial_capital=capital,
            strategy_name=name,
            strategy_params=params,
            full_data=full_data,
            prices_df=prices_df,
            atr_cache=atr_cache,
            st_preloaded=st_preloaded
        )
        return name, result_df
    except Exception as e:
        logger.error(f"参数组 {name} 执行出错: {e}")
        return name, None
import concurrent.futures
from tools.hold_analysis import hold_analyzer
from tools.profit_analysis import profit_analyzer
from tools.return_anlaysis import return_analyzer
from tools.trades_analysis import trades_analyzer
from tools.return_prob_correlation_anlaysis import correlation_analyzer

def run_concurrent_v12(init_capital, data, prices_df, atr_cache, st_preloaded):
    """使用线程池并发执行所有候选参数"""
    max_workers = 16  # 可根据你的 CPU 核心数进行调整
    logger.info(f"开始并发测试所有参数组，最大线程数: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备任务
        futures = []
        for name, params in model_config.STRATEGY_PARAMS_CANDIDATES_V12.items():
            future = executor.submit(
                run_strategy_with_analysis_v12,
                name, params, init_capital, data.copy(),
                prices_df, atr_cache, st_preloaded
            )
            futures.append(future)

        analysis_df_list = []
        # 等待所有任务完成并处理结果
        for future in concurrent.futures.as_completed(futures):
            try:
                strategy_name, result_df = future.result()
                if result_df is not None and not result_df.empty:
                    analysis_df_list.append(result_df)
                    # hold_analyzer(version="v13", param_suffix=strategy_name)
                    # profit_analyzer(version="v13", param_suffix=strategy_name)
                    # return_analyzer(version="v13", param_suffix=strategy_name)
                    # trades_analyzer(version="v13", param_suffix=strategy_name)
                    # correlation_analyzer(version="v13", param_suffix=strategy_name)
                logger.info(f"参数组 {strategy_name} 测试完成")
            except Exception as e:
                logger.error(f"并发获取结果时出错: {e}")

        # 合并所有策略的结果并输出分析文件
        if len(analysis_df_list) > 0:
            final_df = pd.concat(analysis_df_list, ignore_index=True)
            analysis_file = RESULT_DIR / 'strategy_compare_v12_full_analysis.csv'
            os.makedirs(RESULT_DIR, exist_ok=True)
            final_df.to_csv(analysis_file, index=False, encoding='utf-8-sig')
            logger.info(f"所有参数组测试完成，统一分析结果已保存至: {analysis_file}")
        else:
            logger.warning("没有收集到有效的测试结果。")


if __name__ == "__main__":
    processed_data = data_process(dataset_dir=DATASET_DIR, required_files=["test_set.csv", "validation_set.csv"])
    
    prices_df_dict = {}
    if 'code' in processed_data.columns and 'date' in processed_data.columns:
        for code, group in processed_data.groupby('code'):
            g = group.sort_values('date').set_index('date')
            prices_df_dict[code] = g[['open', 'high', 'low', 'close', 'volume']]

    trade_dates = sorted(processed_data['date'].dt.strftime('%Y%m%d').unique())
    
    st_preloaded = _preload_st_cache(trade_dates)
    atr_cache = PrecomputedATR(prices_df_dict, trade_dates, lookbacks=[7, 10, 14, 21])

    selected_param = "param1"
    # 在 V13 中，我们优先拉低 min_probability 让分级仓位起作用
    test_params = model_config.STRATEGY_PARAMS_CANDIDATES_V13.get(selected_param, {}).copy()
    test_params['min_probability'] = 0.35
    test_params['max_positions'] = 5

    result_df = simple_run(
        initial_capital=248526,
        strategy_name=f"{selected_param}_V13",
        strategy_params=test_params,
        full_data=processed_data,
        prices_df=prices_df_dict,
        atr_cache=atr_cache,
        st_preloaded=st_preloaded
    )
    
    analysis_df = RESULT_DIR / f'strategy_{selected_param}_v13_full_analysis.csv'
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_df.to_csv(analysis_df, index=False, encoding='utf-8-sig')
    logger.info("单进程运行 V13 结束")