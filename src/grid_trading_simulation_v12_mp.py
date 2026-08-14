"""
V12 并行参数搜索脚本 (无信息泄露严密对齐版)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import multiprocessing as mp
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config.settings import DATASET_DIR, RESULT_DIR
from grid_trading_simulation_v12 import data_process
from comm_fun import model_config

from strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12, PrecomputedATR

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v12'

_worker_full_data: pd.DataFrame | None = None
_worker_prices_df: dict | None = None
_worker_st_preloaded: dict | None = None
_worker_initial_capital: float | None = None
_worker_atr_cache: PrecomputedATR | None = None


def _preload_st_cache(trade_dates: list[str]) -> dict[str, set[str]]:
    from tinyshare_auth import get_pro_api
    print(">>> [ST预加载] 开始获取 ST 数据...", flush=True)
    pro = get_pro_api()

    st_cache: dict[str, set[str]] = {}
    for date in trade_dates:
        try:
            df = pro.stock_st(trade_date=date)
            st_cache[date] = set(df['ts_code']) if df is not None and len(df) > 0 else set()
        except Exception:
            st_cache[date] = set()
        time.sleep(0.05)
    return st_cache


def _init_worker(full_data, prices_df, st_preloaded, atr_cache, initial_capital):
    global _worker_full_data, _worker_prices_df, _worker_st_preloaded, _worker_atr_cache, _worker_initial_capital
    _worker_full_data = full_data
    _worker_prices_df = prices_df
    _worker_st_preloaded = st_preloaded
    _worker_atr_cache = atr_cache
    _worker_initial_capital = initial_capital

    import logging as _logging
    _logging.basicConfig(level=_logging.CRITICAL)
    _logging.getLogger().setLevel(_logging.CRITICAL)


def _generate_param_grid():
    from comm_fun import model_config
    v8_candidates = model_config.STRATEGY_PARAMS_CANDIDATES_V8
    base_configs = []

    for key in ['param35', 'param1', 'param21', 'param31', 'param20', 'param24', 'param42', 'param5', 'param26', 'param29']:
        if key in v8_candidates:
            p = v8_candidates[key].copy()
            p['name'] = f'v8_{key}'
            base_configs.append(p)

    param_list = []
    for base in base_configs:
        for lookback in [7, 10, 14, 21]:
            for high_thresh in [1.8, 2.0, 2.5, 3.0]:
                for low_thresh in [0.4, 0.6, 0.8]:
                    for profit_mult in [1.2, 1.5, 2.0]:
                        for stop_mult in [1.1, 1.3, 1.5]:
                            for low_profit_mult in [0.6, 0.8, 1.0]:
                                p = base.copy()
                                p.update({
                                    'vol_lookback': lookback, 'vol_high_thresh': high_thresh,
                                    'vol_low_thresh': low_thresh, 'vol_profit_mult': profit_mult,
                                    'vol_stop_mult': stop_mult, 'low_vol_profit_mult': low_profit_mult,
                                    'use_volatility_adaptive': True
                                })
                                param_list.append(p)

        # 基准不自适应参数
        p = base.copy()
        p['use_volatility_adaptive'] = False
        param_list.append(p)

    return param_list


def _run_single_param(param_dict):
    global _worker_full_data, _worker_prices_df, _worker_st_preloaded, _worker_atr_cache, _worker_initial_capital

    name = param_dict.pop('name')
    initial_capital = _worker_initial_capital

    try:
        # 核心：直接在实例化时注入 ST 缓存
        strategy = SmartSniperStrategyV12(
            initial_capital=initial_capital,
            max_positions=param_dict.get('max_positions', 3),
            st_preloaded=_worker_st_preloaded
        )

        # 核心：直接属性注入无泄露 ATR 缓存
        strategy.atr_cache = _worker_atr_cache

        strategy.base_ratio = param_dict.get('base_ratio', 1.0)
        strategy.target_profit = param_dict.get('target_profit', 0.30)
        strategy.max_hold_days = param_dict.get('max_hold_days', 18)
        strategy.hard_stop_loss = param_dict.get('hard_stop_loss', -0.10)
        strategy.min_probability = param_dict.get('min_probability', 0.50)
        strategy.use_volatility_adaptive = param_dict.get('use_volatility_adaptive', True)
        strategy.vol_lookback = param_dict.get('vol_lookback', 14)
        strategy.vol_high_thresh = param_dict.get('vol_high_thresh', 2.5)
        strategy.vol_low_thresh = param_dict.get('vol_low_thresh', 0.6)
        strategy.vol_profit_mult = param_dict.get('vol_profit_mult', 1.5)
        strategy.vol_stop_mult = param_dict.get('vol_stop_mult', 1.3)
        strategy.low_vol_profit_mult = param_dict.get('low_vol_profit_mult', 0.80)
        strategy.use_market_vol = param_dict.get('use_market_vol', False)

        # 传入 full_data.copy() 防止同 worker 参数组脏数据污染
        trade_log, asset_curve = strategy.run(_worker_full_data.copy(), prices_df=_worker_prices_df)

        result = {'strategy_name': name}
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
            daily_std = asset_curve['daily_return'].std()
            annual_volatility = daily_std * np.sqrt(252)
            result['sharpe_ratio'] = annual_return / annual_volatility if annual_volatility != 0 else 0
            result['annual_return'] = annual_return

        if not trade_log.empty:
            close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT']
            closed_trades = trade_log[trade_log['action'].isin(close_actions)]
            if len(closed_trades) > 0:
                win_count = len(closed_trades[closed_trades['profit'] > 0])
                total_count = len(closed_trades)
                result['total_trades'] = total_count
                result['win_count'] = win_count
                result['lose_count'] = total_count - win_count
                result['win_rate'] = win_count / total_count

        result_df = pd.DataFrame([result])
        for k, v in param_dict.items(): result_df[f'param_{k}'] = v
        result_df['param_name'] = name
        return result_df

    except Exception as e:
        # 如果内部发生报错，返回空的df避免阻断整个并行任务
        return pd.DataFrame()


def main():
    logger.info("V12 并行参数搜索开始 (单多进程环境全对齐、无信息泄露版)")

    full_data = data_process(dataset_dir=DATASET_DIR, required_files=["test_set.csv", "validation_set.csv"], start_date=None)

    prices_df = {}
    if 'code' in full_data.columns and 'date' in full_data.columns:
        for code, group in full_data.groupby('code'):
            g = group.sort_values('date').set_index('date')
            prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]

    trade_dates = sorted(full_data['date'].dt.strftime('%Y%m%d').unique())
    st_preloaded = _preload_st_cache(trade_dates)

    logger.info("预计算 ATR 矩阵（向量化且防未来函数）...")
    atr_cache = PrecomputedATR(prices_df, trade_dates, [7, 10, 14, 21])

    initial_capital = 248526
    param_list = _generate_param_grid()

    num_workers = 22
    logger.info(f"使用 {num_workers} 个进程，搜索 {len(param_list)} 组参数...")

    lightweight_prices_dict = {code: None for code in prices_df.keys()}
    ctx = mp.get_context('spawn')
    # 注意这里传入的是 lightweight_prices_dict
    with ctx.Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(full_data, lightweight_prices_dict, st_preloaded, atr_cache, initial_capital)) as pool:
        results = []
        completed = 0
        pbar = tqdm(total=len(param_list), desc="参数搜索", unit="组", mininterval=1.0)

        # 将 chunksize 从 32 降低到 2，让进度条几乎实时更新
        for r in pool.imap(_run_single_param, param_list, chunksize=2):
            results.append(r)
            completed += 1
            pbar.update(1)
            if completed % 500 == 0 or completed == len(param_list):
                pbar.set_postfix_str(f"完成 {completed}/{len(param_list)}")
        pbar.close()

    all_results = [r for r in results if not r.empty]
    if not all_results: return

    final_df = pd.concat(all_results, ignore_index=True)
    final_df['composite_score'] = (
        final_df['return_rate'] / (-final_df['max_drawdown']).clip(lower=0.01) *
        (final_df.get('sharpe_ratio', 1.0).fillna(0) / 2.0)
    )

    final_df = final_df.sort_values('composite_score', ascending=False)
    output_path = RESULT_DIR / 'parameter_optimization_results_concurrent_v12.csv'
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存: {output_path}")

if __name__ == "__main__":
    main()