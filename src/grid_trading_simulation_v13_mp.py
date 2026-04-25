"""
V13 并行参数搜索脚本 (彻底适配狙击手模式)
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
from grid_trading_simulation_v13 import data_process
from comm_fun import model_config

from strategies.smart_sniper_strategy_v13 import SmartSniperStrategyV13, PrecomputedATR

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v13'

_worker_full_data: pd.DataFrame | None = None
_worker_prices_df: dict | None = None
_worker_st_preloaded: dict | None = None
_worker_initial_capital: float | None = None
_worker_atr_cache: PrecomputedATR | None = None  


def _preload_st_cache(trade_dates: list[str]) -> dict[str, set[str]]:
    import tinyshare as ts
    print(">>> [ST预加载] 开始获取 ST 数据...", flush=True)
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
    """
    V13 纯净版全排列参数网格 (解开束缚，寻找全局最优)
    组合总数: 3*3*3(资金与选股) * 2*2*2(基础风控) * 2*2(移动止盈) * 2*2*2*2(波动率倍数) = 13,824 组
    22 核并发预计耗时: 1 ~ 2 小时
    """
    param_list = []
    
    # 强制固定 V13 核心特质
    fixed_params = {
        'use_volatility_adaptive': True,
        'use_trailing_stop': True,
        'vol_low_thresh': 0.6,          # 低波动阈值固定，通常影响不大
        'low_vol_profit_mult': 0.8      # 低波动收缩系数固定
    }

    # --- 维度 1：资金与选股 ---
    min_probs = [0.35, 0.40, 0.50]      # 选股下探深度
    max_positions = [3, 5, 8]           # 仓位集中度
    base_ratios = [0.8, 1.0]       # 阶梯仓位总油门

    # --- 维度 2：基础风控 ---
    hard_stop_losses = [-0.10, -0.12, -0.14]   # 硬止损
    target_profits = [0.25, 0.35, 0.55]       # 基础硬止盈（测试及时落袋 vs 靠移动止盈放飞）
    max_hold_days_list = [15, 18, 20]       # 持仓耐心

    # --- 维度 3：移动止盈 (V13灵魂) ---
    trailing_activations = [0.06, 0.10] # 利润达到多少激活保护锁
    trailing_stop_pcts = [0.08, 0.15]   # 最高点回撤多少出局

    # --- 维度 4：波动率环境 ---
    vol_lookbacks = [14, 21]            # ATR观察窗口
    vol_high_threshs = [2.0, 2.5]       # 妖股界定门槛
    vol_profit_mults = [1.2, 1.5]       # 高波动止盈放大倍数
    vol_stop_mults = [1.2, 1.5]         # 高波动止损放宽倍数

    param_id = 1
    # 开始暴力交叉全排列
    for min_p in min_probs:
        for max_pos in max_positions:
            for b_ratio in base_ratios:
                for stop_loss in hard_stop_losses:
                    for t_profit in target_profits:
                        for hold_days in max_hold_days_list:
                            for t_act in trailing_activations:
                                for t_stop in trailing_stop_pcts:
                                    for v_lookback in vol_lookbacks:
                                        for v_high in vol_high_threshs:
                                            for v_p_mult in vol_profit_mults:
                                                for v_s_mult in vol_stop_mults:
                                                    
                                                    p = fixed_params.copy()
                                                    p.update({
                                                        'name': f'V13_P{param_id:05d}',
                                                        'min_probability': min_p,
                                                        'max_positions': max_pos,
                                                        'base_ratio': b_ratio,
                                                        'hard_stop_loss': stop_loss,
                                                        'target_profit': t_profit,
                                                        'max_hold_days': hold_days,
                                                        'trailing_activation': t_act,
                                                        'trailing_stop_pct': t_stop,
                                                        'vol_lookback': v_lookback,
                                                        'vol_high_thresh': v_high,
                                                        'vol_profit_mult': v_p_mult,
                                                        'vol_stop_mult': v_s_mult
                                                    })
                                                    param_list.append(p)
                                                    param_id += 1
                                                    
    return param_list
   
def _run_single_param(param_dict):
    global _worker_full_data, _worker_prices_df, _worker_st_preloaded, _worker_atr_cache, _worker_initial_capital

    name = param_dict.pop('name')
    initial_capital = _worker_initial_capital

    try:
        strategy = SmartSniperStrategyV13(
            initial_capital=initial_capital, 
            max_positions=param_dict.get('max_positions', 5),
            st_preloaded=_worker_st_preloaded
        )
        strategy.atr_cache = _worker_atr_cache

        # 完全动态接收网格参数
        strategy.base_ratio = param_dict.get('base_ratio', 1.0)
        strategy.target_profit = param_dict.get('target_profit', 0.50)
        strategy.max_hold_days = param_dict.get('max_hold_days', 20)
        strategy.hard_stop_loss = param_dict.get('hard_stop_loss', -0.06)
        strategy.min_probability = param_dict.get('min_probability', 0.35)
        
        # 波动率与移动止盈参数
        strategy.use_volatility_adaptive = param_dict.get('use_volatility_adaptive', True)
        strategy.use_trailing_stop = param_dict.get('use_trailing_stop', True)
        
        strategy.vol_lookback = param_dict.get('vol_lookback', 14)
        strategy.vol_high_thresh = param_dict.get('vol_high_thresh', 2.5)
        strategy.vol_low_thresh = param_dict.get('vol_low_thresh', 0.6)
        strategy.vol_profit_mult = param_dict.get('vol_profit_mult', 1.5)
        strategy.vol_stop_mult = param_dict.get('vol_stop_mult', 1.3)
        strategy.low_vol_profit_mult = param_dict.get('low_vol_profit_mult', 0.80)
        
        strategy.trailing_activation = param_dict.get('trailing_activation', 0.08)
        strategy.trailing_stop_pct = param_dict.get('trailing_stop_pct', 0.10)

        # 执行回测
        trade_log, asset_curve = strategy.run(_worker_full_data.copy(), prices_df=_worker_prices_df)

        # 指标计算
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
            close_actions = ['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT', 'TRAILING_TAKE_PROFIT']
            closed_trades = trade_log[trade_log['action'].isin(close_actions)]
            if len(closed_trades) > 0:
                win_count = len(closed_trades[closed_trades['profit'] > 0])
                total_count = len(closed_trades)
                result['total_trades'] = total_count
                result['win_count'] = win_count
                result['win_rate'] = win_count / total_count

        # 封装结果
        result_df = pd.DataFrame([result])
        for k, v in param_dict.items(): result_df[f'param_{k}'] = v
        result_df['param_name'] = name
        return result_df

    except Exception as e:
        return pd.DataFrame()

def main():
    logger.info("V13 并行参数搜索开始 (集中资金/分级权重/跟踪止损版)")

    full_data = data_process(dataset_dir=DATASET_DIR, required_files=["test_set.csv", "validation_set.csv"], start_date=None)

    prices_df = {}
    if 'code' in full_data.columns and 'date' in full_data.columns:
        for code, group in full_data.groupby('code'):
            g = group.sort_values('date').set_index('date')
            prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]

    trade_dates = sorted(full_data['date'].dt.strftime('%Y%m%d').unique())
    st_preloaded = _preload_st_cache(trade_dates)

    logger.info("预计算 ATR 矩阵（向量化无泄露版）...")
    atr_cache = PrecomputedATR(prices_df, trade_dates, [7, 10, 14, 21])

    initial_capital = 248526  
    param_list = _generate_param_grid()
    
    num_workers = 22
    logger.info(f"使用 {num_workers} 个进程，搜索 {len(param_list)} 组参数...")

    # 内存极速版：传递给 Worker 的 prices_df 掏空实质内容
    lightweight_prices_dict = {code: None for code in prices_df.keys()}

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(full_data, lightweight_prices_dict, st_preloaded, atr_cache, initial_capital)) as pool:
        results = []
        completed = 0
        pbar = tqdm(total=len(param_list), desc="参数搜索", unit="组", mininterval=1.0)
        for r in pool.imap(_run_single_param, param_list, chunksize=8):
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
    output_path = RESULT_DIR / 'parameter_optimization_results_concurrent_v13.csv'
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存: {output_path}")

if __name__ == "__main__":
    main()