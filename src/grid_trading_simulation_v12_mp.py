"""
V12 并行参数搜索脚本
使用 multiprocessing 并行搜索 V12 策略的最佳波动率自适应参数组合

Usage:
    python grid_trading_simulation_v12_mp.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import itertools
import multiprocessing as mp
from functools import partial
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from config.settings import DATASET_DIR, RESULT_DIR
from grid_trading_simulation_v12 import data_process, simple_run
from comm_fun import model_config

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v12'

# ============ V12 参数网格配置 ============
# 使用 V8 候选参数集中的 TOP N（按参数名排序后的前N个）
# 设为 None 则使用全部 V8 候选参数
V8_TOP_N = None  # None = 全部，或指定数字如 10, 20, 35
# ===========================================


def _generate_param_grid():
    """
    生成 V12 参数网格
    分为两部分：
    1. 基础参数（使用 V8 候选参数集的全部参数）
    2. 波动率自适应参数（新）

    使用 comm_fun.model_config.STRATEGY_PARAMS_CANDIDATES_V8 作为基础参数集
    """
    from comm_fun import model_config

    # 使用 V8 候选参数集（支持 TOP N 筛选）
    v8_candidates = model_config.STRATEGY_PARAMS_CANDIDATES_V8

    # 按 key 排序后取前 N 个
    sorted_keys = sorted(v8_candidates.keys())
    if V8_TOP_N is not None:
        sorted_keys = sorted_keys[:V8_TOP_N]
        logger.info(f"使用 V8 TOP {V8_TOP_N} 参数（其余被过滤）")
    else:
        logger.info(f"使用全部 V8 {len(sorted_keys)} 个参数")

    base_configs = []
    for key in sorted_keys:
        params = v8_candidates[key]
        p = params.copy()
        p['name'] = f'v8_{key}'
        base_configs.append(p)

    # 波动率参数网格（扩展版，覆盖更多波动率场景）
    vol_lookbacks = [7, 10, 14, 21]
    vol_high_thresholds = [1.8, 2.0, 2.5, 3.0]
    vol_low_thresholds = [0.4, 0.6, 0.8]
    vol_profit_mults = [1.2, 1.5, 2.0]
    vol_stop_mults = [1.1, 1.3, 1.5]
    low_vol_profit_mults = [0.6, 0.8, 1.0]

    param_list = []

    for base in base_configs:
        for lookback in vol_lookbacks:
            for high_thresh in vol_high_thresholds:
                for low_thresh in vol_low_thresholds:
                    for profit_mult in vol_profit_mults:
                        for stop_mult in vol_stop_mults:
                            for low_profit_mult in low_vol_profit_mults:
                                p = base.copy()
                                p['vol_lookback'] = lookback
                                p['vol_high_thresh'] = high_thresh
                                p['vol_low_thresh'] = low_thresh
                                p['vol_profit_mult'] = profit_mult
                                p['vol_stop_mult'] = stop_mult
                                p['low_vol_profit_mult'] = low_profit_mult
                                p['use_volatility_adaptive'] = True
                                param_list.append(p)

    # 加上不启用波动率自适应的基准组
    for base in base_configs:
        p = base.copy()
        p['use_volatility_adaptive'] = False
        param_list.append(p)

    total = len(param_list)
    logger.info(f"生成 {total} 组参数组合")

    return param_list


def _run_single_param(param_dict, initial_capital, full_data, prices_df):
    """在子进程中运行单组参数"""
    name = param_dict.pop('name')
    try:
        result_df = simple_run(
            initial_capital=initial_capital,
            strategy_name=name,
            strategy_params=param_dict,
            full_data=full_data,
            prices_df=prices_df,
        )
        for k, v in param_dict.items():
            result_df[f'param_{k}'] = v
        result_df['param_name'] = name
        return result_df
    except Exception as e:
        logger.error(f"参数组 {name} 执行出错: {e}")
        return pd.DataFrame()


def main():
    import glob

    # ===== 第一步：准备数据（主进程只做一次） =====
    logger.info("=" * 60)
    logger.info("V12 并行参数搜索开始")
    logger.info("=" * 60)

    logger.info("正在加载数据...")
    full_data = data_process(
        dataset_dir=DATASET_DIR,
        required_files=["test_set.csv", "validation_set.csv"],
        start_date=None,
    )

    # 构建 prices_df 用于 ATR 计算
    prices_df = {}
    if 'code' in full_data.columns and 'date' in full_data.columns:
        logger.info("正在构建价格数据字典（用于ATR计算）...")
        for code, group in full_data.groupby('code'):
            g = group.sort_values('date')
            g = g.set_index('date')
            if 'close' in g.columns:
                prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]
        logger.info(f"价格数据字典构建完成，共 {len(prices_df)} 只股票")

    initial_capital = 248526  # 与V8搜索一致

    # ===== 第二步：生成参数网格 =====
    param_list = _generate_param_grid()

    # ===== 第三步：并行执行 =====
    num_workers = 22
    logger.info(f"使用 {num_workers} 个进程并行搜索...")

    # 使用 multiprocessing Pool
    ctx = mp.get_context('spawn')  # spawn 模式更安全
    with ctx.Pool(processes=num_workers) as pool:
        worker_fn = partial(_run_single_param,
                            initial_capital=initial_capital,
                            full_data=full_data,
                            prices_df=prices_df)
        results = pool.map(worker_fn, param_list)

    # ===== 第四步：合并结果 =====
    all_results = [r for r in results if not r.empty]
    if not all_results:
        logger.error("所有参数组均执行失败")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # 计算综合评分（参考V8的计算方式）
    # composite_score = (annual_return / |max_drawdown|) * sharpe_factor
    final_df['composite_score'] = (
        final_df['return_rate'] / (-final_df['max_drawdown']).clip(lower=0.01) *
        (final_df.get('sharpe_ratio', 1.0).fillna(0) / 2.0)
    )

    # 整理列顺序
    param_cols = ['param_name', 'param_base_ratio', 'param_target_profit', 'param_hard_stop_loss',
                  'param_max_hold_days', 'param_max_positions', 'param_min_probability',
                  'param_vol_lookback', 'param_vol_high_thresh', 'param_vol_low_thresh',
                  'param_vol_profit_mult', 'param_vol_stop_mult', 'param_low_vol_profit_mult',
                  'param_use_volatility_adaptive']
    result_cols = ['final_asset', 'return_rate', 'annual_return', 'max_drawdown',
                    'win_rate', 'total_trades', 'sharpe_ratio', 'composite_score']
    all_cols = [c for c in param_cols if c in final_df.columns] + result_cols

    final_df = final_df[[c for c in all_cols if c in final_df.columns]]

    # 按综合评分排序
    final_df = final_df.sort_values('composite_score', ascending=False)

    # 保存
    output_path = RESULT_DIR / 'parameter_optimization_results_concurrent_v12.csv'
    os.makedirs(RESULT_DIR, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存到: {output_path}")
    logger.info(f"共搜索 {len(final_df)} 组参数")

    # 打印 Top 10
    logger.info("\n===== Top 10 综合评分 =====")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
