# grid_trading_simulation_v9_mp.py
import pandas as pd
import numpy as np
import os
import sys
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from config.settings import DATASET_DIR, RESULT_DIR

# ----------------------------------------------------
# 步骤 1: 导入原文件中的类和函数
# 为了确保能正确导入，需要将原文件所在的目录添加到系统路径
# 假设 grid_trading_simulation_v9_mp.py 和 grid_trading_simulation_v9.py 在同一目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 从V9策略文件导入策略类
    from strategies.smart_sniper_strategy_v9 import SmartSniperStrategy
    # 从V9 simulation文件导入data_process
    from grid_trading_simulation_v9 import data_process
except ImportError as e:
    print(f"导入失败。请检查文件是否存在。错误: {e}")
    sys.exit(1)
# ----------------------------------------------------

# 设置日志级别，避免子进程打印大量日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def _run_single_backtest(params, param_names, df_result, init_capital):
    """
    【核心并发函数】运行单个参数组合的回测并计算绩效指标。
    """
    # 映射参数名称到值
    param_dict = dict(zip(param_names, params))

    try:
        # 1. 初始化策略
        # 必须传入 max_positions 因为 SmartSniperStrategy.__init__ 中会用到
        strategy = SmartSniperStrategy(initial_capital=init_capital, max_positions=param_dict['max_positions'])

        # 2. 设置参数 (复制原 parameter_optimization 中的逻辑)
        strategy.base_ratio = param_dict['base_ratio']
        strategy.target_profit = param_dict['target_profit']
        strategy.hard_stop_loss = param_dict['hard_stop_loss']
        strategy.max_hold_days = param_dict['max_hold_days']
        strategy.min_probability = param_dict['min_probability']

        # V9 新增参数
        strategy.recent_rise_n = param_dict['recent_rise_n']
        strategy.recent_rise_pct = param_dict['recent_rise_pct']
        strategy.rq_window = param_dict['rq_window']
        strategy.rq_shrink_threshold = param_dict['rq_shrink_threshold']
        strategy.rq_recover_threshold = param_dict['rq_recover_threshold']

        # 3. 运行回测 (注意：传入 df_result.copy() 确保数据隔离)
        trade_log, asset_curve = strategy.run(df_result.copy())

        # 4. 计算绩效指标 (复制原 parameter_optimization 中的逻辑)
        final_asset = asset_curve.iloc[-1]['total']
        return_rate = (final_asset - init_capital) / init_capital

        asset_curve['peak'] = asset_curve['total'].cummax()
        asset_curve['drawdown'] = (asset_curve['total'] - asset_curve['peak']) / asset_curve['peak']
        max_drawdown = asset_curve['drawdown'].min()

        closed_trades = trade_log[trade_log['action'].isin(['TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT'])]
        if len(closed_trades) > 0:
            win_rate = len(closed_trades[closed_trades['profit'] > 0]) / len(closed_trades)
            total_trades = len(closed_trades)
        else:
            win_rate = 0
            total_trades = 0

        total_profit = closed_trades['profit'].fillna(0).sum()

        days = (asset_curve['date'].iloc[-1] - asset_curve['date'].iloc[0]).days
        annual_return = (1 + return_rate) ** (365 / days) - 1 if days > 0 else 0

        daily_returns = asset_curve['total'].pct_change().dropna()
        # 假设无风险利率为0，一年250个交易日
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(250) if len(daily_returns) > 1 else 0

        # 5. 存储结果
        result = {
            **param_dict,
            'final_asset': final_asset,
            'total_profit': total_profit,
            'return_rate': return_rate,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'sharpe_ratio': sharpe_ratio
        }
        return result

    except Exception as ex:
        # 如果单个回测失败，返回 None 或包含错误信息的字典
        logger.error(f"参数组合 {param_dict} 测试失败: {ex}")
        return None


def concurrent_parameter_optimization(df_result, init_capital):
    """
    使用 ProcessPoolExecutor 并发执行参数优化。
    """
    # 定义参数搜索空间（V9参数）
    param_grid = {
        # V8 基础参数
        'base_ratio': [0.86, 1.0],
        'target_profit': [0.15, 0.2, 0.25, 0.3, 0.35],
        'hard_stop_loss': [-0.08, -0.1, -0.12, -0.14],
        'max_hold_days': [15, 16, 17, 18, 19, 20, 25, 30],
        'max_positions': [3, 5, 8, 10],
        'min_probability': [0.5, 0.55, 0.65, 0.75, 0.8],
        # V9 新增参数
        'recent_rise_n': [10, 15, 20],
        'recent_rise_pct': [0.2, 0.25, 0.3],
        'rq_window': [5, 10, 15],
        'rq_shrink_threshold': [0.5, 0.6],
        'rq_recover_threshold': [0.7, 0.8, 0.9],
    }

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    print(f"开始参数优化，共 {len(param_combinations)} 种组合...")

    # 设置进程数：建议使用 CPU 核心数减 1
    num_workers = max(1, os.cpu_count() - 1)
    print(f"使用 {num_workers} 个进程进行并发回测...")

    results = []
    total_tasks = len(param_combinations)
    completed_tasks = 0

    # 使用 ProcessPoolExecutor 进行并发
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务。注意：df_result 会被序列化并发送给每个子进程。
        futures = [
            executor.submit(_run_single_backtest, params, param_names, df_result, init_capital)
            for params in param_combinations
        ]

        # 收集结果并显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="================全量并发回测进度================"):
            result = future.result()
            if result is not None:
                results.append(result)
            completed_tasks += 1
            # 每隔固定的任务数（例如 10 组）或在完成时更新进度
            if completed_tasks % 10 == 0 or completed_tasks == total_tasks:
                percentage = (completed_tasks / total_tasks) * 100
                # 使用 '\r' (回车符) 实现进度条单行刷新
                print(f"并发回测进度: 已完成 {completed_tasks}/{total_tasks} 组 ({percentage:.1f}%)", end='\r')

    print("\n所有参数组合测试完毕。")
    return pd.DataFrame(results)


def analyze_optimization_results(results_df):
    """
    分析参数优化结果
    """
    logger.info("\n" + "=" * 80)
    logger.info("参数优化结果分析")
    logger.info("=" * 80)

    if len(results_df) == 0:
        logger.error("没有有效的结果数据")
        return None

    # 按不同指标排序找到最佳参数
    metrics = ['return_rate', 'annual_return', 'sharpe_ratio', 'win_rate']

    best_params = {}
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_params[metric] = results_df.loc[best_idx]

    # 显示最佳参数组合
    logger.info("\n最佳参数组合（按不同指标）:")
    for metric, best in best_params.items():
        logger.info(f"\n按 {metric.upper()} 排序的最佳参数:")
        logger.info(f"  最终收益: {best['return_rate']:.2%}")
        logger.info(f"  年化收益: {best['annual_return']:.2%}")
        logger.info(f"  胜率: {best['win_rate']:.2%}")
        logger.info(f"  夏普比率: {best['sharpe_ratio']:.2f}")
        logger.info(f"  交易次数: {best['total_trades']}")
        logger.info(f"  参数: base_ratio={best['base_ratio']}, "
              f"target_profit={best['target_profit']}, "
              f"stop_loss={best['hard_stop_loss']}, "
              f"hold_days={best['max_hold_days']}")

    # 找到综合表现最好的参数（使用复合评分）
    results_df['composite_score'] = (
            results_df['return_rate'] * 0.3 +
            results_df['win_rate'] * 0.3 +
            (1 - results_df['max_drawdown'].abs()) * 0.2 +
            results_df['sharpe_ratio'] * 0.2
    )

    best_composite = results_df.loc[results_df['composite_score'].idxmax()]

    logger.info(f"\n{'=' * 80}")
    logger.info("综合最佳参数（复合评分）:")
    logger.info(f"{'=' * 80}")
    logger.info(f"最终收益: {best_composite['return_rate']:.2%}")
    logger.info(f"年化收益: {best_composite['annual_return']:.2%}")
    logger.info(f"最大回撤: {best_composite['max_drawdown']:.2%}")
    logger.info(f"胜率: {best_composite['win_rate']:.2%}")
    logger.info(f"夏普比率: {best_composite['sharpe_ratio']:.2f}")
    logger.info(f"交易次数: {best_composite['total_trades']}")
    logger.info(f"复合评分: {best_composite['composite_score']:.4f}")
    logger.info(f"\n推荐参数:")
    logger.info(f"  base_ratio = {best_composite['base_ratio']}")
    logger.info(f"  target_profit = {best_composite['target_profit']}")
    logger.info(f"  hard_stop_loss = {best_composite['hard_stop_loss']}")
    logger.info(f"  max_hold_days = {best_composite['max_hold_days']}")
    # V9新增参数
    logger.info(f"  recent_rise_n = {best_composite['recent_rise_n']}")
    logger.info(f"  recent_rise_pct = {best_composite['recent_rise_pct']}")
    logger.info(f"  rq_window = {best_composite['rq_window']}")
    logger.info(f"  rq_shrink_threshold = {best_composite['rq_shrink_threshold']}")
    logger.info(f"  rq_recover_threshold = {best_composite['rq_recover_threshold']}")

    return best_composite


def finding_best_params_concurrent(init_capital):
    """
    整合数据加载、并发优化和结果分析的函数。
    """
    start_time = time.time()
    print("🚀 正在加载数据并进行模型预测...")
    df_result = data_process(dataset_dir=DATASET_DIR, required_files=["test_set.csv", "validation_set.csv"])

    # ==========================================
    # 参数优化部分 (并发)
    # ==========================================
    optimization_results = concurrent_parameter_optimization(df_result, init_capital)

    # ==========================================
    # 结果分析和保存 (使用原文件中的函数)
    # ==========================================
    if not optimization_results.empty:
        best_params = analyze_optimization_results(optimization_results)

        # 保存结果到文件
        output_file = 'parameter_optimization_results_concurrent_v9.csv'
        optimization_results.to_csv(RESULT_DIR / output_file, index=False)
        print(f"\n优化结果已保存到 {output_file}")
    else:
        print("警告: 没有生成有效的优化结果。")
        best_params = None

    end_time = time.time()
    print(f"\n✅ 并发优化总耗时: {end_time - start_time:.2f} 秒")

    # (可选) 使用最佳参数运行最终回测 - 沿用原文件逻辑
    if best_params is not None:
        print(f"\n使用最佳参数运行最终回测...")
        print(f"最佳参数: {best_params.to_dict()}")


if __name__ == "__main__":
    initial_capital = 248526
    finding_best_params_concurrent(initial_capital)
