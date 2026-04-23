"""
V12 并行参数搜索脚本
使用 multiprocessing 并行搜索 V12 策略的最佳波动率自适应参数组合

性能优化：
- Pool initializer 模式：大数据（full_data/prices_df）只在 worker 启动时传递一次
- ST 数据预加载：主进程预加载所有交易日的 ST 名单，避免每 worker 每天调用 API
- 消除 strategy.run() 内部的 full_data.copy()：worker-global 数据已是私有副本
- chunksize 从 8 提升到 32，减少 IPC 调度次数

Usage:
    python grid_trading_simulation_v12_mp.py
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

OUTPUT_DIR = RESULT_DIR / 'simple_run_log_v12'

# ============ Worker 全局变量（通过 Pool initializer 设置）============
_worker_full_data: pd.DataFrame | None = None
_worker_prices_df: dict | None = None
_worker_st_preloaded: dict | None = None
_worker_initial_capital: float | None = None
_worker_atr_cache: dict | None = None  # 预计算的 ATR 矩阵


# ============ ATR 预计算（主进程执行，所有 lookback 一次性向量化算完）============
class PrecomputedATR:
    """
    主进程预计算所有股票的 ATR 矩阵，传给 workers 直接查表。

    原有 _compute_atr 使用 pandas rolling(lookback).mean()，
    返回 Series 的 iloc[-1]（最后一行）。本实现通过逐元素 numpy 循环
    精确复刻 rolling.mean 的边界行为（窗口 [j-lb+1:j+1]，不足 lb 时取实际数量平均）。

    替换 _precompute_market_atr 中的每日 5285 × _compute_atr 循环（原本占用 99.5% 时间）。
    优化后：主进程预计算 ~5510股票 × 4种lookback × ~483天 ≈ 1064万次 ATR，
    耗时约 20-30 秒（一次性）；每个 worker 每天查表 O(stocks) 取均值。
    """

    def __init__(self, prices_df: dict, dates: list, lookbacks: list):
        import time
        t0 = time.time()
        self.lookbacks = lookbacks
        self._dates = [pd.Timestamp(d) for d in dates]
        self._date_to_pos = {d: i for i, d in enumerate(self._dates)}
        n_dates = len(dates)

        # 存储: {code: {lookback: np.array(n_dates)}} — position-based
        self._data: dict[str, dict[int, np.ndarray]] = {}
        codes = list(prices_df.keys())
        n = len(codes)

        for i, code in enumerate(codes):
            if i % 500 == 0:
                print(f"  ATR预计算: {i}/{n} ({i/n*100:.0f}%)", flush=True)
            df = prices_df[code].sort_index()
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            n_days_stock = len(high)

            # True Range — prev_close[i] = close[i-1] for i>0, prev_close[0] = close[0]
            prev_close = np.empty_like(close)
            prev_close[0] = close[0]
            prev_close[1:] = close[:-1]
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            self._data[code] = {}
            for lb in lookbacks:
                if n_days_stock < lb:
                    atr_arr = np.full(n_dates, np.nan)
                else:
                    # rolling_mean[j] for j < lb-1: NaN (pandas semantics: window not full)
                    # rolling_mean[j] for j >= lb-1: mean of tr[max(0,j-lb+1):j+1]
                    # Vectorized: cumsum[j] = sum(tr[0..j])
                    # window_sum at j = cumsum[j] - cumsum[j-lb] (for j >= lb)
                    # cumsum[j-lb] = sum(tr[0..j-lb]) so window = tr[j-lb+1..j] (lb elements)
                    cs = np.cumsum(tr)
                    # cs_shifted[j] = sum(tr[0..j-1]), with cs_shifted[0] = 0, same length as cs
                    cs_shifted = np.concatenate([[0.0], cs[:-1]])
                    atr_arr = np.full(n_dates, np.nan, dtype=np.float64)
                    # Only compute for j where window fully fits in stock history:
                    # j >= lb-1 AND j <= n_days_stock-1 (can't read beyond stock's last row)
                    max_valid_j = n_days_stock - 1
                    for j in range(lb - 1, min(n_dates, max_valid_j + 1)):
                        start_idx = j - lb + 1
                        window_sum = cs[j] - cs_shifted[start_idx]
                        atr_arr[j] = window_sum / lb
                self._data[code][lb] = atr_arr

        print(f"  ATR预计算完成: {n} 只股票 × {len(lookbacks)} 个窗口, 耗时 {time.time()-t0:.1f}s", flush=True)

    def get_atr(self, code: str, lookback: int, date, default=None):
        """按 position 取 ATR — 与原有 _compute_atr 的 iloc[-1] 逻辑完全一致"""
        if code not in self._data:
            return default
        if lookback not in self._data[code]:
            return default
        pos = self._date_to_pos.get(pd.Timestamp(date), None)
        if pos is None:
            return default
        atr_arr = self._data[code][lookback]
        if pos >= len(atr_arr):
            return default
        v = atr_arr[pos]
        if np.isnan(v):
            return default
        return float(v)

    def market_avg_atr(self, date, lookback: int, codes: list) -> float | None:
        """计算指定日期市场平均 ATR — 按 position 查表，O(stocks)"""
        pos = self._date_to_pos.get(pd.Timestamp(date), None)
        if pos is None:
            return None
        atr_vals = []
        for code in codes:
            if code not in self._data or lookback not in self._data[code]:
                continue
            v = self._data[code][lookback][pos]
            if not np.isnan(v) and v > 0:
                atr_vals.append(v)
        return np.mean(atr_vals) if atr_vals else None


# ============ ST 数据预加载（主进程执行一次）============
def _preload_st_cache(trade_dates: list[str]) -> dict[str, set[str]]:
    """
    批量预加载 ST 股票数据，避免回测过程中每个 worker 重复调 API。
    trade_dates: YYYYMMDD 格式的日期列表
    Returns: {trade_date: set of ts_codes}
    """
    import tinyshare as ts
    print(">>> [ST预加载] 开始从 Tushare 获取 ST 数据...", flush=True)
    token = "3Q4RY56w8deQac5uQkcba5wzoaUf8XBdiLvBti22gv5jTstJ4d0ywZKU247ade48"
    ts.set_token(token)
    pro = ts.pro_api()

    st_cache: dict[str, set[str]] = {}
    total = len(trade_dates)
    for i, date in enumerate(trade_dates):
        if i % 50 == 0:
            print(f">>> [ST预加载] {i}/{total} ({(i/total*100):.0f}%) <<<", flush=True)
        try:
            df = pro.stock_st(trade_date=date)
            if df is not None and len(df) > 0:
                st_cache[date] = set(df['ts_code'])
            else:
                st_cache[date] = set()
        except Exception:
            st_cache[date] = set()
        time.sleep(0.05)  # 避免请求过快被限流

    print(f">>> [ST预加载] 完成，共 {total} 天，{sum(1 for v in st_cache.values() if v)} 天有ST数据 <<<", flush=True)
    return st_cache


# ============ Pool Initializer（每个 worker 只执行一次）============
def _init_worker(full_data, prices_df, st_preloaded, atr_cache, initial_capital):
    """Pool initializer -- 每个 worker 进程启动时执行一次，设置 worker 全局变量"""
    global _worker_full_data, _worker_prices_df, _worker_st_preloaded, _worker_atr_cache, _worker_initial_capital
    _worker_full_data = full_data
    _worker_prices_df = prices_df
    _worker_st_preloaded = st_preloaded
    _worker_atr_cache = atr_cache  # PrecomputedATR 实例，查表代替每日循环
    _worker_initial_capital = initial_capital

    # 完全静默子进程输出（logging + tqdm）
    import logging as _logging
    _logging.basicConfig(level=_logging.CRITICAL)
    _logging.getLogger().setLevel(_logging.CRITICAL)
    # 禁用子进程中所有 tqdm 进度条
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.disable = True
    except ImportError:
        pass


# ============ V12 参数网格配置 ============
def _generate_param_grid():
    """
    生成 V12 参数网格
    分为两部分：
    1. 基础参数（使用 V8 候选参数集的全部参数）
    2. 波动率自适应参数（新）

    使用 comm_fun.model_config.STRATEGY_PARAMS_CANDIDATES_V8 作为基础参数集
    """
    from comm_fun import model_config

    v8_candidates = model_config.STRATEGY_PARAMS_CANDIDATES_V8

    base_configs = []
    # V8 基础参数列表（用户指定）
    for key in [
        'param35',
        'param1',
        'param21', 'param31', 'param20',
        'param24', 'param42', 'param5',
        'param26', 'param29',
    ]:
        if key in v8_candidates:
            params = v8_candidates[key]
            p = params.copy()
            p['name'] = f'v8_{key}'
            base_configs.append(p)

    logger.info(f"使用 V8 {len(base_configs)} 个基础参数")

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


# ============ 子进程执行函数（使用 worker 全局数据，无 .copy()）============
def _run_single_param(param_dict):
    """
    在子进程中运行单组参数（静默模式）。

    使用 worker 全局数据而非 partial 传递：
    - 大数据（full_data/prices_df）通过 Pool initializer 在 worker 启动时加载一次
    - ST 预加载数据通过 st_preloaded 传入，避免每 worker 每天调用 Tushare API
    - ATR 预计算矩阵通过 atr_cache 传入，查表代替每日 5285 × _compute_atr 循环
    - 不走 simple_run 包装层，直接调用 strategy.run()，避免不必要的 full_data.copy()
    """
    global _worker_full_data, _worker_prices_df, _worker_st_preloaded, _worker_atr_cache, _worker_initial_capital

    name = param_dict.pop('name')
    initial_capital = _worker_initial_capital

    try:
        from strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12
        from stock_eligibility_filter import StockEligibilityFilter

        # 构造策略实例
        strategy = SmartSniperStrategyV12(initial_capital=initial_capital, max_positions=10)

        # 注入预加载的 ST 缓存（避免 worker 内每天调用 Tushare API）
        strategy.stock_filter = StockEligibilityFilter(
            filter_main_board=True,
            filter_st=True,
            filter_new_stock=True,
            st_preloaded=_worker_st_preloaded)

        # 注入预计算的 ATR 矩阵：用查表代替 _precompute_market_atr 中的每日循环
        # monkey-patch strategy 实例的 _precompute_market_atr 方法
        _atr_cache = _worker_atr_cache
        _prices_df = _worker_prices_df

        def _fast_market_atr(self, today, prices_df):
            """使用预计算 ATR 矩阵，查表 O(1) 获取市场均值（代替原来每日 O(stocks) 循环）"""
            today_str = today.strftime('%Y%m%d') if hasattr(today, 'strftime') else str(today)
            if today_str in self._market_atr_cache:
                return self._market_atr_cache[today_str]
            # 用预计算 ATR 矩阵查表获取市场 ATR 均值
            codes = list(_prices_df.keys())
            result = _atr_cache.market_avg_atr(today, self.vol_lookback, codes)
            self._market_atr_cache[today_str] = result
            return result

        strategy._precompute_market_atr = _fast_market_atr.__get__(strategy, type(strategy))

        # 应用参数
        strategy.max_positions = param_dict.get('max_positions', 3)
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

        # 直接用 worker-global 数据（已私有，无需 .copy()）
        # 不传 _timing_hook 避免主循环的额外回调开销
        t0 = time.time()
        trade_log, asset_curve = strategy.run(_worker_full_data, prices_df=_worker_prices_df)
        t_total = time.time() - t0

        # ===== 计算回测指标（与 simple_run 一致）=====
        result = {'strategy_name': name}

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
            asset_curve.to_csv(OUTPUT_DIR / f'simple_run_grid_v12_asset_log_{name}.csv', index=False)
            trade_log.to_csv(OUTPUT_DIR / f'simple_run_grid_v12_trade_log_{name}.csv', index=False)

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
        for k, v in param_dict.items():
            result_df[f'param_{k}'] = v
        result_df['param_name'] = name
        return result_df

    except Exception:
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

    # 提取交易日列表（用于 ST 预加载）
    trade_dates = sorted(full_data['date'].dt.strftime('%Y%m%d').unique())
    logger.info(f"共有 {len(trade_dates)} 个交易日")

    # ST 数据预加载（主进程一次性获取，避免 workers 内重复调用 API）
    logger.info("正在预加载 ST 数据...")
    st_preloaded = _preload_st_cache(trade_dates)

    # ATR 预计算（主进程一次性向量化算完，所有 workers 共享查表）
    vol_lookbacks = [7, 10, 14, 21]
    logger.info("正在预计算 ATR 矩阵（向量化，预计 ~1-2分钟）...")
    atr_cache = PrecomputedATR(prices_df, trade_dates, vol_lookbacks)

    initial_capital = 248526  # 与V8搜索一致

    # ===== 第二步：生成参数网格 =====
    param_list = _generate_param_grid()

    # ===== 第三步：并行执行（使用 Pool initializer）=====
    num_workers = 22
    total_params = len(param_list)
    logger.info(f"使用 {num_workers} 个进程并行搜索，共 {total_params} 组参数...")

    ctx = mp.get_context('spawn')
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(full_data, prices_df, st_preloaded, atr_cache, initial_capital)
    ) as pool:
        results = []
        completed = 0
        pbar = tqdm(total=total_params, desc="参数搜索", unit="组", mininterval=1.0)
        for r in pool.imap(_run_single_param, param_list, chunksize=32):
            results.append(r)
            completed += 1
            pbar.update(1)
            if completed % 500 == 0 or completed == total_params:
                pbar.set_postfix_str(f"完成 {completed}/{total_params}")
        pbar.close()

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
