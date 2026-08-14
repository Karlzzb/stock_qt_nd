"""
Phase 2: 参数 × 过滤条件 笛卡尔积回测

对每个过滤条件组合，跑完所有候选参数，输出独立 CSV。
"""
import pandas as pd
import numpy as np
import os
import sys
import concurrent.futures
import multiprocessing as mp
import gc
import time

sys.path.insert(0, os.path.dirname(__file__))

from strategies.smart_sniper_strategy import SmartSniperStrategy
from comm_fun import model_config
from config.settings import STOCK_ND_CSV_DIR, MODEL_DIR, DATASET_DIR, RESULT_DIR
from data_process import prepare_real_daily_features
from predictor_model_v2 import PriceChangePredictor
from feature_pipeline import load_price_data, convert_dict_to_dataframe_from_index
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('phase2')


# 8 种过滤条件组合
FILTER_COMBOS = [
    (False, False, False, "无过滤"),
    (True,  False, False, "不包含创业版"),
    (False, True,  False, "不包含ST"),
    (True,  True,  False, "不包含ST和创业版"),
    (False, False, True,  "不包含次新股"),
    (True,  False, True,  "不包含次新股和创业版"),
    (False, True,  True,  "不包含ST和次新股"),
    (True,  True,  True,  "不包含ST、次新股和创业版"),
]

OUTPUT_DIR = RESULT_DIR / "TOP55参数（测试&验证集）回测对比"


def preload_st_cache(trade_dates: list[str]) -> dict[str, set[str]]:
    """
    批量预加载 ST 股票数据，避免回测过程中每个 worker 重复调 API。
    trade_dates: YYYYMMDD 格式的日期列表
    Returns: {trade_date: set of ts_codes}
    """
    print(">>> [ST预加载] 开始从 Tushare 获取 ST 数据，", flush=True)
    from tinyshare_auth import get_pro_api
    pro = get_pro_api()

    st_cache: dict[str, set[str]] = {}
    total = len(trade_dates)
    for i, date in enumerate(trade_dates):
        if i % 20 == 0:
            print(f">>> [ST预加载] {i}/{total} ({(i/total*100):.0f}%) <<<", flush=True)
        try:
            df = pro.stock_st(trade_date=date)
            if df is not None and len(df) > 0:
                st_cache[date] = set(df['ts_code'])
            else:
                st_cache[date] = set()
        except Exception:
            st_cache[date] = set()
        time.sleep(0.1)  # 避免请求过快被限流

    print(f">>> [ST预加载] 完成，共 {total} 天，{len(st_cache)} 天有ST数据 <<<", flush=True)
    return st_cache


def load_and_prepare_data(required_files=None, start_date=None, end_date=None):
    """加载原始数据（不复制）"""
    if required_files is None:
        required_files = ["test_set.csv", "validation_set.csv"]
    file_list = []
    for filename in required_files:
        file_path = DATASET_DIR / filename
        if file_path.exists():
            file_list.append(str(file_path))
        else:
            logger.warning(f"File not found: {filename}")

    dfs = [pd.read_csv(fp) for fp in file_list]
    combined_df = pd.concat(dfs, ignore_index=True)
    del dfs
    combined_df = prepare_real_daily_features(combined_df)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

    if start_date:
        combined_df = combined_df[combined_df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        combined_df = combined_df[combined_df['timestamp'] <= pd.to_datetime(end_date)]

    return combined_df


def data_process(required_files=None, start_date=None, end_date=None):
    """预测概率并构建完整数据"""
    print(">>> [data_process] Loading raw CSV files... <<<", flush=True)
    raw_df = load_and_prepare_data(required_files, start_date, end_date)

    if start_date is None:
        start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
    if end_date is None:
        end_date = (raw_df['timestamp'].max() + pd.Timedelta(days=60)).strftime("%Y%m%d")

    raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
    print(">>> [data_process] Deduplicated, running predictor... <<<", flush=True)

    predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))
    symbol_info = raw_df['symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(['N/A'] * len(raw_df))
    pred_probabilities = predictor.predict_proba(raw_df)
    df_proba = raw_df[['symbol', 'timestamp']].copy()
    df_proba['y_pred_proba'] = pred_probabilities
    df_proba['symbol'] = symbol_info

    del pred_probabilities
    gc.collect()
    print(">>> [data_process] Probabilities computed, loading price data... <<<", flush=True)

    full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
    full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)
    del full_data_dict
    gc.collect()

    full_data_df = full_data_df[
        (full_data_df['timestamp'] >= pd.to_datetime(start_date)) &
        (full_data_df['timestamp'] <= pd.to_datetime(end_date))
    ]

    full_data_df = full_data_df.merge(
        df_proba[['symbol', 'timestamp', 'y_pred_proba']],
        on=['symbol', 'timestamp'],
        how='left'
    )
    del df_proba
    gc.collect()

    df_result = full_data_df.rename(columns={
        'timestamp': 'date',
        'symbol': 'code',
    })
    df_result['next_open'] = df_result.groupby('code')['open'].shift(-1)
    df_result['next_high'] = df_result.groupby('code')['high'].shift(-1)
    df_result['next_low'] = df_result.groupby('code')['low'].shift(-1)
    df_result['next_close'] = df_result.groupby('code')['close'].shift(-1)
    df_result['entry_date'] = df_result.groupby('code')['date'].shift(-1)

    # 提取所有交易日期（用于 ST 预加载）
    trade_dates = sorted(df_result['date'].dt.strftime('%Y%m%d').unique())

    logger.info(f"Data prepared: {len(df_result):,} rows | {len(trade_dates)} trading days")

    return df_result, trade_dates


def simple_run(task):
    """单次回测，data_copy/st_preloaded 通过 task 显式传入"""
    (name, params), init_capital, fmb, fst, fns, data, st_preloaded = task

    strategy = SmartSniperStrategy(
        initial_capital=init_capital,
        max_positions=5,
        filter_main_board=fmb,
        filter_st=fst,
        filter_new_stock=fns,
        disable_tqdm=True,
        st_preloaded=st_preloaded,
    )

    strategy.max_positions = params['max_positions']
    strategy.base_ratio = params['base_ratio']
    strategy.target_profit = params['target_profit']
    strategy.max_hold_days = params['max_hold_days']
    strategy.hard_stop_loss = params['hard_stop_loss']
    strategy.min_probability = params['min_probability']

    trade_log, asset_curve = strategy.run(data.copy())

    final_asset = asset_curve.iloc[-1]['total']
    return_rate = (final_asset - init_capital) / init_capital

    if not asset_curve.empty:
        asset_curve['peak'] = asset_curve['total'].cummax()
        asset_curve['drawdown'] = (asset_curve['total'] - asset_curve['peak']) / asset_curve['peak']
        max_drawdown = asset_curve['drawdown'].min()
    else:
        max_drawdown = 0.0

    if not asset_curve.empty and len(asset_curve) > 1:
        asset_curve['daily_return'] = asset_curve['total'].pct_change().fillna(0)
        total_days = len(asset_curve)
        annual_return = (1 + return_rate) ** (252 / total_days) - 1 if total_days > 0 else 0
        daily_std = asset_curve['daily_return'].std()
        annual_volatility = daily_std * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0.0
    else:
        annual_return = 0.0
        sharpe_ratio = 0.0

    open_actions = {'OPEN_BUY', 'GRID_ADD'}
    close_actions = {'TAKE_PROFIT', 'INTRADAY_STOP_LOSS', 'TIME_EXIT'}

    if not trade_log.empty:
        open_trades = trade_log[trade_log['action'].isin(open_actions)]
        closed_trades = trade_log[trade_log['action'].isin(close_actions)]
    else:
        open_trades = pd.DataFrame()
        closed_trades = pd.DataFrame()

    total_trades = len(closed_trades)
    win_count = int((closed_trades['profit'] > 0).sum()) if total_trades > 0 else 0
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    tp_trades = closed_trades[closed_trades['action'] == 'TAKE_PROFIT']
    tp_count = len(tp_trades)
    tp_win = int((tp_trades['profit'] > 0).sum()) if tp_count > 0 else 0
    tp_win_rate = tp_win / tp_count if tp_count > 0 else 0.0

    return {
        'param_name': name,
        'return_rate': return_rate,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'open_trades': len(open_trades),
        'take_profit_trades': tp_count,
        'take_profit_win_rate': tp_win_rate,
        'final_asset': final_asset,
        'filter_main_board': fmb,
        'filter_st': fst,
        'filter_new_stock': fns,
    }


def run_concurrent(init_capital, data, st_preloaded, fmb, fst, fns, combo_label, combo_idx, total_combos):
    """并发跑完一组过滤条件下的所有参数"""
    params = list(model_config.STRATEGY_PARAMS_CANDIDATES_V8.items())
    n_params = len(params)

    # GIL 限制下 ThreadPoolExecutor 对 CPU-bound 任务效果差，8 workers 是平衡点
    max_workers = min(18, n_params)

    print(f">>> [{combo_idx}/{total_combos}] {combo_label} starting... <<<", flush=True)
    logger.info(f"[{combo_idx}/{total_combos}] {combo_label} | {n_params} params x {max_workers} workers")

    # 构建任务列表（data/st_preloaded 显式传入，避免跨线程闭包捕获问题）
    tasks = [
        (p, init_capital, fmb, fst, fns, data, st_preloaded)
        for p in params
    ]

    results = []
    done = 0
    last_pct = -1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(simple_run, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            name = task[0][0]
            try:
                result = future.result()
                results.append(result)
                done += 1
                pct = done * 100 // n_params
                if pct >= last_pct + 10:  # 每 10% 打印一次
                    logger.info(f"[{combo_idx}/{total_combos}] {combo_label} {done}/{n_params} ({pct}%) done")
                    last_pct = pct
            except Exception as e:
                logger.error(f"[{combo_label}] {name} FAILED: {e}")
                done += 1

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('return_rate', ascending=False).reset_index(drop=True)

    best = df.iloc[0]
    logger.info(
        f"[{combo_idx}/{total_combos}] {combo_label} BEST: "
        f"#{best['param_name']} return={best['return_rate']:.4f} "
        f"dd={best['max_drawdown']:.4f} sharpe={best['sharpe_ratio']:.4f} "
        f"win={best['win_rate']:.4f} trades={int(best['total_trades'])}"
    )

    return df


def run_all_combos(init_capital, processed_data, st_preloaded, output_dir):
    """按顺序跑完所有 8 种过滤组合"""
    os.makedirs(output_dir, exist_ok=True)
    total = len(FILTER_COMBOS)

    for i, (fmb, fst, fns, label) in enumerate(FILTER_COMBOS, 1):
        csv_path = output_dir / f"Phase2_组合{i}_{label}.csv"

        df = run_concurrent(init_capital, processed_data, st_preloaded, fmb, fst, fns, label, i, total)

        if not df.empty:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"[{i}/{total}] Saved: {csv_path.name}")
        else:
            logger.warning(f"[{i}/{total}] No results for {label}")

        gc.collect()

    logger.info("All 8 filter combos completed.")


if __name__ == "__main__":
    print(">>> Phase2 script started <<<", flush=True)
    print(f">>> CPU cores: {mp.cpu_count()} <<<", flush=True)

    logger.info("=" * 50)
    logger.info(f"Phase 2 start | CPU cores: {mp.cpu_count()}")
    logger.info("=" * 50)

    print(">>> Loading data (this may take a while)... <<<", flush=True)
    processed_data, trade_dates = data_process(
        required_files=["test_set.csv", "validation_set.csv"],
    )
    print(f">>> Data ready: {len(processed_data)} rows, {len(trade_dates)} trading days <<<", flush=True)
    logger.info(f"Data ready: {len(processed_data)} rows | {len(trade_dates)} trading days")

    # 一次性预加载所有 ST 数据，再开始任务
    print(f">>> Preloading ST data for {len(trade_dates)} days... <<<", flush=True)
    st_cache = preload_st_cache(trade_dates)
    logger.info(f"ST cache loaded: {len(st_cache)} days")

    run_all_combos(
        init_capital=248526,
        processed_data=processed_data,
        st_preloaded=st_cache,
        output_dir=OUTPUT_DIR,
    )

    logger.info("Phase 2 finished.")
