"""
SmartSniper 实盘投资脚本 - 人工执行版本 V12
脚本给出投资方案，人工执行后修改状态文件
"""
import pandas as pd
import os
import sys

# 获取当前脚本的绝对路径，并找到项目根目录（src的父目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timedelta
import json
from typing import Optional, Dict, List

from predictor_model_v2 import PriceChangePredictor
import glob
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 解析命令行参数 - 需要在导入config.settings之前执行
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--stock-data-dir', type=str, default=None)
parser.add_argument('--st-stocks-list', type=str, default=None)
parser.add_argument('--new-stocks-list', type=str, default=None)
parser.add_argument('--predict-date', type=str, required=False)
parser.add_argument('--feature-date', type=str, required=False)
parser.add_argument('--update-data', action='store_true')
args, unknown = parser.parse_known_args()

# 如果传入了stock-data-dir，设置环境变量供config.settings和feature_pipeline.py使用
if args.stock_data_dir:
    os.environ['STOCK_DATA_DIR'] = args.stock_data_dir
    logger.info(f"[CONFIG] 已设置STOCK_DATA_DIR={args.stock_data_dir}")

# 现在导入配置，让它读取更新后的环境变量
from config.settings import MODEL_DIR, REAL_TRADING_DIR, DAILY_FEATURE_DIR, STOCK_ND_CSV_DIR
from comm_fun import ALLOCATION_STRATEGY, PROBA_MEAN, PROBA_STD, model_config, label_encoding
STRATEGY_PARAMS = model_config.STRATEGY_PARAMS_V12
from src.stock_eligibility_filter import StockEligibilityFilter
from feature_pipeline import load_price_data, convert_dict_to_dataframe_from_index
from strategies.smart_sniper_strategy_v12 import SmartSniperStrategyV12

class PortfolioState:
    """投资组合状态管理器 - CSV版本，人工执行 V12"""

    def __init__(self,  data_dir = str(REAL_TRADING_DIR / 'investment_data')):
        self.data_dir = data_dir
        self.cash = 0
        self.positions = {}  # {code: {avg_cost, shares, entry_date, stop_loss_price, take_profit_price, actual_sell_price, actual_sell_date, is_sold, volatility_mult}}
        self.trade_suggestions = []
        self.daily_assets = []
        self.last_run_date = None

        os.makedirs(data_dir, exist_ok=True)

        self.cash_file = os.path.join(data_dir, 'portfolio_cash.csv')
        self.positions_file = os.path.join(data_dir, 'portfolio_positions.csv')
        self.suggestions_file_prefix = os.path.join(data_dir, 'trade_suggestions')
        self.assets_file = os.path.join(data_dir, 'portfolio_assets.csv')
        self.summary_file = os.path.join(data_dir, 'portfolio_summary.json')

    def load_state(self) -> bool:
        """从CSV文件加载投资状态"""
        try:
            if not os.path.exists(self.cash_file):
                logger.info(f"✅ CSV不存在，使用初始资金")
                return False

            if os.path.exists(self.cash_file):
                cash_df = pd.read_csv(self.cash_file)
                if not cash_df.empty:
                    self.cash = cash_df.iloc[0]['cash']
                    self.last_run_date = cash_df.iloc[0].get('last_run_date')

            if os.path.exists(self.positions_file):
                positions_df = pd.read_csv(self.positions_file)

                if not positions_df.empty:
                    date_columns = ['entry_date', 'actual_sell_date', 'should_sell_date']
                    for col in date_columns:
                        if col in positions_df.columns:
                            positions_df[col] = pd.to_datetime(positions_df[col], errors='coerce')

                    sold_positions = positions_df[positions_df['is_sold'] == 'YES'].copy()
                    active_positions = positions_df[positions_df['is_sold'] == 'NO'].copy()

                    if not sold_positions.empty:
                        self._archive_sold_positions(sold_positions)
                        positions_df = active_positions

                    for _, row in positions_df.iterrows():
                        code = row['code']
                        self.positions[code] = {
                            'avg_cost': row['avg_cost'],
                            'shares': row['shares'],
                            'entry_date': row.get('entry_date'),
                            'stop_loss_price': row.get('stop_loss_price'),
                            'take_profit_price': row.get('take_profit_price'),
                            'actual_sell_price': row.get('actual_sell_price'),
                            'actual_sell_date': row.get('actual_sell_date'),
                            'is_sold': row.get('is_sold'),
                            'volatility_mult': row.get('volatility_mult', 1.0)
                        }

                    positions_df.to_csv(self.positions_file, index=False)

            if os.path.exists(self.assets_file):
                assets_df = pd.read_csv(self.assets_file)
                self.daily_assets = assets_df.to_dict('records')

            logger.info(f"✅ CSV状态加载成功. 现金: ¥{self.cash:,.2f}, 持仓: {len(self.positions)}")
            return True
        except Exception as e:
            logger.error(f"❌ 加载CSV状态失败: {e}")
            return False

    def _archive_sold_positions(self, sold_positions: pd.DataFrame) -> None:
        try:
            history_file = os.path.join(self.data_dir, 'portfolio_positions_history.csv')
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                combined_df = pd.concat([history_df, sold_positions], ignore_index=True)
                if 'actual_sell_date' in combined_df.columns:
                    combined_df['actual_sell_date'] = pd.to_datetime(combined_df['actual_sell_date'], errors='coerce')
                    combined_df = combined_df.sort_values('actual_sell_date', ascending=False)
                    combined_df['actual_sell_date'] = combined_df['actual_sell_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                combined_df.to_csv(history_file, index=False)
            else:
                if not sold_positions.empty and 'actual_sell_date' in sold_positions.columns:
                    sold_positions = sold_positions.copy()
                    sold_positions['actual_sell_date'] = pd.to_datetime(sold_positions['actual_sell_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                sold_positions.to_csv(history_file, index=False)
        except Exception as e:
            logger.error(f"❌ 归档已卖出持仓失败: {e}")

    def save_state(self, predict_date: datetime) -> bool:
        try:
            cash_df = pd.DataFrame([{
                'cash': self.cash,
                'last_run_date': predict_date.strftime('%Y-%m-%d'),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            cash_df.to_csv(self.cash_file, index=False)

            if self.positions:
                positions_list = []
                for code, pos in self.positions.items():
                    pos_data = {
                        'code': code,
                        'avg_cost': pos['avg_cost'],
                        'shares': pos['shares'],
                        'entry_date': pos.get('entry_date'),
                        'stop_loss_price': pos.get('stop_loss_price'),
                        'take_profit_price': pos.get('take_profit_price'),
                        'should_sell_date': pos.get('entry_date') + pd.Timedelta(days=STRATEGY_PARAMS['max_hold_days']) if pd.notna(pos.get('entry_date')) else None,
                        'actual_sell_price': pos.get('actual_sell_price'),
                        'actual_sell_date': pos.get('actual_sell_date'),
                        'is_sold': pos.get('is_sold'),
                        'volatility_mult': pos.get('volatility_mult', 1.0)
                    }
                    positions_list.append(pos_data)
                pd.DataFrame(positions_list).to_csv(self.positions_file, index=False)
            else:
                pd.DataFrame(columns=['code', 'avg_cost', 'shares', 'entry_date',
                                     'stop_loss_price', 'take_profit_price', 'should_sell_date',
                                     'actual_sell_price', 'actual_sell_date', 'is_sold', 'volatility_mult']).to_csv(self.positions_file, index=False)

            if self.trade_suggestions:
                pd.DataFrame(self.trade_suggestions).to_csv(f"{self.suggestions_file_prefix}_{predict_date.strftime('%Y%m%d')}.csv", index=False)

            if self.daily_assets:
                pd.DataFrame(self.daily_assets).to_csv(self.assets_file, index=False)

            total_assets = self.calculate_total_assets()
            summary = {
                'cash': float(self.cash),
                'positions_count': len([p for p in self.positions.values() if p.get('is_sold') == 'NO']),
                'total_positions_value': float(self.calculate_positions_value()),
                'total_assets': float(total_assets),
                'last_run_date': predict_date.strftime('%Y-%m-%d'),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"❌ 保存CSV状态失败: {e}")
            return False

    def calculate_positions_value(self, price_data: pd.DataFrame = None) -> float:
        total_value = 0
        for code, pos in self.positions.items():
            if pos.get('is_sold') == 'NO':
                if price_data is not None and code in price_data['code'].values:
                    price_row = price_data[price_data['code'] == code].iloc[0]
                    price = price_row['close']
                else:
                    price = pos['avg_cost']
                total_value += price * pos['shares']
        return total_value

    def calculate_total_assets(self, price_data: pd.DataFrame = None) -> float:
        return self.cash + self.calculate_positions_value(price_data)

    def get_summary(self) -> Dict:
        active_positions = [code for code, pos in self.positions.items() if pos.get('is_sold') == 'NO']
        return {
            'cash': self.cash,
            'active_positions_count': len(active_positions),
            'total_positions_count': len(self.positions),
            'active_positions': active_positions,
            'all_positions': list(self.positions.keys()),
            'last_run_date': self.last_run_date
        }


class SmartSniperInvestor:
    def __init__(self, initial_capital: float = 200000, max_positions: int = 5, base_dir = REAL_TRADING_DIR, prices_df_dict = None):
        self.base_dir = str(base_dir)
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.portfolio_state = PortfolioState(str(base_dir / 'investment_data'))
        self.predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))

        # 核心：完全注入 V12 策略逻辑框架
        self.strategy = SmartSniperStrategyV12(initial_capital=initial_capital, max_positions=max_positions)

        # 覆写配置参数
        self.strategy.max_positions = STRATEGY_PARAMS.get('max_positions', 3)
        self.strategy.base_ratio = STRATEGY_PARAMS.get('base_ratio', 1.0)
        self.strategy.target_profit = STRATEGY_PARAMS.get('target_profit', 0.30)
        self.strategy.max_hold_days = STRATEGY_PARAMS.get('max_hold_days', 18)
        self.strategy.hard_stop_loss = STRATEGY_PARAMS.get('hard_stop_loss', -0.10)
        self.strategy.min_probability = STRATEGY_PARAMS.get('min_probability', 0.50)

        self.strategy.use_volatility_adaptive = STRATEGY_PARAMS.get('use_volatility_adaptive', True)
        self.strategy.vol_lookback = STRATEGY_PARAMS.get('vol_lookback', 14)
        self.strategy.vol_high_thresh = STRATEGY_PARAMS.get('vol_high_thresh', 2.5)
        self.strategy.vol_low_thresh = STRATEGY_PARAMS.get('vol_low_thresh', 0.6)
        self.strategy.vol_profit_mult = STRATEGY_PARAMS.get('vol_profit_mult', 1.5)
        self.strategy.vol_stop_mult = STRATEGY_PARAMS.get('vol_stop_mult', 1.3)
        self.strategy.low_vol_profit_mult = STRATEGY_PARAMS.get('low_vol_profit_mult', 0.80)

        # 接收并缓存全局历史数据，用于 ATR 计算
        self.prices_df = prices_df_dict

        if not self.portfolio_state.load_state():
            self.portfolio_state.cash = initial_capital

        self.stock_filter = StockEligibilityFilter(filter_main_board=True,filter_st=True,filter_new_stock=True)

    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        return self.portfolio_state.get_summary()

    @staticmethod
    def _load_prepare_data(file_pattern):
        if isinstance(file_pattern, list):
            file_list = file_pattern
        else:
            file_list = glob.glob(file_pattern)

        all_data = []
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
            except Exception as e:
                logger.error(f"❌ 加载文件出错: {e}")

        if not all_data:
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        required_columns = ['timestamp', model_config.LABEL_COL]
        if any(col not in combined_df.columns for col in required_columns):
            return None

        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        combined_df = combined_df.sort_values(['timestamp', 'symbol', 'confirmation_score']).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        combined_df, _ = label_encoding(combined_df)
        key_columns = ['timestamp'] + model_config.OPTIMIZED_FEATURE_COLS
        combined_df = combined_df.dropna(subset=key_columns)

        return combined_df

    def prepare_realtime_data(self, target_date: datetime) -> Optional[pd.DataFrame]:
        try:
            features_dir = str(DAILY_FEATURE_DIR)
            file_pattern = os.path.join(features_dir, f"realistic_features_{target_date.strftime('%Y%m%d')}.csv")
            raw_df = self._load_prepare_data(file_pattern)

            # 无论今日特征是否存在，先确保构建了历史价格字典用于 ATR 计算
            if self.prices_df is None:
                full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
                full_data_df_all = convert_dict_to_dataframe_from_index(full_data_dict)
                self.prices_df = {}
                for code, group in full_data_df_all.groupby('symbol'):
                    g = group.sort_values('timestamp').set_index('timestamp')
                    self.prices_df[code] = g[['open', 'high', 'low', 'close', 'volume']]
            else:
                full_data_df_all = pd.DataFrame() # 不再重新全加载

            if raw_df is None or len(raw_df) < 1:
                # 只获取目标日的假数据壳
                if full_data_df_all.empty:
                    full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
                    full_data_df_all = convert_dict_to_dataframe_from_index(full_data_dict)
                full_data_df = full_data_df_all[full_data_df_all['timestamp'] == pd.to_datetime(target_date.strftime("%Y%m%d"))]
                df_result = full_data_df.rename(columns={'timestamp': 'predict_date', 'symbol': 'code'})
                df_result['y_pred_proba'] = 0.0
                return df_result

            start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
            end_date = raw_df['timestamp'].max().strftime("%Y%m%d")

            raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
            symbol_info = raw_df['symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(['N/A'] * len(raw_df))
            pred_probabilities = self.predictor.predict_proba(raw_df)
            df_proba = raw_df.copy()
            df_proba['y_pred_proba'] = pred_probabilities
            df_proba['symbol'] = symbol_info

            if full_data_df_all.empty:
                 full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
                 full_data_df_all = convert_dict_to_dataframe_from_index(full_data_dict)

            full_data_df = full_data_df_all[
                (full_data_df_all['timestamp'] >= pd.to_datetime(start_date)) &
                (full_data_df_all['timestamp'] <= pd.to_datetime(end_date))
            ]

            full_data_df = full_data_df.merge(
                df_proba[['symbol', 'timestamp', 'y_pred_proba']],
                on=['symbol', 'timestamp'],
                how='left'
            )

            df_result = full_data_df.rename(columns={'timestamp': 'predict_date', 'symbol': 'code'})
            return df_result

        except Exception as e:
            logger.error(f"❌ 准备实时数据失败: {e}")
            return None

    def execute_daily_investment(self, predict_date: Optional[datetime] = None, feature_date:Optional[datetime] = None) -> bool:
        if predict_date is None or feature_date is None:
            return False

        if (self.portfolio_state.last_run_date and pd.to_datetime(self.portfolio_state.last_run_date).date() == predict_date.date()):
            logger.info(f"⏭️  今天 ({predict_date.date()}) 已经运行过，跳过")
            return True

        realtime_data = self.prepare_realtime_data(feature_date)

        # 核心：执行 V12 自适应目标预计算，将计算基准对齐到 T (feature_date)
        feature_date_ts = pd.Timestamp(feature_date)
        self.strategy.market_avg_atr = self.strategy._precompute_market_atr(feature_date_ts, self.prices_df)

        for code, pos in self.portfolio_state.positions.items():
            if pos.get('is_sold') == 'NO':
                # 动态计算波动率倍数并得出今日止盈止损价，保存以供人类核对
                pos['volatility_mult'] = self.strategy._get_volatility_multiplier(code, feature_date_ts, self.prices_df)
                target_price, stop_price = self.strategy._get_adaptive_targets(pos)
                pos['take_profit_price'] = target_price
                pos['stop_loss_price'] = stop_price

        buy_suggestions = []
        if realtime_data is None or len(realtime_data) == 0:
            logger.error(f"❌ 无法获取{feature_date.strftime('%Y-%m-%d')}数据，不生成买入建议")
        else:
            buy_suggestions = self._generate_buy_suggestions(realtime_data, predict_date, feature_date_ts)

        sell_suggestions = self._generate_sell_suggestions(predict_date)

        self._record_trade_suggestions(predict_date, sell_suggestions, buy_suggestions)
        self._update_portfolio_value(realtime_data, predict_date)
        self.portfolio_state.save_state(predict_date)
        self._generate_daily_report(predict_date, realtime_data, sell_suggestions, buy_suggestions)

        return True

    def _generate_sell_suggestions(self, predict_date: datetime) -> List[Dict]:
        sell_suggestions = []
        for code, pos in self.portfolio_state.positions.items():
            if pos.get('is_sold') == 'YES':
                continue

            current_price = pos['avg_cost']
            days_held = (predict_date - pd.to_datetime(pos['entry_date'])).days if pd.notna(pos.get('entry_date')) else 0

            # 抓取已被 execute_daily_investment 更新过的动态目标价
            stop_loss_price = pos.get('stop_loss_price') or (pos['avg_cost'] * (1 + self.strategy.hard_stop_loss))
            take_profit_price = pos.get('take_profit_price') or (pos['avg_cost'] * (1 + self.strategy.target_profit))

            should_sell = False
            reason = ""
            suggested_price = 0

            if days_held >= self.strategy.max_hold_days:
                should_sell = True
                reason = "TIME_EXIT"
                suggested_price = pos['avg_cost'] * 0.98

            if should_sell:
                sell_suggestions.append({
                    'code': code, 'action': 'SELL', 'reason': reason,
                    'suggested_price': suggested_price, 'current_price': current_price,
                    'shares': pos['shares'], 'suggested_stop_loss': stop_loss_price,
                    'suggested_take_profit': take_profit_price, 'predict_date': predict_date,
                    'avg_cost': pos['avg_cost'], 'days_held': days_held,
                    'volatility_mult': pos.get('volatility_mult', 1.0)
                })
        return sell_suggestions

    def _generate_buy_suggestions(self, daily_data: pd.DataFrame, predict_date: datetime, feature_date_ts: pd.Timestamp) -> List[Dict]:
        buy_suggestions = []
        active_positions = [code for code, pos in self.portfolio_state.positions.items() if pos.get('is_sold') == 'NO']
        available_slots = self.strategy.max_positions - len(active_positions)
        if available_slots <= 0: return buy_suggestions

        current_codes = list(self.portfolio_state.positions.keys())
        candidates = daily_data[
            ~daily_data['code'].isin(current_codes) &
            daily_data['y_pred_proba'].notna() &
            (daily_data['y_pred_proba'] >= self.strategy.min_probability) &
            (daily_data['close'] <= model_config.AFFORDABLE_PRICE)
        ].copy()

        trade_date = predict_date.strftime('%Y%m%d')
        candidates_filter = candidates.set_index('code')
        candidates = self.stock_filter.filter(candidates_filter, trade_date)
        if not candidates.empty: candidates = candidates.reset_index()
        if candidates.empty: return buy_suggestions

        top_candidates = candidates.sort_values(['y_pred_proba', 'code'], ascending=[False, True]).head(int(available_slots * 2))

        total_frozen = 0.0
        for code, row in top_candidates.iterrows():
            remaining_slots = available_slots - len(buy_suggestions)
            if remaining_slots <= 0: break
            available_cash = self.portfolio_state.cash - total_frozen
            if available_cash < 100: break

            signal_price = row['close']
            proba = row['y_pred_proba']

            weight_factor = 1.0
            if ALLOCATION_STRATEGY == 'z_score':
                z_score = (proba - PROBA_MEAN) / PROBA_STD
                weight_factor = max(0.8, min(1.4, 1.0 + (0.2 * z_score)))
            elif ALLOCATION_STRATEGY == 'tiered':
                weight_factor = 1.3 if proba >= 0.72 else 1.0 if proba >= 0.60 else 0.8

            budget = (available_cash / max(1, remaining_slots)) * weight_factor * self.strategy.base_ratio
            planned_shares = int(budget / signal_price / 100) * 100

            if planned_shares <= 100:
                if remaining_slots == 1:
                    planned_shares = int(available_cash / signal_price / 100) * 100
                    if planned_shares <= 100: continue
                else: continue

            required_cash = planned_shares * signal_price
            if required_cash > available_cash:
                max_shares = int(available_cash / signal_price / 100) * 100
                if max_shares < 100: continue
                planned_shares = max_shares

            total_frozen += planned_shares * signal_price

            # 核心：新开仓也要完全贴合 V12 的波动率自适应计算
            vol_mult = 1.0
            if self.strategy.use_volatility_adaptive and self.prices_df is not None:
                vol_mult = self.strategy._get_volatility_multiplier(row['code'], feature_date_ts, self.prices_df)

            tmp_pos = {'avg_cost': signal_price, 'volatility_mult': vol_mult}
            target_price, stop_price = self.strategy._get_adaptive_targets(tmp_pos)

            buy_suggestions.append({
                'code': row['code'], 'action': 'BUY', 'reason': 'SIGNAL',
                'suggested_price': signal_price, 'current_price': row['close'],
                'shares': planned_shares, 'suggested_stop_loss': stop_price,
                'suggested_take_profit': target_price, 'predict_date': predict_date,
                'prediction_probability': row['y_pred_proba'],
                'required_capital': signal_price * planned_shares,
                'volatility_mult': vol_mult
            })

        return buy_suggestions

    def _record_trade_suggestions(self, predict_date: datetime, sell_suggestions: List[Dict], buy_suggestions: List[Dict]):
        for suggestion in sell_suggestions + buy_suggestions:
            suggestion_record = {
                'predict_date': predict_date.strftime('%Y-%m-%d'),
                'code': suggestion['code'], 'action': suggestion['action'],
                'reason': suggestion['reason'], 'suggested_price': suggestion.get('suggested_price'),
                'current_price': suggestion.get('current_price'), 'shares': suggestion.get('shares'),
                'suggested_stop_loss': suggestion.get('suggested_stop_loss'),
                'suggested_take_profit': suggestion.get('suggested_take_profit'),
                'prediction_probability': suggestion.get('prediction_probability', 0.0),
                'required_capital': suggestion.get('required_capital', 0.0),
                'avg_cost': suggestion.get('avg_cost', 0.0), 'days_held': suggestion.get('days_held', 0),
                'volatility_mult': suggestion.get('volatility_mult', 1.0)
            }
            self.portfolio_state.trade_suggestions.append(suggestion_record)

    def _update_portfolio_value(self, daily_data: pd.DataFrame, today: datetime):
        total_value = self.portfolio_state.calculate_total_assets(daily_data)
        self.portfolio_state.daily_assets.append({
            'predict_date': today.strftime('%Y-%m-%d'), 'total': total_value,
            'cash': self.portfolio_state.cash, 'positions_value': total_value - self.portfolio_state.cash
        })

    def _generate_daily_report(self, date: datetime, price_data: pd.DataFrame, sell_suggestions: List[Dict], buy_suggestions: List[Dict]):
        report_dir = os.path.join(self.base_dir, 'investment_reports')
        os.makedirs(report_dir, exist_ok=True)
        report_file = f"{report_dir}/report_{date.strftime('%Y%m%d')}.txt"

        total_assets = self.portfolio_state.calculate_total_assets(price_data)
        total_return = (total_assets - self.initial_capital) / self.initial_capital

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"📊 智能狙击手投资日报（人工执行 V12 波动自适应版）\n")
            f.write(f"📅 日期: {date.strftime('%Y-%m-%d')}\n")
            f.write(f"⏰ 生成时间: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            f.write("【投资组合摘要】\n")
            f.write(f"  现金余额: ¥{self.portfolio_state.cash:,.2f}\n")
            active_positions = [code for code, pos in self.portfolio_state.positions.items() if pos.get('is_sold') == 'NO']
            f.write(f"  未卖出持仓: {len(active_positions)} 只\n")
            f.write(f"  持仓市值: ¥{self.portfolio_state.calculate_positions_value(price_data):,.2f}\n")
            f.write(f"  总资产: ¥{total_assets:,.2f}  (累计收益率: {total_return:.2%})\n\n")

            f.write("【卖出建议 (含到达持有期强制卖出)】\n")
            if sell_suggestions:
                for i, suggestion in enumerate(sell_suggestions, 1):
                    f.write(f"  {i}. {suggestion['code']}: {suggestion['shares']}股 | 理由: {suggestion['reason']}\n")
            else:
                f.write("  今日无卖出建议\n\n")

            f.write("【买入建议】\n")
            if buy_suggestions:
                for i, suggestion in enumerate(buy_suggestions, 1):
                    f.write(f"  {i}. {suggestion['code']}: 建议买入 {suggestion['shares']}股 @ ¥{suggestion['suggested_price']:.2f}\n")
                    f.write(f"     预测概率: {suggestion['prediction_probability']:.2%} | 波动率倍数: {suggestion['volatility_mult']:.2f}x\n")
                    f.write(f"     自适应止损价: ¥{suggestion['suggested_stop_loss']:.2f} | 自适应止盈价: ¥{suggestion['suggested_take_profit']:.2f}\n\n")
            else:
                f.write("  今日无买入建议\n\n")

            f.write("【当前持仓状态 (动态监控)】\n")
            if active_positions:
                for code in active_positions:
                    pos = self.portfolio_state.positions[code]
                    current_price = price_data[price_data['code'] == code].iloc[0]['close'] if price_data is not None and code in price_data['code'].values else pos['avg_cost']
                    pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost']
                    days_held = (date - pd.to_datetime(pos['entry_date'])).days if pd.notna(pos.get('entry_date')) else 0

                    f.write(f"  {code}: {pos['shares']}股 | 持有: {days_held}天 | 浮动盈亏: {pnl_pct:.2%}\n")
                    f.write(f"     成本价: ¥{pos['avg_cost']:.2f} -> 当前价: ¥{current_price:.2f}\n")
                    f.write(f"     波动倍数: {pos.get('volatility_mult', 1.0):.2f}x | 动态止损: ¥{pos.get('stop_loss_price', 0):.2f} | 动态止盈: ¥{pos.get('take_profit_price', 0):.2f}\n\n")
            else:
                f.write("  暂无未卖出持仓\n\n")

            f.write("【操作说明】\n")
            f.write("  交易完成后，请修改 investment_data 下的 portfolio_cash.csv 和 portfolio_positions.csv\n")

def run(predict_date = datetime.now(), feature_date = datetime.now(), real_trading_dir = REAL_TRADING_DIR, prices_df_dict = None):
    investor = SmartSniperInvestor(initial_capital=248526, max_positions=10, base_dir=real_trading_dir, prices_df_dict=prices_df_dict)

    # 增加的 summary 读取与打印日志
    summary = investor.get_portfolio_summary()
    logger.info(f"💰 当前现金: ¥{summary['cash']:,.2f} | 未卖出持仓: {summary['active_positions_count']} 只")

    success = investor.execute_daily_investment(predict_date, feature_date)
    if success:
        logger.debug("✅ 今日投资建议生成完成")

if __name__ == "__main__":
    if args.predict_date:
        PREDICT_DATE = datetime.strptime(args.predict_date, '%Y-%m-%d')
    else:
        PREDICT_DATE = datetime.now()

    if args.feature_date:
        FEATURE_DATE = datetime.strptime(args.feature_date, '%Y-%m-%d')
    else:
        FEATURE_DATE = datetime.now() + timedelta(days=-1)

    try:
        from feature_pipeline import feature_generator
        features = feature_generator(FEATURE_DATE.strftime("%Y-%m-%d"))
        if features is not None:
            run(PREDICT_DATE, FEATURE_DATE)
    except Exception as e:
        logger.error(f"处理发生错误: {e}")