"""
SmartSniper 实盘投资脚本 - 人工执行版本
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

from  predictor_model_v2 import PriceChangePredictor
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
import sys
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
from config.settings import MODEL_DIR, REAL_TRADING_DIR, DAILY_FEATURE_DIR, ST_FILTER_DATA_DIR, STOCK_ND_CSV_DIR
from comm_fun import ALLOCATION_STRATEGY, PROBA_MEAN, PROBA_STD,model_config, label_encoding
STRATEGY_PARAMS = model_config.STRATEGY_PARAMS_V8
from stock_eligibility_filter import StockEligibilityFilter


class PortfolioState:
    """投资组合状态管理器 - CSV版本，人工执行"""

    def __init__(self,  data_dir = str(REAL_TRADING_DIR / 'investment_data')):
        self.data_dir = data_dir
        self.cash = 0
        self.positions = {}  # {code: {avg_cost, shares, entry_date, stop_loss_price, take_profit_price, actual_sell_price, actual_sell_date, is_sold}}
        self.trade_suggestions = []  # 交易建议记录
        self.daily_assets = []  # 每日资产曲线
        self.last_run_date = None

        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 定义CSV文件路径
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

            # 1. 加载现金和最后运行日期
            if os.path.exists(self.cash_file):
                cash_df = pd.read_csv(self.cash_file)
                if not cash_df.empty:
                    self.cash = cash_df.iloc[0]['cash']
                    self.last_run_date = cash_df.iloc[0].get('last_run_date')

            # 2. 加载持仓信息
            if os.path.exists(self.positions_file):
                positions_df = pd.read_csv(self.positions_file)

                # 如果文件有数据才处理
                if not positions_df.empty:
                    # 将字符串日期转换为datetime
                    date_columns = ['entry_date', 'actual_sell_date', 'should_sell_date']
                    for col in date_columns:
                        if col in positions_df.columns:
                            positions_df[col] = pd.to_datetime(positions_df[col], errors='coerce')

                    # 分离已卖出和未卖出的持仓
                    sold_positions = positions_df[positions_df['is_sold'] == 'YES'].copy()
                    active_positions = positions_df[positions_df['is_sold'] == 'NO'].copy()

                    # 将已卖出的股票移动到历史文件
                    if not sold_positions.empty:
                        self._archive_sold_positions(sold_positions)

                        # 只保留未卖出的持仓
                        positions_df = active_positions

                    # 转换为字典格式
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
                        }

                    # 更新positions文件，只保留未卖出的持仓
                    positions_df.to_csv(self.positions_file, index=False)

            # # 3. 加载交易建议
            # if os.path.exists(self.suggestions_file):
            #     suggestions_df = pd.read_csv(self.suggestions_file)
            #     self.trade_suggestions = suggestions_df.to_dict('records')

            # 4. 加载资产曲线
            if os.path.exists(self.assets_file):
                assets_df = pd.read_csv(self.assets_file)
                self.daily_assets = assets_df.to_dict('records')

            logger.info(f"✅ CSV状态加载成功")
            logger.info(f"  现金: ¥{self.cash:,.2f}")
            logger.info(f"  持仓数量: {len(self.positions)}")
            logger.info(f"  最后运行日期: {self.last_run_date}")

            return True

        except Exception as e:
            logger.error(f"❌ 加载CSV状态失败: {e}")
            return False

    def _archive_sold_positions(self, sold_positions: pd.DataFrame) -> None:
        """将已卖出的持仓归档到历史文件"""
        try:
            history_file = os.path.join(self.data_dir, 'portfolio_positions_history.csv')

            # 如果历史文件存在，追加数据
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)

                # 合并新旧数据，避免重复
                combined_df = pd.concat([history_df, sold_positions], ignore_index=True)

                # 按卖出日期降序排列，最新的在前
                if 'actual_sell_date' in combined_df.columns:
                    # 将日期列统一转换为 datetime 类型
                    combined_df['actual_sell_date'] = pd.to_datetime(combined_df['actual_sell_date'], errors='coerce')
                    combined_df = combined_df.sort_values('actual_sell_date', ascending=False)
                    # 将 datetime 转换回字符串以便保存到 CSV
                    combined_df['actual_sell_date'] = combined_df['actual_sell_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                combined_df.to_csv(history_file, index=False)
                logger.info(f"📚 归档 {len(sold_positions)} 条已卖出持仓到历史文件，历史记录总数: {len(combined_df)}")
            else:
                # 如果 sold_positions 中有 datetime 类型，先转换为字符串
                if not sold_positions.empty and 'actual_sell_date' in sold_positions.columns:
                    sold_positions = sold_positions.copy()
                    sold_positions['actual_sell_date'] = pd.to_datetime(sold_positions['actual_sell_date']).dt.strftime(
                        '%Y-%m-%d %H:%M:%S')

                sold_positions.to_csv(history_file, index=False)
                logger.info(f"📚 创建历史文件并归档 {len(sold_positions)} 条已卖出持仓")

        except Exception as e:
            logger.error(f"❌ 归档已卖出持仓失败: {e}")

    def save_state(self, predict_date: datetime) -> bool:
        """保存当前投资状态到CSV文件"""
        try:
            # 1. 保存现金信息
            cash_df = pd.DataFrame([{
                'cash': self.cash,
                'last_run_date': predict_date.strftime('%Y-%m-%d'),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            cash_df.to_csv(self.cash_file, index=False)

            # 2. 保存持仓信息
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
                        'should_sell_date': pos.get('entry_date') + pd.Timedelta(days=STRATEGY_PARAMS['max_hold_days']),
                        'actual_sell_price': pos.get('actual_sell_price'),
                        'actual_sell_date': pos.get('actual_sell_date'),
                        'is_sold': pos.get('is_sold')
                    }

                    positions_list.append(pos_data)

                positions_df = pd.DataFrame(positions_list)
                positions_df.to_csv(self.positions_file, index=False)
            else:
                # 如果没有持仓，创建空文件
                pd.DataFrame(columns=['code', 'avg_cost', 'shares', 'entry_date',
                                     'stop_loss_price', 'take_profit_price',
                                     'actual_sell_price', 'actual_sell_date', 'is_sold']).to_csv(self.positions_file, index=False)

            # 3. 保存交易建议(不追加，每天生成一个)
            if self.trade_suggestions:
                suggestions_df = pd.DataFrame(self.trade_suggestions)
                suggestions_df.to_csv(f"{self.suggestions_file_prefix}_{predict_date.strftime('%Y%m%d')}.csv", index=False)

            # 4. 保存资产曲线（全部覆盖，不追加）
            if self.daily_assets:
                assets_df = pd.DataFrame(self.daily_assets)
                assets_df.to_csv(self.assets_file, index=False)

            # 5. 保存总结信息（JSON格式，方便查看）
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

            logger.info(f"💾 CSV状态已保存")
            logger.info(f"  现金: ¥{self.cash:,.2f}")
            logger.info(f"  未卖出持仓: {summary['positions_count']}只")
            logger.info(f"  总资产: ¥{total_assets:,.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ 保存CSV状态失败: {e}")
            return False

    def calculate_positions_value(self, price_data: pd.DataFrame = None) -> float:
        """计算未卖出持仓的市值"""
        total_value = 0
        for code, pos in self.positions.items():
            # 只计算未卖出的持仓
            if pos.get('is_sold') == 'NO':
                if price_data is not None and code in price_data['code'].values:
                    price_row = price_data[price_data['code'] == code].iloc[0]
                    price = price_row['close']
                else:
                    price = pos['avg_cost']  # 使用成本价作为默认

                total_value += price * pos['shares']
        return total_value

    def calculate_total_assets(self, price_data: pd.DataFrame = None) -> float:
        """计算总资产（现金 + 未卖出持仓市值）"""
        positions_value = self.calculate_positions_value(price_data)
        return self.cash + positions_value

    def get_summary(self) -> Dict:
        """获取投资组合摘要"""
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
    """智能狙击手投资执行器 - 人工执行版本"""

    def __init__(self, initial_capital: float = 200000, max_positions: int = 5, base_dir = REAL_TRADING_DIR):
        self.base_dir = str(base_dir)
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.portfolio_state = PortfolioState(str(base_dir / 'investment_data'))
        self.predictor = PriceChangePredictor(model_dir=str(MODEL_DIR))

        # 策略参数（与回测保持一致）
        self.max_positions = STRATEGY_PARAMS['max_positions']  # 最大持仓
        self.base_ratio = STRATEGY_PARAMS['base_ratio']  # 开仓比例
        self.target_profit = STRATEGY_PARAMS['target_profit']  # 止盈
        self.max_hold_days = STRATEGY_PARAMS['max_hold_days']  # 最长持股
        self.hard_stop_loss = STRATEGY_PARAMS['hard_stop_loss']  # 止损
        self.min_probability = STRATEGY_PARAMS['min_probability']  # 预测阈值

        # 加载历史状态
        if not self.portfolio_state.load_state():
            # 初始化新投资组合
            self.portfolio_state.cash = initial_capital
            logger.info(f"💰 初始化投资组合，起始资金: ¥{initial_capital:,.2f}")

        # 统一股票过滤器
        self.stock_filter = StockEligibilityFilter()


    @staticmethod
    def _load_prepare_data(file_pattern):
        """加载所有文件数据并按时间戳排序"""
        if isinstance(file_pattern, list):
            file_list = file_pattern
        else:
            file_list = glob.glob(file_pattern)

        logger.debug(f"✅找到 {len(file_list)} 个文件")
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
        # 按时间戳排序
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        logger.debug(f"✅数据时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")

        # BUGFIXED 这里没有对数据做预处理，导致模拟和真实场景不一致
        # 1. 去重
        before_count = len(combined_df)
        combined_df = combined_df.sort_values(['timestamp', 'symbol', 'confirmation_score']).drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        after_count = len(combined_df)
        logger.info(f"已删除 {before_count - after_count} 重复行，剩余 {after_count} 行数据")

        # 关键列数据无效剔除
        key_columns = ['timestamp'] + model_config.OPTIMIZED_FEATURE_COLS
        # 只删除关键列为 NaN 的行
        before_count = len(combined_df)
        combined_df = combined_df.dropna(subset=key_columns)
        after_count = len(combined_df)
        logger.info(f"已删除 {before_count - after_count} 个包含关键列 NaN 的行，剩余 {after_count} 行数据")

        return combined_df

    def prepare_realtime_data(self, target_date: datetime) -> Optional[pd.DataFrame]:
        """
        准备实时数据（模拟）
        实际应用中，这里应该连接实时数据API
        """
        logger.debug(f"📊 准备 {target_date.strftime('%Y-%m-%d')} 的数据...")

        try:
            # 这里模拟数据准备，实际应该从数据库或API获取
            features_dir = str(DAILY_FEATURE_DIR)
            file_pattern = os.path.join(features_dir,
                        f"realistic_features_{target_date.strftime('%Y%m%d')}.csv")

            # 加载特征数据
            raw_df = self._load_prepare_data(file_pattern)

            # 特征不存在，继续用真实数据跑逻辑
            if raw_df is None or len(raw_df) < 1:
                from grid_trading_simulation_v8 import load_price_data
                full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
                from grid_trading_simulation_v8 import convert_dict_to_dataframe_from_index
                full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)

                full_data_df = full_data_df[
                    full_data_df['timestamp'] == pd.to_datetime(target_date.strftime("%Y%m%d"))
                    ]
                df_result = full_data_df.rename(columns={
                    'timestamp': 'predict_date',
                    'symbol': 'code',
                })
                df_result['y_pred_proba'] = 0.0
                return df_result

            # 计算时间范围
            start_date = raw_df['timestamp'].min().strftime("%Y%m%d")
            end_date = raw_df['timestamp'].max().strftime("%Y%m%d")

            # 检查 full_data_df 的重复情况
            df_duplicates = raw_df.duplicated(subset=['symbol', 'timestamp']).sum()
            logger.debug(f"DF 中 (symbol, timestamp) 重复数量: {df_duplicates}")

            logger.debug("正在去除 raw_df 中的重复数据...")
            before_count_full = len(raw_df)
            raw_df = raw_df.drop_duplicates(subset=['symbol', 'timestamp'])
            after_count_full = len(raw_df)
            logger.debug(
                f"去重前: {before_count_full} 条, 去重后: {after_count_full} 条, 移除: {before_count_full - after_count_full} 条重复数据")

            # 初始化预测器
            # 保存symbol列
            symbol_info = raw_df[
                'symbol'].copy() if 'symbol' in raw_df.columns else pd.Series(
                ['N/A'] * len(raw_df))
            # 使用模型预测概率（排除symbol列）
            pred_probabilities = self.predictor.predict_proba(raw_df)
            # 直接赋值（顺序不会变）
            df_proba = raw_df.copy()
            df_proba['y_pred_proba'] = pred_probabilities
            # 恢复symbol信息
            df_proba['symbol'] = symbol_info

            from grid_trading_simulation_v8 import load_price_data
            full_data_dict = load_price_data(str(STOCK_ND_CSV_DIR))
            from grid_trading_simulation_v8 import convert_dict_to_dataframe_from_index
            full_data_df = convert_dict_to_dataframe_from_index(full_data_dict)

            full_data_df = full_data_df[
                (full_data_df['timestamp'] >= pd.to_datetime(start_date)) &
                (full_data_df['timestamp'] <= pd.to_datetime(end_date))
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
                'timestamp': 'predict_date',
                'symbol': 'code',
            })

            return df_result
        except Exception as e:
            logger.error(f"❌ 准备实时数据失败: {e}")
            return None

    def execute_daily_investment(self, predict_date: Optional[datetime] = None, feature_date:Optional[datetime] = None) -> bool:
        """
        执行明日投资决策（只生成建议，不实际执行）
        """
        if predict_date is None or feature_date is None:
            logger.error("predict_date或feature_date不存在")
            return False

        # 检查是否已经运行过
        if (self.portfolio_state.last_run_date and
                pd.to_datetime(self.portfolio_state.last_run_date).date() == predict_date.date()):
            logger.info(f"⏭️  今天 ({predict_date.date()}) 已经运行过，跳过")
            return True

        logger.debug(f"🚀 开始 {predict_date.strftime('%Y-%m-%d')} 的投资决策")

        # 1. 准备昨天的数据
        realtime_data = self.prepare_realtime_data(feature_date)


        # BUGFIX如果当天没有特征数据，不生成 买入建议，但要生成卖出建议啊！
        buy_suggestions = []
        if realtime_data is None or len(realtime_data) == 0:
            logger.error(f"❌ 无法获取{feature_date.strftime('%Y-%m-%d')}数据，停止执行")
        else:
            # 2. 生成交易建议（不实际执行）
            buy_suggestions = self._generate_buy_suggestions(realtime_data, predict_date)

        sell_suggestions = self._generate_sell_suggestions(predict_date)

        # 3. 记录交易建议
        self._record_trade_suggestions(predict_date, sell_suggestions, buy_suggestions)

        # 4. 更新投资组合价值（基于当前状态）
        self._update_portfolio_value(realtime_data, predict_date)

        # 5. 保存状态
        self.portfolio_state.save_state(predict_date)

        # 6. 生成投资报告
        self._generate_daily_report(predict_date, realtime_data, sell_suggestions, buy_suggestions)

        return True

    def _current_budget(self):
        return self.portfolio_state.cash / max(1, (self.max_positions - len(self.portfolio_state.positions)))


    def _generate_sell_suggestions(self, predict_date: datetime) -> List[Dict]:
        """
        生成卖出建议（不实际执行）
        """
        sell_suggestions = []

        for code, pos in self.portfolio_state.positions.items():
            # 只检查未卖出的持仓
            if pos.get('is_sold') == 'YES':
                logger.warning(f"{code}已经以{pos.get('actual_sell_price')}卖出了")
                continue


            # BUG FIXED 这个地方没有必要这么判断。已经持仓的股票，如果今天没有开盘，不代表明天不触发卖出信号
            current_price = pos['avg_cost']

            days_held = (predict_date - pos['entry_date']).days if pos['entry_date'] else 0

            # 计算建议止损价和止盈价
            stop_loss_price = pos['avg_cost'] * (1 + self.hard_stop_loss)
            take_profit_price = pos['avg_cost'] * (1 + self.target_profit)

            # OPTIMIZE 更新持仓中的建议价格（第一次操作后应该不更新，后续在优化吧）
            self.portfolio_state.positions[code]['stop_loss_price'] = stop_loss_price
            self.portfolio_state.positions[code]['take_profit_price'] = take_profit_price

            # 检查是否达到卖出条件
            should_sell = False
            reason = ""

            # # 止损检查
            # if stock_data['open'] <= stop_loss_price or stock_data['low'] <= stop_loss_price:
            #     should_sell = True
            #     reason = "STOP_LOSS"
            #     suggested_price = stop_loss_price
            #
            # # 止盈检查
            # elif stock_data['high'] >= take_profit_price:
            #     should_sell = True
            #     reason = "TAKE_PROFIT"
            #     suggested_price = take_profit_price

            # 持有时间检查(前一天提示)
            if days_held >= self.max_hold_days:
                should_sell = True
                reason = "TIME_EXIT"
                # OPTIMIZE 这里给一个建议最后一天的价格，乱写的
                suggested_price = pos['avg_cost'] * 0.98

            if should_sell:
                sell_suggestions.append({
                    'code': code,
                    'action': 'SELL',
                    'reason': reason,
                    'suggested_price': suggested_price,
                    'current_price': current_price,
                    'shares': pos['shares'],
                    'suggested_stop_loss': stop_loss_price,
                    'suggested_take_profit': take_profit_price,
                    'predict_date': predict_date,
                    # 以下是私有字段（和buy_suggestions不一样的）
                    'avg_cost': pos['avg_cost'],
                    'days_held': days_held,

                })

        return sell_suggestions

    def _generate_buy_suggestions(self, daily_data: pd.DataFrame, predict_date: datetime) -> List[Dict]:
        """
        生成买入建议（不实际执行）
        """
        buy_suggestions = []

        # 获取当前活跃持仓数量
        active_positions = [code for code, pos in self.portfolio_state.positions.items()
                          if pos.get('is_sold') == 'NO']
        active_count = len(active_positions)

        available_slots = self.max_positions - active_count
        if available_slots <= 0:
            return buy_suggestions

        # 1. 选股过滤
        current_codes = list(self.portfolio_state.positions.keys())
        candidates = daily_data[
            ~daily_data['code'].isin(current_codes) &
            daily_data['y_pred_proba'].notna() &
            (daily_data['y_pred_proba'] >= self.min_probability) &
            # FIXED BUG  这里过滤，而不在数据预处理过滤，避免simulation与 real_world 执行不一致的问题。
            (daily_data['close'] <= model_config.AFFORDABLE_PRICE)
            ].copy()

        # 统一过滤：主板 + ST + 次新股
        trade_date = predict_date.strftime('%Y%m%d')
        candidates = self.stock_filter.filter(candidates, trade_date)

        if candidates.empty:
            return buy_suggestions

        # 排序选择
        top_candidates = candidates.sort_values(
            ['y_pred_proba', 'code'],
            ascending=[False, True]
        ).head(int(available_slots * 2))

        total_frozen = 0.0
        for code, row in top_candidates.iterrows():
            remaining_slots = available_slots - len(buy_suggestions)
            if remaining_slots <= 0:
                break

            # 可用于分配的真实可用现金 = 当前现金 - 已冻结资金
            available_cash = self.portfolio_state.cash - total_frozen
            if available_cash < 100:  # 现金太少，跳出
                break

            signal_price = row['close']

            # 获取预测概率 (确保列名正确)
            proba = row['y_pred_proba']

            # --------------------------------------------------------------------------
            # [新增核心逻辑] 计算资金分配权重系数
            # --------------------------------------------------------------------------
            weight_factor = 1.0  # 默认为1.0 (即平均分配)

            if ALLOCATION_STRATEGY == 'z_score':
                # 方案一：Z-Score 统计加权 (推荐)
                # 逻辑：利用均值和标准差判断稀缺性。每增加1个标准差，仓位增加20%
                z_score = (proba - PROBA_MEAN) / PROBA_STD
                weight_factor = 1.0 + (0.2 * z_score)
                # 风险控制：限制系数在 [0.8, 1.4] 之间，防止过度偏离
                weight_factor = max(0.8, min(1.4, weight_factor))
            elif ALLOCATION_STRATEGY == 'tiered':
                # 方案二：分箱阶梯法
                if proba >= 0.72:  # > mean + 1std
                    weight_factor = 1.3
                elif proba >= 0.60:  # 核心区
                    weight_factor = 1.0
                else:  # < 0.60 基础区
                    weight_factor = 0.8

            # --------------------------------------------------------------------------
            # [修改资金计算] 应用权重系数
            # --------------------------------------------------------------------------
            # 基础平均预算
            base_budget_per_slot = available_cash / max(1, remaining_slots)

            # 应用权重系数 (核心修改点)
            weighted_budget = base_budget_per_slot * weight_factor

            # 最终预算 (乘用户策略比率)
            budget = weighted_budget * self.base_ratio

            # --------------------------------------------------------------------------
            # 后续原有逻辑保持不变
            # --------------------------------------------------------------------------

            # 以 signal_price 作为限价挂单价格（挂单阶段不知道 next_open）
            planned_shares = int(budget / signal_price / 100) * 100
            if planned_shares <= 100:
                # 如果本slot预算买不到100股，尝试把整个剩余可用现金用在这个slot（若只剩1个slot）
                if remaining_slots == 1:
                    planned_shares = int(available_cash / signal_price / 100) * 100
                    if planned_shares <= 100:
                        continue
                else:
                    continue
            required_cash = planned_shares * signal_price
            # 再次安全检查：如果剩余可用现金不够，降级planned_shares
            if required_cash > available_cash:
                max_shares = int(available_cash / signal_price / 100) * 100
                if max_shares < 100:
                    continue
                planned_shares = max_shares

            required_cash = planned_shares * signal_price
            # 冻结资金（不从 self.cash 扣除）
            total_frozen += required_cash

            suggested_stop_loss = signal_price * (1 + self.hard_stop_loss)
            suggested_take_profit = signal_price * (1 + self.target_profit)

            buy_suggestions.append({
                'code': row['code'],
                'action': 'BUY',
                'reason': 'SIGNAL',
                'suggested_price': signal_price,
                'current_price': row['close'],
                'shares': planned_shares,
                'suggested_stop_loss': suggested_stop_loss,
                'suggested_take_profit': suggested_take_profit,
                'predict_date': predict_date,
                #以下是私有字段（和sell_suggestions不一样的）
                'prediction_probability': row['y_pred_proba'],
                'required_capital': signal_price * planned_shares,

            })

        # 人为填写现金余额，程序不参与
        # self.portfolio_state.cash -= total_frozen

        return buy_suggestions

    def _record_trade_suggestions(self, predict_date: datetime, sell_suggestions: List[Dict], buy_suggestions: List[Dict]):
        """记录交易建议"""
        for suggestion in sell_suggestions + buy_suggestions:
            suggestion_record = {
                'predict_date': predict_date.strftime('%Y-%m-%d'),
                'code': suggestion['code'],
                'action': suggestion['action'],
                'reason': suggestion['reason'],
                'suggested_price': suggestion.get('suggested_price'),
                'current_price': suggestion.get('current_price'),
                'shares': suggestion.get('shares'),
                'suggested_stop_loss': suggestion.get('suggested_stop_loss') ,
                'suggested_take_profit': suggestion.get('suggested_take_profit'),

                # buy_suggestions字段
                'prediction_probability': suggestion.get('prediction_probability', 0.0),
                'required_capital': suggestion.get('required_capital', 0.0),
                # sell_suggestions字段
                'avg_cost': suggestion.get('avg_cost', 0.0),
                'days_held': suggestion.get('days_held', 0),
            }

            self.portfolio_state.trade_suggestions.append(suggestion_record)

    def _update_portfolio_value(self, daily_data: pd.DataFrame, today: datetime):
        """更新投资组合价值（基于当前状态）"""
        total_value = self.portfolio_state.calculate_total_assets(daily_data)

        self.portfolio_state.daily_assets.append({
            'predict_date': today.strftime('%Y-%m-%d'),
            'total': total_value,
            'cash': self.portfolio_state.cash,
            'positions_value': total_value - self.portfolio_state.cash
        })

    def _generate_daily_report(self, date: datetime, price_data: pd.DataFrame,
                             sell_suggestions: List[Dict], buy_suggestions: List[Dict]):
        """生成每日投资报告"""
        report_dir = os.path.join(self.base_dir, 'investment_reports')

        report_file = f"{report_dir}/report_{date.strftime('%Y%m%d')}.txt"

        os.makedirs(report_dir, exist_ok=True)

        # 计算总资产
        total_assets = self.portfolio_state.calculate_total_assets(price_data)
        total_return = (total_assets - self.initial_capital) / self.initial_capital

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"📊 智能狙击手投资日报（人工执行版本）\n")
            f.write(f"📅 日期: {date.strftime('%Y-%m-%d')}\n")
            f.write(f"⏰ 生成时间: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            # 投资组合摘要
            f.write("【投资组合摘要】\n")
            f.write(f"  现金余额: ¥{self.portfolio_state.cash:,.2f}\n")

            active_positions = [code for code, pos in self.portfolio_state.positions.items()
                              if pos.get('is_sold') == 'NO']
            f.write(f"  未卖出持仓: {len(active_positions)} 只\n")
            f.write(f"  总持仓数: {len(self.portfolio_state.positions)} 只\n")
            f.write(f"  持仓市值: ¥{self.portfolio_state.calculate_positions_value(price_data):,.2f}\n")
            f.write(f"  总资产: ¥{total_assets:,.2f}\n")
            f.write(f"  累计收益率: {total_return:.2%}\n")
            f.write(f"  起始资金: ¥{self.initial_capital:,.2f}\n\n")

            # 卖出建议
            f.write("【卖出建议】\n")
            if sell_suggestions:
                for i, suggestion in enumerate(sell_suggestions, 1):
                    f.write(f"  {i}. {suggestion['code']}: {suggestion['shares']}股\n")
                    f.write(f"     建议操作: 卖出\n")
                    f.write(f"     理由: {suggestion['reason']}\n")
                    f.write(f"     建议价格: ¥{suggestion['suggested_price']:.2f}\n")
                    f.write(f"     当前价格: ¥{suggestion['current_price']:.2f}\n")
                    f.write(f"     成本价格: ¥{suggestion['avg_cost']:.2f}\n")
                    f.write(f"     持仓天数: {suggestion['days_held']}天\n")
                    f.write(f"     建议止损价: ¥{suggestion['suggested_stop_loss']:.2f}\n")
                    f.write(f"     建议止盈价: ¥{suggestion['suggested_take_profit']:.2f}\n\n")
            else:
                f.write("  今日无卖出建议\n\n")

            # 买入建议
            f.write("【买入建议】\n")
            if buy_suggestions:
                for i, suggestion in enumerate(buy_suggestions, 1):
                    f.write(f"  {i}. {suggestion['code']}\n")
                    f.write(f"     建议操作: 买入\n")
                    f.write(f"     建议价格: ¥{suggestion['suggested_price']:.2f}\n")
                    f.write(f"     当前价格: ¥{suggestion['current_price']:.2f}\n")
                    f.write(f"     建议数量: {suggestion['shares']}股\n")
                    f.write(f"     预测概率: {suggestion['prediction_probability']:.2%}\n")
                    f.write(f"     所需资金: ¥{suggestion['required_capital']:,.2f}\n")
                    f.write(f"     建议止损价: ¥{suggestion['suggested_stop_loss']:.2f}\n")
                    f.write(f"     建议止盈价: ¥{suggestion['suggested_take_profit']:.2f}\n")
                    f.write(f"     建议卖出日期: {(suggestion['predict_date'] + timedelta(days=self.max_hold_days)).strftime('%Y-%m-%d')}\n\n")
            else:
                f.write("  今日无买入建议\n\n")

            # 当前持仓状态
            f.write("【当前持仓状态】\n")
            if active_positions:
                for code in active_positions:
                    pos = self.portfolio_state.positions[code]

                    # 获取当前价格
                    if price_data is not None and code in price_data['code'].values:
                        stock_row = price_data[price_data['code'] == code].iloc[0]
                        current_price = stock_row['close']
                    else:
                        current_price = pos['avg_cost']
                        logger.warning(f"{code} 的最新价格没找到")

                    market_value = current_price * pos['shares']
                    cost_value = pos['avg_cost'] * pos['shares']
                    pnl = market_value - cost_value
                    pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost']
                    days_held = (date - pos['entry_date']).days if pos['entry_date'] else 0

                    f.write(f"  {code}: {pos['shares']}股\n")
                    f.write(f"     成本价: ¥{pos['avg_cost']:.2f}\n")
                    f.write(f"     当前价: ¥{current_price:.2f}\n")
                    f.write(f"     持仓市值: ¥{market_value:,.2f}\n")
                    f.write(f"     浮动盈亏: ¥{pnl:,.2f} ({pnl_pct:.2%})\n")
                    f.write(f"     持仓天数: {days_held}天\n")

                    # 显示建议价格
                    if pos.get('stop_loss_price'):
                        f.write(f"     建议止损价: ¥{pos['stop_loss_price']:.2f}\n")
                    if pos.get('take_profit_price'):
                        f.write(f"     建议止盈价: ¥{pos['take_profit_price']:.2f}\n")

                    f.write("\n")
            else:
                f.write("  暂无未卖出持仓\n\n")

            # 已卖出持仓
            sold_positions = [code for code, pos in self.portfolio_state.positions.items()
                            if pos.get('is_sold') == 'YES']
            if sold_positions:
                f.write("【已卖出持仓】\n")
                for code in sold_positions:
                    pos = self.portfolio_state.positions[code]
                    profit = (pos['actual_sell_price'] - pos['avg_cost']) * pos['shares']
                    profit_pct = (pos['actual_sell_price'] - pos['avg_cost']) / pos['avg_cost']

                    f.write(f"  {code}: {pos['shares']}股\n")
                    f.write(f"     成本价: ¥{pos['avg_cost']:.2f}\n")
                    f.write(f"     卖出价: ¥{pos['actual_sell_price']:.2f}\n")
                    f.write(f"     卖出日期: {pos['actual_sell_date']}\n")
                    f.write(f"     实际盈亏: ¥{profit:,.2f} ({profit_pct:.2%})\n\n")

            # 操作说明
            f.write("【操作说明】\n")
            f.write("  1. 根据上述建议进行人工交易\n")
            f.write("  2. 交易完成后，请手动修改以下文件：\n")
            f.write("     - investment_data/portfolio_cash.csv: 更新现金余额\n")
            f.write("     - investment_data/portfolio_positions.csv: \n")
            f.write("       * 对于买入：添加新行，填写代码、成本价、股数、买入日期\n")
            f.write("       * 对于卖出：找到对应行，填写actual_sell_price和actual_sell_date\n")
            f.write("  3. 下次运行脚本时将自动加载修改后的状态\n")

        logger.info(f"📄 投资报告已生成: {report_file}")

    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        return self.portfolio_state.get_summary()


def run(predict_date = datetime.now(), feature_date = datetime.now(), real_trading_dir  = REAL_TRADING_DIR):
    """主函数"""

    logger.debug("=" * 60)
    logger.debug(f"🤖 智能狙击手投资系统启动（人工执行版本）")
    logger.debug("=" * 60)

    # 初始化投资器
    investor = SmartSniperInvestor(
        initial_capital=99817,
        max_positions=10,
        base_dir=real_trading_dir
    )

    # 获取投资组合状态
    summary = investor.get_portfolio_summary()
    logger.info(f"💰 当前现金: ¥{summary['cash']:,.2f}")
    logger.info(f"📦 未卖出持仓: {summary['active_positions_count']} 只")
    logger.info(f"📦 总持仓数: {summary['total_positions_count']} 只")

    # 执行明日投资决策（只生成建议）
    success = investor.execute_daily_investment(predict_date, feature_date)

    if success:
        logger.debug("✅ 今日投资建议生成完成")
        logger.debug("📋 请查看 investment_reports/ 目录下的报告文件")
        logger.debug("📝 根据建议进行人工交易后，请手动修改状态文件")

        # 显示状态文件位置
        logger.debug(f"📁 状态文件位置: investment_data/")
        logger.debug(f"  - portfolio_cash.csv: 现金信息（交易后请更新现金）")
        logger.debug(f"  - portfolio_positions.csv: 持仓信息（买卖后请更新）")
        logger.debug(f"  - trade_suggestions.csv: 交易建议记录")
        logger.debug(f"  - portfolio_assets.csv: 资产曲线")
        logger.debug(f"  - portfolio_summary.json: 投资总结")

    logger.debug("=" * 60)
    logger.debug("🤖 智能狙击手投资系统结束")
    logger.debug("=" * 60)

from feature_pipeline import load_price_data, feature_generator

if __name__ == "__main__":
    # 设置环境变量，添加项目根目录到PYTHONPATH
    # import argparse
    
    # 创建参数解析器
    # parser = argparse.ArgumentParser(description='SmartSniper 实盘投资脚本')
    # parser.add_argument('--predict-date', required=False, type=str, help='预测日期 (格式: YYYY-MM-DD)')
    # parser.add_argument('--feature-date', required=False, type=str, help='特征数据日期 (格式: YYYY-MM-DD)')
    
    # args = parser.parse_args()
    
    # 解析日期参数
    if args.predict_date:
        try:
            PREDICT_DATE = datetime.strptime(args.predict_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"预测日期格式错误: {args.predict_date}，应为 YYYY-MM-DD")
            exit(1)
    else:
        PREDICT_DATE = datetime.now() + timedelta(days=1)  # 昨天
    
    if args.feature_date:
        try:
            FEATURE_DATE = datetime.strptime(args.feature_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"特征日期格式错误: {args.feature_date}，应为 YYYY-MM-DD")
            exit(1)
    else:
        FEATURE_DATE = datetime.now() # 默认今天
    
    FEATURE_DATE_STR = FEATURE_DATE.strftime("%Y-%m-%d")

    # 优化后的主流程
    logger.info(f"3.开始特征工程: {FEATURE_DATE_STR}...")

    try:
        # 1. 生成特征
        features = feature_generator(FEATURE_DATE_STR)
        # features = pd.read_csv( DAILY_FEATURE_DIR / f"realistic_features_{FEATURE_DATE.strftime('%Y%m%d')}.csv")

        # 2.开始运行策略
        if features is not None:
            feature_count = len(features) if hasattr(features, '__len__') else "未知"
            logger.info(f"特征生成完成: {FEATURE_DATE_STR}, 生成 {feature_count} 个特征")

            logger.info(f"3.开始运行策略，预测日期: {PREDICT_DATE.date()}, 特征日期: {FEATURE_DATE_STR}")
            run(PREDICT_DATE, FEATURE_DATE)
            logger.info("策略执行完成")

        else:
            logger.error(f"特征生成失败: {FEATURE_DATE_STR}")

    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
    except ValueError as e:
        logger.error(f"数据验证失败: {e}")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        logger.exception("详细错误信息")
