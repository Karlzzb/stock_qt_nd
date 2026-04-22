"""
每日交易自动执行脚本
功能：
1. 自动判断是否为交易日
2. 自动计算 FEATURE_DATE（今天）和 PREDICT_DATE（下一个交易日）
3. 顺序执行：stock_nd.py -> grid_trading_realworld_v8.py
4. 完整的错误处理和日志记录
5. 支持跳过指定步骤（通过参数传递）

使用方法：
    # 正常执行所有步骤
    python daily_trading_executor.py

    # 在 main() 函数中修改参数：
    skip_stock_data = True  # 跳过 股票数据加载
"""


import sys
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path


# 设置日志
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f'daily_executor_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingDayChecker:
    """交易日检查器"""
    
    def __init__(self):
        self.holidays_2026 = [
            # 元旦
            '2026-01-01', '2026-01-02', '2026-01-03',
            # 春节
            '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22',
            # 清明节
            '2026-04-04', '2026-04-05', '2026-04-06',
            # 劳动节
            '2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05',
            # 端午节
            '2026-06-25', '2026-06-26', '2026-06-27',
            # 中秋节
            '2026-10-01', '2026-10-02', '2026-10-03',
            # 国庆节
            '2026-10-01', '2026-10-02', '2026-10-03', '2026-10-04', '2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08',
        ]
        
        # 调休工作日（周末需要上班的日子）
        self.workdays_2026 = [
            '2026-02-15',  # 春节调休
            '2026-02-28',  # 春节调休
            '2026-09-27',  # 国庆调休
            '2026-10-10',  # 国庆调休
        ]
    
    def is_trading_day(self, date: datetime) -> bool:
        """判断是否为交易日"""
        date_str = date.strftime('%Y-%m-%d')
        
        # 1. 如果是调休工作日，是交易日
        if date_str in self.workdays_2026:
            return True
        
        # 2. 如果是法定节假日，不是交易日
        if date_str in self.holidays_2026:
            return False
        
        # 3. 如果是周末，不是交易日
        if date.weekday() >= 5:  # 5=周六, 6=周日
            return False
        
        # 4. 其他情况是交易日
        return True
    
    def get_next_trading_day(self, date: datetime) -> datetime:
        """获取下一个交易日"""
        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """获取上一个交易日"""
        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day


class DailyTradingExecutor:
    """每日交易执行器"""
    
    def __init__(self, skip_stock_data=False):
        """
        初始化执行器

        Args:
            skip_stock_data: 是否跳过股票数据加载步骤
        """
        self.script_dir = Path(__file__).parent
        self.day_checker = TradingDayChecker()

        # 脚本路径
        self.stock_data_script = self.script_dir / 'stock_nd.py'
        self.trading_script = self.script_dir / 'grid_trading_realworld_v8.py'

        # 跳过选项
        self.skip_stock_data = skip_stock_data
    
    def run_script(self, script_path: Path, script_name: str) -> bool:
        """运行Python脚本，实时显示输出"""
        logger.info(f"{'='*60}")
        logger.info(f"开始执行: {script_name}")
        logger.info(f"脚本路径: {script_path}")
        logger.info(f"{'='*60}")
        
        try:
            # 设置环境变量，添加项目根目录到PYTHONPATH
            import os
            env = os.environ.copy()
            project_root = str(self.script_dir.parent)
            
            # 添加项目根目录到PYTHONPATH
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root
            
            logger.info(f"设置 PYTHONPATH: {project_root}")
            
            # 使用Popen实现实时输出
            # -u 参数：强制Python使用无缓冲模式，确保输出立即显示
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                cwd=str(self.script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1,  # 行缓冲
                universal_newlines=True,
                env=env  # 传递修改后的环境变量
            )
            
            # 实时读取并输出stdout
            import threading
            
            def read_stdout():
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.debug(f"  {line}")
            
            def read_stderr():
                for line in process.stderr:
                    line = line.rstrip()
                    if line:
                        logger.info(f"  [stderr] {line}")
            
            # 创建线程读取输出
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            
            stdout_thread.start()
            stderr_thread.start()
            
            # 等待进程结束（最多1小时）
            try:
                returncode = process.wait(timeout=36000)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"❌ {script_name} 执行超时（超过1小时）")
                return False
            
            # 等待输出线程结束
            stdout_thread.join()
            stderr_thread.join()
            
            # 检查返回码
            if returncode == 0:
                logger.info(f"✅ {script_name} 执行成功")
                return True
            else:
                logger.error(f"❌ {script_name} 执行失败，返回码: {returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {script_name} 执行异常: {e}")
            logger.exception("详细错误信息:")
            return False
    
    def check_stock_data_loaded(self) -> bool:
        """检查股票数据是否加载完成"""
        logger.info("="*60)
        logger.info("检查股票数据加载状态...")
        logger.info("="*60)
        
        # 1. 检查 stock_data 目录
        stock_data_dir = self.script_dir.parent / 'stock_data'
        if not stock_data_dir.exists():
            logger.error("❌ stock_data 目录不存在")
            return False
        
        # 2. 统计pkl文件数量
        pkl_files = list(stock_data_dir.glob('*.pkl'))
        logger.info(f"📊 股票数据文件数量: {len(pkl_files)}")
        
        # 3. 检查指数数据（必须存在）
        required_indices = ['000001.SH', '399001.SZ']
        missing_indices = []
        for index in required_indices:
            index_file = stock_data_dir / f'{index}_price_data.pkl'
            if not index_file.exists():
                missing_indices.append(index)
        
        if missing_indices:
            logger.error(f"❌ 缺少指数数据: {missing_indices}")
            return False
        
        # 4. 检查最小股票数量
        min_required_stocks = 100
        if len(pkl_files) < min_required_stocks:
            logger.error(f"❌ 股票数据文件数量不足: {len(pkl_files)} < {min_required_stocks}")
            return False
        
        logger.info("="*60)
        logger.info(f"✅ 股票数据检查通过")
        logger.info("="*60)
        
        return True
    
    def generate_execution_summary(self, feature_date: datetime, predict_date: datetime, 
                                  success: bool, execution_time: float):
        """生成执行摘要报告"""
        summary_dir = self.script_dir / 'logs' / 'summaries'
        summary_dir.mkdir(exist_ok=True)
        
        summary_file = summary_dir / f'execution_summary_{feature_date.strftime("%Y%m%d")}.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("每日交易执行摘要\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"执行日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"特征日期 (FEATURE_DATE): {feature_date.strftime('%Y-%m-%d')}\n")
            f.write(f"预测日期 (PREDICT_DATE): {predict_date.strftime('%Y-%m-%d')}\n")
            f.write(f"执行状态: {'✅ 成功' if success else '❌ 失败'}\n")
            f.write(f"执行耗时: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)\n\n")
            
            f.write(f"详细日志: logs/daily_executor_{feature_date.strftime('%Y%m%d')}.log\n")
        
        logger.info(f"📄 执行摘要已生成: {summary_file}")
    
    def execute_daily_trading(self, feature_date: datetime = None, predict_date: datetime = None):
        """执行每日交易流程"""
        import time
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("🚀 每日交易自动执行脚本启动")
        logger.info("="*80)
        
        # 显示跳过选项
        if self.skip_stock_data:
            logger.info("⚙️  执行选项:")
            if self.skip_stock_data:
                logger.info("   ⏭️  跳过 股票数据加载 (stock_nd.py)")
            logger.info("")
        
        # 1. 确定日期
        if feature_date is None:
            feature_date = datetime.now()
        
        if predict_date is None:
            predict_date = self.day_checker.get_next_trading_day(feature_date)
        
        logger.info(f"📅 特征日期 (FEATURE_DATE): {feature_date.strftime('%Y-%m-%d')}")
        logger.info(f"📅 预测日期 (PREDICT_DATE): {predict_date.strftime('%Y-%m-%d')}")
        
        # 2. 检查今天是否为交易日
        if not self.day_checker.is_trading_day(feature_date):
            logger.info(f"⏭️  今天 ({feature_date.strftime('%Y-%m-%d')}) 不是交易日，跳过执行")
            return False
        
        success = False
        
        try:
            # 3. 执行 stock_nd.py（可选）
            if self.skip_stock_data:
                logger.info("\n" + "="*80)
                logger.info("步骤 1/2: 股票数据加载 [已跳过]")
                logger.info("="*80)
                logger.info("⏭️  根据配置跳过 股票数据加载步骤")
                logger.info("⚠️  警告: 请确保股票数据已经是最新的！")
            else:
                logger.info("\n" + "="*80)
                logger.info("步骤 1/2: 执行股票数据加载")
                logger.info("="*80)
                
                if not self.run_script(self.stock_data_script, "stock_nd.py"):
                    logger.error("❌ 股票数据加载失败，终止执行")
                    return False

            # 4. 检查股票数据是否加载完成（即使跳过加载也要检查）
            if not self.check_stock_data_loaded():
                logger.error("❌ 股票数据加载不完整，终止执行")
                return False

            # 5. 修改 grid_trading_realworld_v8.py 的日期参数并执行
            logger.info("\n" + "="*80)
            logger.info("步骤 2/2: 执行网格交易策略")
            logger.info("="*80)
            
            # 读取原始脚本
            with open(self.trading_script, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # 替换日期参数
            # 查找并替换 PREDICT_DATE 和 FEATURE_DATE
            import re
            
            # 替换 PREDICT_DATE
            predict_pattern = r'PREDICT_DATE\s*=\s*datetime\([^)]+\)'
            predict_replacement = f'PREDICT_DATE = datetime({predict_date.year}, {predict_date.month}, {predict_date.day})'
            script_content = re.sub(predict_pattern, predict_replacement, script_content)
            
            # 替换 FEATURE_DATE
            feature_pattern = r'FEATURE_DATE\s*=\s*datetime\([^)]+\)'
            feature_replacement = f'FEATURE_DATE = datetime({feature_date.year}, {feature_date.month}, {feature_date.day})'
            script_content = re.sub(feature_pattern, feature_replacement, script_content)
            
            # 创建临时脚本
            temp_script = self.script_dir / 'temp_grid_trading.py'
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            try:
                # 执行临时脚本
                success = self.run_script(temp_script, "grid_trading_realworld_v8.py")
                
                if success:
                    logger.info("\n" + "="*80)
                    logger.info("✅ 每日交易执行完成")
                    logger.info("="*80)
                    logger.info(f"📊 请查看报告: real_features_processors_v2/real_trading_data/investment_reports/")
                    logger.info(f"📝 请根据建议进行人工交易，并更新状态文件")
                else:
                    logger.error("❌ 网格交易策略执行失败")
                    
            finally:
                # 清理临时文件
                if temp_script.exists():
                    temp_script.unlink()
                    logger.debug("🗑️  临时脚本已清理")
        
        finally:
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 生成执行摘要
            self.generate_execution_summary(feature_date, predict_date, success, execution_time)
            
            logger.info("\n" + "="*80)
            logger.info(f"⏱️  总执行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")
            logger.info("="*80)
        
        return success


def main():
    """
    主函数

    在这里修改参数来控制执行行为：
    - skip_stock_data: 是否跳过 股票数据加载步骤
    - feature_date: 特征日期（默认为今天）
    - predict_date: 预测日期（默认为下一个交易日）
    """
    try:
        # ==================== 参数设置 ====================
        # 跳过选项
        skip_stock_data = True  # True=跳过股票数据加载, False=执行股票数据加载

        # 日期设置（None表示使用默认值）
        feature_date = None # 特征日期，None=今天，或指定如: datetime(2026, 2, 6)
        predict_date = None  # 预测日期，None=下一个交易日，或指定如: datetime(2026, 2, 9)
        # =================================================

        # 创建执行器
        executor = DailyTradingExecutor(
            skip_stock_data=skip_stock_data
        )
        
        # 执行交易流程
        success = executor.execute_daily_trading(feature_date, predict_date)
        
        if success:
            logger.info("✅ 程序执行成功")
            sys.exit(0)
        else:
            logger.error("❌ 程序执行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
