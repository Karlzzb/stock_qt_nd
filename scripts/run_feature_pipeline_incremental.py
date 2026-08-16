#!/usr/bin/env python3
"""
增量特征计算脚本（改进版）。

改进点
------
1. 分批运行：可指定只跑某个日期范围的子集
2. 进度跟踪：记录已完成/失败的日期到状态文件
3. 断点续跑：出错后可以从上次位置继续
4. 失败重试：单独重试之前失败的日期
5. 更好的日志：每批完成后立即落盘，出错也能看到进度

用法示例
--------
# 1. 按月份分批跑（推荐）
uv run python scripts/run_feature_pipeline_incremental.py --year 2010 --month 1
uv run python scripts/run_feature_pipeline_incremental.py --year 2010 --month 2
...

# 2. 按季度跑
uv run python scripts/run_feature_pipeline_incremental.py --start-date 2010-01-01 --end-date 2010-03-31

# 3. 重试失败的日期
uv run python scripts/run_feature_pipeline_incremental.py --retry-failed

# 4. 查看进度
uv run python scripts/run_feature_pipeline_incremental.py --status

# 5. 继续未完成的工作（跳过已成功的日期）
uv run python scripts/run_feature_pipeline_incremental.py --start-date 2010-01-01 --end-date 2024-12-31 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 状态文件路径
STATE_FILE = REPO_ROOT / ".feature_pipeline_state.json"


class ProgressTracker:
    """跟踪特征计算进度"""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """加载状态文件"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}，使用空状态")
                return {"completed": [], "failed": {}, "last_update": None}
        return {"completed": [], "failed": {}, "last_update": None}

    def _save_state(self):
        """保存状态文件"""
        self.state["last_update"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")

    def mark_completed(self, date: str):
        """标记日期为已完成"""
        if date not in self.state["completed"]:
            self.state["completed"].append(date)
        # 从失败列表中移除
        if date in self.state["failed"]:
            del self.state["failed"][date]
        self._save_state()

    def mark_failed(self, date: str, error: str):
        """标记日期为失败"""
        self.state["failed"][date] = {
            "error": error[:200],  # 只保存前200字符
            "timestamp": datetime.now().isoformat(),
        }
        self._save_state()

    def is_completed(self, date: str) -> bool:
        """检查日期是否已完成"""
        return date in self.state["completed"]

    def get_failed_dates(self) -> List[str]:
        """获取所有失败的日期"""
        return list(self.state["failed"].keys())

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "completed": len(self.state["completed"]),
            "failed": len(self.state["failed"]),
            "last_update": self.state.get("last_update"),
        }

    def clear_failed(self):
        """清除失败记录（用于重试）"""
        self.state["failed"] = {}
        self._save_state()

    def print_status(self):
        """打印状态摘要"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("特征计算进度状态")
        print("=" * 60)
        print(f"✅ 已完成: {stats['completed']} 天")
        print(f"❌ 失败:   {stats['failed']} 天")
        print(f"📅 最后更新: {stats['last_update']}")

        if self.state["failed"]:
            print(f"\n失败的日期（前10个）:")
            for i, (date, info) in enumerate(list(self.state["failed"].items())[:10]):
                print(f"  {i+1}. {date}: {info['error'][:80]}")
            failed_count = len(self.state["failed"])
            if failed_count > 10:
                print(f"  ... 还有 {failed_count - 10} 个失败日期")
        print("=" * 60 + "\n")


def load_price_data_from_parquet(
    parquet_dir: Path,
    start_date: str = "2009-01-01",
    end_date: str | None = None,
) -> dict:
    """从 DAILY_PARQUET_DIR 加载行情数据"""
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"DAILY_PARQUET_DIR ({parquet_dir}) 下无 Parquet 文件"
        )

    logger.info(f"共发现 {len(parquet_files)} 只股票的 Parquet 文件，开始加载…")
    stocks: dict = {}

    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    for pq_file in parquet_files:
        ts_code = pq_file.stem
        try:
            df = pd.read_parquet(pq_file, columns=["trade_date", "open", "high", "low", "close", "vol"])
            df = df.rename(columns={"vol": "volume"})
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype("float64")
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            df["symbol"] = ts_code

            if start_dt is not None:
                df = df[df.index >= start_dt]
            if end_dt is not None:
                df = df[df.index <= end_dt]

            if len(df) > 0:
                stocks[ts_code] = df
        except Exception as exc:
            logger.warning(f"[{ts_code}] 加载失败：{exc}")

    logger.info(f"成功加载 {len(stocks)} 只股票。")
    return stocks


def process_dates_with_tracking(
    full_stocks: dict,
    dates_to_process: List[str],
    tracker: ProgressTracker,
    max_workers: int,
    batch_size: int = 50,
):
    """带进度跟踪的日期处理"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from feature_pipeline_v2 import (
        process_date_standalone_optimized,
        worker_initializer,
    )
    from tqdm import tqdm

    total_dates = len(dates_to_process)
    logger.info(f"总待处理: {total_dates} 天 | 进程数: {max_workers} | 批次大小: {batch_size}")

    success_count = 0
    failed_count = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_initializer,
        initargs=(full_stocks,)
    ) as executor:
        # 分批处理
        for batch_start in range(0, total_dates, batch_size):
            batch_end = min(batch_start + batch_size, total_dates)
            batch_dates = dates_to_process[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_dates + batch_size - 1) // batch_size

            logger.info(f"\n📦 批次 {batch_num}/{total_batches} (包含 {len(batch_dates)} 天)")

            futures = {
                executor.submit(process_date_standalone_optimized, date): date
                for date in batch_dates
            }

            with tqdm(total=len(batch_dates), desc=f"批次{batch_num}", unit="天") as pbar:
                for future in as_completed(futures):
                    date = futures[future]
                    pbar.update(1)

                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        if result is True:
                            tracker.mark_completed(date)
                            success_count += 1
                        else:
                            error_msg = str(result) if result else "未知错误"
                            tracker.mark_failed(date, error_msg)
                            failed_count += 1
                            logger.warning(f"  ⚠️ {date} 失败: {error_msg[:80]}")
                    except Exception as e:
                        error_msg = f"进程级崩溃: {str(e)}"
                        tracker.mark_failed(date, error_msg)
                        failed_count += 1
                        logger.error(f"  ❌ {date} 崩溃: {str(e)[:80]}")

            # 每批完成后打印进度
            logger.info(f"  ✅ 本批成功: {success_count} | ❌ 失败: {failed_count}")

    logger.info(f"\n{'='*60}")
    logger.info(f"处理完成！✅ 成功: {success_count} | ❌ 失败: {failed_count}")
    logger.info(f"{'='*60}\n")


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """生成日期范围列表"""
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return dates


def main():
    parser = argparse.ArgumentParser(description="增量特征计算脚本（改进版）")

    # 日期范围选项
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--year", type=int, help="处理指定年份的所有数据")
    date_group.add_argument("--start-date", help="起始日期 YYYY-MM-DD")

    parser.add_argument("--month", type=int, help="配合 --year 使用，处理指定月份（1-12）")
    parser.add_argument("--end-date", help="截止日期 YYYY-MM-DD")

    # 进度控制选项
    parser.add_argument("--resume", action="store_true", help="跳过已完成的日期，继续处理")
    parser.add_argument("--retry-failed", action="store_true", help="重试所有失败的日期")
    parser.add_argument("--status", action="store_true", help="显示当前进度状态")
    parser.add_argument("--clear-failed", action="store_true", help="清除失败记录")

    # 性能选项
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="并发进程数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="每批处理的天数（默认50，调小可以更频繁保存进度）",
    )

    args = parser.parse_args()

    from config.settings import DAILY_PARQUET_DIR, DAILY_FEATURE_DIR

    tracker = ProgressTracker()

    # 仅显示状态
    if args.status:
        tracker.print_status()
        return

    # 清除失败记录
    if args.clear_failed:
        tracker.clear_failed()
        logger.info("已清除所有失败记录")
        return

    # 确定日期范围
    if args.retry_failed:
        failed_dates = tracker.get_failed_dates()
        if not failed_dates:
            logger.info("没有失败的日期需要重试")
            return
        dates_to_check = failed_dates
        logger.info(f"重试模式：将重试 {len(failed_dates)} 个失败的日期")
        tracker.clear_failed()  # 清除失败记录，准备重试
    else:
        # 确定日期范围
        if args.year:
            if args.month:
                # 指定年月
                start_date = f"{args.year}-{args.month:02d}-01"
                if args.month == 12:
                    end_date = f"{args.year}-12-31"
                else:
                    next_month = datetime(args.year, args.month + 1, 1)
                    end_date = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"处理范围: {args.year} 年 {args.month} 月")
            else:
                # 整年
                start_date = f"{args.year}-01-01"
                end_date = f"{args.year}-12-31"
                logger.info(f"处理范围: {args.year} 年全年")
        else:
            start_date = args.start_date or "2010-01-01"
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')

        dates_to_check = get_date_range(start_date, end_date)

    logger.info(f"日期范围: {dates_to_check[0]} 至 {dates_to_check[-1]} (共 {len(dates_to_check)} 天)")

    # 过滤已完成的日期（如果启用了 resume）
    if args.resume and not args.retry_failed:
        original_count = len(dates_to_check)
        dates_to_check = [d for d in dates_to_check if not tracker.is_completed(d)]
        skipped = original_count - len(dates_to_check)
        if skipped > 0:
            logger.info(f"📋 续跑模式：跳过 {skipped} 个已完成的日期")

    if not dates_to_check:
        logger.info("所有日期都已完成，无需处理")
        tracker.print_status()
        return

    # 加载数据 - 需要加载比目标日期更早的数据以支持特征计算
    # 特征计算需要 FEATURE_NEED_MAX_DAYS (100) + 250 = 350 天历史
    # 为了安全，再往前推 200 天，确保有足够的数据
    earliest_target_date = datetime.strptime(dates_to_check[0], '%Y-%m-%d')
    data_load_start = (earliest_target_date - timedelta(days=550)).strftime('%Y-%m-%d')

    logger.info(f"\n开始加载行情数据...")
    logger.info(f"  Parquet 目录: {DAILY_PARQUET_DIR}")
    logger.info(f"  Feature 目录: {DAILY_FEATURE_DIR}")
    logger.info(f"  数据加载起始: {data_load_start} (目标日期 - 550天)")
    logger.info(f"  并发进程数: {args.workers}")
    logger.info(f"  批次大小: {args.batch_size}")

    full_stocks = load_price_data_from_parquet(
        DAILY_PARQUET_DIR,
        start_date=data_load_start,  # 动态计算加载起始日期
        end_date=None,
    )

    # 处理日期
    logger.info("\n开始批量特征计算...")
    process_dates_with_tracking(
        full_stocks,
        dates_to_check,
        tracker,
        max_workers=args.workers,
        batch_size=args.batch_size,
    )

    # 显示最终状态
    tracker.print_status()


if __name__ == "__main__":
    main()
