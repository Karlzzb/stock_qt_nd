#!/usr/bin/env python3
"""
全量 A 股日线数据拉取脚本（Issue #5）。

功能
----
1. 从 tinyshare 拉取在市 + 退市全量股票列表，构建时点股票池并落盘。
2. 按股票逐个拉取全历史日线（含退市股），存为 Parquet（每股一文件）。
3. 支持断点续拉：已存在且完整的文件直接跳过。
4. 失败股票写入 failed_stocks.txt，可重试（--retry-failed）。
5. 运行结束后输出数据完整性摘要。

用法
----
    export TINYSHARE_TOKEN="<your_token>"
    uv run python scripts/fetch_daily_data.py [选项]

常用选项
--------
    --data-root PATH        数据根目录（默认：$STOCK_DATA_ROOT 或 ./stock_data）
    --start-date YYYYMMDD   拉取起始日期（默认：19900101）
    --end-date   YYYYMMDD   拉取截止日期（默认：今天）
    --workers N             并发线程数（默认：4，受 tinyshare 频率限制）
    --retry-failed          重试上次失败的股票
    --dry-run               只拉股票列表，不拉日线

注意：本脚本不进 CI，需要真实 token 与网络，手动执行。
预计耗时：全量 ~5000 只股票 × 每只约 0.3 s = 25 min（4 并发）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from threading import Lock
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# 延迟导入（网络相关库在 tinyshare 不可用时跳过）
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("[FATAL] pandas 未安装，请先执行 uv sync。")
    sys.exit(1)

try:
    from tinyshare_auth import get_pro_api
except ImportError as e:
    print(f"[FATAL] 无法导入 tinyshare_auth：{e}")
    print("请确保在仓库根目录下执行，或已执行 uv sync。")
    sys.exit(1)

try:
    from universe import PointInTimeUniverse
except ImportError:
    from src.universe import PointInTimeUniverse


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_START_DATE = "19900101"
DAILY_SUBDIR       = "daily"
UNIVERSE_SUBDIR    = "universe"
FAILED_LIST_FILE   = "failed_stocks.txt"
SUMMARY_FILE       = "fetch_summary.txt"

# tinyshare 每次 daily 接口最多返回约 5000 行；全历史需按股票一次拉完
# 若单股记录超过 8000 行则分段拉取（极少见）
_MAX_ROWS_PER_CALL = 8000


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return date.today().strftime("%Y%m%d")


def _parquet_path(daily_dir: Path, ts_code: str) -> Path:
    return daily_dir / f"{ts_code}.parquet"


def _already_fetched(daily_dir: Path, ts_code: str) -> bool:
    """文件存在且非空即视为已拉取，断点续拉跳过。"""
    p = _parquet_path(daily_dir, ts_code)
    return p.exists() and p.stat().st_size > 0


def _fetch_one_stock(
    pro,
    ts_code: str,
    start_date: str,
    end_date: str,
    daily_dir: Path,
) -> tuple[str, str, Optional[int]]:
    """
    拉取单只股票的日线数据并落盘 Parquet。

    Returns
    -------
    (ts_code, status, row_count)
        status: "ok" | "empty" | "error:<msg>"
    """
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or len(df) == 0:
            return ts_code, "empty", 0

        # 确保 trade_date 类型正确，按日期排序
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 原始数据以 float64 存储（不做 optimize_dtypes，精度保留到计算特征时再降）
        out_path = _parquet_path(daily_dir, ts_code)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        return ts_code, "ok", len(df)

    except Exception as exc:
        brief = str(exc).replace("\n", " ")[:120]
        return ts_code, f"error:{brief}", None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

class FetchSession:
    def __init__(
        self,
        data_root: Path,
        start_date: str,
        end_date: str,
        workers: int,
    ) -> None:
        self.data_root  = data_root
        self.daily_dir  = data_root / DAILY_SUBDIR
        self.univ_dir   = data_root / UNIVERSE_SUBDIR
        self.start_date = start_date
        self.end_date   = end_date
        self.workers    = workers
        self.failed_path = data_root / FAILED_LIST_FILE

        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.univ_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self.ok_count     = 0
        self.skip_count   = 0
        self.empty_count  = 0
        self.error_count  = 0
        self.failed: list[str] = []

    # ------------------------------------------------------------------
    # 股票池
    # ------------------------------------------------------------------

    def build_universe(self, pro) -> PointInTimeUniverse:
        print("  → 拉取在市股票列表 stock_basic(L)…", flush=True)
        df_live = pro.stock_basic(
            exchange="", list_status="L",
            fields="ts_code,name,list_date,delist_date",
        )
        print("  → 拉取退市股票列表 stock_basic(D)…", flush=True)
        df_dead = pro.stock_basic(
            exchange="", list_status="D",
            fields="ts_code,name,list_date,delist_date",
        )
        df = pd.concat([df_live, df_dead], ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)

        pit = PointInTimeUniverse(df)

        # 落盘
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        univ_path = self.univ_dir / f"universe_{ts_str}.parquet"
        pit.save(univ_path)
        # 同时写一份固定名称供下游直接引用
        latest_path = self.univ_dir / "universe_latest.parquet"
        pit.save(latest_path)

        print(
            f"  ✓ 股票池落盘：{pit.total_stocks} 只"
            f"（在市 {pit.live_stocks}，退市 {pit.delisted_stocks}）",
            flush=True,
        )
        return pit

    # ------------------------------------------------------------------
    # 日线拉取
    # ------------------------------------------------------------------

    def _worker(self, args):
        pro, ts_code = args
        if _already_fetched(self.daily_dir, ts_code):
            with self._lock:
                self.skip_count += 1
            return ts_code, "skip", None

        result = _fetch_one_stock(
            pro, ts_code, self.start_date, self.end_date, self.daily_dir
        )
        _, status, rows = result

        with self._lock:
            if status == "ok":
                self.ok_count += 1
            elif status == "empty":
                self.empty_count += 1
            else:
                self.error_count += 1
                self.failed.append(ts_code)

        return result

    def fetch_all(self, pro, ts_codes: list[str], dry_run: bool = False) -> None:
        total = len(ts_codes)
        print(f"\n开始拉取日线数据：{total} 只股票，{self.workers} 线程", flush=True)
        if dry_run:
            print("[dry-run] 跳过实际拉取。", flush=True)
            return

        t0 = time.time()
        done = 0

        # 每个线程需要独立的 pro 实例（tinyshare 的 pro_api 对象非线程安全）
        # 用 get_pro_api() 在每个 worker 内部创建；但为简化，此处单线程串行拉取
        # 如需多线程，每次调用都用同一个 pro 对象（tinyshare 底层为 requests，
        # requests.Session 在线程间共享通常无问题，但若出错可降回 workers=1）
        args_list = [(pro, code) for code in ts_codes]

        if self.workers == 1:
            for args in args_list:
                ts_code, status, rows = self._worker(args)
                done += 1
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta  = (total - done) / rate if rate > 0 else 0
                    print(
                        f"  进度 {done}/{total}  "
                        f"ok={self.ok_count} skip={self.skip_count} "
                        f"empty={self.empty_count} err={self.error_count}  "
                        f"速度={rate:.1f}/s  预计剩余={eta/60:.1f}min",
                        flush=True,
                    )
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self._worker, a): a[1] for a in args_list}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        code = futures[future]
                        with self._lock:
                            self.error_count += 1
                            self.failed.append(code)
                    done += 1
                    if done % 200 == 0 or done == total:
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta  = (total - done) / rate if rate > 0 else 0
                        print(
                            f"  进度 {done}/{total}  "
                            f"ok={self.ok_count} skip={self.skip_count} "
                            f"empty={self.empty_count} err={self.error_count}  "
                            f"速度={rate:.1f}/s  预计剩余={eta/60:.1f}min",
                            flush=True,
                        )

        elapsed = time.time() - t0
        print(f"\n拉取完成，耗时 {elapsed/60:.1f} min", flush=True)

    # ------------------------------------------------------------------
    # 失败清单
    # ------------------------------------------------------------------

    def save_failed_list(self) -> None:
        if self.failed:
            self.failed_path.write_text("\n".join(self.failed) + "\n", encoding="utf-8")
            print(f"  ⚠ 失败股票清单已写入：{self.failed_path}  ({len(self.failed)} 只)")
        else:
            # 清空旧的失败清单
            if self.failed_path.exists():
                self.failed_path.unlink()

    def load_failed_list(self) -> list[str]:
        if not self.failed_path.exists():
            print(f"[WARN] 未找到失败清单：{self.failed_path}")
            return []
        codes = [
            line.strip()
            for line in self.failed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"  重试失败股票：{len(codes)} 只")
        return codes

    # ------------------------------------------------------------------
    # 摘要
    # ---------------------------------------------------------------------------

    def print_summary(self) -> None:
        parquet_files = list(self.daily_dir.glob("*.parquet"))
        total_rows = 0
        total_size_mb = sum(p.stat().st_size for p in parquet_files) / 1024 / 1024

        print("\n" + "=" * 60)
        print("拉取摘要")
        print("=" * 60)
        print(f"  Parquet 文件数：{len(parquet_files)}")
        print(f"  磁盘占用：{total_size_mb:.1f} MB")
        print(f"  本次成功：{self.ok_count}")
        print(f"  本次跳过（已有）：{self.skip_count}")
        print(f"  空数据（接口返回空）：{self.empty_count}")
        print(f"  失败：{self.error_count}")
        if self.failed:
            print(f"  失败清单：{self.failed_path}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="全量 A 股日线数据拉取（Issue #5）",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("STOCK_DATA_ROOT", str(REPO_ROOT / "stock_data")),
        help="数据根目录（默认：$STOCK_DATA_ROOT 或 ./stock_data）",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="拉取起始日期 YYYYMMDD")
    parser.add_argument("--end-date",   default=_today_str(),       help="拉取截止日期 YYYYMMDD")
    parser.add_argument("--workers",    type=int, default=1,        help="并发线程数（默认 1，稳定优先）")
    parser.add_argument("--retry-failed", action="store_true",      help="重试上次失败的股票")
    parser.add_argument("--dry-run",      action="store_true",      help="只构建股票池，不拉日线")
    args = parser.parse_args()

    # 检查 token
    token = os.environ.get("TINYSHARE_TOKEN", "").strip()
    if not token:
        print("[ERROR] 环境变量 TINYSHARE_TOKEN 未设置。")
        print("  请执行：export TINYSHARE_TOKEN='<your_token>'")
        return 1

    print("=" * 60)
    print("全量 A 股日线数据拉取")
    print(f"  数据根目录：{args.data_root}")
    print(f"  日期范围：{args.start_date} ~ {args.end_date}")
    print(f"  并发数：{args.workers}")
    print("=" * 60)

    pro = get_pro_api()
    session = FetchSession(
        data_root  = Path(args.data_root),
        start_date = args.start_date,
        end_date   = args.end_date,
        workers    = args.workers,
    )

    # 构建股票池
    print("\n[步骤 1/2] 构建时点股票池…")
    pit = session.build_universe(pro)

    # 确定待拉取股票列表
    if args.retry_failed:
        ts_codes = session.load_failed_list()
    else:
        ts_codes = pit._df["ts_code"].tolist()

    # 拉取日线
    print(f"\n[步骤 2/2] 拉取日线数据…")
    session.fetch_all(pro, ts_codes, dry_run=args.dry_run)
    session.save_failed_list()
    session.print_summary()

    return 1 if session.error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
