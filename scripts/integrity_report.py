#!/usr/bin/env python3
"""
数据完整性报告脚本（Issue #5 验收标准第 5 条）。

对已拉取的日线 Parquet 数据进行统计，输出：
- 总股票数、日期覆盖范围
- 退市股覆盖情况
- 每只股票的记录数分布
- 数据缺口（有上市记录但无 Parquet 文件的股票列表）

用法
----
    uv run python scripts/integrity_report.py [--data-root PATH] [--report-dir PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import pandas as pd
except ImportError:
    print("[FATAL] pandas 未安装，请先执行 uv sync。")
    sys.exit(1)

try:
    from universe import PointInTimeUniverse
except ImportError:
    from src.universe import PointInTimeUniverse


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def build_report(data_root: Path) -> str:
    daily_dir = data_root / "daily"
    univ_dir  = data_root / "universe"

    lines: list[str] = [
        "# A 股日线数据完整性报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据根目录：`{data_root}`",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. 股票池信息
    # ------------------------------------------------------------------
    latest_univ = univ_dir / "universe_latest.parquet"
    pit: PointInTimeUniverse | None = None

    if latest_univ.exists():
        try:
            pit = PointInTimeUniverse.load(latest_univ)
            lines += [
                "## 时点股票池",
                "",
                f"- 来源：`{latest_univ}`",
                f"- 总计：**{pit.total_stocks}** 只",
                f"  - 在市（delist_date 为空）：{pit.live_stocks} 只",
                f"  - 已退市：{pit.delisted_stocks} 只",
                "",
            ]
        except Exception as e:
            lines += [f"⚠ 加载股票池失败：{e}", ""]
    else:
        lines += [
            "## 时点股票池",
            "",
            "⚠ 未找到 `universe_latest.parquet`，请先运行 `fetch_daily_data.py`。",
            "",
        ]

    # ------------------------------------------------------------------
    # 2. Parquet 文件统计
    # ------------------------------------------------------------------
    parquet_files = sorted(daily_dir.glob("*.parquet"))
    n_files = len(parquet_files)
    total_size_mb = sum(p.stat().st_size for p in parquet_files) / 1024 / 1024

    lines += [
        "## Parquet 文件统计",
        "",
        f"- 文件数：**{n_files}**",
        f"- 磁盘占用：{total_size_mb:.1f} MB",
        "",
    ]

    if n_files == 0:
        lines += ["⚠ 未找到任何 Parquet 文件，请先运行 `fetch_daily_data.py`。", ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 3. 日期覆盖范围（抽样读取，全量太慢）
    # ------------------------------------------------------------------
    sample_size = min(200, n_files)
    import random
    sample_files = random.sample(parquet_files, sample_size)

    all_min_dates: list[pd.Timestamp] = []
    all_max_dates: list[pd.Timestamp] = []
    row_counts: list[int] = []

    for p in sample_files:
        try:
            df = pd.read_parquet(p, columns=["trade_date"])
            if len(df) == 0:
                continue
            row_counts.append(len(df))
            dates = pd.to_datetime(df["trade_date"])
            all_min_dates.append(dates.min())
            all_max_dates.append(dates.max())
        except Exception:
            pass

    if all_min_dates:
        global_min = min(all_min_dates).strftime("%Y-%m-%d")
        global_max = max(all_max_dates).strftime("%Y-%m-%d")
        avg_rows   = int(sum(row_counts) / len(row_counts)) if row_counts else 0
        median_rows = sorted(row_counts)[len(row_counts) // 2] if row_counts else 0

        lines += [
            "## 日期覆盖范围（抽样估算）",
            "",
            f"> 抽样 {sample_size} 只股票，结果为估算值。",
            "",
            f"- 最早交易日：{global_min}",
            f"- 最新交易日：{global_max}",
            f"- 单股平均行数：{avg_rows}",
            f"- 单股中位行数：{median_rows}",
            "",
        ]

    # ------------------------------------------------------------------
    # 4. 退市股覆盖情况
    # ------------------------------------------------------------------
    if pit is not None:
        delisted_df = pit._df[pit._df["delist_date"].notna()]
        n_delisted_total = len(delisted_df)
        fetched_codes = {p.stem for p in parquet_files}
        delisted_fetched = delisted_df[delisted_df["ts_code"].isin(fetched_codes)]
        n_delisted_fetched = len(delisted_fetched)
        coverage_pct = n_delisted_fetched / n_delisted_total * 100 if n_delisted_total > 0 else 0

        lines += [
            "## 退市股历史数据覆盖",
            "",
            f"- 退市股总数：{n_delisted_total}",
            f"- 已拉取（有 Parquet 文件）：{n_delisted_fetched}",
            f"- 覆盖率：{coverage_pct:.1f}%",
            "",
        ]

    # ------------------------------------------------------------------
    # 5. 缺口清单（有上市记录但无 Parquet 的股票）
    # ------------------------------------------------------------------
    if pit is not None:
        all_codes = set(pit._df["ts_code"].tolist())
        fetched_codes = {p.stem for p in parquet_files}
        missing_codes = sorted(all_codes - fetched_codes)

        lines += [
            "## 数据缺口",
            "",
        ]
        if missing_codes:
            lines += [
                f"以下 **{len(missing_codes)}** 只股票有上市记录但无 Parquet 文件：",
                "",
            ]
            # 仅显示前 50 条，避免报告过长
            for code in missing_codes[:50]:
                row = pit._df[pit._df["ts_code"] == code].iloc[0]
                name = row.get("name", "")
                delist = row.get("delist_date", "")
                delist_str = f"，退市：{delist}" if pd.notna(delist) and delist else ""
                lines.append(f"- `{code}` {name}{delist_str}")
            if len(missing_codes) > 50:
                lines.append(f"- …（另有 {len(missing_codes)-50} 只，见完整 CSV）")
                # 写出完整 CSV
                missing_df = pit._df[pit._df["ts_code"].isin(missing_codes)]
                csv_path = data_root / "missing_stocks.csv"
                missing_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                lines.append(f"  完整缺口清单：`{csv_path}`")
        else:
            lines.append("✅ 无缺口：所有上市记录均已拉取。")
        lines.append("")

    # ------------------------------------------------------------------
    # 6. 结论
    # ------------------------------------------------------------------
    lines += [
        "---",
        "",
        "## 结论与建议",
        "",
    ]

    issues: list[str] = []
    if pit is None:
        issues.append("股票池文件缺失，请先运行 `fetch_daily_data.py`。")
    if n_files == 0:
        issues.append("无任何 Parquet 日线文件。")

    if pit is not None and n_files > 0:
        coverage = n_files / pit.total_stocks * 100 if pit.total_stocks > 0 else 0
        if coverage < 95:
            issues.append(
                f"整体拉取覆盖率 {coverage:.1f}% < 95%，"
                "建议运行 `fetch_daily_data.py --retry-failed` 补全缺口。"
            )

    if issues:
        for issue in issues:
            lines.append(f"- ⚠ {issue}")
    else:
        lines.append(
            "数据层状态良好，可进入下一步（扶正 v2 特征管线，见 Issue #10）。"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="数据完整性报告（Issue #5）")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("STOCK_DATA_ROOT", str(REPO_ROOT / "stock_data")),
        help="数据根目录（默认：$STOCK_DATA_ROOT 或 ./stock_data）",
    )
    parser.add_argument(
        "--report-dir",
        default=str(REPO_ROOT / "reports"),
        help="报告输出目录（默认：./reports）",
    )
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在生成数据完整性报告（数据根目录：{data_root}）…")
    report_md = build_report(data_root)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = report_dir / f"integrity_report_{ts_str}.md"
    out_path.write_text(report_md, encoding="utf-8")

    # 同时更新 latest
    latest = report_dir / "integrity_report_latest.md"
    latest.write_text(report_md, encoding="utf-8")

    print(f"报告已写入：{out_path}")
    print(f"最新链接：{latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
