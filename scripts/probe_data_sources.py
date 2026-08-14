#!/usr/bin/env python3
"""
数据源连通性探测脚本（issue #3）。

在大规模拉数前逐接口确认 token 权限与退市股历史数据可得性。
此脚本是 go/no-go 门：任何接口不可用都应上报用户，不得跳过进入数据层重建。

用法：
    export TINYSHARE_TOKEN="<your_token>"
    uv run python scripts/probe_data_sources.py [--report-dir ./reports]

脚本不进 CI 默认路径，需要真实 token 与网络，手动执行。
"""

import argparse
import os
import sys
import textwrap
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 将 src/ 加入路径，复用 tinyshare_auth
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from tinyshare_auth import get_pro_api
except ImportError as e:
    print(f"[FATAL] 无法导入 tinyshare_auth：{e}")
    print("请确保在仓库根目录下执行，或已执行 uv sync。")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 探测结果数据类
# ---------------------------------------------------------------------------

class ProbeResult:
    def __init__(self, interface: str, status: str, shape: tuple | None,
                 sample: str, note: str):
        self.interface = interface
        self.status = status          # "OK" | "FAIL" | "EMPTY"
        self.shape = shape
        self.sample = sample          # 首行预览或错误摘要
        self.note = note

    def ok(self) -> bool:
        return self.status == "OK"


# ---------------------------------------------------------------------------
# 抽样用的退市股票列表（手工选取若干知名退市股，探测时从 D 列表动态补充）
# ---------------------------------------------------------------------------

# 这几只是历史上曾退市或被吸收合并的代表性股票，用于兜底抽样
_KNOWN_DELISTED_FALLBACK = [
    "600050.SH",  # 中国联通（老代码，已更名/退市）
    "000001.SZ",  # 实际在市，但日线接口参数测试用
]


# ---------------------------------------------------------------------------
# 核心探测函数
# ---------------------------------------------------------------------------

def _probe(label: str, fn, *args, **kwargs) -> ProbeResult:
    """
    运行单次接口探测，捕获所有异常。
    fn 应返回 pd.DataFrame；探测成功当且仅当返回非空 DataFrame。
    """
    try:
        import pandas as pd

        df = fn(*args, **kwargs)

        if df is None or (hasattr(df, "__len__") and len(df) == 0):
            return ProbeResult(
                interface=label,
                status="EMPTY",
                shape=(0, 0) if df is not None else None,
                sample="（返回空 DataFrame）",
                note="接口可达但无数据；可能是参数范围问题或权限限制。",
            )

        shape = df.shape if hasattr(df, "shape") else (len(df),)
        # 取首行作为样本预览
        if hasattr(df, "iloc"):
            row = df.iloc[0]
            cols = list(df.columns[:6])
            sample = "  ".join(f"{c}={row[c]}" for c in cols if c in row.index)
        else:
            sample = str(df)[:120]

        return ProbeResult(
            interface=label,
            status="OK",
            shape=shape,
            sample=sample,
            note="",
        )

    except Exception as exc:
        tb_last = traceback.format_exc().strip().splitlines()[-1]
        return ProbeResult(
            interface=label,
            status="FAIL",
            shape=None,
            sample=tb_last,
            note=str(exc),
        )


def probe_all(pro) -> tuple[list[ProbeResult], list[str]]:
    """
    按顺序探测所有目标接口，返回结果列表和动态发现的退市股样本列表。
    """
    results: list[ProbeResult] = []
    delisted_samples: list[str] = []

    today = date.today()
    # 取近 5 个交易日的结束日期作为 daily 探测区间
    end_dt = today.strftime("%Y%m%d")
    start_dt = (today - timedelta(days=10)).strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # 1. stock_basic — 在市股票列表
    # ------------------------------------------------------------------
    print("  → stock_basic(list_status=L) …", flush=True)
    r = _probe(
        "stock_basic(list_status=L)",
        lambda: pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,list_date,industry",
        ),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 2. stock_basic — 退市股票列表
    # ------------------------------------------------------------------
    print("  → stock_basic(list_status=D) …", flush=True)
    r_d = _probe(
        "stock_basic(list_status=D)",
        lambda: pro.stock_basic(
            exchange="",
            list_status="D",
            fields="ts_code,name,list_date,delist_date",
        ),
    )
    results.append(r_d)

    # 从退市列表中抽几只股票用于后续日线验证
    if r_d.ok() and r_d.shape and r_d.shape[0] > 0:
        try:
            import pandas as pd
            df_d = pro.stock_basic(
                exchange="",
                list_status="D",
                fields="ts_code,name,list_date,delist_date",
            )
            # 选取有明确退市日期、且历史数据较充足的样本（退市超 3 年）
            if "delist_date" in df_d.columns:
                df_d["delist_date"] = pd.to_datetime(df_d["delist_date"], errors="coerce")
                cutoff = pd.Timestamp("2020-01-01")
                eligible = df_d[df_d["delist_date"] < cutoff]["ts_code"].dropna().tolist()
                delisted_samples = eligible[:3] if eligible else []
        except Exception:
            pass

    if not delisted_samples:
        delisted_samples = _KNOWN_DELISTED_FALLBACK[:2]

    # ------------------------------------------------------------------
    # 3. daily — 日线行情（在市股票，近期）
    # ------------------------------------------------------------------
    print("  → daily (近期日线) …", flush=True)
    # 取第一只在市股票的近期日线
    live_code = "000001.SZ"
    r = _probe(
        f"daily({live_code}, {start_dt}~{end_dt})",
        lambda: pro.daily(ts_code=live_code, start_date=start_dt, end_date=end_dt),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 4. daily_basic — 日频基本面
    # ------------------------------------------------------------------
    print("  → daily_basic …", flush=True)
    r = _probe(
        f"daily_basic({live_code}, {start_dt}~{end_dt})",
        lambda: pro.daily_basic(
            ts_code=live_code,
            start_date=start_dt,
            end_date=end_dt,
            fields="ts_code,trade_date,pe,pb,total_mv,turnover_rate",
        ),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 5. moneyflow — 个股资金流向
    # ------------------------------------------------------------------
    print("  → moneyflow …", flush=True)
    r = _probe(
        f"moneyflow({live_code}, {start_dt}~{end_dt})",
        lambda: pro.moneyflow(ts_code=live_code, start_date=start_dt, end_date=end_dt),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 6. moneyflow_hsgt — 北向资金
    # ------------------------------------------------------------------
    print("  → moneyflow_hsgt …", flush=True)
    r = _probe(
        f"moneyflow_hsgt({start_dt}~{end_dt})",
        lambda: pro.moneyflow_hsgt(start_date=start_dt, end_date=end_dt),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 7. stk_factor — 技术因子
    # ------------------------------------------------------------------
    print("  → stk_factor …", flush=True)
    r = _probe(
        f"stk_factor({live_code}, {start_dt}~{end_dt})",
        lambda: pro.stk_factor(ts_code=live_code, start_date=start_dt, end_date=end_dt),
    )
    results.append(r)

    # ------------------------------------------------------------------
    # 8. 退市股历史日线可得性（抽样验证）
    # ------------------------------------------------------------------
    for code in delisted_samples:
        print(f"  → daily(退市股 {code}, 历史日线抽样) …", flush=True)
        # 抽取该股退市前 1 年的日线数据
        try:
            import pandas as pd
            df_d2 = pro.stock_basic(
                ts_code=code,
                fields="ts_code,list_date,delist_date",
                list_status="D",
            )
            if len(df_d2) and "delist_date" in df_d2.columns:
                delist = pd.to_datetime(df_d2.iloc[0]["delist_date"])
                hist_end = (delist - timedelta(days=1)).strftime("%Y%m%d")
                hist_start = (delist - timedelta(days=365)).strftime("%Y%m%d")
            else:
                hist_end = "20191231"
                hist_start = "20190101"
        except Exception:
            hist_end = "20191231"
            hist_start = "20190101"

        label = f"daily(退市股 {code}, {hist_start}~{hist_end})"
        r = _probe(
            label,
            lambda c=code, s=hist_start, e=hist_end: pro.daily(
                ts_code=c, start_date=s, end_date=e
            ),
        )
        results.append(r)

    return results, delisted_samples


# ---------------------------------------------------------------------------
# Markdown 报告生成
# ---------------------------------------------------------------------------

STATUS_EMOJI = {"OK": "✅", "FAIL": "❌", "EMPTY": "⚠️"}


def build_report(results: list[ProbeResult], delisted_samples: list[str],
                 token_masked: str, report_time: str) -> str:
    """生成 Markdown 格式的权限探测报告。"""

    ok_count = sum(1 for r in results if r.ok())
    fail_count = sum(1 for r in results if r.status == "FAIL")
    empty_count = sum(1 for r in results if r.status == "EMPTY")

    go_nogo = "**🟢 GO**" if fail_count == 0 else "**🔴 NO-GO**"
    if empty_count > 0 and fail_count == 0:
        go_nogo = "**🟡 PARTIAL**（有接口返回空，请核查权限或参数）"

    lines = [
        "# tinyshare 数据源权限探测报告",
        "",
        f"- 探测时间：{report_time}",
        f"- Token（末 6 位）：`…{token_masked}`",
        f"- 汇总：✅ {ok_count} 可用 / ❌ {fail_count} 不可用 / ⚠️ {empty_count} 返回空",
        f"- go/no-go 判定：{go_nogo}",
        "",
        "---",
        "",
        "## 接口探测明细",
        "",
        "| 状态 | 接口 | 行×列 | 首行样本 / 错误 |",
        "| --- | --- | --- | --- |",
    ]

    for r in results:
        emoji = STATUS_EMOJI.get(r.status, "?")
        shape_str = f"{r.shape[0]}×{r.shape[1]}" if r.shape and len(r.shape) == 2 else (str(r.shape) if r.shape else "—")
        sample_escaped = r.sample.replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(f"| {emoji} {r.status} | `{r.interface}` | {shape_str} | {sample_escaped} |")

    lines += [
        "",
        "---",
        "",
        "## 退市股历史日线可得性",
        "",
    ]

    delisted_results = [r for r in results if "退市股" in r.interface]
    if delisted_results:
        lines.append(f"抽样股票：{', '.join(f'`{c}`' for c in delisted_samples)}")
        lines.append("")
        for r in delisted_results:
            emoji = STATUS_EMOJI.get(r.status, "?")
            shape_str = f"{r.shape[0]} 行" if r.shape else "—"
            lines.append(f"- {emoji} `{r.interface}`：{shape_str}")
            if not r.ok():
                lines.append(f"  - 错误：{r.note}")
    else:
        lines.append("（未获得退市股样本，无法验证）")

    lines += [
        "",
        "---",
        "",
        "## 结论与建议",
        "",
    ]

    if fail_count == 0 and empty_count == 0:
        lines.append("所有接口均可用，含退市股历史日线。可进入数据层重建（issue #5）。")
    else:
        if fail_count > 0:
            failed_names = [r.interface for r in results if r.status == "FAIL"]
            lines.append(f"以下接口不可用，**必须上报用户，不得进入数据层重建**：")
            lines.append("")
            for name in failed_names:
                lines.append(f"- `{name}`")
            lines.append("")
        if empty_count > 0:
            empty_names = [r.interface for r in results if r.status == "EMPTY"]
            lines.append("以下接口返回空数据，可能是参数范围问题或权限限制，请人工核查：")
            lines.append("")
            for name in empty_names:
                lines.append(f"- `{name}`")
            lines.append("")
        lines.append("请联系 tinyshare 支持或检查 token 积分/权限等级后重跑本脚本。")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="tinyshare 数据源连通性探测（issue #3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            示例：
              uv run python scripts/probe_data_sources.py
              uv run python scripts/probe_data_sources.py --report-dir ./reports
            """
        ),
    )
    parser.add_argument(
        "--report-dir",
        default="./reports",
        help="Markdown 报告输出目录（默认：./reports）",
    )
    args = parser.parse_args()

    # 检查 token
    token = os.environ.get("TINYSHARE_TOKEN", "").strip()
    if not token:
        print("[ERROR] 环境变量 TINYSHARE_TOKEN 未设置。")
        print("  请执行：export TINYSHARE_TOKEN='<your_token>'")
        return 1

    token_masked = token[-6:] if len(token) >= 6 else "***"

    print("=" * 60)
    print("tinyshare 数据源连通性探测")
    print(f"Token 末 6 位：…{token_masked}")
    print("=" * 60)

    # 初始化 API
    try:
        pro = get_pro_api()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        return 1

    print("\n开始逐接口探测…\n")
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results, delisted_samples = probe_all(pro)

    # 控制台摘要
    print("\n" + "=" * 60)
    print("探测结果摘要")
    print("=" * 60)
    for r in results:
        emoji = STATUS_EMOJI.get(r.status, "?")
        shape_str = (f"{r.shape[0]}×{r.shape[1]}" if r.shape and len(r.shape) == 2
                     else (str(r.shape) if r.shape else "—"))
        print(f"  {emoji} [{r.status:5s}] {r.interface}  shape={shape_str}")
        if not r.ok():
            print(f"          {r.sample[:100]}")

    # 生成报告
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"data_source_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    report_md = build_report(results, delisted_samples, token_masked, report_time)
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n报告已写入：{report_path}")

    fail_count = sum(1 for r in results if r.status == "FAIL")
    if fail_count:
        print(f"\n[WARN] {fail_count} 个接口不可用，请上报用户后再决定是否继续。")
        return 2
    print("\n[OK] 全部接口可达，可进入数据层重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
