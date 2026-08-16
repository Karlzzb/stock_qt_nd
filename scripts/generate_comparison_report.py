#!/usr/bin/env python3
"""
新旧对照报告生成脚本（Issue #12 验收第 4 条）。

功能
----
读取 run_baseline_backtest.py 产生的 summary.json，与旧基线（comm_fun.py
注释中的历史数字）对比，输出 Markdown 格式的新旧对照报告。

报告包含：收益率、最大回撤、夏普比率、胜率、交易数、换手估算。

用法
----
    uv run python scripts/generate_comparison_report.py [选项]

选项
----
    --results-dir PATH     基线结果目录（默认：output/baseline_results）
    --output PATH          报告输出路径（默认：reports/honest_baseline_comparison.md）
    --title TEXT           报告标题（可选）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# 旧基线数字来源：comm_fun.py 参数注释中的历史回测结果（泄露前）
# v8 param35: 回报率:2.1148 | 最大回撤:-0.2924 | 胜率:0.5414 | 夏普:1.2266 | 交易数:133
# v12 param2: 回报率:4.0518 | 最大回撤:-0.2918 | 胜率:0.5827 | 夏普:1.9795 | 交易数:127
OLD_BASELINE: dict = {
    "v8": {
        "return_rate": 2.1148,
        "max_drawdown": -0.2924,
        "win_rate": 0.5414,
        "sharpe_ratio": 1.2266,
        "total_trades": 133,
        "param_key": "param35",
        "note": "旧基线（含泄露：全局统计 + close_wavelet 前视 + 背离锚点漂移）",
    },
    "v12": {
        "return_rate": 4.0518,
        "max_drawdown": -0.2918,
        "win_rate": 0.5827,
        "sharpe_ratio": 1.9795,
        "total_trades": 127,
        "param_key": "param2",
        "note": "旧基线（含泄露：全局统计 + close_wavelet 前视 + 背离锚点漂移）",
    },
}

_METRIC_LABELS = {
    "return_rate":   "总收益率",
    "annual_return": "年化收益率",
    "max_drawdown":  "最大回撤",
    "sharpe_ratio":  "夏普比率",
    "win_rate":      "胜率",
    "total_trades":  "总交易数",
}


def _fmt(val: object, key: str) -> str:
    """格式化数值为可读字符串。"""
    if val is None or val == "N/A":
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if key in ("return_rate", "annual_return", "max_drawdown", "win_rate"):
        return f"{f:.2%}"
    if key == "sharpe_ratio":
        return f"{f:.4f}"
    if key == "total_trades":
        return str(int(round(f)))
    return f"{f:.4f}"


def _delta_arrow(new_val: object, old_val: object, higher_is_better: bool = True) -> str:
    """返回变化方向箭头 + 百分点差。"""
    try:
        n = float(new_val)
        o = float(old_val)
    except (TypeError, ValueError):
        return ""
    diff = n - o
    if abs(diff) < 1e-9:
        return "→ 0"
    arrow = "↑" if (diff > 0) == higher_is_better else "↓"
    sign = "+" if diff > 0 else ""
    return f"{arrow} {sign}{diff:.4f}"


def build_version_section(
    version: str,
    new: dict,
    old: dict,
) -> list[str]:
    """构建单个策略版本的对照表 Markdown。"""
    lines: list[str] = []
    lines.append(f"\n### {version.upper()} — param{old['param_key'].lstrip('param')}")
    lines.append("")
    lines.append(f"> 旧基线说明：{old.get('note', '')}")
    lines.append("")
    lines.append("| 指标 | 旧基线（含泄露） | 诚实基线（修复后） | 变化 |")
    lines.append("|------|-----------------|------------------|------|")

    higher_better = {
        "return_rate": True,
        "annual_return": True,
        "max_drawdown": False,   # 越接近 0 越好
        "sharpe_ratio": True,
        "win_rate": True,
        "total_trades": None,    # 中性
    }
    for key, label in _METRIC_LABELS.items():
        old_val = old.get(key)
        new_val = new.get(key)
        hib = higher_better.get(key)
        delta = _delta_arrow(new_val, old_val, hib) if hib is not None else ""
        lines.append(
            f"| {label} | {_fmt(old_val, key)} | {_fmt(new_val, key)} | {delta} |"
        )

    if "error" in new:
        lines.append(f"\n> ⚠️ 诚实基线跑出错误：`{new['error']}`，以上新值均为 N/A。")

    return lines


def generate_report(results_dir: Path, output_path: Path, title: str) -> None:
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"{summary_file} 不存在，请先运行 scripts/run_baseline_backtest.py"
        )
    with open(summary_file, encoding="utf-8") as f:
        new_results: dict = json.load(f)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 结果来源：`{results_dir.resolve()}`")
    lines.append("")
    lines.append("## 背景")
    lines.append("")
    lines.append(
        "旧回测被四类泄露污染：幸存者偏差、全局统计（跨股票中心化）、"
        "未来函数（close_wavelet）、背离检测器锚点漂移。"
    )
    lines.append(
        "本报告呈现整改后（v2 特征管线 + walk-forward 重训）的诚实基线数字，"
        "作为后续所有新策略的真实对照基准。"
    )
    lines.append("")
    lines.append("## 对照表")

    for version in ("v8", "v12"):
        new = new_results.get(version, {})
        old = OLD_BASELINE.get(version, {})
        lines.extend(build_version_section(version, new, old))

    lines.append("")
    lines.append("## 解读指引")
    lines.append("")
    lines.append(
        "- 若诚实基线显著低于旧基线，说明旧系统高估了策略收益（泄露带来的虚假 alpha）。"
    )
    lines.append(
        "- 若诚实基线与旧基线相近，说明该策略捕捉了真实市场信号。"
    )
    lines.append(
        "- 后续所有新特征实验（Issue #13–#15）必须以本报告的诚实基线为对照，"
        "在 EvaluationLedger 登记后方可进入终审段。"
    )
    lines.append("")
    lines.append("---")
    lines.append(f"*由 `scripts/generate_comparison_report.py` 自动生成。*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成新旧对照报告")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径（默认：reports/honest_baseline_comparison.md）",
    )
    parser.add_argument(
        "--title",
        default="诚实基线：泄露修复前后对照报告",
    )
    args = parser.parse_args()

    from config.settings import RESULT_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULT_DIR / "baseline_results"
    output_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "reports" / "honest_baseline_comparison.md"
    )
    generate_report(results_dir, output_path, args.title)


if __name__ == "__main__":
    main()
