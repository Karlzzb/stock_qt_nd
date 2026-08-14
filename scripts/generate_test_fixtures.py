#!/usr/bin/env python3
"""
scripts/generate_test_fixtures.py
从真实行情数据抽取小型固定样本，写入 tests/fixtures/daily/。

只需运行一次，结果提交到 git，可离线使用。
如需重新生成，删除目标目录后重新运行本脚本。

用法：
    cd /home/karl/repos/personal/stock_qt_nd
    .venv/bin/python scripts/generate_test_fixtures.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DAILY_SRC = REPO_ROOT / "stock_data" / "daily"
FIXTURES_DAILY = REPO_ROOT / "tests" / "fixtures" / "daily"
FIXTURES_DAILY.mkdir(parents=True, exist_ok=True)


def _extract_slice(
    ts_code: str,
    start: str | None = None,
    end: str | None = None,
    max_rows: int = 260,
) -> pd.DataFrame:
    """
    从 stock_data/daily/{ts_code}.parquet 抽取日线切片。

    返回 DataFrame：DatetimeIndex（名称 trade_date），
    列：open, high, low, close, volume（float64）。
    """
    src = DAILY_SRC / f"{ts_code}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"数据文件不存在: {src}")

    df = pd.read_parquet(src)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    df = df.rename(columns={"vol": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].astype(np.float64)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) > max_rows:
        df = df.tail(max_rows)

    return df


def make_live_stock_fixture() -> None:
    """
    000001.SZ（平安银行）：长期在市股，取 2021-01-04 ~ 2021-12-31（约 243 个交易日）。
    用途：特征管线批量入口回归测试的主 fixture。
    """
    df = _extract_slice("000001.SZ", start="2021-01-04", end="2021-12-31")
    out = FIXTURES_DAILY / "000001.SZ.parquet"
    df.to_parquet(out)
    print(
        f"✓ {out.name}: {len(df)} rows  "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )


def make_delisted_stock_fixture() -> None:
    """
    000004.SZ（国华退）：退市股（退市日 2026-07-14），取最后 260 个交易日。
    用途：验证 fixture 能覆盖退市场景，供后续管线与背离测试扩展使用。
    """
    df = _extract_slice("000004.SZ", start=None, end=None, max_rows=260)
    out = FIXTURES_DAILY / "000004.SZ.parquet"
    df.to_parquet(out)
    print(
        f"✓ {out.name}: {len(df)} rows  "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )


def make_divergence_trigger_fixture() -> None:
    """
    合成行情（300 个交易日）：在 idx=165（2020-08-20）确定性地触发
    「价格创新低 + MACD 不创新低」的底背离结构，供背离检测器回归测试使用。

    设计依据（seed=777）：
    - idx=110：第一个局部低点（长期缓慢下跌后触底），MACD 深度负值
    - idx=112-150：强力反弹，MACD 向上修复
    - idx=165：第二个局部低点，价格低于 idx=110，但 MACD 高于 idx=110（底背离成立）
    """
    import talib

    n = 300
    rng = np.random.default_rng(777)
    price = np.ones(n) * 10.0

    # ── 阶段 1（0-50）：平稳震荡，初始化 EMA ─────────────────────
    for i in range(1, 50):
        price[i] = price[i - 1] * (1 + rng.standard_normal() * 0.003)

    # ── 阶段 2（50-110）：长期缓慢下跌 → 第一个谷（idx 110）────────
    for i in range(50, 110):
        price[i] = price[i - 1] * (1 - 0.004 + rng.standard_normal() * 0.003)
    price[110] = price[109] * 0.97   # 尖锐触底
    price[111] = price[110] * 1.02   # 立即回升（确认波谷）
    price[112] = price[111] * 1.02

    # ── 阶段 3（112-150）：强力反弹，让 MACD 充分修复 ─────────────
    for i in range(112, 151):
        price[i] = price[i - 1] * (1 + 0.005 + rng.standard_normal() * 0.003)

    # ── 阶段 4（151-165）：短促下跌 → 第二个谷（idx 165）────────────
    for i in range(151, 165):
        price[i] = price[i - 1] * (1 - 0.006 + rng.standard_normal() * 0.003)
    price[165] = price[164] * 0.975
    if price[165] >= price[110]:     # 保证价格新低
        price[165] = price[110] * 0.97
    price[166] = price[165] * 1.02   # 立即回升（确认波谷）
    price[167] = price[166] * 1.02

    # ── 阶段 5（168-n）：修复走势 ─────────────────────────────────
    for i in range(168, n):
        price[i] = price[i - 1] * (1 + 0.002 + rng.standard_normal() * 0.005)

    volume = np.abs(rng.standard_normal(n) * 500_000 + 1_000_000)
    rng2 = np.random.default_rng(778)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")

    rng_hi = np.abs(rng2.standard_normal(n)) * 0.006
    rng_lo = np.abs(rng2.standard_normal(n)) * 0.006
    df = pd.DataFrame(
        {
            "open":   price * (1 + rng2.standard_normal(n) * 0.003),
            "high":   price * (1 + rng_hi),
            "low":    price * (1 - rng_lo),
            "close":  price,
            "volume": volume,
        },
        index=dates,
        dtype=np.float64,
    )
    df["low"] = df[["low", "close"]].min(axis=1)
    df["high"] = df[["high", "close"]].max(axis=1)
    df.index.name = "trade_date"

    # 验证背离条件
    macd_vals, _, _ = talib.MACD(df["close"].values.astype(np.float64))
    assert price[165] < price[110], (
        f"价格条件不满足：price[165]={price[165]:.4f} >= price[110]={price[110]:.4f}"
    )
    assert macd_vals[165] > macd_vals[110], (
        f"MACD 条件不满足：macd[165]={macd_vals[165]:.6f} <= macd[110]={macd_vals[110]:.6f}"
    )
    divergence_date = dates[165].date()
    print(
        f"  背离验证通过  "
        f"price[110]={price[110]:.4f}  price[165]={price[165]:.4f}  "
        f"macd[110]={macd_vals[110]:.6f}  macd[165]={macd_vals[165]:.6f}"
    )

    out = FIXTURES_DAILY / "div_trigger.parquet"
    df.to_parquet(out)
    print(f"✓ {out.name}: {len(df)} rows  divergence at {divergence_date}")


if __name__ == "__main__":
    print("生成 tests/fixtures/daily/ ...")
    make_live_stock_fixture()
    make_delisted_stock_fixture()
    make_divergence_trigger_fixture()
    print("\n所有 fixture 已写入 tests/fixtures/daily/")
