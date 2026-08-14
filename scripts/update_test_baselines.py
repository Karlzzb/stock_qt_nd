#!/usr/bin/env python3
"""
scripts/update_test_baselines.py
重新计算并覆写回归测试基准文件（tests/fixtures/baselines/）。

何时运行：
  - 每次修改特征逻辑或背离检测逻辑导致预期输出合理变化后，
    需明确运行本脚本以确认变更，而不是放宽测试断言。
  - 运行后请在 commit message 中说明变更原因。

用法：
    cd /home/karl/repos/personal/stock_qt_nd
    .venv/bin/python scripts/update_test_baselines.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import talib

REPO_ROOT = Path(__file__).parent.parent
# scripts/ 从项目根执行时 pythonpath = ["."] 尚未生效；手动添加 src/ 以解决
# divergence_detector_v2 使用 `from comm_fun import ...` 的裸导入
for _p in [str(REPO_ROOT), str(REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DAILY = REPO_ROOT / "tests" / "fixtures" / "daily"
BASELINES_DIR = REPO_ROOT / "tests" / "fixtures" / "baselines"
BASELINES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 辅助：加载 fixture 为 DatetimeIndex + OHLCV DataFrame
# ---------------------------------------------------------------------------

def load_fixture(name: str) -> pd.DataFrame:
    path = FIXTURES_DAILY / name
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# 辅助：特征管线批量入口（纯内存，不依赖文件 I/O）
# ---------------------------------------------------------------------------

def _build_long_df(stock_dict: dict) -> pd.DataFrame:
    """将 {symbol: OHLCV_DataFrame} 转换为多股票长格式 DataFrame。"""
    frames = []
    for symbol, df in stock_dict.items():
        tmp = df.reset_index().copy()
        date_col = tmp.columns[0]
        tmp.rename(columns={date_col: "timestamp"}, inplace=True)
        tmp["symbol"] = symbol
        # 确保 volume 列存在（fixture 列名已是 volume）
        frames.append(tmp)
    big = pd.concat(frames, ignore_index=True)
    big["timestamp"] = pd.to_datetime(big["timestamp"])
    return big.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def compute_features(stock_dict: dict) -> pd.DataFrame:
    """运行完整特征计算流水线，返回带所有特征的 DataFrame。"""
    from src.feature_pipeline_v2 import FeaturePipeline
    from src.divergence_detector_v2 import DivergenceDetectorV2

    pipeline = FeaturePipeline(
        divergence_detector=DivergenceDetectorV2(),
        full_stocks_inner={},
    )

    big = _build_long_df(stock_dict)
    big = pipeline._calculate_basic_technical_features(big)
    if big is None:
        raise RuntimeError("_calculate_basic_technical_features 返回 None，请检查数据量是否 >= 100 行")
    big = pipeline._calculate_advance_technical_features(big)
    big = pipeline._generate_alpha_features(big)
    if big is None:
        raise RuntimeError("_generate_alpha_features 返回 None")
    big = pipeline.generate_structure_features(big)
    if big is None:
        raise RuntimeError("generate_structure_features 返回 None")
    big = pipeline.generate_lag_features(big)
    if big is None:
        raise RuntimeError("generate_lag_features 返回 None")
    return big


# ---------------------------------------------------------------------------
# 生成特征管线基准
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    """将 numpy scalar / NaN 转为 Python float 或 None。"""
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 8)
    except (TypeError, ValueError):
        return None


def generate_feature_pipeline_baseline() -> None:
    print("── 生成特征管线基准 ──")
    df_live = load_fixture("000001.SZ.parquet")
    result = compute_features({"000001.SZ": df_live})

    # 取最后一行作为断言目标
    last_row = result[result["symbol"] == "000001.SZ"].iloc[-1]
    target_date = str(last_row["timestamp"].date())

    # 选取覆盖各管线阶段的代表性特征
    features_to_capture = [
        # ── talib 基础技术指标（_calculate_basic_technical_features）──
        "rsi_14",
        "macd",
        "ma_20",
        "obv",
        "atr",
        # ── 进阶技术特征（_calculate_advance_technical_features）──
        "volume_ratio",
        "price_position",
        # ── alpha 特征（_generate_alpha_features）──
        "pct_change",
        "clv",
        "vol_divergence",
        # ── 结构特征（generate_structure_features）──
        "vol_gk",
        "illiq",
        "ret_overnight",
        "ret_intraday",
        # ── lag 特征（generate_lag_features）──
        "close_lag_5",
        "volume_lag_10",
        "return_lag_1",
    ]

    assertions = []
    for feat in features_to_capture:
        if feat not in last_row.index:
            print(f"  ⚠ 特征 {feat!r} 不存在，跳过")
            continue
        val = _safe_float(last_row[feat])
        if val is None:
            print(f"  ⚠ 特征 {feat!r} 值为 NaN，跳过")
            continue
        assertions.append({"feature": feat, "value": val})
        print(f"  {feat}: {val}")

    baseline = {
        "schema_version": "1",
        "description": (
            "特征管线确定性回归基准。"
            "泄露修复导致的预期数值变化须通过显式运行 "
            "scripts/update_test_baselines.py 更新，不得放宽断言。"
        ),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "fixture": "tests/fixtures/daily/000001.SZ.parquet",
        "symbol": "000001.SZ",
        "target_date": target_date,
        "abs_tol": 1e-4,
        "assertions": assertions,
    }

    out = BASELINES_DIR / "feature_pipeline.json"
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    print(f"✓ 已写入 {out}  ({len(assertions)} 个断言)")


# ---------------------------------------------------------------------------
# 生成背离检测器基准
# ---------------------------------------------------------------------------

def generate_divergence_detector_baseline() -> None:
    print("── 生成背离检测器基准 ──")
    from src.divergence_detector_v2 import DivergenceDetectorV2

    df = load_fixture("div_trigger.parquet")

    # 计算 MACD（检测器的入参 data_with_indicators 需要 macd 列）
    close = df["close"].values.astype(np.float64)
    macd_vals, _, _ = talib.MACD(close)
    df = df.copy()
    df["macd"] = macd_vals
    # 检测器还会访问 volume
    df.index.name = "trade_date"

    # 目标日期：idx=165（div_trigger.parquet 中已验证的背离日 2020-08-20）
    target_date = df.index[165].date()
    detector = DivergenceDetectorV2()
    result = detector.detect_daily_divergence(df, "SYNTH", target_date)

    if result.empty:
        raise RuntimeError(
            f"在 {target_date} 未检测到任何背离信号，请检查 fixture 数据或检测逻辑"
        )

    print(f"  在 {target_date} 检测到 {len(result)} 个背离点")

    # 取第一个背离点
    first = result.iloc[0]
    first_div = {
        "close_current":    _safe_float(first.get("close_current")),
        "close_previous":   _safe_float(first.get("close_previous")),
        "macd_current":     _safe_float(first.get("macd_current")),
        "macd_previous":    _safe_float(first.get("macd_previous")),
        "price_decline_pct": _safe_float(first.get("price_decline_pct")),
        "macd_increase_pct": _safe_float(first.get("macd_increase_pct")),
        "divergence_strength": _safe_float(first.get("divergence_strength")),
    }
    for k, v in first_div.items():
        print(f"  {k}: {v}")

    baseline = {
        "schema_version": "1",
        "description": (
            "背离检测器确定性回归基准。"
            "锚点漂移修复导致的预期变化须通过显式运行 "
            "scripts/update_test_baselines.py 更新。"
        ),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "fixture": "tests/fixtures/daily/div_trigger.parquet",
        "symbol": "SYNTH",
        "target_date": str(target_date),
        "abs_tol": 1e-5,
        "expected_count_min": 1,
        "first_divergence": first_div,
    }

    out = BASELINES_DIR / "divergence_detector.json"
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    print(f"✓ 已写入 {out}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_feature_pipeline_baseline()
    print()
    generate_divergence_detector_baseline()
    print("\n所有基准文件已更新。请将修改纳入 commit 并说明原因。")
