#!/usr/bin/env python3
"""拉取大盘指数数据（000001.SH 上证 / 399001.SZ 深证）"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import ttshare as ts

ts.set_token('QC54t85qlJHyt5n4qC2h3Bkzf68Qem6Ri7DEgJbXbv00F0kUtYr2fAvQ6345f070')

DATA_ROOT = Path(REPO_ROOT / "stock_data")
DAILY_DIR = DATA_ROOT / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# 指数代码
INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
}

def fetch_index(pro, ts_code: str, name: str):
    print(f"拉取 {name} ({ts_code})...", flush=True)

    try:
        # tinyshare index_daily 接口
        df = pro.index_daily(ts_code=ts_code, start_date="19900101")

        if df is None or df.empty:
            print(f"  ✗ {name} 无数据")
            return

        # 重命名列对齐个股格式
        df = df.rename(columns={"vol": "volume"})
        df = df[["trade_date", "open", "high", "low", "close", "volume"]]

        # 保存
        output_path = DAILY_DIR / f"{ts_code}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"  ✓ {name} 完成：{len(df)} 行 → {output_path}")

    except Exception as e:
        print(f"  ✗ {name} 失败：{e}")

def main():
    pro = ts.pro_api()

    for ts_code, name in INDICES.items():
        fetch_index(pro, ts_code, name)

    print("\n完成")

if __name__ == "__main__":
    main()
