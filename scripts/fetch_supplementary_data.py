#!/usr/bin/env python3
"""
特征主表【需补数据】（P2, 15 条）外部数据拉取脚本。

拉取内容（全部走 tinyshare/tushare 兼容接口，token 复用 src/tinyshare_auth.py）：

1. daily_basic  —— 换手率/流通股本/总市值族（P2 #1-5）
   按 trade_date 逐日拉全历史（1990-12-19 → 2026-08-31，对齐本地日线），
   每日一文件：stock_data/daily_basic/YYYYMMDD.parquet
2. stk_limit    —— 涨跌停价格精确版（P2 #14）
   按 trade_date 逐日拉（实测接口覆盖自 2007-01 起），
   每日一文件：stock_data/stk_limit/YYYYMMDD.parquet
3. index_daily  —— 宽基指数（P2 #13）：000300.SH / 000905.SH / 000852.SH / 399006.SZ
   全历史，每指数一文件：stock_data/index/{ts_code}.parquet
4. meta         —— namechange（历史 ST 状态，P2 #12，limit/offset 分页全量）
                  + stock_basic(L+D) 当前行业快照
   → stock_data/meta/namechange.parquet, stock_basic_industry.parquet
5. sw           —— 申万行业（P2 #7-11）：index_classify(SW2021, L1/L2/L3)
                  + 逐指数 index_member（含 in_date/out_date 历史成分）
   → stock_data/meta/sw_index_classify.parquet, sw_index_member.parquet
6. ths          —— 同花顺行业(N)+概念(I)板块及当前成分（ths_member 无历史）
   → stock_data/meta/ths_index.parquet, ths_member.parquet
7. concept      —— tinyshare TS 概念板块清单（实测无成分接口，仅板块列表，降级落盘）
   → stock_data/meta/ts_concept_list.parquet

实测不可用：个股级【成交笔数】（P2 #6 IDEAL_REV）——daily/daily_basic/stk_factor
均无该字段，daily_info 仅有交易所级 trans_count，记为不可补。

限流：实测接口级限流 120 次/分钟（超限返回 429）；脚本每次调用间隔 0.55s，
异常时指数退避重试 5 次。
断点续拉：按日文件已存在即跳过。

用法：
    uv run python scripts/fetch_supplementary_data.py                 # 全部
    uv run python scripts/fetch_supplementary_data.py --only daily_basic,stk_limit
    uv run python scripts/fetch_supplementary_data.py --validate-only # 只做完整性校验
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from tinyshare_auth import get_pro_api

DATA_ROOT = REPO_ROOT / "stock_data"
START_DATE = "19901219"  # 对齐本地日线起点（上交所首个交易日）
END_DATE = "20260831"    # 对齐本地日线终点
STK_LIMIT_START = "20070101"  # 实测 stk_limit 接口 2006 及以前返回空
CALL_INTERVAL = 0.25     # 秒，实测接口级限流 120 次/分钟（429）；间隔+延迟≈80/min 留余量
MAX_RETRY = 5

BROAD_INDEXES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399006.SZ": "创业板指",
}

_pro = None
_last_call = 0.0


def pro():
    global _pro
    if _pro is None:
        _pro = get_pro_api()
    return _pro


def api_call(api_name: str, **kw) -> pd.DataFrame:
    """带限速 + 指数退避重试的接口调用。"""
    global _last_call
    for attempt in range(MAX_RETRY):
        gap = time.time() - _last_call
        if gap < CALL_INTERVAL:
            time.sleep(CALL_INTERVAL - gap)
        try:
            df = pro().query(api_name, **kw)
            _last_call = time.time()
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            _last_call = time.time()
            wait = 2 ** (attempt + 1)
            print(f"    [retry {attempt+1}/{MAX_RETRY}] {api_name} {kw}: {str(e)[:100]} -> sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{api_name} {kw} 重试 {MAX_RETRY} 次仍失败")


def get_trade_dates(start: str, end: str) -> list[str]:
    """用上交所交易日历生成交易日清单（is_open=1）。

    实测 trade_cal 单次调用有行数上限（36 年只返回约 4000 行），
    按 4 年窗口分段拉取后合并去重。
    """
    y0, y1 = int(start[:4]), int(end[:4])
    frames = []
    for y in range(y0, y1 + 1, 4):
        seg_start = max(start, f"{y}0101")
        seg_end = min(end, f"{y + 3}1231")
        if seg_start > seg_end:
            continue
        frames.append(api_call("trade_cal", exchange="SSE", start_date=seg_start, end_date=seg_end))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["cal_date"])
    df = df[df["is_open"] == 1]
    dates = sorted(df["cal_date"].astype(str).tolist())
    return dates


# ---------------------------------------------------------------------------
# 1. daily_basic 逐日
# ---------------------------------------------------------------------------

def fetch_daily_basic() -> None:
    out_dir = DATA_ROOT / "daily_basic"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = get_trade_dates(START_DATE, END_DATE)
    todo = [d for d in dates if not (out_dir / f"{d}.parquet").exists()]
    print(f"[daily_basic] 交易日 {len(dates)} 天，待拉 {len(todo)} 天", flush=True)
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        df = api_call("daily_basic", trade_date=d)
        df.to_parquet(out_dir / f"{d}.parquet", index=False, engine="pyarrow", compression="snappy")
        if i % 200 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60 if rate > 0 else 0
            print(f"  {i}/{len(todo)}  {rate:.2f}/s  ETA {eta:.1f}min", flush=True)


# ---------------------------------------------------------------------------
# 2. stk_limit 逐日
# ---------------------------------------------------------------------------

def fetch_stk_limit() -> None:
    out_dir = DATA_ROOT / "stk_limit"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = get_trade_dates(STK_LIMIT_START, END_DATE)
    todo = [d for d in dates if not (out_dir / f"{d}.parquet").exists()]
    print(f"[stk_limit] 交易日 {len(dates)} 天（自 {STK_LIMIT_START}），待拉 {len(todo)} 天", flush=True)
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        df = api_call("stk_limit", trade_date=d)
        df.to_parquet(out_dir / f"{d}.parquet", index=False, engine="pyarrow", compression="snappy")
        if i % 200 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60 if rate > 0 else 0
            print(f"  {i}/{len(todo)}  {rate:.2f}/s  ETA {eta:.1f}min", flush=True)


# ---------------------------------------------------------------------------
# 3. 宽基指数
# ---------------------------------------------------------------------------

def fetch_index() -> None:
    out_dir = DATA_ROOT / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code, name in BROAD_INDEXES.items():
        df = api_call("index_daily", ts_code=code, start_date="19900101", end_date=END_DATE)
        df = df.sort_values("trade_date").reset_index(drop=True)
        df.to_parquet(out_dir / f"{code}.parquet", index=False, engine="pyarrow", compression="snappy")
        print(f"  {code} {name}: {len(df)} 行, {df['trade_date'].min()} ~ {df['trade_date'].max()}", flush=True)


# ---------------------------------------------------------------------------
# 4. meta: namechange + stock_basic 行业快照
# ---------------------------------------------------------------------------

def fetch_meta() -> None:
    out_dir = DATA_ROOT / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    # namechange 全量（单页上限 10000 行，需 limit/offset 分页）
    pages, offset = [], 0
    while True:
        df = api_call("namechange", limit=10000, offset=offset)
        if len(df) == 0:
            break
        pages.append(df)
        offset += len(df)
        if len(df) < 10000:
            break
    nc = pd.concat(pages, ignore_index=True).drop_duplicates().reset_index(drop=True)
    nc.to_parquet(out_dir / "namechange.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  namechange: {len(nc)} 行, {nc['ts_code'].nunique()} 只股票", flush=True)

    # stock_basic 当前行业快照（在市 + 退市；行业字段为当前口径，无历史）
    parts = []
    for status in ["L", "D"]:
        df = api_call(
            "stock_basic", exchange="", list_status=status,
            fields="ts_code,name,area,industry,market,exchange,list_status,list_date,delist_date",
        )
        parts.append(df)
    sb = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    sb.to_parquet(out_dir / "stock_basic_industry.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  stock_basic: {len(sb)} 只（含行业快照）", flush=True)


# ---------------------------------------------------------------------------
# 5. 申万行业（含历史成分）
# ---------------------------------------------------------------------------

def fetch_sw() -> None:
    out_dir = DATA_ROOT / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_pages = []
    for level in ["L1", "L2", "L3"]:
        df = api_call("index_classify", level=level, src="SW2021")
        cls_pages.append(df)
    cls = pd.concat(cls_pages, ignore_index=True).reset_index(drop=True)
    cls.to_parquet(out_dir / "sw_index_classify.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  sw_index_classify: {len(cls)} 个行业指数", flush=True)

    members = []
    codes = cls["index_code"].tolist()
    for i, ic in enumerate(codes, 1):
        df = api_call("index_member", index_code=ic)
        if len(df):
            members.append(df)
        if i % 100 == 0 or i == len(codes):
            print(f"  index_member {i}/{len(codes)}", flush=True)
    mem = pd.concat(members, ignore_index=True).drop_duplicates().reset_index(drop=True)
    mem.to_parquet(out_dir / "sw_index_member.parquet", index=False, engine="pyarrow", compression="snappy")
    n_hist = mem["out_date"].notna().sum()
    print(f"  sw_index_member: {len(mem)} 行, 历史调出记录 {n_hist} 条", flush=True)


# ---------------------------------------------------------------------------
# 6. 同花顺板块（行业 + 概念，当前成分快照）
# ---------------------------------------------------------------------------

def fetch_ths() -> None:
    out_dir = DATA_ROOT / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_parts = []
    for t in ["N", "I"]:  # N=行业, I=概念
        df = api_call("ths_index", exchange="A", type=t)
        idx_parts.append(df)
    ths_idx = pd.concat(idx_parts, ignore_index=True).reset_index(drop=True)
    ths_idx.to_parquet(out_dir / "ths_index.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  ths_index: {len(ths_idx)} 个板块", flush=True)

    members = []
    codes = ths_idx["ts_code"].tolist()
    for i, ic in enumerate(codes, 1):
        df = api_call("ths_member", ts_code=ic)
        if len(df):
            df = df.copy()
            df["board_code"] = ic
            members.append(df)
        if i % 200 == 0 or i == len(codes):
            print(f"  ths_member {i}/{len(codes)}", flush=True)
    mem = pd.concat(members, ignore_index=True).drop_duplicates().reset_index(drop=True)
    mem.to_parquet(out_dir / "ths_member.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  ths_member: {len(mem)} 行", flush=True)


# ---------------------------------------------------------------------------
# 7. TS 概念板块清单（无成分接口，降级）
# ---------------------------------------------------------------------------

def fetch_concept() -> None:
    out_dir = DATA_ROOT / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = api_call("concept", src="ts", trade_date=END_DATE)
    df.to_parquet(out_dir / "ts_concept_list.parquet", index=False, engine="pyarrow", compression="snappy")
    print(f"  ts_concept_list: {len(df)} 个板块（仅清单，无成分接口）", flush=True)


# ---------------------------------------------------------------------------
# 完整性校验
# ---------------------------------------------------------------------------

def validate() -> dict:
    stats: dict = {"validated_at": datetime.now().isoformat(timespec="seconds")}

    # 本地日线基准
    daily_dir = DATA_ROOT / "daily"
    local_files = sorted(daily_dir.glob("*.parquet"))
    stock_files = [p for p in local_files if p.stem not in ("000001.SH", "399001.SZ")]
    stats["local_daily"] = {"files": len(local_files), "stock_files": len(stock_files)}

    # daily_basic
    db_dir = DATA_ROOT / "daily_basic"
    db_files = sorted(db_dir.glob("*.parquet")) if db_dir.exists() else []
    if db_files:
        dates = [p.stem for p in db_files]
        sample = pd.concat([pd.read_parquet(p) for p in db_files[:: max(1, len(db_files) // 40)]])
        stats["daily_basic"] = {
            "files": len(db_files),
            "date_min": min(dates), "date_max": max(dates),
            "sample_rows": len(sample),
            "sample_stocks": int(sample["ts_code"].nunique()),
            "turnover_rate_missing": float(sample["turnover_rate"].isna().mean()),
            "turnover_rate_f_missing": float(sample["turnover_rate_f"].isna().mean()),
            "float_share_missing": float(sample["float_share"].isna().mean()),
            "total_mv_missing": float(sample["total_mv"].isna().mean()),
        }
        # 全量日期覆盖核对：与交易日历比对
        cal = set(get_trade_dates(START_DATE, END_DATE))
        have = set(dates)
        stats["daily_basic"]["missing_trade_dates"] = sorted(cal - have)

    # stk_limit
    sl_dir = DATA_ROOT / "stk_limit"
    sl_files = sorted(sl_dir.glob("*.parquet")) if sl_dir.exists() else []
    if sl_files:
        dates = [p.stem for p in sl_files]
        sample_frames = []
        for p in sl_files[:: max(1, len(sl_files) // 40)]:
            df = pd.read_parquet(p)
            sample_frames.append(df)
        sample = pd.concat(sample_frames) if sample_frames else pd.DataFrame()
        stats["stk_limit"] = {
            "files": len(sl_files),
            "date_min": min(dates), "date_max": max(dates),
            "sample_rows": len(sample),
            "up_limit_missing": float(sample["up_limit"].isna().mean()) if len(sample) else None,
        }
        cal = set(get_trade_dates(STK_LIMIT_START, END_DATE))
        stats["stk_limit"]["missing_trade_dates"] = sorted(cal - set(dates))

    # index
    idx_dir = DATA_ROOT / "index"
    idx_stats = {}
    for code in BROAD_INDEXES:
        p = idx_dir / f"{code}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            idx_stats[code] = {"rows": len(df), "min": str(df["trade_date"].min()), "max": str(df["trade_date"].max())}
    stats["index"] = idx_stats

    # meta
    meta_dir = DATA_ROOT / "meta"
    meta_stats = {}
    for name in ["namechange", "stock_basic_industry", "sw_index_classify", "sw_index_member",
                 "ths_index", "ths_member", "ts_concept_list"]:
        p = meta_dir / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            meta_stats[name] = {"rows": len(df), "cols": list(df.columns)}
            if name == "namechange":
                meta_stats[name]["stocks"] = int(df["ts_code"].nunique())
            if name == "sw_index_member":
                meta_stats[name]["with_out_date"] = int(df["out_date"].notna().sum())
    stats["meta"] = meta_stats

    # daily_basic 退市股覆盖校验：抽 50 只退市股，
    # 检查其本地最后一个交易日是否出现在 daily_basic 当日文件中
    if db_files:
        u = pd.read_parquet(DATA_ROOT / "universe" / "universe_latest.parquet")
        dead = u[u["delist_date"].notna()]
        dead_sample = dead.sample(min(50, len(dead)), random_state=0)
        hit, miss, miss_codes = 0, 0, []
        for _, row in dead_sample.iterrows():
            code = row["ts_code"]
            fp = daily_dir / f"{code}.parquet"
            if not fp.exists():
                continue
            loc = pd.read_parquet(fp, columns=["trade_date"])
            last = str(loc["trade_date"].max())[:10].replace("-", "")
            dbp = db_dir / f"{last}.parquet"
            if dbp.exists():
                db = pd.read_parquet(dbp, columns=["ts_code"])
                if code in set(db["ts_code"]):
                    hit += 1
                else:
                    miss += 1
                    miss_codes.append(code)
        stats["daily_basic"]["delisted_sample_lastday_hit"] = hit
        stats["daily_basic"]["delisted_sample_lastday_miss"] = miss
        stats["daily_basic"]["delisted_sample_miss_codes"] = miss_codes

    return stats


TASKS = {
    "daily_basic": fetch_daily_basic,
    "stk_limit": fetch_stk_limit,
    "index": fetch_index,
    "meta": fetch_meta,
    "sw": fetch_sw,
    "ths": fetch_ths,
    "concept": fetch_concept,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="P2 需补数据拉取")
    parser.add_argument("--only", default=None, help="逗号分隔的子任务：" + ",".join(TASKS))
    parser.add_argument("--validate-only", action="store_true", help="只做完整性校验")
    args = parser.parse_args()

    if args.validate_only:
        stats = validate()
        out = DATA_ROOT / "SUPPLEMENTARY_DATA_stats.json"
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"\n校验结果已写入 {out}")
        return 0

    names = args.only.split(",") if args.only else list(TASKS)
    for name in names:
        print(f"\n===== {name} =====", flush=True)
        TASKS[name]()
    print("\n全部完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
