#!/usr/bin/env bash
# 战役 #47 M2(issue #49)全链路:四来源 + 主表,确定性双跑 + 独立自检。
# 全程约 3~4.5 小时;心跳落 progress.log。
# 用法: bash run_all.sh [并发数,默认 24]
set -euo pipefail
cd "$(dirname "$0")"
W="${1:-24}"
mkdir -p cache rebuild

echo "== s1 来源 1(事件级词典 + 几何族 40)第一遍 =="
python build_event_dictionary.py --workers "$W"
echo "== s1 第二遍(确定性) =="
python build_event_dictionary.py --workers "$W" --out rebuild/s1_event_dictionary.parquet

echo "== s2 特征工厂 第一遍 =="
python run_factory_v6.py --workers "$W"
echo "== s2 第二遍 =="
python run_factory_v6.py --workers "$W" --out rebuild/s2_factory_full.parquet

echo "== s3 日频快照 第一遍(Pass A 全历史 parts,约 21 GiB) =="
python run_v4daily_v6.py --workers "$W"
echo "== s3 第二遍(全新 parts 目录,核后删副本) =="
python run_v4daily_v6.py --workers "$W" \
  --parts-dir rebuild/v4daily_parts_fullhist --out rebuild/s3_v4daily_snapshot.parquet

echo "== s4 T3 快照 第一遍 =="
python run_t3_v6.py --workers "$W"
echo "== s4 第二遍 =="
python run_t3_v6.py --workers "$W" --out rebuild/s4_t3_snapshot.parquet

echo "== 主表合并 第一遍 =="
python build_master_v6.py
echo "== 主表合并 第二遍 =="
python build_master_v6.py --out rebuild/master_v6.parquet \
  --results rebuild/master_results_v6.json

echo "== 独立自检(含 md5 逐位核验) =="
python selfcheck_acceptance.py --workers "$W"

echo "== 核验通过,删除 rebuild/ 大产物副本(哈希已落 selfcheck 台账) =="
rm -rf rebuild/v4daily_parts_fullhist rebuild/*.parquet
echo "== 全链路完成 =="
