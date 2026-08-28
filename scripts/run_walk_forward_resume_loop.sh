#!/usr/bin/env bash
# Walk-forward 单切分独立进程循环：
# 每个切分一个独立 python 进程，退出即由 OS 完全回收内存，
# 规避 pandas/pyarrow/LightGBM 各 allocator 滞留内存导致的累积 OOM。
# 断点续跑：已有 oos_predictions.parquet 的切分自动跳过。
set -u
cd /home/karl/repos/personal/stock_qt_nd
LOG=/tmp/wf_train5.log
export MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1

for i in $(seq 64 90); do
  ckpt="models/wf_split_$(printf %02d "$i")/oos_predictions.parquet"
  if [ -f "$ckpt" ]; then
    echo "[loop] 切分 $i 已有 checkpoint，跳过 $(date)" >> "$LOG"
    continue
  fi
  echo "[loop] === 启动切分 $i $(date) ===" >> "$LOG"
  uv run python scripts/run_walk_forward_train.py --splits "$i" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[loop] 切分 $i 失败 rc=$rc $(date)，终止循环" >> "$LOG"
    exit "$rc"
  fi
done

echo "[loop] === 全部切分完成，执行 OOS 合并 + 最终模型 $(date) ===" >> "$LOG"
uv run python scripts/run_walk_forward_train.py >> "$LOG" 2>&1
rc=$?
echo "[loop] === 结束 rc=$rc $(date) ===" >> "$LOG"
exit "$rc"
