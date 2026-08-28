#!/bin/bash
# 测试单个月的特征生成（用于验证配置）

set -e
set -o pipefail

# 参数
YEAR=${1:-2022}
MONTH=${2:-10}
WORKERS=${3:-8}
BATCH_SIZE=${4:-15}

# 日志
LOG_DIR="logs/feature_generation"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_${YEAR}$(printf "%02d" ${MONTH})_$(date +%Y%m%d_%H%M%S).log"

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $*" | tee -a "${LOG_FILE}"
}

log "=========================================="
log "测试月份: ${YEAR}-$(printf "%02d" ${MONTH})"
log "配置: workers=${WORKERS}, batch_size=${BATCH_SIZE}"
log "日志文件: ${LOG_FILE}"
log "=========================================="
log ""

# 显示当前状态
log "当前状态:"
uv run python scripts/run_feature_pipeline_incremental.py --status | tee -a "${LOG_FILE}"
log ""

# 运行
log "开始处理..."
start_time=$(date +%s)

if uv run python scripts/run_feature_pipeline_incremental.py \
    --year ${YEAR} --month ${MONTH} \
    --workers ${WORKERS} --batch-size ${BATCH_SIZE} 2>&1 | tee -a "${LOG_FILE}"; then

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    log ""
    log "=========================================="
    log "✅ 测试成功！"
    log "耗时: ${duration}秒 ($(echo "scale=2; ${duration}/60" | bc)分钟)"
    log "=========================================="
else
    exit_code=$?
    log ""
    log "=========================================="
    log "❌ 测试失败，退出码: ${exit_code}"
    log "=========================================="
    exit ${exit_code}
fi

# 最终状态
log ""
log "最终状态:"
uv run python scripts/run_feature_pipeline_incremental.py --status | tee -a "${LOG_FILE}"
log ""
log "日志已保存到: ${LOG_FILE}"
