#!/bin/bash
# 继续生成剩余日期的特征文件（带详细日志）
# 从2022年9月21日开始到2024年12月31日

set -e  # 遇到错误立即退出
set -o pipefail  # 管道命令的错误也会被捕获

# 配置
LOG_DIR="logs/feature_generation"
LOG_FILE="${LOG_DIR}/continue_$(date +%Y%m%d_%H%M%S).log"
WORKERS=8
BATCH_SIZE=15

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ERROR: $*" | tee -a "${LOG_FILE}" >&2
}

log_section() {
    local line="=========================================="
    log "${line}"
    log "$*"
    log "${line}"
}

# 捕获错误
trap 'log_error "脚本在第 $LINENO 行出错，退出码: $?"' ERR

# 开始
log_section "开始特征计算：2018-01 到 2024-12（覆盖模式）"
log "日志文件: ${LOG_FILE}"
log "配置: workers=${WORKERS}, batch_size=${BATCH_SIZE}"
log ""

# 查看当前状态
log "当前状态："
uv run python scripts/run_feature_pipeline_incremental.py --status | tee -a "${LOG_FILE}"
log ""

# 处理函数
process_month() {
    local year=$1
    local month=$2
    local month_str=$(printf "%02d" $month)

    log_section "开始处理 ${year}-${month_str}"
    local start_time=$(date +%s)

    # 运行特征计算，输出到日志
    if uv run python scripts/run_feature_pipeline_incremental.py \
        --year ${year} --month ${month} \
        --workers ${WORKERS} --batch-size ${BATCH_SIZE} 2>&1 | tee -a "${LOG_FILE}"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "✅ ${year}-${month_str} 完成，耗时: ${duration}秒"
    else
        local exit_code=$?
        log_error "${year}-${month_str} 失败，退出码: ${exit_code}"
        return ${exit_code}
    fi

    # 显示当前状态
    log "当前进度:"
    uv run python scripts/run_feature_pipeline_incremental.py --status | tee -a "${LOG_FILE}"
    log ""
}

# 2018年全年
log_section "处理 2018年 (1-12月)"
for month in {1..12}; do
    process_month 2018 ${month}
done

# 2019年全年
log_section "处理 2019年 (1-12月)"
for month in {1..12}; do
    process_month 2019 ${month}
done

# 2020年全年
log_section "处理 2020年 (1-12月)"
for month in {1..12}; do
    process_month 2020 ${month}
done

# 2021年全年
log_section "处理 2021年 (1-12月)"
for month in {1..12}; do
    process_month 2021 ${month}
done

# 2022年全年
log_section "处理 2022年 (1-12月)"
for month in {1..12}; do
    process_month 2022 ${month}
done

# 2023年全年
log_section "处理 2023年 (1-12月)"
for month in {1..12}; do
    process_month 2023 ${month}
done

# 2024年全年
log_section "处理 2024年 (1-12月)"
for month in {1..12}; do
    process_month 2024 ${month}
done

# 最终状态
log ""
log_section "全部完成！"
log "最终状态："
uv run python scripts/run_feature_pipeline_incremental.py --status | tee -a "${LOG_FILE}"

# 统计特征文件
log ""
log "特征文件统计："
file_count=$(ls real_feature_data_daily/*.csv 2>/dev/null | wc -l)
total_size=$(du -sh real_feature_data_daily/ | cut -f1)
first_file=$(ls real_feature_data_daily/*.csv 2>/dev/null | head -1 | xargs basename)
last_file=$(ls real_feature_data_daily/*.csv 2>/dev/null | tail -1 | xargs basename)

log "  文件数量: ${file_count}"
log "  总大小: ${total_size}"
log "  日期范围: ${first_file} 到 ${last_file}"
log ""
log "详细日志已保存到: ${LOG_FILE}"
log ""
log "如有失败，可运行："
log "  uv run python scripts/run_feature_pipeline_incremental.py --retry-failed --workers 24"
