#!/bin/bash

# ==============================================================================
# NPU Monitor Script for RK3588
# Usage: ./npu_monitor.sh
# ==============================================================================

watch -n 0.2 \
    'echo "===== NPU Monitor =====" && \
     echo "Time: $(date "+%Y-%m-%d %H:%M:%S")" && \
     cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo "Cannot access NPU load info" && \
     cat /sys/kernel/debug/clk/clk_summary | grep npu 2>/dev/null || echo "Cannot access NPU frequency info"'
