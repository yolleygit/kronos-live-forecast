#!/usr/bin/env bash
set -euo pipefail

# 使用两个不同的配置同时运行（后台）
# 用法： ./scripts/start_dual_symbols.sh configs/config.btc.yaml configs/config.eth.yaml

BTC_CFG=${1:-configs/config.btc.yaml}
ETH_CFG=${2:-configs/config.eth.yaml}

echo "🚀 启动双币种调度器..."
echo "  • BTC 配置: $BTC_CFG"
echo "  • ETH 配置: $ETH_CFG"

# BTC
KRONOS_CONFIG="$BTC_CFG" nohup python run_prediction.py --config "$BTC_CFG" > logs/btc_scheduler.out 2>&1 &
BTC_PID=$!

# ETH
KRONOS_CONFIG="$ETH_CFG" nohup python run_prediction.py --config "$ETH_CFG" > logs/eth_scheduler.out 2>&1 &
ETH_PID=$!

echo "✅ 已启动: BTC PID=$BTC_PID, ETH PID=$ETH_PID"
echo "📄 日志: logs/btc_scheduler.out, logs/eth_scheduler.out"

