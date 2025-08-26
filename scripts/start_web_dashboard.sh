#!/bin/bash
# Kronos Web Dashboard 启动脚本

set -e

echo "🚀 启动 Kronos Web Dashboard"
echo "================================"

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/web"

# 检查web目录是否存在
if [ ! -d "$WEB_DIR" ]; then
    echo "❌ 错误: web目录不存在: $WEB_DIR"
    echo "请确保已完成Next.js项目设置"
    exit 1
fi

cd "$WEB_DIR"

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "❌ 错误: Node.js 未安装"
    echo "请安装Node.js 18.0或更高版本"
    echo "下载地址: https://nodejs.org/"
    exit 1
fi

# 检查npm是否可用
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: npm 未安装"
    exit 1
fi

# 显示Node.js版本
echo "📌 Node.js 版本: $(node --version)"
echo "📌 npm 版本: $(npm --version)"

# 检查package.json是否存在
if [ ! -f "package.json" ]; then
    echo "❌ 错误: package.json 不存在"
    exit 1
fi

# 检查node_modules是否存在，如果不存在则安装依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
else
    echo "📦 依赖已存在，跳过安装"
fi

# 检查是否需要构建（生产模式）
if [ "$1" = "--production" ] || [ "$1" = "-p" ]; then
    echo "🏗️  构建生产版本..."
    npm run build
    
    echo "🌟 启动生产服务器..."
    echo "访问地址: http://localhost:3000"
    echo "按 Ctrl+C 停止服务器"
    npm start
else
    echo "🛠️  启动开发服务器..."
    echo "访问地址: http://localhost:3000"
    echo "按 Ctrl+C 停止服务器"
    echo "开发模式支持热重载和实时更新"
    npm run dev
fi