#!/usr/bin/env python3
"""
Kronos Live Forecast - 单次运行脚本
避免调度器循环，仅执行一次预测
"""
import sys
from pathlib import Path

# 添加核心模块路径
project_root = Path(__file__).parent
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

# 导入并运行单次预测
from core.single_run import main

if __name__ == "__main__":
    main()