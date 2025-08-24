#!/usr/bin/env python3
"""
Kronos Live Forecast - 主启动脚本
统一入口，处理路径和依赖关系
"""
import sys
import os
from pathlib import Path

# 添加核心模块路径
project_root = Path(__file__).parent
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

# 导入并运行核心预测模块
from core.update_predictions import main

if __name__ == "__main__":
    main()