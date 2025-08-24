#!/usr/bin/env python3
"""
Kronos Live Forecast - 数据预下载脚本
初始化数据缓存和下载历史数据
"""
import sys
from pathlib import Path

# 添加核心模块路径
project_root = Path(__file__).parent.parent
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

# 导入并运行数据预下载器
from data_preloader import main

if __name__ == "__main__":
    main()