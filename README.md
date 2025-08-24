# Kronos Live Forecast

基于Kronos foundation model的BTC/USDT实时价格预测系统。

## 📁 项目结构

```
Kronos-app/
├── configs/                    # 配置文件
│   ├── config.yaml            # 主配置文件
│   └── requirements.txt       # Python依赖
├── core/                      # 核心Python脚本
│   ├── update_predictions.py  # 主预测引擎
│   ├── single_run.py         # 单次运行脚本
│   ├── data_manager.py       # 数据管理器
│   ├── data_preloader.py     # 历史数据预下载器
│   ├── external_data_sources.py # 外部数据源管理
│   └── realtime_data.py      # 实时数据流（暂未启用）
├── frontend/                  # 前端文件
│   ├── index.html            # Web仪表板
│   ├── style.css             # 样式文件
│   └── prediction_chart.png  # 生成的预测图表
├── model/                     # AI模型
│   ├── kronos.py             # Kronos模型实现
│   ├── module.py             # 神经网络模块
│   └── __init__.py          # 模型注册
├── data/                      # 数据缓存目录
├── docs/                      # 文档
│   ├── CLAUDE.md             # 项目开发指南
│   ├── 数据优化方案.md       # 数据优化文档
│   └── 运行文档.md           # 运行说明
├── img/                       # 静态图片
├── scripts/                   # 工具脚本
├── web/                       # Web相关
├── run_prediction.py          # 🚀 主启动脚本
├── run_single.py             # 🔄 单次运行脚本
└── setup_data.py             # 📥 数据初始化脚本
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r configs/requirements.txt
```

### 2. 初始化数据（推荐）
```bash
python setup_data.py --setup
```

### 3. 运行预测
```bash
# 单次运行（推荐用于测试）
python run_single.py

# 持续运行（每小时自动更新）
python run_prediction.py
```

### 4. 查看结果
打开 `frontend/index.html` 查看预测仪表板，或启动本地服务器：
```bash
cd frontend && python -m http.server 8000
```

## ⚙️ 配置说明

主配置文件：`configs/config.yaml`
- 模型设置：支持Kronos-small (24.7M) 和 Kronos-base (102.3M)
- 采样参数：温度、top-p、样本数量
- 数据源：交易所API配置和缓存设置

## 📊 核心功能

- **智能预测**：24小时BTC价格预测，包含概率区间
- **数据缓存**：高效的本地数据管理，减少API调用
- **实时更新**：自动获取最新市场数据并更新预测
- **可视化**：直观的价格走势图和关键指标
- **多数据源**：支持Binance、CoinGecko、Yahoo Finance等

## 🔧 开发模式

```bash
# 数据质量验证
python core/data_preloader.py --verify

# 检查缓存状态
python core/data_manager.py

# 测试外部数据源
python core/external_data_sources.py
```

## 📝 更新日志

- ✅ 修复pandas FutureWarning警告
- ✅ 重组项目目录结构，提高可维护性
- ✅ 创建统一启动脚本，简化使用流程
- ✅ 实现智能数据缓存系统
- ✅ 支持多模型配置和动态切换