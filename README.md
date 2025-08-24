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
├── tools/                     # 数据处理工具
│   ├── setup_data.py         # 📥 数据初始化脚本
│   ├── view_data.py          # 👁️ 数据查看工具
│   ├── download_long_history.py # 📅 长期历史数据下载器
│   ├── data_validator.py     # ✅ 数据验证修正工具
│   └── csv_exporter.py       # 📄 CSV导出器
├── run_prediction.py          # 🚀 主启动脚本
└── run_single.py             # 🔄 单次运行脚本
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r configs/requirements.txt
```

### 2. 初始化数据（推荐）
```bash
python tools/setup_data.py --setup
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
python tools/setup_data.py --verify

# 数据查看和导出
python tools/view_data.py --both

# 长期历史数据下载
python tools/download_long_history.py

# 数据验证和修正
python tools/data_validator.py
```

## 🔄 详细运行流程

### 单次运行模式 (推荐用于开发和测试)
```bash
python run_single.py
```

**完整流程时间分析**:
1. **配置加载** (0.1秒): 读取`configs/config.yaml`
2. **模型加载** (0.4秒): 加载24.7M参数的Kronos-small模型
3. **数据获取** (0.0秒): 智能缓存，有缓存时跳过网络请求  
4. **主要预测** (60秒): 24小时预测，30次蒙特卡洛采样
5. **波动性预测** (60秒): 计算上涨概率和波动性放大
6. **图表生成** (3秒): 生成`frontend/prediction_chart.png`
7. **HTML更新** (0.1秒): 更新`frontend/index.html`
8. **Git提交** (0.5秒): 自动提交更新

**总耗时**: 约124秒 (如有缓存，数据获取几乎瞬时)

### 持续运行模式
```bash
python run_prediction.py
```

**调度机制**:
- 每小时XX:00:05自动触发预测
- 内置5分钟重试机制
- 自动内存清理和垃圾回收
- Ctrl+C优雅退出

### 数据管理优化

**智能缓存策略**:
```bash
# 首次运行或缓存过期
数据获取: 3-8秒 (CCXT多交易所)

# 后续运行 (缓存命中)
数据获取: 0.0秒 (直接使用本地缓存)
```

**数据源优先级**:
1. **本地缓存** (Parquet格式，毫秒级访问)
2. **CCXT交易所API** (Binance → OKX → Bybit → Gate → Huobi)
3. **外部数据源** (CoinGecko → Yahoo Finance → 模拟数据)

## 🚀 性能优化亮点

- **延迟初始化**: CCXT数据源按需连接，避免70秒预初始化
- **智能缓存**: 缓存命中率90%+，大幅减少网络请求
- **文件路径修复**: 解决`frontend/`目录访问问题
- **内存管理**: 自动垃圾回收，支持长期运行

## 🔧 故障排除

### 常见问题

**Q: 运行时出现文件路径错误**
```bash
FileNotFoundError: [Errno 2] No such file or directory: '.../frontend/prediction_chart.png'
```
A: 已修复，确保运行最新版本。项目根目录必须包含`frontend/`文件夹。

**Q: 数据获取缓慢 (>60秒)**  
A: 首次运行会初始化交易所连接，后续运行使用缓存几乎瞬时。可通过以下优化：
```bash
# 检查缓存状态
python tools/view_data.py --both

# 强制使用缓存模式
# 在config.yaml中设置: update_mode: "cache_only"
```

**Q: 模型加载失败**
```bash
FileNotFoundError: ../Kronos_model
```
A: 确保Kronos模型在正确路径。检查`configs/config.yaml`中的`model_path`设置。

**Q: 预测结果异常 (如100%上涨概率)**  
A: 这可能是正常的模型输出，特别是在特定市场条件下。可以：
- 调整采样参数 (`temperature`, `num_samples`)
- 切换到更大的Kronos-base模型
- 检查输入数据质量

### 性能优化建议

**开发阶段**:
```bash
# 减少采样次数加快测试
# config.yaml: num_samples: 10 (默认30)

# 禁用Git自动提交
# config.yaml: auto_commit: false
```

**生产运行**:
```bash
# 使用更大模型提高精度
# config.yaml: model_name: "Kronos-base" 

# 启用全部功能
# config.yaml: num_samples: 30, auto_commit: true
```

## 📝 更新日志

- ✅ 修复pandas FutureWarning警告
- ✅ 重组项目目录结构，提高可维护性
- ✅ 创建统一启动脚本，简化使用流程
- ✅ 实现智能数据缓存系统
- ✅ 支持多模型配置和动态切换
- ✅ CCXT数据源延迟初始化优化
- ✅ 修复文件路径错误，确保图表正常生成