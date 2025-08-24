# CLAUDE.md

<!-- 此文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导 -->
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project
<!-- 项目简介：Kronos 实时预测应用 -->

This is the **Kronos Live Forecast** application - a demo showcasing the Kronos foundation model for financial markets prediction. It provides live BTC/USDT price forecasting using a pre-trained transformer model with probabilistic Monte Carlo sampling.

<!-- 这是 Kronos 实时预测应用 - 展示用于金融市场预测的 Kronos 基础模型的演示。
它使用预训练的transformer模型和概率性蒙特卡洛采样提供实时BTC/USDT价格预测。 -->

The application consists of:
- A static HTML dashboard displaying live forecasts  <!-- 静态HTML仪表板显示实时预测 -->
- Python prediction engine using the Kronos model  <!-- 使用Kronos模型的Python预测引擎 -->
- Automated hourly updates with git commits  <!-- 每小时自动更新并提交git -->
- Real-time chart generation with matplotlib  <!-- 使用matplotlib实时生成图表 -->

## Development Commands
<!-- 开发命令：核心操作和环境配置 -->

### Core Operations
<!-- 核心操作 -->
- **Run prediction update**: `python3 update_predictions.py`  <!-- 运行预测更新 -->
- **Serve local development**: `python3 -m http.server 8000` (serves HTML at localhost:8000)  <!-- 启动本地开发服务器 -->

### Model and Dependencies
<!-- 模型和依赖：环境要求和文件位置 -->
- **Python version**: 3.12.2
- **Key dependencies**: torch, pandas, numpy, matplotlib, python-binance, huggingface_hub, pyyaml, safetensors
- **Model location**: `../Kronos_model` (external to repo) - **已确认存在并可用**
- **Available models**: 
  - **Kronos-small**: 24.7M parameters (默认)
  - **Kronos-base**: 102.3M parameters
  - **Kronos-Tokenizer-base**: 共享的分词器
- **Processing**: 使用 CPU 推理，无需 GPU 加速
- **Configuration**: 通过 `config.yaml` 配置文件管理所有参数
- **Install dependencies**: `pip3 install torch pandas numpy matplotlib python-binance huggingface_hub pyyaml safetensors`

### Testing and Validation
<!-- 测试和验证：通过可视化检查 -->
- No automated tests present - validation is through visual inspection of generated charts and metrics  <!-- 无自动化测试 - 通过可视化检查图表和指标来验证 -->
- Check prediction accuracy by comparing `prediction_chart.png` output  <!-- 通过对比prediction_chart.png输出检查预测准确性 -->
- Validate metrics: upside probability and volatility amplification should be reasonable (0-100%)  <!-- 验证指标：上涨概率和波动性放大应在合理范围(0-100%) -->

## Architecture Overview
<!-- 架构概述：核心组件和数据流程 -->

### Core Components
<!-- 核心组件：前端和预测引擎 -->

**Frontend (Static HTML)**  <!-- 前端静态HTML -->
- `index.html`: Main dashboard with live metrics display  <!-- 主仪表板，显示实时指标 -->
- `style.css`: CSS styling with CSS variables for theming  <!-- CSS样式，支持主题定制 -->
- `prediction_chart.png`: Auto-generated forecast visualization  <!-- 自动生成的预测可视化图表 -->

**Prediction Engine**  <!-- 预测引擎 -->
- `update_predictions.py`: Main orchestrator script  <!-- 主要编排脚本 -->
- `model/kronos.py`: Core Kronos model implementation with tokenizer and predictor classes  <!-- 核心Kronos模型实现，包含分词器和预测器类 -->
- `model/module.py`: Neural network components (transformers, attention, quantization)  <!-- 神经网络组件(transformer, 注意力机制, 量化) -->
- `model/__init__.py`: Model registry and factory functions  <!-- 模型注册和工厂函数 -->

### Key Classes and Functions

**KronosPredictor** (`model/kronos.py:454`): Main interface for generating predictions
- `predict()`: Generates probabilistic forecasts from DataFrame input
- Uses normalized data with clipping and Monte Carlo sampling

**Kronos** (`model/kronos.py:180`): Core transformer model
- Hierarchical token embedding with s1/s2 bits
- Autoregressive generation with temperature and top-p sampling
- Dual-head architecture for conditional prediction

**update_predictions.py workflow**:
1. `fetch_binance_data()`: Gets latest K-line data via Binance API
2. `make_prediction()`: Runs Kronos model inference (30 Monte Carlo samples)  
3. `calculate_metrics()`: Computes upside probability and volatility amplification
4. `create_plot()`: Generates matplotlib chart with historical + forecast data
5. `update_html()`: Updates HTML with new metrics using regex replacement
6. `git_commit_and_push()`: Auto-commits changes with timestamp

### Data Flow

1. **Input**: Binance 1h K-line data (last 360 hours context)
2. **Processing**: Normalization, tokenization via BSQuantizer 
3. **Prediction**: 24-hour horizon with 30 Monte Carlo paths
4. **Output**: Mean forecast + uncertainty bounds + probability metrics
5. **Visualization**: Combined price/volume chart with historical context

### Configuration
<!-- 配置参数：通过config.yaml文件管理 -->

**配置文件**: `config.yaml` - 集中管理所有关键参数

**模型配置**:
- `model_name`: "Kronos-small" (24.7M) 或 "Kronos-base" (102.3M)
- `max_context`: 512 tokens
- `device`: "cpu" 或 "cuda"

**采样配置**:
- `temperature`: 0.6 (预测随机性控制)
- `top_p`: 0.9 (Top-p采样参数)  
- `num_samples`: 30 (蒙特卡洛采样次数)

**数据配置**:
- `history_window`: 360 hours (15天历史数据)
- `forecast_horizon`: 24 hours (预测时间范围)
- `volatility_window`: 24 hours (波动性计算窗口)

### Automation

The application runs on an hourly schedule via `run_scheduler()`:
- Waits until XX:00:05 each hour
- Executes full prediction pipeline
- Handles errors with 5-minute retry logic
- Memory cleanup after each run

## File Structure Notes

- **Model files**: All AI/ML code in `model/` directory
- **Static assets**: HTML, CSS, and generated images at root level  
- **Single script execution**: `update_predictions.py` handles entire pipeline
- **No build process**: Direct Python execution, no compilation needed
- **External model dependency**: 
  - 路径: `../Kronos_model/` (已确认存在)
  - 包含: Kronos-Tokenizer-base 和 Kronos-base 模型
  - 格式: HuggingFace 格式 (config.json + model.safetensors)

## Development Patterns

- Uses HuggingFace transformers pattern with `PyTorchModelHubMixin`
- PyTorch-native implementation with custom modules
- Pandas for time series data handling  
- Git automation for deployment/updates
- Error handling with try/catch and graceful degradation
- Memory management with explicit cleanup and garbage collection

## Common Development Workflows
<!-- 常见开发工作流程：预测逻辑修改和样式更改 -->

When modifying prediction logic, test changes by:  <!-- 修改预测逻辑时，通过以下步骤测试更改： -->
1. Run `python3 update_predictions.py` once manually  <!-- 手动运行一次 -->
2. Check generated `prediction_chart.png` for visual correctness  <!-- 检查生成的预测图表是否正确 -->
3. Verify HTML updates show reasonable probability values  <!-- 验证HTML更新显示合理的概率值 -->
4. Monitor memory usage for long-running scheduler  <!-- 监控长期运行调度器的内存使用 -->

When styling changes are needed:  <!-- 需要样式更改时： -->
1. Edit `style.css` using CSS variables  <!-- 使用CSS变量编辑样式文件 -->
2. Test with local HTTP server  <!-- 用本地HTTP服务器测试 -->
3. Coordinate with automated HTML updates in `update_html()`  <!-- 与update_html()中的自动HTML更新协调 -->

## 重要注意事项 / Important Notes

### 配置文件管理 / Configuration Management
- **主配置文件**: `config.yaml` - 集中管理所有参数
- **模型切换**: 修改 `model.model_name` 在 "Kronos-small" (24.7M) 和 "Kronos-base" (102.3M) 间切换
- **性能调优**: 调整 `sampling.num_samples` 和 `data.history_window` 优化性能
- **参数验证**: 程序启动时会显示加载的模型参数量和配置信息

### 程序运行模式 / Program Execution Modes
- **单次运行 / Single Run**: 注释掉 `run_scheduler()` 调用，避免进入持续循环
- **持续运行 / Continuous Mode**: 默认每小时自动执行预测，包含错误重试机制
- **中断方法 / Interrupt**: 使用 Ctrl+C 正常退出程序

### 性能考虑 / Performance Considerations  
- **Kronos-small**: ~5-8分钟 (24.7M参数，推荐日常使用)
- **Kronos-base**: ~10-15分钟 (102.3M参数，更高精度)
- **内存管理**: 内置垃圾回收和内存清理
- **网络备用**: 自动切换到模拟数据当API不可用时

### 文件输出 / File Outputs
- `prediction_chart.png`: 预测可视化图表
- `index.html`: 自动更新的仪表板 
- `config.yaml`: 集中配置文件
- Git自动提交: 每次运行后自动提交更新