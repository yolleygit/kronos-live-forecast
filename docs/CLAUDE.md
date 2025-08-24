# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

Kronos Live Forecast - 基于Kronos foundation model的BTC/USDT实时价格预测系统，使用预训练transformer模型和概率性蒙特卡洛采样提供24小时价格预测。

**核心特性**：
- 基于transformer架构的Kronos模型进行价格预测
- 蒙特卡洛采样生成概率分布和置信区间 
- 多数据源支持(Binance/CCXT/CoinGecko/Yahoo Finance)和智能降级
- Parquet缓存系统优化数据访问性能
- 静态HTML仪表板实时展示预测结果和可视化图表

## Development Commands

### 核心操作
- **安装依赖**: `pip install -r configs/requirements.txt`
- **初始化数据**: `python tools/setup_data.py --setup` (首次运行推荐)
- **单次预测**: `python run_single.py` (开发测试首选，约2分钟)
- **持续运行**: `python run_prediction.py` (生产模式，每小时XX:00:05自动更新)
- **启动Web服务**: `cd frontend && python -m http.server 8000`

### 数据管理工具
- **数据验证**: `python tools/data_validator.py` - 检查和修复数据完整性
- **查看数据**: `python tools/view_data.py --both` - 检查缓存状态和数据质量
- **数据导出**: `python tools/csv_exporter.py` - 导出历史数据为CSV
- **历史数据**: `python tools/download_long_history.py` - 下载长期历史数据
- **数据设置**: `python tools/setup_data.py --verify` - 验证数据质量

### 环境要求
- **Python**: 3.12.2+
- **关键依赖**: torch>=2.0.0, pandas>=2.0.0, matplotlib>=3.7.0, ccxt>=4.0.0
- **模型依赖**: `../Kronos_model` 目录 (外部模型文件，包含Kronos-small/base)
- **配置管理**: `configs/config.yaml` - 所有参数的中央配置文件

### 验证方法
该项目无传统单元测试，通过以下方式验证：
- 运行 `python run_single.py` 检查完整流程
- 检查生成的 `frontend/prediction_chart.png` 图表合理性
- 验证预测概率在0-100%合理范围内
- 监控长期运行的内存使用情况

## Architecture Overview

### 核心架构设计

**数据流架构**：
```
外部数据源 → 数据管理层 → 模型推理 → 结果输出 → Web展示
    ↓           ↓         ↓        ↓        ↓
多源降级    Parquet缓存  Transformer  图表生成  静态HTML
```

**主要模块**：
- **预测引擎**: `core/update_predictions.py` - 主业务逻辑和调度
- **模型层**: `model/kronos.py` - Transformer架构的Kronos模型实现  
- **数据层**: `core/data_manager.py` + `core/ccxt_data_source.py` - 数据获取和缓存管理
- **工具层**: `tools/` - 数据处理、验证、导出工具集
- **前端**: `frontend/` - 静态HTML仪表板和可视化

### 关键组件深度解析

**KronosPredictor** (`model/kronos.py`)：
- 基于HuggingFace PyTorchModelHubMixin的transformer模型
- `predict()` 方法：蒙特卡洛采样生成概率分布预测
- BSQuantizer: 二进制球面量化器用于数据标准化和压缩

**预测流程** (`core/update_predictions.py`)：
1. **数据获取**: 360小时BTCUSDT 1h K线数据 (多源容错)
2. **预处理**: BSQuantizer标准化和分词
3. **模型推理**: Kronos transformer生成24小时预测(30样本)
4. **后处理**: 计算上涨概率、波动性放大系数
5. **可视化**: matplotlib生成图表，更新HTML仪表板
6. **持久化**: 可选Git自动提交

**多数据源容错策略** (`core/external_data_sources.py`)：
```
Binance API → CCXT交易所池 → CoinGecko → Yahoo Finance → 模拟数据
     ↓              ↓           ↓          ↓         ↓
  主要源        5个交易所     免费API    传统金融   最后保障
```

**智能缓存系统** (`core/data_manager.py`)：
- Parquet格式高效存储 (毫秒级读取)
- 1小时缓存策略，减少90%+ API调用
- 元数据跟踪和完整性验证

### 配置系统架构

**配置文件** `configs/config.yaml` 分层设计：
- **model**: 模型选择(Kronos-small 24.7M/base 102.3M)、设备配置
- **sampling**: 温度(0.6)、top_p(0.9)、采样数(30)、随机性控制
- **data**: 历史窗口(360h)、预测范围(24h)、缓存策略
- **api**: 超时、重试、限流配置  
- **output**: 图表、HTML、Git自动提交控制

## Development Patterns

**核心设计模式**：
- **HuggingFace生态集成**: PyTorchModelHubMixin + transformers架构
- **时间序列处理**: Pandas + NumPy优化的金融数据管道
- **容错设计**: 多数据源降级 + 5分钟重试机制
- **性能优化**: Parquet缓存 + 延迟初始化 + 内存管理
- **配置驱动**: YAML配置文件统一管理所有参数

**运行模式详解**：
- **开发模式**: `python run_single.py` - 单次执行，约2分钟，适合测试
- **生产模式**: `python run_prediction.py` - 每小时XX:00:05自动触发  
- **调试模式**: 通过config.yaml调整日志级别和采样参数
- **缓存模式**: `update_mode: cache_only` 完全使用本地缓存

## Common Development Workflows

**模型或预测逻辑修改**：
1. 编辑相关代码文件 (`model/kronos.py`, `core/update_predictions.py`)
2. 运行 `python run_single.py` 验证更改
3. 检查 `frontend/prediction_chart.png` 图表输出
4. 验证概率值合理性 (0-100%，非异常值)
5. 检查控制台日志无错误或警告

**配置参数调优**：
- **性能 vs 精度**: `num_samples` (10快速测试 ↔ 50高精度)
- **模型选择**: Kronos-small (2-3min) ↔ Kronos-base (5-8min)
- **采样控制**: `temperature` (0.1确定性 ↔ 1.0随机性)
- **数据窗口**: `history_window` (120min ↔ 360h完整)

**前端界面定制**：
- 修改 `frontend/style.css` 样式(支持CSS变量)
- 运行 `cd frontend && python -m http.server 8000` 本地预览
- 注意 `core/update_predictions.py` 中的 `update_html()` 自动更新逻辑

**数据问题诊断**：
1. `python tools/view_data.py --both` - 检查缓存状态
2. `python tools/data_validator.py` - 验证数据完整性
3. `python tools/setup_data.py --verify` - 重新验证设置
4. 检查 `data/metadata.json` 了解数据源状态

## Key Dependencies & External Resources

**关键外部依赖**：
- **Kronos模型**: `../Kronos_model/` 目录(外部模型文件，HuggingFace格式)
- **网络连接**: 获取实时价格数据，支持离线模拟数据降级
- **Git仓库**: 可选的自动提交功能需要配置Git环境

**核心输出文件**：
- `frontend/prediction_chart.png` - 预测可视化图表(matplotlib生成)
- `frontend/index.html` - 自动更新的Web仪表板
- `data/btc_cache.parquet` - Parquet格式的数据缓存
- `data/metadata.json` - 缓存元数据和状态跟踪

## Performance Characteristics

**执行时间分析** (单次运行):
- 配置加载: ~0.1s
- 模型加载: ~0.4s (Kronos-small) / ~1.2s (Kronos-base)
- 数据获取: ~0.0s (缓存命中) / ~3-8s (网络获取)
- 预测推理: ~60s (24h预测) + ~60s (波动性预测)
- 图表生成: ~3s
- 总耗时: ~124s (有缓存时约63s)

**内存使用**:
- Kronos-small: ~200MB 峰值内存
- Kronos-base: ~500MB 峰值内存 
- 包含自动垃圾回收机制支持长期运行

**缓存效率**:
- 缓存命中率: >90% (生产环境)
- Parquet读取: <10ms vs API调用3-8s
- 缓存策略: 1小时有效期，元数据完整性验证