# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

Kronos Live Forecast - 基于Kronos foundation model的BTC/USDT实时价格预测系统，使用预训练transformer模型和概率性蒙特卡洛采样提供24小时价格预测。

## Development Commands

### 核心操作
- **安装Python依赖**: `pip install -r configs/requirements.txt`
- **初始化数据**: `python tools/setup_data.py --setup` (首次运行推荐)
- **单次预测**: `python run_single.py` (开发测试首选，约2分钟)
- **持续运行**: `python run_prediction.py` (生产模式，每小时XX:00:05自动更新)

### Web Dashboard相关
- **启动现代化仪表板**: `./scripts/start_web_dashboard.sh` (Next.js，端口3000)
- **启动传统仪表板**: `cd frontend && python -m http.server 8000`
- **Web开发模式**: `cd web && npm run dev`
- **Web生产构建**: `cd web && npm run build && npm start`
- **Web Lint检查**: `cd web && npm run lint`

### 数据管理工具
- **数据验证**: `python tools/data_validator.py`
- **查看数据**: `python tools/view_data.py --both`
- **数据导出**: `python tools/csv_exporter.py`
- **历史数据下载**: `python tools/download_long_history.py`

### 环境要求
- **Python**: 3.12.2+
- **关键Python依赖**: torch>=2.0.0, pandas>=2.0.0, matplotlib>=3.7.0, ccxt>=4.0.0, pyarrow>=12.0.0
- **Node.js**: 18.0+ (用于web仪表板)
- **外部模型**: `../Kronos_model` 目录包含Kronos-small/base模型文件

## High-Level Architecture

### 核心架构设计
```
外部数据源 → 数据管理层 → 模型推理 → 结果输出 → Web展示
    ↓           ↓         ↓        ↓        ↓
多源降级    Parquet缓存  Transformer  图表生成  双重仪表板
```

**主要模块**：
- **预测引擎**: `core/update_predictions.py` - 主业务逻辑和调度
- **模型层**: `model/kronos.py` - Transformer架构的Kronos模型实现  
- **数据层**: `core/data_manager.py` + `core/ccxt_data_source.py` - 数据获取和缓存管理
- **工具层**: `tools/` - 数据处理、验证、导出工具集
- **前端双系统**: `frontend/` (静态HTML) + `web/` (Next.js现代化)

### 配置系统
- **主配置**: `configs/config.yaml` - 模型、采样、数据源、API配置
- **模型选择**: Kronos-small (24.7M) 或 Kronos-base (102.3M)
- **Web配置**: `web/package.json` - Next.js依赖和脚本

### 数据流设计
1. **数据获取**: 360小时BTCUSDT 1h K线数据，支持Binance → CCXT → CoinGecko → Yahoo Finance多级降级
2. **智能缓存**: Parquet格式存储，1小时缓存策略，90%+缓存命中率
3. **模型推理**: Kronos transformer + 蒙特卡洛采样(30次)
4. **结果输出**: 12个增强指标(上涨概率6个+波动率6个)
5. **双重展示**: 静态HTML(传统) + Next.js(现代化)

## Key Development Patterns

### 测试和验证
项目无传统单元测试，通过以下方式验证：
- `python run_single.py` 检查完整流程
- 检查生成的 `frontend/prediction_chart.png` 图表
- 验证预测概率在0-100%合理范围
- 监控长期运行的内存使用情况

### 调试工作流
1. **快速测试**: `python run_single.py` (约2分钟完整流程)
2. **数据诊断**: `python tools/view_data.py --both`
3. **缓存检查**: 查看 `data/btc_metadata.json`
4. **配置调试**: 修改 `configs/config.yaml` 中的 `num_samples` 和 `temperature`

### 性能调优
- **开发模式**: `num_samples: 10`, `auto_commit: false`
- **生产模式**: `num_samples: 30`, `auto_commit: true`
- **模型切换**: Kronos-small(快速) ↔ Kronos-base(精确)
- **缓存优化**: `update_mode: cache_only` 完全使用本地缓存

### Web开发工作流
- **前端调试**: 修改 `frontend/style.css` + `cd frontend && python -m http.server 8000`
- **现代化开发**: `cd web && npm run dev` (支持热重载)
- **生产部署**: `./scripts/start_web_dashboard.sh --production`

## Performance Characteristics

### 执行时间分析 (单次运行)
- 配置加载: ~0.1s
- 模型加载: ~0.4s (Kronos-small) / ~1.2s (Kronos-base)  
- 数据获取: ~0.0s (缓存命中) / ~3-8s (网络获取)
- 预测推理: ~60s (价格预测) + ~60s (波动率预测)
- 图表生成: ~3s
- **总耗时**: ~124s (有缓存时约63s)

### 内存和缓存效率
- **内存使用**: Kronos-small ~200MB, Kronos-base ~500MB
- **缓存策略**: 1小时有效期，Parquet格式，<10ms读取速度
- **API优化**: 缓存命中率>90%，减少网络调用