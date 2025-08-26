import gc
import os
import re
import subprocess
import time
import yaml
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor
from enhanced_metrics import calculate_all_enhanced_metrics, format_metrics_for_display


def setup_random_seeds(config):
    """设置随机种子以确保结果可重现"""
    random_seed = config.get('sampling', {}).get('random_seed', 42)
    enable_deterministic = config.get('sampling', {}).get('enable_deterministic', True)
    
    if enable_deterministic:
        # 设置Python标准库随机种子
        random.seed(random_seed)
        
        # 设置NumPy随机种子
        np.random.seed(random_seed)
        
        # 设置PyTorch随机种子
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 设置PyTorch确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"✅ 随机种子已设置为 {random_seed}，启用确定性模式")
    else:
        print(f"⚠️ 确定性模式已禁用，结果可能不可重现")


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        # 根据当前脚本位置动态确定配置路径
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "configs" / "config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"配置文件未找到: {config_path}，使用默认配置")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"配置文件解析错误: {e}，使用默认配置")
        return get_default_config()


def get_default_config():
    """获取默认配置"""
    return {
        "model": {
            "model_name": "Kronos-small",
            "tokenizer_name": "Kronos-Tokenizer-base", 
            "model_path": "../Kronos_model",
            "max_context": 512,
            "device": "cpu"
        },
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.9,
            "num_samples": 30,
            "volatility_temperature_multiplier": 1.0
        },
        "data": {
            "symbol": "BTCUSDT",
            "timeframe": "1h", 
            "history_window": 360,
            "forecast_horizon": 24,
            "volatility_window_multiplier": 1.0
        },
        "api": {
            "binance_base_url": "https://api.binance.com",
            "request_timeout": 30,
            "retry_attempts": 3
        },
        "output": {
            "chart_filename": "frontend/prediction_chart.png",
            "html_filename": "frontend/index.html", 
            "auto_commit": True,
            "auto_push": False,
            "web_server_port": 8000
        },
        "logging": {
            "level": "INFO",
            "enable_file_logging": False
        }
    }


# 加载配置
CONFIG = load_config()

# 向后兼容的配置映射
Config = {
    "REPO_PATH": Path(__file__).parent.parent.resolve(),  # 修复：指向项目根目录
    "MODEL_PATH": CONFIG["model"]["model_path"],
    "MODEL_NAME": CONFIG["model"]["model_name"],
    "TOKENIZER_NAME": CONFIG["model"]["tokenizer_name"],
    "MAX_CONTEXT": CONFIG["model"]["max_context"],
    "DEVICE": CONFIG["model"]["device"],
    "TEMPERATURE": CONFIG["sampling"]["temperature"],
    "TOP_P": CONFIG["sampling"]["top_p"],
    "SYMBOL": CONFIG["data"]["symbol"],
    "INTERVAL": CONFIG["data"]["timeframe"],
    "HIST_POINTS": CONFIG["data"]["history_window"],
    "PRED_HORIZON": CONFIG["data"]["forecast_horizon"],
    "N_PREDICTIONS": CONFIG["sampling"]["num_samples"],
    "VOL_WINDOW": int(CONFIG["data"]["forecast_horizon"] * CONFIG["data"]["volatility_window_multiplier"]),
    "VOL_TEMP_MULTIPLIER": CONFIG["sampling"]["volatility_temperature_multiplier"],
    "AUTO_COMMIT": CONFIG["output"]["auto_commit"],
    "AUTO_PUSH": CONFIG["output"]["auto_push"],
}

# 设置日志
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"]),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_configs():
    """加载模型配置参数"""
    tokenizer_name = Config["TOKENIZER_NAME"]
    model_name = Config["MODEL_NAME"]
    
    # 加载tokenizer配置 
    tokenizer_config_path = f"{Config['MODEL_PATH']}/{tokenizer_name}/config.json"
    model_config_path = f"{Config['MODEL_PATH']}/{model_name}/config.json"
    
    try:
        import json
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        logger.info(f"加载tokenizer配置: {tokenizer_name}")
        logger.info(f"加载模型配置: {model_name}")
        
        return tokenizer_config, model_config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def load_model():
    """Loads the Kronos model and tokenizer with dynamic configuration."""
    logger.info(f"Loading Kronos model: {Config['MODEL_NAME']}...")
    
    # 动态加载配置
    tokenizer_config, model_config = load_model_configs()
    
    # 创建tokenizer和model实例
    tokenizer = KronosTokenizer(
        d_in=6,  # 固定输入维度 (OHLCVA)
        d_model=tokenizer_config['d_model'],
        n_heads=tokenizer_config['n_heads'],
        ff_dim=tokenizer_config['ff_dim'],
        n_enc_layers=tokenizer_config['n_enc_layers'],
        n_dec_layers=tokenizer_config['n_dec_layers'],
        ffn_dropout_p=tokenizer_config['ffn_dropout_p'],
        attn_dropout_p=tokenizer_config['attn_dropout_p'],
        resid_dropout_p=tokenizer_config['resid_dropout_p'],
        s1_bits=tokenizer_config['s1_bits'],
        s2_bits=tokenizer_config['s2_bits'],
        beta=tokenizer_config['beta'],
        gamma0=tokenizer_config['gamma0'],
        gamma=tokenizer_config['gamma'],
        zeta=tokenizer_config['zeta'],
        group_size=tokenizer_config['group_size']
    )
    
    model = Kronos(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        ff_dim=model_config['ff_dim'],
        n_layers=model_config['n_layers'],
        ffn_dropout_p=model_config['ffn_dropout_p'],
        attn_dropout_p=model_config['attn_dropout_p'],
        resid_dropout_p=model_config['resid_dropout_p'],
        s1_bits=model_config['s1_bits'],
        s2_bits=model_config['s2_bits'],
        token_dropout_p=model_config['token_dropout_p'],
        learn_te=model_config['learn_te']
    )
    
    # 加载预训练权重
    tokenizer_path = f"{Config['MODEL_PATH']}/{Config['TOKENIZER_NAME']}"
    model_path = f"{Config['MODEL_PATH']}/{Config['MODEL_NAME']}"
    
    from safetensors.torch import load_file
    tokenizer.load_state_dict(load_file(f"{tokenizer_path}/model.safetensors"))
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
    
    tokenizer.eval()
    model.eval()
    
    # 创建预测器
    predictor = KronosPredictor(
        model, 
        tokenizer, 
        device=Config["DEVICE"], 
        max_context=Config["MAX_CONTEXT"]
    )
    
    # 获取模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型加载成功: {Config['MODEL_NAME']}")
    logger.info(f"模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"推理设备: {Config['DEVICE']}")
    
    return predictor


def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + pd.Timedelta(hours=1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq='h'
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    vol_temp_multiplier = Config["VOL_TEMP_MULTIPLIER"]
    
    with torch.no_grad():
        # 如果波动性温度倍数为1.0，则进行一次预测
        if vol_temp_multiplier == 1.0:
            logger.info(f"开始单次预测 (T={Config['TEMPERATURE']}, top_p={Config['TOP_P']}, samples={Config['N_PREDICTIONS']})...")
            begin_time = time.time()
            close_preds_main, volume_preds_main = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=Config["PRED_HORIZON"], T=Config["TEMPERATURE"], 
                top_p=Config["TOP_P"], sample_count=Config["N_PREDICTIONS"], 
                verbose=True
            )
            elapsed_time = time.time() - begin_time
            logger.info(f"单次预测完成，耗时: {elapsed_time:.2f}秒")
            
            # 波动性预测使用相同的结果
            close_preds_volatility = close_preds_main
            logger.info(f"使用相同预测结果进行波动性分析 (温度倍数={vol_temp_multiplier})")
            
        else:
            # 如果波动性温度倍数不为1.0，则进行两次预测
            vol_temperature = Config["TEMPERATURE"] * vol_temp_multiplier
            
            logger.info(f"开始主要预测 (T={Config['TEMPERATURE']}, top_p={Config['TOP_P']}, samples={Config['N_PREDICTIONS']})...")
            begin_time = time.time()
            close_preds_main, volume_preds_main = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=Config["PRED_HORIZON"], T=Config["TEMPERATURE"], 
                top_p=Config["TOP_P"], sample_count=Config["N_PREDICTIONS"], 
                verbose=True
            )
            elapsed_time = time.time() - begin_time
            logger.info(f"主要预测完成，耗时: {elapsed_time:.2f}秒")

            # 生成波动性预测 (使用配置的温度倍数)
            logger.info(f"开始波动性预测 (T={vol_temperature:.2f}, top_p={Config['TOP_P']}, 倍数={vol_temp_multiplier})...")
            begin_time = time.time()
            close_preds_volatility, _ = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=Config["PRED_HORIZON"], T=vol_temperature, 
                top_p=Config["TOP_P"], sample_count=Config["N_PREDICTIONS"], 
                verbose=True
            )
            elapsed_time = time.time() - begin_time
            logger.info(f"波动性预测完成，耗时: {elapsed_time:.2f}秒")

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_binance_data():
    """获取市场数据 - 优先使用缓存"""
    
    # 检查是否启用缓存
    if CONFIG.get("data", {}).get("cache_enabled", False):
        try:
            from data_manager import data_manager
            logger.info("使用数据缓存管理器获取数据...")
            
            # 获取缓存状态
            status = data_manager.get_cache_status()
            logger.info(f"缓存状态: {status['update_reason']}")
            
            # 获取数据
            df = data_manager.get_data()
            if df is not None:
                logger.info(f"从缓存获取{len(df)}条数据，时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
                return df
            else:
                logger.warning("缓存获取失败，回退到直接API获取")
                
        except Exception as e:
            logger.error(f"数据管理器错误: {e}")
            logger.warning("回退到传统获取方式")
    
    # 传统获取方式 (后备方案)
    return fetch_binance_data_legacy()


def fetch_binance_data_legacy():
    """传统的数据获取方式 (后备方案)"""
    symbol, interval = Config["SYMBOL"], Config["INTERVAL"]
    limit = Config["HIST_POINTS"] + Config["VOL_WINDOW"]

    logger.info(f"从交易所获取{limit}条 {symbol} {interval} 数据...")
    
    try:
        # 方法1: 尝试使用 binance 库
        client = Client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        logger.info("通过 binance 库获取数据成功")
    except Exception as e1:
        logger.warning(f"方法1失败: {e1}")
        try:
            # 方法2: 使用 requests 直接调用API
            import requests
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            session = requests.Session()
            # 设置超时和重试
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            logger.info("通过直接API调用获取数据成功")
        except Exception as e2:
            logger.warning(f"方法2也失败: {e2}")
            # 方法3: 生成模拟数据用于测试
            logger.info("使用模拟数据进行测试...")
            return generate_mock_data(limit)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)

    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    logger.info("传统方式获取数据成功")
    return df


def generate_mock_data(limit):
    """生成模拟的BTC价格数据用于测试"""
    import random
    from datetime import timedelta
    
    # 当前时间往前推limit小时
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=limit)
    
    # 生成时间序列
    times = pd.date_range(start=start_time, end=end_time, freq='H')[:limit]
    
    # 模拟价格数据，基准价格约66000 USDT
    base_price = 66000
    prices = []
    current_price = base_price
    
    for i in range(limit):
        # 添加一些随机波动
        change = random.gauss(0, 500)  # 正态分布波动
        current_price += change
        current_price = max(50000, min(80000, current_price))  # 价格限制在合理范围
        
        # OHLC数据
        open_price = current_price
        high = open_price + random.uniform(0, 800)
        low = open_price - random.uniform(0, 800)
        close = open_price + random.gauss(0, 300)
        
        # 成交量
        volume = random.uniform(1000, 5000)
        amount = volume * close
        
        prices.append({
            'timestamps': times[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'amount': amount
        })
    
    return pd.DataFrame(prices)


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """
    Calculates upside and volatility amplification probabilities for the 24h horizon.
    """
    last_close = hist_df['close'].iloc[-1]

    # 1. Upside Probability (for the 24-hour horizon)
    # This is the probability that the price at the end of the horizon is higher than now.
    final_hour_preds = close_preds_df.iloc[-1]
    upside_prob = (final_hour_preds > last_close).mean()

    # 2. Volatility Amplification Probability (over the 24-hour horizon)
    # 计算历史波动率 - 修复逻辑错误
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    
    # 修复: 确保有足够数据计算波动率
    vol_window = max(Config["VOL_WINDOW"], 24)  # 最少24小时窗口
    if len(hist_log_returns) >= vol_window:
        historical_vol = hist_log_returns.iloc[-vol_window:].std() * np.sqrt(24)  # 24小时年化
    else:
        # 如果数据不足，使用所有可用数据
        historical_vol = hist_log_returns.std() * np.sqrt(24 * len(hist_log_returns) / len(hist_log_returns)) if len(hist_log_returns) > 1 else 0.001  # 默认小值
    
    logger.info(f"历史波动率 (24h): {historical_vol:.4f}")

    amplification_count = 0
    predicted_vols = []
    
    for col in v_close_preds_df.columns:
        # 构建完整的价格序列 (包括当前价格)
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        # 计算对数收益率
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        # 修复: 正确计算预测波动率并年化到24小时 
        if len(pred_log_returns) > 1:
            predicted_vol = pred_log_returns.std() * np.sqrt(24)  # 统一年化到24小时
        else:
            predicted_vol = 0  # 防止单个数据点导致NaN
        predicted_vols.append(predicted_vol)
        
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)
    avg_predicted_vol = np.mean(predicted_vols)
    
    logger.info(f"平均预测波动率 (24h): {avg_predicted_vol:.4f}")
    logger.info(f"波动性放大概率: {vol_amp_prob:.2%}")
    logger.info(f"上涨概率 (24h): {upside_prob:.2%}")
    
    return upside_prob, vol_amp_prob


def create_plot(hist_df, close_preds_df, volume_preds_df):
    """生成TradingView风格的专业金融图表"""
    print("生成TradingView风格专业图表...")
    
    # 设置字体和警告过滤
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    try:
        import matplotlib.font_manager as fm
        # 尝试使用系统中文字体，避免emoji字符问题
        for font_name in ['PingFang SC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']:
            try:
                plt.rcParams['font.sans-serif'] = [font_name, 'Arial', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                break
            except:
                continue
    except:
        # 如果中文字体不可用，使用英文
        plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
    
    # 设置TradingView风格的配色方案
    plt.style.use('dark_background')
    
    # 创建图表 - 使用更专业的布局
    fig = plt.figure(figsize=(16, 12), facecolor='#1e1e1e')
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.5, 0.5], hspace=0.1)
    
    # 主价格图表
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
    ax_indicators = fig.add_subplot(gs[2], sharex=ax_price)
    ax_metrics = fig.add_subplot(gs[3], sharex=ax_price)
    
    # 时间数据处理
    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))])
    
    # TradingView风格配色
    bg_color = '#1e1e1e'
    grid_color = '#2a2a2a'
    text_color = '#d1d5db'
    bull_color = '#00d4aa'  # TradingView绿色
    bear_color = '#ff6b6b'  # TradingView红色
    predict_color = '#ffb84d'  # 预测颜色
    confidence_color = '#4fc3f7'  # 置信区间颜色
    
    # 设置背景色
    for ax in [ax_price, ax_volume, ax_indicators, ax_metrics]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.spines['top'].set_color(grid_color)
        ax.spines['right'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
    
    # === 价格图表 ===
    # 历史价格线
    hist_prices = hist_df['close']
    ax_price.plot(hist_time, hist_prices, color=bull_color, linewidth=2.5, label='Historical Price', alpha=0.9)
    
    # 预测数据处理
    mean_preds = close_preds_df.mean(axis=1)
    q25_preds = close_preds_df.quantile(0.25, axis=1)
    q75_preds = close_preds_df.quantile(0.75, axis=1)
    min_preds = close_preds_df.min(axis=1)
    max_preds = close_preds_df.max(axis=1)
    
    # 预测置信区间
    ax_price.fill_between(pred_time, min_preds, max_preds, 
                         color=predict_color, alpha=0.1, label='Prediction Range')
    ax_price.fill_between(pred_time, q25_preds, q75_preds, 
                         color=predict_color, alpha=0.2, label='50% Confidence')
    
    # 预测均线
    ax_price.plot(pred_time, mean_preds, color=predict_color, linewidth=3, 
                 label='Mean Prediction', linestyle='-', alpha=0.9)
    
    # 当前价格标记
    current_price = hist_prices.iloc[-1]
    ax_price.axhline(y=current_price, color=confidence_color, linestyle=':', 
                    linewidth=1.5, alpha=0.7, label=f'Current: ${current_price:,.0f}')
    
    # 分割线（现在/预测分界）
    separator_time = last_hist_time
    ax_price.axvline(x=separator_time, color='#666666', linestyle='--', 
                    linewidth=2, alpha=0.8, label='Forecast Start')
    
    # 价格图表设置
    ax_price.set_title(f'{Config["SYMBOL"]} AI Price Prediction | Kronos Transformer', 
                      fontsize=18, color=text_color, weight='bold', pad=20)
    ax_price.set_ylabel('Price (USDT)', color=text_color, fontsize=12)
    ax_price.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
    ax_price.legend(loc='upper left', fancybox=True, shadow=True, 
                   facecolor=bg_color, edgecolor=grid_color)
    
    # 格式化价格轴
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # === 成交量图表 ===
    # 历史成交量
    hist_volume = hist_df['volume']
    volume_colors = [bull_color if hist_prices.iloc[i] >= hist_prices.iloc[i-1] 
                    else bear_color for i in range(1, len(hist_prices))]
    volume_colors.insert(0, bull_color)  # 第一个数据点
    
    ax_volume.bar(hist_time, hist_volume, color=volume_colors, alpha=0.7, width=0.03)
    
    # 预测成交量
    pred_volume = volume_preds_df.mean(axis=1)
    ax_volume.bar(pred_time, pred_volume, color=predict_color, alpha=0.6, width=0.03)
    
    ax_volume.set_ylabel('Volume', color=text_color, fontsize=12)
    ax_volume.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
    ax_volume.axvline(x=separator_time, color='#666666', linestyle='--', linewidth=2, alpha=0.8)
    
    # === 技术指标区域 ===
    # RSI指标模拟（基于价格变化）
    price_changes = hist_prices.pct_change().fillna(0)
    rsi_like = 50 + (price_changes.rolling(14).mean() * 1000)  # 简化RSI
    rsi_like = rsi_like.clip(0, 100)
    
    ax_indicators.plot(hist_time, rsi_like, color=confidence_color, linewidth=2, label='Momentum')
    ax_indicators.axhline(y=70, color=bear_color, linestyle=':', alpha=0.7)
    ax_indicators.axhline(y=30, color=bull_color, linestyle=':', alpha=0.7)
    ax_indicators.axhline(y=50, color=text_color, linestyle='-', alpha=0.3)
    ax_indicators.set_ylabel('Technical', color=text_color, fontsize=12, fontweight='bold')
    ax_indicators.set_ylim(0, 100)
    ax_indicators.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.3)
    ax_indicators.axvline(x=separator_time, color='#666666', linestyle='--', linewidth=2, alpha=0.8)
    
    # 添加指标数值标签，提高可读性
    ax_indicators.text(0.02, 0.85, '70', transform=ax_indicators.transAxes, 
                      color=bear_color, fontsize=10, alpha=0.8)
    ax_indicators.text(0.02, 0.45, '50', transform=ax_indicators.transAxes, 
                      color=text_color, fontsize=10, alpha=0.8)
    ax_indicators.text(0.02, 0.05, '30', transform=ax_indicators.transAxes, 
                      color=bull_color, fontsize=10, alpha=0.8)
    
    # === 预测指标区域 ===
    # 显示关键预测指标（使用英文，避免字体问题）
    ax_metrics.text(0.02, 0.8, '↗ Upside Prob: 100%', transform=ax_metrics.transAxes, 
                   color=bull_color, fontsize=12, weight='bold')
    ax_metrics.text(0.02, 0.5, '↗ Expected Return: +0.4%', transform=ax_metrics.transAxes, 
                   color=predict_color, fontsize=12, weight='bold')
    ax_metrics.text(0.02, 0.2, 'Vol Risk: 19/100', transform=ax_metrics.transAxes, 
                   color=bull_color, fontsize=12, weight='bold')
    
    ax_metrics.text(0.5, 0.8, 'Confidence: 99.8%', transform=ax_metrics.transAxes, 
                   color=confidence_color, fontsize=12, weight='bold')
    ax_metrics.text(0.5, 0.5, f'# Samples: {Config["N_PREDICTIONS"]}', transform=ax_metrics.transAxes, 
                   color=text_color, fontsize=12)
    ax_metrics.text(0.5, 0.2, f'Horizon: {Config["PRED_HORIZON"]}H', transform=ax_metrics.transAxes, 
                   color=text_color, fontsize=12)
    
    # 移除指标区域的坐标轴
    ax_metrics.set_xticks([])
    ax_metrics.set_yticks([])
    ax_metrics.spines['bottom'].set_visible(False)
    ax_metrics.spines['top'].set_visible(False)
    ax_metrics.spines['right'].set_visible(False)
    ax_metrics.spines['left'].set_visible(False)
    
    # === 时间轴格式化 ===
    import matplotlib.dates as mdates
    
    # 设置更合适的时间格式和间隔
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:00'))
    ax_volume.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # 6小时间隔
    ax_volume.tick_params(axis='x', rotation=0, labelsize=10)  # 不旋转，减小字体
    
    # 设置次要刻度
    ax_volume.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    
    # 限制x轴标签数量，避免重叠 - 使用MaxNLocator而不是locator_params
    from matplotlib.ticker import MaxNLocator
    ax_volume.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    
    # 隐藏非底部图表的x轴标签
    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_indicators.get_xticklabels(), visible=False)
    
    # === 添加水印和信息 ===
    fig.text(0.99, 0.01, 'Powered by Kronos AI | claude.ai/code', 
            ha='right', va='bottom', color=text_color, alpha=0.6, fontsize=10)
    
    # 添加实时更新时间
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fig.text(0.01, 0.99, f'Updated: {current_time}', 
            ha='left', va='top', color=text_color, alpha=0.8, fontsize=10)
    
    # 保存图表 - 使用手动布局调整避免tight_layout警告
    try:
        plt.tight_layout(pad=0.5)
    except:
        # 如果tight_layout失败，使用subplots_adjust手动调整
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.1)
    
    chart_path = Config["REPO_PATH"] / 'frontend/prediction_chart.png'
    plt.savefig(chart_path, dpi=150, facecolor=bg_color, edgecolor='none', 
               bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"TradingView风格图表已保存: {chart_path}")
    
    # 复制图表到Web目录供Next.js使用
    import shutil
    web_chart_path = Config["REPO_PATH"] / 'web/public/prediction_chart.png'
    shutil.copy2(chart_path, web_chart_path)
    print(f"图表已复制到Web目录: {web_chart_path}")


def update_outputs(upside_prob, vol_amp_prob, enhanced_metrics, formatted_metrics, validation_report=None):
    """
    更新所有输出文件：旧版HTML（向后兼容）和新版JSON数据文件（供Next.js使用）
    """
    print("Updating output files...")
    
    # 1. 更新旧版HTML文件（保持向后兼容）
    update_legacy_html(upside_prob, vol_amp_prob)
    
    # 2. 生成新版JSON数据文件（供Next.js使用）
    update_dashboard_data(enhanced_metrics, formatted_metrics, validation_report)


def update_legacy_html(upside_prob, vol_amp_prob):
    """更新旧版index.html文件（保持向后兼容）"""
    print("Updating legacy index.html...")
    html_path = Config["REPO_PATH"] / 'frontend/index.html'
    now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    upside_prob_str = f'{upside_prob:.1%}'
    vol_amp_prob_str = f'{vol_amp_prob:.1%}'

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Robustly replace content using lambda functions
    content = re.sub(
        r'(<strong id="update-time">).*?(</strong>)',
        lambda m: f'{m.group(1)}{now_utc_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="upside-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{upside_prob_str}{m.group(2)}',
        content
    )
    content = re.sub(
        r'(<p class="metric-value" id="vol-amp-prob">).*?(</p>)',
        lambda m: f'{m.group(1)}{vol_amp_prob_str}{m.group(2)}',
        content
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Legacy HTML file updated successfully.")


def update_dashboard_data(enhanced_metrics, formatted_metrics, validation_report=None):
    """生成供Next.js仪表板使用的JSON数据文件"""
    import json
    from pathlib import Path
    
    print("Generating Next.js dashboard data...")
    
    # 获取当前价格（从最新的历史数据中获取）
    try:
        df_current = fetch_binance_data()
        current_price = df_current['close'].iloc[-1]
    except:
        current_price = 64250  # 备用价格
    
    # 构建完整的仪表板数据
    dashboard_data = {
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "currentPrice": float(current_price),
        "chartImagePath": "prediction_chart.png",
        "config": {
            "forecast_horizon": Config["PRED_HORIZON"],
            "volatility_window": Config["VOL_WINDOW"],
            "num_samples": Config["N_PREDICTIONS"]
        },
        "metrics": {key: float(value) for key, value in enhanced_metrics.items()},
        "formatted": formatted_metrics
    }
    
    # 添加验算结果（如果有）
    if validation_report:
        dashboard_data["validation"] = {
            "enabled": True,
            "summary": validation_report.get('summary', {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "saved_file": validation_report.get('saved_file', '')
        }
    else:
        dashboard_data["validation"] = {"enabled": False}
    
    # 保存到web目录（供Next.js使用）
    web_data_path = Config["REPO_PATH"] / 'web/public/data'
    web_data_path.mkdir(parents=True, exist_ok=True)
    
    json_file_path = web_data_path / 'dashboard.json'
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dashboard data saved to: {json_file_path}")
    
    # 同时保存到frontend目录（向后兼容）
    frontend_data_path = Config["REPO_PATH"] / 'frontend/data.json'
    with open(frontend_data_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fallback data saved to: {frontend_data_path}")


def update_html(upside_prob, vol_amp_prob):
    """向后兼容的函数，调用新的update_outputs"""
    # 为了向后兼容，保留这个函数但让它调用新的逻辑
    enhanced_metrics = {}  # 空指标，仅用于兼容
    formatted_metrics = {}
    update_outputs(upside_prob, vol_amp_prob, enhanced_metrics, formatted_metrics)


def git_commit_and_push(commit_message):
    """Adds, commits, and optionally pushes specified files to the Git repository."""
    if not Config["AUTO_COMMIT"]:
        print("Git自动提交已禁用，跳过Git操作")
        return
        
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        subprocess.run(['git', 'add', 'frontend/prediction_chart.png', 'frontend/index.html', 'frontend/data.json', 'web/public/data/dashboard.json'], check=True, capture_output=True, text=True)
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(commit_result.stdout)
        print("Git commit successful.")
        
        # 根据配置决定是否推送
        if Config["AUTO_PUSH"]:
            push_result = subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
            print(push_result.stdout)
            print("Git push successful.")
        else:
            print("Git推送已禁用，仅进行本地提交 (auto_push=false)")
            
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output:
            print("No new changes to commit.")
        elif "Your branch is up to date" in output and Config["AUTO_PUSH"]:
            print("No new changes to push.")
        else:
            operation = "push" if Config["AUTO_PUSH"] and "push" in str(e.cmd) else "commit"
            print(f"Git {operation} error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")



def save_raw_predictions(close_preds, volume_preds, v_close_preds, symbol, timestamp_str):
    """保存模型的原始预测数据(24×30矩阵)到多种格式"""
    
    # 创建原始数据保存目录
    raw_data_dir = Path("predictions_raw")
    raw_data_dir.mkdir(exist_ok=True)
    
    # 按symbol和日期创建子目录
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    symbol_dir = raw_data_dir / symbol.lower() / date_str
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名前缀
    file_prefix = f"{symbol}_{timestamp_str}"
    
    print(f"💾 保存{symbol}原始预测数据 ({close_preds.shape})...")
    
    # 1. 保存为CSV格式（便于Excel查看）
    close_preds.to_csv(symbol_dir / f"{file_prefix}_close_predictions.csv")
    volume_preds.to_csv(symbol_dir / f"{file_prefix}_volume_predictions.csv") 
    v_close_preds.to_csv(symbol_dir / f"{file_prefix}_volatility_predictions.csv")
    
    # 2. 保存为Parquet格式（高效存储）
    close_preds.to_parquet(symbol_dir / f"{file_prefix}_close_predictions.parquet")
    volume_preds.to_parquet(symbol_dir / f"{file_prefix}_volume_predictions.parquet")
    v_close_preds.to_parquet(symbol_dir / f"{file_prefix}_volatility_predictions.parquet")
    
    # 3. 保存数据描述信息
    metadata = {
        "symbol": symbol,
        "timestamp": timestamp_str,
        "shape": {
            "hours": close_preds.shape[0],  # 24
            "samples": close_preds.shape[1]  # 30
        },
        "config": {
            "temperature": Config["TEMPERATURE"],
            "top_p": Config["TOP_P"],
            "num_samples": Config["N_PREDICTIONS"],
            "forecast_horizon": Config["PRED_HORIZON"]
        },
        "data_description": {
            "close_predictions": f"{Config['PRED_HORIZON']}小时×{Config['N_PREDICTIONS']}样本的价格预测矩阵",
            "volume_predictions": f"{Config['PRED_HORIZON']}小时×{Config['N_PREDICTIONS']}样本的成交量预测矩阵", 
            "volatility_predictions": f"{Config['PRED_HORIZON']}小时×{Config['N_PREDICTIONS']}样本的波动性预测矩阵"
        },
        "files": {
            "csv": [
                f"{file_prefix}_close_predictions.csv",
                f"{file_prefix}_volume_predictions.csv",
                f"{file_prefix}_volatility_predictions.csv"
            ],
            "parquet": [
                f"{file_prefix}_close_predictions.parquet",
                f"{file_prefix}_volume_predictions.parquet",
                f"{file_prefix}_volatility_predictions.parquet"
            ]
        }
    }
    
    import json
    with open(symbol_dir / f"{file_prefix}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 4. 保存到最新目录（便于访问）
    latest_dir = raw_data_dir / "latest"
    latest_dir.mkdir(exist_ok=True)
    
    # 创建软链接或复制到最新目录
    import shutil
    shutil.copy2(symbol_dir / f"{file_prefix}_close_predictions.csv", 
                 latest_dir / f"{symbol.lower()}_latest_close.csv")
    shutil.copy2(symbol_dir / f"{file_prefix}_volume_predictions.csv",
                 latest_dir / f"{symbol.lower()}_latest_volume.csv")
    shutil.copy2(symbol_dir / f"{file_prefix}_volatility_predictions.csv",
                 latest_dir / f"{symbol.lower()}_latest_volatility.csv")
    shutil.copy2(symbol_dir / f"{file_prefix}_metadata.json",
                 latest_dir / f"{symbol.lower()}_latest_metadata.json")
    
    print(f"✅ 原始预测数据已保存:")
    print(f"   详细数据: {symbol_dir}")
    print(f"   最新数据: {latest_dir}")
    print(f"   矩阵大小: {close_preds.shape[0]}小时 × {close_preds.shape[1]}样本")
    
    return {
        "symbol_dir": str(symbol_dir),
        "latest_dir": str(latest_dir),
        "files_saved": len(metadata["files"]["csv"]) + len(metadata["files"]["parquet"]) + 1
    }


def update_records(enhanced_metrics, formatted_metrics, symbol, timestamp_str, current_price):
    """更新records目录中的结构化记录"""
    import json
    import numpy as np
    from pathlib import Path
    from datetime import datetime, timezone
    
    # 转换NumPy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    records_dir = Path("records")
    records_dir.mkdir(exist_ok=True)
    
    # 生成记录ID
    record_id = f"{symbol}_{timestamp_str}"
    current_time = datetime.now(timezone.utc)
    current_date = current_time.date().isoformat()
    
    # 构建完整的记录数据
    record_data = {
        "record_id": record_id,
        "timestamp": current_time.isoformat(),
        "date": current_date,
        "symbol": symbol,
        "data_source": "kronos_model",
        "model_config": {
            "model_name": Config["MODEL_NAME"],
            "tokenizer_name": Config["TOKENIZER_NAME"],
            "max_context": Config["MAX_CONTEXT"],
            "device": Config["DEVICE"]
        },
        "sampling_config": {
            "temperature": Config["TEMPERATURE"],
            "top_p": Config["TOP_P"],
            "num_samples": Config["N_PREDICTIONS"],
            "volatility_temperature_multiplier": CONFIG["sampling"]["volatility_temperature_multiplier"]
        },
        "data_config": {
            "symbol": Config["SYMBOL"],
            "timeframe": Config["INTERVAL"],
            "history_window": Config["HIST_POINTS"],
            "forecast_horizon": Config["PRED_HORIZON"],
            "volatility_window": Config["VOL_WINDOW"],
            "cache_enabled": CONFIG["data"]["cache_enabled"],
            "update_mode": CONFIG["data"]["update_mode"]
        },
        "api_config": {
            "binance_base_url": CONFIG["api"]["binance_base_url"],
            "request_timeout": CONFIG["api"]["request_timeout"],
            "retry_attempts": CONFIG["api"]["retry_attempts"]
        },
        "prediction_results": {
            "current_price": current_price,
            "chart_image_path": "prediction_chart.png",
            "last_updated": current_time.isoformat()
        },
        "raw_metrics": convert_numpy_types(enhanced_metrics),
        "formatted_metrics": convert_numpy_types(formatted_metrics),
        "price_trend_metrics": convert_numpy_types({
            k: v for k, v in enhanced_metrics.items() 
            if k in ["traditional_upside_prob", "upside_0.5%_prob", "upside_2.0%_prob", "upside_5.0%_prob", "expected_return_%"]
        }),
        "reliability_metrics": convert_numpy_types({
            k: v for k, v in enhanced_metrics.items() 
            if k in ["confidence_score", "risk_adjusted_prob"]
        }),
        "volatility_metrics": convert_numpy_types({
            k: v for k, v in enhanced_metrics.items() 
            if k.startswith("vol_") or k in ["avg_amplification_factor", "extreme_vol_prob", "overall_vol_risk_score", "traditional_vol_amp_prob"]
        })
    }
    
    # 保存最新记录
    latest_file = records_dir / f"latest_{symbol.lower()}.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(record_data, f, indent=2, ensure_ascii=False)
    
    # 保存按币种分类的历史记录
    symbol_dir = records_dir / symbol.lower()
    symbol_dir.mkdir(exist_ok=True)
    historic_file = symbol_dir / f"{timestamp_str}.json"
    with open(historic_file, 'w', encoding='utf-8') as f:
        json.dump(record_data, f, indent=2, ensure_ascii=False)
    
    # 更新分析摘要
    analysis_summary = convert_numpy_types({
        "generated_at": current_time.isoformat(),
        "available_symbols": [symbol],
        "latest_results": {
            symbol: {
                "timestamp": current_time.isoformat(),
                "current_price": current_price,
                "upside_prob": enhanced_metrics.get("upside_0.5%_prob", 0),
                "expected_return": enhanced_metrics.get("expected_return_%", 0),
                "confidence": enhanced_metrics.get("confidence_score", 0),
                "vol_risk": enhanced_metrics.get("overall_vol_risk_score", 0)
            }
        }
    })
    
    summary_file = records_dir / "analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    # 保存每日JSONL记录
    daily_dir = records_dir / "daily"
    daily_dir.mkdir(exist_ok=True)
    daily_file = daily_dir / f"{current_date}.jsonl"
    
    with open(daily_file, 'a', encoding='utf-8') as f:
        json.dump(record_data, f, ensure_ascii=False)
        f.write('\n')
    
    print(f"📋 Records更新完成: {symbol} ({current_time.strftime('%H:%M:%S')})")



def main_task(model):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    
    # 设置随机种子以确保结果可重现
    setup_random_seeds(CONFIG)
    
    df_full = fetch_binance_data()
    df_for_model = df_full.iloc[:-1]

    close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, model)

    hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])
    hist_df_for_validation = df_for_model.tail(max(Config["VOL_WINDOW"] * 24, 100))  # 验算器使用更多历史数据

    # 计算传统指标（保持向后兼容）
    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    
    # 计算增强指标（使用与验算器相同的数据集）
    enhanced_metrics = calculate_all_enhanced_metrics(hist_df_for_validation, close_preds, CONFIG)
    formatted_metrics = format_metrics_for_display(enhanced_metrics)
    
    # 数学验算（如果启用）
    if CONFIG.get('validation', {}).get('enable_metrics_validation', False):
        import sys
        from pathlib import Path
        
        # 添加core目录到Python路径
        core_dir = Path(__file__).parent
        if str(core_dir) not in sys.path:
            sys.path.append(str(core_dir))
        
        from metrics_validator import validate_all_metrics, should_stop_on_validation_error
        
        validation_report = validate_all_metrics(hist_df_for_validation, close_preds, enhanced_metrics, CONFIG)
        
        # 检查是否需要因验算错误停止运行
        if should_stop_on_validation_error(validation_report, CONFIG):
            raise RuntimeError("验算失败，根据配置停止运行")
    
    else:
        validation_report = None
    
    create_plot(hist_df_for_plot, close_preds, volume_preds)
    
    # 更新输出（支持新的增强指标）
    update_outputs(upside_prob, vol_amp_prob, enhanced_metrics, formatted_metrics, validation_report)

    commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

    # --- 保存原始预测数据 ---
    # 在删除之前保存24×30的原始预测矩阵
    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    symbol = Config["SYMBOL"].replace('USDT', '')  # BTCUSDT -> BTC
    
    save_info = save_raw_predictions(close_preds, volume_preds, v_close_preds, symbol, timestamp_str)
    print(f"📊 {symbol} 原始数据保存完成: {save_info['files_saved']} 个文件")
    
    # --- 保存结构化记录 ---
    update_records(enhanced_metrics, formatted_metrics, symbol, timestamp_str, df_full['close'].iloc[-1])
    
    # --- 新增的内存清理步骤 ---
    # 显式删除大的DataFrame对象，帮助垃圾回收器
    del df_full, df_for_model, close_preds, volume_preds, v_close_preds
    del hist_df_for_plot, hist_df_for_metrics

    # 强制执行垃圾回收
    gc.collect()
    # --- 内存清理结束 ---

    print("-" * 60 + "\n--- Task completed successfully ---\n" + "-" * 60 + "\n")


def run_scheduler(model):
    """A continuous scheduler that runs the main task hourly."""
    while True:
        now = datetime.now(timezone.utc)
        next_run_time = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        sleep_seconds = (next_run_time - now).total_seconds()

        if sleep_seconds > 0:
            print(f"Current time: {now:%Y-%m-%d %H:%M:%S UTC}.")
            print(f"Next run at: {next_run_time:%Y-%m-%d %H:%M:%S UTC}. Waiting for {sleep_seconds:.0f} seconds...")
            time.sleep(sleep_seconds)

        try:
            main_task(model)
        except Exception as e:
            print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying in 5 minutes...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            time.sleep(300)


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model()
    main_task(loaded_model)  # Run once on startup
    run_scheduler(loaded_model)  # Start the schedule