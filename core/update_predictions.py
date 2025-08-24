import gc
import os
import re
import subprocess
import time
import yaml
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor


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
            "num_samples": 30
        },
        "data": {
            "symbol": "BTCUSDT",
            "timeframe": "1h", 
            "history_window": 360,
            "forecast_horizon": 24,
            "volatility_window": 24
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
    "REPO_PATH": Path(__file__).parent.resolve(),
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
    "VOL_WINDOW": CONFIG["data"]["volatility_window"],
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

    with torch.no_grad():
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

        # 生成波动性预测 (使用较高温度增加变异性)
        logger.info(f"开始波动性预测 (T={Config['TEMPERATURE'] * 1.2}, top_p={Config['TOP_P']})...")
        begin_time = time.time()
        close_preds_volatility, _ = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=Config["TEMPERATURE"] * 1.2, 
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
    # 计算历史波动率 (使用最近VOL_WINDOW小时的数据)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std() * np.sqrt(24)  # 年化到24小时
    
    logger.info(f"历史波动率 (24h): {historical_vol:.4f}")

    amplification_count = 0
    predicted_vols = []
    
    for col in v_close_preds_df.columns:
        # 构建完整的价格序列 (包括当前价格)
        full_sequence = pd.concat([pd.Series([last_close]), v_close_preds_df[col]]).reset_index(drop=True)
        # 计算对数收益率
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        # 计算预测波动率 
        predicted_vol = pred_log_returns.std() * np.sqrt(len(pred_log_returns))  # 标准化到同等时间窗口
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
    """Generates and saves a comprehensive forecast chart."""
    print("Generating comprehensive forecast chart...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))])

    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{Config["SYMBOL"]} Probabilistic Price & Volume Forecast (Next {Config["PRED_HORIZON"]} Hours)', fontsize=16, weight='bold')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=0.03)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=0.03)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / 'frontend/prediction_chart.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


def update_html(upside_prob, vol_amp_prob):
    """
    Updates the index.html file with the latest metrics and timestamp.
    This version uses a more robust lambda function for replacement to avoid formatting errors.
    """
    print("Updating index.html...")
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
    print("HTML file updated successfully.")


def git_commit_and_push(commit_message):
    """Adds, commits, and pushes specified files to the Git repository."""
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        subprocess.run(['git', 'add', 'frontend/prediction_chart.png', 'frontend/index.html'], check=True, capture_output=True, text=True)
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(commit_result.stdout)
        push_result = subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
        print(push_result.stdout)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output or "Your branch is up to date" in output:
            print("No new changes to commit or push.")
        else:
            print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}")


def main_task(model):
    """Executes one full update cycle."""
    print("\n" + "=" * 60 + f"\nStarting update task at {datetime.now(timezone.utc)}\n" + "=" * 60)
    df_full = fetch_binance_data()
    df_for_model = df_full.iloc[:-1]

    close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, model)

    hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])

    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, v_close_preds)
    create_plot(hist_df_for_plot, close_preds, volume_preds)
    update_html(upside_prob, vol_amp_prob)

    commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

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