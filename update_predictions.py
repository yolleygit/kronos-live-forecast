import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
import time
import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import numpy as np

# --- 配置区 ---
REPO_PATH = "./"
# 预测参数
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
HIST_POINTS = 360  # 图表上显示的历史数据点数量
PRED_HORIZON = 24  # 预测未来多少个小时
N_PREDICTIONS = 10  # 蒙特卡洛预测次数
VOL_WINDOW = 24  # 计算历史波动率的窗口大小


# --- 1. 定义你的预测模型 (占位符) ---
# !!! 这是一个关键的占位符 !!!
# !!! 你需要用你自己的真实模型替换这部分逻辑 !!!
def your_prediction_model(df, n_preds, pred_horizon):
    """
    一个占位符模型，模拟生成N次、未来H步的预测结果。
    真实模型应在此处加载并执行。

    Args:
        df (pd.DataFrame): 包含'open', 'high', 'low', 'close', 'volume'等的历史数据.
        n_preds (int): 预测次数 N.
        pred_horizon (int): 预测步长 H.

    Returns:
        tuple: (predicted_close_df, predicted_volume_df)
               两个DataFrame，每个的形状都是 (pred_horizon, n_preds).
    """
    print(f"使用占位符模型进行 {n_preds} 次蒙特卡洛预测...")
    last_close = df['close'].iloc[-1]
    last_volume = df['volume'].iloc[-1]

    # 模拟价格的随机游走
    close_preds = {}
    # 使用几何布朗运动的简化形式
    daily_vol = df['close'].pct_change().std() * np.sqrt(24)  # 估算日波动率
    hourly_vol = daily_vol / np.sqrt(24)

    for i in range(n_preds):
        preds = [last_close]
        for _ in range(pred_horizon):
            drift = 0  # 假设无漂移
            random_shock = hourly_vol * np.random.normal()
            new_price = preds[-1] * (1 + drift + random_shock)
            preds.append(new_price)
        close_preds[f'pred-{i + 1}'] = preds[1:]  # 去掉起始点

    # 模拟交易量的随机变化
    volume_preds = {}
    for i in range(n_preds):
        # 简单地在最后一个交易量附近随机波动
        preds = np.random.normal(loc=last_volume, scale=last_volume * 0.3, size=pred_horizon)
        preds[preds < 0] = 0  # 交易量不能为负
        volume_preds[f'pred-{i + 1}'] = preds

    return pd.DataFrame(close_preds), pd.DataFrame(volume_preds)


# --- 2. 获取数据 ---
def fetch_binance_data(symbol, interval, limit):
    print(f"正在从Binance获取 {symbol} {interval} K线数据...")
    client = Client()  # 公开数据无需API Key
    # 获取 limit + VOL_WINDOW 条数据，以确保有足够的数据计算波动率
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit + VOL_WINDOW)

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'amount', ...]
    df = pd.DataFrame(klines, columns=columns[:len(klines[0])])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    print("数据获取成功.")
    return df


# --- 3. 计算额外指标 ---
def calculate_metrics(hist_df, close_preds_df):
    """计算上涨概率和波动率放大概率"""
    # a. 上涨概率
    last_close = hist_df['close'].iloc[-1]
    # 我们只关心下一小时
    next_hour_preds = close_preds_df.iloc[0]
    upside_prob = (next_hour_preds > last_close).mean()

    # b. 波动率放大概率
    # 计算历史波动率 (已实现波动率: log returns的标准差)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-VOL_WINDOW:].std()

    amplification_count = 0
    for col in close_preds_df.columns:
        # 将预测序列与最后一个历史点连接，以计算回报率
        full_sequence = pd.concat([pd.Series([last_close]), close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(close_preds_df.columns)

    print(f"上涨概率: {upside_prob:.2%}, 波动率放大概率: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


# --- 4. 生成图表 ---
def create_plot(hist_df, close_preds_df, volume_preds_df):
    print("正在生成综合图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 历史数据的时间轴
    hist_time = hist_df['open_time']
    # 预测数据的时间轴
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))])

    # --- 上子图: Close Price ---
    # 绘制历史价格
    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price')

    # 计算预测的均值和范围
    mean_preds = close_preds_df.mean(axis=1)
    min_preds = close_preds_df.min(axis=1)
    max_preds = close_preds_df.max(axis=1)

    # 绘制预测均值线
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Prediction')
    # 绘制预测范围阴影
    ax1.fill_between(pred_time, min_preds, max_preds, color='darkorange', alpha=0.2, label='Prediction Range (Min-Max)')

    ax1.set_title(f'{SYMBOL} Close Price Probabilistic Prediction', fontsize=16)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()

    # --- 下子图: Volume ---
    # 绘制历史交易量
    ax2.bar(hist_time, hist_df['volume'], width=0.03, color='skyblue', label='Historical Volume')
    # 绘制预测交易量均值
    mean_vol_preds = volume_preds_df.mean(axis=1)
    ax2.plot(pred_time, mean_vol_preds, color='darkorange', linestyle='-', label='Mean Predicted Volume')

    ax2.set_title(f'{SYMBOL} Volume Prediction', fontsize=16)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()

    # 在两个子图上都绘制分割线
    separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='Prediction Start')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = os.path.join(REPO_PATH, 'prediction_chart.png')
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"图表已保存到: {chart_path}")


# --- 5. 更新HTML文件 ---
def update_html(upside_prob, vol_amp_prob):
    print("正在更新index.html...")
    html_path = os.path.join(REPO_PATH, 'index.html')
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换时间戳
    content = content.replace(
        re.search(r'<span id="update-time">.*</span>', content).group(0),
        f'<span id="update-time">{now_utc}</span>'
    )
    # 替换上涨概率
    content = content.replace(
        re.search(r'<p class="metric-value" id="upside-prob">.*</p>', content).group(0),
        f'<p class="metric-value" id="upside-prob">{upside_prob:.1%}</p>'
    )
    # 替换波动率放大概率
    content = content.replace(
        re.search(r'<p class="metric-value" id="vol-amp-prob">.*</p>', content).group(0),
        f'<p class="metric-value" id="vol-amp-prob">{vol_amp_prob:.1%}</p>'
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML文件更新完成.")


# --- 6. Git操作 ---
def git_commit_and_push(commit_message):
    print("正在执行Git操作...")
    try:
        os.chdir(REPO_PATH)
        subprocess.run(['git', 'add', 'prediction_chart.png', 'index.html'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        print("Git Push: 成功推送到远程仓库.")
    except subprocess.CalledProcessError as e:
        if "nothing to commit" in e.stdout.decode() or "nothing to commit" in e.stderr.decode():
            print("没有检测到更改，无需提交。")
        else:
            print(f"Git操作失败: {e.stderr.decode()}")


# --- 7. 主任务和调度器 ---
def main_task():
    """执行一次完整的更新流程"""
    print("\n" + "=" * 50)
    print(f"开始执行更新任务于 {datetime.now(timezone.utc)}")

    # 获取数据（获取比绘图需要更多的数据，用于模型输入和波动率计算）
    # 我们只需要最后未收盘的K线之前的数据来做预测
    df_full = fetch_binance_data(SYMBOL, INTERVAL, limit=HIST_POINTS + VOL_WINDOW)
    df_for_model = df_full.iloc[:-1]  # 传递给模型的数据不包含当前未走完的K线

    # 运行预测模型
    close_preds, volume_preds = your_prediction_model(df_for_model, N_PREDICTIONS, PRED_HORIZON)

    # 准备用于绘图和指标计算的历史数据
    hist_df_for_plot = df_for_model.tail(HIST_POINTS)

    # 计算指标
    upside_prob, vol_amp_prob = calculate_metrics(df_for_model.tail(VOL_WINDOW), close_preds)

    # 生成图表
    create_plot(hist_df_for_plot, close_preds, volume_preds)

    # 更新网页
    update_html(upside_prob, vol_amp_prob)

    # Git提交与推送
    commit_message = f"Update probabilistic prediction for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

    print("--- 任务完成 ---")
    print("=" * 50 + "\n")


def run_scheduler():
    """持续运行的调度器，在每小时的整点过5秒后执行任务"""
    while True:
        now = datetime.now(timezone.utc)
        # 计算下一个整点
        next_run_time = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)

        sleep_seconds = (next_run_time - now).total_seconds()

        if sleep_seconds > 0:
            print(f"当前时间: {now:%Y-%m-%d %H:%M:%S}. 下次运行时间: {next_run_time:%Y-%m-%d %H:%M:%S}. 等待 {sleep_seconds:.0f} 秒...")
            time.sleep(sleep_seconds)

        try:
            main_task()
        except Exception as e:
            print(f"发生严重错误: {e}")
            print("将在5分钟后重试...")
            time.sleep(300)  # 如果出错，等待5分钟再试


if __name__ == '__main__':
    # 首次启动时立即执行一次
    main_task()
    # 然后进入定时调度循环
    run_scheduler()
