import os
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client

# --- Configuration ---
REPO_PATH = "/path/to/your/local/clone/of/repo"  # IMPORTANT: Set this to your repo's absolute path
# Prediction Parameters
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
HIST_POINTS = 50  # Number of historical data points to display on the chart
PRED_HORIZON = 24  # How many hours into the future to forecast
N_PREDICTIONS = 100  # Number of Monte Carlo predictions to generate
VOL_WINDOW = 24  # Window size for calculating historical volatility


# --- 1. Your Prediction Model (Placeholder) ---
# !!! This is a critical placeholder !!!
# !!! You MUST replace this logic with your actual model !!!
def your_prediction_model(df, n_preds, pred_horizon):
    """
    A placeholder model to simulate N future H-step predictions.
    Your actual model should be loaded and executed here.

    Args:
        df (pd.DataFrame): Historical data with columns like 'open', 'high', 'low', 'close', 'volume'.
        n_preds (int): The number of predictions (N).
        pred_horizon (int): The prediction horizon (H).

    Returns:
        tuple: (predicted_close_df, predicted_volume_df)
               Two DataFrames, each with shape (pred_horizon, n_preds).
    """
    print(f"Running placeholder model for {n_preds} Monte Carlo predictions...")
    last_close = df['close'].iloc[-1]
    last_volume = df['volume'].iloc[-1]

    # Simulate price using a simplified Geometric Brownian Motion
    daily_vol = df['close'].pct_change().std() * np.sqrt(24)  # Estimate daily volatility
    hourly_vol = daily_vol / np.sqrt(24)
    close_preds = {}
    for i in range(n_preds):
        preds = [last_close]
        for _ in range(pred_horizon):
            drift = 0  # Assume zero drift
            random_shock = hourly_vol * np.random.normal()
            new_price = preds[-1] * (1 + drift + random_shock)
            preds.append(new_price)
        close_preds[f'pred-{i + 1}'] = preds[1:]  # Exclude starting point

    # Simulate volume as random fluctuations around the last volume
    volume_preds = {}
    for i in range(n_preds):
        preds = np.random.normal(loc=last_volume, scale=last_volume * 0.3, size=pred_horizon)
        preds[preds < 0] = 0  # Volume cannot be negative
        volume_preds[f'pred-{i + 1}'] = preds

    return pd.DataFrame(close_preds), pd.DataFrame(volume_preds)


# --- 2. Data Fetching ---
def fetch_binance_data(symbol, interval, limit):
    print(f"Fetching {symbol} {interval} K-line data from Binance...")
    client = Client()  # No API key needed for public data
    # Fetch extra data for model input and volatility calculation
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit + VOL_WINDOW)

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount'}, inplace=True)

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    print("Data fetched successfully.")
    return df


# --- 3. Metrics Calculation ---
def calculate_metrics(hist_df, close_preds_df):
    """Calculate upside probability and volatility amplification probability."""
    # a. Upside Probability
    last_close = hist_df['close'].iloc[-1]
    next_hour_preds = close_preds_df.iloc[0]  # Focus on the next hour
    upside_prob = (next_hour_preds > last_close).mean()

    # b. Volatility Amplification Probability
    # Calculate historical realized volatility (std of log returns)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-VOL_WINDOW:].std()

    amplification_count = 0
    for col in close_preds_df.columns:
        # Prepend last historical point to calculate returns for the forecast period
        full_sequence = pd.concat([pd.Series([last_close]), close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(close_preds_df.columns)

    print(f"Upside Probability: {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


# --- 4. Chart Generation ---
def create_plot(hist_df, close_preds_df, volume_preds_df):
    print("Generating comprehensive forecast chart...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Time axes
    hist_time = hist_df['open_time']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(hours=i + 1) for i in range(len(close_preds_df))])

    # --- Top Subplot: Close Price ---
    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price')
    mean_preds = close_preds_df.mean(axis=1)
    min_preds = close_preds_df.min(axis=1)
    max_preds = close_preds_df.max(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, min_preds, max_preds, color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{SYMBOL} Probabilistic Close Price Forecast', fontsize=16)
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()

    # --- Bottom Subplot: Volume ---
    ax2.bar(hist_time, hist_df['volume'], width=0.03, color='skyblue', label='Historical Volume')
    mean_vol_preds = volume_preds_df.mean(axis=1)
    ax2.plot(pred_time, mean_vol_preds, color='darkorange', linestyle='-', label='Mean Forecasted Volume')
    ax2.set_title(f'{SYMBOL} Volume Forecast', fontsize=14)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()

    # Add a separator line to both plots
    separator_time = hist_time.iloc[-1] + timedelta(minutes=30)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5)
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = os.path.join(REPO_PATH, 'prediction_chart.png')
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


# --- 5. HTML Update ---
def update_html(upside_prob, vol_amp_prob):
    print("Updating index.html...")
    html_path = os.path.join(REPO_PATH, 'index.html')
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use regex to replace content based on element IDs
    content = re.sub(r'(<span id="update-time">).*?(</span>)', r'\1' + now_utc + r'\2', content)
    content = re.sub(r'(<p class="metric-value" id="upside-prob">).*?(</p>)', r'\1' + f'{upside_prob:.1%}' + r'\2', content)
    content = re.sub(r'(<p class="metric-value" id="vol-amp-prob">).*?(</p>)', r'\1' + f'{vol_amp_prob:.1%}' + r'\2', content)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML file updated successfully.")


# --- 6. Git Operations ---
def git_commit_and_push(commit_message):
    print("Performing Git operations...")
    try:
        os.chdir(REPO_PATH)
        subprocess.run(['git', 'add', 'prediction_chart.png', 'index.html'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
        subprocess.run(['git', 'push'], check=True, capture_output=True)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode() if e.stdout else e.stderr.decode()
        if "nothing to commit" in output:
            print("No changes to commit.")
        else:
            print(f"A Git error occurred: {output}")


# --- 7. Main Task & Scheduler ---
def main_task():
    """Executes one full update cycle."""
    print("\n" + "=" * 50)
    print(f"Starting update task at {datetime.now(timezone.utc)}")

    # Fetch data, ensuring we have enough for model input and volatility calculations.
    # The model should be trained on data *before* the last, unclosed candle.
    df_full = fetch_binance_data(SYMBOL, INTERVAL, limit=HIST_POINTS + VOL_WINDOW)
    df_for_model = df_full.iloc[:-1]

    # Run the prediction model
    close_preds, volume_preds = your_prediction_model(df_for_model, N_PREDICTIONS, PRED_HORIZON)

    # Prepare historical data for plotting and metrics
    hist_df_for_plot = df_for_model.tail(HIST_POINTS)

    # Calculate metrics
    upside_prob, vol_amp_prob = calculate_metrics(df_for_model.tail(VOL_WINDOW), close_preds)

    # Generate and save the chart
    create_plot(hist_df_for_plot, close_preds, volume_preds)

    # Update the webpage content
    update_html(upside_prob, vol_amp_prob)

    # Commit and push changes to GitHub
    commit_message = f"Update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    git_commit_and_push(commit_message)

    print("--- Task completed ---")
    print("=" * 50 + "\n")


def run_scheduler():
    """A continuous scheduler that runs the task every hour."""
    while True:
        now = datetime.now(timezone.utc)
        # Calculate the next run time: 5 seconds past the next full hour
        next_run_time = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)

        sleep_seconds = (next_run_time - now).total_seconds()

        if sleep_seconds > 0:
            print(f"Current time: {now:%Y-%m-%d %H:%M:%S UTC}. Next run at: {next_run_time:%Y-%m-%d %H:%M:%S UTC}. Waiting for {sleep_seconds:.0f} seconds...")
            time.sleep(sleep_seconds)

        try:
            main_task()
        except Exception as e:
            print(f"A critical error occurred: {e}")
            print("Retrying in 5 minutes...")
            time.sleep(300)


if __name__ == '__main__':
    # Run the task once immediately on startup
    main_task()
    # Then, start the hourly scheduling loop
    run_scheduler()
    