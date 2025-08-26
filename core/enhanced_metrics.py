import numpy as np
import pandas as pd
from typing import Dict, Union


def get_volatility_window(config: Dict) -> int:
    """
    根据配置动态计算波动率窗口大小
    volatility_window = forecast_horizon * volatility_window_multiplier
    """
    forecast_horizon = config.get('data', {}).get('forecast_horizon', 24)
    multiplier = config.get('data', {}).get('volatility_window_multiplier', 1.0)
    return int(forecast_horizon * multiplier)


def calculate_traditional_upside_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算传统的24小时上涨概率（任何上涨 > 0%）"""
    last_close = hist_df['close'].iloc[-1]
    
    final_hour_preds = close_preds_df.iloc[-1]
    upside_count = (final_hour_preds > last_close).sum()
    return upside_count / len(final_hour_preds)


def calculate_upside_05_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算上涨0.5%以上的概率"""
    last_close = hist_df['close'].iloc[-1]
    threshold_price = last_close * (1 + 0.5/100)
    
    final_hour_preds = close_preds_df.iloc[-1]
    upside_count = (final_hour_preds > threshold_price).sum()
    return upside_count / len(final_hour_preds)


def calculate_upside_2_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算上涨2.0%以上的概率"""
    last_close = hist_df['close'].iloc[-1]
    threshold_price = last_close * (1 + 2.0/100)
    
    final_hour_preds = close_preds_df.iloc[-1]
    upside_count = (final_hour_preds > threshold_price).sum()
    return upside_count / len(final_hour_preds)


def calculate_upside_5_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算上涨5.0%以上的概率 - 极端市场机会"""
    last_close = hist_df['close'].iloc[-1]
    threshold_price = last_close * (1 + 5.0/100)
    
    final_hour_preds = close_preds_df.iloc[-1]
    upside_count = (final_hour_preds > threshold_price).sum()
    return upside_count / len(final_hour_preds)


def calculate_expected_return(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算24小时期望收益率"""
    last_close = hist_df['close'].iloc[-1]
    final_hour_preds = close_preds_df.iloc[-1]
    
    returns = ((final_hour_preds / last_close - 1) * 100).values
    return np.mean(returns)


def calculate_confidence_score(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算预测置信度评分"""
    final_hour_preds = close_preds_df.iloc[-1]
    
    prediction_mean = final_hour_preds.mean()
    prediction_std = final_hour_preds.std()
    
    cv = prediction_std / (abs(prediction_mean) + 1e-8)
    confidence_score = 1 / (1 + cv)
    
    return confidence_score


def calculate_risk_adjusted_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> float:
    """计算风险调整后的上涨概率"""
    base_upside_prob = calculate_upside_05_prob(hist_df, close_preds_df)
    confidence_score = calculate_confidence_score(hist_df, close_preds_df)
    
    return base_upside_prob * confidence_score


def calculate_vol_amp_prob_24h(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, volatility_window: int = 24) -> float:
    """计算波动放大概率（基于动态窗口）"""
    last_close = hist_df['close'].iloc[-1]
    
    # 使用动态窗口计算历史波动率
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    
    # 确保有足够数据计算有效的波动率
    if len(hist_log_returns) >= max(volatility_window, 3):
        hist_vol = hist_log_returns.iloc[-volatility_window:].std()
    elif len(hist_log_returns) >= 3:
        hist_vol = hist_log_returns.std()
    else:
        hist_vol = 0.001  # 默认小值，防止除零
    
    amplification_count = 0
    
    for col in close_preds_df.columns:
        # 构建完整价格序列
        full_sequence = pd.concat([
            pd.Series([last_close]), 
            close_preds_df[col]
        ]).reset_index(drop=True)
        
        # 计算预测对数收益率和波动率
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        if len(pred_log_returns) > 0:
            predicted_vol = pred_log_returns.std()
        else:
            predicted_vol = 0
        
        # 判断是否放大
        if predicted_vol > hist_vol:
            amplification_count += 1
    
    return amplification_count / len(close_preds_df.columns)


def calculate_vol_amp_prob_48h(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, volatility_window: int = 24) -> float:
    """计算波动放大概率（使用2倍窗口）"""
    last_close = hist_df['close'].iloc[-1]
    
    # 使用2倍窗口计算历史波动率
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    window_48h = volatility_window * 2
    
    if len(hist_log_returns) >= max(window_48h, 3):
        hist_vol_48h = hist_log_returns.iloc[-window_48h:].std()
    elif len(hist_log_returns) >= 3:
        hist_vol_48h = hist_log_returns.std()
    else:
        hist_vol_48h = 0.001
    
    amplification_count = 0
    
    for col in close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        if len(pred_log_returns) > 0:
            predicted_vol = pred_log_returns.std()
        else:
            predicted_vol = 0
        
        if predicted_vol > hist_vol_48h:
            amplification_count += 1
    
    return amplification_count / len(close_preds_df.columns)


def calculate_avg_amplification_factor(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, volatility_window: int = 24) -> float:
    """计算平均放大倍数"""
    last_close = hist_df['close'].iloc[-1]
    
    # 使用动态窗口计算基准波动率
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    
    if len(hist_log_returns) >= max(volatility_window, 3):
        base_vol = hist_log_returns.iloc[-volatility_window:].std()
    elif len(hist_log_returns) >= 3:
        base_vol = hist_log_returns.std()
    else:
        base_vol = 0.001
    
    amplification_factors = []
    
    for col in close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        if len(pred_log_returns) > 0:
            predicted_vol = pred_log_returns.std()
        else:
            predicted_vol = 0
        
        # 计算放大倍数，避免除零
        amplification_factor = predicted_vol / (base_vol + 1e-8)
        amplification_factors.append(amplification_factor)
    
    return np.mean(amplification_factors)


def calculate_extreme_vol_prob(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, extreme_threshold: float = 2.0, volatility_window: int = 24) -> float:
    """计算极端波动概率（默认2倍阈值）"""
    last_close = hist_df['close'].iloc[-1]
    
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1)).dropna()
    
    # 使用动态窗口计算基准波动率
    if len(hist_log_returns) >= max(volatility_window, 3):
        base_vol = hist_log_returns.iloc[-volatility_window:].std()
    elif len(hist_log_returns) >= 3:
        base_vol = hist_log_returns.std()
    else:
        base_vol = 0.001
    
    extreme_count = 0
    
    for col in close_preds_df.columns:
        full_sequence = pd.concat([pd.Series([last_close]), close_preds_df[col]]).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1)).dropna()
        if len(pred_log_returns) > 0:
            predicted_vol = pred_log_returns.std()
        else:
            predicted_vol = 0
        
        # 判断是否为极端波动（放大2倍以上）
        amplification_factor = predicted_vol / (base_vol + 1e-8)
        if amplification_factor > extreme_threshold:
            extreme_count += 1
    
    return extreme_count / len(close_preds_df.columns)


def calculate_vol_persistence_score(close_preds_df: pd.DataFrame, last_close: float, window_size: int = 6) -> float:
    """计算波动持续性评分"""
    persistence_scores = []
    
    for col in close_preds_df.columns:
        path = close_preds_df[col].values
        full_path = np.concatenate([[last_close], path])
        
        # 计算滚动波动率（6小时窗口）
        rolling_vols = []
        for i in range(window_size, len(full_path)):
            window_data = full_path[i-window_size:i+1]
            window_returns = np.diff(np.log(window_data))
            rolling_vol = np.std(window_returns)
            rolling_vols.append(rolling_vol)
        
        # 计算波动率的自相关系数（持续性指标）
        if len(rolling_vols) > 1:
            # 计算滞后1期的自相关
            try:
                correlation_matrix = np.corrcoef(rolling_vols[:-1], rolling_vols[1:])
                persistence = correlation_matrix[0, 1]
                
                # 处理NaN值
                if not np.isnan(persistence):
                    persistence_scores.append(max(0, persistence))  # 确保非负
                else:
                    persistence_scores.append(0)
            except:
                persistence_scores.append(0)
        else:
            persistence_scores.append(0)
    
    return np.mean(persistence_scores)


def calculate_overall_vol_risk_score(vol_metrics: Dict[str, float]) -> float:
    """计算综合波动风险评分（0-100）"""
    
    # 各项指标权重
    weights = {
        'vol_amp_prob_24h': 0.25,        # 25% - 短期放大概率
        'avg_amplification_factor': 0.20, # 20% - 放大程度
        'extreme_vol_prob': 0.20,         # 20% - 极端风险
        'vol_persistence_score': 0.15,    # 15% - 持续性风险
        'vol_amp_prob_48h': 0.20         # 20% - 中期确认
    }
    
    # 标准化各项指标到0-100区间
    normalized_scores = {}
    
    # 24小时放大概率标准化
    normalized_scores['vol_amp_prob_24h'] = min(100, vol_metrics['vol_amp_prob_24h'] * 100)
    
    # 48小时放大概率标准化  
    normalized_scores['vol_amp_prob_48h'] = min(100, vol_metrics['vol_amp_prob_48h'] * 100)
    
    # 平均放大倍数标准化（以2倍为满分）
    normalized_scores['avg_amplification_factor'] = min(100, (vol_metrics['avg_amplification_factor'] / 2.0) * 100)
    
    # 极端波动概率标准化（以30%为满分）
    normalized_scores['extreme_vol_prob'] = min(100, (vol_metrics['extreme_vol_prob'] / 0.3) * 100)
    
    # 持续性评分标准化
    normalized_scores['vol_persistence_score'] = vol_metrics['vol_persistence_score'] * 100
    
    # 加权计算综合评分
    overall_score = sum(
        normalized_scores[key] * weights[key] 
        for key in weights.keys()
    )
    
    return round(overall_score, 0)


def calculate_enhanced_upside_metrics(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """计算完整的6项Upside Probability指标"""
    
    results = {}
    
    # 1. 基础盈利概率
    results['upside_0.5%_prob'] = calculate_upside_05_prob(hist_df, close_preds_df)
    
    # 2. 显著收益概率  
    results['upside_2.0%_prob'] = calculate_upside_2_prob(hist_df, close_preds_df)
    
    # 3. 极端机会识别
    results['upside_5.0%_prob'] = calculate_upside_5_prob(hist_df, close_preds_df)
    
    # 4. 期望收益率
    results['expected_return_%'] = calculate_expected_return(hist_df, close_preds_df)
    
    # 5. 预测置信度
    results['confidence_score'] = calculate_confidence_score(hist_df, close_preds_df)
    
    # 6. 风险调整概率
    results['risk_adjusted_prob'] = calculate_risk_adjusted_prob(hist_df, close_preds_df)
    
    return results


def calculate_enhanced_volatility_metrics(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, config: Dict = None) -> Dict[str, Union[float, int]]:
    """计算完整的6项波动率指标"""
    
    # 动态计算波动率窗口
    volatility_window = get_volatility_window(config or {})
    
    last_close = hist_df['close'].iloc[-1]
    
    # 计算各项基础指标
    results = {}
    
    # 1. 波动放大概率（单倍窗口）
    results['vol_amp_prob_24h'] = calculate_vol_amp_prob_24h(hist_df, close_preds_df, volatility_window)
    
    # 2. 波动放大概率（双倍窗口）
    results['vol_amp_prob_48h'] = calculate_vol_amp_prob_48h(hist_df, close_preds_df, volatility_window)
    
    # 3. 平均放大倍数
    results['avg_amplification_factor'] = calculate_avg_amplification_factor(hist_df, close_preds_df, volatility_window)
    
    # 4. 极端波动概率
    results['extreme_vol_prob'] = calculate_extreme_vol_prob(hist_df, close_preds_df, 2.0, volatility_window)
    
    # 5. 波动持续性评分
    results['vol_persistence_score'] = calculate_vol_persistence_score(close_preds_df, last_close)
    
    # 6. 综合波动风险评分
    results['overall_vol_risk_score'] = calculate_overall_vol_risk_score(results)
    
    return results


def calculate_all_enhanced_metrics(hist_df: pd.DataFrame, close_preds_df: pd.DataFrame, config: Dict = None) -> Dict[str, Union[float, int]]:
    """计算所有增强指标（12个指标 + 传统指标）"""
    
    # 动态计算波动率窗口
    volatility_window = get_volatility_window(config or {})
    
    # 获取传统指标
    traditional_upside_prob = calculate_traditional_upside_prob(hist_df, close_preds_df)
    traditional_vol_amp_prob = calculate_vol_amp_prob_24h(hist_df, close_preds_df, volatility_window)
    
    # 获取上涨概率指标
    upside_metrics = calculate_enhanced_upside_metrics(hist_df, close_preds_df)
    
    # 获取波动率指标
    volatility_metrics = calculate_enhanced_volatility_metrics(hist_df, close_preds_df, config)
    
    # 合并所有指标，包括传统指标
    all_metrics = {
        # 传统指标
        'traditional_upside_prob': traditional_upside_prob,
        'traditional_vol_amp_prob': traditional_vol_amp_prob,
        # 增强指标
        **upside_metrics, 
        **volatility_metrics
    }
    
    return all_metrics


def format_metrics_for_display(metrics: Dict[str, Union[float, int]]) -> Dict[str, str]:
    """格式化指标用于前端显示"""
    
    formatted = {}
    
    # 传统指标格式化
    if 'traditional_upside_prob' in metrics:
        formatted['traditional_upside_prob'] = f"{metrics['traditional_upside_prob']:.1%}"
    if 'traditional_vol_amp_prob' in metrics:
        formatted['traditional_vol_amp_prob'] = f"{metrics['traditional_vol_amp_prob']:.1%}"
    
    # 上涨概率指标格式化
    if 'upside_0.5%_prob' in metrics:
        formatted['upside_0.5%_prob'] = f"{metrics['upside_0.5%_prob']:.1%}"
    if 'upside_2.0%_prob' in metrics:
        formatted['upside_2.0%_prob'] = f"{metrics['upside_2.0%_prob']:.1%}"
    if 'upside_5.0%_prob' in metrics:
        formatted['upside_5.0%_prob'] = f"{metrics['upside_5.0%_prob']:.1%}"
    if 'expected_return_%' in metrics:
        formatted['expected_return_%'] = f"{metrics['expected_return_%']:.1f}%"
    if 'confidence_score' in metrics:
        formatted['confidence_score'] = f"{metrics['confidence_score']:.1%}"
    if 'risk_adjusted_prob' in metrics:
        formatted['risk_adjusted_prob'] = f"{metrics['risk_adjusted_prob']:.1%}"
    
    # 波动率指标格式化
    if 'vol_amp_prob_24h' in metrics:
        formatted['vol_amp_prob_24h'] = f"{metrics['vol_amp_prob_24h']:.1%}"
    if 'vol_amp_prob_48h' in metrics:
        formatted['vol_amp_prob_48h'] = f"{metrics['vol_amp_prob_48h']:.1%}"
    if 'avg_amplification_factor' in metrics:
        formatted['avg_amplification_factor'] = f"{metrics['avg_amplification_factor']:.1f}x"
    if 'extreme_vol_prob' in metrics:
        formatted['extreme_vol_prob'] = f"{metrics['extreme_vol_prob']:.1%}"
    if 'vol_persistence_score' in metrics:
        formatted['vol_persistence_score'] = f"{metrics['vol_persistence_score']:.1%}"
    if 'overall_vol_risk_score' in metrics:
        formatted['overall_vol_risk_score'] = f"{metrics['overall_vol_risk_score']:.0f}/100"
    
    return formatted