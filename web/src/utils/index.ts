import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function formatNumber(value: number, decimals: number = 1): string {
  return value.toFixed(decimals)
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value)
}

export function getMetricColor(value: number, type: 'upside' | 'volatility' | 'risk'): 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'indigo' {
  switch (type) {
    case 'upside':
      if (value >= 0.7) return 'green'
      if (value >= 0.5) return 'yellow'
      return 'red'
    
    case 'volatility':
      if (value >= 0.7) return 'red'
      if (value >= 0.4) return 'yellow'
      return 'green'
    
    case 'risk':
      if (value >= 80) return 'red'
      if (value >= 60) return 'yellow'
      if (value >= 40) return 'blue'
      return 'green'
    
    default:
      return 'blue'
  }
}

export function getTrendDirection(value: number, type: 'upside' | 'volatility'): 'up' | 'down' | 'neutral' {
  switch (type) {
    case 'upside':
      if (value >= 0.6) return 'up'
      if (value <= 0.4) return 'down'
      return 'neutral'
    
    case 'volatility':
      if (value >= 0.6) return 'up'
      if (value <= 0.3) return 'down'
      return 'neutral'
    
    default:
      return 'neutral'
  }
}

export function getMetricDescription(key: string): string {
  const descriptions: Record<string, string> = {
    'upside_0.5%_prob': '价格上涨至少0.5%的概率，覆盖交易成本的最低收益预期',
    'upside_2.0%_prob': '价格上涨至少2.0%的概率，值得主动操作的收益阈值',
    'upside_5.0%_prob': '价格上涨至少5.0%的概率，识别重大市场机会和异常波动',
    'expected_return_%': '所有预测样本的平均预期收益，量化投资回报预期',
    'confidence_score': '模型预测结果的一致性评分，评估预测可靠性',
    'risk_adjusted_prob': '综合考虑收益和波动风险的上涨概率，提供更稳健的投资决策依据',
    
    'vol_amp_prob_24h': '预测波动率相比过去24小时历史放大的概率',
    'vol_amp_prob_48h': '预测波动率相比过去48小时历史放大的概率，提供更稳定的中期波动基准',
    'avg_amplification_factor': '预测波动率相对历史基准的平均放大倍数，量化波动放大程度',
    'extreme_vol_prob': '预测出现2倍以上波动放大的概率，识别黑天鹅事件和市场极端情况',
    'vol_persistence_score': '预测24小时内波动率的持续程度和自相关性',
    'overall_vol_risk_score': '综合所有波动指标的0-100分风险评分，提供一站式波动风险评估'
  }
  
  return descriptions[key] || '暂无描述'
}

export function getMetricTitle(key: string): string {
  const titles: Record<string, string> = {
    'upside_0.5%_prob': '基础盈利概率',
    'upside_2.0%_prob': '显著收益概率',
    'upside_5.0%_prob': '极端机会识别',
    'expected_return_%': '期望收益率',
    'confidence_score': '预测置信度',
    'risk_adjusted_prob': '风险调整概率',
    
    'vol_amp_prob_24h': '24小时波动放大概率',
    'vol_amp_prob_48h': '48小时波动放大概率',
    'avg_amplification_factor': '平均放大倍数',
    'extreme_vol_prob': '极端波动概率',
    'vol_persistence_score': '波动持续性评分',
    'overall_vol_risk_score': '综合波动风险评分'
  }
  
  return titles[key] || key
}

export function formatDateTime(dateString: string): string {
  try {
    const date = new Date(dateString)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short'
    })
  } catch {
    return dateString
  }
}