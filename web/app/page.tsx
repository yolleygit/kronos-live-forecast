'use client'

import React, { useState, useEffect } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  DollarSign, 
  Shield, 
  BarChart3,
  Activity,
  Zap,
  AlertTriangle,
  Timer,
  Gauge,
  Bitcoin,
  Coins
} from 'lucide-react'

import Header from '@/components/Header'
import MetricGroup from '@/components/MetricGroup'
import ChartSection from '@/components/ChartSection'
import { DashboardData, MetricCardProps } from '@/types'
import { 
  getMetricColor, 
  getTrendDirection, 
  getMetricDescription, 
  getMetricTitle,
  formatPercentage,
  formatNumber
} from '@/utils'

// 定义支持的交易对
type Symbol = 'BTC' | 'ETH'

interface SymbolConfig {
  name: string
  symbol: string
  pair: string
  dataEndpoint: string
  icon: React.ReactNode
  color: string
}

export default function Dashboard() {
  const [activeSymbol, setActiveSymbol] = useState<Symbol>('BTC')
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 交易对配置
  const symbolConfigs: Record<Symbol, SymbolConfig> = {
    'BTC': {
      name: 'Bitcoin',
      symbol: 'BTC',
      pair: 'BTCUSDT',
      dataEndpoint: '/data/dashboard.json',
      icon: <Bitcoin className="w-5 h-5" />,
      color: '#f7931a'
    },
    'ETH': {
      name: 'Ethereum',
      symbol: 'ETH',
      pair: 'ETHUSDT',
      dataEndpoint: '/data/eth_dashboard.json',
      icon: <Coins className="w-5 h-5" />,
      color: '#627eea'
    }
  }

  // Mock data for demonstration - in production this would come from API
  const mockData: DashboardData = {
    lastUpdated: '2025-08-24T15:30:00Z',
    currentPrice: 64250,
    chartImagePath: '/prediction_chart.png', // This should point to the generated chart
    metrics: {
      'upside_0.5%_prob': 0.73,
      'upside_2.0%_prob': 0.45,
      'upside_5.0%_prob': 0.12,
      'expected_return_%': 1.2,
      'confidence_score': 0.82,
      'risk_adjusted_prob': 0.68,
      'vol_amp_prob_24h': 0.35,
      'vol_amp_prob_48h': 0.28,
      'avg_amplification_factor': 1.4,
      'extreme_vol_prob': 0.08,
      'vol_persistence_score': 0.65,
      'overall_vol_risk_score': 42,
      'traditional_upside_prob': 0.85,
      'traditional_vol_amp_prob': 0.35
    },
    formatted: {
      'upside_0.5%_prob': '73.0%',
      'upside_2.0%_prob': '45.0%',
      'upside_5.0%_prob': '12.0%',
      'expected_return_%': '1.2%',
      'confidence_score': '82.0%',
      'risk_adjusted_prob': '68.0%',
      'vol_amp_prob_24h': '35.0%',
      'vol_amp_prob_48h': '28.0%',
      'avg_amplification_factor': '1.4x',
      'extreme_vol_prob': '8.0%',
      'vol_persistence_score': '65.0%',
      'overall_vol_risk_score': '42/100',
      'traditional_upside_prob': '85.0%',
      'traditional_vol_amp_prob': '35.0%'
    }
  }

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const config = symbolConfigs[activeSymbol]
        
        // Try to fetch data from API first
        try {
          const response = await fetch(`/api/dashboard?symbol=${activeSymbol.toLowerCase()}`)
          if (response.ok) {
            const dashboardData = await response.json()
            setData(dashboardData)
            setError(null)
            return
          }
        } catch (apiError) {
          console.warn('API fetch failed, trying direct file access:', apiError)
        }
        
        // Fallback: try to fetch data file directly based on active symbol
        try {
          const response = await fetch(config.dataEndpoint)
          if (response.ok) {
            const dashboardData = await response.json()
            setData(dashboardData)
            setError(null)
            return
          }
        } catch (fileError) {
          console.warn('Direct file fetch failed:', fileError)
        }
        
        // Final fallback: use mock data
        console.log('Using mock data as fallback')
        setData(mockData)
        setError(null)
        
      } catch (err) {
        console.error('Error loading data:', err)
        setError('Failed to load dashboard data')
        // Even if there's an error, provide mock data for demonstration
        setData(mockData)
      } finally {
        setLoading(false)
      }
    }

    loadData()
    
    // Set up auto-refresh every 5 minutes
    const interval = setInterval(loadData, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [activeSymbol])

  const handleRefresh = () => {
    // Manually trigger data refresh
    window.location.reload()
  }

  // Generate price trend metrics (all upside probability thresholds)
  const generatePriceTrendMetrics = (data: DashboardData): MetricCardProps[] => {
    const forecastHours = data.config?.forecast_horizon || 24;
    return [
    {
      title: '任何上涨概率',
      value: data.formatted['traditional_upside_prob'] || '100.0%',
      description: `价格在未来${forecastHours}小时内出现任何程度上涨（>0%）的概率，最基础的趋势判断指标`,
      trend: getTrendDirection(data.metrics['traditional_upside_prob'] || 1.0, 'upside'),
      color: getMetricColor(data.metrics['traditional_upside_prob'] || 1.0, 'upside'),
      icon: <TrendingUp className="w-6 h-6" />
    },
    {
      title: getMetricTitle('upside_0.5%_prob'),
      value: data.formatted['upside_0.5%_prob'],
      description: '价格上涨0.5%以上的概率，覆盖基本交易成本的最低盈利阈值',
      trend: getTrendDirection(data.metrics['upside_0.5%_prob'], 'upside'),
      color: getMetricColor(data.metrics['upside_0.5%_prob'], 'upside'),
      icon: <Target className="w-5 h-5" />
    },
    {
      title: getMetricTitle('upside_2.0%_prob'),
      value: data.formatted['upside_2.0%_prob'],
      description: '价格上涨2.0%以上的概率，值得主动操作的显著收益阈值',
      trend: getTrendDirection(data.metrics['upside_2.0%_prob'], 'upside'),
      color: getMetricColor(data.metrics['upside_2.0%_prob'], 'upside'),
      icon: <TrendingUp className="w-5 h-5" />
    },
    {
      title: getMetricTitle('upside_5.0%_prob'),
      value: data.formatted['upside_5.0%_prob'],
      description: '价格上涨5.0%以上的概率，识别重大市场机会和极端收益可能',
      trend: getTrendDirection(data.metrics['upside_5.0%_prob'], 'upside'),
      color: getMetricColor(data.metrics['upside_5.0%_prob'], 'upside'),
      icon: <Zap className="w-5 h-5" />
    }
  ];
  }

  // Generate prediction reliability metrics
  const generateReliabilityMetrics = (data: DashboardData): MetricCardProps[] => [
    {
      title: getMetricTitle('expected_return_%'),
      value: data.formatted['expected_return_%'],
      description: '所有预测样本的平均期望收益率，量化投资回报预期的数值参考',
      trend: data.metrics['expected_return_%'] > 0 ? 'up' : data.metrics['expected_return_%'] < 0 ? 'down' : 'neutral',
      color: data.metrics['expected_return_%'] > 1 ? 'green' : data.metrics['expected_return_%'] > 0 ? 'blue' : 'red',
      icon: <DollarSign className="w-5 h-5" />
    },
    {
      title: getMetricTitle('confidence_score'),
      value: data.formatted['confidence_score'],
      description: '模型预测结果的一致性评分，评估预测可靠性和稳定性程度',
      trend: getTrendDirection(data.metrics['confidence_score'], 'upside'),
      color: getMetricColor(data.metrics['confidence_score'], 'upside'),
      icon: <Shield className="w-5 h-5" />
    },
    {
      title: getMetricTitle('risk_adjusted_prob'),
      value: data.formatted['risk_adjusted_prob'],
      description: '综合考虑收益和波动风险的上涨概率，提供更稳健的投资决策依据',
      trend: getTrendDirection(data.metrics['risk_adjusted_prob'], 'upside'),
      color: getMetricColor(data.metrics['risk_adjusted_prob'], 'upside'),
      icon: <BarChart3 className="w-5 h-5" />
    }
  ]


  // Generate volatility risk metrics
  const generateVolatilityMetrics = (data: DashboardData): MetricCardProps[] => {
    const volatilityWindow = data.config?.volatility_window || 24;
    const doubleWindow = volatilityWindow * 2;
    return [
    {
      title: '波动放大概率',
      value: data.formatted['traditional_vol_amp_prob'] || data.formatted['vol_amp_prob_24h'] || '3.3%',
      description: `预测波动率相比过去${volatilityWindow}小时历史水平放大的概率，基础波动风险指标`,
      trend: getTrendDirection(data.metrics['traditional_vol_amp_prob'] || data.metrics['vol_amp_prob_24h'] || 0.033, 'volatility'),
      color: getMetricColor(data.metrics['traditional_vol_amp_prob'] || data.metrics['vol_amp_prob_24h'] || 0.033, 'volatility'),
      icon: <Activity className="w-6 h-6" />
    },
    {
      title: '中期波动确认',
      value: data.formatted['vol_amp_prob_48h'],
      description: `基于${doubleWindow}小时更稳定基准的波动放大概率，提供中期波动风险确认`,
      trend: getTrendDirection(data.metrics['vol_amp_prob_48h'], 'volatility'),
      color: getMetricColor(data.metrics['vol_amp_prob_48h'], 'volatility'),
      icon: <Timer className="w-5 h-5" />
    },
    {
      title: getMetricTitle('avg_amplification_factor'),
      value: data.formatted['avg_amplification_factor'],
      description: '预测波动率相对历史基准的平均放大倍数，量化波动放大程度',
      trend: data.metrics['avg_amplification_factor'] > 1.5 ? 'up' : data.metrics['avg_amplification_factor'] < 1.2 ? 'down' : 'neutral',
      color: data.metrics['avg_amplification_factor'] > 2 ? 'red' : data.metrics['avg_amplification_factor'] > 1.5 ? 'yellow' : 'green',
      icon: <TrendingUp className="w-5 h-5" />
    },
    {
      title: getMetricTitle('extreme_vol_prob'),
      value: data.formatted['extreme_vol_prob'],
      description: '预测出现2倍以上波动放大的概率，识别黑天鹅事件和市场极端情况',
      trend: getTrendDirection(data.metrics['extreme_vol_prob'], 'volatility'),
      color: getMetricColor(data.metrics['extreme_vol_prob'], 'volatility'),
      icon: <AlertTriangle className="w-5 h-5" />
    },
    {
      title: getMetricTitle('vol_persistence_score'),
      value: data.formatted['vol_persistence_score'],
      description: `预测${volatilityWindow}小时内波动率的持续程度和自相关性，评估波动延续性`,
      trend: getTrendDirection(data.metrics['vol_persistence_score'], 'volatility'),
      color: getMetricColor(data.metrics['vol_persistence_score'], 'volatility'),
      icon: <BarChart3 className="w-5 h-5" />
    },
    {
      title: '综合风险评分',
      value: data.formatted['overall_vol_risk_score'],
      description: '综合所有波动指标的0-100分风险评分，提供一站式波动风险评估',
      trend: data.metrics['overall_vol_risk_score'] > 60 ? 'up' : data.metrics['overall_vol_risk_score'] < 40 ? 'down' : 'neutral',
      color: getMetricColor(data.metrics['overall_vol_risk_score'], 'risk'),
      icon: <Gauge className="w-5 h-5" />
    }
  ];
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-slate-600 dark:text-slate-400">加载预测数据中...</p>
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto" />
          <p className="text-red-600">{error || '数据加载失败'}</p>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            重试
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <Header
        lastUpdated={data.lastUpdated}
        currentPrice={data.currentPrice}
        isLoading={loading}
        onRefresh={handleRefresh}
        forecastHorizon={data.config?.forecast_horizon || 24}
        symbol={activeSymbol}
        symbolPair={symbolConfigs[activeSymbol].pair}
      />

      {/* Symbol Tabs */}
      <div className="max-w-7xl mx-auto px-6 pt-8">
        <div className="flex flex-wrap gap-2 p-1 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-200/50 dark:border-slate-700/50">
          {Object.entries(symbolConfigs).map(([key, config]) => {
            const isActive = activeSymbol === key
            return (
              <button
                key={key}
                onClick={() => setActiveSymbol(key as Symbol)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm border border-slate-200 dark:border-slate-600'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-slate-700/50'
                }`}
              >
                <div className={`flex items-center justify-center ${isActive ? '' : 'opacity-70'}`}>
                  {config.icon}
                </div>
                <span className="text-sm">
                  {config.name}
                </span>
                {isActive && (
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: config.color }}
                  />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-16">
        {/* Chart Section */}
        <ChartSection 
          chartImagePath={data.chartImagePath}
          forecastHorizon={data.config?.forecast_horizon || 24}
          numSamples={data.config?.num_samples || 30}
          currentPrice={data.currentPrice}
          dashboardData={data}
          symbol={activeSymbol.toLowerCase()}
        />

        {/* Price Trend Analysis */}
        <MetricGroup
          title="价格趋势预测"
          description="不同收益阈值下的上涨概率分析，从任何上涨到极端收益的完整概率分布"
          metrics={generatePriceTrendMetrics(data)}
        />

        {/* Prediction Reliability */}
        <MetricGroup
          title="预测可靠性评估"
          description="评估模型预测结果的置信度和风险调整后的实际投资参考价值"
          metrics={generateReliabilityMetrics(data)}
        />

        {/* Volatility Risk Analysis */}
        <MetricGroup
          title="市场波动风险"
          description="预测未来24-48小时的市场波动特征，识别价格波动放大风险和持续性模式"
          metrics={generateVolatilityMetrics(data)}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-700 py-8">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-slate-600 dark:text-slate-400">
          <p>
            Powered by Kronos Foundation Model | Data Source: Binance API | 
            Last Updated: {new Date(data.lastUpdated).toLocaleString('zh-CN')}
          </p>
          <p className="mt-2">
            ⚠️ 此预测仅供参考，不构成投资建议。加密货币投资存在高风险，请谨慎决策。
          </p>
        </div>
      </footer>
    </div>
  )
}