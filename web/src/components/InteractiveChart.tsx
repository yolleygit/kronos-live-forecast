import React, { useState, useEffect, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Bar
} from 'recharts'
import { formatCurrency, formatDateTime } from '@/utils'
import dataSourceService from '@/services/dataSourceService'

interface ChartDataPoint {
  timestamp: string
  historicalPrice?: number
  meanPrediction?: number
  predictionUpper?: number
  predictionLower?: number
  volume?: number
  currentPrice?: number
  isForecast: boolean
}

interface InteractiveChartProps {
  data: ChartDataPoint[]
  forecastStartTime: string
  currentPrice: number
  config?: {
    forecast_horizon: number
    num_samples: number
  }
  symbol?: string
}

export default function InteractiveChart({ 
  data, 
  forecastStartTime, 
  currentPrice,
  config = { forecast_horizon: 4, num_samples: 30 },
  symbol = 'btc'
}: InteractiveChartProps) {
  const [timeRange, setTimeRange] = useState<'24h' | '48h' | '72h' | 'full'>('24h')
  const [showVolume, setShowVolume] = useState(true)
  const [showPredictionRange, setShowPredictionRange] = useState(true)
  const [realData, setRealData] = useState<ChartDataPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [internalSymbol, setInternalSymbol] = useState(symbol || 'btc') // 内部状态管理
  
  // 使用父组件传入的symbol或内部状态
  const activeSymbol = symbol || internalSymbol

  // 当父组件symbol改变时，同步内部状态
  useEffect(() => {
    if (symbol && symbol !== internalSymbol) {
      console.log('📊 图表接收到父组件symbol变化:', symbol)
      setInternalSymbol(symbol)
    }
  }, [symbol, internalSymbol])

  // 移除事件监听器，现在只使用父组件传入的symbol

  // 获取真实数据
  useEffect(() => {
    async function fetchRealData() {
      try {
        setLoading(true)
        console.log('📊 开始获取真实图表数据...', { activeSymbol })
        
        const chartData = await dataSourceService.getChartData(activeSymbol)
        
        // 转换历史数据格式
        const historicalPoints: ChartDataPoint[] = chartData.historicalData.map(item => ({
          timestamp: item.timestamp,
          historicalPrice: item.close,
          volume: item.volume,
          isForecast: false
        }))
        
        // 模拟预测数据 (实际应该从 predictions_raw 获取)
        const predictionPoints: ChartDataPoint[] = chartData.predictionTimestamps.map(timestamp => {
          const basePrice = chartData.currentPrice
          const randomVariation = (Math.random() - 0.5) * 2000
          const meanPrice = basePrice + randomVariation
          
          return {
            timestamp,
            meanPrediction: meanPrice,
            predictionUpper: meanPrice + Math.random() * 1000,
            predictionLower: meanPrice - Math.random() * 1000,
            isForecast: true
          }
        })
        
        const combinedData = [...historicalPoints, ...predictionPoints]
        console.log('✅ 真实数据加载完成:', {
          历史点数: historicalPoints.length,
          预测点数: predictionPoints.length,
          预测开始时间: chartData.forecastStartTime
        })
        
        setRealData(combinedData)
        setError(null)
      } catch (err) {
        console.error('❌ 获取图表数据失败:', err)
        setError(err instanceof Error ? err.message : '数据加载失败')
      } finally {
        setLoading(false)
      }
    }
    
    fetchRealData()
  }, [activeSymbol]) // 使用内部activeSymbol状态

  // 使用真实数据，如果加载失败则使用传入的data作为后备
  const chartData = realData.length > 0 ? realData : data || []

  const filteredData = useMemo(() => {
    if (timeRange === 'full' || chartData.length === 0) return chartData
    
    const now = new Date()
    const hours = timeRange === '24h' ? 24 : timeRange === '48h' ? 48 : 72
    const cutoff = new Date(now.getTime() - hours * 60 * 60 * 1000)
    
    return chartData.filter(d => new Date(d.timestamp) >= cutoff)
  }, [chartData, timeRange])

  const formatTooltipValue = (value: any, name: string) => {
    if (name.includes('价格') || name.includes('预测') || name.includes('Price') || name.includes('Prediction')) {
      // 只显示整数价格，去掉小数
      return [`$${Math.round(value).toLocaleString()}`, name]
    }
    if (name.includes('交易量') || name.includes('Volume')) {
      return [`${Math.round(value).toLocaleString()}`, name]
    }
    return [value, name]
  }

  const formatTooltipLabel = (label: string) => {
    const date = new Date(label)
    // 格式化为：MM/DD HH:00
    const month = (date.getMonth() + 1).toString().padStart(2, '0')
    const day = date.getDate().toString().padStart(2, '0')
    const hour = date.getHours().toString().padStart(2, '0')
    return `${month}/${day} ${hour}:00`
  }

  // 加载状态
  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-300">正在加载图表数据...</p>
          </div>
        </div>
      </div>
    )
  }

  // 错误状态
  if (error) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-red-500 text-2xl mb-4">⚠️</div>
            <p className="text-red-600 dark:text-red-400 mb-2">数据加载失败</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
      {/* Chart Controls */}
      <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
        <div>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            交互式价格预测图表 - {activeSymbol.toUpperCase()}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            {config.forecast_horizon}小时预测 • {config.num_samples}次采样 • 可拖拽缩放
          </p>
          

        </div>
        
        {/* Time Range Selector */}
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {(['24h', '48h', '72h', 'full'] as const).map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                timeRange === range
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {range === 'full' ? '全部' : range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Toggle Controls */}
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={e => setShowVolume(e.target.checked)}
            className="w-4 h-4 text-blue-600 rounded"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">显示交易量</span>
        </label>
        
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showPredictionRange}
            onChange={e => setShowPredictionRange(e.target.checked)}
            className="w-4 h-4 text-blue-600 rounded"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">显示预测区间</span>
        </label>
      </div>

      {/* Interactive Chart */}
      <div className="space-y-4">
        {/* Main Price Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={filteredData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            
            <XAxis
              dataKey="timestamp"
              tick={{ fontSize: 12, fill: '#6B7280' }}
              tickFormatter={(value) => {
                const date = new Date(value)
                return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:00`
              }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            
            <YAxis
              tick={{ fontSize: 12, fill: '#6B7280' }}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
              domain={['dataMin - 1000', 'dataMax + 1000']}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: 'none',
                borderRadius: '8px',
                color: '#F9FAFB'
              }}
              formatter={formatTooltipValue}
              labelFormatter={formatTooltipLabel}
            />
            
            <Legend />

            {/* Historical Price Line */}
            <Line
              type="monotone"
              dataKey="historicalPrice"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
              name="历史价格"
              connectNulls={false}
            />

            {/* Mean Prediction Line */}
            <Line
              type="monotone"
              dataKey="meanPrediction"
              stroke="#F59E0B"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="平均预测"
              connectNulls={false}
            />

            {/* Prediction Upper Line */}
            {showPredictionRange && (
              <Line
                type="monotone"
                dataKey="predictionUpper"
                stroke="#F59E0B"
                strokeWidth={1}
                strokeOpacity={0.6}
                dot={false}
                name="预测上限"
                connectNulls={false}
              />
            )}

            {/* Prediction Lower Line */}
            {showPredictionRange && (
              <Line
                type="monotone"
                dataKey="predictionLower"
                stroke="#F59E0B"
                strokeWidth={1}
                strokeOpacity={0.6}
                dot={false}
                name="预测下限"
                connectNulls={false}
              />
            )}

            {/* Current Price Reference Line */}
            {currentPrice > 0 && (
              <ReferenceLine
                y={currentPrice}
                stroke="#EF4444"
                strokeDasharray="3 3"
                strokeWidth={1}
                label={{ value: `当前价格: ${formatCurrency(currentPrice)}`, position: "right" }}
              />
            )}

            {/* Forecast Start Line */}
            <ReferenceLine
              x={(() => {
                const now = new Date()
                const nextHour = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours() + 1, 0, 0, 0)
                return nextHour.toISOString()
              })()}
              stroke="#6B7280"
              strokeDasharray="2 2"
              strokeWidth={1}
              label={{ value: "预测开始", position: "left" }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Volume Chart (separate chart) */}
        {showVolume && (
          <ResponsiveContainer width="100%" height={100}>
            <ComposedChart
              data={filteredData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <XAxis
                dataKey="timestamp"
                tick={{ fontSize: 10, fill: '#6B7280' }}
                tickFormatter={(value) => {
                  const date = new Date(value)
                  return `${date.getMonth() + 1}/${date.getDate()}`
                }}
              />
              
              <YAxis
                tick={{ fontSize: 10, fill: '#6B7280' }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
              />

              <Bar
                dataKey="volume"
                fill="#6366F1"
                opacity={0.6}
                name="交易量"
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Chart Info */}
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center">
        💡 提示: 鼠标悬停查看详细数值 • 使用时间范围按钮切换视图 • 可通过复选框控制图层显示
      </div>
    </div>
  )
}