import React, { useState } from 'react'
import { BarChart3, Download, ZoomIn, ToggleLeft, ToggleRight } from 'lucide-react'
import { cn } from '@/utils'
import InteractiveChart from './InteractiveChart'

interface ChartSectionProps {
  chartImagePath?: string
  className?: string
  forecastHorizon?: number
  numSamples?: number
  currentPrice?: number
  dashboardData?: any
  symbol?: string
}

export default function ChartSection({ 
  chartImagePath, 
  className, 
  forecastHorizon = 24, 
  numSamples = 30,
  currentPrice = 0,
  dashboardData,
  symbol = 'btc'
}: ChartSectionProps) {
  console.log('🎨 ChartSection接收到的symbol:', symbol)
  const [isInteractive, setIsInteractive] = useState(true) // 默认启用交互式图表
  const handleDownload = () => {
    // Create a temporary link to download the chart
    if (!chartImagePath) {
      console.warn('No chart image path available for download')
      return
    }
    
    const link = document.createElement('a')
    link.href = chartImagePath.startsWith('/') ? chartImagePath : `/${chartImagePath}`
    link.download = 'kronos-prediction-chart.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleZoom = () => {
    // Open chart in new window for better viewing
    if (!chartImagePath) {
      console.warn('No chart image path available for zoom')
      return
    }
    
    const imageUrl = chartImagePath.startsWith('/') ? chartImagePath : `/${chartImagePath}`
    window.open(imageUrl, '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes')
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Chart Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center space-x-2">
          <BarChart3 className="w-6 h-6 text-slate-700 dark:text-slate-300" />
          <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
            {forecastHorizon}小时价格预测图表
          </h2>
        </div>
        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed">
          基于{numSamples}次蒙特卡洛采样的概率性价格预测，展示预期轨迹和置信区间
        </p>
        
        {/* Chart Mode Toggle */}
        <div className="flex items-center justify-center space-x-3 pt-2">
          <span className={cn(
            "text-sm font-medium transition-colors",
            !isInteractive ? "text-blue-600 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"
          )}>
            静态图表
          </span>
          <button
            onClick={() => setIsInteractive(!isInteractive)}
            className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
            title={isInteractive ? "切换到静态图表" : "切换到交互式图表"}
          >
            {isInteractive ? (
              <ToggleRight className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            ) : (
              <ToggleLeft className="w-6 h-6 text-slate-400 dark:text-slate-500" />
            )}
          </button>
          <span className={cn(
            "text-sm font-medium transition-colors",
            isInteractive ? "text-blue-600 dark:text-blue-400" : "text-slate-500 dark:text-slate-400"
          )}>
            交互式图表
          </span>
        </div>
      </div>

      {/* Chart Container */}
      {isInteractive ? (
        <InteractiveChart
          data={[]} // 这里需要从API获取数据
          forecastStartTime={new Date().toISOString()}
          currentPrice={currentPrice}
          config={{ forecast_horizon: forecastHorizon, num_samples: numSamples }}
          symbol={symbol}
        />
      ) : (
        <div className="relative group">
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
            {/* Chart Actions */}
            <div className="absolute top-4 right-4 z-10 flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <button
                onClick={handleZoom}
                className="p-2 bg-black/20 backdrop-blur-sm text-white rounded-lg hover:bg-black/30 transition-colors"
                title="放大查看"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={handleDownload}
                className="p-2 bg-black/20 backdrop-blur-sm text-white rounded-lg hover:bg-black/30 transition-colors"
                title="下载图表"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>

            {/* Chart Image */}
            <div className="relative">
              <img
                src={chartImagePath && chartImagePath.startsWith('/') ? chartImagePath : `/${chartImagePath || 'prediction_chart.png'}`}
                alt="Kronos BTC/USDT Price Prediction"
                className="w-full h-auto"
                style={{ maxHeight: '600px', objectFit: 'contain' }}
                onError={(e) => {
                  console.error('Failed to load chart image:', chartImagePath)
                  const target = e.target as HTMLImageElement
                  target.style.display = 'none'
                  target.nextElementSibling?.classList.remove('hidden')
                }}
              />
              
              {/* Fallback for missing image */}
              <div className="hidden w-full h-96 flex items-center justify-center bg-slate-100 dark:bg-slate-700">
                <div className="text-center space-y-3">
                  <BarChart3 className="w-12 h-12 text-slate-400 mx-auto" />
                  <div className="space-y-1">
                    <p className="text-slate-600 dark:text-slate-400 font-medium">
                      预测图表生成中...
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-500">
                      请稍候或刷新页面查看最新预测结果
                      <br />
                      图表路径: {chartImagePath}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Chart Footer */}
            <div className="p-4 bg-slate-50 dark:bg-slate-750 border-t border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
                <div className="flex items-center space-x-4">
                  <span>📊 数据源: Binance API</span>
                  <span>🤖 模型: Kronos Transformer</span>
                  <span>📈 采样: {numSamples}次蒙特卡洛</span>
                </div>
                <div>
                  <span>🎯 预测范围: {forecastHorizon}小时</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Chart Legend */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="w-4 h-4 bg-blue-500 rounded mx-auto mb-2"></div>
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300">历史价格</div>
          <div className="text-xs text-slate-500 dark:text-slate-500">过去360小时</div>
        </div>
        
        <div className="text-center">
          <div className="w-4 h-4 bg-red-500 rounded mx-auto mb-2"></div>
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300">预测轨迹</div>
          <div className="text-xs text-slate-500 dark:text-slate-500">未来{forecastHorizon}小时</div>
        </div>
        
        <div className="text-center">
          <div className="w-4 h-4 bg-green-200 rounded mx-auto mb-2"></div>
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300">置信区间</div>
          <div className="text-xs text-slate-500 dark:text-slate-500">95%概率范围</div>
        </div>
        
        <div className="text-center">
          <div className="w-4 h-4 bg-yellow-500 rounded mx-auto mb-2"></div>
          <div className="text-xs font-medium text-slate-700 dark:text-slate-300">关键节点</div>
          <div className="text-xs text-slate-500 dark:text-slate-500">重要价格水平</div>
        </div>
      </div>
    </div>
  )
}