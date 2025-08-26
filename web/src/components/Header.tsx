import React from 'react'
import { Clock, Activity, DollarSign, RefreshCw } from 'lucide-react'
import { cn, formatCurrency } from '@/utils'

interface HeaderProps {
  lastUpdated: string
  currentPrice: number
  isLoading?: boolean
  onRefresh?: () => void
  forecastHorizon?: number
  symbol?: string
  symbolPair?: string
}

export default function Header({ lastUpdated, currentPrice, isLoading = false, onRefresh, forecastHorizon = 24, symbol = 'BTC', symbolPair = 'BTC/USDT' }: HeaderProps) {
  const formatLastUpdated = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    } catch {
      return timestamp
    }
  }

  return (
    <div className="relative">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 opacity-90" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,rgba(255,255,255,0.1),transparent)]" />
      
      <div className="relative px-6 py-8 text-white">
        <div className="max-w-7xl mx-auto">
          {/* Title Section */}
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
              Kronos Live Forecast
            </h1>
            <p className="text-lg text-blue-100 font-light">
              基于Transformer模型的{symbolPair}实时价格预测系统
            </p>
          </div>

          {/* Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Current Price Card */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <DollarSign className="w-5 h-5 text-green-300" />
                  <span className="text-sm font-medium text-blue-100">当前价格</span>
                </div>
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              </div>
              <div className="text-2xl font-bold text-white">
                {formatCurrency(currentPrice)}
              </div>
              <div className="text-xs text-blue-200 mt-1">
                {symbolPair}
              </div>
            </div>

            {/* Last Updated Card */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Clock className="w-5 h-5 text-blue-300" />
                  <span className="text-sm font-medium text-blue-100">最后更新</span>
                </div>
                {onRefresh && (
                  <button
                    onClick={onRefresh}
                    disabled={isLoading}
                    className="p-1 hover:bg-white/10 rounded-full transition-colors disabled:opacity-50"
                  >
                    <RefreshCw className={cn("w-4 h-4 text-blue-200", isLoading && "animate-spin")} />
                  </button>
                )}
              </div>
              <div className="text-lg font-semibold text-white">
                {formatLastUpdated(lastUpdated)}
              </div>
              <div className="text-xs text-blue-200 mt-1">
                Beijing Time (UTC+8)
              </div>
            </div>

            {/* Model Status Card */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-purple-300" />
                  <span className="text-sm font-medium text-blue-100">模型状态</span>
                </div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse-soft" />
              </div>
              <div className="text-lg font-semibold text-white">
                Kronos Active
              </div>
              <div className="text-xs text-blue-200 mt-1">
                {forecastHorizon}H Forecast Ready
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}