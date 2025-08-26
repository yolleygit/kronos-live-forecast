import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { cn } from '@/utils'
import { MetricCardProps } from '@/types'

export default function MetricCard({ 
  title, 
  value, 
  description, 
  trend = 'neutral', 
  color = 'blue',
  icon 
}: MetricCardProps) {
  const getColorClasses = () => {
    switch (color) {
      case 'green':
        return 'from-emerald-500 to-emerald-600 text-white shadow-emerald-500/25'
      case 'red':
        return 'from-red-500 to-red-600 text-white shadow-red-500/25'
      case 'yellow':
        return 'from-amber-500 to-amber-600 text-white shadow-amber-500/25'
      case 'purple':
        return 'from-purple-500 to-purple-600 text-white shadow-purple-500/25'
      case 'indigo':
        return 'from-indigo-500 to-indigo-600 text-white shadow-indigo-500/25'
      default:
        return 'from-blue-500 to-blue-600 text-white shadow-blue-500/25'
    }
  }

  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4" />
      case 'down':
        return <TrendingDown className="w-4 h-4" />
      default:
        return <Minus className="w-4 h-4" />
    }
  }

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-emerald-200'
      case 'down':
        return 'text-red-200'
      default:
        return 'text-slate-200'
    }
  }

  return (
    <div 
      className={cn(
        "relative overflow-hidden rounded-xl p-6 shadow-lg transition-all duration-300 hover:shadow-xl hover:scale-105",
        "bg-gradient-to-br",
        getColorClasses()
      )}
    >
      {/* Background pattern */}
      <div className="absolute inset-0 bg-white/5 bg-[radial-gradient(circle_at_50%_120%,rgba(255,255,255,0.1),transparent)]" />
      
      {/* Header */}
      <div className="relative flex items-start justify-between mb-4">
        <div className="flex items-center space-x-2">
          {icon && <div className="flex-shrink-0">{icon}</div>}
          <h3 className="text-sm font-medium text-white/90">{title}</h3>
        </div>
        <div className={cn("flex items-center space-x-1", getTrendColor())}>
          {getTrendIcon()}
        </div>
      </div>

      {/* Value */}
      <div className="relative mb-4">
        <div className="text-3xl font-bold text-white">
          {value}
        </div>
      </div>

      {/* Description */}
      <div className="relative">
        <p className="text-xs text-white/80 leading-relaxed">
          {description}
        </p>
      </div>

      {/* Glow effect */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-white/20 to-transparent rounded-xl blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-200" />
    </div>
  )
}