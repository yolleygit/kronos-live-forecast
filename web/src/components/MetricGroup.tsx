import React from 'react'
import { cn } from '@/utils'
import { MetricGroupProps } from '@/types'
import MetricCard from './MetricCard'

export default function MetricGroup({ title, description, metrics, className }: MetricGroupProps) {
  return (
    <div className={cn("space-y-6", className)}>
      {/* Group Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
          {title}
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed">
          {description}
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {metrics.map((metric, index) => (
          <MetricCard
            key={index}
            title={metric.title}
            value={metric.value}
            description={metric.description}
            trend={metric.trend}
            color={metric.color}
            icon={metric.icon}
          />
        ))}
      </div>
    </div>
  )
}