import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get('symbol') || 'btc'
    console.log('🎯 Dashboard API接收到symbol参数:', symbol)

    const projectRoot = process.cwd().replace('/web', '')
    const recordsFile = path.join(projectRoot, 'records', `latest_${symbol}.json`)
    const fallbackPath = path.join(process.cwd(), 'public', 'data', 'dashboard.json')
    
    try {
      const recordsData = await fs.readFile(recordsFile, 'utf8')
      const rawData = JSON.parse(recordsData)
      
      const dashboardData = {
        lastUpdated: rawData.timestamp,
        currentPrice: rawData.prediction_results?.current_price || 0,
        config: {
          forecast_horizon: rawData.data_config?.forecast_horizon || 8,
          num_samples: rawData.sampling_config?.num_samples || 30,
          volatility_window: rawData.data_config?.volatility_window || 8
        },
        metrics: rawData.raw_metrics || {},
        formatted: rawData.formatted_metrics || {},
        validation: rawData.validation || null
      }
      
      return NextResponse.json(dashboardData)
      
    } catch (recordsError) {
      try {
        const jsonData = await fs.readFile(fallbackPath, 'utf8')
        const data = JSON.parse(jsonData)
        return NextResponse.json(data)
      } catch (fallbackError) {
        const mockData = {
          lastUpdated: new Date().toISOString(),
          currentPrice: 64250,
          config: {
            forecast_horizon: 8,
            num_samples: 30
          },
          metrics: {
            'upside_0.5%_prob': 0.73,
            'confidence_score': 0.82
          },
          formatted: {
            'upside_0.5%_prob': '73.0%',
            'confidence_score': '82.0%'
          }
        }
        
        return NextResponse.json(mockData)
      }
    }
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load dashboard data' }, { status: 500 })
  }
}