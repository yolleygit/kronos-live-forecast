import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

/**
 * API路由: /api/dashboard 
 * 功能: 从 records/latest_btc.json 读取最新的仪表板数据
 */

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get('symbol') || 'btc'

    // 构建数据路径 (优先使用records目录的结构化数据)
    const projectRoot = process.cwd().replace('/web', '')
    const recordsFile = path.join(projectRoot, 'records', `latest_${symbol}.json`)
    
    // 备用路径: web/public/data/dashboard.json
    const fallbackPath = path.join(process.cwd(), 'public', 'data', 'dashboard.json')
    
    // 尝试读取records数据
    try {
      const recordsData = await fs.readFile(recordsFile, 'utf8')
      const rawData = JSON.parse(recordsData)
      
      // 转换为统一的dashboard格式
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
      
      console.log(`✅ 使用records数据: ${recordsFile}`)
      return NextResponse.json(dashboardData, {
        headers: {
          'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300'
        }
      })
      
    } catch (recordsError) {
      console.warn(`Records文件读取失败，使用备用数据: ${recordsError}`)
      
      // 尝试读取备用数据
      try {
        const jsonData = await fs.readFile(fallbackPath, 'utf8')
        const data = JSON.parse(jsonData)
        console.log(`✅ 使用备用数据: ${fallbackPath}`)
        return NextResponse.json(data)
      } catch (fallbackError) {
        // 最后的fallback - mock数据
        console.warn('所有数据源都失败，使用mock数据:', fallbackError)
        
        const mockData = {
        lastUpdated: new Date().toISOString(),
        currentPrice: 64250,
        chartImagePath: 'prediction_chart.png',
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
          'overall_vol_risk_score': 42
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
          'overall_vol_risk_score': '42/100'
        }
      }
      
      return NextResponse.json(mockData)
    }
  } catch (error) {
    console.error('Error fetching dashboard data:', error)
    return NextResponse.json(
      { error: 'Failed to load dashboard data' },
      { status: 500 }
    )
  }
}