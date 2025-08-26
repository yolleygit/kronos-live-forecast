import { NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'

export async function GET() {
  try {
    // 读取缓存的历史数据
    const cacheDir = path.join(process.cwd(), '../../data')
    const metadataPath = path.join(cacheDir, 'btc_metadata.json')
    
    if (!fs.existsSync(metadataPath)) {
      return NextResponse.json({ error: 'Historical data not found' }, { status: 404 })
    }

    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'))
    const dataPath = path.join(cacheDir, 'btc_cache.parquet')
    
    if (!fs.existsSync(dataPath)) {
      return NextResponse.json({ error: 'Cache data not found' }, { status: 404 })
    }

    // 注意：这里需要使用parquet读取库，暂时返回模拟数据
    // 实际实现中可以使用 node-parquet 或转换为JSON
    
    // 临时解决方案：读取最近的预测数据作为历史参考
    const rawDataDir = path.join(process.cwd(), '../../predictions_raw/latest')
    const latestMetaPath = path.join(rawDataDir, 'btc_latest_metadata.json')
    
    if (!fs.existsSync(latestMetaPath)) {
      return NextResponse.json({ error: 'Latest data not found' }, { status: 404 })
    }

    const latestMeta = JSON.parse(fs.readFileSync(latestMetaPath, 'utf-8'))
    
    // 生成模拟历史数据（实际应用中从parquet读取）
    const now = new Date()
    const historicalData = []
    
    for (let i = 72; i > 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000)
      const basePrice = 111000 + Math.sin(i / 10) * 2000 + (Math.random() - 0.5) * 1000
      
      historicalData.push({
        timestamp: timestamp.toISOString(),
        close: basePrice,
        volume: Math.floor(Math.random() * 5000) + 1000,
        high: basePrice + Math.random() * 500,
        low: basePrice - Math.random() * 500,
        open: basePrice + (Math.random() - 0.5) * 200
      })
    }

    return NextResponse.json({
      data: historicalData,
      metadata: {
        symbol: 'BTCUSDT',
        timeframe: '1h',
        count: historicalData.length,
        last_updated: metadata.last_updated || new Date().toISOString()
      }
    })

  } catch (error) {
    console.error('Error reading historical data:', error)
    return NextResponse.json(
      { error: 'Failed to read historical data' }, 
      { status: 500 }
    )
  }
}