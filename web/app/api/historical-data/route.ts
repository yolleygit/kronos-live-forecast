import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'

export async function GET(request: NextRequest) {
  try {
    // 获取symbol参数
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get('symbol') || 'btc'
    console.log('🔍 历史数据API接收到symbol参数:', symbol)
    
    // 修正路径：从web目录到项目根目录是../
    const projectRoot = path.resolve(process.cwd(), '..')
    const cacheDir = path.join(projectRoot, 'data')
    const metadataPath = path.join(cacheDir, `${symbol}_metadata.json`)
    
    console.log('项目根目录:', projectRoot)
    console.log('数据目录:', cacheDir) 
    console.log('元数据路径:', metadataPath)
    console.log('元数据文件存在:', fs.existsSync(metadataPath))
    
    if (!fs.existsSync(metadataPath)) {
      return NextResponse.json({ error: 'Historical data not found' }, { status: 404 })
    }

    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'))
    
    // 读取最新的CSV数据文件而不是parquet
    const symbolPair = symbol.toUpperCase() === 'BTC' ? 'BTCUSDT' : 'ETHUSDT'
    const csvFiles = fs.readdirSync(cacheDir).filter(f => f.startsWith(`${symbolPair}_1h_`) && f.endsWith('.csv')).sort().reverse()
    
    console.log(`${symbol.toUpperCase()}可用CSV文件:`, csvFiles)
    
    if (csvFiles.length === 0) {
      return NextResponse.json({ error: 'No CSV data files found' }, { status: 404 })
    }
    
    // 读取最新的CSV文件
    const latestCsvPath = path.join(cacheDir, csvFiles[0])
    const csvContent = fs.readFileSync(latestCsvPath, 'utf-8')
    
    // 解析CSV数据
    const lines = csvContent.trim().split('\n')
    const headers = lines[0].split(',')
    console.log('CSV标题行:', headers)
    
    const historicalData = []
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',')
      if (values.length >= 6) {
        // CSV格式: timestamps,open,high,low,close,volume,amount
        const timestampStr = values[0] // 字符串格式: "2025-07-14 03:00:00 UTC"
        const open = parseFloat(values[1])
        const high = parseFloat(values[2])
        const low = parseFloat(values[3])
        const close = parseFloat(values[4])
        const volume = parseFloat(values[5])
        
        // 将UTC时间字符串转换为ISO格式
        const timestamp = new Date(timestampStr.replace(' UTC', 'Z')).toISOString()
        
        historicalData.push({
          timestamp,
          open,
          high,
          low,
          close,
          volume
        })
      }
    }
    
    console.log('解析的历史数据条数:', historicalData.length)
    console.log('最新数据时间戳:', historicalData[historicalData.length - 1]?.timestamp)

    return NextResponse.json({
      data: historicalData,
      metadata: {
        symbol: symbolPair,
        timeframe: '1h',
        count: historicalData.length,
        last_updated: metadata.last_update || new Date().toISOString()
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