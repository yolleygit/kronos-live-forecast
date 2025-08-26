/**
 * 数据源服务 - 统一管理所有数据获取逻辑
 * 负责从不同目录获取数据并提供统一接口
 */

interface DataSourceConfig {
  dataDir: string        // data/ 历史数据目录
  predictionsDir: string // predictions_raw/ 原始预测目录  
  recordsDir: string     // records/ 结构化记录目录
}

interface HistoricalDataPoint {
  timestamp: string
  close: number
  volume: number
  open: number
  high: number
  low: number
}

interface PredictionDataPoint {
  timestamp: string
  close_predictions: number[]
  volume_predictions?: number[]
  volatility_predictions?: number[]
}

interface DashboardData {
  lastUpdated: string
  currentPrice: number
  config: {
    forecast_horizon: number
    num_samples: number
  }
  metrics: Record<string, number>
}

class DataSourceService {
  private config: DataSourceConfig = {
    dataDir: '/data/',
    predictionsDir: '/predictions_raw/',
    recordsDir: '/records/'
  }

  /**
   * 获取最新的历史数据
   * 数据来源: data/btc_cache.parquet (通过API转换)
   */
  async getHistoricalData(symbol = 'btc', hours = 48): Promise<HistoricalDataPoint[]> {
    try {
      const response = await fetch(`/api/historical-data?symbol=${symbol}&hours=${hours}`)
      if (!response.ok) {
        throw new Error(`历史数据获取失败: ${response.statusText}`)
      }
      const result = await response.json()
      // API返回格式: {data: HistoricalDataPoint[], metadata: {...}}
      return result.data || result
    } catch (error) {
      console.error('获取历史数据失败:', error)
      throw error
    }
  }

  /**
   * 获取最新的预测数据
   * 数据来源: predictions_raw/latest/ 或 records/latest_btc.json
   */
  async getLatestPredictions(symbol = 'btc'): Promise<PredictionDataPoint[]> {
    try {
      // 优先使用结构化记录
      const response = await fetch(`/api/prediction-data?symbol=${symbol}&source=latest`)
      if (!response.ok) {
        throw new Error(`预测数据获取失败: ${response.statusText}`)
      }
      
      const result = await response.json()
      // API返回格式: {close_predictions: string, volume_predictions: string, metadata: object}
      return this.parsePredictionCSV(result.close_predictions, result.volume_predictions)
    } catch (error) {
      console.error('获取预测数据失败:', error)
      throw error
    }
  }

  /**
   * 获取仪表板配置数据
   * 数据来源: records/latest_btc.json
   */
  async getDashboardData(symbol = 'btc'): Promise<DashboardData> {
    try {
      const response = await fetch(`/api/dashboard?symbol=${symbol}`)
      if (!response.ok) {
        throw new Error(`仪表板数据获取失败: ${response.statusText}`)
      }
      return await response.json()
    } catch (error) {
      console.error('获取仪表板数据失败:', error)
      throw error
    }
  }

  /**
   * 计算预测开始时间
   * 基于最新历史数据时间的下一个整点
   */
  calculateForecastStartTime(historicalData: HistoricalDataPoint[]): string {
    if (historicalData.length === 0) {
      throw new Error('历史数据为空，无法计算预测开始时间')
    }

    // 获取最新历史数据时间戳
    const latestTime = new Date(historicalData[historicalData.length - 1].timestamp)
    
    // 对齐到下一个整点
    const nextHour = new Date(latestTime)
    nextHour.setMinutes(0, 0, 0)
    nextHour.setHours(nextHour.getHours() + 1)
    
    return nextHour.toISOString()
  }

  /**
   * 生成预测时间戳序列
   */
  generatePredictionTimestamps(startTime: string, forecastHorizon: number): string[] {
    const timestamps: string[] = []
    const start = new Date(startTime)
    
    for (let i = 0; i < forecastHorizon; i++) {
      const timestamp = new Date(start.getTime() + i * 60 * 60 * 1000)
      timestamps.push(timestamp.toISOString())
    }
    
    return timestamps
  }

  /**
   * 解析预测CSV数据
   */
  private parsePredictionCSV(closeCsvContent: string, volumeCsvContent?: string): PredictionDataPoint[] {
    const closeLines = closeCsvContent.trim().split('\n')
    if (closeLines.length <= 1) {
      return []
    }

    const data: PredictionDataPoint[] = []
    
    // 解析成交量预测（如果存在）
    let volumeLines: string[] = []
    if (volumeCsvContent) {
      volumeLines = volumeCsvContent.trim().split('\n')
    }
    
    for (let i = 1; i < closeLines.length; i++) {
      const closeValues = closeLines[i].split(',')
      if (closeValues.length < 2) continue
      
      const timestamp = closeValues[0].trim()
      const closePredictions = closeValues.slice(1).map(v => parseFloat(v)).filter(v => !isNaN(v))
      
      let volumePredictions: number[] = []
      if (volumeLines.length > i) {
        const volumeValues = volumeLines[i].split(',')
        volumePredictions = volumeValues.slice(1).map(v => parseFloat(v)).filter(v => !isNaN(v))
      }
      
      data.push({
        timestamp,
        close_predictions: closePredictions,
        volume_predictions: volumePredictions.length > 0 ? volumePredictions : undefined
      })
    }
    
    return data
  }

  /**
   * 获取完整的图表数据 - 统一入口
   */
  async getChartData(symbol = 'btc') {
    console.log('🔄 开始获取图表数据...', { symbol })
    
    // 并行获取所有必要数据
    const [historicalData, dashboardData] = await Promise.all([
      this.getHistoricalData(symbol, 72), // 获取72小时历史数据
      this.getDashboardData(symbol)
    ])

    // 计算正确的预测开始时间
    const forecastStartTime = this.calculateForecastStartTime(historicalData)
    
    // 生成预测时间戳
    const predictionTimestamps = this.generatePredictionTimestamps(
      forecastStartTime, 
      dashboardData.config.forecast_horizon
    )

    console.log('📊 数据获取完成:', {
      历史数据点数: historicalData.length,
      最新历史时间: historicalData[historicalData.length - 1]?.timestamp,
      预测开始时间: forecastStartTime,
      预测时间点数: predictionTimestamps.length,
      当前价格: dashboardData.currentPrice
    })

    return {
      historicalData,
      predictionTimestamps,
      forecastStartTime,
      currentPrice: dashboardData.currentPrice,
      config: dashboardData.config,
      metrics: dashboardData.metrics,
      lastUpdated: dashboardData.lastUpdated
    }
  }
}

export default new DataSourceService()