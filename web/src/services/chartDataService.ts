interface RawPredictionData {
  timestamp: string
  close_predictions: number[]
  volume_predictions?: number[]
  volatility_predictions?: number[]
}

interface HistoricalData {
  timestamp: string
  close: number
  volume: number
}

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

class ChartDataService {
  /**
   * 从CSV字符串解析预测数据
   */
  private parsePredictionCSV(csvContent: string): RawPredictionData[] {
    const lines = csvContent.trim().split('\n')
    const headers = lines[0].split(',')
    const data: RawPredictionData[] = []

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',')
      const timestamp = values[0]
      const predictions = values.slice(1).map(v => parseFloat(v))
      
      data.push({
        timestamp,
        close_predictions: predictions
      })
    }
    
    return data
  }

  /**
   * 从Parquet数据解析历史数据
   */
  private parseHistoricalData(historicalData: any[]): HistoricalData[] {
    return historicalData.map(item => ({
      timestamp: item.timestamp,
      close: item.close,
      volume: item.volume || 0
    }))
  }

  /**
   * 计算预测统计值
   */
  private calculatePredictionStats(predictions: number[]) {
    const sorted = [...predictions].sort((a, b) => a - b)
    const mean = predictions.reduce((sum, val) => sum + val, 0) / predictions.length
    
    // 使用90%置信区间 (5th - 95th percentile)
    const lowerIndex = Math.floor(predictions.length * 0.05)
    const upperIndex = Math.floor(predictions.length * 0.95)
    
    return {
      mean,
      lower: sorted[lowerIndex],
      upper: sorted[upperIndex],
      median: sorted[Math.floor(predictions.length / 2)],
      std: Math.sqrt(predictions.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / predictions.length)
    }
  }

  /**
   * 合并历史数据和预测数据
   */
  async mergeChartData(
    historicalData: HistoricalData[],
    predictionData: RawPredictionData[],
    currentPrice: number,
    forecastStartTime: string
  ): Promise<ChartDataPoint[]> {
    const chartData: ChartDataPoint[] = []
    const forecastStart = new Date(forecastStartTime)

    // 添加历史数据点
    for (const hist of historicalData) {
      const timestamp = new Date(hist.timestamp)
      
      chartData.push({
        timestamp: hist.timestamp,
        historicalPrice: hist.close,
        volume: hist.volume,
        isForecast: false
      })
    }

    // 添加当前价格点
    chartData.push({
      timestamp: forecastStartTime,
      historicalPrice: currentPrice,
      currentPrice: currentPrice,
      isForecast: false
    })

    // 添加预测数据点
    for (const pred of predictionData) {
      const stats = this.calculatePredictionStats(pred.close_predictions)
      
      chartData.push({
        timestamp: pred.timestamp,
        meanPrediction: stats.mean,
        predictionUpper: stats.upper,
        predictionLower: stats.lower,
        isForecast: true
      })
    }

    // 按时间排序
    chartData.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    
    return chartData
  }

  /**
   * 从API获取完整的图表数据
   */
  async fetchChartData(): Promise<{
    chartData: ChartDataPoint[]
    forecastStartTime: string
    currentPrice: number
    config: any
  }> {
    try {
      // 获取仪表板数据（包含配置和当前价格）
      const dashboardResponse = await fetch('/data/dashboard.json')
      const dashboard = await dashboardResponse.json()

      // 获取历史数据（从缓存API）
      const historyResponse = await fetch('/api/historical-data')
      const historicalData = await historyResponse.json()

      // 获取预测数据CSV
      const predictionResponse = await fetch('/api/prediction-data')
      const predictionCSV = await predictionResponse.text()
      const predictionData = this.parsePredictionCSV(predictionCSV)

      // 确定预测开始时间（最后一个历史数据点的下一小时）
      const lastHistoricalTime = new Date(Math.max(...historicalData.map((d: any) => new Date(d.timestamp).getTime())))
      const forecastStartTime = new Date(lastHistoricalTime.getTime() + 60 * 60 * 1000).toISOString()

      // 合并数据
      const chartData = await this.mergeChartData(
        this.parseHistoricalData(historicalData),
        predictionData,
        dashboard.currentPrice,
        forecastStartTime
      )

      return {
        chartData,
        forecastStartTime,
        currentPrice: dashboard.currentPrice,
        config: dashboard.config
      }
    } catch (error) {
      console.error('获取图表数据失败:', error)
      throw new Error('无法加载图表数据')
    }
  }
}

export default new ChartDataService()