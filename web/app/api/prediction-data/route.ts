import { NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'

export async function GET() {
  try {
    // 修正路径：从web目录到项目根目录是../
    const projectRoot = path.resolve(process.cwd(), '..')
    const rawDataDir = path.join(projectRoot, 'predictions_raw/latest')
    const closePredictionsPath = path.join(rawDataDir, 'btc_latest_close.csv')
    const volumePredictionsPath = path.join(rawDataDir, 'btc_latest_volume.csv')
    const metadataPath = path.join(rawDataDir, 'btc_latest_metadata.json')
    
    console.log('预测数据目录:', rawDataDir)
    console.log('Close预测文件存在:', fs.existsSync(closePredictionsPath))
    console.log('Volume预测文件存在:', fs.existsSync(volumePredictionsPath))
    console.log('元数据文件存在:', fs.existsSync(metadataPath))
    
    if (!fs.existsSync(closePredictionsPath)) {
      return NextResponse.json({ error: 'Prediction data not found' }, { status: 404 })
    }

    const closePredictions = fs.readFileSync(closePredictionsPath, 'utf-8')
    let volumePredictions = ''
    let metadata = null

    try {
      if (fs.existsSync(volumePredictionsPath)) {
        volumePredictions = fs.readFileSync(volumePredictionsPath, 'utf-8')
      }
      if (fs.existsSync(metadataPath)) {
        metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'))
      }
    } catch (error) {
      console.warn('Warning: Could not read optional prediction files:', error)
    }

    return NextResponse.json({
      close_predictions: closePredictions,
      volume_predictions: volumePredictions,
      metadata,
      content_type: 'text/csv'
    })

  } catch (error) {
    console.error('Error reading prediction data:', error)
    return NextResponse.json(
      { error: 'Failed to read prediction data' }, 
      { status: 500 }
    )
  }
}

export async function POST() {
  return NextResponse.json({ error: 'Method not allowed' }, { status: 405 })
}