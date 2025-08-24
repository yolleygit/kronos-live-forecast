#!/usr/bin/env python3
"""
Kronos Live Forecast - 实时数据流
WebSocket实时获取市场数据
"""

import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime, timezone
import logging
from pathlib import Path

from data_manager import DataManager

logger = logging.getLogger(__name__)


class RealtimeDataStream:
    """实时数据流管理器"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1h"
        self.is_running = False
        self.latest_kline = None
    
    async def connect_websocket(self):
        """连接WebSocket数据流"""
        logger.info("连接Binance WebSocket数据流...")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info("WebSocket连接成功")
                self.is_running = True
                
                async for message in websocket:
                    if not self.is_running:
                        break
                        
                    await self.process_message(message)
                    
        except Exception as e:
            logger.error(f"WebSocket连接错误: {e}")
            self.is_running = False
    
    async def process_message(self, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            if 'k' in data:  # K线数据
                kline = data['k']
                
                # 只处理已完成的K线
                if kline['x']:  # x=true表示K线已关闭
                    kline_data = {
                        'timestamps': pd.to_datetime(kline['t'], unit='ms', utc=True),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'amount': float(kline['q'])
                    }
                    
                    self.latest_kline = kline_data
                    logger.info(f"接收到新K线: {kline_data['timestamps']} - 收盘价: {kline_data['close']}")
                    
                    # 更新本地缓存
                    await self.update_cache_with_new_data(kline_data)
                    
        except Exception as e:
            logger.error(f"处理消息出错: {e}")
    
    async def update_cache_with_new_data(self, kline_data):
        """用新数据更新缓存"""
        try:
            # 读取现有缓存
            cache_file = self.data_manager.cache_file
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                
                # 添加新数据
                new_row = pd.DataFrame([kline_data])
                updated_df = pd.concat([df, new_row]).drop_duplicates(
                    subset=['timestamps'], keep='last'
                ).sort_values('timestamps').reset_index(drop=True)
                
                # 保持最新2000条数据
                if len(updated_df) > 2000:
                    updated_df = updated_df.tail(2000)
                
                # 保存更新的缓存
                updated_df.to_parquet(cache_file)
                logger.info(f"缓存已更新，当前{len(updated_df)}条数据")
                
        except Exception as e:
            logger.error(f"更新缓存出错: {e}")
    
    def stop(self):
        """停止数据流"""
        self.is_running = False
        logger.info("实时数据流已停止")


class RealtimeDataManager:
    """实时数据管理器"""
    
    def __init__(self):
        self.stream = RealtimeDataStream()
        self.background_task = None
    
    def start_background_stream(self):
        """在后台启动实时数据流"""
        if self.background_task is None or self.background_task.done():
            loop = asyncio.get_event_loop()
            self.background_task = loop.create_task(self.stream.connect_websocket())
            logger.info("后台实时数据流已启动")
    
    def stop_background_stream(self):
        """停止后台数据流"""
        if self.background_task and not self.background_task.done():
            self.stream.stop()
            self.background_task.cancel()
            logger.info("后台数据流已停止")
    
    def get_latest_data(self):
        """获取最新数据"""
        return self.stream.latest_kline


async def test_realtime_stream():
    """测试实时数据流"""
    stream = RealtimeDataStream()
    
    # 运行5分钟测试
    import signal
    
    def signal_handler(signum, frame):
        stream.stop()
        logger.info("收到停止信号")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await stream.connect_websocket()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"测试出错: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("启动实时数据流测试 (Ctrl+C停止)...")
    asyncio.run(test_realtime_stream())