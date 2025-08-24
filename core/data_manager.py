#!/usr/bin/env python3
"""
Kronos Live Forecast - 数据管理器
实现数据缓存、更新和管理功能
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

from binance.client import Client
try:
    from update_predictions import CONFIG, logger
except ImportError:
    # 如果独立运行，使用默认配置
    CONFIG = {
        "data": {
            "symbol": "BTCUSDT",
            "timeframe": "1h", 
            "history_window": 360,
            "volatility_window": 24
        }
    }
    import logging
    logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器：负责缓存、更新和提供历史数据"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据文件路径
        self.cache_file = self.data_dir / "btc_cache.parquet"
        self.metadata_file = self.data_dir / "metadata.json"
        self.db_file = self.data_dir / "market_data.db"
        
        # 配置参数
        self.symbol = CONFIG["data"]["symbol"]
        self.timeframe = CONFIG["data"]["timeframe"]
        self.required_hours = CONFIG["data"]["history_window"] + CONFIG["data"]["volatility_window"]
        
        logger.info(f"数据管理器初始化完成，缓存目录: {self.data_dir}")
    
    def get_cache_info(self):
        """获取缓存信息"""
        if not self.cache_file.exists():
            return {"status": "empty", "count": 0, "latest_time": None}
            
        try:
            df = pd.read_parquet(self.cache_file)
            return {
                "status": "available",
                "count": len(df),
                "latest_time": df['timestamps'].max(),
                "earliest_time": df['timestamps'].min(),
                "file_size": self.cache_file.stat().st_size
            }
        except Exception as e:
            logger.error(f"读取缓存信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def need_update(self, max_age_hours=1):
        """检查是否需要更新数据"""
        cache_info = self.get_cache_info()
        
        if cache_info["status"] != "available":
            return True, "缓存不存在或损坏"
            
        if cache_info["count"] < self.required_hours:
            return True, f"数据不足，需要{self.required_hours}小时，当前{cache_info['count']}小时"
            
        # 检查数据新鲜度
        latest_time = cache_info["latest_time"]
        if isinstance(latest_time, str):
            latest_time = pd.to_datetime(latest_time)
        
        # 确保时间戳有时区信息
        if latest_time.tz is None:
            latest_time = latest_time.tz_localize('UTC')
        elif latest_time.tz != timezone.utc:
            latest_time = latest_time.tz_convert('UTC')
            
        hours_old = (datetime.now(timezone.utc) - latest_time).total_seconds() / 3600
        
        if hours_old > max_age_hours:
            return True, f"数据过期，已过时{hours_old:.1f}小时"
            
        return False, "缓存数据充足且新鲜"
    
    def fetch_fresh_data(self, limit=1000):
        """优化：优先使用CCXT统一交易所API"""
        logger.info(f"获取{limit}条最新数据...")
        
        # 主要方案: CCXT统一交易所API
        try:
            from ccxt_data_source import CCXTDataSource
            ccxt_source = CCXTDataSource()
            
            # 转换symbol格式 (BTCUSDT -> BTC/USDT)
            symbol_ccxt = self.symbol.replace('USDT', '/USDT').replace('BTC', 'BTC')
            if not '/' in symbol_ccxt:
                symbol_ccxt = 'BTC/USDT'  # 默认
                
            df = ccxt_source.fetch_ohlcv_data(
                symbol=symbol_ccxt, 
                timeframe=self.timeframe, 
                limit=limit
            )
            
            if df is not None:
                logger.info(f"✅ CCXT获取数据成功: {len(df)}条数据")
                logger.info(f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
                return df
            else:
                logger.warning("CCXT返回空数据")
                
        except Exception as e1:
            logger.warning(f"CCXT方法失败: {e1}")
        
        # 备用方案: 外部数据源降级策略
        logger.info("启用多数据源降级策略...")
        try:
            from external_data_sources import MultiSourceDataManager
            external_manager = MultiSourceDataManager()
            days = min(42, limit // 24)  # 根据limit估算天数
            df = external_manager.get_data_with_fallback(days=days)
            if df is not None:
                logger.info(f"外部数据源获取成功: {len(df)}条数据")
                return df
        except Exception as e2:
            logger.error(f"外部数据源也失败: {e2}")
        
        logger.error("所有数据获取方法都失败")
        return None
    
    def _convert_klines_to_df(self, klines):
        """转换K线数据为DataFrame"""
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(klines, columns=cols)
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
        df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)
        
        df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])
            
        return df.sort_values('timestamps').reset_index(drop=True)
    
    def update_cache(self, force_full_update=False):
        """更新本地缓存"""
        cache_info = self.get_cache_info()
        
        if force_full_update or cache_info["status"] != "available":
            # 全量更新
            logger.info("执行全量数据更新...")
            fresh_df = self.fetch_fresh_data(limit=1000)
            if fresh_df is None:
                return False
                
            # 保存到缓存
            fresh_df.to_parquet(self.cache_file)
            self._save_metadata({"last_update": datetime.now(timezone.utc).isoformat()})
            logger.info(f"全量更新完成，缓存了{len(fresh_df)}条数据")
            
            # 自动导出CSV
            self._export_to_csv(fresh_df)
            
            return True
        
        else:
            # 增量更新
            logger.info("执行增量数据更新...")
            existing_df = pd.read_parquet(self.cache_file)
            
            # 获取最新数据
            fresh_df = self.fetch_fresh_data(limit=100)  # 只获取最新100条
            if fresh_df is None:
                return False
            
            # 合并数据，去重
            combined_df = pd.concat([existing_df, fresh_df]).drop_duplicates(
                subset=['timestamps'], keep='last'
            ).sort_values('timestamps').reset_index(drop=True)
            
            # 保留最新的数据（避免文件过大）
            if len(combined_df) > 2000:
                combined_df = combined_df.tail(2000)
            
            # 保存更新的缓存
            combined_df.to_parquet(self.cache_file)
            self._save_metadata({"last_update": datetime.now(timezone.utc).isoformat()})
            logger.info(f"增量更新完成，当前缓存{len(combined_df)}条数据")
            
            # 自动导出CSV
            self._export_to_csv(combined_df)
            
            return True
    
    def get_data(self, hours=None, force_update=False):
        """
        智能获取历史数据
        
        Args:
            hours: 需要的数据小时数
            force_update: 是否强制更新
        """
        if hours is None:
            hours = self.required_hours
            
        # 智能缓存策略
        cache_info = self.get_cache_info()
        
        # 1. 如果有足够的新鲜缓存数据，直接使用
        if not force_update and cache_info["status"] == "available":
            need_update, reason = self.need_update()
            
            if not need_update and cache_info["count"] >= hours:
                logger.info("✅ 使用本地缓存数据，跳过网络请求")
                try:
                    df = pd.read_parquet(self.cache_file)
                    result_df = df.tail(hours).copy()
                    logger.info(f"从缓存获取{len(result_df)}小时数据成功")
                    return result_df
                except Exception as e:
                    logger.error(f"缓存读取失败: {e}")
                    # 继续尝试更新
        
        # 2. 需要更新数据时才连接网络
        logger.info("缓存不足或过期，需要获取新数据...")
        success = self.update_cache(force_full_update=force_update)
        if not success:
            logger.warning("数据更新失败，尝试使用现有缓存...")
        
        # 3. 从缓存读取数据
        if not self.cache_file.exists():
            logger.error("没有可用的缓存数据")
            return None
            
        try:
            df = pd.read_parquet(self.cache_file)
            
            if len(df) >= hours:
                result_df = df.tail(hours).copy()
                logger.info(f"从缓存获取{len(result_df)}小时数据成功")
                return result_df
            else:
                logger.warning(f"数据不足，需要{hours}小时，只有{len(df)}小时")
                return df
                
        except Exception as e:
            logger.error(f"读取缓存数据失败: {e}")
            return None
    
    def _save_metadata(self, metadata):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _export_to_csv(self, df):
        """导出数据为CSV格式"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
            from csv_exporter import CSVExporter
            
            exporter = CSVExporter(data_dir=self.data_dir)
            symbol = self.symbol
            timeframe = self.timeframe
            
            csv_path = exporter.export_dataframe(df, symbol, timeframe)
            if csv_path:
                logger.info(f"📄 CSV数据已导出: {Path(csv_path).name}")
            
            # 清理旧CSV文件，保留最新3个
            exporter.cleanup_old_csv_files(keep_latest=3)
            
        except Exception as e:
            logger.warning(f"CSV导出失败: {e}")
    
    def get_cache_status(self):
        """获取详细的缓存状态"""
        cache_info = self.get_cache_info()
        need_update, reason = self.need_update()
        
        status = {
            "cache_info": cache_info,
            "need_update": need_update,
            "update_reason": reason,
            "required_hours": self.required_hours
        }
        
        return status


# 全局数据管理器实例
data_manager = DataManager()


def get_market_data():
    """统一的数据获取接口"""
    return data_manager.get_data()


if __name__ == "__main__":
    # 测试数据管理器
    logger.info("=== 数据管理器测试 ===")
    
    status = data_manager.get_cache_status()
    print(f"缓存状态: {status}")
    
    # 获取数据测试
    df = data_manager.get_data(hours=50)
    if df is not None:
        print(f"获取数据成功: {len(df)}行")
        print(f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
        print(f"最新价格: {df['close'].iloc[-1]:.2f}")
    else:
        print("获取数据失败")