#!/usr/bin/env python3
"""
Kronos Live Forecast - 外部数据源集成
支持多种数据源：Yahoo Finance, Alpha Vantage, CoinGecko等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


class ExternalDataSources:
    """外部数据源管理器"""
    
    def __init__(self):
        self.sources = {
            'coingecko': self._fetch_coingecko_data,
            'yahoo': self._fetch_yahoo_data,
            'cryptocompare': self._fetch_cryptocompare_data,
        }
    
    def get_data_from_source(self, source='coingecko', symbol='bitcoin', days=30):
        """从指定数据源获取数据"""
        if source not in self.sources:
            raise ValueError(f"不支持的数据源: {source}")
        
        logger.info(f"从{source}获取{symbol}的{days}天数据...")
        return self.sources[source](symbol, days)
    
    def _fetch_coingecko_data(self, coin_id='bitcoin', days=30):
        """从CoinGecko获取数据 (免费，无API密钥)"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # 转换数据格式
            prices = data['prices']
            volumes = data['total_volumes']
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                # CoinGecko返回毫秒时间戳
                dt = pd.to_datetime(timestamp, unit='ms')
                
                # 简化的OHLC数据 (CoinGecko只提供价格点，我们模拟OHLC)
                volume = volumes[i][1] if i < len(volumes) else 0
                
                # 生成简单的OHLC (使用价格点作为收盘价)
                close_price = price
                # 添加小幅随机波动模拟开盘、最高、最低价
                volatility = 0.005  # 0.5%的模拟波动
                noise = np.random.normal(0, volatility, 3)
                
                open_price = close_price * (1 + noise[0])
                high_price = max(open_price, close_price) * (1 + abs(noise[1]))
                low_price = min(open_price, close_price) * (1 - abs(noise[2]))
                
                df_data.append({
                    'timestamps': dt,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'amount': volume * close_price
                })
            
            df = pd.DataFrame(df_data)
            logger.info(f"从CoinGecko获取{len(df)}条数据成功")
            return df
            
        except Exception as e:
            logger.error(f"CoinGecko数据获取失败: {e}")
            return None
    
    def _fetch_yahoo_data(self, symbol='BTC-USD', days=30):
        """从Yahoo Finance获取数据"""
        try:
            # 使用yfinance库（需要安装：pip install yfinance）
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            
            if hist.empty:
                logger.warning("Yahoo Finance没有返回数据")
                return None
            
            # 转换格式
            df = hist.reset_index()
            df = df.rename(columns={
                'Datetime': 'timestamps',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 添加amount列
            df['amount'] = df['volume'] * df['close']
            
            # 确保时间戳格式正确
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            logger.info(f"从Yahoo Finance获取{len(df)}条数据成功")
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            
        except ImportError:
            logger.error("需要安装yfinance: pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"Yahoo Finance数据获取失败: {e}")
            return None
    
    def _fetch_cryptocompare_data(self, symbol='BTC', days=30):
        """从CryptoCompare获取数据"""
        try:
            # CryptoCompare API (需要免费API密钥)
            api_key = "YOUR_CRYPTOCOMPARE_API_KEY"  # 需要注册获取
            
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': days * 24,
                'api_key': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] != 'Success':
                logger.error(f"CryptoCompare API错误: {data.get('Message', 'Unknown error')}")
                return None
            
            # 转换数据格式
            df_data = []
            for item in data['Data']['Data']:
                df_data.append({
                    'timestamps': pd.to_datetime(item['time'], unit='s'),
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volumeto'],  # 使用成交额作为volume
                    'amount': item['volumeto']
                })
            
            df = pd.DataFrame(df_data)
            logger.info(f"从CryptoCompare获取{len(df)}条数据成功")
            return df
            
        except Exception as e:
            logger.error(f"CryptoCompare数据获取失败: {e}")
            return None


class MultiSourceDataManager:
    """多数据源管理器"""
    
    def __init__(self):
        self.external_sources = ExternalDataSources()
        self.fallback_order = ['coingecko', 'yahoo', 'cryptocompare']
    
    def get_data_with_fallback(self, days=30):
        """使用多数据源降级获取数据"""
        
        for source in self.fallback_order:
            try:
                logger.info(f"尝试从{source}获取数据...")
                df = None
                
                if source == 'coingecko':
                    df = self.external_sources.get_data_from_source('coingecko', 'bitcoin', days)
                elif source == 'yahoo':
                    df = self.external_sources.get_data_from_source('yahoo', 'BTC-USD', days)
                elif source == 'cryptocompare':
                    df = self.external_sources.get_data_from_source('cryptocompare', 'BTC', days)
                
                if df is not None and len(df) > 0:
                    logger.info(f"从{source}获取数据成功")
                    return df
                    
            except Exception as e:
                logger.warning(f"{source}获取失败: {e}")
                continue
        
        logger.error("所有外部数据源都失败")
        return None
    
    def compare_data_sources(self, days=7):
        """比较不同数据源的数据质量"""
        results = {}
        
        for source in self.fallback_order:
            try:
                df = self.external_sources.get_data_from_source(source, days=days)
                if df is not None:
                    results[source] = {
                        'count': len(df),
                        'time_range': f"{df['timestamps'].min()} - {df['timestamps'].max()}",
                        'avg_price': df['close'].mean(),
                        'price_std': df['close'].std(),
                        'completeness': (df.notna().sum() / len(df)).mean()
                    }
            except Exception as e:
                results[source] = {'error': str(e)}
        
        return results


def test_external_sources():
    """测试外部数据源"""
    manager = MultiSourceDataManager()
    
    print("=== 数据源比较测试 ===")
    comparison = manager.compare_data_sources(days=3)
    
    for source, result in comparison.items():
        print(f"\n{source.upper()}:")
        if 'error' in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  数据量: {result['count']}")
            print(f"  时间范围: {result['time_range']}")
            print(f"  平均价格: ${result['avg_price']:.2f}")
            print(f"  价格标准差: ${result['price_std']:.2f}")
            print(f"  数据完整度: {result['completeness']:.2%}")
    
    print("\n=== 降级获取测试 ===")
    df = manager.get_data_with_fallback(days=7)
    if df is not None:
        print(f"成功获取{len(df)}条数据")
        print(f"最新价格: ${df['close'].iloc[-1]:.2f}")
    else:
        print("获取数据失败")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_external_sources()