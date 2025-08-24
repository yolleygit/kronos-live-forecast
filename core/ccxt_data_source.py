#!/usr/bin/env python3
"""
CCXT数据源 - 统一的加密货币交易所数据获取
支持多个交易所的标准化API调用
"""
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class CCXTDataSource:
    """基于CCXT的通用数据源"""
    
    def __init__(self):
        self.exchanges = []
        self.current_exchange = None
        self._init_exchanges()
    
    def _init_exchanges(self):
        """初始化支持的交易所"""
        try:
            import ccxt
            
            # 按优先级排序的交易所列表
            exchange_configs = [
                {'name': 'binance', 'class': ccxt.binance, 'rateLimit': 1200},
                {'name': 'okx', 'class': ccxt.okx, 'rateLimit': 2000}, 
                {'name': 'bybit', 'class': ccxt.bybit, 'rateLimit': 1000},
                {'name': 'gate', 'class': ccxt.gate, 'rateLimit': 1000},
                {'name': 'huobi', 'class': ccxt.huobi, 'rateLimit': 2000}
            ]
            
            for config in exchange_configs:
                try:
                    exchange = config['class']({
                        'rateLimit': config['rateLimit'],
                        'timeout': 30000,
                        'enableRateLimit': True,
                    })
                    
                    # 简单测试连通性
                    if hasattr(exchange, 'load_markets'):
                        exchange.load_markets()
                    
                    self.exchanges.append({
                        'name': config['name'],
                        'instance': exchange,
                        'status': 'available'
                    })
                    logger.info(f"✅ {config['name']} 交易所初始化成功")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {config['name']} 交易所初始化失败: {e}")
                    
        except ImportError:
            logger.error("CCXT库未安装，请运行: pip install ccxt")
            
        logger.info(f"CCXT数据源初始化完成，可用交易所: {len(self.exchanges)}")
    
    def get_available_exchanges(self) -> List[str]:
        """获取可用的交易所列表"""
        return [ex['name'] for ex in self.exchanges if ex['status'] == 'available']
    
    def fetch_ohlcv_data(self, symbol='BTC/USDT', timeframe='1h', limit=1000, 
                        exchange_name=None) -> Optional[pd.DataFrame]:
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对符号 (如 'BTC/USDT')
            timeframe: 时间周期 ('1m', '5m', '1h', '1d' 等)
            limit: 数据条数
            exchange_name: 指定交易所名称，None则按优先级尝试
        """
        if not self.exchanges:
            logger.error("没有可用的交易所")
            return None
            
        # 确定要使用的交易所列表
        if exchange_name:
            target_exchanges = [ex for ex in self.exchanges 
                              if ex['name'] == exchange_name and ex['status'] == 'available']
            if not target_exchanges:
                logger.error(f"指定的交易所 {exchange_name} 不可用")
                return None
        else:
            target_exchanges = [ex for ex in self.exchanges if ex['status'] == 'available']
        
        # 按优先级尝试获取数据
        for exchange_info in target_exchanges:
            try:
                exchange = exchange_info['instance']
                name = exchange_info['name']
                
                logger.info(f"尝试从 {name} 获取 {symbol} {timeframe} 数据 (limit={limit})...")
                
                # 获取OHLCV数据
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    logger.warning(f"{name} 返回空数据")
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 处理时间戳
                df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.drop('timestamp', axis=1)
                
                # 添加amount列（成交额 = 价格 × 成交量）
                df['amount'] = df['close'] * df['volume']
                
                # 重新排列列顺序
                df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                
                # 按时间排序
                df = df.sort_values('timestamps').reset_index(drop=True)
                
                logger.info(f"✅ 从 {name} 获取数据成功: {len(df)} 条")
                logger.info(f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
                
                self.current_exchange = name  # 记录成功的交易所
                return df
                
            except Exception as e:
                logger.warning(f"❌ {exchange_info['name']} 获取失败: {e}")
                # 标记该交易所暂时不可用
                exchange_info['status'] = 'error'
                continue
        
        logger.error("所有CCXT交易所都获取失败")
        return None
    
    def fetch_recent_data(self, symbol='BTC/USDT', timeframe='1h', 
                         hours_back=24) -> Optional[pd.DataFrame]:
        """获取最近N小时的数据"""
        return self.fetch_ohlcv_data(symbol, timeframe, limit=hours_back)
    
    def test_connection(self):
        """测试所有交易所连接"""
        results = {}
        
        for exchange_info in self.exchanges:
            name = exchange_info['name']
            try:
                exchange = exchange_info['instance']
                
                # 测试获取少量数据
                ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=5)
                if ohlcv and len(ohlcv) > 0:
                    results[name] = {'status': 'success', 'data_count': len(ohlcv)}
                    exchange_info['status'] = 'available'
                else:
                    results[name] = {'status': 'no_data'}
                    exchange_info['status'] = 'error'
                    
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
                exchange_info['status'] = 'error'
        
        return results


def test_ccxt_data_source():
    """测试CCXT数据源"""
    print("=== CCXT数据源测试 ===")
    
    source = CCXTDataSource()
    
    print(f"\n可用交易所: {source.get_available_exchanges()}")
    
    print("\n=== 连接测试 ===")
    test_results = source.test_connection()
    
    for exchange, result in test_results.items():
        status = result['status']
        if status == 'success':
            print(f"✅ {exchange}: 连接成功，数据条数: {result.get('data_count', 0)}")
        elif status == 'no_data':
            print(f"⚠️ {exchange}: 连接成功但无数据")
        else:
            print(f"❌ {exchange}: 连接失败 - {result.get('error', 'unknown error')}")
    
    print("\n=== 数据获取测试 ===")
    df = source.fetch_ohlcv_data('BTC/USDT', '1h', limit=10)
    
    if df is not None:
        print(f"✅ 获取数据成功: {len(df)} 条")
        print(f"使用的交易所: {source.current_exchange}")
        print(f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
        print(f"最新价格: {df['close'].iloc[-1]:.2f} USDT")
        print("\n最新3条数据:")
        print(df.tail(3).to_string())
    else:
        print("❌ 数据获取失败")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_ccxt_data_source()