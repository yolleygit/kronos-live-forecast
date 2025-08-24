#!/usr/bin/env python3
"""
长期历史数据下载器
从2024年7月开始获取完整的历史数据
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import logging

# 添加核心模块路径
project_root = Path(__file__).parent.parent
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

from ccxt_data_source import CCXTDataSource
from data_manager import DataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_days_from_july_2024():
    """计算从2024年7月1日到现在的天数"""
    start_date = datetime(2024, 7, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    
    days_diff = (now - start_date).days
    logger.info(f"从2024年7月1日到现在: {days_diff} 天")
    return days_diff, start_date

def download_full_history():
    """下载从2024年7月开始的完整历史数据"""
    print("=== 长期历史数据下载 ===")
    
    # 计算需要的天数
    days_needed, start_date = calculate_days_from_july_2024()
    
    # 转换为小时数（每天24小时）
    hours_needed = days_needed * 24
    print(f"需要下载约 {hours_needed} 小时的数据")
    
    # CCXT通常限制单次请求的数量，我们需要分批获取
    max_limit = 1000  # 大多数交易所的限制
    
    if hours_needed <= max_limit:
        print(f"单次请求即可获取所有数据 (需要 {hours_needed} 小时)")
        return download_single_batch(hours_needed)
    else:
        print(f"需要分批下载 (总共 {hours_needed} 小时，单次最多 {max_limit})")
        return download_in_batches(hours_needed, max_limit, start_date)

def download_single_batch(limit):
    """单次下载数据"""
    try:
        ccxt_source = CCXTDataSource()
        
        print("开始下载历史数据...")
        df = ccxt_source.fetch_ohlcv_data('BTC/USDT', '1h', limit=limit)
        
        if df is not None:
            print(f"✅ 下载成功: {len(df)} 条数据")
            print(f"时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
            
            # 保存到缓存
            save_to_cache(df)
            return df
        else:
            print("❌ 下载失败")
            return None
            
    except Exception as e:
        logger.error(f"单次下载失败: {e}")
        return None

def download_in_batches(total_hours, batch_size, start_date):
    """分批下载历史数据"""
    print("⚠️ 注意: CCXT通常只能获取最近1000小时的数据")
    print("对于更长的历史数据，可能需要其他数据源或API")
    
    # 先尝试获取最大可能的数据
    try:
        ccxt_source = CCXTDataSource()
        
        print(f"尝试获取最近 {batch_size} 小时的数据...")
        df = ccxt_source.fetch_ohlcv_data('BTC/USDT', '1h', limit=batch_size)
        
        if df is not None:
            print(f"✅ 获取成功: {len(df)} 条数据")
            print(f"实际时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
            
            # 检查是否覆盖到2024年7月
            earliest_time = df['timestamps'].min()
            target_start = start_date
            
            if earliest_time >= target_start:
                print(f"✅ 数据覆盖了目标起始时间 {target_start}")
            else:
                print(f"⚠️ 数据未完全覆盖目标时间范围")
                print(f"   实际开始: {earliest_time}")
                print(f"   目标开始: {target_start}")
            
            # 保存到缓存
            save_to_cache(df)
            return df
        else:
            print("❌ 获取失败")
            return None
            
    except Exception as e:
        logger.error(f"分批下载失败: {e}")
        return None

def save_to_cache(df):
    """保存数据到缓存"""
    try:
        dm = DataManager()
        
        # 备份现有缓存
        if dm.cache_file.exists():
            backup_file = dm.cache_file.with_suffix('.parquet.backup')
            df_old = pd.read_parquet(dm.cache_file)
            df_old.to_parquet(backup_file)
            logger.info(f"✅ 已备份现有数据: {backup_file}")
        
        # 保存新数据
        df.to_parquet(dm.cache_file)
        dm._save_metadata({"last_update": datetime.now(timezone.utc).isoformat()})
        logger.info(f"✅ 历史数据已保存到缓存: {len(df)} 条")
        
        # 导出CSV
        dm._export_to_csv(df)
        
        return True
        
    except Exception as e:
        logger.error(f"保存缓存失败: {e}")
        return False

def main():
    """主函数"""
    try:
        df = download_full_history()
        
        if df is not None:
            print(f"\n=== 下载完成 ===")
            print(f"📊 总数据量: {len(df)} 条")
            print(f"⏰ 时间跨度: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
            print(f"💰 最新价格: ${df['close'].iloc[-1]:,.2f}")
            
            # 检查2024年7月后的数据
            july_2024 = datetime(2024, 7, 1, tzinfo=timezone.utc)
            data_after_july = df[df['timestamps'] >= july_2024]
            
            print(f"📅 2024年7月后数据: {len(data_after_july)} 条")
            if not data_after_july.empty:
                print(f"   覆盖时间: {data_after_july['timestamps'].min()} 至 {data_after_july['timestamps'].max()}")
        else:
            print("❌ 历史数据下载失败")
            
    except Exception as e:
        logger.error(f"主函数执行失败: {e}")

if __name__ == "__main__":
    main()