#!/usr/bin/env python3
"""
Kronos Live Forecast - 数据预下载器
批量下载历史数据，建立本地数据库
"""

import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager import DataManager
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreloader:
    """数据预下载器"""
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def download_historical_data(self, days_back=30, from_date=None):
        """下载指定天数的历史数据"""
        logger.info(f"开始下载过去{days_back}天的历史数据...")
        
        # 计算需要的数据量
        hours_needed = days_back * 24
        
        try:
            # 强制全量更新
            success = self.data_manager.update_cache(force_full_update=True)
            
            if success:
                # 检查下载结果
                cache_info = self.data_manager.get_cache_info()
                logger.info(f"下载完成！")
                logger.info(f"数据条数: {cache_info['count']}")
                logger.info(f"时间范围: {cache_info['earliest_time']} 至 {cache_info['latest_time']}")
                logger.info(f"文件大小: {cache_info['file_size'] / 1024:.1f} KB")
                
                return True
            else:
                logger.error("历史数据下载失败")
                return False
                
        except Exception as e:
            logger.error(f"下载过程出错: {e}")
            return False
    
    def verify_data_quality(self):
        """验证数据质量"""
        logger.info("验证数据质量...")
        
        df = self.data_manager.get_data()
        if df is None:
            logger.error("没有数据可验证")
            return False
        
        # 检查数据完整性
        issues = []
        
        # 检查时间序列连续性
        time_gaps = df['timestamps'].diff().dt.total_seconds() / 3600  # 转为小时
        large_gaps = time_gaps[time_gaps > 1.5]  # 超过1.5小时的间隔
        if not large_gaps.empty:
            issues.append(f"发现{len(large_gaps)}个时间间隔异常")
        
        # 检查价格数据合理性
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if df[col].isna().any():
                issues.append(f"{col}列存在空值")
            if (df[col] <= 0).any():
                issues.append(f"{col}列存在非正值")
        
        # 检查OHLC逻辑
        invalid_ohlc = (
            (df['low'] > df['high']) |  # Low > High
            (df['open'] > df['high']) | (df['open'] < df['low']) |  # Open不在High-Low区间
            (df['close'] > df['high']) | (df['close'] < df['low'])  # Close不在High-Low区间
        )
        if invalid_ohlc.any():
            issues.append(f"发现{invalid_ohlc.sum()}条OHLC逻辑错误的数据")
        
        # 输出验证结果
        if not issues:
            logger.info("✅ 数据质量验证通过")
            return True
        else:
            logger.warning("⚠️ 数据质量问题:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
    
    def setup_data_pipeline(self):
        """设置完整的数据管道"""
        logger.info("=== 设置数据管道 ===")
        
        # 1. 下载历史数据
        logger.info("步骤1: 下载历史数据")
        download_success = self.download_historical_data(days_back=45)  # 下载45天数据
        
        if not download_success:
            logger.error("历史数据下载失败，无法继续")
            return False
        
        # 2. 验证数据质量
        logger.info("步骤2: 验证数据质量")
        quality_ok = self.verify_data_quality()
        
        if not quality_ok:
            logger.warning("数据质量有问题，但可以继续使用")
        
        # 3. 显示最终状态
        logger.info("步骤3: 数据管道设置完成")
        status = self.data_manager.get_cache_status()
        logger.info(f"缓存状态: {status}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Kronos数据预下载器")
    parser.add_argument('--days', type=int, default=30, help='下载天数 (默认30天)')
    parser.add_argument('--verify', action='store_true', help='验证数据质量')
    parser.add_argument('--setup', action='store_true', help='设置完整数据管道')
    
    args = parser.parse_args()
    
    preloader = DataPreloader()
    
    if args.setup:
        success = preloader.setup_data_pipeline()
        exit(0 if success else 1)
    
    if args.verify:
        quality_ok = preloader.verify_data_quality()
        if quality_ok:
            logger.info("数据质量验证通过")
            exit(0)
        else:
            logger.error("数据质量验证失败")
            exit(1)
    
    # 默认行为：下载数据
    success = preloader.download_historical_data(days_back=args.days)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()