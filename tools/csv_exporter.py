#!/usr/bin/env python3
"""
CSV导出器 - 按时间戳命名导出数据
"""
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CSVExporter:
    """CSV数据导出器"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def generate_filename(self, symbol="BTCUSDT", timeframe="1h", latest_timestamp=None):
        """
        生成CSV文件名
        格式: BTCUSDT_1h_2025082411.csv (表示最新数据时间为2024年8月24日11点)
        """
        if latest_timestamp is None:
            latest_timestamp = datetime.now(timezone.utc)
        
        if isinstance(latest_timestamp, str):
            latest_timestamp = pd.to_datetime(latest_timestamp)
        
        # 确保是UTC时区
        if latest_timestamp.tz is None:
            latest_timestamp = latest_timestamp.tz_localize('UTC')
        elif latest_timestamp.tz != timezone.utc:
            latest_timestamp = latest_timestamp.tz_convert('UTC')
        
        # 格式化时间戳: YYYYMMDDHH
        time_str = latest_timestamp.strftime("%Y%m%d%H")
        
        filename = f"{symbol}_{timeframe}_{time_str}.csv"
        return filename
    
    def export_dataframe(self, df, symbol="BTCUSDT", timeframe="1h", filename=None):
        """导出DataFrame为CSV"""
        if df is None or df.empty:
            logger.error("数据为空，无法导出")
            return None
        
        # 获取最新时间戳
        latest_time = df['timestamps'].max()
        
        # 生成文件名
        if filename is None:
            filename = self.generate_filename(symbol, timeframe, latest_time)
        
        csv_path = self.data_dir / filename
        
        # 准备导出数据
        df_export = df.copy()
        
        # 将时间戳转换为易读格式
        df_export['timestamps'] = df_export['timestamps'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # 导出CSV
        df_export.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ 数据已导出: {csv_path}")
        logger.info(f"📊 数据条数: {len(df_export)}")
        logger.info(f"⏰ 时间范围: {df_export['timestamps'].iloc[0]} 至 {df_export['timestamps'].iloc[-1]}")
        
        return str(csv_path)
    
    def export_from_cache(self, symbol="BTCUSDT", timeframe="1h"):
        """从缓存导出数据"""
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            core_path = project_root / "core"
            sys.path.insert(0, str(core_path))
            
            from data_manager import DataManager
            
            dm = DataManager()
            df = dm.get_data()
            
            if df is not None:
                csv_path = self.export_dataframe(df, symbol, timeframe)
                return csv_path
            else:
                logger.error("缓存中没有数据")
                return None
                
        except Exception as e:
            logger.error(f"从缓存导出失败: {e}")
            return None
    
    def list_csv_files(self):
        """列出所有CSV文件"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        file_info = []
        for csv_file in sorted(csv_files):
            try:
                stat = csv_file.stat()
                file_info.append({
                    'filename': csv_file.name,
                    'size_kb': round(stat.st_size / 1024, 1),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception:
                continue
        
        return file_info
    
    def cleanup_old_csv_files(self, keep_latest=5):
        """清理旧的CSV文件，只保留最新的几个"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if len(csv_files) <= keep_latest:
            logger.info(f"CSV文件数量({len(csv_files)}) <= 保留数量({keep_latest})，无需清理")
            return
        
        # 按修改时间排序
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # 删除多余的文件
        files_to_delete = csv_files[keep_latest:]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                logger.info(f"🗑️ 已删除旧文件: {file_path.name}")
            except Exception as e:
                logger.error(f"删除失败 {file_path.name}: {e}")
        
        logger.info(f"✅ 清理完成，保留了最新的{keep_latest}个CSV文件")


def export_current_data():
    """导出当前缓存数据"""
    exporter = CSVExporter()
    
    print("=== CSV数据导出 ===")
    
    # 导出当前数据
    csv_path = exporter.export_from_cache()
    
    if csv_path:
        print(f"✅ 导出成功: {csv_path}")
    else:
        print("❌ 导出失败")
    
    # 显示所有CSV文件
    print("\n=== 现有CSV文件 ===")
    csv_files = exporter.list_csv_files()
    
    if csv_files:
        for file_info in csv_files:
            print(f"📄 {file_info['filename']} ({file_info['size_kb']} KB, {file_info['modified']})")
    else:
        print("无CSV文件")
    
    # 清理旧文件
    print("\n=== 清理旧文件 ===")
    exporter.cleanup_old_csv_files(keep_latest=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    export_current_data()