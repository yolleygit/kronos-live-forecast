#!/usr/bin/env python3
"""
数据查看和转换工具
可以查看缓存数据并转换为CSV格式
"""
import sys
from pathlib import Path
import pandas as pd
import argparse

# 添加核心模块路径
project_root = Path(__file__).parent.parent
core_path = project_root / "core" 
sys.path.insert(0, str(core_path))

from data_manager import DataManager

def view_data_info():
    """查看数据基本信息"""
    dm = DataManager()
    
    print("=== 数据缓存信息 ===")
    cache_info = dm.get_cache_info()
    
    if cache_info["status"] == "available":
        print(f"✅ 数据状态: {cache_info['status']}")
        print(f"📊 数据条数: {cache_info['count']}")
        print(f"⏰ 时间范围: {cache_info['earliest_time']} 至 {cache_info['latest_time']}")
        print(f"📁 文件大小: {cache_info['file_size']/1024:.1f} KB")
        
        # 读取实际数据查看详情
        df = dm.get_data()
        if df is not None:
            print(f"\n=== 数据详情 ===")
            print(f"列名: {list(df.columns)}")
            print(f"数据类型:")
            for col in df.columns:
                print(f"  {col}: {df[col].dtype}")
            
            print(f"\n=== 最新5条数据 ===")
            print(df.tail().to_string())
            
            print(f"\n=== 数据统计 ===")
            print(df.describe())
            
            return df
    else:
        print(f"❌ 数据状态: {cache_info.get('status', 'unknown')}")
        if 'error' in cache_info:
            print(f"错误信息: {cache_info['error']}")
        return None

def export_to_csv(output_file="data_export.csv"):
    """导出数据为CSV格式"""
    dm = DataManager()
    df = dm.get_data()
    
    if df is not None:
        # 确保时间戳为字符串格式便于CSV读写
        df_export = df.copy()
        df_export['timestamps'] = df_export['timestamps'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        output_path = Path(output_file)
        df_export.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 数据已导出到: {output_path.absolute()}")
        print(f"📊 导出数据量: {len(df_export)} 条")
        return str(output_path.absolute())
    else:
        print("❌ 没有数据可导出")
        return None

def main():
    parser = argparse.ArgumentParser(description="数据查看和转换工具")
    parser.add_argument('--info', action='store_true', help='显示数据信息')
    parser.add_argument('--csv', type=str, help='导出为CSV文件 (指定文件名)')
    parser.add_argument('--both', action='store_true', help='显示信息并导出为CSV')
    
    args = parser.parse_args()
    
    if args.both or args.info or (not args.csv and not args.both):
        view_data_info()
    
    if args.both or args.csv:
        csv_file = args.csv if args.csv else "btc_data_export.csv"
        export_to_csv(csv_file)

if __name__ == "__main__":
    main()