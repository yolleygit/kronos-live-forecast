#!/usr/bin/env python3
"""
数据验证器 - 检查和修正数据中的时间问题
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        pass
    
    def check_data_timerange(self, df):
        """检查数据时间范围"""
        if df is None or df.empty:
            return None
        
        earliest = df['timestamps'].min()
        latest = df['timestamps'].max()
        
        print(f"=== 数据时间范围检查 ===")
        print(f"最早时间: {earliest}")
        print(f"最新时间: {latest}")
        print(f"数据跨度: {latest - earliest}")
        print(f"数据条数: {len(df)}")
        
        # 检查时间是否合理 
        now = datetime.now(timezone.utc)
        
        # 检查是否有未来时间
        if latest > now:
            print(f"⚠️ 警告: 发现未来时间 {latest} > 当前时间 {now}")
        
        # 检查是否有不合理的过去时间
        one_year_ago = now - timedelta(days=365)
        if earliest < one_year_ago:
            print(f"⚠️ 警告: 数据开始时间过早 {earliest} < 一年前 {one_year_ago}")
        
        # 检查2024年7月之后的要求
        cutoff_date = datetime(2024, 7, 1, tzinfo=timezone.utc)
        if earliest < cutoff_date:
            data_after_cutoff = df[df['timestamps'] >= cutoff_date]
            print(f"📅 2024年7月后的数据: {len(data_after_cutoff)} 条")
            print(f"   时间范围: {data_after_cutoff['timestamps'].min()} 至 {data_after_cutoff['timestamps'].max()}")
            return data_after_cutoff
        else:
            print("✅ 数据已符合2024年7月后的要求")
            return df
    
    def fix_year_issues(self, df):
        """修正可能的年份问题"""
        if df is None or df.empty:
            return df
        
        # 检查是否有2025年的数据（可能是错误的）
        now = datetime.now(timezone.utc)
        current_year = now.year
        
        future_data = df[df['timestamps'].dt.year > current_year]
        if not future_data.empty:
            print(f"⚠️ 发现 {len(future_data)} 条未来年份数据")
            
            # 尝试修正：将2025年改为2024年
            df_fixed = df.copy()
            mask = df_fixed['timestamps'].dt.year > current_year
            
            # 将年份减1
            df_fixed.loc[mask, 'timestamps'] = df_fixed.loc[mask, 'timestamps'].apply(
                lambda ts: ts.replace(year=ts.year - 1)
            )
            
            print(f"✅ 已修正年份问题")
            return df_fixed
        else:
            print("✅ 未发现年份问题")
            return df
    
    def filter_data_from_july_2024(self, df):
        """筛选2024年7月后的数据"""
        if df is None or df.empty:
            return df
        
        cutoff_date = datetime(2024, 7, 1, tzinfo=timezone.utc)
        filtered_df = df[df['timestamps'] >= cutoff_date].copy()
        
        print(f"=== 数据筛选 ===")
        print(f"筛选前: {len(df)} 条")
        print(f"筛选后: {len(filtered_df)} 条 (2024年7月后)")
        
        if not filtered_df.empty:
            print(f"新时间范围: {filtered_df['timestamps'].min()} 至 {filtered_df['timestamps'].max()}")
        
        return filtered_df.reset_index(drop=True)
    
    def validate_and_fix_cache(self):
        """验证并修正缓存数据"""
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            core_path = project_root / "core"
            sys.path.insert(0, str(core_path))
            
            from data_manager import DataManager
            
            dm = DataManager()
            
            # 读取原始数据
            df = pd.read_parquet(dm.cache_file)
            print(f"读取缓存数据: {len(df)} 条")
            
            # 检查时间范围
            df_checked = self.check_data_timerange(df)
            
            # 修正年份问题
            df_fixed = self.fix_year_issues(df_checked)
            
            # 筛选2024年7月后的数据
            df_filtered = self.filter_data_from_july_2024(df_fixed)
            
            if df_filtered is not None and not df_filtered.empty and len(df_filtered) != len(df):
                print(f"\n=== 更新缓存 ===")
                
                # 备份原文件
                backup_file = dm.cache_file.with_suffix('.parquet.backup')
                df.to_parquet(backup_file)
                print(f"✅ 原数据已备份: {backup_file}")
                
                # 保存修正后的数据
                df_filtered.to_parquet(dm.cache_file)
                print(f"✅ 缓存已更新: {len(df_filtered)} 条数据")
                
                # 更新元数据
                dm._save_metadata({"last_update": datetime.now(timezone.utc).isoformat()})
                
                # 导出CSV
                dm._export_to_csv(df_filtered)
                
                return df_filtered
            else:
                print("✅ 数据无需修正")
                return df
                
        except Exception as e:
            logger.error(f"验证修正失败: {e}")
            return None


def main():
    """主函数"""
    print("=== 数据验证和修正工具 ===")
    
    validator = DataValidator()
    result_df = validator.validate_and_fix_cache()
    
    if result_df is not None:
        print(f"\n=== 最终结果 ===")
        print(f"数据条数: {len(result_df)}")
        if not result_df.empty:
            print(f"时间范围: {result_df['timestamps'].min()} 至 {result_df['timestamps'].max()}")
            print(f"最新价格: {result_df['close'].iloc[-1]:.2f}")
    else:
        print("❌ 验证修正失败")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()