#!/usr/bin/env python3
"""
Kronos Live Forecast - 主启动脚本
自动化定时预测服务，根据config.yaml中的timeframe参数调度
"""
import sys
import os
import time
import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import argparse

# 添加核心模块路径



project_root = Path(__file__).parent
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

# 导入核心预测模块
from core.update_predictions import load_model, main_task


def load_config():
    """加载配置文件"""
    # 支持通过环境变量或命令行参数覆盖配置路径
    env_cfg = os.environ.get("KRONOS_CONFIG")
    if env_cfg:
        config_path = Path(env_cfg)
    else:
        config_path = project_root / "configs" / "config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        raise


def parse_timeframe(timeframe):
    """
    解析timeframe字符串，返回更新间隔（秒）
    支持: 1h, 4h, 1d, 30m, 15m 等格式
    """
    timeframe = timeframe.lower().strip()
    
    # 提取数字和单位
    match = re.match(r'(\d+)([smhd])', timeframe)
    if not match:
        raise ValueError(f"无效的timeframe格式: {timeframe}")
    
    number = int(match.group(1))
    unit = match.group(2)
    
    # 转换为秒
    unit_multipliers = {
        's': 1,           # 秒
        'm': 60,          # 分钟
        'h': 3600,        # 小时
        'd': 86400        # 天
    }
    
    if unit not in unit_multipliers:
        raise ValueError(f"不支持的时间单位: {unit}")
    
    interval_seconds = number * unit_multipliers[unit]
    
    print(f"📅 检测到timeframe: {timeframe} -> 更新间隔: {interval_seconds}秒 ({interval_seconds/3600:.1f}小时)")
    return interval_seconds


def run_scheduler(model, config):
    """
    根据config.yaml中的timeframe参数运行定时调度器
    """
    timeframe = config['data']['timeframe']
    update_interval = parse_timeframe(timeframe)
    
    print(f"🔄 启动自动化调度器...")
    print(f"⏰ 更新频率: 每 {timeframe} 运行一次")
    print(f"🎯 目标交易对: {config['data']['symbol']}")
    print(f"📊 预测范围: {config['data']['forecast_horizon']} 小时")
    
    # 立即执行一次
    print(f"\n{'='*60}")
    print(f"🚀 立即执行首次预测...")
    print(f"{'='*60}")
    
    try:
        main_task(model)
        print(f"✅ 首次预测任务完成！")
    except Exception as e:
        print(f"❌ 首次预测任务失败: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  将在下次调度时重试...")
    
    # 进入定时循环
    while True:
        try:
            # 计算下次运行时间
            now = datetime.now(timezone.utc)
            
            # 根据timeframe对齐下次运行时间
            if 'h' in timeframe:  # 小时对齐
                hours = int(re.match(r'(\d+)', timeframe).group(1))
                next_hour = ((now.hour // hours) + 1) * hours
                next_run_time = now.replace(hour=next_hour % 24, minute=0, second=5, microsecond=0)
                if next_hour >= 24:
                    next_run_time += timedelta(days=1)
            elif 'm' in timeframe:  # 分钟对齐
                minutes = int(re.match(r'(\d+)', timeframe).group(1))
                next_minute = ((now.minute // minutes) + 1) * minutes
                next_run_time = now.replace(minute=next_minute % 60, second=5, microsecond=0)
                if next_minute >= 60:
                    next_run_time += timedelta(hours=1)
            else:  # 其他情况：简单加间隔
                next_run_time = now + timedelta(seconds=update_interval)
            
            sleep_seconds = (next_run_time - now).total_seconds()
            
            if sleep_seconds > 0:
                print(f"\n{'─'*60}")
                print(f"⏰ 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"⏰ 下次运行: {next_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"⏰ 等待时间: {sleep_seconds:.0f}秒 ({sleep_seconds/60:.1f}分钟)")
                print(f"{'─'*60}")
                
                # 实时倒计时显示
                remaining = int(sleep_seconds)
                while remaining > 0:
                    try:
                        # 计算时分秒
                        hours = remaining // 3600
                        minutes = (remaining % 3600) // 60
                        seconds = remaining % 60
                        
                        # 格式化倒计时显示
                        if hours > 0:
                            countdown_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            countdown_text = f"{minutes:02d}:{seconds:02d}"
                        
                        # 打印倒计时（覆盖前一行）
                        print(f"\r💤 系统休眠中... 倒计时: {countdown_text} ({remaining}秒)", end="", flush=True)
                        
                        time.sleep(1)
                        remaining -= 1
                        
                    except KeyboardInterrupt:
                        print(f"\n⚠️  用户中断倒计时，跳出休眠...")
                        raise  # 重新抛出KeyboardInterrupt
                
                print(f"\n🚀 休眠结束，开始执行任务...")  # 换行结束倒计时
            
            # 执行预测任务
            print(f"\n{'='*60}")
            print(f"🚀 开始定时预测任务 - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*60}")
            
            main_task(model)
            
            print(f"✅ 定时预测任务完成！")
            
        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print(f"⚠️  用户中断，正在安全关闭调度器...")
            print(f"{'='*60}")
            break
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ 预测任务执行失败: {e}")
            print(f"{'='*60}")
            
            import traceback
            traceback.print_exc()
            
            # 失败后等待5分钟再重试
            print(f"⏰ 将在5分钟后重试...")
            time.sleep(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos scheduler")
    parser.add_argument("--config", type=str, help="Path to config.yaml", default=None)
    args = parser.parse_args()
    if args.config:
        os.environ["KRONOS_CONFIG"] = args.config
    try:
        print("🚀 启动 Kronos 自动化预测系统...")
        print("=" * 60)
        
        # 加载配置
        config = load_config()
        
        # 加载模型
        print("🤖 正在加载AI模型...")
        loaded_model = load_model()
        print("✅ 模型加载完成！")
        
        # 启动调度器
        run_scheduler(loaded_model, config)
        
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print("👋 用户主动退出，再见！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"💥 系统启动失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()