#!/usr/bin/env python3
"""
Kronos Live Forecast - 单次运行版本
用于测试和开发，避免进入持续调度循环
"""

import sys
from pathlib import Path

# 导入主要功能
from update_predictions import (
    load_model, main_task, CONFIG, Config, logger
)


def main():
    """单次运行主函数"""
    logger.info("=" * 60)
    logger.info("Kronos Live Forecast - 单次运行模式")
    logger.info("=" * 60)
    
    # 显示配置信息
    logger.info(f"模型: {CONFIG['model']['model_name']}")
    logger.info(f"采样次数: {CONFIG['sampling']['num_samples']}")
    logger.info(f"历史窗口: {CONFIG['data']['history_window']} 小时")
    logger.info(f"预测范围: {CONFIG['data']['forecast_horizon']} 小时")
    
    # 确保模型目录存在
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载模型
        logger.info("正在加载模型...")
        loaded_model = load_model()
        
        # 执行单次预测任务
        logger.info("开始执行预测任务...")
        main_task(loaded_model)
        
        logger.info("=" * 60)
        logger.info("✅ 单次预测任务完成！")
        logger.info("检查生成的文件:")
        logger.info(f"  - {CONFIG['output']['chart_filename']}")
        logger.info(f"  - {CONFIG['output']['html_filename']}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"运行出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()