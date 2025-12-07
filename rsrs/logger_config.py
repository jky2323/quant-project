import logging
import os
from datetime import datetime
from config import CONFIG


def setup_logger(module_name, log_to_file=True, log_to_console=False):
    """
    设置日志记录器
    
    参数:
        module_name: 模块名称，用于日志文件名和记录器名称
        log_to_file: 是否输出到文件 (默认 True)
        log_to_console: 是否输出到控制台 (默认 False)
    
    返回:
        logger 对象
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    if log_to_file:
        os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)
        log_filename = os.path.join(
            CONFIG['LOG_DIR'],
            f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# 创建全局日志记录器供所有模块使用
analysis_logger = setup_logger('analysis', log_to_file=True, log_to_console=False)