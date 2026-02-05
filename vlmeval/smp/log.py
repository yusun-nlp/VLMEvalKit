import logging
import sys
from pathlib import Path
from typing import Optional

LOGGER_NAME = "vlmeval"

def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    配置项目日志系统
    
    Args:
        log_dir: 日志保存目录，None 表示不保存到文件
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: 是否输出到控制台
    
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(LOGGER_NAME)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(filename)s:%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到根日志器
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 子模块名称，会自动加上项目前缀
    
    Returns:
        logger 实例
    """
    if name:
        logger_name = f"{LOGGER_NAME}.{name}"
    else:
        logger_name = LOGGER_NAME

    logger = logging.getLogger(logger_name)

    # 如果作为库使用且未配置，使用 NullHandler
    if not logging.getLogger(LOGGER_NAME).handlers:
        logger.addHandler(logging.NullHandler())

    return logger
