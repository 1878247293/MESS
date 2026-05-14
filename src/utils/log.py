"""
loguru 日志封装。

init_logger 同时挂两个 handler：终端只输出 INFO+ 且屏蔽 API 调试噪声，文件保留
DEBUG 全量。`log` / `log_args` / `log_time` 是 INFO 级薄包装。
"""

import time
import sys

from loguru import logger


def init_logger(file_name):
    """
    初始化 loguru。
    控制台只走 INFO 以上，文件保留 DEBUG 全量。
    """
    file_name = f"logs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) }_{file_name}.log"

    # 干掉默认 stdout handler
    logger.remove()

    # 控制台只想看关键进度，把 API 调试这种噪声过滤掉
    def terminal_filter(record):
        # DEBUG 太碎，ERROR 仅写文件
        if record["level"].name in ["DEBUG", "ERROR"]:
            return False

        logger_name = record.get("name", "")

        message = str(record.get("message", ""))

        # 屏蔽 API 详情类日志
        api_keywords = ["API 响应:", "完整响应对象:", "Prompt 前100字符:",
                       "响应 choices", "批量处理: 任务", "批量解析:"]
        if any(keyword in message for keyword in api_keywords):
            return False

        # 单批失败的详情也别打到屏幕上，只看汇总
        if "⚠️  批次" in message and "处理失败" in message:
            return False

        return True

    logger.add(
        sys.stdout,
        level="INFO",
        filter=terminal_filter,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    logger.add(
        file_name,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return file_name


def log(msg):
    logger.info(msg)


def log_args(args):
    for k, v in args.__dict__.items():
        if str(k).startswith("__"):
            continue
        log(f"{k}: {v}")


def log_time(desc: str, elapsed_time: float):
    logger.info(f"[{desc}]: {elapsed_time:0.4f} seconds")
