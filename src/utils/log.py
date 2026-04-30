import time
import sys

from loguru import logger


def init_logger(file_name):
    """
    初始化日志系统

    配置：
    - 终端：只显示INFO及以上，不显示API详细调试信息
    - 文件：保存所有DEBUG及以上的日志
    """
    file_name = f"logs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) }_{file_name}.log"

    # 移除默认的stdout handler
    logger.remove()

    # 添加终端handler（INFO级别，过滤API调试信息和错误详情）
    def terminal_filter(record):
        """过滤终端输出：只显示关键信息和进度"""
        # 过滤DEBUG和ERROR级别（ERROR只记录到文件）
        if record["level"].name in ["DEBUG", "ERROR"]:
            return False

        # 过滤特定模块的详细日志
        # 检查logger的name（格式：module:function 或 __main__）
        logger_name = record.get("name", "")

        # 过滤包含大量API详细信息的特定消息
        message = str(record.get("message", ""))

        # 过滤API详细信息
        api_keywords = ["API 响应:", "完整响应对象:", "Prompt 前100字符:",
                       "响应 choices", "批量处理: 任务", "批量解析:"]
        if any(keyword in message for keyword in api_keywords):
            return False

        # 过滤批次失败的详细信息（保留最终统计）
        if "⚠️  批次" in message and "处理失败" in message:
            return False

        return True

    logger.add(
        sys.stdout,
        level="INFO",
        filter=terminal_filter,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # 添加文件handler（保存所有DEBUG及以上）
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
