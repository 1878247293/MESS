"""
loguru logging wrapper.

init_logger attaches two handlers: the terminal outputs only INFO+ and suppresses API debug noise, while
the file keeps the full DEBUG output. `log` / `log_args` / `log_time` are thin INFO-level wrappers.
"""

import time
import sys

from loguru import logger


def init_logger(file_name):
    """
    Initialize loguru.
    The console shows INFO and above; the file keeps the full DEBUG output.
    """
    file_name = f"logs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) }_{file_name}.log"

    # remove the default stdout handler
    logger.remove()

    # the console only wants key progress, so filter out noise such as API debug logs
    def terminal_filter(record):
        # DEBUG is too granular; ERROR is written to file only
        if record["level"].name in ["DEBUG", "ERROR"]:
            return False

        logger_name = record.get("name", "")

        message = str(record.get("message", ""))

        # suppress API-detail logs
        api_keywords = ["API response:", "Full response object:", "Prompt first 100 chars:",
                       "Response choices", "Batch processing: task", "Batch parsing:"]
        if any(keyword in message for keyword in api_keywords):
            return False

        # do not print per-batch failure details to the screen either; show only the summary
        if "⚠️  batch" in message and "processing failed" in message:
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
