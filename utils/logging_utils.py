
from loguru import logger
import sys

def setup_logging(level="INFO", log_file=None):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if log_file:
        logger.add(log_file, rotation="50 MB", retention="14 days", enqueue=True)
    return logger

logger = setup_logging()
