"""
utils/logger.py
---------------
Central logging setup used by every module.

Usage
-----
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Return a logger with console + optional rotating-file handler.

    Parameters
    ----------
    name     : typically __name__ of the calling module.
    log_file : optional path to write logs to disk.
                If None, uses LOG_DIR/pipeline.log from config.
    """
    try:
        import config
        level      = getattr(logging, config.LOG_LEVEL, logging.INFO)
        fmt        = config.LOG_FORMAT
        log_dir    = config.LOG_DIR
        default_log = os.path.join(log_dir, "pipeline.log")
    except Exception:
        level       = logging.INFO
        fmt         = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        default_log = "pipeline.log"

    logger = logging.getLogger(name)

    # Avoid adding handlers repeatedly (e.g. during pytest)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        path = log_file or default_log
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass   # never crash just because file logging failed

    return logger