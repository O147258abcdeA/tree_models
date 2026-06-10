"""统一日志模块：每个模块独立 logger，控制台 + 可选文件输出。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured: set[str] = set()


def get_logger(name: str, log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """返回带控制台 handler（可选文件 handler）的 logger。

    Parameters
    ----------
    name : logger 名称（建议用模块名）。
    log_file : 可选日志文件路径。
    level : 日志级别。
    """
    logger = logging.getLogger(name)
    if name in _configured:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(fh)

    _configured.add(name)
    return logger
