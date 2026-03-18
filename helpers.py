from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LoggerConfig:
    verbose_log: int = 2
    verbose_print: int = 0
    log_file: str = "log.txt"
    timestamp_format: str = "%m/%d/%Y %H:%M:%S"


CONFIG = LoggerConfig()


def _stringify(message: Any) -> str:
    if isinstance(message, str):
        return message
    return str(message)


def _timestamp() -> str:
    return datetime.now().strftime(CONFIG.timestamp_format)


def _format_file_line(message: str) -> str:
    # return f"{_timestamp()} | {message}\n"
    return f"{message}\n"


def _format_return_line(message: str) -> str:
    return f"{message}\n"


def _write_file(line: str) -> None:
    Path(CONFIG.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG.log_file, "a", encoding="utf-8") as f:
        f.write(line)

def clear_log_file() -> None:
    Path(CONFIG.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG.log_file).write_text("", encoding="utf-8")

def log(message: Any, level: int = 2) -> str:
    """
    Writes a timestamped line to the local log file if enabled,
    prints to console if enabled,
    and returns a plain line for accumulating UI/download logs.

    Example:
        log_record += log("Starting solver")
    """
    text = _stringify(message)
    file_line = _format_file_line(text)
    return_line = _format_return_line(text)

    if CONFIG.verbose_log >= level:
        _write_file(file_line)

    if CONFIG.verbose_print >= level:
        print(f"LOG: {_timestamp()} {text}")

    return return_line