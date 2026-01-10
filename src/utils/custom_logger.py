from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Union
from uuid import uuid4

#%%
# -------------------------
# Log directory (repo-root)
# -------------------------
def _default_log_dir() -> Path:
    # core/utils/custom_logger.py -> utils -> core -> repo root
    try:
        return Path(__file__).resolve().parents[2] / "logs"
    except Exception:
        return Path("logs")

# -------------------------
# Run id support
# -------------------------
_RUN_ID: str = "-"  # default when not set


def new_run_id() -> str:
    """Generate a short run id for correlating logs of a single execution."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid4().hex[:8]}"


def set_run_id(run_id: str) -> None:
    """Set the current run id that will be injected into all log lines."""
    global _RUN_ID
    _RUN_ID = (run_id or "-").strip() or "-"


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure formatter never breaks if run_id wasn't set
        setattr(record, "run_id", _RUN_ID)
        return True


def _ensure_run_id_filter(logger: logging.Logger) -> None:
    # Add filter to logger
    if not any(isinstance(f, _RunIdFilter) for f in logger.filters):
        logger.addFilter(_RunIdFilter())

    # Add filter to handlers too (covers edge cases)
    for h in logger.handlers:
        if not any(isinstance(f, _RunIdFilter) for f in getattr(h, "filters", [])):
            h.addFilter(_RunIdFilter())


def get_logger(
    name: str,
    log_to_console: bool = False,
    level: Union[int, str] = logging.INFO,
    use_timestamp: bool = False,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> logging.Logger:
    """
    Create and return a logger.

    - Logs to repo-root /logs
    - Injects run_id into every line (set via set_run_id())
    - Safe to call multiple times (won't duplicate handlers)
    """

    LOG_DIR = _default_log_dir()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Convert string levels like "DEBUG" → logging.DEBUG
    if isinstance(level, str):
        level_str = level.upper()
        if level_str not in logging._nameToLevel:
            raise ValueError(
                f"Invalid log level: {level_str}. "
                f"Valid levels are: {list(logging._nameToLevel.keys())}"
            )
        level = logging._nameToLevel[level_str]

    logger.setLevel(level)
    logger.propagate = False  # avoid duplicates via root handlers

    file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s")
    console_fmt = logging.Formatter("%(asctime)s | %(levelname)s | run=%(run_id)s | %(message)s")

    if not logger.handlers:
        # Decide log file name (with or without timestamp)
        if use_timestamp:
            ts = datetime.now().strftime(timestamp_fmt)
            logfile = LOG_DIR / f"{name}_{ts}.log"
        else:
            logfile = LOG_DIR / f"{name}.log"

        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(console_fmt)
            logger.addHandler(console_handler)
    else:
        # Update existing handlers: levels + formatters
        for h in logger.handlers:
            h.setLevel(level)
            if isinstance(h, logging.FileHandler):
                h.setFormatter(file_fmt)
            else:
                h.setFormatter(console_fmt)

    _ensure_run_id_filter(logger)
    return logger

#%%
# Example usage:
# logger = get_logger("my_module", log_to_console=True, level="DEBUG", use_timestamp=True)