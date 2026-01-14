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
    """
    Return the default directory where log files should be written.

    This helper tries to place logs in a consistent "repo-root / logs" location
    by walking up from this file's path:

        core/utils/custom_logger.py  ->  utils  ->  core  ->  repo root

    If that resolution fails for any reason (e.g., unusual packaging, interactive
    environments, missing __file__), it falls back to a local relative "logs"
    directory.

    Returns
    -------
    pathlib.Path
        Path to the log directory.

    Notes
    -----
    - The directory is not created here; callers typically create it with
      mkdir(parents=True, exist_ok=True).
    - In Databricks or other job runners, relative paths may resolve to the
      working directory of the driver/process.
    """
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
    """
    Generate a short run identifier for correlating logs within one execution.

    The run id is intended to be generated once per "run" (script run, notebook
    execution, job run) and then injected into all log lines via `set_run_id()`.

    Returns
    -------
    str
        A run id string in the form:

            YYYYMMDD_HHMMSS_<8-hex-chars>

        Example:
            "20260112_211108_a1b2c3d4"

    See Also
    --------
    set_run_id : Stores the generated run id so it appears in log records.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid4().hex[:8]}"


def set_run_id(run_id: str) -> None:
    """
    Set the current run id that will be injected into all log lines.

    This updates the module-level run id used by the logging filter. Once set,
    any logger created/returned by `get_logger()` (and any other logger that has
    the filter attached) will include this value in formatted output.

    Parameters
    ----------
    run_id : str
        Run id to use for subsequent log records. If empty/whitespace/None-like,
        a safe default "-" is used.

    Examples
    --------
    >>> rid = new_run_id()
    >>> set_run_id(rid)
    >>> logger = get_logger("spacing")
    >>> logger.info("Starting pipeline")  # shows run=<rid> in logs

    Notes
    -----
    - This is a global setting within the Python process.
    - Call this early in your entrypoint (main/notebook/job) for best results.
    """
    global _RUN_ID
    _RUN_ID = (run_id or "-").strip() or "-"


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure formatter never breaks if run_id wasn't set
        setattr(record, "run_id", _RUN_ID)
        return True
    


def _ensure_run_id_filter(logger: logging.Logger) -> None:
    """
    Ensure the run-id injection filter is attached to a logger and its handlers.

    This function makes logging resilient and consistent by:
    - Adding the `_RunIdFilter` to the provided logger (if not already present).
    - Adding the same filter to each of the logger's handlers (FileHandler,
      StreamHandler, etc.) to cover edge cases where handler-level filtering
      is required.

    Parameters
    ----------
    logger : logging.Logger
        The logger to attach the run-id filter to.

    Notes
    -----
    - Safe to call multiple times (idempotent).
    - The filter ensures `record.run_id` always exists so formatters using
      "%(run_id)s" never break, even if `set_run_id()` was never called.
    """
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
    propagate: bool = False,
) -> logging.Logger:
    """
    Create (or return) a configured logger with consistent formatting and file output.

    This helper standardizes logging across your project by:
    - Writing logs to a consistent logs directory (repo-root / "logs" when possible).
    - Attaching a file handler (and optionally a console handler).
    - Injecting a per-run identifier (`run_id`) into every log line for easy
      correlation across modules/classes during one execution.
    - Avoiding duplicate handlers if `get_logger()` is called multiple times
      with the same name.

    Parameters
    ----------
    name : str
        Logger name. This becomes the logger's identity and is typically shown in
        log output via "%(name)s". Names can be hierarchical using dots, e.g.
        "spacing.geo_processing".
    log_to_console : bool, default False
        If True, also add a StreamHandler that prints logs to stdout/stderr.
    level : int or str, default logging.INFO
        Logging level for the logger and its handlers. If a string is provided,
        it must match standard logging names like "DEBUG", "INFO", "WARNING", etc.
    use_timestamp : bool, default False
        If True, create a timestamped log file name (useful for keeping separate
        run logs without overwriting).
    timestamp_fmt : str, default "%Y%m%d_%H%M%S"
        Timestamp format used when `use_timestamp=True`.

    Other Parameters
    ----------------
    propagate : bool, optional
        If you add this parameter to your function signature:
            propagate: bool = False
        it controls whether this logger propagates records to its parent logger.
        Typical patterns:
        - Parent loggers: propagate=False (to avoid duplicating via root handlers)
        - Child loggers created with `logging.getLogger("parent.child")`:
          propagate=True (so they write into the parent's handlers)

    Returns
    -------
    logging.Logger
        A configured logger instance. Calling `get_logger()` again with the same
        name returns the same logger object (logging is a singleton registry).

    What gets created
    -----------------
    - A file handler writing to:
        logs/<name>.log            (default)
        logs/<name>_<timestamp>.log (if use_timestamp=True)
    - Format includes:
        time | level | logger name | run=<run_id> | message

    Run id workflow
    ---------------
    1) Generate a run id:
        >>> rid = new_run_id()
    2) Set it once per run:
        >>> set_run_id(rid)
    3) Create/use loggers:
        >>> logger = get_logger("spacing")
        >>> logger.info("Starting run")

    Examples
    --------
    Basic: write only to file (logs/spacing.log)
    >>> from core.utils.custom_logger import get_logger
    >>> logger = get_logger("spacing", log_to_console=False, level="INFO")
    >>> logger.info("Hello from spacing")

    File + console output
    >>> logger = get_logger("spacing", log_to_console=True, level="DEBUG")
    >>> logger.debug("This prints to console and goes to logs/spacing.log")

    Timestamped log files (avoid overwriting across runs)
    >>> logger = get_logger("spacing", use_timestamp=True, level="INFO")
    >>> logger.info("This run writes to logs/spacing_<timestamp>.log")

    Setting a run id for correlation
    >>> rid = new_run_id()
    >>> set_run_id(rid)
    >>> logger = get_logger("spacing")
    >>> logger.info("Run-aware log line")

    Multi-module pattern (same logger name across files)
    # module_a.py
    >>> logger = get_logger("spacing")
    >>> logger.info("A: step 1")

    # module_b.py
    >>> logger = get_logger("spacing")
    >>> logger.info("B: step 2")
    # Both write into the same logs/spacing.log without duplicated handlers.

    Child logger pattern (Option A: child routes into parent file)
    - Configure ONLY the parent with a file handler:
    >>> parent = get_logger("spacing", level="INFO")   # typically propagate=False
    - Create child loggers with the hierarchical name:
    >>> child = logging.getLogger("spacing.geo_processing")
    >>> child.propagate = True
    >>> child.info("Converting lat/lon to UTM")
    # Result: written into logs/spacing.log, with logger name "spacing.geo_processing".

    Child logger inside a class (dependency injection)
    >>> class GeoSurveyProcessor:
    ...     def __init__(self, logger=None):
    ...         parent = logger or logging.getLogger("spacing")
    ...         self.logger = logging.getLogger(f"{parent.name}.geo_processing")
    ...         self.logger.propagate = True
    ...     def run(self):
    ...         self.logger.info("Geo step running")

    >>> spacing_logger = get_logger("spacing")
    >>> geo = GeoSurveyProcessor(logger=spacing_logger)
    >>> geo.run()

    Separate logger (independent file) when you truly want isolation
    >>> geo_logger = get_logger("geo_processing", level="INFO")
    >>> geo_logger.info("Writes to logs/geo_processing.log")

    Notes
    -----
    - This helper is designed to be safe to call repeatedly. It will not add
      duplicate handlers to the same named logger.
    - By default, the logger typically sets `propagate=False` to prevent duplicate
      output via the root logger's handlers. For parent/child hierarchies where
      children should flow into the parent, create children using `logging.getLogger()`
      and set `child.propagate=True`.
    - If you later change handler formatting/levels, calling `get_logger()` again
      will update existing handlers' level/formatter to the new settings.
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
    logger.propagate = propagate # Whether to pass logs to ancestor loggers

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