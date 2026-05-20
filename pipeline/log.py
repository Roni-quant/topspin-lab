"""Shared structured JSONL logging for all pipeline stages."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOG_DIR, ensure_dirs


class JsonlHandler(logging.Handler):
    """Logging handler that appends one JSON object per line to a file.

    Keeps the file handle open for the handler's lifetime to avoid
    repeated open/close syscalls.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._file = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": getattr(record, "stage", record.name),
            "entity_type": getattr(record, "entity_type", None),
            "entity_id": getattr(record, "entity_id", None),
            "status": getattr(record, "status", record.levelname.lower()),
            "reason": getattr(record, "reason", None),
            "detail": record.getMessage(),
        }
        try:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass
        super().close()


_CONSOLE_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"


def setup_stage_logger(stage_name: str) -> logging.Logger:
    """Create a logger with console output and a JSONL file handler.

    Log file: logs/{stage_name}_{YYYY-MM-DD}.jsonl
    """
    ensure_dirs()

    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    logger.addHandler(console)

    # JSONL file handler
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"{stage_name}_{today}.jsonl"
    jsonl = JsonlHandler(log_path)
    jsonl.setLevel(logging.DEBUG)
    logger.addHandler(jsonl)

    return logger


def log_structured(
    logger: logging.Logger,
    level: int,
    msg: str,
    **extras,
) -> None:
    """Log a message with structured fields attached to the record."""
    record = logger.makeRecord(
        logger.name, level, "(pipeline)", 0, msg, (), None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    logger.handle(record)
