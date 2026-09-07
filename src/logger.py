"""
Structured JSON and console logger with per-execution UUID tracking for FieldSight-Lite.
Provides production-grade logging with execution context, pipeline stage demarcation, and audit logs.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "execution_id": getattr(record, "execution_id", "GLOBAL"),
            "stage": getattr(record, "stage", "SYSTEM"),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_obj["data"] = record.extra_data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class ConsoleFormatter(logging.Formatter):
    """Colored human-readable console formatter for development and CLI runs."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        exec_id = getattr(record, "execution_id", "---")
        if len(exec_id) > 8:
            exec_id = exec_id[:8]
        stage = getattr(record, "stage", "SYS")
        prefix = f"{self.DIM}[{time_str}]{self.RESET} {color}{self.BOLD}[{record.levelname:<5}]{self.RESET} {self.DIM}[{exec_id}|{stage}]{self.RESET}"
        msg = record.getMessage()
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            msg += f" {self.DIM}{record.extra_data}{self.RESET}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return f"{prefix} {msg}"


class FieldSightLoggerAdapter(logging.LoggerAdapter):
    """Context-aware logger adapter injecting execution_id and stage metadata."""

    def __init__(self, logger: logging.Logger, execution_id: Optional[str] = None, stage: str = "PIPELINE"):
        self.execution_id = execution_id or str(uuid.uuid4())
        self.stage = stage
        super().__init__(logger, {"execution_id": self.execution_id, "stage": self.stage})

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = kwargs.setdefault("extra", {})
        extra["execution_id"] = self.execution_id
        extra["stage"] = self.stage
        if "extra_data" in kwargs:
            extra["extra_data"] = kwargs.pop("extra_data")
        return msg, kwargs

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.extra["stage"] = stage


_ROOT_INITIALIZED = False


def setup_root_logger(level: int = logging.INFO, json_output: bool = False) -> None:
    """Configures the root fieldsight logger."""
    global _ROOT_INITIALIZED
    root = logging.getLogger("fieldsight")
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())
    root.addHandler(handler)
    root.propagate = False
    _ROOT_INITIALIZED = True


def get_logger(
    name: str = "fieldsight",
    execution_id: Optional[str] = None,
    stage: str = "SYSTEM",
    json_output: bool = False
) -> FieldSightLoggerAdapter:
    """Returns a structured FieldSightLoggerAdapter instance."""
    global _ROOT_INITIALIZED
    if not _ROOT_INITIALIZED:
        setup_root_logger(logging.INFO, json_output=json_output)
    base_logger = logging.getLogger(name)
    return FieldSightLoggerAdapter(base_logger, execution_id=execution_id, stage=stage)
