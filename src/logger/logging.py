import logging as pylogging
import sys
from pathlib import Path
from typing import Any


LOGS_DIR = Path.cwd() / "logs"
LOG_FILE_PATH = LOGS_DIR / "agentflow.log"
LOGGER_NAME = "agentflow"
INFO = pylogging.INFO
ERROR = pylogging.ERROR


class AgentFlowFormatter(pylogging.Formatter):
    def format(self, record: pylogging.LogRecord) -> str:
        record.seq_id = getattr(record, "seq_id", "-")
        record.session_id = getattr(record, "session_id", "-")
        record.thread_id = getattr(record, "thread_id", "-")
        return super().format(record)


def _build_handler(handler: pylogging.Handler) -> pylogging.Handler:
    handler.setFormatter(
        AgentFlowFormatter(
            fmt=(
                "[%(asctime)s][%(levelname)s]"
                "[%(filename)s:%(lineno)d:%(funcName)s]"
                "|%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _configure_logger() -> pylogging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = pylogging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(pylogging.INFO)
    logger.propagate = False
    logger.addHandler(_build_handler(pylogging.FileHandler(LOG_FILE_PATH, encoding="utf-8")))
    logger.addHandler(_build_handler(pylogging.StreamHandler(sys.stdout)))
    return logger


def compact_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log_event(
    level: int,
    message: str,
    *,
    seq_id: str = "-",
    session_id: str = "-",
    thread_id: str = "-",
) -> None:
    logging.log(
        level,
        message,
        extra={
            "seq_id": seq_id,
            "session_id": session_id,
            "thread_id": thread_id,
        },
    )


logging = _configure_logger()
