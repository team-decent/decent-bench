from __future__ import annotations

import logging
from abc import abstractmethod
from logging import LogRecord
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.managers import SyncManager
from typing import Protocol

from rich.logging import RichHandler

LOGGER = logging.getLogger()


class LogQueue(Protocol):
    """A minimal protocol for queue-like objects used by logging handlers."""

    @abstractmethod
    def put_nowait(self, item: LogRecord, /) -> None:
        """Put item."""

    @abstractmethod
    def get(self) -> LogRecord:
        """Get item."""


def start_log_listener(manager: SyncManager, log_level: int) -> QueueListener:
    """
    Start listener thread which can receive log messages through a queue.

    Args:
        manager: used to create a log queue that can be shared across processes
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Returns:
        :class:`~logging.handlers.QueueListener` which can be used to access the log queue and to stop the listener
        thread

    """
    log_queue = manager.Queue()
    log_listener = QueueListener(log_queue, RichHandler(level=log_level), respect_handler_level=True)
    log_listener.start()
    start_queue_logger(log_listener.queue)
    return log_listener


def start_queue_logger(queue: LogQueue) -> None:
    """Configure the default logger for the current process to put log messages in the *queue*."""
    logging.basicConfig(level=logging.NOTSET, format="%(message)s", handlers=[QueueHandler(queue)])
