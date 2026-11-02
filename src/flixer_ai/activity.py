from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


class ActivityFeed(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._events: deque[dict[str, Any]] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("flixer_ai"):
            return
        try:
            message = record.getMessage()
        except Exception:
            self.handleError(record)
            return
        with self._lock:
            self._events.append(
                {
                    "id": self._next_id,
                    "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                    "level": record.levelname.lower(),
                    "message": message,
                }
            )
            self._next_id += 1

    def read(self) -> dict[str, Any]:
        with self._lock:
            return {"events": list(self._events), "nextId": self._next_id}


activity_feed = ActivityFeed()
ai_logger = logging.getLogger("flixer_ai")
ai_logger.setLevel(logging.INFO)
ai_logger.addHandler(activity_feed)
