from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .schemas import MediaItem

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in ("name", "title", "character", "role"):
            part = value.get(key)
            if isinstance(part, str):
                yield part
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _values(item)


def text_for(item: MediaItem) -> str:
    fields = [
        item.title,
        item.overview or "",
        item.tagline or "",
        item.artist or "",
        item.parent_title or "",
        " ".join(item.genres),
        " ".join(item.parent_genres),
        " ".join(item.studios),
        " ".join(_values(item.cast)),
        " ".join(_values(item.crew)),
    ]
    return " ".join(fields)


def tokens_for(item: MediaItem) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text_for(item)) if len(token) > 1]
