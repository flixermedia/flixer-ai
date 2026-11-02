from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Protocol

from pydantic import ValidationError

from .schemas import RecommendationResponse

logger = logging.getLogger("flixer_ai.cache")

class RecommendationCacheStore(Protocol):
    async def load_recommendation_cache(self, user_id: str) -> dict[str, object]: ...

    async def save_recommendation_cache(self, user_id: str, state: dict[str, object]) -> None: ...

    async def clear_recommendation_cache(self) -> None: ...


class RecommendationRefreshRateLimited(RuntimeError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Recommendation results are being refreshed. Try again after the configured refresh interval.")
        self.retry_after_seconds = retry_after_seconds


class RecommendationCache:
    def __init__(self, refresh_interval_minutes: int = 10, clock: Callable[[], float] = time.time) -> None:
        self._refresh_interval_seconds = refresh_interval_minutes * 60
        self._entries: dict[tuple[str, str], RecommendationResponse] = {}
        self._entry_scopes: dict[tuple[str, str], str] = {}
        self._last_refresh_by_user: dict[tuple[str, str], float] = {}
        self._refresh_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._loaded_users: set[str] = set()
        self._load_tasks: dict[str, asyncio.Task[None]] = {}
        self._store_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self._clock = clock

    @property
    def refresh_interval_minutes(self) -> int:
        return self._refresh_interval_seconds // 60

    @property
    def status(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "refreshesInProgress": len(self._refresh_tasks),
            "refreshIntervalMinutes": self.refresh_interval_minutes,
        }

    def set_refresh_interval(self, minutes: int) -> None:
        if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= 1440:
            raise ValueError("Recommendation cache refresh interval must be between 1 and 1440 minutes.")
        self._refresh_interval_seconds = minutes * 60

    async def get_or_create(
        self,
        user_id: str,
        request_key: str,
        create: Callable[[], Awaitable[RecommendationResponse]],
        store: RecommendationCacheStore | None = None,
        scope: str = "general",
    ) -> RecommendationResponse:
        await self._ensure_loaded(user_id, store)
        key = (user_id, request_key)
        task_to_wait: asyncio.Task[None] | None = None
        async with self._lock:
            cached = self._entries.get(key)
            task = self._refresh_tasks.get(key)
            if cached is not None:
                last_refresh = self._last_refresh_by_user.get((user_id, scope), 0)
                if task is None and self._clock() - last_refresh >= self._refresh_interval_seconds:
                    self._schedule_refresh(key, create, store, scope, preserve_cache=True)
                return cached

            task_to_wait = next(
                (
                    refresh
                    for (entry_user_id, entry_key), refresh in self._refresh_tasks.items()
                    if entry_user_id == user_id and self._entry_scopes.get((entry_user_id, entry_key)) == scope
                ),
                None,
            )
            if task_to_wait is not None:
                task = task_to_wait
            else:
                cached_for_user = self._best_cached_for_user(user_id, scope)
                last_refresh = self._last_refresh_by_user.get((user_id, scope))
                if cached_for_user is not None:
                    if last_refresh is None or self._clock() - last_refresh >= self._refresh_interval_seconds:
                        self._schedule_refresh(key, create, store, scope, preserve_cache=True)
                    return cached_for_user

            if task is None:
                if last_refresh is not None:
                    retry_after = self._refresh_interval_seconds - int(self._clock() - last_refresh)
                    if retry_after > 0:
                        raise RecommendationRefreshRateLimited(retry_after)
                self._schedule_refresh(key, create, store, scope, preserve_cache=False)
                task = self._refresh_tasks[key]

        await task
        async with self._lock:
            if task_to_wait is not None:
                cached_for_user = self._best_cached_for_user(user_id, scope)
                if cached_for_user is not None:
                    return cached_for_user
            return self._entries[key]

    async def force_refresh(
        self,
        user_id: str,
        request_key: str,
        create: Callable[[], Awaitable[RecommendationResponse]],
        store: RecommendationCacheStore | None = None,
        scope: str = "general",
    ) -> RecommendationResponse:
        await self._ensure_loaded(user_id, store)
        key = (user_id, request_key)
        while True:
            async with self._lock:
                task = self._refresh_tasks.get(key)
                scheduled = task is None
                if task is None:
                    self._schedule_refresh(key, create, store, scope, preserve_cache=False)
                    task = self._refresh_tasks[key]
            await task
            if scheduled:
                async with self._lock:
                    return self._entries[key]

    async def _ensure_loaded(self, user_id: str, store: RecommendationCacheStore | None) -> None:
        async with self._lock:
            if user_id in self._loaded_users:
                return
            task = self._load_tasks.get(user_id)
            if task is None:
                task = asyncio.create_task(self._load_user(user_id, store))
                self._load_tasks[user_id] = task
        await task

    async def _load_user(self, user_id: str, store: RecommendationCacheStore | None) -> None:
        try:
            state = await store.load_recommendation_cache(user_id) if store is not None else {}
            entries = state.get("entries", {})
            last_refresh_at = state.get("lastRefreshAt")
            entry_scopes = state.get("entryScopes", {})
            refresh_times = state.get("refreshTimes", {})
            if not isinstance(entries, dict):
                raise RuntimeError("Flixer returned invalid recommendation cache entries.")
            if not isinstance(entry_scopes, dict) or not isinstance(refresh_times, dict):
                raise RuntimeError("Flixer returned invalid recommendation cache metadata.")
            if last_refresh_at is not None and not isinstance(last_refresh_at, str):
                raise RuntimeError("Flixer returned an invalid recommendation cache refresh time.")
            async with self._lock:
                for request_key, response in entries.items():
                    if not isinstance(request_key, str):
                        raise RuntimeError("Flixer returned an invalid recommendation cache key.")
                    try:
                        self._entries[(user_id, request_key)] = RecommendationResponse.model_validate(response)
                    except ValidationError as error:
                        raise RuntimeError("Flixer returned an invalid recommendation cache response.") from error
                    scope = entry_scopes.get(request_key, "movies")
                    if scope not in {"movies", "music", "general"}:
                        raise RuntimeError("Flixer returned an invalid recommendation cache scope.")
                    self._entry_scopes[(user_id, request_key)] = scope
                if refresh_times:
                    for scope, refreshed_at in refresh_times.items():
                        if scope not in {"movies", "music", "general"} or not isinstance(refreshed_at, str):
                            raise RuntimeError("Flixer returned invalid recommendation cache refresh times.")
                        try:
                            timestamp = datetime.fromisoformat(refreshed_at.replace("Z", "+00:00")).timestamp()
                        except ValueError as error:
                            raise RuntimeError("Flixer returned an invalid recommendation cache refresh time.") from error
                        self._last_refresh_by_user[(user_id, scope)] = timestamp
                elif last_refresh_at is not None:
                    try:
                        timestamp = datetime.fromisoformat(last_refresh_at.replace("Z", "+00:00")).timestamp()
                    except ValueError as error:
                        raise RuntimeError("Flixer returned an invalid recommendation cache refresh time.") from error
                    self._last_refresh_by_user[(user_id, "movies")] = timestamp
                self._loaded_users.add(user_id)
        finally:
            async with self._lock:
                if self._load_tasks.get(user_id) is asyncio.current_task():
                    self._load_tasks.pop(user_id)

    def _schedule_refresh(
        self,
        key: tuple[str, str],
        create: Callable[[], Awaitable[RecommendationResponse]],
        store: RecommendationCacheStore | None,
        scope: str,
        *,
        preserve_cache: bool,
    ) -> None:
        self._last_refresh_by_user[(key[0], scope)] = self._clock()
        self._entry_scopes[key] = scope
        self._refresh_tasks[key] = asyncio.create_task(self._populate(key, create, store, scope, preserve_cache))

    async def _populate(
        self,
        key: tuple[str, str],
        create: Callable[[], Awaitable[RecommendationResponse]],
        store: RecommendationCacheStore | None,
        scope: str,
        preserve_cache: bool,
    ) -> None:
        try:
            await self._persist_user(key[0], store)
            result = await create()
            async with self._lock:
                self._entries[key] = result
            await self._save_user(key[0], store)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Could not refresh recommendations for user %s.", key[0])
            if not preserve_cache:
                async with self._lock:
                    self._last_refresh_by_user.pop((key[0], scope), None)
                    if key not in self._entries:
                        self._entry_scopes.pop(key, None)
                try:
                    await self._save_user(key[0], store)
                except Exception:
                    logger.exception("Could not clear failed refresh throttle for user %s.", key[0])
                raise
        finally:
            async with self._lock:
                if self._refresh_tasks.get(key) is asyncio.current_task():
                    self._refresh_tasks.pop(key)

    async def _persist_user(self, user_id: str, store: RecommendationCacheStore | None) -> None:
        if store is None:
            return
        await self._save_user(user_id, store)

    async def _save_user(
        self,
        user_id: str,
        store: RecommendationCacheStore | None,
    ) -> None:
        if store is None:
            return
        store_lock = self._store_locks.setdefault(user_id, asyncio.Lock())
        async with store_lock:
            async with self._lock:
                entries = self._user_entries(user_id)
                refresh_times = {
                    scope: datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                    for (entry_user_id, scope), timestamp in self._last_refresh_by_user.items()
                    if entry_user_id == user_id
                }
                entry_scopes = {
                    key: scope
                    for (entry_user_id, key), scope in self._entry_scopes.items()
                    if entry_user_id == user_id
                }
            last_refresh_at = max(refresh_times.values(), default=None)
            await store.save_recommendation_cache(
                user_id,
                {
                    "entries": {key: response.model_dump(mode="json") for key, response in entries.items()},
                    "lastRefreshAt": last_refresh_at,
                    "entryScopes": entry_scopes,
                    "refreshTimes": refresh_times,
                },
            )

    def _user_entries(self, user_id: str) -> dict[str, RecommendationResponse]:
        return {key: response for (entry_user_id, key), response in self._entries.items() if entry_user_id == user_id}

    def _best_cached_for_user(self, user_id: str, scope: str) -> RecommendationResponse | None:
        entries = {
            key: response
            for (entry_user_id, key), response in self._entries.items()
            if entry_user_id == user_id and self._entry_scopes.get((entry_user_id, key), "movies") == scope
        }
        return max(entries.values(), key=lambda response: len(response.items), default=None)

    async def clear(self, store: RecommendationCacheStore | None = None) -> None:
        async with self._lock:
            tasks = list(self._refresh_tasks.values())
            self._refresh_tasks.clear()
            self._entries.clear()
            self._entry_scopes.clear()
            self._last_refresh_by_user.clear()
            self._loaded_users.clear()
            load_tasks = list(self._load_tasks.values())
            self._load_tasks.clear()
            for task in tasks:
                task.cancel()
            for task in load_tasks:
                task.cancel()
        if tasks or load_tasks:
            await asyncio.gather(*tasks, *load_tasks, return_exceptions=True)
        if store is not None:
            await store.clear_recommendation_cache()

    async def close(self) -> None:
        async with self._lock:
            tasks = list(self._refresh_tasks.values())
            self._refresh_tasks.clear()
            load_tasks = list(self._load_tasks.values())
            self._load_tasks.clear()
            for task in tasks:
                task.cancel()
            for task in load_tasks:
                task.cancel()
        if tasks or load_tasks:
            await asyncio.gather(*tasks, *load_tasks, return_exceptions=True)
