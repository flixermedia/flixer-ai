import asyncio
from datetime import datetime, timezone

from flixer_ai.recommendation_cache import RecommendationCache
from flixer_ai.schemas import RecommendationResponse, ScoredItem


class FakeCacheStore:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, object]] = {}

    async def load_recommendation_cache(self, user_id: str) -> dict[str, object]:
        return self.states.get(user_id, {"entries": {}, "lastRefreshAt": None})

    async def save_recommendation_cache(self, user_id: str, state: dict[str, object]) -> None:
        self.states[user_id] = state

    async def clear_recommendation_cache(self) -> None:
        self.states.clear()


def test_cached_results_refresh_in_background_at_most_once_per_interval() -> None:
    async def run() -> None:
        now = 1_000.0
        cache = RecommendationCache(refresh_interval_minutes=1, clock=lambda: now)
        first = RecommendationResponse(items=[ScoredItem(id="first", score=0.8)])
        refreshed = RecommendationResponse(items=[ScoredItem(id="refreshed", score=0.9)])
        calls = 0
        refresh_started = asyncio.Event()
        allow_refresh = asyncio.Event()

        async def create_first() -> RecommendationResponse:
            nonlocal calls
            calls += 1
            return first

        async def create_refreshed() -> RecommendationResponse:
            nonlocal calls
            calls += 1
            refresh_started.set()
            await allow_refresh.wait()
            return refreshed

        assert await cache.get_or_create("user", "query", create_first) == first
        assert await cache.get_or_create("user", "query", create_refreshed) == first
        assert calls == 1

        now += 60
        assert await cache.get_or_create("user", "query", create_refreshed) == first
        await refresh_started.wait()
        assert calls == 2
        allow_refresh.set()
        while cache.status["refreshesInProgress"]:
            await asyncio.sleep(0)

        assert await cache.get_or_create("user", "query", create_refreshed) == refreshed
        assert calls == 2

        refresh_started.clear()
        now += 60
        assert await cache.get_or_create("user", "query", create_refreshed) == refreshed
        await refresh_started.wait()
        assert calls == 3
        await cache.close()

    asyncio.run(run())


def test_failed_background_refresh_keeps_the_last_cached_results() -> None:
    async def run() -> None:
        now = 1_000.0
        cache = RecommendationCache(refresh_interval_minutes=1, clock=lambda: now)
        original = RecommendationResponse(items=[ScoredItem(id="cached", score=0.7)])

        async def create_original() -> RecommendationResponse:
            return original

        async def fail_refresh() -> RecommendationResponse:
            raise RuntimeError("temporary upstream failure")

        assert await cache.get_or_create("user", "query", create_original) == original
        now += 60
        assert await cache.get_or_create("user", "query", fail_refresh) == original
        while cache.status["refreshesInProgress"]:
            await asyncio.sleep(0)

        async def unexpected_refresh() -> RecommendationResponse:
            raise AssertionError("refresh should be throttled after the failed attempt")

        assert await cache.get_or_create("user", "query", unexpected_refresh) == original
        await cache.close()

    asyncio.run(run())


def test_cache_interval_is_bounded() -> None:
    cache = RecommendationCache()
    for invalid in (0, 1441, True):
        try:
            cache.set_refresh_interval(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid interval {invalid!r} to be rejected.")


def test_failed_initial_refresh_does_not_block_an_immediate_retry() -> None:
    async def run() -> None:
        cache = RecommendationCache(refresh_interval_minutes=1, clock=lambda: 1_000.0)
        calls = 0

        async def fail() -> RecommendationResponse:
            nonlocal calls
            calls += 1
            raise RuntimeError("temporary upstream failure")

        try:
            await cache.get_or_create("user", "query", fail)
        except RuntimeError as error:
            assert str(error) == "temporary upstream failure"
        else:
            raise AssertionError("The initial recommendation failure should be returned to the caller.")

        try:
            await cache.get_or_create("user", "different-query", fail)
        except RuntimeError as error:
            assert str(error) == "temporary upstream failure"
        else:
            raise AssertionError("The retry should attempt a fresh recommendation request.")
        assert calls == 2
        await cache.close()

    asyncio.run(run())


def test_missing_database_cache_is_not_repopulated_until_generation_succeeds() -> None:
    async def run() -> None:
        cache = RecommendationCache()
        store = FakeCacheStore()
        old = RecommendationResponse(items=[ScoredItem(id="old", score=0.7)])
        fresh = RecommendationResponse(items=[ScoredItem(id="fresh", score=0.9)])
        started = asyncio.Event()
        release = asyncio.Event()

        async def original() -> RecommendationResponse:
            return old

        async def generate() -> RecommendationResponse:
            started.set()
            await release.wait()
            return fresh

        await cache.get_or_create("user", "a" * 64, original, store, "movies")
        store.states.clear()
        task = asyncio.create_task(cache.force_refresh("user", "a" * 64, generate, store, "movies", persist_before=False))
        await started.wait()
        assert "user" not in store.states
        release.set()
        assert await task == fresh
        assert store.states["user"]["entries"]["a" * 64]["items"][0]["id"] == "fresh"

        store.states.clear()

        async def fail() -> RecommendationResponse:
            raise RuntimeError("failed")

        try:
            await cache.force_refresh("user", "a" * 64, fail, store, "movies", persist_before=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected generation failure")
        assert "user" not in store.states
        await cache.close()

    asyncio.run(run())


def test_persisted_cache_and_user_throttle_survive_service_restart() -> None:
    async def run() -> None:
        now = 1_000_000.0
        store = FakeCacheStore()
        cached = RecommendationResponse(items=[ScoredItem(id="persisted", score=0.8)])
        calls = 0

        async def create() -> RecommendationResponse:
            nonlocal calls
            calls += 1
            return cached

        first_process = RecommendationCache(refresh_interval_minutes=1, clock=lambda: now)
        assert await first_process.get_or_create("user", "a" * 64, create, store) == cached
        await first_process.close()

        restarted_process = RecommendationCache(refresh_interval_minutes=1, clock=lambda: now + 10)
        assert await restarted_process.get_or_create("user", "a" * 64, create, store) == cached
        assert await restarted_process.get_or_create("user", "b" * 64, create, store) == cached
        assert calls == 1
        await restarted_process.close()

    asyncio.run(run())


def test_cache_miss_for_another_query_serves_the_users_persisted_results() -> None:
    async def run() -> None:
        now = 1_000_000.0
        store = FakeCacheStore()
        cached = RecommendationResponse(items=[ScoredItem(id="cached", score=0.8)])
        store.states["user"] = {
            "entries": {"a" * 64: cached.model_dump(mode="json")},
            "lastRefreshAt": datetime.fromtimestamp(now - 30, timezone.utc).isoformat(),
        }
        cache = RecommendationCache(refresh_interval_minutes=10, clock=lambda: now)

        async def should_not_refresh() -> RecommendationResponse:
            raise AssertionError("A different query should reuse cached user results inside the throttle window.")

        assert await cache.get_or_create("user", "b" * 64, should_not_refresh, store, "movies") == cached
        await cache.close()

    asyncio.run(run())


def test_music_and_screen_results_use_separate_persisted_cache_scopes() -> None:
    async def run() -> None:
        now = 1_000_000.0
        store = FakeCacheStore()
        cache = RecommendationCache(refresh_interval_minutes=10, clock=lambda: now)
        movie_result = RecommendationResponse(items=[ScoredItem(id="movie-id", score=0.8)])
        music_result = RecommendationResponse(items=[ScoredItem(id="album-id", score=0.9)])

        async def create_movie() -> RecommendationResponse:
            return movie_result

        async def create_music() -> RecommendationResponse:
            return music_result

        assert await cache.get_or_create("user", "a" * 64, create_movie, store, "movies") == movie_result
        assert await cache.get_or_create("user", "b" * 64, create_music, store, "music") == music_result
        await cache.close()

        restarted_cache = RecommendationCache(refresh_interval_minutes=10, clock=lambda: now + 30)

        async def should_not_refresh() -> RecommendationResponse:
            raise AssertionError("Music requests should reuse only music recommendations.")

        assert await restarted_cache.get_or_create("user", "c" * 64, should_not_refresh, store, "music") == music_result
        state = store.states["user"]
        assert state["entryScopes"] == {"a" * 64: "movies", "b" * 64: "music"}
        assert set(state["refreshTimes"]) == {"movies", "music"}
        await restarted_cache.close()

    asyncio.run(run())


def test_force_refresh_bypasses_the_user_interval_and_replaces_the_default_set() -> None:
    async def run() -> None:
        now = 1_000.0
        store = FakeCacheStore()
        cache = RecommendationCache(refresh_interval_minutes=10, clock=lambda: now)
        first = RecommendationResponse(items=[ScoredItem(id="old", score=0.6)])
        refreshed = RecommendationResponse(items=[ScoredItem(id="new", score=0.9)])

        async def create_first() -> RecommendationResponse:
            return first

        async def create_refreshed() -> RecommendationResponse:
            return refreshed

        await cache.get_or_create("user", "a" * 64, create_first, store)
        result = await cache.force_refresh("user", "a" * 64, create_refreshed, store)

        assert result == refreshed
        assert store.states["user"]["entries"]["a" * 64] == refreshed.model_dump(mode="json")
        await cache.close()

    asyncio.run(run())
