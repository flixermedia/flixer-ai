from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import tempfile
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from .activity import activity_feed
from .config import Settings
from .flixer_client import FlixerClient, secure_control_url, signature
from .model_manager import ProfileManager
from .log_context import LogContextMiddleware
from .logger_export import start_logger, stop_logger
from .logger_enrollment import run as enroll_logger
from .profiles import ProfileError
from .recommendation_cache import RecommendationCache, RecommendationRefreshRateLimited
from .schemas import (
    MediaItem,
    RecommendRequest,
    RecommendationFilters,
    RecommendationResponse,
    SimilarRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("flixer_ai")

MAX_ADMIN_CLOCK_SKEW_SECONDS = 300


def recommendation_scope(filters: RecommendationFilters | None) -> str:
    types = {media_type.casefold() for media_type in (filters.types if filters else []) or []}
    if types and types <= {"album", "track"}:
        return "music"
    if types and types <= {"movie", "series", "episode", "video"}:
        return "movies"
    return "general"


class AdminConfiguration(BaseModel):
    enabled: bool
    profile: Literal["auto", "classic", "micro", "mini"]
    allow_downloads: bool
    cache_refresh_interval_minutes: int = Field(default=10, ge=1, le=1440)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _profile_manager(request: Request) -> ProfileManager:
    return request.app.state.profile_manager


def _flixer_client(request: Request) -> FlixerClient:
    return request.app.state.flixer_client


def _recommendation_cache(request: Request) -> RecommendationCache:
    return request.app.state.recommendation_cache


async def require_internal_auth(
    settings: Annotated[Settings, Depends(_settings)],
    token: Annotated[str | None, Header(alias="X-Flixer-Internal-Token")] = None,
) -> None:
    expected = settings.internal_token
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Internal authentication is not configured.")
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid internal token.")


async def require_signed_admin(
    request: Request,
    settings: Annotated[Settings, Depends(_settings)],
    timestamp: Annotated[str | None, Header(alias="X-Flixer-AI-Timestamp")] = None,
    provided_signature: Annotated[str | None, Header(alias="X-Flixer-AI-Signature")] = None,
) -> None:
    secret = settings.internal_token
    try:
        issued_at = int(timestamp or "")
    except ValueError:
        issued_at = 0
    if not secret or not provided_signature or abs(int(time.time()) - issued_at) > MAX_ADMIN_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid AI admin signature.")
    body = await request.body()
    expected = signature(secret, str(issued_at), request.method, request.url.path, body)
    if not secrets.compare_digest(provided_signature, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid AI admin signature.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        settings: Settings = app.state.settings
        client = FlixerClient(settings)
        stack.push_async_callback(client.close)
        manual_logger_config = Path("/run/flixer-logger/ai.json")
        if not manual_logger_config.exists() and not manual_logger_config.is_symlink() and (settings.logger_control_socket or secure_control_url(settings.flixer_url)):
            try:
                directory = tempfile.TemporaryDirectory(prefix="flixer-ai-logger-")
            except OSError as error:
                logger.warning("Cannot initialize logger enrollment (%s); export is disabled.", type(error).__name__)
                stack.callback(stop_logger, start_logger())
            else:
                stack.callback(directory.cleanup)
                config_path = Path(directory.name) / "config.json"
                stack.callback(stop_logger, start_logger(config_path))
                enrollment = asyncio.create_task(enroll_logger(client, config_path))

                async def stop_enrollment() -> None:
                    enrollment.cancel()
                    try:
                        await enrollment
                    except asyncio.CancelledError:
                        pass

                stack.push_async_callback(stop_enrollment)
        else:
            stack.callback(stop_logger, start_logger())
        try:
            config = await client.runtime_configuration()
            profile = config.get("profile")
            enabled = config.get("enabled")
            downloads = config.get("allowDownloads")
            refresh_interval = config.get("cacheRefreshIntervalMinutes", 10)
            if (
                profile not in {"auto", "classic", "micro", "mini"}
                or not isinstance(enabled, bool)
                or not isinstance(downloads, bool)
                or not isinstance(refresh_interval, int)
                or isinstance(refresh_interval, bool)
                or not 1 <= refresh_interval <= 1440
            ):
                raise RuntimeError("Flixer returned invalid AI runtime settings.")
            settings = replace(settings, enabled=enabled, recommender_profile=profile, allow_downloads=downloads)
            app.state.settings = settings
            app.state.recommendation_cache.set_refresh_interval(refresh_interval)
            logger.info("Loaded runtime AI configuration from Flixer.")
        except RuntimeError as error:
            logger.warning("Could not load runtime AI configuration from Flixer; using environment defaults: %s", error)
        profile_manager = ProfileManager(settings)
        app.state.profile_manager = profile_manager
        app.state.flixer_client = client
        profile_manager.start()
        stack.push_async_callback(profile_manager.stop)
        stack.push_async_callback(app.state.recommendation_cache.close)
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Flixer AI", version=version("flixer-ai"), lifespan=lifespan)
    app.add_middleware(LogContextMiddleware)
    app.state.settings = settings or Settings.from_env()
    app.state.recommendation_cache = RecommendationCache()

    @app.get("/healthz")
    async def healthz(manager: Annotated[ProfileManager, Depends(_profile_manager)]):
        return {
            "status": "ok",
            "activeProfile": manager.active_profile.name,
            "modelReady": manager.model_manager.model_ready,
        }

    @app.get("/status")
    async def get_status(manager: Annotated[ProfileManager, Depends(_profile_manager)]):
        return manager.status

    @app.get("/admin/status", dependencies=[Depends(require_signed_admin)])
    async def admin_status(
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        recommendation_cache: Annotated[RecommendationCache, Depends(_recommendation_cache)],
    ):
        return {**manager.status, "recommendationCache": recommendation_cache.status}

    @app.get("/admin/events", dependencies=[Depends(require_signed_admin)])
    async def admin_events():
        return activity_feed.read()

    @app.post("/admin/config", dependencies=[Depends(require_signed_admin)])
    async def update_admin_configuration(
        config: Annotated[AdminConfiguration, Body()],
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        recommendation_cache: Annotated[RecommendationCache, Depends(_recommendation_cache)],
        client: Annotated[FlixerClient, Depends(_flixer_client)],
    ):
        profile_configuration_changed = (
            manager.settings.enabled != config.enabled
            or manager.requested_profile != config.profile
            or manager.settings.allow_downloads != config.allow_downloads
        )
        if profile_configuration_changed:
            await manager.configure(config.enabled, config.profile, config.allow_downloads)
            await recommendation_cache.clear(client)
        recommendation_cache.set_refresh_interval(config.cache_refresh_interval_minutes)
        return {
            "accepted": True,
            "status": manager.status,
            "recommendationCache": recommendation_cache.status,
        }

    @app.post("/admin/retry", dependencies=[Depends(require_signed_admin)])
    async def retry_model(
        _: Annotated[dict[str, object], Body()],
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        recommendation_cache: Annotated[RecommendationCache, Depends(_recommendation_cache)],
        client: Annotated[FlixerClient, Depends(_flixer_client)],
    ):
        try:
            await manager.retry()
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        await recommendation_cache.clear(client)
        return {"accepted": True, "status": manager.status}

    async def generate_recommendations(
        user_id: UUID,
        limit: int,
        filters: RecommendationFilters | None,
        manager: ProfileManager,
        client: FlixerClient,
    ) -> RecommendationResponse:
        try:
            context = await client.recommendation_context(user_id)
        except RuntimeError as error:
            logger.warning("Could not load recommendation context for user %s: %s", user_id, error)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not load recommendation data from Flixer.") from error

        profile = manager.active_profile
        inference_started = time.monotonic()
        try:
            items = await asyncio.to_thread(
                profile.recommend,
                context.history,
                context.favorites,
                context.candidates,
                limit,
                filters,
            )
        except ProfileError as error:
            await manager.use_classic_after_error(error)
            items = await asyncio.to_thread(
                manager.active_profile.recommend,
                context.history,
                context.favorites,
                context.candidates,
                limit,
                filters,
            )
        logger.info(
            "Recommendation refresh completed with %d results for user %s.", len(items), user_id,
            extra={"duration_ms": round((time.monotonic() - inference_started) * 1000, 3),
                   "model_profile": manager.active_profile.name, "result_count": len(items)},
        )
        return RecommendationResponse(items=items)

    @app.post("/admin/users/{user_id}/recommendations/refresh", dependencies=[Depends(require_signed_admin)])
    async def force_refresh_recommendations(
        user_id: UUID,
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        client: Annotated[FlixerClient, Depends(_flixer_client)],
        recommendation_cache: Annotated[RecommendationCache, Depends(_recommendation_cache)],
    ):
        if not manager.settings.enabled:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI recommendations are disabled.")

        request = RecommendRequest(
            user_id=user_id,
            limit=20,
            filters=RecommendationFilters(types=["movie", "series", "video"]),
        )
        request_key = hashlib.sha256(request.model_dump_json(exclude={"user_id"}, exclude_none=True).encode("utf-8")).hexdigest()

        async def create() -> RecommendationResponse:
            logger.info("Administrator forced recommendation refresh for user %s.", user_id)
            return await generate_recommendations(user_id, request.limit, request.filters, manager, client)

        try:
            return await recommendation_cache.force_refresh(str(user_id), request_key, create, client, "movies")
        except RuntimeError as error:
            logger.warning("Could not read or store recommendation cache for user %s: %s", user_id, error)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not access the recommendation cache.") from error

    @app.post("/v1/recommend", dependencies=[Depends(require_internal_auth)], response_model=RecommendationResponse)
    async def recommend(
        request: RecommendRequest,
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        client: Annotated[FlixerClient, Depends(_flixer_client)],
        recommendation_cache: Annotated[RecommendationCache, Depends(_recommendation_cache)],
    ):
        if not manager.settings.enabled:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI recommendations are disabled.")

        async def create_recommendations() -> RecommendationResponse:
            logger.info("Recommendation refresh started for user %s.", request.user_id)
            return await generate_recommendations(request.user_id, request.limit, request.filters, manager, client)

        request_key = hashlib.sha256(request.model_dump_json(exclude={"user_id"}, exclude_none=True).encode("utf-8")).hexdigest()
        try:
            return await recommendation_cache.get_or_create(
                str(request.user_id),
                request_key,
                create_recommendations,
                client,
                recommendation_scope(request.filters),
            )
        except RecommendationRefreshRateLimited as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": str(error.retry_after_seconds)},
            ) from error
        except RuntimeError as error:
            logger.warning("Could not read or store recommendation cache for user %s: %s", request.user_id, error)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not access the recommendation cache.") from error

    @app.post("/v1/similar", dependencies=[Depends(require_internal_auth)], response_model=RecommendationResponse)
    async def similar(
        request: SimilarRequest,
        manager: Annotated[ProfileManager, Depends(_profile_manager)],
        client: Annotated[FlixerClient, Depends(_flixer_client)],
    ):
        if not manager.settings.enabled:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI recommendations are disabled.")
        logger.info("Similar-title request started.")
        try:
            context = (
                await client.recommendation_context(request.user_id)
                if request.user_id is not None
                else await client.library_context()
            )
        except RuntimeError as error:
            logger.warning("Could not load similar-item context: %s", error)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not load recommendation data from Flixer.") from error

        source = next((item for item in context.candidates if item.id == str(request.item_id)), None)
        if source is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media item not found.")
        profile = manager.active_profile
        inference_started = time.monotonic()
        try:
            items = await asyncio.to_thread(
                profile.similar,
                source,
                context.candidates,
                request.limit,
                request.filters,
            )
        except ProfileError as error:
            await manager.use_classic_after_error(error)
            items = await asyncio.to_thread(
                manager.active_profile.similar,
                source,
                context.candidates,
                request.limit,
                request.filters,
            )
        logger.info(
            "Similar-title request completed with %d results.", len(items),
            extra={"duration_ms": round((time.monotonic() - inference_started) * 1000, 3),
                   "model_profile": manager.active_profile.name, "result_count": len(items)},
        )
        return RecommendationResponse(items=items)

    return app


app = create_app()
