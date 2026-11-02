from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class RecommendationFilters(BaseModel):
    types: list[str] | None = Field(default=None, max_length=7)
    genres: list[str] | None = Field(default=None, max_length=30)
    year_from: int | None = Field(default=None, ge=1800, le=2200)
    year_to: int | None = Field(default=None, ge=1800, le=2200)

    @model_validator(mode="after")
    def valid_year_range(self) -> RecommendationFilters:
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from cannot be greater than year_to.")
        return self


class RecommendRequest(BaseModel):
    user_id: UUID
    limit: int = Field(default=20, ge=1, le=100)
    filters: RecommendationFilters | None = None


class SimilarRequest(BaseModel):
    item_id: UUID
    user_id: UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)
    filters: RecommendationFilters | None = None


class MediaItem(BaseModel):
    id: str
    type: str
    title: str
    year: int | None = None
    overview: str | None = None
    tagline: str | None = None
    genres: list[str] = Field(default_factory=list)
    studios: list[str] = Field(default_factory=list)
    cast: list[Any] = Field(default_factory=list)
    crew: list[Any] = Field(default_factory=list)
    artist: str | None = None
    parent_title: str | None = None
    parent_genres: list[str] = Field(default_factory=list)


class WatchHistoryItem(MediaItem):
    position_seconds: float = 0
    duration_seconds: float = 0
    completed: bool = False
    updated_at: str | None = None


class FavoriteItem(MediaItem):
    created_at: str | None = None


class RecommendationContext(BaseModel):
    candidates: list[MediaItem] = Field(default_factory=list)
    history: list[WatchHistoryItem] = Field(default_factory=list)
    favorites: list[FavoriteItem] = Field(default_factory=list)


class ScoredItem(BaseModel):
    id: str
    score: float
    reason: str | None = None


class RecommendationResponse(BaseModel):
    items: list[ScoredItem]
