from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from .features import text_for, tokens_for
from .schemas import MediaItem, RecommendationFilters, ScoredItem, WatchHistoryItem


class ProfileError(RuntimeError):
    """A selected recommendation profile could not produce a result."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class RecommenderProfile(Protocol):
    name: str

    def recommend(
        self,
        history: list[WatchHistoryItem],
        favorites: list[MediaItem],
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]: ...

    def similar(
        self,
        item: MediaItem,
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]: ...


def filter_candidates(
    candidates: list[MediaItem],
    filters: RecommendationFilters | None,
) -> list[MediaItem]:
    if filters is None:
        return candidates
    types = {value.casefold() for value in filters.types or []}
    genres = {value.casefold() for value in filters.genres or []}
    result: list[MediaItem] = []
    for item in candidates:
        if types and item.type.casefold() not in types:
            continue
        if filters.year_from is not None and (item.year is None or item.year < filters.year_from):
            continue
        if filters.year_to is not None and (item.year is None or item.year > filters.year_to):
            continue
        if genres and not genres.intersection(value.casefold() for value in item.genres):
            continue
        result.append(item)
    return result


def completion_weight(item: WatchHistoryItem) -> float:
    completion = 1.0 if item.completed else 0.0
    if item.duration_seconds > 0:
        completion = max(completion, min(item.position_seconds / item.duration_seconds, 1.0))
    recency = 1.0
    if item.updated_at:
        try:
            updated_at = datetime.fromisoformat(item.updated_at.replace("Z", "+00:00"))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds() / 86_400)
            recency = 0.25 + 0.75 * math.exp(-age_days / 90)
        except ValueError:
            recency = 0.25
    return (0.35 + 0.65 * completion) * recency


@dataclass(slots=True)
class ClassicProfile:
    name: str = "classic"

    @staticmethod
    def _vectors(documents: list[list[str]]) -> list[dict[str, float]]:
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))
        total_documents = max(len(documents), 1)
        inverse_frequency = {
            token: math.log((total_documents + 1) / (count + 1)) + 1
            for token, count in document_frequency.items()
        }
        vectors: list[dict[str, float]] = []
        for tokens in documents:
            counts = Counter(tokens)
            vector = {token: (1 + math.log(count)) * inverse_frequency[token] for token, count in counts.items()}
            magnitude = math.sqrt(sum(value * value for value in vector.values()))
            vectors.append({token: value / magnitude for token, value in vector.items()} if magnitude else {})
        return vectors

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return max(0.0, sum(value * right.get(token, 0.0) for token, value in left.items()))

    def _rank(
        self,
        sources: list[tuple[MediaItem, float]],
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None,
    ) -> list[ScoredItem]:
        filtered = filter_candidates(candidates, filters)
        if not sources or not filtered:
            return []
        documents = [tokens_for(item) for item, _ in sources] + [tokens_for(item) for item in filtered]
        vectors = self._vectors(documents)
        source_vectors = list(zip(vectors[: len(sources)], (weight for _, weight in sources), strict=True))
        scored: list[ScoredItem] = []
        source_ids = {item.id for item, _ in sources}
        for index, candidate in enumerate(filtered):
            if candidate.id in source_ids:
                continue
            candidate_vector = vectors[len(sources) + index]
            weight_total = sum(weight for _, weight in source_vectors)
            score = (
                sum(self._cosine(source_vector, candidate_vector) * weight for source_vector, weight in source_vectors)
                / weight_total
                if weight_total
                else 0
            )
            if score > 0:
                scored.append(ScoredItem(id=candidate.id, score=min(score, 1.0)))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:limit]

    def recommend(
        self,
        history: list[WatchHistoryItem],
        favorites: list[MediaItem],
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]:
        watched = {item.id for item in history if item.completed or completion_weight(item) >= 0.95}
        eligible = [candidate for candidate in candidates if candidate.id not in watched]
        sources = [(item, completion_weight(item)) for item in history]
        sources.extend((item, 1.5) for item in favorites)
        return self._rank(sources, eligible, limit, filters)

    def similar(
        self,
        item: MediaItem,
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]:
        return self._rank([(item, 1.0)], candidates, limit, filters)


@dataclass(slots=True)
class EmbeddingProfile:
    name: str
    embedder: Embedder

    def _rank(
        self,
        sources: list[tuple[MediaItem, float]],
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None,
    ) -> list[ScoredItem]:
        filtered = filter_candidates(candidates, filters)
        if not sources or not filtered:
            return []
        source_ids = {item.id for item, _ in sources}
        filtered = [item for item in filtered if item.id not in source_ids]
        if not filtered:
            return []
        try:
            source_vectors = self.embedder.embed([text_for(item) for item, _ in sources])
            candidate_vectors = self.embedder.embed([text_for(item) for item in filtered])
        except Exception as error:
            raise ProfileError(f"{self.name} embedding failed.") from error
        weights = np.asarray([weight for _, weight in sources], dtype=np.float32)
        taste = np.average(source_vectors, axis=0, weights=weights)
        taste_norm = np.linalg.norm(taste)
        if taste_norm:
            taste = taste / taste_norm
        scores = np.clip(candidate_vectors @ taste, 0.0, 1.0)
        order = np.argsort(scores)[::-1][:limit]
        return [ScoredItem(id=filtered[int(index)].id, score=float(scores[index])) for index in order if scores[index] > 0]

    def recommend(
        self,
        history: list[WatchHistoryItem],
        favorites: list[MediaItem],
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]:
        watched = {item.id for item in history if item.completed or completion_weight(item) >= 0.95}
        eligible = [candidate for candidate in candidates if candidate.id not in watched]
        sources = [(item, completion_weight(item)) for item in history]
        sources.extend((item, 1.5) for item in favorites)
        return self._rank(sources, eligible, limit, filters)

    def similar(
        self,
        item: MediaItem,
        candidates: list[MediaItem],
        limit: int,
        filters: RecommendationFilters | None = None,
    ) -> list[ScoredItem]:
        return self._rank([(item, 1.0)], candidates, limit, filters)
