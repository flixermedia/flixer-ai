from flixer_ai.profiles import ClassicProfile, EmbeddingProfile, completion_weight
from flixer_ai.schemas import MediaItem, RecommendationFilters, WatchHistoryItem


def media_item(item_id: str, title: str, *, genres: list[str] | None = None, year: int | None = 2020) -> MediaItem:
    return MediaItem(id=item_id, type="movie", title=title, genres=genres or [], year=year)


def test_classic_profile_ranks_related_titles_and_applies_filters() -> None:
    history = [WatchHistoryItem(**media_item("1", "Space adventure", genres=["Sci-Fi"]).model_dump(), completed=True)]
    candidates = [
        media_item("1", "Space adventure", genres=["Sci-Fi"]),
        media_item("2", "A space adventure", genres=["Sci-Fi"]),
        media_item("3", "Small-town romance", genres=["Romance"]),
    ]

    result = ClassicProfile().recommend(
        history,
        [],
        candidates,
        10,
        RecommendationFilters(types=["movie"], genres=["sci-fi"], year_from=2010),
    )

    assert [item.id for item in result] == ["2"]
    assert 0 < result[0].score <= 1


def test_embedding_profile_excludes_sources_and_ranks_cosine_similarity() -> None:
    class FakeEmbedder:
        def embed(self, texts: list[str]):
            import numpy as np

            return np.asarray(
                [[1.0, 0.0] if "space" in text.casefold() else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    source = media_item("1", "Space adventure")
    candidates = [source, media_item("2", "Space opera"), media_item("3", "Cozy romance")]

    result = EmbeddingProfile("test", FakeEmbedder()).similar(source, candidates, 10)

    assert [item.id for item in result] == ["2"]
    assert result[0].score == 1


def test_watch_history_weight_decays_with_age() -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    recent = WatchHistoryItem(
        id="recent",
        type="movie",
        title="Recent",
        completed=True,
        updated_at=now.isoformat(),
    )
    old = WatchHistoryItem(
        id="old",
        type="movie",
        title="Old",
        completed=True,
        updated_at=(now - timedelta(days=365)).isoformat(),
    )

    assert completion_weight(recent) > completion_weight(old)
