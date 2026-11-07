import json
import logging
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from flixer_ai.config import Settings
from flixer_ai.flixer_client import signature
from flixer_ai.main import _flixer_client, _profile_manager, create_app
from flixer_ai.profiles import ClassicProfile
from flixer_ai.schemas import MediaItem, RecommendationContext, WatchHistoryItem

USER_ID = "00000000-0000-4000-8000-000000000001"


class FakeFlixerClient:
    def __init__(self):
        self.recommendation_calls = 0
        self.cache_states = {}

    async def load_recommendation_cache(self, user_id):
        return self.cache_states.get(user_id, {"entries": {}, "lastRefreshAt": None})

    async def save_recommendation_cache(self, user_id, state):
        self.cache_states[user_id] = state

    async def clear_recommendation_cache(self):
        self.cache_states.clear()

    async def recommendation_context(self, user_id):
        assert str(user_id) == USER_ID
        self.recommendation_calls += 1
        return self.context()

    async def library_context(self):
        return self.context()

    @staticmethod
    def context():
        history_item = WatchHistoryItem(
            id="00000000-0000-4000-8000-000000000002",
            type="movie",
            title="Space adventure",
            genres=["Sci-Fi"],
            completed=True,
        )
        return RecommendationContext(
            candidates=[
                MediaItem(id=history_item.id, type="movie", title=history_item.title, genres=["Sci-Fi"]),
                MediaItem(id="00000000-0000-4000-8000-000000000003", type="movie", title="Space journey", genres=["Sci-Fi"]),
                MediaItem(id="00000000-0000-4000-8000-000000000004", type="movie", title="Cozy romance", genres=["Romance"]),
                MediaItem(id="00000000-0000-4000-8000-000000000006", type="album", title="Space soundtrack", genres=["Sci-Fi"]),
            ],
            history=[history_item],
        )


def test_recommendation_inference_reports_bounded_operational_metadata(tmp_path, caplog) -> None:
    settings = Settings(
        port=8094, data_dir=tmp_path, models_dir=tmp_path / "models", recommender_profile="classic",
        allow_downloads=False, flixer_url="http://flixer", internal_token="shared-token", enabled=True,
    )
    app = create_app(settings)
    app.dependency_overrides[_flixer_client] = lambda: FakeFlixerClient()
    app.dependency_overrides[_profile_manager] = lambda: SimpleNamespace(active_profile=ClassicProfile(), settings=settings)
    with caplog.at_level(logging.INFO, logger="flixer_ai"):
        response = TestClient(app).post(
            "/v1/recommend",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"user_id": USER_ID, "limit": 5, "filters": {"types": ["movie"]}},
        )
    assert response.status_code == 200
    records = [record for record in caplog.records if record.getMessage().startswith("Recommendation refresh completed")]
    assert len(records) == 1
    assert records[0].model_profile == "classic"
    assert records[0].result_count == len(response.json()["items"])
    assert records[0].duration_ms >= 0


def test_recommend_requires_auth_and_returns_personalized_results(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
        enabled=True,
    )
    app = create_app(settings)
    flixer_client = FakeFlixerClient()
    app.dependency_overrides[_flixer_client] = lambda: flixer_client

    with TestClient(app) as client:
        unauthorized = client.post("/v1/recommend", json={"user_id": USER_ID})
        assert unauthorized.status_code == 401

        response = client.post(
            "/v1/recommend",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"user_id": USER_ID, "limit": 5, "filters": {"genres": ["sci-fi"]}},
        )
        cached = client.post(
            "/v1/recommend",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"user_id": USER_ID, "limit": 5, "filters": {"genres": ["sci-fi"]}},
        )

    assert response.status_code == 200
    result = response.json()["items"]
    assert [item["id"] for item in result] == ["00000000-0000-4000-8000-000000000003"]
    assert 0 < result[0]["score"] <= 1
    assert cached.json() == response.json()
    assert flixer_client.recommendation_calls == 1


def test_music_and_screen_recommendations_do_not_share_cached_results(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
        enabled=True,
    )
    app = create_app(settings)
    flixer_client = FakeFlixerClient()
    app.dependency_overrides[_flixer_client] = lambda: flixer_client
    app.dependency_overrides[_profile_manager] = lambda: SimpleNamespace(
        active_profile=ClassicProfile(),
        settings=settings,
    )
    headers = {"X-Flixer-Internal-Token": "shared-token"}
    client = TestClient(app)

    movies = client.post(
        "/v1/recommend",
        headers=headers,
        json={"user_id": USER_ID, "limit": 5, "filters": {"types": ["movie"]}},
    )
    music = client.post(
        "/v1/recommend",
        headers=headers,
        json={"user_id": USER_ID, "limit": 5, "filters": {"types": ["album"]}},
    )
    cached_music = client.post(
        "/v1/recommend",
        headers=headers,
        json={"user_id": USER_ID, "limit": 5, "filters": {"types": ["album"]}},
    )

    assert movies.status_code == 200
    assert [item["id"] for item in movies.json()["items"]] == ["00000000-0000-4000-8000-000000000003"]
    assert music.status_code == 200
    assert [item["id"] for item in music.json()["items"]] == ["00000000-0000-4000-8000-000000000006"]
    assert cached_music.json() == music.json()
    assert flixer_client.recommendation_calls == 2


def test_similar_returns_404_for_missing_media_item(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
        enabled=True,
    )
    app = create_app(settings)
    app.dependency_overrides[_flixer_client] = lambda: FakeFlixerClient()

    with TestClient(app) as client:
        response = client.post(
            "/v1/similar",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"item_id": "00000000-0000-4000-8000-000000000005"},
        )

    assert response.status_code == 404


def test_admin_controls_require_signed_requests_and_update_runtime_profile(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
    )
    app = create_app(settings)
    flixer_client = FakeFlixerClient()
    app.dependency_overrides[_flixer_client] = lambda: flixer_client

    def signed_headers(path: str, method: str, body: bytes = b"") -> dict[str, str]:
        timestamp = str(int(time.time()))
        return {
            "X-Flixer-AI-Timestamp": timestamp,
            "X-Flixer-AI-Signature": signature("shared-token", timestamp, method, path, body),
        }

    with TestClient(app) as client:
        assert client.get("/admin/status").status_code == 401

        config_body = json.dumps(
            {
                "enabled": True,
                "profile": "classic",
                "allow_downloads": True,
                "cache_refresh_interval_minutes": 5,
            },
            separators=(",", ":"),
        ).encode()
        configured = client.post(
            "/admin/config",
            content=config_body,
            headers={
                **signed_headers("/admin/config", "POST", config_body),
                "Content-Type": "application/json",
            },
        )
        assert configured.status_code == 200
        assert configured.json()["status"]["allowDownloads"] is True
        assert configured.json()["status"]["requestedProfile"] == "classic"
        assert configured.json()["recommendationCache"]["refreshIntervalMinutes"] == 5

        status_response = client.get("/admin/status", headers=signed_headers("/admin/status", "GET"))
        assert status_response.status_code == 200
        assert status_response.json()["recommendationCache"]["refreshIntervalMinutes"] == 5
        events = client.get("/admin/events", headers=signed_headers("/admin/events", "GET"))
        assert events.status_code == 200
        assert any("AI configuration updated" in event["message"] for event in events.json()["events"])


def test_admin_can_force_refresh_a_users_recommendations(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
        enabled=True,
    )
    app = create_app(settings)
    flixer_client = FakeFlixerClient()
    app.dependency_overrides[_flixer_client] = lambda: flixer_client
    path = f"/admin/users/{USER_ID}/recommendations/refresh"
    timestamp = str(int(time.time()))
    headers = {
        "X-Flixer-AI-Timestamp": timestamp,
        "X-Flixer-AI-Signature": signature("shared-token", timestamp, "POST", path),
    }

    with TestClient(app) as client:
        unauthorized = client.post(path)
        refreshed = client.post(path, headers=headers)
        cached = client.post(
            "/v1/recommend",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"user_id": USER_ID, "limit": 20, "filters": {"types": ["movie", "series", "video"]}},
        )

    assert unauthorized.status_code == 401
    assert refreshed.status_code == 200
    assert cached.json() == refreshed.json()
    assert flixer_client.recommendation_calls == 1


def test_admin_rejects_invalid_recommendation_cache_refresh_interval(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
    )
    app = create_app(settings)

    def signed_headers(path: str, method: str, body: bytes = b"") -> dict[str, str]:
        timestamp = str(int(time.time()))
        return {
            "X-Flixer-AI-Timestamp": timestamp,
            "X-Flixer-AI-Signature": signature("shared-token", timestamp, method, path, body),
        }

    with TestClient(app) as client:
        config_body = json.dumps(
            {
                "enabled": False,
                "profile": "classic",
                "allow_downloads": False,
                "cache_refresh_interval_minutes": 0,
            },
            separators=(",", ":"),
        ).encode()
        response = client.post(
            "/admin/config",
            content=config_body,
            headers={
                **signed_headers("/admin/config", "POST", config_body),
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 422


def test_recommendation_cache_miss_reuses_previous_user_results(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
        enabled=True,
    )
    app = create_app(settings)
    flixer_client = FakeFlixerClient()
    app.dependency_overrides[_flixer_client] = lambda: flixer_client

    with TestClient(app) as client:
        headers = {"X-Flixer-Internal-Token": "shared-token"}
        first = client.post("/v1/recommend", headers=headers, json={"user_id": USER_ID})
        cached_for_different_query = client.post(
            "/v1/recommend",
            headers=headers,
            json={"user_id": USER_ID, "filters": {"genres": ["sci-fi"]}},
        )

    assert first.status_code == 200
    assert cached_for_different_query.status_code == 200
    assert cached_for_different_query.json() == first.json()
    assert flixer_client.recommendation_calls == 1


def test_recommendations_are_rejected_when_ai_is_disabled(tmp_path) -> None:
    settings = Settings(
        port=8094,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        recommender_profile="classic",
        allow_downloads=False,
        flixer_url="http://flixer",
        internal_token="shared-token",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/v1/recommend",
            headers={"X-Flixer-Internal-Token": "shared-token"},
            json={"user_id": USER_ID},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "AI recommendations are disabled."
