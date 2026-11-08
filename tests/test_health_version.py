from importlib.metadata import version
from types import SimpleNamespace

from fastapi.testclient import TestClient

from flixer_ai.main import _profile_manager, create_app


def test_health_reports_installed_version(monkeypatch):
    monkeypatch.delenv("FLIXER_DEVELOPMENT_BUILD", raising=False)
    app = create_app()
    app.dependency_overrides[_profile_manager] = lambda: SimpleNamespace(
        active_profile=SimpleNamespace(name="classic"),
        model_manager=SimpleNamespace(model_ready=True),
    )
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["version"] == version("flixer-ai")


def test_health_reports_development_version(monkeypatch):
    monkeypatch.setenv("FLIXER_DEVELOPMENT_BUILD", "true")
    app = create_app()
    app.dependency_overrides[_profile_manager] = lambda: SimpleNamespace(
        active_profile=SimpleNamespace(name="classic"),
        model_manager=SimpleNamespace(model_ready=True),
    )
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["version"] == f"{version('flixer-ai')}-dev"
    assert app.version == response.json()["version"]
