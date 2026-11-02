from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_secret() -> str | None:
    value = os.getenv("FLIXER_INTERNAL_TOKEN", "").strip()
    if value:
        return value

    secret_file = Path(os.getenv("FLIXER_INTERNAL_TOKEN_FILE", "/run/flixer-secrets/app-secret"))
    try:
        value = secret_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"Could not read the Flixer internal token file: {secret_file}") from error
    return value or None


def _bool(value: str | None, default: bool, setting_name: str) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{setting_name} must be true or false.")


@dataclass(frozen=True, slots=True)
class Settings:
    port: int
    data_dir: Path
    models_dir: Path
    recommender_profile: str
    allow_downloads: bool
    flixer_url: str
    internal_token: str | None
    enabled: bool = False
    flixer_timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("DATA_DIR", "/data")).expanduser()
        models_dir = Path(os.getenv("MODELS_DIR", str(data_dir / "ai" / "models"))).expanduser()
        profile = os.getenv("RECOMMENDER_PROFILE", "auto").strip().lower()
        if profile not in {"auto", "classic", "micro", "mini"}:
            raise ValueError("RECOMMENDER_PROFILE must be auto, classic, micro, or mini.")
        port = int(os.getenv("PORT", "8094"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535.")

        return cls(
            port=port,
            data_dir=data_dir,
            models_dir=models_dir,
            recommender_profile=profile,
            allow_downloads=_bool(os.getenv("ALLOW_DOWNLOADS"), True, "ALLOW_DOWNLOADS"),
            flixer_url=os.getenv("FLIXER_URL", "http://flixer:8080").rstrip("/"),
            internal_token=_read_secret(),
            enabled=_bool(os.getenv("AI_ENABLED"), False, "AI_ENABLED"),
        )
