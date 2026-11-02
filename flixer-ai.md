# flixer-ai

Dedicated recommendation service for the Flixer media platform.

## Purpose

`flixer-ai` is a lightweight, self-hosted recommendation engine.  
It receives requests from `flixer-web` (or the core `flixer` API) and returns ranked media recommendations based on a user’s watch history and the local media library.

It must:

- Run comfortably on Raspberry Pi 4/5 (4 GB and 8 GB)
- Require zero manual model management from the admin
- Automatically download and cache the required model files
- Support multiple recommendation profiles with automatic fallback
- Use the existing Flixer `DATA_DIR` persistent volume
- Be fully offline-capable after the first model download

## Architecture Context

Flixer is a multi-service self-hosted media platform:

- `flixer` (TypeScript) – core backend, users, libraries, watch progress, metadata
- `flixer-web` (Vue) – frontend
- `flixer-streamer` (Go) – playback
- `flixer-indexer` (Go) – scanning & metadata
- `flixer-ai` (this service) – recommendations

`flixer-ai` is an optional service. It does **not** own user or media data. It queries the core `flixer` API for:

- User watch progress / history
- Media item metadata (title, overview, genres, year, cast, etc.)
- Favorites (optional)

It returns ranked lists of media item IDs + scores.

## Core Requirements

### Recommendation Profiles

Support these profiles (selectable via environment variable):

| Profile   | Method                          | Download | Typical RAM | Notes                          |
|-----------|----------------------------------|----------|-------------|--------------------------------|
| `classic` | TF-IDF + cosine similarity      | None     | 150–250 MB  | Always available, no ML model  |
| `micro`   | Very small ONNX embedder        | ~50 MB   | 250–400 MB  | For tighter memory             |
| `mini`    | `all-MiniLM-L6-v2` ONNX         | ~80–90 MB| 350–550 MB  | Best quality that still fits Pi|
| `auto`    | Automatically chooses the best  | —        | —           | **Default**                    |

### Auto Profile Logic (`RECOMMENDER_PROFILE=auto`)

On startup or first request:

1. Detect available system RAM and architecture.
2. Prefer `mini` if there is comfortable free RAM (> ~1.2 GB free).
3. Fall back to `micro` if free RAM is moderate.
4. Fall back to `classic` on low-memory devices or if model loading fails.
5. Never crash the service. Always have a working recommender.

### Model Download & Caching

- Models are stored under the existing Flixer persistent volume:
  ```
  $DATA_DIR/ai/models/
  ```
- On first use of a profile that needs a model:
  - Check if the files already exist and are valid.
  - If missing → download automatically from Hugging Face (or a reliable mirror).
  - Store them permanently in `$DATA_DIR/ai/models/`.
  - Use atomic writes and a simple lock so concurrent requests don’t trigger multiple downloads.
- Subsequent starts reuse the cached files (no re-download).
- If the network is unavailable or download fails → fall back to `classic`.

### Configuration

Environment variables:

```bash
DATA_DIR=/data                          # Required – same volume used by the rest of Flixer
RECOMMENDER_PROFILE=auto                # auto | classic | micro | mini
MODELS_DIR=${DATA_DIR}/ai/models        # Optional override
ALLOW_DOWNLOADS=true                    # Safety switch (default true)
FLIXER_URL=http://flixer:8080           # Core API base URL (internal)
FLIXER_INTERNAL_TOKEN=...               # Shared secret / internal auth token
```

### API Endpoints

Minimum required endpoints:

- `GET /healthz` – health check + basic status
- `GET /status` – detailed status (active profile, model ready, download state, etc.)
- `POST /v1/recommend` – main recommendation endpoint
  - Input: `user_id`, `limit` (default 20), optional filters / context
  - Output: ranked list of `{ id, score, reason? }`
- `POST /v1/similar` – similar items for a given media item ID

All endpoints that return recommendations must work even if the preferred model is still downloading (fall back to `classic`).

### Data Flow

1. Client (web or core) calls `flixer-ai` with a `user_id`.
2. `flixer-ai` fetches the user’s recent watch progress + relevant media metadata from the core `flixer` API.
3. Builds a user taste representation (weighted by completion percentage, recency, favorites).
4. Ranks items from the local library using the active profile.
5. Returns ranked media item IDs + scores.

### Technology Stack

- Language: Python 3.12+
- Web framework: FastAPI
- Classical path: scikit-learn or pure NumPy (TF-IDF)
- Embedding path: ONNX Runtime + small sentence-transformer models exported to ONNX
- HTTP client: `httpx` (to talk to core Flixer API)
- No heavy PyTorch runtime in production images if possible (prefer ONNX Runtime)

### Docker Requirements

- Multi-arch image: `linux/amd64` + `linux/arm64`
- Runs as unprivileged user
- Healthcheck
- Single persistent volume mount: `${DATA_DIR}:/data`
- Lightweight base image (`python:3.12-slim` or better)

### Project Structure (suggested)

```text
flixer-ai/
├── Dockerfile
├── docker-compose.yml          # optional for standalone development
├── .env.example
├── pyproject.toml / requirements.txt
├── README.md
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config.py
│   ├── api/
│   │   ├── health.py
│   │   ├── recommend.py
│   │   └── similar.py
│   ├── profiles/
│   │   ├── base.py             # common interface
│   │   ├── classic.py          # TF-IDF implementation
│   │   ├── micro.py
│   │   └── mini.py
│   ├── model_manager.py        # download, cache, verify models
│   ├── selector.py             # profile selection + fallback logic
│   ├── clients/
│   │   └── flixer.py           # client to talk to core Flixer API
│   └── features/               # feature extraction helpers
└── tests/
```

### Profile Interface

Every profile must implement the same interface:

```python
class RecommenderProfile(Protocol):
    name: str

    async def load(self) -> None: ...
    async def is_ready(self) -> bool: ...

    async def recommend(
        self,
        user_history: list[WatchProgress],
        candidates: list[MediaItem],
        limit: int = 20,
    ) -> list[ScoredItem]: ...

    async def similar(
        self,
        item: MediaItem,
        candidates: list[MediaItem],
        limit: int = 20,
    ) -> list[ScoredItem]: ...
```

### Key Implementation Notes

- Prefer ONNX Runtime over full sentence-transformers + PyTorch for the embedding profiles.
- Model download must be resilient (retries, timeout, checksum verification).
- Service must remain responsive while a model is downloading (serve `classic` in the meantime).
- Keep memory usage low — especially important on Raspberry Pi.
- Log clearly which profile is active and why a fallback occurred.
- The service should be stateless regarding user data (always fetch fresh history from core).

### Success Criteria

- Works out of the box with `RECOMMENDER_PROFILE=auto`
- Automatically downloads and caches models under `$DATA_DIR/ai/models`
- Runs on Raspberry Pi 4 (4 GB) using at least the `classic` profile
- Falls back gracefully when models cannot be loaded
- Provides useful recommendations from real watch history + metadata
- Clean FastAPI service with proper health/status endpoints
- Multi-arch Docker image ready for the Flixer stack

---

Build a production-ready implementation of this service following the requirements above.
```
