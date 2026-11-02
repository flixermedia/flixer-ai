# flixer-ai

An optional, self-hosted recommendation service for Flixer. It starts with a
zero-model TF-IDF recommender and can download small ONNX embedding models for
better semantic matching. Recommendations use fresh watch history, favorites,
and media metadata from the Flixer API; this service does not store user data.

## Run with Flixer

The service is an opt-in Compose profile. From the Flixer deployment directory:

```sh
docker compose --profile ai up -d
```

The service is reachable only on the Compose network at `http://ai:8094`.
Flixer's shared app secret is mounted read-only and used to authenticate both
incoming recommendation requests and requests for recommendation data. Model
files are cached under `${DATA_DIR}/ai/models`.

Signed-in Flixer clients use the core API rather than connecting to this
private service directly:

```http
POST /api/recommendations
Authorization: Bearer <user-token>
Content-Type: application/json

{"limit":20,"filters":{"types":["movie","series"],"genres":["Sci-Fi"]}}
```

`POST /api/recommendations/similar/{itemId}` returns similar titles. The core
API derives the user ID from the authenticated session and forwards requests
to this service using the internal shared token. These routes return HTTP 503
when the optional `ai` Compose profile is not running.

Administrators can enable recommendations, choose the runtime profile, control
model downloads, retry model loading, and view service activity under
**Admin → Settings → AI**. These saved Flixer settings are applied when the
service starts and whenever they are changed; the environment values below
provide defaults if Flixer is temporarily unavailable at startup.

For local development, create a Python 3.12 virtual environment, install
`.[test]`, and run:

```sh
uvicorn flixer_ai.main:app --reload --port 8094
pytest
```

Set `FLIXER_INTERNAL_TOKEN` (or `FLIXER_INTERNAL_TOKEN_FILE`) and
`FLIXER_URL` before calling `/v1/recommend` or `/v1/similar`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `8094` | HTTP listen port |
| `DATA_DIR` | `/data` | Persistent Flixer data mount |
| `MODELS_DIR` | `${DATA_DIR}/ai/models` | ONNX model cache override |
| `AI_ENABLED` | `false` | Enable recommendations when Flixer runtime settings cannot be loaded |
| `RECOMMENDER_PROFILE` | `auto` | `auto`, `classic`, `micro`, or `mini` |
| `ALLOW_DOWNLOADS` | `true` | Disable first-use model downloads |
| `FLIXER_URL` | `http://flixer:8080` | Core Flixer API origin |
| `FLIXER_INTERNAL_TOKEN` | unset | Shared Flixer app secret for service auth |
| `FLIXER_INTERNAL_TOKEN_FILE` | `/run/flixer-secrets/app-secret` | File containing that shared secret |

`auto` chooses `mini` when at least 1.2 GiB of memory is available, `micro`
when at least 512 MiB is available, and `classic` otherwise. Container memory
limits are taken into account. If a selected model cannot be downloaded or
loaded, the next smaller profile is tried, ending at `classic`.

## API

`GET /healthz` is an unauthenticated liveness check. `GET /status` provides the
active profile and model-loading state. Recommendation endpoints require
`X-Flixer-Internal-Token` and are intended for trusted Flixer services on the
private Compose network.

```http
POST /v1/recommend
Content-Type: application/json
X-Flixer-Internal-Token: <shared-secret>

{"user_id":"<uuid>","limit":20,"filters":{"types":["movie","series"]}}
```

`POST /v1/similar` accepts `item_id`, optional `user_id`, `limit`, and filters.
Both endpoints return `{"items":[{"id":"...","score":0.92}]}`. Scores are in
descending order from 0 to 1.

The service fetches data from signed internal Flixer API endpoints; it does not
accept database credentials and does not persist recommendation results.
