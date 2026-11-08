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

For a development checkout, set `FLIXER_DEVELOPMENT_BUILD=true` to report the
installed package version with a `-dev` suffix in the health endpoint.

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
| `SENTRY_ENVIRONMENT` | `production` | Sentry environment label |

When enabled, unhandled HTTP exceptions are reported without request data, exception
text, stack traces, breadcrumbs, user information, or recommendation data. Reporting
is configured in Flixer's General > Error Monitoring dashboard. The global and
AI switches must both be enabled and the AI DSN set. Restart the service after
changing these settings. If the API is unavailable at startup, AI retries with
reporting disabled until it can load the configuration.
Older `SENTRY_DSN` environment values are no longer used; copy the DSN to the
dashboard before restarting an existing installation.

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

## Experimental Flixer Logger export

When Flixer is paired with a logger, AI can enroll automatically using its
existing signed internal Flixer connection, provided `FLIXER_URL` is HTTPS or
literal loopback HTTP, or `FLIXER_LOGGER_CONTROL_SOCKET` names Flixer's
protected local Unix socket. Non-loopback plaintext connections cannot distribute
logger credentials. The credential is application-bound to `flixer-ai`, stored
encrypted by Flixer, and refreshed in a private temporary directory by AI every
minute. It is not a new logger environment file. If enrollment fails, AI
suspends export until a refresh succeeds. Since the internal signature uses the
shared `APP_SECRET`, any service holding that secret and able to reach this
endpoint can request the AI credential; it is not a separate AI identity.
End-to-end TLS (or a trusted local TLS-terminating proxy) is required across
all non-loopback network hops. When configured, the socket takes precedence
and AI does not fall back to plaintext if the socket is unavailable.

Alternatively, an owner-only file at `/run/flixer-logger/ai.json` provisioned
for the AI service user takes precedence at process startup. The
file contains JSON fields `endpoint` (an HTTPS origin, or literal loopback
HTTP for development) and `token` (a deployment ingestion credential). It must
be a regular, non-symlink file owned by the service user with no group or other
permissions. Existing stdout/stderr logging remains enabled. Uvicorn access
exports only the HTTP method and status, not the client address or request URL.
Flixer recommendation calls carry a W3C `traceparent`; AI records inherit its
validated trace ID and record request duration and HTTP status without storing
request paths or bodies. Requests without a valid trace ID receive a new one.
Background cache work can continue under the trace of the request that started
it. These identifiers are correlation hints, not authentication.
Flixer's request logs and AI's completion logs can be retrieved together using
the logger's trace endpoint when both exporters use credentials for the same
deployment. AI-to-Flixer callbacks propagate the same trace ID, and Flixer
request logs accept valid W3C version-00 trace headers. The header is
untrusted correlation metadata and does not authorize requests.
Inference completion events also include elapsed milliseconds, the bounded
profile name (`classic`, `micro` or `mini`), and result count when available;
arbitrary model metadata and request payloads are not exported as fields.

When the exporter started with a valid file, its worker rechecks that file at
most once every five seconds when there are events to send. Atomically replace
the file with another owner-only regular file to rotate a credential. Keep the
old credential valid during the overlap: in-flight requests and retries may
still use it until the worker reloads. An HTTP 401 triggers an immediate
configuration recheck. A changed credential or endpoint discards queued events
from the previous configuration, even when the endpoint is unchanged, so a
credential for a different deployment cannot receive those events. If the file disappears or becomes
invalid, export is suspended and batches are dropped until a valid file is
available. An in-flight batch can still reach the previous endpoint.
When a manual file is selected or automatic enrollment is unavailable, the service
continues without export if that file is absent or invalid and checks for a
valid file every five seconds. When automatic enrollment is selected, adding a
manual file requires restarting AI. Events produced before attachment are
not replayed. If the configuration watcher cannot start, the error is reported
on stderr and AI continues without export; restart AI to try again.

The bounded queue, retries and temporary delivery circuit can drop records;
shutdown does not wait indefinitely. Producer and server redaction are
best-effort and must not be relied on to make secrets safe to log. Automatic
enrollment requires a protected control connection and a configured Flixer
logger. When a credential issued with an ID by this version of Flixer is
revoked through Flixer administration, Flixer clears it and AI obtains a new
one on its next enrollment refresh. Revocation outside Flixer, or of an older
credential whose ID was not recorded, is not automatically detected and
requires reconnecting or re-pairing. This experimental path is not a complete
production deployment setup.
