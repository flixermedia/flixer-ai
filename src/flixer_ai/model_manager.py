from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import platform
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np

from .config import Settings
from .profiles import ClassicProfile, EmbeddingProfile, ProfileError

logger = logging.getLogger("flixer_ai.models")

_MODELS = {
    "micro": {
        "repo": "Xenova/all-MiniLM-L6-v2",
        "revision": "751bff37182d3f1213fa05d7196b954e230abad9",
        "onnx": "onnx/model_quantized.onnx",
    },
    "mini": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "onnx": "onnx/model.onnx",
    },
}
_MAX_MODEL_BYTES = 200 * 1024 * 1024
_MAX_SEQUENCE_LENGTH = 128
_BATCH_SIZE = 16


def _available_memory_bytes() -> int:
    available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        available = 0

    cgroup_available: int | None = None
    for root in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        try:
            limit_text = (root / "memory.max").read_text(encoding="ascii").strip()
            usage_text = (root / "memory.current").read_text(encoding="ascii").strip()
            if limit_text != "max":
                cgroup_available = max(0, int(limit_text) - int(usage_text))
                break
        except (OSError, ValueError):
            pass
        try:
            limit = int((root / "memory.limit_in_bytes").read_text(encoding="ascii").strip())
            usage = int((root / "memory.usage_in_bytes").read_text(encoding="ascii").strip())
            if limit < 1 << 60:
                cgroup_available = max(0, limit - usage)
                break
        except (OSError, ValueError):
            pass
    if cgroup_available is not None:
        return min(available, cgroup_available) if available else cgroup_available
    if available:
        return available
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 0


def select_auto_profile(available_memory: int) -> str:
    if available_memory > 1_200 * 1024 * 1024:
        return "mini"
    if available_memory >= 512 * 1024 * 1024:
        return "micro"
    return "classic"


def supports_onnx_architecture(machine: str | None = None) -> bool:
    return (machine or platform.machine()).lower() in {"aarch64", "arm64", "x86_64", "amd64"}


@dataclass(frozen=True, slots=True)
class DownloadState:
    loading_profile: str | None
    download_state: str
    model_ready: bool
    fallback_reason: str | None


class CachedModelManager:
    def __init__(self, models_dir: Path, allow_downloads: bool) -> None:
        self.models_dir = models_dir
        self.allow_downloads = allow_downloads
        self.loading_profile: str | None = None
        self.download_state = "idle"
        self.model_ready = False
        self.fallback_reason: str | None = None
        self._async_lock = asyncio.Lock()

    async def load_profile(self, profile: str) -> EmbeddingProfile:
        if profile not in _MODELS:
            raise ValueError(f"Unknown embedding profile: {profile}")
        async with self._async_lock:
            self.loading_profile = profile
            self.download_state = "checking"
            self.model_ready = False
            try:
                model_path, tokenizer_path = await asyncio.to_thread(self._ensure_files, profile)
                self.download_state = "loading"
                embedder = await asyncio.to_thread(self._load_embedder, model_path, tokenizer_path)
                self.model_ready = True
                self.download_state = "ready"
                return EmbeddingProfile(name=profile, embedder=embedder)
            except Exception:
                self.download_state = "failed"
                raise
            finally:
                self.loading_profile = None

    def _ensure_files(self, profile: str) -> tuple[Path, Path]:
        spec = _MODELS[profile]
        profile_dir = self.models_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.models_dir / ".download.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                model_path = profile_dir / "model.onnx"
                tokenizer_path = profile_dir / "tokenizer.json"
                manifest_path = profile_dir / "manifest.json"
                if self._cache_is_valid(manifest_path, spec["revision"], model_path, tokenizer_path):
                    return model_path, tokenizer_path
                if not self.allow_downloads:
                    raise RuntimeError(f"Model {profile} is not cached and ALLOW_DOWNLOADS is false.")
                self.download_state = "downloading"
                logger.info("Downloading recommendation model profile %s.", profile)
                files = {
                    spec["onnx"]: model_path,
                    "tokenizer.json": tokenizer_path,
                }
                hashes: dict[str, str] = {}
                for filename, destination in files.items():
                    hashes[destination.name] = self._download_file(spec, filename, destination)
                payload = {"revision": spec["revision"], "files": hashes}
                self._atomic_write(manifest_path, json.dumps(payload, sort_keys=True).encode("utf-8"))
                return model_path, tokenizer_path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _cache_is_valid(manifest: Path, revision: str, *files: Path) -> bool:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("revision") != revision:
                return False
            expected_hashes = payload["files"]
            return all(
                file.is_file()
                and file.stat().st_size > 0
                and expected_hashes.get(file.name) == CachedModelManager._sha256_file(file)
                for file in files
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(path: Path, contents: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(contents)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _download_file(self, spec: dict[str, str], filename: str, destination: Path) -> str:
        url = (
            f"https://huggingface.co/{quote(spec['repo'], safe='/')}/resolve/"
            f"{spec['revision']}/{quote(filename, safe='/')}"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(descriptor, "wb") as output, httpx.Client(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    follow_redirects=True,
                ) as client:
                    with client.stream("GET", url) as response:
                        if response.status_code >= 500:
                            response.raise_for_status()
                        response.raise_for_status()
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > _MAX_MODEL_BYTES:
                            raise RuntimeError(f"Model file {filename} exceeds the 200 MiB download limit.")
                        for chunk in response.iter_bytes(1024 * 1024):
                            size += len(chunk)
                            if size > _MAX_MODEL_BYTES:
                                raise RuntimeError(f"Model file {filename} exceeds the 200 MiB download limit.")
                            digest.update(chunk)
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size == 0:
                    raise RuntimeError(f"Model download returned an empty file: {filename}.")
                # Hugging Face's Xet-backed CDN uses a 64-hex ETag for its CAS
                # object, not the SHA-256 of the downloaded file.
                os.replace(temporary_name, destination)
                return digest.hexdigest()
            except (httpx.HTTPError, OSError, ValueError, RuntimeError) as error:
                last_error = error
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500:
                    raise RuntimeError(f"Could not download model file {filename}: HTTP {error.response.status_code}.") from error
                if attempt < 2:
                    import time

                    time.sleep(2**attempt)
        raise RuntimeError(f"Could not download model file {filename} after three attempts.") from last_error

    @staticmethod
    def _load_embedder(model_path: Path, tokenizer_path: Path) -> _OnnxEmbedder:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            return _OnnxEmbedder(session, tokenizer)
        except Exception as error:
            raise ProfileError("The cached ONNX model could not be loaded.") from error


class _OnnxEmbedder:
    def __init__(self, session: Any, tokenizer: Any) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.input_names = {item.name for item in session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self.input_names):
            raise ProfileError("The ONNX model is missing required tokenizer inputs.")
        self.tokenizer.enable_truncation(max_length=_MAX_SEQUENCE_LENGTH)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoded = [self.tokenizer.encode(text) for text in texts]
        all_vectors: list[np.ndarray] = []
        try:
            for offset in range(0, len(encoded), _BATCH_SIZE):
                batch = encoded[offset : offset + _BATCH_SIZE]
                max_length = max(len(item.ids) for item in batch)
                input_ids = np.zeros((len(batch), max_length), dtype=np.int64)
                attention_mask = np.zeros((len(batch), max_length), dtype=np.int64)
                for index, item in enumerate(batch):
                    input_ids[index, : len(item.ids)] = item.ids
                    attention_mask[index, : len(item.ids)] = 1
                inputs: dict[str, np.ndarray] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                if "token_type_ids" in self.input_names:
                    inputs["token_type_ids"] = np.zeros_like(input_ids)
                output = self.session.run(None, inputs)[0]
                if output.ndim == 3:
                    mask = attention_mask[..., None].astype(np.float32)
                    vectors = (output * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                elif output.ndim == 2:
                    vectors = output
                else:
                    raise ProfileError("The ONNX model returned an unsupported embedding shape.")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                all_vectors.append((vectors / np.maximum(norms, 1e-12)).astype(np.float32))
        except ProfileError:
            raise
        except Exception as error:
            raise ProfileError("ONNX embedding inference failed.") from error
        return np.concatenate(all_vectors, axis=0)


class ProfileManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.active_profile: Any = _ClassicAdapter()
        self.requested_profile = settings.recommender_profile
        self.model_manager = CachedModelManager(settings.models_dir, settings.allow_downloads)
        self._load_task: asyncio.Task[None] | None = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "requestedProfile": self.requested_profile,
            "activeProfile": self.active_profile.name,
            "modelReady": self.model_manager.model_ready,
            "loadingProfile": self.model_manager.loading_profile,
            "downloadState": self.model_manager.download_state,
            "fallbackReason": self.model_manager.fallback_reason,
            "allowDownloads": self.settings.allow_downloads,
            "availableMemoryBytes": _available_memory_bytes(),
        }

    def start(self) -> None:
        if not self.settings.enabled:
            self.model_manager.download_state = "disabled"
            logger.info("AI recommendations are disabled.")
            return
        selected = self.requested_profile
        if selected == "auto":
            if not supports_onnx_architecture():
                selected = "classic"
                self.model_manager.fallback_reason = f"ONNX models are unsupported on {platform.machine()}."
            else:
                selected = select_auto_profile(_available_memory_bytes())
        if selected == "classic":
            logger.info("Using classic recommender profile.")
            return
        self._load_task = asyncio.create_task(self._load_with_fallback(selected))

    async def stop(self) -> None:
        if self._load_task is not None:
            self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass

    async def configure(self, enabled: bool, profile: str, allow_downloads: bool) -> None:
        if profile not in {"auto", "classic", "micro", "mini"}:
            raise ValueError("Profile must be auto, classic, micro, or mini.")
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass

        self.settings = replace(
            self.settings,
            enabled=enabled,
            recommender_profile=profile,
            allow_downloads=allow_downloads,
        )
        self.model_manager.allow_downloads = allow_downloads
        self.requested_profile = profile
        self.active_profile = _ClassicAdapter()
        self.model_manager.model_ready = False
        self.model_manager.loading_profile = None
        self.model_manager.download_state = "idle"
        self.model_manager.fallback_reason = None
        logger.info("AI configuration updated: enabled=%s, profile=%s, downloads=%s.", enabled, profile, allow_downloads)
        self.start()

    async def retry(self) -> None:
        if not self.settings.enabled:
            raise ValueError("AI recommendations are disabled.")
        await self.configure(True, self.requested_profile, self.settings.allow_downloads)

    async def _load_with_fallback(self, selected: str) -> None:
        profiles = ["mini", "micro"] if selected == "mini" else ["micro"] if selected == "micro" else []
        for profile in profiles:
            try:
                loaded = await self.model_manager.load_profile(profile)
                self.active_profile = loaded
                self.model_manager.fallback_reason = None
                logger.info("Loaded recommendation profile %s.", profile)
                return
            except Exception as error:
                reason = f"{profile} unavailable: {error}"
                self.model_manager.fallback_reason = reason
                logger.warning("%s Falling back to a smaller profile.", reason)
        self.active_profile = _ClassicAdapter()
        self.model_manager.model_ready = False
        self.model_manager.download_state = "fallback"
        logger.info("Using classic recommender profile.")

    async def use_classic_after_error(self, error: ProfileError) -> None:
        self.model_manager.fallback_reason = str(error)
        self.active_profile = _ClassicAdapter()
        self.model_manager.model_ready = False
        self.model_manager.download_state = "fallback"
        logger.error(
            "Embedding profile failed during a request; using classic fallback.",
            exc_info=(type(error), error, error.__traceback__),
        )


class _ClassicAdapter:
    name = "classic"

    def __init__(self) -> None:
        self.profile = ClassicProfile()

    def recommend(self, *args: Any, **kwargs: Any) -> Any:
        return self.profile.recommend(*args, **kwargs)

    def similar(self, *args: Any, **kwargs: Any) -> Any:
        return self.profile.similar(*args, **kwargs)
