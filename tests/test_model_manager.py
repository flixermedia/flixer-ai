import hashlib

import httpx

from flixer_ai.model_manager import CachedModelManager, select_auto_profile, supports_onnx_architecture


def test_auto_profile_obeys_memory_thresholds() -> None:
    assert select_auto_profile(511 * 1024 * 1024) == "classic"
    assert select_auto_profile(512 * 1024 * 1024) == "micro"
    assert select_auto_profile(1200 * 1024 * 1024) == "micro"
    assert select_auto_profile(1201 * 1024 * 1024) == "mini"


def test_onnx_architecture_support_is_explicit() -> None:
    assert supports_onnx_architecture("aarch64")
    assert supports_onnx_architecture("AMD64")
    assert not supports_onnx_architecture("armv7l")


def test_model_cache_manifest_detects_changed_files(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"revision":"revision","files":{"model.onnx":"'
        + CachedModelManager._sha256_file(model)
        + '","tokenizer.json":"'
        + CachedModelManager._sha256_file(tokenizer)
        + '"}}',
        encoding="utf-8",
    )

    assert CachedModelManager._cache_is_valid(manifest, "revision", model, tokenizer)
    model.write_bytes(b"changed")
    assert not CachedModelManager._cache_is_valid(manifest, "revision", model, tokenizer)


def test_model_download_does_not_treat_xet_etag_as_file_checksum(tmp_path, monkeypatch) -> None:
    contents = b"model artifact"
    xet_etag = "c96f5f1e2aee643bd8191bb520a3e175db7b05821579a02a70acdf31e655d194"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=contents, headers={"etag": f'"{xet_etag}"'})
        ),
    )
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    destination = tmp_path / "model.onnx"
    manager = CachedModelManager(tmp_path, allow_downloads=True)

    checksum = manager._download_file(
        {"repo": "Xenova/all-MiniLM-L6-v2", "revision": "revision"},
        "onnx/model_quantized.onnx",
        destination,
    )

    assert destination.read_bytes() == contents
    assert checksum == hashlib.sha256(contents).hexdigest()
