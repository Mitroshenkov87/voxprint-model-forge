"""voxprint-common adoption: build number, licence allow-list, components, download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from voxprint_common.buildinfo import load
from voxprint_common.download import DiskSpaceError, download
from voxprint_common.paths import models_dir

from voxprint_forge.components import (
    component_roots,
    model_component_root,
    place_model_component,
    resolve_model_component,
)
from voxprint_forge.errors import ChecksumError, LicenceError
from voxprint_forge.fetch import HfDownloader
from voxprint_forge.licence import decide_licence
from voxprint_forge.plan import PinnedFile


def test_offset_is_added_to_the_run_number(tmp_path: Path) -> None:
    path = tmp_path / "BUILD.json"
    path.write_text(
        json.dumps(
            {
                "app": "model-forge",
                "version": "0.1.0",
                "rc": None,
                "build": None,
                "codename": "Charash",
                "offset": 20,
            }
        ),
        encoding="utf-8",
    )
    recorded = load(path, env={})
    assert recorded.build is None
    assert recorded.label() == '0.1.0 dev "Charash"'
    numbered = load(path, env={"GITHUB_RUN_NUMBER": "3"})
    assert numbered.build == 23
    assert numbered.label() == '0.1.0 build 23 "Charash"'
    assert numbered.build != 3


def test_shared_licence_allow_list() -> None:
    assert decide_licence("apache-2", repo="Qwen/Qwen3-8B") == "apache-2.0"
    assert decide_licence("CC-BY-4.0") == "cc-by-4.0"
    assert decide_licence("cc0") == "cc0-1.0"
    with pytest.raises(LicenceError) as unlicensed:
        decide_licence("unlicense")
    assert unlicensed.value.code == "non_permissive"
    with pytest.raises(LicenceError) as commercial:
        decide_licence("cc-by-nc-4.0", repo="Qwen/Qwen3-8B")
    assert commercial.value.code == "non_commercial"
    with pytest.raises(LicenceError) as copyleft:
        decide_licence("gpl-3.0")
    assert copyleft.value.code == "non_permissive"
    with pytest.raises(LicenceError) as excluded:
        decide_licence("apache-2.0", repo="yandex/YandexGPT-5-Lite-8B-instruct")
    assert excluded.value.code == "excluded_repo"


class _Body:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, count: int) -> bytes:
        chunk = self._payload[:count]
        self._payload = self._payload[count:]
        return chunk

    def close(self) -> None:
        return None


def test_pinned_download_uses_the_shared_downloader(tmp_path: Path) -> None:
    body = b"pinned-bytes"
    digest = hashlib.sha256(body).hexdigest()
    pinned = PinnedFile("source", "F16", "a.gguf", digest, len(body), "org/model", "main")
    seen: list[str] = []

    def opener(request: object, timeout: float) -> _Body:
        del timeout
        seen.append(str(getattr(request, "full_url", "")))
        return _Body(body)

    downloader = HfDownloader(None)
    downloader._open = opener  # type: ignore[method-assign]
    dest = tmp_path / "a.gguf"
    downloader.download_verified(pinned, dest)
    assert dest.read_bytes() == body
    assert seen
    assert "huggingface.co" in seen[0]
    assert "modelscope.cn" not in seen[0]


def test_download_disk_check_names_both_byte_counts(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"pinned-bytes").hexdigest()

    def opener(request: object, timeout: float) -> _Body:
        del request, timeout
        raise AssertionError("the disk check must run before any byte is read")

    with pytest.raises(DiskSpaceError) as disk:
        download(
            tmp_path / "c.gguf",
            ["https://example.com/a.gguf"],
            sha256=digest,
            size=100,
            disk_free=lambda path: 0,
            opener=opener,
        )
    assert "need" in str(disk.value)
    assert "bytes free" in str(disk.value)


def test_pinned_download_refuses_a_bad_digest(tmp_path: Path) -> None:
    body = b"not-the-pin"
    digest = hashlib.sha256(b"pinned-bytes").hexdigest()
    pinned = PinnedFile("source", "F16", "a.gguf", digest, len(body), "org/model", "main")

    def opener(request: object, timeout: float) -> _Body:
        del request, timeout
        return _Body(body)

    downloader = HfDownloader(None)
    downloader._open = opener  # type: ignore[method-assign]
    with pytest.raises(ChecksumError):
        downloader.download_verified(pinned, tmp_path / "a.gguf")
    assert not (tmp_path / "a.gguf").exists()


def test_model_components_stay_under_the_shared_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("VOXPRINT_MODEL_CACHE", str(cache))
    source = tmp_path / "weights.gguf"
    source.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    placed = place_model_component(source, "model-forge/llm/weights.gguf", version="1", sha256=digest)
    assert placed.is_file()
    assert model_component_root() == cache
    assert str(placed).startswith(str(cache))
    assert cache in component_roots() or placed.is_relative_to(cache)
    with pytest.raises(ValueError):
        resolve_model_component("/tmp/outside/weights.gguf")
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE")
    assert resolve_model_component("model-forge/llm/weights.gguf").is_relative_to(models_dir())
