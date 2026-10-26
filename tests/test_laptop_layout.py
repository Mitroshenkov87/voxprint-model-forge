"""Check, repair, and path resolution against the laptop model snapshot.

``tests/fixtures/laptop_layout/rebuild.py`` rebuilds the tree in a temp
directory. Weight files are zero-byte stand-ins. The captured small files stay
byte-for-byte, so this never calls ``ensure_view`` (that would replace them
with the empty blobs).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import voxprint_common.paths
from voxprint_common.gpu_probe import GpuDevice
from voxprint_common.mdl.job import verify_repair, verify_summary
from voxprint_common.mdl.safety import UrlGuard
from voxprint_common.paths import models_dir, projects_dir

from voxprint_forge.components import resolve_model_component

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "laptop_layout"
MDL = ROOT / "src" / "voxprint_forge" / "data" / "mdl.json"
INSTALLED = ("z-image", "qwen3-vl-4b-instruct", "mivolo-v2", "yunet-2023mar")
INDEX_REL = Path("model-forge") / "z-image" / "model_index.json"
INDEX_SHA = "1dbd162955866394e55e1964aee83b367aaaf9304e0591bbb9c9e58c4ddfce32"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


@pytest.fixture
def laptop_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Rebuild the snapshot under a Windows LOCALAPPDATA layout."""
    local = tmp_path / "AppData" / "Local"
    models = local / "VOXPrint AI Suite" / "models"
    monkeypatch.setattr(voxprint_common.paths, "_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.delenv("VOXPRINT_HOME", raising=False)
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)
    subprocess.run([sys.executable, str(FIXTURE / "rebuild.py"), str(models)], check=True)
    return models


def _card() -> GpuDevice:
    return GpuDevice(index=0, name="laptop", driver="0", compute=None, memory_mib=24 * 1024)


def _rows(data: dict[str, object]) -> list[tuple[str, str]]:
    files = data["files"]
    if isinstance(files, dict):
        return [(str(path), str(digest)) for path, digest in files.items()]
    if not isinstance(files, list):
        raise AssertionError("view marker files")
    rows: list[tuple[str, str]] = []
    for item in files:
        if not isinstance(item, dict):
            raise AssertionError("view marker row")
        rows.append((str(item["path"]), str(item["sha256"])))
    return rows


def test_check_reports_corrupt_zero_byte_weights(laptop_models: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = (laptop_models / INDEX_REL).read_bytes()
    blob = laptop_models / "blobs" / INDEX_SHA

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("check must not download")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    outcome = verify_repair(list(INSTALLED), repair=False, manifests=[MDL], devices=[_card()])
    states = {step.item_id: step.state for step in outcome.status.steps}
    assert outcome.code != 0
    assert states == {model_id: "corrupt" for model_id in INSTALLED}
    assert outcome.status.extra["counts"] == {"ok": 0, "missing": 0, "corrupt": 4, "repaired": 0, "failed": 0}
    assert outcome.message == verify_summary(outcome.status.extra["counts"])
    assert (laptop_models / INDEX_REL).read_bytes() == captured
    assert blob.is_file()
    assert blob.stat().st_size == 0
    assert models_dir() == laptop_models


def test_repair_restores_one_captured_file(laptop_models: Path, tmp_path: Path) -> None:
    captured = (laptop_models / INDEX_REL).read_bytes()
    assert hashlib.sha256(captured).hexdigest() == INDEX_SHA
    blob = laptop_models / "blobs" / INDEX_SHA
    assert blob.stat().st_size == 0
    server = _serve(captured)
    port = server.server_address[1]
    manifest = tmp_path / "mdl.json"
    document = {
        "schema": 1,
        "app": "model-forge",
        "models": [
            {
                "id": "z-image",
                "version": "laptop",
                "vram_tier": "any",
                "display_name": "Z-Image",
                "files": [
                    {
                        "path": "model_index.json",
                        "size": len(captured),
                        "sha256": INDEX_SHA,
                        "mirrors": [f"http://127.0.0.1:{port}/model_index.json"],
                    }
                ],
            }
        ],
    }
    manifest.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    try:
        outcome = verify_repair(
            ["z-image"],
            repair=True,
            manifests=[manifest],
            devices=[],
            guard=UrlGuard.loopback(),
        )
    finally:
        server.shutdown()
        server.server_close()
    assert outcome.code == 0
    assert outcome.status.steps[0].state == "repaired"
    assert outcome.status.extra["counts"]["repaired"] == 1
    assert blob.read_bytes() == captured
    assert (laptop_models / INDEX_REL).read_bytes() == captured
    recheck = verify_repair(["z-image"], repair=False, manifests=[manifest], devices=[], guard=UrlGuard.loopback())
    assert recheck.code == 0
    assert recheck.status.steps[0].state == "ok"


def test_model_resolution_stays_inside_the_laptop_tree(laptop_models: Path) -> None:
    view = laptop_models / INDEX_REL
    resolved = resolve_model_component(INDEX_REL.as_posix())
    assert resolved == view
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == INDEX_SHA
    assert resolve_model_component(str(view)) == view
    with pytest.raises(ValueError):
        resolve_model_component("model-forge/../../outside")
    with pytest.raises(ValueError):
        resolve_model_component(str(projects_dir() / "album"))


def test_project_paths_follow_the_suite_layout(laptop_models: Path) -> None:
    assert laptop_models == Path(os.environ["LOCALAPPDATA"]) / "VOXPrint AI Suite" / "models"
    assert models_dir() == laptop_models
    assert projects_dir() == laptop_models.parent / "projects"
    assert projects_dir().parent == models_dir().parent
    matched = 0
    stand_ins = 0
    markers = list(laptop_models.rglob(".voxprint-view.json"))
    assert markers
    for marker in markers:
        for rel, expected in _rows(json.loads(marker.read_text(encoding="utf-8"))):
            target = marker.parent / rel
            name = target.relative_to(laptop_models).as_posix()
            assert resolve_model_component(name) == target
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual == expected:
                matched += 1
            else:
                stand_ins += 1
                assert target.stat().st_size == 0
                assert actual == EMPTY_SHA
    assert matched == 43
    assert stand_ins == 37


def _serve(payload: bytes) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
