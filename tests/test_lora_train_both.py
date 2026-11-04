"""``lora train --base both`` trains one base, releases its encoders, then the next.

An omitted ``--base`` on a legacy project stays on Z-Image and writes
``checkpoints/`` at the project root.
"""

from __future__ import annotations

import json

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.bases import BaseInfo

SEEN: list[object] = []


def _adapter(mdl_id: str, name: str) -> type[TinyAdapter]:
    class Adapter(TinyAdapter):
        info = BaseInfo(mdl_id, f"example/{mdl_id}", "0" * 40, "Apache-2.0", name)

        def load(self, weights_dir, config, device):
            SEEN.append(("load", self.info.mdl_id))
            return super().load(weights_dir, config, device)

        def release_encoders(self):
            SEEN.append(("release", self.info.mdl_id))
            super().release_encoders()

        def encode(self, images, captions, config):
            out = super().encode(images, captions, config)
            SEEN.extend(("cache", key) for key in self._embed_cache)
            return out

    Adapter.__name__ = f"StandIn_{mdl_id.replace('-', '_')}"
    return Adapter


def _envelope(text: str) -> dict[str, object]:
    start = text.rindex("\n{") if "\n{" in text else text.index("{")
    return json.loads(text[start:])


def _install():
    saved = {mdl: bases._REGISTRY[mdl] for mdl in ("moss-image-base", "z-image")}
    bases.register(_adapter("moss-image-base", "MOSS stand-in"))
    bases.register(_adapter("z-image", "Z-Image stand-in"))
    return saved


def _restore(saved) -> None:
    bases._REGISTRY.update(saved)


def _both_project(root):
    project = make_project(root)
    data = project.read()
    data["bases"] = ["moss-image-base", "z-image"]
    project.write(data)
    return project


def _train_args(root, *extra: str) -> list[str]:
    return [
        "--json", "lora", "train", "--project", str(root), "--no-fetch", "--no-previews",
        "--vram", "24", "--resolution", "64", "--batch", "1", "--save-every", "1", *extra,
    ]


def test_both_trains_in_order_and_cache_keys_include_the_base(tmp_path, capsys) -> None:
    SEEN.clear()
    saved = _install()
    try:
        project = _both_project(tmp_path)
        code = main(_train_args(project.root, "--base", "both", "--review-rounds", "0", "--steps", "2"))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0
    result = payload["result"]
    assert result["bases"] == ["moss-image-base", "z-image"]
    assert [run["base"] for run in result["runs"]] == ["moss-image-base", "z-image"]
    assert all(run["state"] == "done" for run in result["runs"])
    events = [item for item in SEEN if isinstance(item, tuple) and item[0] in {"load", "release"}]
    load_z = events.index(("load", "z-image"))
    assert any(item == ("release", "moss-image-base") and index < load_z for index, item in enumerate(events))
    keys = [item[1] for item in SEEN if isinstance(item, tuple) and item[0] == "cache"]
    moss_keys = [key for key in keys if key.startswith("moss-image-base\x1f")]
    z_keys = [key for key in keys if key.startswith("z-image\x1f")]
    assert moss_keys and z_keys and set(moss_keys).isdisjoint(z_keys)
    assert list((tmp_path / "checkpoints" / "moss-image-base").glob("step-*.safetensors"))
    assert list((tmp_path / "checkpoints" / "z-image").glob("step-*.safetensors"))
    assert list((tmp_path / "checkpoints").glob("step-*.safetensors")) == []


def test_both_stops_when_the_first_base_pauses_for_review(tmp_path, capsys) -> None:
    SEEN.clear()
    saved = _install()
    try:
        project = _both_project(tmp_path)
        code = main(_train_args(
            project.root, "--base", "both", "--review-rounds", "2", "--steps", "4", "--first-round-samples", "1",
        ))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0 and payload["result"]["state"] == "review"
    assert ("load", "z-image") not in SEEN
    assert (tmp_path / "checkpoints" / "moss-image-base").is_dir()
    assert not (tmp_path / "checkpoints" / "z-image").exists()


def test_omitted_base_on_a_legacy_project_trains_z_image_at_the_root(tmp_path, capsys) -> None:
    saved = {mdl: bases._REGISTRY[mdl] for mdl in ("z-image",)}
    bases.register(_adapter("z-image", "Z-Image stand-in"))
    try:
        project = make_project(tmp_path)
        code = main(_train_args(project.root, "--review-rounds", "0", "--steps", "2"))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0 and payload["result"]["state"] == "done"
    assert "runs" not in payload["result"]
    assert project.read()["train"]["base"] == "z-image"
    assert "bases" not in project.read()
    assert list((tmp_path / "checkpoints").glob("step-*.safetensors"))
    assert not (tmp_path / "checkpoints" / "z-image").exists()


def test_dry_run_of_one_base_stays_a_single_plan(tmp_path, capsys) -> None:
    saved = _install()
    try:
        project = _both_project(tmp_path / "one")
        code = main(_train_args(project.root, "--base", "moss", "--dry-run", "--steps", "2"))
        single = _envelope(capsys.readouterr().out)["result"]
        capsys.readouterr()
        code_both = main(_train_args(project.root, "--base", "both", "--dry-run", "--steps", "2"))
        both = _envelope(capsys.readouterr().out)["result"]
    finally:
        _restore(saved)
    assert code == 0 and single["state"] == "dry-run" and single["base"] == "moss-image-base" and "runs" not in single
    assert code_both == 0 and both["state"] == "dry-run" and both["bases"] == ["moss-image-base", "z-image"]
    assert [run["steps"] for run in both["runs"]] == [2, 2]
