"""``lora finish``: one device for the final preview, ``--no-preview``, and export after a failed preview.

The RTX 4090 run (garnis-v3) crashed in the final preview with "Expected all tensors to be on
the same device, cuda:0 vs cpu": the adapters keep the transformer on the CPU until
``release_encoders``, and the final preview never called it, so the VAE and the embeddings
went to the card while the transformer and its LoRA stayed behind. These stand-ins use the
``meta`` device as the card, so the same mistake shows up on a CPU-only box.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

from tests.lora_fakes import TinyAdapter, TinyNet, gradient, make_project
from voxprint_forge.lora import bases
from voxprint_forge.lora import cli as lora_cli
from voxprint_forge.lora import train as train_mod
from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.layers import inject
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.project import Project
from voxprint_forge.lora.review import load_state, save_state, transition
from voxprint_forge.lora.train import _one_device, finish_review, train

CARD = "meta"


class CardAdapter(TinyAdapter):
    """Behaves like the real adapters: ``load`` parks the transformer on the CPU, ``release_encoders``
    moves it to the card, and ``preview`` puts its latents and embeddings on the card."""

    info = BaseInfo("card-test", "example/card", "0" * 40, "Apache-2.0", "Card")
    order: list[str] = []
    seen: list[set[str]] = []
    fail = False

    def load(self, weights_dir, config, device):
        type(self).order.append("load")
        self.device = device
        self.model = TinyNet()  # stays on the CPU, as ZImageAdapter.load leaves it
        return self.model

    def encode(self, images, captions, config):
        type(self).order.append("encode")
        return super().encode(images, captions, config)

    def prepare_previews(self, prompts):
        type(self).order.append("prepare_previews")

    def release_encoders(self):
        type(self).order.append("release_encoders")
        self.model.to(self.device)

    def preview(self, model, prompt, seed, config, steps=None):
        type(self).order.append("preview")
        if type(self).fail:
            raise RuntimeError("decode failed")
        latents = torch.zeros(1, device=CARD)
        devices = {parameter.device.type for parameter in model.parameters()} | {latents.device.type}
        type(self).seen.append(devices)
        if devices != {CARD}:
            raise RuntimeError(f"Expected all tensors to be on the same device, {CARD} vs cpu")
        return gradient(seed, 64)


def _config():
    return build_config("quick", "balanced", 24, steps=1, resolution=64, save_every=1, batch=1, previews=False)


def _paused_project(root: Path) -> Project:
    """A project paused at its first review, with a real checkpoint for the tiny model."""
    project = make_project(root, count=10)
    train(project, _config(), TinyAdapter(), Progress())
    state = transition(
        {}, "start", rounds=3, samples_per_round=8, first_round_steps=100, first_round_samples=6,
        sample_steps=28, total_steps=150, prompts=["tcube, a cube"], seed=1, base=CardAdapter.info.mdl_id,
        config=project.read()["train"]["config"],
    )
    save_state(project, transition(state, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=[]))
    data = project.read()
    project.write({**data, "train": {**data["train"], "base": CardAdapter.info.mdl_id}})
    return project


@pytest.fixture
def card(monkeypatch):
    CardAdapter.order, CardAdapter.seen, CardAdapter.fail = [], [], False
    monkeypatch.setattr(train_mod, "_device", lambda adapter, config, plan=None: CARD)
    bases.register(CardAdapter)
    yield CardAdapter
    bases.unregister(CardAdapter.info.mdl_id)


def _envelope(text: str) -> dict[str, object]:
    start = text.rindex("\n{") if "\n{" in text else text.index("{")
    return json.loads(text[start:])


def test_final_preview_puts_the_transformer_and_lora_on_one_device(tmp_path: Path, card) -> None:
    project = _paused_project(tmp_path)
    result = finish_review(project, CardAdapter(), Progress())
    assert result["state"] == "done" and result["preview"] == "drawn"
    assert card.seen == [{CARD}]
    # Same order as a training stage: encoders, prompts, then the transformer to the card, then draw.
    assert card.order.index("release_encoders") < card.order.index("preview")
    assert card.order.index("prepare_previews") < card.order.index("release_encoders")
    assert load_state(project)["phase"] == "finished"


def test_a_lora_layer_left_behind_joins_its_base_and_a_stray_model_is_a_named_error() -> None:
    model = TinyNet()
    layers = inject(model, ("to_q", "to_v"), 4, 4)
    for layer in layers.values():
        layer.base.to(CARD)
    model.to(CARD)
    layers["to_q"].lora_A = torch.nn.Linear(16, 4, bias=False)  # a LoRA layer loaded to the CPU
    assert layers["to_q"].lora_A.weight.device.type == "cpu"
    _one_device(model, layers, CARD)
    assert {parameter.device.type for parameter in model.parameters()} == {CARD}
    stray = TinyNet()
    with pytest.raises(LoraStageError) as caught:
        _one_device(stray, inject(stray, ("to_q", "to_v"), 4, 4), CARD)
    assert caught.value.key == "lora.err.preview_device"


def test_finish_no_preview_skips_the_picture_and_still_exports(tmp_path: Path, card, capsys) -> None:
    from voxprint_forge.cli import main

    project = _paused_project(tmp_path)
    capsys.readouterr()
    assert main(["--json", "lora", "finish", str(project.root), "--no-preview"]) == 0
    result = _envelope(capsys.readouterr().out)["result"]
    assert result["state"] == "done" and result["preview"] == "skipped"
    assert "preview" not in card.order and "load" not in card.order
    assert not list(project.previews.glob("step-000001-*.png"))
    package = Path(str(result["voxlora"]))
    assert package.suffix == ".voxlora" and package.is_file()
    with zipfile.ZipFile(package) as archive:
        assert "model.safetensors" in archive.namelist()
    assert load_state(project)["phase"] == "finished"
    assert project.read()["train"]["state"] == "done"


def test_a_failed_preview_warns_and_still_finishes_and_exports(tmp_path: Path, card, capsys) -> None:
    from voxprint_forge.cli import main

    card.fail = True
    project = _paused_project(tmp_path)
    capsys.readouterr()
    assert main(["--json", "lora", "finish", str(project.root)]) == 0
    out = capsys.readouterr().out
    events = [json.loads(line)["event"] for line in out.splitlines() if line.startswith('{"event"')]
    warning = next(event for event in events if event["message_key"] == "lora.warn.final_preview")
    assert "decode failed" in str(warning["params"]["error"])
    result = _envelope(out)["result"]
    assert result["state"] == "done" and result["preview"] == "failed"
    assert Path(str(result["voxlora"])).is_file()
    assert load_state(project)["phase"] == "finished"
    record = project.read()["train"]
    assert record["state"] == "done" and record["checkpoint"] == "step-000001.safetensors"


def test_a_failed_preview_prints_a_plain_warning(tmp_path: Path, card, capsys) -> None:
    from voxprint_forge.cli import main

    card.fail = True
    project = _paused_project(tmp_path)
    capsys.readouterr()
    assert main(["lora", "finish", str(project.root)]) == 0
    out = capsys.readouterr().out
    assert "final test picture could not be drawn" in out and "--no-preview" in out


def test_finish_help_names_no_preview_and_the_voxlora() -> None:
    from voxprint_forge.cli import build_parser

    parsed = build_parser().parse_args(["lora", "finish", "proj", "--no-preview"])
    assert parsed.preview is False
    assert build_parser().parse_args(["lora", "finish", "proj"]).preview is True


def test_headless_review_pause_prints_how_to_go_on(tmp_path: Path, monkeypatch, capsys) -> None:
    project = Project(tmp_path / "my proj")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    lora_cli._review_hint({"state": "review", "review": {"round": 1, "rounds": 3}}, project)
    err = capsys.readouterr().err
    assert "round 1 of 3" in err
    for command in ("lora review", "lora continue", "--unattended", "lora finish", "--review-rounds 0"):
        assert command in err
    assert f'"{project.root}"' in err
    lora_cli._review_hint({"state": "done"}, project)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    lora_cli._review_hint({"state": "review", "review": {"round": 1, "rounds": 3}}, project)
    assert capsys.readouterr().err == ""


def test_headless_train_hint_goes_to_stderr_and_json_stays_clean(tmp_path: Path, monkeypatch, capsys) -> None:
    from tests.test_lora_review import DrawingAdapter
    from voxprint_forge.cli import main

    project = make_project(tmp_path / "proj", count=10)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    bases.register(DrawingAdapter)
    try:
        capsys.readouterr()
        code = main([
            "--json", "lora", "train", "--project", str(project.root), "--base", "draw-test", "--no-fetch",
            "--review-rounds", "2", "--first-round-steps", "100", "--first-round-samples", "1",
            "--steps", "101", "--resolution", "64", "--batch", "1", "--no-previews",
        ])
    finally:
        bases.unregister("draw-test")
    assert code == 0
    captured = capsys.readouterr()
    assert _envelope(captured.out)["result"]["state"] == "review"
    assert "Paused for review" not in captured.out
    assert "Paused for review (round 1 of 2)" in captured.err and "--review-rounds 0" in captured.err


def test_finish_after_training_reuses_the_base_the_run_recorded(tmp_path: Path, capsys) -> None:
    """``--review-rounds 0`` stores the base on the project and writes no review file.

    Finish used to look only at that missing review base and exit 2 with
    ``lora.err.base_missing`` even though training had just used the same base.
    """
    from voxprint_forge.cli import main

    project = make_project(tmp_path / "proj", count=10)
    bases.register(TinyAdapter)
    try:
        capsys.readouterr()
        trained = main([
            "--json", "lora", "train", "--project", str(project.root), "--base", "tiny-test", "--no-fetch",
            "--review-rounds", "0", "--steps", "2", "--resolution", "64", "--batch", "1", "--save-every", "1",
            "--no-previews", "--vram", "24",
        ])
        assert trained == 0
        assert load_state(project).get("base") == ""
        assert project.read()["train"]["base"] == "tiny-test"
        capsys.readouterr()
        assert main(["--json", "lora", "finish", str(project.root)]) == 0
        result = _envelope(capsys.readouterr().out)["result"]
    finally:
        bases.unregister(TinyAdapter.info.mdl_id)
    assert result["state"] == "done"
    assert Path(str(result["voxlora"])).is_file()
    record = project.read()["train"]
    assert record["base"] == "tiny-test" and record["state"] == "done" and record["step"] == 2


def test_finish_fetches_the_recorded_base_when_its_files_are_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    from voxprint_forge.cli import main

    project = make_project(tmp_path / "proj", count=10)
    bases.register(TinyAdapter)
    calls: list[tuple[list[str], bool]] = []
    installed = {"ready": False}

    def files_for(mdl_id: str, manifest: object = None) -> dict[str, tuple[str, int]]:
        return {"weights.safetensors": ("ab", 1)} if mdl_id == "tiny-test" else {}

    def present(mdl_id: str) -> bool:
        return mdl_id == "tiny-test" and installed["ready"]

    def fetch(ids: list[str], gui: bool = False) -> dict[str, object]:
        calls.append((list(ids), gui))
        installed["ready"] = True
        return {"ok": True, "ready": True, "ids": list(ids)}

    monkeypatch.setattr("voxprint_forge.model_views.files_for", files_for)
    monkeypatch.setattr("voxprint_forge.mdl_app.present", present)
    monkeypatch.setattr("voxprint_forge.mdl_app.fetch", fetch)
    try:
        capsys.readouterr()
        assert main([
            "--json", "lora", "train", "--project", str(project.root), "--base", "tiny-test", "--no-fetch",
            "--review-rounds", "0", "--steps", "1", "--resolution", "64", "--batch", "1", "--save-every", "1",
            "--no-previews", "--vram", "24",
        ]) == 0
        capsys.readouterr()
        assert main(["--json", "lora", "finish", str(project.root)]) == 0
    finally:
        bases.unregister(TinyAdapter.info.mdl_id)
    assert calls == [(["tiny-test"], False)]
    assert _envelope(capsys.readouterr().out)["result"]["state"] == "done"


def test_a_failed_base_fetch_names_the_cli_command(monkeypatch) -> None:
    from voxprint_forge.lora.errors import LoraUsageError

    monkeypatch.setattr(
        "voxprint_forge.model_views.files_for", lambda mdl_id, manifest=None: {"weights.safetensors": ("ab", 1)},
    )
    monkeypatch.setattr("voxprint_forge.mdl_app.present", lambda mdl_id: False)
    monkeypatch.setattr("voxprint_forge.mdl_app.fetch", lambda ids, gui=False: {"ok": False, "ready": False})
    with pytest.raises(LoraUsageError) as caught:
        lora_cli._ensure_base_weights("z-image")
    message = str(caught.value)
    assert caught.value.code == "lora.err.base_missing"
    assert "voxprint-forge-cli models fetch" in message
    assert "Manage models" not in message


def test_cli_errors_do_not_name_gui_screens(tmp_path: Path) -> None:
    """Error text the CLI prints must not send someone to a window in the app."""
    import voxprint_forge.preflight as preflight
    from voxprint_forge.lora.i18n import catalog
    from voxprint_forge.preflight import refset_preflight

    texts: list[str] = []
    for key, value in catalog("en").items():
        if not str(key).startswith("lora.err"):
            continue
        if isinstance(value, dict):
            texts.extend(str(part) for name, part in value.items() if name != "by")
        else:
            texts.append(str(value))
    texts.extend(
        value for name, value in vars(preflight).items() if name.startswith("_FIX_") and isinstance(value, str)
    )
    report = refset_preflight(
        home=tmp_path,
        diffusers_version=lambda: "0.41.0",
        img2img=lambda: True,
        installed=lambda _mdl_id: False,
        vram=lambda: 24 * 2**30,
        ram=lambda: 64 * 2**30,
        mivolo=lambda _python: True,
    )
    texts.extend(str(row.get("fix") or "") for row in report["checks"] if isinstance(row, dict))
    screens = (
        "Manage models",
        "Check this computer",
        "Check and repair",
        "Check images",
        "Check photos",
        "in the GUI",
    )
    leaked = [text for text in texts if any(screen in text for screen in screens)]
    assert leaked == []


def test_train_help_points_people_at_balanced_and_headless_runs() -> None:
    from voxprint_forge.cli import build_parser

    parser = build_parser()
    top = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    lora = top.choices["lora"]
    commands = next(a for a in lora._actions if a.__class__.__name__ == "_SubParsersAction")
    text = " ".join(commands.choices["train"].format_help().split())
    assert "Use balanced for a person LoRA" in text
    assert "--review-rounds 0" in text
    assert ".voxlora" in text and "Canvas does not accept them" in text
