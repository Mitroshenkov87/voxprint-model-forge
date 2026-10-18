"""Reference-to-dataset loop engine (``lora refset``) on CPU stubs: no GPU, no real models, no people."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from voxprint_forge.lora import refset
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import Decision, Gate
from voxprint_forge.lora.project import Project

SUBJECT = Subject("person", name="Shape Demo", trigger="shape_demo", consent_obtained=True, adult_confirmed=True,
                  consent_method="verbal")
PROMPT = "a gradient test pattern on a plain background"


def _refs(folder: Path, count: int, size: int = 640) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        image = Image.linear_gradient("L").resize((size, size)).convert("RGB")
        image = Image.merge("RGB", (image.split()[0], image.split()[1].point(lambda v, i=index: (v + 40 * i) % 256),
                                    Image.new("L", (size, size), 30 * index)))
        path = folder / f"ref-{index}.png"
        image.save(path)
        paths.append(path)
    return paths


class FakeGate(Gate):
    """Keeps every picture unless its top-left pixel is pure red (drop) or pure blue (minor signal)."""

    def __init__(self, log: list[str], *, available: bool = True, fail_all: bool = False) -> None:
        super().__init__(object() if available else None, object())
        self.log = log
        self.fail_all = fail_all
        self.checked = 0

    def check(self, image, *, attested=False):
        self.checked += 1
        pixel = image.getpixel((0, 0))
        if self.fail_all or pixel == (255, 0, 0):
            return Decision(("could_be_under_20",), False, "full")
        if pixel == (0, 0, 255):
            return Decision(("child_or_adolescent",), True, "full")
        return Decision((), False, "full")

    def unload(self):
        self.log.append("gate.unload")


class Loaders:
    """Records the order in which stages load and unload."""

    def __init__(self, *, marks: dict[int, tuple[int, int, int]] | None = None, fail_refs: bool = False,
                 gate_available: bool = True, same_vector: bool = False) -> None:
        self.log: list[str] = []
        self.marks = marks or {}
        self.fail_refs = fail_refs
        self.gate_available = gate_available
        self.same_vector = same_vector
        self.gates = 0

    def gate(self) -> Gate:
        self.gates += 1
        self.log.append("gate.load")
        made = FakeGate(self.log, available=self.gate_available, fail_all=self.fail_refs and self.gates == 1)
        made.judge = _Unloadable(self.log, "judge") if self.gate_available else None
        made.estimator = _Unloadable(self.log, "estimator")
        return made

    def generator(self):
        self.log.append("generator.load")
        return _Generator(self.log, self.marks)

    def embedder(self):
        self.log.append("embedder.load")
        return _Embedder(self.log, self.same_vector)


class _Unloadable:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def unload(self):
        self.log.append(f"{self.name}.unload")


class _Generator:
    name = "stub-turbo"

    def __init__(self, log, marks):
        self.log, self.marks, self.calls = log, marks, []

    def generate(self, image, prompt, *, seed, strength, steps):
        self.calls.append((seed, strength, steps, prompt))
        width, height = refset.target_size(image.size, 256)
        out = image.resize((width, height)).point(lambda v: (v + seed * 7) % 256)
        if seed in self.marks:
            out.putpixel((0, 0), self.marks[seed])
        else:
            out.putpixel((0, 0), (seed % 200 + 1, 9, 9))
        return out

    def unload(self):
        self.log.append("generator.unload")


class _Embedder:
    name = "stub-dino"

    def __init__(self, log, same):
        self.log, self.same = log, same

    def embed(self, images):
        if self.same:
            return [[1.0, 0.0, 0.0] for _ in images]
        return [[float(i + 1), float((i * 7) % 5), float((i * 3) % 11)] for i, _ in enumerate(images)]

    def unload(self):
        self.log.append("embedder.unload")


def _round(project, refs, loaders, settings=None):
    return refset.run_round(
        project, refs, SUBJECT, settings or refset.check_settings(8, 0.5, 9),
        load_gate=loaders.gate, load_generator=loaders.generator, load_embedder=loaders.embedder,
        progress=Progress(), describe=lambda _image: PROMPT,
    )


def test_a_round_checks_refs_then_draws_then_checks_each_candidate_one_stage_at_a_time(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    refs = _refs(tmp_path / "refs", 3)
    loaders = Loaders()
    result = _round(project, refs, loaders)
    assert result["shown"] == 8 and result["round"] == 0
    stages = [entry for entry in loaders.log if entry.endswith(".load") or entry in
              ("generator.unload", "embedder.unload", "estimator.unload")]
    assert stages == ["gate.load", "estimator.unload", "generator.load", "generator.unload", "gate.load",
                      "estimator.unload", "embedder.load", "embedder.unload"]
    state = refset.load_state(project)
    shown = refset.candidates(state)
    assert [item["seed"] for item in shown] == list(range(8))
    assert {item["parent"] for item in shown} == {r["id"] for r in state["refs"]}
    for item in shown:
        assert len(item["id"]) == 12 and int(item["id"], 16) >= 0
        assert item["verdict"] == "pending" and item["strength"] == 0.5 and item["steps"] == 9
        assert (refset.folder(project) / item["file"]).is_file()
        assert item["prompt_id"] == refset.prompt_id(PROMPT)
    assert not (refset.folder(project) / "round-0" / "pending").exists()
    assert refset.load_prompts(project) == {r["id"]: PROMPT for r in state["refs"]}
    assert PROMPT not in refset.state_file(project).read_text(encoding="utf-8")


def test_candidates_that_do_not_pass_are_deleted_silently(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    loaders = Loaders(marks={2: (255, 0, 0), 5: (255, 0, 0)})
    result = _round(project, _refs(tmp_path / "refs", 2), loaders)
    assert result["shown"] == 6
    shown = refset.candidates(refset.load_state(project))
    assert {item["seed"] for item in shown} == {0, 1, 3, 4, 6, 7}
    files = sorted(p.name for p in (refset.folder(project) / "round-0").iterdir())
    assert files == sorted(f"{item['id']}.png" for item in shown)
    assert "drop" not in json.dumps(refset.load_state(project))


def test_a_minor_signal_on_a_candidate_blocks_the_loop_and_shows_nothing(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    refs = _refs(tmp_path / "refs", 2)
    with pytest.raises(LoraBlocked) as caught:
        _round(project, refs, Loaders(marks={3: (0, 0, 255)}))
    assert caught.value.key == "lora.err.refset_blocked"
    state = refset.load_state(project)
    assert state["blocked"] and state["rounds"] == []
    assert not any((refset.folder(project) / "round-0").glob("*.png"))
    with pytest.raises(LoraBlocked):
        _round(project, refs, Loaders())
    assert refset.restart(project) is False and refset.load_state(project)["blocked"]
    with pytest.raises(LoraBlocked):
        refset.finish(project, Progress())


def test_a_reference_that_does_not_pass_blocks_the_loop_for_those_references(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    refs = _refs(tmp_path / "refs", 3)
    loaders = Loaders(fail_refs=True)
    with pytest.raises(LoraBlocked):
        _round(project, refs, loaders)
    assert "generator.load" not in loaders.log
    assert refset.load_state(project)["blocked"]
    again = Loaders()
    with pytest.raises(LoraBlocked):
        _round(project, refs, again)
    assert again.log == []  # the same references stay blocked without a new check
    other = _refs(tmp_path / "other", 2, size=700)
    assert _round(project, other, Loaders())["shown"] == 8


def test_no_gate_means_no_round(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    loaders = Loaders(gate_available=False)
    with pytest.raises(LoraBlocked) as caught:
        _round(project, _refs(tmp_path / "refs", 1), loaders)
    assert caught.value.key == "lora.err.gate_missing"
    assert "generator.load" not in loaders.log


def test_a_missing_teacher_is_a_plain_error(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    loaders = Loaders()
    with pytest.raises(LoraStageError) as caught:
        refset.run_round(project, _refs(tmp_path / "refs", 1), SUBJECT, refset.check_settings(8, 0.5, 9),
                         load_gate=loaders.gate, load_generator=lambda: None, progress=Progress(),
                         describe=lambda _i: PROMPT)
    assert caught.value.key == "lora.err.refset_teacher_missing"


@pytest.mark.parametrize(("count", "strength", "steps", "key"), [
    (7, 0.5, 9, "lora.err.refset_count"),
    (65, 0.5, 9, "lora.err.refset_count"),
    (24, 0.2, 9, "lora.err.refset_strength"),
    (24, 0.95, 9, "lora.err.refset_strength"),
    (24, 0.5, 3, "lora.err.refset_steps"),
    (24, 0.5, 17, "lora.err.refset_steps"),
])
def test_round_settings_out_of_range_are_refused(count, strength, steps, key) -> None:
    with pytest.raises(LoraUsageError) as caught:
        refset.check_settings(count, strength, steps)
    assert caught.value.key == key


def test_reference_count_and_size_are_checked(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    with pytest.raises(LoraUsageError) as caught:
        _round(project, _refs(tmp_path / "many", 11), Loaders())
    assert caught.value.key == "lora.err.refset_refs"
    with pytest.raises(LoraUsageError) as caught:
        _round(project, _refs(tmp_path / "small", 2, size=200), Loaders())
    assert caught.value.key == "lora.err.refset_unreadable"


def test_the_next_round_seeds_from_kept_pictures_and_the_references(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    refs = _refs(tmp_path / "refs", 2)
    _round(project, refs, Loaders())
    shown = refset.candidates(refset.load_state(project))
    refset.set_verdict(project, shown[0]["id"], "keep")
    refset.set_verdict(project, shown[1]["id"], "reject")
    loaders = Loaders()
    result = _round(project, refs, loaders, refset.check_settings(9, 0.35, 8))
    assert "gate.load" in loaders.log and loaders.log.count("gate.load") == 1  # refs are not re-checked
    state = refset.load_state(project)
    latest = refset.candidates(state)
    assert result["round"] == 1
    assert {item["parent"] for item in latest} == {r["id"] for r in state["refs"]} | {shown[0]["id"]}
    assert {item["seed"] for item in latest} == set(range(1000, 1009))
    assert all(item["strength"] == 0.35 and item["steps"] == 8 for item in latest)
    kept_child = next(item for item in latest if item["parent"] == shown[0]["id"])
    assert kept_child["ref"] == shown[0]["ref"]


def test_near_duplicates_are_marked(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    _round(project, _refs(tmp_path / "refs", 2), Loaders(same_vector=True))
    state = refset.load_state(project)
    assert state["near_duplicate_check"] is True
    assert all(item.get("near_duplicate_of") for item in refset.candidates(state))


def test_without_the_embedder_the_round_runs_without_marks(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    loaders = Loaders()
    refset.run_round(project, _refs(tmp_path / "refs", 1), SUBJECT, refset.check_settings(8, 0.5, 9),
                     load_gate=loaders.gate, load_generator=loaders.generator, load_embedder=lambda: None,
                     progress=Progress(), describe=lambda _i: PROMPT)
    state = refset.load_state(project)
    assert state["near_duplicate_check"] is False
    assert not any(item.get("near_duplicate_of") for item in refset.candidates(state))


def test_finish_hands_the_approved_set_to_prepare_and_records_how_it_was_made(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    refs = _refs(tmp_path / "refs", 3)
    _round(project, refs, Loaders())
    with pytest.raises(LoraUsageError) as caught:
        refset.finish(project, Progress())
    assert caught.value.key == "lora.err.refset_none_kept"
    shown = refset.candidates(refset.load_state(project))
    refset.set_verdict(project, shown[0]["id"], "keep")
    with pytest.raises(LoraUsageError) as caught:
        refset.finish(project, Progress())
    assert caught.value.key == "lora.err.too_few"
    for item in shown[1:7]:
        refset.set_verdict(project, item["id"], "keep")
    result = refset.finish(project, Progress(), mode="simple", base="z-image")
    assert result["approved"] == 10
    files = refset.approved_files(project)
    assert [Path(f) for f in result["files"]] == files and len(files) == 10
    metadata_text = (refset.folder(project) / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["base"] == "z-image"
    assert metadata["teacher"]["licence"] == "Apache-2.0"
    assert metadata["teacher"]["repo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert metadata["rounds"][0]["seeds"] == list(range(8)) and metadata["rounds"][0]["kept"] == 7
    assert sum(1 for row in metadata["approved"] if row["origin"] == "reference") == 3
    assert PROMPT not in metadata_text and str(tmp_path) not in metadata_text
    assert "ref-0" not in metadata_text
    data = project.read()
    assert data["refset"]["approved"] == 10 and data["refset"]["teacher"]["mdl_id"] == "z-image-turbo"
    refset.set_verdict(project, shown[0]["id"], "reject")
    assert refset.approved_files(project) == []  # a changed verdict reopens the set


def test_teacher_pin_matches_the_turbo_pin() -> None:
    nunchaku = pytest.importorskip("voxprint_forge.diffusion.nunchaku_zimage")
    assert refset.TEACHER_REPO == nunchaku.ZIMAGE_BASE_REPO
    assert refset.TEACHER_REVISION == nunchaku.ZIMAGE_BASE_REVISION


def test_target_size_keeps_the_aspect_on_a_16_pixel_grid() -> None:
    assert refset.target_size((640, 640)) == (1024, 1024)
    width, height = refset.target_size((3000, 2000))
    assert (width, height) == (1024, 688) and width % 16 == 0 and height % 16 == 0


def test_the_cli_has_refset_round_and_finish(tmp_path) -> None:
    from voxprint_forge.app.engine import commands
    from voxprint_forge.cli import build_parser

    assert "refset" in commands()
    args = build_parser().parse_args(["lora", "refset", "round", "--project", str(tmp_path), "--ref", "a.png",
                                      "--kind", "person", "--candidates", "16", "--strength", "0.4"])
    assert args.refset_command == "round" and args.candidates == 16 and args.strength == 0.4
    assert args.steps == refset.DEFAULT_STEPS
    args = build_parser().parse_args(["lora", "refset", "finish", "--project", str(tmp_path), "--mode", "expert"])
    assert args.refset_command == "finish" and args.base == "z-image"


def test_refset_text_never_says_what_the_check_looks_for() -> None:
    from voxprint_forge.lora.i18n import catalog

    pattern = re.compile(r"\b(minors?|child\w*|age|ages|under\s?(18|20)|несовершеннолет\w*|ребён\w*|возраст\w*|"
                         r"детск\w*)\b", re.IGNORECASE)
    for lang in ("en", "ru"):
        for key, value in catalog(lang).items():
            if "refset" in key:
                for form in (value.values() if isinstance(value, dict) else (value,)):
                    assert not pattern.search(form), (lang, key, form)


def test_refset_preflight_reports_each_requirement(tmp_path) -> None:
    from voxprint_forge.preflight import refset_preflight

    def run(**overrides):
        args = dict(home=tmp_path, diffusers_version=lambda: "0.41.0", img2img=lambda: True,
                    installed=lambda _mdl: True, vram=lambda: 16 * 2**30, ram=lambda: 32 * 2**30,
                    mivolo=lambda _python: True)
        args.update(overrides)
        return refset_preflight(**args)

    report = run()
    assert report["ok"] is True, report["problems"]
    names = {check["name"] for check in report["checks"]}
    assert {"diffusers", "img2img", "model:z-image-turbo", "model:qwen3-vl-4b-instruct", "model:mivolo-v2",
            "mivolo-env", "model:dinov2-base", "cuda", "ram", "disk"} <= names
    assert run(diffusers_version=lambda: "0.42.0")["ok"] is False
    assert run(diffusers_version=lambda: "0.37.1")["ok"] is False
    assert run(diffusers_version=lambda: None)["ok"] is False
    assert run(vram=lambda: None)["ok"] is False
    assert run(vram=lambda: 12 * 2**30)["ok"] is False
    assert run(ram=lambda: 16 * 2**30)["ok"] is False
    assert run(mivolo=lambda _python: False)["ok"] is False
    assert run(installed=lambda mdl: mdl != "z-image-turbo")["ok"] is False
    optional = run(installed=lambda mdl: mdl != "dinov2-base")
    assert optional["ok"] is True
    dino = next(c for c in optional["checks"] if c["name"] == "model:dinov2-base")
    assert dino["ok"] is False and dino["required"] is False


def test_preflight_refset_flag_runs_without_a_repo(monkeypatch, tmp_path) -> None:
    from voxprint_forge import cli, preflight

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preflight, "refset_preflight", lambda: {"ok": False, "problems": ["x"], "checks": []})
    assert cli.main(["--json", "preflight", "--refset"]) == 6
