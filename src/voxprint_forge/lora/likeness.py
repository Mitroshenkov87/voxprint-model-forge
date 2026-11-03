"""Likeness score for a trained LoRA: SFace cosine between generated pictures and the training photos.

This is a quality score only. It never refuses, never blocks export, never filters a picture
and has no part in the safety check. A low score is a warning-level status on the report and
the model card.

* Embeddings: OpenCV YuNet (detector) + SFace (recognizer), MDL ids ``yunet-2023mar`` and
  ``sface-2021dec``. They run outside this process: in a ``voxprint_common.worker`` child when
  that module and OpenCV are importable here, otherwise in a subprocess
  (:mod:`voxprint_forge.lora.likeness_worker`) under a Python that has OpenCV.
* Reference: the mean of the L2-normalised embeddings of the project's kept training photos
  (``prepared/``), normalised again. ``ref_baseline`` is the mean over those photos of each
  photo against the mean of the others (leave one out).
* Per picture: cosine against the reference. A picture with no face is counted in ``no_face``
  and is never a pass. ``mean``, ``median`` and ``n`` cover the pictures with a face.
* Status: ``fail`` when the mean is below :data:`FAIL_BELOW` (the SFace same-person cosine);
  ``warn`` when it is below :data:`WARN_BELOW` or below :data:`BASELINE_SHARE` x ``ref_baseline``,
  or when no picture has a face; else ``pass``. ``skipped`` when the score could not run.

Files: ``eval/likeness.jsonl`` (one row per picture), ``eval/report.json`` and
``eval/report.md`` (the summary), and ``likeness`` on ``project.json``, which export copies
into ``voxprint-model.json`` when it describes the exported step.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from voxprint_forge.lora.project import Project

FAIL_BELOW = 0.363
WARN_BELOW = 0.50
BASELINE_SHARE = 0.6
DETECTOR_ID = "yunet-2023mar"
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_ID = "sface-2021dec"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
MODEL_IDS = (DETECTOR_ID, RECOGNIZER_ID)
EVAL_DIR = "eval"
ROWS_FILE = "likeness.jsonl"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
PROJECT_KEY = "likeness"
CARD_KEYS = ("mean", "median", "ref_baseline", "threshold", "status", "n", "no_face")
WORKER_HANDLER = "voxprint_forge.lora.likeness_worker:run_job"
WORKER_TIMEOUT = 1800.0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_STEP = re.compile(r"^step-(\d+)-\d+$")

Row = dict[str, object]
# paths -> one row per path: {"embedding": [floats] or None, "face_px": int}
Embedder = Callable[[Sequence[Path]], list[Row]]


class LikenessUnavailable(RuntimeError):
    """The score could not run (models absent, no OpenCV, worker failed). Never a refusal."""


# --- numbers -------------------------------------------------------------------------------------

def normalise(vector: Sequence[float]) -> list[float]:
    """``vector`` scaled to length 1. A zero vector stays zero."""
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    return [float(v) / norm for v in vector] if norm > 0 else [0.0 for _ in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of two vectors."""
    return sum(x * y for x, y in zip(normalise(a), normalise(b), strict=True))


def reference(embeddings: Sequence[Sequence[float]]) -> tuple[list[float], float | None, list[float]]:
    """The normalised mean, the leave-one-out baseline and each photo's leave-one-out cosine.

    The baseline is ``None`` with fewer than two photos.
    """
    if not embeddings:
        raise LikenessUnavailable("no training photo has a face")
    units = [normalise(e) for e in embeddings]
    total = [sum(column) for column in zip(*units, strict=True)]
    mean = normalise(total)
    if len(units) < 2:
        return mean, None, []
    loo = [cosine(unit, [t - u for t, u in zip(total, unit, strict=True)]) for unit in units]
    return mean, statistics.fmean(loo), loo


def status_for(mean: float | None, ref_baseline: float | None) -> str:
    """``fail``, ``warn`` or ``pass``. No face in any picture (``mean`` is ``None``) is ``warn``."""
    if mean is None:
        return "warn"
    if mean < FAIL_BELOW:
        return "fail"
    if mean < WARN_BELOW or (ref_baseline is not None and mean < BASELINE_SHARE * ref_baseline):
        return "warn"
    return "pass"


def summarize(cosines: Sequence[float], no_face: int, ref_baseline: float | None) -> dict[str, object]:
    """``mean``, ``median``, ``n``, ``no_face``, ``ref_baseline``, ``threshold`` and ``status``."""
    values = [float(c) for c in cosines]
    mean = statistics.fmean(values) if values else None
    median = statistics.median(values) if values else None
    return {
        "mean": _r(mean),
        "median": _r(median),
        "n": len(values),
        "no_face": int(no_face),
        "ref_baseline": _r(ref_baseline),
        "threshold": FAIL_BELOW,
        "status": status_for(mean, ref_baseline),
    }


def score(ref_rows: Sequence[Row], image_rows: Sequence[tuple[str, Row]]) -> dict[str, object]:
    """Reference from ``ref_rows``; one scored row per ``(name, row)`` image; and the summary."""
    refs = [list(map(float, r["embedding"])) for r in ref_rows if r.get("embedding")]  # type: ignore[arg-type]
    mean, baseline, loo = reference(refs)
    rows: list[Row] = []
    cosines: list[float] = []
    no_face = 0
    for name, row in image_rows:
        emb = row.get("embedding")
        if not emb:
            no_face += 1
            rows.append({"image": name, "cos": None, "face": False, "face_px": 0, "pass": False})
            continue
        value = cosine(list(map(float, emb)), mean)  # type: ignore[arg-type]
        cosines.append(value)
        rows.append({
            "image": name, "cos": _r(value), "face": True, "face_px": int(row.get("face_px") or 0),  # type: ignore[call-overload]
            "pass": value >= FAIL_BELOW,
        })
    summary = summarize(cosines, no_face, baseline)
    summary["refs"] = len(refs)
    summary["refs_no_face"] = len(ref_rows) - len(refs)
    summary["ref_loo_min"] = _r(min(loo)) if loo else None
    return {"summary": summary, "rows": rows}


def _r(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


# --- inputs --------------------------------------------------------------------------------------

def reference_photos(project: Project) -> list[Path]:
    """The kept, prepared training photos of the project."""
    return [project.root / item.file for item in project.items() if (project.root / item.file).is_file()]


def preview_groups(project: Project) -> dict[int, list[Path]]:
    """Preview PNGs by step, from ``previews/step-NNNNNN-I.png`` names."""
    groups: dict[int, list[Path]] = {}
    if not project.previews.is_dir():
        return groups
    for path in sorted(project.previews.glob("step-*.png")):
        match = _STEP.match(path.stem)
        if match:
            groups.setdefault(int(match.group(1)), []).append(path)
    return groups


def folder_images(folder: Path) -> list[Path]:
    """Pictures directly in ``folder``, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# --- running -------------------------------------------------------------------------------------

def model_files() -> tuple[Path, Path] | None:
    """The YuNet and SFace ONNX files when both are installed, else ``None``."""
    from voxprint_forge.lora.backends import installed_dir

    detector_dir, recognizer_dir = installed_dir(DETECTOR_ID), installed_dir(RECOGNIZER_ID)
    if detector_dir is None or recognizer_dir is None:
        return None
    detector, recognizer = detector_dir / DETECTOR_FILE, recognizer_dir / RECOGNIZER_FILE
    return (detector, recognizer) if detector.is_file() and recognizer.is_file() else None


def missing_models() -> list[str]:
    """MDL ids the score needs that are not installed."""
    from voxprint_forge.mdl_app import present

    return [mdl_id for mdl_id in MODEL_IDS if not present(mdl_id)]


def _has_cv2() -> bool:
    return importlib.util.find_spec("cv2") is not None


def _has_common_worker() -> bool:
    try:
        return importlib.util.find_spec("voxprint_common.worker") is not None
    except ModuleNotFoundError:
        return False


def _cv2_python() -> str:
    """This Python when it has OpenCV, else the MiVOLO env's (which ships OpenCV for YuNet)."""
    if _has_cv2():
        return sys.executable
    from voxprint_forge.lora.backends import mivolo_python

    return mivolo_python()


def default_embedder(detector: Path, recognizer: Path) -> Embedder:
    """Embed in a ``voxprint_common.worker`` child when possible, else in a subprocess."""

    def embed(paths: Sequence[Path]) -> list[Row]:
        payload = {"detector": str(detector), "recognizer": str(recognizer), "paths": [str(p) for p in paths]}
        if _has_common_worker() and _has_cv2():
            return _via_worker(payload)
        return _via_subprocess(payload)

    return embed


def _via_worker(payload: dict[str, object]) -> list[Row]:
    from voxprint_common.worker import Handlers, JobKind, JobSpec, JobState, WorkerHost

    with tempfile.TemporaryDirectory(prefix="vxp-likeness-") as state:
        host = WorkerHost(Handlers(run=WORKER_HANDLER), state_dir=Path(state))
        try:
            job = host.submit(JobSpec(kind=JobKind.INFERENCE, payload=payload, phase="lora.likeness"))
            outcome = host.wait(job, timeout=WORKER_TIMEOUT)
        finally:
            host.shutdown()
    if outcome.state != JobState.FINISHED or not isinstance(outcome.value, dict):
        raise LikenessUnavailable(f"worker {outcome.state}: {outcome.message}"[:200])
    return _rows(outcome.value, len(payload["paths"]))  # type: ignore[arg-type]


def _via_subprocess(payload: dict[str, object]) -> list[Row]:
    worker = Path(__file__).resolve().with_name("likeness_worker.py")
    try:
        done = subprocess.run(
            [_cv2_python(), "-I", str(worker)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LikenessUnavailable(type(exc).__name__) from exc
    try:
        reply = json.loads(done.stdout or "{}")
    except ValueError:
        reply = {}
    if not isinstance(reply, dict) or "rows" not in reply:
        error = reply.get("error") if isinstance(reply, dict) else None
        raise LikenessUnavailable(str(error or f"worker exited rc={done.returncode}")[:200])
    return _rows(reply, len(payload["paths"]))  # type: ignore[arg-type]


def _rows(reply: Mapping[str, object], count: int) -> list[Row]:
    rows = reply.get("rows")
    if not isinstance(rows, list) or len(rows) != count or not all(isinstance(r, dict) for r in rows):
        raise LikenessUnavailable("bad worker reply")
    return rows


# --- evaluate and record -------------------------------------------------------------------------

def evaluate(
    project: Project,
    groups: Mapping[str, Sequence[Path]],
    *,
    embedder: Embedder | None = None,
    record_step: int | None = None,
) -> dict[str, object]:
    """Score every group of pictures against the project's training photos and write the report.

    ``groups`` maps a label (``step-000500`` or ``images``) to its pictures. ``record_step`` is
    the step whose summary goes onto ``project.json`` for the model card; ``None`` records the
    last group. The result never raises for a low score. :class:`LikenessUnavailable` is raised
    when the score cannot run at all.
    """
    refs = reference_photos(project)
    if not refs:
        raise LikenessUnavailable("no prepared photos")
    if embedder is None:
        files = model_files()
        if files is None:
            raise LikenessUnavailable("models missing: " + ", ".join(MODEL_IDS))
        embedder = default_embedder(*files)
    labels = [label for label, paths in groups.items() if paths]
    if not labels:
        raise LikenessUnavailable("no pictures to score")
    flat = [path for label in labels for path in groups[label]]
    embedded = embedder([*refs, *flat])
    ref_rows, image_rows = embedded[: len(refs)], embedded[len(refs):]
    results: dict[str, dict[str, object]] = {}
    all_rows: list[Row] = []
    offset = 0
    for label in labels:
        paths = groups[label]
        chunk = image_rows[offset: offset + len(paths)]
        offset += len(paths)
        scored = score(ref_rows, [(_shown(project, p), row) for p, row in zip(paths, chunk, strict=True)])
        results[label] = scored["summary"]  # type: ignore[assignment]
        all_rows.extend({"group": label, **row} for row in scored["rows"])  # type: ignore[union-attr]
    chosen = _chosen_label(labels, record_step)
    best = _best(results)
    report = {
        "metric": "sface-cosine",
        "models": list(MODEL_IDS),
        "thresholds": {"fail_below": FAIL_BELOW, "warn_below": WARN_BELOW, "baseline_share": BASELINE_SHARE},
        "groups": results,
        "recorded": chosen,
        "best": best,
        "summary": results[chosen],
    }
    _write(project, report, all_rows)
    record = {key: results[chosen][key] for key in CARD_KEYS}
    step = _label_step(chosen)
    project.write({**project.read(), PROJECT_KEY: {**record, "step": step, "group": chosen}})
    return report


def _chosen_label(labels: Sequence[str], step: int | None) -> str:
    if step is not None:
        wanted = step_label(step)
        if wanted in labels:
            return wanted
    return labels[-1]


def _best(results: Mapping[str, Mapping[str, object]]) -> str | None:
    scored = [(float(s["mean"]), label) for label, s in results.items() if isinstance(s.get("mean"), float)]
    return max(scored)[1] if scored else None


def step_label(step: int) -> str:
    """``step-000500`` for step 500."""
    return f"step-{int(step):06d}"


def _label_step(label: str) -> int | None:
    match = re.fullmatch(r"step-(\d+)", label)
    return int(match.group(1)) if match else None


def _shown(project: Project, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write(project: Project, report: Mapping[str, object], rows: Sequence[Row]) -> None:
    folder = project.root / EVAL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ROWS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (folder / REPORT_JSON).write_text(
        json.dumps({"likeness": report}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    (folder / REPORT_MD).write_text(report_markdown(report), encoding="utf-8", newline="\n")


def report_markdown(report: Mapping[str, object]) -> str:
    """The likeness table for ``eval/report.md``."""
    groups = report.get("groups") if isinstance(report.get("groups"), dict) else {}
    lines = [
        "# Eval report",
        "",
        "## Likeness (SFace cosine vs the training photos)",
        "",
        f"Quality score only. fail < {FAIL_BELOW}; warn < {WARN_BELOW} or < {BASELINE_SHARE} x baseline. "
        "Pictures with no face count as no_face, never as a pass.",
        "",
        "| group | n | no face | mean | median | baseline | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in groups.items():  # type: ignore[union-attr]
        lines.append(
            f"| {label} | {s['n']} | {s['no_face']} | {_cell(s['mean'])} | {_cell(s['median'])} "
            f"| {_cell(s['ref_baseline'])} | {s['status']} |"
        )
    lines += ["", f"Recorded on the model card: {report.get('recorded')}. Best by mean: {report.get('best')}.", ""]
    return "\n".join(lines)


def _cell(value: object) -> str:
    return "-" if value is None else f"{float(value):.3f}"  # type: ignore[arg-type]


def record_skipped(project: Project, reason: str) -> dict[str, object]:
    """Note a score that could not run on ``project.json``; any earlier score is cleared."""
    data = project.read()
    project.write({**data, PROJECT_KEY: {"status": "skipped", "reason": reason[:200]}})
    return {"status": "skipped", "reason": reason[:200]}


def card_fields(data: Mapping[str, object], step: int | None) -> dict[str, object] | None:
    """The ``likeness`` object for ``voxprint-model.json``, when ``project.json`` has one for ``step``."""
    stored = data.get(PROJECT_KEY)
    if not isinstance(stored, dict) or stored.get("status") not in ("pass", "warn", "fail"):
        return None
    if step is not None and stored.get("step") != step:
        return None
    return {key: stored.get(key) for key in CARD_KEYS}


def score_after_finish(project: Project, step: int, *, embedder: Embedder | None = None) -> dict[str, object]:
    """Score the final previews of ``step``. Any failure is a ``skipped`` status, never an error."""
    pictures = preview_groups(project).get(step, [])
    if not pictures:
        return record_skipped(project, "no previews for the final step")
    try:
        report = evaluate(project, {step_label(step): pictures}, embedder=embedder, record_step=step)
    except LikenessUnavailable as exc:
        return record_skipped(project, str(exc))
    except Exception as exc:  # noqa: BLE001 - a quality score must never stop finish or export
        return record_skipped(project, f"{type(exc).__name__}: {exc}")
    return dict(report["summary"])  # type: ignore[call-overload]


def warning_line(summary: Mapping[str, object]) -> str | None:
    """One plain line for a ``warn`` or ``fail`` score, else ``None``. Advisory only."""
    status = summary.get("status")
    if status not in ("warn", "fail"):
        return None
    return (
        f"Likeness {status}: mean {_cell(summary.get('mean'))} over {summary.get('n')} pictures "
        f"({summary.get('no_face')} without a face), training-photo baseline {_cell(summary.get('ref_baseline'))}. "
        "This is a quality score only; nothing was blocked."
    )
