"""Review rounds between training stages.

A stage trains on the user's photos, saves a checkpoint and draws a fixed set of
samples. The run then pauses. Liked samples join the next stage; disliked ones
are negatives only when the adapter says it can use them, and are otherwise
left out. The pause lives in ``<project>/review/state.json``, so it is still
there after a restart. Sample pictures stay in that project folder and carry
no extra metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never, Protocol

from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.project import Item, Project

SCHEMA = "voxprint-lora-review/1"
PHASES = ("idle", "training", "review", "finished")
VERDICTS = ("unmarked", "like", "dislike")
# Simple mode, when "Extend the set from samples" is on. The user can finish sooner.
DEFAULT_ROUNDS = 3
REVIEW_ROUNDS_MIN = 1
REVIEW_ROUNDS_MAX = 6
DEFAULT_SAMPLES = 8
DEFAULT_FIRST_ROUND_STEPS = 200
FIRST_ROUND_STEPS_MIN = 100
FIRST_ROUND_STEPS_MAX = 500
DEFAULT_FIRST_ROUND_SAMPLES = 6
DEFAULT_SAMPLE_STEPS = 28
# One denoising step, estimated as half a measured training step until a sample is timed.
SAMPLE_STEP_SHARE = 0.5
Action = Literal["start", "pause", "continue", "train_full", "finish", "mark"]


def state_path(project: Project) -> Path:
    """``review/state.json`` inside the project folder."""
    return project.root / "review" / "state.json"


def empty_state() -> dict[str, object]:
    """A project that has not started review rounds."""
    return {
        "schema": SCHEMA,
        "phase": "idle",
        "rounds": 0,
        "samples_per_round": 0,
        "first_round_steps": 0,
        "first_round_samples": 0,
        "sample_steps": 0,
        "total_steps": 0,
        "no_more_reviews": False,
        "index": 0,
        "seed": 0,
        "base": "",
        "checkpoint": "",
        "prompts": [],
        "config": {},
        "samples": [],
        "history": [],
    }


def load_state(project: Project) -> dict[str, object]:
    """The saved review, or an idle state when the project has none."""
    path = state_path(project)
    if not path.is_file():
        return empty_state()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise LoraUsageError("lora.err.review_phase")
    return data


def save_state(project: Project, state: dict[str, object]) -> None:
    """Write the review state atomically. Sample files are not touched."""
    path = state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, **{key: value for key, value in state.items() if key != "schema"}}
    _atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def counts(state: dict[str, object]) -> dict[str, int]:
    """Liked, disliked and unmarked samples of the round that is waiting."""
    samples = _samples(state.get("samples"))
    return {
        "liked": sum(1 for sample in samples if sample.get("verdict") == "like"),
        "disliked": sum(1 for sample in samples if sample.get("verdict") == "dislike"),
        "unmarked": sum(1 for sample in samples if sample.get("verdict", "unmarked") == "unmarked"),
        "round": int(state.get("index") or 0),
        "rounds": int(state.get("rounds") or 0),
    }


def can_continue(state: dict[str, object]) -> bool:
    """True when another stage is still planned and a review is waiting."""
    return state.get("phase") == "review" and int(state.get("index") or 0) < int(state.get("rounds") or 0)


def transition(state: dict[str, object], action: Action, **payload: object) -> dict[str, object]:
    """Move ``state`` by ``action``. An unknown action or a wrong phase is an error."""
    match action:
        case "start":
            return _start(state, payload)
        case "pause":
            return _pause(state, payload)
        case "continue":
            return _continue(state)
        case "train_full":
            return _train_full(state)
        case "finish":
            return _finish(state)
        case "mark":
            return _mark(state, payload)
    return _unexpected(action)


def plan_samples(state: dict[str, object]) -> list[dict[str, object]]:
    """The pictures for the stage in progress: fixed seeds and the project prompts."""
    round_no = int(state.get("index") or 0) + 1
    per_round = samples_for_round(state, round_no)
    seed = int(state.get("seed") or 0)
    prompts = [str(prompt) for prompt in _list(state.get("prompts"))] or [""]
    planned = []
    for index in range(per_round):
        prompt = prompts[index % len(prompts)]
        planned.append(
            {
                "id": f"r{round_no}-s{index:02d}",
                "file": f"review/round-{round_no:04d}/s{index:02d}.png",
                "seed": seed + (round_no - 1) * per_round + index,
                "prompt": prompt,
                "round": round_no,
                "verdict": "unmarked",
            }
        )
    return planned


def samples_for_round(state: dict[str, object], round_no: int) -> int:
    """How many pictures this round draws. The first round has its own count."""
    if round_no <= 1:
        first = int(state.get("first_round_samples") or 0)
        if first > 0:
            return first
    return int(state.get("samples_per_round") or 0)


def sample_steps_of(state: dict[str, object]) -> int:
    """Inference steps for a review picture. Missing values use the default of 28."""
    value = int(state.get("sample_steps") or 0)
    return value if value > 0 else DEFAULT_SAMPLE_STEPS


def total_steps_of(state: dict[str, object]) -> int:
    """The preset step count, from the review or from the saved training config."""
    total = int(state.get("total_steps") or 0)
    if total > 0:
        return total
    raw = state.get("config")
    if isinstance(raw, dict):
        return max(0, int(raw.get("steps") or 0))
    return 0


def step_plan(state: dict[str, object]) -> tuple[int, ...]:
    """Steps in each planned round.

    The first round is ``first_round_steps``, clamped to the preset total.
    Later rounds share the steps that remain, and the last of them takes the
    remainder. A single planned round stops at the first-round length; the
    steps still left are what train-fully runs. Planned rounds plus that
    remainder equal the preset total.
    """
    rounds = int(state.get("rounds") or 0)
    total = total_steps_of(state)
    if rounds < 1 or total < 1:
        return ()
    first = int(state.get("first_round_steps") or 0)
    if first < 1:
        first = min(DEFAULT_FIRST_ROUND_STEPS, total)
    first = min(first, total)
    if rounds == 1:
        return (first,)
    remaining = total - first
    later = rounds - 1
    base, extra = divmod(remaining, later)
    return (first, *([base] * (later - 1)), base + extra)


def steps_completed(state: dict[str, object]) -> int:
    """Steps finished in rounds that have already paused for review."""
    plan = step_plan(state)
    index = int(state.get("index") or 0)
    if index <= 0 or not plan:
        return 0
    return sum(plan[: min(index, len(plan))])


def steps_for_stage(state: dict[str, object]) -> int:
    """Steps for the stage that is running or about to run.

    Train-fully takes every preset step that is not finished yet, in one stage.
    """
    if state.get("no_more_reviews"):
        return max(0, total_steps_of(state) - steps_completed(state))
    plan = step_plan(state)
    index = int(state.get("index") or 0)
    if index < 0 or index >= len(plan):
        return 0
    return plan[index]


def check_round_options(
    *,
    first_round_steps: int,
    first_round_samples: int,
    samples_per_round: int,
    sample_steps: int,
) -> None:
    """Reject a review setup outside the supported range."""
    if samples_per_round < 1 or first_round_samples < 1:
        raise LoraUsageError("lora.err.review_rounds")
    if not FIRST_ROUND_STEPS_MIN <= first_round_steps <= FIRST_ROUND_STEPS_MAX:
        raise LoraUsageError(
            "lora.err.first_round_steps", low=FIRST_ROUND_STEPS_MIN, high=FIRST_ROUND_STEPS_MAX,
        )
    if sample_steps < 1:
        raise LoraUsageError("lora.err.sample_steps")


def check_review_round_count(review_rounds: int) -> None:
    """Reject a review count other than straight-through (0) or 1–6.

    The state machine still accepts any positive count. This check is for a new
    run: the CLI and the GUI offer 1–6, and 0 trains straight through.
    """
    if review_rounds == 0:
        return
    if not REVIEW_ROUNDS_MIN <= review_rounds <= REVIEW_ROUNDS_MAX:
        raise LoraUsageError(
            "lora.err.review_round_count", low=REVIEW_ROUNDS_MIN, high=REVIEW_ROUNDS_MAX,
        )


def first_pictures_eta(
    *,
    steps_left: int,
    seconds_per_step: float,
    samples_left: int,
    sample_steps: int,
    seconds_per_sample: float | None,
) -> dict[str, object]:
    """Seconds until the first review pictures, and whether the sample time is estimated.

    ``steps_left`` times the measured seconds per step, plus the pictures still
    to draw times the seconds per picture. Until one picture has been timed,
    each denoising step is estimated as half a training step.
    """
    estimated = seconds_per_sample is None
    per_sample = (
        max(0.0, seconds_per_step) * SAMPLE_STEP_SHARE * max(0, sample_steps)
        if estimated
        else max(0.0, float(seconds_per_sample))
    )
    total = max(0, steps_left) * max(0.0, seconds_per_step) + max(0, samples_left) * per_sample
    return {
        "first_pictures_seconds": int(round(total)),
        "seconds_per_sample": None if estimated else round(per_sample, 3),
        "sample_seconds_estimated": estimated,
    }


def passes_output_check(image: object) -> bool:
    """Whether a generated picture is kept.

    Training previews keep the picture the adapter returned and drop ``None``.
    Review samples use that same check and no other.
    """
    return image is not None


def save_image(image: object, path: Path) -> None:
    """Write one RGB PNG in the project folder, with no text or camera metadata."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise LoraUsageError("lora.err.review_sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    clean.save(path, format="PNG")


def apply_file(state: dict[str, object], path: Path) -> dict[str, object]:
    """Apply verdicts from a text file (``like id`` / ``dislike id``) or a JSON object."""
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise LoraUsageError("lora.err.review_sample")
        return _apply_groups(
            state,
            like=_strings(data.get("like")),
            dislike=_strings(data.get("dislike")),
            unmarked=_strings(data.get("unmarked")),
        )
    likes: list[str] = []
    dislikes: list[str] = []
    unmarked: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2 or parts[0] not in ("like", "dislike", "unmarked"):
            raise LoraUsageError("lora.err.review_sample")
        target = {"like": likes, "dislike": dislikes, "unmarked": unmarked}[parts[0]]
        target.append(parts[1])
    return _apply_groups(state, like=likes, dislike=dislikes, unmarked=unmarked)


def set_verdicts(
    state: dict[str, object],
    *,
    like: Sequence[str] = (),
    dislike: Sequence[str] = (),
    unmarked: Sequence[str] = (),
) -> dict[str, object]:
    """Set verdicts on the current samples. Other samples keep their marks."""
    return _apply_groups(state, like=list(like), dislike=list(dislike), unmarked=list(unmarked))


class ReviewDecider(Protocol):
    """Chooses like, dislike, or unmarked for each sample of the open review.

    The human GUI and ``lora review`` write the marks. A later automatic decider
    can score the same samples and return the same mapping. The round state
    machine does not grow a new action for that.
    """

    def decide(self, samples: Sequence[Mapping[str, object]]) -> Mapping[str, str]:
        """Sample id to ``like``, ``dislike``, or ``unmarked``."""
        ...


class HumanReviewDecider:
    """Reads the verdicts the GUI or ``lora review`` already stored. It does not score samples."""

    def decide(self, samples: Sequence[Mapping[str, object]]) -> Mapping[str, str]:
        """Sample id to the stored ``like``, ``dislike``, or ``unmarked``."""
        verdicts: dict[str, str] = {}
        for sample in samples:
            sample_id = str(sample.get("id") or "")
            if not sample_id:
                continue
            verdict = str(sample.get("verdict") or "unmarked")
            verdicts[sample_id] = verdict if verdict in VERDICTS else "unmarked"
        return verdicts


def apply_decider(state: dict[str, object], decider: ReviewDecider) -> dict[str, object]:
    """Write ``decider`` verdicts onto the open review.

    Called before ``continue`` and ``train_full``, which archive the samples.
    ``HumanReviewDecider`` is idempotent: marks already on the samples stay.
    """
    chosen = dict(decider.decide(_samples(state.get("samples"))))
    like: list[str] = []
    dislike: list[str] = []
    unmarked: list[str] = []
    for sample_id, verdict in chosen.items():
        if verdict == "like":
            like.append(sample_id)
        elif verdict == "dislike":
            dislike.append(sample_id)
        elif verdict == "unmarked":
            unmarked.append(sample_id)
        else:
            raise LoraUsageError("lora.err.review_sample")
    return set_verdicts(state, like=like, dislike=dislike, unmarked=unmarked)


@dataclass(frozen=True)
class RoundSets:
    """What the next stage trains on, and which disliked ids were left out."""

    include: tuple[Item, ...]
    negatives: tuple[Item, ...]
    excluded: tuple[str, ...]


def build_sets(original: Sequence[Item], state: dict[str, object], *, supports_negatives: bool) -> RoundSets:
    """Original photos plus liked samples. Disliked samples are negatives only when supported.

    Unmarked samples are left out of both sets. When ``supports_negatives`` is false the
    disliked samples are only excluded.
    """
    liked: list[Item] = []
    disliked: list[dict[str, object]] = []
    for sample in _all_samples(state):
        verdict = str(sample.get("verdict") or "unmarked")
        if verdict == "like":
            liked.append(_item(sample))
        elif verdict == "dislike":
            disliked.append(sample)
    excluded = tuple(str(sample["id"]) for sample in disliked)
    negatives = tuple(_item(sample) for sample in disliked) if supports_negatives else ()
    return RoundSets(tuple(original) + tuple(liked), negatives, excluded)


def summary(state: dict[str, object]) -> dict[str, object]:
    """The short review record stored beside the training result."""
    counted = counts(state)
    return {"phase": state.get("phase"), "round": counted["round"], "rounds": counted["rounds"]}


def _start(state: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    phase = str(state.get("phase") or "idle")
    if phase == "review":
        raise LoraUsageError("lora.err.review_pending")
    rounds = int(payload["rounds"])
    samples = int(payload["samples_per_round"])
    if rounds < 1 or samples < 1:
        raise LoraUsageError("lora.err.review_rounds")
    first_samples = int(payload.get("first_round_samples") or 0)
    if first_samples < 1:
        first_samples = samples
    sample_steps = int(payload.get("sample_steps") or 0)
    if sample_steps < 1:
        sample_steps = DEFAULT_SAMPLE_STEPS
    config = payload.get("config")
    stored = dict(config) if isinstance(config, dict) else {}
    total = int(payload.get("total_steps") or 0)
    if total < 1:
        total = int(stored.get("steps") or 0)
    prompts = [str(prompt) for prompt in _list(payload.get("prompts"))]
    return {
        **empty_state(),
        "phase": "training",
        "rounds": rounds,
        "samples_per_round": samples,
        "first_round_steps": int(payload.get("first_round_steps") or 0),
        "first_round_samples": first_samples,
        "sample_steps": sample_steps,
        "total_steps": total,
        "seed": int(payload.get("seed") or 0),
        "base": str(payload.get("base") or ""),
        "prompts": prompts,
        "config": stored,
    }


def _pause(state: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    if state.get("phase") != "training":
        raise LoraUsageError("lora.err.review_phase")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise LoraUsageError("lora.err.review_sample")
    return {
        **state,
        "phase": "review",
        "index": int(state.get("index") or 0) + 1,
        "checkpoint": str(payload.get("checkpoint") or ""),
        "samples": [dict(sample) for sample in samples if isinstance(sample, dict)],
    }


def _archive(state: dict[str, object]) -> dict[str, object]:
    """Move the open round into history and go back to training."""
    history = [dict(entry) for entry in _list(state.get("history")) if isinstance(entry, dict)]
    if state.get("samples"):
        history.append({"round": int(state.get("index") or 0), "samples": list(_samples(state.get("samples")))})
    return {**state, "phase": "training", "samples": [], "history": history}


def _continue(state: dict[str, object]) -> dict[str, object]:
    if state.get("phase") != "review":
        raise LoraUsageError("lora.err.review_phase")
    if not can_continue(state):
        raise LoraUsageError("lora.err.review_finished")
    return {**_archive(state), "no_more_reviews": False}


def _train_full(state: dict[str, object]) -> dict[str, object]:
    """Keep the current marks and train every remaining preset step, with no further review."""
    if state.get("phase") != "review":
        raise LoraUsageError("lora.err.review_phase")
    return {**_archive(state), "no_more_reviews": True}


def _finish(state: dict[str, object]) -> dict[str, object]:
    phase = state.get("phase")
    if phase == "review" or (phase == "training" and state.get("no_more_reviews")):
        return {**state, "phase": "finished"}
    raise LoraUsageError("lora.err.review_phase")


def _mark(state: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    sample_id = str(payload.get("id") or "")
    verdict = str(payload.get("verdict") or "")
    if verdict not in VERDICTS:
        raise LoraUsageError("lora.err.review_sample")
    return _apply_groups(
        state,
        like=[sample_id] if verdict == "like" else [],
        dislike=[sample_id] if verdict == "dislike" else [],
        unmarked=[sample_id] if verdict == "unmarked" else [],
    )


def _apply_groups(
    state: dict[str, object],
    *,
    like: list[str],
    dislike: list[str],
    unmarked: list[str],
) -> dict[str, object]:
    if state.get("phase") != "review":
        raise LoraUsageError("lora.err.review_phase")
    likes, dislikes, clears = set(like), set(dislike), set(unmarked)
    overlap = (likes & dislikes) | (likes & clears) | (dislikes & clears)
    if overlap:
        raise LoraUsageError("lora.err.review_conflict")
    known = {str(sample.get("id")) for sample in _samples(state.get("samples"))}
    unknown = (likes | dislikes | clears) - known
    if unknown:
        raise LoraUsageError("lora.err.review_sample")
    updated = []
    for sample in _samples(state.get("samples")):
        sample_id = str(sample.get("id"))
        verdict = str(sample.get("verdict") or "unmarked")
        if sample_id in likes:
            verdict = "like"
        elif sample_id in dislikes:
            verdict = "dislike"
        elif sample_id in clears:
            verdict = "unmarked"
        updated.append({**sample, "verdict": verdict})
    return {**state, "samples": updated}


def _all_samples(state: dict[str, object]) -> list[dict[str, object]]:
    found = []
    for entry in _list(state.get("history")):
        if isinstance(entry, dict):
            found.extend(_samples(entry.get("samples")))
    found.extend(_samples(state.get("samples")))
    return found


def _item(sample: dict[str, object]) -> Item:
    prompt = str(sample.get("prompt") or "")
    return Item(str(sample["id"]), str(sample["file"]), prompt, prompt, int(sample.get("round") or 0))


def _samples(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(sample) for sample in value if isinstance(sample, dict)]


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _unexpected(action: Never) -> dict[str, object]:
    raise AssertionError(f"unhandled review action {action!r}")


def _atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
