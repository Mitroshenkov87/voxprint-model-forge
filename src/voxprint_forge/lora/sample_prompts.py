"""Review-sample prompts: trigger, the subject, a framing, and one attire word.

The subject is the leading phrase of the short captions (``a woman`` from
``A woman models various stylish outfits``). With the ``mixed`` framing (the
default), framings rotate ``close-up portrait``, ``waist-up``, ``full body``;
the start index is ``(seed + round number) % 3``. The ``portrait`` framing
uses one face close-up for every sample, for judging likeness. ``auto`` resolves to ``unclothed`` when any
caption sets ``unclothed``, and to ``as_photos`` otherwise. ``as_photos`` with
clothed captions and ``clothed`` both use near-skin-tone underwear. The project
name is never used.

Training previews (``--previews``) without ``--preview-prompts`` use :func:`preview_prompt`:
for a person, ``<trigger> woman, head-and-shoulders portrait, face clearly visible, looking at
camera`` (the noun comes from the captions, else ``person``), so every preview shows the face;
for an object or a style, a plain, centred view of the subject.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, Never

from voxprint_forge.lora.captioner import text_marks_unclothed
from voxprint_forge.lora.project import Project
from voxprint_forge.lora.trigger import strip_trigger

# Words that end the subject and start the action or the clothing.
_STOP = frozenset({
    "against", "and", "are", "dressed", "has", "have", "holding", "holds",
    "in", "is", "looking", "looks", "modeling", "modelling", "models", "of",
    "on", "poses", "posing", "showing", "shows", "sitting", "sits", "standing",
    "stands", "walking", "walks", "was", "wearing", "wears", "were", "with", "wore",
})
_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

ATTIRE_MODES = ("auto", "as_photos", "clothed", "unclothed")
FRAMINGS = ("close-up portrait", "waist-up", "full body")
FRAMING_MODES = ("mixed", "portrait")
PORTRAIT_FRAMING = "close-up portrait, face in clear view, looking at the camera, even soft light"
SAMPLE_FRAMING = "sample_framing"
CLOTHED = "wearing near-skin-tone underwear"
UNCLOTHED = "unclothed"
SAMPLE_ATTIRE = "sample_attire"
PERSON_PREVIEW = "head-and-shoulders portrait, face clearly visible, looking at camera"
OBJECT_PREVIEW = "centered, whole subject in clear view, plain background, even soft light"
STYLE_PREVIEW = "a simple scene in this style, clear composition"
_ARTICLES = frozenset({"a", "an", "the"})

AttireMode = Literal["auto", "as_photos", "clothed", "unclothed"]
ResolvedAttire = Literal["as_photos", "clothed", "unclothed"]


def marks_unclothed(caption: Mapping[str, object]) -> bool:
    """True when this caption record marks unclothed.

    A boolean ``unclothed`` field wins. When that field is absent, the long and
    short text are checked.
    """
    flag = caption.get("unclothed")
    if isinstance(flag, bool):
        return flag
    return text_marks_unclothed(str(caption.get("long") or ""), str(caption.get("short") or ""))


def resolve_attire(mode: str, captions: Sequence[Mapping[str, object]]) -> ResolvedAttire:
    """``auto`` becomes ``unclothed`` or ``as_photos``. The other modes stay."""
    choice = _mode(mode)
    marked = any(marks_unclothed(caption) for caption in captions)
    match choice:
        case "auto":
            return "unclothed" if marked else "as_photos"
        case "as_photos":
            return "as_photos"
        case "clothed":
            return "clothed"
        case "unclothed":
            return "unclothed"
        case _ as unexpected:
            return _unexpected_mode(unexpected)


def attire_word(resolved: ResolvedAttire, captions: Sequence[Mapping[str, object]]) -> str:
    """One attire phrase: unclothed, or near-skin-tone underwear."""
    match resolved:
        case "unclothed":
            return UNCLOTHED
        case "clothed":
            return CLOTHED
        case "as_photos":
            return UNCLOTHED if any(marks_unclothed(caption) for caption in captions) else CLOTHED
        case _ as unexpected:
            return _unexpected_resolved(unexpected)


def build_sample_prompts(
    trigger: str,
    *,
    count: int,
    seed: int,
    round_no: int,
    mode: str,
    captions: Sequence[Mapping[str, object]],
    framing: str = "mixed",
) -> list[str]:
    """``count`` prompts for one round. The same seed and round repeat the list.

    ``framing`` is ``mixed`` (rotate the three framings) or ``portrait`` (a face
    close-up for every sample). Seeds still differ per sample.
    """
    if framing not in FRAMING_MODES:
        raise ValueError(framing)
    word = attire_word(resolve_attire(mode, captions), captions)
    subject = subject_phrase(captions, trigger)
    total = max(0, int(count))
    if total == 0:
        return []
    if framing == "portrait":
        return [_compose(trigger, subject, PORTRAIT_FRAMING, word) for _ in range(total)]
    start = (int(seed) + int(round_no)) % len(FRAMINGS)
    prompts = []
    for index in range(total):
        chosen = FRAMINGS[(start + index) % len(FRAMINGS)]
        prompts.append(_compose(trigger, subject, chosen, word))
    return prompts


def preview_prompt(trigger: str, kind: str, captions: Sequence[Mapping[str, object]]) -> str:
    """The default training-preview prompt, by kind (``person``, ``object``, ``style``)."""
    noun = " ".join(w for w in subject_phrase(captions, trigger).split() if w not in _ARTICLES)
    if kind == "person":
        return _compose(f"{trigger} {noun or 'person'}".strip(), "", PERSON_PREVIEW, "")
    if kind == "style":
        return _compose(f"{trigger} style".strip(), "", STYLE_PREVIEW, "")
    return _compose(f"{trigger} {noun or 'object'}".strip(), "", OBJECT_PREVIEW, "")


def project_kind(data: Mapping[str, object]) -> str:
    """``subject.kind`` from ``project.json``, else the top-level ``kind``."""
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    return str(subject.get("kind") or data.get("kind") or "")  # type: ignore[union-attr]


def subject_phrase(captions: Sequence[Mapping[str, object]], trigger: str = "") -> str:
    """The subject shared by the short captions, such as ``a woman``.

    The short text is used first. A short line that is only an action falls
    back to the long text. The most common phrase wins.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for caption in captions:
        phrase = _phrase_from_row(caption, trigger)
        if not phrase:
            continue
        if phrase not in counts:
            order.append(phrase)
        counts[phrase] = counts.get(phrase, 0) + 1
    if not order:
        return ""
    best = max(counts.values())
    for phrase in order:
        if counts[phrase] == best:
            return phrase
    return ""


def read_attire(data: Mapping[str, object]) -> str:
    """The project's sample-attire mode, or ``auto`` when it is missing."""
    raw = data.get(SAMPLE_ATTIRE)
    if isinstance(raw, str) and raw in ATTIRE_MODES:
        return raw
    return "auto"


def store_attire(project: Project, mode: str) -> str:
    """Write ``sample_attire`` on ``project.json`` and return the mode."""
    choice = _mode(mode)
    data = project.read()
    if data.get(SAMPLE_ATTIRE) != choice:
        project.write({**data, SAMPLE_ATTIRE: choice})
    return choice


def read_framing(data: Mapping[str, object]) -> str:
    """The project's sample framing, or ``mixed`` when it is missing."""
    raw = data.get(SAMPLE_FRAMING)
    if isinstance(raw, str) and raw in FRAMING_MODES:
        return raw
    return "mixed"


def store_framing(project: Project, framing: str) -> str:
    """Write ``sample_framing`` on ``project.json`` and return it."""
    if framing not in FRAMING_MODES:
        raise ValueError(framing)
    data = project.read()
    if data.get(SAMPLE_FRAMING) != framing:
        project.write({**data, SAMPLE_FRAMING: framing})
    return framing


def caption_rows(project: Project) -> list[dict[str, object]]:
    """Prepared captions, including ``unclothed`` when prepare stored it."""
    rows: list[dict[str, object]] = []
    for item in project.items():
        row: dict[str, object] = {"long": item.long, "short": item.short}
        if isinstance(item.unclothed, bool):
            row["unclothed"] = item.unclothed
        rows.append(row)
    return rows


def _compose(trigger: str, subject: str, framing: str, word: str) -> str:
    parts = [part.strip() for part in (trigger, subject, framing, word) if part and part.strip()]
    return ", ".join(parts)


def _phrase_from_row(caption: Mapping[str, object], trigger: str) -> str:
    short = strip_trigger(trigger, str(caption.get("short") or ""))
    phrase = _leading_subject(short)
    if phrase:
        return phrase
    long = strip_trigger(trigger, str(caption.get("long") or ""))
    return _leading_subject(long)


def _leading_subject(text: str) -> str:
    kept: list[str] = []
    for word in _WORDS.findall(text):
        low = word.lower()
        if low in _STOP:
            if kept:
                break
            continue
        kept.append(low)
        if len(kept) >= 4:
            break
    return " ".join(kept)


def _mode(mode: str) -> AttireMode:
    if mode not in ATTIRE_MODES:
        raise ValueError(mode)
    match mode:
        case "auto" | "as_photos" | "clothed" | "unclothed":
            return mode
        case _ as unexpected:
            return _unexpected_mode(unexpected)


def _unexpected_mode(mode: Never) -> Never:
    raise AssertionError(mode)


def _unexpected_resolved(mode: Never) -> Never:
    raise AssertionError(mode)
