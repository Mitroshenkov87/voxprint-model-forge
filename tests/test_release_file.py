"""``release.txt`` parsing: the model release file shared by the GUI and ``forge-train`` (no Qt).

Synthetic text only. The file is plain English ``Key: value`` lines; a complete file
(Adult (18+), Consent to training and LoRA training allowed all "yes") answers the
consent prompts, anything less only means the usual prompts are asked.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from voxprint_forge.lora.release_file import (
    RELEASE_FILE,
    TEMPLATE,
    ReleaseInfo,
    find_release,
    parse_release,
    slug,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

COMPLETE = (
    'Model name: "Tatiana Model"\n'
    "Adult (18+): yes\n"
    "No minors in photos: yes\n"
    "Consent to training: yes\n"
    "Consent to generation: yes\n"
    "LoRA training allowed: yes\n"
    "Adult content in training: yes\n"
    "Date: 2026-11-16\n"
)


def _write(folder: Path, text: str = COMPLETE, name: str = RELEASE_FILE) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    target.write_text(text, encoding="utf-8")
    return target


# ----- parse_release ---------------------------------------------------------


def test_parse_release_reads_quoted_and_unquoted_names_with_spaces(tmp_path) -> None:
    quoted = parse_release(_write(tmp_path / "a"))
    assert quoted.model_name == "Tatiana Model"
    unquoted = parse_release(_write(tmp_path / "b", COMPLETE.replace('"Tatiana Model"', "Tatiana Model")))
    assert unquoted.model_name == "Tatiana Model"
    # inner quotes are kept; only one surrounding pair is stripped
    inner = parse_release(_write(tmp_path / "c", COMPLETE.replace('"Tatiana Model"', '"Tatiana "Tanya" Model"')))
    assert inner.model_name == 'Tatiana "Tanya" Model'


def test_parse_release_keys_are_case_insensitive_and_values_trimmed(tmp_path) -> None:
    text = "  MODEL NAME:   Tatiana Model  \nadult (18+):   YES\nConsent To Training: yes\nDate:   2026-11-16  \n"
    info = parse_release(_write(tmp_path / "photos", text))
    assert info.model_name == "Tatiana Model"
    assert info.adult is True
    assert info.consent_training is True
    assert info.lora_allowed is None
    assert info.date == "2026-11-16"


def test_parse_release_ignores_the_explanation_block_even_with_data_lines_inside(tmp_path) -> None:
    text = COMPLETE + '--- How to fill this in ---\nAdult (18+): no\nModel name: "Someone Else"\n'
    info = parse_release(_write(tmp_path / "photos", text))
    assert info.adult is True
    assert info.model_name == "Tatiana Model"


def test_parse_release_missing_or_no_lines_are_not_set_and_incomplete(tmp_path) -> None:
    info = parse_release(_write(tmp_path / "photos", 'Model name: "Tatiana Model"\nAdult (18+): no\n'))
    assert info.adult is False
    assert info.consent_training is None
    assert info.lora_allowed is None
    assert not info.complete_consent
    # the adult content line alone never completes consent
    content_only = parse_release(_write(tmp_path / "other", "Adult content in training: yes\n"))
    assert content_only.adult_content is True
    assert not content_only.complete_consent


def test_parse_release_unknown_keys_blank_lines_and_other_yes_no_values_are_not_set(tmp_path) -> None:
    text = "\nModel agency: example\nAdult (18+): maybe\nAdult (18+): yes\n\nNo minors in photos: NO\n"
    info = parse_release(_write(tmp_path / "photos", text))
    assert info.adult is True  # a later line overrides an unparseable one
    assert info.no_minors is False
    assert info.consent_training is None


def test_parse_release_records_the_sha256_of_the_file_bytes(tmp_path) -> None:
    target = _write(tmp_path / "photos")
    info = parse_release(target)
    assert info.path == target
    assert info.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert ReleaseInfo(path=target, sha256=info.sha256).complete_consent is False


def test_parse_release_reads_beyond_ascii(tmp_path) -> None:
    info = parse_release(_write(tmp_path / "photos", COMPLETE.replace("Tatiana Model", "Татьяна")))
    assert info.model_name == "Татьяна"


# ----- find_release ------------------------------------------------------------


def test_find_release_accepts_case_variants_and_prefers_the_exact_name(tmp_path) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    assert find_release(folder) is None
    variant = _write(folder, name="Release.txt")
    assert find_release(folder) == variant
    exact = _write(folder)
    assert find_release(folder) == exact
    assert find_release(tmp_path / "missing") is None


def test_find_release_ignores_nested_files_directories_and_unreadable_files(tmp_path) -> None:
    folder = tmp_path / "photos"
    nested = folder / "inner"
    nested.mkdir(parents=True)
    (nested / RELEASE_FILE).write_text("nested", encoding="utf-8")
    assert find_release(folder) is None
    (folder / RELEASE_FILE).mkdir()
    assert find_release(folder) is None
    (folder / RELEASE_FILE).rmdir()
    release = _write(folder, "text")
    if os.name != "nt":  # chmod(0) does not drop the owner's read bit on Windows
        release.chmod(0)
        try:
            assert find_release(folder) is None
        finally:
            release.chmod(0o644)
    assert find_release(folder) == release


# ----- helpers -----------------------------------------------------------------


def test_slug_matches_the_forge_train_rule() -> None:
    assert slug("Tatiana Model") == "tatiana-model"
    assert slug("Ada Example!") == "ada-example-"
    assert slug("MIXED case_99") == "mixed-case-99"


def test_template_parses_to_all_yes(tmp_path) -> None:
    target = _write(tmp_path / "photos", TEMPLATE)
    info = parse_release(target)
    assert info.model_name == "Tatiana Model"
    assert info.adult is True
    assert info.no_minors is True
    assert info.consent_training is True
    assert info.consent_generation is True
    assert info.lora_allowed is True
    assert info.adult_content is True
    assert info.adult_generation is True
    assert info.date == "2026-11-15"
    assert info.complete_consent


def test_training_and_generation_adult_lines_are_separate(tmp_path) -> None:
    text = "Adult content in training: no\nAdult content in generation: yes\n"
    info = parse_release(_write(tmp_path / "photos", text))
    assert info.adult_content is False
    assert info.adult_generation is True
    legacy = parse_release(_write(tmp_path / "old", "Adult content allowed: yes\n"))
    assert legacy.adult_content is None and legacy.adult_generation is None


# ----- module CLI ---------------------------------------------------------------


def _bash() -> str:
    """Git bash on Windows. System32\\bash.exe is a WSL stub and is not used."""
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.is_file():
            return str(git_bash)
        pytest.skip("Git bash is not installed")
    return "bash"


def _module(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-m", "voxprint_forge.lora.release_file", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_shell_output_is_eval_safe_with_quotes_and_spaces_in_the_name(tmp_path) -> None:
    name = 'Tatiana "Tanya" Model'
    target = _write(tmp_path / "photos", COMPLETE.replace('"Tatiana Model"', f'"{name}"'))
    listing = _module("--shell", str(target))
    assert listing.returncode == 0, listing.stderr
    env = {**os.environ, "PYTHONPATH": str(SRC), "RELEASE_FILE": str(target), "PY": sys.executable}
    command = (
        'eval "$("$PY" -m voxprint_forge.lora.release_file --shell "$RELEASE_FILE")"; '
        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n%s" '
        '"$RELEASE_ADULT" "$RELEASE_NO_MINORS" "$RELEASE_CONSENT" "$RELEASE_ADULT_CONTENT" "$RELEASE_SHA256"'
    )
    checked = subprocess.run([_bash(), "-c", command], env=env, capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stderr
    adult, no_minors, consent, adult_content, digest = checked.stdout.splitlines()
    assert adult == "1" and no_minors == "1" and consent == "1" and adult_content == "1"
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert f"RELEASE_NAME={shlex.quote(name)}" in listing.stdout


def test_shell_output_marks_incomplete_consent(tmp_path) -> None:
    target = _write(tmp_path / "photos", "Adult (18+): yes\n")
    listing = _module("--shell", str(target))
    assert listing.returncode == 0, listing.stderr
    assert "RELEASE_CONSENT=0" in listing.stdout
    assert "RELEASE_NAME=''" in listing.stdout


def test_json_output_lists_every_field(tmp_path) -> None:
    target = _write(tmp_path / "photos")
    listing = _module("--json", str(target))
    assert listing.returncode == 0, listing.stderr
    payload = json.loads(listing.stdout)
    assert payload["model_name"] == "Tatiana Model"
    assert payload["adult"] is True
    assert payload["no_minors"] is True
    assert payload["consent_training"] is True
    assert payload["consent_generation"] is True
    assert payload["lora_allowed"] is True
    assert payload["adult_content"] is True
    assert payload["date"] == "2026-11-16"
    assert payload["complete_consent"] is True
    assert payload["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert payload["path"] == str(target)


def test_template_flag_prints_the_template() -> None:
    listing = _module("--template")
    assert listing.returncode == 0, listing.stderr
    assert listing.stdout == TEMPLATE


def test_module_cli_requires_one_action() -> None:
    listing = _module()
    assert listing.returncode == 2
