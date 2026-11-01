"""Windows installer: the CLI folder goes on the user Path once and leaves it on uninstall."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "windows" / "voxprint-model-forge.iss").read_text(encoding="utf-8-sig")


def _section(name: str) -> str:
    body = ISS.split(f"[{name}]", 1)[1]
    end = body.find("\n[")
    return body[:end] if end >= 0 else body


def _body(name: str) -> str:
    head = re.search(rf"^(procedure|function) {name}\b", ISS, re.M)
    assert head, name
    rest = ISS[head.start():]
    return rest[: rest.index("\nend;") + 5]


def test_setup_broadcasts_the_environment_change() -> None:
    assert re.search(r"^ChangesEnvironment=yes\s*$", _section("Setup"), re.M)


def test_cli_dir_is_the_venv_scripts_folder() -> None:
    assert "ExpandConstant('{app}\\venv\\Scripts')" in _body("CliDir")


def test_add_uses_the_user_path_and_skips_an_existing_entry() -> None:
    add = _body("AddCliToUserPath")
    assert "HKEY_CURRENT_USER, UserEnvKey, 'Path'" in add
    assert "UserEnvKey = 'Environment'" in ISS
    assert add.index("UserPathHas(Paths, CliDir)") < add.index("RegWriteExpandStringValue")
    assert "Exit;" in add.split("UserPathHas(Paths, CliDir)", 1)[1].split("RegWriteExpandStringValue", 1)[0]
    assert "HKEY_LOCAL_MACHINE" not in ISS and "HKLM" not in ISS


def test_entries_compare_case_insensitive_without_trailing_backslash() -> None:
    same = _body("SamePathEntry")
    assert "CompareText(" in same and "RemoveBackslashUnlessRoot(Trim(" in same


def test_path_is_added_only_after_a_successful_install() -> None:
    step = _body("CurStepChanged")
    assert step.index("ExitProcess(ResultCode)") < step.index("AddCliToUserPath;")
    assert step.index("AddCliToUserPath;") < step.index("if PrefetchPage.Values[0]")


def test_uninstall_removes_only_that_entry() -> None:
    assert "if CurUninstallStep = usUninstall then\n    RemoveCliFromUserPath;" in _body("CurUninstallStepChanged")
    remove = _body("RemoveCliFromUserPath")
    assert "if SamePathEntry(Entry, CliDir) then\n      Found := True\n    else" in remove
    assert "Kept := Kept + Entry;" in remove
    assert "if not Found then\n    Exit;" in remove


def _remove(paths: str, entry: str) -> str:
    """Python model of RemoveCliFromUserPath: drop matching entries, keep the rest verbatim."""

    def same(a: str, b: str) -> bool:
        return a.strip().rstrip("\\").lower() == b.strip().rstrip("\\").lower()

    return ";".join(part for part in paths.split(";") if not same(part, entry))


def test_removal_model_keeps_every_other_entry() -> None:
    cli = r"C:\Program Files\VOXPrint AI Suite\Model Forge\venv\Scripts"
    assert _remove(rf"C:\a;{cli};C:\b;", cli) == r"C:\a;C:\b;"
    assert _remove(rf"{cli}\;C:\a", cli) == r"C:\a"
    assert _remove(r"C:\a;;C:\b", cli) == r"C:\a;;C:\b"


def test_no_cyrillic_in_the_installer() -> None:
    for path in (ROOT / "installer").rglob("*"):
        if path.is_file() and path.suffix in {".iss", ".ps1", ".cs", ".txt", ".py"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            assert not re.search("[\u0400-\u04ff]", text), path
