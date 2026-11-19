"""Default project/photos locations and the ``--photos`` resolution rule (suite rule 2026-11-15).

Every test stubs the Documents location (the ``XDG_DOCUMENTS_DIR`` environment variable, a
fake ``~/.config/user-dirs.dirs``, or the internal resolvers); nothing here reads the
machine's real Documents folder. Person folders use generic example names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voxprint_forge import paths
from voxprint_forge.paths import (
    PhotosResolutionError,
    default_photos_root,
    default_projects_dir,
    documents_dir,
    person_folders,
    resolve_photos_arg,
)

PROJECTS_TAIL = Path("VOXPrint Projects") / "Model Forge"


def _documents_via_env(monkeypatch, tmp_path: Path, name: str) -> Path:
    """Point XDG_DOCUMENTS_DIR at a tmp folder and return it."""
    docs = tmp_path / name
    docs.mkdir(parents=True)
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(docs))
    return docs


def test_documents_dir_follows_xdg_documents_env_with_non_ascii_name(monkeypatch, tmp_path: Path) -> None:
    docs = _documents_via_env(monkeypatch, tmp_path, "Документы")
    assert documents_dir() == docs  # the env value wins, whatever the folder is called


def test_default_projects_and_photos_dirs_use_the_env_documents_dir(monkeypatch, tmp_path: Path) -> None:
    docs = _documents_via_env(monkeypatch, tmp_path, "文档")
    assert default_projects_dir() == docs / PROJECTS_TAIL
    assert default_photos_root() == docs / PROJECTS_TAIL / "Photos"


def test_documents_dir_reads_user_dirs_file_expanding_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    home = tmp_path / "profile"
    config = home / ".config"
    config.mkdir(parents=True)
    (config / "user-dirs.dirs").write_text(
        'XDG_DOWNLOAD_DIR="$HOME/Downloads"\nXDG_DOCUMENTS_DIR="$HOME/文档"\n', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert documents_dir() == home / "文档"


def test_documents_dir_falls_back_to_home_documents(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(paths, "_xdg_user_dir_documents", lambda: None)
    monkeypatch.setattr(paths, "_windows_documents_dir", lambda: None)
    assert documents_dir() == home / "Documents"


def test_person_folders_lists_only_directories_sorted(tmp_path: Path) -> None:
    root = tmp_path / "Photos"
    (root / "Zed Example").mkdir(parents=True)
    (root / "Ada Example").mkdir()
    (root / "文档 Example").mkdir()
    (root / "notes.txt").write_text("not a person folder\n", encoding="utf-8")
    assert [folder.name for folder in person_folders(root)] == ["Ada Example", "Zed Example", "文档 Example"]
    assert person_folders(tmp_path / "missing") == []


def test_resolve_photos_absolute_path_is_used_as_is(tmp_path: Path) -> None:
    folder = tmp_path / "elsewhere" / "Ada Example"
    folder.mkdir(parents=True)
    assert resolve_photos_arg(folder, root=tmp_path / "Photos") == folder


def test_resolve_photos_person_name_finds_the_folder_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "Photos"
    person = root / "Ada Example"
    person.mkdir(parents=True)
    assert resolve_photos_arg("Ada Example", root=root) == person
    assert resolve_photos_arg(Path("Ada Example"), root=root) == person


def test_resolve_photos_accepts_a_relative_path_including_photos_from_the_projects_dir(
    tmp_path: Path,
) -> None:
    projects = tmp_path / PROJECTS_TAIL
    person = projects / "Photos" / "Ada Example"
    person.mkdir(parents=True)
    assert resolve_photos_arg(Path("Photos") / "Ada Example", root=projects / "Photos", cwd=projects) == person


def test_resolve_photos_unknown_name_names_the_expected_root(tmp_path: Path) -> None:
    root = tmp_path / "Photos"
    root.mkdir(parents=True)
    with pytest.raises(PhotosResolutionError) as err:
        resolve_photos_arg("Nobody Example", root=root)
    assert str(root) in str(err.value)
    assert "Nobody Example" in str(err.value)


def test_resolve_photos_without_value_uses_the_single_person_folder(tmp_path: Path) -> None:
    root = tmp_path / "Photos"
    person = root / "Ada Example"
    person.mkdir(parents=True)
    assert resolve_photos_arg(None, root=root) == person


def test_resolve_photos_without_value_lists_names_and_refuses_to_pick_one(tmp_path: Path) -> None:
    root = tmp_path / "Photos"
    (root / "Ada Example").mkdir(parents=True)
    (root / "Zed Example").mkdir()
    with pytest.raises(PhotosResolutionError) as err:
        resolve_photos_arg(None, root=root)
    assert err.value.names == ["Ada Example", "Zed Example"]
    message = str(err.value)
    assert "Ada Example" in message and "Zed Example" in message
    assert "--photos" in message


def test_resolve_photos_without_value_errors_clearly_when_no_person_folder_exists(tmp_path: Path) -> None:
    with pytest.raises(PhotosResolutionError) as err:
        resolve_photos_arg(None, root=tmp_path / "Photos")
    assert "Photos" in str(err.value)
    root = tmp_path / "Photos"
    root.mkdir(parents=True)
    with pytest.raises(PhotosResolutionError) as empty_err:
        resolve_photos_arg(None, root=root)
    assert str(root) in str(empty_err.value)
