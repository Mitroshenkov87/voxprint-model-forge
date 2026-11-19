"""Photos-step status cards: what the picked folder's release.txt and training.txt say.

Both cards read the folder the photos came from. ``release.txt`` is parsed by the
shared parser (:mod:`voxprint_forge.lora.release_file`); when it is missing, the
card offers the approved template — the exact text ``python -m
voxprint_forge.lora.release_file --template`` prints. ``training.txt`` is shown
as it is: the first lines, with no parser.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from voxprint_forge.app.text import tr
from voxprint_forge.lora.release_file import RELEASE_FILE, TEMPLATE, ReleaseInfo, find_release, parse_release

_PREVIEW_LINES = 6


def _title(text_key: str, column: QtWidgets.QVBoxLayout) -> None:
    title = QtWidgets.QLabel(tr(text_key))
    title.setObjectName("settingGroup")
    column.addWidget(title)


class ReleaseCard(QtWidgets.QFrame):
    """The photo folder's ``release.txt``: missing, found, or complete.

    A complete file (18+, consent to training and LoRA allowed all ``yes``) answers
    the consent prompts by itself. The card lists the fields Forge reads — the model
    name, 18+, consent, no-minors and the date — and which of them came from the file.
    When the file is missing, a button writes the approved template into the folder.
    """

    def __init__(
        self,
        folder: Path,
        on_template_saved: Callable[[Path], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("forge.photos.release")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._folder = Path(folder)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(8)
        _title("forge.app.photos.release.title", column)
        release = find_release(self._folder)
        if release is None:
            self._build_missing(column, on_template_saved)
            return
        info = parse_release(release)
        self._build_found(column, info)

    def _build_missing(self, column: QtWidgets.QVBoxLayout, on_template_saved: Callable[[Path], None]) -> None:
        self.setProperty("checkGate", "run")
        note = QtWidgets.QLabel(tr("forge.app.photos.release.missing"))
        note.setObjectName("forge.photos.release.note")
        note.setWordWrap(True)
        column.addWidget(note)
        button = QtWidgets.QPushButton(tr("forge.app.photos.release.template"))
        button.setObjectName("forge.photos.release.template")
        button.setToolTip(tr("forge.app.photos.release.template.tip"))
        button.setAutoDefault(False)
        button.clicked.connect(lambda _checked=False: self._save_template(on_template_saved))
        column.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

    def _save_template(self, on_template_saved: Callable[[Path], None]) -> None:
        target = self._folder / RELEASE_FILE
        target.write_text(TEMPLATE, encoding="utf-8", newline="\n")
        on_template_saved(self._folder)

    def _build_found(self, column: QtWidgets.QVBoxLayout, info: ReleaseInfo) -> None:
        missing = self._missing_fields(info)
        self.setProperty("checkGate", "go" if not missing else "run")
        status = QtWidgets.QLabel(
            tr("forge.app.photos.release.complete") if not missing
            else tr("forge.app.photos.release.incomplete", fields=", ".join(missing))
        )
        status.setObjectName("forge.photos.release.status")
        status.setWordWrap(True)
        column.addWidget(status)
        rows = QtWidgets.QGridLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(12)
        rows.setVerticalSpacing(4)
        for index, (key, value) in enumerate(self._fields(info)):
            name = QtWidgets.QLabel(tr(key))
            name.setObjectName("forge.photos.release.field")
            shown = QtWidgets.QLabel(value)
            shown.setObjectName("forge.photos.release.value")
            rows.addWidget(name, index, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            rows.addWidget(shown, index, 1, QtCore.Qt.AlignmentFlag.AlignTop)
        column.addLayout(rows)
        taken = self._taken(info)
        if taken:
            line = QtWidgets.QLabel(tr("forge.app.photos.release.taken", items=", ".join(taken)))
            line.setObjectName("forge.photos.release.taken")
            line.setWordWrap(True)
            column.addWidget(line)

    def _fields(self, info: ReleaseInfo) -> list[tuple[str, str]]:
        """The five fields the card shows, each with its value word."""
        word_yes = tr("forge.app.photos.release.yes")
        word_no = tr("forge.app.photos.release.no")
        word_missing = tr("forge.app.photos.release.missing_value")

        def yes_no(value: bool | None) -> str:
            return word_missing if value is None else (word_yes if value else word_no)

        return [
            ("forge.app.photos.release.field.name", info.model_name or word_missing),
            ("forge.app.photos.release.field.adult", yes_no(info.adult)),
            ("forge.app.photos.release.field.consent", yes_no(info.consent_training)),
            ("forge.app.photos.release.field.no_minors", yes_no(info.no_minors)),
            ("forge.app.photos.release.field.date", info.date or word_missing),
        ]

    def _missing_fields(self, info: ReleaseInfo) -> list[str]:
        """The labels of the shown fields that are not set (the name and the date may be empty)."""
        return [tr(key) for key, value in self._fields(info) if value == tr("forge.app.photos.release.missing_value")]

    def _taken(self, info: ReleaseInfo) -> list[str]:
        """What the app took from the file: the fields that are set."""
        return [tr(key) for key, value in self._fields(info) if value != tr("forge.app.photos.release.missing_value")]


class TrainingTextCard(QtWidgets.QFrame):
    """The optional ``training.txt`` next to the photos: its first lines, or "not present".

    The file is free text the trainer reads verbatim; the card shows it as it is.
    """

    def __init__(self, folder: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("forge.photos.training")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(8)
        _title("forge.app.photos.training.title", column)
        path = Path(folder) / "training.txt"
        if path.is_file():
            self.setProperty("checkGate", "go")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:_PREVIEW_LINES]
            body = QtWidgets.QLabel("\n".join(lines))
            body.setObjectName("forge.photos.training.body")
            body.setWordWrap(True)
            body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            column.addWidget(body)
        else:
            self.setProperty("checkGate", "run")
            note = QtWidgets.QLabel(tr("forge.app.photos.training.missing"))
            note.setObjectName("forge.photos.training.note")
            note.setWordWrap(True)
            column.addWidget(note)
