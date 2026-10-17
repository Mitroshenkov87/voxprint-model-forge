"""The Model Forge window on the shared Yitro shell (voxprint-common v0.2.5).

Surface ``model-forge`` (tool type). Two sections: "Train a model" with the
five-step wizard (photos, subject and consent, check and prepare, train,
result) and "Check this computer". The shell draws the chrome, the tab row,
the step title and counter, Back and Next, the overflow menu and the Expert
panel. This module fills each step's body, gates Next with a plain reason and
feeds the engine's progress sentences into the step.

v0.2.5 has no public hook for a custom wizard body, so :class:`ForgeWidget`
reaches a few documented-nowhere attributes of ``SimpleImageWidget``. They
are all read in :meth:`ForgeWidget._parts`; a test fails if the pinned shell
changes them.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from voxprint_common.i18n import add_catalogs, get_language, tr
from voxprint_common.paths import state_dir
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.sections import BodyKind, Section, SectionSet, Wizard, WizardStep
from voxprint_common.ui.simple_image import SimpleImageWidget, SimpleLayout
from voxprint_common.ui.theme import build_shell_style
from voxprint_common.ui.variants import state_names, surface_for

from voxprint_forge.app import engine
from voxprint_forge.app.draft import KINDS, STEPS, Draft, Prepared
from voxprint_forge.app.settings import PRESET_GROUP, SECTION, WIZARD, config_values, lora_settings, preset_file

APP_ID = "model-forge"
LOCALES = Path(__file__).resolve().parent / "locales"


def register_catalogs() -> None:
    """GUI strings, then the engine's ``lora.*`` strings (plan: add_catalogs)."""
    add_catalogs(LOCALES)
    try:
        from voxprint_forge.lora import i18n as lora_i18n

        add_catalogs(Path(lora_i18n.__file__).resolve().parent)
    except ImportError:
        pass


def section_set(settings: object) -> SectionSet:
    """The two sections and the trainer wizard."""
    steps = tuple(WizardStep(step, f"lora.step.{step}.title") for step in STEPS)
    return SectionSet(
        sections=(
            Section(SECTION, "forge.app.section.train"),
            Section("check", "forge.app.section.check", body=BodyKind.NOTE, note_key="forge.app.check.note"),
        ),
        wizards=(Wizard(WIZARD, "forge.app.wizard.title", SECTION, steps),),
        settings=settings,  # type: ignore[arg-type]
    )


class ForgeWidget(SimpleImageWidget):
    """The shared screen plus a hook that runs after every shell repaint."""

    after_refresh: Callable[[], None] | None = None

    def refresh(self, *, prompt: bool = False) -> None:
        """Let the shell paint, then fill the step body and gate Next."""
        super().refresh(prompt=prompt)
        if self.after_refresh is not None:
            self.after_refresh()

    def _parts(self) -> dict[str, object]:
        """The shell widgets this app extends (pinned to v0.2.5)."""
        return {
            "body": self._wizard_body,  # type: ignore[attr-defined]
            "signature": self._wizard_signature,  # type: ignore[attr-defined]
            "next": self._wizard_next,  # type: ignore[attr-defined]
            "back": self._wizard_back,  # type: ignore[attr-defined]
            "start": self._wizard_start,  # type: ignore[attr-defined]
            "cancel": self._wizard_cancel,  # type: ignore[attr-defined]
            "note": self._note_page,  # type: ignore[attr-defined]
            "overflow": self._overflow,  # type: ignore[attr-defined]
            "expert": self._expert_toggle,  # type: ignore[attr-defined]
            "tabs": self._section_bar,  # type: ignore[attr-defined]
        }

    EXPERT_WIDTH = 380

    def _fit_expert_panel(self) -> None:
        """Open the Expert panel wide enough for its labels, values and the Reset button."""
        super()._fit_expert_panel()  # type: ignore[misc]
        split = self._centre_split  # type: ignore[attr-defined]
        total = split.width() - split.handleWidth()
        if total > self.EXPERT_WIDTH * 2:
            split.setSizes([total - self.EXPERT_WIDTH, self.EXPERT_WIDTH])


class ForgeApp:
    """Builds the window and runs the steps. ``runner`` runs CLI jobs (a stub in tests)."""

    def __init__(
        self,
        *,
        runner: object | None = None,
        home: Path | None = None,
        commands: frozenset[str] | None = None,
        open_path: Callable[[Path], None] | None = None,
        launch: Callable[[list[str]], None] | None = None,
    ) -> None:
        from voxprint_forge.app.runner import ProcessRunner

        register_catalogs()
        self.surface = surface_for(APP_ID)
        self.settings = lora_settings()
        self.home = home if home is not None else state_dir() / APP_ID
        self.runner = runner if runner is not None else ProcessRunner()
        self.commands = engine.commands() if commands is None else commands
        self._open_path = open_path or _open_path
        self._launch = launch or (lambda argv: subprocess.Popen(argv))  # noqa: S603 - fixed argv
        self.expert = ExpertMode()
        self.draft = Draft(expert=self.expert.enabled())
        self.simple: dict[str, str] = {"preset": "balanced", "likeness": "balanced", "vram": "auto"}
        self.simple["base"] = self._bases()[0][0] if self._bases() else ""
        self.previews = True
        self.project = self.home / "lora" / "untitled"
        self.status: tuple[str, dict[str, object], bool] | None = None
        self.progress_text = ""
        self.progress_percent: float | None = None
        self.estimate_minutes: int | None = None
        self.paused = False
        self.help_open: dict[str, bool] = {}
        settings_name, history_name = state_names(self.surface)
        self.layout = SimpleLayout(
            presets=preset_file(self.settings),
            settings=self.settings,
            state_dir=self.home,
            state_name=settings_name,
            history_name=history_name,
            section_set=section_set(self.settings),
            variant=self.surface.variant,
        )
        self.widget = ForgeWidget(
            self.layout, self.settings, on_command=self.command, surface=self.surface
        )
        self.widget.after_refresh = self._after_refresh
        self.window = QtWidgets.QMainWindow()
        self.window.setObjectName("shell")
        self.window.setWindowTitle(tr(f"splash.app.{APP_ID}"))
        self.window.setStyleSheet(build_shell_style(False))
        self.window.setCentralWidget(self.widget.root)
        self.window.resize(1200, 900)
        self._painted: object = None
        self._check_built = False
        self._extend_menu()
        self._unsubscribe = self.expert.subscribe(self._on_expert)
        self.layout.sections.open_wizard(WIZARD)
        self.apply_simple()
        self.widget.set_expert(self.expert.enabled())
        self.widget.refresh()

    # ----- shell glue -------------------------------------------------
    def _after_refresh(self) -> None:
        parts = self.widget._parts()
        deck = self.layout.sections
        if deck.shows_wizard() and deck.step is not None:
            signature = (parts["signature"], deck.step.step_id, id(parts["body"]))
            missing = parts["body"].findChild(QtWidgets.QWidget, "forgeStep") is None  # type: ignore[attr-defined]
            if signature != self._painted or self._stale or missing:
                if signature != self._painted and not self.runner.busy:  # type: ignore[attr-defined]
                    self.progress_text, self.progress_percent, self.status = "", None, None
                self._fill_step(deck.step.step_id)
                self._painted = signature
            self._gate()
        elif deck.current.section_id == "check" and not self._check_built:
            self._build_check()
        self._tips()

    _stale = False

    def repaint(self) -> None:
        """Rebuild the open step from the draft."""
        self._stale = True
        self.widget.refresh()
        self._stale = False

    def _gate(self) -> None:
        parts = self.widget._parts()
        step = self.layout.sections.step
        if step is None:
            return
        result = self.draft.check(step.step_id)
        busy = self.runner.busy  # type: ignore[attr-defined]
        last = step.step_id == STEPS[-1]
        parts["next"].setEnabled(result.ok and not busy and not last)  # type: ignore[attr-defined]
        parts["back"].setEnabled(self.layout.sections.index > 0 and not busy)  # type: ignore[attr-defined]
        label = getattr(self, "_status_label", None)
        if label is None:
            return
        if self.status is not None:
            key, params, warning = self.status
        elif not result.ok or result.warning:
            key, params, warning = result.key, result.params, result.warning
        else:
            key, params, warning = "forge.app.ok", {}, False
        label.setText(tr(key, **params))
        label.setObjectName("warn" if (warning or not result.ok) else "ok")
        label.style().unpolish(label)
        label.style().polish(label)

    def _tips(self) -> None:
        parts = self.widget._parts()
        tips = {
            "next": "forge.app.next.tip",
            "back": "forge.app.back.tip",
            "overflow": "forge.app.overflow.tip",
            "expert": "forge.app.expert.toggle.tip",
        }
        for name, key in tips.items():
            parts[name].setToolTip(tr(key))  # type: ignore[attr-defined]
        for button in parts["tabs"].findChildren(QtWidgets.QAbstractButton):  # type: ignore[attr-defined]
            if not button.toolTip():
                button.setToolTip(button.text().replace("&", ""))
        save = self.widget.root.findChild(QtWidgets.QPushButton, "savePicture")
        if save is not None:
            save.setVisible(False)  # Save lives on the Result step and in the overflow menu.
        folder = self.widget.root.findChild(QtWidgets.QPushButton, "openFolder")
        if folder is not None:
            folder.setToolTip(tr("forge.app.folder.tip"))
        self._fit_expert_fields()

    def _fit_expert_fields(self) -> None:
        """Give Expert combos and spins room for their longest value (bf16, Постоянное)."""
        panel = self.widget.root.findChild(QtWidgets.QWidget, "expertPanel")
        if panel is None:
            return
        for combo in panel.findChildren(QtWidgets.QComboBox):
            metrics = combo.fontMetrics()
            widest = max((metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
            combo.setMinimumWidth(widest + 48)
        for spin in panel.findChildren(QtWidgets.QAbstractSpinBox):
            spin.setMinimumWidth(max(88, spin.fontMetrics().horizontalAdvance(spin.text()) + 44))

    def _extend_menu(self) -> None:
        menu = self.widget._parts()["overflow"].menu()  # type: ignore[attr-defined]
        first = menu.actions()[0] if menu.actions() else None
        items = (
            ("forge.app.menu.new", "lora.menu.new.desc", self.new_model),
            ("forge.app.menu.extend", "lora.menu.extend.desc", self.choose_extend),
            ("forge.app.menu.open", "lora.menu.open.desc", self.choose_project),
            ("forge.app.menu.export", "lora.menu.export.desc", self.save),
            ("forge.app.menu.guide", "lora.menu.guide.desc", self.open_guide),
        )
        for label, desc, slot in items:
            action = QtGui.QAction(tr(label), menu)
            action.setStatusTip(tr(desc))
            action.setToolTip(tr(desc))
            action.triggered.connect(slot)
            menu.insertAction(first, action)
        menu.setToolTipsVisible(True)
        if first is not None:
            menu.insertSeparator(first)

    def command(self, command_id: str) -> None:
        """Shell commands: Save, Open folder, Manage models, Expert mode."""
        if command_id == "view.toggle_expert_mode":
            self.expert.toggle()
        elif command_id == "models.manage":
            self.manage_models()
        elif command_id == "file.save":
            self.save()
        elif command_id == "file.open_folder":
            self.project.mkdir(parents=True, exist_ok=True)
            self._open_path(self.project)

    def _on_expert(self, enabled: bool) -> None:
        self.draft.expert = enabled
        self.widget.set_expert(enabled)
        self.repaint()

    def manage_models(self) -> None:
        """Start Manage models (``voxprint-mdl --gui --app model-forge``)."""
        from voxprint_common.mdl import CLI_NAME

        self._launch([CLI_NAME, "--gui", "--app", APP_ID])

    # ----- steps --------------------------------------------------------
    def _fill_step(self, step: str) -> None:
        body = self.widget._parts()["body"]
        layout = body.layout()  # type: ignore[attr-defined]
        old = body.findChild(QtWidgets.QWidget, "forgeStep")  # type: ignore[attr-defined]
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self._intro(column, step)
        builder = {
            "photos": self._photos,
            "subject": self._subject,
            "prepare": self._prepare,
            "train": self._train,
            "result": self._result,
        }[step]
        builder(column)
        column.addStretch(1)
        status = QtWidgets.QLabel()
        status.setWordWrap(True)
        status.setObjectName("ok")
        self._status_label = status
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setToolTip(tr(f"lora.step.{step}.desc"))
        page.setObjectName("forgeStepPage")
        scroll.setWidget(page)
        outer = QtWidgets.QWidget()
        outer.setObjectName("forgeStep")
        stack = QtWidgets.QVBoxLayout(outer)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(12)
        status.setContentsMargins(0, 4, 0, 4)
        status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        stack.addWidget(scroll, 1)
        stack.addWidget(status)
        for index in reversed(range(layout.count())):  # type: ignore[attr-defined]
            item = layout.itemAt(index)  # type: ignore[attr-defined]
            if item is not None and item.spacerItem() is not None:
                layout.removeItem(item)  # type: ignore[attr-defined]
        layout.insertWidget(0, outer, 1)  # type: ignore[attr-defined]

    def _intro(self, column: QtWidgets.QVBoxLayout, step: str) -> None:
        row = QtWidgets.QHBoxLayout()
        desc = QtWidgets.QLabel(tr(f"lora.step.{step}.desc"))
        desc.setObjectName("panel")
        desc.setWordWrap(True)
        row.addWidget(desc, 1)
        toggle = QtWidgets.QPushButton(tr("forge.app.help.button"))
        toggle.setObjectName("disclosure")
        toggle.setCheckable(True)
        toggle.setToolTip(tr("forge.app.help.tip"))
        row.addWidget(toggle)
        column.addLayout(row)
        box = QtWidgets.QFrame()
        box.setObjectName("card")
        inner = QtWidgets.QVBoxLayout(box)
        title = QtWidgets.QLabel(tr("forge.app.help.title"))
        title.setObjectName("settingGroup")
        text = QtWidgets.QLabel(tr(f"lora.step.{step}.help"))
        text.setWordWrap(True)
        text.setObjectName("simpleNote")
        inner.addWidget(title)
        inner.addWidget(text)
        column.addWidget(box)
        opened = self.help_open.get(step, self._first_run(step))
        box.setVisible(opened)
        toggle.setChecked(opened)

        def flip(checked: bool) -> None:
            box.setVisible(checked)
            self.help_open[step] = checked
            self._remember_help(step, checked)

        toggle.toggled.connect(flip)

    def _first_run(self, step: str) -> bool:
        seen = self.home / "help-seen.txt"
        names = set(seen.read_text(encoding="utf-8").split()) if seen.is_file() else set()
        return step not in names

    def _remember_help(self, step: str, opened: bool) -> None:
        seen = self.home / "help-seen.txt"
        names = set(seen.read_text(encoding="utf-8").split()) if seen.is_file() else set()
        if opened:
            names.discard(step)
        else:
            names.add(step)
        seen.parent.mkdir(parents=True, exist_ok=True)
        seen.write_text("\n".join(sorted(names)), encoding="utf-8")

    def _photos(self, column: QtWidgets.QVBoxLayout) -> None:
        drop = _DropList(self._dropped)
        drop.setObjectName("sourceDrop")
        drop.setToolTip(tr("forge.app.photos.list.tip"))
        drop.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        drop.setIconSize(QtCore.QSize(72, 72))
        drop.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        drop.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        drop.setMinimumHeight(220)
        for path in self.draft.photos:
            item = QtWidgets.QListWidgetItem(_thumb(path), path.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(tr("lora.photos.remove.tip"))
            drop.addItem(item)
        if not self.draft.photos:
            hint = QtWidgets.QListWidgetItem(tr("forge.app.photos.drop"))
            hint.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            drop.addItem(hint)
        column.addWidget(drop, 1)
        row = QtWidgets.QHBoxLayout()
        add = _button("forge.app.photos.add", "lora.photos.add.tip", self.choose_photos, primary=True)
        folder = _button("forge.app.photos.add_folder", "forge.app.photos.add_folder.tip", self.choose_folder)
        remove = _button("forge.app.photos.remove", "lora.photos.remove.tip", lambda: self._remove(drop))
        count_key = "forge.app.photos.count_expert" if self.draft.expert else "forge.app.photos.count"
        count = QtWidgets.QLabel(tr(count_key, count=len(self.draft.photos)))
        count.setObjectName("presetLabel")
        count.setToolTip(tr("lora.photos.count_expert.tip" if self.draft.expert else "lora.photos.count.tip"))
        for widget in (add, folder, remove):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(count)
        column.addLayout(row)
        kind_label = QtWidgets.QLabel(tr("forge.app.kind.label"))
        kind_label.setObjectName("settingGroup")
        column.addWidget(kind_label)
        column.addLayout(self._chips(KINDS, "lora.kind.{}", self.draft.kind, self._set_kind, tip="lora.kind.tip"))

    def _chips(
        self,
        values: tuple[str, ...],
        label: str,
        current: str,
        on_pick: Callable[[str], None],
        *,
        tip: str = "",
        tips: str = "",
    ) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        group = QtWidgets.QButtonGroup(row)
        group.setExclusive(True)
        for value in values:
            chip = QtWidgets.QPushButton(tr(label.format(value)))
            chip.setObjectName(f"forge.chip.{value}")
            chip.setProperty("chip", "true")
            chip.setCheckable(True)
            chip.setChecked(value == current)
            bold = QtGui.QFont(chip.font())
            bold.setBold(True)
            chip.setMinimumWidth(QtGui.QFontMetrics(bold).horizontalAdvance(chip.text()) + 36)
            chip.setProperty("value", value)
            chip.setToolTip(tr(tips.format(value)) if tips else tr(tip))
            chip.clicked.connect(lambda _checked=False, v=value: on_pick(v))
            group.addButton(chip)
            row.addWidget(chip)
        row.addStretch(1)
        row._group = group  # type: ignore[attr-defined]
        return row

    def _subject(self, column: QtWidgets.QVBoxLayout) -> None:
        form = QtWidgets.QFormLayout()
        name = _edit(self.draft.name, "lora.subject.name.tip", "forge.app.subject.name.field")
        stage = _edit(self.draft.stage_name, "lora.subject.stage_name.tip", "forge.app.subject.stage_name.field")
        trigger = _edit(self.draft.trigger, "lora.subject.trigger.tip", "forge.app.subject.trigger.field")

        def names(_text: str = "") -> None:
            self.draft.set_names(name.text(), stage.text())
            if not self.draft.trigger_edited:
                trigger.blockSignals(True)
                trigger.setText(self.draft.trigger)
                trigger.blockSignals(False)
            self._gate()

        def edited(text: str) -> None:
            self.draft.trigger = text.strip()
            self.draft.trigger_edited = True
            self._gate()

        name.textChanged.connect(names)
        stage.textChanged.connect(names)
        trigger.textChanged.connect(edited)
        form.addRow(tr("forge.app.subject.name"), name)
        form.addRow(tr("forge.app.subject.stage_name"), stage)
        form.addRow(tr("forge.app.subject.trigger"), trigger)
        column.addLayout(form)
        kind = self.draft.kind
        if kind == "person":
            boxes = (
                ("forge.app.subject.consent", "lora.subject.consent.tip", "consent"),
                ("forge.app.subject.adult", "lora.subject.adult.tip", "adult"),
            )
        elif kind == "object":
            boxes = (("lora.subject.rights_object", "lora.subject.rights_object.tip", "rights"),)
        elif kind == "style":
            boxes = (("lora.subject.rights_style", "lora.subject.rights_style.tip", "rights"),)
        else:
            boxes = ()
            hint = QtWidgets.QLabel(tr("forge.app.subject.choose_kind"))
            hint.setObjectName("warn")
            column.addWidget(hint)
        for label, tip, field in boxes:
            box = QtWidgets.QCheckBox(tr(label))
            box.setObjectName(f"forge.subject.{field}")
            box.setToolTip(tr(tip))
            box.setChecked(bool(getattr(self.draft, field)))
            box.toggled.connect(lambda checked, f=field: (setattr(self.draft, f, checked), self._gate()))
            column.addWidget(box)
        release_row = QtWidgets.QHBoxLayout()
        attach = _button("forge.app.subject.release", "lora.subject.release.tip", self.choose_release)
        release_row.addWidget(attach)
        shown = self.draft.release
        state = QtWidgets.QLabel(
            tr("forge.app.subject.release_set", name=shown.name) if shown else tr("forge.app.subject.release_none")
        )
        state.setObjectName("panel")
        release_row.addWidget(state)
        if shown is not None:
            release_row.addWidget(
                _button("forge.app.subject.release_clear", "forge.app.subject.release_clear.tip", self.clear_release)
            )
        release_row.addStretch(1)
        column.addLayout(release_row)
        note = _edit(self.draft.note, "lora.subject.note.tip", "forge.app.subject.note.field")
        note.textChanged.connect(lambda text: setattr(self.draft, "note", text))
        notes = QtWidgets.QFormLayout()
        notes.addRow(tr("forge.app.subject.note"), note)
        column.addLayout(notes)

    def _prepare(self, column: QtWidgets.QVBoxLayout) -> None:
        crop_label = QtWidgets.QLabel(tr("forge.app.prepare.crop"))
        crop_label.setObjectName("settingGroup")
        column.addWidget(crop_label)
        column.addLayout(
            self._chips(("subject", "whole"), "forge.app.prepare.crop.{}", self.draft.crop, self._set_crop,
                        tip="lora.prepare.crop.tip")
        )
        run = _button("forge.app.prepare.run", "forge.app.prepare.run.tip", self.prepare, primary=True)
        run.setEnabled(not self.runner.busy)  # type: ignore[attr-defined]
        column.addWidget(run, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self._progress_widgets(column)
        prepared = self.draft.prepared
        if prepared is None:
            return
        kept = QtWidgets.QLabel(tr("forge.app.prepare.kept", count=len(prepared.kept)))
        kept.setObjectName("settingGroup")
        column.addWidget(kept)
        review = QtWidgets.QTableWidget(len(prepared.kept), 2)
        review.setObjectName("forge.prepare.review")
        review.setToolTip(tr("lora.prepare.review.tip"))
        review.horizontalHeader().setVisible(False)
        review.verticalHeader().setVisible(False)
        review.horizontalHeader().setStretchLastSection(True)
        review.setIconSize(QtCore.QSize(56, 56))
        for row, item in enumerate(prepared.kept):
            picture = QtWidgets.QTableWidgetItem(_thumb(Path(str(item.get("file", "")))), "")
            picture.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            caption = QtWidgets.QTableWidgetItem(str(item.get("short") or item.get("long") or ""))
            caption.setData(QtCore.Qt.ItemDataRole.UserRole, str(item.get("id", "")))
            review.setItem(row, 0, picture)
            review.setItem(row, 1, caption)
            review.setRowHeight(row, 60)
        review.itemChanged.connect(self._caption_edited)
        column.addWidget(review, 1)
        dropped = QtWidgets.QLabel(tr("forge.app.prepare.dropped", count=len(prepared.drops)))
        dropped.setObjectName("settingGroup")
        column.addWidget(dropped)
        drops = QtWidgets.QListWidget()
        drops.setObjectName("forge.prepare.dropped")
        drops.setToolTip(tr("lora.prepare.dropped.tip"))
        drops.setMaximumHeight(120)
        for item in prepared.drops:
            name = str(item.get("name") or Path(str(item.get("file", ""))).name)
            reason = str(item.get("reason") or "")
            drops.addItem(f"{name}: {tr(reason)}" if reason else name)
        column.addWidget(drops)
        final = QtWidgets.QLabel(tr("forge.app.prepare.safety_final"))
        final.setObjectName("simpleNote")
        final.setWordWrap(True)
        column.addWidget(final)

    def _train(self, column: QtWidgets.QVBoxLayout) -> None:
        busy = self.runner.busy  # type: ignore[attr-defined]
        for title, values, label, current, pick, tips in (
            ("forge.app.preset.label", ("quick", "balanced", "best"), "lora.preset.{}", self.simple["preset"],
             lambda v: self._set_simple("preset", v), "lora.preset.{}.tip"),
            ("forge.app.likeness.label", ("flexible", "balanced", "close"), "forge.app.likeness.{}",
             self.simple["likeness"], lambda v: self._set_simple("likeness", v), "forge.app.likeness.{}.tip"),
            ("forge.app.memory.label", ("auto", "16", "24"), "forge.app.memory.{}", self.simple["vram"],
             lambda v: self._set_simple("vram", v), ""),
        ):
            heading = QtWidgets.QLabel(tr(title))
            heading.setObjectName("settingGroup")
            column.addWidget(heading)
            tip = "lora.memory.tip" if title == "forge.app.memory.label" else "lora.preset.tip"
            row = self._chips(values, label, current, pick, tip=tip, tips=tips)
            if title == "forge.app.preset.label" and self.custom():
                custom = QtWidgets.QLabel(tr("forge.app.preset.custom"))
                custom.setObjectName("warn")
                row.insertWidget(row.count() - 1, custom)
                row.insertWidget(
                    row.count() - 1, _button("forge.app.preset.reset", "forge.app.preset.reset.tip", self.apply_simple)
                )
            column.addLayout(row)
        form = QtWidgets.QFormLayout()
        base = QtWidgets.QComboBox()
        base.setObjectName("forge.train.base")
        base.setToolTip(tr("lora.base.tip"))
        for mdl_id, label, permissive in self._bases():
            base.addItem(label, mdl_id)
            if not permissive:
                base.setItemData(base.count() - 1, tr("lora.base.external.tip"), QtCore.Qt.ItemDataRole.ToolTipRole)
        index = base.findData(self.simple["base"])
        base.setCurrentIndex(max(0, index))
        base.currentIndexChanged.connect(lambda _i: self.simple.__setitem__("base", str(base.currentData() or "")))
        form.addRow(tr("forge.app.base.label"), base)
        column.addLayout(form)
        previews = QtWidgets.QCheckBox(tr("forge.app.previews.label"))
        previews.setToolTip(tr("lora.previews.tip"))
        previews.setChecked(self.previews)
        previews.toggled.connect(lambda checked: setattr(self, "previews", checked))
        column.addWidget(previews)
        row = QtWidgets.QHBoxLayout()
        estimate = _button("forge.app.train.estimate", "forge.app.train.estimate.tip", self.estimate)
        start_key, start_tip = (
            ("forge.app.train.continue", "forge.app.train.continue.tip")
            if self.paused
            else ("forge.app.train.start", "lora.train.start.tip")
        )
        start = _button(start_key, start_tip, self.train, primary=True)
        pause = _button("forge.app.train.pause", "lora.train.pause.tip", self.pause)
        stop = _button("forge.app.train.stop", "lora.train.stop.tip", self.stop)
        estimate.setEnabled(not busy)
        start.setEnabled(not busy)
        pause.setEnabled(busy)
        stop.setEnabled(busy or self.paused)
        for widget in (estimate, start, pause, stop):
            row.addWidget(widget)
        row.addStretch(1)
        column.addLayout(row)
        if self.estimate_minutes is not None:
            note = QtWidgets.QLabel(tr("forge.app.train.estimate_result", minutes=self.estimate_minutes))
            note.setObjectName("panel")
            column.addWidget(note)
        self._progress_widgets(column, idle="forge.app.train.idle")

    def _result(self, column: QtWidgets.QVBoxLayout) -> None:
        form = QtWidgets.QFormLayout()
        pick = QtWidgets.QComboBox()
        pick.setObjectName("forge.result.checkpoint")
        pick.setToolTip(tr("lora.result.checkpoint.tip"))
        pick.addItem(tr("forge.app.result.final"), None)
        for step in self._checkpoints():
            pick.addItem(tr("forge.app.result.step", step=step), step)
        index = pick.findData(self.draft.checkpoint)
        pick.setCurrentIndex(max(0, index))
        pick.currentIndexChanged.connect(lambda _i: setattr(self.draft, "checkpoint", pick.currentData()))
        form.addRow(tr("forge.app.result.checkpoint"), pick)
        column.addLayout(form)
        heading = QtWidgets.QLabel(tr("forge.app.result.previews"))
        heading.setObjectName("settingGroup")
        column.addWidget(heading)
        grid = QtWidgets.QListWidget()
        grid.setObjectName("forge.result.previews")
        grid.setToolTip(tr("forge.app.result.previews.tip"))
        grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        grid.setIconSize(QtCore.QSize(160, 160))
        grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        found = sorted((self.project / "previews").glob("*.png")) + sorted((self.project / "previews").glob("*.webp"))
        for path in found:
            grid.addItem(QtWidgets.QListWidgetItem(_thumb(path, 160), path.stem))
        if not found:
            grid.addItem(tr("forge.app.result.no_previews"))
        column.addWidget(grid, 1)
        row = QtWidgets.QHBoxLayout()
        busy = self.runner.busy  # type: ignore[attr-defined]
        for label, tip, slot, primary in (
            ("forge.app.result.save", "lora.result.save.tip", self.save, True),
            ("forge.app.result.canvas", "lora.result.canvas.tip", self.install, False),
            ("forge.app.result.extend", "lora.result.extend.tip", self.extend_this, False),
        ):
            button = _button(label, tip, slot, primary=primary)
            button.setEnabled(not busy)
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)
        self._progress_widgets(column)

    def _progress_widgets(self, column: QtWidgets.QVBoxLayout, idle: str = "") -> None:
        text = QtWidgets.QLabel(self.progress_text or (tr(idle) if idle else ""))
        text.setObjectName("progressText")
        text.setWordWrap(True)
        bar = QtWidgets.QProgressBar()
        bar.setObjectName("progress")
        bar.setTextVisible(False)
        bar.setRange(0, 1000)
        bar.setValue(int((self.progress_percent or 0) * 10))
        bar.setVisible(self.progress_percent is not None)
        column.addWidget(text)
        column.addWidget(bar)
        self._progress_text, self._progress_bar = text, bar

    def _build_check(self) -> None:
        note = self.widget._parts()["note"]
        layout = note.layout()  # type: ignore[attr-defined]
        box = QtWidgets.QWidget()
        box.setObjectName("forgeCheck")
        column = QtWidgets.QVBoxLayout(box)
        column.setContentsMargins(0, 8, 0, 0)
        self._check_lines = QtWidgets.QLabel()
        self._check_lines.setObjectName("panel")
        self._check_lines.setWordWrap(True)
        column.addWidget(self._check_lines)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("forge.app.check.run", "forge.app.check.run.tip", self.run_check, primary=True))
        row.addWidget(_button("forge.app.check.models", "forge.app.check.models.tip", self.manage_models))
        row.addStretch(1)
        column.addLayout(row)
        layout.insertWidget(1, box)  # type: ignore[attr-defined]
        self._check_built = True
        self.run_check()

    def run_check(self) -> None:
        """Graphics card tier, memory, disk and model readiness (library calls, no GPU work)."""
        from voxprint_common.mdl import ready

        from voxprint_forge import disk
        from voxprint_forge.lora.presets import detect_vram_gb
        from voxprint_forge.preflight import memory_status

        lines = []
        gib = 2**30
        vram = _vram_total()
        if vram is None:
            lines.append(tr("lora.err.no_gpu"))
        else:
            lines.append(tr("forge.app.check.gpu", gb=round(vram / gib), tier=detect_vram_gb()))
        total, free = memory_status()
        lines.append(tr("forge.app.check.ram", free=round(free / gib), total=round(total / gib)))
        self.home.mkdir(parents=True, exist_ok=True)
        lines.append(tr("forge.app.check.disk", free=round(disk.free_bytes(self.home) / gib)))
        lines.append(tr("forge.app.check.models_ready" if ready(APP_ID) else "forge.app.check.models_missing"))
        for mdl_id, label, _permissive in self._bases():
            key = "forge.app.check.base_ready" if _base_installed(mdl_id) else "forge.app.check.base_missing"
            lines.append(tr(key, name=label))
        self._check_lines.setText("\n".join(lines))

    # ----- actions ------------------------------------------------------
    def _set_kind(self, kind: str) -> None:
        if kind != self.draft.kind:
            self.draft.kind = kind
            self.draft.consent = self.draft.adult = self.draft.rights = False
            self.draft.prepared = None
        self._gate()

    def _set_crop(self, crop: str) -> None:
        if crop != self.draft.crop:
            self.draft.crop = crop
            self.draft.prepared = None
        self.repaint()

    def _set_simple(self, key: str, value: str) -> None:
        self.simple[key] = value
        self.apply_simple()
        self.repaint()

    def apply_simple(self) -> None:
        """Write the live Expert values for the Simple choices (Reset uses this too)."""
        state = self.layout.state
        if state is None:
            return
        state.choose(PRESET_GROUP, self.simple["preset"])
        for key, value in config_values(self.simple["preset"], self.simple["likeness"], self.simple["vram"]).items():
            if state.has(key) and state.get(key) != value:
                state.set_value(key, value)
        self._applied = self.expert_values()
        if self.layout.sections.step is not None and self.layout.sections.step.step_id == "train":
            self.repaint()

    _applied: dict[str, object] = {}

    def expert_values(self) -> dict[str, object]:
        """The live values the Expert panel shows."""
        state = self.layout.state
        return {spec.setting_id: state.get(spec.setting_id) for spec in self.settings.visible(expert=True)
                if state is not None and state.has(spec.setting_id)}

    def custom(self) -> bool:
        """True when an Expert edit moved a value away from the chosen preset."""
        return bool(self._applied) and self.expert_values() != self._applied

    def _dropped(self, paths: list[Path]) -> None:
        self.draft.add_photos(paths)
        self.repaint()

    def _remove(self, drop: QtWidgets.QListWidget) -> None:
        for item in drop.selectedItems():
            value = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if value:
                self.draft.remove_photo(Path(value))
        self.repaint()

    def choose_photos(self) -> None:
        """Pick photo files."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.window, tr("forge.app.photos.add"), "", tr("forge.app.photos.filter")
        )
        self._dropped([Path(item) for item in files])

    def choose_folder(self) -> None:
        """Pick a folder of photos."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self.window, tr("forge.app.photos.add_folder"))
        if folder:
            self._dropped([Path(folder)])

    def choose_release(self) -> None:
        """Attach a model release; only its existence and hash go into the model."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, tr("forge.app.subject.release"), "", tr("forge.app.subject.release_filter")
        )
        if path:
            self.draft.release = Path(path)
            self.repaint()

    def clear_release(self) -> None:
        """Detach the model release."""
        self.draft.release = None
        self.repaint()

    def _caption_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        item_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not item_id or self.draft.prepared is None:
            return
        for row in self.draft.prepared.kept:
            if str(row.get("id")) == item_id:
                row["short"] = item.text()
        edits = self.project / "captions-edit.jsonl"
        lines = [json.dumps({"id": str(row.get("id")), "short": str(row.get("short", "")),
                             "long": str(row.get("long", ""))}, ensure_ascii=False)
                 for row in self.draft.prepared.kept]
        edits.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _needs(self, command: str) -> bool:
        if command in self.commands:
            return True
        self.status = ("forge.app.err.missing_command", {}, True)
        self._gate()
        return False

    def _run(self, args: list[str], done: Callable[[engine.Outcome], None]) -> None:
        if self.runner.busy:  # type: ignore[attr-defined]
            self.status = ("forge.app.err.busy", {}, True)
            self._gate()
            return
        self.status = None
        self.progress_text = ""
        self.progress_percent = None
        self.project.mkdir(parents=True, exist_ok=True)

        def finished(outcome: engine.Outcome) -> None:
            done(outcome)
            if not outcome.ok and self.status is None:
                self.status = _failure(outcome)
            self.repaint()

        self.runner.start(args, self.on_event, finished)  # type: ignore[attr-defined]
        self.repaint()

    def on_event(self, event: engine.EngineEvent) -> None:
        """Show the engine's sentence: what it does, why, and the time left."""
        parts = [tr(event.message_key, **event.params)]
        if event.reason_key:
            parts.append(tr(event.reason_key))
        if event.state == "running" and event.percent is not None:
            if event.eta_seconds is None:
                parts.append(tr("lora.progress.eta_unknown"))
            else:
                parts.append(tr("lora.progress.eta", minutes=max(1, round(event.eta_seconds / 60))))
        self.progress_text = " — ".join(parts)
        self.progress_percent = event.percent
        if event.state == "paused":
            self.paused = True
        text = getattr(self, "_progress_text", None)
        bar = getattr(self, "_progress_bar", None)
        try:
            if text is not None:
                text.setText(self.progress_text)
            if bar is not None:
                bar.setVisible(event.percent is not None)
                bar.setValue(int((event.percent or 0) * 10))
        except RuntimeError:
            pass
        window = self.widget.root.window()
        if window is not None:
            window.setWindowTitle(f"{tr('splash.app.' + APP_ID)} — {self.progress_text}")

    def prepare(self) -> None:
        """``lora prepare``: filter, safety gate, crop, captions. The gate result is final."""
        if not self._needs("prepare"):
            return
        args = engine.prepare_args(self.draft, self.project)
        if self.draft.expert:
            args += ["--caption-model", str(self.expert_values().get("lora.caption_model", "qwen3-vl-4b-instruct"))]

        def done(outcome: engine.Outcome) -> None:
            if outcome.ok:
                kept, drops = engine.prepared_from(self.project)
                self.draft.prepared = Prepared(kept, drops, False)
                self.progress_text = tr("forge.app.prepare.done", kept=len(kept), drops=len(drops))
                self.progress_percent = None
            elif outcome.error_code == "lora.err.blocked":
                self.draft.prepared = Prepared([], engine.prepared_from(self.project)[1], True)
            else:
                self.draft.prepared = None

        self._run(args, done)

    def _train_args(self, *, resume: bool = False, dry_run: bool = False) -> list[str]:
        values: dict[str, object] = dict(self.expert_values())
        values.update({"lora.preset": self.simple["preset"], "lora.likeness": self.simple["likeness"],
                       "lora.vram": self.simple["vram"], "lora.base": self.simple["base"],
                       "lora.previews": self.previews})
        args = engine.train_args(self.project, values, extend_from=self.draft.extend_from, resume=resume,
                                 dry_run=dry_run, expert=self.draft.expert)
        edits = self.project / "captions-edit.jsonl"
        if edits.is_file():
            args += ["--captions", str(edits)]
        return args

    def estimate(self) -> None:
        """``--dry-run``: the plan and the time estimate, nothing trained."""
        command = "extend" if self.draft.extend_from else "train"
        if not self._needs(command):
            return

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            seconds = result.get("eta_seconds") or result.get("seconds")
            if outcome.ok and isinstance(seconds, int | float):
                self.estimate_minutes = max(1, round(seconds / 60))

        self._run(self._train_args(dry_run=True), done)

    def train(self) -> None:
        """Start (or continue after a pause) training on this computer's graphics card."""
        command = "extend" if self.draft.extend_from else "train"
        if not self._needs(command):
            return
        resume = self.paused

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            if outcome.ok and result.get("state", "done") == "done":
                self.draft.trained = True
                self.paused = False
            elif outcome.ok and result.get("state") == "paused":
                self.paused = True
            engine.clear_pause(self.project)

        engine.clear_pause(self.project)
        self._run(self._train_args(resume=resume), done)

    def pause(self) -> None:
        """Pause: create ``PAUSE``; the engine saves a version and exits. Continue resumes from it."""
        if not self.runner.busy:  # type: ignore[attr-defined]
            return
        self.paused = True
        engine.request_pause(self.project)

    def stop(self) -> None:
        """Stop now and keep the versions saved so far."""
        if self.runner.busy:  # type: ignore[attr-defined]
            self.runner.stop()  # type: ignore[attr-defined]
        engine.clear_pause(self.project)
        self.paused = False
        if self._checkpoints():
            self.draft.trained = True
        self.repaint()

    def _export(self, out: Path | None, install: bool) -> None:
        if not self._needs("export"):
            return

        def done(outcome: engine.Outcome) -> None:
            if outcome.ok:
                key = "forge.app.result.installed" if install else "forge.app.result.saved"
                self.status = (key, {"name": out.name if out else ""}, False)

        self._run(engine.export_args(self.draft, self.project, out, checkpoint=self.draft.checkpoint,
                                     install=install), done)

    def save(self) -> None:
        """Save one .safetensors file with its ``voxprint-model/1`` description."""
        name = (self.draft.trigger or "model") + ".safetensors"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, tr("forge.app.result.save"), str(Path.home() / name), tr("forge.app.result.filter")
        )
        if path:
            self._export(Path(path), False)

    def install(self) -> None:
        """Open in Canvas: install into the MDL user-models folder."""
        self._export(None, True)

    def extend_this(self) -> None:
        """Add more photos: back to step 1 with this model as the starting point."""
        final = self.project / "lora.safetensors"
        self.draft.extend_from = final if final.is_file() else None
        self.draft.trained = False
        self.draft.prepared = None
        self.layout.sections.open_wizard(WIZARD)
        self.repaint()

    def choose_extend(self) -> None:
        """Pick an existing LoRA to teach more."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, tr("forge.app.menu.extend"), "", tr("forge.app.result.filter")
        )
        if path:
            self.new_model()
            self.draft.extend_from = Path(path)
            self.repaint()

    def choose_project(self) -> None:
        """Open a saved training project folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self.window, tr("forge.app.menu.open"),
                                                            str(self.home / "lora"))
        if folder:
            self.open_project(Path(folder))

    def open_project(self, folder: Path) -> None:
        """Load the subject fields from ``project.json`` and continue on step 1."""
        self.new_model()
        self.project = folder
        data_file = folder / "project.json"
        try:
            data = json.loads(data_file.read_text(encoding="utf-8")) if data_file.is_file() else {}
        except (OSError, ValueError):
            data = {}
        subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
        self.draft.kind = str(data.get("kind") or subject.get("kind") or "")
        self.draft.name = str(subject.get("name") or "")
        self.draft.stage_name = str(subject.get("stage_name") or "")
        self.draft.trigger = str(subject.get("trigger") or "")
        self.draft.trigger_edited = bool(self.draft.trigger)
        photos = folder / "photos"
        if photos.is_dir():
            self.draft.add_photos([photos])
        self.repaint()

    def new_model(self) -> None:
        """Start over with an empty draft in a new project folder."""
        self.draft = Draft(expert=self.expert.enabled())
        self.paused = False
        self.estimate_minutes = None
        self.status = None
        self.progress_text = ""
        self.progress_percent = None
        root = self.home / "lora"
        index = 1
        while (root / f"model-{index}").exists():
            index += 1
        self.project = root / f"model-{index}"
        self.layout.sections.open_wizard(WIZARD)
        self.repaint()

    def open_guide(self) -> None:
        """Open the offline user guide shipped with the app, in the app's language."""
        folder = Path(__file__).resolve().parent / "guide"
        guide = folder / f"{get_language()}.html"
        if not guide.is_file():
            guide = folder / "index.html"
        if guide.is_file():
            self._open_path(guide)
        else:
            self.status = ("forge.app.err.no_guide", {}, True)
            self._gate()

    def _checkpoints(self) -> list[int]:
        steps = []
        for path in (self.project / "checkpoints").glob("*"):
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            if digits:
                steps.append(int(digits))
        return sorted(set(steps))

    def _bases(self) -> list[tuple[str, str, bool]]:
        try:
            from voxprint_forge.lora import bases
        except ImportError:
            return []
        return [(info.mdl_id, info.display_name, info.permissive) for info in bases.available()]


class _DropList(QtWidgets.QListWidget):
    """Photo list that accepts dropped files and folders."""

    def __init__(self, on_drop: Callable[[list[Path]], None]) -> None:
        super().__init__()
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        """Accept file drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # noqa: N802 - Qt name
        """Keep accepting while the files move over the list."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802 - Qt name
        """Add the dropped files and folders."""
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        event.acceptProposedAction()
        self._on_drop(paths)


def _button(label: str, tip: str, slot: Callable[[], None], *, primary: bool = False) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(tr(label))
    button.setObjectName("primary" if primary else label)
    button.setToolTip(tr(tip))
    button.setAutoDefault(False)
    button.clicked.connect(lambda _checked=False: slot())
    return button


def _edit(text: str, tip: str, name: str) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(text)
    edit.setObjectName(name)
    edit.setToolTip(tr(tip))
    return edit


def _thumb(path: Path, size: int = 72) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(str(path))
    if pixmap.isNull():
        return QtGui.QIcon()
    return QtGui.QIcon(pixmap.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                     QtCore.Qt.TransformationMode.SmoothTransformation))


def _failure(outcome: engine.Outcome) -> tuple[str, dict[str, object], bool]:
    code = outcome.error_code
    if code.startswith(("lora.", "forge.app.")):
        return code, {}, True
    return "forge.app.err.unexpected", {"message": outcome.error_message or code}, True


def _vram_total() -> int | None:
    try:
        from voxprint_forge.gui.gauges import vram_status
    except Exception:  # noqa: BLE001 - the probe is optional
        return None
    found = vram_status()
    return found[0] if found else None


def _base_installed(mdl_id: str) -> bool:
    from voxprint_forge.lora.bases import weights_dir

    folder = weights_dir(mdl_id)
    return folder.is_dir() and any(folder.iterdir())


def _open_path(path: Path) -> None:
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
