"""The Model Forge window on the shared Yitro shell (voxprint-common v0.2.6).

Surface ``model-forge`` (tool type). Two sections: "Train a model" with the
five-step wizard (photos, subject and consent, check and prepare, train,
result) and "Check this computer". The shell draws the chrome, the tab row,
the step title and counter, Back and Next, the overflow menu and the Expert
panel. This module fills each step's body, gates Next with a plain reason and
feeds the engine's progress sentences into the step.

v0.2.5 and v0.2.6 have no public hook for a custom wizard body, so :class:`ForgeWidget`
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
from voxprint_common.i18n import add_catalogs, get_language
from voxprint_common.paths import state_dir
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.sections import BodyKind, Section, SectionSet, Wizard, WizardStep
from voxprint_common.ui.simple_image import SimpleImageWidget, SimpleLayout
from voxprint_common.ui.theme import build_shell_style
from voxprint_common.ui.tokens import load as load_tokens
from voxprint_common.ui.variants import state_names, surface_for

from voxprint_forge.app import basepicker, engine
from voxprint_forge.app.details import Details
from voxprint_forge.app.draft import KINDS, STEPS, Draft, Prepared
from voxprint_forge.app.flow import FlowRow
from voxprint_forge.app.settings import PRESET_GROUP, SECTION, WIZARD, config_values, lora_settings, preset_file
from voxprint_forge.app.text import tr
from voxprint_forge.lora import refset
from voxprint_forge.lora import review as rounds
from voxprint_forge.lora.attr import HIGH, LOW, accepted
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.limits import expert_range_differs
from voxprint_forge.lora.progress_log import read_progress
from voxprint_forge.lora.project import Project, package_stem
from voxprint_forge.lora.sample_prompts import read_attire, store_attire
from voxprint_forge.mdl_app import download_argv, manage_argv, missing_id_text, missing_ids

APP_ID = "model-forge"
CARD_PICTURE = 136  # px, a review-grid picture
KEEP_MARK = 30  # px, the checkmark toggle on a picture
REVIEW_MARK = 36  # px, the include / leave-out toggles on a review sample


def button_radius_px() -> int:
    """Square-button radius from the suite token, or 4 px when that token is absent."""
    try:
        value = getattr(load_tokens().density, "button_radius_px", None)
    except Exception:  # noqa: BLE001 - a missing token uses the 4 px fallback
        value = None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 4


_BUTTON_RADIUS = button_radius_px()


def format_span(seconds: float) -> str:
    """A duration in hours and minutes, for the training progress line."""
    whole = max(0, int(seconds))
    if whole == 0:
        return tr("lora.progress.span.minutes", count=0)
    if whole < 30:
        return tr("lora.progress.span.under")
    minutes = (whole + 30) // 60
    if minutes < 60:
        return tr("lora.progress.span.minutes", count=minutes)
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return tr("lora.progress.span.hours", count=hours)
    return tr("lora.progress.span.hours_minutes", hours=hours, minutes=rest)


def _control_width(button: QtWidgets.QAbstractButton) -> int:
    """The label width plus the chip padding, measured in the heavier checked face."""
    density = load_tokens().density
    button.ensurePolished()
    font = QtGui.QFont(button.font())
    font.setPixelSize(density.body_px)
    font.setWeight(QtGui.QFont.Weight.DemiBold)
    advance = QtGui.QFontMetrics(font).horizontalAdvance(button.text())
    return advance + (density.chip_pad_px + density.border_px) * 2 + 4


# The shell's action bar (Open folder) gets the wizard card's 16 px inset, so it lines up with Back.
ACTION_BAR_MARGINS = (16, 0, 16, 16)
FORGE_STYLE = """
QPushButton#wizardNext:disabled, QPushButton#wizardBack:disabled {
    background: #22222b; color: #6e6e7e; border-color: #33333f; }
QPushButton[keepMark="true"] { background: #1c1c24; color: #9a9aa8; border: 2px solid #c4c4d0;
    border-radius: @@RADIUS@@px; padding: 0px; min-height: 0px; font-size: 16px; font-weight: 700; }
QPushButton[keepMark="true"]:checked { background: #f5d76e; color: #1c1604; border-color: #fbe8a6; }
QFrame#card[kept="true"] { border: 2px solid #f5d76e; border-radius: 6px; }
QFrame#card[kept="false"] { border: 2px solid transparent; border-radius: 6px; }
QFrame#card[verdict="like"] { border: 2px solid #f5d76e; border-radius: 6px; }
QFrame#card[verdict="dislike"] { border: 2px solid #c45c5c; border-radius: 6px; }
QFrame#card[verdict="unmarked"] { border: 2px solid transparent; border-radius: 6px; }
QLabel[reviewBadge="like"] {
    background: #f5d76e; color: #1c1604; border-radius: @@RADIUS@@px; font-weight: 700; font-size: 14px; }
QLabel[reviewBadge="dislike"] {
    background: #b55252; color: #ffffff; border-radius: @@RADIUS@@px; font-weight: 700; font-size: 14px; }
QPushButton[reviewMark="true"] {
    background: #1c1c24; color: #e8e6e3; border: 2px solid #4a4a58;
    border-radius: @@RADIUS@@px; padding: 0; margin: 0;
    min-width: @@MARK@@px; max-width: @@MARK@@px; min-height: @@MARK@@px; max-height: @@MARK@@px;
    font-size: 18px; }
QPushButton[reviewMark="true"]:checked { background: #f5d76e; color: #1c1604; border-color: #fbe8a6; }
QPushButton[reviewMark="true"][down="true"]:checked {
    background: #5a3030; color: #f3d0d0; border-color: #c45c5c; }
QLabel[similar="true"] { color: #fbbf24; }
QWidget#sourceDrop { background: #16161c; border: 2px dashed #6e6e80; border-radius: 12px; }
QLabel[emptyTitle="true"] { font-size: 16px; font-weight: 600; padding: 0; min-height: 28px; }
QLabel[muted="true"] { color: #9a9aa8; }
QLabel[kindNote="true"] { color: #9a9aa8; font-size: 12px; }
QFrame#kindSegments { background: #1c1c24; border: 1px solid #4a4a58; border-radius: @@RADIUS@@px; }
QFrame#kindSegments QPushButton[segment="true"] {
    background: transparent; color: #e8e6e3; border: none; border-radius: 0;
    padding: 8px 10px; min-height: 34px; }
QFrame#kindSegments QPushButton[segment="true"][end="left"] {
    border-top-left-radius: @@RADIUS@@px; border-bottom-left-radius: @@RADIUS@@px; }
QFrame#kindSegments QPushButton[segment="true"][end="right"] {
    border-top-right-radius: @@RADIUS@@px; border-bottom-right-radius: @@RADIUS@@px; }
QPushButton[chip="true"], QPushButton#overflow, QPushButton#expertToggle { border-radius: @@RADIUS@@px; }
QFrame#kindSegments QPushButton[segment="true"][end="mid"],
QFrame#kindSegments QPushButton[segment="true"][end="right"] { border-left: 1px solid #4a4a58; }
QFrame#kindSegments QPushButton[segment="true"]:checked {
    background: #f5d76e; color: #1c1604; font-weight: 600; border-left-color: #fbe8a6; }
QPushButton[photosAdd="true"] {
    background: #f5d76e; border: 1px solid #fbe8a6; font-weight: 600; color: #1c1604; }
QPushButton[photosAdd="true"]:hover { background: #f9e291; }
QPushButton[textButton="true"] {
    background: transparent; border: none; color: #c8c8d4; padding: 4px 2px; min-height: 28px; font-weight: 500; }
QPushButton[textButton="true"]:hover { background: transparent; color: #f5d76e; }
QToolButton[photoRemove="true"] {
    background: transparent; border: none; border-radius: @@RADIUS@@px; padding: 0; min-width: 0; min-height: 0; }
QToolButton[photoRemove="true"][shown="true"] {
    background: rgba(12, 12, 16, 210); border: 1px solid #6a6a78; }
QToolButton[selectionRemove="true"] {
    background: transparent; border: 1px solid #4a4a58; border-radius: @@RADIUS@@px;
    padding: 0; min-width: 0; min-height: 0; }
QToolButton[selectionRemove="true"]:hover { background: #2a2a35; border-color: #f5d76e; }
QFrame[photoTile="true"] { background: #1c1c24; border: 2px solid transparent; border-radius: 8px; }
QFrame[photoTile="true"][selected="true"] { border-color: #f5d76e; }
QPushButton[reviewFinish="true"] {
    background: #2a2a33; color: #f4f1ea; border: 1px solid #3a3a46;
    border-radius: 4px; padding: 2px 12px; min-height: 28px; font-weight: 500; }
QPushButton[reviewFinish="true"]:hover { background: #34343f; }
QPushButton[reviewFinish="true"]:disabled {
    background: #22222b; color: #6e6e7e; border-color: #33333f; }
QMenu { background: #2a2a33; color: #f4f1ea; border: 1px solid #3a3a46; padding: 4px; }
QMenu::item { padding: 6px 18px; background: transparent; color: #f4f1ea; }
QMenu::item:selected { background: #3a3a46; color: #f4f1ea; }
QMenu::separator { height: 1px; background: #3a3a46; margin: 4px 8px; }
QDialog { background: #1c1c24; color: #f4f1ea; }
QDialog QLabel { color: #f4f1ea; background: transparent; }
QPushButton:disabled, QPushButton:disabled:hover {
    background: #22222b; color: #6e6e7e; border: 1px solid #33333f; }
QPushButton#primary:disabled, QPushButton#primary:disabled:hover {
    background: #2a2a33; color: #6e6e7e; border: 1px solid #3a3a46; }
QPushButton[keepMark="true"][down="true"]:checked { background: #3a3a46; color: #e8e6e3; border-color: #6e6e80; }
QToolTip {
    background-color: #2b2d36; color: #f4f1ea; border: 1px solid #c8c8d4;
    border-radius: 4px; padding: 6px 8px; margin: 0px; max-width: 420px;
    qproperty-margin: 0;
}
"""
FORGE_STYLE = FORGE_STYLE.replace("@@RADIUS@@", str(_BUTTON_RADIUS)).replace("@@MARK@@", str(REVIEW_MARK))
_DIALOG_STYLE = """
QDialog { background: transparent; }
QFrame#forgeDialogCard { background: #1c1c24; border: 2px solid #c8c8d4; border-radius: 12px; }
QLabel#forgeDialogTitle { color: #f4f1ea; font-size: 16px; font-weight: 600; background: transparent; }
QLabel#qt_msgbox_label { color: #f4f1ea; font-size: 14px; background: transparent; }
QPushButton { background: #2a2a33; color: #f4f1ea; border: 1px solid #5c5c6c; border-radius: 4px;
    padding: 6px 16px; min-height: 32px; min-width: 96px; }
QPushButton:hover { background: #34343f; }
QPushButton[danger="true"] { background: #9a3030; color: #fff8f8; border: 1px solid #f0b0b0; font-weight: 600; }
QPushButton[danger="true"]:hover { background: #b23c3c; }
"""
_MENU_STYLE = """
QMenu { background: #2a2a33; color: #f4f1ea; border: 1px solid #3a3a46; padding: 4px; }
QMenu::item { padding: 6px 18px; background: transparent; color: #f4f1ea; }
QMenu::item:selected { background: #3a3a46; color: #f4f1ea; }
QMenu::separator { height: 1px; background: #3a3a46; margin: 4px 8px; }
"""
# The finish menu matches the discard dialog's card: dark fill, a light border, a short shadow.
# ``_DANGER_TEXT`` is the text colour on the dislike / danger state.
_FINISH_PANEL = "#1c1c24"
_FINISH_BORDER = "#c8c8d4"
_FINISH_INK = "#f4f1ea"
_DANGER_TEXT = "#f3d0d0"
_FINISH_RADIUS = 4
_FINISH_SHADOW = 8
# Tooltip copy wraps inside 420 px. Padding and the 1 px border sit outside that.
_TOOLTIP_TEXT = 420
_TOOLTIP_PAD_X = 8
_TOOLTIP_PAD_Y = 6
_TOOLTIP_BORDER = 1
_TOOLTIP_GAP = 2
_CHEVRON_W = 28
_LABEL_PAD = 16
_LABEL_GAP = 14
_FADE_MS = 150
_ROW_RADIUS = 4
LOCALES = Path(__file__).resolve().parent / "locales"


def register_catalogs() -> None:
    """GUI strings, then the engine's ``lora.*`` strings (plan: add_catalogs)."""
    add_catalogs(LOCALES)
    try:
        from voxprint_forge.lora import i18n as lora_i18n

        add_catalogs(Path(lora_i18n.__file__).resolve().parent)
    except ImportError:
        pass


def _install_tooltip_placement() -> None:
    """Open every tooltip under its control, once per process."""
    app = QtWidgets.QApplication.instance()
    if app is None or app.findChild(QtCore.QObject, "forge.tooltip.placement") is not None:
        return
    placement = _TooltipPlacement(app)
    placement.setObjectName("forge.tooltip.placement")
    app.installEventFilter(placement)


class _TooltipPlacement(QtCore.QObject):
    """Replace the cursor-shaped system tip so it does not cover the control."""

    def __init__(self, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._busy = False

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 - Qt override
        if self._busy or event.type() != QtCore.QEvent.Type.ToolTip:
            return False
        if not isinstance(watched, QtWidgets.QWidget) or not isinstance(event, QtGui.QHelpEvent):
            return False
        text, local = _tooltip_subject(watched, event.pos())
        if not text:
            return False
        self._busy = True
        try:
            anchor = QtCore.QRect(watched.mapToGlobal(local.topLeft()), local.size())
            below = QtCore.QPoint(anchor.x(), anchor.y() + anchor.height() + _TOOLTIP_GAP)
            QtWidgets.QToolTip.showText(below, text, watched, local)
            _settle_tooltip(watched, anchor)
        finally:
            self._busy = False
        return True


def _tooltip_subject(widget: QtWidgets.QWidget, pos: QtCore.QPoint) -> tuple[str, QtCore.QRect]:
    """The tip text and the rectangle it must not cover, in ``widget`` coordinates."""
    if isinstance(widget, QtWidgets.QMenu) and widget.toolTipsVisible():
        action = widget.actionAt(pos)
        if action is not None and action.toolTip().strip():
            return action.toolTip(), widget.actionGeometry(action)
        return "", QtCore.QRect()
    text = widget.toolTip().strip()
    if text:
        return widget.toolTip(), widget.rect()
    return "", QtCore.QRect()


def _visible_tooltip() -> QtWidgets.QWidget | None:
    """The live tip label, which Qt keeps off the window that owns the control."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    for widget in app.topLevelWidgets():
        if not widget.isVisible():
            continue
        name = widget.metaObject().className()
        if widget.windowType() == QtCore.Qt.WindowType.ToolTip or "Tip" in name:
            return widget
    return None


def _settle_tooltip(widget: QtWidgets.QWidget, anchor: QtCore.QRect) -> None:
    """Wrap the tip to 420 px, round its corners, and sit it under the control."""
    tip = _visible_tooltip()
    if tip is None or not isinstance(tip, QtWidgets.QLabel):
        return
    tip.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    tip.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    font = tip.font()
    font.setStyleStrategy(QtGui.QFont.StyleStrategy.NoSubpixelAntialias)
    tip.setFont(font)
    tip.setWordWrap(True)
    text = tip.text()
    flags = int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.TextFlag.TextWordWrap)
    rect = tip.fontMetrics().boundingRect(0, 0, _TOOLTIP_TEXT, 2000, flags, text)
    width = rect.width() + _TOOLTIP_PAD_X * 2 + _TOOLTIP_BORDER * 2
    height = rect.height() + _TOOLTIP_PAD_Y * 2 + _TOOLTIP_BORDER * 2
    tip.setFixedSize(max(width, 1), max(height, 1))
    x = anchor.x()
    y = anchor.y() + anchor.height() + _TOOLTIP_GAP
    window = widget.window()
    if isinstance(window, QtWidgets.QWidget):
        origin = window.mapToGlobal(QtCore.QPoint(0, 0))
        if y + tip.height() > origin.y() + window.height():
            y = anchor.y() - tip.height() - _TOOLTIP_GAP
        limit = origin.x() + window.width() - tip.width()
        x = min(max(x, origin.x()), max(limit, origin.x()))
        y = max(y, origin.y())
    tip.move(x, y)


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
        """The shell widgets this app extends (pinned to v0.2.6; unchanged since v0.2.5)."""
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
            "expert_form": self._expert_form,  # type: ignore[attr-defined]
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
        _install_tooltip_placement()
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
        self.first_round_steps = rounds.DEFAULT_FIRST_ROUND_STEPS
        self.review_rounds = rounds.DEFAULT_ROUNDS
        self.refset_settings: dict[str, float] = {
            "count": refset.DEFAULT_CANDIDATES, "strength": refset.DEFAULT_STRENGTH, "steps": refset.DEFAULT_STEPS,
        }
        self.project = self.home / "lora" / "untitled"
        self.status: tuple[str, dict[str, object], bool] | None = None
        self.progress_text = ""
        self.progress_percent: float | None = None
        self.estimate_minutes: int | None = None
        self.paused = False
        self.help_open: dict[str, bool] = {}
        self.photo_selection: set[str] = set()
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
        self.window.setStyleSheet(build_shell_style(False) + FORGE_STYLE)
        self.window.setCentralWidget(self.widget.root)
        self.window.resize(1200, 900)
        self._painted: object = None
        self._check_built = False
        self._open_required_models()
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
        reviewing = step.step_id == "train" and self._review_waiting() and not busy
        parts["next"].setVisible(not reviewing)  # type: ignore[attr-defined]
        parts["back"].setVisible(not reviewing)  # type: ignore[attr-defined]
        last = step.step_id == STEPS[-1]
        parts["next"].setEnabled(result.ok and not busy and not last and not reviewing)  # type: ignore[attr-defined]
        parts["back"].setEnabled(self.layout.sections.index > 0 and not busy and not reviewing)  # type: ignore[attr-defined]
        label = getattr(self, "_status_label", None)
        if label is None:
            return
        if reviewing:
            self._paint_review_footer(rounds.counts(self._review_state()))
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
        self._place_open_folder()
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
        _style_menu(menu)

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
        """Start Manage models (``voxprint-mdl --gui --app model-forge --manifest <mdl.json>``)."""
        self._launch(manage_argv())

    def _open_required_models(self) -> None:
        """Open the MDL progress window for missing required ids before any job."""
        try:
            missing = missing_ids()
        except (OSError, ValueError):
            return
        if missing:
            self._launch(download_argv(tuple(missing), gui=True))

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
        self._loss_label = None
        self._review_actions = None
        # A paused review replaces the training intro; Continue, train fully and Finish drive that screen.
        if not (step == "train" and self._review_waiting() and not self.runner.busy):  # type: ignore[attr-defined]
            self._intro(column, step)
        builder = {
            "photos": self._photos,
            "subject": self._subject,
            "refset": self._refset,
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
        # Always on: with "as needed" the page is sized for the width before the bar appears, so wrapped
        # text (the Details boxes, the base hint) gets too little height and is cut.
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
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
        # Continue / Finish stay under the scroll, so an 800 px window shows the heading and the row together.
        if self._review_actions is not None:
            stack.addWidget(self._review_actions)
        # The footer (status line, then the shell's Back/Next) is pinned under the scrolling step body;
        # a thin line marks where the body ends, so a row cut by the scroll edge reads as "scroll for more".
        line = QtWidgets.QFrame()
        line.setObjectName("forgeStepFooterLine")
        line.setFixedHeight(1)
        line.setAutoFillBackground(True)
        line.setBackgroundRole(QtGui.QPalette.ColorRole.Mid)
        stack.addWidget(line)
        stack.addWidget(status)
        for index in reversed(range(layout.count())):  # type: ignore[attr-defined]
            item = layout.itemAt(index)  # type: ignore[attr-defined]
            if item is not None and item.spacerItem() is not None:
                layout.removeItem(item)  # type: ignore[attr-defined]
        layout.insertWidget(0, outer, 1)  # type: ignore[attr-defined]
        self._expert_extra(step)

    def _expert_extra(self, step: str) -> None:
        """Step controls that live in the shell's Expert panel (the Build a set round settings)."""
        form = self.widget._parts()["expert_form"]
        host = form.parentWidget()  # type: ignore[attr-defined]
        if host is not None:
            for old in host.findChildren(QtWidgets.QWidget, "forge.refset.expert"):
                form.removeWidget(old)  # type: ignore[attr-defined]
                old.hide()
                old.setParent(None)
                old.deleteLater()
        if not (self.draft.expert and step == "refset" and self.draft.source == "refs"):
            return
        box = QtWidgets.QWidget()
        box.setObjectName("forge.refset.expert")
        column = QtWidgets.QVBoxLayout(box)
        column.setContentsMargins(0, 8, 0, 0)
        column.setSpacing(8)
        heading = QtWidgets.QLabel(tr("lora.step.refset.title"))
        heading.setObjectName("settingGroup")
        column.addWidget(heading)
        self._refset_settings(column, self._refset_state())
        index = form.count()  # type: ignore[attr-defined]
        last = form.itemAt(index - 1) if index else None  # type: ignore[attr-defined]
        if last is not None and last.spacerItem() is not None:
            index -= 1  # above the shell's closing stretch
        form.insertWidget(index, box)  # type: ignore[attr-defined]

    def _refset_state(self) -> dict[str, object]:
        state = refset.load_state(self.project)
        if state.get("refs") and not refset.same_refs(state, self.draft.photos):
            state = refset.empty_state()  # new references: the engine starts a new loop on the next round
        return state

    def _photos_count_key(self) -> str:
        """The one count line: recommended photos, or 1–10 samples, for the chosen subject."""
        base = "forge.app.step.photos.short_refs" if self.draft.source == "refs" else "forge.app.step.photos.short"
        if self.draft.kind in KINDS:
            return f"{base}.{self.draft.kind}"
        return base

    def _place_open_folder(self) -> None:
        """Name the project-folder button and sit it on the Back/Next row while the wizard is open."""
        folder = self.widget.root.findChild(QtWidgets.QPushButton, "openFolder")
        if folder is None:
            return
        folder.setText(tr("forge.app.folder"))
        folder.setToolTip(tr("forge.app.folder.tip"))
        folder.setProperty("textButton", "true")
        folder.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        folder.style().unpolish(folder)
        folder.style().polish(folder)
        back = self.widget.root.findChild(QtWidgets.QPushButton, "wizardBack")
        bar = self.widget.root.findChild(QtWidgets.QWidget, "actionBar")
        row = _button_row(back) if self.layout.sections.shows_wizard() else None
        if row is not None:
            if row.indexOf(folder) < 0:
                row.insertWidget(0, folder)
            row.setAlignment(folder, QtCore.Qt.AlignmentFlag.AlignVCenter)
            if bar is not None:
                bar.hide()
            return
        if bar is None or bar.layout() is None:
            return
        if folder.parentWidget() is not bar:
            bar.layout().insertWidget(0, folder)
        bar.layout().setContentsMargins(*ACTION_BAR_MARGINS)
        bar.show()

    def _intro(self, column: QtWidgets.QVBoxLayout, step: str) -> None:
        running = step == "train" and bool(self.runner.busy)  # type: ignore[attr-defined]
        key = self._photos_count_key() if step == "photos" else f"forge.app.step.{step}.short"
        short = QtWidgets.QLabel(self._running_title() if running else tr(key))
        short.setObjectName("panel")
        short.setWordWrap(True)
        short.setToolTip(tr(f"lora.step.{step}.desc"))
        column.addWidget(short)
        paragraphs = (tr(f"lora.step.{step}.desc"), tr(f"lora.step.{step}.help"))
        if step == "refset" and self.draft.source == "refs":
            # One Details for the whole step: the round notes join the step's own explanation.
            notes = ("rounds", "strength", "similar", "check")
            paragraphs += tuple(tr(f"forge.app.refset.more.{key}") for key in notes)
        # Photos keeps this text in the empty drop zone; the other steps keep Details.
        if step != "photos":
            details = self._details(f"step.{step}", paragraphs)
            column.addWidget(details)
            if step == "train":
                self._review_rounds_field(details)
                self._first_round_steps_field(details)
            if running:
                loss = QtWidgets.QLabel(self._loss_text())
                loss.setObjectName("forge.train.loss")
                loss.setWordWrap(True)
                details.body.layout().addWidget(loss)
                self._loss_label = loss

    def _review_rounds_field(self, details: Details) -> None:
        """How many review pauses, under Details on the training step."""
        if self._review_rounds() is None:
            return
        stored = int(self._review_state().get("rounds") or 0)
        if rounds.REVIEW_ROUNDS_MIN <= stored <= rounds.REVIEW_ROUNDS_MAX:
            phase = str(self._review_state().get("phase") or "")
            if phase in ("training", "review"):
                self.review_rounds = stored
        label = QtWidgets.QLabel(tr("forge.app.train.review_rounds"))
        label.setObjectName("forge.train.review_rounds.label")
        spin = QtWidgets.QSpinBox()
        spin.setObjectName("forge.train.review_rounds")
        spin.setRange(rounds.REVIEW_ROUNDS_MIN, rounds.REVIEW_ROUNDS_MAX)
        spin.setValue(self.review_rounds)
        spin.setToolTip(tr("forge.app.train.review_rounds.tip"))
        running = bool(self.runner.busy)  # type: ignore[attr-defined]
        spin.setEnabled(not running)
        label.setEnabled(not running)
        spin.valueChanged.connect(self._set_review_rounds)
        holder = QtWidgets.QWidget()
        holder.setObjectName("forge.train.review_rounds.row")
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 4, 0, 0)
        row.addWidget(label)
        row.addWidget(spin)
        row.addStretch(1)
        layout = details.body.layout()
        if layout is not None:
            layout.addWidget(holder)

    def _set_review_rounds(self, value: int) -> None:
        """Remember how many review pauses were chosen under Details."""
        self.review_rounds = int(value)

    def _first_round_steps_field(self, details: Details) -> None:
        """Steps before the first review, under Details on the training step."""
        if self._review_rounds() is None:
            return
        stored = int(self._review_state().get("first_round_steps") or 0)
        if rounds.FIRST_ROUND_STEPS_MIN <= stored <= rounds.FIRST_ROUND_STEPS_MAX:
            self.first_round_steps = stored
        label = QtWidgets.QLabel(tr("forge.app.train.first_round_steps"))
        label.setObjectName("forge.train.first_round_steps.label")
        spin = QtWidgets.QSpinBox()
        spin.setObjectName("forge.train.first_round_steps")
        spin.setRange(rounds.FIRST_ROUND_STEPS_MIN, rounds.FIRST_ROUND_STEPS_MAX)
        spin.setValue(self.first_round_steps)
        spin.setToolTip(tr("forge.app.train.first_round_steps.tip"))
        running = bool(self.runner.busy)  # type: ignore[attr-defined]
        spin.setEnabled(not running)
        label.setEnabled(not running)
        spin.valueChanged.connect(self._set_first_round_steps)
        holder = QtWidgets.QWidget()
        holder.setObjectName("forge.train.first_round_steps.row")
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 4, 0, 0)
        row.addWidget(label)
        row.addWidget(spin)
        row.addStretch(1)
        layout = details.body.layout()
        if layout is not None:
            layout.addWidget(holder)

    def _set_first_round_steps(self, value: int) -> None:
        """Remember the first-round length chosen under Details."""
        self.first_round_steps = int(value)

    def _details(self, key: str, paragraphs: tuple[str, ...]) -> Details:
        """A collapsed Details expander; its open state survives repaints of the step."""
        return Details(f"forge.details.{key}", paragraphs, opened=self.help_open.get(key, False),
                       on_toggle=lambda checked: self.help_open.__setitem__(key, checked))

    def _photos(self, column: QtWidgets.QVBoxLayout) -> None:
        self.photo_selection &= {str(path) for path in self.draft.photos}
        kind_label = QtWidgets.QLabel(tr("forge.app.kind.label"))
        kind_label.setObjectName("settingGroup")
        column.addWidget(kind_label)
        column.addLayout(self._kind_choices())
        tiles = [
            _PhotoTile(path, str(path) in self.photo_selection, self._select_photo, self._remove_paths, self._dropped)
            for path in self.draft.photos
        ]
        drop = _PhotoBoard(self._dropped, tiles)
        drop.setToolTip(tr("forge.app.photos.list.tip"))
        column.addWidget(drop, 0)
        row = QtWidgets.QHBoxLayout()
        remove = _icon_button(
            "forge.photos.remove_selected", "forge.app.photos.remove_selected",
            "forge.app.photos.remove_selected.tip", self._remove_selected, "trash", "selectionRemove",
        )
        remove.setVisible(bool(self.photo_selection))
        row.addWidget(self._add_button())
        row.addWidget(remove)
        row.addStretch(1)
        column.addLayout(row)
        extend = _Switch(tr("forge.app.photos.extend"))
        extend.setObjectName("forge.photos.extend")
        extend.setToolTip(tr("forge.app.photos.extend.tip"))
        extend.setChecked(self.draft.source == "refs")
        extend.toggled.connect(lambda checked: self._set_source("refs" if checked else "photos"))
        column.addWidget(extend)
        hint = QtWidgets.QLabel(tr("forge.app.photos.extend.hint"))
        hint.setObjectName("forge.photos.extend.hint")
        hint.setProperty("muted", "true")
        hint.setWordWrap(True)
        column.addWidget(hint)

    def _add_button(self) -> QtWidgets.QPushButton:
        """One button whose menu adds photo files or every photo in a folder."""
        button = QtWidgets.QPushButton(tr("forge.app.photos.add"))
        button.setObjectName("forge.photos.add")
        button.setProperty("photosAdd", "true")
        button.setToolTip(tr("lora.photos.add.tip"))
        button.setAutoDefault(False)
        menu = _menu(button, "forge.photos.add_menu")
        files = menu.addAction(tr("forge.app.photos.add_files"))
        files.triggered.connect(self.choose_photos)
        folder = menu.addAction(tr("forge.app.photos.add_folder"))
        folder.triggered.connect(self.choose_folder)
        button.setMenu(menu)
        return button

    def _kind_choices(self) -> QtWidgets.QVBoxLayout:
        """Person, object and style as one segmented choice, with a note under each segment."""
        outer = QtWidgets.QVBoxLayout()
        outer.setSpacing(6)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(0)
        buttons.setContentsMargins(0, 0, 0, 0)
        notes = QtWidgets.QHBoxLayout()
        notes.setSpacing(8)
        frame = QtWidgets.QFrame()
        frame.setObjectName("kindSegments")
        frame.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        frame.setLayout(buttons)
        group = QtWidgets.QButtonGroup(frame)
        group.setExclusive(True)
        ends = {"person": "left", "object": "mid", "style": "right"}
        for value in KINDS:
            chip = QtWidgets.QPushButton(tr(f"lora.kind.{value}"))
            chip.setObjectName(f"forge.chip.{value}")
            chip.setProperty("segment", "true")
            chip.setProperty("end", ends[value])
            chip.setCheckable(True)
            chip.setChecked(value == self.draft.kind)
            chip.setAutoDefault(False)
            chip.setMinimumWidth(_control_width(chip))
            chip.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            chip.setToolTip(tr("lora.kind.tip"))
            chip.clicked.connect(lambda _checked=False, v=value: self._set_kind(v))
            group.addButton(chip)
            buttons.addWidget(chip, 1)
            note = QtWidgets.QLabel(tr(f"forge.app.kind.{value}.note"))
            note.setObjectName(f"forge.kind.note.{value}")
            note.setProperty("kindNote", "true")
            note.setProperty("muted", "true")
            note.setWordWrap(True)
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            notes.addWidget(note, 1)
        outer.addWidget(frame)
        outer.addLayout(notes)
        outer._group = group  # type: ignore[attr-defined]
        return outer

    def _chips(
        self,
        values: tuple[str, ...],
        label: str,
        current: str,
        on_pick: Callable[[str], None],
        *,
        tip: str = "",
        tips: str = "",
        enabled: bool = True,
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
            chip.setFixedWidth(_control_width(chip))
            chip.setEnabled(enabled)
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
            consent = self._box("forge.app.subject.consent", "lora.subject.consent.tip", "consent")
            verbal = self._box("forge.app.subject.verbal", "lora.subject.verbal.tip", "verbal", self._set_verbal)
            content = self._box("forge.app.subject.adult_content", "lora.subject.adult_content.tip", "adult_content")
            content.setVisible(self.draft.adult)
            attr = _attr_edit(self.draft, self._gate)
            adult = self._box(
                "forge.app.subject.adult", "lora.subject.adult.tip", "adult",
                lambda checked: self._set_adult(checked, content, attr),
            )
            column.addWidget(_flow_row("forge.subject.consent_row", consent, verbal))
            column.addWidget(_flow_row("forge.subject.adult_row", adult, content))
            attr_form = QtWidgets.QFormLayout()
            attr_form.addRow(tr("forge.app.subject.attr"), attr)
            column.addLayout(attr_form)
        elif kind == "object":
            column.addWidget(self._box("lora.subject.rights_object", "lora.subject.rights_object.tip", "rights"))
        elif kind == "style":
            column.addWidget(self._box("forge.app.subject.rights_style", "lora.subject.rights_style", "rights"))
        else:
            hint = QtWidgets.QLabel(tr("forge.app.subject.choose_kind"))
            hint.setObjectName("warn")
            column.addWidget(hint)
        verbal_only = kind == "person" and self.draft.verbal
        release_row = QtWidgets.QHBoxLayout()
        attach = _button("forge.app.subject.release", "lora.subject.release.tip", self.choose_release)
        attach.setObjectName("forge.subject.release")
        attach.setEnabled(not verbal_only)
        release_row.addWidget(attach)
        shown = self.draft.release
        if verbal_only:
            text = tr("forge.app.subject.release_verbal")
        elif shown:
            text = tr("forge.app.subject.release_set", name=shown.name)
        else:
            text = tr("forge.app.subject.release_none")
        state = QtWidgets.QLabel(text)
        state.setObjectName("panel")
        state.setWordWrap(True)
        release_row.addWidget(state, 1)
        if shown is not None and not verbal_only:
            release_row.addWidget(
                _button("forge.app.subject.release_clear", "forge.app.subject.release_clear.tip", self.clear_release)
            )
        column.addLayout(release_row)
        details = {
            "person": ("lora.subject.consent.tip", "lora.subject.verbal.tip", "lora.subject.release.tip",
                       "lora.subject.adult.tip", "lora.subject.adult_content.tip", "forge.app.subject.attr.tip"),
            "object": ("lora.subject.rights_object.tip", "lora.subject.release.tip"),
            "style": ("lora.subject.rights_style", "lora.subject.rights_style.tip", "lora.subject.release.tip"),
        }.get(kind)
        if details:
            column.addWidget(self._details(f"subject.{kind}", tuple(tr(key) for key in details)))
        note = _edit(self.draft.note, "lora.subject.note.tip", "forge.app.subject.note.field")
        note.textChanged.connect(lambda text: setattr(self.draft, "note", text))
        notes = QtWidgets.QFormLayout()
        notes.addRow(tr("forge.app.subject.note"), note)
        column.addLayout(notes)

    def _box(
        self, label: str, tip: str, field: str, on_toggle: Callable[[bool], None] | None = None
    ) -> QtWidgets.QCheckBox:
        box = QtWidgets.QCheckBox(tr(label))
        box.setObjectName(f"forge.subject.{field}")
        box.setToolTip(tr(tip))
        box.setChecked(bool(getattr(self.draft, field)))
        if on_toggle is not None:
            box.toggled.connect(on_toggle)
        else:
            box.toggled.connect(lambda checked, f=field: (setattr(self.draft, f, checked), self._gate()))
        return box

    def _set_verbal(self, checked: bool) -> None:
        self.draft.set_verbal(checked)
        QtCore.QTimer.singleShot(0, self.repaint)  # the release row changes; rebuild after this signal

    def _set_adult(self, checked: bool, content: QtWidgets.QCheckBox, attr: QtWidgets.QLineEdit) -> None:
        """The adult-content box shows only while 18 or older is ticked; the attribute field is enabled only then."""
        self.draft.set_adult(checked)
        if not checked:
            content.setChecked(False)
            attr.clear()
        content.setVisible(checked)
        attr.setEnabled(checked)
        row = content.parentWidget()
        if isinstance(row, FlowRow):
            row.relayout()
        self._gate()

    # ----- build a set from references ------------------------------------
    def _refset(self, column: QtWidgets.QVBoxLayout) -> None:
        """References, round settings, Make pictures / This is it, and the Keep/Reject review grid."""
        if self.draft.source != "refs":
            skip = QtWidgets.QLabel(tr("forge.app.refset.skip"))
            skip.setObjectName("panel")
            skip.setWordWrap(True)
            column.addWidget(skip)
            switch = _button("forge.app.refset.switch", "forge.app.source.refs.tip",
                             lambda: self._set_source("refs"))
            column.addWidget(switch, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            return
        self._kept_label = self._finish_button = None
        state = self._refset_state()
        busy = self.runner.busy  # type: ignore[attr-defined]
        blocked = bool(state.get("blocked"))
        heading = QtWidgets.QLabel(tr("forge.app.refset.refs", count=len(self.draft.photos)))
        heading.setObjectName("settingGroup")
        column.addWidget(heading)
        strip = QtWidgets.QListWidget()
        strip.setObjectName("forge.refset.refs")
        strip.setToolTip(tr("forge.app.refset.refs.tip"))
        strip.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        strip.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        strip.setWrapping(False)
        strip.setIconSize(QtCore.QSize(72, 72))
        strip.setFixedHeight(104)
        strip.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for path in self.draft.photos:
            strip.addItem(QtWidgets.QListWidgetItem(_thumb(path), ""))
        column.addWidget(strip)
        if not self.draft.expert:
            summary = QtWidgets.QLabel(tr("forge.app.refset.simple", count=refset.DEFAULT_CANDIDATES))
            summary.setObjectName("panel")
            summary.setWordWrap(True)
            column.addWidget(summary)
        rounds = state.get("rounds") or []
        row = QtWidgets.QHBoxLayout()
        make_key = "forge.app.refset.next_round" if rounds else "forge.app.refset.make"
        make = _button(make_key, "forge.app.refset.make.tip", self.make_round, primary=True)
        make.setObjectName("primary")
        finish = _button("forge.app.refset.finish", "forge.app.refset.finish.tip", self.finish_set)
        finish.setObjectName("forge.refset.finish")
        restart = _button("forge.app.refset.restart", "forge.app.refset.restart.tip", self.restart_set)
        restart.setObjectName("forge.refset.restart")
        make.setEnabled(not busy and not blocked)
        finish.setEnabled(not busy and not blocked and bool(refset.kept(state)))
        self._finish_button = finish
        restart.setEnabled(not busy and not blocked and bool(rounds))
        # One action row: draw, approve, and (set apart on the right) start over. Next stays off until This is it.
        row.addWidget(make)
        row.addWidget(finish)
        row.addStretch(1)
        row.addWidget(restart)
        column.addLayout(row)
        if blocked:
            if self.status is None:
                self.status = ("lora.err.refset_blocked", {}, True)
            stop = QtWidgets.QLabel(tr("lora.err.refset_blocked"))
            stop.setObjectName("warn")
            stop.setWordWrap(True)
            column.addWidget(stop)
        self._progress_widgets(column)
        if blocked or not rounds:
            return
        self._kept_label = QtWidgets.QLabel()
        self._kept_label.setObjectName("presetLabel")
        self._kept_label.setWordWrap(True)
        column.addWidget(self._kept_label)
        shown = refset.candidates(state)
        if not shown:
            empty = QtWidgets.QLabel(tr("forge.app.refset.none_shown"))
            empty.setObjectName("panel")
            empty.setWordWrap(True)
            column.addWidget(empty)
        root = refset.folder(self.project)
        cards = tuple(self._candidate_card(item, root / str(item["file"]), busy) for item in shown)
        if cards:
            column.addWidget(_flow_row("forge.refset.grid", *cards))
        self._update_kept()

    def _refset_settings(self, column: QtWidgets.QVBoxLayout, state: dict[str, object]) -> None:
        """Expert round settings and descriptions; built into the Expert panel by :meth:`_expert_extra`."""
        values = self.refset_settings
        form = QtWidgets.QFormLayout()
        count = QtWidgets.QSpinBox()
        count.setObjectName("forge.refset.count")
        count.setRange(*refset.CANDIDATE_RANGE)
        count.setValue(int(values["count"]))
        count.setToolTip(tr("forge.app.refset.count.tip"))
        count.valueChanged.connect(lambda v: values.__setitem__("count", int(v)))
        strength = QtWidgets.QDoubleSpinBox()
        strength.setObjectName("forge.refset.strength")
        strength.setRange(refset.STRENGTH_MIN, refset.STRENGTH_MAX)
        strength.setSingleStep(0.05)
        strength.setDecimals(2)
        strength.setValue(float(values["strength"]))
        strength.setToolTip(tr("forge.app.refset.strength.tip"))
        strength.valueChanged.connect(lambda v: values.__setitem__("strength", round(float(v), 2)))
        steps = QtWidgets.QSpinBox()
        steps.setObjectName("forge.refset.steps")
        steps.setRange(*refset.STEPS_RANGE)
        steps.setValue(int(values["steps"]))
        steps.setToolTip(tr("forge.app.refset.steps.tip"))
        steps.valueChanged.connect(lambda v: values.__setitem__("steps", int(v)))
        form.addRow(tr("forge.app.refset.count"), count)
        form.addRow(tr("forge.app.refset.strength"), strength)
        form.addRow(tr("forge.app.refset.steps"), steps)
        column.addLayout(form)
        heading = QtWidgets.QLabel(tr("forge.app.refset.prompts"))
        heading.setObjectName("settingGroup")
        column.addWidget(heading)
        prompts = refset.load_prompts(self.project) if state.get("refs_checked") else {}
        refs = [(str(r["id"]), refset.folder(self.project) / str(r["file"])) for r in state.get("refs") or []
                if str(r.get("id")) in prompts]
        if not refs:
            later = QtWidgets.QLabel(tr("forge.app.refset.prompts_later"))
            later.setObjectName("simpleNote")
            later.setWordWrap(True)
            column.addWidget(later)
            return
        table = QtWidgets.QTableWidget(len(refs), 2)
        table.setObjectName("forge.refset.prompts")
        table.setToolTip(tr("forge.app.refset.prompts.tip"))
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setIconSize(QtCore.QSize(48, 48))
        table.setWordWrap(True)
        for index, (rid, path) in enumerate(refs):
            picture = QtWidgets.QTableWidgetItem(_thumb(path, 48), "")
            picture.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            text = QtWidgets.QTableWidgetItem(prompts[rid])
            text.setData(QtCore.Qt.ItemDataRole.UserRole, rid)
            table.setItem(index, 0, picture)
            table.setItem(index, 1, text)
            table.setRowHeight(index, 56)
        table.setColumnWidth(0, 60)
        table.setFixedHeight(min(4, len(refs)) * 56 + 6)
        table.itemChanged.connect(self._prompt_edited)
        column.addWidget(table)

    def _candidate_card(self, item: dict[str, object], path: Path, busy: bool) -> QtWidgets.QFrame:
        """One picture with a checkmark toggle on it (ticked = kept) and a reserved line for the ≈ badge."""
        cid = str(item["id"])
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card.setProperty("candidate", cid)
        inner = QtWidgets.QVBoxLayout(card)
        inner.setContentsMargins(6, 6, 6, 6)
        inner.setSpacing(4)
        keep = QtWidgets.QPushButton("\u2713")
        keep.setObjectName(f"forge.refset.keep.{cid}")
        keep.setProperty("keepMark", "true")
        keep.setCheckable(True)
        keep.setChecked(str(item.get("verdict", "pending")) == "keep")
        keep.setFixedSize(KEEP_MARK, KEEP_MARK)
        keep.setToolTip(tr("forge.app.refset.keep.tip"))
        keep.setAccessibleName(tr("forge.app.refset.keep"))
        keep.setEnabled(not busy)
        keep.setAutoDefault(False)
        keep.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        picture = _ClickLabel(keep)
        picture.setFixedSize(CARD_PICTURE, CARD_PICTURE)
        picture.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(str(path))
        if not pixmap.isNull():
            picture.setPixmap(pixmap.scaled(CARD_PICTURE, CARD_PICTURE, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                            QtCore.Qt.TransformationMode.SmoothTransformation))
        picture.setToolTip(tr("forge.app.refset.picture.tip", seed=item.get("seed", ""),
                              round=int(item.get("round", 0)) + 1))
        stack = QtWidgets.QGridLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(picture, 0, 0)
        corner = QtWidgets.QWidget()
        corner.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        corner_row = QtWidgets.QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 6, 6, 0)
        corner_row.addWidget(keep)
        stack.addWidget(corner, 0, 0, QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        inner.addLayout(stack)
        similar = bool(item.get("near_duplicate_of"))
        mark = QtWidgets.QLabel(tr("forge.app.refset.similar_mark") if similar else "")
        mark.setObjectName("forge.refset.similar")
        mark.setProperty("similar", "true" if similar else "false")
        mark.setWordWrap(True)
        mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        # Two lines are always reserved, so cards with and without the badge line up in the grid.
        mark.setFixedHeight(mark.fontMetrics().lineSpacing() * 2 + 2)
        if similar:
            mark.setToolTip(tr("forge.app.refset.similar_mark.tip"))
        inner.addWidget(mark)
        card.setProperty("kept", "true" if keep.isChecked() else "false")
        keep.toggled.connect(lambda checked, c=card: self._verdict(cid, checked, c))
        card.setFixedWidth(CARD_PICTURE + 16)
        return card

    def _verdict(self, cid: str, checked: bool, card: QtWidgets.QFrame | None = None) -> None:
        refset.set_verdict(self.project, cid, "keep" if checked else "pending")
        if card is not None:
            card.setProperty("kept", "true" if checked else "false")
            card.style().unpolish(card)
            card.style().polish(card)
        self.draft.approved = []
        self.draft.prepared = None
        self._update_kept()
        self._gate()

    _kept_label: QtWidgets.QLabel | None = None
    _finish_button: QtWidgets.QPushButton | None = None

    def _update_kept(self) -> None:
        label = getattr(self, "_kept_label", None)
        if label is None:
            return
        state = refset.load_state(self.project)
        busy = self.runner.busy  # type: ignore[attr-defined]
        try:
            drawn = sum(len(r.get("candidates") or []) for r in state.get("rounds") or [])
            # One recommended range (lora.limits) in both modes; Expert adds its allowed range only if it differs.
            expert = self.draft.expert and expert_range_differs()
            key = "forge.app.refset.kept_expert" if expert else "forge.app.refset.kept"
            label.setText(tr(key, kept=len(refset.kept(state)), shown=drawn))
            if self._finish_button is not None:
                self._finish_button.setEnabled(not busy and not state.get("blocked") and bool(refset.kept(state)))
        except RuntimeError:
            pass

    def _prompt_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        rid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not rid:
            return
        prompts = refset.load_prompts(self.project)
        prompts[str(rid)] = item.text().strip()
        refset.save_prompts(self.project, prompts)

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
        if self._review_waiting() and not self.runner.busy:  # type: ignore[attr-defined]
            self._review_screen(column)
            return
        busy = self.runner.busy  # type: ignore[attr-defined]
        if busy:
            self._progress_widgets(column)
            self._apply_train_progress()
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
            row = self._chips(values, label, current, pick, tip=tip, tips=tips, enabled=not busy)
            if title == "forge.app.preset.label" and self.custom():
                custom = QtWidgets.QLabel(tr("forge.app.preset.custom"))
                custom.setObjectName("warn")
                reset = _button("forge.app.preset.reset", "forge.app.preset.reset.tip", self.apply_simple)
                reset.setEnabled(not busy)
                row.insertWidget(row.count() - 1, custom)
                row.insertWidget(row.count() - 1, reset)
            column.addLayout(row)
        heading = QtWidgets.QLabel(tr("forge.app.base.label"))
        heading.setObjectName("settingGroup")
        column.addWidget(heading)
        base = basepicker.build_picker(self._base_options(), self.simple["base"])
        base.currentIndexChanged.connect(lambda _i: self.simple.__setitem__("base", str(base.currentData() or "")))
        base.setEnabled(not busy)
        column.addWidget(basepicker.hint_label())
        column.addWidget(base, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        memory = basepicker.memory_details()
        column.addWidget(self._details("train.base", (tr("forge.app.base.details"),) + ((memory,) if memory else ())))
        previews = _Switch(tr("forge.app.previews.label"))
        previews.setObjectName("forge.train.previews")
        previews.setToolTip(tr("lora.previews.tip"))
        previews.setChecked(self.previews)
        previews.setEnabled(not busy)
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
        stop = _button("forge.app.train.stop", "lora.train.stop.tip", self._stop_clicked)
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
        if not busy:
            self._progress_widgets(column, idle="forge.app.train.idle")

    def _result(self, column: QtWidgets.QVBoxLayout) -> None:
        counted = self._finished_steps()
        if counted is not None:
            done, planned = counted
            label = QtWidgets.QLabel(tr("forge.app.result.steps", done=done, planned=planned))
            label.setObjectName("forge.result.steps")
            label.setWordWrap(True)
            column.addWidget(label)
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
            ("forge.app.result.export_lora", "forge.app.result.export_lora.tip", self.export_lora, True),
            ("forge.app.result.save", "lora.result.save.tip", self.save, False),
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
        shown = self.progress_percent is not None
        bar.setVisible(shown)
        percent = QtWidgets.QLabel("" if self.progress_percent is None else f"{int(round(self.progress_percent))}%")
        percent.setObjectName("progressPercent")
        percent.setVisible(shown)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(bar, 1)
        row.addWidget(percent)
        column.addWidget(text)
        column.addLayout(row)
        self._progress_text, self._progress_bar, self._progress_percent_label = text, bar, percent

    def _running_title(self) -> str:
        """The training-step heading while a stage is running."""
        state = self._review_state()
        total = int(state.get("rounds") or 0)
        phase = str(state.get("phase") or "")
        if total > 0 and phase in ("training", "review"):
            index = int(state.get("index") or 0)
            current = index if phase == "review" else index + 1
            current = min(total, max(1, current))
            return tr("forge.app.train.running.round", round=current, rounds=total)
        return tr("forge.app.train.running")

    def _loss_text(self) -> str:
        """The loss sentence kept in Details while a stage is running."""
        row = read_progress(self.project)
        loss = None if row is None else row.get("loss")
        shown = "—" if not isinstance(loss, int | float) else f"{float(loss):.4f}"
        return tr("lora.progress.loss", loss=shown)

    def _review_state(self) -> dict[str, object]:
        """The saved review, or an idle state when this project has none."""
        project = Project(self.project)
        if not rounds.state_path(project).is_file():
            return rounds.empty_state()
        try:
            return rounds.load_state(project)
        except (OSError, ValueError, LoraUsageError):
            return rounds.empty_state()

    def _review_waiting(self) -> bool:
        """True when a stage has finished and the samples are waiting for marks."""
        return self._review_state().get("phase") == "review"

    def _review_rounds(self) -> tuple[int, int] | None:
        """Rounds and samples when the sample-set switch is on, or a review is already open."""
        state = self._review_state()
        phase = str(state.get("phase") or "idle")
        planned = int(state.get("rounds") or 0)
        if phase in ("training", "review") and planned > 0:
            return planned, int(state.get("samples_per_round") or rounds.DEFAULT_SAMPLES)
        if self.draft.source == "refs":
            return self.review_rounds, rounds.DEFAULT_SAMPLES
        return None

    def _finished_steps(self) -> tuple[int, int] | None:
        """Steps the saved LoRA got, and the preset total, once a run has recorded both."""
        try:
            data = Project(self.project).read()
        except (OSError, ValueError, LoraUsageError):
            return None
        train = data.get("train")
        if not isinstance(train, dict):
            return None
        done, planned = train.get("step"), train.get("steps_planned")
        if isinstance(done, bool) or isinstance(planned, bool):
            return None
        if not isinstance(done, int) or not isinstance(planned, int) or planned < 1:
            return None
        return done, planned

    def _review_screen(self, column: QtWidgets.QVBoxLayout) -> None:
        """Original photos, generated samples, and continue with reinforcement, just continue, or finish."""
        state = self._review_state()
        counted = rounds.counts(state)
        busy = self.runner.busy  # type: ignore[attr-defined]
        # A flush first line sits under the viewport edge and reads as a cut-off header.
        margins = column.contentsMargins()
        column.setContentsMargins(margins.left(), max(margins.top(), 12), margins.right(), margins.bottom())
        heading = QtWidgets.QLabel(tr("forge.app.review.heading"))
        heading.setObjectName("forge.review.heading")
        heading.setProperty("emptyTitle", "true")
        heading.setWordWrap(True)
        heading.setMinimumHeight(max(36, heading.fontMetrics().height() + 10))
        column.addWidget(heading)
        lead = QtWidgets.QLabel(tr("forge.app.review.lead"))
        lead.setObjectName("forge.review.lead")
        lead.setProperty("muted", "true")
        lead.setWordWrap(True)
        column.addWidget(lead)
        round_label = QtWidgets.QLabel(tr("forge.app.review.round", round=counted["round"], rounds=counted["rounds"]))
        round_label.setObjectName("forge.review.round")
        round_label.setWordWrap(True)
        column.addWidget(round_label)
        counts = QtWidgets.QLabel(tr(
            "forge.app.review.counts",
            liked=counted["liked"], disliked=counted["disliked"], unmarked=counted["unmarked"],
        ))
        counts.setObjectName("forge.review.counts")
        counts.setWordWrap(True)
        column.addWidget(counts)
        self._review_counts = counts
        originals = QtWidgets.QLabel(tr("forge.app.review.originals"))
        originals.setObjectName("settingGroup")
        column.addWidget(originals)
        tiles = tuple(self._review_original(path) for path in self._review_originals())
        if tiles:
            column.addWidget(_flow_row("forge.review.originals", *tiles))
        samples_label = QtWidgets.QLabel(tr("forge.app.review.samples"))
        samples_label.setObjectName("settingGroup")
        column.addWidget(samples_label)
        samples = [sample for sample in state.get("samples") or [] if isinstance(sample, dict)]
        cards = tuple(self._sample_card(sample, busy) for sample in samples)
        if cards:
            column.addWidget(_flow_row("forge.review.samples", *cards))
        tools = QtWidgets.QHBoxLayout()
        like_all = _button("forge.app.review.like_all", "forge.app.review.like_all.tip", self._like_all)
        like_all.setObjectName("forge.review.like_all")
        like_all.setProperty("textButton", "true")
        reset = _button("forge.app.review.reset", "forge.app.review.reset.tip", self._reset_marks)
        reset.setObjectName("forge.review.reset")
        reset.setProperty("textButton", "true")
        for button in (like_all, reset):
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setEnabled(not busy)
            tools.addWidget(button)
        tools.addStretch(1)
        column.addLayout(tools)
        row = QtWidgets.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.setSpacing(12)
        cont = self._continue_control(busy)
        finish = self._finish_menu(busy)
        note = _ElidedLabel()
        note.setObjectName("forge.review.note")
        note.setProperty("muted", "true")
        note.setWordWrap(False)
        note.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self._review_note = note
        self._show_review_note(cont.mode())
        cont.toggled.connect(self._show_review_note)
        row.addWidget(cont, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(note, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        row.addWidget(self._review_attire(busy), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(finish, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        bar = QtWidgets.QWidget()
        bar.setObjectName("forge.review.actions")
        pinned = QtWidgets.QVBoxLayout(bar)
        pinned.setContentsMargins(0, 8, 0, 0)
        pinned.setSpacing(0)
        pinned.addLayout(row)
        self._review_actions = bar
        self._review_continue = cont

    def _review_attire(self, busy: bool) -> QtWidgets.QWidget:
        """Attire for the next round, on the continue row and to the right of the help line."""
        row = QtWidgets.QWidget()
        row.setObjectName("forge.review.attire.row")
        row.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        label = _FullWidthLabel(tr("forge.app.review.attire"))
        label.setObjectName("forge.review.attire.label")
        label.setStyleSheet("color: #f4f1ea; background: transparent;")
        combo = _AttireCombo()
        tip = tr("forge.app.review.attire.tip")
        combo.setToolTip(tip)
        label.setToolTip(tip)
        label.setBuddy(combo)
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        for key, mode in (
            ("forge.app.review.attire.as_photos", "auto"),
            ("forge.app.review.attire.clothed", "clothed"),
            ("forge.app.review.attire.unclothed", "unclothed"),
        ):
            combo.addItem(tr(key), mode)
        combo.fit_to_items()
        stored = self._stored_sample_attire()
        shown = stored if stored in ("clothed", "unclothed") else "auto"
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(shown)))
        combo.blockSignals(False)
        combo.setEnabled(not busy)
        combo.currentIndexChanged.connect(self._store_sample_attire)
        self._attire_combo = combo
        layout.addWidget(label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(combo, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        return row

    def _stored_sample_attire(self) -> str:
        """The attire mode saved on the project, or ``auto``."""
        try:
            return read_attire(Project(self.project).read())
        except (OSError, LoraUsageError, json.JSONDecodeError, ValueError):
            return "auto"

    def _store_sample_attire(self, index: int) -> None:
        """Keep the combo's choice on the project. The open round is left as it is."""
        combo = getattr(self, "_attire_combo", None)
        if not isinstance(combo, QtWidgets.QComboBox):
            return
        mode = combo.itemData(index)
        if mode not in ("auto", "clothed", "unclothed"):
            return
        project = Project(self.project)
        try:
            data = project.read()
        except (OSError, LoraUsageError, json.JSONDecodeError, ValueError):
            return
        if data.get("sample_attire") == mode:
            return
        store_attire(project, str(mode))

    def _show_review_note(self, mode: str) -> None:
        """One muted line for the mode the main button would start."""
        note = getattr(self, "_review_note", None)
        if not isinstance(note, QtWidgets.QLabel):
            return
        key = "forge.app.review.note.plain" if mode == "plain" else "forge.app.review.note.reinforced"
        note.setText(tr(key))
        note.setToolTip(tr(key))

    def _continue_control(self, busy: bool) -> "_ContinueToggle":
        """Large primary button. ▾ swaps reinforcement and plain continue."""
        control = _ContinueToggle(
            tr("forge.app.review.continue"),
            tr("forge.app.review.unattended"),
            tr("forge.app.review.continue.tip"),
            tr("forge.app.review.unattended.tip"),
            self._stored_continue_mode(),
        )
        control.setEnabled(not busy)
        control.activated.connect(self._start_continue)
        control.toggled.connect(self._store_continue_mode)
        return control

    def _stored_continue_mode(self) -> str:
        """``plain`` when the project saved that choice, otherwise reinforcement."""
        try:
            data = Project(self.project).read()
        except (OSError, LoraUsageError, json.JSONDecodeError, ValueError):
            return "reinforced"
        return "plain" if data.get("continue_mode") == "plain" else "reinforced"

    def _store_continue_mode(self, mode: str) -> None:
        """Write the continue choice onto ``project.json``."""
        if mode not in ("plain", "reinforced"):
            return
        project = Project(self.project)
        try:
            data = project.read()
        except (OSError, LoraUsageError, json.JSONDecodeError, ValueError):
            return
        if data.get("continue_mode") == mode:
            return
        project.write({**data, "continue_mode": mode})

    def _start_continue(self) -> None:
        """Enter on the main face starts the mode the ▾ last selected."""
        control = getattr(self, "_review_continue", None)
        if control is not None and control.mode() == "plain":
            self.continue_unattended()
            return
        self.continue_review()

    def _finish_menu(self, busy: bool) -> QtWidgets.QPushButton:
        """Small Finish button. Its menu saves or discards."""
        button = _FinishButton(tr("forge.app.review.finish"))
        button.setObjectName("forge.review.finish")
        button.setProperty("reviewFinish", "true")
        button.setToolTip(tr("forge.app.review.finish.tip"))
        button.setAutoDefault(False)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(28)
        button.setMaximumHeight(32)
        button.setEnabled(not busy)
        button.style().unpolish(button)
        button.style().polish(button)
        menu = _FinishMenu(button)
        save = menu.addAction(tr("forge.app.review.finish.save"))
        save.setObjectName("forge.review.finish.save")
        save.setToolTip(tr("forge.app.review.finish.save.tip"))
        save.triggered.connect(self.finish_review)
        discard = menu.addAction(tr("forge.app.review.finish.discard"))
        discard.setObjectName("forge.review.finish.discard")
        discard.setToolTip(tr("forge.app.review.finish.discard.tip"))
        discard.triggered.connect(self.discard_review)
        menu.setToolTipsVisible(True)
        button.setMenu(menu)
        return button

    def _review_originals(self) -> list[Path]:
        """The user's prepared photos, or the photos still on the draft."""
        project = Project(self.project)
        paths = []
        for item in project.items():
            path = Path(item.file)
            if not path.is_absolute():
                path = project.root / path
            if path.is_file():
                paths.append(path)
        return paths or [path for path in self.draft.photos if path.is_file()]

    def _review_original(self, path: Path) -> QtWidgets.QLabel:
        """One original photo, shown smaller than a generated sample."""
        picture = QtWidgets.QLabel()
        picture.setObjectName("forge.review.original")
        picture.setFixedSize(96, 96)
        picture.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(str(path))
        if not pixmap.isNull():
            picture.setPixmap(pixmap.scaled(
                96, 96, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation,
            ))
        return picture

    def _sample_card(self, sample: dict[str, object], busy: bool) -> QtWidgets.QFrame:
        """One generated picture with include and leave-out toggles."""
        sample_id = str(sample.get("id") or "")
        verdict = str(sample.get("verdict") or "unmarked")
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card.setProperty("verdict", verdict)
        inner = QtWidgets.QVBoxLayout(card)
        inner.setContentsMargins(6, 6, 6, 6)
        inner.setSpacing(6)
        inner.addWidget(self._sample_picture(sample, verdict))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        like = self._verdict_button(sample_id, "like", "\U0001f44d", verdict == "like", busy)
        dislike = self._verdict_button(sample_id, "dislike", "\U0001f44e", verdict == "dislike", busy)
        dislike.setProperty("down", "true")
        like.toggled.connect(lambda checked, sid=sample_id, frame=card: self._mark_sample(
            sid, "like" if checked else "unmarked", frame,
        ))
        dislike.toggled.connect(lambda checked, sid=sample_id, frame=card: self._mark_sample(
            sid, "dislike" if checked else "unmarked", frame,
        ))
        row.addWidget(like)
        row.addWidget(dislike)
        row.addStretch(1)
        inner.addLayout(row)
        card.setFixedWidth(CARD_PICTURE + 16)
        return card

    def _verdict_button(
        self, sample_id: str, verdict: str, glyph: str, checked: bool, busy: bool,
    ) -> QtWidgets.QPushButton:
        """A square include or leave-out toggle for one sample."""
        button = QtWidgets.QPushButton(glyph)
        button.setObjectName(f"forge.review.{verdict}.{sample_id}")
        button.setProperty("reviewMark", "true")
        button.setCheckable(True)
        button.setChecked(checked)
        button.setFixedSize(REVIEW_MARK, REVIEW_MARK)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        button.setToolTip(tr(f"forge.app.review.{verdict}.tip"))
        button.setAccessibleName(tr(f"forge.app.review.{verdict}"))
        button.setEnabled(not busy)
        button.setAutoDefault(False)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return button

    def _sample_picture(self, sample: dict[str, object], verdict: str) -> QtWidgets.QWidget:
        """The picture, a corner badge, and a dimming effect when the picture is left out."""
        holder = QtWidgets.QWidget()
        holder.setFixedSize(CARD_PICTURE, CARD_PICTURE)
        picture = QtWidgets.QLabel(holder)
        picture.setObjectName("forge.review.picture")
        picture.setFixedSize(CARD_PICTURE, CARD_PICTURE)
        picture.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        path = self.project / str(sample.get("file") or "")
        pixmap = QtGui.QPixmap(str(path))
        if not pixmap.isNull():
            picture.setPixmap(pixmap.scaled(
                CARD_PICTURE, CARD_PICTURE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            ))
        self._paint_picture_state(picture, verdict)
        badge = QtWidgets.QLabel(holder)
        badge.setObjectName("forge.review.badge")
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(22, 22)
        badge.move(CARD_PICTURE - 26, 4)
        self._paint_badge(badge, verdict)
        badge.raise_()
        return holder

    def _paint_picture_state(self, picture: QtWidgets.QLabel, verdict: str) -> None:
        """Dim a left-out picture. The toggles stay on the card, outside this effect."""
        if verdict != "dislike":
            picture.setGraphicsEffect(None)
            return
        effect = picture.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(picture)
            picture.setGraphicsEffect(effect)
        effect.setOpacity(0.5)

    def _paint_badge(self, badge: QtWidgets.QLabel, verdict: str) -> None:
        """A check on an included picture, a cross on a left-out one, nothing when unmarked."""
        marks = {"like": ("\u2713", "like"), "dislike": ("\u00d7", "dislike")}
        if verdict not in marks:
            badge.hide()
            badge.setProperty("reviewBadge", "")
        else:
            text, kind = marks[verdict]
            badge.setText(text)
            badge.setProperty("reviewBadge", kind)
            badge.show()
        badge.style().unpolish(badge)
        badge.style().polish(badge)

    def _mark_sample(self, sample_id: str, verdict: str, card: QtWidgets.QFrame) -> None:
        """Store one mark and refresh the counter. The other toggle follows."""
        state = rounds.transition(self._review_state(), "mark", id=sample_id, verdict=verdict)
        rounds.save_state(Project(self.project), state)
        card.setProperty("verdict", verdict)
        card.style().unpolish(card)
        card.style().polish(card)
        picture = card.findChild(QtWidgets.QLabel, "forge.review.picture")
        badge = card.findChild(QtWidgets.QLabel, "forge.review.badge")
        if picture is not None:
            self._paint_picture_state(picture, verdict)
        if badge is not None:
            self._paint_badge(badge, verdict)
        for name, on in (("like", verdict == "like"), ("dislike", verdict == "dislike")):
            button = card.findChild(QtWidgets.QPushButton, f"forge.review.{name}.{sample_id}")
            if button is None:
                continue
            button.blockSignals(True)
            button.setChecked(on)
            button.blockSignals(False)
        self._paint_review_counts()

    def _paint_review_counts(self) -> None:
        """Rewrite the liked / disliked / unmarked line from the saved review."""
        label = getattr(self, "_review_counts", None)
        if label is None:
            return
        counted = rounds.counts(self._review_state())
        try:
            label.setText(tr(
                "forge.app.review.counts",
                liked=counted["liked"], disliked=counted["disliked"], unmarked=counted["unmarked"],
            ))
            button = getattr(self, "_review_continue", None)
            if button is not None:
                # Both modes stay available while the review is open. The click checks the round itself.
                button.setEnabled(not self.runner.busy)  # type: ignore[attr-defined]
            self._paint_review_footer(counted)
        except RuntimeError:
            pass

    def _paint_review_footer(self, counted: dict[str, int]) -> None:
        """The pinned status line: liked, disliked and still unmarked."""
        label = getattr(self, "_status_label", None)
        if label is None:
            return
        try:
            label.setText(tr(
                "forge.app.review.footer",
                liked=counted["liked"], disliked=counted["disliked"], unmarked=counted["unmarked"],
            ))
            label.setObjectName("ok")
            label.style().unpolish(label)
            label.style().polish(label)
        except RuntimeError:
            pass

    def _like_all(self) -> None:
        """Mark every current sample as included."""
        self._set_review_group("like")

    def _reset_marks(self) -> None:
        """Clear every mark on the current samples."""
        self._set_review_group("unmarked")

    def _set_review_group(self, verdict: str) -> None:
        state = self._review_state()
        ids = [str(sample.get("id") or "") for sample in state.get("samples") or [] if isinstance(sample, dict)]
        if verdict == "like":
            state = rounds.set_verdicts(state, like=ids)
        elif verdict == "unmarked":
            state = rounds.set_verdicts(state, unmarked=ids)
        else:
            return
        rounds.save_state(Project(self.project), state)
        self.repaint()

    def _confirm_unmarked(self, title_key: str = "forge.app.review.continue") -> bool:
        """Ask before a pass that would drop pictures the user has not marked."""
        box = _message_box(self.window, "forge.review.unmarked")
        box.setWindowTitle(tr(title_key))
        box.setText(tr("forge.app.review.unmarked"))
        go = box.addButton(
            tr("forge.app.review.unmarked.continue"), QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        cancel = box.addButton(
            tr("forge.app.review.unmarked.cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is go

    def continue_review(self) -> None:
        """``lora continue``: the next round on the liked pictures, then another review."""
        if not self._needs("continue"):
            return
        if not rounds.can_continue(self._review_state()):
            return
        if rounds.counts(self._review_state())["unmarked"] > 0 and not self._confirm_unmarked():
            return

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            if not outcome.ok:
                return
            if result.get("state") == "done":
                self.draft.trained = True
                self.paused = False
            elif result.get("state") == "review":
                self.draft.trained = False
                self.paused = False

        self._run(["--json", "lora", "continue", str(self.project)], done)

    def continue_unattended(self) -> None:
        """``lora continue --unattended``: the remaining steps, with no more pauses."""
        if not self._needs("continue"):
            return
        if rounds.counts(self._review_state())["unmarked"] > 0 and not self._confirm_unmarked(
            "forge.app.review.unattended",
        ):
            return

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            if not outcome.ok:
                return
            if result.get("state") == "done":
                self.draft.trained = True
                self.paused = False
            elif result.get("state") == "review":
                self.draft.trained = False
                self.paused = False

        self._run(["--json", "lora", "continue", str(self.project), "--unattended"], done)

    def finish_review(self) -> None:
        """``lora finish``: confirm the steps, then save the LoRA and draw a final preview."""
        if not self._needs("finish"):
            return
        if not self._confirm_finish():
            return

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            if outcome.ok and result.get("state") == "done":
                self.draft.trained = True
                self.paused = False

        self._run(["--json", "lora", "finish", str(self.project)], done)

    def discard_review(self) -> None:
        """``lora finish --discard --yes``: confirm, then remove temporary files. Photos stay."""
        if not self._needs("finish"):
            return
        if not self._confirm_discard():
            return

        def done(outcome: engine.Outcome) -> None:
            result = outcome.result if isinstance(outcome.result, dict) else {}
            if outcome.ok and result.get("state") == "discarded":
                self.draft.trained = False
                self.paused = False

        self._run(["--json", "lora", "finish", str(self.project), "--discard", "--yes"], done)

    def _confirm_discard(self) -> bool:
        """Ask before a discard that removes temporary versions and samples."""
        box = _message_box(self.window, "forge.review.finish.discard.confirm")
        box.setWindowTitle(tr("forge.app.review.finish.discard"))
        box.setText(tr("forge.app.review.finish.discard.confirm"))
        go = box.addButton(tr("forge.app.review.finish.discard"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(
            tr("forge.app.review.unmarked.cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is go

    def _confirm_finish(self) -> bool:
        """Ask before ending the run, and show the steps done out of the plan."""
        state = self._review_state()
        box = _message_box(self.window, "forge.review.finish.confirm")
        box.setWindowTitle(tr("forge.app.review.finish.save"))
        box.setText(tr(
            "forge.app.review.finish.confirm",
            done=rounds.steps_completed(state), planned=rounds.total_steps_of(state),
        ))
        go = box.addButton(tr("forge.app.review.finish.save"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(
            tr("forge.app.review.unmarked.cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is go

    def _restore_review(self, subject: dict[str, object]) -> None:
        """Put a restarted review back on the training step, with consent and captions restored."""
        self.draft.consent = bool(subject.get("consent_obtained"))
        self.draft.adult = bool(subject.get("adult_confirmed"))
        self.draft.rights = bool(subject.get("rights_confirmed"))
        self.draft.adult_content = bool(subject.get("adult_content"))
        self.draft.verbal = subject.get("consent_method") == "verbal"
        if subject.get("attr") is not None:
            try:
                self.draft.attr = int(str(subject.get("attr")))
            except (TypeError, ValueError):
                self.draft.attr = None
        if (self.project / "captions.jsonl").is_file():
            kept, drops = engine.prepared_from(self.project)
            if kept or drops:
                self.draft.prepared = Prepared(kept, drops, False)
        if not self.draft.photos:
            for item in Project(self.project).items():
                path = Path(item.file)
                if not path.is_absolute():
                    path = self.project / path
                if path.is_file():
                    self.draft.photos.append(path)
        deck = self.layout.sections
        deck._index = STEPS.index("train")  # type: ignore[attr-defined]
        deck._save()  # type: ignore[attr-defined]

    def _arm_progress(self) -> None:
        """Poll ``train.json`` once a second while a training job is running."""
        timer = getattr(self, "_progress_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self.window)
            timer.setInterval(1000)
            timer.timeout.connect(self._read_live_progress)
            self._progress_timer = timer
        if not timer.isActive():
            timer.start()

    def _read_live_progress(self) -> None:
        """Show the latest flushed step, loss, elapsed time and ETA."""
        if not self.runner.busy:  # type: ignore[attr-defined]
            timer = getattr(self, "_progress_timer", None)
            if timer is not None:
                timer.stop()
            return
        self._apply_train_progress()

    def _apply_train_progress(self) -> None:
        """Paint ``train.json`` onto the training step: step, total, loss and ETA."""
        row = read_progress(self.project)
        if row is None:
            return
        total = int(row.get("total") or 0)
        step = int(row.get("step") or 0)
        elapsed = format_span(float(row.get("elapsed_seconds") or 0))
        eta = row.get("eta_seconds")
        pictures = row.get("first_pictures_seconds")
        sample = row.get("seconds_per_sample")
        estimated = bool(row.get("sample_seconds_estimated", True))
        if isinstance(pictures, int | float) and int(row.get("round") or 0) == 1:
            shown = format_span(float(pictures))
            if isinstance(sample, int | float) and not estimated:
                self.progress_text = tr(
                    "lora.progress.first_pictures_rate",
                    step=step, steps=total, elapsed=elapsed, eta=shown, seconds=max(1, round(float(sample))),
                )
            else:
                self.progress_text = tr(
                    "lora.progress.first_pictures", step=step, steps=total, elapsed=elapsed, eta=shown,
                )
        elif isinstance(eta, int | float):
            self.progress_text = tr(
                "lora.progress.status", step=step, steps=total, elapsed=elapsed, eta=format_span(float(eta)),
            )
        else:
            self.progress_text = tr("lora.progress.status_open", step=step, steps=total, elapsed=elapsed)
        self.progress_percent = None if total <= 0 else (step / total) * 100
        loss = getattr(self, "_loss_label", None)
        if loss is not None:
            try:
                loss.setText(self._loss_text())
            except RuntimeError:
                self._loss_label = None
        self._paint_progress()

    def _paint_progress(self) -> None:
        """Copy the current sentence and bar onto the step, ignoring a widget that was rebuilt."""
        try:
            text = getattr(self, "_progress_text", None)
            bar = getattr(self, "_progress_bar", None)
            if text is not None:
                text.setText(self.progress_text)
            shown = self.progress_percent is not None
            if bar is not None:
                bar.setValue(int((self.progress_percent or 0) * 10))
                bar.setVisible(shown)
            percent = getattr(self, "_progress_percent_label", None)
            if percent is not None:
                percent.setText("" if not shown else f"{int(round(self.progress_percent or 0))}%")
                percent.setVisible(shown)
            window = self.widget.root.window()
            if window is not None and self.progress_text:
                window.setWindowTitle(f"{tr('splash.app.' + APP_ID)} — {self.progress_text}")
        except RuntimeError:
            pass

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
        from voxprint_forge.lora.backends import installed_dir

        teacher = installed_dir(refset.TEACHER_ID) is not None
        lines.append(tr("forge.app.check.teacher_ready" if teacher else "forge.app.check.teacher_missing"))
        self._check_lines.setText("\n".join(lines))

    # ----- actions ------------------------------------------------------
    def _set_kind(self, kind: str) -> None:
        if kind == self.draft.kind:
            self._gate()
            return
        self.draft.kind = kind
        self.draft.consent = self.draft.adult = self.draft.rights = False
        self.draft.verbal = self.draft.adult_content = False
        self.draft.attr = None
        self.draft.prepared = None
        self.repaint()

    def _set_source(self, source: str) -> None:
        if source != self.draft.source:
            self.draft.set_source(source)
        self.repaint()

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

    def _select_photo(self, path: Path, modifiers: QtCore.Qt.KeyboardModifier) -> None:
        """Select one photo, or toggle it when Control is held."""
        key = str(path)
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            if key in self.photo_selection:
                self.photo_selection.remove(key)
            else:
                self.photo_selection.add(key)
        else:
            self.photo_selection = {key}
        self.repaint()

    def _remove_paths(self, paths: list[Path]) -> None:
        """Leave these photos out of training. The files on disk stay."""
        for path in paths:
            self.draft.remove_photo(path)
            self.photo_selection.discard(str(path))
        self.repaint()

    def _remove_selected(self) -> None:
        """Leave the selected photos out of training."""
        chosen = [path for path in self.draft.photos if str(path) in self.photo_selection]
        self._remove_paths(chosen)

    def choose_photos(self) -> None:
        """Pick photo files."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.window, tr("forge.app.photos.add_files"), "", tr("forge.app.photos.filter")
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
        self._arm_progress()
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
        if self.runner.busy and read_progress(self.project) is not None:  # type: ignore[attr-defined]
            self._apply_train_progress()
            return
        self._paint_progress()

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

    def make_round(self) -> None:
        """``lora refset round``: check the references (first time), draw candidates, check each one."""
        if not self._needs("refset"):
            return
        values = self.refset_settings if self.draft.expert else {
            "count": refset.DEFAULT_CANDIDATES, "strength": refset.DEFAULT_STRENGTH, "steps": refset.DEFAULT_STEPS}
        args = engine.refset_round_args(self.draft, self.project, count=int(values["count"]),
                                        strength=float(values["strength"]), steps=int(values["steps"]))

        def done(outcome: engine.Outcome) -> None:
            self.draft.approved = []
            self.draft.prepared = None
            if outcome.ok:
                rounds = refset.load_state(self.project).get("rounds") or [{}]
                last = rounds[-1]
                self.progress_text = tr("lora.progress.refset_done", round=int(last.get("round", 0)) + 1,
                                        shown=len(last.get("candidates") or []))
                self.progress_percent = None

        self._run(args, done)

    def finish_set(self) -> None:
        """"This is it": hand the references and the kept pictures to the normal preparation."""
        if not self._needs("refset"):
            return

        def done(outcome: engine.Outcome) -> None:
            if outcome.ok:
                self.draft.approved = refset.approved_files(self.project)
                self.draft.prepared = None
                self.status = ("forge.app.refset.finished", {"count": len(self.draft.approved)}, False)

        self._run(engine.refset_finish_args(self.draft, self.project, self.simple.get("base", "")), done)

    def restart_set(self) -> None:
        """Start over with the same references (a blocked set stays blocked)."""
        if self.runner.busy:  # type: ignore[attr-defined]
            return
        if refset.restart(self.project):
            self.draft.approved = []
            self.draft.prepared = None
        self.repaint()

    def _train_args(self, *, resume: bool = False, dry_run: bool = False) -> list[str]:
        values: dict[str, object] = dict(self.expert_values())
        values.update({"lora.preset": self.simple["preset"], "lora.likeness": self.simple["likeness"],
                       "lora.vram": self.simple["vram"], "lora.base": self.simple["base"],
                       "lora.previews": self.previews})
        args = engine.train_args(self.project, values, extend_from=self.draft.extend_from, resume=resume,
                                 dry_run=dry_run, expert=self.draft.expert)
        planned = None if dry_run or self.draft.extend_from is not None else self._review_rounds()
        if planned is not None:
            count, samples = planned
            args += [
                "--review-rounds", str(count),
                "--samples-per-round", str(samples),
                "--first-round-steps", str(self.first_round_steps),
                "--first-round-samples", str(rounds.DEFAULT_FIRST_ROUND_SAMPLES),
                "--sample-steps", str(rounds.DEFAULT_SAMPLE_STEPS),
            ]
        elif not dry_run and self.draft.extend_from is None:
            # The CLI default is 3. An explicit 0 keeps a straight train when the switch is off.
            args += ["--review-rounds", "0"]
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
            if outcome.ok and result.get("state") == "review":
                self.draft.trained = False
                self.paused = False
            elif outcome.ok and result.get("state", "done") == "done":
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

    def _stop_clicked(self) -> None:
        """Ask, then stop. Versions saved so far stay in the project."""
        if not self._confirm_stop():
            return
        self.stop()

    def _confirm_stop(self) -> bool:
        """True when the user confirms ending the run."""
        box = _message_box(self.window, "forge.train.stop.ask")
        box.setWindowTitle(tr("forge.app.train.stop"))
        box.setText(tr("forge.app.train.stop.ask"))
        stop = box.addButton(tr("forge.app.train.stop.confirm"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(tr("forge.app.train.stop.cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is stop

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

    def export_lora(self) -> None:
        """Save the small ``.voxlora`` package. Source photos are not included."""
        if not self._needs("export"):
            return
        stem = package_stem(Project(self.project).read())
        path, _chosen = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            tr("forge.app.result.export_lora"),
            str(Path.home() / f"{stem}.voxlora"),
            tr("forge.app.result.export_filter"),
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".voxlora":
            target = Path(f"{target}.voxlora") if not target.suffix else target.with_suffix(".voxlora")

        def done(outcome: engine.Outcome) -> None:
            if outcome.ok:
                self.status = ("forge.app.result.exported", {"name": target.name}, False)

        self._run(engine.package_args(self.project, target), done)

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
        """Load the subject fields from ``project.json`` and continue on step 1.

        A project paused on a review opens on the training step with that review.
        """
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
        if self._review_waiting():
            self._restore_review(subject)
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
        return [(info.mdl_id, info.display_name, info.permissive) for info in bases.trainable()]

    def _base_options(self) -> list[tuple[str, str, str, str, bool]]:
        try:
            from voxprint_forge.lora import bases
        except ImportError:
            return []
        return [(i.mdl_id, i.display_name, i.repo, i.licence, i.permissive) for i in bases.trainable()]


def _button_row(back: QtWidgets.QPushButton | None) -> QtWidgets.QLayout | None:
    """The horizontal row that holds Back, which lives inside the wizard's vertical layout."""
    if back is None or back.parentWidget() is None or back.parentWidget().layout() is None:
        return None
    outer = back.parentWidget().layout()
    for index in range(outer.count()):
        nested = outer.itemAt(index).layout()
        if nested is not None and nested.indexOf(back) >= 0:
            return nested
    return None


def _file_drop(event: QtGui.QDragEnterEvent | QtGui.QDragMoveEvent | QtGui.QDropEvent) -> list[Path]:
    """Local paths from a file drag, accepting the event when it carries any."""
    if not event.mimeData().hasUrls():
        return []
    event.acceptProposedAction()
    return [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]


class _PhotoBoard(QtWidgets.QWidget):
    """Drop zone: a short hint while empty, otherwise one tile per photo, sized to the tiles."""

    def __init__(self, on_drop: Callable[[list[Path]], None], tiles: list[QtWidgets.QWidget]) -> None:
        super().__init__()
        self._on_drop = on_drop
        self.setObjectName("sourceDrop")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        if tiles:
            layout.addWidget(FlowRow("forge.photos.grid", tuple(tiles)))
            return
        self.setMinimumHeight(188)
        layout.addStretch(1)
        box = QtWidgets.QWidget()
        box.setObjectName("forge.photos.empty")
        box.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        inner = QtWidgets.QVBoxLayout(box)
        inner.setSpacing(6)
        icon = QtWidgets.QLabel()
        icon.setObjectName("forge.photos.empty_icon")
        icon.setPixmap(_drop_icon())
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title = QtWidgets.QLabel(tr("forge.app.photos.drop"))
        title.setObjectName("forge.photos.empty_title")
        title.setProperty("emptyTitle", "true")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        hint = QtWidgets.QLabel(tr("forge.app.photos.empty_hint"))
        hint.setObjectName("forge.photos.empty_hint")
        hint.setProperty("muted", "true")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        for label in (icon, title, hint):
            label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        inner.addWidget(icon)
        inner.addWidget(title)
        inner.addWidget(hint)
        layout.addWidget(box)
        layout.addStretch(1)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        """Accept file and folder drops."""
        _file_drop(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # noqa: N802 - Qt name
        """Keep accepting while the files move over the zone."""
        _file_drop(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802 - Qt name
        """Add the dropped files and folders."""
        paths = _file_drop(event)
        if paths:
            self._on_drop(paths)


class _PhotoTile(QtWidgets.QFrame):
    """One photo. A remove button sits on the thumbnail corner and shows on hover or focus."""

    def __init__(
        self,
        path: Path,
        selected: bool,
        on_select: Callable[[Path, QtCore.Qt.KeyboardModifier], None],
        on_remove: Callable[[list[Path]], None],
        on_drop: Callable[[list[Path]], None],
    ) -> None:
        super().__init__()
        self._path = path
        self._on_select = on_select
        self._on_drop = on_drop
        self.setObjectName("forge.photos.item")
        self.setProperty("path", str(path))
        self.setProperty("photoTile", "true")
        self.setProperty("selected", "true" if selected else "false")
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFixedWidth(104)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(4)
        host = QtWidgets.QWidget()
        host.setObjectName("forge.photos.thumb")
        host.setFixedSize(88, 88)
        thumb = QtWidgets.QLabel(host)
        thumb.setPixmap(_thumb(path).pixmap(88, 88))
        thumb.setGeometry(0, 0, 88, 88)
        thumb.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._remove = _icon_button(
            "forge.photos.remove", "forge.app.photos.remove", "lora.photos.remove.tip",
            lambda: on_remove([path]), "close", "photoRemove",
        )
        self._remove.setParent(host)
        self._remove.setProperty("path", str(path))
        self._remove.move(88 - 22 - 4, 4)
        self._remove.installEventFilter(self)
        self._show_remove(False)
        column.addWidget(host, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        name = QtWidgets.QLabel()
        name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        shown = QtGui.QFontMetrics(name.font()).elidedText(path.name, QtCore.Qt.TextElideMode.ElideMiddle, 90)
        name.setText(shown)
        name.setToolTip(path.name)
        name.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        column.addWidget(name)

    def _show_remove(self, shown: bool) -> None:
        """Show the corner button while the pointer or keyboard focus is on this photo."""
        self._remove.setIcon(_mark_icon("close") if shown else QtGui.QIcon())
        self._remove.setProperty("shown", "true" if shown else "false")
        self._remove.style().unpolish(self._remove)
        self._remove.style().polish(self._remove)

    def _remove_wanted(self) -> bool:
        return self.underMouse() or self.hasFocus() or self._remove.hasFocus()

    def enterEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        self._show_remove(True)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        self._show_remove(self._remove_wanted())
        super().leaveEvent(event)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:  # noqa: N802 - Qt override
        self._show_remove(True)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # noqa: N802 - Qt override
        self._show_remove(self._remove_wanted())
        super().focusOutEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 - Qt override
        """Keep the corner button up while it holds keyboard focus."""
        if watched is self._remove and event.type() in (QtCore.QEvent.Type.FocusIn, QtCore.QEvent.Type.FocusOut):
            self._show_remove(self._remove_wanted() or event.type() == QtCore.QEvent.Type.FocusIn)
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Select this photo. Control toggles it in the current selection."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._on_select(self._path, event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802 - Qt name
        """Accept a drop on top of a photo."""
        _file_drop(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:  # noqa: N802 - Qt name
        """Keep accepting while the files move over a photo."""
        _file_drop(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802 - Qt name
        """Add files dropped on a photo."""
        paths = _file_drop(event)
        if paths:
            self._on_drop(paths)


def _icon_button(
    name: str, label: str, tip: str, slot: Callable[[], None], glyph: str, role: str,
) -> QtWidgets.QToolButton:
    """A small icon button with a tooltip and an accessible name, and no visible text."""
    button = QtWidgets.QToolButton()
    button.setObjectName(name)
    button.setProperty(role, "true")
    button.setIcon(_mark_icon(glyph))
    button.setIconSize(QtCore.QSize(14, 14))
    button.setFixedSize(22, 22)
    button.setAutoRaise(True)
    button.setToolTip(tr(tip))
    button.setAccessibleName(tr(label))
    button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    button.clicked.connect(lambda _checked=False: slot())
    return button


def _mark_icon(kind: str) -> QtGui.QIcon:
    """A close mark or a trash can, drawn so it shows without a theme icon."""
    image = QtGui.QImage(16, 16, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor("#f4f1ea"))
    pen.setWidthF(1.6)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    if kind == "trash":
        painter.drawLine(3, 4, 13, 4)
        painter.drawLine(6, 4, 6, 2)
        painter.drawLine(6, 2, 10, 2)
        painter.drawLine(10, 2, 10, 4)
        painter.drawRoundedRect(QtCore.QRectF(4, 6, 8, 8), 1.2, 1.2)
        painter.drawLine(7, 8, 7, 12)
        painter.drawLine(9, 8, 9, 12)
    elif kind == "close":
        painter.drawLine(4, 4, 12, 12)
        painter.drawLine(12, 4, 4, 12)
    else:
        raise ValueError(kind)
    painter.end()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image))


def _drop_icon() -> QtGui.QPixmap:
    """A small picture mark for the empty drop zone."""
    image = QtGui.QImage(36, 36, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor("#9a9aa8"))
    pen.setWidthF(1.6)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QtCore.QRectF(4, 7, 28, 22), 3, 3)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor("#9a9aa8"))
    painter.drawEllipse(QtCore.QRectF(9, 11, 5, 5))
    path = QtGui.QPainterPath()
    path.moveTo(8, 25)
    path.lineTo(15, 17)
    path.lineTo(20, 22)
    path.lineTo(24, 18)
    path.lineTo(30, 25)
    path.closeSubpath()
    painter.drawPath(path)
    painter.end()
    return QtGui.QPixmap.fromImage(image)


class _Switch(QtWidgets.QCheckBox):
    """A pill switch with the label to the right of the track."""

    _TRACK_W = 36
    _TRACK_H = 20

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setProperty("switch", "true")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        text = self.fontMetrics().horizontalAdvance(self.text())
        return QtCore.QSize(self._TRACK_W + 12 + text + 4, max(self._TRACK_H + 6, self.fontMetrics().height() + 6))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        return self.sizeHint()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Draw the track, the knob and the label. A focus ring follows keyboard focus."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = (self.height() - self._TRACK_H) / 2
        track = QtCore.QRectF(1, top, self._TRACK_W, self._TRACK_H)
        on = self.isChecked()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#f5d76e" if on else "#2a2a35"))
        painter.drawRoundedRect(track, 10, 10)
        knob = 14.0
        left = self._TRACK_W - 3 - knob if on else 4
        painter.setBrush(QtGui.QColor("#1c1604" if on else "#e8e6e3"))
        painter.drawEllipse(QtCore.QRectF(left, top + 3, knob, knob))
        if self.hasFocus():
            pen = QtGui.QPen(QtGui.QColor("#f5d76e"))
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(track.adjusted(-1, -1, 1, 1), 11, 11)
        painter.setPen(QtGui.QColor("#e8e6e3"))
        painter.drawText(
            QtCore.QRectF(self._TRACK_W + 12, 0, self.width() - self._TRACK_W - 12, self.height()),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        painter.end()


class _ClickLabel(QtWidgets.QLabel):
    """A candidate picture: a click on it flips the checkmark toggle laid over it."""

    def __init__(self, toggle: QtWidgets.QAbstractButton) -> None:
        super().__init__()
        self._toggle = toggle
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._toggle.isEnabled():
            self._toggle.click()
            event.accept()
            return
        super().mousePressEvent(event)


def _page_color() -> QtGui.QColor:
    """The wizard card behind the continue button, so the rounded corners match it."""
    try:
        return QtGui.QColor(str(load_tokens().colors["CARD_PLAIN"]))
    except Exception:  # noqa: BLE001 - a missing token still paints the card colour
        return QtGui.QColor("#202029")


def _mix(start: QtGui.QColor, end: QtGui.QColor, t: float) -> QtGui.QColor:
    """Linear blend of two colours. ``t`` is 0 at ``start`` and 1 at ``end``."""
    clamped = max(0.0, min(1.0, t))
    return QtGui.QColor(
        int(start.red() + (end.red() - start.red()) * clamped),
        int(start.green() + (end.green() - start.green()) * clamped),
        int(start.blue() + (end.blue() - start.blue()) * clamped),
    )


class _FlatButton(QtWidgets.QPushButton):
    """Focusable hit target. The parent paints the face."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.setObjectName(name)
        self.setFlat(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAutoDefault(False)
        self.setStyleSheet(
            "QPushButton, QPushButton:hover, QPushButton:pressed, QPushButton:focus, QPushButton:disabled {"
            " background: transparent; border: none; color: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Draw nothing, so the parent's crossfade stays visible."""
        del event

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:  # noqa: N802 - Qt override
        """Ask the parent to draw the focus mark."""
        super().focusInEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            parent.update()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # noqa: N802 - Qt override
        """Clear the parent's focus mark."""
        super().focusOutEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            parent.update()


class _ContinueMain(_FlatButton):
    """The wide face. Enter starts the current mode."""

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override
        """Enter starts. Other keys keep the button's own behaviour."""
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)


class _ContinueChevron(_FlatButton):
    """The narrow ▾ edge. Arrow keys swap the mode."""

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override
        """Arrow keys toggle. Enter does not start the run."""
        if event.key() in (
            QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down,
        ):
            self.click()
            event.accept()
            return
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)


class _ContinueToggle(QtWidgets.QWidget):
    """One primary button: the label and colour swap when ▾ is pressed."""

    toggled = QtCore.Signal(str)
    activated = QtCore.Signal()

    def __init__(self, reinforced: str, plain: str, reinforced_tip: str, plain_tip: str, mode: str) -> None:
        super().__init__()
        self.setObjectName("forge.review.continue.host")
        self.setAutoFillBackground(False)
        self._reinforced = reinforced
        self._plain = plain
        self._reinforced_tip = reinforced_tip
        self._plain_tip = plain_tip
        self._mode = "plain" if mode == "plain" else "reinforced"
        self._blend = 1.0 if self._mode == "plain" else 0.0
        self._main = _ContinueMain("forge.review.continue")
        self._chevron = _ContinueChevron("forge.review.continue.chevron")
        self._main.setParent(self)
        self._chevron.setParent(self)
        self._chevron.setAccessibleName("▾")
        self._apply_text()
        self._main.clicked.connect(lambda _checked=False: self.activated.emit())
        self._chevron.clicked.connect(lambda _checked=False: self.toggle())
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(_FADE_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self._anim.valueChanged.connect(self._on_blend)
        self._fit()

    def mode(self) -> str:
        """``reinforced`` or ``plain``."""
        return self._mode

    def text(self) -> str:
        """The label Enter would start."""
        return self._main.text()

    def toggle(self) -> None:
        """Swap reinforcement and plain continue, then crossfade the label."""
        self.set_mode("reinforced" if self._mode == "plain" else "plain")

    def set_mode(self, mode: str) -> None:
        """Show ``reinforced`` or ``plain``. An unknown value stays on reinforcement."""
        target = "plain" if mode == "plain" else "reinforced"
        changed = target != self._mode
        self._mode = target
        self._apply_text()
        end = 1.0 if target == "plain" else 0.0
        self._anim.stop()
        if changed and abs(self._blend - end) > 0.001:
            self._anim.setStartValue(float(self._blend))
            self._anim.setEndValue(end)
            self._anim.start()
        else:
            self._blend = end
            self.update()
        if changed:
            self.toggled.emit(target)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        """The longer label, the chevron, and the padding. Both modes share it."""
        return QtCore.QSize(self._needed_width(), self._needed_height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        """Do not let the row shrink the label."""
        return self.sizeHint()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 - Qt override
        """Measure again once the shell font is applied."""
        super().showEvent(event)
        self._fit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt override
        """Keep the ▾ segment on the right edge."""
        super().resizeEvent(event)
        self._place()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        """Repaint when the row is enabled or disabled."""
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.Type.EnabledChange,
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        ):
            self._fit()
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Gold or a calmer sand, with both labels crossfading."""
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _page_color())
        blend = self._blend
        if self.isEnabled():
            fill = _mix(QtGui.QColor("#f5d76e"), QtGui.QColor("#d2c4a4"), blend)
            border = _mix(QtGui.QColor("#fbe8a6"), QtGui.QColor("#e6dcc4"), blend)
            ink = _mix(QtGui.QColor("#1c1604"), QtGui.QColor("#2a2416"), blend)
            rule = _mix(QtGui.QColor("#c4a84a"), QtGui.QColor("#a89878"), blend)
        else:
            fill = QtGui.QColor("#4a4430")
            border = fill
            ink = QtGui.QColor("#6e6e7e")
            rule = fill
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, _ROW_RADIUS, _ROW_RADIUS)
        chevron = self._chevron.geometry()
        painter.setPen(QtGui.QPen(rule, 1))
        painter.drawLine(chevron.left(), 6, chevron.left(), max(6, self.height() - 7))
        painter.setPen(ink)
        painter.setFont(self.font())
        painter.drawText(chevron, int(QtCore.Qt.AlignmentFlag.AlignCenter), "▾")
        label_right = chevron.left() - _LABEL_GAP
        label = QtCore.QRectF(_LABEL_PAD, 0, max(0.0, label_right - _LABEL_PAD), float(self.height()))
        align = int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        painter.setOpacity(max(0.0, 1.0 - blend))
        painter.drawText(label, align, self._reinforced)
        painter.setOpacity(max(0.0, blend))
        painter.drawText(label, align, self._plain)
        painter.setOpacity(1.0)
        if self._chevron.hasFocus():
            pen = QtGui.QPen(ink)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QtCore.QRectF(chevron.adjusted(3, 4, -4, -5)), 2, 2)
        painter.end()

    def _apply_text(self) -> None:
        if self._mode == "plain":
            self._main.setText(self._plain)
            self._main.setToolTip(self._plain_tip)
            self._chevron.setToolTip(self._reinforced_tip)
            return
        self._main.setText(self._reinforced)
        self._main.setToolTip(self._reinforced_tip)
        self._chevron.setToolTip(self._plain_tip)

    def _text_width(self) -> int:
        font = self.font()
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        metrics = QtGui.QFontMetrics(font)
        advance = max(metrics.horizontalAdvance(self._reinforced), metrics.horizontalAdvance(self._plain))
        ink = max(metrics.boundingRect(self._reinforced).width(), metrics.boundingRect(self._plain).width())
        return max(advance, ink) + 4

    def _needed_width(self) -> int:
        return self._text_width() + _LABEL_PAD + _LABEL_GAP + _CHEVRON_W

    def _needed_height(self) -> int:
        font = self.font()
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        return max(40, QtGui.QFontMetrics(font).height() + 18)

    def _fit(self) -> None:
        if not hasattr(self, "_main"):
            return
        font = self.font()
        if font.weight() != QtGui.QFont.Weight.DemiBold:
            font.setWeight(QtGui.QFont.Weight.DemiBold)
            self.setFont(font)
            self._main.setFont(font)
        needed = QtCore.QSize(self._needed_width(), self._needed_height())
        if self.minimumSize() != needed:
            self.setMinimumSize(needed)
            self.updateGeometry()
        if self.width() < needed.width() or self.height() < needed.height():
            self.resize(max(self.width(), needed.width()), max(self.height(), needed.height()))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self._place()

    def _place(self) -> None:
        width = self.width()
        height = self.height()
        chevron = min(_CHEVRON_W, width)
        self._chevron.setGeometry(width - chevron, 0, chevron, height)
        self._main.setGeometry(0, 0, max(0, width - chevron), height)

    def _on_blend(self, value: float) -> None:
        self._blend = float(value)
        self.update()


def _style_menu(menu: QtWidgets.QMenu) -> None:
    """Dark popup. A menu is top-level and does not inherit the window sheet."""
    menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
    menu.setStyleSheet(_MENU_STYLE)
    palette = menu.palette()
    ground = QtGui.QColor("#2a2a33")
    ink = QtGui.QColor("#f4f1ea")
    palette.setColor(QtGui.QPalette.ColorRole.Window, ground)
    palette.setColor(QtGui.QPalette.ColorRole.Base, ground)
    palette.setColor(QtGui.QPalette.ColorRole.Button, ground)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, ink)
    palette.setColor(QtGui.QPalette.ColorRole.Text, ink)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, ink)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#3a3a46"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, ink)
    menu.setPalette(palette)


def _menu(parent: QtWidgets.QWidget, name: str) -> QtWidgets.QMenu:
    """A dark menu owned by ``parent``."""
    menu = QtWidgets.QMenu(parent)
    menu.setObjectName(name)
    _style_menu(menu)
    return menu


_ATTIRE_DROP_W = 22
_ATTIRE_TEXT_PAD = 8
_ATTIRE_ARROW = Path(__file__).resolve().parent / "review-chevron.svg"
_ATTIRE_COMBO_STYLE = """
QComboBox {
    background: #2a2a33; color: #f4f1ea; border: 1px solid #5c5c6c; border-radius: 4px;
    padding: 2px 4px 2px 8px; min-height: 28px; max-height: 28px;
}
QComboBox:hover { border-color: #c8c8d4; }
QComboBox:disabled { background: #22222b; color: #6e6e7e; border-color: #33333f; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none;
    border-top-right-radius: 4px; border-bottom-right-radius: 4px;
}
QComboBox::down-arrow { image: url(@@ARROW@@); width: 12px; height: 8px; }
QComboBox QAbstractItemView {
    background: #1c1c24; color: #f4f1ea; border: 1px solid #c8c8d4; border-radius: 4px;
    selection-background-color: #3a3a46; selection-color: #f4f1ea; outline: 0; padding: 2px;
}
"""
_ATTIRE_POPUP_STYLE = """
QFrame { background: #1c1c24; color: #f4f1ea; border: 1px solid #c8c8d4; border-radius: 4px; }
"""
_ATTIRE_VIEW_STYLE = """
QAbstractItemView {
    background: #1c1c24; color: #f4f1ea; border: none;
    selection-background-color: #3a3a46; selection-color: #f4f1ea; outline: 0; padding: 2px;
}
"""


def _attire_combo_style() -> str:
    """Dark combo face. The ▾ is the down-arrow image."""
    return _ATTIRE_COMBO_STYLE.replace("@@ARROW@@", _ATTIRE_ARROW.as_posix())


class _ElidedLabel(QtWidgets.QLabel):
    """A one-line label that gives way to its neighbours and ends with an ellipsis when short of room."""

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self._full = ""
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = text
        self._refit()

    def text(self) -> str:
        return self._full

    def shown_text(self) -> str:
        return super().text()

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(self.fontMetrics().horizontalAdvance(self._full) + 2, hint.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refit()

    def _refit(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight, max(0, self.width()))
        super().setText(shown)


class _FullWidthLabel(QtWidgets.QLabel):
    """A label that never gets narrower than its text in the font it ends up with."""

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        return QtCore.QSize(max(hint.width(), self.fontMetrics().horizontalAdvance(self.text()) + 2), hint.height())

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        return QtCore.QSize(max(hint.width(), self.minimumSizeHint().width()), hint.height())

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.FontChange, QtCore.QEvent.Type.StyleChange):
            self.updateGeometry()


class _AttireCombo(QtWidgets.QComboBox):
    """Dark attire list. The popup opens under the box."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("forge.review.attire")
        self.setStyleSheet(_attire_combo_style())
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setFixedHeight(28)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.view().setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 - Qt override
        """Size the box to its longest item once the font is real."""
        super().showEvent(event)
        self.fit_to_items()

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802 - Qt override
        """Measure again when the font changes."""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self.fit_to_items()

    def fit_to_items(self) -> None:
        """Minimum width is the longest item, plus the arrow and the padding."""
        self.ensurePolished()
        metrics = self.fontMetrics()
        widest = 0
        for index in range(self.count()):
            text = self.itemText(index)
            widest = max(widest, metrics.horizontalAdvance(text), metrics.boundingRect(text).width())
        floor = widest + _ATTIRE_TEXT_PAD + _ATTIRE_TEXT_PAD + _ATTIRE_DROP_W
        self.setMinimumWidth(floor)
        self.updateGeometry()

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        """Open the dark list directly under the box, sharing its left edge."""
        view = self.view()
        view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        super().showPopup()
        popup = view.window()
        if popup is None or popup is self.window():
            return
        popup.setStyleSheet(_ATTIRE_POPUP_STYLE)
        view.setStyleSheet(_ATTIRE_VIEW_STYLE)
        self._place_popup(popup)

    def _place_popup(self, popup: QtWidgets.QWidget) -> None:
        """Put the list under the box. The extra scroll buttons stay hidden."""
        view = self.view()
        for child in popup.findChildren(QtWidgets.QWidget):
            if child.metaObject().className() == "QComboBoxPrivateScroller":
                child.hide()
                child.setFixedHeight(0)
        layout = popup.layout()
        if layout is not None:
            layout.setContentsMargins(1, 1, 1, 1)
        metrics = self.fontMetrics()
        widest = 0
        for index in range(self.count()):
            widest = max(widest, metrics.horizontalAdvance(self.itemText(index)))
        width = max(self.width(), widest + _ATTIRE_TEXT_PAD * 2)
        row = view.sizeHintForRow(0) if self.count() else metrics.height() + 6
        height = row * max(1, self.count()) + 2
        popup.setFixedSize(width, height)
        view.setFixedSize(max(1, width - 2), max(1, height - 2))
        origin = self.mapToGlobal(QtCore.QPoint(0, 0))
        left = origin.x()
        top = origin.y() + self.height()
        host = self.window()
        if host is not None:
            frame = QtCore.QRect(host.mapToGlobal(QtCore.QPoint(0, 0)), host.size())
            left = min(max(left, frame.left()), max(frame.left(), frame.right() - popup.width() + 1))
        popup.move(left, top)


class _FinishMenu(QtWidgets.QMenu):
    """Dark finish menu. The panel is drawn here so a grab shows the border, radius and shadow."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("forgeReviewMenu")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowFlag(QtCore.Qt.WindowType.NoDropShadowWindowHint, True)
        self.setProperty("forgeShadowBelow", _FINISH_SHADOW)

    def shadow_below(self) -> int:
        """Pixels of shadow under the panel. Zero when the menu opens above the button."""
        value = self.property("forgeShadowBelow")
        return value if isinstance(value, int) else 0

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Fill, light border, shadow under the panel, and the two actions."""
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        font = self.font()
        # Subpixel antialiasing turns the danger pink into orange and cyan fringes.
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.NoSubpixelAntialias)
        painter.setFont(font)
        shadow = self.shadow_below()
        panel = QtCore.QRectF(0, 0, self.width(), max(1, self.height() - shadow))
        if shadow:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for step in range(shadow, 0, -1):
                painter.setBrush(QtGui.QColor(0, 0, 0, 16))
                painter.drawRoundedRect(panel.adjusted(1, 2, -1, step), _FINISH_RADIUS, _FINISH_RADIUS)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(_FINISH_BORDER))
        painter.drawRoundedRect(panel, _FINISH_RADIUS, _FINISH_RADIUS)
        painter.setBrush(QtGui.QColor(_FINISH_PANEL))
        painter.drawRoundedRect(panel.adjusted(1, 1, -1, -1), max(0, _FINISH_RADIUS - 1), max(0, _FINISH_RADIUS - 1))
        for action in self.actions():
            if action.isSeparator() or not action.isVisible():
                continue
            rect = self.actionGeometry(action)
            if rect.isEmpty():
                continue
            if self.activeAction() is action:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor("#3a3a46"))
                painter.drawRect(rect.adjusted(2, 1, -2, -1))
            color = _DANGER_TEXT if action.objectName() == "forge.review.finish.discard" else _FINISH_INK
            painter.setPen(QtGui.QColor(color))
            flags = QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft
            painter.drawText(rect.adjusted(16, 0, -8, 0), flags, action.text())
        painter.end()


class _FinishButton(QtWidgets.QPushButton):
    """Finish opens its menu from the button's right edge, inside the window."""

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Open the menu aligned to this button instead of past the window edge."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.menu() is not None:
            popup_menu(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt override
        """Space, Enter and Down open the same aligned menu."""
        if self.menu() is not None and event.key() in (
            QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Down,
        ):
            popup_menu(self)
            event.accept()
            return
        super().keyPressEvent(event)


def _menu_width(menu: QtWidgets.QMenu) -> int:
    metrics = menu.fontMetrics()
    widest = 0
    for action in menu.actions():
        if not action.isSeparator():
            widest = max(widest, metrics.horizontalAdvance(action.text()))
    return widest + 48


def _menu_panel_size(menu: QtWidgets.QMenu) -> tuple[int, int]:
    """Width and height of the menu panel, without the shadow strip under a finish menu."""
    menu.setMinimumWidth(_menu_width(menu))
    if not menu.isVisible():
        menu.adjustSize()
    hint = menu.sizeHint()
    return max(hint.width(), menu.minimumWidth()), max(hint.height(), 1)


def menu_popup_point(button: QtWidgets.QAbstractButton) -> QtCore.QPoint:
    """Global top-left: the menu's right edge meets the button's, and it hangs just below.

    It flips above the button only when that panel would pass the bottom of the window.
    The anchor is the window. The offscreen screen is smaller than a 1280 px window, and
    clamping to the screen leaves the menu in the middle, over the help text.
    """
    menu = button.menu()
    if menu is None:
        return button.mapToGlobal(QtCore.QPoint(0, button.height()))
    if isinstance(menu, _FinishMenu):
        menu.setProperty("forgeShadowBelow", _FINISH_SHADOW)
    width, panel = _menu_panel_size(menu)
    shadow = menu.shadow_below() if isinstance(menu, _FinishMenu) else 0
    height = panel + shadow
    right = button.mapToGlobal(QtCore.QPoint(button.width(), 0)).x()
    bottom = button.mapToGlobal(QtCore.QPoint(0, button.height())).y()
    top = button.mapToGlobal(QtCore.QPoint(0, 0)).y()
    x = right - width
    y = bottom
    window = button.window()
    if window is not None:
        bounds = QtCore.QRect(window.mapToGlobal(QtCore.QPoint(0, 0)), window.size())
        if y + height - 1 > bounds.bottom():
            if isinstance(menu, _FinishMenu):
                menu.setProperty("forgeShadowBelow", 0)
            y = top - panel
    return QtCore.QPoint(x, y)


def popup_menu(button: QtWidgets.QAbstractButton) -> None:
    """Show ``button``'s menu anchored to its right edge."""
    menu = button.menu()
    if menu is None:
        return
    menu.popup(menu_popup_point(button))
    # QMenu::popup() pulls the menu onto the screen. Move it back onto the button.
    point = menu_popup_point(button)
    width, panel = _menu_panel_size(menu)
    shadow = menu.shadow_below() if isinstance(menu, _FinishMenu) else 0
    menu.resize(max(width, menu.width()), panel + shadow)
    right = button.mapToGlobal(QtCore.QPoint(button.width(), 0)).x()
    menu.move(right - menu.width(), point.y())
    menu.setActiveAction(None)


class _DarkDialog(QtWidgets.QDialog):
    """A dark modal with a title, a border and a shadow. The frame is drawn, not the platform's."""

    def __init__(self, parent: QtWidgets.QWidget, name: str) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._clicked: QtWidgets.QPushButton | None = None
        self._roles: dict[QtWidgets.QPushButton, QtWidgets.QMessageBox.ButtonRole] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        card = QtWidgets.QFrame()
        card.setObjectName("forgeDialogCard")
        body = QtWidgets.QVBoxLayout(card)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        self._title = QtWidgets.QLabel()
        self._title.setObjectName("forgeDialogTitle")
        self._title.setWordWrap(True)
        self._text = QtWidgets.QLabel()
        self._text.setObjectName("qt_msgbox_label")
        self._text.setWordWrap(True)
        self._text.setMinimumWidth(420)
        self._row = QtWidgets.QHBoxLayout()
        self._row.addStretch(1)
        self._button_list: list[QtWidgets.QPushButton] = []
        body.addWidget(self._title)
        body.addWidget(self._text)
        body.addLayout(self._row)
        outer.addWidget(card)
        self.setStyleSheet(_DIALOG_STYLE)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Draw the shadow in the margin so a grab shows it. Effects do not."""
        del event
        card = self.findChild(QtWidgets.QFrame, "forgeDialogCard")
        if card is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        rect = QtCore.QRectF(card.geometry())
        for step in range(14, 0, -1):
            painter.setBrush(QtGui.QColor(0, 0, 0, 16))
            painter.drawRoundedRect(
                rect.adjusted(-step * 0.4, 1 + step * 0.2, step * 0.4, 3 + step * 0.7), 16, 16,
            )
        painter.end()

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt override
        """Show the title inside the card. A frameless window has no title bar."""
        super().setWindowTitle(title)
        self._title.setText(title)

    def setText(self, text: str) -> None:
        """The sentence under the title."""
        self._text.setText(text)

    def text(self) -> str:
        """The sentence under the title."""
        return self._text.text()

    def addButton(self, label: str, role: QtWidgets.QMessageBox.ButtonRole) -> QtWidgets.QPushButton:
        """A button on the card. The discard confirmation's accept button is danger-coloured."""
        button = QtWidgets.QPushButton(label)
        button.setAutoDefault(False)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if role == QtWidgets.QMessageBox.ButtonRole.AcceptRole and self.objectName().endswith("discard.confirm"):
            button.setProperty("danger", "true")
            button.style().unpolish(button)
            button.style().polish(button)
        button.clicked.connect(lambda _checked=False, chosen=button: self._choose(chosen))
        self._roles[button] = role
        self._button_list.append(button)
        self._row.addWidget(button)
        return button

    def buttons(self) -> list[QtWidgets.QPushButton]:
        """The buttons on the card, in the order they were added."""
        return list(self._button_list)

    def clickedButton(self) -> QtWidgets.QPushButton | None:
        """The button that closed the dialog, or none when it was dismissed."""
        return self._clicked

    def setDefaultButton(self, button: QtWidgets.QPushButton) -> None:
        """The button Enter activates."""
        button.setDefault(True)
        button.setAutoDefault(True)

    def exec(self) -> int:  # noqa: N802 - Qt override
        """Centre the card on the parent window, then run it."""
        self.adjustSize()
        window = self.parentWidget().window() if self.parentWidget() is not None else None
        if window is not None:
            frame = window.frameGeometry()
            self.move(frame.center().x() - self.width() // 2, frame.center().y() - self.height() // 2)
        return super().exec()

    def _choose(self, button: QtWidgets.QPushButton) -> None:
        self._clicked = button
        if self._roles.get(button) == QtWidgets.QMessageBox.ButtonRole.AcceptRole:
            self.accept()
            return
        self.reject()


def _message_box(parent: QtWidgets.QWidget, name: str) -> _DarkDialog:
    """A dark modal: title, border and shadow, on the app theme."""
    return _DarkDialog(parent, name)


def _button(label: str, tip: str, slot: Callable[[], None], *, primary: bool = False) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(tr(label))
    button.setObjectName("primary" if primary else label)
    button.setToolTip(tr(tip))
    button.setAutoDefault(False)
    button.clicked.connect(lambda _checked=False: slot())
    return button


def _flow_row(name: str, *widgets: QtWidgets.QWidget) -> FlowRow:
    """Widgets side by side that wrap onto the next line in a narrow window."""
    return FlowRow(name, widgets)


def _edit(text: str, tip: str, name: str) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(text)
    edit.setObjectName(name)
    edit.setToolTip(tr(tip))
    return edit


def _attr_edit(draft: Draft, on_change: Callable[[], None]) -> QtWidgets.QLineEdit:
    """Optional whole number, enabled only while 18 or older is ticked."""
    edit = QtWidgets.QLineEdit()
    edit.setObjectName("forge.subject.attr")
    edit.setToolTip(tr("forge.app.subject.attr.tip"))
    edit.setPlaceholderText("18-99")
    edit.setValidator(QtGui.QIntValidator(LOW, HIGH, edit))
    edit.setEnabled(draft.adult)
    if draft.adult and accepted(draft.attr) is not None:
        edit.setText(str(draft.attr))

    def changed(text: str) -> None:
        stripped = text.strip()
        if not stripped:
            draft.attr = None
        else:
            try:
                draft.attr = int(stripped)
            except ValueError:
                draft.attr = None
        on_change()

    edit.textChanged.connect(changed)
    return edit


def _thumb(path: Path, size: int = 72) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(str(path))
    if pixmap.isNull():
        return QtGui.QIcon()
    return QtGui.QIcon(pixmap.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                     QtCore.Qt.TransformationMode.SmoothTransformation))


def _failure(outcome: engine.Outcome) -> tuple[str, dict[str, object], bool]:
    code = outcome.error_code
    if code == "lora.err.gate_missing":
        ids = missing_id_text()
        if not ids:
            return "lora.err.not_prepared", {}, True
        return code, {"ids": ids}, True
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
