"""The base-model picker on the Train step: a short name per option, details on a muted second line.

Only bases the engine can train on are offered (:func:`voxprint_forge.lora.bases.trainable`).
Video memory shown per base comes from measured runs when there are any, else
from the lowest tier the presets target (``VRAM_TIERS``), shown as "16 GB+".
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from voxprint_forge.app.text import tr
from voxprint_forge.lora.presets import VRAM_TIERS

# Peak training memory in GB per base and tier, from LORA-CALIB (``train16`` / ``train24`` peak_vram_gb).
# TODO(lora-calib): LORA-CALIB has not produced training numbers yet (the 17 Oct run stopped at prepare).
# Fill this from its summary.json, e.g. {"z-image": {16: <peak>, 24: <peak>}}; until then the list shows
# the preset tier ("16 GB+"), never a guessed figure, and Details and the tooltip have no memory line.
MEASURED_PEAK_GB: dict[str, dict[int, float]] = {}

DETAIL_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
MIN_WIDTH = 220  # px, the closed picker
MAX_WIDTH = 360
POPUP_MAX_WIDTH = 520  # px, the open list


def muted_color() -> QtGui.QColor:
    """The shell's muted text colour for the active theme (dark or light)."""
    try:
        from voxprint_common.ui.theme import resolved_shell_light
        from voxprint_common.ui.tokens import load

        tokens = load()
        colors = tokens.light_colors if resolved_shell_light() else tokens.colors
        return QtGui.QColor(colors["TEXT_MUTED"])
    except Exception:  # noqa: BLE001 - a missing token falls back to the palette
        return QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)


def vram_text(mdl_id: str) -> str:
    """``16 GB+`` (preset tier), or the measured peak once calibration has run."""
    measured = MEASURED_PEAK_GB.get(mdl_id) or {}
    if measured:
        return tr("forge.app.base.vram_measured", gb=f"{min(measured.values()):g}")
    return tr("forge.app.base.vram_tier", gb=min(VRAM_TIERS))


def memory_details() -> str | None:
    """The measured video memory sentence for Details, or None (no line at all) until calibration has run."""
    if not MEASURED_PEAK_GB:
        return None
    return tr("forge.app.base.vram_measured.tip", vram=", ".join(vram_text(mdl_id) for mdl_id in MEASURED_PEAK_GB))


def detail_line(mdl_id: str) -> str:
    """The muted second line: what the base is best for, then video memory."""
    key = f"forge.app.base.{mdl_id}.note"
    note = tr(key)
    vram = vram_text(mdl_id)
    return vram if note == key else f"{note} · {vram}"


def tooltip(mdl_id: str, name: str, repo: str, licence: str) -> str:
    """Full details for one option; the video memory sentence only when it was measured."""
    memory = tr("forge.app.base.vram_measured.tip", vram=vram_text(mdl_id)) if MEASURED_PEAK_GB.get(mdl_id) else ""
    return tr("forge.app.base.option.tip", name=name, repo=repo, licence=licence, detail=detail_line(mdl_id),
              memory=memory).rstrip()


class TwoLineDelegate(QtWidgets.QStyledItemDelegate):
    """Name on the first line, the detail line below it in a muted colour."""

    PAD = 6

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802 - Qt API
        """Qt hook: two text lines plus padding."""
        metrics = QtGui.QFontMetrics(option.font)
        small = QtGui.QFontMetrics(_small(option.font))
        width = max(metrics.horizontalAdvance(str(index.data() or "")),
                    small.horizontalAdvance(str(index.data(DETAIL_ROLE) or ""))) + 4 * self.PAD
        return QtCore.QSize(min(width, POPUP_MAX_WIDTH), metrics.height() + small.height() + 3 * self.PAD)

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        """Qt hook: draw the item."""
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        selected = bool(option.state & QtWidgets.QStyle.StateFlag.State_Selected)
        rect = option.rect.adjusted(2 * self.PAD, self.PAD, -2 * self.PAD, -self.PAD)
        metrics = QtGui.QFontMetrics(option.font)
        painter.save()
        text_role = QtGui.QPalette.ColorRole.HighlightedText if selected else QtGui.QPalette.ColorRole.Text
        painter.setPen(option.palette.color(text_role))
        painter.setFont(option.font)
        name = metrics.elidedText(str(index.data() or ""), QtCore.Qt.TextElideMode.ElideRight, rect.width())
        painter.drawText(QtCore.QRect(rect.left(), rect.top(), rect.width(), metrics.height()),
                         int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter), name)
        small = _small(option.font)
        small_metrics = QtGui.QFontMetrics(small)
        muted = muted_color()
        painter.setPen(option.palette.color(text_role) if selected else muted)
        painter.setFont(small)
        detail = small_metrics.elidedText(str(index.data(DETAIL_ROLE) or ""), QtCore.Qt.TextElideMode.ElideRight,
                                          rect.width())
        top = rect.top() + metrics.height() + self.PAD // 2
        painter.drawText(QtCore.QRect(rect.left(), top, rect.width(), small_metrics.height()),
                         int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter), detail)
        painter.restore()


def _small(font: QtGui.QFont) -> QtGui.QFont:
    small = QtGui.QFont(font)
    size = font.pointSizeF()
    if size > 0:
        small.setPointSizeF(max(7.0, size * 0.88))
    else:
        small.setPixelSize(max(9, round(font.pixelSize() * 0.88)))
    return small


class DownwardCombo(QtWidgets.QComboBox):
    """A combo whose list always opens below the field, left edges aligned.

    Some styles open the list over the field (on the current item), which hides
    the label and the controls above it.
    """

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        """Qt hook: open, then move the list under the field."""
        super().showPopup()
        self._place_popup()
        # Opening focuses the field and the step may scroll it into view; place the list again after that.
        QtCore.QTimer.singleShot(0, self._place_popup)

    def _place_popup(self) -> None:
        popup = self.view().window()
        if popup.isVisible():
            popup.move(self.mapToGlobal(QtCore.QPoint(0, self.height())))


def build_picker(options: list[tuple[str, str, str, str, bool]], current: str) -> QtWidgets.QComboBox:
    """A sane-width combo of ``(mdl_id, name, repo, licence, permissive)`` with two-line options."""
    combo = DownwardCombo()
    combo.setObjectName("forge.train.base")
    combo.setToolTip(tr("lora.base.tip"))
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setMinimumWidth(MIN_WIDTH)
    combo.setMaximumWidth(MAX_WIDTH)
    combo.setItemDelegate(TwoLineDelegate(combo))
    for mdl_id, name, repo, licence, permissive in options:
        combo.addItem(name, mdl_id)
        row = combo.count() - 1
        combo.setItemData(row, detail_line(mdl_id), DETAIL_ROLE)
        tip = tooltip(mdl_id, name, repo, licence)
        if not permissive:
            tip = f"{tip}\n{tr('lora.base.external.tip')}"
        combo.setItemData(row, tip, QtCore.Qt.ItemDataRole.ToolTipRole)
    view = combo.view()
    delegate = combo.itemDelegate()
    option = QtWidgets.QStyleOptionViewItem()
    option.font = combo.font()
    widest = max((delegate.sizeHint(option, combo.model().index(i, 0)).width() for i in range(combo.count())),
                 default=0)
    view.setMinimumWidth(min(POPUP_MAX_WIDTH, max(widest, combo.sizeHint().width())))
    view.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
    combo.setCurrentIndex(max(0, combo.findData(current)))
    return combo


def hint_label() -> QtWidgets.QLabel:
    """The muted line under the label: what a base is and that the LoRA only works with it."""
    hint = QtWidgets.QLabel(tr("forge.app.base.hint"))
    hint.setObjectName("forge.train.base.hint")
    hint.setWordWrap(True)
    hint.setStyleSheet(f"color: {muted_color().name()}; font-size: 12px; background: transparent;")
    return hint
