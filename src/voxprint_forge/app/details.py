"""A minimal "Details" / «Подробнее» expander: one disclosure button and a body that starts collapsed.

voxprint-common 0.2.5 has no expander widget, only the ``QPushButton#disclosure``
style (used by Manage models for its own details). This reuses that style.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6 import QtCore, QtWidgets
from voxprint_common.i18n import tr


class Details(QtWidgets.QWidget):
    """``Details ▸`` that shows short paragraphs when opened. Collapsed unless ``opened``."""

    def __init__(
        self,
        name: str,
        paragraphs: Sequence[str],
        *,
        opened: bool = False,
        on_toggle: Callable[[bool], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(name)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.button = QtWidgets.QPushButton()
        self.button.setObjectName("disclosure")
        self.button.setCheckable(True)
        self.button.setToolTip(tr("forge.app.details.tip"))
        self.button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        layout.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.body = QtWidgets.QFrame()
        self.body.setObjectName("card")
        inner = QtWidgets.QVBoxLayout(self.body)
        for text in paragraphs:
            label = QtWidgets.QLabel(text)
            label.setWordWrap(True)
            label.setObjectName("simpleNote")
            inner.addWidget(label)
        layout.addWidget(self.body)
        self._on_toggle = on_toggle
        self.button.toggled.connect(self._flip)
        self.button.setChecked(opened)
        self._flip(opened, notify=False)

    @property
    def opened(self) -> bool:
        """True while the body is shown."""
        return self.button.isChecked()

    def _flip(self, checked: bool, *, notify: bool = True) -> None:
        self.body.setVisible(checked)
        self.button.setText(tr("forge.app.details.hide" if checked else "forge.app.details"))
        self.updateGeometry()
        if notify and self._on_toggle is not None:
            self._on_toggle(checked)
        if notify and checked:
            # After the page has grown, scroll so the opened text is in view (not left under the footer).
            QtCore.QTimer.singleShot(0, self.reveal)

    def reveal(self) -> None:
        """Scroll the nearest scroll area so the button and the opened body are visible."""
        area = self.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if area is None or not self.isVisible():
            return
        area.ensureWidgetVisible(self.body if self.opened else self.button, 0, 8)
