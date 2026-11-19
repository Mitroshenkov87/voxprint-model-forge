"""A left-to-right row that wraps its widgets onto the next line when the window is narrow."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Places items in reading order and starts a new line when the next one does not fit."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, spacing: int = 16) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802 - Qt API
        """Qt hook: append an item."""
        self._items.append(item)

    def count(self) -> int:
        """Qt hook: number of items."""
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 - Qt API
        """Qt hook: item by index."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 - Qt API
        """Qt hook: remove an item."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802 - Qt API
        """Qt hook: the row never asks for extra space."""
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        """Qt hook: the height depends on the width (wrapping)."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        """Qt hook: height needed at ``width``."""
        return self._place(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802 - Qt API
        """Qt hook: lay the items out inside ``rect``."""
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        """Qt hook: everything on one line."""
        sizes = [_size(item) for item in self._visible()]
        width = sum(size.width() for size in sizes) + self._gap * max(0, len(sizes) - 1)
        height = max((size.height() for size in sizes), default=0)
        return QtCore.QSize(width, height)

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        """Qt hook: the widest single item, so no label is ever cut."""
        size = QtCore.QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize()).expandedTo(_size(item))
        return size

    def _visible(self) -> list[QtWidgets.QLayoutItem]:
        return [item for item in self._items if not (item.widget() is not None and item.widget().isHidden())]

    def _place(self, rect: QtCore.QRect, *, move: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._visible():
            hint = _size(item)
            if x > rect.x() and x + hint.width() > rect.right() + 1:
                x, y, line = rect.x(), y + line + self._gap // 2, 0
            if move:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._gap
            line = max(line, hint.height())
        return y + line - rect.y()


def _size(item: QtWidgets.QLayoutItem) -> QtCore.QSize:
    """The hinted size, made tall enough for wrapped text at the hinted width.

    A word-wrapped label hints a height for a width of its own choosing, so a tile placed at its
    plain hint can be a line short; how short depends on the platform's font metrics.
    """
    hint = item.sizeHint()
    if item.hasHeightForWidth():
        hint.setHeight(max(hint.height(), item.heightForWidth(hint.width())))
    return hint


class FlowRow(QtWidgets.QWidget):
    """A widget holding a :class:`FlowLayout`; it grows taller when its items wrap."""

    def __init__(self, name: str, widgets: tuple[QtWidgets.QWidget, ...], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(name)
        self.flow = FlowLayout(self)
        for widget in widgets:
            self.flow.addWidget(widget)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def relayout(self) -> None:
        """Fit the height to the current width (call after showing or hiding an item)."""
        height = self.flow.heightForWidth(max(1, self.width()))
        if height != self.minimumHeight():
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
        self.flow.invalidate()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Qt hook: keep the height in step with the width."""
        super().resizeEvent(event)
        self.relayout()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Qt hook: fit once the real width is known."""
        super().showEvent(event)
        self.relayout()
