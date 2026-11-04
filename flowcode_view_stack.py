"""
flowcode_view_stack.py
----------------------
Result views chosen from a drop-down instead of a tab bar, for pages with
more views than a tab bar can show. ``ViewStack`` is the page stack; its
``selector`` (the drop-down with Previous and Next buttons) is placed by the
page wherever it fits, for example on a row of other choices.

The drop-down lists the views in order under headings (the *section_of*
callable gives each view's heading). Headings are greyed out and cannot be
chosen; the Previous and Next buttons and the arrow keys skip them.

``ViewStack`` exposes the part of the QTabWidget API the pages use
(addTab, insertTab, removeTab, widget, tabText, count, currentIndex,
setCurrentIndex, currentWidget, indexOf and currentChanged), so it replaces a
QTabWidget without other changes. Indices count views only, never headings.

Not named ``*_tab.py``, so the plugin loader does not load it as a tab.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QSizePolicy, QStackedWidget, QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget,
)

_VIEW_ROLE = Qt.UserRole + 1
_INDENT = '    '


class _IndentDelegate(QStyledItemDelegate):
    """Draws the drop-down's views indented under their headings, while
    the closed drop-down shows the plain name."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(_VIEW_ROLE) is not None and index.siblingAtRow(0).data(_VIEW_ROLE) is None:
            option.text = _INDENT + option.text


class ViewStack(QWidget):
    """A page stack whose page is chosen in ``selector``."""

    currentChanged = Signal(int)

    def __init__(self, section_of=None, parent=None):
        super().__init__(parent)
        self._section_of = section_of or (lambda _widget: '')
        self._pages: list[QWidget] = []
        self._titles: list[str] = []
        self._current = -1
        self._stack = QStackedWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._stack)

        self.combo = QComboBox()
        self.combo.setModel(QStandardItemModel(self.combo))
        self.combo.setItemDelegate(_IndentDelegate(self.combo))
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.setToolTip('The result shown below.')
        self.combo.currentIndexChanged.connect(self._on_row)
        self.btn_previous = QToolButton()
        self.btn_previous.setArrowType(Qt.LeftArrow)
        self.btn_previous.setToolTip('Previous view')
        self.btn_previous.clicked.connect(lambda: self._step(-1))
        self.btn_next = QToolButton()
        self.btn_next.setArrowType(Qt.RightArrow)
        self.btn_next.setToolTip('Next view')
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.selector = QWidget()
        row = QHBoxLayout(self.selector)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(self.combo, stretch=1)
        row.addWidget(self.btn_previous)
        row.addWidget(self.btn_next)
        self._update_buttons()

    # QTabWidget API ----------------------------------------------------------

    def addTab(self, widget: QWidget, title: str) -> int:
        return self.insertTab(len(self._pages), widget, title)

    def insertTab(self, index: int, widget: QWidget, title: str) -> int:
        index = max(0, min(int(index), len(self._pages)))
        shown = self.currentWidget()
        self._pages.insert(index, widget)
        self._titles.insert(index, str(title))
        self._stack.insertWidget(index, widget)
        if shown is None:
            self._set_current(index, rebuild=True)
        else:
            self._current = self._pages.index(shown)
            self._rebuild()
        return index

    def removeTab(self, index: int):
        if not 0 <= index < len(self._pages):
            return
        widget = self._pages.pop(index)
        self._titles.pop(index)
        self._stack.removeWidget(widget)
        if not self._pages:
            self._set_current(-1, rebuild=True)
        elif index == self._current:
            self._set_current(min(index, len(self._pages) - 1), rebuild=True)
        else:
            if index < self._current:
                self._current -= 1
            self._rebuild()

    def widget(self, index: int):
        return self._pages[index] if 0 <= index < len(self._pages) else None

    def tabText(self, index: int) -> str:
        return self._titles[index] if 0 <= index < len(self._titles) else ''

    def count(self) -> int:
        return len(self._pages)

    def currentIndex(self) -> int:
        return self._current

    def currentWidget(self):
        return self.widget(self._current)

    def indexOf(self, widget) -> int:
        return self._pages.index(widget) if widget in self._pages else -1

    def setCurrentIndex(self, index: int):
        if 0 <= index < len(self._pages) and index != self._current:
            self._set_current(index)

    # Drop-down -----------------------------------------------------------------

    def _set_current(self, index: int, rebuild: bool = False):
        changed = index != self._current
        self._current = index
        if index >= 0:
            self._stack.setCurrentWidget(self._pages[index])
        if rebuild:
            self._rebuild()
        else:
            self._select_row()
        if changed:
            self.currentChanged.emit(index)

    def _rebuild(self):
        """Refill the drop-down: the views in order, a heading before each
        run of views with the same section."""
        model = self.combo.model()
        self.combo.blockSignals(True)
        model.clear()
        heading_font = QFont(self.combo.font())
        heading_font.setBold(True)
        last = None
        for i, (widget, title) in enumerate(zip(self._pages, self._titles)):
            section = self._section_of(widget) or ''
            if section and section != last:
                heading = QStandardItem(section)
                heading.setFlags(Qt.NoItemFlags)
                heading.setFont(heading_font)
                model.appendRow(heading)
            last = section
            item = QStandardItem(title)
            item.setData(i, _VIEW_ROLE)
            model.appendRow(item)
        self.combo.blockSignals(False)
        self._select_row()

    def _row_of(self, index: int) -> int:
        model = self.combo.model()
        for row in range(model.rowCount()):
            if model.item(row).data(_VIEW_ROLE) == index:
                return row
        return -1

    def _select_row(self):
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(self._row_of(self._current))
        self.combo.blockSignals(False)
        self._update_buttons()

    def _on_row(self, row: int):
        index = self.combo.itemData(row, _VIEW_ROLE)
        if index is None:
            self._select_row()
            return
        self.setCurrentIndex(int(index))

    def _step(self, step: int):
        self.setCurrentIndex(self._current + step)

    def _update_buttons(self):
        self.btn_previous.setEnabled(self._current > 0)
        self.btn_next.setEnabled(0 <= self._current < len(self._pages) - 1)
