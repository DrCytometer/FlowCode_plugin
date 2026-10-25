"""
flowcode_figure_panel.py
------------------------
A matplotlib figure on a FlowCode page (section 4 figures, section 6
results): the main app's ScalableFigureWidget (Pop Out, Export Graphic and
a Size slider) with Export Graphic writing ``<folder>/<name>.<format>``, and
the pop-out window, which draws a fresh copy of the figure from its maker
and resizes it with the window (as Honeycluster's Pop Out does).

Message figures ("Run the characterisation…") are shown at their own size
without the controls.

``scrolling_help`` is the app's help toggle with its text in a scroll area,
for page heads that sit above tabs.

Not named ``*_tab.py``, so the plugin loader does not load it as a tab.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QSizePolicy, QVBoxLayout, QWidget

import honeychrome.settings as hc_settings
from honeychrome.view_components.scalable_figure_widget import (
    ScalableFigureWidget, initial_scale_for, scrollable_canvas_class,
)

_HINT_STYLE = 'color: grey; font-style: italic;'
HELP_HEIGHT_FRACTION = 0.45
HELP_MIN_HEIGHT = 160
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def scrolling_help(text: str) -> QWidget:
    """
    The app's help toggle (``HelpToggleWidget``) with its text in a scroll
    area: when shown, the text takes the height it needs up to
    ``HELP_HEIGHT_FRACTION`` of the window (at least ``HELP_MIN_HEIGHT``
    pixels) and scrolls beyond that, so long help is not cropped by the
    tabs below it.
    """
    from PySide6.QtWidgets import QFrame, QScrollArea

    from honeychrome.view_components.help_toggle_widget import HelpToggleWidget

    widget = HelpToggleWidget(text=text)
    label = widget.help_label
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(label)
    area.setVisible(False)
    widget.layout().addWidget(area)

    def toggled():
        shown = not label.isHidden()
        if shown:
            cap = max(HELP_MIN_HEIGHT, int(HELP_HEIGHT_FRACTION * widget.window().height()))
            width = max(200, widget.width() - area.verticalScrollBar().sizeHint().width() - 4)
            area.setFixedHeight(min(cap, label.heightForWidth(width) + 8))
        area.setVisible(shown)

    widget.help_button.clicked.connect(toggled)
    return widget


class _PanelFigure(ScalableFigureWidget):
    """ScalableFigureWidget whose Export Graphic is the panel's export."""

    def __init__(self, panel: 'FigurePanel', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panel = panel

    def export_graphic(self):
        self._panel.export()


def _connect_hover(canvas, fig):
    handler = getattr(fig, '_hover_handler', None)
    if handler is not None:
        canvas.mpl_connect('motion_notify_event', handler)


def open_figure_window(parent: QWidget, fig, title: str, connect=None) -> QDialog:
    """Show *fig* (a figure of its own) in a window with the matplotlib
    toolbar; the figure is redrawn at the window's size. *connect*, when
    given, is called with the window's canvas and the figure."""
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    width, height = fig.get_size_inches() * fig.get_dpi()
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    canvas = scrollable_canvas_class()(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    _connect_hover(canvas, fig)
    if connect is not None:
        connect(canvas, fig)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)
    layout.addWidget(NavigationToolbar2QT(canvas, dialog))
    layout.addWidget(canvas, stretch=1)
    screen = parent.screen() if parent is not None else None
    if screen is not None:
        area = screen.availableGeometry()
        max_w, max_h = area.width() - 40, area.height() - 120
    else:
        max_w, max_h = 1400, 1000
    dialog.resize(min(int(width), max_w) + 8, min(int(height), max_h) + 60)
    dialog.show()
    return dialog


class FigurePanel(QWidget):
    """
    One figure, replaced by ``set_figure``. *folder_fn* returns the folder
    Export Graphic writes to (created when missing); *name* is the file
    name without extension. With a *make* callable, Pop Out opens a window
    with the figure it returns. ``connect_canvas``, when set, is called with
    each canvas showing the figure (inline and popped out) and its figure,
    e.g. to connect mouse clicks.
    """

    def __init__(self, name: str, folder_fn, parent=None):
        super().__init__(parent)
        self.name = name
        self._folder_fn = folder_fn
        self.figure = None
        self.view = None
        self.exportable = False
        self._make = None
        self.connect_canvas = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.holder = QVBoxLayout()
        lay.addLayout(self.holder, stretch=1)
        self.export_msg = QLabel('')
        self.export_msg.setStyleSheet(_HINT_STYLE)
        self.export_msg.setWordWrap(True)
        lay.addWidget(self.export_msg)

    @property
    def canvas(self):
        return None if self.view is None else getattr(self.view, 'canvas', self.view)

    def set_figure(self, fig, exportable: bool = True, make=None, name: str | None = None):
        self._drop_view()
        self.figure = fig
        self.exportable = bool(exportable)
        self._make = make if exportable else None
        if name:
            self.name = name
        if exportable:
            self.view = _PanelFigure(self, fig, title=self.name, experiment_dir='',
                                     pop_out=self.pop_out if make is not None else None,
                                     initial_scale=initial_scale_for(fig), parent=self)
            self.holder.addWidget(self.view, stretch=1)
        else:
            self.view = scrollable_canvas_class()(fig)
            width, height = fig.get_size_inches() * fig.get_dpi()
            self.view.setFixedSize(int(round(width)), int(round(height)))
            self.holder.addWidget(self.view, alignment=Qt.AlignLeft | Qt.AlignTop)
            self.view.draw_idle()
        _connect_hover(self.canvas, fig)
        if self.connect_canvas is not None and self.exportable:
            self.connect_canvas(self.canvas, fig)
        self.export_msg.setText('')

    def clear(self):
        self._drop_view()
        self.figure = None
        self.exportable = False
        self._make = None
        self.export_msg.setText('')

    def _drop_view(self):
        if self.view is not None:
            self.holder.removeWidget(self.view)
            self.view.setParent(None)
            self.view.deleteLater()
            self.view = None

    def file_path(self) -> Path:
        stem = _INVALID_NAME_CHARS.sub('_', self.name).strip() or 'figure'
        return Path(self._folder_fn()) / f'{stem}.{hc_settings.graphics_export_format_retrieved}'

    def export(self):
        if self.figure is None or not self.exportable:
            return
        path = self.file_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(path, bbox_inches='tight', facecolor=self.figure.get_facecolor())
        except (OSError, ValueError) as exc:
            self.export_msg.setText(f'Could not export: {exc}')
            return
        self.export_msg.setText(f'Saved {path}')

    def pop_out(self):
        if self._make is None:
            return
        error = 'no figure was made'
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fig = self._make()
        except Exception as exc:
            fig = None
            error = str(exc) or type(exc).__name__
        finally:
            QApplication.restoreOverrideCursor()
        if fig is None:
            QMessageBox.warning(self, 'Pop Out', f'The figure could not be drawn: {error}')
            return
        open_figure_window(self, fig, self.name, connect=self.connect_canvas)
