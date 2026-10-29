"""
flowcode_gating_view.py
-----------------------
The *Gating* tab of the FlowCode Analysis page: the experiment's Unmixed
Data plots and gates over the FlowCode-unmixed events of the sample
selected in the main window's sample browser, drawn with Honeychrome's
``CytometryPlotWidget`` on a private event bus (as section 3's gate plots
are). View only: the plots' gates, axes and channels cannot be changed
here; gates are edited in the main window. Data are built in
flowcode_gating.py. Not named ``*_tab.py``, so the plugin loader does not
load it as a tab of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMenu, QScrollArea, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

import honeychrome.settings as hc_settings
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.event_bus import EventBus

import flowcode_gating as fcg

logger = logging.getLogger(__name__)

TASK = 'analysis_gating'
MODE = 'flowcode_gating'
COLUMNS = 4
HIGHLIGHT_COLOUR = (255, 59, 48)
INTRO = ('The experiment\'s Unmixed Data plots and gates over the FlowCode-unmixed events of the sample selected '
         'in the sample browser, gated exactly as the Analysis counts them (the sample\'s own gate overrides '
         'included). <i>Uncorrected</i> is the experiment\'s own unmix of the raw file; <i>FlowCode standard</i> '
         'adds the FRET correction (and tandem); <i>FlowCode calibrated</i> also the per-sample combination '
         'calibration (section 5, Advanced settings). Highlighted events are drawn in red; the table gives each '
         'gate\'s % of its parent and how many highlighted events change side against the uncorrected view, '
         'which is unmixed as well for that comparison. View only: edit gates in the main window.')
TABLE_COLUMNS = ('Gate', 'Parent', '% of parent', 'Events', 'Changing side')

_ReadOnlyPlot = None


def _plot_class():
    """``CytometryPlotWidget`` that cannot change its plot: no channel, axis,
    source-gate or new-plot menus, no zoom, and gate shapes that cannot be
    moved, renamed or deleted."""
    global _ReadOnlyPlot
    if _ReadOnlyPlot is not None:
        return _ReadOnlyPlot
    from honeychrome.view_components.cytometry_plot_widget import CytometryPlotWidget

    class ReadOnlyPlot(CytometryPlotWidget):
        def select_plot_on_parent_grid(self):
            pass

        def mouseDoubleClickEvent(self, event):
            event.accept()

        def set_up_plot(self, item=None, parent=None):
            pass

        def set_axis_transform(self, n, parent):
            pass

        def set_source_gate(self, n, parent):
            pass

        def new_plot_on_gate(self, source_gate):
            pass

        def apply_zoom(self, axis_name):
            axis = self.axis_bottom if axis_name == 'x' else self.axis_left
            axis._pending_delta = 0

        def configure_title(self):
            super().configure_title()
            self.plot_title.leftClickMenuItems = []

        def right_click_menu(self, ev):
            menu = QMenu()
            menu.addAction('Export Image', self.export_image)
            menu.exec(ev.screenPos().toPoint())

        def configure_rois(self, mode, index):
            super().configure_rois(mode, index)
            if mode == self.mode and index == self.n_in_plot_sequence:
                for roi in self.rois:
                    _freeze(roi)

        def refresh_custom_rois(self, mode):
            pass

        def _restore(self, *args):
            QTimer.singleShot(0, lambda: self.configure_rois(self.mode, self.n_in_plot_sequence))

        update_polygon = update_rectangle = update_ellipse = update_range = update_quad = _restore
        remove_gate_and_roi = _restore

    _ReadOnlyPlot = ReadOnlyPlot
    return ReadOnlyPlot


def _freeze(roi):
    """Stop a gate shape (and its handles, lines and label) taking mouse input."""
    if roi is None:
        return
    items = [roi] + [getattr(roi, a, None) for a in ('label', 'vx', 'vy', 'v1', 'v2')]
    seen = set()
    while items:
        item = items.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, 'translatable'):
            item.translatable = False
        if hasattr(item, 'setMovable'):
            item.setMovable(False)
        if hasattr(item, 'setAcceptedMouseButtons'):
            item.setAcceptedMouseButtons(Qt.NoButton)
        if hasattr(item, 'setAcceptHoverEvents'):
            item.setAcceptHoverEvents(False)
        if hasattr(item, 'childItems'):
            items.extend(item.childItems())


class GatingSection(QWidget):
    """The Gating tab; *page* is the ``AnalysisPage``."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self._token = None
        self._pending = False
        self._bus = None
        self._plots = []
        self.result = None
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)
        row = QHBoxLayout()
        self.sample_label = QLabel('')
        row.addWidget(self.sample_label)
        self.view_combo = QComboBox()
        for key, text in fcg.VIEWS.items():
            self.view_combo.addItem(text, key)
        self.view_combo.setCurrentIndex(self.view_combo.findData(fcg.VIEW_CALIBRATED))
        self.highlight_combo = QComboBox()
        self.highlight_combo.setToolTip('Events drawn in red and counted for "changing side".')
        for text, w in (('View', self.view_combo), ('Highlight', self.highlight_combo)):
            row.addWidget(QLabel(text))
            row.addWidget(w)
        row.addStretch()
        lay.addLayout(row)
        self.view_combo.currentIndexChanged.connect(lambda _i: self.start())
        self.highlight_combo.currentIndexChanged.connect(lambda _i: self.start())
        self.status = QLabel('')
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.table = CopyableTableWidget([], [])
        self.table.setColumnCount(len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setMaximumHeight(180)
        lay.addWidget(self.table)
        area = QScrollArea()
        area.setWidgetResizable(True)
        holder = QWidget()
        self.grid = QGridLayout(holder)
        self.grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        area.setWidget(holder)
        lay.addWidget(area, stretch=1)

    # -- sample and choices ----------------------------------------------------

    def current_sample(self) -> str | None:
        """The sample selected in the main window's sample browser."""
        path = getattr(self.page._controller(), 'current_sample_path', None)
        return str(path).replace('\\', '/') if path else None

    def is_shown(self) -> bool:
        return self.page.section_tabs.currentWidget() is self

    def refresh(self):
        """The selected sample's name and the combinations to highlight."""
        page = self.page
        key = self.current_sample()
        self.sample_label.setText(f'<b>{page._name(key)}</b> (pick a sample in the sample browser)' if key else
                                  'Pick a sample in the sample browser.')
        chosen = self.highlight_combo.currentData()
        self.highlight_combo.blockSignals(True)
        self.highlight_combo.clear()
        self.highlight_combo.addItem('Barcoded events', fcg.HIGHLIGHT_BARCODED)
        self.highlight_combo.addItem('Nothing', fcg.HIGHLIGHT_NONE)
        design = page._current_design()
        for i, name in enumerate(design.combo_names if design is not None else []):
            self.highlight_combo.addItem(name, i + 1)
        index = self.highlight_combo.findData(chosen)
        self.highlight_combo.setCurrentIndex(index if index >= 0 else 0)
        self.highlight_combo.blockSignals(False)

    def on_shown(self):
        """The tab became visible: draw the selected sample when it is not drawn."""
        self.refresh()
        result = self.result or {}
        if (result.get('key'), result.get('view'), result.get('highlight')) != (
                self.current_sample(), self.view_combo.currentData(), self.highlight_combo.currentData()):
            self.start()

    def sample_changed(self):
        """The main window selected another sample."""
        if self.is_shown():
            self.refresh()
            self.start()

    # -- run ---------------------------------------------------------------------

    def start(self):
        import flowcode_analysis as fa
        import flowcode_export as fce

        if not self.is_shown():
            return
        page = self.page
        plugin = page.plugin
        if plugin.task_running(TASK):
            self._pending = True
            return
        self._pending = False
        key = self.current_sample()
        if key is None:
            self.status.setText('Pick a sample in the sample browser.')
            return
        raw_path = Path(page._controller().experiment_dir) / key
        if not raw_path.exists() or not plugin.live_af_ok([key]).get(key):
            self.status.setText(f'{page._name(key)} cannot be FlowCode-unmixed: its raw file or its AF library '
                                '(at least 2 spectra) is missing.')
            return
        view = self.view_combo.currentData()
        views = [view] + ([fcg.VIEW_UNCORRECTED] if view != fcg.VIEW_UNCORRECTED else [])
        counted = list(page.used_files())
        keys = counted if key in counted else [key]
        try:
            # the counted samples' unmixing signature, so the view the counts use shares their cache
            inputs = {v: plugin.live_unmix_inputs(keys, fcg.VIEW_OVERRIDES[v]) for v in views}
        except ValueError as exc:
            self.status.setText(f'Cannot unmix: {exc}')
            return
        gate = getattr(page._controller(), 'job_gate', None)
        token = None
        if gate is not None:
            token = gate.try_acquire('FlowCode gating viewer')
            if token is None:
                busy = getattr(gate, 'busy_text', None)
                self.status.setText(busy() if callable(busy) else 'Another job is running.')
                return
        snapshot = page._gating_snapshot()
        experiment = page._controller().experiment
        plots = [dict(p) for p in (getattr(experiment, 'cytometry', None) or {}).get('plots') or []]
        unmixed = experiment.settings['unmixed']
        fluorescence = [unmixed['event_channels_pnn'][i] for i in unmixed.get('fluorescence_channel_ids') or []]
        highlight = self.highlight_combo.currentData()
        cache_keys = {v: fce.live_cache_key(inputs[v]['signature'], key, fce.file_signature(raw_path)) for v in views}
        steps = len(views) + 1

        def work(progress):
            events = {}
            for i, v in enumerate(views):
                progress(i, steps, f'unmixing, {fcg.VIEWS[v]}')
                inp = inputs[v]
                events[v] = fce.live_sample_events(inp['layout'], inp['af_by_sample'][key], inp['unmix_fn'], key,
                                                   inp['pnn'], inp['n_extra'], cache_key=cache_keys[v],
                                                   max_cache_bytes=inp['max_cache_bytes'])
            progress(len(views), steps, 'gating')
            pnn = inputs[view]['pnn']
            n_combos = len(fa.combo_names_in_file(pnn))
            data = fcg.plot_data(events[view], pnn, key, snapshot, plots, fluorescence)
            focus = fcg.focus_mask(fa.event_combo_ids(events[view], pnn)[0], highlight, n_combos)
            reference = None
            if view != fcg.VIEW_UNCORRECTED:
                other = events.pop(fcg.VIEW_UNCORRECTED)
                reference = snapshot.memberships(key, other, pnn)
                focus = focus | fcg.focus_mask(fa.event_combo_ids(other, pnn)[0], highlight, n_combos)
                del other
            rows = fcg.gate_rows(data, reference, focus)
            overlay = fcg.highlight_overlay(data, focus)
            progress(steps, steps, '')
            return {'key': key, 'view': view, 'highlight': highlight, 'data': data, 'rows': rows,
                    'overlay': overlay}

        if plugin._start_task(TASK, work):
            self._token = token
            self.status.setText(f'Unmixing {page._name(key)} ({", ".join(fcg.VIEWS[v] for v in views)})…')
        elif token is not None:
            token.release()

    def _release(self):
        token, self._token = self._token, None
        if token is not None:
            token.release()

    def on_finished(self, result):
        self._release()
        self.result = result
        self._fill_table(result['rows'])
        self._draw(result['data'], result['overlay'])
        n_plots = len(result['data']['plots'])
        self.status.setText(f"{self.page._name(result['key'])}, {fcg.VIEWS[result['view']]}: {n_plots} plot(s)."
                            + ('' if n_plots else ' The experiment has no Unmixed Data plots on these channels.'))
        if self._pending:
            QTimer.singleShot(0, self.start)

    def on_error(self, message):
        self._release()
        self.status.setText(f'Could not show the gating: {message}')
        if self._pending:
            QTimer.singleShot(0, self.start)

    def on_progress(self, n, total, text):
        if text:
            self.status.setText(f'{text[0].upper()}{text[1:]}… ({n + 1}/{total})')

    # -- display -----------------------------------------------------------------

    def _fill_table(self, rows: list):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, g in enumerate(rows):
            values = (g['name'], 'all events' if g['parent'] == 'root' else g['parent'],
                      f"{g['percent']:.2f}", f"{g['n_in']:,}", '' if g['changed'] is None else f"{g['changed']:,}")
            for c, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _clear_plots(self):
        for plot in self._plots:
            plot.setParent(None)
            plot.deleteLater()
        self._plots = []
        self._bus = None

    def _draw(self, data: dict, overlay: list):
        import pyqtgraph as pg

        self._clear_plots()
        plot_class = _plot_class()
        self._bus = EventBus()
        size = int(getattr(hc_settings, 'cytometry_plot_width_target_retrieved', 220) or 220)
        for i, plot in enumerate(data['plots']):
            widget = plot_class(bus=self._bus, mode=MODE, n_in_plot_sequence=i, plot=plot,
                                data_for_cytometry_plots=data)
            widget.setFixedSize(size, size)
            self.grid.addWidget(widget, i // COLUMNS, i % COLUMNS)
            self._plots.append(widget)
        self._bus.histsStatsRecalculated.emit(MODE, list(range(len(data['plots']))))
        for widget, extra in zip(self._plots, overlay):
            if extra is None:
                continue
            if extra[0] == 'points':
                item = pg.ScatterPlotItem(extra[1], extra[2], size=2, pen=None, brush=pg.mkBrush(*HIGHLIGHT_COLOUR, 160))
            else:
                step = data['transformations'][widget.plot['channel_x']].step_scale
                item = pg.PlotDataItem(step, extra[1], stepMode='center', pen=pg.mkPen(HIGHLIGHT_COLOUR, width=1.5))
            item.setZValue(30)
            widget.vb.addItem(item)
