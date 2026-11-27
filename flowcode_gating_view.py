"""
flowcode_gating_view.py
-----------------------
The *Gating* tab of the FlowCode Analysis page: the experiment's Unmixed
Data plots and gates over the FlowCode-unmixed events of the sample
selected in the main window's sample browser, drawn with Honeychrome's own
plot grid, plot toolbar and gating hierarchy on a private event bus.

The plots work as in Unmixed Data (channels, axes, gates, new plots) on the
tab's own copy of the experiment's gating, axes and plot list
(``flowcode_gating.LocalGating``): edits never reach the experiment or the
Analysis counts. Until something is edited the copy follows Unmixed Data;
*Reset to Unmixed Data* discards the edits. After each edit the shown
events are gated again on a worker thread. Data are built in
flowcode_gating.py. Not named ``*_tab.py``, so the plugin loader does not
load it as a tab of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidgetItem, QVBoxLayout, QWidget,
)

from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.cytometry_grid_widget import CytometryGridWidget
from honeychrome.view_components.cytometry_toolbar import CytometryToolbar
from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.focus_mode import mark_focus_chrome
from honeychrome.view_components.gating_hierarchy_widget import GatingHierarchyWidget

import flowcode_gating as fcg

logger = logging.getLogger(__name__)

TASK = 'analysis_gating'
MODE = 'flowcode_gating'
HIGHLIGHT_COLOUR = (255, 209, 220)
EDIT_DELAY_MS = 150
INTRO = ('The experiment\'s Unmixed Data plots and gates over the FlowCode-unmixed events of the sample selected '
         'in the sample browser, gated as the Analysis counts them (the sample\'s own gate overrides included) '
         'until they are edited here. <i>Uncorrected</i> is the experiment\'s own unmix of the raw file; '
         '<i>FlowCode standard</i> adds the FRET correction (and tandem); <i>FlowCode calibrated</i> also the '
         'per-sample combination calibration (section 5, Advanced settings). With the source set to <i>Exported '
         'files</i> (top of the page), the plots show the sample\'s exported file instead, compared with the '
         'uncorrected unmix when the raw file and AF library are available. Highlighted events (barcoded, one '
         'combination, or a phenotype population: membership at least one half, inside the FlowCode gates) are '
         'drawn in pink; the table gives each gate\'s % of its parent, how many highlighted events it holds and how '
         'many change side against the uncorrected view, which is unmixed as well for that comparison. The plots '
         'work as in Unmixed Data: change channels and axes, draw, move, rename or delete gates, add, move and '
         'delete plots. These edits belong to this tab only and change neither the experiment\'s gates nor the '
         'Analysis counts; <i>Reset to Unmixed Data</i> takes the experiment\'s gating again.')
TABLE_COLUMNS = ('Gate', 'Parent', '% of parent', 'Events', 'Highlighted', 'Changing side')
EDITED_NOTE = ' Edited here: the gates, axes or plots differ from Unmixed Data, which the Analysis counts use.'
PER_SAMPLE_NOTE = ('Per-sample gates are set in Unmixed Data. Gates edited here change only this tab\'s copy of '
                   'the gating.')


class GatingSection(QWidget):
    """The Gating tab; *page* is the ``AnalysisPage``."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self._token = None
        self._pending = False
        self._regate_pending = False
        self._drawing = False
        self._version = 0
        self._hidden = []
        self._overlay_items = []
        self.local = None
        self.result = None
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)
        mark_focus_chrome(intro)
        row = QHBoxLayout()
        self.sample_label = QLabel('')
        row.addWidget(self.sample_label)
        self.view_combo = QComboBox()
        for key, text in fcg.VIEWS.items():
            self.view_combo.addItem(text, key)
        self.view_combo.setCurrentIndex(self.view_combo.findData(fcg.VIEW_CALIBRATED))
        self.highlight_combo = QComboBox()
        self.highlight_combo.setToolTip('Events drawn in pink and counted for "changing side".')
        self.view_label = QLabel('View')
        row.addWidget(self.view_label)
        row.addWidget(self.view_combo)
        row.addWidget(QLabel('Highlight'))
        row.addWidget(self.highlight_combo)
        row.addStretch()
        self.reset_btn = QPushButton('Reset to Unmixed Data')
        self.reset_btn.setToolTip('Take the experiment\'s Unmixed Data gates, axes and plots again, discarding the '
                                  'edits made on this tab.')
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_to_unmixed)
        row.addWidget(self.reset_btn)
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

        self._bus = EventBus()
        self.tree = GatingHierarchyWidget(self._bus, mode=MODE)
        self.grid = CytometryGridWidget(self._bus, mode=MODE, gating_tree=self.tree)
        self.toolbar = CytometryToolbar(self._bus, parent=self.grid)
        self.grid.set_toolbar(self.toolbar)
        self.tree.set_toolbar(self.toolbar)
        self.toolbar.setEnabled(False)
        lay.addWidget(self.toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.grid)
        splitter.addWidget(self.tree)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1, 0])
        lay.addWidget(splitter, stretch=1)

        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(EDIT_DELAY_MS)
        self._edit_timer.timeout.connect(self.regate)
        self._connect_bus()

    def _connect_bus(self):
        """What the plots, toolbar and hierarchy ask for: in Unmixed Data the
        controller answers these signals; here this tab does, on its copy."""
        bus = self._bus
        bus.changedGatingHierarchy.connect(self._edited)
        bus.plotChangeRequested.connect(self._edited)
        bus.axisTransformed.connect(self._axis_transformed)
        bus.axesReset.connect(self._axes_reset)
        bus.newPlotRequested.connect(self._new_plot)
        bus.showNewPlot.connect(self._plot_added)
        bus.autoSaveRequested.connect(self._plots_changed)
        bus.statusMessage.connect(self.status.setText)
        for signal in (bus.customiseGateRequested, bus.revertGateRequested, bus.adoptGateRequested,
                       bus.applyGateToGroupRequested):
            signal.connect(lambda _mode, _name: self.status.setText(PER_SAMPLE_NOTE))

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
        for name in self._phenotypes():
            self.highlight_combo.addItem(f'Phenotype: {name}', fcg.phenotype_highlight(name))
        design = page._current_design()
        for i, name in enumerate(design.combo_names if design is not None else []):
            self.highlight_combo.addItem(name, i + 1)
        index = self.highlight_combo.findData(chosen)
        self.highlight_combo.setCurrentIndex(index if index >= 0 else 0)
        self.highlight_combo.blockSignals(False)

    def _phenotypes(self) -> dict:
        """``{population name: Resolved}`` of every resolved definition (split
        by congenic marker when the Data tab asks), the ticked ones first."""
        import flowcode_phenotype as fp
        page = self.page
        resolved = page.resolved_phenotypes()
        keys = [k for k, r in resolved.items() if isinstance(r, fp.Resolved)]
        ticked = page.phenotype_populations()
        out = dict(ticked)
        out.update({name: res for name, res in page.phenotype_populations(keys).items() if name not in out})
        return out

    def _local_out_of_date(self) -> bool:
        """The unedited copy no longer matches the experiment's Unmixed Data."""
        return (self.local is not None and not self.local.edited
                and self.local.base != fcg.unmixed_signature(self.page._controller()))

    def on_shown(self):
        """The tab became visible: draw the selected sample when it is not
        drawn, or when Unmixed Data changed and nothing was edited here."""
        self.refresh()
        result = self.result or {}
        view = self.view_combo.currentData() if self.page.is_live() else fcg.VIEW_EXPORTED
        if self._local_out_of_date() or (result.get('key'), result.get('view'), result.get('highlight')) != (
                self.current_sample(), view, self.highlight_combo.currentData()):
            self.start()

    def sample_changed(self):
        """The main window selected another sample."""
        if self.is_shown():
            self.refresh()
            self.start()

    def reset_to_unmixed(self):
        """Discard the edits made here and draw with the experiment's gating."""
        self.local = None
        self._version += 1
        self.reset_btn.setEnabled(False)
        self.start()

    def reset_session(self):
        """Forget the previous experiment's sample, copy and plots."""
        self._edit_timer.stop()
        self.local = None
        self.result = None
        self._hidden = []
        self._overlay_items = []
        self._version += 1
        self._regate_pending = False
        self.grid.selected_plot = None
        self.grid.clear_layout(self.grid.layout)
        self.grid.plot_widgets = []
        self.grid.data_for_cytometry_plots = None
        self.toolbar.update_button_state(None)
        self.toolbar.setEnabled(False)
        self.table.setRowCount(0)
        self.reset_btn.setEnabled(False)
        self.status.setText('')

    # -- run ---------------------------------------------------------------------

    def _exported_file(self, key: str):
        """(path, problem) of the sample's exported file when the Analysis
        source is Exported files: the path when it is linked and usable,
        else None and why."""
        import flowcode_analysis as fa
        info = ((self.page.links or {}).get('files') or {}).get(key)
        if not info or info.get('path') is None:
            return None, f'{self.page._name(key)} has no exported file in {self.page.export_folder()}.'
        if info['status'] == fa.LINK_UNREADABLE:
            return None, f'The exported file of {self.page._name(key)} cannot be read: {info.get("note") or ""}'
        return info['path'], None

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
        exported = not page.is_live()
        self._show_view_choice(not exported)
        exported_path = None
        if exported:
            exported_path, problem = self._exported_file(key)
            if exported_path is None:
                self.status.setText(problem)
                return
        view = fcg.VIEW_EXPORTED if exported else self.view_combo.currentData()
        raw_path = Path(page._controller().experiment_dir) / key
        live_ok = raw_path.exists() and bool(plugin.live_af_ok([key]).get(key))
        if not exported and not live_ok:
            self.status.setText(f'{page._name(key)} cannot be FlowCode-unmixed: its raw file or its AF library '
                                '(at least 2 spectra) is missing.')
            return
        live_views = ([] if exported else [view]) + ([fcg.VIEW_UNCORRECTED] if view != fcg.VIEW_UNCORRECTED
                                                     and live_ok else [])
        try:
            inputs = {v: plugin.live_unmix_inputs([key], fcg.VIEW_OVERRIDES[v]) for v in live_views}
        except ValueError as exc:
            if not exported:
                self.status.setText(f'Cannot unmix: {exc}')
                return
            logger.info('FlowCode gating: no uncorrected comparison for the exported file: %s', exc)
            inputs, live_views = {}, []
        highlight = self.highlight_combo.currentData()
        name = fcg.highlighted_phenotype(highlight)
        if name is not None and exported:
            self.status.setText('Highlighting a phenotype needs Live unmix: exported files carry no negative model.')
            return
        gate = getattr(page._controller(), 'job_gate', None)
        token = None
        if gate is not None:
            token = gate.try_acquire('FlowCode gating viewer')
            if token is None:
                busy = getattr(gate, 'busy_text', None)
                self.status.setText(busy() if callable(busy) else 'Another job is running.')
                return
        if self.local is None or self._local_out_of_date():
            self.local = fcg.LocalGating.from_controller(page._controller())
            self._version += 1
        snapshot, plots = self.local.worker_snapshot(key)
        version = self._version
        experiment = page._controller().experiment
        unmixed = experiment.settings['unmixed']
        fluorescence = [unmixed['event_channels_pnn'][i] for i in unmixed.get('fluorescence_channel_ids') or []]
        spectral_model = list(experiment.process.get('spectral_model') or [])
        phenotype = None
        if name is not None:
            import flowcode_phenotype as fp
            import flowcode_state as fcs
            resolved = self._phenotypes().get(name)
            if resolved is None:
                self.status.setText(f'The phenotype {name} is no longer resolved on this panel.')
                if token is not None:
                    token.release()
                return
            inp = inputs[view]
            model = fp.sample_negative_model(inp['af_by_sample'][key], list(inp['layout'].fluor_names),
                                             inp['plan'].state, key)
            if model is None:
                self.status.setText(f'{page._name(key)} has no negative model for phenotypes: no AF library and '
                                    'no noise model.')
                if token is not None:
                    token.release()
                return
            records = [plugin.state.gates[n] for n in fcs.GATE_NAMES if n in plugin.state.gates]
            phenotype = (resolved, model, records)
        cache_keys = {v: fce.live_cache_key(inputs[v]['sample_signatures'][key], key, fce.file_signature(raw_path))
                      for v in live_views}
        steps = len(live_views) + (2 if exported else 1)

        def focus_of(view_events, pnn, n_combos):
            if phenotype is None:
                return fcg.focus_mask(fa.event_combo_ids(view_events, pnn)[0], highlight, n_combos)
            import flowcode_phenotype as fp
            resolved, model, records = phenotype
            return fp.population_mask(view_events, pnn, resolved, model, records)

        def work(progress):
            events = {}
            step = 0
            if exported:
                progress(step, steps, 'reading the exported file')
                events[view] = fa.read_exported_events(exported_path)
                step += 1
            for v in live_views:
                progress(step, steps, f'unmixing, {fcg.VIEWS[v]}')
                inp = inputs[v]
                events[v] = (fce.live_sample_events(inp['layout'], inp['af_by_sample'][key], inp['unmix_fn'], key,
                                                    inp['pnn'], inp['n_extra'], cache_key=cache_keys[v],
                                                    max_cache_bytes=inp['max_cache_bytes']), inp['pnn'])
                step += 1
            progress(step, steps, 'gating')
            shown, pnn = events.pop(view)
            focus = focus_of(shown, pnn, len(fa.combo_names_in_file(pnn)))
            reference = None
            if fcg.VIEW_UNCORRECTED in events:
                other, other_pnn = events.pop(fcg.VIEW_UNCORRECTED)
                if len(other) == len(shown):
                    reference = (other, other_pnn)
                    focus = focus | focus_of(other, other_pnn, len(fa.combo_names_in_file(other_pnn)))
            out = fcg.gate_view(shown, pnn, key, snapshot, plots, fluorescence, spectral_model, reference, focus)
            progress(steps, steps, '')
            out.update({'key': key, 'view': view, 'highlight': highlight, 'n_focus': int(focus.sum()),
                        'focus': focus, 'reference': reference, 'fluorescence': fluorescence,
                        'spectral_model': spectral_model, 'version': version})
            return out

        if plugin._start_task(TASK, work):
            self._token = token
            what = [fcg.VIEW_LABELS[view]] if exported else []
            what += [fcg.VIEWS[v] for v in live_views]
            self.status.setText(f'{"Reading" if exported else "Unmixing"} {page._name(key)} ({", ".join(what)})…')
        elif token is not None:
            token.release()

    def regate(self):
        """Gate the shown events again with the tab's copy after an edit, on
        a worker thread (no unmixing)."""
        if self.result is None or self.local is None:
            return
        plugin = self.page.plugin
        if plugin.task_running(TASK):
            self._regate_pending = True
            return
        self._regate_pending = False
        result = self.result
        key = result['key']
        data = result['data']
        snapshot, plots = self.local.worker_snapshot(key)
        events, pnn = data['event_data'], data['pnn']
        reference, focus = result['reference'], result['focus']
        fluorescence, spectral_model = result['fluorescence'], result['spectral_model']
        version = self._version

        def work(progress):
            progress(0, 1, 'gating')
            out = fcg.gate_view(events, pnn, key, snapshot, plots, fluorescence, spectral_model, reference, focus)
            progress(1, 1, '')
            out.update({'regate': True, 'key': key, 'version': version})
            return out

        if not plugin._start_task(TASK, work):
            self._regate_pending = True

    def _show_view_choice(self, shown: bool):
        """The View drop-down applies to Live unmix only; with Exported files
        the plots show the exported file."""
        self.view_label.setVisible(shown)
        self.view_combo.setVisible(shown)

    def _release(self):
        token, self._token = self._token, None
        if token is not None:
            token.release()

    def on_finished(self, result):
        if result.get('regate'):
            self._apply_regate(result)
        else:
            self._release()
            if result['version'] != self._version:
                self._pending = True
            else:
                self._show(result)
        self._follow_up()

    def on_error(self, message):
        self._release()
        self.status.setText(f'Could not show the gating: {message}')
        self._follow_up()

    def _follow_up(self):
        if self._pending:
            QTimer.singleShot(0, self.start)
        elif self._regate_pending:
            self._edit_timer.start()

    def on_progress(self, n, total, text):
        if text:
            self.status.setText(f'{text[0].upper()}{text[1:]}… ({n + 1}/{total})')

    # -- edits on the plots ------------------------------------------------------

    def _edited(self, *_args):
        """A plot, axis or gate changed on the tab: gate the events again."""
        if self._drawing or self.result is None or self.local is None:
            return
        self.local.edited = True
        self.reset_btn.setEnabled(True)
        self._version += 1
        self._edit_timer.start()

    def _axis_transformed(self, channel: str):
        """An axis was changed on a plot (its menu, a drag, Fit Axes): keep it
        in the copy, where a per-sample axis is otherwise remade per sample."""
        if self._drawing or self.result is None or self.local is None:
            return
        axis = self.result['data']['transformations'].get(channel)
        if axis is not None:
            self.local.transforms[channel] = axis
        self._edited()

    def _axes_reset(self, channels: list):
        """Reset Axes on a plot: the experiment's default axes, as in Unmixed
        Data."""
        if self.result is None or self.local is None:
            return
        data = self.result['data']
        fresh = fcg.default_axes(self.page._controller(), channels, data['event_data'], data['pnn'])
        for ch in channels:
            if ch in fresh:
                data['transformations'][ch] = fresh[ch]
                self.local.transforms[ch] = fresh[ch]
        self._edited()

    def _new_plot(self, channel_x: str, channel_y: str):
        """The new-plot tile asked for a plot, made as Unmixed Data makes one."""
        if self.result is None:
            return
        data = self.result['data']
        pnn = data['pnn']
        if channel_x in pnn and channel_y in pnn and channel_x != channel_y:
            plot = {'type': 'hist2d', 'channel_x': channel_x, 'channel_y': channel_y, 'source_gate': 'root',
                    'child_gates': []}
        elif channel_x in pnn and (channel_y == 'Count' or channel_y == channel_x or channel_y is None):
            plot = {'type': 'hist1d', 'channel_x': channel_x, 'source_gate': 'root', 'child_gates': []}
        else:
            plot = {'type': 'ribbon', 'source_gate': 'root', 'child_gates': [], 'width': 3}
        data['plots'].append(plot)
        self._bus.showNewPlot.emit(MODE)

    def _plot_added(self, mode: str):
        """A plot was added (the new-plot tile, or a new plot on a gate): give
        it an empty histogram until the events are gated again."""
        from honeychrome.controller_components.functions import initialise_hists

        if mode != MODE or self.result is None:
            return
        data = self.result['data']
        missing = len(data['plots']) - len(data['histograms'])
        if missing > 0:
            data['histograms'] += initialise_hists(data['plots'][-missing:], data)
        self._plots_changed()

    def _plots_changed(self, *_args):
        """Plots were added, deleted or moved: the copy's plot list follows,
        with the plots on channels these events lack kept at the end."""
        if self._drawing or self.result is None or self.local is None:
            return
        self.local.plots[:] = list(self.result['data']['plots']) + list(self._hidden)
        self._edited()

    # -- display -----------------------------------------------------------------

    def _show(self, result: dict):
        """A sample was unmixed and gated: draw it on the copy's own objects."""
        self.result = result
        self._hidden = fcg.bind(result['data'], self.local, result['key'])
        self._fill_table(result['rows'])
        self._draw(result['data'], result['overlay'])
        self._set_status()

    def _apply_regate(self, out: dict):
        """The shown events were gated again after an edit: update the
        histograms, statistics, table and highlights in place, or draw afresh
        when the plot list no longer matches."""
        if self.result is None or self.local is None or out['key'] != self.result['key']:
            return
        if out['version'] != self._version:
            self._regate_pending = True
            return
        data = self.result['data']
        new = out['data']
        plots = [self.local.plots[i] for i in new['plot_index'] if i < len(self.local.plots)]
        self.result.update(rows=out['rows'], overlay=out['overlay'])
        self._fill_table(out['rows'])
        if len(plots) != len(data['plots']) or any(a is not b for a, b in zip(plots, data['plots'])):
            self.result['data'] = new
            self._hidden = fcg.bind(new, self.local, out['key'])
            self._draw(new, out['overlay'])
        else:
            data['gate_membership'] = new['gate_membership']
            data['statistics'] = new['statistics']
            data['histograms'][:] = new['histograms']
            self._drawing = True
            try:
                self._bus.histsStatsRecalculated.emit(MODE, list(range(len(data['plots']))))
                self._draw_overlay(out['overlay'])
            finally:
                self._drawing = False
        self._set_status()

    def _set_status(self):
        result = self.result
        n_plots = len(result['data']['plots'])
        highlighted = result.get('n_focus')
        edited = self.local is not None and self.local.edited
        text = (f"{self.page._name(result['key'])}, {fcg.VIEW_LABELS[result['view']]}: {n_plots} plot(s)"
                + (f"; {highlighted:,} highlighted events" if highlighted else '') + '.')
        capped = max((extra[3] for extra in result.get('overlay') or ()
                      if extra is not None and extra[0] == 'points' and extra[3] > fcg.MAX_HIGHLIGHT_POINTS),
                     default=0)
        if capped:
            text += (f' Plots with more than {fcg.MAX_HIGHLIGHT_POINTS:,} highlighted events draw a random '
                     f'{fcg.MAX_HIGHLIGHT_POINTS:,} of them in pink.')
        if not n_plots:
            text += ' No plots on these channels: add one from the toolbar.'
        if edited:
            text += EDITED_NOTE
        self.status.setText(text)
        self.reset_btn.setEnabled(edited)

    def _fill_table(self, rows: list):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, g in enumerate(rows):
            values = (g['name'], 'all events' if g['parent'] == 'root' else g['parent'],
                      f"{g['percent']:.2f}", f"{g['n_in']:,}", '' if g.get('n_focus') is None else f"{g['n_focus']:,}",
                      '' if g['changed'] is None else f"{g['changed']:,}")
            for c, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def _draw(self, data: dict, overlay: list):
        """Build the plots, toolbar state and hierarchy for *data*."""
        self._drawing = True
        try:
            self._overlay_items = []
            self.grid.init_plots(data)
            self.toolbar.update_button_state(None)
            self.toolbar.setEnabled(True)
            self.tree.init_data(data)
            self._bus.histsStatsRecalculated.emit(MODE, list(range(len(data['plots']))))
            self._draw_overlay(overlay)
        finally:
            self._drawing = False

    def _draw_overlay(self, overlay: list):
        """The highlighted events in pink over each plot's density."""
        import pyqtgraph as pg

        for widget, item in self._overlay_items:
            try:
                widget.vb.removeItem(item)
            except RuntimeError:
                continue
        self._overlay_items = []
        transforms = self.result['data']['transformations']
        for widget, extra in zip(self.grid.plot_widgets, overlay):
            if extra is None:
                continue
            if extra[0] == 'points':
                item = pg.ScatterPlotItem(extra[1], extra[2], size=2, pen=None, brush=pg.mkBrush(*HIGHLIGHT_COLOUR, 160))
            else:
                step = transforms[widget.plot['channel_x']].step_scale
                item = pg.PlotDataItem(step, extra[1], stepMode='center', pen=pg.mkPen(HIGHLIGHT_COLOUR, width=1.5))
            item.setZValue(30)
            widget.vb.addItem(item)
            self._overlay_items.append((widget, item))
