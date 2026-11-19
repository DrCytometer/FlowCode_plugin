"""
flowcode_synopsis_view.py
-------------------------
The *Synopsis* tab of the FlowCode Analysis page: every readout of the page
per guide, in one place (flowcode_synopsis.py; figures in
flowcode_synopsis_plots.py). It runs nothing of its own. It reads the
statistics shown on the Results tab (counts, frequencies, markers, gene
groups) and the run shown on the Phenotype shift tab (shift and immune
states), and is put together again, on a worker, when the tab is shown and
one of them, or a choice on this tab, has changed.

Views
    Overview        the summary in words and one row per guide with its
                    evidence class, call counts and sentence
    Atlas           groups × guides: accumulation as colour, phenotype
                    calls as dots
    Guide dossier   one guide across the groups, one panel per readout
    Calls           one row per guide and group

Not named ``*_tab.py``, so the plugin loader does not load it as a tab of
its own.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from honeychrome.view_components.copyable_table_widget import CopyableTableWidget

import flowcode_figure_panel as fcfp
import flowcode_synopsis as fsy
import flowcode_synopsis_plots as fsyp
import flowcode_view_stack as fvs

logger = logging.getLogger(__name__)

TASK = 'analysis_synopsis'
TASKS = (TASK,)
TITLE = 'Synopsis'
_WARN_STYLE = 'color: #cc6600;'
_HINT_STYLE = 'color: grey; font-style: italic;'
STALE_TEXT = ('Out of date: the statistics or the phenotype shift shown here were made with other counts, samples '
              'or settings. Run them again to update the synopsis.')
EMPTY_TEXT = 'Run statistics (Statistics tab) or the phenotype shift (Phenotype shift tab) first.'
GUIDE_HEADERS = {
    'gene_group': 'gene group', 'groups_more': 'more cells (groups)', 'groups_fewer': 'fewer cells (groups)',
    'groups_tests_agree': 'counts and frequencies agree (groups)', 'groups_shift': 'shift (groups)',
    'state_calls': 'immune-state calls', 'marker_calls': 'marker calls',
    'groups_two_phenotype_readouts': '2+ phenotype readouts (groups)',
    'groups_accumulation_tested': 'accumulation tested (groups)',
    'groups_phenotype_tested': 'phenotype tested (groups)'}
CALL_COLUMNS = ['guide', 'group', 'logFC', 'q_accumulation', 'accumulation', 'accumulation_other', 'tests_agree',
                'unbounded', 'cells', 'shift_angle', 'q_shift', 'shift', 'states_tested', 'states_towards',
                'states_away', 'markers_tested', 'markers_higher', 'markers_lower', 'phenotype_readouts',
                'phenotype_tested']

HELP = (
    '<p><b>In short.</b> One place to see what each guide does across every group, from the results already made '
    'on this page. Nothing is tested here: every call is the one its own test made, with the thresholds set on the '
    '<i>Statistics</i> and <i>Phenotype shift</i> tabs.</p>'
    '<p><b>Two kinds of question.</b> <i>Accumulation</i> asks how many of a guide\'s cells a group holds compared '
    'with a baseline group (the counts and frequencies tests): are knockout cells lost from, or piling up in, a '
    'tissue or condition? <i>Phenotype</i> asks what a guide\'s cells look like inside one group, compared with the '
    'guide\'s reference in the same sample (the marker test, the phenotype shift and the immune states): have the '
    'cells that are there changed kind? A guide can be normal in number and changed in kind, or the reverse.</p>'
    '<p><b>What to do here.</b> Choose the <i>Population</i>, the <i>Baseline</i> group the cell numbers are '
    'compared with (the reference group of the statistics; any group when every pair was compared) and, when both '
    'were run, whether <i>Cells from</i> the counts or the frequencies test are shown. Start with the '
    '<i>Atlas</i>, then double-click a guide for its <i>dossier</i>.</p>'
    '<p><b>Atlas.</b> Groups down, guides across. Colour: the log2 fold change of the guide\'s cells in the group '
    'against the baseline (red: more, blue: fewer), where the test calls it; <i>Show every fold change</i> colours '
    'every tested cell and outlines the calls. Dot: a phenotype call in that group, larger when two or all three '
    'of shift, immune states and markers make one. Grey: no phenotype readout could be tested there (too few '
    'cells), so a missing dot says nothing. Cross: the test made a call but the guide has no cell at all in the '
    'group (or in the baseline), so there is no fold change to show. The numbers on the right are the median '
    'barcoded cells per sample: a group with few cells can show only large losses or gains. Guides are ordered by '
    'gene group when a gene-group table is loaded (the strip above the cells), by effect, or by name.</p>'
    '<p><b>Guide dossier.</b> One guide across the groups, panels sharing the group axis: the fold change with its '
    '95% interval for both tests (filled: called), the guide\'s cells counted in the group, the shift angle, the '
    'immune-state scores and the markers with a call (ringed: called; grey: not tested).</p>'
    '<p><b>Overview.</b> The summary in words, and one row per guide. The <i>class</i> says which kinds of result '
    'exist for the guide: accumulation and phenotype, accumulation only, phenotype only, or no call. It does not '
    'say how the gene works. The columns beside it count the groups with a call per readout, the groups where the '
    'counts and frequencies tests agree, and the groups where two or more phenotype readouts agree; sort by any '
    'of them. Each guide\'s sentence is built from fixed templates, so the same results always give the same '
    'words. Double-click a row for the dossier.</p>'
    '<p><b>Reading it with care.</b> The counts and frequencies tests use the same counts, and the three phenotype '
    'readouts read the same cells: agreement within a family is consistency, not independent confirmation. '
    'Accumulation against phenotype is the comparison between different measurements. Phenotype calls use the '
    'phenotype shift\'s FDR, so with many guides and groups some calls by a single readout are expected by '
    'chance.</p>'
    '<p><b>Left out for now.</b> Population contrasts and comparisons between populations (<i>Compare: '
    'Populations</i>) have no baseline group and are not shown. With <i>Split by</i>, each level is listed as a '
    'population of its own and shares its population\'s phenotype readouts.</p>'
    '<p><b>Export.</b> The Export tab adds <tt>synopsis.txt</tt>, <tt>synopsis_guides.csv</tt> and '
    '<tt>synopsis_calls.csv</tt>, the atlas of each population, and the dossier of every guide with both an '
    'accumulation and a phenotype call. Any other dossier can be saved from its own Export Graphic button.</p>'
    '<p><b>Glossary.</b> <i>Baseline</i>: the group the cell numbers of every other group are compared with. '
    '<i>Call</i>: a result its test found significant at the thresholds set for that test. <i>Not detected</i>: no '
    'cell of the guide in any sample of the group.</p>'
)


def _label(text: str = '', style: str = _HINT_STYLE) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.PlainText)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(style)
    return lbl


def _no_wheel(*widgets):
    from flowcode_shift_view import _no_wheel as no_wheel
    no_wheel(*widgets)


def _fill(table, frame, formats: dict | None = None):
    from flowcode_shift_view import _fill_table
    _fill_table(table, frame, formats)


class SynopsisSection(QWidget):
    """The Synopsis tab; *page* is the ``AnalysisPage``."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.synopsis = None
        self.guide = None
        self._key = None
        self._scopes = []
        self._scope_notes = []
        self._pending = False
        self._filling = False
        folder, options = page.analysis_root, page.figure_options_store
        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.addWidget(fcfp.scrolling_help(HELP))
        self.stale_label = _label(STALE_TEXT, _WARN_STYLE)
        self.stale_label.setVisible(False)
        outer.addWidget(self.stale_label)
        self.status = _label('')
        outer.addWidget(self.status)

        pick = QHBoxLayout()
        self.cb_scope, self.cb_baseline = QComboBox(), QComboBox()
        self.cb_test, self.cb_order = QComboBox(), QComboBox()
        self.cb_baseline.setToolTip('The group the cell numbers of every other group are compared with.')
        self.cb_test.setToolTip('The test the fold changes and accumulation calls are taken from.')
        for key, text in fsy.ORDERS.items():
            self.cb_order.addItem(text, key)
        self.chk_all = QCheckBox('Show every fold change')
        self.chk_all.setToolTip('Colour every tested cell of the atlas by its fold change and outline the calls, '
                                'instead of colouring the calls only.')
        self.views = fvs.ViewStack()
        self.lbl_test = QLabel('<b>Cells from:</b>')
        for text, widget in (('<b>Population:</b>', self.cb_scope), ('<b>Baseline:</b>', self.cb_baseline),
                             (None, self.cb_test), ('<b>Guides:</b>', self.cb_order)):
            pick.addWidget(self.lbl_test if text is None else QLabel(text))
            pick.addWidget(widget, stretch=1)
        pick.addWidget(self.chk_all)
        pick.addWidget(QLabel('<b>View:</b>'))
        pick.addWidget(self.views.selector, stretch=1)
        _no_wheel(self.cb_scope, self.cb_baseline, self.cb_test, self.cb_order)
        self.cb_scope.currentIndexChanged.connect(lambda _i: self._on_choice())
        self.cb_baseline.currentIndexChanged.connect(lambda _i: self._set('baseline', self.cb_baseline.currentData()))
        self.cb_test.currentIndexChanged.connect(lambda _i: self._set('test', self.cb_test.currentData()))
        self.cb_order.currentIndexChanged.connect(lambda _i: self._set('order', self.cb_order.currentData()))
        self.chk_all.toggled.connect(lambda v: self._set('show_all', bool(v)))
        outer.addLayout(pick)

        self.overview = QLabel('')
        self.overview.setTextFormat(Qt.PlainText)
        self.overview.setWordWrap(True)
        self.overview.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.overview.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(self.overview)
        self.guide_table = CopyableTableWidget([], [])
        self.guide_table.setToolTip('Double-click a guide to show its dossier.')
        self.guide_table.cellDoubleClicked.connect(self._on_table_double_clicked)
        split = QSplitter(Qt.Vertical)
        split.addWidget(area)
        split.addWidget(self.guide_table)
        split.setSizes([260, 400])
        self.views.addTab(split, 'Overview')
        self.atlas = fcfp.FigurePanel('Synopsis atlas', folder, options_store=options)
        self.atlas.figure_key = 'synopsis_atlas'
        self.atlas.connect_canvas = self._connect_atlas
        self.views.addTab(self.atlas, 'Atlas')
        self.dossier = fcfp.FigurePanel('Guide dossier', folder, options_store=options)
        self.dossier.figure_key = 'synopsis_dossier'
        self.cb_guide = QComboBox()
        _no_wheel(self.cb_guide)
        self.cb_guide.currentIndexChanged.connect(lambda _i: self._pick_guide(self.cb_guide.currentData()))
        self.dossier_view = QWidget()
        lay = QVBoxLayout(self.dossier_view)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel('<b>Guide:</b>'))
        row.addWidget(self.cb_guide, stretch=1)
        lay.addLayout(row)
        lay.addWidget(self.dossier, stretch=1)
        self.views.addTab(self.dossier_view, 'Guide dossier')
        self.calls_table = CopyableTableWidget([], [])
        self.calls_table.setToolTip('Double-click a row to show that guide\'s dossier.')
        self.calls_table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.views.addTab(self.calls_table, 'Calls')
        outer.addWidget(self.views, stretch=1)
        self.figure_msg = _label('')
        outer.addWidget(self.figure_msg)

    # -- settings ---------------------------------------------------------------

    @property
    def settings(self) -> dict:
        """The saved choices (``state.analysis['synopsis']``), defaults filled in."""
        s = self.page.settings
        merged = fsy.synopsis_settings(s.get('synopsis'))
        if s.get('synopsis') != merged:
            s['synopsis'] = merged
        return s['synopsis']

    def _set(self, key, value):
        if self._filling or value is None or self.settings.get(key) == value:
            return
        self.settings[key] = value
        self.page._changed()
        self.refresh()

    def _on_choice(self):
        if not self._filling:
            self.refresh()

    # -- what is read -------------------------------------------------------------

    def is_shown(self) -> bool:
        return self.page.section_tabs.currentWidget() is self

    def _sources(self) -> tuple:
        return self.page.stats, self.page.shift_view.result

    def _stale(self) -> bool:
        stats, shift = self._sources()
        return bool((stats is not None and self.page._stats_stale)
                    or (shift is not None and self.page.shift_view._stale))

    def _inputs(self, scope: dict | None) -> dict:
        """The arguments of ``flowcode_synopsis.assemble`` for *scope*."""
        page, s = self.page, self.settings
        return dict(scope=scope, baseline=s['baseline'], test=s['test'], order=s['order'],
                    fdr_scope=page.settings.get('fdr_scope', 'per_comparison'),
                    group_order=list(page.settings.get('group_names') or []),
                    marker_labels=page.shift_view._short_labels(), gene_groups=page._group_result())

    def _source_key(self) -> tuple:
        """Changes whenever the synopsis shown would differ: the results
        (by identity: each run or re-thresholding makes new tables), the
        choices on this tab and what names and colours the figures."""
        stats, shift = self._sources()
        page, s = self.page, self.settings
        parts = [id(stats), id(shift), self.cb_scope.currentText(), tuple(sorted(s.items())),
                 page.settings.get('fdr_scope'), tuple(page.settings.get('group_names') or []),
                 tuple(page.gene_group_order()), tuple(sorted(page.gene_group_colours().items())),
                 int(page.settings.get('heatmap_rows', 1)), page.plugin.is_dark(), page.uncorrected()]
        for source, keys in ((stats, ('counts', 'freq', 'markers', 'gene_groups')), (shift, ('results', 'fsea'))):
            parts += [id(source.get(k)) if source is not None else None for k in keys]
        return tuple(parts)

    def _scope(self) -> dict | None:
        i = self.cb_scope.currentData()
        return self._scopes[i] if i is not None and 0 <= i < len(self._scopes) else None

    # -- refresh ------------------------------------------------------------------

    def reset_session(self):
        self.synopsis = None
        self.guide = None
        self._key = None
        self._scopes = []
        self._pending = False
        self.stale_label.setVisible(False)
        self.status.setText('')
        self._clear_views()

    def _clear_views(self):
        self.overview.setText('')
        for table in (self.guide_table, self.calls_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        for panel in (self.atlas, self.dossier):
            panel.clear()
        self.cb_guide.blockSignals(True)
        self.cb_guide.clear()
        self.cb_guide.blockSignals(False)
        self.figure_msg.setText('')

    def refresh(self):
        """Put the synopsis together again when the tab is shown and its
        results or choices have changed since it was last made."""
        if not self.is_shown():
            return
        stats, shift = self._sources()
        self.stale_label.setVisible(self._stale())
        if stats is None and shift is None:
            self.synopsis, self._key = None, None
            self._clear_views()
            self.status.setText(EMPTY_TEXT)
            return
        self._fill_choices(stats, shift)
        key = self._source_key()
        if key == self._key:
            return
        self._start(key)

    def _fill_choices(self, stats, shift):
        scopes, notes = fsy.scopes(stats, shift, self.page.counted_population())
        s = self.settings
        self._filling = True
        try:
            current = self.cb_scope.currentText()
            self._scopes = scopes
            self.cb_scope.clear()
            for i, scope in enumerate(scopes):
                self.cb_scope.addItem(scope['label'], i)
            if current:
                self.cb_scope.setCurrentIndex(max(self.cb_scope.findText(current), 0))
            scope = self._scope()
            level = scope['level'] if scope else None
            self.cb_baseline.clear()
            for group in fsy.baselines(stats, level):
                self.cb_baseline.addItem(str(group), group)
            chosen = fsy.resolve_baseline(stats, level, s['baseline'])
            self.cb_baseline.setCurrentIndex(max(self.cb_baseline.findData(chosen), 0))
            self.cb_baseline.setEnabled(self.cb_baseline.count() > 1)
            tests = fsy.available_tests(stats)
            self.cb_test.clear()
            for test in tests:
                self.cb_test.addItem(fsy.TESTS[test].capitalize(), test)
            self.cb_test.setCurrentIndex(max(self.cb_test.findData(fsy.resolve_test(stats, s['test'])), 0))
            for widget in (self.cb_test, self.lbl_test):
                widget.setVisible(len(tests) > 1)
            self.cb_order.setCurrentIndex(max(self.cb_order.findData(s['order']), 0))
            self.chk_all.setChecked(bool(s['show_all']))
        finally:
            self._filling = False
        self._scope_notes = notes

    def _start(self, key):
        if self.page.plugin.task_running(TASK):
            self._pending = True
            return
        self._pending = False
        stats, shift = self._sources()
        inputs = self._inputs(self._scope())
        if inputs['scope'] is None:
            self.synopsis, self._key = None, key
            self._clear_views()
            self.status.setText('; '.join(self._scope_notes) or EMPTY_TEXT)
            return
        maker = self._atlas_maker(self.page.plugin.is_dark())

        def work(_progress):
            synopsis = fsy.assemble(stats, shift, **inputs)
            make = (lambda: maker(synopsis))
            try:
                fig, error = make(), None
            except Exception as exc:
                logger.exception('FlowCode synopsis: the atlas could not be drawn')
                fig, error = None, str(exc)
            return {'key': key, 'synopsis': synopsis, 'atlas': fig, 'make': make, 'error': error}

        if self.page.plugin._start_task(TASK, work):
            self.figure_msg.setText('Putting the synopsis together…')

    def _atlas_maker(self, is_dark: bool):
        """A function drawing the atlas of a synopsis with the page's
        gene-group colours and heatmap panels."""
        colours = self.page.gene_group_colours()
        n_rows = int(self.page.settings.get('heatmap_rows', 1))
        show_all = bool(self.settings['show_all'])
        return lambda synopsis: fsyp.atlas_figure(synopsis, colours, is_dark, '', n_rows, show_all)

    # -- task results -------------------------------------------------------------

    def on_finished(self, _key, result):
        if self._pending or result['key'] != self._source_key():
            self._pending = False
            self._key = None
            self.refresh()
            return
        self._key = result['key']
        self.synopsis = result['synopsis']
        self.figure_msg.setText(f"Atlas: {result['error']}" if result['error'] else '')
        self._show(result)

    def on_error(self, _key, message):
        self._pending = False
        self.figure_msg.setText(f'The synopsis could not be put together: {message}')

    def _show(self, result):
        synopsis = self.synopsis
        scope = synopsis['scope']
        table = synopsis['guide_table']
        notes = list(self._scope_notes)
        parts = [f"{scope['label']}: {len(synopsis['guides'])} guide(s) in {len(synopsis['groups'])} group(s)"]
        for kind in fsy.CLASSES[:3]:
            parts.append(f"{int((table['class'] == kind).sum())} {kind.lower()}")
        text = ', '.join(parts) + '.'
        if self.page.uncorrected():
            self.status.setStyleSheet('color: #d62728; font-weight: bold;')
            text += ' (no FRET correction)'
        else:
            self.status.setStyleSheet(_HINT_STYLE)
        self.status.setText(' '.join([text] + notes))
        self.overview.setText(fsy.overview_text(synopsis))
        _fill(self.guide_table, table.rename(columns=GUIDE_HEADERS))
        calls = synopsis['calls']
        _fill(self.calls_table, calls[[c for c in CALL_COLUMNS if c in calls.columns]],
              {'logFC': '{:+.2f}', 'shift_angle': '{:+.1f}', 'cells': '{:.0f}'})
        if result['atlas'] is not None:
            self.atlas.name = f"{scope['label']} - synopsis atlas"
            fig, make = self.page._stamped(result['atlas'], result['make'])
            self.atlas.set_figure(fig, make=make)
        else:
            self.atlas.clear()
        guides = list(synopsis['guides'])
        if self.guide not in guides:
            called = table[table['class'] == fsy.BOTH]
            self.guide = called['guide'].iloc[0] if len(called) else (guides[0] if guides else None)
        self.cb_guide.blockSignals(True)
        try:
            self.cb_guide.clear()
            for guide in sorted(guides, key=lambda g: str(g).lower()):
                self.cb_guide.addItem(str(guide), guide)
            if self.guide is not None:
                self.cb_guide.setCurrentIndex(max(self.cb_guide.findData(self.guide), 0))
        finally:
            self.cb_guide.blockSignals(False)
        self.draw_dossier()

    # -- dossier ------------------------------------------------------------------

    def _pick_guide(self, guide):
        if guide is not None and guide != self.guide:
            self.guide = guide
            self.draw_dossier()

    def draw_dossier(self):
        synopsis, guide = self.synopsis, self.guide
        if synopsis is None or guide is None:
            self.dossier.clear()
            return
        is_dark = self.page.plugin.is_dark()

        def make():
            return fsyp.dossier_figure(synopsis, guide, is_dark, '')
        try:
            fig = make()
        except Exception as exc:
            logger.exception('FlowCode synopsis: the dossier could not be drawn')
            self.dossier.clear()
            self.figure_msg.setText(f'Guide dossier: {exc}')
            return
        self.dossier.name = f"{synopsis['scope']['label']} - {guide} dossier"
        fig, make = self.page._stamped(fig, make)
        self.dossier.set_figure(fig, make=make)

    def show_dossier(self, guide: str):
        """Show *guide*'s dossier in the Guide dossier view."""
        if self.synopsis is None or guide not in self.synopsis['guides']:
            return
        self.guide = guide
        self.cb_guide.blockSignals(True)
        self.cb_guide.setCurrentIndex(max(self.cb_guide.findData(guide), 0))
        self.cb_guide.blockSignals(False)
        self.draw_dossier()
        self.views.setCurrentIndex(self.views.indexOf(self.dossier_view))

    def _on_table_double_clicked(self, row: int, _col: int):
        item = self.sender().item(row, 0)
        if item is not None:
            self.show_dossier(item.text())

    def _connect_atlas(self, canvas, fig):
        if getattr(fig, '_atlas_points', None):
            canvas.mpl_connect('button_press_event', lambda event, fig=fig: self._on_atlas_click(event, fig))
            canvas.setToolTip('Double-click a cell to show that guide\'s dossier.')

    def _on_atlas_click(self, event, fig):
        if not (event.dblclick and event.button == 1) or event.x is None:
            return
        for ax, points in fig._atlas_points:
            if event.inaxes is not ax or not points:
                continue
            xy = ax.transData.transform([(x, y) for x, y, _g in points])
            dist = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
            i = int(np.argmin(dist))
            if dist[i] <= 14:
                self.show_dossier(points[i][2])
            return

    # -- export -------------------------------------------------------------------

    def _export_inputs(self) -> list:
        """(stats, shift, assemble arguments) per population, from results
        that are current (out-of-date ones are left out)."""
        page = self.page
        stats = page.stats if not page._stats_stale else None
        shift = page.shift_view.result if page.shift_view.exportable() else None
        if stats is None and shift is None:
            return []
        scopes, _notes = fsy.scopes(stats, shift, page.counted_population())
        return [(stats, shift, self._inputs(scope)) for scope in scopes]

    def export_plan(self, is_dark: bool = False) -> tuple:
        """
        ([(name, make)], write_tables) for the page's export: the atlas of
        every population and the dossier of every guide with both an
        accumulation and a phenotype call, then a function writing the
        tables into a folder.
        """
        maker = self._atlas_maker(is_dark)
        synopses = [fsy.assemble(stats, shift, **inputs) for stats, shift, inputs in self._export_inputs()]
        figures = []
        for synopsis in synopses:
            label = synopsis['scope']['label']
            figures.append((f'Synopsis {label} - atlas', lambda synopsis=synopsis: maker(synopsis)))
            table = synopsis['guide_table']
            for guide in table.loc[table['class'] == fsy.BOTH, 'guide']:
                figures.append((f'Synopsis {label} - {guide} dossier',
                                lambda synopsis=synopsis, guide=guide: fsyp.dossier_figure(synopsis, guide, is_dark,
                                                                                           '')))

        def write_tables(folder) -> list:
            return fsy.write_tables(folder, synopses)
        return figures, write_tables

    def settings_text(self) -> str:
        """Lines for the export's settings.txt."""
        jobs = self._export_inputs()
        if not jobs:
            return ''
        s = self.settings
        lines = ['', 'Synopsis']
        for stats, _shift, inputs in jobs:
            scope = inputs['scope']
            baseline = fsy.resolve_baseline(stats, scope['level'], s['baseline']) if stats is not None else None
            test = fsy.resolve_test(stats, s['test'])
            lines.append(f"  {scope['label']}: " + (f'cells against {baseline} ({fsy.TESTS[test]} test)'
                                                   if baseline and test else 'no statistics; phenotype only'))
        lines.append(f"  guide order: {fsy.ORDERS[s['order']].lower()}; calls as made by each test at its own "
                     'thresholds')
        return '\n'.join(lines)
