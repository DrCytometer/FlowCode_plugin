"""
flowcode_shift_view.py
----------------------
The *Phenotype shift* tab of the FlowCode Analysis page: settings, the run,
saved results and the figures of the per-guide phenotype shift
(flowcode_phenotype_shift.py, figures in flowcode_shift_plots.py), and the
per-guide immune states made in the same run (flowcode_fsea_view.py). It
uses the page's samples (Data), populations (each ticked gate of Data) and
sample groups (Groups). Not named ``*_tab.py``, so the plugin loader does
not load it as a tab of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSpinBox, QTableWidgetItem, QVBoxLayout, QWidget,
)

from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.help_toggle_widget import WheelBlocker

import flowcode_analysis as fa
import flowcode_figure_panel as fcfp
import flowcode_fsea as ffsea
import flowcode_fsea_view as ffv
import flowcode_phenotype_shift as fsh
import flowcode_shift_plots as fsp
import flowcode_tag_correction as ftc
import flowcode_view_stack as fvs

logger = logging.getLogger(__name__)

TASK = 'analysis_shift'
TASK_CHECK = 'analysis_shift_check'
TASK_RESTORE = 'analysis_shift_restore'
TASK_FIGURES = 'analysis_shift_figures'
TASKS = (TASK, TASK_CHECK, TASK_RESTORE, TASK_FIGURES, ffv.TASK)
AUTO = ''
OFF = '\x1foff'

_ERROR_STYLE = 'color: #d62728;'
_DONE_STYLE = 'color: #2ca02c;'
_WARN_STYLE = 'color: #cc6600;'
_HINT_STYLE = 'color: grey; font-style: italic;'
STALE_TEXT = ('Out of date: these saved results were made with other samples, gates, unmixing, groups or settings. '
              'Run again to update them.')
FIGURES = (('shift_plot', 'Shift plot'), ('direction_map', 'Direction map'), ('heatmap', 'Heatmap'))
SUMMARY_KEY, SUMMARY_TITLE = 'summary_heatmap', 'Summary heatmap'
# Settings that only change the statistics: applied to the stored cluster
# counts again without clustering.
STATS_KEYS = ('reference', 'min_cells', 'min_mice', 'fdr')

HELP = (
    '<p><b>In short.</b> For every guide, asks whether the <i>mix of phenotypes</i> among its cells differs from '
    'what its tags predict in the same mouse and tissue, how large the shift is and towards which phenotypes. '
    'The counts test asks whether a guide has more or fewer cells, the marker test whether a marker is brighter or '
    'dimmer; this asks whether the guide\'s cells have become a different kind of cell.</p>'
    '<p><b>What to do here.</b> You do not need to count events in Data first: the run unmixes (or reads) the '
    'samples itself. Tick the populations in Data (each is clustered on its own) and set the sample groups in Groups (each '
    'group, e.g. a tissue, is tested on its own, with one sample per mouse). Choose the clustering channels and the '
    'number of clusters, then <b>Run phenotype shift</b>. <b>Check calibration</b> afterwards runs the statistics '
    'again on shuffled guide labels.</p>'
    '<p><b>How it works.</b> The barcoded cells of each population are clustered together (FlowSOM, a 10 × 10 grid '
    'of nodes merged into consensus metaclusters, on the display scale). Each guide in each mouse then has a '
    '<i>cluster fingerprint</i>: the share of its cells in each cluster. It is compared with a <i>reference</i> from '
    'the same mouse: by default the <i>tag model</i>, what the guide\'s three tags predict from the other guides of '
    'that mouse, so a tag\'s spillover that moves every combination carrying it is not taken for an effect (the '
    'control combinations or all other barcoded cells can be chosen instead). In one mouse, with 50–100 cells, '
    'chance alone moves a fingerprint by about 11–15°, so one mouse cannot show a shift. But chance points a '
    'different way in each mouse, while a real shift points the same way in every mouse: the test asks whether '
    'the mice agree.</p>'
    '<p><b>The statistic.</b> For each mouse, the <i>displacement</i> is the guide\'s cluster shares minus the '
    'reference\'s, each cluster divided by 2√(its pooled share), so rare clusters count as much as common ones. '
    'U is the mean, over every pair of different mice, of the product of their displacements: about 0 when the '
    'mice point in unrelated directions, positive when they agree. z = U / (its spread if each mouse\'s direction '
    'were flipped at random), read against a t distribution with (mice − 1) degrees of freedom; p-values are '
    'adjusted (Benjamini–Hochberg) within each population and group. On JA_007 label shuffles this gave '
    '2.7–3.6 % at p < 0.05, and detected 10 % of a guide\'s cells moving into one cluster about 80 % of the time '
    '(5–7 mice, 50–115 cells per guide per mouse). With few mice the smallest possible p is limited: z can be at '
    'most √(pairs of mice), so the smallest p is about 0.017 with 5 mice, 0.006 with 6, 0.002 with 7 and 0.0006 '
    'with 8.</p>'
    '<p><b>Glossary.</b> <i>Shift angle</i>: the size of the shift, 2·asin(√|U| / 2) in degrees, signed like U. '
    '0° is no shift; noise centres on 0 (negative when the mice disagree), and 90° would mean the guide\'s cells '
    'sit in entirely different clusters from the reference\'s. <i>Direction</i>: the cluster each guide gains and '
    'loses most, and per cluster the log2 ratio of the guide\'s mean share to the reference\'s. <i>Direction '
    'map</i>: guides on the first two principal components of their mean displacements; guides pointing the same '
    'way share a phenotype effect, and the arrows show which clusters each direction moves cells into. '
    '<i>Summary heatmap</i> (All groups): every guide\'s shift angle in every group (tissue), one row per group '
    '(and population), coloured where the guide\'s shift is significant and left blank otherwise, like the '
    'Analysis page\'s comparison heatmap, whose order and rows settings it follows: a guide that shifts in '
    'several tissues shows as a column of colour.</p>'
    '<p><b>Spillover correction</b> (on by default; needs <i>Live unmix</i>). A guide\'s tags cannot change where '
    'truly negative cells sit on a marker. If a combination\'s negative cells move with how bright its tags are, '
    'that movement is spillover or leftover FRET error. It is measured per sample group, combination and marker '
    '(pooled over the group\'s mice, up to 2,000 of the combination\'s cells per mouse in the <i>fit gate</i>, '
    'by default the viable gate) from the three tag amounts (abundance ÷ threshold) and their pairwise products, '
    'with a robust fit that follows the main (negative) population, and subtracted cell by cell before the '
    'populations are gated and clustered; cells without tags are untouched. A marker is corrected only when at '
    'least 30 % of the untransduced cells of the <i>eligibility gate</i> are negative (by default the deepest gate '
    'that still holds 80 % of the barcoded cells, e.g. T cells), and a combination\'s marker only when its tag '
    'dependence is significant (p below the evidence threshold, 0.01), so estimation noise is not added where '
    'there is nothing to correct. <i>Anchored markers</i> (Foxp3 and Helios by default) are fitted on the '
    'combination\'s CD8 T cells, where they are absent, and applied everywhere; without a CD8 gate the anchoring '
    'is off. The coefficients are listed in the <i>Correction</i> view. The correction is used by this readout '
    'only.</p>'
    '<p><b>Kept events (speed).</b> The first run reads (with Live unmix: unmixes) every sample once and keeps only '
    'what this readout needs: the barcoded events and up to 50,000 untransduced ones inside the deepest gate above '
    'the populations that the correction cannot change (one drawn on scatter, the viability dye or tags, e.g. the '
    'viable gate): on OTB01 a quarter of each sample\'s events, about 35 MB. These are kept in the experiment\'s '
    '<tt>cache/flowcode_analysis/shift_events</tt> folder, and later runs read only them. A sample is read in full '
    'again only when its file, the unmixing, or that gate and the gates above it change.</p>'
    '<p><b>Saved results.</b> Each run is saved when it finishes and shown again when the experiment is reopened. '
    'It is marked <i>Out of date</i> when the samples, gates, unmixing, groups or settings have changed since. '
    'Double-click a guide in the table, shift plot or direction map to show its cluster composition in every '
    'mouse beside its reference\'s.</p>'
) + ffv.HELP


def _label(text: str = '', style: str = _HINT_STYLE) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.PlainText)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(style)
    return lbl


def _no_wheel(*widgets):
    for w in widgets:
        w.setFocusPolicy(Qt.StrongFocus)
        w.installEventFilter(WheelBlocker(w))


def _item(text) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


def _fill_table(table, frame: pd.DataFrame, formats: dict | None = None):
    formats = formats or {}
    table.setSortingEnabled(False)
    table.setUpdatesEnabled(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    try:
        cols = list(frame.columns)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([str(c) for c in cols])
        table.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples(index=False)):
            for j, c in enumerate(cols):
                v = row[j]
                if c in formats and isinstance(v, (float, np.floating, int, np.integer)):
                    text = formats[c].format(v)
                elif isinstance(v, (float, np.floating)):
                    text = '' if not np.isfinite(v) else f'{v:.3g}'
                else:
                    text = str(v)
                table.setItem(r, j, _item(text))
        table.resizeColumnsToContents()
    finally:
        table.setSortingEnabled(True)
        table.setUpdatesEnabled(True)


class ShiftSection(QWidget):
    """The Phenotype shift tab; *page* is the ``AnalysisPage``."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.result = None
        self._saved_inputs = None
        self._stale = False
        self._restored = False
        self._token = None
        self._filling = False
        self._figures_pending = False
        self._composition = None
        self._panel_cache = None
        self.fsea = ffv.FseaViews(self)
        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.addWidget(fcfp.scrolling_help(HELP))
        row = QHBoxLayout()
        self.settings_box = self._build_settings()
        row.addWidget(self.settings_box, stretch=0)
        right = QVBoxLayout()
        self.settings_btn = QPushButton('Hide settings')
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip('Show or hide the settings, to give the figures more room.')
        self.settings_btn.toggled.connect(self._on_settings_toggled)
        shown = str(page.plugin.qsettings.value('shift_settings_shown', 'true')).lower() != 'false'
        self.settings_btn.setChecked(not shown)
        self.settings_box.setVisible(shown)
        self.run_btn = QPushButton('Run phenotype shift')
        self.run_btn.setToolTip('Unmix (or read) every used sample, correct it, gate the populations, cluster them '
                                'and test every guide, and its immune states when they are on. Holds the heavy-job '
                                'gate.')
        self.run_btn.clicked.connect(self.start)
        self.check_btn = QPushButton('Check calibration (5 label shuffles)')
        self.check_btn.setToolTip('Shuffle the guide labels among the barcoded cells of each sample five times and '
                                  'test again: about 5 % of tests (or fewer) should reach p < 0.05.')
        self.check_btn.clicked.connect(self.start_check)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.check_btn)
        buttons.addStretch()
        buttons.addWidget(self.settings_btn)
        right.addLayout(buttons)
        self.status = _label('')
        right.addWidget(self.status)
        self.check_status = _label('')
        right.addWidget(self.check_status)
        self.notes = _label('')
        right.addWidget(self.notes)
        right.addStretch()
        row.addLayout(right, stretch=1)
        outer.addLayout(row)
        self.stale_label = _label(STALE_TEXT, _WARN_STYLE)
        self.stale_label.setVisible(False)
        outer.addWidget(self.stale_label)
        pick = QHBoxLayout()
        pick.addWidget(QLabel('<b>Population:</b>'))
        self.cb_population = QComboBox()
        pick.addWidget(self.cb_population, stretch=1)
        pick.addWidget(QLabel('<b>Group:</b>'))
        self.cb_group = QComboBox()
        pick.addWidget(self.cb_group, stretch=1)
        self.views = fvs.ViewStack(lambda w: getattr(w, 'view_section', ''))
        pick.addWidget(QLabel('<b>View:</b>'))
        pick.addWidget(self.views.selector, stretch=1)
        _no_wheel(self.cb_population, self.cb_group)
        self.cb_population.currentIndexChanged.connect(lambda _i: self.show_result())
        self.cb_group.currentIndexChanged.connect(lambda _i: self.show_result())
        outer.addLayout(pick)
        self.table = CopyableTableWidget([], [])
        self.table.setToolTip('Double-click a guide to show its cluster composition in every mouse.')
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.table.view_section = 'Results'
        self.views.addTab(self.table, 'Table')
        self.panels = {}
        for key, title in FIGURES:
            panel = fcfp.FigurePanel(title, self.page.analysis_root,
                                     options_store=self.page.figure_options_store)
            panel.view_section = 'Results'
            if key in ('shift_plot', 'direction_map'):
                panel.connect_canvas = self._connect_clicks
            self.panels[key] = panel
            self.views.addTab(panel, title)
        self.composition_panel = fcfp.FigurePanel('Composition', self.page.analysis_root,
                                                  options_store=self.page.figure_options_store)
        self.composition_panel.view_section = 'Results'
        self.views.addTab(self.composition_panel, 'Composition')
        summary = fcfp.FigurePanel(SUMMARY_TITLE, self.page.analysis_root,
                                   options_store=self.page.figure_options_store)
        summary.figure_key = 'shift_summary_heatmap'
        summary.view_section = 'All groups'
        self.panels[SUMMARY_KEY] = summary
        self.views.addTab(summary, SUMMARY_TITLE)
        self.correction_table = CopyableTableWidget([], [])
        self.correction_table.view_section = 'Details'
        self.views.addTab(self.correction_table, 'Correction')
        self.cluster_table = CopyableTableWidget([], [])
        self.cluster_table.view_section = 'Details'
        self.views.addTab(self.cluster_table, 'Clusters')
        self.fsea.add_views(self.views)
        self.panels[ffv.HEATMAP_KEY] = self.fsea.heatmap
        outer.addWidget(self.views, stretch=1)
        self.figure_msg = _label('')
        outer.addWidget(self.figure_msg)

    # -- settings ---------------------------------------------------------------

    def _build_settings(self) -> QWidget:
        box = QGroupBox('Settings')
        box.setMaximumWidth(520)
        form = QFormLayout(box)
        self.defaults_btn = QPushButton('Restore defaults')
        self.defaults_btn.setToolTip('Put every phenotype-shift setting back to its default value.')
        self.defaults_btn.clicked.connect(self._restore_defaults)
        form.addRow(self.defaults_btn)
        self.lbl_scope = _label('')
        form.addRow(self.lbl_scope)
        self.cb_reference = QComboBox()
        for key, text in fsh.REFERENCES.items():
            self.cb_reference.addItem(text, key)
        self.cb_reference.setToolTip('What each guide\'s cluster fingerprint is compared with in its mouse. Tag '
                                     'model (recommended): what its own tags predict from the other guides of the '
                                     'mouse. Control combinations: those ticked for Control normalisation '
                                     '(Statistics). All other barcoded cells.')
        form.addRow('Reference:', self.cb_reference)
        self.sb_k = QSpinBox()
        self.sb_k.setRange(*fsh.K_RANGE)
        self.sb_k.setToolTip('Number of consensus metaclusters (phenotypes) the 10 × 10 FlowSOM grid is merged '
                             'into, 5 to 20.')
        self.lbl_k = _label('')
        k_row = QHBoxLayout()
        k_row.addWidget(self.sb_k)
        k_row.addWidget(self.lbl_k, stretch=1)
        form.addRow('Clusters:', k_row)
        self.list_channels = QListWidget()
        self.list_channels.setMaximumHeight(130)
        self.list_channels.setToolTip('Clustering channels. By default the panel\'s markers without the tags, the '
                                      'viability dye, congenic markers and the channels of the gates that define the '
                                      'populations.')
        form.addRow('Channels:', self.list_channels)
        self.chk_correction = QCheckBox('Correct tag spillover before clustering (live unmix)')
        self.chk_correction.setToolTip('Subtract each combination\'s brightness-dependent tag spillover from its '
                                       'cells\' markers before the populations are gated and clustered (see Help).')
        form.addRow(self.chk_correction)
        self.cb_fit_gate = QComboBox()
        self.cb_fit_gate.setToolTip('The combination\'s cells the correction is fitted on (up to 2,000 per mouse).')
        form.addRow('Fit gate:', self.cb_fit_gate)
        self.cb_eligibility_gate = QComboBox()
        self.cb_eligibility_gate.setToolTip('A marker is corrected only when at least 30 % of the untransduced cells '
                                            'of this gate are negative. Automatic: the deepest gate above the '
                                            'populations that still holds 80 % of the barcoded cells of the fit '
                                            'gate.')
        form.addRow('Eligibility gate:', self.cb_eligibility_gate)
        self.list_anchored = QListWidget()
        self.list_anchored.setMaximumHeight(90)
        self.list_anchored.setToolTip('Markers whose negatives are better seen in the anchor gate (Foxp3 and Helios '
                                      'by default, absent from CD8 T cells): fitted there, applied everywhere.')
        form.addRow('Anchored markers:', self.list_anchored)
        self.cb_anchor_gate = QComboBox()
        self.cb_anchor_gate.setToolTip('Where the anchored markers are fitted. Automatic: the CD8 T-cell gate; '
                                       'anchoring is off when there is none.')
        form.addRow('Anchor gate:', self.cb_anchor_gate)
        self.sb_evidence = QDoubleSpinBox()
        self.sb_evidence.setDecimals(4)
        self.sb_evidence.setRange(0.0001, 1.0)
        self.sb_evidence.setSingleStep(0.005)
        self.sb_evidence.setToolTip('A combination\'s marker is corrected only when the p-value of its tag '
                                    'dependence is below this (1 corrects every fit).')
        form.addRow('Evidence p below:', self.sb_evidence)
        self.sb_min_cells = QSpinBox()
        self.sb_min_cells.setRange(1, 100000)
        self.sb_min_cells.setToolTip('A guide (or its reference) with fewer cells in a mouse has no value there.')
        self.sb_min_mice = QSpinBox()
        self.sb_min_mice.setRange(3, 1000)
        self.sb_min_mice.setToolTip('A guide is tested in a group only with at least this many mice.')
        self.sb_fdr = QDoubleSpinBox()
        self.sb_fdr.setDecimals(3)
        self.sb_fdr.setRange(0.001, 1.0)
        self.sb_fdr.setSingleStep(0.01)
        self.sb_fdr.setToolTip('A guide is significant when its Benjamini–Hochberg q is below this.')
        mins = QHBoxLayout()
        for text, w in (('cells per mouse', self.sb_min_cells), ('mice', self.sb_min_mice), ('FDR', self.sb_fdr)):
            mins.addWidget(QLabel(text))
            mins.addWidget(w)
        mins.addStretch()
        form.addRow('Minimum:', mins)
        _no_wheel(self.cb_reference, self.sb_k, self.cb_fit_gate, self.cb_eligibility_gate, self.cb_anchor_gate,
                  self.sb_evidence, self.sb_min_cells, self.sb_min_mice, self.sb_fdr)
        self.cb_reference.currentIndexChanged.connect(
            lambda _i: self._set('reference', self.cb_reference.currentData()))
        self.sb_k.valueChanged.connect(lambda v: self._set('k', int(v)))
        self.chk_correction.toggled.connect(lambda v: self._set('correction', bool(v)))
        for key, combo in (('fit_gate', self.cb_fit_gate), ('eligibility_gate', self.cb_eligibility_gate),
                           ('anchor_gate', self.cb_anchor_gate)):
            combo.currentIndexChanged.connect(lambda _i, key=key, combo=combo: self._set(key, combo.currentData()))
        self.sb_evidence.valueChanged.connect(lambda v: self._set('evidence_p', float(v)))
        self.sb_min_cells.valueChanged.connect(lambda v: self._set('min_cells', int(v)))
        self.sb_min_mice.valueChanged.connect(lambda v: self._set('min_mice', int(v)))
        self.sb_fdr.valueChanged.connect(lambda v: self._set('fdr', float(v)))
        self.list_channels.itemChanged.connect(lambda _it: self._set_list('channels', self.list_channels))
        self.list_anchored.itemChanged.connect(lambda _it: self._set_list('anchor_markers', self.list_anchored))
        form.addRow(self.fsea.settings_box)
        return box

    def _on_settings_toggled(self, hidden: bool):
        self.settings_box.setVisible(not hidden)
        self.settings_btn.setText('Show settings' if hidden else 'Hide settings')
        self.page.plugin.qsettings.setValue('shift_settings_shown', 'false' if hidden else 'true')

    @property
    def settings(self) -> dict:
        """The saved settings (``state.analysis['shift']``), defaults filled in."""
        s = self.page.settings
        merged = fsh.shift_settings(s.get('shift'))
        if s.get('shift') != merged:
            s['shift'] = merged
        return s['shift']

    def _set(self, key, value):
        if self._filling:
            return
        if self.settings.get(key) == value:
            return
        self.settings[key] = value
        self.page._changed()
        if key in STATS_KEYS and self.result is not None:
            self._reanalyse()
        self._after_setting()

    def _set_list(self, key, widget: QListWidget):
        if self._filling:
            return
        ticked = [widget.item(i).data(Qt.UserRole) for i in range(widget.count())
                  if widget.item(i).checkState() == Qt.Checked]
        self._set(key, ticked)

    def _after_setting(self):
        self.defaults_btn.setEnabled(self.settings != fsh.shift_settings(None))
        self._update_stale()
        self._update_scope()

    def _reanalyse(self):
        """The stored cluster counts tested again with the current
        statistics settings (no clustering)."""
        result = self.result
        settings = dict(result['settings'])
        settings.update({k: self.settings[k] for k in STATS_KEYS})
        if settings == result['settings']:
            return
        try:
            results, direction, notes = fsh.analyse(result['counts'], result['design'], settings)
        except Exception as exc:
            self.figure_msg.setText(f'The statistics could not be run again: {exc}')
            return
        kept = [n for n in result.get('notes') or [] if n not in result.get('stat_notes', [])]
        self.result = {**result, 'settings': settings, 'results': results, 'direction': direction,
                       'notes': kept + notes, 'stat_notes': notes, 'calibration': None}
        self._set_result(self.result, self._saved_inputs, fresh=False, when='statistics updated')
        self.fsea.start_reanalyse()

    def _restore_defaults(self):
        self.page.settings['shift'] = fsh.shift_settings(None)
        self.page._changed()
        self.refresh()

    # -- what the run uses --------------------------------------------------------

    def populations(self) -> list:
        return list(self.page.settings.get('gates') or [])

    def sample_groups(self) -> dict:
        """{used sample: group}: the Groups section's groups; every sample in
        one group when none is assigned."""
        keys = list(self.page.used_files())
        groups = self.page.settings.get('sample_groups') or {}
        if not any(groups.get(k) for k in keys):
            return {k: fsh.ALL_SAMPLES for k in keys}
        return {k: groups[k] for k in keys if groups.get(k)}

    def _panel(self):
        try:
            return self.page._panel()
        except Exception as exc:
            logger.warning('FlowCode phenotype shift: panel not available: %s', exc)
            return None

    def _tag_fluors(self) -> list:
        design = self.page._current_design()
        return list(design.fluors) if design is not None else []

    def _channel_labels(self) -> dict:
        panel = self._panel()
        if panel is None:
            return {}
        return {f: (f'{panel.antigens.get(f)} ({f})' if panel.antigens.get(f) else f) for f in panel.fluorophores}

    def _short_labels(self) -> dict:
        panel = self._panel()
        if panel is None:
            return {}
        return {f: (panel.antigens.get(f) or f) for f in panel.fluorophores}

    def resolve(self, snapshot=None) -> dict:
        """The run's resolved choices from the settings and the experiment:
        'populations', 'channels', 'fit_gate', 'candidates', 'eligibility_gate',
        'anchor_gate', 'anchored', 'marker_fluors' and 'notes'."""
        s = self.settings
        snapshot = snapshot or self.page._gating_snapshot()
        pops = self.populations()
        panel = self._panel()
        fluors = list(panel.fluorophores) if panel is not None else []
        viability = list(panel.viability) if panel is not None else []
        tags = self._tag_fluors()
        congenic = []
        if panel is not None:
            import flowcode_phenotype as fp
            for name in fp.CONGENIC_MARKERS:
                congenic.extend(panel.markers.get(name, ()))
        defaults = fsh.default_channels(fluors, panel.antigens if panel else {}, tags, viability, congenic,
                                        fsh.population_channels(snapshot, pops))
        channels = [c for c in (s['channels'] if s['channels'] is not None else defaults) if c in fluors]
        notes = []
        fit = s['fit_gate'] or fsh.viable_gate(snapshot, pops, viability)
        if not s['fit_gate'] and fit is None:
            notes.append('No viable gate was found above the populations: the correction is fitted on every event '
                         '(set a Fit gate).')
        candidates = fsh.eligibility_candidates(snapshot, pops, fit)
        if s['anchor_gate'] == OFF:
            anchor = None
        else:
            anchor = s['anchor_gate'] or fsh.anchor_gate(snapshot.gate_names())
            if anchor is None:
                notes.append('No CD8 T-cell gate was found, so the anchored markers are not corrected (choose an '
                             'Anchor gate to fit them there).')
        markers = [f for f in fluors if f not in tags and f not in viability]
        anchored = [f for f in (s['anchor_markers'] if s['anchor_markers'] is not None
                                else fsh.default_anchored(panel.markers if panel else {})) if f in markers]
        return {'populations': pops, 'channels': channels, 'default_channels': defaults, 'fit_gate': fit,
                'candidates': candidates, 'eligibility_gate': s['eligibility_gate'] or '',
                'anchor_gate': anchor, 'anchored': anchored if anchor else [], 'anchored_wanted': anchored,
                'marker_fluors': markers, 'notes': notes}

    def inputs_digest(self, snapshot=None) -> str:
        """What a run depends on: the used samples and their files, the
        populations and the gating, the unmixing, the groups, the library
        setup, the control combinations and the settings."""
        files = self.page.used_files()
        snapshot = snapshot or self.page._gating_snapshot()
        live = self.page.is_live()
        unmixing = self.page.plugin.live_unmixing_signature(list(files)) if live else None
        settings = {k: v for k, v in self.settings.items() if k not in STATS_KEYS and k != 'fsea'}
        settings['fsea'] = ffsea.scoring_key(self.settings['fsea'])
        if settings['fsea']['enabled']:
            settings['fsea']['resolved_roots'] = self.fsea.resolved_roots()
        payload = {'counts': fa.counts_signature(files, self.populations(), snapshot.signature(), unmixing),
                   'groups': self.sample_groups(), 'settings': settings, 'live': live,
                   'library': self.page._library_setup_key(),
                   'controls': list(self.page.settings.get('control_combos') or [])}
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    # -- filling ------------------------------------------------------------------

    def reset_session(self):
        self.result = None
        self._stale = False
        self._restored = False
        self._composition = None
        self.status.setText('')
        self.check_status.setText('')
        self.notes.setText('')
        self.stale_label.setVisible(False)
        self._clear_views()

    def refresh(self):
        """Bring the controls up to date with the experiment and the page."""
        try:
            snapshot = self.page._gating_snapshot()
            resolved = self.resolve(snapshot)
            gate_names = list(snapshot.gate_names())
        except Exception as exc:
            logger.warning('FlowCode phenotype shift: settings could not be resolved: %s', exc)
            return
        s = self.settings
        labels = self._channel_labels()
        self._filling = True
        try:
            self.cb_reference.setCurrentIndex(max(self.cb_reference.findData(s['reference']), 0))
            self.sb_k.setValue(int(s['k']))
            self.chk_correction.setChecked(bool(s['correction']))
            self.sb_evidence.setValue(float(s['evidence_p']))
            self.sb_min_cells.setValue(int(s['min_cells']))
            self.sb_min_mice.setValue(int(s['min_mice']))
            self.sb_fdr.setValue(float(s['fdr']))
            panel = self._panel()
            tags = set(self._tag_fluors())
            self.list_channels.clear()
            for f in (panel.fluorophores if panel else ()):
                if f in tags:
                    continue
                it = QListWidgetItem(labels.get(f, f))
                it.setData(Qt.UserRole, f)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if f in resolved['channels'] else Qt.Unchecked)
                self.list_channels.addItem(it)
            self.list_anchored.clear()
            for f in resolved['marker_fluors']:
                it = QListWidgetItem(labels.get(f, f))
                it.setData(Qt.UserRole, f)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if f in resolved['anchored_wanted'] else Qt.Unchecked)
                self.list_anchored.addItem(it)
            fit_auto = resolved['fit_gate'] if not s['fit_gate'] else None
            for combo, key, auto_text, extra in (
                    (self.cb_fit_gate, 'fit_gate',
                     f'Automatic (viable gate: {fit_auto})' if fit_auto else 'Automatic (no viable gate: every event)',
                     ()),
                    (self.cb_eligibility_gate, 'eligibility_gate',
                     'Automatic (holds 80 % of the barcoded cells)', ()),
                    (self.cb_anchor_gate, 'anchor_gate',
                     f"Automatic (CD8 gate: {fsh.anchor_gate(gate_names)})" if fsh.anchor_gate(gate_names)
                     else 'Automatic (no CD8 gate: off)', ((OFF, 'Off'),))):
                combo.clear()
                combo.addItem(auto_text, AUTO)
                for data, text in extra:
                    combo.addItem(text, data)
                for g in gate_names:
                    combo.addItem(g, g)
                combo.setCurrentIndex(max(combo.findData(s[key]), 0))
            self.fsea.refresh()
        finally:
            self._filling = False
        self.defaults_btn.setEnabled(s != fsh.shift_settings(None))
        self._update_scope(resolved)
        self._restore_saved()
        self._update_stale()
        self._update_buttons()

    def _update_scope(self, resolved=None):
        try:
            resolved = resolved or self.resolve()
        except Exception:
            return
        pops = resolved['populations']
        present = set(self.sample_groups().values())
        groups = ([g for g in self.page.settings.get('group_names') or [] if g in present]
                  + sorted(present - set(self.page.settings.get('group_names') or [])))
        text = (f"Populations (Data): {', '.join(pops) if pops else 'none ticked'}. "
                f"Groups: {', '.join(groups) if groups else 'none'}.")
        if not self.page.is_live() and self.settings['correction']:
            text += ' The correction needs Live unmix (Data); with exported files it is skipped.'
        if self.settings['correction'] and resolved['notes']:
            text += ' ' + ' '.join(resolved['notes'])
        self.lbl_scope.setText(text)
        self.fsea._update_rows_label()

    def _update_buttons(self):
        running = any(self.page.plugin.task_running(t) for t in (TASK, TASK_CHECK))
        self.run_btn.setEnabled(not running and bool(self.populations()))
        self.check_btn.setEnabled(not running and self.result is not None and not self._stale)

    def _update_stale(self):
        if self.result is None:
            self._stale = False
        else:
            try:
                self._stale = self._saved_inputs != self.inputs_digest()
            except Exception:
                self._stale = True
        self.stale_label.setVisible(self._stale)
        self._update_buttons()

    # -- run ------------------------------------------------------------------------

    def _job(self) -> tuple:
        """(job, load, token, description) on the main thread; raises
        ValueError with the reason when the run cannot start."""
        page, plugin = self.page, self.page.plugin
        files = page.used_files()
        groups = self.sample_groups()
        keys = [k for k in files if k in groups]
        if not keys:
            raise ValueError('No used sample has a group.')
        snapshot = page._gating_snapshot()
        resolved = self.resolve(snapshot)
        if not resolved['populations']:
            raise ValueError('Tick at least one population (gate) in Data.')
        if not resolved['channels']:
            raise ValueError('Choose at least one clustering channel.')
        design = page._current_design()
        if design is None:
            raise ValueError('The FlowCode tags and combinations are not set.')
        settings = dict(self.settings)
        live = page.is_live()
        notes = []
        if settings['correction'] and not live:
            settings['correction'] = False
            notes.append('The spillover correction was skipped: it needs Live unmix.')
        tag_fluor = dict(zip(design.tags, design.fluors))
        combo_fluors = {n: tuple(tag_fluor[t] for t in tags) for n, tags in zip(design.combo_names, design.combo_tags)}
        thresholds = {}
        fc = plugin.saved_spectra()
        if fc is not None:
            thresholds = {f: float(t) for f, t in zip(fc.design.fluors, np.asarray(fc.thresholds, dtype=float))}
        if settings['correction'] and not thresholds:
            settings['correction'] = False
            notes.append('The spillover correction was skipped: no saved characterisation (section 4) gives the '
                         'tag thresholds.')
        feature_of = library_of = None
        if page.multi_library():
            fmap = page.feature_map() or {}
            library_of = {k: page.library_of(k) for k in keys}
            feature_of = {k: dict(fmap.get(library_of[k]) or {}) for k in keys}
        if live:
            inputs = plugin.live_unmix_inputs(keys)
            load, model, source_keys = self._live_loader(inputs, {k: files[k] for k in keys})
            fluors = list(inputs['layout'].fluor_names)
            what = f'Unmixing where needed ({inputs["plan"].label}),'
        else:
            load, model = self._file_loader(files), (lambda _key: None)
            source_keys = {k: json.dumps([str(files[k]), fa.file_signature(files[k])]) for k in keys}
            fluors = list(resolved['marker_fluors'])
            what = 'Reading where needed,'
        cache = {'dir': fsh.events_path(page._controller().experiment_dir), 'source_keys': source_keys,
                 'keep': fsh.UNTRANSDUCED_KEEP}
        spec = None
        if settings['fsea']['enabled']:
            if not live:
                notes.append('The immune states were not tested: they need Live unmix (the negative model).')
            else:
                try:
                    spec = self.fsea.spec()
                except Exception as exc:
                    notes.append(f'The immune states were not tested: {exc}')
                if spec is not None and not spec['markers']:
                    notes.append('The immune states were not tested: no library row can be measured in the '
                                 'populations on this panel.')
                    notes += list(spec['notes'])
        job = {'samples': keys, 'group_of': groups, 'populations': resolved['populations'], 'snapshot': snapshot,
               'transforms': snapshot.transforms, 'channels': resolved['channels'], 'settings': settings,
               'combo_fluors': combo_fluors, 'combo_tags': dict(zip(design.combo_names, design.combo_tags)),
               'thresholds': thresholds, 'fluors': fluors,
               'marker_fluors': [f for f in resolved['marker_fluors'] if f in fluors],
               'af_channel': 'AF Abundance', 'negative_model': model, 'fit_gate': resolved['fit_gate'],
               'eligibility_candidates': resolved['candidates'], 'eligibility_gate': resolved['eligibility_gate'],
               'anchor_gate': resolved['anchor_gate'], 'anchored': resolved['anchored'], 'feature_of': feature_of,
               'library_of': library_of, 'triplet_tags': dict(zip(design.combo_names, design.combo_tags)),
               'tag_names': list(design.tags), 'controls': list(page.settings.get('control_combos') or []),
               'fsea': spec}
        return job, load, cache, notes + (resolved['notes'] if settings['correction'] else []), what

    def _live_loader(self, inputs: dict, files: dict) -> tuple:
        import flowcode_export as fce
        import flowcode_phenotype as fp

        unmixing = inputs['signature']
        cache_keys = {k: fce.live_cache_key(unmixing, k, fce.file_signature(p)) for k, p in files.items()}
        fluor_names = list(inputs['layout'].fluor_names)

        def load(key):
            events = fce.live_sample_events(inputs['layout'], inputs['af_by_sample'][key], inputs['unmix_fn'], key,
                                            inputs['pnn'], inputs['n_extra'], cache_key=cache_keys[key],
                                            max_cache_bytes=inputs['max_cache_bytes'])
            return np.asarray(events, dtype=np.float64), list(inputs['pnn'])

        def model(key):
            return fp.sample_negative_model(inputs['af_by_sample'][key], fluor_names, inputs['plan'].state, key)
        return load, model, cache_keys

    @staticmethod
    def _file_loader(files: dict):
        def load(key):
            from honeychrome.controller_components.functions import sample_from_fcs
            sample = sample_from_fcs(str(files[key]))
            return np.asarray(sample.get_events(source='raw'), dtype=np.float64), list(sample.pnn_labels)
        return load

    def start(self):
        try:
            job, load, cache, notes, what = self._job()
        except Exception as exc:
            self.status.setStyleSheet(_ERROR_STYLE)
            self.status.setText(f'Cannot run: {exc}')
            return
        gate = getattr(self.page._controller(), 'job_gate', None)
        token = None
        if gate is not None:
            token = gate.try_acquire('FlowCode phenotype shift')
            if token is None:
                busy = getattr(gate, 'busy_text', None)
                self.status.setText(busy() if callable(busy) else 'Another job is running.')
                return
        inputs = self.inputs_digest(job['snapshot'])
        path = fsh.result_path(self.page._controller().experiment_dir)

        def work(progress):
            result = fsh.run(job, load, progress, cache)
            result['notes'] = notes + result['notes']
            try:
                fsh.save_result(path, result, inputs)
            except Exception as exc:
                logger.exception('FlowCode phenotype shift: could not save to %s', path)
                result['notes'].append(f'The results could not be saved: {exc}')
            return {'result': result, 'inputs': inputs}

        if self.page.plugin._start_task(TASK, work):
            self._token = token
            self.status.setStyleSheet(_HINT_STYLE)
            self.status.setText(f'{what} correcting, clustering and testing {len(job["samples"])} sample(s)…')
            self._update_buttons()
        elif token is not None:
            token.release()

    def _release(self):
        token, self._token = self._token, None
        if token is not None:
            token.release()

    def start_check(self):
        if self.result is None:
            return
        result = self.result
        inputs = self._saved_inputs
        path = fsh.result_path(self.page._controller().experiment_dir)

        def work(progress):
            calibration = fsh.shuffle_check(result, progress=progress)
            saved = {**result, 'calibration': calibration}
            fsea_calibration = None
            if result.get('fsea'):
                fsea_calibration = ffsea.shuffle_check(result['fsea'], result['design'], result['settings'],
                                                       progress=progress)
                saved['fsea'] = {**result['fsea'], 'calibration': fsea_calibration}
            try:
                fsh.save_result(path, saved, inputs)
            except Exception:
                logger.exception('FlowCode phenotype shift: could not save the calibration check')
            return calibration, fsea_calibration

        if self.page.plugin._start_task(TASK_CHECK, work):
            self.check_status.setText('Checking calibration…')
            self._update_buttons()

    def _restore_saved(self):
        """Once per experiment: the last saved run, marked out of date when
        what it was made from has changed."""
        if self._restored or self.result is not None:
            return
        self._restored = True
        try:
            path = fsh.result_path(self.page._controller().experiment_dir)
        except Exception:
            return
        if not path.exists():
            return

        def work(_progress):
            return fsh.load_result(path)

        if not self.page.plugin._start_task(TASK_RESTORE, work):
            self._restored = False

    # -- task results -------------------------------------------------------------

    def on_finished(self, key, result):
        if key == TASK:
            self._release()
            self._set_result(result['result'], result['inputs'], fresh=True)
        elif key == TASK_CHECK:
            calibration, fsea_calibration = result
            if self.result is not None:
                self.result['calibration'] = calibration
                if self.result.get('fsea'):
                    self.result['fsea'] = {**self.result['fsea'], 'calibration': fsea_calibration}
            self.check_status.setText(self._calibration_text(calibration, fsea_calibration))
        elif key == TASK_RESTORE:
            if result is None or self.result is not None:
                return
            res, inputs, when = result
            self._set_result(res, inputs, fresh=False, when=when)
            if any(res['settings'].get(k) != self.settings[k] for k in STATS_KEYS):
                self._reanalyse()
            elif self.fsea.needs_reanalysis(res):
                self.fsea.start_reanalyse()
        elif key == TASK_FIGURES:
            self._on_figures(result)
        elif key == ffv.TASK:
            self.fsea.on_finished(result)
        self._update_buttons()

    def on_error(self, key, message):
        if key == TASK:
            self._release()
            self.status.setStyleSheet(_ERROR_STYLE)
            self.status.setText(f'Phenotype shift failed: {message}')
        elif key == TASK_CHECK:
            self.check_status.setText(f'Calibration check failed: {message}')
        elif key == TASK_RESTORE:
            logger.warning('FlowCode phenotype shift: could not read the saved results: %s', message)
        elif key == TASK_FIGURES:
            self.figure_msg.setText(f'Figures failed: {message}')
        elif key == ffv.TASK:
            self.figure_msg.setText(f'The immune states could not be tested again: {message}')
        self._update_buttons()

    def on_progress(self, key, n, total, text):
        if key == TASK and text:
            self.status.setText(f'Phenotype shift: {text}… ({n + 1}/{total})')
        elif key == TASK_CHECK and text:
            self.check_status.setText(f'Checking calibration: {text} of {total}…')

    def _set_result(self, result: dict, inputs: str, fresh: bool, when: str | None = None):
        self.result = result
        self._saved_inputs = inputs
        self._update_stale()
        res = result['results']
        n_sig = int(res['significant'].sum()) if not res.empty else 0
        head = 'Done' if fresh else f'Saved results ({when})'
        self.status.setStyleSheet(_DONE_STYLE if fresh else _HINT_STYLE)
        text = (f"{head}: {len(res)} guide test(s) in {len(result['populations'])} population(s), "
                f"{n_sig} significant (q < {result['settings']['fdr']:g}).{self.fsea.status_text(result)}")
        if self.page.uncorrected():
            self.status.setStyleSheet('color: #d62728; font-weight: bold;')
            text += ' (no FRET correction)'
        self.status.setText(text)
        corr = result.get('correction')
        note_lines = list(result.get('notes') or []) + self.fsea.notes(result)
        if corr:
            table = corr['table']
            applied = int((table['status'] == ftc.STATUS_OK).sum()) if not table.empty else 0
            gates = corr['gates']
            where = {'root': 'every event'}
            note_lines.insert(0, f"Correction: {applied} of {len(table)} combination × marker fits applied "
                                 f"(fit gate {where.get(gates['fit'], gates['fit'])}, eligibility gate "
                                 f"{where.get(gates['eligibility'], gates['eligibility'])}, anchor gate "
                                 f"{gates['anchor'] or 'off'}).")
        kept = result.get('kept')
        if kept and kept.get('events'):
            share = [kept['kept'][s] / max(kept['events'][s], 1) for s in kept['events']]
            note_lines.insert(0, f"Kept events: inside {'every event' if kept['gate'] == 'root' else kept['gate']}, "
                                 f"median {100 * float(np.median(share)):.1f}% of each sample; "
                                 f"{kept['read']} sample(s) read in full this run, the others from the cache.")
        self.notes.setText('\n'.join(note_lines))
        self.check_status.setText(self._calibration_text(result.get('calibration'),
                                                         (result.get('fsea') or {}).get('calibration')))
        suggested = {p: c.get('suggested_k') for p, c in result['clusters'].items()}
        self.lbl_k.setText('Suggested (consensus stability): ' + ', '.join(
            f'{p} {k}' for p, k in suggested.items() if k) if any(suggested.values()) else '')
        self._fill_choices()
        self.show_result()

    @staticmethod
    def _calibration_text(calibration, fsea_calibration) -> str:
        text = fsp.calibration_text(calibration) or 'No test in the shuffles.'
        if fsea_calibration is not None and not fsea_calibration.empty:
            text += '\nImmune states: ' + fsp.calibration_text(fsea_calibration).replace('significant guide',
                                                                                     'call')
        return text if calibration is not None else ''

    def _fill_choices(self):
        result = self.result
        for combo, values in ((self.cb_population, result['populations']),
                              (self.cb_group, list(dict.fromkeys(result['design']['group_of'][s]
                                                                 for s in result['design']['samples'])))):
            current = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                for v in values:
                    combo.addItem(str(v), v)
                if current:
                    combo.setCurrentIndex(max(combo.findText(current), 0))
            finally:
                combo.blockSignals(False)

    # -- views --------------------------------------------------------------------

    def _clear_views(self):
        for table in (self.table, self.correction_table, self.cluster_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        for panel in list(self.panels.values()) + [self.composition_panel]:
            panel.clear()
        self.fsea.clear()
        self.figure_msg.setText('')

    def current(self) -> tuple:
        return self.cb_population.currentData(), self.cb_group.currentData()

    def _k(self, population) -> int:
        return int(self.result['counts'][population].shape[2])

    def _descriptions(self, population) -> dict:
        return fsp.cluster_descriptions(self.result['clusters'][population]['medians'], self._short_labels())

    def show_result(self):
        if self.result is None:
            self._clear_views()
            return
        pop, group = self.current()
        if pop is None or group is None:
            return
        res = self.result['results']
        view = res[(res['population'] == pop) & (res['group'] == group)].sort_values('p')
        cols = ['guide', 'angle_deg', 'p', 'q', 'significant', 'gain_cluster', 'loss_cluster', 'mice', 'median_cells',
                'U', 'z']
        desc = self._descriptions(pop)
        shown = view[cols].copy()
        shown['gain_cluster'] = [desc.get(c, c) for c in shown['gain_cluster']]
        shown['loss_cluster'] = [desc.get(c, c) for c in shown['loss_cluster']]
        _fill_table(self.table, shown, {'angle_deg': '{:+.1f}', 'median_cells': '{:.0f}', 'mice': '{:d}'})
        corr = self.result.get('correction')
        if corr and not corr['table'].empty:
            t = corr['table']
            t = t[t['group'] == group]
            _fill_table(self.correction_table, t)
        else:
            self.correction_table.setRowCount(0)
            self.correction_table.setColumnCount(0)
        cl = self.result['clusters'][pop]
        med = cl['medians'].rename(columns=self._channel_labels())
        med.insert(0, 'events', cl['sizes'])
        med.insert(0, 'phenotype', [desc.get(c, c) for c in med.index])
        med.insert(0, 'cluster', list(med.index))
        _fill_table(self.cluster_table, med.reset_index(drop=True), {'events': '{:d}'})
        self.fsea.show(pop, group)
        self.start_figures()
        if self._composition is not None:
            self.show_composition(self._composition)

    def _run_label(self) -> str:
        return 'WARNING: no spillover performed! Results invalid!' if self.page.uncorrected() else ''

    def start_figures(self):
        if self.result is None:
            return
        if self.page.plugin.task_running(TASK_FIGURES):
            self._figures_pending = True
            return
        self._figures_pending = False
        result = self.result
        pop, group = self.current()
        if pop is None:
            return
        k = self._k(pop)
        desc = self._descriptions(pop)
        labels = self._short_labels()
        fdr = float(result['settings']['fdr'])
        is_dark = self.page.plugin.is_dark()
        makers = self._makers(result, pop, group, k, desc, labels, fdr, is_dark)
        makers[SUMMARY_KEY] = self._summary_maker(result, fdr, is_dark, self.page.settings)
        heatmap = self.fsea.heatmap_maker(result, pop, group, fdr, is_dark)
        if heatmap is not None:
            makers[ffv.HEATMAP_KEY] = heatmap
        else:
            self.fsea.heatmap.clear()

        def work(_progress):
            out = {}
            for key, make in makers.items():
                try:
                    out[key] = {'fig': make(), 'make': make, 'error': None}
                except Exception as exc:
                    out[key] = {'fig': None, 'make': make, 'error': str(exc)}
            return {'figures': out, 'shown': (pop, group)}

        if self.page.plugin._start_task(TASK_FIGURES, work):
            self.figure_msg.setText('Drawing figures…')

    @staticmethod
    def _makers(result, pop, group, k, desc, labels, fdr, is_dark) -> dict:
        res, direction = result['results'], result['direction']
        medians = result['clusters'][pop]['medians']
        return {
            'shift_plot': lambda: fsp.shift_figure(res, pop, group, k, is_dark, '', desc, fdr),
            'direction_map': lambda: fsp.direction_figure(direction, res, pop, group, k, is_dark, '', desc),
            'heatmap': lambda: fsp.heatmap_figure(direction, res, medians, pop, group, k, is_dark, '', labels),
        }

    @staticmethod
    def _summary_maker(result, fdr, is_dark, settings: dict):
        """The summary heatmap over every group and population, in the order
        and rows of the Analysis page's comparison heatmap settings."""
        groups = list(dict.fromkeys(result['design']['group_of'][s] for s in result['design']['samples']))
        order, rows = settings.get('heatmap_order', 'effect'), int(settings.get('heatmap_rows', 1))
        return lambda: fsp.summary_heatmap_figure(result['results'], list(result['populations']), groups, is_dark,
                                                  '', order, rows, fdr)

    def _on_figures(self, result):
        if self._figures_pending:
            self.start_figures()
            return
        if result['shown'] != self.current():
            self.start_figures()
            return
        errors = []
        for key, item in result['figures'].items():
            panel = self.panels[key]
            title = {SUMMARY_KEY: SUMMARY_TITLE, ffv.HEATMAP_KEY: ffv.HEATMAP_TITLE}.get(key) or dict(FIGURES)[key]
            if item['fig'] is None:
                errors.append(f"{title}: {item['error']}")
                panel.clear()
                continue
            pop, group = result['shown']
            panel.name = ('phenotype shift summary heatmap' if key == SUMMARY_KEY
                          else f'{pop} - {group} - immune-state heatmap' if key == ffv.HEATMAP_KEY
                          else f'{pop} - {group} - phenotype shift {title.lower()}')
            fig, make = self.page._stamped(item['fig'], item['make'])
            panel.set_figure(fig, make=make)
        self.figure_msg.setText('; '.join(errors))

    def _connect_clicks(self, canvas, fig):
        if getattr(fig, '_points', None):
            canvas.mpl_connect('button_press_event', lambda event, fig=fig: self._on_click(event, fig))
            canvas.setToolTip('Double-click a guide to show its cluster composition in every mouse.')

    def _on_click(self, event, fig):
        if not (event.dblclick and event.button == 1) or event.inaxes is not getattr(fig, '_point_axes', None):
            return
        points = fig._points or []
        if not points or event.x is None:
            return
        ax = fig._point_axes
        xy = ax.transData.transform([(x, y) for x, y, _g in points])
        dist = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(dist))
        if dist[i] <= 12:
            self.show_composition(points[i][2])

    def _on_table_double_clicked(self, row: int, _col: int):
        item = self.table.item(row, 0)
        if item is not None:
            self.show_composition(item.text())

    def show_composition(self, guide: str):
        """Show *guide*'s cluster composition per mouse in the Composition view."""
        if self.result is None:
            return
        pop, group = self.current()
        self._composition = guide
        result, k = self.result, self._k(pop)
        desc, is_dark = self._descriptions(pop), self.page.plugin.is_dark()
        names = self.page._display_names()

        def make():
            comp = fsh.composition(result, pop, group, guide)
            return fsp.composition_figure(comp, guide, pop, group, k, names, is_dark, '', desc)
        try:
            fig = make()
        except KeyError:
            self.composition_panel.clear()
            self.figure_msg.setText(f'{guide} is not in {pop}, {group}.')
            return
        except Exception as exc:
            self.figure_msg.setText(f'Composition: {exc}')
            return
        self.composition_panel.name = f'{pop} - {group} - {guide} composition'
        fig, make = self.page._stamped(fig, make)
        self.composition_panel.set_figure(fig, make=make)
        self.views.setCurrentIndex(self.views.indexOf(self.composition_panel))

    # -- export -------------------------------------------------------------------

    def exportable(self) -> bool:
        return self.result is not None and not self._stale

    def export_figures(self, is_dark: bool = False) -> list:
        """(name, make) of every population's and group's figures, for the
        page's PDF report."""
        if not self.exportable():
            return []
        result = self.result
        labels = self._short_labels()
        fdr = float(result['settings']['fdr'])
        out = [('Phenotype shift summary heatmap', self._summary_maker(result, fdr, is_dark, self.page.settings))]
        for pop in result['populations']:
            k = self._k(pop)
            desc = fsp.cluster_descriptions(result['clusters'][pop]['medians'], labels)
            for group in dict.fromkeys(result['design']['group_of'][s] for s in result['design']['samples']):
                makers = self._makers(result, pop, group, k, desc, labels, fdr, is_dark)
                for key, title in FIGURES:
                    out.append((f'Phenotype shift {pop} {group} - {title.lower()}', makers[key]))
        return out + self.fsea.export_figures(result, is_dark)

    def write_tables(self, folder) -> list:
        return fsh.write_tables(folder, self.result) if self.exportable() else []

    def settings_text(self) -> str:
        """Lines for the export's settings.txt."""
        if not self.exportable():
            return ''
        r = self.result
        s = r['settings']
        lines = ['', 'Phenotype shift', f"  populations: {', '.join(r['populations'])}",
                 f"  reference: {fsh.REFERENCES.get(s['reference'], s['reference'])}",
                 f"  clusters: {s['k']} (FlowSOM {fsh.SOM_X} x {fsh.SOM_Y}, consensus metaclusters)",
                 f"  channels: {', '.join(r['channels'])}",
                 f"  minimum cells per mouse {s['min_cells']}, mice {s['min_mice']}, FDR {s['fdr']}",
                 f"  spillover correction: {'on' if r.get('correction') else 'off'}"]
        corr = r.get('correction')
        if corr:
            g = corr['gates']
            lines.append(f"    fit gate {g['fit']}, eligibility gate {g['eligibility']}, anchor gate "
                         f"{g['anchor'] or 'off'}, anchored {', '.join(g['anchored']) or 'none'}, "
                         f"evidence p < {s['evidence_p']}")
        for note in r.get('notes') or []:
            lines.append(f'  note: {note}')
        cal = fsp.calibration_text(r.get('calibration'))
        if cal:
            lines += ['  calibration check:'] + [f'    {line}' for line in cal.splitlines()]
        lines += self.fsea.settings_lines(r)
        return '\n'.join(lines)
