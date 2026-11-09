"""
flowcode_fsea_view.py
---------------------
The immune-state (Flow Set Enrichment Analysis) part of the FlowCode
*Phenotype shift* tab: its settings group and its views (state × guide
heatmap, state volcano, guide profile, a member per mouse, and two tables).
The scores are made in the phenotype shift's run, so ``FseaViews`` belongs
to a ``ShiftSection`` and uses its result, populations, groups and figure
task. Statistics (flowcode_fsea.py) and figures (flowcode_fsea_plots.py)
have no Qt. Not named ``*_tab.py``, so the plugin loader does not load it
as a tab of its own.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QSpinBox, QVBoxLayout, QWidget,
)

from honeychrome.view_components.copyable_table_widget import CopyableTableWidget

import flowcode_figure_panel as fcfp
import flowcode_fsea as ffsea
import flowcode_fsea_plots as ffp

logger = logging.getLogger(__name__)

TASK = 'analysis_shift_fsea'
SECTION = 'Immune states'
HEATMAP_KEY, HEATMAP_TITLE = 'fsea_heatmap', 'Immune-state heatmap'
AUTO = ''
NONE = '\x1fnone'
_HINT_STYLE = 'color: grey; font-style: italic;'

HELP = (
    '<p><b>Immune states</b> (views under <i>Immune states</i>; on by default, needs <i>Live unmix</i>). The '
    'shift says <i>whether</i> a guide\'s cells have changed kind; the immune states say <i>what kind</i> of change '
    'it is, in the terms of the Flow Set Enrichment library: for example "this guide pushes Tregs towards an '
    'effector state". An immune state is a short list of expected changes, e.g. effector Treg expansion: more '
    'effector Tregs; more CD44+, ICOS+, KLRG1+ and CD103+ cells among Tregs; fewer CD62L+. Each population is read '
    'as a cell type of the cell-type database (by its gate name, or as chosen in the settings), and only the rows '
    'that this panel can measure inside it are used. Each item is measured on the guide\'s cells from the same '
    'positivity scores as the phenotype populations, after the spillover correction: a cell type\'s share of the '
    'population, or the share of a marker\'s positive cells within a cell type. It is compared with the guide\'s '
    'reference in the same mouse (the reference set above). A state scores high when its items move the listed '
    'way together and consistently across mice.</p>'
    '<p><b>The statistic.</b> Per item, the per-mouse differences from the reference (on the logit scale) give a '
    'moderated t (Smyth 2004: each item\'s spread is pulled towards the spread typical of all items, which steadies '
    'it at 5–8 mice), turned into a z-score. A state\'s score is the mean of its items\' z, each signed by the '
    'direction the state lists: positive means towards the state, negative away from it. The p-value comes from a '
    '<i>rotation test</i> (limma\'s roast; Wu et al. 2010): the guide\'s per-mouse differences are rotated at '
    'random many times (9,999 by default), which keeps how items move together but removes any consistent '
    'direction, and the score is computed again each time. The p-values are adjusted by Benjamini–Hochberg over '
    'guides × states within each population and group. Each guide is scored twice: with every item, and without '
    'the items whose gene is the guide\'s own target (a Gata3 guide loses its GATA-3 item), because a knocked-out '
    'protein that disappears is already the marker test\'s result. The second is the main result. A dot in the '
    'heatmap marks a call that holds only with the target\'s items. In label shuffles on a Treg screen with 8 mice, '
    '4.6–4.9 % of tests reached p < 0.05, and a 2-fold rise of a common phenotype was found about half the time.</p>'
    '<p><b>Minimums.</b> An item is tested for a guide when, over its mice, the median of the smaller of its '
    '<i>expected</i> positive and negative cell counts is at least 5 (Minimum count), so an item never rests on '
    'one or two cells. Expected means what the reference predicts for the guide\'s cells (the reference\'s '
    'positive share × the guide\'s cells in that cell type), not what the guide shows, so a guide that removes '
    'nearly all of a rare positive population is still tested on it. '
    'A cell type\'s share needs at least two stained markers beyond the population, so a cell type does not rest '
    'on a single marker. A state needs 3 tested items; under 5 it is marked as small. Cells per mouse and mice are '
    'the shift\'s minimums.</p>'
    '<p><b>Views.</b> <i>Immune-state heatmap</i>: states × guides, coloured by score and ringed where significant. '
    'Double-click a cell to open that state\'s volcano and that guide\'s profile. <i>State volcano</i>: one '
    'state\'s guides, score against −log10 FDR. <i>Guide profile</i>: one guide\'s states, and each state\'s items '
    'as t × the listed direction (to the right: moved the way the state says; hatched: the guide\'s own target). '
    'Double-click an item for its values in every mouse beside the reference\'s. <i>Immune-state table</i>: every '
    'guide × state, with the shift angle beside it. <i>States</i>: per state, the rows the panel can score, the '
    'rows it cannot, and the calls.</p>'
    '<p><b>Glossary.</b> <i>Immune state</i>: a named programme in the library, a list of expected changes with '
    'cited sources. <i>Self-contained test</i>: asks whether the guide moved along the state at all (not whether '
    'it moved more than along other states). <i>Mixed</i>: the same rotation test of the mean z², asking whether '
    'the items changed in any direction. <i>Target rows</i>: a state\'s items whose gene is the guide\'s target.</p>'
)


def _no_wheel(*widgets):
    from flowcode_shift_view import _no_wheel as no_wheel
    no_wheel(*widgets)


def _fill(table, frame: pd.DataFrame, formats: dict | None = None):
    from flowcode_shift_view import _fill_table
    _fill_table(table, frame, formats)


class FseaViews:
    """The immune-state settings and views of *section* (a ``ShiftSection``)."""

    def __init__(self, section):
        self.section = section
        self.page = section.page
        self.state = None
        self.guide = None
        self._pending = False
        self.settings_box = self._build_settings()
        self.table = CopyableTableWidget([], [])
        self.table.setToolTip('Double-click a row to show that guide\'s profile and that state\'s volcano.')
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.table.view_section = SECTION
        self.states_table = CopyableTableWidget([], [])
        self.states_table.view_section = SECTION
        folder, options = self.page.analysis_root, self.page.figure_options_store
        self.heatmap = fcfp.FigurePanel(HEATMAP_TITLE, folder, options_store=options)
        self.heatmap.figure_key = 'fsea_heatmap'
        self.heatmap.view_section = SECTION
        self.heatmap.connect_canvas = self._connect_heatmap
        self.volcano = fcfp.FigurePanel('State volcano', folder, options_store=options)
        self.volcano.figure_key = 'fsea_volcano'
        self.volcano.connect_canvas = self._connect_volcano
        self.profile = fcfp.FigurePanel('Guide profile', folder, options_store=options)
        self.profile.figure_key = 'fsea_profile'
        self.profile.connect_canvas = self._connect_profile
        self.member = fcfp.FigurePanel('Member per mouse', folder, options_store=options)
        self.member.figure_key = 'fsea_member'
        self.member.view_section = SECTION
        self.cb_state = QComboBox()
        self.cb_guide = QComboBox()
        _no_wheel(self.cb_state, self.cb_guide)
        self.cb_state.currentIndexChanged.connect(lambda _i: self._pick_state(self.cb_state.currentData()))
        self.cb_guide.currentIndexChanged.connect(lambda _i: self._pick_guide(self.cb_guide.currentData()))
        self.volcano_view = self._with_picker('State:', self.cb_state, self.volcano)
        self.profile_view = self._with_picker('Guide:', self.cb_guide, self.profile)

    @staticmethod
    def _with_picker(text: str, combo: QComboBox, panel) -> QWidget:
        w = QWidget()
        w.view_section = SECTION
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(f'<b>{text}</b>'))
        row.addWidget(combo, stretch=1)
        lay.addLayout(row)
        lay.addWidget(panel, stretch=1)
        return w

    def add_views(self, views):
        """Add the views to the tab's view stack."""
        views.addTab(self.heatmap, HEATMAP_TITLE)
        views.addTab(self.volcano_view, 'State volcano')
        views.addTab(self.profile_view, 'Guide profile')
        views.addTab(self.member, 'Member per mouse')
        views.addTab(self.table, 'Immune-state table')
        views.addTab(self.states_table, 'States')

    # -- settings ---------------------------------------------------------------

    @property
    def settings(self) -> dict:
        return self.section.settings['fsea']

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox('Immune states (Flow Set Enrichment)')
        form = QFormLayout(box)
        self.chk_enabled = QCheckBox('Test immune states in the same run (live unmix)')
        self.chk_enabled.setToolTip('Score every guide against the immune states of the FSEA library that this '
                                    'panel can measure inside each population (see Help).')
        form.addRow(self.chk_enabled)
        self.cb_species = QComboBox()
        for s in ffsea.SPECIES:
            self.cb_species.addItem(s.capitalize(), s)
        self.cb_species.setToolTip('Which species\' cell types and immune states are used.')
        form.addRow('Library:', self.cb_species)
        self.roots_box = QWidget()
        self.roots_form = QFormLayout(self.roots_box)
        self.roots_form.setContentsMargins(0, 0, 0, 0)
        form.addRow('Populations as:', self.roots_box)
        self.list_states = QListWidget()
        self.list_states.setMaximumHeight(120)
        self.list_states.setToolTip('The immune states tested. States the panel cannot measure in any population '
                                    'are not tested whether ticked or not.')
        form.addRow('States:', self.list_states)
        self.sb_members = QSpinBox()
        self.sb_members.setRange(2, 30)
        self.sb_members.setToolTip('A state is tested for a guide only with at least this many tested items.')
        self.sb_flag = QSpinBox()
        self.sb_flag.setRange(2, 30)
        self.sb_flag.setToolTip('States tested with fewer items than this are marked as small.')
        row = QHBoxLayout()
        for text, w in (('members', self.sb_members), ('flag under', self.sb_flag)):
            row.addWidget(QLabel(text))
            row.addWidget(w)
        row.addStretch()
        form.addRow('Minimum:', row)
        self.sb_count = QDoubleSpinBox()
        self.sb_count.setRange(0.0, 1000.0)
        self.sb_count.setDecimals(1)
        self.sb_count.setToolTip('An item is tested for a guide when, over its mice, the median of the smaller of '
                                 'its expected positive and negative cell counts (from the reference\'s share) '
                                 'reaches this (0: no minimum).')
        self.sb_signs = QSpinBox()
        self.sb_signs.setRange(1, 6)
        self.sb_signs.setToolTip('A cell type\'s share is used only when it rests on at least this many stained '
                                 'markers beyond the population.')
        row = QHBoxLayout()
        for text, w in (('count', self.sb_count), ('cell-type markers', self.sb_signs)):
            row.addWidget(QLabel(text))
            row.addWidget(w)
        row.addStretch()
        form.addRow('Minimum:', row)
        self.sb_rotations = QSpinBox()
        self.sb_rotations.setRange(*ffsea.ROTATION_RANGE)
        self.sb_rotations.setSingleStep(1000)
        self.sb_rotations.setToolTip('Random rotations per guide for the p-values; the smallest possible p is '
                                     '1 / (rotations + 1).')
        form.addRow('Rotations:', self.sb_rotations)
        self.lbl_rows = QLabel('')
        self.lbl_rows.setStyleSheet(_HINT_STYLE)
        self.lbl_rows.setWordWrap(True)
        form.addRow(self.lbl_rows)
        _no_wheel(self.cb_species, self.sb_members, self.sb_flag, self.sb_count, self.sb_signs, self.sb_rotations)
        self.chk_enabled.toggled.connect(lambda v: self._set('enabled', bool(v)))
        self.cb_species.currentIndexChanged.connect(lambda _i: self._set('species', self.cb_species.currentData()))
        self.sb_members.valueChanged.connect(lambda v: self._set('min_members', int(v)))
        self.sb_flag.valueChanged.connect(lambda v: self._set('flag_members', int(v)))
        self.sb_count.valueChanged.connect(lambda v: self._set('min_count', float(v)))
        self.sb_signs.valueChanged.connect(lambda v: self._set('min_abund_signs', int(v)))
        self.sb_rotations.editingFinished.connect(lambda: self._set('rotations', int(self.sb_rotations.value())))
        self.list_states.itemChanged.connect(lambda _it: self._on_states_ticked())
        return box

    def _set(self, key, value):
        if self.section._filling or self.settings.get(key) == value:
            return
        self.settings[key] = value
        self.page._changed()
        if key == 'species':
            self.refresh()
        if key not in ffsea.SCORING_KEYS:
            self.start_reanalyse()
        self.section._after_setting()

    def _on_states_ticked(self):
        if self.section._filling:
            return
        items = [self.list_states.item(i) for i in range(self.list_states.count())]
        ticked = [it.data(Qt.UserRole) for it in items if it.checkState() == Qt.Checked]
        self._set('states', None if len(ticked) == len(items) else ticked)

    def _on_root_chosen(self, population: str, combo: QComboBox):
        if self.section._filling:
            return
        roots = dict(self.settings['roots'])
        value = combo.currentData()
        if value == AUTO:
            roots.pop(population, None)
        else:
            roots[population] = value
        self._set('roots', roots)

    def resolved_roots(self) -> dict:
        """{population: database cell type or None} for the ticked populations."""
        s = self.settings
        out = {}
        for pop in self.section.populations():
            chosen = s['roots'].get(pop, AUTO)
            if chosen == NONE:
                out[pop] = None
            elif chosen:
                out[pop] = chosen
            else:
                out[pop] = ffsea.default_root(pop, s['species'])
        return out

    def spec(self) -> dict:
        """The library rows each ticked population can score on this panel."""
        panel = self.section._panel()
        if panel is None:
            raise ValueError('the panel is not available')
        return ffsea.panel_library(panel, self.section._tag_fluors(), self.resolved_roots(), self.settings['species'])

    def refresh(self):
        """Bring the settings group up to date (called inside the section's
        filling guard)."""
        s = self.settings
        was = self.section._filling
        self.section._filling = True
        try:
            self.chk_enabled.setChecked(bool(s['enabled']))
            self.cb_species.setCurrentIndex(max(self.cb_species.findData(s['species']), 0))
            self.sb_members.setValue(int(s['min_members']))
            self.sb_flag.setValue(int(s['flag_members']))
            self.sb_count.setValue(float(s['min_count']))
            self.sb_signs.setValue(int(s['min_abund_signs']))
            self.sb_rotations.setValue(int(s['rotations']))
            while self.roots_form.rowCount():
                self.roots_form.removeRow(0)
            try:
                choices = ffsea.root_choices(s['species'])
                states = ffsea.library_states(s['species'])
            except Exception as exc:
                logger.warning('FlowCode immune states: library not available: %s', exc)
                choices, states = [], []
            for pop in self.section.populations():
                combo = QComboBox()
                auto = ffsea.default_root(pop, s['species'])
                combo.addItem(f'Automatic ({auto})' if auto else 'Automatic (no match: not tested)', AUTO)
                combo.addItem('Not tested', NONE)
                for c in choices:
                    combo.addItem(c, c)
                combo.setCurrentIndex(max(combo.findData(s['roots'].get(pop, AUTO)), 0))
                _no_wheel(combo)
                combo.currentIndexChanged.connect(lambda _i, pop=pop, combo=combo: self._on_root_chosen(pop, combo))
                self.roots_form.addRow(f'{pop}:', combo)
            self.list_states.clear()
            for state in states:
                it = QListWidgetItem(state)
                it.setData(Qt.UserRole, state)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if s['states'] is None or state in s['states'] else Qt.Unchecked)
                self.list_states.addItem(it)
        finally:
            self.section._filling = was
        self._update_rows_label()

    def _update_rows_label(self):
        if not self.settings['enabled']:
            self.lbl_rows.setText('Off.')
            return
        try:
            spec = self.spec()
        except Exception as exc:
            self.lbl_rows.setText(f'The library could not be matched to the panel: {exc}')
            return
        parts = []
        for pop, sp in spec['populations'].items():
            if sp['root'] is None:
                parts.append(f'{pop}: not tested')
                continue
            rows, _notes = ffsea.active_rows(sp, self.settings)
            counts = rows.groupby('state').size() if not rows.empty else pd.Series(dtype=int)
            ok = int((counts >= int(self.settings['min_members'])).sum())
            parts.append(f'{pop} as {sp["root"]}: {ok} state(s) with at least {self.settings["min_members"]} '
                         f'measurable items')
        self.lbl_rows.setText('; '.join(parts) + '.')

    # -- statistics again ------------------------------------------------------

    def start_reanalyse(self):
        """The stored scores tested again with the current settings."""
        result = self.section.result
        if result is None or not result.get('fsea'):
            return
        fsea, design = result['fsea'], result['design']
        settings = dict(self.settings)
        shift = dict(result['settings'])
        if self.page.plugin.task_running(TASK):
            self._pending = True
            return
        self._pending = False

        def work(_progress):
            return ffsea.reanalyse(fsea, design, settings, shift)

        if self.page.plugin._start_task(TASK, work):
            self.section.figure_msg.setText('Testing the immune states again…')

    def needs_reanalysis(self, result: dict) -> bool:
        fsea = result.get('fsea')
        if not fsea:
            return False
        stored = {k: v for k, v in fsea['settings'].items() if k not in ffsea.SCORING_KEYS}
        current = {k: v for k, v in self.settings.items() if k not in ffsea.SCORING_KEYS}
        return stored != current

    def on_finished(self, fsea: dict):
        if self._pending:
            self.start_reanalyse()
            return
        if self.section.result is None or not self.section.result.get('fsea'):
            return
        self.section.result = {**self.section.result, 'fsea': fsea}
        self.section.figure_msg.setText('')
        self.section.show_result()

    # -- views ------------------------------------------------------------------

    def clear(self):
        for table in (self.table, self.states_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        for panel in (self.heatmap, self.volcano, self.profile, self.member):
            panel.clear()
        for combo in (self.cb_state, self.cb_guide):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)

    def status_text(self, result: dict) -> str:
        fsea = result.get('fsea')
        if not fsea:
            return ''
        res = fsea['results']
        tested = res[np.isfinite(res['p'].astype(float))] if not res.empty else res
        return (f" Immune states: {len(tested)} guide × state test(s), {int(res['significant'].sum()) if len(res) else 0}"
                f" significant.")

    def notes(self, result: dict) -> list:
        fsea = result.get('fsea')
        return [f'Immune states: {n}' for n in fsea.get('notes') or []] if fsea else []

    def show(self, population, group):
        """Fill the tables and the picked-state and picked-guide figures."""
        result = self.section.result
        fsea = (result or {}).get('fsea')
        if not fsea:
            self.clear()
            return
        res = fsea['results']
        sub = res[(res['population'] == population) & (res['group'] == group)]
        shift = result['results']
        angle = shift[(shift['population'] == population) & (shift['group'] == group)].set_index('guide')['angle_deg']
        table = sub.assign(shift_angle=sub['guide'].map(angle)).sort_values('p')
        cols = ['guide', 'state', 'score', 'p', 'q', 'significant', 'members', 'q_mixed', 'score_all', 'q_all',
                'members_all', 'target_rows', 'depends_on_target', 'small_set', 'mice', 'shift_angle']
        _fill(self.table, table[cols], {'score': '{:+.2f}', 'score_all': '{:+.2f}', 'shift_angle': '{:+.1f}'})
        _fill(self.states_table, ffsea.states_table(fsea, population, group))
        self._fill_pickers(sub)
        self.draw_volcano()
        self.draw_profile()

    def _fill_pickers(self, sub: pd.DataFrame):
        tested = sub[np.isfinite(sub['score'].astype(float))]
        states = list(dict.fromkeys(sub['state']))
        calls = tested[tested['significant']]
        guides = list(dict.fromkeys(list(calls.sort_values('q')['guide']) + sorted(set(sub['guide']))))
        if self.state not in states:
            counts = calls.groupby('state').size()
            self.state = counts.idxmax() if len(counts) else (states[0] if states else None)
        if self.guide not in guides:
            self.guide = guides[0] if guides else None
        for combo, values, current in ((self.cb_state, states, self.state), (self.cb_guide, guides, self.guide)):
            combo.blockSignals(True)
            try:
                combo.clear()
                for v in values:
                    combo.addItem(str(v), v)
                if current is not None:
                    combo.setCurrentIndex(max(combo.findData(current), 0))
            finally:
                combo.blockSignals(False)

    def _pick_state(self, state):
        if state is not None and state != self.state:
            self.state = state
            self.draw_volcano()

    def _pick_guide(self, guide):
        if guide is not None and guide != self.guide:
            self.guide = guide
            self.draw_profile()

    def _context(self):
        result = self.section.result
        pop, group = self.section.current()
        return result, pop, group, float(result['settings']['fdr']), self.page.plugin.is_dark()

    def _put(self, panel, name: str, make):
        try:
            fig = make()
        except Exception as exc:
            panel.clear()
            self.section.figure_msg.setText(f'{name}: {exc}')
            return False
        fig, make = self.page._stamped(fig, make)
        panel.set_figure(fig, make=make, name=name)
        return True

    def draw_volcano(self):
        result, pop, group, fdr, dark = self._context()
        if not result.get('fsea') or self.state is None:
            self.volcano.clear()
            return
        res, state = result['fsea']['results'], self.state
        self._put(self.volcano, f'{pop} - {group} - immune state volcano',
                  lambda: ffp.state_volcano_figure(res, pop, group, state, fdr, dark, ''))

    def draw_profile(self):
        result, pop, group, fdr, dark = self._context()
        if not result.get('fsea') or self.guide is None:
            self.profile.clear()
            return
        fsea, guide = result['fsea'], self.guide
        rows = {s: ffsea.state_members(fsea, pop, s) for s in dict.fromkeys(fsea['results']['state'])} \
            if pop in fsea['spec']['populations'] else {}
        self._put(self.profile, f'{pop} - {group} - {guide} immune-state profile',
                  lambda: ffp.guide_profile_figure(fsea['results'], fsea['members'], rows, pop, group, guide, fdr,
                                                   dark, ''))

    def show_member(self, feature: str):
        result, pop, group, _fdr, dark = self._context()
        fsea, guide = result['fsea'], self.guide
        rows = fsea['spec']['populations'][pop]['rows']
        label = dict(zip(rows['feature'], rows['label'])).get(feature, feature)
        names = self.page._display_names()
        shift = dict(result['settings'])
        if self._put(self.member, f'{pop} - {group} - {guide} {label}',
                     lambda: ffp.member_figure(ffsea.member_values(fsea, result['design'], shift, pop, group, guide,
                                                                   feature),
                                               guide, label, pop, group, names, dark, '')):
            self.section.views.setCurrentIndex(self.section.views.indexOf(self.member))

    def heatmap_maker(self, result, pop, group, fdr, is_dark):
        """(key, make) of the heatmap for the section's figure task, or None."""
        if not result.get('fsea'):
            return None
        res = result['fsea']['results']
        return lambda: ffp.state_heatmap_figure(res, pop, group, fdr, is_dark, '')

    def export_figures(self, result, is_dark: bool) -> list:
        """(name, make) of each population's and group's heatmap and of the
        volcano of every state with a call, for the page's PDF report."""
        fsea = result.get('fsea')
        if not fsea:
            return []
        res = fsea['results']
        fdr = float(result['settings']['fdr'])
        out = []
        for (pop, group), sub in res.groupby(['population', 'group'], sort=False):
            out.append((f'Immune states {pop} {group} - heatmap',
                        lambda pop=pop, group=group: ffp.state_heatmap_figure(res, pop, group, fdr, is_dark, '')))
            for state in dict.fromkeys(sub.loc[sub['significant'], 'state']):
                out.append((f'Immune states {pop} {group} - {state} volcano',
                            lambda pop=pop, group=group, state=state: ffp.state_volcano_figure(
                                res, pop, group, state, fdr, is_dark, '')))
        return out

    def settings_lines(self, result) -> list:
        fsea = result.get('fsea')
        if not fsea:
            return []
        s = fsea['settings']
        lines = ['', 'Immune states (Flow Set Enrichment)',
                 f"  library: {fsea['spec'].get('species')} (SHA-256 {fsea['spec'].get('library', '')[:12]})",
                 "  populations as: " + ', '.join(f"{p} = {sp['root'] or 'not tested'}"
                                                    for p, sp in fsea['spec']['populations'].items()),
                 f"  states: {'all' if s['states'] is None else ', '.join(s['states'])}",
                 f"  minimum members {s['min_members']} (flagged under {s['flag_members']}), minimum count "
                 f"{s['min_count']:g}, cell-type markers {s['min_abund_signs']}, rotations {s['rotations']}",
                 '  main result without the guide\'s own target rows; BH over guides x states per population and '
                 'group']
        for note in fsea.get('notes') or []:
            lines.append(f'  note: {note}')
        cal = fsea.get('calibration')
        if cal is not None and not cal.empty:
            for r in cal.itertuples(index=False):
                lines.append(f'  calibration {r.population}, {r.group}: {r.pct_p_below_0_05:.1f}% of {r.tests} '
                             f'shuffled tests at p < 0.05; {r.shuffles_with_a_hit} of {r.shuffles} shuffles with '
                             'a call')
        return lines

    # -- clicks -----------------------------------------------------------------

    def _connect_heatmap(self, canvas, fig):
        if getattr(fig, '_points', None):
            canvas.mpl_connect('button_press_event', lambda e, fig=fig: self._on_point(e, fig, 'heatmap'))
            canvas.setToolTip('Double-click a cell to show that state\'s volcano and that guide\'s profile.')

    def _connect_volcano(self, canvas, fig):
        if getattr(fig, '_points', None):
            canvas.mpl_connect('button_press_event', lambda e, fig=fig: self._on_point(e, fig, 'volcano'))
            canvas.setToolTip('Double-click a guide to show its immune-state profile.')

    def _connect_profile(self, canvas, fig):
        if getattr(fig, '_member_features', None):
            canvas.mpl_connect('button_press_event', lambda e, fig=fig: self._on_member(e, fig))
            canvas.setToolTip('Double-click an item to show its value in every mouse.')

    def _on_point(self, event, fig, kind: str):
        if not (event.dblclick and event.button == 1) or event.inaxes is not getattr(fig, '_point_axes', None):
            return
        points = fig._points or []
        if not points or event.x is None:
            return
        xy = fig._point_axes.transData.transform([(x, y) for x, y, _k in points])
        dist = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(dist))
        if dist[i] > 14:
            return
        key = points[i][2]
        if kind == 'heatmap':
            self.guide, self.state = key
            self._sync_pickers()
            self.draw_volcano()
            self.draw_profile()
            self.section.views.setCurrentIndex(self.section.views.indexOf(self.profile_view))
        else:
            self.guide = key
            self._sync_pickers()
            self.draw_profile()
            self.section.views.setCurrentIndex(self.section.views.indexOf(self.profile_view))

    def _on_member(self, event, fig):
        if not (event.dblclick and event.button == 1) or event.inaxes is not getattr(fig, '_member_axes', None):
            return
        if event.ydata is None:
            return
        i = int(round(event.ydata))
        features = fig._member_features
        if 0 <= i < len(features):
            self.show_member(features[i])

    def _sync_pickers(self):
        for combo, value in ((self.cb_state, self.state), (self.cb_guide, self.guide)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(value), 0))
            combo.blockSignals(False)

    def _on_table_double_clicked(self, row: int, _col: int):
        guide, state = self.table.item(row, 0), self.table.item(row, 1)
        if guide is None or state is None:
            return
        self.guide, self.state = guide.text(), state.text()
        self._sync_pickers()
        self.draw_volcano()
        self.draw_profile()
        self.section.views.setCurrentIndex(self.section.views.indexOf(self.profile_view))
