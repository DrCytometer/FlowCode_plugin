"""
test_flowcode_tab.py
--------------------
The FlowCode plugin's widgets (flowcode_unmix_tab.py), sections 1 to 5, on
the synthetic experiment in flowcode_synthetic.py written to raw FCS files.

* The gate panel on its own: Honeychrome's CytometryPlotWidget on a private
  event bus and data dictionary, with a polygon edited, the whole FlowCode
  Cells gate scaled with its handle, an axis transform changed, a gate
  renamed and a gate deleted.
* The plugin end to end: the default and pooled backbones and the matched
  unstained, AF library, tags and combos, the combinations CSV, automatic
  scatter gates, the experiment-gate fallback, the debarcoding space and
  thresholds, run records, and restoring the session from the saved state.
* Section 4: the backbone characterisation, its saved files and figures,
  the linked biplots with barcoded events drawn as points, invalidation by a
  threshold edit, and rebuilding the figures in a new session.
* Section 5: the FlowCode export of two samples, control samples listed only
  on request, the written FCS channels and keywords, the summary files, and
  invalidation by a combo rename; the tandem tag check box, its backbone
  statistic and the tandem columns of the summary files.

Needs pytest-qt (pip install pytest-qt); without it the module is skipped.

The plugin lives outside the package; its folder defaults to
user_plugins/flowcode in this repository and can be set with
FLOWCODE_PLUGIN_DIR.

Usage:
    pytest tests/test_flowcode_tab.py -m ui
    HONEYCHROME_SHOW_UI=1 pytest tests/test_flowcode_tab.py -m ui -s     # watch it run
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pytestqt', reason='pytest-qt is not installed: pip install pytest-qt')

_REPO = Path(__file__).resolve().parents[1]
_FLOWCODE_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR', _REPO / 'user_plugins' / 'flowcode'))
if str(_FLOWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWCODE_DIR))
_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))
if os.environ.get('HONEYCHROME_SHOW_UI', '').lower() not in ('1', 'true', 'yes', 'on'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import flowcode_core as fcc  # noqa: E402
import flowcode_state as fcs  # noqa: E402
import flowcode_synthetic as syn  # noqa: E402

pytestmark = pytest.mark.ui

SINGLETS, CELLS = fcs.SINGLETS_GATE, fcs.CELLS_GATE
LONG = 120_000


class FlowCodeController:
    """The Controller attributes the FlowCode plugin reads."""

    def __init__(self, experiment_dir):
        from flowkit import Dimension, GatingStrategy, gates
        from honeychrome.controller_components.functions import (
            assign_default_transforms, generate_transformations,
        )

        raw = syn.raw_settings()
        unmixed = syn.unmixed_settings()
        self.experiment_dir = Path(experiment_dir)
        self.experiment = SimpleNamespace(
            experiment_path=str(self.experiment_dir / 'FlowCode test.kit'),
            settings={'raw': raw, 'unmixed': unmixed, 'unmixing_method': 'OLS'},
            samples=syn.samples(),
            process=syn.process(),
        )
        self.filtered_raw_fluorescence_channel_ids = list(raw['fluorescence_channel_ids'])
        self.raw_transformations = generate_transformations(assign_default_transforms(raw))
        self.unmixed_transformations = generate_transformations(assign_default_transforms(unmixed, n_af_spectra=3))
        self.autospectral_variants = {}
        self.autospectral_unmixed_pos_thresholds = None
        self.af_precomputed_cache = {}

        gating = GatingStrategy()
        for ch in syn.SCATTER:
            gating.transformations[ch] = self.raw_transformations[ch].xform
        dim_x = Dimension('FSC-A', range_min=0.1, range_max=0.5, transformation_ref='FSC-A')
        dim_y = Dimension('SSC-A', range_min=0.05, range_max=0.3, transformation_ref='SSC-A')
        gating.add_gate(gates.RectangleGate('Cells', dimensions=[dim_x, dim_y]), gate_path=('root',))
        self.raw_gating = gating

    def precomputed_for_profile(self, name):
        """Per-profile AF matrices; the synthetic profiles carry no negative envelope."""
        from honeychrome.controller_components.autospectral_functions import build_af_precomputed
        entry = self.experiment.process['af_profiles'].get(name)
        if entry is None:
            return None
        return build_af_precomputed(syn.spectra(), np.asarray(entry['spectra'], dtype=np.float64))

    def _build_fluor_spectra(self):
        return syn.spectra()

    def n_af_spectra(self):
        return 3


@pytest.fixture
def fp(qapp):
    import flowcode_unmix_tab as module
    module._ensure_qt_imports()
    return module


@pytest.fixture
def experiment(tmp_path):
    syn.write_experiment(tmp_path)
    return tmp_path


def _plugin(fp, qtbot, controller):
    from honeychrome.view_components.event_bus import EventBus

    bus = EventBus()
    plugin = fp.PluginWidget(bus=bus, controller=controller)
    qtbot.addWidget(plugin)
    plugin.resize(1400, 900)
    plugin.show()
    bus.modeChangeRequested.emit(fp.plugin_name)
    return plugin, bus


def _browser_requests(fp, qtbot, controller):
    """The plugin, opened, and the sample-browser visibility it asked for."""
    from honeychrome.view_components.event_bus import EventBus

    bus = EventBus()
    requests = []
    bus.sampleBrowserVisibilityRequested.connect(requests.append)
    plugin = fp.PluginWidget(bus=bus, controller=controller)
    qtbot.addWidget(plugin)
    bus.modeChangeRequested.emit(fp.plugin_name)
    return plugin, requests


def _check_pop_out(panel, qtbot):
    """Pop Out opens a window with a fresh copy of the panel's figure,
    redrawn at the window's size."""
    from PySide6.QtWidgets import QDialog

    panel.view.size_slider.setValue(50)
    qtbot.waitUntil(lambda: panel.canvas.width() < panel.figure.get_figwidth() * panel.figure.dpi * 0.6,
                    timeout=5000)
    panel.pop_out()
    dialogs = [w for w in panel.findChildren(QDialog) if w.isVisible()]
    assert len(dialogs) == 1
    canvas = dialogs[0].findChildren(type(panel.canvas))[0]
    assert canvas.figure is not panel.figure
    dialogs[0].resize(500, 400)
    qtbot.waitUntil(lambda: canvas.figure.get_figwidth() * canvas.figure.dpi < 500, timeout=5000)
    dialogs[0].close()


def _settle(plugin, qtbot):
    """Wait for every background task and its thread to finish."""
    qtbot.waitUntil(lambda: not plugin._tasks and all(t.isFinished() for t in plugin._threads), timeout=LONG)


def _backbone_items(plugin) -> dict:
    from PySide6.QtCore import Qt

    page = plugin.pages[1]
    return {page.backbone_list.item(i).data(Qt.UserRole): page.backbone_list.item(i)
            for i in range(page.backbone_list.count())}


def _choose_backbone(plugin, qtbot, keys=(syn.BACKBONE,)):
    """Tick exactly *keys* (experiment order) and wait for the matching."""
    from PySide6.QtCore import Qt

    for key, item in _backbone_items(plugin).items():
        want = Qt.Checked if key in keys else Qt.Unchecked
        if item.checkState() != want:
            item.setCheckState(want)
    qtbot.waitUntil(lambda: plugin.state.match_backbone_keys == list(keys)
                    and not plugin.task_running('match') and not plugin._change_pending, timeout=LONG)


def _load_gates(plugin, qtbot):
    plugin.tabs.setCurrentIndex(2)
    plugin.load_backbone()
    qtbot.waitUntil(lambda: plugin.backbone_data is not None and not plugin.task_running('auto_cells')
                    and CELLS in plugin.state.gates and len(plugin.pages[3].panel._plots) == 2, timeout=LONG)


def _confirm(plugin, qtbot):
    from PySide6.QtCore import Qt

    qtbot.mouseClick(plugin.pages[3].confirm_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: plugin.state.run_status(3, plugin.fingerprint())[0] == 'done'
                    and plugin.debarcode_space_current(), timeout=LONG)


# ---------------------------------------------------------------------------
# Gate panel on a private bus
# ---------------------------------------------------------------------------

def _panel_data(fp):
    bb = syn.make_events('backbone', 20000, 1)
    un = syn.make_events('unstained', 8000, 2)
    cols = [syn.RAW_PNN.index(ch) for ch in syn.SCATTER]
    return fp.BackboneData(
        backbone_keys=(syn.BACKBONE,), unstained_key=syn.UNSTAINED_MATCHED, scatter_channels=syn.SCATTER,
        scatter=bb['raw'][:, cols], fl=bb['raw'][:, 4:].astype(np.float32), n_file=20000,
        un_scatter=un['raw'][:, cols], un_fl=un['raw'][:, 4:].astype(np.float32), un_n_file=8000,
    )


def _panel(fp, qtbot, tmp_path):
    controller = FlowCodeController(tmp_path)
    transforms = {ch: controller.raw_transformations[ch] for ch in syn.SCATTER}
    records = {
        SINGLETS: fcs.gate_record('FSC-A', 'FSC-H', fcc.singlet_polygon(1.1, syn.MAGNITUDE_CEILING,
                                                                         syn.MAGNITUDE_CEILING), 'auto'),
        CELLS: fcs.gate_record('FSC-A', 'SSC-A', fcc.rectangle_vertices(6e5, 1.4e6, 3e5, 9e5), 'auto'),
    }
    panel = fp.GatePanel()
    qtbot.addWidget(panel)
    panel.resize(1100, 420)
    panel.show()
    data = _panel_data(fp)
    panel.set_backbone(data, records, transforms)
    return panel, data, records


def test_gate_panel_draws_both_gates_with_stats(fp, qtbot, tmp_path):
    panel, data, records = _panel(fp, qtbot, tmp_path)
    assert len(panel._plots) == 2
    singlet_plot, cell_plot = panel._plots
    assert singlet_plot.plot['child_gates'] == [SINGLETS]
    assert cell_plot.plot['source_gate'] == SINGLETS
    assert len(singlet_plot.rois) == 1 and len(cell_plot.rois) == 1
    hists = panel._data['histograms']
    assert len(hists) == 2 and hists[0].sum() > 0 and hists[1].sum() > 0
    for widget in (singlet_plot, cell_plot, panel.preview):
        assert (widget.width(), widget.height()) == (fp.GATE_PLOT_SIZE, fp.GATE_PLOT_SIZE)

    a, h, ssc = data.scatter.T
    singlets = fcc.points_in_polygon(a, h, records[SINGLETS]['vertices_raw'])
    cells = singlets & fcc.points_in_polygon(a, ssc, records[CELLS]['vertices_raw'])
    masks = panel.masks()
    np.testing.assert_array_equal(masks[SINGLETS], singlets)
    np.testing.assert_array_equal(masks[CELLS], cells)
    stats = panel._data['statistics']
    assert stats[CELLS]['n_events_gate'] == cells.sum()
    assert stats[CELLS]['p_gate_parent'] == pytest.approx(cells.sum() / singlets.sum())


def test_dragging_a_gate_updates_its_raw_vertices(fp, qtbot, tmp_path):
    from PySide6.QtCore import QPointF

    panel, data, records = _panel(fp, qtbot, tmp_path)
    before = np.asarray(panel.records[CELLS]['vertices_raw'])
    roi = panel._plots[1].rois[0]
    with qtbot.waitSignal(panel.gateEdited, timeout=5000) as blocker:
        roi.setPos(roi.pos() + QPointF(0.02, 0.0))
    assert blocker.args == [CELLS]
    after = np.asarray(panel.records[CELLS]['vertices_raw'])
    tr = panel._transforms['FSC-A']
    shift = tr.xform.inverse(np.array([0.52]))[0] - tr.xform.inverse(np.array([0.5]))[0]
    np.testing.assert_allclose(after[:, 0] - before[:, 0], shift, rtol=1e-6)
    np.testing.assert_allclose(after[:, 1], before[:, 1], rtol=1e-9)
    assert panel.records[CELLS]['source'] == 'manual'
    a, h, ssc = data.scatter.T
    expected = fcc.points_in_polygon(a, h, records[SINGLETS]['vertices_raw']) & \
        fcc.points_in_polygon(a, ssc, after)
    np.testing.assert_array_equal(panel.masks()[CELLS], expected)


def test_the_handle_scales_the_whole_cells_gate(fp, qtbot, tmp_path):
    panel, data, records = _panel(fp, qtbot, tmp_path)
    handle = panel._scale_handle
    assert handle is not None and handle.isVisible()
    shown = panel._display_vertices(panel.records[CELLS])
    centre = fcc.polygon_centroid(shown)
    start = np.array([handle.pos().x(), handle.pos().y()])
    assert np.min(np.linalg.norm(shown - start, axis=1)) < 1e-9
    n_before = int(panel.masks()[CELLS].sum())

    target = centre + 1.25 * (start - centre)
    handle.setPos(float(target[0]), float(target[1]))
    assert len(panel._scale_preview.getData()[0]) == len(shown) + 1
    np.testing.assert_allclose(panel.records[CELLS]['vertices_raw'], records[CELLS]['vertices_raw'])
    with qtbot.waitSignal(panel.gateEdited, timeout=5000) as blocker:
        handle.sigPositionChangeFinished.emit(handle)
    assert blocker.args == [CELLS]
    scaled = panel._display_vertices(panel.records[CELLS])
    np.testing.assert_allclose(scaled, centre + 1.25 * (shown - centre), rtol=1e-9, atol=1e-12)
    assert panel.records[CELLS]['source'] == 'manual' and '1.25x' in panel.records[CELLS]['note']
    assert int(panel.masks()[CELLS].sum()) > n_before
    gate_shown = np.asarray(panel._data['gating'].get_gate(CELLS).vertices, dtype=float)
    np.testing.assert_allclose(gate_shown, scaled, rtol=1e-9)
    assert len(panel._plots[1].rois) == 1
    moved = np.array([handle.pos().x(), handle.pos().y()])
    assert np.min(np.linalg.norm(scaled - moved, axis=1)) < 1e-9

    panel.scale_cells_gate(0.8)
    np.testing.assert_allclose(panel._display_vertices(panel.records[CELLS]), shown, rtol=1e-9, atol=1e-12)


def test_changing_an_axis_transform_keeps_the_gate_in_raw_units(fp, qtbot, tmp_path):
    panel, data, records = _panel(fp, qtbot, tmp_path)
    raw_before = np.asarray(panel.records[CELLS]['vertices_raw'])
    masks_before = panel.masks()[CELLS].copy()
    tr = panel._transforms['SSC-A']
    tr.set_transform(id=1)
    panel._bus.axisTransformed.emit('SSC-A')
    np.testing.assert_array_equal(np.asarray(panel.records[CELLS]['vertices_raw']), raw_before)
    shown = np.asarray(panel._data['gating'].get_gate(CELLS).vertices, dtype=float)
    np.testing.assert_allclose(shown[:, 1], tr.xform.apply(raw_before[:, 1]), rtol=1e-9)
    np.testing.assert_array_equal(panel.masks()[CELLS], masks_before)
    assert len(panel._plots[1].rois) == 1

    panel._bus.axesReset.emit(['SSC-A'])
    assert panel._transforms['SSC-A'].id == 0


def test_renaming_a_flowcode_gate_is_undone(fp, qtbot, tmp_path):
    panel, _data, _records = _panel(fp, qtbot, tmp_path)
    label = panel._plots[1].rois[0].label
    label.rename_gate('Something else')
    qtbot.waitUntil(lambda: panel._data is not None
                    and {g[0] for g in panel._data['gating'].get_gate_ids()} == {SINGLETS, CELLS}, timeout=5000)
    assert len(panel._plots) == 2


def test_deleting_a_flowcode_gate_asks_for_a_reset(fp, qtbot, tmp_path):
    panel, _data, _records = _panel(fp, qtbot, tmp_path)
    with qtbot.waitSignal(panel.gateResetRequested, timeout=5000) as blocker:
        panel._plots[0].rois[0].request_remove()
    assert blocker.args == [SINGLETS]


# ---------------------------------------------------------------------------
# Plugin, sections 1 to 3
# ---------------------------------------------------------------------------

def test_view_stack_lists_views_under_headings_and_steps(qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    import flowcode_view_stack as fvs

    stack = fvs.ViewStack(lambda widget: widget.section)
    qtbot.addWidget(stack)
    qtbot.addWidget(stack.selector)
    pages = {}
    for title, section in (('Table', 'This comparison'), ('Volcano', 'This comparison'),
                           ('Spider', 'Gene groups'), ('Members', 'Gene groups')):
        pages[title] = QLabel(title)
        pages[title].section = section
        stack.addTab(pages[title], title)
    combo = stack.combo
    assert [combo.itemText(r).strip() for r in range(combo.count())] == [
        'This comparison', 'Table', 'Volcano', 'Gene groups', 'Spider', 'Members']
    assert combo.model().item(0).flags() == Qt.NoItemFlags and combo.model().item(3).flags() == Qt.NoItemFlags
    assert stack.currentIndex() == 0 and combo.currentIndex() == 1
    assert not stack.btn_previous.isEnabled() and stack.btn_next.isEnabled()

    changed = []
    stack.currentChanged.connect(changed.append)
    stack.btn_next.click()
    stack.btn_next.click()
    assert stack.currentIndex() == 2 and stack.currentWidget() is pages['Spider'] and changed == [1, 2]
    assert combo.currentText().strip() == 'Spider'
    combo.setCurrentIndex(3)
    assert stack.currentIndex() == 2 and combo.currentText().strip() == 'Spider'
    combo.setCurrentIndex(5)
    assert stack.currentWidget() is pages['Members'] and not stack.btn_next.isEnabled()

    stack.removeTab(3)
    assert stack.count() == 3 and stack.currentWidget() is pages['Spider']
    combination = QLabel('Combination')
    combination.section = 'This comparison'
    stack.insertTab(1, combination, 'Combination')
    assert stack.currentWidget() is pages['Spider'] and stack.tabText(stack.currentIndex()) == 'Spider'
    assert stack.indexOf(pages['Spider']) == 3
    stack.removeTab(1)
    stack.removeTab(0)
    assert stack.count() == 2 and stack.tabText(0) == 'Volcano' and stack.currentWidget() is pages['Spider']


def test_plugin_is_disabled_without_an_unmixing_matrix(fp, qtbot, tmp_path):
    controller = FlowCodeController(tmp_path)
    controller.experiment.process['unmixing_matrix'] = None
    plugin, _bus = _plugin(fp, qtbot, controller)
    assert plugin.label_disabled.isVisible()
    assert not plugin.content.isVisible()


def test_the_tab_hides_the_sample_browser_once_usable(fp, qtbot, experiment, tmp_path):
    controller = FlowCodeController(tmp_path)
    controller.experiment.process['unmixing_matrix'] = None
    _plugin_off, requests = _browser_requests(fp, qtbot, controller)
    assert requests == []
    plugin, requests = _browser_requests(fp, qtbot, FlowCodeController(experiment))
    assert requests == [False]
    _settle(plugin, qtbot)


def test_sections_one_and_two(fp, qtbot, experiment):
    from PySide6.QtCore import Qt

    controller = FlowCodeController(experiment)
    controller.experiment.samples['all_samples']['Controls/A1 HA BV421 (Cells).fcs'] = 'A1 HA BV421 (Cells)'
    plugin, _bus = _plugin(fp, qtbot, controller)
    state = plugin.state
    assert state.backbones() == [syn.BACKBONE]          # the backbone-named sample is the default
    assert list(_backbone_items(plugin)) == [syn.BACKBONE, syn.OTHER_SAMPLE]
    _backbone_items(plugin)[syn.BACKBONE].setCheckState(Qt.Unchecked)
    qtbot.waitUntil(lambda: state.backbone_keys == [] and not plugin._change_pending, timeout=5000)
    assert plugin.section_status(1) == ('error', 'Choose at least one backbone sample.')
    assert plugin.section_status(2)[0] == 'waiting'

    _choose_backbone(plugin, qtbot)
    assert state.unstained_key == syn.UNSTAINED_MATCHED
    assert state.unstained_scores[syn.UNSTAINED_MATCHED] > 0.9 > state.unstained_scores[syn.UNSTAINED_SHIFTED]
    assert state.af_profile_names == [syn.AF_PROFILE]
    assert plugin.section_status(1) == ('done', 'Done')

    plugin.pages[1]._on_unstained_clicked(1, 0)
    qtbot.waitUntil(lambda: state.unstained_key == syn.UNSTAINED_SHIFTED, timeout=5000)
    assert state.af_profile_names == []
    assert 'No AF profile extracted' in plugin.section_status(1)[1]
    plugin.choose_best_unstained()
    assert state.unstained_key == syn.UNSTAINED_MATCHED and state.af_profile_names == [syn.AF_PROFILE]

    plugin.tabs.setCurrentIndex(1)
    assert state.tag_channels() == list(syn.TAG_FLUORS)
    assert [state.tag_names[f] for f in syn.TAG_FLUORS] == list(syn.TAGS)
    assert len(state.combos) == 10
    assert plugin.section_status(2) == ('done', 'Done')
    page = plugin.pages[2]
    assert page.combo_table.rowCount() == 10
    assert page.tag_table.rowCount() == 5

    page.combo_table.item(0, 0).setCheckState(Qt.Unchecked)
    qtbot.waitUntil(lambda: not state.combos[0]['include'], timeout=5000)
    assert len(state.included_combos()) == 9
    page.combo_table.item(1, 1).setText('Gata3')
    qtbot.waitUntil(lambda: state.combos[1]['name'] == 'Gata3', timeout=5000)
    page.combo_table.item(2, 1).setText('Gata3')
    qtbot.waitUntil(lambda: plugin.section_status(2)[0] == 'error', timeout=5000)
    assert 'used twice' in plugin.section_status(2)[1]
    page.combo_table.item(2, 1).setText('Tbet')
    qtbot.waitUntil(lambda: plugin.section_status(2)[0] == 'done', timeout=5000)

    page._checkboxes['BV421'].setChecked(False)
    qtbot.waitUntil(lambda: 'BV421' not in state.tag_channels() and not plugin._change_pending, timeout=5000)
    assert len(state.combos) == 4
    page._checkboxes['FITC'].setChecked(True)
    qtbot.waitUntil(lambda: 'FITC' in state.tag_channels() and not plugin._change_pending, timeout=5000)
    assert state.tag_names['FITC'] == 'CD4'
    page._on_auto_tags()
    assert state.tag_channels() == list(syn.TAG_FLUORS)
    kept = [row for row in state.combos if row['name'] == 'Gata3']
    assert kept, 'combo names survive a change of tag selection'

    saved = controller.experiment.process[fcs.PROCESS_KEY]
    json.dumps(saved)
    assert fcs.FlowCodeState.from_json(saved).backbone_keys == [syn.BACKBONE]
    _settle(plugin, qtbot)


def test_combinations_csv(fp, qtbot, experiment):
    controller = FlowCodeController(experiment)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    good = experiment / 'combinations.csv'
    good.write_text('Id,Procode.tag1,Procode.tag2,Procode.tag3\n'
                    'Gata3,HA,V5,AU1\nTbet,HA,S,C\nFoxp3,v5,S,C\n', encoding='utf-8')
    assert plugin.pages[1].load_csv(good)
    state = plugin.state
    assert [row['name'] for row in state.combos] == ['Gata3', 'Tbet', 'Foxp3']
    assert state.combos[2]['tags'] == ['V5', 'S', 'C']
    assert state.tag_channels() == list(syn.TAG_FLUORS)
    assert plugin.section_status(2) == ('done', 'Done')

    duplicate = experiment / 'duplicate.csv'
    duplicate.write_text('Id,a,b,c\nGata3,HA,V5,AU1\nGata3,HA,S,C\n', encoding='utf-8')
    assert not plugin.pages[1].load_csv(duplicate)
    assert 'used twice' in plugin.pages[1].csv_msg.text()
    assert [row['name'] for row in state.combos] == ['Gata3', 'Tbet', 'Foxp3']

    unknown = experiment / 'unknown.csv'
    unknown.write_text('Id,a,b,c\nGata3,HA,V5,Ollas\n', encoding='utf-8')
    assert plugin.pages[1].load_csv(unknown)
    assert plugin.section_status(2)[0] == 'error'
    assert any('Ollas' in e for e in plugin.tag_problems()[0])

    plugin.pages[1]._on_clear_csv()
    assert state.csv_combos is None
    assert state.tag_channels() == list(syn.TAG_FLUORS)
    assert plugin.section_status(2) == ('done', 'Done')
    _settle(plugin, qtbot)


def test_section_three_gates_thresholds_and_restore(fp, qtbot, experiment):
    from PySide6.QtCore import QPointF, Qt

    controller = FlowCodeController(experiment)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    assert plugin.section_status(3) == ('pending', 'Not run yet')
    _load_gates(plugin, qtbot)
    state = plugin.state
    assert state.gates[SINGLETS]['source'] == 'auto'
    assert state.gates[CELLS]['source'] == 'auto', state.gates[CELLS]['note']
    assert 'enlarged 1.5x' in state.gates[CELLS]['note']
    data = plugin.backbone_data
    assert data.n_clean == data.n_file == syn.SAMPLE_EVENTS[syn.BACKBONE][1]
    masks = plugin.pages[3].panel.masks()
    assert 0.8 < masks[SINGLETS].mean() < 0.9
    assert 0.1 < masks[CELLS].mean() < 0.6
    assert plugin.pages[3].panel.unstained_mask().sum() > 500

    _confirm(plugin, qtbot)
    assert plugin.pages[3].section_tabs.currentIndex() == fp.GatesPage.SECTION_THRESHOLDS
    assert set(state.thresholds) == set(syn.TAGS)
    for tag, rec in state.thresholds.items():
        assert 200 < rec['raw'] < 2000, (tag, rec)
        assert rec['method'] in ('mixture', 'tail', 'floor')
    assert plugin.debarcode_space.engine == 'per-cell AF'
    assert plugin.debarcode_space.rescue_floor_source == 'unstained'
    partners = plugin.tag_partners()
    assert all(state.y_channels[t] == state.tag_map()[partners[t]] for t in syn.TAGS)
    panel = plugin.pages[3].thresholds
    assert panel.table.rowCount() == 5 and panel.isEnabled()
    assert (panel.biplot.width(), panel.biplot.height()) == (fp.THRESHOLD_PLOT_SIZE, fp.THRESHOLD_PLOT_SIZE)
    saved_done = deepcopy(controller.experiment.process[fcs.PROCESS_KEY])

    # R thresholds file: matched by tag, then by fluorophore; imported values survive a re-confirm
    auto = {t: rec['raw'] for t, rec in state.thresholds.items()}
    r_file = experiment / 'thresholds.csv'
    r_file.write_text('"Fluor","Tag","Threshold_Raw","Threshold_Transformed"\n'
                      f'"BV421","HA",{auto["HA"] + 100},1\n"PE","",{auto["AU1"] + 50},1\n"X","Ollas",9,1\n')
    assert plugin.import_thresholds(r_file)
    assert state.thresholds['HA']['raw'] == pytest.approx(auto['HA'] + 100)
    assert state.thresholds['AU1']['method'] == 'imported'
    assert 'Ollas' in panel.import_msg.text() and panel.import_msg.isVisible()
    plugin.compute_debarcode_space(confirm=True)
    assert plugin.task_running('space')
    qtbot.waitUntil(lambda: not plugin.task_running('space') and plugin.debarcode_space_current(), timeout=LONG)
    assert state.thresholds['HA']['method'] == 'imported'
    assert state.thresholds['HA']['raw'] == pytest.approx(auto['HA'] + 100)
    plugin.on_thresholds_auto_all()

    inputs4 = state.section_inputs(4, plugin.fingerprint())
    panel.table.item(0, 3).setText('1234.5')
    qtbot.waitUntil(lambda: state.thresholds['HA']['raw'] == pytest.approx(1234.5), timeout=5000)
    assert state.thresholds['HA']['method'] == 'manual'
    assert plugin.section_status(3) == ('done', 'Done')
    assert state.section_inputs(4, plugin.fingerprint()) != inputs4
    panel.biplot.set_threshold(800.0)
    panel.biplot._on_line_moved()
    assert state.thresholds['HA']['raw'] == pytest.approx(800.0, rel=1e-6)
    plugin.on_thresholds_auto_all()
    assert state.thresholds['HA']['raw'] == state.thresholds['HA']['auto_raw']

    plugin.use_experiment_gate(CELLS)
    rec = state.gates[CELLS]
    assert rec['source'] == 'experiment'
    tr_x, tr_y = controller.raw_transformations['FSC-A'], controller.raw_transformations['SSC-A']
    v = np.asarray(rec['vertices_raw'])
    assert v[:, 0].min() == pytest.approx(tr_x.xform.inverse(np.array([0.1]))[0])
    assert v[:, 1].max() == pytest.approx(tr_y.xform.inverse(np.array([0.3]))[0])
    status, text = plugin.section_status(3)
    assert status == 'stale' and 'gates changed' in text
    assert not plugin.debarcode_space_current()

    plugin.auto_gate(CELLS)
    qtbot.waitUntil(lambda: not plugin.task_running('auto_cells') and state.gates[CELLS]['source'] == 'auto',
                    timeout=LONG)
    roi = plugin.pages[3].panel._plots[1].rois[0]
    roi.setPos(roi.pos() + QPointF(0.005, 0.0))
    qtbot.waitUntil(lambda: state.gates[CELLS]['source'] == 'manual', timeout=5000)
    assert plugin.section_status(3)[0] == 'stale'

    # A new session reads no events until Load events, which restores the
    # confirmed state without changing thresholds.
    controller2 = FlowCodeController(experiment)
    controller2.experiment.process[fcs.PROCESS_KEY] = saved_done
    plugin2, _bus2 = _plugin(fp, qtbot, controller2)
    assert plugin2.section_status(3) == ('done', 'Done')
    plugin2.tabs.setCurrentIndex(2)
    assert plugin2.backbone_data is None and not plugin2.task_running('load')
    qtbot.mouseClick(plugin2.pages[3].load_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: plugin2.debarcode_space_current(), timeout=LONG)
    assert plugin2.state.thresholds == saved_done['thresholds']
    assert plugin2.state.gates == saved_done['gates']
    _settle(plugin, qtbot)
    _settle(plugin2, qtbot)


def test_pooled_backbones_are_matched_and_loaded_together(fp, qtbot, experiment):
    controller = FlowCodeController(experiment)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    single = dict(plugin.state.unstained_scores)
    pooled = (syn.BACKBONE, syn.OTHER_SAMPLE)
    _choose_backbone(plugin, qtbot, pooled)
    assert plugin.state.backbones() == list(pooled)
    assert plugin.state.unstained_scores != single
    assert plugin.state.unstained_key == syn.UNSTAINED_MATCHED
    _load_gates(plugin, qtbot)
    data = plugin.backbone_data
    assert data.backbone_keys == pooled
    assert data.n_file == data.n_clean == sum(syn.SAMPLE_EVENTS[k][1] for k in pooled)
    assert 'Backbone, Mouse 1:' in plugin.pages[3].load_status.text()

    plugin.on_setting_changed('backbone_max_events', 4000)
    plugin.tabs.setCurrentIndex(2)
    plugin.load_backbone()
    qtbot.waitUntil(lambda: plugin.backbone_data is not None and not plugin.task_running('load')
                    and plugin.backbone_data.n_clean == 8000, timeout=LONG)
    assert plugin.backbone_data.n_file == 35000
    _settle(plugin, qtbot)


def test_spectral_model_change_marks_section_three(fp, qtbot, experiment):
    controller = FlowCodeController(experiment)
    plugin, bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    profiles = controller.experiment.process['profiles']
    profiles['BV421'] = (np.asarray(profiles['BV421']) * 0.98).tolist()
    original = controller._build_fluor_spectra
    controller._build_fluor_spectra = lambda: np.array([profiles[f] for f in syn.FLUORS])
    bus.spectralProcessRefreshed.emit()
    status, text = plugin.section_status(3)
    assert status == 'stale' and 'spectral model changed' in text
    controller._build_fluor_spectra = original
    _settle(plugin, qtbot)


# ---------------------------------------------------------------------------
# Section 4
# ---------------------------------------------------------------------------

def _characterise(plugin, qtbot):
    from PySide6.QtCore import Qt

    plugin.tabs.setCurrentIndex(3)
    qtbot.mouseClick(plugin.pages[4].run_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: plugin.section_status(4) == ('done', 'Done') and plugin.characterisation is not None
                    and not plugin.characterise_busy(), timeout=LONG)


def test_section_four_characterises_saves_and_restores(fp, qtbot, experiment):
    import csv

    from PySide6.QtCore import Qt

    controller = FlowCodeController(experiment)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    assert plugin.section_status(4) == ('waiting', 'Waiting for section 3')
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    assert plugin.section_status(4) == ('pending', 'Not run yet')
    _characterise(plugin, qtbot)

    state = plugin.state
    output = plugin.characterisation
    fc = output.result.spectra
    assert fc.design.combo_names == tuple(row['name'] for row in state.included_combos())
    assert all(e is not None for e in fc.combo_error), fcc.fit_table_rows(fc)
    summary = output.summary
    barcoded = sum(summary['per_combo'].values())
    assert barcoded > 0.6 * summary['total_events']
    assert summary['total_events'] == len(plugin.debarcode_space.backbone)
    assert len(output.after['unmixed']) == summary['total_events']
    assert len(output.before['unmixed']) == summary['total_events']
    # the figures' events go through the final unmix, without and with correction
    shown = np.isfinite(output.after['unmixed']).all(axis=1)
    assert shown[output.display_idx].all() and np.isfinite(output.before['unmixed'][shown]).all()
    assert all(e.shape == (6, len(fc.detectors)) for e in fc.combo_error)
    assert fc.pair_model is not None
    assert state.characterisation['error_model'] == fcc.ERROR_MODEL
    assert output.engine in plugin.pages[4].biplot_msg.text()

    cache = experiment / 'state' / fp.SPECTRA_FILE
    assert cache.exists()
    with open(experiment / fp.OUTPUT_SUBDIR / 'thresholds.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ['Fluor', 'Tag', 'Threshold_Raw', 'Threshold_Transformed']
    assert [(r[0], r[1]) for r in rows[1:]] == list(zip(syn.TAG_FLUORS, syn.TAGS))
    assert [float(r[2]) for r in rows[1:]] == [state.thresholds[t]['raw'] for t in syn.TAGS]
    with open(experiment / fp.OUTPUT_SUBDIR / 'combinations.csv', newline='') as fh:
        assert len(list(csv.reader(fh))) == 11
    saved = deepcopy(controller.experiment.process[fcs.PROCESS_KEY])
    json.dumps(saved)
    assert saved['characterisation']['fingerprint'] == fc.fingerprint()

    counts = plugin.combo_backbone_counts()
    assert sum(counts.values()) == barcoded
    plugin.tabs.setCurrentIndex(1)
    table = plugin.pages[2].combo_table
    assert table.item(0, 5).text() == f'{counts[state.combos[0]["name"]]:,}'
    plugin.tabs.setCurrentIndex(3)

    page = plugin.pages[4]
    assert page.section_tabs.count() == 6
    assert page.section_tabs.tabText(page.SECTION_ASSESS) == 'Assess correction'
    assert page.frequency.figure is not None and page.heatmap.figure is not None
    assert page.expression.figure is not None and page.expression.exportable
    assert len(page.expression.figure.axes) == 3 * len(syn.FLUORS)
    assert output.expression['n']['before'] == barcoded
    assert page.table.rowCount() == len(fc.design.combo_names)
    assert len(page.biplot_before._event_data) == len(output.display_idx)
    for plot in (page.biplot_before, page.biplot_after):
        assert (plot.width(), plot.height()) == (fp.ASSESS_PLOT_SIZE, fp.ASSESS_PLOT_SIZE)
    ids = np.asarray(output.result.ids)[output.display_idx]
    n_barcoded = int(((ids >= 1) & (ids <= fc.design.n_combos)).sum())
    assert n_barcoded > 0
    page.highlight_check.setChecked(True)
    for plot in (page.biplot_before, page.biplot_after):
        assert plot.highlighted_count() == n_barcoded
        assert len(plot._highlight_item.getData()[0]) == n_barcoded
    page.highlight_check.setChecked(False)
    assert len(page.biplot_before._highlight_item.getData()[0]) == 0
    page.highlight_check.setChecked(True)
    page.highlight_size.setValue(6)
    assert len(page.biplot_after._highlight_item.getData()[0]) == n_barcoded
    x, _y = page.biplot_before.channels()
    assert x in syn.TAG_FLUORS
    menu = page.biplot_before._menu_channels
    assert {'FITC', 'FSC-A', 'SSC-A', 'AF Abundance'} <= set(menu) and 'Time' not in menu
    page.biplot_before._set_channel_x(menu.index('SSC-A'), None)
    assert page.biplot_after.channels()[0] == 'SSC-A'
    after = page.biplot_after
    raw_ssc = after._event_data[after._highlight, syn.unmixed_pnn().index('SSC-A')]
    np.testing.assert_allclose(after._highlight_item.getData()[0],
                               fp._forward(after._transformations['SSC-A'])(raw_ssc), rtol=1e-6)
    ssc = syn.unmixed_pnn().index('SSC-A')
    assert np.all(page.biplot_before._event_data[:, ssc] > 0)
    page.heat_space.setCurrentIndex(1)
    labels = [t.get_text() for t in page.heatmap.figure.axes[0].get_xticklabels()]
    assert labels == list(syn.DETECTORS)
    page.heatmap.export()
    assert 'Saved' in page.heatmap.export_msg.text()
    assert page.heatmap.file_path().name.startswith('FlowCode spillover.')
    _check_pop_out(page.heatmap, qtbot)

    plugin.on_threshold_changed('HA', 2000.0)
    status, text = plugin.section_status(4)
    assert status == 'stale' and 'thresholds changed' in text
    plugin.on_thresholds_auto_all()
    assert plugin.section_status(4) == ('done', 'Done')

    # A new session shows the saved run at once, starts nothing, and rebuilds
    # the event plots on Show event figures without changing the state.
    controller2 = FlowCodeController(experiment)
    controller2.experiment.process[fcs.PROCESS_KEY] = deepcopy(saved)
    plugin2, _bus2 = _plugin(fp, qtbot, controller2)
    assert plugin2.section_status(4) == ('done', 'Done')
    view = plugin2.characterisation_view()
    assert view is not None and view[2] is None
    plugin2.tabs.setCurrentIndex(3)
    assert plugin2.pages[4].heatmap.figure is not None
    assert plugin2.backbone_data is None and not plugin2.characterise_busy()
    assert not plugin2.pages[4].figures_btn.isHidden()
    qtbot.mouseClick(plugin2.pages[4].figures_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: plugin2.characterisation is not None and not plugin2.characterise_busy(), timeout=LONG)
    assert plugin2.pages[4].figures_btn.isHidden()
    assert plugin2.state.thresholds == saved['thresholds']
    assert plugin2.state.gates == saved['gates']
    fc2 = plugin2.characterisation.result.spectra
    for a, b in zip(fc.combo_error, fc2.combo_error):
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-9)
    assert plugin2.state.characterisation == saved['characterisation']
    assert plugin2.pages[4].biplot_after._event_data is not None

    cache.unlink()
    plugin2._saved_cache = None
    plugin2.characterisation = None
    status, text = plugin2.section_status(4)
    assert status == 'stale' and 'not found' in text
    _settle(plugin, qtbot)
    _settle(plugin2, qtbot)


def test_characterisation_failure_is_reported(fp, qtbot, experiment):
    controller = FlowCodeController(experiment)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    plugin.on_setting_changed('min_cells_per_combo', 1_000_000)
    plugin.tabs.setCurrentIndex(3)
    plugin.characterise()
    qtbot.waitUntil(lambda: not plugin.characterise_busy(), timeout=LONG)
    assert 'No combo had' in plugin.pages[4].run_status.text()
    assert plugin.section_status(4) == ('pending', 'Not run yet')
    assert not (experiment / 'state' / fp.SPECTRA_FILE).exists()
    _settle(plugin, qtbot)


# ---------------------------------------------------------------------------
# Section 5
# ---------------------------------------------------------------------------

def _text_value(text, key):
    for candidate in (key, key.lower(), f'${key}', f'${key.lower()}'):
        if candidate in text:
            return text[candidate]
    raise KeyError(key)


def test_section_five_adds_the_threshold_af_profiles_and_warns(fp, qtbot, experiment):
    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(4)
    page = plugin.pages[5]
    assert plugin.state.af_profile_names == [syn.AF_PROFILE]
    assert [k for k, _n, ok in plugin.export_candidates() if ok] == [syn.BACKBONE, syn.OTHER_SAMPLE]
    note = plugin.export_af_note()
    assert 'Mouse 1' in note and 'Backbone' not in note and 'Autofluorescence page of the AutoSpectral tab' in note
    assert page.af_warn.text() == note
    controller.experiment.samples['sample_af_profiles'][syn.OTHER_SAMPLE] = [syn.AF_PROFILE]
    assert plugin.export_af_note() == ''


def test_section_five_exports_flowcode_channels_and_summaries(fp, qtbot, experiment):
    import csv
    from PySide6.QtCore import Qt
    from honeychrome.controller_components.functions import sample_from_fcs

    control = 'Raw/Reference Group/A1 HA BV421 (Cells).fcs'
    controller = FlowCodeController(experiment)
    controller.experiment.samples['all_samples'][control] = 'A1 HA BV421 (Cells)'
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE],
                                                           control: [syn.AF_PROFILE]}
    plugin, bus = _plugin(fp, qtbot, controller)
    popups = []
    bus.popupMessage.connect(popups.append)
    _choose_backbone(plugin, qtbot)
    plugin.tabs.setCurrentIndex(4)
    assert plugin.section_status(5) == ('waiting', 'Waiting for section 4')
    assert not plugin.pages[5].run_btn.isEnabled()
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)

    plugin.tabs.setCurrentIndex(4)
    page = plugin.pages[5]
    assert plugin.section_status(5) == ('pending', 'Not run yet')
    assert [k for k, _n, ok in plugin.export_candidates() if ok] == [syn.BACKBONE, syn.OTHER_SAMPLE]
    assert page.sample_list.count() == 2 and plugin.export_selection() == [syn.BACKBONE, syn.OTHER_SAMPLE]
    page.controls_check.setChecked(True)
    qtbot.waitUntil(lambda: page.sample_list.count() == 5, timeout=5000)
    # Samples without an AF assignment (the shifted unstained) are exported
    # with the profiles the thresholds were set with, so every listed sample
    # is selectable.
    assert plugin.export_selection() == [syn.BACKBONE, syn.UNSTAINED_MATCHED, syn.UNSTAINED_SHIFTED,
                                         syn.OTHER_SAMPLE, control]
    assert plugin.section_status(5)[0] == 'pending'
    page.controls_check.setChecked(False)
    qtbot.waitUntil(lambda: page.sample_list.count() == 2, timeout=5000)
    assert plugin.export_selection() == [syn.BACKBONE, syn.OTHER_SAMPLE]
    assert 'Final unmix: <b>AutoSpectral per-cell AF</b>' in page.method_line.text()
    assert page.run_btn.isEnabled(), page.run_status.text()
    qtbot.mouseClick(page.run_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert plugin.section_status(5) == ('done', 'Done'), page.run_status.text()
    assert popups and 'Exported 2 sample(s)' in popups[-1]

    folder = experiment / fp.EXPORT_SUBDIR
    sample = sample_from_fcs(folder / 'Backbone FlowCode (Unmixed).fcs')
    pnn = list(sample.pnn_labels)
    events = np.asarray(sample.get_events(source='raw'), dtype=np.float64)
    text = sample.get_metadata()
    design = plugin.design()
    extra = ['FlowCode', 'FlowCode ID'] + [f'FC {n}' for n in design.combo_names]
    assert pnn[-len(extra):] == extra
    assert _text_value(text, 'FLOWCODEUNMIX') == fcs.PLUGIN_VERSION
    assert f'FC {design.combo_names[0]}={fcc.combo_key(design.combo_tags[0])}' in _text_value(text, 'FLOWCODECOMBOS')
    assert _text_value(text, 'FLOWCODESOURCE') == syn.BACKBONE
    assert _text_value(text, 'FLOWCODECHARACTERISATION') == plugin.saved_spectra().characterisation_id()
    combos = events[:, [pnn.index(ch) for ch in extra[2:]]]
    flowcode = events[:, pnn.index('FlowCode')]
    ids = events[:, pnn.index('FlowCode ID')]
    assert (combos != 0).sum(axis=1).max() == 1
    np.testing.assert_allclose(combos.sum(axis=1), flowcode, rtol=1e-6)
    barcoded = (ids >= 1) & (ids <= design.n_combos)
    assert np.all((combos[barcoded] != 0).sum(axis=1) == 1)
    np.testing.assert_array_equal(np.argmax(combos[barcoded] != 0, axis=1) + 1, ids[barcoded])
    assert np.all(combos[~barcoded] == 0)
    n_backbone = syn.SAMPLE_EVENTS[syn.BACKBONE][1]
    assert len(events) == n_backbone

    with open(folder / fp.SUMMARY_FILE, newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(fcc.SUMMARY_COLUMNS) + list(design.combo_names) + ['Calibrated combos']
    assert [r[0] for r in rows[1:]] == ['Backbone', 'Mouse 1']
    assert int(rows[1][1]) == n_backbone
    assert sum(int(v) for v in rows[1][len(fcc.SUMMARY_COLUMNS):-1]) == int(np.count_nonzero(flowcode))
    assert int(np.count_nonzero(flowcode)) > 0.6 * n_backbone
    run = json.loads((folder / fp.RUN_FILE).read_text())
    assert run['backbone'] == [syn.BACKBONE] and len(run['samples']) == 2
    assert 'af_fret_term' not in run['settings'] and run['settings']['correction'] == 'combo'
    assert run['calibration'] == 'on' and run['settings']['calib_combos'] is True
    assert all({'combos', 'not_identifiable', 'weighting'} <= set(s['combo_scales']) for s in run['samples'])
    assert _text_value(text, 'FLOWCODECALIBRATION') == 'on'
    assert 'Combination calibration' in page.result.text()
    assert run['settings']['rescue_toofew'] is True
    assert run['af_profiles'] == {syn.BACKBONE: [syn.AF_PROFILE], syn.OTHER_SAMPLE: [syn.AF_PROFILE]}
    assert [t['Tag'] for t in run['thresholds']] == list(design.tags)
    assert 'Backbone' in page.result.text() or str(folder) in page.result.text()

    plugin.tabs.setCurrentIndex(1)
    plugin.pages[2].combo_table.item(0, 1).setText('Renamed')
    qtbot.waitUntil(lambda: plugin.state.combos[0]['name'] == 'Renamed' and not plugin._change_pending,
                    timeout=5000)
    assert plugin.section_status(4) == ('done', 'Done')
    status, text = plugin.section_status(5)
    assert status == 'stale' and 'combo names changed' in text
    assert plugin.export_spectra().design.combo_names[0] == 'Renamed'
    plugin.tabs.setCurrentIndex(4)
    page._select_all(False)
    assert plugin.export_problem() == 'Select at least one sample.' and not page.run_btn.isEnabled()
    _settle(plugin, qtbot)


def test_section_five_checks_tandem_tags(fp, qtbot, experiment, monkeypatch):
    import csv
    from PySide6.QtCore import Qt

    db = [{'fluorophore': f, 'synonym1': '', 'synonym2': '', 'synonym3': '', 'synonym4': '',
           'class': 'Protein tandem' if f == 'BUV395' else 'Small molecule',
           'tandem.parent': 'FITC' if f == 'BUV395' else ''} for f in syn.FLUORS]
    monkeypatch.setattr(fcc, '_fluorophore_db', lambda: db)
    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    ref = plugin.characterisation.result.spectra.diagnostics['tandem']['BUV395']
    assert ref['parent'] == 'FITC' and ref['tag'] == 'C' and ref['source'] == 'panel' and ref['n_null'] == 4
    assert 'Tandem C (BUV395' in plugin.pages[4].summary.text()

    plugin.tabs.setCurrentIndex(4)
    page = plugin.pages[5]
    box = page.tandem_checks['BUV395']
    assert list(page.tandem_checks) == ['BUV395'] and box.isChecked() and box.isEnabled()
    assert 'C: backbone FITC per unit C' in page.tandem_status.text()
    box.setChecked(False)
    assert plugin.state.settings['tandem_tags'] == [] and plugin.export_tandems()[0] == []
    box.setChecked(True)
    assert plugin.state.settings['tandem_tags'] == ['BUV395']
    page.tandem_adv['tandem_min_combos'].setValue(3)
    assert plugin.state.settings['tandem_min_combos'] == 3
    assert page.run_btn.isEnabled(), page.run_status.text()
    qtbot.mouseClick(page.run_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert plugin.section_status(5) == ('done', 'Done'), page.run_status.text()

    folder = experiment / fp.EXPORT_SUBDIR
    with open(folder / fp.SUMMARY_FILE, newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:len(fcc.SUMMARY_COLUMNS)] == list(fcc.SUMMARY_COLUMNS)
    assert rows[0][-4:] == ['BUV395 -> FITC per unit', 'BUV395 -> FITC z', 'BUV395 -> FITC at median C',
                            'Calibrated combos']
    run = json.loads((folder / fp.RUN_FILE).read_text())
    assert run['tandem_tags'][0]['parent'] == 'FITC' and run['settings']['tandem_min_combos'] == 3
    assert [s['tandem'][0]['fluor'] for s in run['samples']] == ['BUV395', 'BUV395']
    assert 'Tandem breakdown (BUV395)' in page.result.text()

    # combination calibration controls (Advanced settings)
    calib = page.calib_checks
    assert calib['calib_combos'].isChecked() and not calib['calib_brightness'].isChecked()
    assert calib['calib_brightness'].isEnabled() and page.calib_adv['calib_z_min'].value() == 3.0
    calib['calib_brightness'].setChecked(True)
    assert plugin.state.settings['calib_brightness'] is True
    assert plugin.section_status(5)[0] == 'stale' and plugin.calibration_mode(plugin.saved_spectra()) == 'brightness'
    page.calib_adv['calib_min_events'].setValue(60)
    assert plugin.state.settings['calib_min_events'] == 60
    calib['calib_combos'].setChecked(False)
    page.refresh()
    assert not calib['calib_brightness'].isEnabled() and not page.calib_adv['calib_z_min'].isEnabled()
    assert plugin.calibration_mode(plugin.saved_spectra()) == 'off'
    calib['calib_combos'].setChecked(True)
    page.correction.setCurrentIndex(page.correction.findData('pairwise'))
    page.refresh()
    assert not calib['calib_combos'].isEnabled() and 'Correction: combo' in page.calib_status.text()
    assert plugin.calibration_mode(plugin.saved_spectra()) == 'off'
    _settle(plugin, qtbot)


def test_section_five_overrides_switch_off_or_scale_a_combination_in_the_export(fp, qtbot, experiment):
    from PySide6.QtCore import Qt
    from honeychrome.controller_components.functions import sample_from_fcs

    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(4)
    page = plugin.pages[5]
    design = plugin.design()
    names = design.combo_names
    keys = [fcc.combo_key(t) for t in design.combo_tags]
    assert page.override_table.rowCount() == design.n_combos
    assert page.override_table.item(0, 0).text() == names[0]
    assert 'No overrides' in page.override_status.text() and not page.override_clear.isEnabled()

    mode, _spin = page._override_widgets[keys[0]]
    mode.setCurrentIndex(mode.findData('off'))
    assert plugin.state.settings['combo_overrides'] == {keys[0]: 0.0}
    mode, spin = page._override_widgets[keys[1]]
    assert not spin.isEnabled()
    mode.setCurrentIndex(mode.findData('scaled'))
    assert plugin.state.settings['combo_overrides'][keys[1]] == 1.0 and spin.isEnabled()
    spin.setValue(0.5)
    assert plugin.state.settings['combo_overrides'] == {keys[0]: 0.0, keys[1]: 0.5}
    assert f'{names[0]} off' in page.override_status.text() and f'{names[1]} \u00d70.50' in page.override_status.text()
    assert fcs.DEFAULT_SETTINGS['combo_overrides'] == {}

    assert page.run_btn.isEnabled(), page.run_status.text()
    qtbot.mouseClick(page.run_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert plugin.section_status(5) == ('done', 'Done'), page.run_status.text()
    folder = experiment / fp.EXPORT_SUBDIR
    run = json.loads((folder / fp.RUN_FILE).read_text())
    assert run['fret_overrides'] == {names[0]: 0.0, names[1]: 0.5}
    assert run['settings']['combo_overrides'] == {keys[0]: 0.0, keys[1]: 0.5}
    for s in run['samples']:
        rec = s['combo_scales']['combos']
        assert rec[names[0]]['override'] and rec[names[0]]['applied'] == [0.0, 0.0]
        assert rec[names[1]]['override'] and rec[names[1]]['applied'] == [0.5, 0.0]
    text = sample_from_fcs(folder / 'Backbone FlowCode (Unmixed).fcs').get_metadata()
    assert _text_value(text, 'FLOWCODEOVERRIDES') == f'{names[0]}=0;{names[1]}=0.5'
    assert 'FRET correction overridden' in page.result.text() and f'{names[0]} off' in page.result.text()

    signature = plugin.live_unmixing_signature([syn.BACKBONE])
    spin.setValue(0.75)
    assert plugin.section_status(5) == ('stale', 'Needs re-run: FRET correction overrides changed')
    assert plugin.live_unmixing_signature([syn.BACKBONE]) != signature
    qtbot.mouseClick(page.override_clear, Qt.LeftButton)
    assert plugin.state.settings['combo_overrides'] == {}
    assert page._override_widgets[keys[0]][0].currentData() == 'backbone'
    page._override_widgets[keys[2]][0].setCurrentIndex(mode.findData('off'))
    assert plugin.state.settings['combo_overrides'] == {keys[2]: 0.0}
    qtbot.mouseClick(page.restore_btn, Qt.LeftButton)
    assert plugin.state.settings['combo_overrides'] == {}
    page._override_widgets[keys[2]][0].setCurrentIndex(mode.findData('off'))
    page.correction.setCurrentIndex(page.correction.findData('pairwise'))
    page.refresh()
    assert not page._override_widgets[keys[2]][0].isEnabled() and 'Correction: combo' in page.override_status.text()
    assert plugin.export_overrides(plugin.saved_spectra()) == {}
    _settle(plugin, qtbot)


def test_live_unmix_equals_the_export_and_is_cached(fp, qtbot, experiment, monkeypatch):
    """The Analysis page's live unmix gives exactly the exported file's events,
    and a second unmix reads the cache instead of unmixing again."""
    import flowcode_analysis as fa
    import flowcode_export as fce
    from honeychrome.controller_components.functions import sample_from_fcs

    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(4)
    plugin.start_export()
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)

    keys = [syn.BACKBONE, syn.OTHER_SAMPLE]
    assert plugin.live_af_ok(keys) == {k: True for k in keys}
    links = fa.live_links(keys + ['Raw/missing.fcs'], experiment, {**plugin.live_af_ok(keys), 'Raw/missing.fcs': True})
    assert [links['files'][k]['status'] for k in keys] == [fa.LINK_OK, fa.LINK_OK]
    assert links['files']['Raw/missing.fcs']['status'] == fa.LINK_MISSING
    inputs = plugin.live_unmix_inputs(keys)
    # a sample's cache key does not depend on the other samples in the run
    assert plugin.live_unmix_inputs([syn.OTHER_SAMPLE])['sample_signatures'][syn.OTHER_SAMPLE] == \
        inputs['sample_signatures'][syn.OTHER_SAMPLE]
    folder = experiment / fp.EXPORT_SUBDIR
    for key, name in ((syn.BACKBONE, 'Backbone'), (syn.OTHER_SAMPLE, 'Mouse 1')):
        sample = sample_from_fcs(folder / f'{name} FlowCode (Unmixed).fcs')
        exported = np.asarray(sample.get_events(source='raw'))
        assert list(sample.pnn_labels) == inputs['pnn']
        cache_key = fce.live_cache_key(inputs['sample_signatures'][key], key, fce.file_signature(experiment / key))
        live = fce.live_sample_events(inputs['layout'], inputs['af_by_sample'][key], inputs['unmix_fn'], key,
                                      inputs['pnn'], inputs['n_extra'], cache_key=cache_key)
        np.testing.assert_array_equal(live.astype(np.float64), exported)
        assert fce.live_cache_file(experiment, cache_key).exists()
        calls = []
        unmix_fn = inputs['unmix_fn']

        def counting(*args, **kwargs):
            calls.append(1)
            return unmix_fn(*args, **kwargs)
        again = fce.live_sample_events(inputs['layout'], inputs['af_by_sample'][key], counting, key,
                                       inputs['pnn'], inputs['n_extra'], cache_key=cache_key)
        assert not calls
        np.testing.assert_array_equal(again, live)
    # a changed unmixing gives a different signature, so no stale cache is read
    plugin.state.settings['calib_brightness'] = True
    assert plugin.live_unmixing_signature(keys) != inputs['signature']
    plugin.state.settings['calib_brightness'] = False
    assert plugin.live_unmixing_signature(keys) == inputs['signature']
    plugin.state.settings['correction'] = 'pairwise'
    assert plugin.live_unmixing_signature(keys) != inputs['signature']
    _settle(plugin, qtbot)


def _with_unmixed_gate(controller):
    """An unmixed 'Cells' rectangle on FSC-A / SSC-A, as the Analysis page's population."""
    from flowkit import Dimension, GatingStrategy, gates

    gating = GatingStrategy()
    for ch in syn.SCATTER:
        gating.transformations[ch] = controller.unmixed_transformations[ch].xform
    dim_x = Dimension('FSC-A', range_min=0.1, range_max=0.5, transformation_ref='FSC-A')
    dim_y = Dimension('SSC-A', range_min=0.05, range_max=0.3, transformation_ref='SSC-A')
    gating.add_gate(gates.RectangleGate('Cells', dimensions=[dim_x, dim_y]), gate_path=('root',))
    controller.unmixed_gating = gating
    return controller


def test_analysis_counts_live_without_exporting_and_matches_the_export(fp, qtbot, experiment):
    import flowcode_analysis as fa

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    assert page.source() == fa.SOURCE_LIVE and not page.export_box.isVisible()
    assert 'unmixed from its raw file' in page.source_note.text()
    page.settings['gates'] = ['Cells']
    page.refresh()
    linked = {k for k, v in page.links['files'].items() if v['status'] == fa.LINK_OK}
    assert {syn.BACKBONE, syn.OTHER_SAMPLE} <= linked
    page.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page.links['files']}
    page.start_count()
    qtbot.waitUntil(lambda: page.count_btn.isEnabled(), timeout=LONG)
    live = page.settings['counts']
    assert live['source'] == fa.SOURCE_LIVE and not live['errors'], live['errors']
    assert live['populations'] == ['Cells'] and set(live['records']['Cells']) == {syn.BACKBONE, syn.OTHER_SAMPLE}
    assert all(r['barcoded'] > 0 and r['in_population'] < r['events'] for r in live['records']['Cells'].values())
    assert all('marker_sums' not in r for r in live['records']['Cells'].values())
    markers = fa.load_marker_sums(fa.marker_cache_path(experiment), live['signature'])
    assert markers is not None and markers['channels'] == plugin.analysis_marker_channels()
    assert list(page.counts.index) == [syn.BACKBONE, syn.OTHER_SAMPLE]
    design = plugin.design()
    tag_matrix, spread, note = page._marker_tag_inputs(True)
    assert tag_matrix.shape == (len(design.combo_names), len(design.tags))
    assert (tag_matrix.sum(axis=1) == 3).all()
    if spread is None:
        assert 'noise model' in note
        import flowcode_core as fcc
        out = fcc.spillover_spread(plugin.saved_spectra(), np.ones(len(plugin.saved_spectra().detectors)))
        assert out['values'].shape == (len(design.combo_names), len(plugin.saved_spectra().fluor_names))
        assert np.isfinite(out['values']).all() and (out['values'] >= 0).all()
    assert 'Counts from' in page.count_status.text() and 'out of date' not in page.count_status.text()

    plugin.tabs.setCurrentIndex(4)
    plugin.start_export()
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    plugin.tabs.setCurrentIndex(5)
    page.source_combo.setCurrentIndex(page.source_combo.findData(fa.SOURCE_EXPORTED))
    assert page.source() == fa.SOURCE_EXPORTED and 'out of date' in page.count_status.text()
    page.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page.links['files']}
    page.start_count()
    qtbot.waitUntil(lambda: page.count_btn.isEnabled(), timeout=LONG)
    exported = page.settings['counts']
    assert exported['source'] == fa.SOURCE_EXPORTED
    assert exported['records'] == live['records']
    _settle(plugin, qtbot)


def test_gating_viewer_follows_the_sample_browser_with_the_experiments_plots(fp, qtbot, experiment):
    import flowcode_analysis as fa
    import flowcode_export as fce
    import flowcode_gating as fcg

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    controller.experiment.cytometry = {'plots': [
        {'type': 'hist2d', 'channel_x': 'FSC-A', 'channel_y': 'SSC-A', 'source_gate': 'root', 'child_gates': ['Cells']},
        {'type': 'hist1d', 'channel_x': 'PE', 'source_gate': 'Cells', 'child_gates': []},
        {'type': 'hist1d', 'channel_x': 'Not a channel', 'source_gate': 'root', 'child_gates': []}]}
    controller.current_sample_path = None
    plugin, bus = _plugin(fp, qtbot, controller)
    browser, progress = [], []
    bus.sampleBrowserVisibilityRequested.connect(browser.append)
    bus.progress.connect(lambda n, total: progress.append((n, total)))
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    assert page.section_tabs.tabText(0) == 'Gating' and page.section_tabs.tabText(1) == 'Data'
    page.section_tabs.setCurrentIndex(1)
    assert browser[-1] is False
    page.settings['gates'] = ['Cells']
    page.refresh()
    page.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page.links['files']}
    page.start_count()
    qtbot.waitUntil(lambda: page.count_btn.isEnabled(), timeout=LONG)
    records = page.settings['counts']['records']['Cells']
    cache_dir = experiment / fce.LIVE_CACHE_SUBDIR
    cached = set(cache_dir.iterdir())
    gating_before = fa.SampleGating.from_controller(controller).signature()

    view = page.gating_view
    page.section_tabs.setCurrentWidget(view)
    assert browser[-1] is True and 'sample browser' in view.status.text()
    assert view.view_combo.currentData() == fcg.VIEW_CALIBRATED
    assert view.highlight_combo.count() == 2 + len(view._phenotypes()) + len(plugin.design().combo_names)

    def picked(key):
        view.result = None
        controller.current_sample_path = key
        plugin._on_sample_selected(key)
        qtbot.waitUntil(lambda: view.result is not None or 'Could not' in view.status.text(), timeout=LONG)
        assert view.result is not None, view.status.text()
        return {g['name']: g for g in view.result['rows']}

    progress.clear()
    rows = picked(syn.OTHER_SAMPLE)
    assert progress[-1] == (3, 3)                               # two unmixes and the gating, then done
    cells = rows['Cells']
    assert cells['n_in'] == records[syn.OTHER_SAMPLE]['in_population'] and cells['parent'] == 'root'
    assert cells['percent'] == pytest.approx(100.0 * cells['n_in'] / records[syn.OTHER_SAMPLE]['events'])
    assert cells['changed'] is not None
    assert view.table.rowCount() == 1 and view.table.item(0, 0).text() == 'Cells'
    assert [p.plot['channel_x'] for p in view._plots] == ['FSC-A', 'PE']   # the plot on a missing channel is left out
    assert type(view._plots[0]).__mro__[1].__name__ == 'CytometryPlotWidget'
    roi = view._plots[0].rois[0]
    assert roi.label.gate_name == 'Cells' and roi.translatable is False
    assert len(set(cache_dir.iterdir()) - cached) == 1          # the calibrated view is the counts' cache

    view.result = None
    view.view_combo.setCurrentIndex(view.view_combo.findData(fcg.VIEW_STANDARD))
    qtbot.waitUntil(lambda: view.result is not None, timeout=LONG)
    standard = {g['name']: g for g in view.result['rows']}['Cells']
    inputs = plugin.live_unmix_inputs([syn.OTHER_SAMPLE], fcg.VIEW_OVERRIDES[fcg.VIEW_STANDARD])
    events = fce.live_sample_events(inputs['layout'], inputs['af_by_sample'][syn.OTHER_SAMPLE], inputs['unmix_fn'],
                                    syn.OTHER_SAMPLE, inputs['pnn'], inputs['n_extra'])
    counted = fa.count_populations(events, inputs['pnn'], syn.OTHER_SAMPLE, page._gating_snapshot(), ['Cells'])
    assert standard['n_in'] == counted['Cells']['in_population']

    view.result = None
    view.view_combo.setCurrentIndex(view.view_combo.findData(fcg.VIEW_UNCORRECTED))
    qtbot.waitUntil(lambda: view.result is not None, timeout=LONG)
    assert view.result['rows'][0]['changed'] is None
    backbone = picked(syn.BACKBONE)['Cells']
    assert backbone['n_in'] == records[syn.BACKBONE]['in_population'] or view.view_combo.currentData() != \
        fcg.VIEW_CALIBRATED
    assert fa.SampleGating.from_controller(controller).signature() == gating_before
    page.section_tabs.setCurrentIndex(1)
    assert browser[-1] is False
    _settle(plugin, qtbot)


def test_no_fret_backbone_debarcodes_without_correction_and_marks_every_output(fp, qtbot, experiment, monkeypatch):
    import csv
    import flowcode_analysis as fa
    import flowcode_stats as fst
    from honeychrome.controller_components.functions import sample_from_fcs

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    asked = []
    answer = {'value': False}
    monkeypatch.setattr(plugin, 'confirm_no_fret', lambda action: asked.append(action) or answer['value'])
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    assert plugin.section_status(4) == ('done', 'Done')

    page1 = plugin.pages[1]
    plugin.tabs.setCurrentIndex(0)
    page1.no_fret_check.setChecked(True)
    assert len(asked) == 1 and not plugin.state.no_fret and not page1.no_fret_check.isChecked()
    answer['value'] = True
    page1.no_fret_check.setChecked(True)
    qtbot.waitUntil(lambda: not plugin._change_pending, timeout=5000)
    assert len(asked) == 2 and plugin.state.no_fret and plugin.state.no_fret_acknowledged
    assert page1.backbone_group.title() == fp.NO_FRET_BACKBONE_TITLE
    assert page1.no_fret_msg.text() == fcs.NO_FRET_WARNING
    assert plugin.section_status(3) == ('done', 'Done')
    status, text = plugin.section_status(4)
    assert status == 'stale' and 'FRET correction' in text

    plugin.tabs.setCurrentIndex(3)
    page4 = plugin.pages[4]
    assert page4.run_btn.text() == 'Debarcode (no FRET correction)'
    from PySide6.QtCore import Qt
    qtbot.mouseClick(page4.run_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: plugin.section_status(4) == ('done', 'Done') and not plugin.characterise_busy(),
                    timeout=LONG)
    fc = plugin.saved_spectra()
    assert not fc.fret_corrected and all(e is None for e in fc.combo_error)
    assert sum(plugin.characterisation.summary['per_combo'].values()) > 0
    assert fcs.NO_FRET_WARNING in page4.status.text() and '#d62728' in page4.status.styleSheet()
    assert fcs.NO_FRET_WARNING in page4.summary.text()
    assert 'not performed' in page4.biplot_msg.text()

    plugin.tabs.setCurrentIndex(4)
    page5 = plugin.pages[5]
    assert not page5.correction.isEnabled() and not page5.fix_btn.isEnabled()
    n_asked = len(asked)
    answer['value'] = False
    plugin.start_export()
    assert len(asked) == n_asked + 1 and not plugin.export_running()
    assert page5.run_status.text() == 'Export cancelled.'
    answer['value'] = True
    plugin.start_export()
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert plugin.section_status(5) == ('done', 'Done')
    assert fcs.NO_FRET_WARNING in page5.status.text() and fcs.NO_FRET_WARNING in page5.result.text()

    folder = experiment / fp.EXPORT_SUBDIR
    sample = sample_from_fcs(folder / 'Backbone FlowCode (Unmixed).fcs')
    text = sample.get_metadata()
    assert _text_value(text, 'FLOWCODECORRECTION') == fa.NO_CORRECTION_KEYWORD
    assert _text_value(text, 'FLOWCODEWARNING') == fcs.NO_FRET_WARNING
    assert _text_value(text, 'FLOWCODECHARACTERISATION') == fc.characterisation_id()
    run = json.loads((folder / fp.RUN_FILE).read_text())
    assert run['fret_correction'] is False and run['warning'] == fcs.NO_FRET_WARNING
    assert run['tandem_tags'] == [] and {c['error'] for c in run['combinations']} == {'none'}
    with open(folder / fp.SUMMARY_FILE, newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-1] == 'Warning' and all(r[-1] == fcs.NO_FRET_WARNING for r in rows[1:])

    plugin.tabs.setCurrentIndex(5)
    page6 = plugin.pages[6]
    assert page6.is_live() and page6.uncorrected()
    page6.source_combo.setCurrentIndex(page6.source_combo.findData(fa.SOURCE_EXPORTED))
    page6.settings['gates'] = ['Cells']
    page6.refresh()
    assert page6.links['files'][syn.BACKBONE]['uncorrected'] and page6.uncorrected()
    page6.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page6.links['files']}
    page6.start_count()
    qtbot.waitUntil(lambda: page6.count_btn.isEnabled(), timeout=LONG)
    page6.refresh_status()
    assert page6.no_fret_label.isVisible() and fcs.NO_FRET_WARNING in page6.no_fret_label.text()
    assert '#d62728' in page6.status.styleSheet()

    stamped = []
    original = fst.stamp_invalid
    monkeypatch.setattr(fst, 'stamp_invalid', lambda fig, text=None: stamped.append(fig) or original(fig, text))
    n_asked = len(asked)
    page6.start_export()
    assert len(asked) == n_asked + 1
    qtbot.waitUntil(lambda: page6.export_btn.isEnabled(), timeout=LONG)
    out = page6._export_folder
    assert (out / 'settings.txt').read_text().splitlines()[0] == fcs.NO_FRET_WARNING
    assert list(out.glob('*report.pdf')) and len(stamped) >= 3

    plugin.tabs.setCurrentIndex(0)
    page1.no_fret_check.setChecked(False)
    qtbot.waitUntil(lambda: not plugin._change_pending, timeout=5000)
    assert not plugin.state.no_fret and plugin.state.no_fret_acknowledged is None
    assert plugin.section_status(4)[0] == 'stale' and plugin.section_status(3) == ('done', 'Done')
    _settle(plugin, qtbot)


def test_prerequisites_panel_and_opening_the_autospectral_pages(fp, qtbot, experiment):
    from honeychrome.controller_components import spectral_correction as sc

    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, bus = _plugin(fp, qtbot, controller)
    opened = []
    bus.showAutoSpectralPage.connect(opened.append)
    _choose_backbone(plugin, qtbot)
    rows = {r['key']: r for r in plugin.prerequisites()}
    assert list(rows) == ['af', 'noise', 'flowcode', 'corrections', 'export']
    assert rows['af']['state'] == 'ok' and '2 of 2 sample(s)' in rows['af']['text']
    assert rows['noise']['state'] == 'warn' and 'Not fitted' in rows['noise']['text']
    assert rows['flowcode']['state'] == 'todo' and rows['corrections']['state'] == 'info'
    panel = plugin.pages[1].prerequisites
    mark, _name, text, button = panel.rows['noise']
    assert mark.text() == '⚠' and button.isEnabled() and button.text() == 'Open Noise'
    button.click()
    assert opened == ['noise']
    plugin.open_prerequisite('corrections')
    assert opened == ['noise', 'correction']
    fixed = sc.load_setup(controller.experiment.process).fixed
    assert fixed == plugin.state.tag_channels() and fixed
    plugin.open_prerequisite('corrections')
    assert sc.load_setup(controller.experiment.process).fixed == fixed
    _settle(plugin, qtbot)


def test_fix_unmix_from_section_five_reads_flowcode_corrected_events(fp, qtbot, experiment, monkeypatch):
    from PySide6.QtCore import Qt
    from honeychrome.controller_components import spectral_correction as sc
    import flowcode_fix_unmix as fcfu

    controller = FlowCodeController(experiment)
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    process = controller.experiment.process
    sc.save_setup(process, sc.CorrectionSetup(samples=[syn.BACKBONE, syn.OTHER_SAMPLE], gate='root',
                                              n_events=4000, unstained={sc.ALL_GROUP: syn.UNSTAINED_MATCHED}))
    reads, runs = [], []
    loader_of = fcfu.corrected_loader

    def recording_loader(unmix_fn, af_by_sample, *args, **kwargs):
        inner = loader_of(unmix_fn, af_by_sample, *args, **kwargs)

        def loader(experiment_dir, path, *rest):
            fl, scatter = inner(experiment_dir, path, *rest)
            reads.append((path, path in af_by_sample))
            return fl, scatter
        return loader

    def recording_run(snap, progress, loader=None):
        runs.append(snap['settings'])
        return run_corrections(snap, progress, loader)

    run_corrections = sc.run_corrections
    monkeypatch.setattr(fcfu, 'corrected_loader', recording_loader)
    monkeypatch.setattr(sc, 'run_corrections', recording_run)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    assert plugin.correction_state()[0] == 'info'
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)

    plugin.tabs.setCurrentIndex(4)
    page = plugin.pages[5]
    assert page.fix_btn.isEnabled()
    qtbot.mouseClick(page.fix_btn, Qt.LeftButton)
    qtbot.waitUntil(lambda: not plugin.fix_running(), timeout=LONG)
    _settle(plugin, qtbot)
    assert page.fix_status.text().startswith('Done'), page.fix_status.text()
    tags = plugin.state.tag_channels()
    assert set(tags) <= set(runs[0].fixed_fluorophores) and runs[0].exclude_fixed_targets
    assert (syn.BACKBONE, True) in reads and (syn.OTHER_SAMPLE, True) in reads
    assert (syn.UNSTAINED_MATCHED, False) in reads
    record = fcfu.correction_record(process, sc.STORE_KEY)
    assert record['characterisation'] == plugin.saved_spectra().characterisation_id() and record['tags'] == tags
    kind, text = plugin.correction_state()
    assert (kind == 'ok' and 'FRET-corrected' in text) or (kind == 'info' and not process[sc.STORE_KEY]['groups']
                                                            [sc.ALL_GROUP]['fluorophores'])


# ---------------------------------------------------------------------------
# Several libraries: export and Analysis
# ---------------------------------------------------------------------------

class _LibraryQuestion:
    """QMessageBox stand-in for the export's question about samples without
    a library: clicks the button named ``choice``."""
    choice = ''
    Warning = ActionRole = AcceptRole = Cancel = 0

    def __init__(self, *_args):
        self.buttons = []

    def __getattr__(self, _name):
        return lambda *_a, **_k: None

    def addButton(self, button, *_role):
        made = SimpleNamespace(text=button if isinstance(button, str) else 'Cancel')
        self.buttons.append(made)
        return made

    def exec(self):
        return 0

    def clickedButton(self):
        return next(b for b in self.buttons if b.text == self.choice)


def test_several_libraries_export_names_and_analysis(fp, qtbot, experiment, monkeypatch):
    import csv
    import itertools

    import flowcode_analysis as fa
    from honeychrome.controller_components.functions import sample_from_fcs

    triplets = [list(t) for t in itertools.combinations(syn.TAGS, 3)]
    keys = [fcc.combo_key(t) for t in triplets]
    nt = 'NON-Targetting 1'
    libraries = {'lib1.csv': [(f'g{i}', triplets[i]) for i in range(9)] + [(nt, triplets[9])],
                 'lib2.csv': [(f'h{i}', triplets[i]) for i in range(6)] + [(nt, triplets[6])]}
    for name, rows in libraries.items():
        with open(experiment / name, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['ID', 'FlowCode tag1', 'FlowCode tag2', 'FlowCode tag3'])
            for guide, tags in rows:
                writer.writerow([guide, *tags])

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    state = plugin.state
    plugin.set_multi_library(True)
    assert plugin.load_library_csv(0, experiment / 'lib1.csv') == []
    assert plugin.load_library_csv(1, experiment / 'lib2.csv') == []
    state.libraries[1]['label'] = 'Metab B'
    plugin.state_changed()
    assert [row['name'] for row in state.combos] == keys
    assert plugin.section_status(4) == ('done', 'Done')
    assert plugin.library_samples() == [syn.OTHER_SAMPLE]
    assert plugin.export_library_gaps(plugin.export_selection()) == [syn.OTHER_SAMPLE]

    # Export blocked for the sample without a library
    monkeypatch.setattr(fp, 'QMessageBox', _LibraryQuestion)
    _LibraryQuestion.choice = 'Go to section 2'
    plugin.tabs.setCurrentIndex(4)
    plugin.start_export()
    assert not plugin.export_running() and plugin.tabs.currentIndex() == 1
    assert 'assign a library' in plugin.pages[5].run_status.text()
    _LibraryQuestion.choice = 'Assign as Other and export'
    plugin.tabs.setCurrentIndex(4)
    plugin.start_export()
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert state.sample_libraries[syn.OTHER_SAMPLE] == fcs.OTHER_LIBRARY
    assert plugin.section_status(5) == ('done', 'Done'), plugin.pages[5].run_status.text()
    folder = experiment / fp.EXPORT_SUBDIR
    mouse = sample_from_fcs(folder / 'Mouse 1 FlowCode (Unmixed).fcs')
    fc_at = {ch: i for i, ch in enumerate(mouse.pnn_labels) if ch.startswith('FC ')}
    assert {ch: mouse.pns_labels[i] for ch, i in fc_at.items()} == {f'FC {k}': k for k in keys}
    text = mouse.get_metadata()
    assert _text_value(text, 'FLOWCODELIBRARY') == 'Other'
    with pytest.raises(KeyError):
        _text_value(text, 'FLOWCODEGUIDES')

    # Exported with Library 2: guide names per file, summaries and the run record
    state.sample_libraries[syn.OTHER_SAMPLE] = 'Library 2'
    plugin.state_changed()
    assert plugin.section_status(5)[0] == 'stale'
    plugin.tabs.setCurrentIndex(4)
    plugin.start_export()
    qtbot.waitUntil(lambda: not plugin.export_running(), timeout=LONG)
    assert plugin.section_status(5) == ('done', 'Done')
    mouse = sample_from_fcs(folder / 'Mouse 1 FlowCode (Unmixed).fcs')
    guides = dict(zip(keys[:6], [f'h{i}' for i in range(6)]), **{keys[6]: nt})
    assert {ch: mouse.pns_labels[i] for ch, i in fc_at.items()} == {f'FC {k}': guides.get(k, k) for k in keys}
    text = mouse.get_metadata()
    assert _text_value(text, 'FLOWCODELIBRARY') == 'Library 2'
    assert _text_value(text, 'FLOWCODELIBRARYLABEL') == 'Metab B'
    assert _text_value(text, 'FLOWCODEGUIDES') == ';'.join(f'FC {k}={g}' for k, g in guides.items())
    assert _text_value(text, 'FLOWCODECOMBOS') == ';'.join(f'FC {k}={k}' for k in keys)
    backbone = sample_from_fcs(folder / 'Backbone FlowCode (Unmixed).fcs')
    assert _text_value(backbone.get_metadata(), 'FLOWCODELIBRARY') == 'Other'
    with open(folder / fp.SUMMARY_FILE, newline='') as fh:
        rows = list(csv.reader(fh))
    n = len(fcc.SUMMARY_COLUMNS)
    assert rows[0][n:n + 2] == ['Library', 'Out of library'] and rows[0][n + 2:-1] == keys
    by_name = {r[0]: r for r in rows[1:]}
    assert by_name['Backbone'][n:n + 2] == ['Other', '']
    outside = sum(int(v) for v in by_name['Mouse 1'][n + 2 + 7:-1])
    assert by_name['Mouse 1'][n:n + 2] == ['Library 2', str(outside)] and outside > 0
    run = json.loads((folder / fp.RUN_FILE).read_text())
    assert [lib['name'] for lib in run['libraries']] == ['Library 1', 'Library 2'] and run['merge_same_names']
    assert run['sample_libraries'] == {syn.BACKBONE: None, syn.OTHER_SAMPLE: 'Library 2'}
    with open(experiment / fp.OUTPUT_SUBDIR / 'combinations Library 2.csv', newline='') as fh:
        written = list(csv.reader(fh))[1:]
    assert [r[0] for r in written] == [f'h{i}' for i in range(6)] + [nt]

    # Analysis: counts named by the sample's library; the backbone is Other
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    page.settings['gates'] = ['Cells']
    page.refresh()
    assert set(page.used_files()) == {syn.OTHER_SAMPLE}
    page.start_count()
    qtbot.waitUntil(lambda: page.count_btn.isEnabled(), timeout=LONG)
    record = page.settings['counts']['records']['Cells'][syn.OTHER_SAMPLE]
    assert set(record['counts']) == set(keys)
    assert list(page.counts.index) == [syn.OTHER_SAMPLE]
    row = page.counts.loc[syn.OTHER_SAMPLE]
    assert row[[f'g{i}' for i in range(9)]].isna().all()
    assert row[f'h{1}'] == record['counts'][keys[1]] and row[nt] == record['counts'][keys[6]]
    assert page.meta.loc[syn.OTHER_SAMPLE, 'out_of_library'] == sum(record['counts'][k] for k in keys[7:])
    headers = [page.sample_table.horizontalHeaderItem(j).text() for j in range(page.sample_table.columnCount())]
    assert headers[2] == 'Library' and 'Out of library' in headers
    labels = {page.sample_table.item(r, 1).text(): page.sample_table.item(r, 2).text()
              for r in range(page.sample_table.rowCount())}
    assert labels['Mouse 1'] == 'Metab B' and labels['Backbone'] == 'Other'
    assert page.samples_table.horizontalHeaderItem(2).text() == 'Library (section 2)'
    assert page.cb_split.findData(fa.SPLIT_BY_LIBRARY) > 0

    # A new assignment names the same counts again; no library refuses the statistics
    state.sample_libraries[syn.OTHER_SAMPLE] = 'Library 1'
    plugin.state_changed()
    page.refresh()
    assert page.counts.loc[syn.OTHER_SAMPLE, [f'g{i}' for i in range(9)]].notna().all()
    assert page.counts.loc[syn.OTHER_SAMPLE, 'h1':'h5'].isna().all()
    state.sample_libraries.pop(syn.OTHER_SAMPLE)
    plugin.state_changed()
    page.refresh()
    assert page.counts is None and 'without a library' in page.filter_note.text()
    assert not page.run_btn.isEnabled() and 'have no library' in page.stats_status.text()

    # Exported files: a file exported for another library is noted, and used
    state.sample_libraries[syn.OTHER_SAMPLE] = 'Library 1'
    plugin.state_changed()
    page.source_combo.setCurrentIndex(page.source_combo.findData(fa.SOURCE_EXPORTED))
    info = page.links['files'][syn.OTHER_SAMPLE]
    assert info['status'] == fa.LINK_OK and 'Exported as Library 2; counted as Library 1' in info['note']
    _settle(plugin, qtbot)


def test_several_libraries_statistics_figures_and_export_on_the_analysis_page(fp, qtbot, experiment):
    """Saved counts of 24 samples (Spleen and Gut of six mice per library)
    through the Analysis page: names per library, statistics with per-library
    pairing and library sizes, the marker test within each library, figures
    and the analysis export."""
    import csv
    import itertools

    import flowcode_analysis as fa

    triplets = [list(t) for t in itertools.combinations(syn.TAGS, 3)]
    keys = [fcc.combo_key(t) for t in triplets]
    nt = 'NON-Targetting 1'
    libraries = {'lib1.csv': [(f'g{i}', triplets[i]) for i in range(9)] + [(nt, triplets[9])],
                 'lib2.csv': [(f'h{i}', triplets[i]) for i in range(6)] + [(nt, triplets[6])]}
    for name, rows in libraries.items():
        with open(experiment / name, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['ID', 'FlowCode tag1', 'FlowCode tag2', 'FlowCode tag3'])
            for guide, tags in rows:
                writer.writerow([guide, *tags])
    plugin, _bus = _plugin(fp, qtbot, _with_unmixed_gate(FlowCodeController(experiment)))
    state = plugin.state
    plugin.set_multi_library(True)
    assert plugin.load_library_csv(0, experiment / 'lib1.csv') == []
    assert plugin.load_library_csv(1, experiment / 'lib2.csv') == []

    rng = np.random.default_rng(2)
    channels = ['FITC']
    records, groups, mouse = {}, {}, {}
    for lib, file, mice in (('Library 1', 'lib1.csv', range(1, 7)), ('Library 2', 'lib2.csv', range(1, 7))):
        used = [fcc.combo_key(t) for _g, t in libraries[file]]
        for m in mice:
            base = rng.lognormal(0.0, 0.3, len(used))
            for tissue in ('Spleen', 'Gut'):
                w = base * (np.where(np.arange(len(used)) == 1, 4.0, 1.0) if tissue == 'Gut' else 1.0)
                n = rng.multinomial(3000, w / w.sum())
                counts = {k: int(c) for k, c in zip(used, n)}
                key = f'Raw/{tissue} {lib} M{m}.fcs'
                sums = np.array([[rng.normal(0.4, 0.01) * counts.get(k, 0)] for k in keys])
                records[key] = {'counts': {k: counts.get(k, 0) for k in keys}, 'events': 4000,
                                'in_population': 3500, 'barcoded': 3000, 'untransduced': 400,
                                'unresolved': 100, 'not_barcoded': 500, 'marker_sums': sums,
                                'marker_channels': channels}
                state.sample_libraries[key] = lib
                groups[key], mouse[key] = tissue, f'M{m}'
    fa.save_marker_sums(fa.marker_cache_path(experiment), 'sig', {fa.ALL_EVENTS: records}, [fa.ALL_EVENTS])
    plugin.state_changed()
    page = plugin.pages[6]
    page.settings.update(counts={'signature': 'sig', 'records': {fa.ALL_EVENTS: records},
                                 'populations': [fa.ALL_EVENTS], 'source': fa.SOURCE_LIVE, 'errors': {}},
                         group_names=['Spleen', 'Gut'], sample_groups=groups, covariates={'Mouse': mouse},
                         pairing='Mouse', reference='Spleen', run_counts=True, run_freq=True, run_markers=True,
                         marker_reference='all', remove_clones=False)
    page.counts = None
    plugin.tabs.setCurrentIndex(5)
    page.refresh()
    assert page.counts.shape == (24, 16) and page.filtered.counts.shape[0] == 24
    page.start_stats()
    qtbot.waitUntil(lambda: page.stats is not None or 'failed' in page.stats_status.text(), timeout=LONG)
    stats = page.stats
    assert stats is not None, page.stats_status.text()
    assert stats['comparisons'] == ['Gut vs Spleen'] and set(stats['libraries'].values()) == {'Library 1', 'Library 2'}
    assert len(set(stats['pairing'].values())) == 12                 # M1 of Library 1 is not M1 of Library 2
    res = stats['counts'].set_index('feature')
    assert np.isfinite(res['P.Value']).all() and len(res) == 16
    assert bool(res.loc['g1', 'significant']) and bool(res.loc['h1', 'significant'])
    markers = stats['markers']
    assert markers is not None and markers['reference'] == 'all', stats['notes']
    assert np.isnan(markers['means'].loc['Raw/Gut Library 1 M1.fcs', 'h0: FITC'])      # not in Library 1
    qtbot.waitUntil(lambda: not plugin.task_running('analysis_figures') and bool(page.figures), timeout=LONG)
    _settle(plugin, qtbot)
    assert {'sample_pca Library 1', 'sample_pca Library 2', 'heatmap_all'} <= set(page.figures)
    assert all(item['fig'] is not None or key in ('marker_table', 'pathway_table', 'gene_group_table')
               for key, item in page.figures.items()), \
        {k: v['error'] for k, v in page.figures.items() if v['fig'] is None}
    assert page.loo_btn.isEnabled()
    page.start_leave_one_out()
    qtbot.waitUntil(lambda: not plugin.task_running('analysis_loo'), timeout=LONG)
    loo = page.stats.get('leave_one_out')
    assert loo is not None and len(loo['levels']) == 12 and not loo['failed'], page.loo_status.text()
    assert page.loo_status.text().startswith('Leave one out: 12 refit(s)')
    headers = [page.results_table.horizontalHeaderItem(j).text() for j in range(page.results_table.columnCount())]
    assert 'leave one out' in headers
    _settle(plugin, qtbot)
    page.start_export()
    qtbot.waitUntil(lambda: page.export_btn.isEnabled(), timeout=LONG)
    folder = page._export_folder
    samples = (folder / 'samples.csv').read_text()
    assert 'library' in samples.splitlines()[0] and 'Library 2' in samples
    settings = (folder / 'settings.txt').read_text()
    assert 'Libraries (section 2' in settings and 'Guides with the same name in several libraries: merged' in settings
    assert any(p.name.startswith('Sample_PCA_(Library_2)') for p in (folder / 'figures').iterdir())
    assert 'loo_flag' in (folder / 'results.csv').read_text().splitlines()[0]
    assert 'Synopsis\n  All events: cells against Spleen (counts test)' in settings
    assert (folder / 'synopsis.txt').read_text(encoding='utf-8').startswith('All events: 16 guides, 2 groups.')
    assert len((folder / 'synopsis_guides.csv').read_text().splitlines()) == 17
    assert any(p.name.startswith('Synopsis_All_events_-_atlas') for p in (folder / 'figures').iterdir())
    _settle(plugin, qtbot)

    # A new session reads the saved results on a worker, so the page opens
    # at once; they come back current and with the thresholds applied.
    import flowcode_analysis_page as fap

    plugin.save_state()
    controller2 = _with_unmixed_gate(FlowCodeController(experiment))
    controller2.experiment.process[fcs.PROCESS_KEY] = deepcopy(plugin.controller.experiment.process[fcs.PROCESS_KEY])
    plugin2, _bus2 = _plugin(fp, qtbot, controller2)
    page2 = plugin2.pages[6]
    plugin2.tabs.setCurrentIndex(5)
    assert page2.counts.shape == (24, 16)
    assert page2.stats is None and page2.stats_status.text() == fap.RESTORING_TEXT
    qtbot.waitUntil(lambda: page2.stats is not None, timeout=LONG)
    assert not page2._stats_stale and page2.stats_status.text().startswith('Saved results')
    assert page2.stats['comparisons'] == ['Gut vs Spleen']
    assert list(page2.stats['counts']['significant']) == list(page.stats['counts']['significant'])
    qtbot.waitUntil(lambda: not plugin2.task_running('analysis_figures') and bool(page2.figures), timeout=LONG)
    assert {'sample_pca Library 1', 'heatmap_all'} <= set(page2.figures)
    _settle(plugin2, qtbot)


def test_analysis_counts_a_phenotype_population_live_with_soft_counts(fp, qtbot, experiment, monkeypatch):
    """A marker definition ticked beside the gates: counted from the live
    unmix as memberships summed, inside the FlowCode gates, with the Counts
    test withheld and the Frequencies test run on the fractional counts."""
    import flowcode_phenotype as fph
    from PySide6.QtCore import Qt
    from honeychrome.controller_components.autospectral_functions import (
        build_af_precomputed, build_runtime_envelope, make_envelope_entry,
    )

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE],
                                                           syn.UNSTAINED_MATCHED: [syn.AF_PROFILE]}
    unstained = syn.make_events('unstained', 15000, 2)['raw'][:, 4:4 + syn.N_DET]
    entry, _sample = make_envelope_entry(unstained, syn.FLUORS, syn.spectra(), syn.af_spectra())
    envelope = build_runtime_envelope(entry, None, syn.FLUORS, syn.spectra(), syn.MAGNITUDE_CEILING)

    def precomputed_with_envelope(name):
        profile = controller.experiment.process['af_profiles'].get(name)
        return None if profile is None else build_af_precomputed(
            syn.spectra(), np.asarray(profile['spectra'], dtype=np.float64), envelope)

    controller.precomputed_for_profile = precomputed_with_envelope
    definitions = [fph.Definition('CD4 cells', 'mouse', ('CD4',), ('HA',)),
                   fph.Definition('CD3 cells', 'mouse', ('CD3',), ())]
    monkeypatch.setattr(fph, 'load_definitions', lambda *a, **k: definitions)
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    page.settings['gates'] = ['Cells']
    page.settings['phenotypes'] = ['CD4 cells (mouse)', 'CD3 cells (mouse)']
    page.refresh()
    items = {page.phenotype_list.item(i).data(Qt.UserRole): page.phenotype_list.item(i)
             for i in range(page.phenotype_list.count())}
    assert items['CD4 cells (mouse)'].checkState() == Qt.Checked
    assert not items['CD3 cells (mouse)'].flags() & Qt.ItemIsEnabled
    assert items['CD3 cells (mouse)'].checkState() == Qt.Unchecked and 'none of its markers is on the panel' in items['CD3 cells (mouse)'].toolTip()
    assert page.phenotype_names() == ['CD4 cells'] and page.population_name() == 'Cells, CD4 cells'
    assert 'CD4+ (FITC) HA- (BV421)' in page.phenotype_note.text()
    page.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page.links['files']}
    before = page._counts_signature()
    page.settings['phenotypes'] = []
    assert page._counts_signature() != before
    page.settings['phenotypes'] = ['CD4 cells (mouse)']
    page.start_count()
    qtbot.waitUntil(lambda: page.count_btn.isEnabled(), timeout=LONG)
    live = page.settings['counts']
    assert not live['errors'], live['errors']
    assert live['populations'] == ['Cells', 'CD4 cells'] and list(live['phenotypes']) == ['CD4 cells']
    soft = live['records']['CD4 cells'][syn.BACKBONE]
    hard = live['records']['Cells'][syn.BACKBONE]
    assert isinstance(soft['in_population'], float) and 0 < soft['in_population'] < soft['events']
    assert abs(sum(soft['counts'].values()) - soft['barcoded']) < 1e-6
    assert all(isinstance(v, int) for v in hard['counts'].values())
    assert page.soft_counts() and page.several_populations()
    assert page.counts.dtypes.iloc[0] == np.float64
    assert 'CD4 cells' in page.settings_text() and 'memberships summed' in page.settings_text()

    page.settings.update(group_names=['A', 'B'], reference='A', run_counts=True, run_freq=True, run_markers=False,
                         sample_groups={syn.BACKBONE: 'A', syn.OTHER_SAMPLE: 'B'})
    page.refresh()
    assert not page.chk_counts.isEnabled() and not page.counts_note.isHidden() and not page.run_counts()
    assert page.run_btn.isEnabled()
    page.settings['run_freq'] = False
    page.refresh()
    assert not page.run_btn.isEnabled() and 'phenotype population' in page.stats_status.text()

    page.congenic_check.setChecked(True)
    assert page.settings['congenic_split'] and page.phenotype_names() == ['CD4 cells']
    assert 'Not split' in page.phenotype_note.text()
    page.congenic_check.setChecked(False)

    import flowcode_gating as fcg
    view = page.gating_view
    page.section_tabs.setCurrentWidget(view)
    assert view.highlight_combo.findData(fcg.phenotype_highlight('CD4 cells')) == 2
    view.highlight_combo.setCurrentIndex(2)
    view.result = None
    controller.current_sample_path = syn.BACKBONE
    plugin._on_sample_selected(syn.BACKBONE)
    qtbot.waitUntil(lambda: view.result is not None or 'Could not' in view.status.text(), timeout=LONG)
    assert view.result is not None, view.status.text()
    rows = {g['name']: g for g in view.result['rows']}
    assert view.result['highlight'] == 'phenotype:CD4 cells' and view.result['n_focus'] > 0
    assert 0 < rows['Cells']['n_focus'] <= min(rows['Cells']['n_in'], view.result['n_focus'])
    assert view.result['n_focus'] < soft['events']
    assert view.table.horizontalHeaderItem(4).text() == 'Highlighted'
    assert view.table.item(0, 4).text() == f"{rows['Cells']['n_focus']:,}"
    assert 'highlighted events' in view.status.text()
    page.section_tabs.setCurrentIndex(1)
    _settle(plugin, qtbot)


def test_phenotype_shift_tab_runs_saves_and_restores(fp, qtbot, experiment):
    from PySide6.QtCore import Qt

    import flowcode_phenotype_shift as fsh
    import flowcode_shift_view as fshv

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    controller.experiment.samples['sample_af_profiles'] = {syn.BACKBONE: [syn.AF_PROFILE],
                                                           syn.OTHER_SAMPLE: [syn.AF_PROFILE]}
    plugin, _bus = _plugin(fp, qtbot, controller)
    _choose_backbone(plugin, qtbot)
    _load_gates(plugin, qtbot)
    _confirm(plugin, qtbot)
    _characterise(plugin, qtbot)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    titles = [page.section_tabs.tabText(i) for i in range(page.section_tabs.count())]
    assert titles.index('Phenotype shift') == titles.index('Results') + 1
    page.settings['gates'] = ['Cells']
    page.refresh()
    page.settings['use'] = {k: k in (syn.BACKBONE, syn.OTHER_SAMPLE) for k in page.links['files']}
    view = page.shift_view
    view.refresh()
    channels = [view.list_channels.item(i).data(Qt.UserRole) for i in range(view.list_channels.count())]
    assert channels == ['FITC'] and view.list_channels.item(0).checkState() == Qt.Checked
    assert 'Cells' in view.lbl_scope.text()

    view.start()
    qtbot.waitUntil(lambda: not plugin.task_running(fshv.TASK), timeout=LONG)
    _settle(plugin, qtbot)
    assert view.result is not None, view.status.text()
    assert view.result['populations'] == ['Cells']
    assert view.result['counts']['Cells'].shape == (2, len(plugin.design().combo_names), fsh.DEFAULT_K)
    assert 'fewer than the minimum of 3 mice' in view.notes.text()
    assert fsh.result_path(experiment).exists() and not view.stale_label.isVisible()

    view.reset_session()
    view.refresh()
    _settle(plugin, qtbot)
    assert view.result is not None and 'Saved results' in view.status.text()
    assert not view._stale
    view.sb_k.setValue(fsh.DEFAULT_K - 2)
    assert view._stale and not view.stale_label.isHidden()
    view.sb_k.setValue(fsh.DEFAULT_K)
    assert not view._stale


def test_immune_state_views_draw_and_respond(fp, qtbot, experiment):
    import flowcode_fsea_view as ffv
    import test_flowcode_fsea as tfs

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    plugin, _bus = _plugin(fp, qtbot, controller)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    view = page.shift_view
    result = tfs._run(tfs._mice(100, convert={tfs.CONVERTED: (0.2, tfs.M1_M3_STATE)}), False)
    view._set_result(result, 'synthetic', fresh=True)
    _settle(plugin, qtbot)
    fsea = view.fsea
    assert 'Immune states:' in view.status.text()
    assert fsea.heatmap.figure is not None and fsea.heatmap.figure._points
    calls = result['fsea']['results']
    calls = calls[calls['significant']]
    assert fsea.state in set(calls['state']) and fsea.guide in set(calls['guide'])
    assert fsea.volcano.figure is not None and fsea.profile.figure is not None
    assert fsea.table.rowCount() == len(result['fsea']['results'])
    assert fsea.states_table.rowCount() == 3

    fsea.cb_guide.setCurrentIndex(fsea.cb_guide.findData(f'G{tfs.CONVERTED}'))
    assert fsea.guide == f'G{tfs.CONVERTED}'
    assert 'abund:M1+ M3+ T' in fsea.profile.figure._member_features
    fsea.show_member('abund:M1+ M3+ T')
    assert view.views.currentWidget() is fsea.member and fsea.member.figure is not None

    fsea.sb_members.setValue(4)
    qtbot.waitUntil(lambda: not plugin.task_running(ffv.TASK), timeout=LONG)
    _settle(plugin, qtbot)
    retested = view.result['fsea']
    assert retested['settings']['min_members'] == 4 and (retested['results']['members_all'] >= 4).all()
    names = [name for name, _make in fsea.export_figures(view.result, False)]
    assert names[0] == 'Immune states T Spleen - heatmap'
    assert any(n.endswith(f'{syn.FSEA_PLANTED} volcano') for n in names)
    assert 'Immune states (Flow Set Enrichment)' in '\n'.join(fsea.settings_lines(view.result))


def test_synopsis_tab_reads_the_results_and_draws(fp, qtbot, experiment):
    from PySide6.QtCore import Qt

    import flowcode_synopsis as fsy
    import flowcode_synopsis_view as fsyv
    import test_flowcode_fsea as tfs
    import test_flowcode_synopsis as tsy

    controller = _with_unmixed_gate(FlowCodeController(experiment))
    plugin, _bus = _plugin(fp, qtbot, controller)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    titles = [page.section_tabs.tabText(i) for i in range(page.section_tabs.count())]
    assert titles.index('Synopsis') == titles.index('Phenotype shift') + 1 == titles.index('Export') - 1
    view = page.synopsis_view
    _settle(plugin, qtbot)
    page.section_tabs.setCurrentWidget(view)
    assert view.synopsis is None and view.status.text() == fsyv.EMPTY_TEXT

    page.stats = tsy._statistics()
    view.refresh()
    qtbot.waitUntil(lambda: view.synopsis is not None, timeout=LONG)
    _settle(plugin, qtbot)
    assert view.synopsis['baseline'] == 'A' and view.synopsis['groups'] == ['A', 'B', 'C']
    assert view.cb_baseline.count() == 1 and not view.cb_baseline.isEnabled()
    assert view.cb_test.count() == 2 and not view.cb_test.isHidden()
    assert view.guide_table.rowCount() == len(tsy.GUIDES) and view.calls_table.rowCount() == 16
    assert 'against A (counts test)' in view.overview.text()
    view.calls_table.sortItems(8, Qt.AscendingOrder)
    cells = [float(view.calls_table.item(r, 8).text()) for r in range(view.calls_table.rowCount())]
    assert view.calls_table.horizontalHeaderItem(8).text() == 'cells' and cells == sorted(cells) and cells[-1] > 1000
    assert view.atlas.figure is not None and view.atlas.figure._atlas_points
    assert view.dossier.figure is not None and not view.stale_label.isVisible()

    made = view.synopsis
    view.refresh()
    assert not plugin.task_running(fsyv.TASK) and view.synopsis is made

    result = tfs._run(tfs._mice(100, convert={tfs.CONVERTED: (0.2, tfs.M1_M3_STATE)}), False)
    page.shift_view._set_result(result, 'synthetic', fresh=True)
    qtbot.waitUntil(lambda: view.synopsis is not made and not plugin.task_running(fsyv.TASK), timeout=LONG)
    _settle(plugin, qtbot)
    assert view.synopsis['groups'] == ['A', 'B', 'C', 'Spleen']
    table = view.synopsis['guide_table'].set_index('guide')
    assert table.loc[tsy.GAINED, 'class'] == fsy.BOTH and table.loc[tsy.GAINED, 'groups_shift'] == 1

    view.views.setCurrentIndex(0)
    view.show_dossier(tsy.ABSENT)
    assert view.guide == tsy.ABSENT and view.views.currentWidget() is view.dossier_view
    assert view.dossier.name == f'T - {tsy.ABSENT} dossier'

    view.cb_test.setCurrentIndex(view.cb_test.findData('freq'))
    qtbot.waitUntil(lambda: view.synopsis['test'] == 'freq', timeout=LONG)
    _settle(plugin, qtbot)
    assert page.settings['synopsis']['test'] == 'freq' and view.guide == tsy.ABSENT

    assert page.shift_view._stale and view.export_plan(False)[0][0][0] == 'Synopsis All events - atlas'
    page.shift_view._stale = False
    figures, write_tables = view.export_plan(False)
    names = [name for name, _make in figures]
    assert names[0] == 'Synopsis T - atlas' and f'Synopsis T - {tsy.GAINED} dossier' in names
    assert page._report_figure_key(names[0]) == 'synopsis_atlas'
    assert page._report_figure_key(names[1]) == 'synopsis_dossier'
    assert [p.name for p in write_tables(experiment / 'out')] == ['synopsis_guides.csv', 'synopsis_calls.csv',
                                                                 'synopsis.txt']
    assert 'T: cells against A (frequencies test)' in view.settings_text()

    page._set_stats_stale(True)
    view.refresh()
    assert not view.stale_label.isHidden()
    assert 'no statistics; phenotype only' in view.settings_text()
    page.reset_session()
    assert view.synopsis is None and view.atlas.figure is None
