"""
test_flowcode_state.py
----------------------
Qt-free parts of the FlowCode plugin's first four sections: the experiment
state and its run records (flowcode_state.py), the input rules for the
backbone, matched unstained, AF library, tags and combos, the scatter gate,
cleaning and debarcoding-space helpers in flowcode_core.py, and the backbone
counts kept after the characterisation.

The plugin lives outside the package; its folder defaults to
user_plugins/flowcode in this repository and can be set with
FLOWCODE_PLUGIN_DIR.

Usage:
    pytest tests/test_flowcode_state.py -m numpy_only
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FLOWCODE_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR', _REPO / 'user_plugins' / 'flowcode'))
if str(_FLOWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWCODE_DIR))

import flowcode_core as fc  # noqa: E402
import flowcode_state as fs  # noqa: E402

pytestmark = pytest.mark.numpy_only

TAGS = ('HA', 'V5', 'AU1', 'S', 'C', 'Flag')


def _recognise(antigen):
    """Stand-in for flowcode_core.recognise_tag: exact canonical names."""
    text = str(antigen).strip()
    if text.lower().endswith(' tag'):
        text = text[:-4]
    for tag in fc.FLOWCODE_TAGS:
        if tag.upper() == text.upper():
            return tag
    return None


def _state_with_design(n_tags=4):
    state = fs.FlowCodeState(backbone_keys=['Raw/bb.fcs'], unstained_key='Raw/Unstained.fcs',
                             af_profile_names=['Unstained AutoSpectral AF'])
    channels = [f'Fluor{i}' for i in range(n_tags)]
    state.tag_selection = channels
    state.tag_names = {ch: tag for ch, tag in zip(channels, TAGS)}
    state.combos = fs.generate_combo_rows(list(state.tag_map()))
    return state


# ---------------------------------------------------------------------------
# State persistence and run records
# ---------------------------------------------------------------------------

def test_state_round_trips_through_json():
    state = _state_with_design()
    state.gates[fs.CELLS_GATE] = fs.gate_record('FSC-A', 'SSC-A', [[0, 0], [1, 0], [1, 1]], 'auto')
    state.thresholds['HA'] = fs.threshold_record(512.0, 'mixture', 42.0)
    state.mark_completed(3, 'abc')
    text = json.dumps(state.to_json())
    back = fs.FlowCodeState.from_json(json.loads(text))
    assert back.to_json() == state.to_json()
    assert back.run_status(3, 'abc') == ('done', '')


def test_from_json_fills_new_settings_and_ignores_unknown_keys():
    back = fs.FlowCodeState.from_json({'backbone_keys': ['x', 'y'], 'settings': {'fit_min_fold': 5.0},
                                       'something_else': 1})
    assert back.backbones() == ['x', 'y']
    assert back.settings['fit_min_fold'] == 5.0
    assert back.settings['backbone_max_events'] == fs.DEFAULT_SETTINGS['backbone_max_events']
    assert not hasattr(back, 'something_else')


def test_from_empty_is_default():
    assert fs.FlowCodeState.from_json(None).to_json() == fs.FlowCodeState().to_json()
    assert fs.FlowCodeState().backbone_keys is None


def test_single_backbone_states_are_read_and_keep_their_run_records():
    old = {'backbone_key': 'Raw/bb.fcs', 'match_backbone_key': 'Raw/bb.fcs', 'unstained_key': 'Raw/Unstained.fcs',
           'af_profile_names': ['Unstained AutoSpectral AF']}
    back = fs.FlowCodeState.from_json(old)
    assert back.backbone_keys == ['Raw/bb.fcs'] and back.match_backbone_keys == ['Raw/bb.fcs']
    assert 'backbone_key' not in back.to_json()
    # a single backbone digests as its key, so runs recorded before pooling stay done
    recorded = back.section_inputs(3, 'fp')['backbone']
    assert recorded == fs._digest('Raw/bb.fcs')


def test_pooled_backbones_digest_without_order():
    a = _state_with_design()
    a.backbone_keys = ['Raw/b.fcs', 'Raw/a.fcs']
    b = _state_with_design()
    b.backbone_keys = ['Raw/a.fcs', 'Raw/b.fcs']
    assert a.section_inputs(3, 'fp') == b.section_inputs(3, 'fp')
    b.backbone_keys = ['Raw/a.fcs']
    assert a.section_inputs(3, 'fp')['backbone'] != b.section_inputs(3, 'fp')['backbone']


@pytest.mark.parametrize('change, section_states', [
    (lambda s: setattr(s, 'backbone_keys', ['Raw/other.fcs']), ('stale', 'stale', 'stale')),
    (lambda s: s.backbone_keys.append('Raw/second.fcs'), ('stale', 'stale', 'stale')),
    (lambda s: s.settings.update(spill_ridge=0.1), ('done', 'stale', 'stale')),
    (lambda s: s.settings.update(export_include_controls=True), ('done', 'done', 'stale')),
    (lambda s: s.af_profile_names.append('More AF'), ('stale', 'stale', 'stale')),
    (lambda s: s.combos[0].update(include=False), ('stale', 'stale', 'stale')),
    (lambda s: s.gates[fs.CELLS_GATE]['vertices_raw'].append([2.0, 2.0]), ('stale', 'stale', 'stale')),
    (lambda s: s.thresholds['HA'].update(raw=999.0), ('done', 'stale', 'stale')),
    (lambda s: s.settings.update(fit_min_fold=4.0), ('done', 'stale', 'stale')),
    (lambda s: s.combos[0].update(name='Renamed'), ('done', 'done', 'stale')),
    (lambda s: s.settings.update(correction='pairwise'), ('done', 'done', 'stale')),
    (lambda s: s.settings.update(tag_shift_min_n=50), ('done', 'done', 'stale')),
    (lambda s: s.settings.update(expression_max_events=500), ('done', 'done', 'done')),
    (lambda s: s.settings.update(landmark_expand=2.0), ('done', 'done', 'done')),
    (lambda s: s.settings.update(debarcode_method='experiment'), ('stale', 'stale', 'stale')),
    (lambda s: s.unmixing_inputs.update(token='M2pg token'), ('done', 'done', 'done')),
    (lambda s: s.unmixing_inputs.update(final={'token': 'M2pg token', 'sets': {}}), ('done', 'done', 'stale')),
    (lambda s: setattr(s, 'export_samples', ['Raw/a.fcs']), ('done', 'done', 'stale')),
    (lambda s: setattr(s, 'output_folder', '/tmp/out'), ('done', 'done', 'stale')),
])
def test_changes_invalidate_the_sections_that_depend_on_them(change, section_states):
    state = _state_with_design()
    state.gates[fs.CELLS_GATE] = fs.gate_record('FSC-A', 'SSC-A', [[0, 0], [1, 0], [1, 1]], 'auto')
    state.thresholds = {t: fs.threshold_record(100.0, 'tail', 1.0) for t in state.tag_map()}
    for section in (3, 4, 5):
        state.mark_completed(section, 'fp')
    change(state)
    assert tuple(state.run_status(s, 'fp')[0] for s in (3, 4, 5)) == section_states


def test_spectral_model_change_is_reported():
    state = _state_with_design()
    state.mark_completed(3, 'old')
    status, reason = state.run_status(3, 'new')
    assert status == 'stale' and 'spectral model changed' in reason
    assert state.fingerprint == 'old'


def test_model_fingerprint_follows_spectra_tags_and_detectors():
    rng = np.random.default_rng(0)
    spectra = rng.random((5, 8))
    names = [f'F{i}' for i in range(5)]
    det = [f'D{i}' for i in range(8)]
    base = fs.model_fingerprint(spectra, names, ['F0', 'F1', 'F2'], det)
    assert base == fs.model_fingerprint(spectra.copy(), names, ['F0', 'F1', 'F2'], det)
    tweaked = spectra.copy()
    tweaked[4, 3] += 1e-3
    assert fs.model_fingerprint(tweaked, names, ['F0', 'F1', 'F2'], det) != base
    assert fs.model_fingerprint(spectra, names, ['F0', 'F1', 'F3'], det) != base
    assert fs.model_fingerprint(spectra, names, ['F0', 'F1', 'F2'], det[::-1]) != base


# ---------------------------------------------------------------------------
# Section 1
# ---------------------------------------------------------------------------

SAMPLES = {
    'all_samples': {
        'Raw/Unstained cells.fcs': 'Unstained cells',
        'Raw/Unstained beads.fcs': 'Unstained beads',
        'Raw/US_spleen.fcs': 'US_spleen',
        'Raw/Single stain controls/CD4 BV421.fcs': 'CD4 BV421',
        'Raw/Backbone.fcs': 'Backbone',
        'Raw/Mouse 1.fcs': 'Mouse 1',
    },
    'unstained_samples': ['Raw/US_spleen.fcs'],
    'single_stain_controls': ['Raw/Single stain controls/CD4 BV421.fcs'],
}

# Teresa FlowCode layout: the reference group appears twice, once tagged as
# controls and once inside the raw samples folder.
SAMPLES_COPIED_CONTROLS = {
    'all_samples': {
        'Raw/FC control/F1 FC_FRET_01.fcs': 'F1 FC_FRET_01',
        'Raw/FC control/F2 FC_FRET_02.fcs': 'F2 FC_FRET_02',
        'Raw/FC control/F6 Unstained.fcs': 'F6 Unstained',
        'Raw/Fully stained/A5 T13.fcs': 'A5 T13',
        'Raw/Reference Group/A8 HA BV421 (Cells)_Plate_001.fcs': 'A8 HA BV421 (Cells)_Plate_001',
        'Raw/Reference Group/E7 Unstained (Cells)_Plate_001.fcs': 'E7 Unstained (Cells)_Plate_001',
        'Raw/Reference Group/G1 CXCR5 Spark Blue 574 (Beads).fcs': 'G1 CXCR5 Spark Blue 574 (Beads)',
        'Raw/Old controls/CD4 BV421.fcs': 'CD4 BV421',
        'Controls/A8 HA BV421 (Cells)_Plate_001.fcs': 'A8 HA BV421 (Cells)_Plate_001',
        'Controls/CD4 BV421.fcs': 'CD4 BV421',
        'Raw/Pool backbone.fcs': 'Pool backbone',
    },
    'unstained_samples': ['Raw/FC control/F6 Unstained.fcs'],
    'single_stain_controls': ['Controls/A8 HA BV421 (Cells)_Plate_001.fcs', 'Controls/CD4 BV421.fcs'],
}


def test_unstained_candidates_follow_the_optimization_rules():
    assert fs.unstained_candidates(SAMPLES) == ['Raw/Unstained cells.fcs', 'Raw/US_spleen.fcs']


def test_backbone_candidates_exclude_controls_and_unstained():
    assert fs.backbone_candidates(SAMPLES) == ['Raw/Backbone.fcs', 'Raw/Mouse 1.fcs']


def test_copies_of_controls_are_controls_too():
    assert fs.backbone_candidates(SAMPLES_COPIED_CONTROLS) == [
        'Raw/FC control/F1 FC_FRET_01.fcs', 'Raw/FC control/F2 FC_FRET_02.fcs', 'Raw/Fully stained/A5 T13.fcs',
        'Raw/Pool backbone.fcs']
    controls = fs.control_sample_keys(SAMPLES_COPIED_CONTROLS)
    assert 'Raw/Old controls/CD4 BV421.fcs' in controls
    assert 'Raw/Reference Group/G1 CXCR5 Spark Blue 574 (Beads).fcs' in controls
    assert 'Raw/FC control/F6 Unstained.fcs' in controls


def test_backbone_and_fret_samples_are_the_default_backbones():
    assert fs.default_backbone_keys(SAMPLES_COPIED_CONTROLS) == [
        'Raw/FC control/F1 FC_FRET_01.fcs', 'Raw/FC control/F2 FC_FRET_02.fcs', 'Raw/Pool backbone.fcs']
    assert fs.default_backbone_keys(SAMPLES) == ['Raw/Backbone.fcs']
    assert fs.default_backbone_keys({'all_samples': {'Raw/Mouse 1.fcs': 'Mouse 1'}}) == []


def test_af_profiles_are_matched_to_their_source_sample(tmp_path):
    profiles = {
        'A': {'source_fcs': 'Raw/US_spleen.fcs', 'spectra': [[1, 0], [0, 1]]},
        'B': {'source_fcs': str(tmp_path / 'Raw' / 'US_spleen.fcs'), 'spectra': [[1, 1]]},
        'C': {'source_fcs': '/elsewhere/US_spleen.fcs', 'spectra': [[1, 1]]},
        'D': {'source_fcs': 'Raw/Unstained cells.fcs', 'spectra': [[1, 1]]},
    }
    assert fs.af_profiles_for_sample(profiles, 'Raw/US_spleen.fcs', tmp_path) == ['A', 'B']


def test_af_library_problems():
    det = ['D1', 'D2']
    profiles = {
        'one': {'spectra': [[1, 0]], 'channel_names': det},
        'two': {'spectra': [[1, 0], [0, 1]], 'channel_names': det},
        'other_det': {'spectra': [[1, 0], [0, 1]], 'channel_names': ['X', 'Y']},
    }
    assert fs.af_library_problem(profiles, [], det, 'US')[1].startswith("No AF profile extracted from 'US'")
    assert fs.af_library_problem(profiles, ['one'], det)[1] is not None
    assert fs.af_library_problem(profiles, ['two'], det) == (2, None)
    assert fs.af_library_problem(profiles, ['one', 'two'], det) == (3, None)
    assert 'different detectors' in fs.af_library_problem(profiles, ['other_det'], det)[1]
    assert fs.af_library_spectra(profiles, ['one', 'two']).shape == (3, 2)


def test_export_af_library_adds_the_threshold_profiles():
    assert fs.export_af_profile_names(['E7'], ['F6']) == ['E7', 'F6']
    assert fs.export_af_profile_names(['F6', 'E7'], ['F6']) == ['F6', 'E7']
    assert fs.export_af_profile_names(None, ['F6']) == ['F6']
    assert fs.export_af_profile_names(['E7'], []) == ['E7']
    assigned = {'s1': ['E7'], 's2': ['F6'], 'control': ['E7']}
    out = fs.export_af_assignments(assigned, ['s1', 's2', 's3'], ['F6'])
    assert out == {'s1': ['E7', 'F6'], 's2': ['F6'], 's3': ['F6'], 'control': ['E7']}
    assert assigned['s1'] == ['E7']


def test_best_match_breaks_ties_by_experiment_order():
    assert fc.best_match({'b': 0.9, 'a': 0.9, 'c': 0.5}, ['a', 'b', 'c']) == 'a'
    assert fc.best_match({'b': 0.95, 'a': 0.9}, ['a', 'b']) == 'b'
    assert fc.best_match({'a': np.nan}, ['a']) is None


def test_matching_scores_identical_scatter_highest():
    rng = np.random.default_rng(3)
    lim = (0.0, 1.0)

    def hist(centre, n=20000):
        xy = rng.normal(centre, 0.08, (n, 2))
        return fc.scatter_histogram(xy[:, 0], xy[:, 1], lim, lim)

    backbone = hist(0.4)
    scores = {'same': fc.bhattacharyya(backbone, hist(0.4)), 'shifted': fc.bhattacharyya(backbone, hist(0.6))}
    assert scores['same'] > 0.95 > scores['shifted']
    assert fc.best_match(scores, ['shifted', 'same']) == 'same'


# ---------------------------------------------------------------------------
# Section 2
# ---------------------------------------------------------------------------

CHANNELS = ['BV421', 'BV510', 'PE', 'APC', 'FITC', 'PerCP']
ANTIGENS = {'BV421': 'HA', 'BV510': 'V5 tag', 'PE': 'AU1', 'APC': 'CD4', 'FITC': 'S', 'PerCP': ''}


def test_auto_tags_match_canonical_names_only():
    assert fs.auto_tag_channels(CHANNELS, ANTIGENS, _recognise) == ['BV421', 'BV510', 'PE', 'FITC']
    assert fs.tag_name_for_channel('BV510', 'V5 tag', _recognise) == 'V5'
    assert fs.tag_name_for_channel('APC', 'CD4', _recognise) == 'CD4'
    assert fs.tag_name_for_channel('PerCP', '', _recognise) == 'PerCP'


def test_generated_combos_keep_edits_for_the_same_tag_set():
    rows = fs.generate_combo_rows(['V5', 'HA', 'AU1', 'S'])
    assert len(rows) == 4
    assert rows[0] == {'include': True, 'name': 'AU1_HA_S', 'tags': ['AU1', 'HA', 'S']}
    rows[0].update(include=False, name='Gata3')
    again = fs.generate_combo_rows(['V5', 'HA', 'AU1', 'S', 'C'], previous=rows)
    assert len(again) == 10
    kept = [r for r in again if sorted(r['tags']) == ['AU1', 'HA', 'S']][0]
    assert kept == {'include': False, 'name': 'Gata3', 'tags': ['AU1', 'HA', 'S']}


def test_ten_tags_give_120_combos():
    assert len(fs.generate_combo_rows(fc.FLOWCODE_TAGS)) == 120


def test_csv_tags_map_to_channels_and_report_unknown_tags():
    csv = [('Gata3', ('HA', 'v5', 'AU1')), ('Tbet', ('HA', 'S', 'Ollas'))]
    picked, wanted, unmatched = fs.csv_tag_channels(csv, CHANNELS, ANTIGENS, _recognise)
    assert picked == ['BV421', 'BV510', 'PE', 'FITC']
    assert wanted['V5'] == 'BV510'
    assert unmatched == ['Ollas']
    tag_for_csv = {k: fs.tag_name_for_channel(ch, ANTIGENS[ch], _recognise) for k, ch in wanted.items()}
    rows = fs.csv_combo_rows(csv, tag_for_csv)
    assert rows[0] == {'include': True, 'name': 'Gata3', 'tags': ['HA', 'V5', 'AU1']}


def test_two_channels_with_the_same_tag_are_reported():
    assert fs.duplicate_tag_problems({'BV421': 'HA', 'PE': 'V5', 'APC': 'ha'}) == [
        'BV421, APC all carry tag HA; untick all but one.']
    assert fs.duplicate_tag_problems({'BV421': 'HA', 'PE': 'V5'}) == []


def test_tag_and_combo_problems():
    errors, warnings = fs.tag_problems({'HA': 'BV421', 'V5': 'BV510'}, CHANNELS)
    assert errors and not warnings
    errors, warnings = fs.tag_problems({'HA': 'BV421', 'V5': 'BV510', 'AU1': 'PE'}, CHANNELS)
    assert not errors and warnings
    errors, _ = fs.tag_problems({'HA': 'BV421', 'V5': 'BV510', 'AU1': 'Missing', 'S': 'FITC'}, CHANNELS)
    assert any('Missing' in e for e in errors)

    tags = ['HA', 'V5', 'AU1', 'S']
    rows = fs.generate_combo_rows(tags)
    assert fs.combo_problems(rows, tags, CHANNELS) == []
    rows[1]['name'] = rows[0]['name']
    assert any('used twice' in p for p in fs.combo_problems(rows, tags, CHANNELS))
    rows[1]['name'] = 'a,b'
    assert any('commas' in p for p in fs.combo_problems(rows, tags, CHANNELS))
    rows[1]['name'] = 'BV421'
    assert fs.combo_problems(rows, tags, ['FC BV421']) != []
    for row in rows:
        row['include'] = False
    assert fs.combo_problems(rows, tags, CHANNELS) == ['Include at least one combination.']


def test_design_from_state_uses_included_combos():
    state = _state_with_design(4)
    state.combos[0]['include'] = False
    design = fs.design_from_state(state)
    assert design.tags == ('HA', 'V5', 'AU1', 'S')
    assert design.fluors == ('Fluor0', 'Fluor1', 'Fluor2', 'Fluor3')
    assert design.n_combos == 3


# ---------------------------------------------------------------------------
# Section 3 helpers in flowcode_core
# ---------------------------------------------------------------------------

def test_cleaning_drops_saturated_detectors_and_scatter():
    fl = np.array([[10.0, 20.0], [262144.0, 5.0], [10.0, 10.0], [10.0, 10.0]])
    sc = np.array([[100.0, 100.0], [100.0, 100.0], [4194304.0, 100.0], [100.0, 4194000.0]])
    keep = fc.clean_events_mask(fl, sc, 262144.0, [4194304.0, 4194304.0])
    assert keep.tolist() == [True, False, False, False]


def test_densified_polygon_keeps_the_shape():
    square = fc.rectangle_vertices(0, 2, 0, 1)
    dense = fc.densify_closed_polygon(square, 8)
    assert dense.shape == (32, 2)
    assert fc.polygon_area(dense) == pytest.approx(2.0)
    x, y = np.array([1.0, 3.0, 1.0]), np.array([0.5, 0.5, -0.1])
    assert fc.points_in_polygon(x, y, dense).tolist() == [True, False, False]
    assert fc.points_in_polygon(x, y, square).tolist() == [True, False, False]


def test_map_vertices_round_trips():
    v = np.array([[1.0, 10.0], [100.0, 1000.0], [10.0, 100.0]])
    to = fc.map_vertices(v, np.log10, np.sqrt)
    back = fc.map_vertices(to, lambda a: 10 ** a, np.square)
    np.testing.assert_allclose(back, v)


def _scatter_population(rng, n_single=6000, n_double=600):
    area_s = rng.normal(60000, 8000, n_single)
    height_s = area_s / rng.normal(1.05, 0.03, n_single)
    area_d = rng.normal(120000, 15000, n_double)
    height_d = area_d / rng.normal(1.9, 0.1, n_double)
    area = np.concatenate([area_s, area_d])
    height = np.concatenate([height_s, height_d])
    is_single = np.r_[np.ones(n_single, bool), np.zeros(n_double, bool)]
    return area, height, is_single


def test_auto_singlet_gate_separates_doublets():
    rng = np.random.default_rng(7)
    area, height, is_single = _scatter_population(rng)
    cutoff, poly = fc.auto_singlet_gate(area, height, 4194304.0, 4194304.0, quantile=0.85)
    inside = fc.points_in_polygon(area, height, poly)
    assert inside[is_single].mean() > 0.9
    assert inside[~is_single].mean() < 0.02
    assert cutoff == pytest.approx(fc.singlet_ratio_cutoff(area, height, 0.85))


def _landmark_population(rng, n=8000, n_bright=600):
    n_det, tags = 16, 4
    ch = np.arange(n_det)
    spectra = np.array([np.exp(-0.5 * ((ch - c) / 1.2) ** 2) for c in (2, 5, 8, 11, 14)])
    abund = rng.gamma(0.5, 100.0, (n, 5))
    bright = rng.choice(n, n_bright, replace=False)
    abund[bright[:, None], rng.permuted(np.tile(np.arange(tags), (n_bright, 1)), axis=1)[:, :3]] += 5000.0
    raw = abund @ spectra + rng.normal(0, 5, (n, n_det))
    scatter = rng.normal([50000, 30000], [15000, 12000], (n, 2))
    scatter[bright] = rng.normal([80000, 20000], [3000, 2000], (n_bright, 2))
    return raw, scatter, spectra, bright


def test_landmark_gate_surrounds_the_triple_positive_events():
    rng = np.random.default_rng(11)
    raw, scatter, spectra, bright = _landmark_population(rng)
    result = fc.auto_landmark_gate(raw, scatter, spectra, [0, 1, 2, 3], n_per_tag=100)
    assert result.reason is None
    assert np.isin(result.landmark_idx, bright).mean() > 0.95
    inside = fc.points_in_polygon(scatter[:, 0], scatter[:, 1], result.vertices)
    assert inside[bright].mean() > 0.5
    assert inside.sum() < 0.3 * len(inside)


def test_landmark_gate_falls_back_with_a_reason():
    rng = np.random.default_rng(12)
    raw, scatter, spectra, _ = _landmark_population(rng)
    assert fc.auto_landmark_gate(raw, scatter, spectra, [0, 1]).reason == 'fewer than 3 tags'
    few = fc.auto_landmark_gate(raw[:40], scatter[:40], spectra, [0, 1, 2, 3])
    assert few.vertices is None and 'landmark events' in few.reason
    tiny = fc.auto_landmark_gate(raw, scatter, spectra, [0, 1, 2, 3], to_display=lambda v: v * 1e-9)
    assert tiny.vertices is None and 'too small' in tiny.reason


def test_rescue_floor_prefers_optimization_thresholds():
    names = ['A', 'B', 'C']
    un = np.arange(300, dtype=float).reshape(100, 3)
    floor, source = fc.rescue_floor([1.0, 2.0, 3.0], names, ['C', 'A'], un[:, [2, 0]])
    assert source == 'optimization' and floor.tolist() == [3.0, 1.0]
    floor, source = fc.rescue_floor([1.0, 2.0], names, ['C', 'A'], un[:, [2, 0]])
    assert source == 'unstained'
    np.testing.assert_allclose(floor, np.quantile(un[:, [2, 0]], 0.995, axis=0))
    assert fc.rescue_floor(None, names, ['A'], un[:, [0]])[1] == 'unstained'


def test_initial_thresholds_are_raw_and_respect_the_floor():
    rng = np.random.default_rng(5)
    neg = rng.normal(0, 150, 20000)
    pos = rng.lognormal(np.log(20000), 0.3, 6000)
    bimodal = np.concatenate([neg, pos])
    flat = rng.normal(0, 150, 26000)
    values = np.column_stack([bimodal, flat, flat])
    floor = np.array([np.nan, np.nan, 5000.0])
    out = fc.initial_thresholds(values, floor, 262144.0)
    assert out[0]['method'] == 'mixture'
    assert 500 < out[0]['raw'] < 15000
    assert out[1]['method'] in ('tail', 'mixture') and out[1]['raw'] < 2000
    assert out[2]['method'] == 'floor' and out[2]['raw'] == pytest.approx(5000.0, rel=1e-3)
    inverse = fc.logicle_inverse(262144.0)
    for item in out:
        assert inverse(np.array([item['transformed']]))[0] == pytest.approx(item['raw'], rel=1e-9)


def test_debarcoding_space_shapes_and_floors():
    rng = np.random.default_rng(9)
    n_det = 12
    ch = np.arange(n_det)
    spectra = np.array([np.exp(-0.5 * ((ch - c) / 1.5) ** 2) for c in (2, 5, 8, 10)])
    af = np.array([np.exp(-0.5 * ((ch - c) / w) ** 2) for c, w in ((3, 4.0), (6, 5.0), (4, 6.0))])
    names = ['F0', 'F1', 'F2', 'F3']
    bb = rng.gamma(0.8, 2000.0, (3000, 4)) @ spectra + rng.gamma(2.0, 300.0, 3000)[:, None] * af[rng.integers(0, 3, 3000)]
    un = rng.gamma(2.0, 300.0, 2000)[:, None] * af[rng.integers(0, 3, 2000)] + rng.normal(0, 10, (2000, n_det))
    space = fc.debarcoding_space(bb, un, spectra, names, ['F1', 'F3', 'F0'], af)
    assert space.backbone.shape == (3000, 4) and space.unstained.shape == (2000, 4)
    assert space.af_idx.min() >= 1 and space.af_idx.max() <= 3
    assert np.max(np.abs(space.af_mean)) == pytest.approx(1.0)
    np.testing.assert_allclose(space.floor, np.quantile(space.unstained[:, [1, 3, 0]], 0.995, axis=0))
    assert space.rescue_floor_source == 'unstained'
    space2 = fc.debarcoding_space(bb, un, spectra, names, ['F1', 'F3', 'F0'], af, pos_thresholds=[1, 2, 3, 4])
    assert space2.rescue_floor.tolist() == [2.0, 4.0, 1.0]


# ---------------------------------------------------------------------------
# Section 4 records
# ---------------------------------------------------------------------------

def test_settings_of_the_earlier_error_model_are_migrated():
    state = fs.FlowCodeState.from_json({'settings': {'af_fret_term': True, 'correction': 'global'}})
    assert 'af_fret_term' not in state.settings and state.settings['correction'] == 'pairwise'
    kept = fs.FlowCodeState.from_json({'settings': {'correction': 'none'}})
    assert kept.settings['correction'] == 'none'


def test_final_unmix_settings_of_earlier_versions_are_dropped():
    state = fs.FlowCodeState.from_json({'settings': {'use_variants': True, 'n_passes': 3, 'threads': 4}})
    assert not set(fs.RETIRED_SETTINGS) & set(state.settings)
    assert state.settings['debarcode_method'] == 'r1'


def test_section_three_records_from_before_the_debarcode_setting_stay_done():
    state = _state_with_design()
    state.mark_completed(3, 'fp')
    del state.completed['3']['debarcode']
    assert state.run_status(3, 'fp')[0] == 'done'
    state.settings['debarcode_method'] = 'experiment'
    assert state.run_status(3, 'fp') == ('stale', 'debarcoding unmix changed')
    state.mark_completed(3, 'fp')
    state.unmixing_inputs = {'token': 'M2pg'}
    assert state.run_status(3, 'fp') == ('stale', 'debarcoding unmix changed')
    state.settings['debarcode_method'] = 'r1'
    assert state.run_status(3, 'fp') == ('stale', 'debarcoding unmix changed')


def test_unmixing_inputs_are_not_saved():
    state = _state_with_design()
    state.unmixing_inputs = {'debarcode': 'x', 'final': {'token': 'y'}}
    data = state.to_json()
    assert 'unmixing_inputs' not in data
    assert fs.FlowCodeState.from_json(data).unmixing_inputs == {}


def test_section_settings_are_split_between_characterisation_and_export():
    assert set(fs.SECTION4_SETTINGS) == {f.name for f in dataclasses.fields(fc.CharacteriseSettings)}
    assert set(fs.SECTION5_SETTINGS) == {'tag_shift_min_n', 'correction', 'regressor', 'tandem_tags',
                                         'tandem_z_min', 'tandem_min_combos', 'tandem_max_events',
                                         'debarcode_after_correction'}
    assert not set(fs.RETIRED_SETTINGS) & set(fs.DEFAULT_SETTINGS)
    assert not set(fs.SECTION4_SETTINGS) & set(fs.SECTION5_SETTINGS)
    for key in fs.SECTION3_SETTINGS + fs.SECTION4_SETTINGS + fs.SECTION5_SETTINGS:
        assert key in fs.DEFAULT_SETTINGS


def test_too_few_rescue_settings_default_on_and_stale_the_characterisation():
    assert fs.DEFAULT_SETTINGS['rescue_toofew'] is True
    assert fc.TooFewSettings.from_settings(fs.DEFAULT_SETTINGS) == fc.TooFewSettings()
    state = _state_with_design(4)
    state.mark_completed(4, 'fp')
    state.settings['rescue_toofew_min_frac'] = 0.5
    assert state.run_status(4, 'fp') == ('stale', 'section 4 settings changed')


def test_tandem_settings_default_to_automatic_ticks_and_only_stale_the_export():
    assert fs.DEFAULT_SETTINGS['tandem_tags'] is None
    assert fs.DEFAULT_SETTINGS['tandem_z_min'] == 3.0 and fs.DEFAULT_SETTINGS['tandem_min_combos'] == 5
    state = _state_with_design(4)
    state.mark_completed(4, 'fp')
    state.mark_completed(5, 'fp')
    state.settings['tandem_tags'] = ['BV421']
    assert state.run_status(4, 'fp') == ('done', '')
    assert state.run_status(5, 'fp') == ('stale', 'section 5 settings changed')
    back = fs.FlowCodeState.from_json(json.loads(json.dumps(state.to_json())))
    assert back.settings['tandem_tags'] == ['BV421']
    assert fs.FlowCodeState.from_json({'settings': {}}).settings['tandem_max_events'] == 2_000_000


def test_calibration_settings_default_on_and_stale_only_the_export():
    assert fs.DEFAULT_SETTINGS['calib_combos'] is True and fs.DEFAULT_SETTINGS['calib_brightness'] is False
    assert {k: fs.DEFAULT_SETTINGS[k] for k in fs.CALIBRATION_SETTINGS} == fc.CALIB_DEFAULTS
    state = _state_with_design(4)
    state.mark_completed(4, 'fp')
    state.mark_completed(5, 'fp')
    state.settings['calib_brightness'] = True
    assert state.run_status(4, 'fp') == ('done', '')
    assert state.run_status(5, 'fp') == ('stale', 'combination calibration settings changed')
    # an export recorded before the calibration existed
    state.settings['calib_brightness'] = False
    del state.completed['5']['calibration']
    assert state.run_status(5, 'fp') == ('stale', 'combination calibration added')
    old = fs.FlowCodeState.from_json({'settings': {'correction': 'combo'}})
    assert old.settings['calib_combos'] is True and old.settings['calib_min_events'] == 40


def test_combo_overrides_are_an_export_input_and_leave_the_default_untouched():
    assert fs.DEFAULT_SETTINGS['combo_overrides'] == {} and fs.OVERRIDE_SETTINGS == ('combo_overrides',)
    state = _state_with_design(4)
    state.mark_completed(4, 'fp')
    state.mark_completed(5, 'fp')
    assert state.section_inputs(5, 'fp')['overrides'] is None
    new = state.with_combo_override('C_S_V5', 0.0)
    assert new == {'C_S_V5': 0.0} and state.settings['combo_overrides'] == {}
    assert fs.DEFAULT_SETTINGS['combo_overrides'] == {}
    assert state.with_combo_override('C_S_V5', 7.0) == {'C_S_V5': fc.CALIB_MAX_SCALE}
    assert state.with_combo_override('C_S_V5', -1.0) == {'C_S_V5': 0.0}
    state.settings['combo_overrides'] = new
    assert state.run_status(4, 'fp') == ('done', '')
    assert state.run_status(5, 'fp') == ('stale', 'FRET correction overrides changed')
    back = fs.FlowCodeState.from_json(json.loads(json.dumps(state.to_json())))
    assert back.settings['combo_overrides'] == {'C_S_V5': 0.0}
    assert back.with_combo_override('C_S_V5', None) == {}
    state.settings['combo_overrides'] = state.with_combo_override('C_S_V5', None)
    assert state.run_status(5, 'fp') == ('done', '')
    old = fs.FlowCodeState.from_json({'settings': {'correction': 'combo'}})
    assert old.settings['combo_overrides'] == {}


def test_sample_thresholds_follow_their_mode_and_only_stale_the_export():
    state = _state_with_design(3)
    tags = list(state.tag_map())
    assert state.sample_threshold_mode == fs.SAMPLE_THRESHOLDS_AUTO
    assert state.sample_threshold_values(tags) is None
    state.sample_thresholds = {t: fs.threshold_record(100.0 * (i + 1), 'mixture', 50.0)
                               for i, t in enumerate(tags)}
    assert state.sample_threshold_values(tags) == [100.0, 200.0, 300.0]
    state.mark_completed(4, 'fp')
    state.mark_completed(5, 'fp')
    state.sample_thresholds[tags[0]].update(raw=150.0, method='manual')
    assert state.sample_threshold_values(tags) == [100.0, 200.0, 300.0]
    assert state.run_status(5, 'fp') == ('done', '')
    state.sample_threshold_mode = fs.SAMPLE_THRESHOLDS_INDIVIDUAL
    assert state.sample_threshold_values(tags) == [150.0, 200.0, 300.0]
    assert state.run_status(4, 'fp') == ('done', '')
    assert state.run_status(5, 'fp') == ('stale', 'sample debarcoding thresholds changed')
    state.sample_threshold_mode = fs.SAMPLE_THRESHOLDS_BACKBONE
    assert state.sample_threshold_values(tags) is None
    assert state.section_inputs(5, 'fp')['sample_thresholds'] is None
    back = fs.FlowCodeState.from_json(json.loads(json.dumps(state.to_json())))
    assert back.sample_threshold_mode == fs.SAMPLE_THRESHOLDS_BACKBONE and back.sample_thresholds == state.sample_thresholds


def test_states_from_before_sample_thresholds_reuse_the_backbone_thresholds():
    old = fs.FlowCodeState.from_json({'backbone_keys': ['Raw/bb.fcs']})
    assert old.sample_threshold_mode == fs.SAMPLE_THRESHOLDS_BACKBONE
    assert fs.FlowCodeState.from_json({'sample_threshold_mode': 'x'}).sample_threshold_mode == \
        fs.SAMPLE_THRESHOLDS_AUTO


def test_sample_threshold_inputs_follow_samples_libraries_and_the_debarcoding_space():
    state = _state_with_design(3)
    lists = {'Raw/a.fcs': ['AF a'], 'Raw/b.fcs': ['AF b']}
    base = state.sample_threshold_inputs('fp', lists)
    assert base == state.sample_threshold_inputs('fp', dict(reversed(list(lists.items()))))
    assert base != state.sample_threshold_inputs('fp', {'Raw/a.fcs': ['AF a']})
    assert base != state.sample_threshold_inputs('fp', {'Raw/a.fcs': ['AF a'], 'Raw/b.fcs': ['AF c']})
    assert base != state.sample_threshold_inputs('other', lists)
    state.gates[fs.CELLS_GATE] = {'vertices_raw': [[0, 0], [1, 0], [1, 1]]}
    assert base != state.sample_threshold_inputs('fp', lists)


def test_backbone_counts_follow_combos_through_a_rename():
    state = _state_with_design(4)
    design = fs.design_from_state(state)
    ids = np.array([0, 1, 1, 2, 2, 2, 4, 5, 5])
    summary = fc.debarcode_summary(ids, design, rescued=3, rescued_toofew=4)
    state.backbone_summary = fs.backbone_summary(summary, design)
    assert state.backbone_summary['untransduced'] == 1
    assert state.backbone_summary['unresolved'] == 2
    assert state.backbone_summary['rescued'] == 3 and state.backbone_summary['rescued_toofew'] == 4
    older = {k: v for k, v in summary.items() if k != 'rescued_toofew'}
    assert fs.backbone_summary(older, design)['rescued_toofew'] == 0
    counts = state.combo_backbone_counts()
    assert counts == {design.combo_names[0]: 2, design.combo_names[1]: 3, design.combo_names[2]: 0,
                      design.combo_names[3]: 1}

    state.combos[1]['name'] = 'Gata3'
    assert state.combo_backbone_counts()['Gata3'] == 3
    back = fs.FlowCodeState.from_json(json.loads(json.dumps(state.to_json())))
    assert back.combo_backbone_counts() == state.combo_backbone_counts()


def test_backbone_counts_are_empty_before_a_run():
    assert _state_with_design(4).combo_backbone_counts() == {}


# ---------------------------------------------------------------------------
# Several libraries
# ---------------------------------------------------------------------------

def test_default_library_pattern_finds_the_library_after_librar():
    import re

    one = re.compile(fs.default_library_pattern('Library 1'), re.IGNORECASE)
    for name in ('A1 SILPL_Library1_LSRB31.3a_001_SI LPL', 'Spleen Library 1 m3', 'axLN_library_1', 'Libraries1'):
        assert one.search(name), name
    for name in ('Spleen_Library10_m1', 'Spleen_Library2_m1', 'JA_007 Metab'):
        assert not one.search(name), name
    letter = re.compile(fs.default_library_pattern('Library A'), re.IGNORECASE)
    assert letter.search('LibraryA_m1') and not letter.search('LibraryAB_m1')
    named = re.compile(fs.default_library_pattern('Metab'), re.IGNORECASE)
    assert named.search('Liver_Library_Metab_m2')


def test_assign_libraries_by_pattern():
    libraries = [fs.new_library([])]
    libraries.append(fs.new_library(libraries))
    assert [lib['name'] for lib in libraries] == ['Library 1', 'Library 2']
    names = {'a': 'A1 Spleen_Library1_001', 'b': 'B1 Spleen_Library2_008', 'c': 'C1 FMO CD8',
             'd': 'D1 Liver_Library2_009'}
    keys = list(names)
    out, matched = fs.assign_libraries(keys, names, libraries, {})
    assert out == {'a': 'Library 1', 'b': 'Library 2', 'd': 'Library 2'}
    assert matched == {'Library 1': 1, 'Library 2': 2}
    current = {'a': 'Library 2', 'c': fs.OTHER_LIBRARY, 'gone': 'Library 9'}
    filled, _m = fs.assign_libraries(keys, names, libraries, current, only_unassigned=True)
    assert filled == {'a': 'Library 2', 'b': 'Library 2', 'c': fs.OTHER_LIBRARY, 'd': 'Library 2'}
    again, _m = fs.assign_libraries(keys, names, libraries, current)
    assert again == {'a': 'Library 1', 'b': 'Library 2', 'c': fs.OTHER_LIBRARY, 'd': 'Library 2'}
    libraries[0]['pattern'] = '('
    with pytest.raises(ValueError):
        fs.assign_libraries(keys, names, libraries, {})


def test_library_rows_are_the_union_of_triplets_named_by_triplet():
    state = _state_with_design(5)
    state.combos[0]['include'] = False
    first = sorted(state.combos[0]['tags'], key=str.upper)
    state.enable_libraries()
    state.libraries[0]['combos'] = [['Nrip1', first], ['NT 1', ['HA', 'V5', 'S']]]
    state.libraries[1]['combos'] = [['Enpp5', [t.upper() for t in first]], ['NT 1', ['V5', 'S', 'C']]]
    tag_for_csv = {t.upper(): t for t in state.tag_map()}
    state.combos = fs.library_combo_rows(state.libraries, tag_for_csv, previous=state.combos)
    assert [row['name'] for row in state.combos] == [fc.combo_key(first), 'HA_S_V5', 'C_S_V5']
    assert state.combos[0]['names'] == {'Library 1': 'Nrip1', 'Library 2': 'Enpp5'}
    assert state.combos[0]['include'] is False and state.combos[1]['include'] is True
    assert state.combos[2]['names'] == {'Library 2': 'NT 1'}
    assert fs.shared_guide_names(state.libraries) == {'NT 1': ['Library 1', 'Library 2']}
    design = fs.design_from_state(state)
    assert list(design.combo_names) == ['HA_S_V5', 'C_S_V5']
    assert state.csv_source() == state.libraries[0]['combos'] + state.libraries[1]['combos']


def test_library_changes_only_stale_the_export():
    state = _state_with_design(4)
    for n in (3, 4, 5):
        state.mark_completed(n, 'fp')
    assert state.section_inputs(5, 'fp')['libraries'] is None
    state.enable_libraries()
    assert state.run_status(5, 'fp') == ('stale', 'libraries or library assignments changed')
    state.mark_completed(5, 'fp')
    state.libraries[0]['label'] = 'Metabolism A'
    state.sample_libraries['Raw/s1.fcs'] = 'Library 1'
    assert state.run_status(3, 'fp') == ('done', '')
    assert state.run_status(4, 'fp') == ('done', '')
    assert state.run_status(5, 'fp') == ('stale', 'libraries or library assignments changed')


def test_enable_libraries_seeds_library_one_and_round_trips():
    state = _state_with_design(4)
    state.csv_combos = [['Nrip1', ['HA', 'V5', 'AU1']]]
    state.combos_csv_path = '/Volumes/T7/Metab library 1.csv'
    state.enable_libraries()
    assert state.libraries[0]['combos'] == [['Nrip1', ['HA', 'V5', 'AU1']]]
    assert state.libraries[0]['csv_path'] == '/Volumes/T7/Metab library 1.csv'
    assert state.libraries[1]['name'] == 'Library 2' and state.libraries[1]['combos'] == []
    state.sample_libraries = {'Raw/a.fcs': 'Library 1', 'Raw/b.fcs': fs.OTHER_LIBRARY, 'Raw/c.fcs': 'Library 9'}
    back = fs.FlowCodeState.from_json(json.loads(json.dumps(state.to_json())))
    assert back.to_json() == state.to_json()
    assert [back.library_of(k) for k in ('Raw/a.fcs', 'Raw/b.fcs', 'Raw/c.fcs', 'Raw/d.fcs')] == \
        ['Library 1', fs.OTHER_LIBRARY, None, None]
    old = fs.FlowCodeState.from_json({'combos': []})
    assert (old.multi_library, old.libraries, old.sample_libraries, old.merge_same_names) == (False, [], {}, True)


def test_library_problems():
    libraries = [fs.new_library([], combos=[['Nrip1', ['HA', 'V5', 'AU1']]])]
    assert fs.library_problems(libraries) == []
    libraries.append({'name': 'library 1', 'label': '', 'pattern': '(', 'combos': []})
    libraries.append({'name': 'Other', 'label': '', 'pattern': '', 'combos': [['g', ['HA', 'V5', 'S']]]})
    problems = fs.library_problems(libraries)
    assert 'Library name "library 1" is used twice.' in problems
    assert 'library 1: load its combinations file.' in problems
    assert any(p.startswith('library 1: the sample-name pattern is not valid') for p in problems)
    assert any(p.startswith('"Other" is reserved') for p in problems)
    assert fs.library_problems([]) == ['Add at least one library.']


def test_library_samples_leave_out_controls_beads_and_backbones():
    samples = {'all_samples': {'Raw/bb.fcs': 'A1 FRET CTRL 1', 'Raw/s1.fcs': 'A1 Spleen_Library1_001',
                               'Raw/u.fcs': 'B1 Unstained', 'Raw/cd4.fcs': 'C1 CD4 BV421 (Cells)',
                               'Raw/beads.fcs': 'D1 UltraComp', 'Raw/fmo.fcs': 'E1 FMO CD8'},
               'bead_samples': ['Raw/beads.fcs']}
    assert fs.library_sample_keys(samples, ['Raw/bb.fcs']) == ['Raw/s1.fcs', 'Raw/fmo.fcs']
