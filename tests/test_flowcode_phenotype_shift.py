"""
test_flowcode_phenotype_shift.py
--------------------------------
The FlowCode phenotype-shift readout (flowcode_phenotype_shift.py) and its
tag spillover correction (flowcode_tag_correction.py) on synthetic
FlowCode-unmixed events (``flowcode_synthetic.shift_sample``): 56
combinations of eight tags, 100 cells each per mouse, T cells in four
states on three markers.

    pytest tests/test_flowcode_phenotype_shift.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flowcode_phenotype_shift as fsh  # noqa: E402
import flowcode_synthetic as syn  # noqa: E402
import flowcode_tag_correction as ftc  # noqa: E402

MICE = 8
CONVERTED, ARTEFACT, KNOCKOUT = 0, 4, 7
M1_M3_STATE = 3
ARTEFACT_K = 150.0
SHUFFLES = 40


def _job(events: dict, correction: bool, time_gate: bool = False) -> dict:
    snapshot = syn.shift_gating(time_gate)
    combos = syn.shift_combos()
    tag_fluor = dict(zip(syn.SHIFT_TAGS, syn.SHIFT_TAG_FLUORS))
    keys = list(events)
    return {
        'samples': keys, 'group_of': {k: 'Spleen' for k in keys}, 'populations': ['T'], 'snapshot': snapshot,
        'transforms': snapshot.transforms,
        'channels': fsh.default_channels(syn.SHIFT_FLUORS, {}, syn.SHIFT_TAG_FLUORS, [syn.SHIFT_VIABILITY], [],
                                         fsh.population_channels(snapshot, ['T'])),
        'settings': {'correction': correction},
        'combo_fluors': {n: tuple(tag_fluor[t] for t in tags) for n, tags in combos},
        'combo_tags': dict(combos), 'thresholds': {f: syn.SHIFT_THRESHOLD for f in syn.SHIFT_TAG_FLUORS},
        'fluors': list(syn.SHIFT_FLUORS), 'marker_fluors': list(syn.SHIFT_MARKER_FLUORS),
        'af_channel': 'AF Abundance', 'negative_model': lambda _k: syn.shift_negative_model(),
        'fit_gate': fsh.viable_gate(snapshot, ['T'], [syn.SHIFT_VIABILITY]),
        'eligibility_candidates': fsh.eligibility_candidates(snapshot, ['T'], 'Viable'), 'eligibility_gate': '',
        'anchor_gate': fsh.anchor_gate(snapshot.gate_names()), 'anchored': [], 'feature_of': None,
        'library_of': None, 'triplet_tags': dict(combos), 'tag_names': list(syn.SHIFT_TAGS), 'controls': []}


def _run(events: dict, correction: bool) -> dict:
    pnn = syn.shift_pnn()
    return fsh.run(_job(events, correction), lambda key: (events[key], pnn))


def _mice(seed0: int, **planted) -> dict:
    return {f'Raw/Spleen/M{i}.fcs': syn.shift_sample(seed0 + i, **planted) for i in range(MICE)}


@pytest.fixture(scope='module')
def planted():
    events = _mice(100, convert={CONVERTED: (0.2, M1_M3_STATE)}, artefact={ARTEFACT: ARTEFACT_K},
                   knockout={KNOCKOUT: 0.9})
    return {'events': events, 'corrected': _run(events, True), 'uncorrected': _run(events, False)}


def _p(result, guide_index: int) -> float:
    res = result['results'].set_index('guide')
    return float(res.loc[f'G{guide_index}', 'p'])


def test_gates_are_found_from_the_hierarchy():
    snapshot = syn.shift_gating()
    assert fsh.viable_gate(snapshot, ['T'], [syn.SHIFT_VIABILITY]) == 'Viable'
    assert fsh.eligibility_candidates(snapshot, ['T'], 'Viable') == ['Viable', 'T']
    assert fsh.anchor_gate(['CD4', 'CD4+CD8-', 'CD8 T cells']) == 'CD8 T cells'
    assert fsh.anchor_gate(['CD4', 'CD8neg']) is None


def test_a_planted_conversion_is_detected_towards_its_phenotype(planted):
    result = planted['corrected']
    assert _p(result, CONVERTED) < 0.01
    row = result['results'].set_index('guide').loc[f'G{CONVERTED}']
    assert row['angle_deg'] > 10 and row['significant']
    medians = result['clusters']['T']['medians']
    gained = medians.loc[row['gain_cluster']]
    assert gained['FITC'] > 0.5 and gained['APC-Cy7'] > 0.5 and gained['PerCP'] < 0.3


def test_a_brightness_dependent_artefact_is_removed_and_a_knockout_kept(planted):
    assert _p(planted['uncorrected'], ARTEFACT) < 0.01
    assert _p(planted['corrected'], ARTEFACT) > 0.05
    assert _p(planted['corrected'], KNOCKOUT) < 0.05
    table = planted['corrected']['correction']['table']
    row = table[(table['combination'] == f'G{ARTEFACT}') & (table['fluor'] == 'PerCP')].iloc[0]
    assert row['status'] == ftc.STATUS_OK and row['t12'] == pytest.approx(ARTEFACT_K, rel=0.25)
    knocked = table[(table['combination'] == f'G{KNOCKOUT}') & (table['fluor'] == 'APC-Cy7')].iloc[0]
    assert knocked['status'] != ftc.STATUS_OK or abs(knocked[list(ftc.TERM_COLUMNS)]).max() < 40
    gates = planted['corrected']['correction']['gates']
    assert gates['fit'] == 'Viable' and gates['eligibility'] == 'T'
    eligibility = planted['corrected']['correction']['eligibility'].set_index('fluor')
    assert not eligibility.loc['BV605', 'eligible'] and eligibility.loc['PerCP', 'eligible']


def test_a_correction_needs_evidence_over_all_the_fits_of_its_group(planted):
    table = planted['corrected']['correction']['table']
    fitted = table[np.isfinite(table['evidence_q'])]
    assert len(fitted) > 200 and (fitted['evidence_q'] >= fitted['evidence_p']).all()
    applied = fitted['status'] == ftc.STATUS_OK
    assert (fitted.loc[applied, 'evidence_q'] < 0.01).all() and (fitted.loc[~applied, 'evidence_q'] >= 0.01).all()
    assert (fitted.loc[~applied, 'status'] == ftc.STATUS_NO_EVIDENCE).all()
    assert fitted.loc[~applied, list(ftc.TERM_COLUMNS)].isna().all().all()
    assert ((fitted['evidence_p'] < 0.01) & ~applied).any()


def test_shuffled_labels_stay_calibrated(planted):
    check = fsh.shuffle_check(planted['corrected'], reps=SHUFFLES)
    row = check.iloc[0]
    assert row['tests'] == SHUFFLES * 56
    assert row['pct_p_below_0_05'] <= 5.0
    assert row['shuffles_with_a_hit'] <= 1


def test_correction_does_not_inflate_the_null():
    p = np.concatenate([_run(_mice(1000 * rep), True)['results']['p'].to_numpy() for rep in range(3)])
    assert len(p) == 3 * 56
    assert np.mean(p < 0.05) <= 0.075


def test_a_marker_near_half_positive_gets_no_runaway_correction():
    """FITC (M1) is 40 % positive: a few brightness bins have more
    positives than negatives, which must not tilt the starting line."""
    rng = np.random.default_rng(3)
    n = 600
    amounts = rng.lognormal(np.log(3.0), 0.5, (n, 3))
    positive = rng.random(n) < 0.42
    values = rng.normal(0.0, 100.0, n) + np.where(positive, rng.normal(5000.0, 600.0, n), 0.0)
    start = ftc.fit_slope(values, amounts.sum(axis=1))
    assert abs(start[0]) < 300 and abs(start[1]) < 40
    _c0, six, _p = ftc.fit_six(values, ftc.tag_terms(amounts), start)
    assert np.abs(six).max() < 40


def test_libraries_merge_guides_by_name_and_keep_their_own_tags():
    counts = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    feature_of = {'a': {'HA_V5_S': 'Ctrl', 'HA_V5_C': 'Tbx21'}, 'b': {'HA_V5_S': 'Gata3', 'HA_V5_C': 'Ctrl'}}
    triplet_tags = {'HA_V5_S': ('HA', 'V5', 'S'), 'HA_V5_C': ('HA', 'V5', 'C'), 'HA_S_C': ('HA', 'S', 'C')}
    out = fsh.feature_counts(counts, ['a', 'b'], ['HA_V5_S', 'HA_V5_C', 'HA_S_C'], feature_of,
                             {'a': 'L1', 'b': 'L2'}, triplet_tags, ['HA', 'V5', 'S', 'C'])
    assert out['features'] == ['Ctrl', 'Tbx21', 'Gata3']
    np.testing.assert_array_equal(out['counts'][0, 0], counts[0, 0])
    np.testing.assert_array_equal(out['counts'][1, 0], counts[1, 1])
    assert np.isnan(out['counts'][0, 2]).all() and np.isnan(out['counts'][1, 1]).all()
    np.testing.assert_array_equal(out['tags']['L1'][0], [1, 1, 1, 0])
    np.testing.assert_array_equal(out['tags']['L2'][0], [1, 1, 0, 1])


def test_results_are_saved_and_read_back(planted, tmp_path):
    result = planted['corrected']
    fsh.save_result(tmp_path / 'shift', result, 'digest')
    back, inputs, _when = fsh.load_result(tmp_path / 'shift')
    assert inputs == 'digest'
    assert back['results'].equals(result['results'])
    np.testing.assert_array_equal(back['counts']['T'], result['counts']['T'])
    np.testing.assert_array_equal(back['cells']['T']['labels'], result['cells']['T']['labels'])
    assert back['correction']['table'].shape == result['correction']['table'].shape
    composition = fsh.composition(back, 'T', 'Spleen', f'G{CONVERTED}')
    assert len(composition) == MICE * fsh.DEFAULT_K
    written = fsh.write_tables(tmp_path / 'export', back)
    assert {p.name for p in written} >= {'shift_results.csv', 'shift_directions.csv', 'shift_cluster_counts.csv',
                                        'shift_cluster_medians.csv', 'shift_correction.csv'}


def test_figures_draw(planted):
    import flowcode_shift_plots as fsp

    result = planted['corrected']
    k = result['settings']['k']
    desc = fsp.cluster_descriptions(result['clusters']['T']['medians'])
    shift = fsp.shift_figure(result['results'], 'T', 'Spleen', k, descriptions=desc)
    assert any(g == f'G{CONVERTED}' for _x, _y, g in shift._points)
    fsp.direction_figure(result['direction'], result['results'], 'T', 'Spleen', k, descriptions=desc)
    fsp.heatmap_figure(result['direction'], result['results'], result['clusters']['T']['medians'], 'T', 'Spleen', k)
    comp = fsh.composition(result, 'T', 'Spleen', f'G{CONVERTED}')
    fsp.composition_figure(comp, f'G{CONVERTED}', 'T', 'Spleen', k, descriptions=desc)
    summary = fsp.summary_heatmap_figure(result['results'], ['T'], ['Spleen'])
    assert f'G{CONVERTED}' in [t.get_text() for t in summary.axes[0].get_xticklabels()]
    assert [t.get_text() for t in summary.axes[0].get_yticklabels()] == ['Spleen']


def test_kept_events_are_gated_as_in_the_whole_sample():
    """Kept events fall in the same gates as in the whole sample, a per-sample
    Time gate included, because the latest event is kept so that the Time
    axis spans the same range (without it the range, and so the Time bins,
    would differ)."""
    snapshot, pnn = syn.shift_gating(time_gate=True), syn.shift_pnn()
    events = syn.shift_sample(7, untransduced=40000)
    kept = fsh.keep_events(events, pnn, 'M7', snapshot, 'Viable', None)
    rows = kept['rows']
    assert int(np.argmax(events[:, pnn.index('Time')])) in rows
    full = snapshot.memberships('M7', events, pnn, ['T'])
    part = snapshot.memberships('M7', kept['events'].astype(float), pnn, ['T'])
    for gate in ('Stable', 'Viable', 'T'):
        np.testing.assert_array_equal(np.asarray(part[gate]), np.asarray(full[gate])[rows])
    whole = snapshot._sample_transforms(events, pnn)['Time'].limits
    assert snapshot._sample_transforms(kept['events'], pnn)['Time'].limits == whole
    analysed = kept['events'][kept['analysed']]
    assert snapshot._sample_transforms(analysed, pnn)['Time'].limits != whole


def _kept_run(events: dict, folder, keep):
    """A run through the kept-event cache in *folder*; returns (result,
    samples read in full)."""
    pnn = syn.shift_pnn()
    reads = []

    def load(key):
        reads.append(key)
        return events[key], pnn
    cache = {'dir': folder, 'source_keys': {k: k for k in events}, 'keep': keep}
    return fsh.run(_job(events, True, time_gate=True), load, cache=cache), reads


@pytest.fixture(scope='module')
def timed():
    events = _mice(300, convert={CONVERTED: (0.2, M1_M3_STATE)}, artefact={ARTEFACT: ARTEFACT_K})
    pnn = syn.shift_pnn()
    return {'events': events, 'full': fsh.run(_job(events, True, time_gate=True), lambda key: (events[key], pnn))}


def _same_clusters(a: dict, b: dict):
    np.testing.assert_array_equal(a['counts']['T'], b['counts']['T'])
    for key in ('ids', 'labels', 'offsets'):
        np.testing.assert_array_equal(a['cells']['T'][key], b['cells']['T'][key])


def test_kept_events_give_the_results_of_full_reads(timed, tmp_path):
    full = timed['full']
    kept, reads = _kept_run(timed['events'], tmp_path, None)
    assert sorted(reads) == sorted(timed['events'])
    assert kept['kept']['gate'] == 'Viable'
    pd.testing.assert_frame_equal(kept['correction']['table'], full['correction']['table'])
    pd.testing.assert_frame_equal(kept['correction']['eligibility'], full['correction']['eligibility'])
    _same_clusters(kept, full)
    pd.testing.assert_frame_equal(kept['results'], full['results'])
    again, reads = _kept_run(timed['events'], tmp_path, None)
    assert reads == []
    _same_clusters(again, full)


def test_the_untransduced_cap_changes_only_the_eligibility_shares(timed, tmp_path):
    full = timed['full']
    capped, _reads = _kept_run(timed['events'], tmp_path, 500)
    ids = syn.shift_pnn().index('FlowCode ID')
    assert all(capped['kept']['kept'][s] <= (ev[:, ids] >= 1).sum() + 500 for s, ev in timed['events'].items())
    a = capped['correction']['eligibility'].set_index('fluor')
    b = full['correction']['eligibility'].set_index('fluor')
    assert (a['eligible'] == b['eligible']).all()
    assert np.abs(a['negative_share'] - b['negative_share']).max() < 0.05
    pd.testing.assert_frame_equal(capped['correction']['table'], full['correction']['table'])
    _same_clusters(capped, full)
    assert len(list(tmp_path.glob('*.npz'))) == MICE
