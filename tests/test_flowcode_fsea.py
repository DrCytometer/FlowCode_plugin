"""
test_flowcode_fsea.py
---------------------
Flow Set Enrichment Analysis per guide (flowcode_fsea.py) inside the
phenotype-shift run, on synthetic FlowCode-unmixed events
(``flowcode_synthetic.shift_sample``: 56 combinations of eight tags, 100
cells each per mouse, T cells in four states on three markers) and the
synthetic immune states of ``flowcode_synthetic.fsea_spec``; and the
matching of the bundled library to a Treg panel.

    pytest tests/test_flowcode_fsea.py
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

import flowcode_fsea as ffsea  # noqa: E402
import flowcode_phenotype_shift as fsh  # noqa: E402
import flowcode_synthetic as syn  # noqa: E402

MICE = 8
CONVERTED, ARTEFACT, KNOCKOUT = 0, 4, 7
M1_M3_STATE = 3
ARTEFACT_K = 40.0
SHUFFLES = 20


def _job(events: dict, correction: bool, fsea: dict | None = None) -> dict:
    snapshot = syn.shift_gating()
    combos = syn.shift_combos()
    tag_fluor = dict(zip(syn.SHIFT_TAGS, syn.SHIFT_TAG_FLUORS))
    keys = list(events)
    return {
        'samples': keys, 'group_of': {k: 'Spleen' for k in keys}, 'populations': ['T'], 'snapshot': snapshot,
        'transforms': snapshot.transforms,
        'channels': fsh.default_channels(syn.SHIFT_FLUORS, {}, syn.SHIFT_TAG_FLUORS, [syn.SHIFT_VIABILITY], [],
                                         fsh.population_channels(snapshot, ['T'])),
        'settings': {'correction': correction, 'fsea': fsea},
        'combo_fluors': {n: tuple(tag_fluor[t] for t in tags) for n, tags in combos},
        'combo_tags': dict(combos), 'thresholds': {f: syn.SHIFT_THRESHOLD for f in syn.SHIFT_TAG_FLUORS},
        'fluors': list(syn.SHIFT_FLUORS), 'marker_fluors': list(syn.SHIFT_MARKER_FLUORS),
        'af_channel': 'AF Abundance', 'negative_model': lambda _k: syn.shift_negative_model(),
        'fit_gate': fsh.viable_gate(snapshot, ['T'], [syn.SHIFT_VIABILITY]),
        'eligibility_candidates': fsh.eligibility_candidates(snapshot, ['T'], 'Viable'), 'eligibility_gate': '',
        'anchor_gate': fsh.anchor_gate(snapshot.gate_names()), 'anchored': [], 'feature_of': None,
        'library_of': None, 'triplet_tags': dict(combos), 'tag_names': list(syn.SHIFT_TAGS), 'controls': [],
        'fsea': syn.fsea_spec(f'G{KNOCKOUT}')}


def _run(events: dict, correction: bool, fsea: dict | None = None) -> dict:
    pnn = syn.shift_pnn()
    return fsh.run(_job(events, correction, fsea), lambda key: (events[key], pnn))


def _mice(seed0: int, **planted) -> dict:
    return {f'Raw/Spleen/M{i}.fcs': syn.shift_sample(seed0 + i, **planted) for i in range(MICE)}


@pytest.fixture(scope='module')
def planted():
    events = _mice(100, convert={CONVERTED: (0.2, M1_M3_STATE)}, artefact={ARTEFACT: ARTEFACT_K},
                   knockout={KNOCKOUT: 0.9})
    return {'events': events, 'corrected': _run(events, True), 'uncorrected': _run(events, False)}


def _row(result, guide: int, state: str) -> pd.Series:
    res = result['fsea']['results']
    return res[(res['guide'] == f'G{guide}') & (res['state'] == state)].iloc[0]


def test_settings_defaults_and_bad_values():
    s = ffsea.fsea_settings({'min_members': 1, 'rotations': 10, 'species': 'fish', 'states': 'all'})
    assert s == ffsea.FSEA_DEFAULTS
    assert fsh.shift_settings(None)['fsea'] == ffsea.FSEA_DEFAULTS
    assert ffsea.scoring_key({'min_members': 4}) == {'enabled': True, 'species': 'mouse', 'roots': {}}


def test_a_planted_state_shift_is_detected(planted):
    result = planted['corrected']
    row = _row(result, CONVERTED, syn.FSEA_PLANTED)
    assert row['score'] > 0 and row['p'] < 0.01 and row['significant']
    res = result['fsea']['results']
    others = res[(res['state'] == syn.FSEA_PLANTED) & ~res['guide'].isin([f'G{CONVERTED}', f'G{KNOCKOUT}'])]
    assert others['significant'].mean() <= 0.05
    members = result['fsea']['members']
    m = members[(members['guide'] == f'G{CONVERTED}')].set_index('feature')
    assert m.loc['abund:M1+ M3+ T', 'mean'] > 0.3 and m.loc['marker:T|M2', 'mean'] < 0


def test_a_planted_tag_artefact_is_removed_by_the_correction(planted):
    assert _row(planted['uncorrected'], ARTEFACT, syn.FSEA_ARTEFACT)['p'] < 0.01
    assert _row(planted['corrected'], ARTEFACT, syn.FSEA_ARTEFACT)['p'] > 0.05


def test_the_target_rows_are_left_out_of_the_main_result(planted):
    res = planted['corrected']['fsea']['results'].set_index(['guide', 'state'])
    target = res.loc[(f'G{KNOCKOUT}', syn.FSEA_TARGET)]
    assert target['target_rows'] == 1 and target['members_all'] == 3 and target['members'] == 2
    assert target['p_all'] < 0.01 and target['significant_all']
    assert np.isnan(target['p']) and not target['significant'] and target['depends_on_target']
    shared = res.loc[(f'G{KNOCKOUT}', syn.FSEA_PLANTED)]
    assert shared['members'] == shared['members_all'] - 1
    other = res.loc[(f'G{CONVERTED}', syn.FSEA_PLANTED)]
    assert other['target_rows'] == 0 and other['p'] == other['p_all']


def test_the_minimum_count_keeps_a_feature_whose_positive_cells_a_guide_removes(planted):
    """The minimum count is taken on the expected counts (from the
    reference): a guide that removes nearly all of a rare positive
    population is still tested on it, and the loss is found."""
    members = planted['corrected']['fsea']['members']
    knocked = members[members['guide'] == f'G{KNOCKOUT}'].set_index('feature')
    assert {'marker:T|M3', 'abund:M1+ M3+ T', 'marker:T|M1'} <= set(knocked.index)
    assert knocked.loc['marker:T|M3', 't'] < -3


def test_thin_abundance_rows_and_minimums(planted):
    fsea = planted['corrected']['fsea']
    assert any('M2+ T' in n for n in fsea['notes'])
    assert 'abund:M2+ T' not in set(fsea['members']['feature'])
    loose = ffsea.reanalyse(fsea, planted['corrected']['design'], {**fsea['settings'], 'min_abund_signs': 1},
                            planted['corrected']['settings'])
    assert 'abund:M2+ T' in set(loose['members']['feature'])
    strict = ffsea.reanalyse(fsea, planted['corrected']['design'], {**fsea['settings'], 'min_count': 1e6},
                             planted['corrected']['settings'])
    assert strict['results'].empty and strict['notes']
    small = fsea['results']
    assert small['small_set'].all()


def test_shuffled_labels_stay_calibrated(planted):
    result = planted['corrected']
    check = ffsea.shuffle_check(result['fsea'], result['design'], result['settings'], reps=SHUFFLES)
    row = check.iloc[0]
    assert row['tests'] > SHUFFLES * 56 * 2
    assert row['pct_p_below_0_05'] <= 5.5
    assert row['shuffles_with_a_hit'] <= 2


def test_immune_states_off_skip_the_scores():
    events = _mice(500)
    result = _run(events, False, {'enabled': False})
    assert result['fsea'] is None


def test_other_references_detect_the_planted_shift(planted):
    result = planted['corrected']
    shift = {**result['settings'], 'reference': 'all'}
    out = ffsea.reanalyse(result['fsea'], result['design'], result['fsea']['settings'], shift)
    res = out['results']
    row = res[(res['guide'] == f'G{CONVERTED}') & (res['state'] == syn.FSEA_PLANTED)].iloc[0]
    assert row['score'] > 0 and row['significant']


def test_member_values_and_tables(planted, tmp_path):
    result = planted['corrected']
    values = ffsea.member_values(result['fsea'], result['design'], result['settings'], 'T', 'Spleen',
                                 f'G{CONVERTED}', 'abund:M1+ M3+ T')
    assert len(values) == MICE
    assert (values['guide_share'] > values['reference_share']).mean() >= 0.75
    with pytest.raises(KeyError):
        ffsea.member_values(result['fsea'], result['design'], result['settings'], 'T', 'Spleen', 'nope',
                            'abund:M1+ M3+ T')
    table = ffsea.states_table(result['fsea'], 'T', 'Spleen').set_index('state')
    assert table.loc[syn.FSEA_PLANTED, 'calls_up'] >= 1 and table.loc[syn.FSEA_PLANTED, 'not_scored'] == 1
    written = {p.name for p in fsh.write_tables(tmp_path, result)}
    assert {'immune_state_results.csv', 'immune_state_features.csv', 'immune_state_rows.csv'} <= written


def test_results_are_saved_and_read_back(planted, tmp_path):
    result = planted['corrected']
    fsh.save_result(tmp_path / 'shift', result, 'abc')
    back, inputs, _when = fsh.load_result(tmp_path / 'shift')
    assert inputs == 'abc'
    a, b = result['fsea'], back['fsea']
    pd.testing.assert_frame_equal(a['results'].reset_index(drop=True), b['results'].reset_index(drop=True),
                                  check_dtype=False)
    again = ffsea.reanalyse(b, back['design'], b['settings'], back['settings'])
    np.testing.assert_allclose(again['results']['p'].to_numpy(float), a['results']['p'].to_numpy(float))


def _treg_panel():
    import flowcode_phenotype as fp
    antigens = ['CD3', 'CD4', 'CD8', 'FoxP3', 'Helios', 'GATA-3', 'T-bet', 'RORgt', 'c-MAF', 'IRF4', 'Blimp-1',
                'Nur77', 'ICOS', 'PD-1', 'KLRG1', 'CD103', 'CD69', 'CD44', 'CD62L', 'CD25', 'CD19']
    model = [{'label': f'F{i}', 'antigen': a} for i, a in enumerate(antigens)]
    model += [{'label': 'BV421', 'antigen': 'HA'}, {'label': 'BV510', 'antigen': 'V5'}]
    return fp.panel_from_spectral_model(model)


def test_the_library_is_matched_to_a_treg_panel():
    assert ffsea.default_root('Treg') == 'CD4 Treg'
    assert ffsea.default_root('no such cells') is None
    spec = ffsea.panel_library(_treg_panel(), ['BV421', 'BV510'], {'Treg': 'CD4 Treg', 'Other': None})
    pop = spec['populations']['Treg']
    rows = pop['rows']
    states = set(rows['state'])
    assert {'Effector Treg expansion', 'Tissue Treg programme'} <= states
    effector = rows[rows['state'] == 'Effector Treg expansion']
    assert {'CD44', 'CD62L', 'ICOS', 'CD103', 'KLRG1'} <= set(effector['marker'])
    assert effector.set_index('marker').loc['KLRG1', 'gene'] == 'Klrg1'
    assert effector.set_index('marker').loc['CD62L', 'sign'] == -1
    left = pop['left_out']
    assert ((left['state'] == 'Effector Treg expansion') & left['reason'].str.contains('not stained')).any()
    assert not rows['marker'].isin(['HA', 'V5']).any()
    assert set(spec['markers']) >= {'KLRG1', 'CD44'} and spec['fluor_of']['KLRG1'].startswith('F')
    assert spec['populations']['Other']['rows'].empty and spec['notes']
