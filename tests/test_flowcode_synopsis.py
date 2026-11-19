"""
test_flowcode_synopsis.py
-------------------------
The Synopsis of the FlowCode Analysis page (flowcode_synopsis.py and its
figures, flowcode_synopsis_plots.py) on a small statistics run (three
groups of six samples, eight guides, one planted gain and one guide with no
cell in a group) with hand-made marker, phenotype-shift and immune-state
tables.

    pytest tests/test_flowcode_synopsis.py
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

import flowcode_stats as fst  # noqa: E402
import flowcode_synopsis as fsy  # noqa: E402
import flowcode_synopsis_plots as fsyp  # noqa: E402

GROUPS = ('A', 'B', 'C')
GUIDES = [f'G{i}' for i in range(8)]
SAMPLES = [f'Raw/{g}/M{i}.fcs' for g in GROUPS for i in range(6)]
GROUP_OF = {s: s.split('/')[1] for s in SAMPLES}
GAINED, ABSENT, SHIFTED, QUIET = 'G0', 'G1', 'G2', 'G7'
SCOPE = {'label': 'T', 'level': None, 'population': 'T'}


def _statistics(contrast_mode: str = 'reference') -> dict:
    rng = np.random.default_rng(7)
    counts = pd.DataFrame(rng.poisson(200, (len(SAMPLES), len(GUIDES))).astype(float), index=SAMPLES,
                          columns=GUIDES)
    in_b = [s for s in SAMPLES if GROUP_OF[s] == 'B']
    in_c = [s for s in SAMPLES if GROUP_OF[s] == 'C']
    counts.loc[in_b, GAINED] *= 6
    counts.loc[in_c, ABSENT] = 0
    stats = fst.run_statistics(counts, counts.sum(axis=1), GROUP_OF, group_order=list(GROUPS),
                               contrast_mode=contrast_mode, reference='A', run_counts=True, run_freq=True)
    rows = [('B', SHIFTED, 'M1', -0.2, True), ('B', SHIFTED, 'M2', 0.1, True), ('B', GAINED, 'M1', 0.01, False),
            ('C', SHIFTED, 'M1', -0.05, False)]
    stats['markers'] = {
        'sets': {g: {'group': g, 'level': None, 'population': ''} for g in ('B', 'C')},
        'results': pd.DataFrame({'comparison': [r[0] for r in rows], 'combination': [r[1] for r in rows],
                                 'channel': [r[2] for r in rows], 'logFC': [r[3] for r in rows],
                                 'significant': [r[4] for r in rows], 'stagewise.adj.P.Val': 0.01,
                                 'spillover_dominant': False})}
    return stats


def _shift() -> dict:
    rows = [('B', GAINED, 30.0, True), ('B', SHIFTED, 45.0, True), ('A', SHIFTED, 12.0, False),
            ('B', QUIET, 5.0, False)]
    results = pd.DataFrame({'population': 'T', 'group': [r[0] for r in rows], 'guide': [r[1] for r in rows],
                            'angle_deg': [r[2] for r in rows], 'p': 0.01, 'q': 0.02,
                            'significant': [r[3] for r in rows], 'gain_cluster': 'C1', 'loss_cluster': 'C2',
                            'mice': 6, 'median_cells': 150.0})
    states = pd.DataFrame({'population': 'T', 'group': ['B', 'B', 'A'], 'guide': [SHIFTED, SHIFTED, SHIFTED],
                           'state': ['Effector', 'Resting', 'Effector'], 'score': [2.0, -1.5, np.nan],
                           'p': 0.001, 'q': 0.01, 'significant': [True, True, False],
                           'depends_on_target': False, 'members': 4})
    return {'populations': ['T'], 'results': results, 'fsea': {'results': states}}


@pytest.fixture(scope='module')
def stats():
    return _statistics()


@pytest.fixture(scope='module')
def synopsis(stats):
    return fsy.assemble(stats, _shift(), SCOPE, group_order=list(GROUPS), marker_labels={'M1': 'CD25'})


def _call(synopsis, guide, group) -> pd.Series:
    calls = synopsis['calls']
    return calls[(calls['guide'] == guide) & (calls['group'] == group)].iloc[0]


def test_settings_defaults_and_bad_values():
    assert fsy.synopsis_settings(None) == fsy.SYNOPSIS_DEFAULTS
    assert fsy.synopsis_settings({'test': 'wald', 'order': 'size', 'baseline': None, 'other': 1}) \
        == fsy.SYNOPSIS_DEFAULTS


def test_scopes_tests_and_baselines(stats):
    scopes, notes = fsy.scopes(stats, _shift(), 'Gate')
    assert scopes == [SCOPE] and not notes
    assert fsy.scopes(None, _shift())[0] == [SCOPE]
    assert fsy.scopes(None, None) == ([], [])
    assert fsy.available_tests(stats) == ['counts', 'freq']
    assert fsy.resolve_test(stats) == 'counts' and fsy.resolve_test(stats, 'freq') == 'freq'
    assert fsy.baselines(stats, None) == ['A'] and fsy.resolve_baseline(stats, None, 'B') == 'A'
    pairwise = _statistics('pairwise')
    assert sorted(fsy.baselines(pairwise, None)) == ['A', 'B', 'C']
    assert fsy.resolve_baseline(pairwise, None, 'B') == 'B'


def test_accumulation_is_read_against_the_baseline(stats, synopsis):
    assert synopsis['baseline'] == 'A' and synopsis['test'] == 'counts'
    assert synopsis['groups'] == ['A', 'B', 'C']
    gained = _call(synopsis, GAINED, 'B')
    table = stats['counts']
    direct = table[(table['feature'] == GAINED) & (table['comparison'] == 'B vs A')].iloc[0]
    assert gained['logFC'] == pytest.approx(direct['logFC']) and gained['logFC'] > 2
    assert gained['accumulation'] == 1 and gained['tests_agree'] and not gained['unbounded']
    assert not _call(synopsis, QUIET, 'B')['accumulation']
    assert synopsis['samples'] == {'A': 6, 'B': 6, 'C': 6}


def test_another_baseline_reverses_the_sign_and_the_interval():
    pairwise = _statistics('pairwise')
    from_a = fsy.assemble(pairwise, None, SCOPE, baseline='A')['accumulation']
    from_b = fsy.assemble(pairwise, None, SCOPE, baseline='B')['accumulation']
    a = from_a[(from_a['guide'] == GAINED) & (from_a['group'] == 'B') & (from_a['test'] == 'counts')].iloc[0]
    b = from_b[(from_b['guide'] == GAINED) & (from_b['group'] == 'A') & (from_b['test'] == 'counts')].iloc[0]
    assert b['logFC'] == pytest.approx(-a['logFC'])
    assert (b['CI.L'], b['CI.R']) == pytest.approx((-a['CI.R'], -a['CI.L']))
    assert b['significant'] == a['significant'] and b['q'] == pytest.approx(a['q'])


def test_a_guide_with_no_cell_in_a_group_has_no_fold_change(synopsis):
    absent = _call(synopsis, ABSENT, 'C')
    assert absent['cells'] == 0 and absent['unbounded'] and absent['accumulation'] == -1
    assert not _call(synopsis, ABSENT, 'B')['unbounded']
    sentence = fsy.guide_sentence(ABSENT, synopsis)
    assert 'not detected in C' in sentence and 'fewer in' not in sentence


def test_phenotype_calls_count_the_readouts(synopsis):
    shifted = _call(synopsis, SHIFTED, 'B')
    assert shifted['shift'] and shifted['states_towards'] == 1 and shifted['states_away'] == 1
    assert shifted['markers_lower'] == 1 and shifted['markers_higher'] == 1 and shifted['phenotype_readouts'] == 3
    assert _call(synopsis, GAINED, 'B')['phenotype_readouts'] == 1
    baseline = _call(synopsis, SHIFTED, 'A')
    assert baseline['phenotype_tested'] and not baseline['accumulation_tested']
    assert baseline['phenotype_readouts'] == 0 and baseline['states_tested'] == 0
    assert not _call(synopsis, SHIFTED, 'C')['shift_tested'] and _call(synopsis, SHIFTED, 'C')['phenotype_tested']
    assert not _call(synopsis, QUIET, 'C')['phenotype_tested']


def test_classes_and_sentences(synopsis):
    table = synopsis['guide_table'].set_index('guide')
    assert table.loc[GAINED, 'class'] == fsy.BOTH and table.loc[ABSENT, 'class'] == fsy.ACCUMULATION
    assert table.loc[SHIFTED, 'class'] == fsy.PHENOTYPE and table.loc[QUIET, 'class'] == fsy.NONE
    assert table.loc[SHIFTED, 'groups_two_phenotype_readouts'] == 1 and table.loc[GAINED, 'groups_more'] == 1
    gained = table.loc[GAINED, 'synopsis']
    assert gained.startswith(f'{fsy.BOTH}. Cells against A (counts, log2 fold change): more in B (+')
    assert 'Counts and frequencies agree in 1 of 1 group.' in gained and 'Phenotype shift in B (30°).' in gained
    shifted = table.loc[SHIFTED, 'synopsis']
    assert 'Immune states: towards Effector (B); away from Resting (B).' in shifted
    assert 'Markers: 2 changes in 1 group; largest CD25 lower in B (-0.20).' in shifted
    assert table.loc[QUIET, 'synopsis'].startswith(f'{fsy.NONE}.')
    assert fsy.guide_sentence(GAINED, synopsis) == gained


def test_guide_order(stats):
    groups = {'assignment': pd.DataFrame({'feature': GUIDES, 'group': ['late'] * 4 + ['early'] * 4}),
              'group_order': ['early', 'late']}
    by_group = fsy.assemble(stats, None, SCOPE, gene_groups=groups)
    assert set(by_group['guides'][:4]) == set(GUIDES[4:]) and by_group['gene_group'][GAINED] == 'late'
    assert by_group['guides'][-1] == GAINED
    assert fsy.assemble(stats, None, SCOPE, order='alphabetical', gene_groups=groups)['guides'] == GUIDES


def test_comparisons_between_populations_are_not_read(stats):
    between = {**stats, 'strata': {s: 'T' for s in stats['samples']}}
    scopes, notes = fsy.scopes(between, _shift())
    assert scopes == [SCOPE] and 'no baseline group' in notes[0]
    synopsis = fsy.assemble(between, _shift(), SCOPE)
    assert synopsis['baseline'] is None and synopsis['accumulation'].empty
    assert not synopsis['calls']['accumulation_tested'].any() and synopsis['calls']['shift'].sum() == 2


def test_overview_and_tables(synopsis, tmp_path):
    text = fsy.overview_text(synopsis)
    assert text.startswith('T: 8 guides, 3 groups.') and 'against A (counts test)' in text
    assert f'{fsy.BOTH} (1): {GAINED}.' in text and '  A (baseline;' in text
    assert '1 guide × group cell had no cell on one side' in text
    written = fsy.write_tables(tmp_path, [synopsis])
    assert [p.name for p in written] == ['synopsis_guides.csv', 'synopsis_calls.csv', 'synopsis.txt']
    guides = pd.read_csv(tmp_path / 'synopsis_guides.csv')
    assert list(guides.columns[:4]) == ['population', 'baseline', 'accumulation_test', 'guide']
    assert len(guides) == len(GUIDES) and set(guides['accumulation_test']) == {'counts'}
    assert f'{GAINED}: {fsy.BOTH}.' in (tmp_path / 'synopsis.txt').read_text(encoding='utf-8')
    assert fsy.write_tables(tmp_path, [fsy.assemble(None, None, SCOPE)]) == []


def test_figures(synopsis):
    atlas = fsyp.atlas_figure(synopsis, {}, n_rows=2, show_all=True)
    assert len(atlas._atlas_points) == 2
    assert {key for _ax, points in atlas._atlas_points for _x, _y, key in points} == set(GUIDES)
    dossier = fsyp.dossier_figure(synopsis, SHIFTED, is_dark=True)
    assert len(dossier.axes) >= 5
    empty = fsy.assemble(None, None, SCOPE)
    assert not hasattr(fsyp.atlas_figure(empty), '_atlas_points')
    assert len(fsyp.dossier_figure(synopsis, 'missing').axes) == 1
