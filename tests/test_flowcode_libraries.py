"""
test_flowcode_libraries.py
--------------------------
FlowCode Analysis with several libraries (flowcode_analysis.py,
flowcode_stats.py, differential_plots.make_heatmap_figure). No Qt.

Synthetic counts: Spleen and Gut of six mice per library on six tags.
Library 1 puts guides A0-A9 on ten triplets and NON-Targetting 1 on an
eleventh; Library 2 puts guides B0-B7 on eight of the same triplets and
NON-Targetting 1 on another triplet (the same name on a different triplet,
as in Metab library 4). A1 is enriched 4-fold in the gut, B2 depleted
4-fold. Library 1 samples also have events on a triplet neither library
uses (out of library).

    pytest tests/test_flowcode_libraries.py
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))

import flowcode_analysis as fa  # noqa: E402
import flowcode_core as fcc  # noqa: E402
import flowcode_state as fcs  # noqa: E402
import flowcode_stats as fst  # noqa: E402

TAGS = ('HA', 'V5', 'AU1', 'S', 'C', 'FLAG')
TRIPLETS = [list(t) for t in itertools.combinations(TAGS, 3)]
KEYS = [fcc.combo_key(t) for t in TRIPLETS]
NT = 'NON-Targetting 1'
L1 = [[f'A{i}', TRIPLETS[i]] for i in range(10)] + [[NT, TRIPLETS[10]]]
L2 = [[f'B{i}', TRIPLETS[i]] for i in range(8)] + [[NT, TRIPLETS[11]]]
UNUSED = KEYS[15]
MICE = {'Library 1': [f'M{i}' for i in range(1, 7)], 'Library 2': [f'M{i}' for i in range(7, 13)]}
TISSUES = ('Spleen', 'Gut')


def make_state(merge=True, label2=''):
    lib1 = fcs.new_library([], 'lib1.csv', L1)
    lib2 = fcs.new_library([lib1], 'lib2.csv', L2)
    lib2['label'] = label2
    state = fcs.FlowCodeState(multi_library=True, libraries=[lib1, lib2], merge_same_names=merge)
    state.combos = fcs.library_combo_rows(state.libraries, {})
    return state


def unit(tissue, lib, mouse):
    return f'Raw/{tissue}_{lib.replace(" ", "")}_{mouse}.fcs'


def triplet_records(seed=0, size=4000):
    """{sample: count record} per triplet, and {sample: library}."""
    rng = np.random.default_rng(seed)
    records, library = {}, {}
    for lib, guides in (('Library 1', L1), ('Library 2', L2)):
        for mouse in MICE[lib]:
            base = rng.lognormal(0.0, 0.3, len(guides))
            for tissue in TISSUES:
                w = base.copy()
                if tissue == 'Gut' and lib == 'Library 1':
                    w[1] *= 4.0
                if tissue == 'Gut' and lib == 'Library 2':
                    w[2] /= 4.0
                n = rng.multinomial(size, w / w.sum())
                counts = {fcc.combo_key(t): int(c) for (_g, t), c in zip(guides, n)}
                if lib == 'Library 1':
                    counts[UNUSED] = 37
                key = unit(tissue, lib, mouse)
                barcoded = sum(counts.values())
                records[key] = {'counts': counts, 'events': barcoded + 500, 'in_population': barcoded + 200,
                                'barcoded': barcoded, 'untransduced': 150, 'unresolved': 50, 'not_barcoded': 200}
                library[key] = lib
    return records, library


def feature_counts(state=None, seed=0, extra=None):
    """units_table + library_features of the synthetic records."""
    state = state or make_state()
    records, library = triplet_records(seed)
    records.update(extra or {})
    counts, meta = fa.units_table({fa.ALL_EVENTS: records}, [fa.ALL_EVENTS], combo_order=KEYS)
    return fa.library_features(counts, meta, library, fa.library_feature_map(state)), library, state


def design(units):
    groups = {u: Path(u).stem.split('_')[0] for u in units}
    lib_of = {u: f"Library {Path(u).stem.split('_')[1][-1]}" for u in units}
    mouse = {u: f"{lib_of[u]}·{Path(u).stem.split('_')[2]}" for u in units}
    return groups, mouse, lib_of


def feature_libraries(state):
    out = {}
    for lib, fmap in fa.library_feature_map(state).items():
        for f in fmap.values():
            out.setdefault(f, []).append(lib)
    return out


def run(counts, denom, libraries=True, **kwargs):
    groups, mouse, lib_of = design(list(counts.index))
    state = make_state()
    extra = dict(libraries=lib_of, feature_libraries=feature_libraries(state)) if libraries else {}
    kwargs.setdefault('pairing', mouse)
    return fst.run_statistics(counts, denom, groups, group_order=list(TISSUES), reference='Spleen',
                              **extra, **kwargs)


# ---------------------------------------------------------------------------
# Naming the counts
# ---------------------------------------------------------------------------

def test_each_library_names_its_own_triplets():
    fmap = fa.library_feature_map(make_state())
    assert fmap['Library 1'][KEYS[1]] == 'A1' and fmap['Library 2'][KEYS[1]] == 'B1'
    assert fmap['Library 1'][KEYS[10]] == NT and fmap['Library 2'][KEYS[11]] == NT
    assert KEYS[9] not in fmap['Library 2'] and UNUSED not in fmap['Library 1']
    apart = fa.library_feature_map(make_state(merge=False, label2='Metab B'))
    assert apart['Library 1'][KEYS[10]] == f'{NT} (Library 1)' and apart['Library 2'][KEYS[11]] == f'{NT} (Metab B)'
    assert apart['Library 2'][KEYS[0]] == 'B0'


def test_counts_are_nan_outside_the_library_and_out_of_library_events_are_kept_apart():
    other = {'Raw/FMO.fcs': {'counts': {KEYS[0]: 5}, 'events': 10, 'in_population': 9, 'barcoded': 5,
                             'untransduced': 1, 'unresolved': 1, 'not_barcoded': 4}}
    (counts, meta, left_out), library, _state = feature_counts(extra=other)
    assert left_out == {'Raw/FMO.fcs': ''}
    assert list(counts.columns) == [g for g, _t in L1] + [g for g, _t in L2 if g != NT]
    u1, u2 = unit('Gut', 'Library 1', 'M1'), unit('Gut', 'Library 2', 'M7')
    assert counts.loc[u1, [f'B{i}' for i in range(8)]].isna().all()
    assert counts.loc[u2, [f'A{i}' for i in range(10)]].isna().all() and counts.loc[u2].notna().sum() == 9
    records, _lib = triplet_records()
    assert counts.loc[u2, NT] == records[u2]['counts'][KEYS[11]]
    assert meta.loc[u1, 'out_of_library'] == 37 and meta.loc[u2, 'out_of_library'] == 0
    assert meta.loc[u1, 'barcoded'] == records[u1]['barcoded'] - 37 == counts.loc[u1].sum()
    assert meta.loc[u1, 'library'] == 'Library 1'
    qc = fa.qc_table(counts, meta)
    assert qc.loc[u1, 'out_of_library'] == 37 and np.isfinite(qc['gini']).all()
    assert qc.loc[u2, 'barcoded'] == int(counts.loc[u2].sum())
    filtered = fa.filter_counts(counts, min_combo_events=0, min_sample_events=0)
    assert filtered.denominator[u1] == counts.loc[u1].sum()
    # a guide's minimum is over the samples that carry it
    rare = counts.copy()
    rare.loc[[u for u in rare.index if library[u] == 'Library 2'], 'B5'] = 1.0
    assert 'B5' in fa.filter_counts(rare, min_combo_events=20).rare_combos


def test_a_file_exported_with_another_library_is_noted(tmp_path):
    from honeychrome.controller_components import functions as fn

    key = 'Raw/Gut_M1.fcs'
    pnn = ['FSC-A', 'FlowCode ID', f'FC {KEYS[0]}']
    kw = {'FLOWCODECOMBOS': 'x', 'FLOWCODESOURCE': key, 'FLOWCODELIBRARY': 'Library 2'}
    for i, name in enumerate(pnn, start=1):
        kw.update({f'$P{i}N': name, f'$P{i}B': '32', f'$P{i}E': '0,0', f'$P{i}R': '262144'})
    fn.write_fcs(np.zeros((3, 3)), kw, tmp_path / 'Gut_M1 FlowCode (Unmixed).fcs')
    links = fa.link_exports(tmp_path, [key], 'Raw', {key: 'Gut_M1'}, ' FlowCode', 'x', None, {key: 'Library 1'})
    info = links['files'][key]
    assert info['status'] == fa.LINK_OK and 'Exported as Library 2; counted as Library 1' in info['note']
    same = fa.link_exports(tmp_path, [key], 'Raw', {key: 'Gut_M1'}, ' FlowCode', 'x', None, {key: 'Library 2'})
    assert same['files'][key]['note'] == ''


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_merged_fit_tests_every_guide_and_matches_the_split_frequencies():
    (counts, _meta, _left), _library, _state = feature_counts()
    filt = fa.filter_counts(counts)
    merged = run(filt.counts, filt.denominator, run_counts=True, run_freq=True)
    assert merged['comparisons'] == ['Gut vs Spleen']
    for key in ('counts', 'freq'):
        res = merged[key].set_index('feature')
        assert np.isfinite(res['P.Value']).all() and len(res) == counts.shape[1]
        assert bool(res.loc['A1', 'significant']) and res.loc['A1', 'logFC'] > 1.0
        assert bool(res.loc['B2', 'significant']) and res.loc['B2', 'logFC'] < -1.0
    assert not merged['removed'].values.any()
    assert merged['libraries'][unit('Gut', 'Library 2', 'M7')] == 'Library 2'
    _groups, _mouse, lib_of = design(list(filt.counts.index))
    # The unweighted fit gives every guide the same estimate merged or split by library; the capped
    # weights do not, since their prior count and shared variance depend on the samples fitted.
    split = run(filt.counts, filt.denominator, split_by=lib_of, run_counts=False, run_freq=True,
                freq_weights=None)
    assert split['comparisons'] == ['Gut vs Spleen [Library 1]', 'Gut vs Spleen [Library 2]']
    unweighted = run(filt.counts, filt.denominator, run_counts=False, run_freq=True, freq_weights=None)
    m = unweighted['freq'].set_index('feature')['logFC']
    for lib, own in (('Library 1', [f'A{i}' for i in range(10)]), ('Library 2', [f'B{i}' for i in range(8)])):
        s = split['freq'][split['freq']['comparison'] == f'Gut vs Spleen [{lib}]'].set_index('feature')['logFC']
        np.testing.assert_allclose(m.loc[own].values, s.loc[own].values, atol=1e-10)
    np.testing.assert_allclose(merged['library_sizes'].values, split['library_sizes'].loc[merged['samples']].values)
    # library sizes within each library: a library's samples alone give the same sizes
    l1 = [u for u in filt.counts.index if lib_of[u] == 'Library 1']
    alone = run(filt.counts.loc[l1], filt.denominator.loc[l1], libraries=False)
    np.testing.assert_allclose(merged['library_sizes'].loc[l1].values, alone['library_sizes'].values, rtol=1e-12)


def test_a_library_with_too_few_samples_in_a_group_is_left_out_there():
    (counts, _meta, _left), _library, _state = feature_counts()
    drop = [unit('Gut', 'Library 2', m) for m in ('M7', 'M8', 'M9', 'M10')]
    counts = counts.drop(index=drop)
    filt = fa.filter_counts(counts)
    stats = run(filt.counts, filt.denominator, run_counts=True, run_freq=True)
    res = stats['counts'].set_index('feature')
    assert res.loc[[f'B{i}' for i in range(8)], 'logFC'].isna().all()
    assert np.isfinite(res.loc[[f'A{i}' for i in range(10)], 'P.Value']).all()
    assert np.isfinite(res.loc[NT, 'P.Value'])                      # 6 + 2 gut samples carry it
    notes = ' '.join(stats['notes'])
    assert 'Library 2: 2 sample(s) in Gut, fewer than 3; its guides are not compared in Gut there.' in notes
    assert 'Gut vs Spleen: no Library 2 guide could be tested.' not in notes      # NON-Targetting 1 was
    assert not stats['removed'].values.any()


def test_clones_and_control_normalisation_are_worked_out_within_each_library():
    (counts, _meta, _left), _library, _state = feature_counts()
    clone = unit('Gut', 'Library 1', 'M2')
    counts.loc[clone, 'A5'] = 3000.0
    filt = fa.filter_counts(counts)
    stats = run(filt.counts, filt.denominator, clone_filter={})
    clones = stats['clones']
    assert set(clones['library']) <= {'Library 1', 'Library 2'}
    hit = clones[(clones['combination'] == 'A5') & (clones['action'] == 'removed')]
    assert list(hit['sample']) == [clone]
    removed = stats['removed']
    assert removed.loc[[clone, unit('Spleen', 'Library 1', 'M2')], 'A5'].all()
    assert int(removed.values.sum()) == int(sum((clones['action'] == 'removed') * 2))
    assert not (removed & filt.counts.loc[removed.index].isna()).values.any()     # never a guide outside it
    with pytest.raises(RuntimeError, match='Library 2: Control normalisation: none of the chosen'):
        run(filt.counts, filt.denominator, normalisation='control', control_combos=['A0'])
    ok = run(filt.counts, filt.denominator, normalisation='control', control_combos=[NT])
    assert np.isfinite(ok['library_sizes']).all()


def test_figures_keep_each_library_apart():
    import matplotlib
    matplotlib.use('Agg')
    (counts, _meta, _left), _library, _state = feature_counts()
    filt = fa.filter_counts(counts)
    stats = run(filt.counts, filt.denominator, run_counts=True, run_freq=True,
                library_labels={'Library 2': 'Metab B'})
    summary = fst.summary_figures(stats, False, 'CD4')
    assert {'sample_pca Library 1', 'sample_pca Library 2'} <= set(summary)
    assert summary['sample_pca Library 2']['title'] == 'Sample PCA (Metab B)'
    assert all(item['fig'] is not None for item in summary.values()), summary
    pca = fst.sample_pca(stats, library='Library 2')
    assert all('Library2' in s for s in pca['scores'].index) and not pca['loadings'].index.str.startswith('A').any()
    pair = stats['pairs'][0]
    figs = fst.comparison_figures(stats, pair, 0.05, 1.0, 'per_comparison', False, 'CD4', combination='A1')
    for key in ('heatmap_all', 'heatmap_significant', 'combination'):
        assert figs[key]['fig'] is not None, figs[key]['error']
    ax = figs['heatmap_all']['fig']._row_axes
    columns = [t.get_text() for t in ax.get_xticklabels()]
    gap = columns.index('')
    assert all('Library1' in c for c in columns[:gap]) and all('Library2' in c for c in columns[gap + 1:])
    assert len(columns) == len(stats['samples']) + 1
    values = fst.combination_values(stats, 'A1', pair)
    assert len(values) == 12 and all('Library1' in s for s in values['sample'])


def test_heatmap_without_blocks_is_unchanged():
    import matplotlib
    matplotlib.use('Agg')
    from honeychrome.view_components import differential_plots as dp

    rng = np.random.default_rng(3)
    samples = [f's{i}' for i in range(6)]
    data = pd.DataFrame(rng.normal(size=(6, 4)), index=samples, columns=list('wxyz'))
    res = pd.DataFrame({'feature': list('wxyz'), 'significant': True})
    groups = dict(zip(samples, ['a', 'a', 'a', 'b', 'b', 'b']))
    plain = dp.make_heatmap_figure(res, data, 'Frequencies', groups, 'a', 'b')
    blocked = dp.make_heatmap_figure(res, data, 'Frequencies', groups, 'a', 'b',
                                     sample_blocks={s: ('L1' if i % 2 else 'L2') for i, s in enumerate(samples)},
                                     block_colours={'L1': '#66c2a5', 'L2': '#fc8d62'})
    plain_cols = [t.get_text() for t in plain._row_axes.get_xticklabels()]
    blocked_cols = [t.get_text() for t in blocked._row_axes.get_xticklabels()]
    assert sorted(plain_cols) == samples and len(plain_cols) == 6
    assert blocked_cols[3] == '' and set(blocked_cols[:3]) == {'s1', 's3', 's5'}
    assert plain.get_size_inches()[0] < blocked.get_size_inches()[0]


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def test_marker_deltas_are_worked_out_within_each_library():
    (counts, _meta, _left), _library, state = feature_counts()
    fmap = fa.library_feature_map(state)
    _groups, _mouse, lib_of = design(list(counts.index))
    rng = np.random.default_rng(5)
    channels = ['M0', 'M1']
    sums = rng.uniform(0.2, 0.6, size=(len(counts), len(KEYS), len(channels))) * 50.0
    data = {'units': list(counts.index), 'combos': KEYS, 'channels': channels, 'sums': sums}
    tag_matrix = pd.DataFrame([[1.0 if t in trip else 0.0 for t in TAGS] for trip in TRIPLETS], index=KEYS,
                              columns=list(TAGS))
    deltas, meta, used, means, refs, spill = fst.marker_deltas_by_library(
        data, counts, channels, lib_of, fmap, 'all', [NT], 10, tag_matrix=tag_matrix)
    assert used == 'all' and spill is None
    l1 = [u for u in counts.index if lib_of[u] == 'Library 1']
    own = [g for g, _t in L1]
    triplet_of = {f: k for k, f in fmap['Library 1'].items()}
    alone = fa.marker_sums_for(data, l1, [triplet_of[g] for g in own], channels)
    expected, _m, _u = fst.marker_deltas(alone, counts.loc[l1, own], channels, 'all', [NT], 10)
    np.testing.assert_allclose(deltas.loc[l1, expected.columns].values, expected.values)
    assert deltas.loc[l1, 'B0: M0'].isna().all() and deltas.loc[l1, f'{NT}: M0'].notna().all()
    assert list(meta['feature'][:2]) == ['A0: M0', 'A0: M1'] and len(meta) == deltas.shape[1]
    res = fst.marker_statistics(deltas, meta, design(list(counts.index))[0], group_order=list(TISSUES))
    assert set(res['comparisons']) == {'Spleen', 'Gut'}


# ---------------------------------------------------------------------------
# Export summary
# ---------------------------------------------------------------------------

def test_debarcode_summary_lists_the_library_and_out_of_library_events(tmp_path):
    import csv

    names = KEYS[:3]
    d = fcc.FlowCodeDesign.build({t: t for t in TAGS}, [(k, tuple(TRIPLETS[i])) for i, k in enumerate(names)])
    base = {'total_events': 100, 'untransduced': 10, 'unresolved': 5, 'rescued': 0, 'rescued_toofew': 0,
            'pairwise_corrected': 0, 'per_combo': {names[0]: 7, names[1]: 3, names[2]: 1}}
    rows = [('a', dict(base, library='Library 1', out_of_library=1)), ('b', dict(base, library='Other',
                                                                                    out_of_library=None))]
    fcc.write_debarcode_summary_csv(tmp_path / 's.csv', rows, d)
    with open(tmp_path / 's.csv', newline='') as fh:
        out = list(csv.reader(fh))
    assert out[0] == list(fcc.SUMMARY_COLUMNS) + ['Library', 'Out of library'] + list(names)
    assert out[1][8:10] == ['Library 1', '1'] and out[2][8:10] == ['Other', '']
    fcc.write_debarcode_summary_csv(tmp_path / 'one.csv', [('a', base)], d)
    with open(tmp_path / 'one.csv', newline='') as fh:
        assert next(csv.reader(fh)) == list(fcc.SUMMARY_COLUMNS) + list(names)
