"""
test_flowcode_analysis.py
-------------------------
The FlowCode Analysis page's data and statistics (flowcode_analysis.py,
flowcode_stats.py) on synthetic exported files: 3 tissues x 6 mice, six
combinations, Pancreas enriched 4-fold for G1, treated mice depleted 4-fold
for G2 in LN only. Events sit well inside or outside every gate, so the
gated counts are exact.

    pytest tests/test_flowcode_analysis.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flowkit import Dimension, GatingStrategy, gates

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))

import flowcode_analysis as fa  # noqa: E402
import flowcode_stats as fst  # noqa: E402
from honeychrome.controller_components import functions as fn  # noqa: E402
from honeychrome.controller_components.functions import define_polygon_gate, define_rectangle_gate  # noqa: E402
from honeychrome.controller_components.transform import Transform  # noqa: E402

TISSUES = ('Spleen', 'LN', 'Pancreas')
MICE = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6')
TREATED = ('M1', 'M2', 'M3')
COMBOS = ('G1', 'G2', 'G3', 'G4', 'G5', 'G6')
COMBOS_KEYWORD = ';'.join(f'FC {c}=K{c}' for c in COMBOS)
CHARACTERISATION = 'abc123abc123'
SUFFIX = ' FlowCode'
PNN = ['Time', 'FSC-A', 'SSC-A', 'CD8', 'CD4', 'FlowCode', 'FlowCode ID'] + [f'FC {c}' for c in COMBOS]
N_IN, N_OUT = 600, 300
SHARE_UNTRANSDUCED, SHARE_UNRESOLVED = 0.2, 0.05


def sample_key(tissue, mouse):
    return f'Raw/{tissue}/CD8_{tissue}_a_b_c_d_{mouse}_x_y.fcs'


SAMPLES = [sample_key(t, m) for t in TISSUES for m in MICE]
DISPLAY = {k: Path(k).stem for k in SAMPLES}


def barcode_probs(tissue, mouse):
    w = np.ones(len(COMBOS))
    if tissue == 'Pancreas':
        w[0] *= 4.0
    if tissue == 'LN' and mouse in TREATED:
        w[1] /= 4.0
    return w / w.sum()


def make_events(tissue, mouse, seed, t_max=1000.0):
    """Exported events and the truth: (events, in_cd8, early, ids)."""
    rng = np.random.default_rng(seed)
    n = N_IN + N_OUT
    in_cd8 = np.zeros(n, dtype=bool)
    in_cd8[:N_IN] = True
    fsc = rng.uniform(30_000, 120_000, n)
    ssc = rng.uniform(30_000, 120_000, n)
    cd8 = np.where(in_cd8, rng.uniform(20_000, 60_000, n), rng.uniform(-50, 200, n))
    cd4 = rng.uniform(-50, 200, n)
    # Time avoids [0.45, 0.55] x t_max so the 500-unit early gate is exact for t_max = 1000
    early_half = rng.random(n) < 0.5
    time = np.where(early_half, rng.uniform(0, 0.45 * t_max, n), rng.uniform(0.55 * t_max, t_max, n))
    p = barcode_probs(tissue, mouse)
    kind = rng.random(n)
    ids = 1 + rng.choice(len(COMBOS), size=n, p=p)
    ids = np.where(kind < SHARE_UNTRANSDUCED, 0, ids)
    ids = np.where((kind >= SHARE_UNTRANSDUCED) & (kind < SHARE_UNTRANSDUCED + SHARE_UNRESOLVED),
                   len(COMBOS) + 1, ids)
    fc_cols = np.zeros((n, len(COMBOS)))
    barcoded = (ids >= 1) & (ids <= len(COMBOS))
    level = rng.uniform(5_000, 9_000, n)
    fc_cols[np.flatnonzero(barcoded), ids[barcoded] - 1] = level[barcoded]
    flowcode = np.where(barcoded, level, 0.0)
    events = np.column_stack([time, fsc, ssc, cd8, cd4, flowcode, ids.astype(float), fc_cols])
    return events, in_cd8, time < 500, ids


def write_export(path, events, pnn, keywords):
    kw = dict(keywords)
    for i, name in enumerate(pnn, start=1):
        kw[f'$P{i}N'] = name
        kw[f'$P{i}B'] = '32'
        kw[f'$P{i}E'] = '0,0'
        kw[f'$P{i}R'] = '262144'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fn.write_fcs(np.asarray(events, dtype=np.float64), kw, path)


def export_path(folder, key):
    return fa.expected_export_path(folder, key, 'Raw', DISPLAY[key], SUFFIX)


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    folder = tmp_path_factory.mktemp('exp') / 'FlowCode Unmixed'
    truth = {}
    for i, key in enumerate(SAMPLES):
        tissue, mouse = key.split('/')[1], key.split('_')[-3]
        events, in_cd8, early, ids = make_events(tissue, mouse, seed=100 + i)
        kw = {'FLOWCODECOMBOS': COMBOS_KEYWORD, 'FLOWCODECHARACTERISATION': CHARACTERISATION,
              'FLOWCODESOURCE': key}
        write_export(export_path(folder, key), events, PNN, kw)
        truth[key] = {'in_cd8': in_cd8, 'early': early, 'ids': ids}
    return {'folder': folder, 'truth': truth}


def _transforms():
    lin = {}
    for ch in ('FSC-A', 'SSC-A'):
        lin[ch] = Transform()
        lin[ch].set_transform(id=0, limits=[0, 1])
    for ch in ('CD8', 'CD4'):
        lin[ch] = Transform()
        lin[ch].set_transform(id=1, limits=[-0.1, 1.0])
    lin['Time'] = Transform()
    lin['Time'].set_transform(id='default', limits=[0, 5000.0])
    return lin


def make_snapshot(overrides=None):
    tr = _transforms()
    g = GatingStrategy()
    pts, dx, dy = define_polygon_gate([(0.05, 0.05), (0.6, 0.05), (0.6, 0.6), (0.05, 0.6)], 'FSC-A', 'SSC-A', tr)
    g.add_gate(gates.PolygonGate('Cells', [dx, dy], pts), ('root',))
    g.add_gate(gates.RectangleGate('CD8+', list(define_rectangle_gate((0.6, -0.1), (0.5, 0.5), 'CD8', 'CD4', tr))),
               ('root', 'Cells'))
    g.add_gate(gates.RectangleGate('Early', [Dimension('Time', range_min=0, range_max=500)]),
               ('root', 'Cells', 'CD8+'))
    return fa.SampleGating(gating=g, transforms=tr, overrides=overrides or {})


def _expected_counts(truth, mask):
    ids = truth['ids'][mask]
    return {c: int(np.sum(ids == i + 1)) for i, c in enumerate(COMBOS)}


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def test_events_sit_clear_of_the_gate_boundaries():
    tr = _transforms()
    events, in_cd8, _early, _ids = make_events('Spleen', 'M1', seed=1)
    cd8 = tr['CD8'].xform.apply(events[:, 3])
    fsc = tr['FSC-A'].xform.apply(events[:, 1])
    assert cd8[in_cd8].min() > 0.62 and cd8[~in_cd8].max() < 0.58
    assert 0.05 < fsc.min() and fsc.max() < 0.6


def test_link_by_keyword_and_name_with_status(experiment, tmp_path):
    folder = tmp_path / 'exports'
    keys = SAMPLES[:5]
    base_kw = {'FLOWCODECOMBOS': COMBOS_KEYWORD, 'FLOWCODECHARACTERISATION': CHARACTERISATION}
    events = make_events('Spleen', 'M1', seed=3)[0]
    write_export(folder / 'renamed.fcs', events, PNN, dict(base_kw, FLOWCODESOURCE=keys[0]))
    write_export(export_path(folder, keys[1]), events, PNN, base_kw)              # name only
    write_export(export_path(folder, keys[2]), events, PNN,
                 {'FLOWCODECOMBOS': 'FC X=K', 'FLOWCODESOURCE': keys[2]})          # other combos
    write_export(export_path(folder, keys[3]), events, PNN,
                 dict(base_kw, FLOWCODECHARACTERISATION='old', FLOWCODESOURCE=keys[3]))
    write_export(folder / 'stray.fcs', events, PNN, base_kw)
    out = fa.link_exports(folder, keys, 'Raw', DISPLAY, SUFFIX, COMBOS_KEYWORD, CHARACTERISATION)
    files = out['files']
    assert files[keys[0]]['status'] == fa.LINK_OK and files[keys[0]]['path'].endswith('renamed.fcs')
    assert files[keys[1]]['status'] == fa.LINK_OK
    assert files[keys[2]]['status'] == fa.LINK_OTHER_COMBOS
    assert files[keys[3]]['status'] == fa.LINK_OTHER_CHARACTERISATION
    assert files[keys[4]]['status'] == fa.LINK_MISSING and files[keys[4]]['path'] is None
    assert [Path(p).name for p in out['unlinked']] == ['stray.fcs']


def test_expected_export_path_mirrors_the_raw_folder(tmp_path):
    p = fa.expected_export_path(tmp_path, 'Raw/Spleen/a.fcs', 'Raw', 'Mouse a', ' FlowCode')
    assert p == tmp_path / 'Spleen' / 'Mouse a FlowCode (Unmixed).fcs'


# ---------------------------------------------------------------------------
# Gating and counting
# ---------------------------------------------------------------------------

def test_counts_in_the_population_are_exact(experiment):
    snap = make_snapshot()
    for key in SAMPLES[:4]:
        rec = fa.count_sample(export_path(experiment['folder'], key), key, snap, ['CD8+'])
        truth = experiment['truth'][key]
        assert rec['events'] == N_IN + N_OUT and rec['in_population'] == N_IN
        assert rec['counts'] == _expected_counts(truth, truth['in_cd8'])
        ids = truth['ids'][truth['in_cd8']]
        assert rec['untransduced'] == int(np.sum(ids == 0))
        assert rec['unresolved'] == int(np.sum(ids == len(COMBOS) + 1))
        assert rec['barcoded'] == sum(rec['counts'].values())
        assert rec['not_barcoded'] == rec['untransduced'] + rec['unresolved']


def test_time_gate_uses_each_sample_range(experiment, tmp_path):
    snap = make_snapshot()
    key = SAMPLES[0]
    rec = fa.count_sample(export_path(experiment['folder'], key), key, snap, ['Early'])
    truth = experiment['truth'][key]
    assert rec['counts'] == _expected_counts(truth, truth['in_cd8'] & truth['early'])
    # a run three times longer: the Time axis follows the file, the 500-unit gate stays exact
    events, in_cd8, _e, ids = make_events('Spleen', 'M1', seed=7, t_max=3000.0)
    path = tmp_path / 'long.fcs'
    write_export(path, events, PNN, {})
    rec = fa.count_sample(path, key, snap, ['Early'])
    early = events[:, 0] < 500
    assert rec['counts'] == {c: int(np.sum(ids[in_cd8 & early] == i + 1)) for i, c in enumerate(COMBOS)}


def test_each_sample_uses_its_own_gate_override(experiment):
    tr = _transforms()
    moved = gates.RectangleGate('CD8+', list(define_rectangle_gate((0.0, 0.6), (0.3, 0.3), 'CD8', 'CD4', tr)))
    snap = make_snapshot(overrides={SAMPLES[1]: {'CD8+': moved}})
    rec_other = fa.count_sample(export_path(experiment['folder'], SAMPLES[0]), SAMPLES[0], snap, ['CD8+'])
    rec_moved = fa.count_sample(export_path(experiment['folder'], SAMPLES[1]), SAMPLES[1], snap, ['CD8+'])
    rec_after = fa.count_sample(export_path(experiment['folder'], SAMPLES[2]), SAMPLES[2], snap, ['CD8+'])
    assert rec_other['in_population'] == N_IN and rec_after['in_population'] == N_IN
    assert rec_moved['in_population'] == 0


def test_lookup_gating_matches_flowkit_exact_gating(experiment):
    """The rasterised boundaries agree with FlowKit's exact gating on every event here."""
    from flowkit import Sample

    snap = make_snapshot()
    key = SAMPLES[0]
    sample = fn.sample_from_fcs(str(export_path(experiment['folder'], key)))
    events = np.asarray(sample.get_events(source='raw'), dtype=np.float64)
    mask = snap.population_mask(key, events, list(sample.pnn_labels), ['CD8+'])
    gs = GatingStrategy()
    gs.add_gate(snap.gating.get_gate('Cells'), ('root',))
    gs.add_gate(snap.gating.get_gate('CD8+'), ('root', 'Cells'))
    for ch in ('FSC-A', 'SSC-A', 'CD8', 'CD4'):
        gs.transformations[ch] = snap.transforms[ch].xform
    fk = Sample(events[:, :5], channel_labels=PNN[:5], sample_id='x')
    exact = gs.gate_sample(fk).get_gate_membership('CD8+')
    np.testing.assert_array_equal(mask, exact)


def test_legacy_files_count_barcoded_events_from_the_combo_channels(experiment, tmp_path):
    events, in_cd8, _e, ids = make_events('LN', 'M4', seed=9)
    keep = [i for i, ch in enumerate(PNN) if ch != 'FlowCode ID']
    path = tmp_path / 'legacy.fcs'
    write_export(path, events[:, keep], [PNN[i] for i in keep], {})
    rec = fa.count_sample(path, SAMPLES[0], make_snapshot(), ['CD8+'])
    assert rec['counts'] == {c: int(np.sum(ids[in_cd8] == i + 1)) for i, c in enumerate(COMBOS)}
    assert rec['untransduced'] is None and rec['unresolved'] is None
    assert rec['not_barcoded'] == int(np.sum((ids[in_cd8] == 0) | (ids[in_cd8] == len(COMBOS) + 1)))


def test_a_gate_on_a_missing_channel_is_reported(experiment, tmp_path):
    events = make_events('LN', 'M4', seed=9)[0]
    keep = [i for i, ch in enumerate(PNN) if ch != 'CD4']
    path = tmp_path / 'no_cd4.fcs'
    write_export(path, events[:, keep], [PNN[i] for i in keep], {})
    with pytest.raises(ValueError, match='CD8\\+'):
        fa.count_sample(path, SAMPLES[0], make_snapshot(), ['CD8+'])


# ---------------------------------------------------------------------------
# Tables, filters and QC
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def counted(experiment):
    snap = make_snapshot()
    records = {k: fa.count_sample(export_path(experiment['folder'], k), k, snap, ['CD8+']) for k in SAMPLES}
    counts, meta = fa.count_table(records, combo_order=list(COMBOS))
    return counts, meta


def test_count_table_keeps_combination_order(counted):
    counts, meta = counted
    assert list(counts.columns) == list(COMBOS) and list(counts.index) == SAMPLES
    assert (meta['in_population'] == N_IN).all()
    assert (counts.sum(axis=1) == meta['barcoded']).all()


def test_filters_drop_rare_combinations_then_small_samples():
    counts = pd.DataFrame({'A': [10, 12, 0], 'B': [3, 4, 2], 'C': [5, 1, 3]}, index=['s1', 's2', 's3'])
    out = fa.filter_counts(counts, exclude_combos=['C'], min_combo_events=20, min_sample_events=10)
    assert out.excluded_combos == ['C'] and out.rare_combos == {'B': 9}
    assert out.small_samples == {'s3': 0}
    assert list(out.counts.columns) == ['A'] and list(out.counts.index) == ['s1', 's2']
    assert out.denominator.tolist() == [10.0, 12.0]
    assert len(out.notes) == 3


@pytest.mark.parametrize('values, expected', [([1, 1, 1, 1], 0.0), ([0, 0, 0, 1], 0.75),
                                              ([1, 2, 3, 4], 0.25)])
def test_gini_matches_ineq(values, expected):
    assert fa.gini(values) == pytest.approx(expected)


def test_qc_table(counted):
    counts, meta = counted
    qc = fa.qc_table(counts, meta, DISPLAY)
    key = SAMPLES[0]
    assert qc.loc[key, 'name'] == DISPLAY[key]
    assert qc.loc[key, 'pct_barcoded'] == pytest.approx(100.0 * counts.loc[key].sum() / N_IN)
    assert qc.loc[key, 'top_share_pct'] == pytest.approx(100.0 * counts.loc[key].max() / counts.loc[key].sum())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _groups(by):
    return {k: (k.split('/')[1] if by == 'tissue' else ('treated' if k.split('_')[-3] in TREATED else 'control'))
            for k in SAMPLES}


MOUSE = {k: k.split('_')[-3] for k in SAMPLES}
TISSUE = {k: k.split('/')[1] for k in SAMPLES}


def test_tissue_vs_spleen_paired_by_mouse_finds_the_enriched_combination(counted):
    counts, _meta = counted
    filt = fa.filter_counts(counts)
    stats = fst.run_statistics(filt.counts, filt.denominator, _groups('tissue'), group_order=list(TISSUES),
                               reference='Spleen', pairing=MOUSE, run_counts=True, run_freq=True)
    assert stats['comparisons'] == ['LN vs Spleen', 'Pancreas vs Spleen']
    # 4-fold weight among six combinations: share 4/9 against 1/6, log2 = 1.415. The Frequencies
    # test reports log2 odds, which for a combination holding most of the library is larger:
    # about 0.6 of the robust library against 0.18, odds ratio about 7
    for key, (low, high) in (('counts', (1.0, 1.9)), ('freq', (2.2, 3.2))):
        res = stats[key]
        pan = res[res['comparison'] == 'Pancreas vs Spleen'].set_index('feature')
        assert bool(pan.loc['G1', 'significant'])
        assert low < pan.loc['G1', 'logFC'] < high
        assert not pan.drop(index='G1')['significant'].any()


def test_frequencies_are_capped_by_default_and_can_run_unweighted(counted):
    counts, _meta = counted
    filt = fa.filter_counts(counts)
    kwargs = dict(group_order=list(TISSUES), reference='Spleen', pairing=MOUSE, run_counts=False, run_freq=True)
    capped = fst.run_statistics(filt.counts, filt.denominator, _groups('tissue'), **kwargs)
    plain = fst.run_statistics(filt.counts, filt.denominator, _groups('tissue'), freq_weights=None, **kwargs)
    assert capped['freq_weights'] == 'capped' and plain['freq_weights'] is None
    assert any(n.startswith('Frequencies: each sample weighted') for n in capped['notes'])
    assert not any(n.startswith('Frequencies:') for n in plain['notes'])
    assert set(fst.results_table(capped)['test']) == {fst.FREQ_TEST_NAMES['capped']}
    assert set(fst.results_table(plain)['test']) == {fst.FREQ_TEST_NAMES[None]}
    for stats in (capped, plain):
        pan = stats['freq'][stats['freq']['comparison'] == 'Pancreas vs Spleen'].set_index('feature')
        assert bool(pan.loc['G1', 'significant'])
    assert fa.analysis_settings({'freq_weights': 'voom'})['freq_weights'] == 'capped'
    assert fa.analysis_settings({'freq_weights': None})['freq_weights'] is None


def test_split_by_runs_the_comparison_within_each_level(counted):
    counts, _meta = counted
    filt = fa.filter_counts(counts)
    stats = fst.run_statistics(filt.counts, filt.denominator, _groups('treatment'),
                               group_order=['control', 'treated'], reference='control',
                               split_by=TISSUE, run_counts=True)
    assert stats['comparisons'] == ['treated vs control [Spleen]', 'treated vs control [LN]',
                                    'treated vs control [Pancreas]']
    res = stats['counts']
    sig = res[res['significant']]
    assert set(zip(sig['comparison'], sig['feature'])) == {('treated vs control [LN]', 'G2')}
    ln = res[(res['comparison'] == 'treated vs control [LN]') & (res['feature'] == 'G2')].iloc[0]
    assert ln['logFC'] < -1.2
    assert fst.comparison_samples(stats, ('control', 'treated', 'LN')) == [
        k for k in SAMPLES if TISSUE[k] == 'LN']


def test_too_few_samples_per_group_is_explained(counted):
    counts, _meta = counted
    filt = fa.filter_counts(counts)
    few = {k: TISSUE[k] for k in SAMPLES if MOUSE[k] in ('M1', 'M2')}
    with pytest.raises(RuntimeError, match='Fewer than two groups'):
        fst.run_statistics(filt.counts, filt.denominator, few, reference='Spleen')


def test_thresholds_reapply_without_refitting(counted):
    counts, _meta = counted
    filt = fa.filter_counts(counts)
    stats = fst.run_statistics(filt.counts, filt.denominator, _groups('tissue'), reference='Spleen',
                               pairing=MOUSE)
    loose = fst.reapply_thresholds(stats, 0.05, 0.1, 'per_comparison', False)
    assert loose['counts']['significant'].sum() >= stats['counts']['significant'].sum()
    np.testing.assert_allclose(loose['counts']['logFC'], stats['counts']['logFC'])


def test_pca_figures_and_export(counted, tmp_path):
    counts, meta = counted
    filt = fa.filter_counts(counts)
    groups = _groups('tissue')
    stats = fst.run_statistics(filt.counts, filt.denominator, groups, reference='Spleen', pairing=MOUSE,
                               run_freq=True)
    pca = fst.sample_pca(stats)
    assert list(pca['scores'].index) == list(stats['samples'])
    assert pca['groups'][0] == 'Spleen'
    figs = fst.comparison_figures(stats, stats['pairs'][1], 0.05, 1.0, 'per_comparison', False, 'CD8+', DISPLAY)
    assert {'counts_volcano', 'freq_volcano', 'fold_change_bars', 'heatmap_significant', 'heatmap_all'} <= set(figs)
    assert all(f['fig'] is not None for f in figs.values()), {k: f['error'] for k, f in figs.items()}
    summary = fst.summary_figures(stats, False, 'CD8+', sample_labels=DISPLAY)
    assert summary['comparison_heatmap']['fig'] is not None and summary['sample_pca']['fig'] is not None
    assert 'counts' in summary['comparison_heatmap']['fig'].axes[0].get_title()
    assert 'frequencies' in summary['comparison_heatmap_freq']['fig'].axes[0].get_title()
    qc = fa.qc_table(counts, meta, DISPLAY)
    folder = fst.default_export_folder(tmp_path, 'CD8+ / T cells')
    assert folder.parent == tmp_path and folder.name.startswith('CD8')
    written = fst.write_analysis_folder(
        folder, population='CD8+', raw_counts=counts, meta=meta, qc=qc, filtered=filt, stats=stats,
        sample_table=pd.DataFrame({'group': groups}), settings_text='settings',
        figures=[('volcano', lambda: figs['counts_volcano']['fig']),
                 ('qc', lambda: fst.qc_figure(qc, filt.counts, groups, False, 'CD8+'))],
        sample_labels=DISPLAY)
    names = {p.name for p in written}
    assert {'counts_all_samples.csv', 'counts_analysed.csv', 'percent_of_barcoded.csv', 'pct_matrix.csv',
            'qc.csv', 'samples.csv', 'results.csv', 'settings.txt', 'volcano.png', 'qc.png'} <= names
    assert any(n.endswith('report.pdf') for n in names)
    pct = pd.read_csv(folder / 'pct_matrix.csv')
    assert pct.columns[0] == 'sample' and pct['sample'].iloc[0] == DISPLAY[SAMPLES[0]]
    np.testing.assert_allclose(pct.iloc[:, 1:].sum(axis=1), 100.0)
    res = pd.read_csv(folder / 'results.csv')
    assert set(res['test']) == {'Counts (NB GLM, quasi-likelihood F test)', fst.FREQ_TEST_NAMES['capped']}
    sizes = pd.read_csv(folder / 'library_sizes.csv', index_col=0)
    np.testing.assert_allclose(sizes['library_size'] / sizes['barcoded'], sizes['factor'])


def _clone_counts(seed=0, n_mice=6, clone_share=0.6):
    """Spleen and tissue per mouse, 30 combinations; combination C0 makes up
    *clone_share* of every tissue sample, the rest keep their spleen shares."""
    rng = np.random.default_rng(seed)
    combos = [f'C{i}' for i in range(30)] + ['NON-Targetting 1', 'NON-Targetting 2']
    rows, groups, mouse = {}, {}, {}
    for m in range(n_mice):
        share = rng.lognormal(0, 0.5, len(combos))
        share /= share.sum()
        tissue = share.copy()
        tissue[0] = 0.0
        tissue *= (1 - clone_share) / tissue.sum()
        tissue[0] = clone_share
        for name, p in (('Spleen', share), ('Gut', tissue)):
            key = f'{name}_M{m}'
            rows[key] = rng.poisson(3000 * p)
            groups[key], mouse[key] = name, f'M{m}'
    return pd.DataFrame(rows, index=combos).T, groups, mouse


def test_robust_library_sizes_keep_the_other_combinations_level():
    counts, groups, mouse = _clone_counts()
    denom = counts.sum(axis=1)
    medians = {}
    for norm in ('total', 'robust'):
        stats = fst.run_statistics(counts, denom, groups, reference='Spleen', pairing=mouse,
                                   normalisation=norm)
        res = stats['counts'].set_index('feature')
        medians[norm] = float(res.drop(index='C0')['logFC'].median())
        assert bool(res.loc['C0', 'significant'])
    assert medians['total'] < -1.0          # log2(0.4) = -1.32: every other combination pushed down
    assert abs(medians['robust']) < 0.15


def test_leave_one_out_flags_a_hit_that_rests_on_single_mice(tmp_path):
    """C5 is up in every gut sample, and much more in mouse M0: without some
    mice it is no longer significant, and M0 moves its log2FC most. C0 holds
    in every refit. The check is judged again for new thresholds, survives
    saving, and shows in the results table and the combination plot."""
    counts, groups, mouse = _clone_counts(clone_share=0.15)
    gut = [k for k in counts.index if k.startswith('Gut')]
    counts.loc[gut, 'C5'] = np.round(counts.loc[gut, 'C5'] * 1.8)
    counts.loc['Gut_M0', 'C5'] = np.round(counts.loc['Gut_M0', 'C5'] * 4)
    denom = counts.sum(axis=1)
    kwargs = dict(reference='Spleen', pairing=mouse, fc_threshold=0.0)
    stats = fst.run_statistics(counts, denom, groups, **kwargs)
    steps = []
    loo = fst.leave_one_out(counts, denom, groups, mouse, kwargs,
                            progress=lambda n, total, text='': steps.append((n, total)))
    assert loo['levels'] == [f'M{m}' for m in range(6)] and not loo['failed']
    assert steps[-1] == (6, 6)                  # the progress bar closes
    stats['leave_one_out'] = dict(loo, labels={'M0': 'mouse 0'})
    fst.save_stats(tmp_path / 'stats', fst.reapply_thresholds(stats, 0.05, 0.0, 'per_comparison', False), 'x')
    stats = fst.reapply_thresholds(fst.load_stats(tmp_path / 'stats')[0], 0.05, 0.0, 'per_comparison', False)
    summary = stats['leave_one_out']['summary'].set_index('feature')
    assert summary.loc['C0', ['loo_significant', 'loo_refits', 'loo_flag']].tolist() == [6, 6, '']
    c5 = summary.loc['C5']
    assert c5['loo_flag'] == '⚠' and c5['loo_significant'] < 6 and c5['loo_influential'] == 'M0'
    table = fst.results_table(stats).set_index('feature')
    assert table.loc['C5', 'loo_influential'] == 'mouse 0'
    assert table['loo_refits'].notna().sum() == int(stats['counts']['significant'].sum())
    fig = fst.combination_figure(stats, 'C5', stats['pairs'][0], 'per_comparison', False)
    assert 'most influential mouse 0' in fig.axes[0].get_title()
    strict = fst.reapply_thresholds(stats, 0.001, 0.0, 'per_comparison', False)
    assert 'C5' not in set(strict['leave_one_out']['summary']['feature'])


def test_clone_filter_removes_the_clone_in_that_mouse_only():
    counts, groups, mouse = _clone_counts(clone_share=0.15)
    counts.loc['Gut_M0', 'C5'] = 2000                      # one mouse's expanded clone
    denom = counts.sum(axis=1)
    found = fst.expanded_clones(counts, denom, [k for k in counts.index if groups[k] == 'Gut'])
    assert set(zip(found['sample'], found['combination'])) >= {('Gut_M0', 'C5')}
    masked, table = fst.remove_clones(counts, found, mouse, max_levels=2, groups=groups)
    assert masked.loc[['Gut_M0', 'Spleen_M0'], 'C5'].isna().all()
    assert masked.loc['Gut_M1', 'C5'] == counts.loc['Gut_M1', 'C5']
    # C0 is large in every mouse's gut: no mouse stands out, so it is no clone
    assert 'C0' not in set(found['combination'])
    assert not masked['C0'].isna().any()
    # C7 expanded in two mice: kept as an effect when more than one level is allowed to share it
    counts.loc[['Gut_M1', 'Gut_M2'], 'C7'] = 2000
    denom = counts.sum(axis=1)
    found = fst.expanded_clones(counts, denom, [k for k in counts.index if groups[k] == 'Gut'], groups)
    assert set(found.loc[found['combination'] == 'C7', 'sample']) == {'Gut_M1', 'Gut_M2'}
    masked, table = fst.remove_clones(counts, found, mouse, max_levels=1, groups=groups)
    assert table.loc[table['combination'] == 'C7', 'action'].str.startswith('kept').all()
    assert not masked['C7'].isna().any()
    counts.loc[['Gut_M1', 'Gut_M2'], 'C7'] = counts.loc[['Spleen_M1', 'Spleen_M2'], 'C7'].values
    denom = counts.sum(axis=1)
    stats = fst.run_statistics(counts, denom, groups, reference='Spleen', pairing=mouse, clone_filter={})
    assert (stats['clones']['action'] == 'removed').sum() == 1
    assert stats['denominator']['Gut_M0'] == denom['Gut_M0'] - 2000


def test_control_normalisation_leaves_out_expanded_controls():
    counts, groups, mouse = _clone_counts()
    assert fst.control_candidates(counts.columns) == ['NON-Targetting 1', 'NON-Targetting 2']
    counts.loc['Gut_M2', 'NON-Targetting 1'] = 3000          # a clone among the controls
    denom = counts.sum(axis=1)
    stats = fst.run_statistics(counts, denom, groups, reference='Spleen', pairing=mouse,
                               normalisation='control', control_combos=fst.control_candidates(counts.columns))
    assert any('expanded control clone' in n for n in stats['notes'])
    res = stats['counts'].set_index('feature')
    assert abs(float(res.drop(index=['C0', 'NON-Targetting 1', 'NON-Targetting 2'])['logFC'].median())) < 0.3
    with pytest.raises(RuntimeError, match='control combination'):
        fst.run_statistics(counts, denom, groups, reference='Spleen', normalisation='control',
                           control_combos=[])


def test_assign_by_pattern_takes_the_first_match_case_insensitive():
    names = {'a': 'A1 Spleen_M1', 'b': 'B2 pp_M1', 'c': 'C3 Liver_M1', 'd': 'D4 spleen pp'}
    groups, matched = fa.assign_by_pattern(list(names), names, ['Spleen', 'PP', 'Empty'],
                                           {'Spleen': 'SPLEEN', 'PP': r'\bPP', 'Empty': ''})
    assert groups == {'a': 'Spleen', 'b': 'PP', 'd': 'Spleen'}
    assert matched == {'Spleen': 2, 'PP': 1}
    with pytest.raises(ValueError, match='PP'):
        fa.assign_by_pattern(list(names), names, ['PP'], {'PP': '('})


def test_group_colours_are_kept_and_new_groups_get_unused_ones():
    settings = {'group_names': ['A', 'B', 'C'], 'group_colors': {'B': '#d60000'}}
    assert fa.resolve_group_colours(settings)
    colours = settings['group_colors']
    assert colours['B'] == '#d60000' and len(set(colours.values())) == 3
    assert not fa.resolve_group_colours(settings)


def test_fold_change_bars_and_combination_plot_values():
    counts, groups, mouse = _clone_counts(clone_share=0.15)
    counts.loc['Gut_M0', 'C5'] = 2000
    denom = counts.sum(axis=1)
    stats = fst.run_statistics(counts, denom, groups, reference='Spleen', pairing=mouse, clone_filter={},
                               run_freq=True)
    pair = stats['pairs'][0]
    figs = fst.comparison_figures(stats, pair, 0.05, 1.0, 'per_comparison', False, 'CD8+',
                                  group_colours={'Gut': '#123456', 'Spleen': '#abcdef'}, combination='C5',
                                  group_order=['Spleen', 'Gut'])
    assert all(f['fig'] is not None for f in figs.values()), {k: f['error'] for k, f in figs.items()}
    view = stats['counts'][stats['counts']['comparison'] == fst.comparison_label(*pair)]
    bars = fst.fold_change_bars(view)
    assert bars['logFC'].is_monotonic_increasing
    assert figs['fold_change_bars']['fig']._bar_features == bars['feature'].tolist()
    values = fst.combination_values(stats, 'C5', pair, ['Spleen', 'Gut'])
    assert values['group'].tolist() == ['Spleen'] * 6 + ['Gut'] * 6
    assert dict(zip(values['sample'], values['pair'])) == {k: mouse[k] for k in values['sample']}
    assert set(values.loc[values['removed'], 'sample']) == {'Gut_M0', 'Spleen_M0'}
    expected = 2.0 ** stats['log2_values'].loc['Gut_M1', 'C5']
    assert values.set_index('sample').loc['Gut_M1', 'percent'] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Several populations, population contrasts, marker intensities
# ---------------------------------------------------------------------------

def test_each_gate_is_counted_as_its_own_population_with_marker_sums(experiment):
    snap = make_snapshot()
    key = SAMPLES[0]
    path = export_path(experiment['folder'], key)
    out = fa.count_sample_populations(path, key, snap, ['CD8+', 'Early'], ['CD8', 'CD4', 'Time', 'absent'])
    assert set(out) == {'CD8+', 'Early'}
    for gate in ('CD8+', 'Early'):
        single = fa.count_sample(path, key, snap, [gate])
        assert {k: out[gate][k] for k in single} == single
    rec = out['CD8+']
    assert rec['marker_channels'] == ['CD8', 'CD4']             # Time has a per-sample axis; absent is not in the file
    sample = fn.sample_from_fcs(str(path))
    events = np.asarray(sample.get_events(source='raw'), dtype=np.float64)
    truth = experiment['truth'][key]
    cd4 = snap.transforms['CD4'].xform.apply(events[:, PNN.index('CD4')])
    for i, combo in enumerate(COMBOS):
        rows = truth['in_cd8'] & (truth['ids'] == i + 1)
        assert rec['marker_sums'][i, 1] == pytest.approx(cd4[rows].sum())
    whole = fa.count_sample_populations(path, key, snap, [])
    assert list(whole) == [fa.ALL_EVENTS] and whole[fa.ALL_EVENTS]['in_population'] == N_IN + N_OUT


def test_units_table_saved_records_and_marker_cache(experiment, tmp_path):
    snap = make_snapshot()
    pops = ['CD8+', 'Early']
    records = {p: {} for p in pops}
    for key in SAMPLES[:3]:
        out = fa.count_sample_populations(export_path(experiment['folder'], key), key, snap, pops, ['CD8', 'CD4'])
        for p in pops:
            records[p][key] = out[p]
    path = tmp_path / 'cache' / fa.MARKERS_FILE
    assert fa.save_marker_sums(path, 'sig1', records, pops)
    assert all('marker_sums' not in rec for p in pops for rec in records[p].values())
    assert fa.load_marker_sums(path, 'other') is None
    saved = fa.load_marker_sums(path, 'sig1')
    assert saved['channels'] == ['CD8', 'CD4'] and saved['combos'] == list(COMBOS)
    counts, meta = fa.units_table(records, pops, combo_order=list(COMBOS))
    assert len(counts) == 6 and set(meta['population']) == set(pops)
    unit = fa.unit_id(SAMPLES[1], 'Early', True)
    assert fa.unit_parts(unit) == (SAMPLES[1], 'Early')
    assert counts.loc[unit].to_dict() == records['Early'][SAMPLES[1]]['counts']
    sums = fa.marker_sums_for(saved, [unit, 'missing'], ['G2', 'nope'], ['CD4'])
    assert sums.shape == (2, 2, 1) and sums[1].sum() == 0 and sums[0, 1, 0] == 0
    assert sums[0, 0, 0] == pytest.approx(saved['sums'][saved['units'].index(unit), 1, 1])
    # counts saved before several populations were counted read as one population
    old = {'records': records['CD8+'], 'population': 'CD8+'}
    assert fa.saved_records(old) == (['CD8+'], {'CD8+': records['CD8+']})
    single, _meta = fa.units_table({'CD8+': records['CD8+']}, ['CD8+'])
    assert list(single.index) == SAMPLES[:3]


def _two_population_counts(seed=1, effect=4.0):
    """6 mice x Spleen/Pancreas/Liver x CD8/Tconv; G0 enriched in the CD8
    pancreas only, G1 enriched in the liver of both populations."""
    rng = np.random.default_rng(seed)
    combos = [f'G{j}' for j in range(30)]
    base = rng.lognormal(0, 0.7, size=len(combos))
    rows, groups, mouse, pop = {}, {}, {}, {}
    for m in range(6):
        own = base * rng.lognormal(0, 0.5, size=len(combos))
        for p in ('CD8', 'Tconv'):
            for t in ('Spleen', 'Pancreas', 'Liver'):
                w = own.copy()
                if p == 'CD8' and t == 'Pancreas':
                    w[0] *= effect
                if t == 'Liver':
                    w[1] *= 3.0
                mu = 3000 * w / w.sum()
                unit = fa.unit_id(f'{t}_M{m}', p, True)
                rows[unit] = rng.negative_binomial(5, 5 / (5 + mu))
                groups[unit], mouse[unit], pop[unit] = t, f'M{m}', p
    return pd.DataFrame(rows, index=combos).T, groups, mouse, pop


def test_population_contrast_tests_the_interaction_paired_by_mouse():
    counts, groups, mouse, pop = _two_population_counts()
    denom = counts.sum(axis=1)
    stats = fst.run_statistics(counts, denom, groups, group_order=['Spleen', 'Pancreas', 'Liver'],
                               reference='Spleen', pairing=mouse, strata=pop, population_contrasts='Tconv',
                               run_freq=True)
    assert stats['comparisons'] == ['Pancreas vs Spleen [CD8]', 'Liver vs Spleen [CD8]',
                                    'Pancreas vs Spleen [Tconv]', 'Liver vs Spleen [Tconv]',
                                    'CD8 vs Tconv [Pancreas vs Spleen]', 'CD8 vs Tconv [Liver vs Spleen]']
    for key in ('counts', 'freq'):
        res = stats[key]
        pancreas = res[res['comparison'] == 'CD8 vs Tconv [Pancreas vs Spleen]'].set_index('feature')
        liver = res[res['comparison'] == 'CD8 vs Tconv [Liver vs Spleen]'].set_index('feature')
        assert pancreas.loc['G0', 'significant'] and pancreas['significant'].sum() == 1
        assert not liver['significant'].any()                         # G1's liver effect is shared
        assert abs(liver.loc['G1', 'logFC']) < 0.6
    # the moderated LM estimate is the mean over mice of the difference of the two populations' changes
    l2 = stats['log2_values']
    diff = [(l2.loc[fa.unit_id(f'Pancreas_M{m}', 'CD8', True), 'G0'] - l2.loc[fa.unit_id(f'Spleen_M{m}', 'CD8', True), 'G0'])
            - (l2.loc[fa.unit_id(f'Pancreas_M{m}', 'Tconv', True), 'G0']
               - l2.loc[fa.unit_id(f'Spleen_M{m}', 'Tconv', True), 'G0']) for m in range(6)]
    freq = stats['freq'].set_index(['comparison', 'feature'])
    assert freq.loc[('CD8 vs Tconv [Pancreas vs Spleen]', 'G0'), 'logFC'] == pytest.approx(np.mean(diff))
    # library sizes are worked out within each population
    assert any(n.startswith('CD8: Library sizes') for n in stats['notes'])
    pair = stats['pairs'][4]
    assert len(fst.comparison_samples(stats, pair)) == 24
    values = fst.combination_values(stats, 'G0', pair)
    assert list(dict.fromkeys(values['group'])) == ['Tconv · Spleen', 'Tconv · Pancreas', 'CD8 · Spleen',
                                                     'CD8 · Pancreas']
    figs = fst.comparison_figures(stats, pair, 0.05, 1.0, 'per_comparison', False, 'CD8, Tconv', combination='G0')
    assert all(f['fig'] is not None for f in figs.values()), {k: f['error'] for k, f in figs.items()}


def test_populations_compared_within_each_sample():
    counts, groups, mouse, pop = _two_population_counts()
    denom = counts.sum(axis=1)
    sample = {u: fa.unit_parts(u)[0] for u in counts.index}
    stats = fst.run_statistics(counts, denom, dict(pop), group_order=['CD8', 'Tconv'], reference='Tconv',
                               pairing=sample, split_by=groups, strata=pop, within_strata=False,
                               clone_groups=groups, clone_reference=None)
    assert stats['comparisons'] == ['CD8 vs Tconv [Spleen]', 'CD8 vs Tconv [Pancreas]', 'CD8 vs Tconv [Liver]']
    res = stats['counts'].set_index(['comparison', 'feature'])
    assert res.loc[('CD8 vs Tconv [Pancreas]', 'G0'), 'significant']
    assert not res.loc[('CD8 vs Tconv [Spleen]', 'G0'), 'significant']


def test_marker_differences_against_controls_or_all_cells():
    rng = np.random.default_rng(3)
    units = [f'{t}_M{m}' for m in range(6) for t in ('Spleen', 'Pancreas')]
    groups = {u: u.split('_')[0] for u in units}
    combos = ['G0', 'G1', 'G2', 'G3', 'NON-Targetting 1', 'NON-Targetting 2']
    channels = ['CD44', 'CD69']
    counts = pd.DataFrame(rng.integers(40, 200, size=(len(units), len(combos))), index=units, columns=combos)
    sums = np.zeros((len(units), len(combos), len(channels)))
    for i, u in enumerate(units):
        for j, c in enumerate(combos):
            mu = np.array([0.5, 0.3]) + (np.array([0.08, 0.0]) if (c == 'G0' and groups[u] == 'Pancreas') else 0)
            sums[i, j] = rng.normal(mu, 0.05, size=(counts.iloc[i, j], 2)).sum(axis=0)
    controls = fst.control_candidates(combos)
    deltas, meta, used = fst.marker_deltas(sums, counts, channels, 'controls', controls)
    assert used == 'controls' and list(meta['cluster_id'][:2]) == ['G0', 'G0']
    mean = sums / counts.values[:, :, None]
    ctrl = sums[:, 4:].sum(axis=1) / counts.values[:, 4:].sum(axis=1)[:, None]
    assert deltas.loc[units[1], 'G0: CD44'] == pytest.approx(mean[1, 0, 0] - ctrl[1, 0])
    other = (sums[:, 5] / counts.values[:, 5, None])                 # a control is compared with the other controls
    assert deltas.loc[units[1], 'NON-Targetting 1: CD69'] == pytest.approx(mean[1, 4, 1] - other[1, 1])
    all_ref = fst.marker_deltas(sums, counts, channels, 'all')[0]
    rest = (sums.sum(axis=1)[1] - sums[1, 2]) / (counts.values[1].sum() - counts.values[1, 2])
    assert all_ref.loc[units[1], 'G2: CD44'] == pytest.approx(mean[1, 2, 0] - rest[0])
    assert fst.marker_deltas(sums, counts, channels, 'controls', [])[2] == 'all'
    few = counts.copy()
    few.loc[units[0], 'G1'] = 3
    removed = pd.DataFrame(False, index=units, columns=combos)
    removed.loc[units[2], 'G3'] = True
    masked = fst.marker_deltas(sums, few, channels, 'controls', controls, min_events=10, removed=removed)[0]
    assert np.isnan(masked.loc[units[0], 'G1: CD44']) and np.isnan(masked.loc[units[2], 'G3: CD69'])
    markers = fst.marker_statistics(deltas, meta, groups, group_order=['Spleen', 'Pancreas'])
    res = markers['results']
    assert markers['comparisons'] == ['Spleen', 'Pancreas']
    hits = res[res['significant']]
    assert list(zip(hits['comparison'], hits['feature'])) == [('Pancreas', 'G0: CD44')]
    assert 'stagewise.adj.P.Val' in res.columns
    figs = fst.marker_figures(markers, 'Pancreas', 0.05, 0.03, 'per_comparison', False, 'CD8', 'controls')
    assert all(f['fig'] is not None for f in figs.values()), {k: f['error'] for k, f in figs.items()}
    # per-sample plot: each sample's combination and reference means in every group of the level
    _d, _m, _u, means, refs = fst.marker_deltas(sums, counts, channels, 'controls', controls, return_means=True)
    markers.update(means=means, reference_means=refs)
    assert markers['sets']['Pancreas']['units'] == [u for u in units if groups[u] == 'Pancreas']
    mouse = {u: u.split('_')[1] for u in units}
    values = fst.marker_values(markers, 'G0: CD44', 'Pancreas', mouse, ['Spleen', 'Pancreas'])
    assert values['group'].tolist() == ['Spleen'] * 6 + ['Pancreas'] * 6
    row = values.set_index('sample').loc[units[1]]
    assert row['combination'] == pytest.approx(mean[1, 0, 0]) and row['reference'] == pytest.approx(ctrl[1, 0])
    assert row['delta'] == pytest.approx(deltas.loc[units[1], 'G0: CD44']) and row['pair'] == 'M0'
    figs = fst.marker_figures(markers, 'Pancreas', 0.05, 0.03, 'per_comparison', False, 'CD8', 'controls',
                              feature='G0: CD44', pairing=mouse)
    assert figs['marker_combination']['fig'] is not None, figs['marker_combination']['error']


def _tagged_marker_data(seed=4):
    """6 mice x Spleen/Pancreas, 56 triplets of 8 tags; tag T5 lowers marker
    M0 by 0.05 in every combination carrying it (spillover), and the controls
    are mostly T5 combinations; combination 7 raises M1 by 0.08 in the
    pancreas only (a real effect)."""
    import itertools
    rng = np.random.default_rng(seed)
    tags = [f'T{i}' for i in range(8)]
    triplets = list(itertools.combinations(range(8), 3))
    combos = [f'C{j}' for j in range(len(triplets))]
    controls = [c for c, t in zip(combos, triplets) if 5 in t][:3] + [c for c, t in zip(combos, triplets) if 5 not in t][-1:]
    names = {c: (f'NON-Targetting {i + 1}' if c in controls else c) for i, c in enumerate(combos)}
    combos = [names[c] for c in combos]
    tag_matrix = pd.DataFrame([[1.0 if i in t else 0.0 for i in range(8)] for t in triplets], index=combos,
                              columns=tags)
    units = [f'{t}_M{m}' for m in range(6) for t in ('Spleen', 'Pancreas')]
    groups = {u: u.split('_')[0] for u in units}
    counts = pd.DataFrame(rng.integers(40, 160, size=(len(units), len(combos))), index=units, columns=combos)
    channels = ['M0', 'M1', 'M2']
    sums = np.zeros((len(units), len(combos), len(channels)))
    for i, u in enumerate(units):
        for j, t in enumerate(triplets):
            mu = np.array([0.5, 0.3, 0.4]) + rng.normal(0, 0.004, 3)
            if 5 in t:
                mu[0] -= 0.05
            if j == 7 and groups[u] == 'Pancreas':
                mu[1] += 0.08
            sums[i, j] = rng.normal(mu, 0.05, size=(counts.iloc[i, j], 3)).sum(axis=0)
    return counts, sums, channels, groups, tag_matrix, fst.control_candidates(combos), combos


def test_tag_model_reference_removes_a_tag_spillover_shared_by_combinations():
    counts, sums, channels, groups, tag_matrix, controls, combos = _tagged_marker_data()
    hits = {}
    for reference in ('controls', 'tags'):
        d, meta, used = fst.marker_deltas(sums, counts, channels, reference, controls, tag_matrix=tag_matrix)
        assert used == reference
        res = fst.marker_statistics(d, meta, groups, group_order=['Spleen', 'Pancreas'])['results']
        hits[reference] = set(zip(res.loc[res['significant'], 'comparison'], res.loc[res['significant'], 'feature']))
    assert sum(f.endswith(': M0') for _g, f in hits['controls']) > 20          # the T5 block against T5-rich controls
    assert hits['tags'] == {('Pancreas', f'{combos[7]}: M1')}
    # without a tag matrix the tag model falls back to the controls
    assert fst.marker_deltas(sums, counts, channels, 'tags', controls)[2] == 'controls'
    # each residual is the combination against what the other combinations predict for its tags
    d, _m, _u, means, refs = fst.marker_deltas(sums, counts, channels, 'tags', controls, tag_matrix=tag_matrix,
                                               return_means=True)
    np.testing.assert_allclose((means - refs).values, d.values, equal_nan=True)


def _linear_transforms(channels):
    from honeychrome.controller_components.transform import Transform

    tr = {}
    for ch in channels:
        tr[ch] = Transform(scale_t=1.0)
        tr[ch].set_transform(id=0, limits=[0, 1])
    return tr


def _marker_inputs(spread_cells: dict, control_shift: float = 0.0, seed: int = 4):
    """_tagged_marker_data on the tag model with a spillover spread (raw
    units; about 100 raw units are 1.0 on the linear display scale) of
    *spread_cells* {(combination index or name, channel): spread};
    *control_shift* moves marker M2 of the first control combination in the
    pancreas (an artefact on the null)."""
    counts, sums, channels, groups, tag_matrix, controls, combos = _tagged_marker_data(seed)
    if control_shift:
        j = combos.index(controls[0])
        for i, u in enumerate(counts.index):
            if groups[u] == 'Pancreas':
                sums[i, j, 2] += control_shift * counts.iloc[i, j]
    tr = _linear_transforms(channels)
    spread = pd.DataFrame(0.0, index=combos, columns=channels)
    for (c, ch), v in spread_cells.items():
        spread.loc[combos[c] if isinstance(c, int) else c, ch] = v
    spill = fst.marker_spillover(spread, sums, counts, channels, tr)
    d, meta, _u = fst.marker_deltas(sums, counts, channels, 'tags', controls, tag_matrix=tag_matrix)
    return d, meta, groups, spill, controls, combos, (spread, sums, counts, channels, tr)


def test_marker_spillover_is_the_spread_on_the_display_scale():
    _d, _m, groups, spill, _c, combos, (spread, sums, counts, channels, tr) = _marker_inputs({(7, 'M1'): 20.0})
    lin = tr['M1'].xform
    level = sums[:, :, 1].sum(axis=1) / counts.values.sum(axis=1)
    expected = np.abs(lin.apply(lin.inverse(level) + 20.0) - level)
    np.testing.assert_allclose(spill[f'{combos[7]}: M1'].values, expected, rtol=1e-9)
    assert (spill[f'{combos[0]}: M1'] < 1e-12).all()


def test_spillover_k_zero_gives_the_moderated_statistics():
    d, meta, groups, spill, controls, combos, _x = _marker_inputs({(7, 'M1'): 20.0, (3, 'M0'): 5.0})
    plain = fst.marker_statistics(d, meta, groups, spillover=None, control_combos=controls)['results']
    zero = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k=0.0, control_combos=controls)
    res = zero['results']
    assert zero['spillover_k'] == {'': 0.0} and not zero['spillover_k_auto']
    for col in ('logFC', 't', 'CI.L', 'CI.R', 'SE', 'P.Value', 'adj.P.Val', 'stagewise.adj.P.Val'):
        np.testing.assert_allclose(res[col].values, plain[col].values, rtol=1e-12, equal_nan=True, err_msg=col)
    assert (res['significant'].values == plain['significant'].values).all()
    np.testing.assert_allclose(res['se_total'].values, res['se_moderated'].values, rtol=0)
    assert 'tested' not in res.columns and not res['spillover_dominant'].any()


def test_spillover_inflates_the_standard_error_of_the_feature_it_reaches():
    d, meta, groups, spill, controls, combos, _x = _marker_inputs({(7, 'M1'): 20.0})
    feature = f'{combos[7]}: M1'
    clean = fst.marker_statistics(d, meta, groups, spillover=spill * 0.0, spillover_k=1.0, control_combos=controls)
    spilt = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k=1.0, control_combos=controls)
    a = clean['results'].set_index(['comparison', 'feature']).loc[('Pancreas', feature)]
    b = spilt['results'].set_index(['comparison', 'feature']).loc[('Pancreas', feature)]
    assert a['significant'] and not b['significant']                  # effect 0.08, spillover about 0.2
    assert b['se_total'] == pytest.approx(np.hypot(b['se_moderated'], b['spillover']))
    assert b['t'] == pytest.approx(b['logFC'] / b['se_total'])
    assert b['spillover_dominant'] and b['CI.R'] - b['CI.L'] > a['CI.R'] - a['CI.L']
    figs = fst.marker_figures(spilt, 'Pancreas', 0.05, 0.03, 'per_comparison', False, 'CD8', 'tags', feature=feature)
    assert all(f['fig'] is not None for f in figs.values()), {k: f['error'] for k, f in figs.items()}


def controls_first():
    return fst.control_candidates(_tagged_marker_data()[-1])[0]


def test_auto_k_is_the_smallest_that_clears_the_controls_and_grows_with_the_artefact():
    chosen = []
    for shift in (0.0, 0.03, 0.06, 0.12):
        d, meta, groups, spill, controls, combos, _x = _marker_inputs({(controls_first(), 'M2'): 2.0},
                                                                      control_shift=shift)
        markers = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k='auto', control_combos=controls)
        k = markers['spillover_k']['']
        assert markers['spillover_k_auto'] and markers['spillover_controls'] == {'': len(controls)}
        res = markers['results']
        assert not res.loc[res['combination'].isin(controls), 'significant'].any()
        if k > 0:
            lower = fst.reapply_marker_thresholds(markers, 0.05, 0.03, 'per_comparison', False, k - 0.25)['results']
            assert lower.loc[lower['combination'].isin(controls), 'significant'].any()
        chosen.append(k)
    assert chosen[0] == 0.0 and chosen[1] > 0.0
    assert chosen == sorted(chosen) and chosen[-1] > chosen[1]


def test_auto_k_without_controls_is_one_and_says_so():
    d, meta, groups, spill, _controls, _c, _x = _marker_inputs({(7, 'M1'): 20.0})
    markers = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k='auto', control_combos=[])
    assert markers['spillover_k'] == {'': fst.SPILLOVER_K_NO_CONTROLS}
    assert any('no control combinations' in n for n in markers['notes'])


def test_spillover_k_reapplies_like_a_refit_and_reads_old_results():
    d, meta, groups, spill, controls, combos, _x = _marker_inputs({(7, 'M1'): 20.0, (controls_first(), 'M2'): 2.0},
                                                                  control_shift=0.06)
    for k in (0.5, 'auto'):
        refit = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k=k, control_combos=controls)
        base = fst.marker_statistics(d, meta, groups, spillover=spill, spillover_k=0.0, control_combos=controls)
        again = fst.reapply_marker_thresholds(base, 0.05, 0.03, 'per_comparison', False, k)
        assert again['spillover_k'] == refit['spillover_k']
        pd.testing.assert_frame_equal(again['results'], refit['results'])
    # results saved with the earlier spillover limit: a 'tested' column and the moderated SE only
    old = dict(base, results=base['results'].drop(columns=['se_moderated', 'se_total', 'k', 'spillover_dominant'])
               .assign(tested=False))
    redone = fst.reapply_marker_thresholds(old, 0.05, 0.03, 'per_comparison', False, 0.5)['results']
    expected = fst.reapply_marker_thresholds(base, 0.05, 0.03, 'per_comparison', False, 0.5)['results']
    pd.testing.assert_frame_equal(redone[sorted(redone.columns)], expected[sorted(expected.columns)])


def test_spillover_k_per_population():
    d, meta, groups, spill, controls, combos, _x = _marker_inputs({(controls_first(), 'M2'): 2.0},
                                                                  control_shift=0.06)
    units = list(d.index)
    strata = {u: 'CD8' for u in units}
    d2 = pd.concat([d, d.rename(index=lambda u: f'{u}|CD4')])
    spill2 = pd.concat([spill, (spill * 0.0).rename(index=lambda u: f'{u}|CD4')])
    groups2 = {**groups, **{f'{u}|CD4': g for u, g in groups.items()}}
    strata.update({f'{u}|CD4': 'CD4' for u in units})
    markers = fst.marker_statistics(d2, meta, groups2, strata=strata, spillover=spill2, spillover_k='auto',
                                    control_combos=controls)
    assert set(markers['spillover_k']) == {'CD8', 'CD4'}
    assert markers['spillover_k']['CD8'] > 0.0                       # the CD8 controls carry the spillover
    assert markers['spillover_k']['CD4'] == fst.SPILLOVER_K_GRID[-1]  # same artefact, no spillover to explain it
    assert any(n.startswith('CD4: control combinations still have hits') for n in markers['notes'])
    res = markers['results']
    assert set(res['population']) == {'CD8', 'CD4'} and (res.loc[res['population'] == 'CD4', 'k'] == 4.0).all()


def test_old_settings_with_a_spillover_limit_load_with_auto_k():
    out = fa.analysis_settings({'marker_spillover': True, 'marker_spillover_limit': 0.02})
    assert out['marker_spillover_k'] == 'auto' and 'marker_spillover_limit' not in out
    assert fa.analysis_settings({'marker_spillover_k': 0.75})['marker_spillover_k'] == 0.75
    assert fa.analysis_settings({'marker_spillover_k': 'x'})['marker_spillover_k'] == 'auto'
    assert fa.analysis_settings({'marker_spillover_k': -1})['marker_spillover_k'] == 'auto'
