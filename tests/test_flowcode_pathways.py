"""
test_flowcode_pathways.py
-------------------------
Signed pathway analysis of FlowCode screens (flowcode_pathways.py). No Qt.

Synthetic screen: Spleen and Gut of six mice per library, negative-binomial
counts. Library 1 carries guides G01-G24, Library 2 guides H01-H24, and both
carry NON-Targetting 1-4 (merged, as in the Metab libraries). The planted
pathway has three enhancers (G01, G02, H01; sign +1) depleted 2-fold in the
gut and three inhibitors (G03, G04, H02; sign -1) enriched 2-fold: process X
promotes accumulation in the gut. Each library has as many enhancers as
inhibitors, so the pathway's unsigned version (every sign +1) cancels within
each library.

    pytest tests/test_flowcode_pathways.py
"""

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats as sstats

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))

import flowcode_pathways as fp  # noqa: E402
import flowcode_stats as fst  # noqa: E402
from honeychrome.controller_components import differential_stats as ds  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'
LIBRARIES = {'Library 1': [f'G{i:02d}' for i in range(1, 25)], 'Library 2': [f'H{i:02d}' for i in range(1, 25)]}
CONTROLS = [f'NON-Targetting {i}' for i in range(1, 5)]
ENHANCERS = ('G01', 'G02', 'H01')
INHIBITORS = ('G03', 'G04', 'H02')
PLANTED = [(g, 1) for g in ENHANCERS] + [(g, -1) for g in INHIBITORS]
GUT = 'Gut vs Spleen'


def screen(seed=0, effects=True, tissues=('Spleen', 'Gut'), mice=6, size=6000, alpha=0.05, noisy=0.0,
           fold=2.0):
    """(counts, denominator, groups, pairing, libraries, feature_libraries).
    Enhancers are depleted and inhibitors enriched *fold*-fold in the gut.
    A share *noisy* of the guides has extra gamma noise (shape 0.25) on every
    count, which the shrunk dispersions do not absorb: the quasi-dispersion
    prior then has finite degrees of freedom."""
    rng = np.random.default_rng(seed)
    rows, groups, pairing, libraries = {}, {}, {}, {}
    for lib, guides in LIBRARIES.items():
        features = guides + CONTROLS
        base = np.exp(rng.normal(0.0, 0.4, len(features)))
        extra = rng.random(len(features)) < noisy
        for m in range(mice):
            mouse = f'{lib[-1]}M{m + 1}'
            w = base * np.exp(rng.normal(0.0, 0.2, len(features)))
            for tissue in tissues:
                change = np.ones(len(features))
                if effects and tissue == 'Gut':
                    change[[features.index(g) for g in ENHANCERS if g in features]] = 1.0 / fold
                    change[[features.index(g) for g in INHIBITORS if g in features]] = fold
                mean = size * w * change / np.sum(w * change)
                mean = np.where(extra, mean * rng.gamma(0.25, 4.0, len(features)), mean)
                unit = f'{tissue}_{lib}_{mouse}'
                rows[unit] = dict(zip(features, rng.poisson(rng.gamma(1.0 / alpha, alpha * mean))))
                groups[unit], pairing[unit], libraries[unit] = tissue, mouse, lib
    counts = pd.DataFrame(rows).T.astype(float)
    counts = counts[[g for guides in LIBRARIES.values() for g in guides] + CONTROLS]
    denominator = counts.sum(axis=1)
    feature_libraries = {g: [lib] for lib, guides in LIBRARIES.items() for g in guides}
    feature_libraries.update({c: list(LIBRARIES) for c in CONTROLS})
    return counts, denominator, groups, pairing, libraries, feature_libraries


def run_stats(data, **kwargs):
    counts, denominator, groups, pairing, libraries, feature_libraries = data
    log = []
    stats = fst.run_statistics(counts, denominator, groups, group_order=['Spleen', 'Gut', 'Lung'],
                               reference='Spleen', pairing=pairing, libraries=libraries,
                               feature_libraries=feature_libraries, fit_log=log, **kwargs)
    return stats, log


def sets_table(extra=()):
    rows = [('GO:1', 'planted process', g, s, 'experimental') for g, s in PLANTED]
    rows += [('GO:2', 'planted process, unsigned', g, 1, 'IEA') for g, _s in PLANTED]
    rows += list(extra)
    return pd.DataFrame(rows, columns=['set_id', 'set_name', 'gene', 'sign', 'evidence_class'])


def matches_of(stats):
    features = list(stats['counts']['feature'].unique())
    symbols = {f.lower(): f for f in features if not f.startswith('NON')}
    return fp.match_features(features, symbols, {})


# ---------------------------------------------------------------------------
# Rebuilt fit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode, clones', [('reference', False), ('pairwise', False), ('reference', True)])
def test_rebuilt_fit_gives_the_statistics_runs_t_for_every_guide(mode, clones):
    clone_filter = {'min_count': 50, 'min_pct': 2.0, 'max_levels': 2} if clones else None
    stats, log = run_stats(screen(13, tissues=('Spleen', 'Gut', 'Lung'), alpha=0.03, noisy=0.25),
                           contrast_mode=mode, clone_filter=clone_filter)
    assert np.isfinite(stats['counts']['df.total'].astype(float)).any()
    if clones:
        assert stats['removed'].values.any()
    results = stats['counts'].set_index(['comparison', 'feature'])
    checked = 0
    for record in log:
        for fit, comparisons in fp._fits_of_record(record):
            for label, _base, other in comparisons:
                coef = fit.group_columns[other]
                for feature, part in fit.features.items():
                    row = results.loc[(label, feature)]
                    k = int(np.flatnonzero(part.columns == coef)[0])
                    X = fit.design[np.ix_(part.observed, part.design_columns)]
                    O = np.log(np.maximum(record['offset_df'].loc[fit.samples, feature].values[part.observed], 1.0))
                    reduced = ds._fit_rows(part.y[None, :], np.delete(X, k, axis=1), O[None, :],
                                           np.array([part.alpha]))[0]
                    a = np.array([part.alpha])
                    gain = ds._glm_deviance(part.y[None, :], reduced[1][None, :], a) - \
                        ds._glm_deviance(part.y[None, :], part.mu[None, :], a)
                    t = np.sign(row['logFC']) * np.sqrt(max(float(gain[0]), 0.0) / part.quasi_dispersion)
                    assert t == pytest.approx(row['t'], rel=1e-8, abs=1e-10)
                    assert part.residual_df + fit.prior_df == pytest.approx(row['df.total'], rel=1e-10)
                    checked += 1
    assert checked == len(stats['counts'].dropna(subset=['t']))


def test_a_t_that_does_not_match_the_statistics_run_is_refused():
    stats, log = run_stats(screen(2))
    stats['counts'].loc[stats['counts']['feature'] == 'G05', 't'] *= 1.01
    with pytest.raises(fp.PathwayError, match='G05'):
        fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=99)


def test_a_df_that_does_not_match_the_statistics_run_is_refused():
    stats, log = run_stats(screen(13, tissues=('Spleen', 'Gut', 'Lung'), alpha=0.03, noisy=0.25))
    assert np.isfinite(stats['counts']['df.total'].astype(float)).all()
    stats['counts'].loc[stats['counts']['feature'] == 'G05', 'df.total'] += 0.5
    with pytest.raises(fp.PathwayError, match='G05'):
        fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=99)


def test_the_wald_test_and_a_missing_record_are_refused():
    stats, log = run_stats(screen(2), counts_method='wald')
    with pytest.raises(fp.PathwayError, match='quasi-likelihood'):
        fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=99)
    stats, _log = run_stats(screen(2))
    with pytest.raises(fp.PathwayError, match='record'):
        fp.run_pathways(stats, [], sets_table(), matches_of(stats), rotations=99)


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def haar(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def test_rotation_matches_explicit_rotation_of_every_guide():
    rng = np.random.default_rng(3)
    d, n_guides, n_rot = 7, 9, 200
    raw = rng.standard_normal((n_guides, d))
    raw[:, 0] *= 3.0
    omegas = [haar(rng, d) for _ in range(n_rot)]
    rotated = np.array([[o @ v for v in raw] for o in omegas])
    t = rotated[:, :, 0] / np.sqrt(np.sum(rotated[:, :, 1:] ** 2, axis=2) / (d - 1))
    brute = fp.t_to_z(t, d - 1)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    fast = fp.rotated_z(unit, np.array([o[0] for o in omegas]))
    assert fast == pytest.approx(brute, rel=1e-9, abs=1e-9)

    signs = np.array([1, 1, -1, 1, 0, 0, -1, 0, 0], dtype=float)
    weights = (signs / np.count_nonzero(signs))[:, None]
    observed = fp.t_to_z(raw[:, 0] / np.sqrt(np.sum(raw[:, 1:] ** 2, axis=1) / (d - 1)), d - 1)
    score, p = fp.rotation_pvalues(observed, unit, weights, np.array([o[0] for o in omegas]))
    brute_scores = brute @ weights[:, 0]
    assert score[0] == pytest.approx(float(observed @ weights[:, 0]))
    assert p[0] == pytest.approx((np.sum(np.abs(brute_scores) >= abs(score[0]) * (1 - 1e-12)) + 1) / (n_rot + 1))


def test_the_unrotated_guide_gives_back_its_observed_z():
    rng = np.random.default_rng(4)
    basis = np.linalg.qr(rng.standard_normal((10, 6)))[0]
    for z in (-7.5, -2.26, -0.1, 0.0, 0.8, 3.0, 8.0):
        v = fp.guide_vector(z, rng.standard_normal(10), basis)
        assert np.linalg.norm(v) == pytest.approx(1.0)
        e1 = np.zeros((1, v.size))
        e1[0, 0] = 1.0
        assert fp.rotated_z(v[None, :], e1)[0, 0] == pytest.approx(z, abs=1e-8)


@pytest.mark.parametrize('z', [30.0, -37.0])
def test_a_guide_far_in_the_tail_keeps_a_finite_vector(z):
    rng = np.random.default_rng(11)
    for n, p in ((8, 3), (18, 8)):
        basis = np.linalg.qr(rng.standard_normal((n, n - p)))[0]
        v = fp.guide_vector(z, rng.standard_normal(n), basis)
        assert np.isfinite(v).all() and np.linalg.norm(v) == pytest.approx(1.0)
        draws = fp.rotated_z(v[None, :], fp.random_directions(rng, 20000, v.size))[:, 0]
        assert np.isfinite(draws).all() and sstats.kstest(draws, 'norm').pvalue > 0.01


@pytest.mark.parametrize('z', [0.0, 2.5, -6.0])
def test_each_guides_rotated_z_is_standard_normal(z):
    rng = np.random.default_rng(5)
    basis = np.linalg.qr(rng.standard_normal((8, 5)))[0]
    v = fp.guide_vector(z, rng.standard_normal(8), basis)
    draws = fp.rotated_z(v[None, :], fp.random_directions(rng, 20000, v.size))[:, 0]
    assert sstats.kstest(draws, 'norm').pvalue > 0.01


def test_t_to_z_keeps_the_tail_probability():
    for t, df in ((2.0, 5.0), (-3.0, 6.3), (12.0, 4.0), (1.5, np.inf)):
        z = float(fp.t_to_z(t, df))
        tail = special.ndtr(-abs(t)) if np.isinf(df) else special.stdtr(df, -abs(t))
        assert special.ndtr(-abs(z)) == pytest.approx(tail, rel=1e-9)
        assert np.sign(z) == np.sign(t)


# ---------------------------------------------------------------------------
# CAMERA and Stouffer
# ---------------------------------------------------------------------------

def limma_camera(z, directions, members, df):
    """limma's camera arithmetic (two-sample form, estimated correlation)."""
    n, m = z.size, members.size
    m2 = n - m
    delta = n / m2 * (np.mean(z[members]) - np.mean(z))
    pooled = ((n - 1) * np.var(z, ddof=1) - delta ** 2 * m * m2 / n) / (n - 2)
    u = directions[members] * np.sqrt(directions.shape[1])
    vif = max(m * np.mean(np.mean(u, axis=0) ** 2), 1.0)
    t = delta / np.sqrt(pooled * (vif / m + 1.0 / m2))
    return t, 2.0 * special.stdtr(min(df, n - 2), -abs(t))


def test_signed_camera_with_positive_signs_is_limmas_camera():
    rng = np.random.default_rng(6)
    z = rng.normal(0.0, 1.0, 40)
    z[:5] += 1.0
    directions = rng.standard_normal((40, 8))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    members = np.arange(5)
    out = fp.signed_camera(z, directions, members, np.ones(5), 8)
    t, p = limma_camera(z, directions, members, 8)
    assert out['t'] == pytest.approx(t, rel=1e-12)
    assert out['p'] == pytest.approx(p, rel=1e-12)


def test_a_negative_sign_reflects_the_member_about_the_background():
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1.0, 30)
    directions = rng.standard_normal((30, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    members = np.array([0, 1, 2, 3])
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    centre = np.mean(np.delete(z, members))
    flipped_z, flipped_dirs = z.copy(), directions.copy()
    flipped_z[[1, 3]] = 2 * centre - z[[1, 3]]
    flipped_dirs[[1, 3]] *= -1
    signed = fp.signed_camera(z, directions, members, signs, 6)
    plain = fp.signed_camera(flipped_z, flipped_dirs, members, np.ones(4), 6)
    assert signed['delta'] == pytest.approx(plain['delta'])
    assert signed['correlation'] == pytest.approx(plain['correlation'])
    se_signed = signed['delta'] / signed['t']
    se_plain = plain['delta'] / plain['t']
    assert se_signed ** 2 < se_plain ** 2          # mean sign 0: the background mean adds no variance


def test_camera_matches_limma_on_the_reference_data():
    reference = DATA / 'pathway_camera_limma.csv'
    if not reference.exists():
        pytest.skip('run tests/data/pathway_camera_limma.R once to make the limma reference')
    y = pd.read_csv(DATA / 'pathway_camera_input.csv', index_col=0)
    design = np.column_stack([np.ones(y.shape[1]), (np.arange(y.shape[1]) >= y.shape[1] // 2).astype(float)])
    expected = pd.read_csv(reference)
    genes = pd.read_csv(DATA / 'pathway_camera_limma_genes.csv')
    q = np.linalg.qr(design, mode='complete')[0][:, design.shape[1]:]
    residual = (q.T @ y.values.T).T
    directions = residual / np.linalg.norm(residual, axis=1, keepdims=True)
    z = genes['z'].values
    df = y.shape[1] - design.shape[1]
    for _i, row in expected.iterrows():
        members = np.array([int(x) - 1 for x in str(row['index']).split(';')])
        out = fp.signed_camera(z, directions, members, np.ones(members.size), df)
        assert out['correlation'] == pytest.approx(row['Correlation'], rel=1e-6, abs=1e-10)
        assert out['p'] == pytest.approx(row['PValue'], rel=1e-6)
        assert ('Up' if out['delta'] > 0 else 'Down') == row['Direction']


def test_stouffer_weights_libraries_by_their_members():
    z, p = fp.stouffer(np.array([2.0, 1.0]), np.array([4, 1]))
    assert z == pytest.approx(np.sqrt(5.0))
    assert p == pytest.approx(2.0 * special.ndtr(-np.sqrt(5.0)))
    assert fp.stouffer(np.array([-1.7, np.nan]), np.array([3, 2]))[0] == pytest.approx(-1.7)
    assert fp.signed_z_of_p(0.05, -0.3) == pytest.approx(-1.959963984540054)


# ---------------------------------------------------------------------------
# Gene sets and names
# ---------------------------------------------------------------------------

def test_names_match_by_symbol_then_unique_synonym():
    symbols = {'sult2b1': 'Sult2b1', 'slc15a2': 'Slc15a2', 'acads': 'Acads', 'crot': 'Crot'}
    synonyms = {'gm5897': {'Sult2b1'}, 'shared': {'Acads', 'Crot'}}
    table = fp.match_features(['slc15a2', 'ACADS', 'Gm5897', 'Shared', 'Mystery', 'NON-Targetting 2', 'Odd'],
                              symbols, synonyms, overrides={'Odd': 'Crot'})
    got = dict(zip(table['feature'], zip(table['gene'], table['match'])))
    assert got['slc15a2'] == ('Slc15a2', 'symbol')
    assert got['ACADS'] == ('Acads', 'symbol')
    assert got['Gm5897'] == ('Sult2b1', 'synonym')
    assert got['Shared'] == ('', 'ambiguous synonym')
    assert got['Mystery'] == ('', 'no match')
    assert got['NON-Targetting 2'] == ('', 'control')
    assert got['Odd'] == ('Crot', 'override')


def test_sets_are_cut_filtered_and_identical_ones_merged():
    rows = pd.DataFrame([
        ('GO:10', 'broad process', 'A', 1, 'IEA'), ('GO:10', 'broad process', 'B', -1, 'IEA'),
        ('GO:10', 'broad process', 'C', 1, 'IEA'), ('GO:10', 'broad process', 'Untested', 1, 'IEA'),
        ('GO:10', 'broad process', 'Other', 1, 'IEA'),
        ('GO:11', 'narrow process', 'A', 1, 'IEA'), ('GO:11', 'narrow process', 'B', -1, 'IEA'),
        ('GO:11', 'narrow process', 'C', 1, 'IEA'),
        ('GO:12', 'opposite signs', 'A', 1, 'IEA'), ('GO:12', 'opposite signs', 'B', 1, 'IEA'),
        ('GO:12', 'opposite signs', 'C', 1, 'IEA'),
        ('GO:13', 'too small', 'A', 1, 'IEA'), ('GO:13', 'too small', 'B', 1, 'IEA'),
        ('GO:14', 'too large', 'A', 1, 'IEA'), ('GO:14', 'too large', 'B', 1, 'IEA'),
        ('GO:14', 'too large', 'C', 1, 'IEA'), ('GO:14', 'too large', 'D', 1, 'IEA'),
        ('GO:14', 'too large', 'E', 1, 'IEA'), ('GO:14', 'too large', 'F', 1, 'IEA'),
    ], columns=['set_id', 'set_name', 'gene', 'sign', 'evidence_class'])
    genes = {'a1': 'A', 'b1': 'B', 'c1': 'C', 'c2': 'C', 'd1': 'D', 'e1': 'E', 'f1': 'F', 'x1': 'X', 'y1': 'Y',
             'z1': 'Z', 'w1': 'W'}
    out = fp.collapse_sets(rows, genes, min_size=3, max_fraction=0.5)
    by_key = out.set_index('set_key')
    assert set(by_key.index) == {'GO:11', 'GO:12'}
    assert by_key.loc['GO:11', 'go_ids'] == ('GO:11', 'GO:10')
    assert by_key.loc['GO:11', 'go_names'] == ('narrow process', 'broad process')
    assert by_key.loc['GO:11', 'members'] == (('a1', 1), ('b1', -1), ('c1', 1), ('c2', 1))


def test_bundled_table_header_and_evidence_filter(tmp_path):
    path = tmp_path / 'sets.csv.gz'
    with gzip.open(path, 'wt') as fh:
        fh.write('# GO releases/2026-07-26; MGI GAF 2026-05-21; signed BP sets, CC BY 4.0\n')
        sets_table().to_csv(fh, index=False)
    rows, meta = fp.load_signed_sets(path, evidence='experimental')
    assert meta['go_release'] == 'releases/2026-07-26'
    assert meta['gaf_date'] == '2026-05-21'
    assert set(rows['set_id']) == {'GO:1'}
    assert int(meta['full_size']['GO:2']) == 6


# ---------------------------------------------------------------------------
# Synthetic screens
# ---------------------------------------------------------------------------

def test_planted_signed_pathway_is_found_and_its_unsigned_version_cancels():
    """The unsigned version's score, concordance and competitive test show no
    direction. Its self-contained p is not tested: that test asks whether
    any member changes, and these members change strongly both ways."""
    stats, log = run_stats(screen(8))
    out = fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=1999, min_size=3)
    table = out['pathways'].set_index('set_key')
    planted, unsigned = table.loc['GO:1'], table.loc['GO:2']
    assert planted['comparison'] == GUT
    assert planted['score'] > 1.0
    assert planted['P.Value'] < 0.005 and planted['FDR'] < 0.05
    assert planted['competitive_P.Value'] < 0.05 and planted['competitive_z'] > 0
    assert planted['concordance'] == 1.0
    assert planted['members'] == 6 and planted['libraries'] == 2
    assert abs(unsigned['score']) < 0.25 * planted['score']
    assert unsigned['competitive_P.Value'] > 0.05
    assert 0.25 <= unsigned['concordance'] <= 0.75
    libraries = out['libraries'].set_index(['set_key', 'library'])
    per_library = libraries.loc['GO:1']
    assert list(per_library['members']) == [4, 2]
    assert (per_library['score'] > 0).all() and (per_library['P.Value'] < 0.01).all()
    members = out['members'][out['members']['set_key'] == 'GO:1'].set_index('feature')
    assert (members.loc[list(ENHANCERS), 'logFC'] < 0).all()
    assert (members.loc[list(INHIBITORS), 'logFC'] > 0).all()
    assert (members['contribution'] > 0).all()


def test_figures_tables_and_settings_lines():
    stats, log = run_stats(screen(8))
    out = fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=499)
    table = fp.pathway_table(out, GUT)
    assert list(table['set_key']) == ['GO:1', 'GO:2'] and 'per_library' in table.columns
    either = [k for k, own, comp in zip(table['set_key'], table['FDR'], table['competitive_FDR'])
              if own <= 0.05 or comp <= 0.05]
    assert fp.top_set(out, GUT) == 'GO:1' and fp.export_sets(out, GUT, 0.05) == either
    assert fp.dot_plot_sets(out, 0.05)[0] == 'GO:1'
    dot = fp.dot_plot_figure(out, stats['comparisons'], 0.05)
    assert (GUT, 'GO:1') in dot._dot_keys
    members = fp.members_figure(out, GUT, 'GO:1', None, 0.05)
    assert sorted(members._member_features) == sorted(g for g, _s in PLANTED)
    assert 'planted process' in members.axes[0].get_title()
    lines = '\n'.join(fp.settings_lines(out, {}))
    assert '499 rotations' in lines and 'Score = mean of -sign x z' in lines


def test_weak_signed_effects_add_up():
    stats, log = run_stats(screen(32, fold=1.3))
    guides = stats['counts'][stats['counts']['feature'].isin([g for g, _s in PLANTED])]
    assert int((guides['P.Value'] < 0.05).sum()) <= 2
    out = fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=1999)
    table = out['pathways'].set_index('set_key')
    assert table.loc['GO:1', 'P.Value'] < 0.005 and table.loc['GO:1', 'competitive_P.Value'] < 0.01
    assert table.loc['GO:2', 'competitive_P.Value'] > 0.05


def test_non_targetting_pseudo_set_is_rotated_over_both_libraries():
    stats, log = run_stats(screen(9))
    out = fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=999)
    qc = out['qc']
    assert list(qc['library']) == ['Library 1 + Library 2']
    assert int(qc['members'].iloc[0]) == 4
    assert qc['P.Value'].iloc[0] > 0.01


def test_null_screens_give_uniform_rotation_pvalues():
    rng = np.random.default_rng(10)
    genes = [g for guides in LIBRARIES.values() for g in guides]
    pvals, competitive = [], []
    for seed in range(4):
        stats, log = run_stats(screen(100 + seed, effects=False))
        extra = []
        for k in range(40):
            chosen = rng.choice(genes, size=6, replace=False)
            extra += [(f'GO:R{k}', f'random {k}', g, int(rng.choice([-1, 1])), 'IEA') for g in chosen]
        out = fp.run_pathways(stats, log, pd.DataFrame(extra, columns=sets_table().columns),
                              matches_of(stats), rotations=999, max_fraction=1.0)
        pvals.extend(out['pathways']['P.Value'])
        competitive.extend(out['pathways']['competitive_P.Value'])
    pvals, competitive = np.array(pvals), np.array(competitive)
    assert pvals.size == 160
    assert 0.01 <= np.mean(pvals < 0.05) <= 0.10
    assert np.mean(competitive < 0.05) <= 0.10
    assert sstats.kstest(pvals, 'uniform').pvalue > 0.001


def test_population_contrasts_are_tested_from_their_own_fit():
    counts, denominator, groups, pairing, libraries, feature_libraries = screen(11)
    rows, strata = {}, {}
    rng = np.random.default_rng(12)
    for unit in counts.index:
        for pop in ('Tconv', 'Treg'):
            key = f'{unit}\x1f{pop}'
            values = counts.loc[unit].copy()
            if pop == 'Treg':
                values = values.where(values.isna(), rng.poisson(np.maximum(values.fillna(0).values, 0)))
            rows[key] = values
            strata[key] = pop
    units = pd.DataFrame(rows).T
    expand = {f'{u}\x1f{p}': v for u, v in groups.items() for p in ('Tconv', 'Treg')}
    log = []
    stats = fst.run_statistics(units, units.sum(axis=1), expand, group_order=['Spleen', 'Gut'],
                               reference='Spleen',
                               pairing={k: pairing[k.split('\x1f')[0]] for k in units.index},
                               strata=strata, population_contrasts='Tconv',
                               libraries={k: libraries[k.split('\x1f')[0]] for k in units.index},
                               feature_libraries=feature_libraries, fit_log=log)
    out = fp.run_pathways(stats, log, sets_table(), matches_of(stats), rotations=499)
    labels = set(out['pathways']['comparison'])
    contrast = fst.comparison_label('Tconv', 'Treg', 'Gut vs Spleen')
    assert contrast in labels
    assert f'{GUT} [Tconv]' in labels and f'{GUT} [Treg]' in labels
    planted = out['pathways'].set_index(['comparison', 'set_key'])
    assert planted.loc[(f'{GUT} [Tconv]', 'GO:1'), 'P.Value'] < 0.01
