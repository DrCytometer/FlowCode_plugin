"""
test_flowcode_gene_groups.py
----------------------------
Gene-group effects of FlowCode screens (flowcode_gene_groups.py) and the
any-change statistic of the pathway engine. No Qt.

Synthetic screen: Spleen and Gut of six mice per library, negative-binomial
counts. Library 1 carries guides G01-G24, Library 2 guides H01-H24, and both
carry NON-Targetting 1-4 (merged). Planted groups:

* Lost: G01-G04, H01, H02 all depleted 2-fold in the gut.
* Mixed: G05, G06, H03 depleted and G07, G08, H04 enriched 2.5-fold, so the
  family moves both ways and its directional score cancels within each
  library.
* Quiet: G09-G16, unchanged.
* Rest: every other guide (28 of the 48, more than half), unchanged.

    pytest tests/test_flowcode_gene_groups.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))

import flowcode_gene_groups as fgg  # noqa: E402
import flowcode_pathways as fpw  # noqa: E402
import flowcode_stats as fst  # noqa: E402

LIBRARIES = {'Library 1': [f'G{i:02d}' for i in range(1, 25)], 'Library 2': [f'H{i:02d}' for i in range(1, 25)]}
CONTROLS = [f'NON-Targetting {i}' for i in range(1, 5)]
LOST = ['G01', 'G02', 'G03', 'G04', 'H01', 'H02']
MIXED_DOWN = ['G05', 'G06', 'H03']
MIXED_UP = ['G07', 'G08', 'H04']
QUIET = [f'G{i:02d}' for i in range(9, 17)]
GUT = 'Gut vs Spleen'


def screen(seed=0, effects=True, mice=6, size=6000, alpha=0.05):
    """(counts, denominator, groups, pairing, libraries, feature_libraries)."""
    rng = np.random.default_rng(seed)
    folds = {g: 0.5 for g in LOST}
    folds.update({g: 0.4 for g in MIXED_DOWN})
    folds.update({g: 2.5 for g in MIXED_UP})
    rows, groups, pairing, libraries = {}, {}, {}, {}
    for lib, guides in LIBRARIES.items():
        features = guides + CONTROLS
        base = np.exp(rng.normal(0.0, 0.4, len(features)))
        for m in range(mice):
            mouse = f'{lib[-1]}M{m + 1}'
            w = base * np.exp(rng.normal(0.0, 0.2, len(features)))
            for tissue in ('Spleen', 'Gut'):
                change = np.array([folds.get(f, 1.0) if effects and tissue == 'Gut' else 1.0 for f in features])
                mean = size * w * change / np.sum(w * change)
                unit = f'{tissue}_{lib}_{mouse}'
                rows[unit] = dict(zip(features, rng.poisson(rng.gamma(1.0 / alpha, alpha * mean))))
                groups[unit], pairing[unit], libraries[unit] = tissue, mouse, lib
    counts = pd.DataFrame(rows).T.astype(float)
    counts = counts[[g for guides in LIBRARIES.values() for g in guides] + CONTROLS]
    feature_libraries = {g: [lib] for lib, guides in LIBRARIES.items() for g in guides}
    feature_libraries.update({c: list(LIBRARIES) for c in CONTROLS})
    return counts, counts.sum(axis=1), groups, pairing, libraries, feature_libraries


def run_stats(data):
    counts, denominator, groups, pairing, libraries, feature_libraries = data
    log = []
    stats = fst.run_statistics(counts, denominator, groups, group_order=['Spleen', 'Gut'], reference='Spleen',
                               pairing=pairing, libraries=libraries, feature_libraries=feature_libraries,
                               fit_log=log)
    return stats, log


def planted_table():
    genes = [g for guides in LIBRARIES.values() for g in guides]
    group = {g: 'Lost' for g in LOST}
    group.update({g: 'Mixed' for g in MIXED_DOWN + MIXED_UP})
    group.update({g: 'Quiet' for g in QUIET})
    return pd.DataFrame({'gene': genes, 'group': [group.get(g, 'Rest') for g in genes], 'colour': '', 'area': ''})


def write(tmp_path, text, name='groups.csv'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Group tables and matching
# ---------------------------------------------------------------------------

def test_table_columns_any_case_and_order_blank_rows_skipped(tmp_path):
    path = write(tmp_path, 'Category,Group,GENES,Colour\nx,Integrin,Itga4,#1f78b4\n'
                           'x,,Itgb7,\nx,Chemokine receptor,Ccr9,blue\nx,Integrin,ITGA4,\n')
    with pytest.raises(fgg.GroupTableError):
        fgg.read_group_table(path)
    path = write(tmp_path, 'group,gene,colour,extra\nIntegrin,Itga4,#1f78b4,a\nIntegrin,,,b\n'
                           'Chemokine receptor,Ccr9,blue,c\nIntegrin,ITGA4,,d\n')
    table = fgg.read_group_table(path)
    assert list(table['gene']) == ['Itga4', 'Ccr9']
    assert list(table['colour']) == ['#1f78b4', '']
    assert fgg.group_order(table) == ['Integrin', 'Chemokine receptor']


def test_a_gene_in_two_groups_is_refused(tmp_path):
    path = write(tmp_path, 'gene,group\nDsc1,Cadherin\nDSC1,IgSF CAMs\n')
    with pytest.raises(fgg.GroupTableError, match='Dsc1'):
        fgg.read_group_table(path)


def test_matching_by_override_name_then_mgi_symbol():
    symbols = {'sult2b1': 'Sult2b1', 'crot': 'Crot', 'acaa1a': 'Acaa1a', 'dnah10': 'Dnah10'}
    synonyms = {'gm5897': {'Sult2b1'}}
    table = pd.DataFrame({'gene': ['Sult2b1', 'Crot', 'Acaa1a'], 'group': ['Sterol', 'Fatty acid', 'Fatty acid'],
                          'colour': '', 'area': ''})
    features = ['Gm5897', 'CROT', 'Dnah10', 'Unknown1', 'NON-Targetting 1']
    out = fgg.assign_groups(features, table, symbols, synonyms, {'Dnah10': 'Acaa1a'}).set_index('feature')
    assert out.loc['Gm5897', ['gene', 'group', 'match']].tolist() == ['Sult2b1', 'Sterol', fgg.MATCH_GENE]
    assert out.loc['CROT', ['gene', 'group', 'match']].tolist() == ['Crot', 'Fatty acid', fgg.MATCH_NAME]
    assert out.loc['Dnah10', ['gene', 'group', 'match']].tolist() == ['Acaa1a', 'Fatty acid', fgg.MATCH_OVERRIDE]
    assert out.loc['Unknown1', 'match'] == fgg.MATCH_UNASSIGNED and out.loc['Unknown1', 'group'] == ''
    assert out.loc['NON-Targetting 1', 'match'] == fgg.MATCH_CONTROL


def test_group_sets_are_signed_plus_one():
    sets = fgg.group_sets(planted_table())
    assert set(sets['sign']) == {1}
    assert set(sets['set_id']) == {'group:Lost', 'group:Mixed', 'group:Quiet', 'group:Rest'}


# ---------------------------------------------------------------------------
# The any-change statistic
# ---------------------------------------------------------------------------

def test_mixed_rotation_statistic_matches_explicit_rotation():
    rng = np.random.default_rng(3)
    d, guides = 7, 9
    vectors = rng.normal(size=(guides, d))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    z = np.array([fpw.rotated_z(vectors[[i]], np.eye(d)[[0]])[0, 0] for i in range(guides)])
    weights = np.zeros((guides, 2))
    weights[:4, 0] = -1.0 / 4
    weights[4:, 1] = -1.0 / 5
    directions = fpw.random_directions(rng, 499, d)
    score, p, mixed, mixed_p = fpw.rotation_pvalues(z, vectors, weights, directions, mixed=True)
    score2, p2 = fpw.rotation_pvalues(z, vectors, weights, directions)
    assert np.allclose(score, score2) and np.allclose(p, p2)
    rotated = fpw.rotated_z(vectors, directions)
    for j, idx in enumerate((np.arange(4), np.arange(4, 9))):
        assert mixed[j] == pytest.approx(np.mean(z[idx] ** 2))
        expected = (np.sum(np.mean(rotated[:, idx] ** 2, axis=1) >= mixed[j] * (1 - 1e-12)) + 1) / 500
        assert mixed_p[j] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Synthetic screens
# ---------------------------------------------------------------------------

def test_planted_groups_directional_and_any_change():
    """The Mixed group's self-contained directional p is not tested: that
    test's null is that no member changes, and with strong opposite changes
    it can pass with a score near 0. Direction is read from the score with
    the competitive test."""
    stats, log = run_stats(screen(5))
    out = fgg.run_gene_groups(stats, log, planted_table(), rotations=1999)
    table = out['pathways'].set_index('set_key')
    lost, mixed, quiet, rest = (table.loc[f'group:{g}'] for g in ('Lost', 'Mixed', 'Quiet', 'Rest'))
    assert lost['comparison'] == GUT and lost['set_name'] == 'Lost'
    assert lost['score'] > 1.0 and lost['FDR'] < 0.05 and lost['mixed_FDR'] < 0.05
    assert lost['members'] == 6 and lost['libraries'] == 2 and lost['concordance'] == 1.0
    assert abs(mixed['score']) < 0.3 * lost['score'] and mixed['concordance'] == 0.5
    assert mixed['mixed_P.Value'] < 0.005 and mixed['mixed_FDR'] < 0.05
    assert lost['competitive_P.Value'] < 0.05 and mixed['competitive_P.Value'] > 0.05
    assert quiet['P.Value'] > 0.05 and quiet['mixed_P.Value'] > 0.05
    assert rest['members'] == 28
    libraries = out['libraries'].set_index(['set_key', 'library'])
    assert list(libraries.loc['group:Mixed', 'members']) == [4, 2]
    assert (libraries.loc['group:Lost', 'mixed_P.Value'] < 0.05).all()
    assert out['group_order'] == ['Lost', 'Mixed', 'Quiet', 'Rest']
    assert set(out['assignment']['match']) == {fgg.MATCH_NAME, fgg.MATCH_CONTROL}
    assert int(out['qc']['members'].iloc[0]) == 4


def test_unassigned_guides_are_counted_and_not_tested():
    stats, log = run_stats(screen(6))
    table = planted_table()
    table = table[table['group'] != 'Rest']
    out = fgg.run_gene_groups(stats, log, table, rotations=199)
    assert out['settings']['unassigned'] == 28
    assert any('28 combination(s) have no gene group' in n for n in out['notes'])
    assert set(out['pathways']['set_name']) == {'Lost', 'Mixed', 'Quiet'}


def test_null_screens_give_uniform_directional_and_any_change_pvalues():
    rng = np.random.default_rng(20)
    genes = [g for guides in LIBRARIES.values() for g in guides]
    directional, mixed = [], []
    for seed in range(6):
        stats, log = run_stats(screen(200 + seed, effects=False))
        order = rng.permutation(genes)
        table = pd.DataFrame({'gene': order, 'group': [f'R{i // 6}' for i in range(len(order))],
                              'colour': '', 'area': ''})
        out = fgg.run_gene_groups(stats, log, table, rotations=999)
        directional.extend(out['pathways']['P.Value'])
        mixed.extend(out['pathways']['mixed_P.Value'])
    directional, mixed = np.array(directional), np.array(mixed)
    assert directional.size == mixed.size == 48
    assert np.mean(directional < 0.05) <= 0.125 and np.mean(mixed < 0.05) <= 0.125
    assert sstats.kstest(directional, 'uniform').pvalue > 0.001
    assert sstats.kstest(mixed, 'uniform').pvalue > 0.001


# ---------------------------------------------------------------------------
# Colours, allocation, figures and tables
# ---------------------------------------------------------------------------

def test_group_colours_up_to_eight_then_area_families_then_paired():
    eight = [f'g{i}' for i in range(8)]
    assert list(fgg.group_colours(eight).values()) == list(fgg.CATEGORICAL)
    many = [f'g{i}' for i in range(10)]
    areas = {g: ('Lipid' if i < 4 else 'Transport') for i, g in enumerate(many)}
    by_area = fgg.group_colours(many, areas)
    assert [by_area[g] for g in many[:4]] == [fgg.AREA_FAMILIES[0][i] for i in (0, 1, 3, 5)]
    assert all(by_area[g] in fgg.AREA_FAMILIES[1] for g in many[4:])
    no_areas = fgg.group_colours(many, {g: '' for g in many})
    assert len(set(no_areas.values())) == 10 and not set(no_areas.values()) & set(fgg.AREA_FAMILIES[0])
    chosen = fgg.group_colours(eight, None, {'g2': '#123456', 'g3': 'blue', 'other': '#654321'})
    assert chosen['g2'] == '#123456' and chosen['g3'] == fgg.CATEGORICAL[3] and 'other' not in chosen


def test_allocation_counts_each_librarys_guides_and_pools_them():
    table = pd.DataFrame({'gene': ['Ctnnd2', 'Itga4', 'Ccr9'], 'group': ['Cadherin', 'Integrin', 'Chemokine'],
                          'colour': '', 'area': ''})
    libraries = {'Set 1': ['Ctnnd2', 'Ccr9', 'NON-Targetting 1', 'Mystery'], 'Set 3': ['Ctnnd2', 'ITGA4']}
    out = fgg.allocation(libraries, table, {}, {}).set_index('group')
    assert out.loc['Cadherin', ['guides', 'Set 1', 'Set 3']].tolist() == [2, 1, 1]
    assert out.loc['Integrin', 'guides'] == 1 and out.loc['Chemokine', 'guides'] == 1
    assert out.loc[fgg.CONTROL_SLICE, 'guides'] == 1 and out.loc[fgg.UNASSIGNED_SLICE, 'guides'] == 1
    assert list(out.index[:3]) == ['Cadherin', 'Integrin', 'Chemokine']
    fig = fgg.donut_figure(out.reset_index(), fgg.group_colours(['Cadherin', 'Integrin', 'Chemokine']))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert '6' in texts and any('2 libraries' in t for t in texts)


def test_spider_comparisons_default_to_the_reference_group():
    comparisons = ['Gut vs Spleen', 'LN vs Spleen', 'LN vs Gut']
    pairs = [('Spleen', 'Gut', None), ('Spleen', 'LN', None), ('Gut', 'LN', None)]
    assert fgg.spider_comparisons(comparisons, [], pairs, 'Spleen') == ['Gut vs Spleen', 'LN vs Spleen']
    assert fgg.spider_comparisons(comparisons, ['LN vs Gut', 'gone'], pairs, 'Spleen') == ['LN vs Gut']
    assert fgg.spider_comparisons(comparisons, [], pairs, 'Lung') == comparisons
    colours = fgg.comparison_colours(comparisons, pairs, {'Gut': '#d95f02', 'LN': '#d95f02'})
    assert colours['Gut vs Spleen'] == '#d95f02' and colours['LN vs Spleen'] != '#d95f02'
    assert len(set(colours.values())) == 3


def test_figures_tables_and_settings_lines():
    stats, log = run_stats(screen(5))
    libraries = {lib: guides + CONTROLS for lib, guides in LIBRARIES.items()}
    out = fgg.run_gene_groups(stats, log, planted_table(), rotations=199, library_guides=libraries)
    alloc = out['allocation'].set_index('group')
    assert alloc.loc['Rest', 'guides'] == 28 and alloc.loc[fgg.CONTROL_SLICE, 'guides'] == 8
    table = fgg.group_table(out, GUT)
    assert list(table['set_name']) == ['Lost', 'Mixed', 'Quiet', 'Rest'] and 'per_library' in table.columns
    assert fgg.export_groups(out, GUT, 0.05)[:2] == ['group:Lost', 'group:Mixed']
    colours = fgg.group_colours(out['group_order'])
    items = fgg.figures(out, out['allocation'], colours, {}, [GUT], {GUT: '#d95f02'}, GUT, 'group:Mixed', None,
                        0.05, False, 'CD4')
    assert set(items) == {'guide_allocation', 'gene_group_spider', 'gene_group_members'}
    assert all(item['fig'] is not None for item in items.values()), {k: v['error'] for k, v in items.items()}
    spider = items['gene_group_spider']['fig']
    assert sorted(k for _c, k in spider._spider_keys) == ['group:Lost', 'group:Mixed', 'group:Quiet', 'group:Rest']
    assert [t.get_text() for t in spider._spider_axes.texts if t.get_text()] == ['Lost', 'Mixed', 'Quiet', 'Rest']
    members = items['gene_group_members']['fig']
    assert sorted(members._member_features) == sorted(MIXED_DOWN + MIXED_UP)
    assert 'any-change FDR' in members.axes[0].get_title()
    lines = '\n'.join(fgg.settings_lines(out, {'file': 'groups.csv', 'digest': fgg.table_digest(planted_table())}))
    assert 'table groups.csv (digest ' in lines and '4 groups' in lines and '199 rotations' in lines
    assert 'competitive test: CAMERA' in lines and 'ringed points' in lines


# ---------------------------------------------------------------------------
# Built-in tables
# ---------------------------------------------------------------------------

def test_builtin_tables_are_installed_and_readable():
    installed = dict(fgg.builtin_tables())
    assert list(installed) == [name for name, _f in fgg.BUILTIN_TABLES]
    sizes = {name: (len(fgg.builtin_table(f)), fgg.builtin_table(f)['group'].nunique())
             for name, f in installed.items()}
    assert sizes == {'Homing (sets 1–3)': (157, 7), 'Treg transcription factors (OTB01)': (56, 8),
                     'Metabolism (Metab 1–4)': (186, 14)}
    metab = fgg.builtin_table('flowcode_gene_groups_metab.csv')
    assert metab['area'].ne('').all()


def test_best_builtin_picks_the_covering_table_only_above_the_minimum():
    symbols, synonyms, _meta = fpw.bundled_names()
    metab = ['Hmgcs1', 'Hmgcr', 'Ldlr', 'Gm5897', 'slc15a2', 'NON-Targetting 1']
    best = fgg.best_builtin(metab, symbols, synonyms)
    assert best[:2] == ('Metabolism (Metab 1–4)', 'flowcode_gene_groups_metab.csv') and best[2] == 1.0
    assert fgg.best_builtin(['Ccr7', 'Itga4', 'Sell'], symbols, synonyms)[0] == 'Homing (sets 1–3)'
    assert fgg.best_builtin(['Batf', 'Irf4', 'Tbx21', 'Myc'], symbols, synonyms)[0] == \
        'Treg transcription factors (OTB01)'
    half = ['Hmgcs1', 'Hmgcr', 'Mystery1', 'Mystery2']
    assert fgg.coverage(half, fgg.builtin_table('flowcode_gene_groups_metab.csv'), symbols, synonyms) == 0.5
    assert fgg.best_builtin(half, symbols, synonyms) is None
    assert fgg.best_builtin(half, symbols, synonyms, minimum=0.5)[0] == 'Metabolism (Metab 1–4)'


def test_builtin_of_tells_a_picked_table_from_an_earlier_version():
    table = fgg.builtin_table('flowcode_gene_groups_tx_otb01.csv')
    assert fgg.builtin_of('flowcode_gene_groups_tx_otb01.csv', table) == 'same'
    assert fgg.builtin_of('flowcode_gene_groups_tx_otb01.csv', table.iloc[1:]) == 'older'
    assert fgg.builtin_of('own_table.csv', table) is None
    assert fgg.builtin_of('flowcode_gene_groups_tx_otb01.csv', None) is None
