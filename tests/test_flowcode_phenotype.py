"""
test_flowcode_phenotype.py
--------------------------
The FlowCode phenotype module (flowcode_phenotype.py): definition parsing,
resolution against a panel, the positivity score, membership and soft
counts, on the synthetic FlowCode panel and events with known truth.

    pytest tests/test_flowcode_phenotype.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flowcode_phenotype as fp  # noqa: E402
import flowcode_synthetic as syn  # noqa: E402

FLUORS = syn.FLUORS
F = len(FLUORS)
SD = np.array([30.0, 40.0, 25.0, 35.0, 50.0, 20.0])


def definition(positive, negative=(), cell_type='Test', species='mouse'):
    return fp.Definition(cell_type=cell_type, species=species, positive=tuple(positive), negative=tuple(negative))


def panel(spectral_model=None):
    model = spectral_model if spectral_model is not None else syn.process()['spectral_model']
    return fp.panel_from_spectral_model(model, FLUORS)


def model(spread=None):
    null_var = np.vstack([SD ** 2, np.zeros(F), np.zeros(F)])
    return fp.NegativeModel(tuple(FLUORS), null_var, spread)


# -- definitions --------------------------------------------------------------

def test_definitions_are_read_and_canonicalised(tmp_path):
    path = tmp_path / 'definitions.csv'
    path.write_text(
        'cell_type,species,positive_markers,negative_markers,notes\n'
        'Treg,mouse,CD3e; Foxp3 ;CD4,Cd8a;CD19,first\n'
        'Empty,mouse,,,\n')
    defs = fp.read_definitions(path, fp.SOURCE_BUNDLED)
    assert [d.cell_type for d in defs] == ['Treg']
    assert defs[0].positive == ('CD3', 'FoxP3', 'CD4')
    assert defs[0].negative == ('CD8', 'CD19')
    assert defs[0].text() == 'CD3+ FoxP3+ CD4+ CD8- CD19-'


def test_definitions_come_from_the_cell_type_database():
    defs = fp.load_definitions()
    keys = [d.key for d in defs]
    assert len(keys) == len(set(keys)) and all(d.source == fp.SOURCE_BUNDLED for d in defs)
    assert {'CD4 Treg (mouse)', 'CD4 (mouse)', 'CD8 (mouse)'} <= set(keys)
    assert set(fp.RENAMED.values()) <= set(keys)


# -- resolution ---------------------------------------------------------------

def test_panel_maps_antigens_to_canonical_markers():
    p = panel()
    assert p.markers['CD4'] == ('FITC',)
    assert p.markers['V5'] == ('BV510',)
    assert p.unmatched == ()
    assert p.viability == ()


def test_positive_marker_the_panel_lacks_is_not_checked():
    out = fp.resolve(definition(['CD3', 'CD4'], ['CD19']), panel())
    assert isinstance(out, fp.Resolved)
    assert out.markers == (('CD4', 'FITC', 1),)
    assert out.not_checked == ('CD3', 'CD19')


def test_a_definition_nothing_on_the_panel_identifies_is_unresolved():
    none = fp.resolve(definition(['CD3'], ['CD19']), panel())
    assert isinstance(none, fp.Unresolved) and 'none of its markers' in none.reason
    against = fp.resolve(definition(['CD3'], ['CD4']), panel())
    assert isinstance(against, fp.Unresolved) and 'only markers that count against it' in against.reason


def test_definitions_the_panel_cannot_tell_apart_are_one_phenotype():
    defs = [definition(['CD3', 'CD4'], ['HA'], cell_type='CD4 T cell'),
            definition(['CD3', 'CD4', 'TCRgd'], ['HA'], cell_type='CD4 gdT'),
            definition(['CD4', 'V5'], [], cell_type='V5 CD4'),
            definition(['CD3', 'CD4'], ['HA'], cell_type='CD4 T cell', species='human'),
            definition(['CD3'], [], cell_type='T cell')]
    groups, aliases = fp.phenotype_groups(defs, panel())
    assert list(groups) == ['CD4+ HA-: CD4 T cell / CD4 gdT (mouse)', 'V5 CD4 (mouse)', 'CD4 T cell (human)',
                            'T cell (mouse)']
    pair = groups['CD4+ HA-: CD4 T cell / CD4 gdT (mouse)']
    assert isinstance(pair, fp.Resolved)
    assert pair.markers == (('CD4', 'FITC', 1), ('HA', 'BV421', -1))
    assert pair.definition.members == ('CD4 T cell (mouse)', 'CD4 gdT (mouse)')
    assert pair.definition.separated_by == ('TCRgd',) and pair.not_checked == ('CD3', 'TCRgd')
    assert isinstance(groups['T cell (mouse)'], fp.Unresolved)
    assert aliases['CD4 gdT (mouse)'] == aliases['CD4 T cell (mouse)'] == 'CD4+ HA-: CD4 T cell / CD4 gdT (mouse)'
    assert aliases['V5 CD4 (mouse)'] == 'V5 CD4 (mouse)'


def test_renamed_definitions_carry_over_to_their_phenotype():
    defs = [definition(['CD3', 'CD4'], ['HA'], cell_type='CD4')]
    _groups, aliases = fp.phenotype_groups(defs, panel())
    assert aliases['CD4 T cell (mouse)'] == 'CD4 (mouse)'


def test_negative_marker_the_panel_lacks_is_not_checked():
    out = fp.resolve(definition(['CD4'], ['CD19', 'HA']), panel())
    assert isinstance(out, fp.Resolved)
    assert out.markers == (('CD4', 'FITC', 1), ('HA', 'BV421', -1))
    assert out.not_checked == ('CD19',)
    assert out.summary() == 'CD4+ (FITC) HA- (BV421); not checked: CD19'


def test_marker_on_two_fluorophores_is_ambiguous():
    model_ = syn.process()['spectral_model']
    model_[0] = dict(model_[0], antigen='CD4')
    out = fp.resolve(definition(['CD4']), panel(model_))
    assert isinstance(out, fp.Unresolved)
    assert out.ambiguous == (('CD4', ('BV421', 'FITC')),)


def test_viability_dye_is_appended_as_a_negative_marker():
    model_ = syn.process()['spectral_model']
    model_[1] = {'label': 'BV510', 'antigen': 'Live/Dead'}
    from flowcode_core import _ensure_bundled_plugins_on_path
    _ensure_bundled_plugins_on_path()
    import ag_core
    ag_core._VIABILITY_CACHE.clear()
    saved = ag_core._viability_lookup
    try:
        ag_core._viability_lookup = lambda channels: tuple(c for c in channels if c == 'BV510')
        p = panel(model_)
    finally:
        ag_core._viability_lookup = saved
    assert p.viability == ('BV510',)
    out = fp.resolve(definition(['CD4']), p)
    assert out.markers[-1] == ('viability', 'BV510', -1) and out.viability == 'BV510'
    without = fp.resolve(definition(['CD4']), p, viability=False)
    assert without.markers == (('CD4', 'FITC', 1),)


def test_signature_parts_change_with_the_definition():
    a = fp.resolve(definition(['CD4'], ['HA']), panel()).signature_parts()
    b = fp.resolve(definition(['CD4'], ['V5']), panel()).signature_parts()
    assert a != b and a['score'] == fp.SCORE_VERSION


# -- the score ----------------------------------------------------------------

def test_positivity_is_a_soft_threshold_in_negative_sds():
    m = model()
    z = np.array([0.0, -5.0, 10.0, 1.0, 3.0, 4.0])
    a = np.zeros((len(z), F))
    a[:, 0] = z * SD[0]
    pi = fp.positivity(a, np.zeros(len(z)), [0], m)[:, 0]
    expit = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    assert pi == pytest.approx(expit((z - fp.POSITIVE_Z) / fp.POSITIVE_WIDTH))
    assert pi[4] == pytest.approx(0.5)
    assert pi[0] < 0.003 and pi[1] < 1e-6 and pi[2] > 0.999999
    assert pi[3] == pytest.approx(0.018, abs=1e-3) and pi[5] == pytest.approx(0.881, abs=1e-3)


def test_af_abundance_widens_the_negative():
    null_var = np.vstack([SD ** 2, 2.0 * SD ** 2, np.zeros(F)])
    m = fp.NegativeModel(tuple(FLUORS), null_var)
    a = np.zeros((2, F))
    a[:, 2] = 3 * SD[2]
    pi = fp.positivity(a, np.array([0.0, 4.0]), [2], m)
    assert pi[0, 0] == pytest.approx(0.5)
    assert pi[1, 0] == pytest.approx(1.0 / (1.0 + np.exp(-(1.0 - 3.0) / 0.5)), abs=1e-6)


def test_spillover_spread_widens_the_negative_only_when_the_source_is_positive():
    spread = np.zeros((F, F))
    spread[0, 1] = 100.0
    m = model(spread)
    a = np.zeros((3, F))
    a[:, 1] = 2 * SD[1]
    a[1, 0] = 1000.0
    a[2, 0] = -1000.0
    pi = fp.positivity(a, np.zeros(3), [1], m)
    assert pi[1, 0] < pi[0, 0]
    assert pi[2, 0] == pi[0, 0]


def test_membership_is_the_product_of_the_signed_terms():
    pi = np.array([[0.9, 0.2, 0.5], [1.0, 0.0, 0.0]])
    out = fp.membership(pi, [1, -1, -1])
    assert out == pytest.approx([0.9 * 0.8 * 0.5, 1.0])


def test_soft_count_equals_the_hard_count_when_the_markers_are_decided():
    rng = np.random.default_rng(0)
    n = 2000
    a = np.zeros((n, F))
    truth_pos = rng.random(n) < 0.3
    truth_neg = rng.random(n) < 0.5
    a[:, 5] = np.where(truth_pos, 12.0, -3.0) * SD[5]
    a[:, 0] = np.where(truth_neg, 12.0, -3.0) * SD[0]
    pnn = ['Time', 'FSC-A', 'FSC-H', 'SSC-A'] + list(FLUORS) + ['AF Abundance', 'AF Index', 'FC A', 'FC B',
                                                               'FlowCode ID']
    events = np.zeros((n, len(pnn)))
    events[:, 4:4 + F] = a
    ids = rng.integers(0, 4, n)
    events[:, pnn.index('FlowCode ID')] = ids
    res = fp.resolve(definition(['CD4'], ['HA']), panel())
    scored = fp.score_events(events, pnn, res, model())
    hard = truth_pos & ~truth_neg
    assert scored['membership'] == pytest.approx(hard.astype(float), abs=1e-5)
    soft = fp.soft_counts(scored['membership'], ids, 2)
    assert soft == pytest.approx(np.bincount(ids[hard], minlength=4), abs=0.01)
    assert soft.sum() == pytest.approx(hard.sum(), abs=0.02)


def test_negative_model_from_an_af_envelope_and_from_the_noise_model():
    envelope = {'null_var': np.stack([np.vstack([SD ** 2, np.ones(F), np.zeros(F)])]), 'spread': None,
                'mean_rows': np.array([0])}
    af = SimpleNamespace(precomputed={'envelope': envelope})
    m = fp.negative_model(af, FLUORS)
    assert m.origin == 'AF envelope' and m.null_var.shape == (3, F) and m.null_var[1, 0] == 1.0
    state = SimpleNamespace(noise=SimpleNamespace(cell_null_var=SD ** 2, bead_null_var=None),
                            spread_no_af=np.eye(F))
    m2 = fp.negative_model(None, FLUORS, state)
    assert m2.origin == 'noise model' and np.array_equal(m2.null_var[0], SD ** 2) and m2.spread.shape == (F, F)
    assert fp.negative_model(None, FLUORS, None) is None
    with pytest.raises(ValueError):
        fp.negative_model(af, FLUORS[:-1])


def test_count_populations_scores_a_phenotype_inside_the_flowcode_gates():
    import flowcode_analysis as fa
    rng = np.random.default_rng(1)
    n = 3000
    a = np.full((n, F), -3.0) * SD
    positive = rng.random(n) < 0.4
    a[:, 5] = np.where(positive, 12.0, -3.0) * SD[5]
    pnn = ['Time', 'FSC-A', 'FSC-H', 'SSC-A'] + list(FLUORS) + ['AF Abundance', 'AF Index', 'FlowCode',
                                                               'FlowCode ID', 'FC A', 'FC B']
    events = np.zeros((n, len(pnn)))
    events[:, 4:4 + F] = a
    events[:, 1] = rng.uniform(0, 10, n)
    events[:, 3] = rng.uniform(0, 10, n)
    ids = rng.integers(0, 4, n)
    events[:, pnn.index('FlowCode ID')] = ids
    records = [{'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [5, 0], [5, 10], [0, 10]]}]
    inside = events[:, 1] <= 5
    res = fp.resolve(definition(['CD4'], ['HA']), panel())
    snapshot = SimpleNamespace(transforms={}, memberships=lambda *a, **k: {})
    out = fa.count_populations(events, pnn, 'S', snapshot, ['CD4 cells'], (), phenotypes={'CD4 cells': (res, model())},
                               gate_records=records)
    rec = out['CD4 cells']
    want = positive & inside
    assert rec['events'] == n and rec['in_population'] == pytest.approx(want.sum(), abs=0.05)
    assert rec['barcoded'] == pytest.approx((want & (ids >= 1) & (ids <= 2)).sum(), abs=0.05)
    assert rec['counts']['A'] == pytest.approx((want & (ids == 1)).sum(), abs=0.05)
    assert rec['untransduced'] == pytest.approx((want & (ids == 0)).sum(), abs=0.05)
    assert isinstance(rec['in_population'], float) or isinstance(rec['in_population'], int)
    counts, meta = fa.units_table({'CD4 cells': {'S': rec}}, ['CD4 cells'])
    assert counts.dtypes.iloc[0] == np.float64 or np.mod(counts.values, 1.0).sum() == 0
    with pytest.raises(ValueError):
        fa.count_populations(events, pnn, 'S', snapshot, ['Missing'], (), phenotypes={'CD4 cells': (res, model())})


def test_frequencies_run_on_fractional_counts():
    import pandas as pd
    import flowcode_stats as fst
    rng = np.random.default_rng(3)
    samples = [f'M{i} {g}' for g in ('A', 'B') for i in range(1, 5)]
    combos = [f'G{j}' for j in range(8)]
    base = rng.lognormal(0, 0.3, len(combos))
    rows = []
    for s in samples:
        w = base * (np.where(np.arange(8) == 2, 3.0, 1.0) if s.endswith('B') else 1.0)
        rows.append(rng.multinomial(2000, w / w.sum()) * rng.uniform(0.6, 0.95, len(combos)))
    counts = pd.DataFrame(np.array(rows), index=samples, columns=combos)
    assert np.mod(counts.values, 1.0).any()
    groups = {s: s[-1] for s in samples}
    stats = fst.run_statistics(counts, counts.sum(axis=1), groups, group_order=['A', 'B'], reference='A',
                               run_counts=False, run_freq=True)
    assert stats['counts'] is None and stats['freq'] is not None
    assert stats['comparisons'] == ['B vs A'] and len(stats['freq']) == len(combos)
    top = stats['freq'].sort_values('P.Value').iloc[0]
    assert top['feature'] == 'G2' and top['logFC'] > 1


def test_congenic_split_makes_two_variants_when_both_markers_are_in_the_panel():
    model_ = syn.process()['spectral_model']
    model_[1] = dict(model_[1], antigen='CD45.1')
    model_[2] = dict(model_[2], antigen='CD45.2')
    p = panel(model_)
    res = fp.resolve(definition(['CD4'], ['HA'], cell_type='CD4 T cell'), p)
    variants = fp.split_congenic(res, p)
    assert list(variants) == ['CD4 T cell CD45.1', 'CD4 T cell CD45.2']
    one = variants['CD4 T cell CD45.1']
    assert one.markers == (('CD4', 'FITC', 1), ('HA', 'BV421', -1), ('CD45.1', 'BV510', 1), ('CD45.2', 'PE', -1))
    assert one.definition.positive == ('CD4', 'CD45.1') and one.definition.negative == ('HA', 'CD45.2')
    two = variants['CD4 T cell CD45.2']
    assert two.markers[-2:] == (('CD45.2', 'PE', 1), ('CD45.1', 'BV510', -1))
    assert one.signature_parts() != two.signature_parts() != res.signature_parts()
    assert fp.split_congenic(res, panel()) == {}
    already = fp.resolve(definition(['CD4', 'CD45.2'], ['CD45.1']), p)
    assert fp.split_congenic(already, p) == {}


def test_population_mask_is_the_half_membership_inside_the_flowcode_gates():
    n = 500
    a = np.full((n, F), -3.0) * SD
    a[:250, 5] = 12 * SD[5]
    pnn = ['Time', 'FSC-A', 'FSC-H', 'SSC-A'] + list(FLUORS) + ['AF Abundance', 'AF Index']
    events = np.zeros((n, len(pnn)))
    events[:, 4:4 + F] = a
    events[:, 1] = np.arange(n) % 10
    res = fp.resolve(definition(['CD4'], ['HA']), panel())
    mask = fp.population_mask(events, pnn, res, model())
    assert mask.sum() == 250 and mask[:250].all()
    records = [{'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[-1, -1], [4.5, -1], [4.5, 1], [-1, 1]]}]
    gated = fp.population_mask(events, pnn, res, model(), records)
    assert gated.sum() == 125 and not gated[250:].any()


def test_library_features_takes_fractional_counts():
    import flowcode_analysis as fa
    rec = lambda a, b, c: {'counts': {'A': a, 'B': b, 'C': c}, 'events': 100, 'in_population': a + b + c + 0.3,  # noqa: E731
                           'barcoded': a + b + c, 'untransduced': 0.3, 'unresolved': 0, 'not_barcoded': 0.3}
    records = {'Raw/S1.fcs': rec(0.1 + 0.2, 0.7, 0.0), 'Raw/S2.fcs': rec(3.0, 4.0, 0.25)}
    counts, meta = fa.units_table({'P': records}, ['P'])
    assert counts.dtypes.iloc[0] == np.float64
    out, new_meta, left_out = fa.library_features(counts, meta, {'Raw/S1.fcs': 'L1', 'Raw/S2.fcs': 'L1'},
                                                  {'L1': {'A': 'g1', 'B': 'g2'}})
    assert left_out == {} and list(out.columns) == ['g1', 'g2']
    assert new_meta.loc['Raw/S1.fcs', 'out_of_library'] == 0
    assert new_meta.loc['Raw/S2.fcs', 'out_of_library'] == pytest.approx(0.25)
    assert new_meta.loc['Raw/S2.fcs', 'barcoded'] == 7
    assert fa.whole_or_float(-1.4210854715202004e-14) == 0 and fa.whole_or_float(2.5) == 2.5
