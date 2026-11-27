"""
test_flowcode_core.py
---------------------
Numerics of the FlowCode plugin (flowcode_core.py, a port of FlowCodeUnmix
0.6.0) on synthetic data with known unmixing errors: design and combos
file handling, debarcoding and the too-many and too-few-tag rescues, the error fits and
correction, backbone characterisation and its summaries (spillover
heatmap matrix, panel expression data, fit table), staged sample unmixing
and its export function, scatter gates, thresholds (including reading R
thresholds files) and persistence.

The plugin lives outside the package; its folder defaults to
user_plugins/flowcode in this repository and can be set with
FLOWCODE_PLUGIN_DIR.

Usage:
    pytest tests/test_flowcode_core.py -m numpy_only
    pytest tests/test_flowcode_core.py -k benchmark -s     # AF-only speed, kernel vs Python
"""

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FLOWCODE_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR', _REPO / 'user_plugins' / 'flowcode'))
if str(_FLOWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWCODE_DIR))

import flowcode_core as fc  # noqa: E402

import af_synthetic as syn  # noqa: E402

pytestmark = pytest.mark.numpy_only


# ---------------------------------------------------------------------------
# Synthetic FlowCode experiment
# ---------------------------------------------------------------------------

N_DET = 24
TAGS = ('HA', 'V5', 'AU1', 'S', 'C', 'Flag')
TAG_FLUORS = ('BV421', 'BV510', 'PE', 'APC', 'BUV395', 'PerCP')
OTHER_FLUORS = ('FITC', 'PE-Cy7')
FLUORS = TAG_FLUORS + OTHER_FLUORS
DETECTORS = tuple(f'D{i + 1}-A' for i in range(N_DET))
THRESHOLD = 300.0


def _spectra():
    ch = np.arange(N_DET)
    centres = np.linspace(1.5, N_DET - 2.5, len(FLUORS))
    s = np.array([np.exp(-0.5 * ((ch - c) / 1.3) ** 2) + 0.05 * np.exp(-0.5 * ((ch - c - 6) / 2.0) ** 2)
                  for c in centres])
    return s / s.max(axis=1, keepdims=True)


def _af_mean():
    ch = np.arange(N_DET)
    af = np.exp(-0.5 * ((ch - 4) / 6.0) ** 2) + 0.2
    return af / af.max()


def _design(n_combos=10):
    triplets = fc.all_triplets(TAGS)[:n_combos]
    return fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), triplets)


def _true_errors(design, spectra, seed=1234):
    """Per combo: on-tags x detectors error = FRET into an untagged
    fluorophore's spectrum (in the basis span) + an out-of-span shape. Fixed
    seed, so a backbone and a sample share the same errors."""
    rng = np.random.default_rng(seed)
    basis = np.vstack([spectra, _af_mean()])
    q, _r = np.linalg.qr(basis.T)
    errors = []
    for g in range(design.n_combos):
        k = len(TAG_FLUORS) + g % len(OTHER_FLUORS)
        fret = rng.uniform(0.005, 0.03, 3)[:, None] * spectra[k]
        shape = rng.normal(0, 1, N_DET)
        shape -= q @ (q.T @ shape)
        out_of_span = rng.uniform(0.002, 0.01, 3)[:, None] * shape / np.abs(shape).max()
        errors.append(fret + out_of_span)
    return errors


def _backbone(design, n_per_combo=600, n_untransduced=800, noise=15.0, seed=0, af_scale=200.0):
    """Raw events with known per-combo errors, plus the truth."""
    rng = np.random.default_rng(seed)
    spectra = _spectra()
    af_mean = _af_mean()
    errors = _true_errors(design, spectra)
    tag_rows = spectra[:len(TAGS)]
    raws, codes, abundances = [], [], []
    for g in range(design.n_combos):
        on = design.on_tags(g)
        a = np.zeros((n_per_combo, len(TAGS)))
        a[:, on] = rng.lognormal(np.log(4000), 0.6, (n_per_combo, 3))
        raw = a @ tag_rows + a[:, on] @ errors[g]
        raws.append(raw)
        codes.append(np.full(n_per_combo, g + 1))
        abundances.append(a)
    raws.append(np.zeros((n_untransduced, N_DET)))
    codes.append(np.zeros(n_untransduced, dtype=int))
    abundances.append(np.zeros((n_untransduced, len(TAGS))))
    raw = np.vstack(raws)
    af = rng.gamma(2.0, af_scale / 2.0, len(raw))
    raw = raw + np.outer(af, af_mean) + rng.normal(0, noise, raw.shape)
    return {
        'raw': raw, 'codes': np.concatenate(codes), 'abundances': np.vstack(abundances),
        'errors': errors, 'spectra': spectra, 'af_mean': af_mean, 'af': af,
    }


def _identity_unmix(spectra, af_mean):
    """Debarcoding-space stand-in: OLS on fluorophores + one AF column."""
    basis = np.vstack([spectra, af_mean])

    def unmix(raw):
        coef = fc.ols_unmix(raw, basis)
        n = len(raw)
        return {'unmixed': coef[:, :len(spectra)], 'af_scale': coef[:, -1], 'af_idx': np.ones(n, dtype=np.int64)}

    return unmix


def _transform(x):
    return np.arcsinh(np.asarray(x, dtype=float) / 150.0)


def _characterise(design, bb, **settings):
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    return fc.characterise_backbone(
        raw=bb['raw'], debarcode_unmixed=unmix(bb['raw'])['unmixed'], design=design,
        thresholds=np.full(len(TAGS), THRESHOLD), spectra=bb['spectra'], fluor_names=FLUORS,
        detectors=DETECTORS, af_mean=bb['af_mean'], transform=_transform,
        transform_params={'magnitude_ceiling': 262144.0}, rescue_floor=np.full(len(TAGS), 150.0),
        unstained_raw=_unstained(bb['af_mean']),
        settings=fc.CharacteriseSettings(**settings),
    )


def _unstained(af_mean, n=2000, seed=9):
    rng = np.random.default_rng(seed)
    return np.outer(rng.gamma(2.0, 100.0, n), af_mean) + rng.normal(0, 15, (n, N_DET))


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------
# Design and combos file
# ---------------------------------------------------------------------------

def test_design_matches_tags_case_insensitively_and_builds_the_logical_matrix():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)),
                                     [('GATA-3', ('au1', 'S', 'c')), ('Tbet', ('HA', 'V5', 'Flag'))])
    assert design.combo_tags[0] == ('AU1', 'S', 'C')
    np.testing.assert_array_equal(design.logical, [[0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1]])
    np.testing.assert_array_equal(design.on_tags(1), [0, 1, 5])
    assert design.unresolved_code == 3


@pytest.mark.parametrize('combos, message', [
    ([('X', ('HA', 'V5', 'Myc'))], 'unknown tag'),
    ([('X', ('HA', 'V5'))], 'three different tags'),
    ([('X', ('HA', 'HA', 'V5'))], 'three different tags'),
    ([('X', ('HA', 'V5', 'S')), ('X', ('HA', 'V5', 'C'))], 'used twice'),
    ([('X', ('HA', 'V5', 'S')), ('Y', ('S', 'HA', 'v5'))], 'same tags'),
    ([], 'No FlowCode combos'),
])
def test_design_rejects_invalid_combos(combos, message):
    with pytest.raises(ValueError, match=message):
        fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), combos)


def test_design_needs_three_tags_with_their_own_fluorophores():
    with pytest.raises(ValueError, match='At least 3'):
        fc.FlowCodeDesign.build({'HA': 'PE', 'V5': 'APC'}, [])
    with pytest.raises(ValueError, match='own fluorophore'):
        fc.FlowCodeDesign.build({'HA': 'PE', 'V5': 'PE', 'S': 'APC'}, [('X', ('HA', 'V5', 'S'))])


def test_all_triplets():
    assert len(fc.all_triplets(TAGS)) == 20
    assert len(fc.all_triplets(fc.FLOWCODE_TAGS)) == 120
    name, tags = fc.all_triplets(['S', 'au1', 'C'])[0]
    assert tags == ('au1', 'C', 'S') and name == 'au1_C_S'
    assert fc.combo_key(('S', 'au1', 'C')) == fc.combo_key(('C', 'S', 'AU1'))


def test_combinations_csv_round_trip_and_errors(tmp_path):
    path = tmp_path / 'combinations.csv'
    path.write_text('﻿Id,Procode.tag1,Procode.tag2,Procode.tag3\n'
                    'GATA-3,AU1,S,C\n\n  Tbet , HA ,V5,Flag\n', encoding='utf-8')
    combos = fc.read_combinations_csv(path)
    assert combos == [('GATA-3', ('AU1', 'S', 'C')), ('Tbet', ('HA', 'V5', 'Flag'))]

    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), combos)
    out = tmp_path / 'out.csv'
    fc.write_combinations_csv(out, design)
    assert fc.read_combinations_csv(out) == combos

    bad = tmp_path / 'bad.csv'
    bad.write_text('Id,a,b,c\nX,HA,V5,\n,HA,V5,S\nY,HA,V5,S\n', encoding='utf-8')
    with pytest.raises(ValueError, match='line\\(s\\) 2, 3'):
        fc.read_combinations_csv(bad)


def test_combinations_csv_without_header(tmp_path):
    path = tmp_path / 'library.csv'
    path.write_text('Atf3,AU1,Flag,V5,\nBatf,AU1,HA,Ollas,\n\nCebpa,AU1,HA,S,\n', encoding='utf-8')
    assert fc.read_combinations_csv(path) == [
        ('Atf3', ('AU1', 'Flag', 'V5')), ('Batf', ('AU1', 'HA', 'Ollas')), ('Cebpa', ('AU1', 'HA', 'S'))]

    single = tmp_path / 'single.csv'
    single.write_text('Ahr,AU1,Flag,Ollas\n', encoding='utf-8')
    assert fc.read_combinations_csv(single) == [('Ahr', ('AU1', 'Flag', 'Ollas'))]

    bad = tmp_path / 'bad_headerless.csv'
    bad.write_text('Atf3,AU1,Flag,V5\n\nBatf,AU1,HA,\n', encoding='utf-8')
    with pytest.raises(ValueError, match='line\\(s\\) 3'):
        fc.read_combinations_csv(bad)


def test_validate_combo_names():
    assert fc.validate_combo_names(['GATA-3', 'Tbet'], ['PE', 'FSC-A']) == []
    problems = fc.validate_combo_names(['A,B', 'Tbet', 'Tbet', ' ', 'Foxp3'], ['FC Foxp3'])
    assert len(problems) == 4
    assert fc.combo_channel_name('GATA-3') == 'FC GATA-3'


@pytest.mark.parametrize('antigen, tag', [
    ('AU1', 'AU1'), ('AU1 tag', 'AU1'), ('S', 'S'), ('C-tag', 'C'), ('FLAG', 'Flag'),
    ('HA Tag', 'HA'), ('VSV-g', 'VSVg'), ('OLLAS', 'Ollas'),
    ('CD4', None), ('SSC', None), ('', None), ('CD45', None),
])
def test_recognise_tag(antigen, tag):
    assert fc.recognise_tag(antigen) == tag


def test_cosine_partner_is_the_most_similar_other_row():
    s = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.2, 1.0]])
    np.testing.assert_array_equal(fc.cosine_partners(s), [1, 0, 1])


# ---------------------------------------------------------------------------
# Small statistics
# ---------------------------------------------------------------------------

def test_weighted_median_takes_the_first_value_reaching_half_the_weight():
    assert fc.weighted_median([3, 1, 2], [1, 1, 1]) == 2
    assert fc.weighted_median([1, 2], [1, 1]) == 1
    assert fc.weighted_median([5, 1, 2], [10, 1, 1]) == 5


def test_group_medians_leaves_empty_groups_nan():
    mat = np.array([[1.0, 10], [3, 30], [5, 50], [7, 70]])
    out = fc.group_medians(mat, np.array([1, 1, 3, 3]), 3)
    np.testing.assert_array_equal(out[0], [2, 20])
    assert np.all(np.isnan(out[1]))
    np.testing.assert_array_equal(out[2], [6, 60])
    assert np.all(np.isnan(fc.group_medians(mat[:0], np.array([], dtype=int), 2)))


def test_ols_unmix_matches_least_squares():
    s = _spectra()
    a = np.random.default_rng(1).normal(1000, 200, (50, len(s)))
    np.testing.assert_allclose(fc.ols_unmix(a @ s, s), a, rtol=1e-8)


# ---------------------------------------------------------------------------
# Debarcoding and rescue
# ---------------------------------------------------------------------------

def _tag_rows(design, *rows):
    out = np.zeros((len(rows), design.n_tags))
    for i, row in enumerate(rows):
        for tag, value in row.items():
            out[i, design.tags.index(tag)] = value
    return out


def test_debarcode_codes():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)),
                                     [('A', ('HA', 'V5', 'AU1')), ('B', ('S', 'C', 'Flag'))])
    thr = np.full(design.n_tags, 100.0)
    events = _tag_rows(
        design,
        {},                                                        # untransduced
        {'HA': 500},                                               # one tag
        {'HA': 500, 'V5': 500},                                    # two tags
        {'HA': 500, 'V5': 500, 'AU1': 500},                        # combo A
        {'S': 500, 'C': 500, 'Flag': 500},                         # combo B
        {'HA': 500, 'V5': 500, 'S': 500},                          # triplet, not a combo
        {'HA': 900, 'V5': 800, 'AU1': 700, 'S': 150},              # 4 tags: 600 > 2 x 50 -> A
        {'HA': 900, 'V5': 800, 'AU1': 300, 'S': 200},              # 4 tags: 200 <= 2 x 100
        {'HA': 100, 'V5': 500, 'AU1': 500, 'S': 50},               # at threshold is negative
    )
    np.testing.assert_array_equal(fc.debarcode(events, thr, design), [0, 3, 3, 1, 2, 3, 1, 3, 3])


def test_rescue_assigns_four_tag_events_to_the_nearest_combo_shape():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), [
        ('A', ('HA', 'V5', 'AU1')), ('B', ('S', 'C', 'Flag')), ('C3', ('HA', 'S', 'Flag')),
    ])
    rng = np.random.default_rng(2)
    thr = np.full(design.n_tags, 100.0)
    blocks, truth = [], []
    for g in range(3):
        a = np.zeros((300, design.n_tags))
        a[:, design.on_tags(g)] = rng.lognormal(np.log(3000), 0.2, (300, 3))
        blocks.append(a)
        truth.append(np.full(300, g + 1))
    # combo A plus a fourth tag too bright for the trim (2900 <= 2 x 1450)
    extra = np.zeros((50, design.n_tags))
    extra[:, design.on_tags(0)] = 3000.0
    extra[:, design.tags.index('C')] = 1550.0
    events = np.vstack(blocks + [extra])

    ids_before = fc.debarcode(events, thr, design)
    assert np.all(ids_before[-50:] == design.unresolved_code)

    result = fc.debarcode_rescue(events, thr, design, _transform, floor=np.full(design.n_tags, 50.0))
    np.testing.assert_array_equal(result.ids[:900], np.concatenate(truth))
    assert np.all(result.ids[-50:] == 1)
    assert result.rescued.size == 50 and result.rescue['n_rescued'] == 50
    np.testing.assert_array_equal(result.confident, np.arange(900))

    skipped = fc.debarcode_rescue(events, thr, design, _transform, floor=None)
    assert skipped.rescue is None and np.all(skipped.ids[-50:] == design.unresolved_code)


def test_rescue_rejects_events_halfway_between_two_combos():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), [
        ('A', ('HA', 'V5', 'AU1')), ('B', ('HA', 'V5', 'S')),
    ])
    rng = np.random.default_rng(3)
    blocks = []
    for g in range(2):
        a = np.zeros((300, design.n_tags))
        a[:, design.on_tags(g)] = rng.lognormal(np.log(3000), 0.1, (300, 3))
        blocks.append(a)
    ambiguous = _tag_rows(design, *[{'HA': 3000, 'V5': 3000, 'AU1': 3000, 'S': 3000}] * 20)
    result = fc.debarcode_rescue(np.vstack(blocks + [ambiguous]), np.full(design.n_tags, 100.0),
                                 design, _transform, floor=np.full(design.n_tags, 50.0))
    assert result.rescued.size == 0


def test_rescue_shifts_reference_medians_by_the_sample_offset():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), [
        ('A', ('HA', 'V5', 'AU1')), ('B', ('S', 'C', 'Flag')),
    ])
    rng = np.random.default_rng(4)
    blocks = []
    for g in range(2):
        a = np.zeros((400, design.n_tags))
        a[:, design.on_tags(g)] = rng.lognormal(np.log(3000), 0.1, (400, 3))
        blocks.append(a)
    reference = fc.group_medians(np.vstack(blocks), np.repeat([1, 2], 400), 2)
    sample = np.vstack(blocks)
    sample[:, design.tags.index('HA')] *= 2.0
    out = fc.rescue_too_many(sample, np.repeat([1, 2], 400), np.zeros(0, dtype=int), reference,
                             design, _transform, np.full(design.n_tags, 50.0), tag_shift_min_n=100)
    shift = out['diagnostics']['tag_shift']
    assert shift[design.tags.index('HA')] == pytest.approx(_transform(6000) - _transform(3000), rel=0.05)
    assert abs(shift[design.tags.index('V5')]) < 0.02
    assert out['diagnostics']['tag_support'][design.tags.index('HA')] == 400


TOOFEW_COMBOS = [('A', ('HA', 'V5', 'AU1')), ('B', ('HA', 'V5', 'S')), ('C3', ('S', 'C', 'Flag')),
                 ('D', ('AU1', 'C', 'Flag'))]


def _toofew_data(combos=TOOFEW_COMBOS, n_confident=300, n_untransduced=2000, seed=21, skip=()):
    """Tag abundances (threshold 1000, untransduced noise SD 50): untransduced
    events, then confident events of each combo not in *skip*."""
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), combos)
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(0, 50, (n_untransduced, design.n_tags))]
    for g in range(design.n_combos):
        if design.combo_names[g] in skip:
            continue
        a = rng.normal(0, 50, (n_confident, design.n_tags))
        a[:, design.on_tags(g)] = rng.lognormal(np.log(5000), 0.2, (n_confident, 3))
        blocks.append(a)
    return design, np.vstack(blocks), np.full(design.n_tags, 1000.0)


def _toofew_run(design, events, thr, settings=None, **kwargs):
    ids = fc.debarcode(events, thr, design)
    return ids, fc.rescue_too_few(events, ids, thr, design, settings or fc.TooFewSettings(), **kwargs)


def test_too_few_rescue_assigns_dimmed_events_to_their_combo():
    design, events, thr = _toofew_data()
    n = len(events)
    dim = _tag_rows(design, *[{'HA': 1500, 'V5': 1400, 'AU1': 800}] * 20,
                    *[{'S': 1200, 'C': 700, 'Flag': 700}] * 20)
    ids, out = _toofew_run(design, np.vstack([events, dim]), thr)
    assert np.all(ids[n:] == design.unresolved_code)
    np.testing.assert_array_equal(out['rescued_idx'], np.arange(n, n + 40))
    np.testing.assert_array_equal(out['assigned'], [1] * 20 + [3] * 20)
    assert out['diagnostics']['n_candidates'] == 40 and out['diagnostics']['n_rescued'] == 40
    np.testing.assert_array_equal(out['diagnostics']['rescued_per_combo'], [20, 0, 20, 0])


def test_too_few_rescue_rejects_noise_ties_and_weak_missing_tags():
    design, events, thr = _toofew_data()
    n = len(events)
    single = _tag_rows(design, *[{'S': 1200}] * 10)
    tie = _tag_rows(design, *[{'HA': 1500, 'V5': 1500, 'AU1': 700, 'S': 700}] * 10)
    weak = _tag_rows(design, *[{'HA': 1500, 'V5': 1500, 'AU1': 180}] * 10)
    _ids, out = _toofew_run(design, np.vstack([events, single, tie, weak]), thr)
    assert out['diagnostics']['n_candidates'] >= 30
    assert not np.any(out['rescued_idx'] >= n)
    loose = fc.TooFewSettings(flat_threshold=0.8, gate_frac=0.0, min_frac=0.1)
    _ids, out = _toofew_run(design, np.vstack([events, weak]), thr, loose)
    np.testing.assert_array_equal(out['rescued_idx'], np.arange(n, n + 10))
    assert np.all(out['assigned'] == 1)
    _ids, out = _toofew_run(design, np.vstack([events, weak]), thr,
                            fc.TooFewSettings(flat_threshold=0.8, gate_frac=0.0))
    assert out['rescued_idx'].size == 0


def test_too_few_rescue_skips_saturated_and_af_dominated_events():
    design, events, thr = _toofew_data()
    n = len(events)
    dim = _tag_rows(design, *[{'HA': 1500, 'V5': 1400, 'AU1': 800}] * 20)
    data = np.vstack([events, dim])
    af_scale = np.full(len(data), 100.0)
    af_scale[n:n + 10] = 5000.0
    usable = np.ones(len(data), dtype=bool)
    usable[n + 10:n + 15] = False
    _ids, out = _toofew_run(design, data, thr, af_scale=af_scale, usable=usable)
    np.testing.assert_array_equal(out['rescued_idx'], np.arange(n + 15, n + 20))
    assert out['diagnostics']['n_af_dominated'] == 10
    _ids, out = _toofew_run(design, data, thr, fc.TooFewSettings(af_ratio_max=10.0), af_scale=af_scale,
                            usable=usable)
    assert out['rescued_idx'].size == 15


def test_too_few_rescue_falls_back_to_the_reference_for_absent_combos():
    design, backbone, thr = _toofew_data(seed=22)
    bb_ids = fc.debarcode(backbone, thr, design)
    confident = np.flatnonzero((bb_ids >= 1) & (bb_ids <= design.n_combos))
    reference = fc.group_medians(backbone[confident], bb_ids[confident], design.n_combos)
    background = np.median(backbone[bb_ids == 0], axis=0)
    _d, events, _t = _toofew_data(seed=23, skip=('C3',))
    n = len(events)
    dim = _tag_rows(design, *[{'S': 1200, 'C': 700, 'Flag': 700}] * 15)
    data = np.vstack([events, dim])
    _ids, own = _toofew_run(design, data, thr)
    assert own['rescued_idx'].size == 0
    _ids, out = _toofew_run(design, data, thr, reference_median=reference, reference_background=background)
    np.testing.assert_array_equal(out['rescued_idx'], np.arange(n, n + 15))
    assert np.all(out['assigned'] == 3)
    weight = out['diagnostics']['reference_weight']
    assert weight[2] == 0.0 and weight[0] == pytest.approx(300 / 350)


def test_too_few_reference_scales_the_reference_per_tag():
    design, events, thr = _toofew_data(seed=24)
    ids = fc.debarcode(events, thr, design)
    confident = np.flatnonzero((ids >= 1) & (ids <= design.n_combos))
    untransduced = np.flatnonzero(ids == 0)
    reference = fc.group_medians(events[confident], ids[confident], design.n_combos)
    half = reference.copy()
    half[:, design.tags.index('HA')] *= 0.5
    ref = fc.too_few_reference(events, ids, design, thr, untransduced, confident, reference_median=half,
                               reference_background=np.zeros(design.n_tags), tag_shift_min_n=100)
    assert ref.tag_factor[design.tags.index('HA')] == pytest.approx(2.0, rel=0.05)
    assert ref.tag_factor[design.tags.index('V5')] == pytest.approx(1.0, rel=0.05)
    np.testing.assert_allclose(ref.span, thr - ref.background)
    assert np.all(ref.rival < 1.0)


def test_too_few_settings_from_plugin_settings():
    assert fc.TooFewSettings.from_settings({'rescue_toofew': False}) is None
    assert fc.TooFewSettings.from_settings({}) == fc.TooFewSettings()
    s = fc.TooFewSettings.from_settings({'rescue_toofew_flat_threshold': 0.8, 'rescue_toofew_min_z': 3,
                                         'rescue_toofew_min_frac': 0.5, 'rescue_toofew_af_ratio': 2,
                                         'rescue_toofew_gate_frac': 0.25, 'rescue_toofew_n0': 10})
    assert s == fc.TooFewSettings(flat_threshold=0.8, gate_frac=0.25, min_z=3.0, min_frac=0.5,
                                  af_ratio_max=2.0, n0=10.0)
    assert set(vars(fc.TooFewSettings())) and fc.CharacteriseSettings().rescue_toofew


def test_debarcode_rescue_runs_the_too_few_rescue_after_the_too_many():
    design, events, thr = _toofew_data()
    n = len(events)
    dim = _tag_rows(design, *[{'HA': 1500, 'V5': 1400, 'AU1': 800}] * 12)
    four = _tag_rows(design, *[{'HA': 5000, 'V5': 5000, 'AU1': 5000, 'C': 3100}] * 8)
    data = np.vstack([events, dim, four])
    result = fc.debarcode_rescue(data, thr, design, _transform, floor=np.full(design.n_tags, 200.0),
                                 too_few=fc.TooFewSettings())
    np.testing.assert_array_equal(result.rescued_toofew, np.arange(n, n + 12))
    assert np.all(np.isin(result.rescued, np.arange(n + 12, n + 20)))
    assert np.all(result.ids[n:n + 12] == 1)
    assert result.rescue_toofew['n_rescued'] == 12
    off = fc.debarcode_rescue(data, thr, design, _transform, floor=np.full(design.n_tags, 200.0))
    assert off.rescued_toofew.size == 0 and off.rescue_toofew is None
    assert np.all(off.ids[n:n + 12] == design.unresolved_code)


# ---------------------------------------------------------------------------
# Error fits and correction
# ---------------------------------------------------------------------------

def test_combo_features_and_slope():
    a = np.array([[2.0, 3.0, 5.0], [-1.0, 4.0, 2.0]])
    np.testing.assert_array_equal(fc.combo_features(a), [[2, 3, 5, 6, 10, 15], [-1, 4, 2, 0, 0, 8]])
    error = np.arange(12, dtype=float).reshape(6, 2)
    slope = fc.combo_error_slope(error, [2.0, 3.0, 5.0])
    # d/da0 = L0 + a1 P01 + a2 P02, and so on
    np.testing.assert_allclose(slope[0], error[0] + 3 * error[3] + 5 * error[4])
    np.testing.assert_allclose(slope[1], error[1] + 2 * error[3] + 5 * error[5])
    np.testing.assert_allclose(slope[2], error[2] + 2 * error[4] + 3 * error[5])


def test_combo_error_fit_recovers_a_convex_error_through_the_origin():
    rng = np.random.default_rng(5)
    thr = np.array([300.0, 500.0, 200.0])
    a = rng.lognormal(np.log(4000), 0.6, (2000, 3))
    true = np.vstack([rng.normal(0, 0.05, (3, N_DET)), rng.normal(0, 2e-6, (3, N_DET))])
    y = fc.combo_features(a) @ true + rng.normal(0, 1, (2000, N_DET))
    fit = fc.combo_error_fit(a, y, thr, min_cells=50)
    assert fit.shape == (6, N_DET)
    assert _rel(fit[:3], true[:3]) < 0.01 and _rel(fit[3:], true[3:]) < 0.01
    # no constant offset left at the data or at zero abundance
    resid = y - fc.combo_features(a) @ fit
    assert np.abs(resid.mean(axis=0)).max() < 0.2

    assert fc.combo_error_fit(a[:40], y[:40], thr, min_cells=50) is None
    assert fc.combo_error_fit(a, y[:, :0], thr, min_cells=50) is None
    ridged = fc.combo_error_fit(a, y, thr, min_cells=50, ridge=0.5)
    scale = fc._combo_scale(thr)[:, None]
    assert np.linalg.norm(ridged * scale) < np.linalg.norm(fit * scale)
    singular = np.column_stack([a[:, 0], a[:, 0], a[:, 1]])
    assert fc.combo_error_fit(singular, y, thr, min_cells=50) is None


def test_centred_linear_fit_of_a_convex_error_leaves_a_negative_offset():
    """Why the fit goes through the origin: a centred straight line through
    a donor x acceptor error over-subtracts at every abundance."""
    rng = np.random.default_rng(8)
    a = rng.lognormal(np.log(4000), 0.6, (3000, 3))
    y = (2e-6 * a[:, 0] * a[:, 1])[:, None] * np.ones((1, 2))
    x = a - a.mean(axis=0)
    slopes = np.linalg.lstsq(x, y - y.mean(axis=0), rcond=None)[0]
    centred_offset = np.median(y - a @ slopes, axis=0)
    fit = fc.combo_error_fit(a, y, np.full(3, 300.0), min_cells=50)
    origin_offset = np.median(y - fc.combo_features(a) @ fit, axis=0)
    assert np.all(centred_offset < -0.2 * np.median(y))
    assert np.all(np.abs(origin_offset) < 1e-6 * np.median(y))


def _pair_truth(design, rng):
    n_tags = design.n_tags
    linear = rng.normal(0, 0.03, (n_tags, n_tags, N_DET))
    product = np.zeros((n_tags, n_tags, N_DET))
    for i, j in itertools.combinations(range(n_tags), 2):
        product[i, j] = product[j, i] = rng.normal(0, 1e-6, N_DET)
    for i in range(n_tags):
        linear[i, i] = 0.0
    return fc.PairModel(linear=linear, product=product, support=np.ones((n_tags, n_tags), dtype=bool))


def test_pair_model_fit_recovers_shared_pair_terms():
    rng = np.random.default_rng(6)
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), fc.all_triplets(TAGS))
    truth = _pair_truth(design, rng)
    rows = []
    for g in range(design.n_combos):
        a = np.zeros((150, design.n_tags))
        a[:, design.on_tags(g)] = rng.lognormal(np.log(4000), 0.6, (150, 3))
        rows.append(a)
    a = np.vstack(rows)
    y = truth.predict(a, a != 0) + rng.normal(0, 1, (len(a), N_DET))
    fit = fc.pair_model_fit(a, y, np.full(design.n_tags, THRESHOLD), min_cells=50)
    off_diagonal = ~np.eye(design.n_tags, dtype=bool)
    assert fit.support[off_diagonal].all() and not fit.support.diagonal().any()
    assert _rel(fit.linear[off_diagonal], truth.linear[off_diagonal]) < 0.02
    assert _rel(fit.product[off_diagonal], truth.product[off_diagonal]) < 0.02
    np.testing.assert_allclose(fit.product, np.transpose(fit.product, (1, 0, 2)))
    assert fc.pair_model_fit(a[:10], y[:10], np.full(design.n_tags, THRESHOLD), min_cells=50) is None


def test_pair_model_predicts_triplets_pairs_singles_and_unseen_partners():
    rng = np.random.default_rng(7)
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), fc.all_triplets(TAGS))
    model = _pair_truth(design, rng)
    model.support[0, 5] = model.support[5, 0] = False
    a = np.array([[100.0, 200.0, 300.0, 0, 0, 0],
                  [100.0, 200.0, 0, 0, 0, 0],
                  [100.0, 0, 0, 0, 0, 0],
                  [100.0, 0, 0, 0, 0, 50.0],
                  [100.0, 200.0, 300.0, 0, 0, 0]])
    active = a > 0
    active[4] = False
    out = model.predict(a, active)
    lin, prod = model.linear, model.product
    np.testing.assert_allclose(out[0], 100 * (lin[0, 1] + lin[0, 2]) + 200 * (lin[1, 0] + lin[1, 2])
                               + 300 * (lin[2, 0] + lin[2, 1]) + 100 * 200 * prod[0, 1]
                               + 100 * 300 * prod[0, 2] + 200 * 300 * prod[1, 2])
    np.testing.assert_allclose(out[1], 100 * 2 * lin[0, 1] + 200 * 2 * lin[1, 0] + 100 * 200 * prod[0, 1])
    seen_0 = [j for j in range(1, 5)]
    np.testing.assert_allclose(out[2], 100 * 2 * lin[0, seen_0].mean(axis=0))
    # tags 0 and 5 were never seen together: each uses its average partner
    seen_5 = [1, 2, 3, 4]
    np.testing.assert_allclose(out[3], 100 * 2 * lin[0, seen_0].mean(axis=0)
                               + 50 * 2 * lin[5, seen_5].mean(axis=0) + 100 * 50 * prod[0, 5])
    np.testing.assert_array_equal(out[4], 0.0)


def test_apply_error_correction_modes():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)),
                                     [('A', ('HA', 'V5', 'AU1')), ('B', ('S', 'C', 'Flag'))])
    raw = np.full((4, 3), 100.0)
    tags = np.array([[10, 20, 30, 0, 0, 0],
                     [0, 0, 0, 1, 2, 3],
                     [50, 5, 0, 0, 0, 0],
                     [10, 20, 30, 0, 0, 0]], dtype=float)
    ids = np.array([1, 2, 3, 1])
    combo_error = [np.vstack([np.ones((3, 3)), np.zeros((3, 3))]), None]
    n_tags = len(TAGS)
    pair = fc.PairModel(linear=np.full((n_tags, n_tags, 3), 0.25), product=np.zeros((n_tags, n_tags, 3)),
                        support=~np.eye(n_tags, dtype=bool))
    thr = np.full(6, 8.0)

    out = fc.apply_error_correction(raw, tags, ids, design, combo_error, pair, thr)
    np.testing.assert_allclose(out['raw'][0], 100 - 60)
    np.testing.assert_allclose(out['raw'][1], 100 - 0.5 * 6)    # B uses the pairwise model on its own tags
    np.testing.assert_allclose(out['raw'][2], 100 - 0.5 * 50)   # only the above-threshold tag
    np.testing.assert_array_equal(out['applied_combo'], [True, False])
    assert out['pairwise_cells'] == 2
    np.testing.assert_array_equal(raw, 100.0)

    pairwise = fc.apply_error_correction(raw, tags, ids, design, combo_error, pair, thr, 'pairwise')
    np.testing.assert_allclose(pairwise['raw'][0], 100 - 0.5 * 60)
    assert pairwise['pairwise_cells'] == 4 and not pairwise['applied_combo'].any()
    none = fc.apply_error_correction(raw, tags, ids, design, combo_error, pair, thr, 'none')
    np.testing.assert_array_equal(none['raw'], raw)
    with pytest.raises(ValueError):
        fc.apply_error_correction(raw, tags, ids, design, combo_error, pair, thr, 'global')
    without = fc.apply_error_correction(raw, tags, ids, design, combo_error, None, thr)
    np.testing.assert_allclose(without['raw'][1:3], 100.0)


# ---------------------------------------------------------------------------
# Backbone characterisation
# ---------------------------------------------------------------------------

def _without_af(m, af_mean):
    unit = af_mean / np.linalg.norm(af_mean)
    return m - np.outer(m @ unit, unit)


def _prediction_error(result, bb, g):
    """Relative error of combo g's fitted error over its fit events, outside
    the AF direction (where each event's own AF variation is noise)."""
    spectra = result.spectra
    on = spectra.design.on_tags(g)
    idx = result.fit_idx[result.ids[result.fit_idx] == g + 1]
    a = result.fixed[np.ix_(idx, spectra.tag_columns()[on])] - spectra.tag_baseline()[on]
    predicted = fc.combo_features(a) @ spectra.combo_error[g]
    truth = bb['abundances'][np.ix_(idx, on)] @ bb['errors'][g]
    return _rel(_without_af(predicted, bb['af_mean']), _without_af(truth, bb['af_mean']))


def test_characterisation_recovers_injected_errors_and_cleans_off_target_channels():
    design = _design(10)
    bb = _backbone(design, noise=2.5)
    result = _characterise(design, bb)
    spectra = result.spectra

    np.testing.assert_array_equal(result.ids[bb['codes'] > 0], bb['codes'][bb['codes'] > 0])
    assert np.all(result.ids[bb['codes'] == 0] == 0)
    for g in range(design.n_combos):
        assert spectra.combo_error[g] is not None and spectra.combo_error[g].shape == (6, N_DET)
        assert _prediction_error(result, bb, g) < 0.02, design.combo_names[g]

    fit_rows = [row for row in spectra.diagnostics['validation'] if row['set'] == 'fit']
    assert len(fit_rows) == design.n_combos
    for row in fit_rows:
        assert row['off_rms_after'] < 0.1 * row['off_rms_before'], row
        assert row['off_spill_after'] < 0.1 * row['off_spill_before'], row

    table = spectra.diagnostics['fit']
    assert np.all(table['n_fit'] <= table['n_confident'])
    assert np.all(np.isfinite(table['kappa'])) and np.all(table['out_span'] > 0)
    assert np.all(np.isfinite(table['af_confound']))
    # the injected errors are linear
    assert np.all(table['curvature'] < 0.02)
    assert spectra.pair_model is not None and spectra.pair_model.linear.shape == (len(TAGS), len(TAGS), N_DET)
    untransduced = bb['codes'] == 0
    np.testing.assert_allclose(spectra.raw_baseline, np.median(bb['raw'][untransduced], axis=0))
    assert spectra.diagnostics['baseline_events'] == int(untransduced.sum())
    debarcoded = _identity_unmix(bb['spectra'], bb['af_mean'])(bb['raw'])['unmixed'][:, spectra.tag_columns()]
    np.testing.assert_allclose(spectra.tag_background, np.median(debarcoded[untransduced], axis=0))


def test_characterisation_error_is_unbiased_and_shrinks_with_events():
    """Recovery error is sampling noise: 4x the events (and untransduced
    events for the baseline) halves it (1/sqrt(n))."""
    design = _design(10)
    medians = []
    for n in (600, 2400):
        bb = _backbone(design, n_per_combo=n, n_untransduced=n)
        result = _characterise(design, bb)
        medians.append(np.median([_prediction_error(result, bb, g) for g in range(design.n_combos)]))
    assert medians[1] < 0.7 * medians[0]


def test_characterisation_fit_events_need_every_on_tag_above_the_fold():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    result = _characterise(design, bb, fit_min_fold=3.0)
    fixed_tags = result.fixed[:, :len(TAGS)]
    for i in result.fit_idx:
        on = design.on_tags(result.ids[i] - 1)
        assert np.all(fixed_tags[i, on] >= 3.0 * THRESHOLD)


def test_characterisation_removes_a_convex_fret_error_without_an_offset():
    """A donor x acceptor FRET error into an untagged fluorophore is removed
    at every abundance: its median after correction is near zero, where a
    centred linear fit leaves a negative offset."""
    design = _design(4)
    bb = _backbone(design, n_per_combo=1500, noise=2.5)
    g = 1
    on = design.on_tags(g)
    k = FLUORS.index('FITC')
    rows = bb['codes'] == g + 1
    a = bb['abundances'][np.ix_(rows, on)]
    fret = 1.25e-4 * a[:, 0] * a[:, 1]
    raw = bb['raw'].copy()
    raw[rows] += np.outer(fret, bb['spectra'][k])
    convex = dict(bb, raw=raw)
    result = _characterise(design, convex)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    before = unmix(raw)['unmixed'][rows, k]
    after = unmix(result.corrected_raw)['unmixed'][rows, k]
    assert np.median(before) > 1000
    assert abs(np.median(after)) < 0.01 * np.median(before)
    order = np.argsort(a.sum(axis=1))
    for part in np.array_split(order, 4):
        assert abs(np.median(after[part])) < 0.02 * np.median(before[part])
    curvature = result.spectra.diagnostics['fit']['curvature']
    assert curvature[g] > 0.05 and curvature[g] > 5 * np.delete(curvature, g).max()


def test_characterisation_keeps_an_af_shaped_error():
    """An error along the mean AF spectrum, growing with the tags, is part
    of the combo's error and is removed; the untransduced AF sets the
    baseline it is measured against."""
    design = _design(4)
    bb = _backbone(design, n_per_combo=1500, noise=2.5)
    g = 2
    rows = bb['codes'] == g + 1
    a = bb['abundances'][rows].sum(axis=1)
    raw = bb['raw'].copy()
    raw[rows] += np.outer(0.05 * a, bb['af_mean'])
    result = _characterise(design, dict(bb, raw=raw))
    af_col = len(FLUORS)
    after = fc.ols_unmix(result.corrected_raw, np.vstack([bb['spectra'], bb['af_mean']]))[:, af_col]
    untransduced = bb['codes'] == 0
    assert np.median(result.fixed[rows, af_col]) > 400 + np.median(result.fixed[untransduced, af_col])
    assert abs(np.median(after[rows]) - np.median(after[untransduced])) < 40
    assert result.spectra.diagnostics['fit']['af_excess'][g] > 0.03


def test_characterisation_keeps_off_tag_background_staining():
    """Non-specific staining that every event carries in a tag channel is
    the reference, not error: after correction, combos without that tag sit
    with the untransduced events at every abundance instead of being pulled
    towards zero as their on-tags grow."""
    design = _design(10)
    bb = _backbone(design, n_per_combo=1500, noise=2.5)
    k = FLUORS.index('PerCP')
    rng = np.random.default_rng(5)
    raw = bb['raw'] + np.outer(rng.normal(200.0, 20.0, len(bb['raw'])), bb['spectra'][k])
    result = _characterise(design, dict(bb, raw=raw))
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    before = unmix(raw)['unmixed'][:, k]
    after = unmix(result.corrected_raw)['unmixed'][:, k]
    untransduced = bb['codes'] == 0
    reference = np.median(after[untransduced])
    assert reference == pytest.approx(200.0, abs=5)
    assert result.spectra.tag_baseline()[TAGS.index('Flag')] == pytest.approx(200.0, abs=5)
    without = [g for g in range(design.n_combos) if TAGS.index('Flag') not in design.on_tags(g)]
    assert len(without) >= 5
    for g in without:
        rows = np.flatnonzero(bb['codes'] == g + 1)
        order = rows[np.argsort(bb['abundances'][rows].sum(axis=1))]
        for part in np.array_split(order, 4):
            assert abs(np.median(before[part]) - reference) < 10
            assert abs(np.median(after[part]) - reference) < 10, design.combo_names[g]


def test_apply_error_correction_measures_tags_from_the_baseline():
    design = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)),
                                     [('A', ('HA', 'V5', 'AU1')), ('B', ('S', 'C', 'Flag'))])
    raw = np.full((3, 3), 100.0)
    tags = np.array([[15, 25, 35, 5, 5, 5],
                     [5, 5, 5, 6, 7, 8],
                     [55, 10, 5, 5, 5, 5]], dtype=float)
    ids = np.array([1, 2, 3])
    combo_error = [np.vstack([np.ones((3, 3)), np.zeros((3, 3))]), None]
    n_tags = len(TAGS)
    pair = fc.PairModel(linear=np.full((n_tags, n_tags, 3), 0.25), product=np.zeros((n_tags, n_tags, 3)),
                        support=~np.eye(n_tags, dtype=bool))
    baseline = np.full(n_tags, 5.0)
    out = fc.apply_error_correction(raw, tags, ids, design, combo_error, pair, np.full(6, 8.0),
                                    tag_baseline=baseline)
    np.testing.assert_allclose(out['raw'][0], 100 - 60)
    np.testing.assert_allclose(out['raw'][1], 100 - 0.5 * 6)
    # active tags come from the abundances themselves (V5 = 10 > 8), the model from the shifted ones
    np.testing.assert_allclose(out['raw'][2], 100 - 0.5 * 50 - 0.5 * 5)


def test_characterisation_uses_the_pairwise_model_for_small_combos():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    keep = (bb['codes'] != 2) | (np.cumsum(bb['codes'] == 2) <= 30)
    small = dict(bb, raw=bb['raw'][keep], codes=bb['codes'][keep], af=bb['af'][keep],
                 abundances=bb['abundances'][keep])
    result = _characterise(design, small, min_cells_per_combo=50)
    errors = result.spectra.combo_error
    assert errors[1] is None
    assert all(errors[g] is not None for g in (0, 2, 3))

    model = result.spectra.pair_model
    corrected = fc.apply_error_correction(small['raw'], result.fixed[:, :len(TAGS)], result.ids, design,
                                          errors, model, result.spectra.thresholds)
    assert not corrected['applied_combo'][1]
    rest = ~np.isin(result.ids, [1, 3, 4])
    above = (result.fixed[:, :len(TAGS)] > result.spectra.thresholds).any(axis=1)
    valid = result.ids == 2
    assert corrected['pairwise_cells'] == int(np.sum(rest & (above | valid)))
    # combo 2's own events are corrected through its tags
    rows = np.flatnonzero(valid)
    on = design.on_tags(1)
    active = np.zeros((rows.size, len(TAGS)), dtype=bool)
    active[:, on] = True
    expected = small['raw'][rows] - model.predict(result.fixed[rows][:, :len(TAGS)], active)
    np.testing.assert_allclose(corrected['raw'][rows], expected)

    with pytest.raises(ValueError, match='No combo had'):
        _characterise(design, bb, min_cells_per_combo=10_000)


# ---------------------------------------------------------------------------
# Characterisation summaries
# ---------------------------------------------------------------------------

def _coef_map(spectra, af_mean):
    basis = np.vstack([spectra, af_mean])
    return np.linalg.solve(basis @ basis.T, basis)


def test_spillover_matrix_in_unmixed_space_recovers_the_injected_fret():
    """Each combo's injected FRET goes into one untagged fluorophore; the
    heatmap value there is 100 x the mean FRET fraction of its on-tags, to
    within 0.05 percentage points, and every other fluorophore is within
    0.05 (the AF column carries each event's AF variation)."""
    design = _design(10)
    bb = _backbone(design, noise=2.5)
    spectra = _characterise(design, bb).spectra
    m = fc.spillover_matrix(spectra, 'unmixed')
    assert m['rows'] == list(design.combo_names) and m['combo_index'] == list(range(10))
    assert m['columns'] == list(FLUORS) + ['AF']
    assert m['pairwise_combos'] == []
    u = _coef_map(bb['spectra'], bb['af_mean'])
    for g in range(design.n_combos):
        truth = 100.0 * (bb['errors'][g] @ u.T).mean(axis=0)
        np.testing.assert_allclose(m['values'][g, :-1], truth[:-1], atol=0.05)
        k = len(TAG_FLUORS) + g % len(OTHER_FLUORS)
        assert m['values'][g, k] > 0.4
        on = [FLUORS.index(design.fluors[t]) for t in design.on_tags(g)]
        assert sorted(np.flatnonzero(m['on'][g])) == sorted(on)


def test_spillover_matrix_matches_its_definition_and_lists_pairwise_combos():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    spectra.combo_error[2] = None
    u = _coef_map(spectra.spectra, spectra.af_mean)

    def slope(g):
        on = design.on_tags(g)
        return fc.combo_error_slope(spectra.combo_error[g],
                                    spectra.combo_raw_median[g, on] - spectra.tag_baseline()[on])

    m = fc.spillover_matrix(spectra, 'unmixed')
    assert m['rows'] == [design.combo_names[g] for g in (0, 1, 3)]
    assert m['pairwise_combos'] == [design.combo_names[2]]
    np.testing.assert_allclose(m['values'][2], 100.0 * (slope(3) @ u.T).mean(axis=0), rtol=1e-12)

    d = fc.spillover_matrix(spectra, 'detector')
    assert d['columns'] == list(DETECTORS) and not d['on'].any()
    tags = design.on_tags(1)
    peaks = spectra.spectra[spectra.tag_columns()[tags]].max(axis=1)
    np.testing.assert_allclose(d['values'][1], 100.0 * (slope(1) / peaks[:, None]).mean(axis=0), rtol=1e-12)
    with pytest.raises(ValueError):
        fc.spillover_matrix(spectra, 'raw')


def test_spillover_spread_matches_its_definition():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    gains = np.linspace(0.5, 2.0, len(DETECTORS))
    out = fc.spillover_spread(spectra, gains)
    assert out['rows'] == list(design.combo_names) and out['columns'] == list(spectra.fluor_names)
    u = _coef_map(spectra.spectra, spectra.af_mean)[:len(spectra.fluor_names)]
    g = 2
    on = design.on_tags(g)
    a = np.maximum(spectra.combo_raw_median[g, on] - spectra.tag_baseline()[on], 0.0)
    tag_spectra = np.maximum(spectra.spectra[spectra.tag_columns()[on]], 0.0)
    truth = np.sqrt(((u ** 2 * gains) @ tag_spectra.T) @ a)
    np.testing.assert_allclose(out['values'][g], truth, rtol=1e-12)
    np.testing.assert_allclose(fc.spillover_spread(spectra, 4.0 * gains)['values'], 2.0 * out['values'], rtol=1e-12)
    with pytest.raises(ValueError):
        fc.spillover_spread(spectra, gains[:-1])


def _linear_axes(lo=-1.0, hi=9.0):
    return {f: (_transform, (lo, hi), [(0.0, '0')]) for f in FLUORS}


def test_panel_expression_data_groups_traces_and_medians():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    result = _characterise(design, bb)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    before = unmix(bb['raw'])['unmixed']
    after = unmix(result.corrected_raw)['unmixed']
    labels = {f: f'{t} ({f})' for t, f in zip(TAGS, TAG_FLUORS)}
    data = fc.panel_expression_data(before, after, result.ids, design, FLUORS, _linear_axes(), labels,
                                    max_events=100_000, bins=50, smooth_bins=0.0)

    assert data['markers'] == [labels.get(f, f) for f in FLUORS]
    assert data['is_tag'] == [f in design.fluors for f in FLUORS]
    n_barcoded = int(np.sum((result.ids >= 1) & (result.ids <= design.n_combos)))
    assert data['n'] == {'untransduced': int(np.sum(result.ids == 0)), 'before': n_barcoded, 'after': n_barcoded}
    j = FLUORS.index('FITC')
    negative = np.flatnonzero(result.ids == 0)
    counts, _ = np.histogram(np.clip(_transform(before[negative, j]), -1.0, 9.0), bins=np.linspace(-1, 9, 51))
    np.testing.assert_allclose(data['traces']['untransduced'][j], counts / counts.max())
    assert data['medians']['untransduced'][j] == pytest.approx(float(np.median(_transform(before[negative, j]))))
    for group in ('untransduced', 'before', 'after'):
        assert all(t.shape == (50,) and t.max() == pytest.approx(1.0) for t in data['traces'][group])
    assert data['limits'][0] == (-1.0, 9.0) and data['ticks'][0] == [(0.0, '0')]

    smooth = fc.panel_expression_data(before, after, result.ids, design, FLUORS, _linear_axes(),
                                      max_events=500, bins=50, smooth_bins=2.0)
    assert smooth['n'] == {'untransduced': 500, 'before': 500, 'after': 500}
    again = fc.panel_expression_data(before, after, result.ids, design, FLUORS, _linear_axes(),
                                     max_events=500, bins=50, smooth_bins=2.0)
    np.testing.assert_array_equal(smooth['traces']['before'][2], again['traces']['before'][2])
    rough = fc.panel_expression_data(before, after, result.ids, design, FLUORS, _linear_axes(),
                                     max_events=500, bins=50, smooth_bins=0.0)
    total_variation = [np.abs(np.diff(d['traces']['before'][j])).sum() for d in (rough, smooth)]
    assert total_variation[1] < total_variation[0]

    none = fc.panel_expression_data(before, after, np.full(len(before), design.unresolved_code), design,
                                    FLUORS, _linear_axes())
    assert none['n']['before'] == 0 and all(t is None for t in none['traces']['before'])


def test_fit_table_rows_read_fresh_and_saved_diagnostics(tmp_path):
    import json
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    rows = fc.fit_table_rows(spectra)
    assert [r['Combo'] for r in rows] == list(design.combo_names)
    fit = spectra.diagnostics['fit']
    for g, row in enumerate(rows):
        assert row['Error'] == 'combo'
        assert row['Confident'] == fit['n_confident'][g] and row['Fit'] == fit['n_fit'][g]
        assert row['Kappa'] == pytest.approx(fit['kappa'][g])
        assert row['Off RMS after'] < 0.1 * row['Off RMS before']
        assert row['Held-out n'] == 0 and np.isnan(row['Held-out RMS before'])

    spectra.combo_error[1] = None
    meta = json.loads(json.dumps(fc.save_flowcode_spectra(spectra, tmp_path / 'fc.npz')))
    loaded = fc.fit_table_rows(fc.load_flowcode_spectra(meta, tmp_path / 'fc.npz'))
    assert loaded[1]['Error'] == 'pairwise'
    assert all(np.isfinite(r['Curvature']) for r in rows)
    for a, b in zip(rows, loaded):
        for key, value in a.items():
            if key == 'Error':
                continue
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(b[key]), key
            else:
                assert b[key] == pytest.approx(value), key


def test_write_thresholds_csv(tmp_path):
    import csv
    design = _design(4)
    raw = np.array([300.0, 450.5, 1200.0, 80.0, 999.0, 10.0])
    fc.write_thresholds_csv(tmp_path / 'thresholds.csv', design, raw, _transform)
    with open(tmp_path / 'thresholds.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ['Fluor', 'Tag', 'Threshold_Raw', 'Threshold_Transformed']
    assert [r[:2] for r in rows[1:]] == [[f, t] for f, t in zip(TAG_FLUORS, TAGS)]
    np.testing.assert_array_equal([float(r[2]) for r in rows[1:]], raw)
    np.testing.assert_allclose([float(r[3]) for r in rows[1:]], _transform(raw), rtol=1e-15)


def test_read_thresholds_csv_reads_the_r_format(tmp_path):
    path = tmp_path / 'thresholds.csv'
    path.write_text('"Fluor","Tag","Threshold_Raw","Threshold_Transformed"\n'
                    '"Alexa Fluor 647","AU1",4998.90611919185,104.25737233888\n'
                    '"DyLight 680","Flag",10478.9992918387,126.269813319527\n\n')
    assert fc.read_thresholds_csv(path) == [('Alexa Fluor 647', 'AU1', 4998.90611919185),
                                            ('DyLight 680', 'Flag', 10478.9992918387)]
    path.write_text('fluor,threshold_raw\nPE,12\n')
    assert fc.read_thresholds_csv(path) == [('PE', '', 12.0)]
    for text, message in (('Fluor,Tag\nPE,HA\n', 'no Threshold_Raw'),
                          ('Threshold_Raw\n12\n', 'Fluor or a Tag'),
                          ('Fluor,Threshold_Raw\nPE,high\nAPC,3\n', 'line\\(s\\) 2'),
                          ('Fluor,Threshold_Raw\n', 'no thresholds')):
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            fc.read_thresholds_csv(path)


def test_match_thresholds_by_tag_then_fluorophore():
    rows = [('BV421', 'ha', 10.0), ('PE', '', 20.0), ('APC', 'V5', 30.0), ('BV421', 'HA', 99.0),
            ('Nope', 'ZZ', 1.0)]
    matched, unmatched = fc.match_thresholds(rows, TAGS, TAG_FLUORS)
    assert matched == {'HA': 10.0, 'AU1': 20.0, 'V5': 30.0}
    assert unmatched == ['ZZ']


# ---------------------------------------------------------------------------
# Staged sample unmixing
# ---------------------------------------------------------------------------

def test_unmix_staged_corrects_fret_and_builds_flowcode_channels():
    design = _design(10)
    bb = _backbone(design)
    spectra = _characterise(design, bb).spectra
    sample = _backbone(design, n_per_combo=300, seed=11)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])

    result = fc.unmix_staged(sample['raw'], spectra, bb['spectra'], FLUORS, unmix, unmix, _transform,
                             ref_mask=np.ones(len(sample['raw']), dtype=bool))
    assert result.diagnostics['applied_combo'].all()
    assert result.diagnostics['gate_events'] == len(sample['raw'])

    plain = unmix(sample['raw'])['unmixed']
    other = [FLUORS.index(f) for f in OTHER_FLUORS]
    barcoded = sample['codes'] > 0
    before = np.abs(np.median(plain[barcoded][:, other], axis=0))
    after = np.abs(np.median(result.unmixed[barcoded][:, other], axis=0))
    assert np.all(after < 0.1 * before)

    tag_cols = spectra.tag_columns()
    for g in (0, 5):
        idx = np.flatnonzero(result.ids == g + 1)
        expected = np.median(result.unmixed[np.ix_(idx, tag_cols[design.on_tags(g)])], axis=1)
        np.testing.assert_allclose(result.flowcode[idx], expected)
        np.testing.assert_allclose(result.combo_channels[idx, g], expected)
        assert np.all(np.delete(result.combo_channels[idx], g, axis=1) == 0)
    untransduced = result.ids == 0
    assert np.all(result.flowcode[untransduced] == 0)

    summary = fc.debarcode_summary(result.ids, design)
    assert summary['total_events'] == len(sample['raw'])
    assert summary['untransduced'] == int(untransduced.sum())
    assert sum(summary['per_combo'].values()) == int(((result.ids >= 1) & (result.ids <= 10)).sum())


def test_unmix_staged_with_no_correction_returns_the_plain_unmix():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    result = fc.unmix_staged(bb['raw'], spectra, bb['spectra'], FLUORS, unmix, unmix, _transform,
                             correction='none')
    np.testing.assert_allclose(result.unmixed, unmix(bb['raw'])['unmixed'])
    with pytest.raises(ValueError):
        fc.unmix_staged(bb['raw'], spectra, bb['spectra'], FLUORS, unmix, unmix, _transform, regressor='x')


def test_unmix_staged_rescues_dimmed_events_unless_saturated():
    design = _design(10)
    bb = _backbone(design)
    spectra = _characterise(design, bb).spectra
    sample = _backbone(design, n_per_combo=300, seed=11)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    dim = 0.12 * sample['raw'][sample['codes'] == 1][:200]
    raw = np.vstack([sample['raw'], dim])
    n = len(sample['raw'])
    too_few = fc.TooFewSettings.from_settings(spectra.settings)

    result = fc.unmix_staged(raw, spectra, bb['spectra'], FLUORS, unmix, unmix, _transform, too_few=too_few)
    plain = fc.unmix_staged(raw, spectra, bb['spectra'], FLUORS, unmix, unmix, _transform)
    record = result.diagnostics['rescue_toofew']
    rescued = record['similarity']['idx'][record['similarity']['accepted']]
    assert record['n_rescued'] >= 20 and np.all(rescued >= n)
    assert np.mean(result.ids[rescued] == 1) > 0.95
    assert np.all(plain.ids[rescued] == design.unresolved_code)
    assert fc.staged_summary(result, design)['rescued_toofew'] == record['n_rescued']
    assert plain.diagnostics['rescue_toofew'] is None

    saturated = raw.copy()
    saturated[rescued, 0] = spectra.transform_params['magnitude_ceiling']
    again = fc.unmix_staged(saturated, spectra, bb['spectra'], FLUORS, unmix, unmix, _transform, too_few=too_few)
    assert not np.any(np.isin(rescued, again.diagnostics['rescue_toofew']['similarity']['idx']))


def test_staged_export_fn_debarcodes_with_the_sample_thresholds(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    design = _design(10)
    bb = _backbone(design)
    spectra = _characterise(design, bb).spectra
    sample = _backbone(design, n_per_combo=200, seed=11)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    monkeypatch.setattr(fc, 'af_only_unmix', lambda raw, *a, **k: unmix(raw))
    pnn = ('FSC-A', 'SSC-A') + DETECTORS
    raw_chunk = np.column_stack([np.full((len(sample['raw']), 2), 10.0), sample['raw']])
    gate = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 100], [0, 100]]}
    settings = {'tag_shift_min_n': 100, 'correction': 'combo', 'regressor': 'fixed'}
    af = SimpleNamespace(spectra=np.zeros((2, N_DET)))
    higher = np.asarray(spectra.thresholds, dtype=float) * 4.0

    def run(fc_spectra, thresholds=None):
        unmix_fn = fc.staged_export_fn(fc_spectra, bb['spectra'], FLUORS, lambda raw, af, _key=None: unmix(raw),
                                       [gate], pnn, settings, thresholds=thresholds)
        unmix_fn.prepare(sample['raw'], raw_chunk, af)
        return unmix_fn(sample['raw'], raw_chunk, af)

    plain = run(spectra)
    np.testing.assert_array_equal(run(spectra, spectra.thresholds)['extra'], plain['extra'])
    moved = run(spectra, higher)
    np.testing.assert_array_equal(moved['extra'], run(replace(spectra, thresholds=higher))['extra'])
    assert np.any(moved['extra'][:, 1] != plain['extra'][:, 1])


def test_staged_export_fn_returns_the_exporter_columns_and_counts(monkeypatch):
    from types import SimpleNamespace

    design = _design(10)
    bb = _backbone(design)
    spectra = _characterise(design, bb).spectra
    sample = _backbone(design, n_per_combo=200, seed=11)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    monkeypatch.setattr(fc, 'af_only_unmix', lambda raw, *a, **k: unmix(raw))
    pnn = ('FSC-A', 'SSC-A') + DETECTORS
    rng = np.random.default_rng(3)
    scatter = rng.uniform(0, 100, (len(sample['raw']), 2))
    raw_chunk = np.column_stack([scatter, sample['raw']])
    gate = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 100], [0, 100]]}
    blocks = []
    unmix_fn = fc.staged_export_fn(spectra, bb['spectra'], FLUORS, lambda raw, af, _key=None: unmix(raw), [gate], pnn,
                                   {'tag_shift_min_n': 100, 'correction': 'combo', 'regressor': 'fixed'},
                                   on_block=blocks.append)
    out = unmix_fn(sample['raw'], raw_chunk, SimpleNamespace(spectra=np.zeros((2, N_DET))))

    expected = fc.unmix_staged(sample['raw'], spectra, bb['spectra'], FLUORS, unmix, unmix,
                               fc.logicle_transform(262144.0), ref_mask=scatter[:, 0] < 50,
                               too_few=fc.TooFewSettings.from_settings(spectra.settings))
    np.testing.assert_allclose(out['unmixed'], expected.unmixed)
    assert out['extra'].shape == (len(sample['raw']), 2 + design.n_combos)
    np.testing.assert_allclose(out['extra'][:, 0], expected.flowcode)
    np.testing.assert_array_equal(out['extra'][:, 1], expected.ids)
    np.testing.assert_allclose(out['extra'][:, 2:], expected.combo_channels)
    barcoded = (expected.ids >= 1) & (expected.ids <= design.n_combos)
    np.testing.assert_array_equal(np.argmax(out['extra'][barcoded, 2:] != 0, axis=1) + 1, expected.ids[barcoded])
    assert len(blocks) == 1 and blocks[0]['total_events'] == len(sample['raw'])
    assert sum(blocks[0]['per_combo'].values()) == int(np.sum(out['extra'][:, 0] != 0))
    assert fc.gate_records_mask(raw_chunk, pnn, [gate]).sum() == int(np.sum(scatter[:, 0] <= 50))
    assert fc.gate_records_mask(raw_chunk, ('X',) + pnn[1:], [gate]) is None


def test_characterisation_id_tracks_thresholds_and_combos():
    design = _design()
    bb = _backbone(design)
    spectra = _characterise(design, bb).spectra
    cid = spectra.characterisation_id()
    assert len(cid) == 12 and cid == spectra.characterisation_id()
    from dataclasses import replace as dc_replace
    moved = dc_replace(spectra, thresholds=np.asarray(spectra.thresholds) * 1.01)
    assert moved.characterisation_id() != cid
    renamed = dc_replace(spectra, design=dc_replace(
        design, combo_names=('X' + design.combo_names[0],) + tuple(design.combo_names[1:])))
    assert renamed.characterisation_id() != cid


def test_debarcoding_without_fret_correction_leaves_raw_data_unchanged(tmp_path):
    import json
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    full = _characterise(design, bb)
    unmix = _identity_unmix(bb['spectra'], bb['af_mean'])
    plain = fc.characterise_backbone(
        raw=bb['raw'], debarcode_unmixed=unmix(bb['raw'])['unmixed'], design=design,
        thresholds=np.full(len(TAGS), THRESHOLD), spectra=bb['spectra'], fluor_names=FLUORS,
        detectors=DETECTORS, af_mean=bb['af_mean'], transform=_transform,
        transform_params={'magnitude_ceiling': 262144.0}, rescue_floor=np.full(len(TAGS), 150.0),
        unstained_raw=_unstained(bb['af_mean']), fit_error=False)
    spectra = plain.spectra
    assert not spectra.fret_corrected and full.spectra.fret_corrected
    assert all(e is None for e in spectra.combo_error) and spectra.pair_model is None
    np.testing.assert_array_equal(plain.ids, full.ids)
    np.testing.assert_allclose(spectra.combo_raw_median, full.spectra.combo_raw_median)
    np.testing.assert_array_equal(plain.corrected_raw, bb['raw'])
    assert {r['Error'] for r in fc.fit_table_rows(spectra)} == {'none'}
    assert spectra.characterisation_id() != full.spectra.characterisation_id()

    meta = json.loads(json.dumps(fc.save_flowcode_spectra(spectra, tmp_path / 'fc.npz')))
    loaded = fc.load_flowcode_spectra(meta, tmp_path / 'fc.npz')
    assert not loaded.fret_corrected
    assert loaded.characterisation_id() == spectra.characterisation_id()
    renamed = fc.rename_combos(loaded, design)
    assert not renamed.fret_corrected

    sample = _backbone(design, n_per_combo=200, seed=11)
    result = fc.unmix_staged(sample['raw'], spectra, bb['spectra'], FLUORS, unmix, unmix, _transform,
                             correction='combo')
    np.testing.assert_allclose(result.unmixed, unmix(sample['raw'])['unmixed'])
    assert result.diagnostics['correction'] == 'none'
    assert not result.diagnostics['applied_combo'].any()


def test_summaries_add_up_and_write_one_row_per_sample(tmp_path):
    import csv
    design = _design(4)
    a = fc.debarcode_summary(np.array([0, 1, 1, 2, 5]), design, rescued=1, pairwise_cells=2, rescued_toofew=2)
    b = fc.debarcode_summary(np.array([3, 3, 0]), design, rescued=0, pairwise_cells=1, rescued_toofew=1)
    total = fc.add_summaries(fc.add_summaries(None, a), b)
    assert total['total_events'] == 8 and total['untransduced'] == 2 and total['unresolved'] == 1
    assert total['rescued'] == 1 and total['pairwise_corrected'] == 3 and total['rescued_toofew'] == 3
    assert list(total['per_combo'].values()) == [2, 1, 2, 0]
    assert a['per_combo'][design.combo_names[0]] == 2
    fc.write_debarcode_summary_csv(tmp_path / 's.csv', [('One', total), ('Two', b)], design)
    with open(tmp_path / 's.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == list(fc.SUMMARY_COLUMNS) + list(design.combo_names)
    assert rows[1] == ['One', '8', '2', '1', '1', '3', '0', '3', '2', '1', '2', '0']
    assert rows[2][0] == 'Two' and rows[2][8:] == ['0', '0', '2', '0']
    fc.write_debarcode_summary_csv(tmp_path / 'w.csv', [('One', total)], design, warning='Invalid')
    with open(tmp_path / 'w.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-1] == 'Warning' and rows[1][-1] == 'Invalid' and rows[1][:-1] == ['One', '8', '2', '1', '1',
                                                                                     '3', '0', '3', '2', '1', '2',
                                                                                     '0']


def test_rename_combos_maps_by_tag_set():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    renamed = [(f'Combo {g}', tuple(reversed(t))) for g, t in enumerate(design.combo_tags)][::-1]
    current = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), renamed)
    out = fc.rename_combos(spectra, current)
    assert out.design.combo_names == ('Combo 3', 'Combo 2', 'Combo 1', 'Combo 0')
    for g_new, g_old in enumerate((3, 2, 1, 0)):
        np.testing.assert_array_equal(out.combo_error[g_new], spectra.combo_error[g_old])
        np.testing.assert_array_equal(out.combo_raw_median[g_new], spectra.combo_raw_median[g_old])
        assert set(current.combo_tags[g_new]) == set(design.combo_tags[g_old])
    assert out.pair_model is spectra.pair_model
    assert out.raw_baseline is spectra.raw_baseline
    assert out.tag_background is spectra.tag_background
    fewer = fc.FlowCodeDesign.build(dict(zip(TAGS, TAG_FLUORS)), renamed[:3])
    with pytest.raises(ValueError, match='combinations changed'):
        fc.rename_combos(spectra, fewer)
    other_tags = fc.FlowCodeDesign.build(dict(zip(TAGS[::-1], TAG_FLUORS[::-1])), renamed)
    with pytest.raises(ValueError, match='tags changed'):
        fc.rename_combos(spectra, other_tags)


def test_check_spectra_match():
    design = _design(4)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb).spectra
    fc.check_spectra_match(spectra, bb['spectra'], FLUORS, DETECTORS)

    changed_other = bb['spectra'].copy()
    changed_other[FLUORS.index('FITC'), 3] += 0.1
    fc.check_spectra_match(spectra, changed_other, FLUORS, DETECTORS)

    changed_tag = bb['spectra'].copy()
    changed_tag[0, 3] += 0.01
    with pytest.raises(ValueError, match='Re-characterise'):
        fc.check_spectra_match(spectra, changed_tag, FLUORS, DETECTORS)
    with pytest.raises(ValueError, match='Re-characterise'):
        fc.check_spectra_match(spectra, bb['spectra'], FLUORS, DETECTORS[::-1])


def test_flowcode_spectra_persist_round_trip(tmp_path):
    import json
    design = _design(10)
    bb = _backbone(design, n_per_combo=300)
    spectra = _characterise(design, bb, min_cells_per_combo=50).spectra
    spectra.combo_error[3] = None
    meta = json.loads(json.dumps(fc.save_flowcode_spectra(spectra, tmp_path / 'cache' / 'fc.npz')))
    loaded = fc.load_flowcode_spectra(meta, tmp_path / 'cache' / 'fc.npz')

    assert loaded.design.combo_names == design.combo_names
    np.testing.assert_array_equal(loaded.design.logical, design.logical)
    np.testing.assert_array_equal(loaded.combo_raw_median, spectra.combo_raw_median)
    assert loaded.combo_error[3] is None
    for g in (0, 9):
        np.testing.assert_array_equal(loaded.combo_error[g], spectra.combo_error[g])
    for key in ('linear', 'product', 'support'):
        np.testing.assert_array_equal(getattr(loaded.pair_model, key), getattr(spectra.pair_model, key))
    assert loaded.pair_model.support.dtype == bool
    np.testing.assert_array_equal(loaded.raw_baseline, spectra.raw_baseline)
    np.testing.assert_array_equal(loaded.tag_background, spectra.tag_background)
    np.testing.assert_array_equal(loaded.tag_baseline(), spectra.tag_baseline())
    assert loaded.fingerprint() == meta['fingerprint'] == spectra.fingerprint()
    assert loaded.settings == spectra.settings and 'af_fret_term' not in loaded.settings

    older = dict(meta)
    del older['error_model']
    with pytest.raises(ValueError, match='earlier error model'):
        fc.load_flowcode_spectra(older, tmp_path / 'cache' / 'fc.npz')


# ---------------------------------------------------------------------------
# Tandem tags
# ---------------------------------------------------------------------------

TT_TAGS = ('HA', 'V5', 'AU1', 'S', 'C', 'Flag', 'VSVg', 'Ollas')
TT_TAG_FLUORS = ('BV421', 'PE-Cy7', 'APC', 'BUV395', 'FITC', 'PerCP', 'BV510', 'BV605')
TT_FLUORS = TT_TAG_FLUORS + ('PE', 'APC-Cy7')
TT_DET = 30
TT_DETECTORS = tuple(f'D{i + 1}-A' for i in range(TT_DET))
TT_CLASS = {'PE-Cy7': ('Protein tandem', 'PE'), 'BV605': ('SIRIGEN tandem', ''), 'APC-Cy7': ('Protein tandem', 'APC'),
            'PE': ('Protein', ''), 'APC': ('Protein', '')}
TT_V5 = TT_TAGS.index('V5')
TT_LEVEL = 200_000.0


def _tt_db(extra=()):
    rows = []
    for f in TT_FLUORS + tuple(extra):
        cls, parent = TT_CLASS.get(f, ('Small molecule', ''))
        rows.append({'fluorophore': f, 'synonym1': '', 'synonym2': '', 'synonym3': '', 'synonym4': '',
                     'class': cls, 'tandem.parent': parent})
    return rows


def _tt_spectra():
    ch = np.arange(TT_DET)
    centres = np.linspace(1.5, TT_DET - 2.5, len(TT_FLUORS))
    s = np.array([np.exp(-0.5 * ((ch - c) / 1.2) ** 2) + 0.05 * np.exp(-0.5 * ((ch - c - 5) / 2.0) ** 2)
                  for c in centres])
    return s / s.max(axis=1, keepdims=True)


def _tt_af():
    ch = np.arange(TT_DET)
    af = np.exp(-0.5 * ((ch - 5) / 7.0) ** 2) + 0.2
    return af / af.max()


def _tt_design():
    return fc.FlowCodeDesign.build(dict(zip(TT_TAGS, TT_TAG_FLUORS)), fc.all_triplets(TT_TAGS))


def _tt_tandem():
    spectra = _tt_spectra()
    return fc.TandemTag(tag=TT_V5, fluor='PE-Cy7', parent='PE', parent_label='PE',
                        spectrum=spectra[TT_FLUORS.index('PE')].copy())


def _tt_events(design, n_per_combo=150, n_untransduced=1500, breakdown=0.0, parent_biology=False,
               trend=600.0, seed=0):
    """Raw events of every combo of eight co-expressed tags (one protein:
    each tag is the cell's expression level times a small own factor); V5 is
    PE-Cy7, whose breakdown adds *breakdown* PE per unit of V5. With
    *parent_biology*, PE (the parent's marker) also has a mean per combo and
    a trend with expression shared by all combos."""
    rng = np.random.default_rng(seed)
    spectra = _tt_spectra()
    pe = spectra[TT_FLUORS.index('PE')]
    combo_mean = np.random.default_rng(99).normal(4000.0, 2000.0, design.n_combos)
    raws, codes, v5 = [], [], []
    for g in range(design.n_combos):
        on = design.on_tags(g)
        level = rng.lognormal(np.log(TT_LEVEL), 0.5, n_per_combo)
        a = np.zeros((n_per_combo, len(TT_TAGS)))
        a[:, on] = level[:, None] * rng.lognormal(0.0, 0.15, (n_per_combo, 3))
        marker = np.zeros(n_per_combo)
        if parent_biology:
            marker = combo_mean[g] + trend * (level / TT_LEVEL) + rng.normal(0, 300.0, n_per_combo)
        raw = a @ spectra[:len(TT_TAGS)] + np.outer(marker + breakdown * a[:, TT_V5], pe)
        raws.append(raw)
        codes.append(np.full(n_per_combo, g + 1))
        v5.append(a[:, TT_V5])
    raws.append(np.zeros((n_untransduced, TT_DET)))
    codes.append(np.zeros(n_untransduced, dtype=int))
    v5.append(np.zeros(n_untransduced))
    raw = np.vstack(raws)
    raw = raw + np.outer(rng.gamma(2.0, 100.0, len(raw)), _tt_af()) + rng.normal(0, 15.0, raw.shape)
    return {'raw': raw, 'codes': np.concatenate(codes), 'v5': np.concatenate(v5), 'combo_mean': combo_mean}


def _tt_unmix():
    basis = np.vstack([_tt_spectra(), _tt_af()])

    def unmix(raw):
        coef = fc.ols_unmix(raw, basis)
        return {'unmixed': coef[:, :len(TT_FLUORS)], 'af_scale': coef[:, -1],
                'af_idx': np.ones(len(raw), dtype=np.int64)}
    return unmix


def _tt_characterise(design, raw, tandems):
    unmix = _tt_unmix()
    return fc.characterise_backbone(
        raw=raw, debarcode_unmixed=unmix(raw)['unmixed'], design=design,
        thresholds=np.full(len(TT_TAGS), THRESHOLD), spectra=_tt_spectra(), fluor_names=TT_FLUORS,
        detectors=TT_DETECTORS, af_mean=_tt_af(), transform=_transform,
        transform_params={'magnitude_ceiling': 262144.0}, rescue_floor=None,
        settings=fc.CharacteriseSettings(rescue_toomany=False), tandems=tandems,
    ).spectra


def _tt_estimate(fcs_, raw, tandems, **kwargs):
    unmix = _tt_unmix()
    return fc.estimate_tandem_shifts(raw, fcs_, _tt_spectra(), TT_FLUORS, unmix, _transform, tandems,
                                     rescue=False, **kwargs)


def test_tandem_candidates_read_class_parent_panel_and_library(monkeypatch):
    import pandas as pd

    design = fc.FlowCodeDesign.build(dict(zip(('HA', 'V5', 'AU1', 'S'), ('BV421', 'PE-Cy7', 'APC-Cy7', 'BV605'))),
                                     [('A', ('HA', 'V5', 'AU1')), ('B', ('HA', 'V5', 'S'))])
    spectra = _tt_spectra()
    library = pd.DataFrame([np.linspace(0, 1, TT_DET)], index=['APC'], columns=list(TT_DETECTORS))
    monkeypatch.setattr(fc, '_reference_library', lambda key: library if key == 'Aurora' else None)
    panel = [f for f in TT_FLUORS if f != 'APC']
    rows = fc.tandem_candidates(design, panel, spectra[[TT_FLUORS.index(f) for f in panel]], TT_DETECTORS,
                                'Aurora', fluor_db=_tt_db())
    by = {r['fluor']: r for r in rows}
    assert [r['fluor'] for r in rows] == ['PE-Cy7', 'APC-Cy7', 'BV605']
    pecy7 = by['PE-Cy7']['tandem']
    assert pecy7.tag == 1 and pecy7.parent == 'PE' and pecy7.parent_label == 'PE' and pecy7.source == 'panel'
    np.testing.assert_array_equal(pecy7.spectrum, spectra[TT_FLUORS.index('PE')])
    apccy7 = by['APC-Cy7']['tandem']
    assert apccy7.parent_label is None and apccy7.source == 'library'
    np.testing.assert_allclose(apccy7.spectrum, np.linspace(0, 1, TT_DET))
    assert by['BV605']['tandem'] is None and 'no tandem.parent' in by['BV605']['problem']
    assert by['PE-Cy7']['fluor_class'] == 'Protein tandem'

    no_library = fc.tandem_candidates(design, panel, spectra[[TT_FLUORS.index(f) for f in panel]], TT_DETECTORS,
                                      None, fluor_db=_tt_db())
    assert next(r for r in no_library if r['fluor'] == 'APC-Cy7')['problem'] == 'no APC spectrum for these detectors'
    tag_parent = fc.FlowCodeDesign.build(dict(zip(('HA', 'V5', 'AU1'), ('BV421', 'PE-Cy7', 'PE'))),
                                         [('A', ('HA', 'V5', 'AU1'))])
    row = fc.tandem_candidates(tag_parent, TT_FLUORS, spectra, TT_DETECTORS, None, fluor_db=_tt_db())[0]
    assert row['tandem'] is None and row['problem'] == 'the parent PE is a FlowCode tag'


def test_tandem_statistic_separates_breakdown_from_biology_and_the_shared_trend():
    design = _tt_design()
    tandem = _tt_tandem()
    spectra, af = _tt_spectra(), _tt_af()
    tag_cols = np.arange(len(TT_TAGS))
    for breakdown in (0.0, 0.005):
        ev = _tt_events(design, breakdown=breakdown, parent_biology=True, trend=1500.0, seed=4)
        tags = fc.ols_unmix(ev['raw'], np.vstack([spectra, af]))[:, tag_cols]
        parent = fc.tandem_parent_values(ev['raw'], spectra, TT_FLUORS, af, tandem)
        stat = fc.tandem_statistic(parent, tags, ev['codes'], design, TT_V5, (), 50)
        assert stat['n_combos'] == 21 and stat['n_null'] == 35
        assert abs(stat['beta'] - breakdown) < max(3 * stat['se'], 0.1 * breakdown), stat
        assert stat['gamma'] > 500
        assert len(stat['combo_beta']) == 21 and len(stat['combo_null']) == 35
    # combos with another same-parent tandem tag are left out of both groups
    flag = TT_TAGS.index('Flag')
    stat = fc.tandem_statistic(parent, tags, ev['codes'], design, TT_V5, [flag], 50)
    assert stat['n_combos'] == 15 and stat['n_null'] == 20


def test_tandem_shift_pairs_combos_and_soft_thresholds_on_z():
    stat = {'beta': 0.010, 'se': 0.002, 'n_combos': 6, 'n_null': 6,
            'combo_beta': {f'c{i}': 0.010 + 0.05 * i for i in range(6)},
            'combo_null': {f'n{i}': 0.001 * (-1) ** i for i in range(6)}}
    unpaired = fc.tandem_shift(stat, {'beta': 0.004, 'se': 0.0}, z_min=3.0, min_combos=5)
    assert unpaired['paired'] == 0 and unpaired['delta'] == pytest.approx(0.006)
    assert unpaired['z'] == pytest.approx(3.0) and unpaired['beta'] == 0.0 and unpaired['reason'] == '|z| below 3'
    reference = {'beta': 0.0, 'se': 0.0, 'combo_beta': {f'c{i}': 0.05 * i for i in range(6)},
                 'combo_null': {f'n{i}': 0.0 for i in range(6)}}
    paired = fc.tandem_shift(stat, reference, z_min=3.0, min_combos=5)
    assert paired['paired'] == 6 and paired['delta'] == pytest.approx(0.010)
    assert paired['z'] > 3.0
    assert paired['beta'] == pytest.approx(paired['weight'] * 0.010)
    assert paired['weight'] == pytest.approx(1.0 - (3.0 / paired['z']) ** 2)
    few = fc.tandem_shift(dict(stat, n_null=4), reference, min_combos=5)
    assert few['beta'] == 0.0 and 'at least 5 of each' in few['reason']
    no_reference = fc.tandem_shift(stat, None, z_min=3.0)
    assert no_reference['delta'] == pytest.approx(0.010) and no_reference['z'] == pytest.approx(5.0)
    assert no_reference['beta'] == pytest.approx((1 - 0.36) * 0.010)


def test_apply_tandem_correction_reaches_every_event_carrying_the_tag():
    design = _tt_design()
    tandem = _tt_tandem()
    with_v5 = next(g for g in range(design.n_combos) if TT_V5 in design.on_tags(g))
    without = next(g for g in range(design.n_combos) if TT_V5 not in design.on_tags(g))
    ids = np.array([with_v5 + 1, without + 1, design.unresolved_code, design.unresolved_code, 0])
    tags = np.zeros((5, len(TT_TAGS)))
    tags[:, TT_V5] = [5000.0, 5000.0, 5000.0, 100.0, 0.0]
    raw = np.zeros((5, TT_DET))
    baseline = np.zeros(len(TT_TAGS))
    baseline[TT_V5] = 1000.0
    out = fc.apply_tandem_correction(raw, tags, ids, design, np.full(len(TT_TAGS), 300.0), baseline,
                                     [(tandem, 0.01)])
    np.testing.assert_allclose(out[0], -40.0 * tandem.spectrum)
    np.testing.assert_allclose(out[2], -40.0 * tandem.spectrum)
    assert not out[[1, 3, 4]].any()
    np.testing.assert_array_equal(fc.apply_tandem_correction(raw, tags, ids, design, np.full(8, 300.0), baseline,
                                                             [(tandem, 0.0)]), raw)


def test_tandem_breakdown_beyond_the_backbone_is_found_and_removed():
    design = _tt_design()
    tandem = _tt_tandem()
    backbone = _tt_events(design, n_per_combo=200, seed=1)
    char = _tt_characterise(design, backbone['raw'], [tandem])
    ref = char.diagnostics['tandem']['PE-Cy7']
    assert ref['parent'] == 'PE' and ref['tag'] == 'V5' and ref['source'] == 'panel'
    assert ref['n_combos'] == 21 and ref['n_null'] == 35 and abs(ref['beta']) < 0.002
    assert len(ref['combo_beta']) == 21 and np.isfinite(ref['beta_uncorrected'])

    clean = _tt_events(design, parent_biology=True, seed=2)
    est = _tt_estimate(char, clean['raw'], [tandem])[0]
    assert est['paired'] == 21 and abs(est['z']) < 3 and est['beta'] == 0.0 and est['has_reference']

    broken = _tt_events(design, breakdown=0.005, parent_biology=True, seed=2)
    est = _tt_estimate(char, broken['raw'], [tandem])[0]
    assert est['z'] > 10 and est['delta'] == pytest.approx(0.005, rel=0.15)
    assert est['beta'] == pytest.approx(est['weight'] * est['delta']) and est['weight'] > 0.9
    assert est['at_median'] == pytest.approx(est['beta'] * est['median_tag'])
    assert set(est) >= {'fluor', 'tag', 'parent', 'source', 'sample_beta', 'sample_se', 'n_combos', 'n_null',
                        'reference_beta', 'reason', 'events'}

    without = fc.FlowCodeSpectra(**{**vars(char), 'diagnostics': {}})
    est_none = _tt_estimate(without, broken['raw'], [tandem])[0]
    assert not est_none['has_reference'] and est_none['beta'] == 0.0 and est_none['z'] > 10
    assert est_none['reason'] == 'no backbone statistic: run section 4 again'

    unmix = _tt_unmix()
    pe = TT_FLUORS.index('PE')
    kwargs = dict(rescue=False, ref_mask=np.ones(len(broken['raw']), dtype=bool))
    plain = fc.unmix_staged(broken['raw'], char, _tt_spectra(), TT_FLUORS, unmix, unmix, _transform, **kwargs)
    fixed = fc.unmix_staged(broken['raw'], char, _tt_spectra(), TT_FLUORS, unmix, unmix, _transform,
                            tandem_shifts=[(tandem, est['beta'])], **kwargs)
    assert fixed.diagnostics['tandem'] == {'PE-Cy7': est['beta']}
    np.testing.assert_array_equal(fixed.ids, plain.ids)
    truth = fc.unmix_staged(clean['raw'], char, _tt_spectra(), TT_FLUORS, unmix, unmix, _transform, **kwargs)
    v5_combo = np.array([TT_V5 in design.on_tags(g) for g in range(design.n_combos)])
    carriers = (broken['codes'] > 0) & v5_combo[np.maximum(broken['codes'] - 1, 0)]
    before = np.median(plain.unmixed[carriers, pe] - truth.unmixed[carriers, pe])
    after = np.median(fixed.unmixed[carriers, pe] - truth.unmixed[carriers, pe])
    assert before > 500.0 and abs(after) < 0.1 * before
    others = (broken['codes'] > 0) & ~carriers
    np.testing.assert_allclose(fixed.unmixed[others], plain.unmixed[others])


def test_tandem_reference_survives_saving_and_renaming(tmp_path):
    import json
    design = _tt_design()
    char = _tt_characterise(design, _tt_events(design, n_per_combo=120, seed=1)['raw'], [_tt_tandem()])
    meta = json.loads(json.dumps(fc.save_flowcode_spectra(char, tmp_path / 'fc.npz')))
    loaded = fc.load_flowcode_spectra(meta, tmp_path / 'fc.npz')
    ref, back = char.diagnostics['tandem']['PE-Cy7'], loaded.diagnostics['tandem']['PE-Cy7']
    assert back['combo_beta'] == pytest.approx(ref['combo_beta'])
    assert back['beta'] == pytest.approx(ref['beta']) and back['parent'] == 'PE'
    renamed = fc.FlowCodeDesign.build(dict(zip(TT_TAGS, TT_TAG_FLUORS)),
                                      [(f'R{g}', t) for g, t in enumerate(design.combo_tags)])
    assert fc.rename_combos(loaded, renamed).diagnostics['tandem'] == loaded.diagnostics['tandem']


def test_staged_export_prepare_estimates_once_per_sample_and_the_chunks_apply_it(monkeypatch):
    from types import SimpleNamespace

    design = _tt_design()
    tandem = _tt_tandem()
    char = _tt_characterise(design, _tt_events(design, n_per_combo=200, seed=1)['raw'], [tandem])
    sample = _tt_events(design, breakdown=0.005, parent_biology=True, seed=2)
    unmix = _tt_unmix()
    monkeypatch.setattr(fc, 'af_only_unmix', lambda raw, *a, **k: unmix(raw))
    pnn = ('FSC-A', 'SSC-A') + TT_DETECTORS
    raw_all = np.column_stack([np.full((len(sample['raw']), 2), 10.0), sample['raw']])
    gate = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 50], [0, 50]]}
    seen = []
    settings = {'tag_shift_min_n': 100, 'correction': 'combo', 'regressor': 'fixed', 'tandem_z_min': 3.0,
                'tandem_min_combos': 5, 'tandem_max_events': 1_000_000, 'calib_combos': False}
    char.settings['rescue_toomany'] = False
    unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda raw, af, _key=None: unmix(raw), [gate], pnn,
                                   settings, tandems=[tandem], on_tandem=seen.append)
    af = SimpleNamespace(spectra=np.zeros((2, TT_DET)))
    unmix_fn.prepare(sample['raw'], raw_all, af)
    assert len(seen) == 1 and seen[0][0]['fluor'] == 'PE-Cy7' and 'tandem' not in seen[0][0]
    beta = seen[0][0]['beta']
    assert beta > 0.004
    half = len(sample['raw']) // 2
    out = np.vstack([unmix_fn(sample['raw'][s], raw_all[s], af)['unmixed']
                     for s in (slice(0, half), slice(half, None))])
    expected = fc.unmix_staged(sample['raw'], char, _tt_spectra(), TT_FLUORS, unmix, unmix,
                               fc.logicle_transform(262144.0), ref_mask=np.ones(len(sample['raw']), dtype=bool),
                               rescue=False, tandem_shifts=[(tandem, beta)])
    np.testing.assert_allclose(out, expected.unmixed)

    plain_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda raw, af, _key=None: unmix(raw), [gate], pnn,
                                   settings, on_tandem=seen.append)
    assert len(seen) == 1
    no_tandem = fc.unmix_staged(sample['raw'], char, _tt_spectra(), TT_FLUORS, unmix, unmix,
                                fc.logicle_transform(262144.0), ref_mask=np.ones(len(sample['raw']), dtype=bool),
                                rescue=False)
    np.testing.assert_allclose(plain_fn(sample['raw'], raw_all, af)['unmixed'], no_tandem.unmixed)


def test_corrected_raw_is_the_final_unmix_input_and_the_fix_unmix_loader_reads_it(monkeypatch):
    from types import SimpleNamespace
    import flowcode_fix_unmix as fcfu

    design = _tt_design()
    tandem = _tt_tandem()
    char = _tt_characterise(design, _tt_events(design, n_per_combo=200, seed=1)['raw'], [tandem])
    sample = _tt_events(design, breakdown=0.005, parent_biology=True, seed=2)
    unmix = _tt_unmix()
    monkeypatch.setattr(fc, 'af_only_unmix', lambda raw, *a, **k: unmix(raw))
    pnn = ('FSC-A', 'SSC-A') + TT_DETECTORS
    scatter = np.full((len(sample['raw']), 2), 10.0)
    raw_all = np.column_stack([scatter, sample['raw']])
    gate = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 50], [0, 50]]}
    settings = {'tag_shift_min_n': 100, 'correction': 'combo', 'regressor': 'fixed', 'tandem_z_min': 3.0,
                'tandem_min_combos': 5, 'tandem_max_events': 1_000_000}
    char.settings['rescue_toomany'] = False
    unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda raw, af, _key=None: unmix(raw), [gate], pnn,
                                   settings, tandems=[tandem])
    af = SimpleNamespace(spectra=np.zeros((2, TT_DET)))

    def base(_dir, path, _gate, _layout, _gating):
        return sample['raw'], scatter

    loader = fcfu.corrected_loader(unmix_fn, {'fc.fcs': af}, TT_DETECTORS, ('FSC-A', 'SSC-A'), base_loader=base)
    fl, sc = loader('', 'fc.fcs', 'root', None, None)
    assert sc is scatter and not np.allclose(fl, sample['raw'])
    np.testing.assert_allclose(unmix(fl)['unmixed'], unmix_fn(sample['raw'], raw_all, af)['unmixed'])
    other, _sc = loader('', 'unstained.fcs', 'root', None, None)
    assert other is sample['raw']

    s = SimpleNamespace(fixed_fluorophores=('PE',), exclude_fixed_targets=False)
    fcfu.keep_tags_out(s, ['PE-Cy7', 'PE'])
    assert s.fixed_fluorophores == ('PE', 'PE-Cy7') and s.exclude_fixed_targets is True
    with pytest.raises(ValueError):
        fcfu.keep_tags_out(SimpleNamespace(fixed_fluorophores=()), ['PE'])


# ---------------------------------------------------------------------------
# Per-sample combination calibration
# ---------------------------------------------------------------------------

def _cal_af():
    return np.vstack([_tt_af(), 0.8 * _tt_af() + 0.1])


def _cal_events(design, scale=None, slope=None, in_span=(), n_per_combo=200, n_untransduced=3000, seed=0):
    """Raw events of every combination of the eight tandem-test tags, each
    combination carrying an error of 1 % of its on-tag sum along its own
    fixed random detector pattern (mostly outside the span of the spectra
    and AF), times *scale* per combination, plus *slope* times the log
    on-tag fold sum about its median. The patterns of the *in_span*
    combinations are the sum of the two marker spectra (PE, APC-Cy7).
    Returns the raw events, their codes and the same events without any
    error."""
    spectra = _tt_spectra()
    patterns = np.random.default_rng(5).normal(0.0, 1.0, (design.n_combos, TT_DET))
    for g in in_span:
        patterns[g] = spectra[len(TT_TAGS):].sum(axis=0) * np.abs(patterns[g]).mean()
    scale = np.ones(design.n_combos) if scale is None else np.asarray(scale, dtype=float)
    slope = np.zeros(design.n_combos) if slope is None else np.asarray(slope, dtype=float)
    rng = np.random.default_rng(seed)
    raws, errs, codes = [], [], []
    for g in range(design.n_combos):
        on = design.on_tags(g)
        level = rng.lognormal(np.log(TT_LEVEL), 0.5, n_per_combo)
        a = np.zeros((n_per_combo, len(TT_TAGS)))
        a[:, on] = level[:, None] * rng.lognormal(0.0, 0.15, (n_per_combo, 3))
        u = np.log((a[:, on] / THRESHOLD).sum(axis=1))
        factor = scale[g] + slope[g] * (u - np.median(u))
        raws.append(a @ spectra[:len(TT_TAGS)])
        errs.append((factor * 0.01 * a[:, on].sum(axis=1))[:, None] * patterns[g])
        codes.append(np.full(n_per_combo, g + 1))
    raws.append(np.zeros((n_untransduced, TT_DET)))
    errs.append(np.zeros((n_untransduced, TT_DET)))
    codes.append(np.zeros(n_untransduced, dtype=int))
    clean = np.vstack(raws)
    noise = np.outer(rng.gamma(2.0, 100.0, len(clean)), _tt_af()) + rng.normal(0, 15.0, clean.shape)
    return clean + noise + np.vstack(errs), np.concatenate(codes), clean + noise


def _cal_char(design, in_span=()):
    char = _tt_characterise(design, _cal_events(design, in_span=in_span, seed=1)[0], [])
    char.settings['rescue_toomany'] = False
    return char


def _cal_scales(char, raw, rng_seed=0, **settings):
    stage = fc._debarcode_and_correct(raw, char, _tt_spectra(), list(TT_FLUORS), _tt_unmix(), _transform, None,
                                      char.thresholds, False, 0.9, 0.5, 100, 'combo', 'fixed', None)
    scales = fc.estimate_combination_scales(raw, stage['dc'].ids, stage['tag_abundances'], stage['tag_baseline'],
                                            char, _tt_spectra(), _cal_af(), fc.poisson_weights(raw),
                                            settings=settings, rng=np.random.default_rng(rng_seed))
    return scales, stage


def _cal_unmix(char, raw, scales=None, correction='combo'):
    return fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, _tt_unmix(), _tt_unmix(), _transform, rescue=False,
                           ref_mask=np.ones(len(raw), dtype=bool), combo_scales=scales, correction=correction)


def test_combination_scales_recover_a_sample_scale_and_flatten_the_correction():
    design = _tt_design()
    char = _cal_char(design)
    truth = np.ones(design.n_combos)
    truth[:10] = 0.5
    raw, codes, clean = _cal_events(design, truth, seed=2)
    scales, stage = _cal_scales(char, raw)
    assert np.mean(stage['dc'].ids == codes) > 0.99
    rec = [scales['combos'][design.combo_names[g]] for g in range(design.n_combos)]
    assert all(abs(r['s'] - 0.5) < 0.05 and r['weight'] > 0.99 for r in rec[:10])
    assert all(abs(r['applied'][0] - 1.0) < 0.01 for r in rec[10:])
    assert all(0.05 <= r['share'] <= 1.0 and r['n'] == 200 and r['slope'] == 0.0 for r in rec)
    plain = _cal_unmix(char, raw)
    calibrated = _cal_unmix(char, raw, scales)
    true_unmix = _cal_unmix(char, clean, correction='none')
    off = [TT_FLUORS.index(f) for f in ('PE', 'APC-Cy7')]
    halved = np.isin(codes, np.arange(1, 11))
    before = np.abs(np.median(plain.unmixed[halved][:, off] - true_unmix.unmixed[halved][:, off], axis=0))
    after = np.abs(np.median(calibrated.unmixed[halved][:, off] - true_unmix.unmixed[halved][:, off], axis=0))
    assert np.all(after < 0.2 * before), (before, after)
    assert calibrated.diagnostics['calibrated'] == sum(r['weight'] > 0 for r in rec)


def test_combination_scales_leave_a_backbone_like_sample_unchanged():
    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, seed=2)
    scales, stage = _cal_scales(char, raw)
    assert len(scales['combos']) == design.n_combos and not scales['not_identifiable']
    assert all(round(r['applied'][0], 2) == 1.0 for r in scales['combos'].values())
    corrected = stage['corrected']['raw']
    again = fc.apply_combination_scales(corrected, stage['dc'].ids, stage['tag_abundances'], stage['tag_baseline'],
                                        char, scales)
    removed = np.abs(raw - corrected).max()
    assert np.abs(again - corrected).max() < 0.01 * removed


def test_combinations_whose_error_is_inside_the_span_are_not_identifiable():
    design = _tt_design()
    char = _cal_char(design, in_span=(0, 1))
    truth = np.full(design.n_combos, 0.5)
    raw, _codes, _clean = _cal_events(design, truth, in_span=(0, 1), seed=2)
    scales, stage = _cal_scales(char, raw)
    names = design.combo_names
    assert scales['not_identifiable'] == [names[0], names[1]]
    assert names[0] not in scales['combos'] and names[2] in scales['combos']
    out = fc.apply_combination_scales(stage['corrected']['raw'], stage['dc'].ids, stage['tag_abundances'],
                                      stage['tag_baseline'], char, scales)
    kept = np.isin(stage['dc'].ids, (1, 2))
    np.testing.assert_array_equal(out[kept], stage['corrected']['raw'][kept])
    few, _stage = _cal_scales(char, raw, calib_min_events=500)
    assert not few['combos'] and len(few['too_few']) == design.n_combos and not few['not_identifiable']


def test_combination_calibration_is_off_without_fret_correction_or_the_combo_correction():
    from types import SimpleNamespace

    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, np.full(design.n_combos, 0.5), seed=2)
    scales, _stage = _cal_scales(char, raw)
    no_fret = fc.FlowCodeSpectra(**{**vars(char), 'fret_corrected': False})
    for spectra, correction in ((no_fret, 'combo'), (char, 'none'), (char, 'pairwise')):
        plain = fc.staged_corrected_raw(raw, spectra, _tt_spectra(), TT_FLUORS, _tt_unmix(), _transform,
                                        rescue=False, correction=correction)[1]
        with_scales = fc.staged_corrected_raw(raw, spectra, _tt_spectra(), TT_FLUORS, _tt_unmix(), _transform,
                                              rescue=False, correction=correction, combo_scales=scales)[1]
        np.testing.assert_array_equal(with_scales, plain)
        seen = []
        unmix_fn = fc.staged_export_fn(spectra, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: r, [],
                                       ('FSC-A',) + TT_DETECTORS, {'correction': correction},
                                       debarcode_fn=lambda r, af, _key=None: _tt_unmix()(r), on_scales=seen.append)
        unmix_fn.prepare(raw, np.column_stack([np.zeros(len(raw)), raw]), SimpleNamespace(spectra=_cal_af()), 's')
        assert not seen


def test_export_chunks_and_the_fix_unmix_loader_apply_the_same_calibration(monkeypatch):
    from types import SimpleNamespace
    import flowcode_fix_unmix as fcfu

    design = _tt_design()
    char = _cal_char(design)
    truth = np.ones(design.n_combos)
    truth[:10] = 0.5
    raw, _codes, _clean = _cal_events(design, truth, seed=2)
    unmix = _tt_unmix()
    monkeypatch.setattr(fc, 'af_only_unmix', lambda r, *a, **k: unmix(r))
    pnn = ('FSC-A', 'SSC-A') + TT_DETECTORS
    scatter = np.column_stack([np.where(np.arange(len(raw)) % 10 == 0, 80.0, 10.0), np.full(len(raw), 10.0)])
    raw_all = np.column_stack([scatter, raw])
    singlets = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 50], [0, 50]]}
    seen = []
    unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: unmix(r), [singlets],
                                   pnn, {'correction': 'combo'}, calibration_record=singlets, on_scales=seen.append)
    af = SimpleNamespace(spectra=_cal_af())
    unmix_fn.prepare(raw, raw_all, af, 'sample.fcs')
    assert len(seen) == 1 and seen[0]['weighting'] == 'poisson'
    assert seen[0]['events'] == int(np.sum(scatter[:, 0] < 50))
    scales = seen[0]
    assert all(abs(scales['combos'][design.combo_names[g]]['s'] - 0.5) < 0.05 for g in range(10))
    half = len(raw) // 2
    chunks = np.vstack([unmix_fn(raw[s], raw_all[s], af, 'sample.fcs')['unmixed']
                        for s in (slice(0, half), slice(half, None))])
    expected = fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                               ref_mask=scatter[:, 0] < 50, rescue=False, combo_scales=scales)
    np.testing.assert_allclose(chunks, expected.unmixed)
    standard = fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                               ref_mask=scatter[:, 0] < 50, rescue=False)
    assert not np.allclose(chunks, standard.unmixed)

    def base(_dir, path, _gate, _layout, _gating):
        return raw, scatter
    loader = fcfu.corrected_loader(unmix_fn, {'sample.fcs': af}, TT_DETECTORS, ('FSC-A', 'SSC-A'), base_loader=base)
    fl, _sc = loader('', 'sample.fcs', 'root', None, None)
    np.testing.assert_allclose(unmix(fl)['unmixed'], chunks)


def test_brightness_calibration_is_an_option_and_off_is_the_one_scale_path():
    design = _tt_design()
    char = _cal_char(design)
    slope = np.zeros(design.n_combos)
    slope[:10] = -0.3
    raw, _codes, _clean = _cal_events(design, np.ones(design.n_combos), slope=slope, seed=2)
    one, stage = _cal_scales(char, raw)
    off, _stage = _cal_scales(char, raw, calib_brightness=False)
    assert one == off and not one['brightness']
    assert all(r['slope'] == 0.0 and r['applied'][1] == 0.0 and r['u_median'] is None for r in one['combos'].values())
    bright, _stage = _cal_scales(char, raw, calib_brightness=True)
    assert bright['brightness']
    for g in range(design.n_combos):
        r = bright['combos'][design.combo_names[g]]
        if g < 10:
            assert abs(r['slope'] + 0.3) < 0.05 and abs(r['s'] - 1.0) < 0.05 and r['weight'] > 0.99
        else:
            assert abs(r['applied'][1]) < 0.01
    args = (stage['corrected']['raw'], stage['dc'].ids, stage['tag_abundances'], stage['tag_baseline'], char)
    flat = fc.apply_combination_scales(*args, one)
    varying = fc.apply_combination_scales(*args, bright)
    g0 = stage['dc'].ids == 1
    assert not np.allclose(flat[g0], varying[g0])


def test_overrides_switch_off_or_scale_a_combination_in_place_of_its_calibration():
    design = _tt_design()
    char = _cal_char(design)
    truth = np.ones(design.n_combos)
    truth[:10] = 0.5
    raw, _codes, _clean = _cal_events(design, truth, seed=2)
    scales, stage = _cal_scales(char, raw)
    names = design.combo_names
    keys = [fc.combo_key(t) for t in design.combo_tags]
    overrides = fc.combination_overrides({keys[0]: 0.0, keys[1].lower(): 0.25, keys[12]: 9.0, 'NOT_A_COMBO': 0.0,
                                          keys[13]: None}, design)
    assert overrides == {names[0]: 0.0, names[1]: 0.25, names[12]: fc.CALIB_MAX_SCALE}
    merged = fc.with_combination_overrides(scales, overrides)
    assert merged['combos'][names[0]]['override'] and merged['combos'][names[0]]['applied'] == [0.0, 0.0]
    assert merged['combos'][names[2]] == scales['combos'][names[2]]
    assert 'override' not in scales['combos'][names[0]]
    assert fc.with_combination_overrides(scales, {}) is scales
    ids = stage['dc'].ids
    corrected = stage['corrected']['raw']
    args = (corrected, ids, stage['tag_abundances'], stage['tag_baseline'], char)
    out = fc.apply_combination_scales(*args, merged)
    calibrated = fc.apply_combination_scales(*args, scales)
    np.testing.assert_allclose(out[ids == 1], raw[ids == 1])
    np.testing.assert_allclose(raw[ids == 2] - out[ids == 2], 0.25 * (raw[ids == 2] - corrected[ids == 2]),
                               atol=1e-6 * np.abs(raw).max())
    np.testing.assert_allclose(raw[ids == 13] - out[ids == 13], 1.5 * (raw[ids == 13] - corrected[ids == 13]),
                               atol=1e-6 * np.abs(raw).max())
    other = ~np.isin(ids, (1, 2, 13))
    np.testing.assert_array_equal(out[other], calibrated[other])
    summary = fc.calibration_summary(merged, design)
    assert summary['overridden'] == [(names[0], 0.0), (names[1], 0.25), (names[12], fc.CALIB_MAX_SCALE)]
    assert summary['identifiable'] == design.n_combos - 3
    assert not {names[0], names[1], names[12]} & set(summary['calibrated'])
    assert _cal_unmix(char, raw, merged).diagnostics['calibrated'] == len(summary['calibrated'])


def test_overrides_apply_without_calibration_and_to_pairwise_combinations():
    design = _tt_design()
    char = _cal_char(design)
    assert char.pair_model is not None
    raw, _codes, _clean = _cal_events(design, seed=2)
    names = design.combo_names
    alone = fc.with_combination_overrides(None, {names[0]: 0.0})
    assert alone['combos'][names[0]]['override'] and not alone['not_identifiable']
    pairwise = fc.FlowCodeSpectra(**{**vars(char), 'combo_error': [None] + list(char.combo_error[1:])})
    for spectra in (char, pairwise):
        _scales, stage = _cal_scales(spectra, raw)
        ids = stage['dc'].ids
        assert np.abs(raw[ids == 1] - stage['corrected']['raw'][ids == 1]).max() > 0
        out = fc.apply_combination_scales(stage['corrected']['raw'], ids, stage['tag_abundances'],
                                          stage['tag_baseline'], spectra, alone)
        np.testing.assert_allclose(out[ids == 1], raw[ids == 1])
        np.testing.assert_array_equal(out[ids != 1], stage['corrected']['raw'][ids != 1])
    calibrated_only = fc.with_combination_overrides(None, {})
    assert calibrated_only is None


def test_export_applies_overrides_with_the_combo_correction_only():
    from types import SimpleNamespace

    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, np.full(design.n_combos, 0.5), seed=2)
    unmix = _tt_unmix()
    key0 = fc.combo_key(design.combo_tags[0])
    af = SimpleNamespace(spectra=_cal_af())
    pnn = ('FSC-A',) + TT_DETECTORS
    raw_all = np.column_stack([np.zeros(len(raw)), raw])
    expected_scales = fc.with_combination_overrides(None, {design.combo_names[0]: 0.0})
    for settings, called in (({'correction': 'combo', 'calib_combos': False, 'combo_overrides': {key0: 0.0}}, True),
                             ({'correction': 'combo', 'calib_combos': False, 'combo_overrides': {}}, False),
                             ({'correction': 'pairwise', 'calib_combos': False, 'combo_overrides': {key0: 0.0}}, False)):
        seen = []
        unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: unmix(r), [], pnn,
                                       settings, debarcode_fn=lambda r, af, _key=None: unmix(r),
                                       on_scales=seen.append)
        unmix_fn.prepare(raw, raw_all, af, 's')
        assert bool(seen) == called
        if not called:
            continue
        assert seen[0] == expected_scales
        chunks = unmix_fn(raw, raw_all, af, 's')['unmixed']
        direct = fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                                 rescue=False, combo_scales=expected_scales)
        np.testing.assert_allclose(chunks, direct.unmixed)
        plain = fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                                rescue=False)
        g1 = direct.ids == 1
        assert not np.allclose(chunks[g1], plain.unmixed[g1])
        np.testing.assert_allclose(chunks[direct.ids == 2], plain.unmixed[direct.ids == 2])
        with_calibration = fc.staged_export_fn(
            char, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: unmix(r), [], pnn,
            {**settings, 'calib_combos': True}, debarcode_fn=lambda r, af, _key=None: unmix(r), on_scales=seen.append)
        with_calibration.prepare(raw, raw_all, af, 's')
        both = seen[-1]
        assert both['combos'][design.combo_names[0]]['override']
        assert abs(both['combos'][design.combo_names[1]]['s'] - 0.5) < 0.05


def test_combination_scales_are_deterministic_per_sample():
    from types import SimpleNamespace

    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, np.full(design.n_combos, 0.7), seed=2)
    first, _stage = _cal_scales(char, raw, rng_seed=7)
    second, _stage = _cal_scales(char, raw, rng_seed=7)
    assert first == second
    unmix = _tt_unmix()
    seen = []
    unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: unmix(r), [],
                                   ('FSC-A',) + TT_DETECTORS, {'correction': 'combo'},
                                   debarcode_fn=lambda r, af, _key=None: unmix(r), on_scales=seen.append)
    af = SimpleNamespace(spectra=_cal_af())
    raw_all = np.column_stack([np.zeros(len(raw)), raw])
    for key in ('A1.fcs', 'A1.fcs', 'D3.fcs'):
        unmix_fn.prepare(raw, raw_all, af, key)
    assert seen[0] == seen[1]
    assert [r['se'] for r in seen[0]['combos'].values()] != [r['se'] for r in seen[2]['combos'].values()]


def test_debarcoding_in_blocks_equals_one_pass_and_rows_slice_it():
    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, seed=2)
    char.settings['rescue_toomany'] = False
    too_few = fc.TooFewSettings.from_settings(char.settings)
    args = (raw, char, _tt_spectra(), list(TT_FLUORS), _tt_unmix(), _transform, None, char.thresholds, False, 0.9,
            0.5, 100, 'fixed', too_few)
    whole, blocks = fc._debarcode(*args), fc._debarcode(*args, block_size=1000)
    np.testing.assert_array_equal(blocks['dc'].ids, whole['dc'].ids)
    np.testing.assert_allclose(blocks['tag_abundances'], whole['tag_abundances'])
    half = len(raw) // 2
    parts = [fc.debarcoded_rows(whole, slice(0, half)), fc.debarcoded_rows(whole, slice(half, len(raw)))]
    np.testing.assert_array_equal(np.concatenate([p['dc'].ids for p in parts]), whole['dc'].ids)
    assert sum(len(p['dc'].rescued_toofew) for p in parts) == len(whole['dc'].rescued_toofew)
    picked = np.sort(np.random.default_rng(1).choice(len(raw), 500, replace=False))
    sub = fc.debarcoded_rows(whole, picked)
    np.testing.assert_array_equal(sub['dc'].ids, whole['dc'].ids[picked])
    np.testing.assert_array_equal(picked[sub['dc'].rescued_toofew],
                                  np.intersect1d(picked, whole['dc'].rescued_toofew))


def test_second_debarcoding_after_the_correction_recalls_only_unbarcoded_events():
    design = _tt_design()
    char = _cal_char(design)
    raw, _codes, _clean = _cal_events(design, seed=2)
    too_few = fc.TooFewSettings.from_settings(char.settings)
    unmix = _tt_unmix()
    args = (raw, char, _tt_spectra(), list(TT_FLUORS), unmix, _transform, None, char.thresholds, True, 0.9,
            0.5, 100, 'fixed', too_few)
    first = fc._debarcode(*args)
    second = fc._debarcode(*args, second_pass=True)
    ids1, ids2 = first['dc'].ids, second['dc'].ids
    barcoded = (ids1 >= 1) & (ids1 <= design.n_combos)
    np.testing.assert_array_equal(ids2[barcoded], ids1[barcoded])

    corrected = fc.apply_error_correction(raw, first['tag_abundances'], ids1, design, char.combo_error,
                                          char.pair_model, char.thresholds, 'combo',
                                          tag_baseline=first['tag_baseline'])['raw']
    out = unmix(corrected)
    tag_cols = [list(TT_FLUORS).index(f) for f in design.fluors]
    again = fc.debarcode_rescue(out['unmixed'][:, tag_cols], char.thresholds, design, _transform,
                                reference_median=char.combo_raw_median, floor=char.rescue_floor, too_few=too_few,
                                af_scale=out['af_scale'], usable=fc.unsaturated_mask(raw, char),
                                reference_background=char.tag_background)
    np.testing.assert_array_equal(ids2[~barcoded], again.ids[~barcoded])
    gained = np.flatnonzero(~barcoded & (again.ids >= 1) & (again.ids <= design.n_combos))
    np.testing.assert_array_equal(second['dc'].rescued_corrected, gained)
    np.testing.assert_array_equal(fc._debarcode(*args, second_pass=True, correction='none')['dc'].ids, ids1)


def test_prepare_debarcodes_the_sample_once_and_the_chunks_reuse_it():
    from types import SimpleNamespace

    design = _tt_design()
    char = _cal_char(design)
    truth = np.ones(design.n_combos)
    truth[:10] = 0.5
    raw, _codes, _clean = _cal_events(design, truth, seed=2)
    unmix = _tt_unmix()
    calls, blocks, seen = [], [], []

    def debarcode(r, af, _key=None):
        calls.append(len(r))
        return unmix(r)
    pnn = ('FSC-A', 'SSC-A') + TT_DETECTORS
    scatter = np.column_stack([np.where(np.arange(len(raw)) % 7 == 0, 80.0, 10.0), np.full(len(raw), 10.0)])
    raw_all = np.column_stack([scatter, raw])
    gate = {'x': 'FSC-A', 'y': 'SSC-A', 'vertices_raw': [[0, 0], [50, 0], [50, 50], [0, 50]]}
    unmix_fn = fc.staged_export_fn(char, _tt_spectra(), TT_FLUORS, lambda r, af, _key=None: unmix(r), [gate], pnn,
                                   {'correction': 'combo', 'debarcode_after_correction': False},
                                   on_block=blocks.append, debarcode_fn=debarcode,
                                   calibration_record=gate, on_scales=seen.append)
    af = SimpleNamespace(spectra=_cal_af())
    unmix_fn.prepare(raw, raw_all, af, 'sample.fcs')
    assert calls == [len(raw)]                                  # one debarcoding pass for the whole sample
    third = len(raw) // 3
    out = np.vstack([unmix_fn(raw[s], raw_all[s], af, 'sample.fcs')['unmixed']
                     for s in (slice(0, third), slice(third, 2 * third), slice(2 * third, len(raw)))])
    assert calls == [len(raw)]                                  # the chunks reuse it
    too_few = fc.TooFewSettings.from_settings(char.settings)
    expected = fc.unmix_staged(raw, char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                               ref_mask=scatter[:, 0] < 50, rescue=False, too_few=too_few, combo_scales=seen[0])
    np.testing.assert_allclose(out, expected.unmixed)          # the chunks as one whole-sample block
    whole = fc.staged_summary(expected, design)
    assert sum(b['rescued_toofew'] for b in blocks) == whole['rescued_toofew']
    assert sum(sum(b['per_combo'].values()) for b in blocks) == sum(whole['per_combo'].values())
    # a block that is not the prepared sample's rows is debarcoded on its own, with its own gated events
    calls.clear()
    alone = unmix_fn(raw[::2], raw_all[::2], af, 'sample.fcs')['unmixed']
    assert calls == [len(raw[::2])]
    own = fc.unmix_staged(raw[::2], char, _tt_spectra(), TT_FLUORS, unmix, unmix, fc.logicle_transform(262144.0),
                          ref_mask=scatter[::2, 0] < 50, rescue=False, too_few=too_few, combo_scales=seen[0])
    np.testing.assert_allclose(alone, own.unmixed)


def test_summary_csv_counts_the_calibrated_combinations(tmp_path):
    import csv
    design = _design(4)
    names = design.combo_names
    scales = {'combos': {names[0]: {'weight': 0.9, 'applied': [0.4, 0.0]}, names[1]: {'weight': 0.0,
                                                                                      'applied': [1.0, 0.0]}},
              'not_identifiable': [names[2]]}
    a = dict(fc.debarcode_summary(np.array([0, 1, 1]), design), combo_scales=scales)
    b = fc.debarcode_summary(np.array([2, 0]), design)
    fc.write_debarcode_summary_csv(tmp_path / 's.csv', [('One', a), ('Two', b)], design)
    with open(tmp_path / 's.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-1] == 'Calibrated combos' and rows[1][-1] == '1/2' and rows[2][-1] == ''
    summary = fc.calibration_summary(scales, design)
    assert summary == {'calibrated': [names[0]], 'identifiable': 2, 'not_identifiable': [names[2]],
                       'largest': [(names[0], 0.4)], 'overridden': []}


def test_summary_csv_adds_tandem_columns(tmp_path):
    import csv
    design = _design(4)
    a = fc.debarcode_summary(np.array([0, 1, 1]), design)
    b = fc.debarcode_summary(np.array([2, 0]), design)
    rec = {'fluor': 'PE-Cy7', 'parent': 'PE', 'tag': 'V5', 'beta': 0.0123456789, 'z': 7.5, 'at_median': 9000.0}
    fc.write_debarcode_summary_csv(tmp_path / 's.csv', [('One', dict(a, tandem=[rec])),
                                                        ('Two', dict(b, tandem=[dict(rec, z=float('nan'))]))],
                                   design)
    with open(tmp_path / 's.csv', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-3:] == ['PE-Cy7 -> PE per unit', 'PE-Cy7 -> PE z', 'PE-Cy7 -> PE at median V5']
    assert rows[0][:len(fc.SUMMARY_COLUMNS)] == list(fc.SUMMARY_COLUMNS)
    assert rows[1][-3:] == ['0.0123457', '7.5', '9000'] and rows[2][-2] == ''


# ---------------------------------------------------------------------------
# Scatter gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cutoff, n_vertices', [(1.3, 4), (0.5, 3)])
def test_singlet_polygon_holds_events_below_the_ratio_cutoff(cutoff, n_vertices):
    from matplotlib.path import Path as MplPath
    poly = fc.singlet_polygon(cutoff, area_max=1000.0, height_max=1000.0)
    assert len(poly) == n_vertices
    rng = np.random.default_rng(7)
    pts = rng.uniform(1, 999, (5000, 2))
    ratio = pts[:, 0] / pts[:, 1]
    clear = np.abs(ratio - cutoff) > 0.01
    inside = MplPath(poly).contains_points(pts)
    np.testing.assert_array_equal(inside[clear], ratio[clear] < cutoff)


def test_singlet_ratio_cutoff_is_the_area_height_quantile():
    area = np.array([10.0, 20, 30, 40])
    height = np.array([10.0, 10, 10, 10])
    assert fc.singlet_ratio_cutoff(area, height, 0.5) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        fc.singlet_polygon(0.0, 1, 1)


def test_landmark_indices():
    tags = np.array([[9, 8, 7, 0], [9, 9, 1, 0], [5, 5, 5, 5], [np.nan, 3, 3, 3]], dtype=float)
    np.testing.assert_array_equal(fc.landmark_indices(tags, n_per_tag=1), [0, 2, 3, 1])
    np.testing.assert_array_equal(fc.landmark_indices(tags, n_per_tag=1, method='per.tag'), [0, 1, 2])
    with pytest.raises(ValueError):
        fc.landmark_indices(tags[:, :2])


def test_landmark_gate_covers_the_robust_core_and_ignores_outliers():
    from matplotlib.path import Path as MplPath
    rng = np.random.default_rng(8)
    core = rng.multivariate_normal([1e5, 5e4], [[1e8, 3e7], [3e7, 4e7]], 900)
    outliers = rng.uniform([2e5, 1.5e5], [2.5e5, 2e5], (100, 2))
    hull = fc.landmark_gate(np.vstack([core, outliers]))
    inside = MplPath(hull).contains_points(np.vstack([core, outliers]), radius=1e-6)
    assert 0.6 <= inside[:900].mean() <= 0.8
    assert not inside[900:].any()
    assert fc.polygon_area(hull) > 0
    with pytest.raises(ValueError):
        fc.landmark_gate(np.column_stack([np.arange(50.0), np.arange(50.0)]))


def test_polygon_area():
    assert fc.polygon_area([[0, 0], [2, 0], [2, 3], [0, 3]]) == 6.0


def test_expand_polygon_scales_about_the_area_centroid():
    square = fc.rectangle_vertices(0.0, 2.0, 0.0, 2.0)
    big = fc.expand_polygon(square, 1.5)
    np.testing.assert_allclose(big, [[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]])
    assert fc.polygon_area(big) == pytest.approx(2.25 * fc.polygon_area(square))
    l_shape = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], dtype=float)
    centre = np.array([4 * 2.0 + 3 * 0.5, 4 * 0.5 + 3 * 2.5]) / 7.0     # area centroid, not the vertex mean
    np.testing.assert_allclose(fc.expand_polygon(l_shape, 2.0), centre + 2.0 * (l_shape - centre))
    np.testing.assert_array_equal(fc.expand_polygon(l_shape, 1.0), l_shape)


def test_polygon_centroid_is_the_area_centroid():
    l_shape = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], dtype=float)
    np.testing.assert_allclose(fc.polygon_centroid(l_shape), np.array([9.5, 9.5]) / 7.0)
    np.testing.assert_allclose(fc.polygon_centroid(l_shape[::-1]), np.array([9.5, 9.5]) / 7.0)
    np.testing.assert_allclose(fc.polygon_centroid([[0, 0], [1, 1], [2, 2]]), [1.0, 1.0])


def test_bhattacharyya_scatter_similarity():
    rng = np.random.default_rng(10)
    x, y = rng.uniform(0, 1, 200_000), rng.uniform(0, 1, 200_000)
    p = fc.scatter_histogram(x, y, (0, 1), (0, 1))
    q = fc.scatter_histogram(x[::-1], y, (0, 1), (0, 1))
    assert fc.bhattacharyya(p, p) == pytest.approx(1.0)
    assert fc.bhattacharyya(p, q) > 0.98
    shifted = fc.scatter_histogram(x * 0.4, y * 0.4, (0, 1), (0, 1))
    far = fc.scatter_histogram(0.6 + x * 0.4, 0.6 + y * 0.4, (0, 1), (0, 1))
    assert fc.bhattacharyya(shifted, far) == 0.0
    assert fc.scatter_histogram(np.array([5.0]), np.array([5.0]), (0, 1), (0, 1)).sum() == 0


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_initial_threshold_methods():
    rng = np.random.default_rng(11)
    bimodal = np.concatenate([rng.normal(0.2, 0.04, 6000), rng.normal(0.7, 0.05, 4000)])
    t, method, dbic = fc.initial_threshold(bimodal)
    assert method == 'mixture' and 0.3 < t < 0.6 and dbic > 10

    unimodal = rng.normal(0.2, 0.04, 8000)
    t, method, _dbic = fc.initial_threshold(unimodal)
    assert method == 'tail' and 0.25 < t < 0.35

    t, method, _dbic = fc.initial_threshold(bimodal, floor_t=0.65)
    assert method == 'floor' and t == 0.65


def test_logicle_transform_round_trip():
    pytest.importorskip('flowkit')
    forward = fc.logicle_transform(262144.0)
    inverse = fc.logicle_inverse(262144.0)
    x = np.array([-500.0, 0.0, 300.0, 5000.0, 2e5])
    np.testing.assert_allclose(inverse(forward(x)), x, rtol=1e-6, atol=1e-6)
    assert np.all(np.diff(forward(x)) > 0)


# ---------------------------------------------------------------------------
# Per-cell AF-only unmixing
# ---------------------------------------------------------------------------

def _af_events(n, seed=12):
    raw = syn.raw_events(n, seed=seed)[:, 3:]
    return raw, syn.fluor_spectra(), syn.af_spectra()


def test_af_only_unmix_python_path():
    raw, fl, af = _af_events(5000)
    out = fc.af_only_unmix(raw, fl, syn.FLUOR_LABELS, af)
    assert out['engine'] == 'python'
    assert out['unmixed'].shape == (5000, syn.N_FLUOR)
    assert out['af_idx'].min() >= 1 and out['af_idx'].max() <= len(af)
    with pytest.raises(ValueError):
        fc.af_only_unmix(raw, fl, syn.FLUOR_LABELS, af[:1])


def test_af_only_unmix_scores_af_with_the_profiles_envelope():
    from honeychrome.controller_components import autospectral_functions as asf
    raw, fl, af = _af_events(5000)
    names = list(syn.FLUOR_LABELS)
    entry, _ = asf.make_envelope_entry(raw[:2000], names, fl, af)
    envelope = asf.build_runtime_envelope(entry, None, names, fl, 1e12)
    out = fc.af_only_unmix(raw, fl, names, af, envelope=envelope)
    expected = asf.apply_af_unmixing(raw, asf.build_af_precomputed(fl, af, envelope), af)
    np.testing.assert_array_equal(out['af_idx'], expected['af_idx'])
    np.testing.assert_array_equal(out['unmixed'], expected['unmixed'])
    joint = fc.af_only_unmix(raw, fl, names, af)
    assert np.any(joint['af_idx'] != out['af_idx'])
    space = fc.debarcoding_space(raw, raw[:500], fl, names, names[:3], af, envelope=envelope)
    np.testing.assert_array_equal(space.af_idx, out['af_idx'])


def test_af_only_unmix_kernel_and_python_agree():
    if not fc.kernel_available():
        pytest.skip('compiled AutoSpectral Optimization kernel not built')
    raw, fl, af = _af_events(20000)
    cpp = fc.af_only_unmix(raw, fl, syn.FLUOR_LABELS, af, n_threads=2, use_kernel=True)
    py = fc.af_only_unmix(raw, fl, syn.FLUOR_LABELS, af)
    assert cpp['engine'] == 'kernel'
    same = cpp['af_idx'] == py['af_idx']
    assert same.mean() >= 0.98
    np.testing.assert_allclose(cpp['unmixed'][same], py['unmixed'][same], rtol=1e-6, atol=1e-6)


def _realistic_af_problem(n, n_det=64, n_fluor=25, n_af=40, seed=13):
    """Gaussian spectra at panel-like dimensions, with per-cell AF."""
    rng = np.random.default_rng(seed)
    ch = np.arange(n_det)
    fl = np.array([np.exp(-0.5 * ((ch - c) / rng.uniform(1.5, 3.0)) ** 2)
                   for c in np.linspace(1, n_det - 2, n_fluor)])
    af = np.array([np.exp(-0.5 * ((ch - c) / w) ** 2) + 0.1
                   for c, w in zip(rng.uniform(0, 20, n_af), rng.uniform(6, 15, n_af))])
    fl /= fl.max(axis=1, keepdims=True)
    af /= af.max(axis=1, keepdims=True)
    abund = rng.gamma(0.8, 2000.0, (n, n_fluor)) * (rng.random((n, n_fluor)) < 0.3)
    raw = abund @ fl + rng.gamma(2.0, 400.0, n)[:, None] * af[rng.integers(0, n_af, n)]
    raw += rng.normal(0.0, 20.0, raw.shape)
    return raw, fl, af, [f'F{i}' for i in range(n_fluor)]


def test_benchmark_af_only_kernel_versus_python():
    """Events/sec for per-cell AF-only unmixing, 200,000 events x 64
    detectors, 25 fluorophores, 40 AF spectra: kernel on 1 thread and on
    CPU count - 1 threads, and the Python path (run with -s to see them)."""
    if not fc.kernel_available():
        pytest.skip('compiled AutoSpectral Optimization kernel not built')
    raw, fl, af, names = _realistic_af_problem(200_000)
    n_threads = max(1, (os.cpu_count() or 2) - 1)
    runs = (('kernel, 1 thread', True, 1), (f'kernel, {n_threads} threads', True, n_threads),
            ('python', False, None))
    rates = {}
    for label, use_kernel, threads in runs:
        start = time.perf_counter()
        out = fc.af_only_unmix(raw, fl, names, af, n_threads=threads, use_kernel=use_kernel)
        rates[label] = len(raw) / (time.perf_counter() - start)
        assert out['engine'] == ('kernel' if use_kernel else 'python')
    lines = '\n'.join(f'  {label:<20} {rate:>12,.0f} events/s' for label, rate in rates.items())
    scaling = rates[f'kernel, {n_threads} threads'] / rates['kernel, 1 thread']
    print(f"\nAF-only unmixing, 200,000 events x 64 detectors, 25 fluorophores, 40 AF spectra:\n"
          f"{lines}\n  kernel thread scaling: {scaling:.2f}x on {n_threads} threads")
    assert all(np.isfinite(list(rates.values())))
