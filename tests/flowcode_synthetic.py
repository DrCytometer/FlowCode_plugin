"""
flowcode_synthetic.py
---------------------
Synthetic FlowCode experiment for the FlowCode plugin tests: five epitope
tags and one other marker, Gaussian spectra, a three-spectrum AF library,
a backbone in which most cells carry three tags, and two unstained samples
(one with the backbone's scatter, one shifted). No Qt.

``make_events`` builds the arrays; ``write_experiment`` writes them as raw
FCS files and returns the experiment description the UI tests turn into a
stand-in controller.
"""

from pathlib import Path

import numpy as np

TAGS = ('HA', 'V5', 'AU1', 'S', 'C')
TAG_FLUORS = ('BV421', 'BV510', 'PE', 'APC', 'BUV395')
OTHER_FLUORS = ('FITC',)
FLUORS = TAG_FLUORS + OTHER_FLUORS
ANTIGENS = dict(zip(FLUORS, ('HA', 'V5 tag', 'AU1', 'S', 'C', 'CD4')))
N_DET = 16
DETECTORS = tuple(f'D{i + 1}-A' for i in range(N_DET))
SCATTER = ('FSC-A', 'FSC-H', 'SSC-A')
RAW_PNN = ('Time',) + SCATTER + DETECTORS
MAGNITUDE_CEILING = 4194304.0
POSITIVE_LEVEL = 6000.0

BACKBONE = 'Raw/Backbone.fcs'
UNSTAINED_MATCHED = 'Raw/Unstained cells.fcs'
UNSTAINED_SHIFTED = 'Raw/Unstained thymus.fcs'
OTHER_SAMPLE = 'Raw/Mouse 1.fcs'
AF_PROFILE = 'Unstained cells AutoSpectral AF'


def spectra() -> np.ndarray:
    ch = np.arange(N_DET)
    centres = np.linspace(1.0, N_DET - 2.0, len(FLUORS))
    s = np.array([np.exp(-0.5 * ((ch - c) / 1.1) ** 2) for c in centres])
    return s / s.max(axis=1, keepdims=True)


def af_spectra() -> np.ndarray:
    ch = np.arange(N_DET)
    s = np.array([np.exp(-0.5 * ((ch - c) / w) ** 2) + 0.1 for c, w in ((3, 5.0), (6, 6.0), (4, 8.0))])
    return s / s.max(axis=1, keepdims=True)


def combos():
    import itertools
    return [tuple(c) for c in itertools.combinations(range(len(TAGS)), 3)]


def _scatter(rng, n, centre=(1.0e6, 6.0e5), doublet_frac=0.08):
    area = rng.normal(centre[0], 0.12 * centre[0], n)
    ratio = rng.normal(1.05, 0.03, n)
    doublet = rng.random(n) < doublet_frac
    area[doublet] *= 1.8
    ratio[doublet] = rng.normal(1.9, 0.1, doublet.sum())
    height = area / ratio
    ssc = rng.normal(centre[1], 0.15 * centre[1], n)
    out = np.column_stack([np.abs(area), np.abs(height), np.abs(ssc)])
    return np.minimum(out, MAGNITUDE_CEILING), doublet


def make_events(kind: str, n: int, seed: int) -> dict:
    """
    kind: 'backbone', 'unstained' or 'shifted' (unstained with a different
    scatter). Returns ``raw`` (n x len(RAW_PNN)), ``tags`` (n x 5 true tag
    abundances), ``combo`` (combo index, -1 untransduced) and ``doublet``.
    """
    rng = np.random.default_rng(seed)
    centre = (1.8e6, 3.0e5) if kind == 'shifted' else (1.0e6, 6.0e5)
    scatter, doublet = _scatter(rng, n, centre)
    abund = np.zeros((n, len(FLUORS)))
    combo = np.full(n, -1)
    if kind == 'backbone':
        triplets = combos()
        transduced = rng.random(n) < 0.8
        combo[transduced] = rng.integers(0, len(triplets), transduced.sum())
        for i in np.flatnonzero(transduced):
            for t in triplets[combo[i]]:
                abund[i, t] = rng.lognormal(np.log(POSITIVE_LEVEL), 0.35)
        abund[:, len(TAGS)] = rng.gamma(1.0, 800.0, n)
        bright = transduced & (abund[:, :len(TAGS)].sum(axis=1) > 3.3 * POSITIVE_LEVEL)
        scatter[bright, 0] = rng.normal(1.0e6, 5.0e4, bright.sum())
        scatter[bright, 1] = scatter[bright, 0] / rng.normal(1.05, 0.03, bright.sum())
        scatter[bright, 2] = rng.normal(6.0e5, 4.0e4, bright.sum())
    af = af_spectra()[rng.integers(0, 3, n)] * rng.gamma(4.0, 150.0, n)[:, None]
    detectors = abund @ spectra() + af + rng.normal(0.0, 15.0, (n, N_DET))
    time = np.sort(rng.uniform(0, 100, n))
    raw = np.column_stack([time, scatter, detectors])
    return {'raw': raw, 'tags': abund[:, :len(TAGS)], 'combo': combo, 'doublet': doublet}


def raw_settings() -> dict:
    pnn = list(RAW_PNN)
    return {
        'raw_samples_subdirectory': 'Raw',
        'event_channels_pnn': pnn,
        'whitelisted_pnn': pnn,
        'scatter_channel_ids': [1, 2, 3],
        'scatter_param': ['FSC-A', 'SSC-A'],
        'time_channel_id': 0,
        'event_id_channel_id': None,
        'fluorescence_channel_ids': list(range(4, 4 + N_DET)),
        'n_scatter_channels': 3,
        'cytometer': 'Cytek Aurora',
        'cytometer_db_col': 'Aurora',
        'magnitude_ceiling': MAGNITUDE_CEILING,
        'width_ceiling': 1024.0,
        'default_ceiling': 100.0,
        'width_channels': [],
        'channel_pnr': [100.0] + [MAGNITUDE_CEILING] * (len(pnn) - 1),
    }


def unmixed_pnn() -> list:
    return ['Time'] + list(SCATTER) + list(FLUORS) + ['AF Abundance', 'AF Index']


def unmixed_settings() -> dict:
    pnn = unmixed_pnn()
    return {
        'unmixed_samples_subdirectory': 'Unmixed',
        'event_channels_pnn': pnn,
        'time_channel_id': 0,
        'event_id_channel_id': None,
        'scatter_channel_ids': [1, 2, 3],
        'n_scatter_channels': 3,
        'fluorescence_channel_ids': list(range(4, 4 + len(FLUORS))),
        'n_fluorophore_channels': len(FLUORS),
        'magnitude_ceiling': MAGNITUDE_CEILING,
        'width_ceiling': 1024.0,
        'default_ceiling': 100.0,
        'width_channels': [],
    }


def samples() -> dict:
    keys = (BACKBONE, UNSTAINED_MATCHED, UNSTAINED_SHIFTED, OTHER_SAMPLE)
    return {
        'all_samples': {k: Path(k).stem for k in keys},
        'unstained_samples': [],
        'single_stain_controls': [],
        'sample_af_profiles': {},
    }


def process() -> dict:
    s = spectra()
    return {
        'unmixing_matrix': np.linalg.solve(s @ s.T, s).tolist(),
        'spillover': np.eye(len(FLUORS)).tolist(),
        'spectral_model': [{'label': f, 'antigen': ANTIGENS[f]} for f in FLUORS],
        'profiles': {f: s[i].tolist() for i, f in enumerate(FLUORS)},
        'af_profiles': {
            AF_PROFILE: {'spectra': af_spectra().tolist(), 'source_fcs': UNSTAINED_MATCHED,
                         'channel_names': list(DETECTORS)},
        },
    }


SAMPLE_EVENTS = {
    BACKBONE: ('backbone', 30000, 1),
    UNSTAINED_MATCHED: ('unstained', 15000, 2),
    UNSTAINED_SHIFTED: ('shifted', 15000, 3),
    OTHER_SAMPLE: ('backbone', 5000, 4),
}


def write_experiment(experiment_dir) -> dict:
    """Write every sample as a raw FCS file; returns {key: make_events dict}."""
    from honeychrome.controller_components.functions import write_fcs

    experiment_dir = Path(experiment_dir)
    out = {}
    for key, (kind, n, seed) in SAMPLE_EVENTS.items():
        events = make_events(kind, n, seed)
        keywords = {'$CYT': 'Aurora'}
        for i, name in enumerate(RAW_PNN, start=1):
            keywords[f'$P{i}N'] = name
            keywords[f'$P{i}B'] = '32'
            keywords[f'$P{i}E'] = '0,0'
            keywords[f'$P{i}R'] = '100' if name == 'Time' else str(int(MAGNITUDE_CEILING))
        path = experiment_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        write_fcs(np.asarray(events['raw'], dtype=np.float64), keywords, path)
        out[key] = events
    return out


# ---------------------------------------------------------------------------
# FlowCode-unmixed events for the phenotype-shift readout
# ---------------------------------------------------------------------------

SHIFT_TAGS = ('HA', 'V5', 'AU1', 'S', 'C', 'FLAG', 'VSVg', 'Ollas')
SHIFT_TAG_FLUORS = ('BV421', 'BV510', 'PE', 'APC', 'BUV395', 'BUV496', 'BV650', 'PE-Cy5')
SHIFT_MARKERS = ('CD3', 'CD8', 'M1', 'M2', 'M3')
SHIFT_MARKER_FLUORS = ('BV605', 'BV711', 'FITC', 'PerCP', 'APC-Cy7')
SHIFT_VIABILITY = 'Viability'
SHIFT_FLUORS = SHIFT_TAG_FLUORS + SHIFT_MARKER_FLUORS + (SHIFT_VIABILITY,)
SHIFT_THRESHOLD = 1000.0
SHIFT_NEGATIVE_SD = 100.0
SHIFT_POSITIVE = 5000.0
SHIFT_CD8_SHARE = 0.35
SHIFT_TIME = 1000.0
# T-cell states on M1, M2, M3 (1 = positive) and their baseline shares
SHIFT_STATES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1))
SHIFT_SHARES = (0.4, 0.3, 0.2, 0.1)


def shift_triplets() -> list:
    """The 56 triplets of the eight shift tags (tag indices)."""
    import itertools
    return [tuple(c) for c in itertools.combinations(range(len(SHIFT_TAGS)), 3)]


def shift_combos() -> list:
    """(name, tag names) of the 56 tag triplets, named G0..G55."""
    return [(f'G{i}', tuple(SHIFT_TAGS[t] for t in c)) for i, c in enumerate(shift_triplets())]


def shift_pnn() -> list:
    names = [n for n, _t in shift_combos()]
    return (['Time', 'FSC-A', 'SSC-A'] + list(SHIFT_FLUORS) + ['AF Abundance', 'FlowCode ID']
            + [f'FC {n}' for n in names])


def shift_sample(seed: int, cells_per_combo: int = 100, untransduced: int = 3000, convert: dict | None = None,
                 artefact: dict | None = None, knockout: dict | None = None) -> np.ndarray:
    """
    One mouse's FlowCode-unmixed events (columns ``shift_pnn``): viable and
    dead cells, untransduced T cells and non-T cells, and *cells_per_combo*
    barcoded T cells of each of the 56 combinations. T cells take one of
    ``SHIFT_STATES`` on M1–M3 (baseline ``SHIFT_SHARES``) and are CD8+ with
    probability ``SHIFT_CD8_SHARE``.

    convert  : {combination index: (share, state)} — that share of the
               combination's cells is put in the given state (a phenotype
               shift).
    artefact : {combination index: k} — M2 of the combination's cells rises
               by k × (amount of its first tag × amount of its second tag),
               a FRET-like, brightness-dependent spillover.
    knockout : {combination index: share} — that share of the combination's
               M3-positive cells lose M3 (a knockout of a marker most cells
               are negative for).
    """
    rng = np.random.default_rng(seed)
    pnn = shift_pnn()
    triplets = shift_triplets()
    n_combo = len(triplets)
    ids = [np.full(cells_per_combo, c + 1) for c in range(n_combo)]
    n_bar = n_combo * cells_per_combo
    n_t_un = untransduced // 2
    n_dead = untransduced // 5
    n = n_bar + untransduced + n_dead
    ids = np.concatenate(ids + [np.zeros(untransduced + n_dead, dtype=int)])
    col = {name: j for j, name in enumerate(pnn)}
    ev = np.zeros((n, len(pnn)))
    clock = np.random.default_rng([seed, 1])
    ev[:, col['Time']] = clock.uniform(0.0, SHIFT_TIME * (1.0 + clock.random()), n)
    ev[:, col['FSC-A']] = rng.normal(1.0e5, 1.0e4, n)
    ev[:, col['SSC-A']] = rng.normal(5.0e4, 5.0e3, n)
    for f in SHIFT_FLUORS:
        ev[:, col[f]] = rng.normal(0.0, SHIFT_NEGATIVE_SD, n)
    ev[:, col['AF Abundance']] = rng.gamma(4.0, 50.0, n)
    is_t = np.zeros(n, dtype=bool)
    is_t[:n_bar + n_t_un] = True
    dead = np.zeros(n, dtype=bool)
    dead[n_bar + untransduced:] = True
    is_t[dead] = rng.random(dead.sum()) < 0.5
    ev[dead, col[SHIFT_VIABILITY]] += SHIFT_POSITIVE * 2
    ev[is_t, col['BV605']] += rng.normal(SHIFT_POSITIVE, 600.0, is_t.sum())
    cd8 = is_t & (rng.random(n) < SHIFT_CD8_SHARE)
    ev[cd8, col['BV711']] += rng.normal(SHIFT_POSITIVE, 600.0, cd8.sum())
    state = np.where(is_t, rng.choice(len(SHIFT_STATES), size=n, p=SHIFT_SHARES), -1)
    for c, (share, to) in (convert or {}).items():
        m = np.flatnonzero(ids == c + 1)
        moved = m[rng.random(len(m)) < share]
        state[moved] = to
    for c, share in (knockout or {}).items():
        m = np.flatnonzero((ids == c + 1) & np.isin(state, [i for i, s in enumerate(SHIFT_STATES) if s[2]]))
        lost = m[rng.random(len(m)) < share]
        state[lost] = [i for i, s in enumerate(SHIFT_STATES) if s == (1, 0, 0)][0]
    for j, f in enumerate(('FITC', 'PerCP', 'APC-Cy7')):
        pos = np.array([SHIFT_STATES[s][j] if s >= 0 else 0 for s in state], dtype=bool)
        ev[pos, col[f]] += rng.normal(SHIFT_POSITIVE, 600.0, pos.sum())
    amounts = np.zeros((n, len(SHIFT_TAGS)))
    for c, trip in enumerate(triplets):
        m = ids == c + 1
        for t in trip:
            amounts[m, t] = rng.lognormal(np.log(3.0), 0.5, m.sum())
            ev[m, col[SHIFT_TAG_FLUORS[t]]] += amounts[m, t] * SHIFT_THRESHOLD
    for c, k in (artefact or {}).items():
        m = ids == c + 1
        a, b = triplets[c][0], triplets[c][1]
        ev[m, col['PerCP']] += k * amounts[m, a] * amounts[m, b]
    ev[:, col['FlowCode ID']] = ids
    for c in range(n_combo):
        ev[ids == c + 1, col[f'FC G{c}']] = 1.0
    return ev


def shift_negative_model():
    """``flowcode_phenotype.NegativeModel`` of the shift events: the
    negative SD of every fluorophore, no AF term, no spread."""
    import flowcode_phenotype as fp
    var = np.full(len(SHIFT_FLUORS), SHIFT_NEGATIVE_SD ** 2)
    return fp.NegativeModel(tuple(SHIFT_FLUORS), np.vstack([var, np.zeros_like(var), np.zeros_like(var)]), None)


def shift_gating(time_gate: bool = False):
    """``SampleGating`` of the shift events: Viable (viability dye low) →
    T (CD3+) → CD8 (CD8+), on logicle display transforms; with
    *time_gate*, under a Stable gate on Time (0 to 0.9 × ``SHIFT_TIME``) on
    a per-sample axis, as the main window gates Time."""
    from flowkit import GatingStrategy, gates
    from honeychrome.controller_components.functions import define_rectangle_gate
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating
    from honeychrome.controller_components.transform import Transform

    tr = {}
    for ch in ('FSC-A', 'SSC-A'):
        tr[ch] = Transform()
        tr[ch].set_transform(id=0, limits=[0, 1])
    for f in SHIFT_FLUORS:
        tr[f] = Transform()
        tr[f].set_transform(id=1, limits=[-0.1, 1.0])
    lo, hi = float(tr[SHIFT_VIABILITY].xform.apply(np.array([2000.0]))[0]), 1.0
    g = GatingStrategy()
    top = ('root',)
    if time_gate:
        from flowkit import Dimension
        tr['Time'] = Transform()
        tr['Time'].set_transform(id='default', limits=[0, 5 * SHIFT_TIME])
        g.add_gate(gates.RectangleGate('Stable', [Dimension('Time', range_min=0, range_max=0.9 * SHIFT_TIME)]), top)
        top = ('root', 'Stable')
    g.add_gate(gates.RectangleGate('Viable', list(define_rectangle_gate(
        (-0.1, -0.1), (lo + 0.1, 1.1), SHIFT_VIABILITY, 'FSC-A', tr))), top)
    g.add_gate(gates.RectangleGate('T', list(define_rectangle_gate(
        (lo, -0.1), (hi - lo, 1.1), 'BV605', 'FSC-A', tr))), top + ('Viable',))
    g.add_gate(gates.RectangleGate('CD8', list(define_rectangle_gate(
        (lo, -0.1), (hi - lo, 1.1), 'BV711', 'FSC-A', tr))), top + ('Viable', 'T'))
    return SampleGating(gating=g, transforms=tr, overrides={})


# ---------------------------------------------------------------------------
# Immune states on the phenotype-shift events
# ---------------------------------------------------------------------------

FSEA_MARKERS = {'M1': 'FITC', 'M2': 'PerCP', 'M3': 'APC-Cy7'}
FSEA_PLANTED = 'Planted programme'
FSEA_ARTEFACT = 'M2 programme'
FSEA_TARGET = 'Target-only programme'


def fsea_spec(target_guide: str = 'G7') -> dict:
    """A ``flowcode_fsea.panel_library`` result for the shift events, with
    the population 'T' as the root and three states:

    * FSEA_PLANTED: more M1+ M3+ T cells, more M1+ and M3+, fewer M2+ (the
      state ``shift_sample``'s conversion into M1+ M3+ moves along); its M3
      row is the gene of *target_guide*;
    * FSEA_ARTEFACT: more M2+ M1− T cells, more M2+ (what an M2 artefact
      fakes) and fewer M3+, plus an abundance row on M2 alone (too thin);
    * FSEA_TARGET: more M3+ (gene of *target_guide*), M1+ and fewer M2+,
      which has fewer than three members without the target's row.
    """
    import pandas as pd
    import flowcode_fsea as ffsea

    sets = {'T': ([], []), 'M1+ M3+ T': (['M1', 'M3'], []), 'M2+ M1- T': (['M2'], ['M1']),
            'M2+ T': (['M2'], [])}

    def row(state, cell, marker, sign, gene=''):
        kind = 'Marker' if marker else 'Abund'
        feature = f'marker:{cell}|{marker}' if marker else f'abund:{cell}'
        label = f'{marker}+ in {cell}' if marker else f'{cell} share'
        return {'state': state, 'feature': feature, 'label': label, 'kind': kind, 'cell_type': cell,
                'marker': marker, 'sign': sign, 'weight': 'High', 'gene': gene,
                'signs': len(sets[cell][0]) + len(sets[cell][1]), 'description': '', 'citation': ''}

    rows = pd.DataFrame([
        row(FSEA_PLANTED, 'M1+ M3+ T', '', 1), row(FSEA_PLANTED, 'T', 'M1', 1),
        row(FSEA_PLANTED, 'T', 'M3', 1, target_guide), row(FSEA_PLANTED, 'T', 'M2', -1),
        row(FSEA_ARTEFACT, 'M2+ M1- T', '', 1), row(FSEA_ARTEFACT, 'T', 'M2', 1), row(FSEA_ARTEFACT, 'T', 'M3', -1),
        row(FSEA_ARTEFACT, 'M2+ T', '', 1),
        row(FSEA_TARGET, 'T', 'M3', 1, target_guide), row(FSEA_TARGET, 'T', 'M1', 1),
        row(FSEA_TARGET, 'T', 'M2', -1),
    ], columns=list(ffsea.ROW_COLUMNS))
    left = pd.DataFrame([(FSEA_PLANTED, 'M4 in T', 'M4 is not stained')], columns=['state', 'row', 'reason'])
    return {'populations': {'T': {'root': 'T', 'rows': rows, 'sets': sets, 'markers': sorted(FSEA_MARKERS),
                                  'left_out': left}},
            'fluor_of': dict(FSEA_MARKERS), 'markers': sorted(FSEA_MARKERS), 'notes': [], 'library': '',
            'species': 'mouse'}
