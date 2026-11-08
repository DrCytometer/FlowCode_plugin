"""
flowcode_phenotype_shift.py
---------------------------
Per-guide phenotype shift for the FlowCode Analysis page: is the mix of
phenotypes among a guide's cells different from what that guide's tags
predict in the same mouse and tissue, how large is the shift, and towards
which phenotypes? No Qt.

In plain terms
    The barcoded cells of each analysed population are clustered together
    (FlowSOM, 10 × 10 grid, then consensus metaclusters), so each guide in
    each mouse has a cluster fingerprint: the share of its cells in each
    cluster. The fingerprint is compared with a reference from the same
    mouse; by default the tag model, which is what the guide's three tags
    predict from the other guides of that mouse, so a tag's spillover that
    moves every combination carrying it is not taken for an effect. A
    single mouse is too noisy to show a shift (with 50–100 cells the noise
    alone is about 11–15°), but noise points in a different direction in
    each mouse while a real shift points the same way in every mouse. The
    test therefore asks whether the mice agree on a direction.

The statistic
    For guide g in mouse i, the displacement over the K clusters is

        d_i = (p_guide − p_reference) / (2 √r̄)

    with r̄ the pooled cluster shares of the group (population and sample
    group). Dividing by √r̄ puts every cluster on the same noise scale, so a
    rare cluster is not drowned by a large one. The agreement between mice
    is

        U = mean over pairs of different mice (i ≠ j) of d_i · d_j

    which is about 0 when the mice point in unrelated directions and
    positive when they agree. Its spread under no effect is estimated from
    the products themselves (as if each mouse's sign were flipped at
    random), z = U / SD, and z is read against a t distribution with
    (mice − 1) degrees of freedom (one-sided: only agreement is a shift).
    On JA_007 label shuffles this gave 2.7–3.6 % at p < 0.05 and detected a
    10 % conversion of a guide's cells into one cluster about 80 % of the
    time. The shift angle 2·asin(√|U| / 2), signed like U, is the size: 0°
    for no shift, near 90° when the cells have moved to entirely different
    clusters. P-values are adjusted by Benjamini–Hochberg within each
    population and sample group. The direction is each cluster's mean
    displacement, shown as the log2 ratio of the guide's mean share to the
    reference's.

References (as the marker test)
    'tags'     the tag model (``flowcode_stats.tag_model_residuals`` on the
               cluster shares), within each library;
    'controls' the pooled control combinations of the sample (a control is
               compared with the other controls);
    'all'      every other barcoded cell of the sample.

Spillover correction
    With the correction on (``flowcode_tag_correction``), each sample's
    events are corrected before the populations are gated and clustered:
    the coefficients are estimated in a first pass over the samples, from
    each combination's cells in the fit population (the viable gate) and
    the anchor population, and applied in a second pass. The correction
    needs the live unmix, whose AF library gives the negative model used to
    decide which markers can be corrected.

Kept events
    Only a small part of each sample is ever needed: the barcoded events
    and some untransduced ones inside the *selection gate*, the deepest
    gate above every analysed population (and the fit, eligibility and
    anchor gates) that is drawn only on channels the correction never
    changes (scatter, the viability dye, tags). The correction can move an
    event across a gate drawn on a marker (Foxp3 decides who is a Treg),
    but not across these, so every event that can end up in a population
    after correction is kept. The first run reads (and, live, unmixes) each
    sample once and keeps its barcoded events, up to 50,000 random
    untransduced ones (the 30 % check needs no more) and, for each per-sample axis (Time), the event with
    the largest value, so that per-sample gates are scaled exactly as on the
    whole sample. The kept events are stored in the experiment's cache
    (``cache/flowcode_analysis/shift_events``; on OTB01 about 35 MB per sample)
    and later runs read only them; they are made again only when the
    unmixing, the sample's file or the selection gate and the gates above
    it change. Without the cap on untransduced events, the kept events give
    exactly the results of reading every sample in full.

Libraries
    Counts are kept per tag triplet and named by each sample's library
    when the statistics are made; a guide name used by several libraries is
    one guide (as in the counts test). The tag model is fitted within each
    library.

Deliberate simplifications of this first version
    * one clustering per population, on up to 10,000 barcoded events per
      sample, after correction;
    * the correction is used by this readout only (not by the counts,
      marker test, phenotype populations or exports);
    * the calibration check is five within-sample label shuffles, run on
      demand.

``run`` keeps the events (``prepare_events``), does both passes, the
clustering and the statistics; ``analyse``
recomputes the statistics from stored counts; ``shuffle_check`` and
``composition`` work on a stored result; ``save_result`` and
``load_result`` keep it in the experiment's cache.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import flowcode_analysis as fa
import flowcode_tag_correction as ftc

SEED = 20261105
SOM_X = SOM_Y = 10
SOM_EPOCHS = 10
DEFAULT_K = 20
K_RANGE = (5, 20)
TRAIN_PER_SAMPLE = 10_000
ELIGIBILITY_SUBSAMPLE = 20_000
FIT_SHARE = 0.8
SHUFFLES = 5
UNTRANSDUCED_KEEP = 50_000
EVENTS_VERSION = 1
ALL_SAMPLES = 'All samples'
REFERENCES = {'tags': 'Tag model (other combinations, same tags)', 'controls': 'Control combinations',
              'all': 'All other barcoded cells'}
SHIFT_DEFAULTS = {
    'channels': None,
    'k': DEFAULT_K,
    'reference': 'tags',
    'correction': True,
    'fit_gate': '',
    'eligibility_gate': '',
    'anchor_markers': None,
    'anchor_gate': '',
    'min_cells': 10,
    'min_mice': 3,
    'fdr': 0.1,
    'evidence_p': 0.01,
}
FORMAT_VERSION = 1
RESULT_COLUMNS = ('population', 'group', 'guide', 'mice', 'median_cells', 'angle_deg', 'U', 'z', 'p', 'q',
                  'significant', 'gain_cluster', 'loss_cluster')
_ANCHOR_GATE = re.compile(r'cd8(?!\d)', re.IGNORECASE)
_NOT_ANCHOR = re.compile(r'cd8\s*(-|−|neg)', re.IGNORECASE)


def shift_settings(stored: dict | None) -> dict:
    """The phenotype-shift settings with defaults filled in (values of the
    wrong kind are put back to their defaults)."""
    out = deepcopy(SHIFT_DEFAULTS)
    if not isinstance(stored, dict):
        return out
    for key, value in stored.items():
        if key in out:
            out[key] = deepcopy(value)
    if out['reference'] not in REFERENCES:
        out['reference'] = SHIFT_DEFAULTS['reference']
    if not isinstance(out['k'], int) or not K_RANGE[0] <= out['k'] <= K_RANGE[1]:
        out['k'] = DEFAULT_K
    for key in ('channels', 'anchor_markers'):
        if out[key] is not None and not isinstance(out[key], list):
            out[key] = None
    for key in ('fit_gate', 'eligibility_gate', 'anchor_gate'):
        if not isinstance(out[key], str):
            out[key] = ''
    return out


def cluster_name(k: int) -> str:
    """Display name of 0-based metacluster *k*."""
    return f'C{int(k) + 1}'


# ---------------------------------------------------------------------------
# Gates and channels
# ---------------------------------------------------------------------------

def gate_chain(snapshot, gate: str) -> list:
    """*gate* and its ancestors, top first ('root' left out); empty when the
    gate is not in the hierarchy."""
    try:
        paths = snapshot.gating.find_matching_gate_paths(gate)
    except Exception:
        return []
    if not paths:
        return []
    return [g for g in paths[0] if g != 'root'] + [gate]


def common_chain(snapshot, populations: list) -> list:
    """The gates above (and including) every population: the shared top of
    their chains."""
    chains = [gate_chain(snapshot, p) for p in populations]
    chains = [c for c in chains if c]
    if not chains:
        return []
    out = []
    for level in zip(*chains):
        if len(set(level)) != 1:
            break
        out.append(level[0])
    return out


def gate_channels(snapshot, gate: str) -> list:
    """Channels a gate is drawn on."""
    from honeychrome.controller_components.per_sample_gating_functions import _gate_channels
    try:
        return list(_gate_channels(snapshot.gating.get_gate(gate)))
    except Exception:
        return []


def population_channels(snapshot, populations: list) -> set:
    """Channels of the populations' gates and of every gate above them."""
    out = set()
    for pop in populations:
        for gate in gate_chain(snapshot, pop):
            out.update(gate_channels(snapshot, gate))
    return out


def viable_gate(snapshot, populations: list, viability) -> str | None:
    """The deepest gate above every population drawn on a viability dye."""
    dyes = set(viability or ())
    found = None
    for gate in common_chain(snapshot, populations):
        if dyes & set(gate_channels(snapshot, gate)):
            found = gate
    return found


def anchor_gate(gate_names) -> str | None:
    """The first gate named like a CD8 T-cell gate ('CD8', 'CD8 T cells',
    'CD4-CD8+'), not a CD8-negative one."""
    for name in gate_names:
        text = str(name)
        if _ANCHOR_GATE.search(text) and not _NOT_ANCHOR.search(text):
            return name
    return None


def eligibility_candidates(snapshot, populations: list, fit_gate: str | None) -> list:
    """Gates the eligibility check can use, top first: the fit gate (or
    every event, 'root') and every gate below it above or at every
    population."""
    chain = common_chain(snapshot, populations)
    if fit_gate in chain:
        chain = chain[chain.index(fit_gate):]
    else:
        chain = ['root'] + chain
    return chain


def default_channels(fluors, antigens: dict, tag_fluors, viability, congenic, gate_channel_set) -> list:
    """Clustering channels: the panel's fluorophores without the tags, the
    viability dye, congenic markers and the channels of the gates that
    define the analysed populations."""
    leave = set(tag_fluors) | set(viability or ()) | set(congenic or ()) | set(gate_channel_set or ())
    return [f for f in fluors if f not in leave]


def default_anchored(markers: dict) -> list:
    """Fluorophores of the default anchored markers (Foxp3, Helios) on this
    panel; *markers* is ``flowcode_phenotype.Panel.markers``."""
    out = []
    for name in ftc.DEFAULT_ANCHORED_MARKERS:
        out.extend(markers.get(name, ()))
    return out


# ---------------------------------------------------------------------------
# Counts per guide
# ---------------------------------------------------------------------------

def feature_counts(counts: np.ndarray, samples: list, combos: list, feature_of: dict | None = None,
                   library_of: dict | None = None, triplet_tags: dict | None = None,
                   tag_names: list | None = None) -> dict:
    """
    Triplet counts (samples × combinations × clusters) named by each
    sample's library: ``{'counts'`` (samples × guides × clusters, NaN where
    a guide is not in the sample's library), ``'features'``, ``'library'``
    (per sample) and ``'tags'`` ({library: guides × tags 0/1, from each
    guide's first triplet in that library})``}``. Without *feature_of*
    every sample has every combination.
    """
    counts = np.asarray(counts, dtype=np.float64)
    tag_names = list(tag_names or [])
    triplet_tags = triplet_tags or {}
    if not feature_of:
        features = list(combos)
        tags = np.array([[1.0 if t in triplet_tags.get(c, ()) else 0.0 for t in tag_names] for c in combos])
        return {'counts': counts.copy(), 'features': features, 'library': [None] * len(samples),
                'tags': {None: tags.reshape(len(combos), len(tag_names))}}
    features = []
    for s in samples:
        for c in combos:
            f = (feature_of.get(s) or {}).get(c)
            if f is not None and f not in features:
                features.append(f)
    out = np.full((len(samples), len(features), counts.shape[2]), np.nan)
    libs = [(library_of or {}).get(s) for s in samples]
    first: dict = {}
    for i, s in enumerate(samples):
        fmap = feature_of.get(s) or {}
        for j, c in enumerate(combos):
            f = fmap.get(c)
            if f is None:
                continue
            col = features.index(f)
            if np.isnan(out[i, col, 0]):
                out[i, col] = 0.0
            out[i, col] += counts[i, j]
            first.setdefault((libs[i], f), c)
    tags = {}
    for lib in dict.fromkeys(libs):
        rows = [[1.0 if t in triplet_tags.get(first.get((lib, f)), ()) else 0.0 for t in tag_names]
                for f in features]
        tags[lib] = np.array(rows).reshape(len(features), len(tag_names))
    return {'counts': out, 'features': features, 'library': libs, 'tags': tags}


def displacement(cnt: np.ndarray, libraries: list, tags: dict, reference: str = 'tags', controls=(),
                 features: list | None = None, min_cells: int = SHIFT_DEFAULTS['min_cells']) -> dict:
    """
    Displacements of one group: *cnt* is samples × guides × clusters (NaN
    where a guide is not in the sample's library). Returns 'd' (samples ×
    guides × clusters, NaN where the guide or its reference has fewer than
    *min_cells* cells), 'n' (cells per sample and guide), 'p' (shares) and
    'ref' (reference shares).
    """
    cnt = np.asarray(cnt, dtype=np.float64)
    n = cnt.sum(axis=2)
    enough = np.isfinite(n) & (n >= min_cells)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = cnt / n[:, :, None]
    p[~enough] = np.nan
    nn = np.where(enough, n, 0.0)
    ref = np.full(p.shape, np.nan)
    if reference == 'tags':
        from flowcode_stats import tag_model_residuals
        libs = np.array([str(lib) for lib in libraries], dtype=object)
        for lib in dict.fromkeys(libraries):
            rows = np.flatnonzero(libs == str(lib))
            cols = np.flatnonzero(np.isfinite(n[rows]).any(axis=0))
            if not len(rows) or not len(cols):
                continue
            sub_p = p[np.ix_(rows, cols)]
            res = tag_model_residuals(sub_p, nn[np.ix_(rows, cols)], np.asarray(tags[lib])[cols])
            ref[np.ix_(rows, cols)] = sub_p - res
    else:
        c0 = np.nan_to_num(cnt)
        features = list(features or range(cnt.shape[1]))
        pool = np.zeros(cnt.shape[1], dtype=bool)
        if reference == 'controls':
            pool[[features.index(c) for c in controls if c in features]] = True
        else:
            pool[:] = True
        total = c0[:, pool].sum(axis=1)
        ref_cnt = total[:, None, :] - c0 * pool[None, :, None]
        ref_n = ref_cnt.sum(axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            ref = np.where((ref_n >= min_cells)[:, :, None], ref_cnt / ref_n[:, :, None], np.nan)
        ref[~enough] = np.nan
    pooled = np.nansum(cnt, axis=(0, 1))
    rbar = np.maximum(pooled / max(pooled.sum(), 1e-12), 1e-6)
    d = (p - ref) / (2.0 * np.sqrt(rbar))
    return {'d': d, 'n': n, 'p': p, 'ref': ref}


def shift_test(dv: np.ndarray) -> dict:
    """U, z, the one-sided t (mice − 1) p and the signed angle of one
    guide's displacements *dv* (mice × clusters, every row finite)."""
    s = len(dv)
    gram = dv @ dv.T
    u = (gram.sum() - np.trace(gram)) / (s * (s - 1))
    iu = np.triu_indices(s, 1)
    sd = np.sqrt(4.0 * np.sum(gram[iu] ** 2)) / (s * (s - 1))
    z = u / sd if sd > 0 else 0.0
    angle = float(np.degrees(2 * np.arcsin(min(np.sqrt(abs(u)) / 2, 1.0))) * np.sign(u))
    return {'U': float(u), 'z': float(z), 'p': float(stats.t.sf(z, s - 1)), 'angle_deg': angle}


def group_tests(disp: dict, features: list, min_mice: int = SHIFT_DEFAULTS['min_mice']) -> tuple:
    """(rows, direction rows) of one group: a test per guide with at least
    *min_mice* mice, and each tested guide's per-cluster direction."""
    d, n, p, ref = disp['d'], disp['n'], disp['p'], disp['ref']
    rows, direction = [], []
    for c, feature in enumerate(features):
        ok = np.isfinite(d[:, c]).all(axis=1)
        s = int(ok.sum())
        if s < max(int(min_mice), 2):
            continue
        dv = d[ok, c]
        test = shift_test(dv)
        mean = dv.mean(axis=0)
        cells = float(np.median(n[ok, c]))
        guide = p[ok, c].mean(axis=0)
        reference = np.clip(ref[ok, c], 0.0, None).mean(axis=0)
        eps = 0.5 / max(cells, 1.0)
        log2fc = np.log2((guide + eps) / (reference + eps))
        rows.append({'guide': feature, 'mice': s, 'median_cells': cells, **test,
                     'gain_cluster': cluster_name(int(np.argmax(mean))),
                     'loss_cluster': cluster_name(int(np.argmin(mean)))})
        for k in range(d.shape[2]):
            direction.append({'guide': feature, 'cluster': cluster_name(k), 'mean_displacement': float(mean[k]),
                              'log2fc': float(log2fc[k]), 'guide_share': float(guide[k]),
                              'reference_share': float(reference[k])})
    return rows, direction


def _bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return stats.false_discovery_control(p) if len(p) else p


def analyse(counts: dict, design: dict, settings: dict) -> tuple:
    """
    (results, direction, notes) from stored cluster counts.

    counts : {population: samples × combinations × clusters}.
    design : 'samples', 'group_of', 'combos', 'feature_of', 'library_of',
             'triplet_tags', 'tag_names', 'controls'.
    """
    rows, direction, notes = [], [], []
    samples = list(design['samples'])
    group_of = design['group_of']
    groups = list(dict.fromkeys(group_of[s] for s in samples if group_of.get(s) is not None))
    for pop, cnt in counts.items():
        fc = feature_counts(cnt, samples, design['combos'], design.get('feature_of'), design.get('library_of'),
                            design.get('triplet_tags'), design.get('tag_names'))
        for group in groups:
            idx = [i for i, s in enumerate(samples) if group_of.get(s) == group]
            if len(idx) < int(settings['min_mice']):
                notes.append(f'{pop}, {group}: {len(idx)} sample(s), fewer than the minimum of '
                             f'{settings["min_mice"]} mice; not tested.')
                continue
            disp = displacement(fc['counts'][idx], [fc['library'][i] for i in idx], fc['tags'],
                                settings['reference'], design.get('controls') or (), fc['features'],
                                int(settings['min_cells']))
            g_rows, g_dir = group_tests(disp, fc['features'], int(settings['min_mice']))
            if not g_rows:
                notes.append(f'{pop}, {group}: no guide had {settings["min_cells"]} cells in '
                             f'{settings["min_mice"]} mice.')
                continue
            q = _bh([r['p'] for r in g_rows])
            for r, qv in zip(g_rows, q):
                r.update({'population': pop, 'group': group, 'q': float(qv),
                          'significant': bool(qv < float(settings['fdr']))})
            for r in g_dir:
                r.update({'population': pop, 'group': group})
            rows += g_rows
            direction += g_dir
    results = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
    direction = pd.DataFrame(direction, columns=['population', 'group', 'guide', 'cluster', 'mean_displacement',
                                                 'log2fc', 'guide_share', 'reference_share'])
    return results, direction, notes


def similarity(direction: pd.DataFrame, population: str, group: str, guides=None) -> pd.DataFrame:
    """Cosine similarity of the guides' mean displacements in one group:
    guides pointing the same way share a phenotype effect."""
    sub = direction[(direction['population'] == population) & (direction['group'] == group)]
    m = sub.pivot(index='guide', columns='cluster', values='mean_displacement')
    if guides is not None:
        m = m.reindex([g for g in guides if g in m.index])
    v = m.to_numpy(float)
    norm = np.linalg.norm(v, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = (v @ v.T) / np.outer(norm, norm)
    return pd.DataFrame(cos, index=m.index, columns=m.index)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _sample_rng(sample_key: str, stream: int):
    """A random stream of one sample: the same for that sample whatever
    else is in the run."""
    import zlib
    return np.random.default_rng([SEED, zlib.crc32(str(sample_key).encode('utf-8')), stream])


def _subsample(rng, idx: np.ndarray, cap: int) -> np.ndarray:
    return idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False))


def _display(events: np.ndarray, pnn: list, channels: list, transforms: dict) -> np.ndarray:
    out = np.empty((len(events), len(channels)), dtype=np.float32)
    for j, ch in enumerate(channels):
        out[:, j] = transforms[ch].xform.apply(np.asarray(events[:, pnn.index(ch)], dtype=np.float64))
    return out


def _combos_of(pnn: list, combos: list | None) -> list:
    names = fa.combo_names_in_file(pnn)
    if not names:
        raise ValueError('no FlowCode combination channels')
    if combos is not None and names != combos:
        raise ValueError('its FlowCode combinations differ from the other samples\'')
    return names


def _sample_events(source, key: str, combos: list | None) -> tuple:
    """(events, pnn, combination ids, combinations) of one sample from
    *source* (sample → (events, pnn, analysed or None)); events kept only
    for the per-sample axis ranges get id −1, so no pass uses them."""
    events, pnn, analysed = source(key)
    combos = _combos_of(pnn, combos)
    ids, _has = fa.event_combo_ids(events, pnn)
    if analysed is not None:
        ids = np.where(analysed, ids, -1)
    return events, pnn, ids, combos


# ---------------------------------------------------------------------------
# Kept events
# ---------------------------------------------------------------------------

def selection_gate(snapshot, gates, correctable) -> str | None:
    """The deepest gate above (or at) every gate of *gates* drawn, like all
    its ancestors, on none of the *correctable* channels; None (every
    event) when there is none."""
    leave = set(correctable)
    found = None
    for gate in common_chain(snapshot, [g for g in gates if g and g != 'root']):
        if leave & set(gate_channels(snapshot, gate)):
            break
        found = gate
    return found


def per_sample_channels(snapshot, pnn) -> list:
    """Channels of *pnn* whose axis range is taken from each sample (Time)."""
    return [ch for ch, tr in snapshot.transforms.items() if getattr(tr, 'id', None) == 'default' and ch in pnn]


def keep_events(events: np.ndarray, pnn: list, sample_key: str, snapshot, gate: str | None,
                keep: int | None = UNTRANSDUCED_KEEP) -> dict:
    """
    The events of one sample the readout needs: inside *gate* (every event
    when None), every barcoded event and up to *keep* random untransduced
    ones (all of them with None), plus for each per-sample axis the event
    with the largest value. Returns 'rows' (their positions in the sample,
    in file order), 'events' (float32 when that loses nothing), 'analysed'
    (False for rows kept only for an axis range), 'n_events' and
    'n_inside'.
    """
    pnn = list(pnn)
    combos = _combos_of(pnn, None)
    ids, _has = fa.event_combo_ids(events, pnn)
    if gate is None:
        inside = np.ones(len(events), dtype=bool)
    else:
        inside = np.asarray(snapshot.memberships(sample_key, events, pnn, [gate])[gate], dtype=bool)
    barcoded = np.flatnonzero(inside & (ids >= 1) & (ids <= len(combos)))
    untransduced = np.flatnonzero(inside & (ids == 0))
    if keep is not None and len(untransduced) > keep:
        untransduced = np.sort(_sample_rng(sample_key, 0).choice(untransduced, keep, replace=False))
    analysed_rows = np.union1d(barcoded, untransduced)
    ranges = [int(np.argmax(events[:, pnn.index(ch)])) for ch in per_sample_channels(snapshot, pnn)
              if len(events)]
    rows = np.union1d(analysed_rows, np.asarray(ranges, dtype=np.int64))
    kept = np.asarray(events[rows])
    small = kept.astype(np.float32)
    if np.array_equal(small.astype(kept.dtype), kept):
        kept = small
    return {'rows': rows, 'events': kept, 'analysed': np.isin(rows, analysed_rows), 'n_events': int(len(events)),
            'n_inside': int(inside.sum())}


def events_path(experiment_dir) -> Path:
    """Folder of the kept events, in the experiment's cache."""
    return Path(experiment_dir) / 'cache' / 'flowcode_analysis' / 'shift_events'


def events_key(source_key: str, gate: str | None, chain_signature: str, keep: int | None) -> str:
    """File name (without extension) of one sample's kept events, from what
    decides them: its unmixed values (*source_key*), the selection gate,
    that gate's and its ancestors' boundaries for the sample, and the cap."""
    import hashlib
    import json

    text = json.dumps([EVENTS_VERSION, source_key, gate, chain_signature, keep])
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:40]


def _write_kept(path: Path, kept: dict, pnn: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp.npz')
    np.savez(tmp, events=kept['events'], analysed=kept['analysed'], pnn=np.array(list(pnn), dtype=str),
             counts=np.array([kept['n_events'], kept['n_inside']], dtype=np.int64))
    tmp.replace(path)


def _read_kept(path: Path) -> tuple:
    with np.load(path, allow_pickle=False) as z:
        return (np.asarray(z['events'], dtype=np.float64), [str(c) for c in z['pnn']],
                np.asarray(z['analysed'], dtype=bool), tuple(int(c) for c in z['counts']))


def prepare_events(job: dict, load, cache: dict, progress, total: int) -> tuple:
    """
    Make sure every sample's kept events are in the cache (*cache*: 'dir',
    'source_keys' {sample: key of its unmixed values}, 'keep'), reading in
    full (*load*) only the samples whose kept events are missing; files no
    longer used by these samples are removed. Returns (source for the
    passes, {'gate', 'read', 'kept', 'events'}).
    """
    snapshot, samples = job['snapshot'], list(job['samples'])
    gates = (list(job['populations']) + [job.get('fit_gate'), job.get('anchor_gate')]
             + [job.get('eligibility_gate') or None])
    gate = selection_gate(snapshot, gates, job['marker_fluors'])
    keep = cache.get('keep', UNTRANSDUCED_KEEP)
    folder = Path(cache['dir'])
    files = {}
    for s in samples:
        chain = snapshot.chain_signature(s, [gate]) if gate else ''
        files[s] = folder / f"{events_key(cache['source_keys'][s], gate, chain, keep)}.npz"
    missing = [s for s in samples if not files[s].exists()]
    for i, s in enumerate(missing):
        progress(i, total, f'reading {Path(s).stem}')
        events, pnn = load(s)
        _write_kept(files[s], keep_events(events, pnn, s, snapshot, gate, keep), pnn)
        del events
    used = {f.name for f in files.values()}
    for old in folder.glob('*.npz'):
        if old.name not in used:
            try:
                old.unlink()
            except OSError:
                pass
    info = {'gate': gate or 'root', 'read': len(missing), 'kept': {}, 'events': {}}

    def source(key):
        events, pnn, analysed, (n_events, _n_inside) = _read_kept(files[key])
        info['kept'][key] = int(analysed.sum())
        info['events'][key] = n_events
        return events, pnn, analysed
    return source, info


def estimate_correction(job: dict, load, progress, step: int, total: int) -> dict:
    """
    First pass: pool each combination's cells in the fit and anchor
    populations and each candidate gate's untransduced cells, choose the
    eligibility gate and estimate the coefficients
    (``flowcode_tag_correction.estimate``). *load* is sample → (events,
    pnn, analysed or None). Returns {'table', 'eligibility', 'gates',
    'notes', 'combos'}. Each sample draws its fit cells and its
    untransduced cells from random streams of its own, so a sample's draws
    do not depend on the other samples, nor its fit cells on how many
    untransduced cells were kept.
    """
    samples, group_of, snapshot = job['samples'], job['group_of'], job['snapshot']
    fluors, markers = list(job['fluors']), list(job['marker_fluors'])
    fit_gate, anchor = job['fit_gate'], job['anchor_gate']
    candidates = list(job['eligibility_candidates'])
    chosen = job.get('eligibility_gate') or None
    if chosen and chosen not in candidates:
        candidates.append(chosen)
    wanted = [g for g in dict.fromkeys([fit_gate, anchor] + candidates) if g and g != 'root']
    anchored = [f for f in job['anchored'] if f in markers] if anchor else []
    fit_cells, anchor_cells = {}, {}
    shares = {g: {} for g in candidates}
    barcoded_in = dict.fromkeys(candidates, 0)
    notes, combos = [], None
    for i, key in enumerate(samples):
        progress(step + i, total, f'correction: {Path(key).stem}')
        group = group_of[key]
        events, pnn, ids, combos = _sample_events(load, key, combos)
        n = len(combos)
        fit_rng, untransduced_rng = (_sample_rng(key, stream) for stream in (1, 2))
        mem = snapshot.memberships(key, events, pnn, wanted) if wanted else {}
        everything = np.ones(len(events), dtype=bool)
        missing = [g for g in wanted if g not in mem]
        if missing:
            raise ValueError(f'{Path(key).stem}: gate(s) {", ".join(missing)} could not be applied')
        in_fit = mem.get(fit_gate, everything) if fit_gate else everything
        in_anchor = mem.get(anchor) if anchor else None
        mcols = [pnn.index(f) for f in markers]
        for c, combo in enumerate(combos):
            tag_cols = [pnn.index(f) for f in job['combo_fluors'][combo]]
            thr = [job['thresholds'][f] for f in job['combo_fluors'][combo]]
            for mask, store in ((in_fit, fit_cells), (in_anchor, anchor_cells)):
                if mask is None:
                    continue
                idx = _subsample(fit_rng, np.flatnonzero(mask & (ids == c + 1)), ftc.CELLS_PER_SAMPLE)
                cells = store.setdefault((group, combo), ftc.FitCells())
                cells.add(events[np.ix_(idx, mcols)], ftc.tag_amounts(events[np.ix_(idx, tag_cols)], thr))
        barcoded = (ids >= 1) & (ids <= n) & in_fit
        model = job['negative_model'](key)
        if model is None:
            notes.append(f'{Path(key).stem}: no negative model, so it does not count towards eligibility.')
        fcols = [pnn.index(f) for f in fluors]
        af = pnn.index(job['af_channel']) if job.get('af_channel') in pnn else None
        for gate in candidates:
            in_gate = everything if gate == 'root' else mem[gate]
            barcoded_in[gate] += int((barcoded & in_gate).sum())
            if model is None:
                continue
            u = _subsample(untransduced_rng, np.flatnonzero(in_gate & (ids == 0)), ELIGIBILITY_SUBSAMPLE)
            ab = events[np.ix_(u, fcols)]
            af_values = events[u, af] if af is not None else np.zeros(len(u))
            pi = fpositivity(ab, af_values, [fluors.index(f) for f in markers], model)
            for j, f in enumerate(markers):
                shares[gate].setdefault(group, {}).setdefault(f, []).append(ftc.negative_share(pi[:, j]))
        del events
    top = candidates[0] if candidates else 'root'
    if chosen is None:
        chosen = top
        for gate in candidates:
            if barcoded_in.get(top) and barcoded_in[gate] >= FIT_SHARE * barcoded_in[top]:
                chosen = gate
    eligibility, eligible = [], {}
    for group in dict.fromkeys(group_of[s] for s in samples):
        table = ftc.eligible_markers((shares.get(chosen) or {}).get(group) or {f: [] for f in markers})
        table.insert(0, 'group', group)
        eligibility.append(table)
        eligible[group] = set(table.loc[table['eligible'], 'fluor'])
    eligibility = pd.concat(eligibility, ignore_index=True) if eligibility else pd.DataFrame()
    table = ftc.estimate(fit_cells, anchor_cells, markers, eligible, anchored,
                         {c: tuple(job['combo_tags'].get(c, ())) for c in combos or []},
                         evidence_p=float(shift_settings(job['settings'])['evidence_p']))
    if job['anchored'] and not anchor:
        notes.append('Anchored markers were fitted on all of the fit population: no CD8 T-cell gate was found '
                     '(set one in Anchor gate).')
    share = {g: (barcoded_in[g] / barcoded_in[top] if barcoded_in.get(top) else float('nan')) for g in candidates}
    return {'table': table, 'eligibility': eligibility, 'notes': notes, 'combos': combos,
            'gates': {'fit': fit_gate or 'root', 'eligibility': chosen, 'anchor': anchor,
                      'anchored': anchored, 'barcoded_share': share}}


def fpositivity(abundances, af_abundance, columns, model):
    """``flowcode_phenotype.positivity`` (imported when needed)."""
    import flowcode_phenotype as fp
    return fp.positivity(abundances, af_abundance, columns, model)


def run(job: dict, load, progress=None, cache: dict | None = None) -> dict:
    """
    The phenotype-shift readout of every population and sample group.

    job  : 'samples', 'group_of' (sample → group), 'populations' (gate
           names), 'snapshot' (``SampleGating``), 'transforms' (display
           scale), 'channels' (clustering), 'settings' (``shift_settings``),
           'combo_fluors' and 'combo_tags' (per file combination, in its
           tag order), 'thresholds' (tag fluorophore → raw threshold),
           'fluors' (the negative model's fluorophores), 'marker_fluors'
           (markers that may be corrected), 'af_channel', 'negative_model'
           (sample → ``NegativeModel`` or None), 'fit_gate',
           'eligibility_candidates', 'eligibility_gate' ('' for automatic),
           'anchor_gate', 'anchored', and for the statistics 'feature_of',
           'library_of', 'triplet_tags', 'tag_names', 'controls'.
    load : sample → (events, pnn), FlowCode-unmixed.
    cache: 'dir', 'source_keys' and 'keep' for ``prepare_events``: each
           sample is read in full only when its kept events are not in the
           cache; without *cache* every sample is read in full in both
           passes.

    Returns the result dict that ``save_result`` stores.
    """
    progress = progress or (lambda *_a: None)
    settings = shift_settings(job['settings'])
    samples, group_of, snapshot = list(job['samples']), job['group_of'], job['snapshot']
    populations, channels = list(job['populations']), list(job['channels'])
    if not channels:
        raise ValueError('No clustering channels are chosen.')
    correct = bool(settings['correction'])
    total = len(samples) * (2 if correct else 1) + len(populations)
    notes = []
    kept = None
    if cache is not None:
        total += len(samples)
        source, kept = prepare_events(job, load, cache, progress, total)
        start = len(samples)
    else:
        def source(key):
            events, pnn = load(key)
            return events, pnn, None
        start = 0
    correction = None
    if correct:
        correction = estimate_correction(job, source, progress, start, total)
        notes += correction['notes']
    step = start + (len(samples) if correct else 0)
    rng = np.random.default_rng(SEED)
    held = {pop: [] for pop in populations}
    combos = None
    fixers = {}
    for i, key in enumerate(samples):
        progress(step + i, total, Path(key).stem)
        events, pnn, ids, combos = _sample_events(source, key, combos)
        n = len(combos)
        if correction is not None:
            group = group_of[key]
            if group not in fixers:
                fixers[group] = ftc.Correction.from_table(correction['table'], group, pnn, combos,
                                                          job['combo_fluors'], job['thresholds'])
            if fixers[group]:
                events = fixers[group].apply(events, ids)
        mem = snapshot.memberships(key, events, pnn, populations)
        barcoded = (ids >= 1) & (ids <= n)
        for pop in populations:
            if pop not in mem:
                raise ValueError(f'{Path(key).stem}: gate "{pop}" could not be applied')
            m = np.asarray(mem[pop], dtype=bool) & barcoded
            held[pop].append((_display(events[m], pnn, channels, job['transforms']), (ids[m] - 1).astype(np.int16)))
        del events
    _bundled_on_path()
    import flowsom_consensus as fsc

    k = int(settings['k'])
    counts, clusters, cells = {}, {}, {}
    for j, pop in enumerate(populations):
        progress(step + len(samples) + j, total, f'clustering {pop}')
        parts = [x for x, _ids in held[pop]]
        train = np.vstack([x[_subsample(rng, np.arange(len(x)), TRAIN_PER_SAMPLE)] for x in parts if len(x)])
        if len(train) < SOM_X * SOM_Y * 2:
            notes.append(f'{pop}: only {len(train)} barcoded events, too few to cluster.')
            continue
        weights = fsc.train_som(train.astype(np.float64), SOM_X, SOM_Y, SOM_EPOCHS, seed=SEED)
        meta = np.asarray(fsc.consensus_metacluster(weights, k, seed=SEED), dtype=np.int64)
        try:
            best_k, pac = fsc.consensus_stability(weights, range(K_RANGE[0], K_RANGE[1] + 1), seed=SEED)
        except Exception:
            best_k, pac = None, {}
        labels = [meta[fsc.assign_to_nodes(weights, x.astype(np.float64))] if len(x) else np.zeros(0, np.int64)
                  for x in parts]
        cnt = np.zeros((len(samples), len(combos), k))
        for i, (lab, (_x, ids)) in enumerate(zip(labels, held[pop])):
            np.add.at(cnt[i], (ids.astype(np.int64), lab), 1.0)
        everything = np.vstack(parts)
        all_labels = np.concatenate(labels)
        medians = pd.DataFrame([np.median(everything[all_labels == c], axis=0) if (all_labels == c).any()
                                else np.full(len(channels), np.nan) for c in range(k)],
                               index=[cluster_name(c) for c in range(k)], columns=channels)
        counts[pop] = cnt
        clusters[pop] = {'codebook': weights, 'metaclusters': meta, 'medians': medians,
                         'sizes': np.bincount(all_labels, minlength=k).astype(np.int64),
                         'suggested_k': best_k, 'pac': {int(a): float(b) for a, b in pac.items()}}
        offsets = np.cumsum([0] + [len(lab) for lab in labels]).astype(np.int64)
        cells[pop] = {'ids': np.concatenate([ids for _x, ids in held[pop]]).astype(np.int16),
                      'labels': all_labels.astype(np.int16), 'offsets': offsets}
        del held[pop]
    design = {'samples': samples, 'group_of': {s: group_of[s] for s in samples}, 'combos': combos or [],
              'feature_of': job.get('feature_of'), 'library_of': job.get('library_of'),
              'triplet_tags': {c: list(t) for c, t in (job.get('triplet_tags') or {}).items()},
              'tag_names': list(job.get('tag_names') or []), 'controls': list(job.get('controls') or [])}
    results, direction, stat_notes = analyse(counts, design, settings)
    progress(total, total, '')
    return {'settings': settings, 'channels': channels, 'populations': [p for p in populations if p in counts],
            'design': design, 'counts': counts, 'clusters': clusters, 'cells': cells, 'results': results,
            'direction': direction, 'correction': correction, 'notes': notes + stat_notes, 'calibration': None,
            'stat_notes': stat_notes, 'kept': kept}


def _bundled_on_path():
    from flowcode_stats import _bundled_plugins_on_path
    _bundled_plugins_on_path()


# ---------------------------------------------------------------------------
# On a stored result
# ---------------------------------------------------------------------------

def shuffle_check(result: dict, reps: int = SHUFFLES, seed: int = SEED, progress=None) -> pd.DataFrame:
    """
    Calibration check: the guide labels are shuffled among the barcoded
    cells of each sample (keeping every guide's cell number and removing
    any guide effect) *reps* times, and the statistics are run again. Per
    population and group: the tests, the share with p < 0.05 (about 5 % or
    less when the test is calibrated) and the shuffles with any q below the
    FDR threshold.
    """
    rng = np.random.default_rng(seed)
    settings, design = result['settings'], result['design']
    n_samples, n_combos = len(design['samples']), len(design['combos'])
    rows = []
    for rep in range(reps):
        if progress is not None:
            progress(rep, reps, f'shuffle {rep + 1}')
        counts = {}
        for pop, cell in result['cells'].items():
            k = result['counts'][pop].shape[2]
            cnt = np.zeros((n_samples, n_combos, k))
            off = cell['offsets']
            for i in range(n_samples):
                ids = rng.permutation(cell['ids'][off[i]:off[i + 1]].astype(np.int64))
                np.add.at(cnt[i], (ids, cell['labels'][off[i]:off[i + 1]].astype(np.int64)), 1.0)
            counts[pop] = cnt
        res, _direction, _notes = analyse(counts, design, settings)
        res['shuffle'] = rep + 1
        rows.append(res)
    if progress is not None:
        progress(reps, reps, '')
    if not rows:
        return pd.DataFrame()
    allr = pd.concat(rows, ignore_index=True)
    out = []
    for (pop, group), sub in allr.groupby(['population', 'group'], sort=False):
        out.append({'population': pop, 'group': group, 'shuffles': reps, 'tests': len(sub),
                    'pct_p_below_0_05': 100.0 * float((sub['p'] < 0.05).mean()) if len(sub) else np.nan,
                    'shuffles_with_a_hit': int(sub.groupby('shuffle')['significant'].any().sum())})
    return pd.DataFrame(out)


def composition(result: dict, population: str, group: str, guide: str) -> pd.DataFrame:
    """One guide's cluster shares in each mouse of a group, with its
    reference's: one row per sample and cluster ('sample', 'cluster',
    'guide_share', 'reference_share', 'cells')."""
    design, settings = result['design'], result['settings']
    samples = list(design['samples'])
    fc = feature_counts(result['counts'][population], samples, design['combos'], design.get('feature_of'),
                        design.get('library_of'), design.get('triplet_tags'), design.get('tag_names'))
    if guide not in fc['features']:
        raise KeyError(guide)
    idx = [i for i, s in enumerate(samples) if design['group_of'].get(s) == group]
    disp = displacement(fc['counts'][idx], [fc['library'][i] for i in idx], fc['tags'], settings['reference'],
                        design.get('controls') or (), fc['features'], int(settings['min_cells']))
    c = fc['features'].index(guide)
    rows = []
    for r, i in enumerate(idx):
        if not np.isfinite(disp['d'][r, c]).all():
            continue
        for k in range(disp['p'].shape[2]):
            rows.append({'sample': samples[i], 'cluster': cluster_name(k), 'guide_share': disp['p'][r, c, k],
                         'reference_share': disp['ref'][r, c, k], 'cells': disp['n'][r, c]})
    return pd.DataFrame(rows, columns=['sample', 'cluster', 'guide_share', 'reference_share', 'cells'])


def counts_table(result: dict) -> pd.DataFrame:
    """Long table of the cluster counts: population, sample, combination,
    cluster, events."""
    rows = []
    design = result['design']
    for pop, cnt in result['counts'].items():
        s_idx, c_idx, k_idx = np.nonzero(cnt)
        for i, j, k in zip(s_idx, c_idx, k_idx):
            rows.append({'population': pop, 'sample': design['samples'][i], 'combination': design['combos'][j],
                         'cluster': cluster_name(k), 'events': int(cnt[i, j, k])})
    return pd.DataFrame(rows, columns=['population', 'sample', 'combination', 'cluster', 'events'])


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def result_path(experiment_dir) -> Path:
    """Folder of the last phenotype-shift run, in the experiment's cache."""
    return Path(experiment_dir) / 'cache' / 'flowcode_analysis' / 'shift'


def save_result(path, result: dict, inputs: str, when=None):
    """Write *result* (``run``) to the bundle at *path* with *inputs*, the
    digest of what it was made from."""
    from datetime import datetime
    _bundled_on_path()
    import drc_persist

    when = when or datetime.now()
    drc_persist.save_bundle(Path(path), {'format_version': FORMAT_VERSION, 'inputs': inputs,
                                         'saved': when.isoformat(timespec='seconds'), 'result': result})


def load_result(path) -> tuple | None:
    """(result, inputs digest, time saved) of the run at *path*, or None."""
    _bundled_on_path()
    import drc_persist

    payload = drc_persist.load_bundle(Path(path))
    if not payload or payload.get('format_version') != FORMAT_VERSION:
        return None
    return payload['result'], payload.get('inputs'), payload.get('saved')


def write_tables(folder, result: dict) -> list:
    """The readout's CSV tables in *folder*; returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []

    def put(name, frame):
        if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
            return
        p = folder / name
        frame.to_csv(p, index=isinstance(frame.index, pd.Index) and frame.index.name is not None)
        written.append(p)

    put('shift_results.csv', result['results'])
    put('shift_directions.csv', result['direction'])
    put('shift_cluster_counts.csv', counts_table(result))
    medians = []
    for pop, cl in result['clusters'].items():
        m = cl['medians'].copy()
        m.insert(0, 'events', cl['sizes'])
        m.insert(0, 'cluster', m.index)
        m.insert(0, 'population', pop)
        medians.append(m.reset_index(drop=True))
    if medians:
        put('shift_cluster_medians.csv', pd.concat(medians, ignore_index=True))
    corr = result.get('correction')
    if corr:
        put('shift_correction.csv', corr['table'])
        put('shift_correction_eligibility.csv', corr['eligibility'])
    put('shift_calibration.csv', result.get('calibration'))
    return written
