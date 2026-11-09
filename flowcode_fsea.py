"""
flowcode_fsea.py
----------------
Flow Set Enrichment Analysis per guide for the FlowCode Analysis page: does
a guide move its cells along a known immune state (for example the effector
Treg programme), compared with what its tags predict in the same mouse?
No Qt.

In plain terms
    An immune state from the FSEA library is a short list of expected
    changes, e.g. effector Treg expansion: more effector Tregs; more CD44+,
    ICOS+, KLRG1+ and CD103+ cells among Tregs; fewer CD62L+. For each guide
    and mouse, every item on the list is measured on the guide's cells and
    compared with the guide's reference in the same mouse (by default the
    tag model, as in the phenotype shift). A state scores high when its
    items move in the listed directions, together and consistently across
    mice. "Consistently across mice" is what makes it a test rather than a
    description.

What is measured (features)
    Everything is read inside the analysed population (the root, a gate
    ticked on Data), from the per-event positivity scores of the phenotype
    populations (``flowcode_phenotype.positivity``), after the phenotype
    shift's spillover correction when that is on. With soft counts (sums of
    per-event memberships) for the guide's cells in one mouse:

        Abund row   logit of a cell type's share of the root;
                    membership = product over the cell type's stained
                    signs beyond the root's of π (positive sign) or 1 − π;
        Marker row  logit of the share marker+ within a cell type:
                    Σ(membership × π_marker) / Σ membership.

    Half an event is added to each count. Rows are matched to the panel as
    the cell-type database resolves them (stained markers only; a cell type
    the panel cannot tell apart from another definition is not counted).

The statistics
    For guide g in mouse i, d_i = logit(guide share) − logit(reference
    share) for every feature. Per feature, the one-sample moderated t of
    the d_i (Smyth 2004), with a variance prior shared by all guides'
    features of the population and sample group. A state's score is the
    mean of its members' signed z-scores (z with the t's tail
    probability, sign + for an UP row, − for a DOWN row). Its p-value comes
    from a rotation test (limma's roast; Wu et al. 2010): the guide's
    per-mouse differences are rotated at random, which keeps the
    correlation between features but removes any consistent direction,
    and the score is recomputed each time. A "mixed" version (mean z²)
    asks whether the members change in any direction. P-values are
    adjusted by Benjamini–Hochberg over guides × states within each
    population and sample group.

    Each guide is scored twice: with all rows, and without the rows whose
    gene (the library's ``gene_mapping``) is the guide's own target, so a
    state does not score only because the knocked-out protein is gone. The
    second is the main result; a state whose call depends on the target's
    rows is flagged.

Minimums
    A feature is kept for a guide when, over the guide's mice, the median of
    the smaller of its expected marker+ and marker− counts reaches
    ``min_count`` (default 5). The expected counts are what the reference
    predicts for the guide's cells: reference share × the guide's cells in
    the feature's cell type, and (1 − reference share) × those cells. They
    are taken from the reference rather than the guide's own counts, so a
    guide that removes nearly all of a rare positive population is still
    tested on it, while a feature on too few cells is not. An Abund row needs at least ``min_abund_signs`` (default 2)
    stained signs beyond the root, so a cell type does not rest on a single
    marker. A state needs ``min_members`` (default 3) tested members; one
    under ``flag_members`` (5) is flagged as small. Cells per mouse, mice
    and the reference are the phenotype shift's settings.

Deliberate simplifications of this first version
    * positivity shares only (no marker means);
    * no competitive test (in label-shuffle and spike-in checks it added no calls);
    * each guide is compared with its reference within each sample; how a
      guide's state differs between tissues comes later.

The positivity scores are made in the phenotype shift's run
(``score_sample``, called from ``flowcode_phenotype_shift.run``) and kept
with its result, so the statistics can be run again (``analyse``) when a
statistics setting changes.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261108
MIN_PARENT_WEIGHT = 5.0
VERSION_MAIN, VERSION_ALL = 'no_target', 'all'
SPECIES = ('mouse', 'human')
FSEA_DEFAULTS = {
    'enabled': True,
    'species': 'mouse',
    'roots': {},
    'states': None,
    'min_members': 3,
    'flag_members': 5,
    'min_count': 5.0,
    'min_abund_signs': 2,
    'rotations': 9999,
}
# Settings that decide which markers are scored: changing one needs a new run.
SCORING_KEYS = ('enabled', 'species', 'roots')
ROTATION_RANGE = (999, 99999)
SHUFFLES = 5
_TIE = 1e-12
RESULT_COLUMNS = ('population', 'group', 'guide', 'state', 'mice', 'members', 'members_all', 'target_rows',
                  'score', 'p', 'q', 'p_mixed', 'q_mixed', 'significant', 'score_all', 'p_all', 'q_all',
                  'significant_all', 'small_set', 'depends_on_target')
MEMBER_COLUMNS = ('population', 'group', 'guide', 'feature', 'label', 'kind', 'cell_type', 'marker', 'mice',
                  'mean', 't', 'z')
ROW_COLUMNS = ('state', 'feature', 'label', 'kind', 'cell_type', 'marker', 'sign', 'weight', 'gene', 'signs',
               'description', 'citation')


def fsea_settings(stored: dict | None) -> dict:
    """The FSEA settings with defaults filled in (values of the wrong kind
    are put back to their defaults)."""
    out = deepcopy(FSEA_DEFAULTS)
    if not isinstance(stored, dict):
        return out
    for key, value in stored.items():
        if key in out:
            out[key] = deepcopy(value)
    if out['species'] not in SPECIES:
        out['species'] = FSEA_DEFAULTS['species']
    if not isinstance(out['roots'], dict):
        out['roots'] = {}
    if out['states'] is not None and not isinstance(out['states'], list):
        out['states'] = None
    out['enabled'] = bool(out['enabled'])
    for key, low in (('min_members', 2), ('flag_members', 2), ('min_abund_signs', 1)):
        if not isinstance(out[key], int) or out[key] < low:
            out[key] = FSEA_DEFAULTS[key]
    if not isinstance(out['min_count'], (int, float)) or out['min_count'] < 0:
        out['min_count'] = FSEA_DEFAULTS['min_count']
    if not isinstance(out['rotations'], int) or not ROTATION_RANGE[0] <= out['rotations'] <= ROTATION_RANGE[1]:
        out['rotations'] = FSEA_DEFAULTS['rotations']
    return out


def scoring_key(settings: dict) -> dict:
    """The part of the FSEA settings a run's scores depend on."""
    s = fsea_settings(settings)
    return {k: s[k] for k in SCORING_KEYS}


def _bundled_on_path():
    from flowcode_stats import _bundled_plugins_on_path
    _bundled_plugins_on_path()


# ---------------------------------------------------------------------------
# Library rows the panel can score
# ---------------------------------------------------------------------------

def _stained(panel, tag_fluors) -> set:
    tags = set(tag_fluors or ())
    return {m for m, f in panel.markers.items() if len(f) == 1 and f[0] not in tags}


def _definitions(species: str) -> list:
    import flowcode_phenotype as fp
    return [d for d in fp.load_definitions() if d.species == species]


def _definition_finder(definitions: list):
    """name → database definition, by the definition's own name or any name
    the FSEA cell-type table gives the same cell type."""
    _bundled_on_path()
    import ag_fsea_library as fl
    by_name = {d.cell_type.lower(): d for d in definitions}
    cell_types = fl.load_cell_types()
    names = {}
    for _i, r in cell_types.iterrows():
        group = [r['cell_type']] + list(r['synonyms'])
        for n in group:
            names.setdefault(fl._norm(n), group)

    def find(name: str):
        if not name:
            return None
        if name.lower() in by_name:
            return by_name[name.lower()]
        for n in names.get(fl._norm(name), ()):
            if n.lower() in by_name:
                return by_name[n.lower()]
        return None
    return find


def root_choices(species: str = 'mouse') -> list:
    """The database cell types a population can be read as."""
    return [d.cell_type for d in _definitions(species)]


def default_root(gate: str, species: str = 'mouse') -> str | None:
    """The database cell type a gate's name denotes (by the cell type's name
    or a synonym in the FSEA cell-type table), or None."""
    d = _definition_finder(_definitions(species))(str(gate or '').strip())
    return d.cell_type if d is not None else None


def library_states(species: str = 'mouse', library_path=None) -> list:
    """The immune states of the library for *species*, in library order."""
    _bundled_on_path()
    import ag_fsea_library as fl
    lib, _sha, _msg = fl.load_library(library_path, species=species)
    return list(dict.fromkeys(lib['immune_state']))


def panel_library(panel, tag_fluors, roots: dict, species: str = 'mouse', library_path=None) -> dict:
    """
    The library rows each population can score on *panel*
    (``flowcode_phenotype.Panel``; the FlowCode tag fluorophores are not
    markers). *roots* is {population gate: database cell type or None}.

    A Marker row is scored when its marker is stained (once, not a tag)
    and its cell type is the root or resolves inside it; an Abund row when
    its cell type resolves inside the root, is not the root, and the panel
    can tell it apart from every other definition. Inside: the cell type's
    stained positive signs include the root's, and no sign contradicts it.

    Returns {'populations': {gate: {'root', 'rows' (``ROW_COLUMNS``),
    'sets' {cell type: (positive, negative) markers beyond the root's},
    'markers', 'left_out' (state, row, reason)}}, 'fluor_of' {marker:
    fluorophore}, 'markers' (every scored marker), 'notes', 'library'
    (SHA-256), 'species'}.
    """
    _bundled_on_path()
    import ag_fsea_library as fl
    import flowcode_phenotype as fp

    stained = _stained(panel, tag_fluors)
    definitions = _definitions(species)
    find = _definition_finder(definitions)
    groups, aliases = fp.phenotype_groups(definitions, panel)
    lib, sha, _messages = fl.load_library(library_path, species=species)

    def status(defn):
        res = groups.get(aliases.get(defn.key, defn.key))
        if res is None or isinstance(res, fp.Unresolved):
            return 'unresolved'
        return 'grouped' if len(res.definition.members) > 1 else 'resolved'

    def signs(defn):
        return ({m for m in defn.positive if m in stained}, {m for m in defn.negative if m in stained})

    def inside(defn, root):
        rp, rn = signs(root)
        dp, dn = signs(defn)
        return rp <= dp and not (rn & dp) and not (rp & dn)

    out, notes, used = {}, [], set()
    for gate, root_name in roots.items():
        root = find(root_name) if root_name else None
        if root is None:
            notes.append(f'{gate}: no cell-type definition is chosen for this population, so it has no immune '
                         'states (choose one in the immune-state settings).')
            out[gate] = {'root': None, 'rows': pd.DataFrame(columns=list(ROW_COLUMNS)), 'sets': {}, 'markers': [],
                         'left_out': pd.DataFrame(columns=['state', 'row', 'reason'])}
            continue
        rp, rn = signs(root)
        rows, sets, markers, left = [], {}, set(), []
        for _i, r in lib.iterrows():
            cell, kind = r['cell_type_canonical'], r['feature_kind']
            what = f"{cell} share" if kind == 'Abund' else f"{r['marker_canonical']} in {cell}"
            defn = find(cell)
            if defn is None:
                left.append((r['immune_state'], what, 'no definition in the cell-type database'))
                continue
            if defn.key == root.key:
                rel = (set(), set())
            elif inside(defn, root) and status(defn) != 'unresolved':
                dp, dn = signs(defn)
                rel = (dp - rp, dn - rn)
            else:
                left.append((r['immune_state'], what, f'not inside {root.cell_type} on this panel'))
                continue
            if kind == 'Abund':
                if defn.key == root.key:
                    left.append((r['immune_state'], what, 'the population itself'))
                    continue
                if status(defn) != 'resolved':
                    left.append((r['immune_state'], what, 'the panel cannot tell it apart from another cell type'))
                    continue
                if not (rel[0] or rel[1]):
                    left.append((r['immune_state'], what, 'no stained marker beyond the population'))
                    continue
                feature, label, marker = f'abund:{cell}', f'{cell} share', ''
            else:
                marker = r['marker_canonical']
                if marker not in stained:
                    left.append((r['immune_state'], what, f'{marker} is not stained'))
                    continue
                feature = f'marker:{cell}|{marker}'
                label = f'{marker}+ in {cell}'
                markers.add(marker)
            sets[cell] = (sorted(rel[0]), sorted(rel[1]))
            markers |= rel[0] | rel[1]
            rows.append({'state': r['immune_state'], 'feature': feature, 'label': label, 'kind': kind,
                         'cell_type': cell, 'marker': marker, 'sign': 1 if r['direction'] == 'UP' else -1,
                         'weight': r['weight'], 'gene': str(r.get('gene_mapping') or ''),
                         'signs': len(rel[0]) + len(rel[1]), 'description': r['description'],
                         'citation': r['citation']})
        frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
        frame = _drop_conflicts(frame)
        out[gate] = {'root': root.cell_type, 'rows': frame, 'sets': sets, 'markers': sorted(markers),
                     'left_out': pd.DataFrame(left, columns=['state', 'row', 'reason'])}
        used |= markers
    fluor_of = {m: panel.markers[m][0] for m in sorted(used)}
    return {'populations': out, 'fluor_of': fluor_of, 'markers': sorted(used), 'notes': notes, 'library': sha,
            'species': species}


def _drop_conflicts(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per state and feature; a feature asked for in both directions
    by one state is left out of it."""
    if rows.empty:
        return rows
    keep = []
    for (_state, _feature), g in rows.groupby(['state', 'feature'], sort=False):
        if g['sign'].nunique() == 1:
            keep.append(g.index[0])
    return rows.loc[sorted(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Positivity scores (inside the phenotype shift's run)
# ---------------------------------------------------------------------------

def new_scores(spec: dict) -> dict:
    """An empty store for ``score_sample``."""
    return {pop: {'samples': [], 'ids': [], 'pi': []} for pop, p in spec['populations'].items() if p['markers']}


def score_sample(store: dict, spec: dict, key: str, events: np.ndarray, pnn: list, ids: np.ndarray, n_combos: int,
                 memberships: dict, model, fluors: list, af_channel: str):
    """
    Add one sample's barcoded events of each population to *store*: their
    combination (0-based) and the positivity of each scored marker
    (``spec['markers']`` order). *events* are the (corrected) FlowCode-
    unmixed events, *memberships* the populations' event masks, *model* the
    sample's ``NegativeModel`` over *fluors*.
    """
    import flowcode_phenotype as fp
    if not store:
        return
    need = [spec['fluor_of'][m] for m in spec['markers']]
    missing = [f for f in need if f not in fluors]
    if missing:
        raise ValueError(f'markers without a negative-model fluorophore: {", ".join(missing)}')
    fcols = [pnn.index(f) for f in fluors]
    af = events[:, pnn.index(af_channel)] if af_channel in pnn else np.zeros(len(events))
    barcoded = (ids >= 1) & (ids <= n_combos)
    pi = None
    for pop in store:
        m = np.asarray(memberships[pop], dtype=bool) & barcoded
        if pi is None:
            pi = fp.positivity(events[:, fcols], af, [fluors.index(f) for f in need], model).astype(np.float32)
        store[pop]['samples'].append(key)
        store[pop]['ids'].append((ids[m] - 1).astype(np.int16))
        store[pop]['pi'].append(pi[m])


def pack_scores(store: dict) -> dict:
    """*store* as flat arrays for saving: {pop: {'samples', 'ids', 'pi',
    'offsets'}}."""
    out = {}
    for pop, s in store.items():
        sizes = [len(i) for i in s['ids']]
        n_markers = s['pi'][0].shape[1] if s['pi'] else 0
        out[pop] = {'samples': list(s['samples']),
                    'ids': np.concatenate(s['ids']).astype(np.int16) if s['ids'] else np.zeros(0, np.int16),
                    'pi': (np.vstack(s['pi']).astype(np.float32) if s['pi']
                           else np.zeros((0, n_markers), np.float32)),
                    'offsets': np.cumsum([0] + sizes).astype(np.int64)}
    return out


def _items(packed: dict) -> list:
    """[(sample, ids, pi)] of one population's packed scores."""
    off = packed['offsets']
    return [(s, packed['ids'][off[i]:off[i + 1]], packed['pi'][off[i]:off[i + 1]])
            for i, s in enumerate(packed['samples'])]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _membership(pi: np.ndarray, markers: list, pos, neg) -> np.ndarray:
    w = np.ones(len(pi))
    for m in pos:
        w *= pi[:, markers.index(m)]
    for m in neg:
        w *= 1.0 - pi[:, markers.index(m)]
    return w


def features(items: list, n_combos: int, rows: pd.DataFrame, sets: dict, markers: list,
             ids_override: list | None = None) -> tuple:
    """(features, num, den, root): soft counts per sample, combination and
    feature (num: in the numerator, den: in the denominator) and the
    root's cells per sample and combination."""
    feats = list(dict.fromkeys(rows['feature']))
    s = len(items)
    num = np.zeros((s, n_combos, len(feats)))
    den = np.zeros((s, n_combos, len(feats)))
    root = np.zeros((s, n_combos))
    for i, (_key, ids, pi) in enumerate(items):
        if ids_override is not None:
            ids = ids_override[i]
        ok = ids >= 0
        ids, pi = ids[ok].astype(np.int64), np.asarray(pi[ok], dtype=np.float64)
        root[i] = np.bincount(ids, minlength=n_combos)[:n_combos]
        cache = {}
        for j, f in enumerate(feats):
            kind, rest = f.split(':', 1)
            cell, mk = (rest, None) if kind == 'abund' else rest.split('|')
            if cell not in cache:
                cache[cell] = _membership(pi, markers, *sets[cell])
            w = cache[cell]
            if mk is None:
                num[i, :, j] = np.bincount(ids, w, minlength=n_combos)[:n_combos]
                den[i, :, j] = root[i]
            else:
                num[i, :, j] = np.bincount(ids, w * pi[:, markers.index(mk)], minlength=n_combos)[:n_combos]
                den[i, :, j] = np.bincount(ids, w, minlength=n_combos)[:n_combos]
    return feats, num, den, root


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def _combo_names(design: dict, sample: str) -> list:
    """The guide name of each combination in *sample* (None: not in its
    library)."""
    feature_of = design.get('feature_of')
    if not feature_of:
        return list(design['combos'])
    fmap = feature_of.get(sample) or {}
    return [fmap.get(c) for c in design['combos']]


def differences(num, den, root, samples: list, design: dict, reference: str, min_cells: int) -> dict:
    """
    Per-mouse logit differences from the reference: 'd', 'p' (guide share)
    and 'ref' (reference share), samples × combinations × features, NaN
    where the guide has fewer than *min_cells* root cells or fewer than
    ``MIN_PARENT_WEIGHT`` soft cells in the feature's cell type.
    """
    from flowcode_stats import tag_model_residuals
    p = (num + 0.5) / (den + 1.0)
    p[den < MIN_PARENT_WEIGHT] = np.nan
    p[root < min_cells] = np.nan
    ref = np.full(p.shape, np.nan)
    combos = design['combos']
    if reference == 'tags':
        tag_names = list(design.get('tag_names') or [])
        triplets = design.get('triplet_tags') or {}
        tags = np.array([[1.0 if t in triplets.get(c, ()) else 0.0 for t in tag_names] for c in combos])
        tags = tags.reshape(len(combos), len(tag_names))
        library_of = design.get('library_of') or {}
        libraries = {}
        for i, s in enumerate(samples):
            libraries.setdefault(str(library_of.get(s)), []).append(i)
        for rows in libraries.values():
            sub = p[rows]
            n = np.where(root[rows] >= min_cells, root[rows], 0.0)
            res = tag_model_residuals(sub, n, tags)
            ref[rows] = np.clip(sub - res, 1e-4, 1 - 1e-4)
    else:
        controls = set(design.get('controls') or ())
        for i, s in enumerate(samples):
            names = _combo_names(design, s)
            pool = np.array([n is not None and (reference == 'all' or n in controls) for n in names])
            enough = root[i] >= min_cells
            pool &= enough
            rn = (num[i] * pool[:, None]).sum(axis=0)[None, :] - num[i] * pool[:, None]
            rd = (den[i] * pool[:, None]).sum(axis=0)[None, :] - den[i] * pool[:, None]
            r = (rn + 0.5) / (rd + 1.0)
            r[rd < MIN_PARENT_WEIGHT] = np.nan
            ref[i] = r
        ref[np.isnan(p)] = np.nan
    return {'d': _logit(p) - _logit(ref), 'p': p, 'ref': ref}


def guide_matrices(diff: dict, num, den, samples: list, design: dict, feats: list, min_mice: int,
                   min_count: float) -> dict:
    """
    {guide: (mice × features differences, features, samples)} of one group,
    guides merged by name across libraries. A feature is kept for a guide
    when it has a value in every one of the guide's mice and, over those
    mice, the median of the smaller of its expected marker+ and marker−
    counts (reference share × the guide's cells in the cell type, and its
    complement) reaches *min_count*; a guide needs *min_mice* mice and a
    kept feature. *num* is not used for the minimum: what the guide does to
    the feature must not decide whether it is tested.
    """
    d = diff['d']
    ref = diff['ref']
    minor = np.minimum(ref, 1.0 - ref) * den
    rows, small, keys = {}, {}, {}
    for i, s in enumerate(samples):
        for c, name in enumerate(_combo_names(design, s)):
            if name is None or not np.isfinite(d[i, c]).any():
                continue
            rows.setdefault(name, []).append(d[i, c])
            small.setdefault(name, []).append(minor[i, c])
            keys.setdefault(name, []).append(s)
    out = {}
    for name, r in rows.items():
        mat = np.vstack(r)
        keep = np.isfinite(mat).all(axis=0)
        if min_count > 0:
            keep &= np.median(np.vstack(small[name]), axis=0) >= min_count
        if mat.shape[0] >= min_mice and keep.any():
            out[name] = (mat[:, keep], [f for f, k in zip(feats, keep) if k], keys[name])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def variance_prior(mats: dict) -> tuple:
    """(d0, s0²) of the moderated t, from every guide's features in *mats*;
    (0, 0) (ordinary t) when there is nothing to estimate it from."""
    from honeychrome.controller_components import moderated_stats as ms
    s2, dfs = [], []
    for mat, _f, _k in mats.values():
        if mat.shape[0] < 2:
            continue
        s2.append(mat.var(axis=0, ddof=1))
        dfs.append(np.full(mat.shape[1], mat.shape[0] - 1.0))
    if not s2:
        return 0.0, 0.0
    d0, s0_2 = ms.fit_prior_variance(np.concatenate(s2), np.concatenate(dfs))
    if not (np.isfinite(s0_2) and s0_2 > 0) or np.isnan(d0):
        return 0.0, 0.0
    return float(d0), float(s0_2)


def _moderated_z(mean, s2, n: int, d0: float, s0_2: float) -> tuple:
    """(t, z) of the one-sample moderated t of means *mean* and variances
    *s2* (n mice); z has the t's tail probability."""
    from flowcode_pathways import t_to_z
    if np.isinf(d0):
        post, df = np.full(np.shape(s2), s0_2), np.inf
    else:
        post, df = (d0 * s0_2 + (n - 1) * s2) / (d0 + n - 1), d0 + n - 1
    t = mean / np.sqrt(np.maximum(post, 1e-12) / n)
    return t, t_to_z(t, df)


def guide_statistics(mat: np.ndarray, d0: float, s0_2: float, directions: np.ndarray) -> dict:
    """
    Per-feature mean, t and z of one guide's mice × features differences,
    and z under each rotation (*directions*: unit vectors, one row per
    rotation). A rotation keeps each feature's sum of squares and turns its
    mean into u·d/√n with the same u for every feature, so the correlation
    between features is kept.
    """
    n = mat.shape[0]
    mean = mat.mean(axis=0)
    ss = (mat ** 2).sum(axis=0)
    t, z = _moderated_z(mean, mat.var(axis=0, ddof=1), n, d0, s0_2)
    m_star = (directions @ mat) / np.sqrt(n)
    s2_star = np.maximum(ss[None, :] - n * m_star ** 2, 0.0) / (n - 1)
    _t, z_star = _moderated_z(m_star, s2_star, n, d0, s0_2)
    return {'mean': mean, 't': t, 'z': z, 'z_star': z_star}


def state_test(st: dict, columns: np.ndarray, signs: np.ndarray) -> dict:
    """Score, directional p, mixed score and mixed p of one state's members
    (*columns* of the guide's features, *signs* +1 UP / −1 DOWN)."""
    z, z_star = st['z'][columns], st['z_star'][:, columns]
    score = float(np.mean(signs * z))
    null = (z_star * signs).mean(axis=1)
    total = len(null) + 1.0
    p = (1.0 + np.sum(np.abs(null) >= abs(score) * (1.0 - _TIE))) / total
    mixed = float(np.mean(z ** 2))
    p_mixed = (1.0 + np.sum((z_star ** 2).mean(axis=1) >= mixed * (1.0 - _TIE))) / total
    return {'score': score, 'p': float(p), 'mixed': mixed, 'p_mixed': float(p_mixed)}


def _target_genes(guide: str) -> set:
    return {str(guide).strip().lower()}


def _bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok])
    return out


def active_rows(spec_pop: dict, settings: dict) -> tuple:
    """(rows used, notes) of one population: the ticked states, without
    Abund rows resting on fewer than ``min_abund_signs`` stained signs."""
    rows = spec_pop['rows']
    notes = []
    if rows.empty:
        return rows, notes
    if settings['states'] is not None:
        rows = rows[rows['state'].isin(settings['states'])]
    thin = rows['kind'].eq('Abund') & (rows['signs'] < int(settings['min_abund_signs']))
    if thin.any():
        names = ', '.join(dict.fromkeys(rows.loc[thin, 'cell_type']))
        notes.append(f"Abundance rows left out (fewer than {settings['min_abund_signs']} stained markers beyond "
                     f"the population): {names}.")
    return rows[~thin].reset_index(drop=True), notes


def _directions(rotations: int, sizes, seed: int = SEED) -> dict:
    from flowcode_pathways import random_directions
    rng = np.random.default_rng(seed)
    return {n: random_directions(rng, rotations, n) for n in sorted(set(sizes))}


def analyse(fsea: dict, design: dict, settings: dict, shift: dict, ids_override: dict | None = None,
            seed: int = SEED) -> dict:
    """
    The statistics of every population and sample group from the stored
    scores (*fsea*: 'spec', 'scores'). *settings* are the FSEA settings,
    *shift* the phenotype shift's ('reference', 'min_cells', 'min_mice',
    'fdr'). With *ids_override* ({population: [ids per sample]}) the guide
    labels are replaced (calibration check).

    Returns {'results' (``RESULT_COLUMNS``), 'members' (``MEMBER_COLUMNS``,
    the main version's tested features), 'notes'}.
    """
    settings = fsea_settings(settings)
    spec, scores = fsea['spec'], fsea['scores']
    group_of = design['group_of']
    markers = list(spec['markers'])
    min_mice, min_cells = int(shift['min_mice']), int(shift['min_cells'])
    min_members, fdr = int(settings['min_members']), float(shift['fdr'])
    results, members, notes = [], [], []
    for pop, packed in scores.items():
        sp = spec['populations'].get(pop)
        if sp is None:
            continue
        rows, row_notes = active_rows(sp, settings)
        notes += [f'{pop}: {n}' for n in row_notes]
        if rows.empty:
            notes.append(f'{pop}: no immune-state row can be scored on this panel.')
            continue
        items = _items(packed)
        override = (ids_override or {}).get(pop)
        feats, num, den, root = features(items, len(design['combos']), rows, sp['sets'], markers, override)
        samples = [s for s, _i, _p in items]
        labels = dict(zip(rows['feature'], rows['label']))
        info = rows.drop_duplicates('feature').set_index('feature')
        for group in dict.fromkeys(group_of[s] for s in samples):
            idx = [i for i, s in enumerate(samples) if group_of[s] == group]
            sub = [samples[i] for i in idx]
            diff = differences(num[idx], den[idx], root[idx], sub, design, shift['reference'], min_cells)
            mats = guide_matrices(diff, num[idx], den[idx], sub, design, feats, min_mice,
                                  float(settings['min_count']))
            if not mats:
                notes.append(f'{pop}, {group}: no guide had enough cells for an immune-state feature in '
                             f'{min_mice} mice.')
                continue
            d0, s0_2 = variance_prior(mats)
            directions = _directions(int(settings['rotations']), [m.shape[0] for m, _f, _k in mats.values()], seed)
            g_rows = []
            for guide, (mat, fn, _keys) in mats.items():
                st = guide_statistics(mat, d0, s0_2, directions[mat.shape[0]])
                for j, f in enumerate(fn):
                    members.append({'population': pop, 'group': group, 'guide': guide, 'feature': f,
                                    'label': labels.get(f, f), 'kind': info.loc[f, 'kind'],
                                    'cell_type': info.loc[f, 'cell_type'], 'marker': info.loc[f, 'marker'],
                                    'mice': mat.shape[0], 'mean': float(st['mean'][j]), 't': float(st['t'][j]),
                                    'z': float(st['z'][j])})
                index = {f: j for j, f in enumerate(fn)}
                targets = _target_genes(guide)
                for state, g in rows.groupby('state', sort=False):
                    g = g[g['feature'].isin(index)]
                    if len(g) < min_members:
                        continue
                    on_target = g['gene'].str.lower().isin(targets).to_numpy()
                    test_all = state_test(st, np.array([index[f] for f in g['feature']]), g['sign'].to_numpy(float))
                    rec = {'population': pop, 'group': group, 'guide': guide, 'state': state, 'mice': mat.shape[0],
                           'members_all': len(g), 'target_rows': int(on_target.sum()),
                           'score_all': test_all['score'], 'p_all': test_all['p']}
                    main = g[~on_target]
                    if not on_target.any():
                        rec.update(members=len(g), score=test_all['score'], p=test_all['p'],
                                   p_mixed=test_all['p_mixed'])
                    elif len(main) >= min_members:
                        t = state_test(st, np.array([index[f] for f in main['feature']]),
                                       main['sign'].to_numpy(float))
                        rec.update(members=len(main), score=t['score'], p=t['p'], p_mixed=t['p_mixed'])
                    else:
                        rec.update(members=len(main), score=np.nan, p=np.nan, p_mixed=np.nan)
                    g_rows.append(rec)
            if not g_rows:
                notes.append(f'{pop}, {group}: no state had {min_members} tested members for any guide.')
                continue
            frame = pd.DataFrame(g_rows)
            frame['q'] = _bh(frame['p'])
            frame['q_mixed'] = _bh(frame['p_mixed'])
            frame['q_all'] = _bh(frame['p_all'])
            results.append(frame)
    if results:
        res = pd.concat(results, ignore_index=True)
        res['significant'] = res['q'] < fdr
        res['significant_all'] = res['q_all'] < fdr
        res['small_set'] = res['members'] < int(settings['flag_members'])
        res['depends_on_target'] = (res['target_rows'] > 0) & res['significant_all'] & ~res['significant']
        res = res[list(RESULT_COLUMNS)]
    else:
        res = pd.DataFrame(columns=list(RESULT_COLUMNS))
    return {'results': res, 'members': pd.DataFrame(members, columns=list(MEMBER_COLUMNS)), 'notes': notes}


def finish(store: dict, spec: dict, design: dict, settings: dict, shift: dict) -> dict:
    """The FSEA part of a phenotype-shift result from the scores collected
    during the run."""
    fsea = {'spec': spec, 'scores': pack_scores(store), 'settings': fsea_settings(settings), 'calibration': None}
    fsea.update(analyse(fsea, design, settings, shift))
    fsea['notes'] = list(spec.get('notes') or []) + fsea['notes']
    return fsea


def reanalyse(fsea: dict, design: dict, settings: dict, shift: dict) -> dict:
    """*fsea* with its statistics made again under *settings* and *shift*."""
    out = {**fsea, 'settings': fsea_settings(settings), 'calibration': None}
    out.update(analyse(out, design, settings, shift))
    out['notes'] = list(fsea['spec'].get('notes') or []) + out['notes']
    return out


# ---------------------------------------------------------------------------
# On a stored result
# ---------------------------------------------------------------------------

def shuffle_check(fsea: dict, design: dict, shift: dict, reps: int = SHUFFLES, seed: int = SEED,
                  progress=None) -> pd.DataFrame:
    """
    Calibration check: the guide labels are shuffled among the barcoded
    cells of each sample *reps* times and the states tested again. Per
    population and group: the tests, the share with p < 0.05 (about 5 % or
    less when the test is calibrated) and the shuffles with any call.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        if progress is not None:
            progress(rep, reps, f'immune states, shuffle {rep + 1}')
        override = {pop: [rng.permutation(ids) for _s, ids, _pi in _items(packed)]
                    for pop, packed in fsea['scores'].items()}
        res = analyse(fsea, design, fsea['settings'], shift, ids_override=override, seed=seed + rep + 1)['results']
        res['shuffle'] = rep + 1
        rows.append(res)
    if progress is not None:
        progress(reps, reps, '')
    allr = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = []
    for (pop, group), sub in (allr.groupby(['population', 'group'], sort=False) if len(allr) else ()):
        tested = sub[np.isfinite(sub['p'].astype(float))]
        out.append({'population': pop, 'group': group, 'shuffles': reps, 'tests': len(tested),
                    'pct_p_below_0_05': 100.0 * float((tested['p'] < 0.05).mean()) if len(tested) else np.nan,
                    'shuffles_with_a_hit': int(sub.groupby('shuffle')['significant'].any().sum())})
    return pd.DataFrame(out, columns=['population', 'group', 'shuffles', 'tests', 'pct_p_below_0_05',
                                      'shuffles_with_a_hit'])


def member_values(fsea: dict, design: dict, shift: dict, population: str, group: str, guide: str,
                  feature: str) -> pd.DataFrame:
    """One guide's value of one feature in every mouse of a group, with its
    reference: 'sample', 'guide_share', 'reference_share', 'difference'
    (logit), 'cells' (root cells of the guide)."""
    sp = fsea['spec']['populations'][population]
    rows = sp['rows'][sp['rows']['feature'] == feature]
    if rows.empty:
        raise KeyError(feature)
    items = [it for it in _items(fsea['scores'][population]) if design['group_of'][it[0]] == group]
    samples = [s for s, _i, _p in items]
    _f, num, den, root = features(items, len(design['combos']), rows, sp['sets'], list(fsea['spec']['markers']))
    diff = differences(num, den, root, samples, design, shift['reference'], int(shift['min_cells']))
    out = []
    for i, s in enumerate(samples):
        for c, name in enumerate(_combo_names(design, s)):
            if name == guide and np.isfinite(diff['d'][i, c, 0]):
                out.append({'sample': s, 'guide_share': float(diff['p'][i, c, 0]),
                            'reference_share': float(diff['ref'][i, c, 0]),
                            'difference': float(diff['d'][i, c, 0]), 'cells': float(root[i, c])})
    if not out:
        raise KeyError(guide)
    return pd.DataFrame(out, columns=['sample', 'guide_share', 'reference_share', 'difference', 'cells'])


def state_members(fsea: dict, population: str, state: str) -> pd.DataFrame:
    """The scored rows of one state in one population."""
    rows, _notes = active_rows(fsea['spec']['populations'][population], fsea['settings'])
    return rows[rows['state'] == state].reset_index(drop=True)


def states_table(fsea: dict, population: str, group: str) -> pd.DataFrame:
    """Per state in one population and group: library rows, rows scored on
    the panel, guides tested, median tested members, calls (main version)."""
    sp = fsea['spec']['populations'].get(population)
    if sp is None:
        return pd.DataFrame()
    rows, _notes = active_rows(sp, fsea['settings'])
    res = fsea['results']
    res = res[(res['population'] == population) & (res['group'] == group)]
    left = sp['left_out']
    out = []
    for state in dict.fromkeys(list(rows['state']) + list(left['state'])):
        r = res[res['state'] == state]
        tested = r[np.isfinite(r['p'].astype(float))]
        out.append({'state': state, 'scored_rows': int((rows['state'] == state).sum()),
                    'not_scored': int((left['state'] == state).sum()), 'guides_tested': len(tested),
                    'median_members': float(tested['members'].median()) if len(tested) else np.nan,
                    'calls_up': int((tested['significant'] & (tested['score'] > 0)).sum()),
                    'calls_down': int((tested['significant'] & (tested['score'] < 0)).sum()),
                    'depend_on_target': int(r['depends_on_target'].sum())})
    return pd.DataFrame(out)


def write_tables(folder, fsea: dict) -> list:
    """The immune-state CSV tables in *folder*; returns the paths written."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    tables = {'immune_state_results.csv': fsea.get('results'), 'immune_state_features.csv': fsea.get('members'),
              'immune_state_calibration.csv': fsea.get('calibration')}
    rows, left = [], []
    for pop, sp in fsea['spec']['populations'].items():
        r = sp['rows'].copy()
        r.insert(0, 'population', pop)
        r.insert(1, 'root', sp['root'])
        rows.append(r)
        lo = sp['left_out'].copy()
        lo.insert(0, 'population', pop)
        left.append(lo)
    tables['immune_state_rows.csv'] = pd.concat(rows, ignore_index=True) if rows else None
    tables['immune_state_rows_not_scored.csv'] = pd.concat(left, ignore_index=True) if left else None
    for name, frame in tables.items():
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        p = folder / name
        frame.to_csv(p, index=False)
        written.append(p)
    return written
