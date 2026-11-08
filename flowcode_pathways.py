"""
flowcode_pathways.py
--------------------
Signed pathway analysis of FlowCode knockout screens. No Qt.

What it asks
~~~~~~~~~~~~
A pathway (a GO Biological Process term, "process X") has member genes with
a sign: +1 when the gene is part of X or drives it, −1 when it holds X
back. Knocking out a +1 gene lowers X; knocking out a −1 gene raises X. If
X matters for the readout (for example, cells need X to accumulate in the
gut), the two kinds of knockout move in opposite directions, and the
pathway test adds them up instead of letting them cancel.

**Reading the score.** For a comparison "Gut vs Spleen", a positive score
means process X promotes accumulation in the gut: knockouts that lower X
are depleted there and knockouts that raise X are enriched. A negative
score means X works against accumulation in the gut. Each guide's
contribution is −sign × z, where z is the guide's statistic from the
statistics run; the score is their mean over the pathway's guides.

Example: X promotes gut accumulation. An enhancer knockout (sign +1) is
depleted in the gut (z = −2) and contributes +2; an inhibitor knockout
(sign −1) is enriched (z = +1.5) and contributes +1.5. Score +1.75.

Two tests per pathway and comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Self-contained** (the main result): is there a coordinated signed
  change in this pathway at all? A rotation test (after limma's roast,
  Wu et al. 2010) makes many "nothing is happening" versions of the data
  by re-mixing each guide's leftover mouse-to-mouse wobble in a random
  direction, the same direction for every guide, so guides whose wobble
  rises and falls together (compositional counts) keep doing so. The p
  value is the share of these versions whose score is at least as far
  from zero as the real one.
* **Competitive** (second column): does this pathway's signed change
  stand out from the other guides of the same library? CAMERA (Wu &
  Smyth 2012) compares the pathway's guides with the rest, allowing for
  correlation between the pathway's guides.
* **Any change** (third column, ``mixed_``): is anything happening in
  these guides, whatever the direction? Opposite movements add up instead
  of cancelling, so a gene family whose knockouts are lost from the gut
  for some members and gained for others still shows up. One strongly
  changed guide is enough to pass it, so read it with the leading guides.
  It uses the same rotations as the self-contained test (limma's "mixed"
  alternative).

Several libraries are tested one by one (their mice are different) and
combined by a weighted Stouffer sum; each library's result is also shown.

The statistics behind it
~~~~~~~~~~~~~~~~~~~~~~~~
Per-guide statistic. Each guide's observed statistic is the
quasi-likelihood t of the statistics run (signed √F, low-count adjusted,
on df_total = residual df + d0), turned into a z-score with the same
p-value: z = Φ⁻¹(F_t(t; df_total)). Guides with different degrees of
freedom are then on one scale (limma's roast and camera also work on z).

The fit behind each t is rebuilt here from the inputs the statistics run
recorded (``flowcode_stats.fit_record``) with the same functions of
``differential_stats``; its residual df + d0 must equal the run's df.total
for every guide, or the analysis stops.

Percentile-matched rotation. For one library (rotation unit) with n
samples and design X (p estimable columns, the tested one last), Q is the
orthonormal basis of the residual space of X (n − p columns), the same for
every guide. A guide's leftover wobble is its vector of Pearson residuals
r = (y − μ)/√(μ(1 + αμ)) from the QL fit (zero where the guide was not
measured, e.g. a removed clone), and e = Qᵀr / |Qᵀr| its direction. With
D = n − p + 1 and a₀ chosen so that √(D−1)·a₀/√(1 − a₀²) has, on t_{D−1},
the same lower-tail probability as z on N(0, 1), the guide is the unit
vector v = (a₀, √(1 − a₀²)·e) in R^D. A rotation is a random unit vector u
(uniform on the sphere); the guide's rotated z is

    z* = Φ⁻¹( F_t( √(D−1)·a/√(1 − a²); D−1 ) ),   a = u·v.

For any v, a is distributed as the first coordinate of a uniform unit
vector, so z* is exactly N(0, 1) under the rotations: every guide's null
matches its own QL test, also for guides with few events or a group with
no events. The same u for every guide keeps their correlation
(v_g·v_h). The unrotated direction (first coordinate 1, the rest 0) gives
back the observed z, so the observed
data are one of the rotations and p = (b + 1)/(R + 1), b the number of R
rotations with |score*| ≥ |score|.

Combination over libraries. Library l with n_l pathway guides gives
z_l = Φ⁻¹(1 − p_l/2) × sign(score_l); the combined
z = Σ √n_l·z_l / √Σ n_l, two-sided. Mice never span libraries, so the
libraries are independent.

Signed CAMERA. In one library with G tested guides, m of them in the
pathway with knockout signs k = −sign: x_g = k_g (z_g − z̄_other),
δ = mean(x), pooled variance σ² of x within the pathway and of z within
the others (on G − 2 df), variance inflation VIF = m × mean_i(mean_g k_g
U_gi)² from the guides' residual directions U (scaled to mean square 1;
VIF floored at 1), and t = δ / √(σ² (VIF/m + mean(k)²/(G − m))) on
G − 2 df. With every k = +1 this is limma's camera with an estimated
inter-gene correlation, except for the df: limma uses min(n − p, G − 2),
which with few mice (n − p of 3 to 6 in one comparison's fit) holds the
test far below its nominal rate; on G − 2 df it was close to nominal on
JA_007 nulls. ``signed_camera(..., guides_df=False)`` gives limma's df.

Any change. In one library, m = mean of z_g² over the set's guides, and
its rotated value mean of z*_g² on every rotation; p = (b + 1)/(R + 1)
with b the rotations whose value is at least m (one-sided: only large
values count). Library l gives z_l = Φ⁻¹(1 − p_l); the combined
z = Σ √n_l·z_l / √Σ n_l has the one-sided p 1 − Φ(z). The table's
mixed_score is the mean of z² over all the set's guides.

Deliberate v1 simplifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The rotation does not re-estimate each guide's noise level on every
  rotation (limma's roast does); that uncertainty enters through each
  guide's own QL p-value instead.
* A feature shared by several libraries (merged NON-Targetting guides) is
  never a pathway member; it is used only for the NON-Targetting check,
  rotated over the samples of all its libraries.
* The evidence filter keeps a gene in a set when its best annotation in
  that set has the chosen class; its sign is still taken from all its
  annotations.
"""

from __future__ import annotations

import gzip
import re
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import special

from honeychrome.controller_components import differential_stats as ds
from honeychrome.controller_components import moderated_stats as ms

import flowcode_stats as fst

GO_TABLE = Path(__file__).with_name('flowcode_go_signed_sets.csv.gz')
NAMES_TABLE = Path(__file__).with_name('flowcode_mgi_gene_names.csv.gz')

DEFAULT_ROTATIONS = 9999
DEFAULT_SEED = 20261103
DEFAULT_MIN_SIZE = 3
MIN_SIZE_FLOOR = 2
MAX_SET_FRACTION = 0.5
MIN_GROUP_SAMPLES = 2
EVIDENCE_ALL = 'all'
EVIDENCE_EXPERIMENTAL = 'experimental'
EVIDENCE_FILTERS = (EVIDENCE_ALL, EVIDENCE_EXPERIMENTAL)
LEADING_MEMBERS = 3

MATCH_OVERRIDE = 'override'
MATCH_SYMBOL = 'symbol'
MATCH_SYNONYM = 'synonym'
MATCH_AMBIGUOUS = 'ambiguous synonym'
MATCH_NONE = 'no match'
MATCH_CONTROL = 'control'

_ALONE = ''                         # rotation unit of a run with one library
_A_LIMIT = 1.0 - 1e-15
_TIE = 1e-12
_SET_CHUNK = 256
_HEADER = re.compile(r'GO\s+(?P<go>[^;]+);\s*MGI GAF\s+(?P<gaf>[^;]+);')

NEEDS_LOW_COUNT = ("The pathway analysis needs the quasi-likelihood test with the low-count adjustment, "
                   "per-comparison fits and the trended dispersion, which this version of Honeychrome "
                   "does not have.")


class PathwayError(RuntimeError):
    """The pathway analysis cannot run on these statistics."""


# ---------------------------------------------------------------------------
# Gene sets and gene names
# ---------------------------------------------------------------------------

def _table_header(path) -> dict:
    with gzip.open(path, 'rt') as fh:
        first = fh.readline().strip()
    match = _HEADER.search(first)
    return {'header': first.lstrip('# '), 'go_release': match.group('go').strip() if match else '',
            'gaf_date': match.group('gaf').strip() if match else ''}


def load_signed_sets(path=None, evidence: str = EVIDENCE_ALL) -> tuple:
    """
    The bundled signed GO BP table: (rows, meta). Rows have set_id,
    set_name, gene, sign (+1/−1) and evidence_class (experimental, curated
    or IEA, the best of the gene's annotations in that set). *evidence*
    'experimental' keeps only rows whose class is experimental. meta holds
    the GO release and the GAF date from the file's first line.
    """
    if evidence not in EVIDENCE_FILTERS:
        raise ValueError(f"evidence must be one of {EVIDENCE_FILTERS}, got {evidence!r}")
    path = Path(path) if path is not None else GO_TABLE
    meta = _table_header(path)
    rows = pd.read_csv(path, comment='#', dtype={'set_id': str, 'set_name': str, 'gene': str,
                                                 'sign': int, 'evidence_class': str})
    meta['full_size'] = rows.groupby('set_id').size()
    if evidence == EVIDENCE_EXPERIMENTAL:
        rows = rows[rows['evidence_class'] == EVIDENCE_EXPERIMENTAL]
    meta['evidence'] = evidence
    return rows.reset_index(drop=True), meta


def load_gene_names(path=None) -> tuple:
    """
    The bundled MGI symbol and synonym table: (symbols by lower-case name,
    symbols by lower-case synonym, meta). Each symbol maps to itself; a
    synonym maps to every symbol that lists it.
    """
    path = Path(path) if path is not None else NAMES_TABLE
    meta = _table_header(path)
    table = pd.read_csv(path, comment='#', dtype=str, keep_default_na=False)
    symbols = {s.lower(): s for s in table['symbol'].unique()}
    synonyms: dict[str, set] = {}
    for symbol, name in zip(table['symbol'], table['synonym']):
        if name and name.lower() not in symbols:
            synonyms.setdefault(name.lower(), set()).add(symbol)
    return symbols, synonyms, meta


def match_features(features, symbols: dict, synonyms: dict, overrides: dict | None = None) -> pd.DataFrame:
    """
    The MGI symbol of each feature (combination name): a user override
    first, then the symbol (ignoring case), then a synonym that belongs to
    one symbol only. Non-targeting combinations are controls and never
    genes. Returns feature, gene ('' when none) and match (override,
    symbol, synonym, ambiguous synonym, no match, control).
    """
    overrides = {str(k): str(v).strip() for k, v in (overrides or {}).items() if str(v).strip()}
    rows = []
    for feature in features:
        name = str(feature).strip()
        if fst.control_candidates([name]):
            rows.append((feature, '', MATCH_CONTROL))
        elif name in overrides:
            rows.append((feature, overrides[name], MATCH_OVERRIDE))
        elif name.lower() in symbols:
            rows.append((feature, symbols[name.lower()], MATCH_SYMBOL))
        elif len(synonyms.get(name.lower(), ())) == 1:
            rows.append((feature, next(iter(synonyms[name.lower()])), MATCH_SYNONYM))
        elif synonyms.get(name.lower()):
            rows.append((feature, '', MATCH_AMBIGUOUS))
        else:
            rows.append((feature, '', MATCH_NONE))
    return pd.DataFrame(rows, columns=['feature', 'gene', 'match'])


def collapse_sets(rows: pd.DataFrame, feature_genes: dict, min_size: int = DEFAULT_MIN_SIZE,
                  max_fraction: float = MAX_SET_FRACTION, full_size: pd.Series | None = None) -> pd.DataFrame:
    """
    The pathways that can be tested: each set cut to the features in
    *feature_genes* ({feature: gene}; a gene carried by several features
    gives each of them the gene's sign), kept when it has at least
    *min_size* members and no more than *max_fraction* of the features,
    and sets with identical signed members merged into one row. The row's
    GO terms are ordered most specific first (fewest genes in the full
    table, *full_size*). Returns set_key (the first GO id), go_ids,
    go_names and members (a tuple of (feature, sign) pairs, sorted by
    feature).
    """
    columns = ['set_key', 'go_ids', 'go_names', 'members']
    by_gene: dict[str, list] = {}
    for feature, gene in feature_genes.items():
        if gene:
            by_gene.setdefault(gene, []).append(feature)
    cut = rows[rows['gene'].isin(by_gene)]
    if cut.empty:
        return pd.DataFrame(columns=columns)
    if full_size is None:
        full_size = rows.groupby('set_id').size()
    limit = max_fraction * len(feature_genes)
    names = rows.drop_duplicates('set_id').set_index('set_id')['set_name']
    merged: dict[tuple, list] = {}
    for set_id, block in cut.groupby('set_id', sort=False):
        members = tuple(sorted((f, int(s)) for g, s in zip(block['gene'], block['sign']) for f in by_gene[g]))
        if min_size <= len(members) <= limit:
            merged.setdefault(members, []).append(set_id)
    out = []
    for members, ids in merged.items():
        ids = sorted(ids, key=lambda i: (int(full_size.get(i, 0)), str(names.get(i, i)), i))
        out.append((ids[0], tuple(ids), tuple(str(names.get(i, i)) for i in ids), members))
    return pd.DataFrame(out, columns=columns).sort_values('set_key', ignore_index=True)


# ---------------------------------------------------------------------------
# The quasi-likelihood fit of the statistics run, rebuilt
# ---------------------------------------------------------------------------

class FeatureFit(NamedTuple):
    """One feature's part of a rebuilt QL fit."""
    observed: np.ndarray        # positions of the fit's samples where it was measured
    design_columns: np.ndarray  # design columns of its sub-design
    columns: np.ndarray         # the same, −1 where the coefficient is not estimable
    y: np.ndarray
    offset: np.ndarray          # log library sizes
    mu: np.ndarray              # fitted means of the negative-binomial fit
    alpha: float                # negative-binomial dispersion (trended with abundance)
    residual_df: float          # low-count adjusted residual df
    quasi_dispersion: float     # moderated quasi-dispersion (the F test's denominator)


class QLFit(NamedTuple):
    """A rebuilt QL fit: its samples, design and features."""
    samples: list
    group_vec: list
    design: np.ndarray
    group_columns: dict
    features: dict              # feature -> FeatureFit
    prior_df: float             # d0 of the quasi-dispersion prior


def _log_offsets(counts_df: pd.DataFrame, offset_df: pd.DataFrame | None) -> np.ndarray:
    """Log library sizes (features × samples), as ``run_glm_counts`` makes them."""
    features = list(counts_df.columns)
    if offset_df is None:
        totals = counts_df.sum(axis=1).values.astype(float)
        return np.tile(np.log(np.maximum(totals, 1.0)), (len(features), 1))
    denom = offset_df.reindex(index=counts_df.index, columns=counts_df.columns)
    return np.log(np.maximum(denom[features].values.T.astype(float), 1.0))


def rebuild_ql_fit(counts_df: pd.DataFrame, group_vec: list, base: str, pairing_vec: list | None,
                   covariates: pd.DataFrame | None, offset_df: pd.DataFrame | None,
                   alpha: np.ndarray | None = None, design: np.ndarray | None = None,
                   group_cols: dict | None = None) -> QLFit:
    """
    The quasi-likelihood fit of ``differential_stats._glm_ql_tests`` for
    these inputs (those of one comparison's fit), with the parts the
    rotation needs: fitted means, the dispersions (*alpha*, one per column
    of *counts_df*, else estimated on these samples), residual df and d0.
    *design* and *group_cols* are the comparison's design
    (``ql_comparisons``); without them it is built from the groups, pairing
    and covariates. Each step is the statistics run's own function, in the
    same order.
    """
    if not hasattr(ds, 'ql_block_dispersions'):
        raise PathwayError(NEEDS_LOW_COUNT)
    if design is None:
        design, _names, group_cols = ds.build_design(group_vec, base, pairing_vec, covariates)
    features = list(counts_df.columns)
    counts = counts_df[features].values.T.astype(float)
    log_offsets = _log_offsets(counts_df, offset_df)

    blocks, mus = ds._ql_blocks(counts, design, log_offsets)
    out: dict = {}
    prior_df = np.nan
    if blocks:
        alphas, mus = ds.ql_block_dispersions(blocks, mus, alpha)
        full = [ds._fit_rows(b.Y, b.X, b.O, a) for b, a in zip(blocks, alphas)]
        mus = [np.array([f[1] if f is not None else m for f, m in zip(fb, mu)])
               for fb, mu in zip(full, mus)]
        positions, kept = [], []
        for b in blocks:
            observed = np.zeros(counts.shape[1], dtype=bool)
            pattern = np.isfinite(counts[b.rows[0]]) & np.isfinite(log_offsets[b.rows[0]])
            observed[pattern] = True
            columns, _estimable = ms.estimable_columns(design[observed])
            positions.append(np.flatnonzero(observed))
            kept.append(columns)
        adjusted = [ds._adjusted_deviance(b.Y, mu, b.X, a) for b, mu, a in zip(blocks, mus, alphas)]
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = np.concatenate([np.where(df > 0, d / df, 0.0) for d, df in adjusted])
        df_all = np.concatenate([df for _d, df in adjusted])
        prior_df, s0 = ms.fit_prior_variance(phi, df_all)
        post = np.full(phi.shape, s0) if np.isinf(prior_df) else (prior_df * s0 + df_all * phi) / (prior_df + df_all)
        start = 0
        for b, mu, a, (_dev, rdf), obs, cols in zip(blocks, mus, alphas, adjusted, positions, kept):
            for r, i in enumerate(b.rows):
                out[features[i]] = FeatureFit(obs, cols, b.columns, b.Y[r], b.O[r], mu[r], float(a[r]),
                                              float(rdf[r]), float(post[start + r]))
            start += b.rows.size
    return QLFit(list(counts_df.index), list(group_vec), design, group_cols, out, float(prior_df))


def rebuilt_t(fit: QLFit, coef: int, features: list) -> dict:
    """
    |t| of the QL F test of design column *coef* for *features* from the
    rebuilt fit: √((deviance without the column − deviance with it) /
    moderated quasi-dispersion), as the statistics run computes it.
    Features sharing a sub-design are refitted together.
    """
    groups: dict[tuple, list] = {}
    for f in features:
        part = fit.features[f]
        groups.setdefault((part.observed.tobytes(), part.design_columns.tobytes()), []).append(f)
    out = {}
    for names in groups.values():
        first = fit.features[names[0]]
        k = int(np.flatnonzero(first.columns == coef)[0])
        X = np.delete(fit.design[np.ix_(first.observed, first.design_columns)], k, axis=1)
        Y = np.array([fit.features[f].y for f in names])
        O = np.array([fit.features[f].offset for f in names])
        mu = np.array([fit.features[f].mu for f in names])
        alpha = np.array([fit.features[f].alpha for f in names])
        reduced = ds._fit_rows(Y, X, O, alpha)
        mu_red = np.array([r[1] if r is not None else m for r, m in zip(reduced, mu)])
        gain = np.maximum(ds._glm_deviance(Y, mu_red, alpha) - ds._glm_deviance(Y, mu, alpha), 0.0)
        for f, g, r in zip(names, gain, reduced):
            out[f] = float(np.sqrt(g / fit.features[f].quasi_dispersion)) if r is not None else np.nan
    return out


def _fits_of_record(record: dict) -> list:
    """(rebuilt fit, comparisons as [(results label, base, other)]) for one
    recorded ``run_glm_counts`` call (QL): one fit per comparison, on the
    samples of its two groups, with the dispersions estimated on every
    sample of the call, as the statistics run fits it."""
    labels = record['labels']
    counts_df, group_vec = record['counts'], record['group_vec']
    pairing_vec, covariates = record['pairing_vec'], record['covariates']
    full_design, _names, _cols = ds.build_design(group_vec, record['contrasts'][0][0], pairing_vec, covariates)
    alpha = ds.ql_dispersion(counts_df.values.T.astype(float), full_design,
                             _log_offsets(counts_df, record['offset_df']))
    out = []
    for base, other, mask, design, coef in ds.ql_comparisons(group_vec, record['contrasts'], pairing_vec,
                                                             covariates):
        sub = counts_df.loc[counts_df.index[mask]]
        offsets = record['offset_df'].loc[sub.index] if record['offset_df'] is not None else None
        groups = [g for g, m in zip(group_vec, mask) if m]
        fit = rebuild_ql_fit(sub, groups, base, None, None, offsets, alpha=alpha, design=design,
                             group_cols={other: coef})
        out.append((fit, [(labels[f'{other} vs {base}'], base, other)]))
    return out


# ---------------------------------------------------------------------------
# z-scores and rotations
# ---------------------------------------------------------------------------

def t_to_z(t, df) -> np.ndarray:
    """The z-score with the same tail probability as t on df degrees of
    freedom (df may be infinite)."""
    t = np.asarray(t, dtype=float)
    df = np.broadcast_to(np.asarray(df, dtype=float), t.shape)
    infinite = np.isinf(df)
    with np.errstate(invalid='ignore'):
        lower = np.where(infinite, special.ndtr(-np.abs(t)),
                         special.stdtr(np.where(infinite, 1.0, df), -np.abs(t)))
    z = -special.ndtri(np.clip(lower, 1e-300, 0.5))
    return np.copysign(z, t)


def _z_to_x(z, d: int) -> np.ndarray:
    """x = 1 − a₀² of the a₀ whose t (√d·a₀/√(1 − a₀²) on t_d) has the tail
    probability of z on N(0, 1). The tail of t_d at that t is
    ½·I_x(d/2, ½) (regularized incomplete beta), so x is its inverse at
    twice the tail probability; it stays exact far into the tails, where
    a₀ itself rounds to ±1."""
    lower = np.clip(special.ndtr(-np.abs(np.asarray(z, dtype=float))), 1e-300, 0.5)
    return special.betaincinv(d / 2.0, 0.5, 2.0 * lower)


def rotated_z(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """
    Rotated z of each guide (columns) for each direction (rows): a = u·v
    for unit vectors v (guides × D) and u (directions × D). Its t on D − 1
    df, √(D−1)·a/√(1 − a²), has the tail probability ½·I_x((D−1)/2, ½)
    with x = 1 − a², which is mapped to N(0, 1).
    """
    d = vectors.shape[1] - 1
    a = np.clip(directions @ vectors.T, -_A_LIMIT, _A_LIMIT)
    lower = np.clip(0.5 * special.betainc(d / 2.0, 0.5, (1.0 - a) * (1.0 + a)), 1e-300, 0.5)
    return np.copysign(-special.ndtri(lower), a)


def random_directions(rng: np.random.Generator, rotations: int, dimension: int) -> np.ndarray:
    """*rotations* unit vectors uniform on the sphere in R^dimension."""
    u = rng.standard_normal((rotations, dimension))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def residual_basis(design: np.ndarray, coef: int) -> np.ndarray | None:
    """Orthonormal basis (samples × (n − p)) of the residual space of
    *design* (these samples' rows), or None when the tested column is not
    estimable or no residual df is left."""
    columns, estimable = ms.estimable_columns(design)
    position = np.flatnonzero(columns == coef)
    if position.size == 0 or not estimable[int(position[0])]:
        return None
    if design.shape[0] - columns.size < 1:
        return None
    order = [int(c) for c in columns if c != coef] + [coef]
    q = np.linalg.qr(design[:, order], mode='complete')[0]
    return q[:, columns.size:]


def guide_vector(z: float, residuals: np.ndarray, basis: np.ndarray) -> np.ndarray | None:
    """The unit vector (a₀, √(1 − a₀²)·e) of one guide, e the direction of
    its residuals in *basis*; None when the residuals have no direction
    there."""
    e = basis.T @ residuals
    norm = float(np.linalg.norm(e))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    x = float(_z_to_x(z, basis.shape[1]))
    return np.concatenate([[np.copysign(np.sqrt(1.0 - x), z)], np.sqrt(x) * e / norm])


def pearson_residuals(fit: FeatureFit) -> np.ndarray:
    """(y − μ)/√(μ(1 + αμ)) of a feature's fit; 0 where μ is 0."""
    var = fit.mu * (1.0 + fit.alpha * fit.mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(var > 0, (fit.y - fit.mu) / np.sqrt(var), 0.0)


def rotation_pvalues(z: np.ndarray, vectors: np.ndarray, weights: np.ndarray,
                     directions: np.ndarray, mixed: bool = False) -> tuple:
    """
    Self-contained rotation test of every set at once in one unit.
    *weights* (guides × sets) holds each member's knockout sign divided by
    the set's number of members there. Returns (scores, p); with *mixed*
    also the any-change statistic (mean z² of the members) and its
    one-sided p from the same rotations: (scores, p, mixed_scores, mixed_p).
    """
    observed = z @ weights
    beyond = np.zeros(weights.shape[1])
    if mixed:
        sizes = np.abs(weights)
        observed_mixed = (z * z) @ sizes
        beyond_mixed = np.zeros(weights.shape[1])
    for start in range(0, directions.shape[0], 1024):
        rotated = rotated_z(vectors, directions[start:start + 1024])
        for j in range(0, weights.shape[1], _SET_CHUNK):
            scores = rotated @ weights[:, j:j + _SET_CHUNK]
            limit = np.abs(observed[j:j + _SET_CHUNK]) * (1.0 - _TIE)
            beyond[j:j + _SET_CHUNK] += np.sum(np.abs(scores) >= limit, axis=0)
            if mixed:
                squares = (rotated * rotated) @ sizes[:, j:j + _SET_CHUNK]
                beyond_mixed[j:j + _SET_CHUNK] += np.sum(
                    squares >= observed_mixed[j:j + _SET_CHUNK] * (1.0 - _TIE), axis=0)
    total = directions.shape[0] + 1.0
    if mixed:
        return observed, (beyond + 1.0) / total, observed_mixed, (beyond_mixed + 1.0) / total
    return observed, (beyond + 1.0) / total


def signed_camera(z: np.ndarray, directions: np.ndarray, members: np.ndarray, signs: np.ndarray,
                  residual_df: float, guides_df: bool = False) -> dict:
    """
    Signed CAMERA of one set in one unit. *z*: every tested guide's z;
    *directions*: unit residual directions (guides × (n − p)); *members*:
    positions of the set's guides in *z*; *signs*: their knockout signs.
    The t is read on min(*residual_df*, guides − 2) df, as limma's camera
    does, or with *guides_df* on guides − 2 df (the pathway analysis).
    Returns delta, t, df, p (two-sided), correlation (NaN for one member)
    and vif.
    """
    nan = {'delta': np.nan, 't': np.nan, 'df': np.nan, 'p': np.nan, 'correlation': np.nan, 'vif': np.nan}
    n = z.size
    m = members.size
    others = np.ones(n, dtype=bool)
    others[members] = False
    m2 = int(others.sum())
    if m < 1 or m2 < 1 or n < 3:
        return nan
    centre = float(np.mean(z[others]))
    x = signs * (z[members] - centre)
    delta = float(np.mean(x))
    pooled = (float(np.sum((x - delta) ** 2)) + float(np.sum((z[others] - centre) ** 2))) / (n - 2)
    if m > 1:
        scaled = signs[:, None] * directions[members] * np.sqrt(directions.shape[1])
        vif = m * float(np.mean(np.mean(scaled, axis=0) ** 2))
        correlation = (vif - 1.0) / (m - 1.0)
        vif = max(vif, 1.0)
    else:
        vif, correlation = 1.0, np.nan
    se = np.sqrt(pooled * (vif / m + float(np.mean(signs)) ** 2 / m2))
    df = float(n - 2 if guides_df else min(residual_df, n - 2))
    t = delta / se if se > 0 else np.nan
    p = float(2.0 * special.stdtr(df, -abs(t))) if np.isfinite(t) else np.nan
    return {'delta': delta, 't': float(t), 'df': df, 'p': p, 'correlation': correlation, 'vif': vif}


def stouffer(z: np.ndarray, n: np.ndarray) -> tuple:
    """Weighted Stouffer combination Σ √n·z / √Σ n and its two-sided p."""
    z, n = np.asarray(z, dtype=float), np.asarray(n, dtype=float)
    use = np.isfinite(z) & (n > 0)
    if not use.any():
        return np.nan, np.nan
    combined = float(np.sum(np.sqrt(n[use]) * z[use]) / np.sqrt(np.sum(n[use])))
    return combined, float(2.0 * special.ndtr(-abs(combined)))


def signed_z_of_p(p: float, direction: float) -> float:
    """Φ⁻¹(1 − p/2) carrying the sign of *direction* (0 for no direction)."""
    if not np.isfinite(p):
        return np.nan
    return float(np.sign(direction) * -special.ndtri(min(max(p, 1e-300), 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Bundled tables, read once
# ---------------------------------------------------------------------------

_LOADED: dict = {}


def bundled_sets(evidence: str = EVIDENCE_ALL) -> tuple:
    """``load_signed_sets`` of the bundled table, kept after the first read."""
    key = ('sets', evidence)
    if key not in _LOADED:
        _LOADED[key] = load_signed_sets(None, evidence)
    return _LOADED[key]


def bundled_names() -> tuple:
    """``load_gene_names`` of the bundled table, kept after the first read."""
    if 'names' not in _LOADED:
        _LOADED['names'] = load_gene_names(None)
    return _LOADED['names']


def run_from_bundle(stats: dict, fit_log: list, *, evidence: str = EVIDENCE_ALL, overrides: dict | None = None,
                    rotations: int = DEFAULT_ROTATIONS, seed: int = DEFAULT_SEED,
                    min_size: int = DEFAULT_MIN_SIZE) -> dict:
    """``run_pathways`` with the bundled GO sets and gene names; the result's
    settings also carry the GO release, GAF date, evidence filter and the
    name overrides used."""
    rows, meta = bundled_sets(evidence)
    symbols, synonyms, _names_meta = bundled_names()
    features = list(dict.fromkeys(f for r in fit_log for f in r['counts'].columns))
    matches = match_features(features, symbols, synonyms, overrides)
    out = run_pathways(stats, fit_log, rows, matches, rotations=rotations, seed=seed, min_size=min_size,
                       full_size=meta['full_size'])
    out['settings'].update(go_release=meta['go_release'], gaf_date=meta['gaf_date'], evidence=evidence,
                           overrides=dict(overrides or {}))
    return out


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

DOT_PLOT_SETS = 40
DOT_PLOT_FALLBACK = 15
MEMBER_FIGURES_PER_COMPARISON = 10
SIGN_COLOURS = {1: '#1b9e77', -1: '#d95f02'}       # RColorBrewer Dark2
SIGN_NAMES = {1: 'part of / drives the process (+1)', -1: 'holds the process back (−1)'}
_FDR_SIZE_CAP = 4.0


def significant(row, fdr_threshold: float) -> tuple:
    """(self-contained, competitive) FDR at or below *fdr_threshold*."""
    return (bool(float(row['FDR']) <= fdr_threshold), bool(float(row['competitive_FDR']) <= fdr_threshold))


def library_summaries(result: dict, comparison: str | None = None) -> dict:
    """{(comparison, set key): each library's members, score and rotation z},
    e.g. 'Metab 1: 3 members, z +2.10; Metab 2: 2, z -0.40', for one
    comparison or all."""
    lib = result['libraries']
    if comparison is not None:
        lib = lib[lib['comparison'] == comparison]
    parts = {}
    for c, k, name, members, z in zip(lib['comparison'], lib['set_key'], lib['library'], lib['members'], lib['z']):
        parts.setdefault((c, k), []).append(f"{name or 'library'}: {int(members)} member(s), z {z:+.2f}")
    return {key: '; '.join(texts) for key, texts in parts.items()}


def library_summary(result: dict, comparison: str, set_key: str) -> str:
    """``library_summaries`` of one pathway and comparison."""
    return library_summaries(result, comparison).get((comparison, set_key), '')


def pathway_table(result: dict, comparison: str | None = None) -> pd.DataFrame:
    """The pathway results (of one comparison, or all), with each library's
    summary, ordered by competitive FDR then self-contained FDR."""
    table = result['pathways']
    if comparison is not None:
        table = table[table['comparison'] == comparison]
    summaries = library_summaries(result, comparison)
    table = table.assign(per_library=[summaries.get((c, k), '')
                                      for c, k in zip(table['comparison'], table['set_key'])])
    return table.sort_values(['competitive_FDR', 'FDR', 'set_name'], ignore_index=True)


def top_set(result: dict, comparison: str) -> str | None:
    """The first pathway of ``pathway_table`` for *comparison*."""
    table = pathway_table(result, comparison)
    return None if table.empty else str(table['set_key'].iloc[0])


def dot_plot_sets(result: dict, fdr_threshold: float, max_sets: int = DOT_PLOT_SETS) -> list:
    """The pathways shown in the dot plot: those with either FDR at or below
    the threshold in any comparison, lowest competitive FDR first (at most
    *max_sets*); without any, the DOT_PLOT_FALLBACK lowest."""
    table = result['pathways']
    if table.empty:
        return []
    best = table.groupby('set_key').agg(competitive=('competitive_FDR', 'min'), own=('FDR', 'min'))
    best = best.sort_values(['competitive', 'own'])
    hits = best[(best['competitive'] <= fdr_threshold) | (best['own'] <= fdr_threshold)]
    chosen = hits if not hits.empty else best.head(DOT_PLOT_FALLBACK)
    return list(chosen.index[:max_sets])


def _size(fdr) -> np.ndarray:
    value = -np.log10(np.clip(np.asarray(fdr, dtype=float), 10.0 ** -_FDR_SIZE_CAP, 1.0))
    return 20.0 + 280.0 * np.nan_to_num(value) / _FDR_SIZE_CAP


def _short(text: str, width: int = 60) -> str:
    text = str(text)
    return text if len(text) <= width else text[:width - 1].rstrip() + '…'


def dot_plot_figure(result: dict, comparisons: list, fdr_threshold: float, is_dark: bool = False,
                    population: str = ''):
    """
    Pathways × comparisons. Colour: the score (red: the process promotes
    accumulation in the comparison's first group; blue: it works against
    it). Size: −log10 competitive FDR (capped at 10⁻⁴). Ringed: the
    self-contained FDR is at or below the threshold. Double-clicking a dot
    needs ``fig._dot_points`` (data x, y), ``_dot_keys`` and ``_dot_axes``.
    """
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    keys = dot_plot_sets(result, fdr_threshold)
    table = result['pathways']
    table = table[table['set_key'].isin(keys)]
    if table.empty:
        return dp.message_figure('No pathway could be tested.', 'Pathways', is_dark)
    comps = [c for c in comparisons if c in set(table['comparison'])]
    names = table.drop_duplicates('set_key').set_index('set_key')['set_name']
    order = list(reversed(keys))
    fig = Figure(figsize=(max(5.0, 1.3 * len(comps) + 5.5), max(3.2, 0.3 * len(order) + 2.0)), layout='constrained')
    ax = fig.add_subplot(111)
    lim = max(1.0, float(np.nanmax(np.abs(table['score'].astype(float)))))
    ring = 'white' if is_dark else 'black'
    rows = [r for _i, r in table.iterrows() if r['comparison'] in comps]
    xs = np.array([comps.index(r['comparison']) for r in rows], dtype=float)
    ys = np.array([order.index(r['set_key']) for r in rows], dtype=float)
    edges = [ring if significant(r, fdr_threshold)[0] else 'none' for r in rows]
    sc = ax.scatter(xs, ys, c=[float(r['score']) for r in rows], s=_size([r['competitive_FDR'] for r in rows]),
                    cmap='RdBu_r', vmin=-lim, vmax=lim, edgecolors=edges, linewidths=1.4)
    labels = [f"{r['set_name']} — {r['comparison']}\nscore {r['score']:+.2f}, {int(r['members'])} member(s) in "
              f"{int(r['libraries'])} library(ies), agreement {r['concordance']:.0%}\n"
              f"self-contained p {r['P.Value']:.3g}, FDR {r['FDR']:.3g}\n"
              f"competitive p {r['competitive_P.Value']:.3g}, FDR {r['competitive_FDR']:.3g}\n"
              f"leading: {r['leading']}" for r in rows]
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha='right')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_short(names.get(k, k)) for k in order])
    ax.set_xlim(-0.6, len(comps) - 0.4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label('Score (red: promotes accumulation in the first group)')
    for fdr in (0.05, 0.01, 0.001):
        ax.scatter([], [], s=_size([fdr]), c='grey', label=f'competitive FDR {fdr:g}')
    ax.scatter([], [], s=_size([0.05]), facecolors='none', edgecolors=ring, linewidths=1.4,
               label=f'ringed: self-contained FDR ≤ {fdr_threshold:g}')
    handles, texts = ax.get_legend_handles_labels()
    fig.legend(handles, texts, loc='outside right upper', fontsize=8, frameon=False, labelspacing=1.2)
    shown = f'{len(order)} pathway(s)' + ('' if len(order) < DOT_PLOT_SETS else f' (top {DOT_PLOT_SETS})')
    ax.set_title(f'{population}: signed GO pathways, {shown}' if population else f'Signed GO pathways, {shown}',
                 fontsize=10)
    fg = dp.style_figure_theme(fig, is_dark)
    cb.ax.yaxis.label.set_color(fg)
    cb.ax.tick_params(colors=fg)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, labels, is_dark)
    fig._dot_points = np.column_stack([xs, ys])
    fig._dot_keys = [(r['comparison'], r['set_key']) for r in rows]
    fig._dot_axes = ax
    return fig


def members_figure(result: dict, comparison: str, set_key: str, ci: pd.DataFrame | None, fdr_threshold: float,
                   is_dark: bool = False, population: str = '', groups: bool = False):
    """
    One pathway in one comparison: each member's log2 fold change with its
    95% interval (*ci*: CI.L and CI.R by feature), coloured by sign and
    grouped by library (a dotted line between libraries; each library's
    result is in ``library_summary``). Within a library, members are listed
    by sign (+1 first) and then by gene name, so a pathway's rows keep the
    same order across comparisons. The shaded side of each row is the
    direction the pathway's score predicts for that member (−sign × sign of
    the score).
    With *groups* (gene groups, every sign +1) the legend has no signs and
    the title gives the competitive and any-change FDR. Double-clicking a
    member needs ``fig._member_features`` and ``fig._member_axes``.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from honeychrome.view_components import differential_plots as dp

    members = result['members']
    view = members[(members['comparison'] == comparison) & (members['set_key'] == set_key)]
    rows = result['pathways']
    row = rows[(rows['comparison'] == comparison) & (rows['set_key'] == set_key)]
    if view.empty or row.empty:
        what = ('This group', 'Gene group members') if groups else ('This pathway', 'Pathway members')
        return dp.message_figure(f'{what[0]} was not tested in this comparison.', what[1], is_dark)
    row = row.iloc[0]
    direction = 1.0 if float(row['score']) >= 0 else -1.0
    view = view.assign(_name=[str(g or f).lower() for g, f in zip(view['gene'], view['feature'])])
    view = view.sort_values(['library', 'sign', '_name', 'feature'], ascending=[True, False, True, True],
                            ignore_index=True).drop(columns='_name')
    fig = Figure(figsize=(7.5, max(2.8, 0.32 * len(view) + 2.2)), layout='constrained')
    ax = fig.add_subplot(111)
    lfc = view['logFC'].astype(float).values
    lo = hi = lfc
    if ci is not None:
        lo = ci.reindex(view['feature'])['CI.L'].astype(float).values
        hi = ci.reindex(view['feature'])['CI.R'].astype(float).values
    extent = float(np.nanmax(np.abs(np.concatenate([lo, hi, lfc, [1.0]]))))
    extent = 1.05 * extent if np.isfinite(extent) else 2.0
    y = np.arange(len(view))[::-1]
    shade_colour, shade_alpha = ('white', 0.16) if is_dark else ('grey', 0.12)
    for yi, sign in zip(y, view['sign'].astype(int)):
        side = -sign * direction
        ax.axhspan(yi - 0.45, yi + 0.45, xmin=0.5 if side > 0 else 0.0, xmax=1.0 if side > 0 else 0.5,
                   color=shade_colour, alpha=shade_alpha, lw=0)
    for sign, colour in SIGN_COLOURS.items():
        pick = view['sign'].astype(int).values == sign
        if not pick.any():
            continue
        ax.errorbar(lfc[pick], y[pick], xerr=[lfc[pick] - lo[pick], hi[pick] - lfc[pick]], fmt='o', color=colour,
                    ecolor=colour, elinewidth=1.2, capsize=2, ms=6)
    ax.axvline(0.0, color='grey', lw=0.8)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-0.6, len(view) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} ({f})" if g and g.lower() != str(f).lower() else str(f)
                        for g, f in zip(view['gene'], view['feature'])])
    for lib in dict.fromkeys(view['library']):
        at = y[(view['library'] == lib).values]
        if at.min() > 0:
            ax.axhline(at.min() - 0.5, color='grey', lw=0.6, ls=':')
    ax.set_xlabel(f'log2 fold change ({comparison})')
    own, comp = significant(row, fdr_threshold)
    if groups:
        mixed = bool(float(row['mixed_FDR']) <= fdr_threshold)
        ax.set_title(f"{_short(row['set_name'], 80)}\nscore {row['score']:+.2f}, agreement {row['concordance']:.0%}; "
                     f"competitive FDR {row['competitive_FDR']:.3g}{' *' if comp else ''}; any-change FDR "
                     f"{row['mixed_FDR']:.3g}{' *' if mixed else ''}", fontsize=9)
        handles = [Patch(color=SIGN_COLOURS[1], label='guide of the group')]
    else:
        ax.set_title(f"{_short(row['set_name'], 80)}\nscore {row['score']:+.2f}, agreement {row['concordance']:.0%}; "
                     f"self-contained FDR {row['FDR']:.3g}{' *' if own else ''}; competitive FDR "
                     f"{row['competitive_FDR']:.3g}{' *' if comp else ''}", fontsize=9)
        handles = [Patch(color=c, label=SIGN_NAMES[s]) for s, c in SIGN_COLOURS.items()]
    handles.append(Patch(color=shade_colour, alpha=2.5 * shade_alpha, label='direction the score predicts'))
    fig.legend(handles=handles, loc='outside lower center', fontsize=8, frameon=False, ncol=3)
    dp.style_figure_theme(fig, is_dark)
    if population:
        dp.stamp_run_label(fig, population)
    fig._member_features = list(view['feature'])[::-1]
    fig._member_axes = ax
    return fig


def figures(result: dict, comparisons: list, comparison: str | None, set_key: str | None, ci: pd.DataFrame | None,
            fdr_threshold: float, is_dark: bool, population: str, dot_plot: bool = True) -> dict:
    """{key: {'fig', 'title', 'error', 'make', 'export_title'}} for the
    page: the dot plot (with *dot_plot*) and the chosen pathway's members in
    *comparison*."""
    out = {}
    makers = []
    if dot_plot:
        makers.append(('pathway_dot_plot', 'Pathway dot plot', 'Pathway dot plot',
                       lambda: dot_plot_figure(result, comparisons, fdr_threshold, is_dark, population)))
    if comparison is not None and set_key is not None:
        makers.append(('pathway_members', 'Pathway members', f'Pathway members - {set_name(result, set_key)}',
                       lambda: members_figure(result, comparison, set_key, ci, fdr_threshold, is_dark, population)))
    for key, title, export_title, make in makers:
        try:
            out[key] = {'fig': make(), 'title': title, 'error': None, 'make': make, 'export_title': export_title}
        except Exception as exc:
            out[key] = {'fig': None, 'title': title, 'error': str(exc) or type(exc).__name__}
    return out


def set_name(result: dict, set_key: str) -> str:
    """The pathway's shown name (its most specific GO term)."""
    table = result['pathways']
    names = table.loc[table['set_key'] == set_key, 'set_name']
    return str(names.iloc[0]) if len(names) else str(set_key)


def export_sets(result: dict, comparison: str, fdr_threshold: float,
                limit: int = MEMBER_FIGURES_PER_COMPARISON) -> list:
    """The pathways of *comparison* whose members are drawn on export:
    either FDR at or below the threshold, in ``pathway_table`` order, at
    most *limit*."""
    table = pathway_table(result, comparison)
    hits = table[(table['competitive_FDR'] <= fdr_threshold) | (table['FDR'] <= fdr_threshold)]
    return list(hits['set_key'][:limit])


def settings_lines(result: dict | None, settings: dict) -> list:
    """settings.txt lines of a pathway run (or of the chosen settings
    when there is no result)."""
    run = (result or {}).get('settings') or {}
    evidence = run.get('evidence', settings.get('evidence', EVIDENCE_ALL))
    lines = [
        f"  Pathways: signed GO Biological Process sets (GO {run.get('go_release', '-')}, MGI GAF "
        f"{run.get('gaf_date', '-')}; regulation precedence; evidence: "
        f"{'all annotations, IEA included' if evidence == EVIDENCE_ALL else 'experimental only'}); "
        f"min. {run.get('min_size', settings.get('min_size'))} members; "
        f"{run.get('rotations', settings.get('rotations'))} rotations, seed {run.get('seed', DEFAULT_SEED)}",
        '  Pathway tests: self-contained rotation test on the QL z-scores (percentile matching), signed CAMERA '
        '(estimated inter-gene correlation, guides - 2 df) as the competitive score; libraries combined by '
        'weighted Stouffer; '
        'BH FDR within each comparison. Score = mean of -sign x z: positive means the process promotes '
        'accumulation in the first group.',
    ]
    overrides = run.get('overrides', settings.get('overrides')) or {}
    if overrides:
        lines.append('  Pathway gene names set by hand: ' + ', '.join(f'{k} = {v}' for k, v in overrides.items()))
    matches = (result or {}).get('matches')
    if matches is not None and not matches.empty:
        syn = matches[matches['match'] == MATCH_SYNONYM]
        if not syn.empty:
            lines.append('  Pathway gene names matched by MGI synonym: '
                         + ', '.join(f'{f} = {g}' for f, g in zip(syn['feature'], syn['gene'])))
        missing = matches[matches['match'].isin([MATCH_NONE, MATCH_AMBIGUOUS])]
        if not missing.empty:
            lines.append('  Combinations with no MGI gene (not in any pathway): ' + ', '.join(missing['feature']))
    return lines


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

class _Unit(NamedTuple):
    """The guides of one rotation unit (library) in one comparison."""
    name: str
    features: list
    z: np.ndarray
    vectors: np.ndarray        # guides × D
    residual_df: int


def _unit_name(carriers: tuple, labels: dict) -> str:
    return ' + '.join(labels.get(c, c) for c in carriers) if carriers != (_ALONE,) else _ALONE


def _carriers(stats: dict, features) -> dict:
    """{feature: libraries carrying it (sorted tuple)}; one shared unit
    without libraries."""
    by_feature = stats.get('feature_libraries')
    if not stats.get('libraries') or not by_feature:
        return {f: (_ALONE,) for f in features}
    return {f: tuple(sorted(str(lib) for lib in by_feature.get(f) or ())) for f in features}


def _comparison_units(fit: QLFit, coef: int, base: str, other: str, results: pd.DataFrame,
                      carriers: dict, sample_library: dict | None, labels: dict, seed_label: str,
                      notes: list) -> dict:
    """{carriers: _Unit} of one comparison of a rebuilt fit, with the
    guides that can take part: tested in the results, coefficient
    estimable, measured in at least MIN_GROUP_SAMPLES samples of each
    compared group, residuals with a direction in the unit's basis."""
    tested = results.set_index('feature')
    candidates = []
    for feature, part in fit.features.items():
        if feature not in tested.index or not carriers.get(feature) or coef not in part.columns:
            continue
        t, df = float(tested.loc[feature, 't']), float(tested.loc[feature, 'df.total'])
        if np.isfinite(t) and not np.isnan(df):
            candidates.append((feature, part, t, df))
    again = rebuilt_t(fit, coef, [c[0] for c in candidates])
    by_unit: dict[tuple, list] = {}
    for feature, part, t, df in candidates:
        expected = part.residual_df + fit.prior_df
        if not (np.isclose(df, expected, rtol=1e-6, atol=1e-6)
                and np.isclose(abs(t), again[feature], rtol=1e-6, atol=1e-8)):
            raise PathwayError(
                f"The pathway analysis could not reproduce the statistics run's test of {feature} "
                f"({seed_label}: t {t:.6g} on {df:.6g} df against {again[feature]:.6g} on {expected:.6g}). "
                "Run the statistics again.")
        groups = [fit.group_vec[i] for i in part.observed]
        if groups.count(base) < MIN_GROUP_SAMPLES or groups.count(other) < MIN_GROUP_SAMPLES:
            continue
        by_unit.setdefault(carriers[feature], []).append((feature, part, float(t_to_z(t, df))))
    units = {}
    for unit, members in by_unit.items():
        if sample_library is None or unit == (_ALONE,):
            positions = np.arange(len(fit.samples))
        else:
            positions = np.array([i for i, s in enumerate(fit.samples) if sample_library.get(s) in unit])
        basis = residual_basis(fit.design[positions], coef)
        name = _unit_name(unit, labels)
        if basis is None:
            notes.append(f"{seed_label}: {name or 'the library'} has no residual degrees of freedom; "
                         "its guides are not tested in pathways.")
            continue
        index = {int(p): r for r, p in enumerate(positions)}
        features, zs, vectors = [], [], []
        for feature, part, z in members:
            if any(int(p) not in index for p in part.observed):
                continue
            residuals = np.zeros(positions.size)
            residuals[[index[int(p)] for p in part.observed]] = pearson_residuals(part)
            vector = guide_vector(z, residuals, basis)
            if vector is None:
                continue
            features.append(feature)
            zs.append(z)
            vectors.append(vector)
        if features:
            units[unit] = _Unit(name, features, np.array(zs), np.array(vectors), basis.shape[1])
    return units


def _seeded(seed: int, *labels: str) -> np.random.Generator:
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(x).encode()) for x in labels])


def run_pathways(stats: dict, fit_log: list, sets: pd.DataFrame, matches: pd.DataFrame, *,
                 rotations: int = DEFAULT_ROTATIONS, seed: int = DEFAULT_SEED,
                 min_size: int = DEFAULT_MIN_SIZE, max_fraction: float = MAX_SET_FRACTION,
                 full_size: pd.Series | None = None) -> dict:
    """
    Signed pathway analysis of a statistics run (``flowcode_stats.run_statistics``
    with *fit_log*, counts tested by the QL test).

    sets    : rows of ``load_signed_sets``.
    matches : ``match_features`` of the run's combinations.

    Returns {'pathways', 'libraries', 'members', 'qc', 'matches', 'notes',
    'settings'}: one row per pathway and comparison (combined over
    libraries), per pathway, comparison and library, per member, and the
    NON-Targetting pseudo-set check per comparison and library.
    """
    if stats.get('counts') is None:
        raise PathwayError('The pathway analysis needs the counts test.')
    if stats.get('counts_method') != 'ql':
        raise PathwayError('The pathway analysis needs the quasi-likelihood counts test.')
    if not fit_log:
        raise PathwayError('The statistics run kept no record of its fits; run the statistics again.')
    min_size = max(int(min_size), MIN_SIZE_FLOOR)
    results = stats['counts']
    labels = dict(stats.get('library_labels') or {})
    sample_library = stats.get('libraries')
    all_features = list(dict.fromkeys(f for r in fit_log for f in r['counts'].columns))
    carriers = _carriers(stats, all_features)
    match_of = matches.set_index('feature')
    genes = {f: str(match_of.loc[f, 'gene']) for f in all_features
             if f in match_of.index and str(match_of.loc[f, 'gene']) and len(carriers.get(f) or ()) == 1}
    tested = set(results.loc[np.isfinite(results['t'].astype(float)), 'feature'])
    genes = {f: g for f, g in genes.items() if f in tested}
    controls = [f for f in all_features if f in match_of.index and match_of.loc[f, 'match'] == MATCH_CONTROL]
    pathways = collapse_sets(sets, genes, min_size, max_fraction, full_size)
    notes: list = []

    rows, lib_rows, member_rows, qc_rows = [], [], [], []
    by_label = {lbl: res for lbl, res in results.groupby('comparison', sort=False)}
    for record in fit_log:
        if record.get('method') != 'ql':
            continue
        for fit, comparisons in _fits_of_record(record):
            for label, base, other in comparisons:
                coef = fit.group_columns[other]
                view = by_label.get(label)
                if view is None:
                    continue
                units = _comparison_units(fit, coef, base, other, view, carriers, sample_library, labels,
                                          label, notes)
                logfc = view.set_index('feature')['logFC'].astype(float)
                rows_c, lib_c, mem_c = _test_sets(pathways, units, logfc, genes, label, rotations, seed, min_size)
                rows.extend(rows_c)
                lib_rows.extend(lib_c)
                member_rows.extend(mem_c)
                qc_rows.extend(_control_check(units, controls, label, rotations, seed))

    table = pd.DataFrame(rows, columns=_PATHWAY_COLUMNS)
    for prefix in ('', 'competitive_', 'mixed_'):
        column = f'{prefix}FDR'
        table[column] = np.nan
        for _label, idx in table.groupby('comparison', sort=False).groups.items():
            table.loc[idx, column] = ms.bh_adjust(table.loc[idx, f'{prefix}P.Value'].values)
    return {
        'pathways': table,
        'libraries': pd.DataFrame(lib_rows, columns=_LIBRARY_COLUMNS),
        'members': pd.DataFrame(member_rows, columns=_MEMBER_COLUMNS),
        'qc': pd.DataFrame(qc_rows, columns=_QC_COLUMNS),
        'matches': matches,
        'notes': list(dict.fromkeys(notes)),
        'settings': {'rotations': int(rotations), 'seed': int(seed), 'min_size': min_size,
                     'max_fraction': float(max_fraction), 'pathways': len(pathways)},
    }


_PATHWAY_COLUMNS = ['comparison', 'set_key', 'set_name', 'go_ids', 'go_names', 'members', 'libraries', 'score',
                    'z', 'P.Value', 'competitive_z', 'competitive_P.Value', 'concordance', 'leading',
                    'mixed_score', 'mixed_z', 'mixed_P.Value']
_LIBRARY_COLUMNS = ['comparison', 'set_key', 'library', 'members', 'score', 'P.Value', 'z',
                    'competitive_t', 'competitive_P.Value', 'competitive_z', 'correlation',
                    'mixed_score', 'mixed_P.Value', 'mixed_z']
_MEMBER_COLUMNS = ['comparison', 'set_key', 'library', 'feature', 'gene', 'sign', 'logFC', 'z', 'contribution']
_QC_COLUMNS = ['comparison', 'library', 'members', 'mean_z', 'P.Value', 'competitive_P.Value']


def _test_sets(pathways: pd.DataFrame, units: dict, logfc: pd.Series, genes: dict, label: str, rotations: int,
               seed: int, min_size: int) -> tuple:
    """Rows of the pathway, library and member tables for one comparison."""
    per_set: dict[str, list] = {}
    lib_rows, member_rows = [], []
    for unit, data in units.items():
        position = {f: i for i, f in enumerate(data.features)}
        keys, member_lists = [], []
        for key, members in zip(pathways['set_key'], pathways['members']):
            inside = [(position[f], -float(s), f, s) for f, s in members if f in position]
            if inside:
                keys.append(key)
                member_lists.append(inside)
        if not keys:
            continue
        weights = np.zeros((len(data.features), len(keys)))
        for j, inside in enumerate(member_lists):
            for i, k, _f, _s in inside:
                weights[i, j] = k / len(inside)
        directions = random_directions(_seeded(seed, label, data.name), rotations, data.vectors.shape[1])
        scores, pvals, mixed, mixed_p = rotation_pvalues(data.z, data.vectors, weights, directions, mixed=True)
        residual_dirs = data.vectors[:, 1:] / np.maximum(
            np.linalg.norm(data.vectors[:, 1:], axis=1, keepdims=True), 1e-300)
        for j, (key, inside) in enumerate(zip(keys, member_lists)):
            idx = np.array([i for i, _k, _f, _s in inside])
            ks = np.array([k for _i, k, _f, _s in inside])
            camera = signed_camera(data.z, residual_dirs, idx, ks, data.residual_df, guides_df=True)
            z_rot = signed_z_of_p(pvals[j], scores[j])
            z_cam = signed_z_of_p(camera['p'], camera['delta'])
            z_mixed = float(-special.ndtri(min(max(float(mixed_p[j]), 1e-300), 1.0)))
            lib_rows.append((label, key, data.name, len(inside), float(scores[j]), float(pvals[j]), z_rot,
                             camera['t'], camera['p'], z_cam, camera['correlation'], float(mixed[j]),
                             float(mixed_p[j]), z_mixed))
            per_set.setdefault(key, []).append((len(inside), z_rot, z_cam, z_mixed))
            for i, k, feature, sign in inside:
                member_rows.append((label, key, data.name, feature, genes.get(feature, ''), int(sign),
                                    float(logfc.get(feature, np.nan)), float(data.z[i]), float(k * data.z[i])))
    members_of = {}
    for row in member_rows:
        members_of.setdefault(row[1], []).append(row)
    rows = []
    names = pathways.set_index('set_key')
    for key, parts in per_set.items():
        n = np.array([p[0] for p in parts])
        if n.sum() < min_size:
            continue
        z, p = stouffer(np.array([p[1] for p in parts]), n)
        zc, pc = stouffer(np.array([p[2] for p in parts]), n)
        zm, _two_sided = stouffer(np.array([p[3] for p in parts]), n)
        pm = float(special.ndtr(-zm)) if np.isfinite(zm) else np.nan
        mem = members_of[key]
        contributions = np.array([r[8] for r in mem])
        score = float(np.mean(contributions))
        mixed_score = float(np.mean([r[7] * r[7] for r in mem]))
        direction = np.sign(score) if score != 0 else 1.0
        agree = [np.sign(-r[5] * r[6]) == direction for r in mem if np.isfinite(r[6]) and r[6] != 0]
        order = np.argsort(-direction * contributions)[:LEADING_MEMBERS]
        leading = ', '.join(mem[i][4] or mem[i][3] for i in order)
        go_names = names.loc[key, 'go_names']
        rows.append((label, key, go_names[0], ';'.join(names.loc[key, 'go_ids']), ';'.join(go_names), int(n.sum()),
                     len(parts), score, z, p, zc, pc, float(np.mean(agree)) if agree else np.nan, leading,
                     mixed_score, zm, pm))
    return rows, lib_rows, member_rows


def _control_check(units: dict, controls: list, label: str, rotations: int, seed: int) -> list:
    """The NON-Targetting pseudo-set (all signs +1) in each unit that has
    two or more of them: it should not score."""
    out = []
    for _unit, data in units.items():
        idx = np.array([i for i, f in enumerate(data.features) if f in controls])
        if idx.size < 2:
            continue
        weights = np.zeros((len(data.features), 1))
        weights[idx, 0] = 1.0 / idx.size
        directions = random_directions(_seeded(seed, label, data.name, 'controls'), rotations,
                                       data.vectors.shape[1])
        score, p = rotation_pvalues(data.z, data.vectors, weights, directions)
        residual_dirs = data.vectors[:, 1:] / np.maximum(
            np.linalg.norm(data.vectors[:, 1:], axis=1, keepdims=True), 1e-300)
        camera = signed_camera(data.z, residual_dirs, idx, np.ones(idx.size), data.residual_df, guides_df=True)
        out.append((label, data.name, int(idx.size), float(score[0]), float(p[0]), camera['p']))
    return out
