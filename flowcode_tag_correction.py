"""
flowcode_tag_correction.py
--------------------------
Data-level correction of combination-specific tag spillover, for the
phenotype-shift readout of the FlowCode Analysis page. No Qt.

In plain terms
    A guide's tags cannot change where truly negative cells sit on a
    marker. If the negative cells of one combination move up or down a
    marker with how bright that combination's tags are, the movement is
    spillover or leftover FRET error, not biology. It is measured per
    combination and marker from the combination's own cells and subtracted
    cell by cell; cells without tags are not changed.

The model
    For one sample group, combination and marker, pooled over the group's
    samples, the marker value y of the combination's cells is modelled on
    six terms of its three tags: the tag amounts a1, a2, a3 (each tag's
    abundance divided by its debarcoding threshold) and their pairwise
    products a1·a2, a1·a3, a2·a3, the same terms as the backbone's FRET
    error model:

        y = c0 + b1·a1 + b2·a2 + b3·a3 + b12·a1·a2 + b13·a1·a3 + b23·a2·a3

    The fit is a Tukey-biweight robust regression (tuning 4.685): cells more
    than 4.685 robust SDs from the current fit get no weight, so on a marker
    whose main population is negative the fit follows the negatives and
    ignores the positives. It starts from a straight line through the
    half-sample modes (the densest point) of ten brightness bins, drawn as
    the median of the slopes between every pair of bins (Theil–Sen), so a
    bin whose main peak happens to be the positives cannot tilt it; a small
    ridge (0.001 × the weight total, on the standardised terms) keeps the
    correlated product terms stable. Only the six tag terms are subtracted
    (c0 is kept), so the correction is zero for a cell without tags.

Only where there is evidence
    The coefficients are estimated once per group from the pooled mice, so
    their estimation noise would move a combination's cells the same way
    in every mouse, which the shift test would read as a consistent shift.
    A combination's marker is therefore corrected only when its six tag
    terms are jointly significant, p below 0.01 by default: an F test
    comparing the weighted residual sum of squares of the fit with that of
    an intercept alone, both with the final biweight weights (the cells the
    fit follows). On synthetic negatives with no tag dependence it gave
    1.0–2.5 % at p < 0.01, against 5–10 % for a Wald test with a sandwich
    covariance, which underestimates the spread of the product terms.

Which markers
    A marker is corrected only when at least 30 % of the untransduced cells
    of the eligibility population are negative by the plugin's negative
    model (``flowcode_phenotype.positivity`` below one half). Otherwise its
    main peak may be the positives, and moving it would erase real
    expression. Markers whose negatives are better seen in another
    population (by default Foxp3 and Helios, absent from CD8 T cells) are
    fitted on the combination's cells in that anchor population and applied
    everywhere.

Deliberate simplifications of this first version
    * one robust fit per sample group, combination and marker, pooled over
      the group's samples (no per-sample term), applied only with evidence;
    * eligibility is judged once per group from the untransduced cells'
      negative share, with a fixed 30 % rule;
    * anchoring is a user list of markers fitted on one user gate; it is
      off when that gate is missing.

``estimate`` returns the coefficient table and ``Correction.from_table``
turns one group's rows into the per-cell correction applied by
``Correction.apply``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_CELLS_FIT = 200
CELLS_PER_SAMPLE = 2000
ELIGIBLE_NEGATIVE = 0.30
MIN_UNTRANSDUCED = 50
TUNING = 4.685
RIDGE = 1e-3
EVIDENCE_P = 0.01
ITERATIONS = 30
N_BINS = 10
DEFAULT_ANCHORED_MARKERS = ('FoxP3', 'Helios')

STATUS_OK = 'corrected'
STATUS_NOT_ELIGIBLE = 'not eligible'
STATUS_FEW_CELLS = 'too few cells'
STATUS_NO_FIT = 'no fit'
STATUS_NO_EVIDENCE = 'no evidence'
STATUSES = (STATUS_OK, STATUS_NO_EVIDENCE, STATUS_NOT_ELIGIBLE, STATUS_FEW_CELLS, STATUS_NO_FIT)
TERM_COLUMNS = ('t1', 't2', 't3', 't12', 't13', 't23')


def half_sample_mode(x) -> float:
    """Centre of the densest half of the values, halved again until three
    remain (Bickel and Frühwirth 2006): a robust estimate of the main peak."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    while len(x) > 3:
        h = (len(x) + 1) // 2
        widths = x[h - 1:] - x[:len(x) - h + 1]
        j = int(np.argmin(widths))
        x = x[j:j + h]
    return float(x.mean())


def tag_amounts(tag_values, thresholds) -> np.ndarray:
    """(n, 3) tag amounts: each tag's abundance divided by its threshold."""
    return np.asarray(tag_values, dtype=np.float64) / np.asarray(thresholds, dtype=np.float64)


def tag_terms(amounts) -> np.ndarray:
    """(n, 6) regressors of one combination's cells: its three tag amounts
    and their three pairwise products (order of ``TERM_COLUMNS``)."""
    x = np.asarray(amounts, dtype=np.float64)
    pairs = [x[:, a] * x[:, b] for a, b in itertools.combinations(range(x.shape[1]), 2)]
    return np.column_stack([x] + pairs)


def fit_slope(values, brightness) -> tuple | None:
    """Starting line (intercept, slope) of a marker on the combination's
    total tag amount: the half-sample modes of up to ten equal-count
    brightness bins (at least 10 cells each), joined by the median of the
    slopes between every pair of bins and the median intercept (Theil–Sen),
    so one or two bins whose main peak is the positives do not tilt it.
    None with fewer than three usable bins."""
    values = np.asarray(values, dtype=np.float64)
    brightness = np.asarray(brightness, dtype=np.float64)
    n_bins = int(min(N_BINS, len(values) // 20))
    if n_bins < 3:
        return None
    edges = np.quantile(brightness, np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (brightness >= lo) & (brightness <= hi)
        if m.sum() >= 10:
            xs.append(np.median(brightness[m]))
            ys.append(half_sample_mode(values[m]))
    if len(xs) < 3:
        return None
    x, y = np.array(xs), np.array(ys)
    i, j = np.triu_indices(len(x), 1)
    run = x[j] - x[i]
    keep = run > 0
    if not keep.any():
        return None
    gamma = float(np.median((y[j] - y[i])[keep] / run[keep]))
    alpha = float(np.median(y - gamma * x))
    return alpha, gamma


def fit_six(values, terms, start, iterations: int = ITERATIONS, tuning: float = TUNING,
            ridge: float = RIDGE) -> tuple | None:
    """
    Robust fit of a marker's main population on the six tag terms.

    Tukey-biweight weighted least squares starting from *start* (intercept,
    slope on each tag amount, none on the products). Returns (intercept,
    the six tag coefficients in the units of *terms*, the p-value that all
    six are zero), or None when the robust scale collapses or fewer than
    50 cells keep weight. The p-value is an F test with the final biweight
    weights held fixed: the weighted residual sum of squares of the six
    terms against an intercept alone, on 6 and (cells with weight − 7)
    degrees of freedom.
    """
    values = np.asarray(values, dtype=np.float64)
    scale_cols = np.maximum(terms.std(axis=0), 1e-12)
    design = np.column_stack([np.ones(len(values)), terms / scale_cols])
    beta = np.zeros(design.shape[1])
    beta[0] = start[0]
    beta[1:4] = start[1] * scale_cols[:3]
    for _ in range(iterations):
        r = values - design @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not s > 0:
            return None
        u = r / (tuning * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 50:
            return None
        penalty = np.eye(design.shape[1]) * ridge * w.sum()
        penalty[0, 0] = 0.0
        new = np.linalg.solve(design.T @ (design * w[:, None]) + penalty, design.T @ (w * values))
        if np.allclose(new, beta, rtol=1e-6, atol=1e-6):
            beta = new
            break
        beta = new
    return float(beta[0]), beta[1:] / scale_cols, _evidence(values, design, beta, tuning)


def _evidence(values, design, beta, tuning: float) -> float:
    """F-test p-value that the tag terms of *design* explain nothing, with
    the biweight weights of the fit *beta* held fixed."""
    from scipy.stats import f as f_dist

    r = values - design @ beta
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    if not s > 0:
        return float('nan')
    u = r / (tuning * s)
    w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    keep = w > 0
    df2 = int(keep.sum()) - design.shape[1]
    if df2 < 1:
        return float('nan')
    root = np.sqrt(w[keep])
    x, y = design[keep] * root[:, None], values[keep] * root
    rss1 = float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    rss0 = float(np.sum((y - x[:, :1] @ np.linalg.lstsq(x[:, :1], y, rcond=None)[0]) ** 2))
    if not rss1 > 0:
        return 0.0
    df1 = design.shape[1] - 1
    return float(f_dist.sf(((rss0 - rss1) / df1) / (rss1 / df2), df1, df2))


def negative_share(pi) -> float:
    """Share of untransduced cells that are negative (positivity below one
    half) in one sample; NaN with fewer than ``MIN_UNTRANSDUCED`` cells."""
    pi = np.asarray(pi, dtype=np.float64)
    if len(pi) < MIN_UNTRANSDUCED:
        return float('nan')
    return float(np.mean(pi < 0.5))


def eligible_markers(shares: dict, threshold: float = ELIGIBLE_NEGATIVE) -> pd.DataFrame:
    """
    *shares* is ``{fluorophore: [negative share of each sample]}`` (NaN for
    samples with too few untransduced cells). Returns one row per
    fluorophore: 'negative_share' (median over samples) and 'eligible'
    (at least *threshold*; False when no sample had enough cells).
    """
    rows = []
    for fluor, values in shares.items():
        v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
        share = float(np.median(v)) if v.size else float('nan')
        rows.append({'fluor': fluor, 'negative_share': share, 'eligible': bool(v.size and share >= threshold)})
    return pd.DataFrame(rows, columns=['fluor', 'negative_share', 'eligible'])


@dataclass
class FitCells:
    """One sample group's and combination's pooled cells for the fits:
    ``markers`` (n, markers) raw marker values in the order of the
    estimate's *fluors*, ``amounts`` (n, 3) tag amounts in the combination's
    tag order."""
    markers: list = field(default_factory=list)
    amounts: list = field(default_factory=list)

    def add(self, markers, amounts):
        if len(markers):
            self.markers.append(np.asarray(markers, dtype=np.float64))
            self.amounts.append(np.asarray(amounts, dtype=np.float64))

    def arrays(self) -> tuple:
        if not self.markers:
            return np.zeros((0, 0)), np.zeros((0, 3))
        return np.vstack(self.markers), np.vstack(self.amounts)


def fit_marker(values, amounts, evidence_p: float = EVIDENCE_P) -> tuple:
    """(status, start, six, p) of one marker on one combination's cells
    (*amounts*: their tag amounts); *six* is None unless the marker is
    corrected."""
    if len(values) < MIN_CELLS_FIT:
        return STATUS_FEW_CELLS, None, None, np.nan
    start = fit_slope(values, amounts.sum(axis=1))
    if start is None:
        return STATUS_NO_FIT, None, None, np.nan
    fit = fit_six(values, tag_terms(amounts), start)
    if fit is None:
        return STATUS_NO_FIT, start, None, np.nan
    _c0, six, p = fit
    if not p < evidence_p:
        return STATUS_NO_EVIDENCE, start, None, p
    return STATUS_OK, start, six, p


def estimate(fit_cells: dict, anchor_cells: dict, fluors: list, eligible: dict, anchored=(),
             combo_tags: dict | None = None, evidence_p: float = EVIDENCE_P) -> pd.DataFrame:
    """
    The coefficient table.

    fit_cells   : {(group, combination): FitCells} from the fit population.
    anchor_cells: the same from the anchor population (may be empty; an
                  anchored marker without anchor cells is not corrected).
    fluors      : marker fluorophores, the columns of ``FitCells.markers``.
    eligible    : {group: set of eligible fluorophores}.
    anchored    : fluorophores fitted on the anchor cells.
    combo_tags  : {combination: its three tag names}, for the table.
    evidence_p  : a fit is applied only when the Wald p-value of its six
                  tag terms is below this (1 applies every fit).

    One row per group, combination and marker: 'cells' (cells fitted),
    'anchored', 'status' (``STATUSES``), 'alpha' and 'gamma' (the starting
    line), 'evidence_p' and the six coefficients ``TERM_COLUMNS`` (NaN
    unless corrected).
    """
    anchored = set(anchored)
    rows = []
    for (group, combo), cells in fit_cells.items():
        values, amounts = cells.arrays()
        a_cells = anchor_cells.get((group, combo))
        a_values, a_amounts = a_cells.arrays() if a_cells is not None else (np.zeros((0, 0)), np.zeros((0, 3)))
        ok = eligible.get(group, set())
        for j, fluor in enumerate(fluors):
            anchor = fluor in anchored
            row = {'group': group, 'combination': combo,
                   'tags': ' '.join((combo_tags or {}).get(combo, ())), 'fluor': fluor, 'anchored': anchor}
            none = (STATUS_FEW_CELLS, None, None, np.nan)
            if fluor not in ok:
                (status, start, six, p), n = (STATUS_NOT_ELIGIBLE, None, None, np.nan), len(values)
            elif anchor:
                n = len(a_values)
                status, start, six, p = fit_marker(a_values[:, j], a_amounts, evidence_p) if n else none
            else:
                n = len(values)
                status, start, six, p = fit_marker(values[:, j], amounts, evidence_p) if n else none
            row.update({'cells': int(n), 'status': status,
                        'alpha': np.nan if start is None else start[0],
                        'gamma': np.nan if start is None else start[1], 'evidence_p': p})
            for k, name in enumerate(TERM_COLUMNS):
                row[name] = float(six[k]) if six is not None else np.nan
            rows.append(row)
    columns = ['group', 'combination', 'tags', 'fluor', 'anchored', 'cells', 'status', 'alpha', 'gamma',
               'evidence_p', *TERM_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


@dataclass
class Correction:
    """The correction of one sample group, ready to apply to events.

    ``terms`` maps a combination id (1-based, the ``FlowCode ID`` of the
    events) to (tag columns (3,), thresholds (3,), marker columns (m,),
    coefficients (6, m))."""
    terms: dict

    @classmethod
    def from_table(cls, table: pd.DataFrame, group, pnn: list, combo_names: list, combo_fluors: dict,
                   thresholds: dict) -> 'Correction':
        """One group's corrected rows of *table* (``estimate``) on events
        with columns *pnn*: *combo_names* are the file's combinations in
        ``FlowCode ID`` order, *combo_fluors* gives each combination's three
        tag fluorophores in its tag order and *thresholds* each tag
        fluorophore's raw threshold."""
        pnn = list(pnn)
        terms = {}
        if table is None or table.empty:
            return cls(terms)
        rows = table[(table['group'] == group) & (table['status'] == STATUS_OK)]
        for combo, sub in rows.groupby('combination', sort=False):
            if combo not in combo_names or combo not in combo_fluors:
                continue
            fl = list(combo_fluors[combo])
            if any(f not in pnn for f in fl) or any(f not in thresholds for f in fl):
                continue
            sub = sub[sub['fluor'].isin(pnn)]
            if sub.empty:
                continue
            terms[combo_names.index(combo) + 1] = (
                np.array([pnn.index(f) for f in fl]), np.array([float(thresholds[f]) for f in fl]),
                np.array([pnn.index(f) for f in sub['fluor']]), sub[list(TERM_COLUMNS)].to_numpy(float).T)
        return cls(terms)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def apply(self, events: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """A copy of *events* with each combination's tag terms subtracted
        from its corrected markers; other events are unchanged."""
        out = np.array(events, dtype=np.float64, copy=True)
        for cid, (tag_cols, thr, marker_cols, coef) in self.terms.items():
            m = ids == cid
            if not m.any():
                continue
            shift = tag_terms(tag_amounts(out[np.ix_(m, tag_cols)], thr)) @ coef
            out[np.ix_(m, marker_cols)] -= shift
        return out
