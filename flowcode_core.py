"""
flowcode_core.py
----------------
Numerics for the FlowCode plugin: a port of the FlowCodeUnmix 0.6.0 R
package's AF-matched staged pipeline. No Qt.

Pipeline
    1. Scatter gates on the backbone: an FSC Area/Height ratio singlet gate
       (``singlet_ratio_cutoff``, ``singlet_polygon``) and a landmark
       FSC/SSC gate around events clearly positive for three tags
       (``landmark_indices``, ``landmark_gate``).
    2. Per-cell AF unmixing of the gated backbone (``af_only_unmix``), scored
       with the AF profiles' negative envelopes as in the AutoSpectral AF
       tab: the debarcoding space in which positivity thresholds are set
       (``initial_threshold``).
    3. ``characterise_backbone``: debarcode with too-many and too-few-tag
       rescue, then model each combo's raw-space unmixing error (FRET plus
       residual spillover) as linear in its on-tag abundances plus their
       pairwise products, through the origin (``combo_error_fit``), plus a
       shared pairwise model for combos with too few fit events and events
       without a valid barcode (``PairModel``). Returns a ``FlowCodeSpectra``.
    4. ``unmix_staged`` on each sample: debarcode, subtract each event's
       characterised error from its raw data, then unmix the corrected data
       (per-cell AF and, optionally, fluorophore variants), and build the
       FlowCode and per-combo channels.

Conventions
    * Tag matrices (abundances, thresholds, ``logical``) have one column per
      tag, in ``FlowCodeDesign.tags`` order.
    * Combo codes per event: 0 = no positive tag (untransduced),
      1..G = combo ``code - 1`` of the design, G + 1 = unresolved (1, 2 or
      more than 3 positive tags, or a triplet that is not a combo).
    * Spectra are fluorophores x detectors; raw data are events x detectors
      in the same detector order.
    * All abundances are uncompensated.

R correspondence
    debarcode()                        -> debarcode
    .flowcode.debarcode.rescue()       -> debarcode_rescue
    .flowcode.rescue.toomany()         -> rescue_too_many
    (no R counterpart)                 -> rescue_too_few
    .weighted.median(), .node.medians  -> weighted_median, group_medians
    .flowcode.raw.error()              -> raw_error
    .flowcode.error.apply()            -> apply_error_correction
    get.flowcode.spectra()             -> characterise_backbone
    The error models (``combo_error_fit``, ``pair_model_fit``) replace R's
    centred per-combo fit and global error.
    unmix.flowcode.staged()            -> unmix_staged
    define.gate()                      -> landmark_indices + landmark_gate
    AutoSpectral::unmix.ols()          -> ols_unmix
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from honeychrome.controller_components.spectral_refinement import residual_spillover

logger = logging.getLogger(__name__)

# Epitope tags recognised automatically from channel antigens.
FLOWCODE_TAGS = ('HA', 'V5', 'AU1', 'AU5', 'S', 'C', 'Flag', 'NWS', 'VSVg', 'Ollas')

UNTRANSDUCED = 0
COMBO_CHANNEL_PREFIX = 'FC '
FLOWCODE_CHANNEL = 'FlowCode'
# Exported per-event barcode: 0 untransduced, 1..n combination (design order),
# n + 1 unresolved.
FLOWCODE_ID_CHANNEL = 'FlowCode ID'

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Bundled-plugin imports (kernel wrapper, automated gating thresholds)
# ---------------------------------------------------------------------------

def _ensure_bundled_plugins_on_path():
    import honeychrome
    bundled = str(Path(honeychrome.__file__).resolve().parent / 'bundled_plugins')
    if bundled not in sys.path:
        sys.path.insert(0, bundled)


def kernel_available() -> bool:
    """True when the compiled AutoSpectral Optimization kernel is built."""
    _ensure_bundled_plugins_on_path()
    try:
        from autospectral_opt_kernel_wrapper import AUTOSPECTRAL_OPT_KERNEL_AVAILABLE
    except ImportError:
        return False
    return bool(AUTOSPECTRAL_OPT_KERNEL_AVAILABLE)


# ---------------------------------------------------------------------------
# Design: tags, fluorophores and combos
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FlowCodeDesign:
    """FlowCode tags, their fluorophores and the valid triplet combos.

    ``logical`` is combos x tags, 1 where the combo carries the tag.
    """
    tags: tuple
    fluors: tuple
    combo_names: tuple
    combo_tags: tuple
    logical: np.ndarray

    @classmethod
    def build(cls, tag_to_fluor: dict, combos) -> 'FlowCodeDesign':
        """
        tag_to_fluor : {tag name: fluorophore label}, in tag order.
        combos       : iterable of (name, (tag1, tag2, tag3)); tags are
                       matched to ``tag_to_fluor`` case-insensitively.

        Raises ValueError for fewer than 3 tags, tags whose names differ only
        in case, combos that are not three distinct known tags, and duplicate
        names or tag triplets.
        """
        tags = tuple(str(t) for t in tag_to_fluor)
        fluors = tuple(str(tag_to_fluor[t]) for t in tag_to_fluor)
        if len(tags) < 3:
            raise ValueError('At least 3 FlowCode tags are needed.')
        upper = {t.upper(): t for t in tags}
        if len(upper) != len(tags):
            raise ValueError('FlowCode tag names must differ by more than case.')
        if len(set(fluors)) != len(fluors):
            raise ValueError('Each FlowCode tag needs its own fluorophore.')

        names, triplets, rows, seen = [], [], [], {}
        for name, combo in combos:
            name = str(name).strip()
            combo = [str(t).strip() for t in combo]
            unknown = [t for t in combo if t.upper() not in upper]
            if unknown:
                raise ValueError(f'Combo "{name}": unknown tag(s) {", ".join(unknown)}.')
            canonical = tuple(upper[t.upper()] for t in combo)
            if len(canonical) != 3 or len(set(canonical)) != 3:
                raise ValueError(f'Combo "{name}" must have three different tags.')
            if name in names:
                raise ValueError(f'Combo name "{name}" is used twice.')
            key = combo_key(canonical)
            if key in seen:
                raise ValueError(f'Combos "{seen[key]}" and "{name}" have the same tags.')
            seen[key] = name
            names.append(name)
            triplets.append(canonical)
            rows.append([1 if t in canonical else 0 for t in tags])
        if not names:
            raise ValueError('No FlowCode combos defined.')
        return cls(tags=tags, fluors=fluors, combo_names=tuple(names),
                   combo_tags=tuple(triplets), logical=np.array(rows, dtype=np.int8))

    @property
    def n_tags(self) -> int:
        return len(self.tags)

    @property
    def n_combos(self) -> int:
        return len(self.combo_names)

    @property
    def unresolved_code(self) -> int:
        return self.n_combos + 1

    def on_tags(self, g: int) -> np.ndarray:
        """Tag indices carried by combo *g* (0-based)."""
        return np.flatnonzero(self.logical[g])


def combo_key(tags) -> str:
    """Order-independent key of a tag set, as debarcode() builds it in R."""
    return '_'.join(sorted(str(t).upper() for t in tags))


def combo_channel_name(name: str) -> str:
    """Exported channel name for a combo."""
    return f'{COMBO_CHANNEL_PREFIX}{name}'


def combos_keyword(design) -> str:
    """The ``FLOWCODECOMBOS`` keyword of an export: ``FC <name>=<tag key>``
    for each combination, joined by ';'."""
    return ';'.join(f'{combo_channel_name(n)}={combo_key(t)}'
                    for n, t in zip(design.combo_names, design.combo_tags))


def all_triplets(tags) -> list:
    """Every 3-tag combination, as (name, tags), named TAG1_TAG2_TAG3."""
    ordered = sorted(tags, key=lambda t: str(t).upper())
    return [('_'.join(c), tuple(c)) for c in itertools.combinations(ordered, 3)]


def read_combinations_csv(path) -> list:
    """
    Combos from a FlowCodeUnmix combinations CSV: a header row, then one row
    per combo with its name in the first column (``Id``) and its three tags
    in the next three (``Procode.tag1..3``). Blank rows are skipped.

    Returns a list of (name, (tag1, tag2, tag3)). Raises ValueError, naming
    the offending rows, for missing names or tags.
    """
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError('The combinations file is empty.')
    header, body = rows[0], rows[1:]
    if len(header) < 4:
        raise ValueError('The combinations file needs an Id column and three tag columns.')
    combos, problems = [], []
    for line_no, row in enumerate(body, start=2):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        cells = (cells + [''] * 4)[:4]
        if not all(cells):
            problems.append(line_no)
            continue
        combos.append((cells[0], tuple(cells[1:4])))
    if problems:
        raise ValueError('Missing Id or tag on line(s) ' + ', '.join(map(str, problems)) + '.')
    if not combos:
        raise ValueError('The combinations file has no combos.')
    return combos


def write_combinations_csv(path, design: FlowCodeDesign):
    """Write the design's combos in the FlowCodeUnmix CSV format."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Id', 'Procode.tag1', 'Procode.tag2', 'Procode.tag3'])
        for name, tags in zip(design.combo_names, design.combo_tags):
            writer.writerow([name, *tags])


def validate_combo_names(names, existing_channels) -> list:
    """Problems with combo names as exported channel names (empty if none)."""
    problems = []
    existing = set(existing_channels)
    seen = set()
    for name in names:
        name = str(name)
        if not name.strip():
            problems.append('A combo has no name.')
            continue
        if ',' in name:
            problems.append(f'"{name}": commas are not allowed in channel names.')
        if name in seen:
            problems.append(f'"{name}" is used twice.')
        seen.add(name)
        channel = combo_channel_name(name)
        if channel in existing:
            problems.append(f'"{channel}" is already a channel name.')
    return problems


def recognise_tag(antigen: str, marker_db=None) -> str | None:
    """The FlowCode tag an antigen label names (canonical spelling), or None.

    Uses the marker database's synonyms (e.g. "AU1 tag" -> "AU1"), then
    requires the canonical marker to be a FlowCode tag, so single-letter tags
    never match other markers by substring.
    """
    if not antigen or not str(antigen).strip():
        return None
    from honeychrome.controller_components.label_matching import get_marker_db, match_marker
    canonical = match_marker(str(antigen), marker_db if marker_db is not None else get_marker_db())
    if canonical is None:
        return None
    for tag in FLOWCODE_TAGS:
        if tag.upper() == canonical.upper():
            return tag
    return None


def cosine_partners(spectra: np.ndarray) -> np.ndarray:
    """For each row, the index of the other row with the highest cosine
    similarity (the threshold plot's y-axis channel)."""
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    unit = spectra / np.maximum(norms, _EPS)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    return np.argmax(sim, axis=1)


# ---------------------------------------------------------------------------
# Unmixing helpers
# ---------------------------------------------------------------------------

def ols_unmix(raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """OLS abundances via the SVD pseudoinverse (AutoSpectral::unmix.ols)."""
    return raw @ np.linalg.pinv(basis.T).T


def af_mean_spectrum(unstained_raw: np.ndarray) -> np.ndarray:
    """Mean raw spectrum of the gated matched unstained, max-|value| normalised."""
    mean = unstained_raw.mean(axis=0)
    return mean / np.max(np.abs(mean))


def af_only_unmix(raw_fl: np.ndarray, spectra: np.ndarray, fluor_names, af_spectra: np.ndarray,
                  cell_weight: bool = False, n_threads: int | None = None,
                  use_kernel: bool = False, envelope: dict | None = None) -> dict:
    """
    Per-cell AF unmixing without fluorophore variants.

    By default Honeychrome's Python per-cell AF (batched matrix products),
    which ignores ``cell_weight``. With ``use_kernel`` and the kernel built,
    the joint kernel in AF-only mode instead, as
    ``unmix.autospectral.rcpp(pipeline = "joint", spectra.variants = NULL)``
    does in R. ``envelope`` is the AF library's negative envelope
    (``af_envelope_for_profiles``): with it both score the AF candidates by
    the positivity-weighted term and the Python path re-fits events with a
    saturated detector on the remaining detectors; without it they use the
    joint L2 score. The Python path was about twice as fast in the benchmark.

    Returns ``unmixed`` (n x F), ``af_scale``, ``af_idx`` (1-based into
    ``af_spectra``) and ``engine`` ('kernel' or 'python').
    """
    if af_spectra.shape[0] < 2:
        raise ValueError('Per-cell AF unmixing needs at least 2 AF spectra.')
    n_fluor = spectra.shape[0]
    if use_kernel and kernel_available():
        from autospectral_opt_kernel_wrapper import unmix_autospectral_joint
        threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
        result = unmix_autospectral_joint(
            raw_data_in=raw_fl, spectra=spectra, af_spectra=af_spectra,
            fluor_names=list(fluor_names), pos_thresholds=np.zeros(n_fluor), variants=[],
            n_threads=threads, cell_weight=cell_weight, envelope=envelope,
        )
        return {'unmixed': result[:, :n_fluor], 'af_scale': result[:, n_fluor],
                'af_idx': result[:, n_fluor + 1].astype(np.int64), 'engine': 'kernel'}

    from honeychrome.controller_components.autospectral_functions import (
        apply_af_unmixing, build_af_precomputed,
    )
    precomputed = build_af_precomputed(spectra, af_spectra, envelope)
    result = apply_af_unmixing(np.ascontiguousarray(raw_fl, dtype=np.float64), precomputed, af_spectra)
    return {'unmixed': result['unmixed'], 'af_scale': result['af_scale'],
            'af_idx': np.asarray(result['af_idx'], dtype=np.int64), 'engine': 'python'}


def logicle_transform(magnitude_ceiling: float, w: float = 0.5, m: float = 4.5, a: float = 0.0):
    """Fixed logicle used for the rescue and initial thresholds (FlowKit)."""
    from flowkit import transforms
    xform = transforms.LogicleTransform(param_t=float(magnitude_ceiling), param_w=w, param_m=m, param_a=a)
    return lambda values: np.asarray(xform.apply(np.asarray(values, dtype=np.float64)), dtype=np.float64)


def logicle_inverse(magnitude_ceiling: float, w: float = 0.5, m: float = 4.5, a: float = 0.0):
    """Inverse of ``logicle_transform`` with the same parameters."""
    from flowkit import transforms
    xform = transforms.LogicleTransform(param_t=float(magnitude_ceiling), param_w=w, param_m=m, param_a=a)
    return lambda values: np.asarray(xform.inverse(np.asarray(values, dtype=np.float64)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Small statistics
# ---------------------------------------------------------------------------

def weighted_median(x, w) -> float:
    """First value at which the cumulative weight reaches half the total."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x, kind='stable')
    cw = np.cumsum(w[order]) / np.sum(w)
    return float(x[order][np.flatnonzero(cw >= 0.5)[0]])


def group_medians(mat: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Row g holds the column medians of the rows of *mat* with code g + 1;
    NaN for groups with no rows."""
    out = np.full((n_groups, mat.shape[1]), np.nan)
    codes = np.asarray(codes)
    if codes.size == 0:
        return out
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    groups, starts = np.unique(sorted_codes, return_index=True)
    ends = np.append(starts[1:], len(sorted_codes))
    for g, s, e in zip(groups, starts, ends):
        if 1 <= g <= n_groups:
            out[int(g) - 1] = np.median(mat[order[s:e]], axis=0)
    return out


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve ``a x = b``; None when *a* is computationally singular.

    Singular means a reciprocal condition number (smallest / largest
    singular value) below ``max(a.shape) * eps``, as R's solve() rejects a
    reciprocal condition number below eps. Using singular values rather than
    LAPACK's zero-pivot report gives the same answer on every BLAS.
    """
    sv = np.linalg.svd(a, compute_uv=False)
    if sv.size == 0 or not np.all(np.isfinite(sv)) or sv[-1] <= sv[0] * max(a.shape) * _EPS:
        return None
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None


def _transform_finite(transform, x: np.ndarray) -> np.ndarray:
    """Apply *transform* to finite values only; NaN elsewhere."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.any():
        out[ok] = transform(x[ok])
    return out


def _transform_columns(transform, mat: np.ndarray) -> np.ndarray:
    return np.column_stack([_transform_finite(transform, mat[:, j]) for j in range(mat.shape[1])])


# ---------------------------------------------------------------------------
# Debarcoding
# ---------------------------------------------------------------------------

def debarcode(tag_abundances: np.ndarray, thresholds: np.ndarray, design: FlowCodeDesign) -> np.ndarray:
    """
    Combo code per event (see module docstring for codes).

    An event with more than three tags above threshold keeps only its three
    strongest when the third-strongest signal over threshold is more than
    twice the fourth. Only events with exactly three positive tags can match
    a combo.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    above = tag_abundances > thresholds
    count = above.sum(axis=1)

    multi = np.flatnonzero(count > 3)
    if multi.size:
        signal = tag_abundances[multi] - thresholds
        order = np.argsort(-signal, axis=1, kind='stable')
        ranked = np.take_along_axis(signal, order, axis=1)
        trim = ranked[:, 2] > 2.0 * ranked[:, 3]
        rows = multi[trim]
        above[rows[:, None], order[trim, 3:]] = False
        count = above.sum(axis=1)

    ids = np.full(len(tag_abundances), design.unresolved_code, dtype=np.int64)
    ids[count == 0] = UNTRANSDUCED

    bits = 1 << np.arange(design.n_tags, dtype=np.int64)
    combo_masks = design.logical.astype(np.int64) @ bits
    lookup = {}
    for g, mask in enumerate(combo_masks):
        lookup.setdefault(int(mask), g + 1)
    triplet = np.flatnonzero(count == 3)
    if triplet.size:
        event_masks = above[triplet].astype(np.int64) @ bits
        ids[triplet] = [lookup.get(int(m), design.unresolved_code) for m in event_masks]
    return ids


@dataclass
class DebarcodeResult:
    """Combo codes after rescue, plus the indices and medians the
    characterisation needs. ``rescued`` are the too-many-tag rescues,
    ``rescued_toofew`` the too-few-tag rescues."""
    ids: np.ndarray
    confident: np.ndarray
    rescued: np.ndarray
    combo_median: np.ndarray
    rescue: dict | None = None
    rescued_toofew: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    rescue_toofew: dict | None = None


def rescue_too_many(tag_abundances: np.ndarray, ids: np.ndarray, toomany_idx: np.ndarray,
                    reference_median: np.ndarray, design: FlowCodeDesign, transform,
                    floor: np.ndarray, tag_shift_min_n: int = 100, flat_threshold: float = 0.9,
                    gate_frac: float = 0.5, ref_idx: np.ndarray | None = None) -> dict:
    """
    Assign events with more than three positive tags to the combo whose
    reference median they most resemble.

    Similarity is cosine similarity in transformed space, with each tag
    scaled from its noise floor (``floor``) to the 99.5th percentile of all
    events. Reference medians are shifted per tag by this data's median
    offset from the reference (estimated on confident events, restricted to
    ``ref_idx`` when given, once a tag has ``tag_shift_min_n`` supporting
    events). An event is accepted when its best similarity is at least
    ``flat_threshold`` and beats the runner-up by ``gate_frac`` of the best
    combo's distance (1 − cosine) to its nearest rival combo.

    Returns ``rescued_idx``, ``assigned`` (combo codes) and ``diagnostics``.
    """
    n_combo = design.n_combos
    valid_idx = np.flatnonzero((ids >= 1) & (ids <= n_combo))
    if ref_idx is not None:
        valid_idx = np.intersect1d(valid_idx, ref_idx)
    confident_combo = ids[valid_idx]
    sample_n = np.bincount(confident_combo, minlength=n_combo + 1)[1:n_combo + 1]
    sample_median = group_medians(tag_abundances[valid_idx], confident_combo, n_combo)

    ref_t = _transform_columns(transform, np.asarray(reference_median, dtype=float))
    sample_t = _transform_columns(transform, sample_median)

    shift = np.zeros(design.n_tags)
    support = np.zeros(design.n_tags, dtype=np.int64)
    for f in range(design.n_tags):
        combos_f = np.flatnonzero(design.logical[:, f] == 1)
        have = combos_f[(sample_n[combos_f] > 0) & np.isfinite(sample_t[combos_f, f])
                        & np.isfinite(ref_t[combos_f, f])]
        support[f] = int(sample_n[have].sum())
        if have.size and support[f] >= tag_shift_min_n:
            shift[f] = weighted_median(sample_t[have, f] - ref_t[have, f], sample_n[have])

    shifted = ref_t + design.logical * shift

    tag_t = _transform_columns(transform, tag_abundances)
    floor_t = _transform_finite(transform, np.asarray(floor, dtype=float))
    ceil_t = np.nanquantile(tag_t, 0.995, axis=0)
    span = np.maximum(ceil_t - floor_t, _EPS)

    def scale_rows(x):
        return np.maximum((x - floor_t) / span, 0.0)

    ref_scaled = scale_rows(shifted)
    ref_scaled[~np.isfinite(ref_scaled)] = 0.0
    ref_norm = np.maximum(np.sqrt(np.sum(ref_scaled ** 2, axis=1)), _EPS)

    combo_sim = (ref_scaled @ ref_scaled.T) / np.outer(ref_norm, ref_norm)
    np.fill_diagonal(combo_sim, -np.inf)
    rival = combo_sim.max(axis=1)
    rival[~np.isfinite(rival)] = 0.0

    toomany_idx = np.asarray(toomany_idx, dtype=np.int64)
    rescued_idx = np.zeros(0, dtype=np.int64)
    assigned = np.zeros(0, dtype=np.int64)
    similarity = None
    if toomany_idx.size:
        events = scale_rows(tag_t[toomany_idx])
        events_norm = np.sqrt(np.sum(events ** 2, axis=1))
        sim = (events @ ref_scaled.T) / np.outer(np.maximum(events_norm, _EPS), ref_norm)
        sim[events_norm <= 0] = 0.0

        rows = np.arange(len(sim))
        best = np.argmax(sim, axis=1)
        s1 = sim[rows, best]
        sim2 = sim.copy()
        sim2[rows, best] = -np.inf
        s2 = sim2.max(axis=1)
        best2 = np.argmax(sim2, axis=1)

        gate = gate_frac * (1.0 - rival[best])
        accept = (s1 >= flat_threshold) & ((s1 - s2) >= gate)
        rescued_idx = toomany_idx[accept]
        assigned = best[accept] + 1
        similarity = {'idx': toomany_idx, 'best': best + 1, 'best2': best2 + 1,
                      's1': s1, 's2': s2, 'gate': gate, 'accepted': accept}

    return {
        'rescued_idx': rescued_idx,
        'assigned': assigned,
        'diagnostics': {
            'n_candidates': int(toomany_idx.size),
            'n_rescued': int(rescued_idx.size),
            'rescued_per_combo': np.bincount(assigned, minlength=n_combo + 1)[1:],
            'tag_shift': shift,
            'tag_support': support,
            'rival': rival,
            'similarity': similarity,
        },
    }


@dataclass
class TooFewSettings:
    """Settings of the too-few-tag rescue (``rescue_too_few``)."""
    flat_threshold: float = 0.9
    gate_frac: float = 0.5
    min_z: float = 2.0
    min_frac: float = 0.25
    af_ratio_max: float = 1.0
    n0: float = 50.0

    @classmethod
    def from_settings(cls, settings: dict) -> 'TooFewSettings | None':
        """From plugin / characterisation settings; None when the rescue is off."""
        settings = settings or {}
        if not settings.get('rescue_toofew', True):
            return None
        default = cls()
        return cls(
            flat_threshold=float(settings.get('rescue_toofew_flat_threshold', default.flat_threshold)),
            gate_frac=float(settings.get('rescue_toofew_gate_frac', default.gate_frac)),
            min_z=float(settings.get('rescue_toofew_min_z', default.min_z)),
            min_frac=float(settings.get('rescue_toofew_min_frac', default.min_frac)),
            af_ratio_max=float(settings.get('rescue_toofew_af_ratio', default.af_ratio_max)),
            n0=float(settings.get('rescue_toofew_n0', default.n0)),
        )


def af_ratio(tag_abundances: np.ndarray, af_scale: np.ndarray, thresholds: np.ndarray,
             background: np.ndarray) -> np.ndarray:
    """Per-cell AF abundance over the event's largest above-threshold tag
    abundance (measured from *background*); inf without a positive tag."""
    pos = np.where(tag_abundances > thresholds, tag_abundances - background, 0.0).max(axis=1)
    return np.where(pos > 0, np.asarray(af_scale, dtype=float) / np.maximum(pos, _EPS), np.inf)


@dataclass(eq=False)
class TooFewReference:
    """Background, scales and combo reference directions of the too-few rescue."""
    background: np.ndarray       # untransduced median per tag
    noise: np.ndarray            # untransduced robust SD per tag
    span: np.ndarray             # threshold minus background per tag
    centroid: np.ndarray         # combos x tags, abundance over background
    unit: np.ndarray             # centroid / span, unit rows
    rival: np.ndarray            # cosine to the nearest other combo
    sample_n: np.ndarray         # confident events per combo
    weight: np.ndarray           # weight of this data's own medians
    tag_factor: np.ndarray       # this data / reference on-tag level per tag


def too_few_reference(tag_abundances: np.ndarray, ids: np.ndarray, design: FlowCodeDesign,
                      thresholds: np.ndarray, untransduced_idx: np.ndarray, confident_idx: np.ndarray,
                      reference_median: np.ndarray | None = None,
                      reference_background: np.ndarray | None = None, n0: float = 50.0,
                      tag_shift_min_n: int = 100) -> TooFewReference:
    """
    Reference vector of each combo for the too-few rescue: the median
    abundance over the untransduced median of this data's confident events,
    shrunk towards the reference (backbone) combo median with weight
    n / (n + n0). The reference is measured from ``reference_background``
    (this data's untransduced median when None) and scaled per tag to this
    data's expression level, estimated over every combo carrying the tag
    once those combos have ``tag_shift_min_n`` events. Without
    ``reference_median`` this data's own medians are used.
    """
    n_combo = design.n_combos
    thresholds = np.asarray(thresholds, dtype=float)
    background = np.median(tag_abundances[untransduced_idx], axis=0)
    noise = np.maximum(1.4826 * np.median(np.abs(tag_abundances[untransduced_idx] - background), axis=0), _EPS)
    span = np.maximum(thresholds - background, noise)

    codes = ids[confident_idx]
    sample_n = np.bincount(codes, minlength=n_combo + 1)[1:n_combo + 1]
    own = group_medians(tag_abundances[confident_idx] - background, codes, n_combo)
    factor = np.ones(design.n_tags)
    if reference_median is None:
        prior = np.nan_to_num(own)
        weight = np.ones(n_combo)
    else:
        ref_bg = background if reference_background is None else np.asarray(reference_background, dtype=float)
        prior = np.nan_to_num(np.asarray(reference_median, dtype=float) - ref_bg)
        for f in range(design.n_tags):
            on = np.flatnonzero(design.logical[:, f] == 1)
            ok = on[(sample_n[on] > 0) & (np.nan_to_num(own[on, f]) > 0) & (prior[on, f] > 0)]
            if ok.size and sample_n[ok].sum() >= tag_shift_min_n:
                factor[f] = np.exp(weighted_median(np.log(own[ok, f] / prior[ok, f]), sample_n[ok]))
        prior = prior * np.where(design.logical == 1, factor[None, :], 1.0)
        weight = sample_n / (sample_n + float(n0)) if n0 > 0 else (sample_n > 0).astype(float)
    own = np.where(np.isfinite(own), own, prior)
    centroid = weight[:, None] * own + (1.0 - weight[:, None]) * prior

    unit = centroid / span
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), _EPS)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    rival = sim.max(axis=1) if n_combo > 1 else np.zeros(n_combo)
    return TooFewReference(background=background, noise=noise, span=span, centroid=centroid, unit=unit,
                           rival=rival, sample_n=sample_n, weight=weight, tag_factor=factor)


def too_few_score(tag_abundances: np.ndarray, above: np.ndarray, ref: TooFewReference,
                  design: FlowCodeDesign) -> dict:
    """
    Cosine similarity of each event (abundance over the background, in
    units of threshold minus background) to every combo reference, among
    the combos that contain all of the event's positive tags (*above*):
    ``best`` (0-based combo), ``s1`` and the runner-up ``s2``. For the best
    combo's on-tags below threshold, ``z_dropped`` is the smallest
    abundance over the background in untransduced SDs and ``frac_dropped``
    the smallest as a fraction of threshold minus background.
    """
    n = len(tag_abundances)
    if n == 0:
        empty = np.zeros(0)
        return {'best': np.zeros(0, dtype=np.int64), 's1': empty, 's2': empty, 'z_dropped': empty,
                'frac_dropped': empty}
    delta = tag_abundances - ref.background
    e = delta / ref.span
    norm = np.linalg.norm(e, axis=1)
    sim = (e @ ref.unit.T) / np.maximum(norm, _EPS)[:, None]
    conflict = above.astype(np.int64) @ (design.logical == 0).astype(np.int64).T > 0
    sim[conflict] = -np.inf
    rows = np.arange(n)
    best = np.argmax(sim, axis=1)
    s1 = sim[rows, best]
    sim[rows, best] = -np.inf
    s2 = sim.max(axis=1)
    s2[~np.isfinite(s2)] = -1.0
    dropped = (design.logical[best] == 1) & ~above
    z_dropped = np.where(dropped, delta / ref.noise, np.inf).min(axis=1)
    frac_dropped = np.where(dropped, delta / ref.span, np.inf).min(axis=1)
    return {'best': best, 's1': s1, 's2': s2, 'z_dropped': z_dropped, 'frac_dropped': frac_dropped}


def too_few_accept(score: dict, ref: TooFewReference, settings: TooFewSettings) -> np.ndarray:
    """Events of ``too_few_score`` that pass every criterion of *settings*."""
    s1, best = score['s1'], score['best']
    if s1.size == 0:
        return np.zeros(0, dtype=bool)
    gate = settings.gate_frac * (1.0 - ref.rival[best])
    return (np.isfinite(s1) & (s1 >= settings.flat_threshold) & ((s1 - score['s2']) >= gate)
            & (score['z_dropped'] >= settings.min_z) & (score['frac_dropped'] >= settings.min_frac))


def rescue_too_few(tag_abundances: np.ndarray, ids: np.ndarray, thresholds: np.ndarray, design: FlowCodeDesign,
                   settings: TooFewSettings, af_scale: np.ndarray | None = None,
                   usable: np.ndarray | None = None, confident_idx: np.ndarray | None = None,
                   reference_median: np.ndarray | None = None,
                   reference_background: np.ndarray | None = None, tag_shift_min_n: int = 100) -> dict:
    """
    Assign unresolved events with one or two positive tags to the combo
    whose tag pattern they share at a lower expression level.

    Candidates are unresolved events with one or two tags above threshold
    that are *usable* (e.g. no saturated detector) and not AF-dominated
    (``af_ratio`` above ``settings.af_ratio_max``; skipped without
    *af_scale*). Each candidate's abundance over the untransduced median,
    in units of threshold minus that median, is compared by cosine
    similarity with each combo's reference (``too_few_reference``, built
    from the confident events that are usable and not AF-dominated) among
    the combos that contain all of its positive tags. It is accepted when
    the similarity is at least ``flat_threshold``, beats the next such
    combo by ``gate_frac`` x (1 - the cosine between the best combo and its
    nearest rival), and every on-tag of that combo below threshold is at
    least ``min_z`` untransduced SDs and ``min_frac`` x (threshold - median)
    above the untransduced median. Scale-free in each event's expression
    level, so a uniformly dimmer cell of a combo scores as its bright cells
    do.

    Returns ``rescued_idx``, ``assigned`` (combo codes) and ``diagnostics``.
    """
    n_combo = design.n_combos
    thresholds = np.asarray(thresholds, dtype=float)
    usable = np.ones(len(ids), dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    if confident_idx is None:
        confident_idx = np.flatnonzero((ids >= 1) & (ids <= n_combo))
    untransduced = np.flatnonzero((ids == UNTRANSDUCED) & usable)
    empty = np.zeros(0, dtype=np.int64)
    out = {'rescued_idx': empty, 'assigned': empty,
           'diagnostics': {'n_candidates': 0, 'n_rescued': 0, 'n_af_dominated': 0,
                           'rescued_per_combo': np.zeros(n_combo, dtype=np.int64)}}
    if untransduced.size < 2:
        return out
    background = np.median(tag_abundances[untransduced], axis=0)
    clean = usable.copy()
    if af_scale is not None:
        clean &= af_ratio(tag_abundances, af_scale, thresholds, background) <= settings.af_ratio_max
    above = tag_abundances > thresholds
    count = above.sum(axis=1)
    unresolved = ids == design.unresolved_code
    candidates = np.flatnonzero(unresolved & (count >= 1) & (count <= 2) & clean)
    confident = np.asarray(confident_idx, dtype=np.int64)
    confident = confident[clean[confident]]
    if candidates.size == 0 or confident.size == 0:
        out['diagnostics']['n_candidates'] = int(candidates.size)
        return out

    ref = too_few_reference(tag_abundances, ids, design, thresholds, untransduced, confident,
                            reference_median=reference_median, reference_background=reference_background,
                            n0=settings.n0, tag_shift_min_n=tag_shift_min_n)
    score = too_few_score(tag_abundances[candidates], above[candidates], ref, design)
    accept = too_few_accept(score, ref, settings)
    rescued_idx = candidates[accept]
    assigned = score['best'][accept] + 1
    return {
        'rescued_idx': rescued_idx,
        'assigned': assigned,
        'diagnostics': {
            'n_candidates': int(candidates.size),
            'n_rescued': int(rescued_idx.size),
            'n_af_dominated': int(np.sum(unresolved & usable & ~clean)),
            'rescued_per_combo': np.bincount(assigned, minlength=n_combo + 1)[1:],
            'tag_factor': ref.tag_factor,
            'reference_weight': ref.weight,
            'rival': ref.rival,
            'similarity': {'idx': candidates, 'best': score['best'] + 1, 's1': score['s1'], 's2': score['s2'],
                           'z_dropped': score['z_dropped'], 'frac_dropped': score['frac_dropped'],
                           'accepted': accept},
        },
    }


def debarcode_rescue(tag_abundances: np.ndarray, thresholds: np.ndarray, design: FlowCodeDesign,
                     transform=None, reference_median: np.ndarray | None = None,
                     floor: np.ndarray | None = None, rescue: bool = True,
                     flat_threshold: float = 0.9, gate_frac: float = 0.5,
                     tag_shift_min_n: int = 100, ref_idx: np.ndarray | None = None,
                     too_few: TooFewSettings | None = None, af_scale: np.ndarray | None = None,
                     usable: np.ndarray | None = None,
                     reference_background: np.ndarray | None = None) -> DebarcodeResult:
    """
    Debarcode, rescue events with more than three positive tags, then
    (with ``too_few``) events with one or two (``rescue_too_few``).

    Without ``reference_median`` (the backbone) the reference is this data's
    own confident-event medians, so the per-tag shift is zero. The
    too-many rescue is skipped when ``floor`` is None (no positivity noise
    floor available). ``af_scale``, ``usable`` and ``reference_background``
    (the reference's untransduced median) are passed to the too-few rescue.
    """
    n_combo = design.n_combos
    ids = debarcode(tag_abundances, thresholds, design)
    confident = np.flatnonzero((ids >= 1) & (ids <= n_combo))
    own_median = group_medians(tag_abundances[confident], ids[confident], n_combo)
    external_reference = reference_median is not None
    if reference_median is None:
        reference_median = own_median

    above = tag_abundances > np.asarray(thresholds, dtype=float)
    toomany_idx = np.flatnonzero((ids == design.unresolved_code) & (above.sum(axis=1) > 3))

    if rescue and floor is None:
        logger.info('Too-many-tag rescue skipped: no positivity noise floor available.')
        rescue = False

    rescued = np.zeros(0, dtype=np.int64)
    diagnostics = None
    if rescue and toomany_idx.size:
        result = rescue_too_many(
            tag_abundances, ids, toomany_idx, reference_median, design, transform, floor,
            tag_shift_min_n=tag_shift_min_n, flat_threshold=flat_threshold,
            gate_frac=gate_frac, ref_idx=ref_idx,
        )
        rescued = result['rescued_idx']
        ids = ids.copy()
        ids[rescued] = result['assigned']
        diagnostics = result['diagnostics']
        logger.info(f'Too-many-tag rescue: {rescued.size}/{toomany_idx.size} candidates rescued.')

    rescued_toofew = np.zeros(0, dtype=np.int64)
    toofew_diagnostics = None
    if too_few is not None:
        result = rescue_too_few(
            tag_abundances, ids, thresholds, design, too_few, af_scale=af_scale, usable=usable,
            confident_idx=confident, reference_median=reference_median if external_reference else None,
            reference_background=reference_background if external_reference else None,
            tag_shift_min_n=tag_shift_min_n,
        )
        rescued_toofew = result['rescued_idx']
        if rescued_toofew.size:
            ids = ids.copy()
            ids[rescued_toofew] = result['assigned']
        toofew_diagnostics = result['diagnostics']
        logger.info(f'Too-few-tag rescue: {rescued_toofew.size}/{toofew_diagnostics["n_candidates"]} '
                    f'candidates rescued.')

    return DebarcodeResult(ids=ids, confident=confident, rescued=rescued,
                           combo_median=own_median, rescue=diagnostics,
                           rescued_toofew=rescued_toofew, rescue_toofew=toofew_diagnostics)


# ---------------------------------------------------------------------------
# Error models and correction
# ---------------------------------------------------------------------------

# On-tag position pairs (in ``design.on_tags`` order) of a combo's product rows.
PAIR_ROWS = ((0, 1), (0, 2), (1, 2))


def raw_error(raw: np.ndarray, tag_on: np.ndarray, tag_spectra: np.ndarray,
              raw_baseline: np.ndarray) -> np.ndarray:
    """Raw data minus the untransduced baseline spectrum and the fixed-basis
    reconstruction of the on-tags above it (*tag_on*, abundances minus the
    baseline's): each event's unmixing error plus noise and its own AF and
    background variation."""
    return raw - raw_baseline - tag_on @ tag_spectra


def baseline_tag_abundances(raw_baseline: np.ndarray, spectra: np.ndarray, af_mean: np.ndarray,
                            tag_cols: np.ndarray) -> np.ndarray:
    """Fixed-basis (spectra plus mean AF) tag abundances of the baseline
    spectrum: the origin of the error models in tag-abundance space."""
    basis = np.vstack([spectra, af_mean])
    return ols_unmix(np.asarray(raw_baseline, dtype=float)[None], basis)[0, np.asarray(tag_cols)]


def combo_features(a: np.ndarray) -> np.ndarray:
    """Regressors of the combo error model: the three on-tag abundances, then
    the products of each on-tag pair (``PAIR_ROWS``; negative abundances
    count as zero in the products)."""
    p = np.maximum(a, 0.0)
    return np.column_stack([a] + [p[:, i] * p[:, j] for i, j in PAIR_ROWS])


def _combo_scale(thresholds_on: np.ndarray) -> np.ndarray:
    t = np.asarray(thresholds_on, dtype=float)
    return np.concatenate([t, [t[i] * t[j] for i, j in PAIR_ROWS]])


def combo_error_fit(a: np.ndarray, y: np.ndarray, thresholds_on: np.ndarray, min_cells: int,
                    ridge: float = 0.0) -> np.ndarray | None:
    """
    Through-origin OLS of the raw error *y* (events x detectors) on the
    on-tag abundances *a* (events x 3, measured from the untransduced
    baseline) and their pairwise products.

    FRET grows with donor x acceptor, so the error is convex in abundance; a
    line through the origin with product terms follows it down to the
    untransduced baseline, where a centred linear fit would leave a constant
    offset.
    The fit is done in units of each tag's threshold for conditioning.
    ``ridge`` adds that fraction of the mean diagonal of X'X to its diagonal.

    Returns the error per unit of each regressor (6 x detectors: three
    on-tag rows, then the ``PAIR_ROWS`` product rows), in raw abundance
    units, or None if under-supported or singular.
    """
    if y.shape[1] == 0 or a.shape[0] < max(min_cells, 8):
        return None
    scale = _combo_scale(thresholds_on)
    x = combo_features(a) / scale
    xtx = x.T @ x
    if ridge > 0:
        xtx = xtx + np.eye(len(xtx)) * ridge * np.mean(np.diag(xtx))
    b = _solve(xtx, x.T @ y)
    if b is None or not np.all(np.isfinite(b)):
        return None
    return b / scale[:, None]


def combo_error_slope(error: np.ndarray, a_on) -> np.ndarray:
    """Error per unit of each on-tag (3 x detectors) at on-tag abundances
    *a_on*: the derivative of the combo error model there."""
    slope = np.array(error[:3], dtype=float, copy=True)
    p = np.maximum(np.asarray(a_on, dtype=float), 0.0)
    for r, (i, j) in enumerate(PAIR_ROWS):
        slope[i] += p[j] * error[3 + r]
        slope[j] += p[i] * error[3 + r]
    return slope


@dataclass(eq=False)
class PairModel:
    """
    Shared pairwise error model, fitted on the fit events of every combo.

    The error of an event whose active tags are A is
    ``sum_i a_i c_i + sum_{i<j in A} a_i a_j product[i, j]``, with ``c_i``
    twice the mean of ``linear[i, j]`` over the partners j in A that were
    seen with i (so a full triplet gets ``linear[i, j] + linear[i, k]``),
    or over every partner seen with i when none of them is active.

    linear  : tags x tags x detectors; error per unit of tag i with partner j.
    product : tags x tags x detectors, symmetric; error per unit of a_i a_j.
    support : tags x tags bool; tag pairs seen together in fit events.
    """
    linear: np.ndarray
    product: np.ndarray
    support: np.ndarray

    def _coefficients(self, active: np.ndarray) -> np.ndarray:
        n_tags, n_det = self.linear.shape[0], self.linear.shape[2]
        coef = np.zeros((n_tags, n_det))
        for i in np.flatnonzero(active):
            seen = self.support[i].copy()
            seen[i] = False
            partners = np.flatnonzero(active & seen)
            if partners.size == 0:
                partners = np.flatnonzero(seen)
            if partners.size:
                coef[i] = 2.0 * self.linear[i, partners].mean(axis=0)
        return coef

    def predict(self, tag_abundances: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Error of each event (events x detectors) from its tag abundances
        (events x tags) and active tags (events x tags, bool)."""
        active = np.asarray(active, dtype=bool)
        out = np.zeros((len(tag_abundances), self.linear.shape[2]))
        if not len(out):
            return out
        bits = 1 << np.arange(active.shape[1], dtype=np.int64)
        masks = active.astype(np.int64) @ bits
        for mask in np.unique(masks):
            rows = np.flatnonzero(masks == mask)
            on = active[rows[0]]
            if not on.any():
                continue
            a = tag_abundances[rows]
            out[rows] = (a * on) @ self._coefficients(on)
            p = np.maximum(a, 0.0)
            for i, j in itertools.combinations(np.flatnonzero(on), 2):
                out[rows] += np.outer(p[:, i] * p[:, j], self.product[i, j])
        return out


def pair_model_fit(tag_on: np.ndarray, y: np.ndarray, thresholds: np.ndarray,
                   min_cells: int) -> PairModel | None:
    """
    Fit the ``PairModel`` by through-origin least squares (minimum norm) on
    events whose on-tags (exactly three) are the nonzero columns of
    *tag_on* (events x tags); *y* is their raw error. Columns are scaled by
    the tag thresholds for conditioning. None if under-supported.
    """
    n, n_tags = tag_on.shape
    if n < max(min_cells, 8) or y.shape[1] == 0:
        return None
    thr = np.asarray(thresholds, dtype=float)
    on = tag_on != 0
    pairs = list(itertools.combinations(range(n_tags), 2))
    seen = np.array([np.count_nonzero(on[:, i] & on[:, j]) for i, j in pairs])
    support = np.zeros((n_tags, n_tags), dtype=bool)
    for (i, j), count in zip(pairs, seen):
        support[i, j] = support[j, i] = count > 0
    lin_cols = [(i, j) for i in range(n_tags) for j in range(n_tags) if i != j and support[i, j]]
    prod_cols = [p for p, count in zip(pairs, seen) if count > 0]
    if not lin_cols:
        return None
    p = np.maximum(tag_on, 0.0)
    x = np.column_stack(
        [np.where(on[:, i] & on[:, j], tag_on[:, i], 0.0) / thr[i] for i, j in lin_cols]
        + [p[:, i] * p[:, j] / (thr[i] * thr[j]) for i, j in prod_cols]
    )
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    if not np.all(np.isfinite(b)):
        return None
    linear = np.zeros((n_tags, n_tags, y.shape[1]))
    product = np.zeros((n_tags, n_tags, y.shape[1]))
    for k, (i, j) in enumerate(lin_cols):
        linear[i, j] = b[k] / thr[i]
    for k, (i, j) in enumerate(prod_cols, start=len(lin_cols)):
        product[i, j] = product[j, i] = b[k] / (thr[i] * thr[j])
    return PairModel(linear=linear, product=product, support=support)


def apply_error_correction(raw: np.ndarray, tag_abundances: np.ndarray, ids: np.ndarray,
                           design: FlowCodeDesign, combo_error: list, pair_model: PairModel | None,
                           thresholds: np.ndarray, correction: str = 'combo',
                           tag_baseline: np.ndarray | None = None) -> dict:
    """
    Subtract the characterised error from raw data.

    'combo'    : events with a valid barcode and a combo error get their
                 combo model; every other event gets the pairwise model,
                 through its combo's tags when it has a valid barcode, else
                 through its above-threshold tags.
    'pairwise' : the pairwise model for every event.
    'none'     : unchanged.

    The models are evaluated on *tag_abundances* minus *tag_baseline* (the
    untransduced baseline's tag abundances, ``baseline_tag_abundances``);
    above-threshold tags are found on *tag_abundances* itself.

    Returns ``raw`` (corrected copy), ``applied_combo`` (bool per combo) and
    ``pairwise_cells``: events given the pairwise model with at least one
    active tag.
    """
    if correction not in ('combo', 'pairwise', 'none'):
        raise ValueError(f'Unknown correction "{correction}".')
    applied = np.zeros(design.n_combos, dtype=bool)
    out = np.array(raw, dtype=np.float64, copy=True)
    if correction == 'none':
        return {'raw': out, 'applied_combo': applied, 'pairwise_cells': 0}
    shifted = tag_abundances if tag_baseline is None else tag_abundances - np.asarray(tag_baseline, dtype=float)

    combo_cell = np.zeros(len(raw), dtype=bool)
    if correction == 'combo':
        for g in range(design.n_combos):
            error = combo_error[g]
            if error is None:
                continue
            idx = np.flatnonzero(ids == g + 1)
            if idx.size == 0:
                continue
            on = design.on_tags(g)
            out[idx] -= combo_features(shifted[np.ix_(idx, on)]) @ error
            applied[g] = True
            combo_cell[idx] = True

    rest = np.flatnonzero(~combo_cell)
    pairwise_cells = 0
    if pair_model is not None and rest.size:
        active = tag_abundances[rest] > np.asarray(thresholds, dtype=float)
        valid = (ids[rest] >= 1) & (ids[rest] <= design.n_combos)
        active[valid] = design.logical[ids[rest][valid] - 1].astype(bool)
        out[rest] -= pair_model.predict(shifted[rest], active)
        pairwise_cells = int(np.count_nonzero(active.any(axis=1)))
    return {'raw': out, 'applied_combo': applied, 'pairwise_cells': pairwise_cells}


# ---------------------------------------------------------------------------
# Tandem tags: sample-specific breakdown
# ---------------------------------------------------------------------------
#
# A tag on a tandem fluorophore (PE-Cy7, say) that breaks down in a sample
# emits part of its signal as its parent (PE). When the parent is in the
# panel this lands on the parent's marker in every combo carrying the tag,
# in proportion to the tag's abundance, and no per-cell residual shows it.
# The barcodes identify it: within each combo (combo means removed), the
# parent's rise with expression is compared between combos that carry the
# tag and combos that carry no tandem tag with the same parent. The
# backbone's value of the same statistic is the reference, so only the
# sample's departure from the backbone is corrected.

TANDEM_QUANTILE = 0.25


@dataclass(frozen=True, eq=False)
class TandemTag:
    """
    A FlowCode tag on a tandem fluorophore.

    tag          : index of the tag in ``FlowCodeDesign.tags``.
    fluor        : the tag's fluorophore label.
    parent       : the parent fluorophore (fluorophore database name).
    parent_label : the parent's label in the panel, or None when the parent
                   is not in the panel (its spectrum then comes from the
                   cytometer's reference library).
    spectrum     : the parent's spectrum (detectors), scaled as the panel
                   spectra.
    """
    tag: int
    fluor: str
    parent: str
    parent_label: str | None
    spectrum: np.ndarray

    @property
    def source(self) -> str:
        return 'panel' if self.parent_label is not None else 'library'


def _fluorophore_db() -> list:
    from honeychrome.controller_components.label_matching import get_fluorophore_db
    return get_fluorophore_db()


def _reference_library(cytometer_key):
    from honeychrome.controller_components.spectral_reference_library import load_reference_library
    return load_reference_library(cytometer_key) if cytometer_key else None


def tandem_candidates(design: FlowCodeDesign, fluor_names, spectra: np.ndarray, detectors,
                      cytometer_key: str | None = None, fluor_db: list | None = None) -> list:
    """
    The tag fluorophores the fluorophore database calls tandems or gives a
    parent (``tandem.parent`` column), in tag order.

    Each entry: ``fluor``, ``tag`` (name), ``tag_index``, ``fluor_class``,
    ``parent``, ``tandem`` (a ``TandemTag``, or None) and ``problem`` (why
    there is no ``TandemTag``: no parent in the database, the parent is a
    FlowCode tag, or no parent spectrum for these detectors).
    """
    from honeychrome.controller_components.label_matching import match_fluorophore

    db = _fluorophore_db() if fluor_db is None else fluor_db
    by_name = {row.get('fluorophore', '').strip().upper(): row for row in db}
    fluor_names = list(fluor_names)
    spectra = np.asarray(spectra, dtype=float)
    canonical = {label: (match_fluorophore(label, db) or label).strip().upper() for label in fluor_names}
    tag_canonical = {(match_fluorophore(f, db) or f).strip().upper() for f in design.fluors}
    out = []
    for t, fluor in enumerate(design.fluors):
        name = (match_fluorophore(fluor, db) or fluor).strip().upper()
        row = by_name.get(name) or {}
        fluor_class = (row.get('class') or '').strip()
        parent = (row.get('tandem.parent') or '').strip()
        if 'tandem' not in fluor_class.lower() and not parent:
            continue
        entry = {'fluor': fluor, 'tag': design.tags[t], 'tag_index': t, 'fluor_class': fluor_class,
                 'parent': parent, 'tandem': None, 'problem': None}
        out.append(entry)
        if not parent:
            entry['problem'] = 'no tandem.parent in the fluorophore database'
            continue
        if parent.upper() in tag_canonical:
            entry['problem'] = f'the parent {parent} is a FlowCode tag'
            continue
        label = next((lbl for lbl in fluor_names if canonical[lbl] == parent.upper()), None)
        if label is not None:
            spectrum = spectra[fluor_names.index(label)]
        else:
            library = _reference_library(cytometer_key)
            if library is None or parent not in library.index \
                    or any(d not in library.columns for d in detectors):
                entry['problem'] = f'no {parent} spectrum for these detectors'
                continue
            spectrum = library.loc[parent, list(detectors)].to_numpy(dtype=float)
        entry['tandem'] = TandemTag(tag=t, fluor=fluor, parent=parent, parent_label=label,
                                    spectrum=np.array(spectrum, dtype=float))
    return out


def tandem_parent_values(raw: np.ndarray, spectra: np.ndarray, fluor_names, af_mean: np.ndarray,
                         tandem: TandemTag) -> np.ndarray:
    """Fixed-basis abundance of the tandem's parent per event; a parent not
    in the panel is added to the basis."""
    basis = np.vstack([spectra, af_mean])
    if tandem.parent_label is None:
        return ols_unmix(raw, np.vstack([basis, tandem.spectrum]))[:, -1]
    return ols_unmix(raw, basis)[:, list(fluor_names).index(tandem.parent_label)]


def tandem_exclusions(tandems, tandem: TandemTag) -> list:
    """Other tandem tags with the same parent: their combos are left out of
    both groups, as their own breakdown also adds the parent."""
    return [o.tag for o in tandems if o is not tandem and o.parent == tandem.parent]


def tandem_statistic(parent: np.ndarray, tag_shifted: np.ndarray, ids: np.ndarray, design: FlowCodeDesign,
                     tag: int, exclude=(), min_cells: int = 50) -> dict:
    """
    Parent abundance per unit of tag *tag* specific to the combos carrying
    it.

    parent      : the parent's abundance per event (``tandem_parent_values``).
    tag_shifted : tag abundances measured from the untransduced baseline
                  (events x tags).
    exclude     : tags whose combos are left out (``tandem_exclusions``).

    In each combo with *min_cells* events, an expression index E (mean of
    the on-tags other than *tag*, each as a fold of its combo median) splits
    the events into quartiles; the top-minus-bottom differences of the
    medians of the parent (dy), E (dE) and the tag (da) are robust to the
    tails and remove the combo's mean. Combos without the tag give the
    expression trend shared by all combos, gamma = median(dy / dE). Each
    combo with the tag gives beta_g = (dy - gamma dE) / da; ``beta`` is
    their median. Combos without the tag, scaled by the median da, give
    null values (``combo_null``) and their dispersion; ``se`` combines it
    (or the tag combos' own spread, when larger) with the error of gamma.
    ``combo_beta`` and ``combo_null`` are kept per combo name for the
    paired comparison in ``tandem_shift``.
    """
    q = TANDEM_QUANTILE
    skip = set(exclude)
    tag_rows, null_rows = [], []
    for g in range(design.n_combos):
        on = design.on_tags(g)
        if skip.intersection(on.tolist()):
            continue
        idx = np.flatnonzero(ids == g + 1)
        if idx.size < max(min_cells, 8):
            continue
        expr = [t for t in on if t != tag]
        a = tag_shifted[np.ix_(idx, expr)]
        med = np.median(a, axis=0)
        if np.any(med <= 0):
            continue
        e = (a / med).mean(axis=1)
        lo_cut, hi_cut = np.quantile(e, [q, 1.0 - q])
        lo, hi = e <= lo_cut, e >= hi_cut
        y = parent[idx]
        dy = np.median(y[hi]) - np.median(y[lo])
        de = np.median(e[hi]) - np.median(e[lo])
        if de <= 0:
            continue
        if tag in on:
            at = tag_shifted[idx, tag]
            da = np.median(at[hi]) - np.median(at[lo])
            if da > 0:
                tag_rows.append((g, dy, de, da, float(np.median(at))))
        else:
            null_rows.append((g, dy, de))

    out = {'beta': float('nan'), 'se': float('nan'), 'gamma': float('nan'), 'sigma_null': float('nan'),
           'sigma_tag': float('nan'), 'n_combos': len(tag_rows), 'n_null': len(null_rows),
           'median_tag': float('nan'), 'combo_beta': {}, 'combo_null': {}}
    if not tag_rows or len(null_rows) < 2:
        return out
    null = np.array([r[1:] for r in null_rows], dtype=float)
    rows = np.array([r[1:5] for r in tag_rows], dtype=float)
    gamma = float(np.median(null[:, 0] / null[:, 1]))
    da_ref = float(np.median(rows[:, 2]))
    pseudo = (null[:, 0] - gamma * null[:, 1]) / da_ref
    beta_g = (rows[:, 0] - gamma * rows[:, 1]) / rows[:, 2]
    beta = float(np.median(beta_g))
    sigma_null = 1.4826 * float(np.median(np.abs(pseudo)))
    sigma_tag = 1.4826 * float(np.median(np.abs(beta_g - beta))) if len(beta_g) > 1 else 0.0
    sigma = max(sigma_null, sigma_tag)
    se = float(np.sqrt(0.5 * np.pi * (sigma ** 2 / len(beta_g) + sigma_null ** 2 / len(pseudo))))
    out.update(beta=beta, se=se, gamma=gamma, sigma_null=sigma_null, sigma_tag=sigma_tag,
               median_tag=float(np.median(rows[:, 3])),
               combo_beta={design.combo_names[r[0]]: float(b) for r, b in zip(tag_rows, beta_g)},
               combo_null={design.combo_names[r[0]]: float(v) for r, v in zip(null_rows, pseudo)})
    return out


def _paired_difference(stat: dict, ref: dict, min_combos: int):
    """Median over combos of the sample's minus the backbone's per-combo
    values, and its standard error, or None with fewer than *min_combos*
    shared combos in either group. Pairing cancels each combo's own
    residual from the correction, which the sample shares with the
    backbone."""
    tag_ref, null_ref = ref.get('combo_beta') or {}, ref.get('combo_null') or {}
    d = np.array([v - tag_ref[c] for c, v in (stat.get('combo_beta') or {}).items() if c in tag_ref], dtype=float)
    dn = np.array([v - null_ref[c] for c, v in (stat.get('combo_null') or {}).items() if c in null_ref],
                  dtype=float)
    if d.size < min_combos or dn.size < min_combos:
        return None
    delta = float(np.median(d))
    sigma_null = 1.4826 * float(np.median(np.abs(dn - np.median(dn))))
    sigma = max(sigma_null, 1.4826 * float(np.median(np.abs(d - delta))))
    se = float(np.sqrt(0.5 * np.pi * (sigma ** 2 / d.size + sigma_null ** 2 / dn.size)))
    return delta, se, int(d.size)


def tandem_shift(stat: dict, reference: dict | None, z_min: float = 3.0, min_combos: int = 5) -> dict:
    """
    The breakdown correction of one tandem tag in a sample: its statistic
    minus the backbone's (*reference*; zero when None), soft-thresholded on
    z = difference / standard error: the weight is max(0, 1 - (z_min / z)^2),
    so nothing is applied below z_min and almost all of it well above.
    Needs *min_combos* combos with and without the tag.

    With per-combo values in the reference for at least *min_combos* of the
    sample's combos in each group, the difference is paired by combo
    (``_paired_difference``); otherwise it is the difference of the two
    medians, with their standard errors combined.

    Returns ``beta`` (parent abundance removed per unit of tag above the
    baseline), ``delta``, ``se``, ``z``, ``weight``, ``paired`` (combos
    paired, 0 when unpaired), ``reference_beta``, ``reference_se`` and
    ``reason`` (why nothing was applied, or '').
    """
    ref = reference or {}
    ref_beta = ref.get('beta')
    ref_beta = float(ref_beta) if ref_beta is not None and np.isfinite(ref_beta) else 0.0
    ref_se = ref.get('se')
    ref_se = float(ref_se) if ref_se is not None and np.isfinite(ref_se) else 0.0
    out = {'beta': 0.0, 'delta': float('nan'), 'se': float('nan'), 'z': float('nan'), 'weight': 0.0,
           'paired': 0, 'reference_beta': ref_beta, 'reference_se': ref_se, 'reason': ''}
    if stat['n_combos'] < min_combos or stat['n_null'] < min_combos or not np.isfinite(stat['beta']):
        out['reason'] = (f'{stat["n_combos"]} combination(s) with the tag and {stat["n_null"]} without '
                         f'(at least {min_combos} of each needed)')
        return out
    paired = _paired_difference(stat, ref, min_combos) if reference else None
    if paired is not None:
        delta, se, out['paired'] = paired
    else:
        delta = stat['beta'] - ref_beta
        se = float(np.hypot(stat['se'], ref_se))
    out['se'] = se
    z = delta / se if se > 0 else (float('inf') if delta else 0.0)
    weight = max(0.0, 1.0 - (z_min / abs(z)) ** 2) if z else 0.0
    out.update(beta=weight * delta, delta=delta, z=z, weight=weight)
    if weight == 0.0:
        out['reason'] = f'|z| below {z_min:g}'
    return out


def apply_tandem_correction(raw: np.ndarray, tag_abundances: np.ndarray, ids: np.ndarray,
                            design: FlowCodeDesign, thresholds: np.ndarray, tag_baseline: np.ndarray,
                            shifts) -> np.ndarray:
    """
    Remove each tandem's breakdown, ``beta`` x (tag abundance above the
    baseline) x the parent spectrum, from the raw data of the events
    carrying the tag: barcoded events of combos with the tag, and other
    events with the tag above threshold. *shifts*: (``TandemTag``, beta)
    pairs. Returns a corrected copy.
    """
    out = np.array(raw, dtype=np.float64, copy=True)
    valid = (ids >= 1) & (ids <= design.n_combos)
    for tandem, beta in shifts:
        if not beta:
            continue
        t = tandem.tag
        carries = np.zeros(len(raw), dtype=bool)
        carries[valid] = design.logical[ids[valid] - 1, t].astype(bool)
        carries[~valid] = tag_abundances[~valid, t] > thresholds[t]
        a = np.maximum(tag_abundances[carries, t] - tag_baseline[t], 0.0)
        out[carries] -= beta * np.outer(a, tandem.spectrum)
    return out


# ---------------------------------------------------------------------------
# Backbone characterisation
# ---------------------------------------------------------------------------

@dataclass
class CharacteriseSettings:
    """Backbone characterisation settings."""
    rescue_toomany: bool = True
    rescue_flat_threshold: float = 0.9
    rescue_gate_frac: float = 0.5
    rescue_toofew: bool = True
    rescue_toofew_flat_threshold: float = 0.9
    rescue_toofew_gate_frac: float = 0.5
    rescue_toofew_min_z: float = 2.0
    rescue_toofew_min_frac: float = 0.25
    rescue_toofew_af_ratio: float = 1.0
    rescue_toofew_n0: float = 50.0
    fit_min_fold: float = 3.0
    min_cells_per_combo: int = 50
    spill_ridge: float = 0.0


# Version of the error model stored with a characterisation.
ERROR_MODEL = 'bilinear-2'


@dataclass(eq=False)
class FlowCodeSpectra:
    """Everything a sample needs to be unmixed with FlowCode correction.

    ``combo_error[g]`` is the combo error model (6 x detectors, see
    ``combo_error_fit``), or None for combos that use ``pair_model``.
    ``raw_baseline`` is the raw spectrum the errors were measured against:
    the per-detector median of the untransduced events (their AF and
    non-specific staining); the models' tag abundances are measured from its
    ``baseline_tag_abundances``. None means a zero baseline.
    ``tag_background`` is the untransduced median per tag in debarcoding
    space, the background of ``combo_raw_median`` for the too-few rescue
    (None: each sample's own).
    ``fret_corrected`` False marks a debarcoding-only characterisation
    (no FRET backbone): it has no error model, and samples unmixed with it
    get no FRET, spillover or tandem correction, so their results are not
    valid.
    """
    design: FlowCodeDesign
    thresholds: np.ndarray
    spectra: np.ndarray
    fluor_names: tuple
    detectors: tuple
    af_mean: np.ndarray
    combo_raw_median: np.ndarray
    combo_error: list
    pair_model: PairModel | None
    rescue_floor: np.ndarray | None
    transform_params: dict
    raw_baseline: np.ndarray | None = None
    tag_background: np.ndarray | None = None
    settings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    fret_corrected: bool = True

    def tag_columns(self) -> np.ndarray:
        """Indices of the tag fluorophores in ``fluor_names``."""
        return np.array([self.fluor_names.index(f) for f in self.design.fluors])

    def tag_baseline(self, spectra: np.ndarray | None = None, fluor_names=None) -> np.ndarray:
        """Tag abundances of ``raw_baseline`` in the fixed basis of *spectra*
        (rows in *fluor_names* order; default the characterisation's)."""
        if self.raw_baseline is None:
            return np.zeros(self.design.n_tags)
        if spectra is None:
            spectra, fluor_names = self.spectra, self.fluor_names
        tag_cols = [list(fluor_names).index(f) for f in self.design.fluors]
        return baseline_tag_abundances(self.raw_baseline, spectra, self.af_mean, tag_cols)

    def fingerprint(self) -> str:
        return spectra_fingerprint(self.spectra[self.tag_columns()], self.detectors)

    def characterisation_id(self) -> str:
        """Short identifier of this characterisation: the tag spectra, the
        thresholds and the combination table. Written to each exported file
        so later analyses can tell which characterisation produced it."""
        h = hashlib.sha1(self.fingerprint().encode('utf-8'))
        h.update(np.round(np.asarray(self.thresholds, dtype=np.float64), 6).tobytes())
        for name, tags in zip(self.design.combo_names, self.design.combo_tags):
            h.update(f'{name}={combo_key(tags)};'.encode('utf-8'))
        if not self.fret_corrected:
            h.update(b'no FRET correction')
        return h.hexdigest()[:12]


@dataclass(eq=False)
class CharacterisationResult:
    spectra: FlowCodeSpectra
    ids: np.ndarray
    confident: np.ndarray
    rescued: np.ndarray
    fit_idx: np.ndarray
    fixed: np.ndarray
    corrected_raw: np.ndarray
    rescued_toofew: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def spectra_fingerprint(tag_spectra: np.ndarray, detectors) -> str:
    """Stable fingerprint of the tag spectra and detector list."""
    h = hashlib.sha1(np.round(np.asarray(tag_spectra, dtype=np.float64), 8).tobytes())
    h.update('|'.join(map(str, detectors)).encode('utf-8'))
    return h.hexdigest()


def check_spectra_match(fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names, detectors):
    """Raise ValueError if the tag spectra or detectors differ from those the
    backbone was characterised with (mean relative difference > 1.5e-8, as
    R's all.equal)."""
    fluor_names = list(fluor_names)
    missing = [f for f in fc.design.fluors if f not in fluor_names]
    if tuple(detectors) != tuple(fc.detectors) or missing:
        raise ValueError('The FlowCode spectra or detectors differ from those the backbone was '
                         'characterised with. Re-characterise the backbone with the current spectra.')
    now = spectra[[fluor_names.index(f) for f in fc.design.fluors]]
    then = fc.spectra[fc.tag_columns()]
    diff = np.mean(np.abs(now - then)) / max(np.mean(np.abs(then)), _EPS)
    if diff > 1.5e-8:
        raise ValueError('The FlowCode spectra or detectors differ from those the backbone was '
                         'characterised with. Re-characterise the backbone with the current spectra.')


def characterise_backbone(raw: np.ndarray, debarcode_unmixed: np.ndarray, design: FlowCodeDesign,
                          thresholds: np.ndarray, spectra: np.ndarray, fluor_names, detectors,
                          af_mean: np.ndarray, transform, transform_params: dict | None = None,
                          rescue_floor: np.ndarray | None = None,
                          unstained_raw: np.ndarray | None = None,
                          settings: CharacteriseSettings | None = None,
                          tandems=None, debarcode_af_scale: np.ndarray | None = None,
                          fit_error: bool = True) -> CharacterisationResult:
    """
    Measure the FlowCode unmixing error on the gated backbone.

    raw               : gated backbone raw data, events x detectors.
    debarcode_unmixed : the same events in debarcoding space, events x
                        fluorophores (``fluor_names`` order).
    debarcode_af_scale: their per-cell AF abundance, for the too-few
                        rescue's AF-dominated check (skipped when None).
    thresholds        : positivity thresholds per tag (debarcoding space).
    spectra           : fluorophores x detectors, no AF row.
    af_mean           : ``af_mean_spectrum`` of the gated matched unstained.
    rescue_floor      : per-tag noise floor for the rescue; None skips it.
    unstained_raw     : gated matched unstained raw data, for the AF/size
                        confound diagnostic (NaN without it).
    tandems           : ``TandemTag`` list; each gets its backbone
                        ``tandem_statistic`` after the correction (the
                        reference for samples) and before it (the backbone's
                        breakdown against the reference spectra), in
                        ``diagnostics['tandem']``.
    fit_error         : False debarcodes only, for samples acquired without
                        a FRET backbone: no error model, no tandem reference
                        and no correction (``fret_corrected`` False). The
                        combo medians and untransduced baseline that the
                        rescues use are kept.

    The error of an event is its raw data minus the untransduced baseline
    (the per-detector median raw spectrum of the untransduced events) and
    the fixed-basis reconstruction of its on-tags above the baseline's, and
    the models run through that baseline. Off-tag background staining and
    AF that every event carries are therefore not error: a corrected event
    keeps them and sits with the untransduced events, while an AF-shaped
    error stays in the error rather than being taken for the event's own
    AF. Each combo with ``min_cells_per_combo``
    fit events (every on-tag at least ``fit_min_fold`` x threshold) gets its
    own model (``combo_error_fit``); the shared ``PairModel``, fitted on the
    fit events of every combo, serves the other combos and events without a
    valid barcode.

    Raises ValueError if no combo has enough fit events (with *fit_error*).
    """
    settings = settings or CharacteriseSettings()
    if not fit_error:
        tandems = None
    fluor_names = tuple(fluor_names)
    thresholds = np.asarray(thresholds, dtype=float)
    tag_cols = np.array([fluor_names.index(f) for f in design.fluors])
    n_combo = design.n_combos
    min_cells = settings.min_cells_per_combo

    dc = debarcode_rescue(
        debarcode_unmixed[:, tag_cols], thresholds, design, transform,
        reference_median=None, floor=rescue_floor if settings.rescue_toomany else None,
        rescue=settings.rescue_toomany, flat_threshold=settings.rescue_flat_threshold,
        gate_frac=settings.rescue_gate_frac, too_few=TooFewSettings.from_settings(vars(settings)),
        af_scale=debarcode_af_scale,
    )
    ids, confident = dc.ids, dc.confident

    basis = np.vstack([spectra, af_mean])
    fixed = ols_unmix(raw, basis)
    af_col = spectra.shape[0]
    coef_map = _solve(basis @ basis.T, basis)
    if coef_map is None:
        raise ValueError('The fluorophore spectra plus the mean AF spectrum are collinear.')
    fixed_tags = fixed[:, tag_cols]

    fold = fixed_tags[confident] / thresholds
    fold[design.logical[ids[confident] - 1] == 0] = np.inf
    min_fold = fold.min(axis=1) if fold.size else np.zeros(0)
    fit_idx = confident[min_fold >= settings.fit_min_fold]
    logger.info(f'{confident.size} confident events, {fit_idx.size} with every on-tag '
                f'>= {settings.fit_min_fold}x threshold used for fitting')

    n_conf = np.bincount(ids[confident], minlength=n_combo + 1)[1:n_combo + 1]
    n_fit = np.bincount(ids[fit_idx], minlength=n_combo + 1)[1:n_combo + 1]
    table = {key: np.full(n_combo, np.nan) for key in
             ('kappa', 'in_span', 'out_span', 'curvature', 'af_slope', 'af_excess', 'af_confound')}
    fit_by = [fit_idx[ids[fit_idx] == g + 1] for g in range(n_combo)]

    untransduced = np.flatnonzero(ids == UNTRANSDUCED)
    base_pool = untransduced if untransduced.size >= min_cells else np.arange(len(raw))
    if base_pool.size:
        raw_baseline = np.median(raw[base_pool], axis=0)
        reference = np.median(fixed[base_pool], axis=0)
        tag_background = np.median(debarcode_unmixed[np.ix_(base_pool, tag_cols)], axis=0)
    else:
        raw_baseline = np.zeros(raw.shape[1])
        reference = np.zeros(fixed.shape[1])
        tag_background = np.zeros(design.n_tags)
    baseline_fixed = ols_unmix(raw_baseline[None], basis)[0]
    tag_baseline = baseline_fixed[tag_cols]
    shifted_tags = fixed_tags - tag_baseline
    for g in range(n_combo):
        idx = fit_by[g]
        if idx.size < max(min_cells, 3):
            continue
        s = fixed_tags[np.ix_(idx, design.on_tags(g))].sum(axis=1)
        if np.var(s, ddof=1) > 0:
            table['af_slope'][g] = np.cov(fixed[idx, af_col], s)[0, 1] / np.var(s, ddof=1)
    sloped = np.flatnonzero(np.isfinite(table['af_slope']))
    af_slope_reference = (weighted_median(table['af_slope'][sloped], n_fit[sloped])
                          if sloped.size else 0.0)
    table['af_excess'] = table['af_slope'] - af_slope_reference

    def error_of(idx, tag_on):
        return raw_error(raw[idx], tag_on, spectra[tag_cols], raw_baseline)

    combo_error = [None] * n_combo
    for g in range(n_combo):
        idx = fit_by[g]
        if idx.size < min_cells or not fit_error:
            continue
        on = design.on_tags(g)
        tag_on = np.zeros((idx.size, design.n_tags))
        tag_on[:, on] = shifted_tags[np.ix_(idx, on)]
        combo_error[g] = combo_error_fit(tag_on[:, on], error_of(idx, tag_on), thresholds[on], min_cells,
                                         ridge=settings.spill_ridge)

    fitted = np.flatnonzero([e is not None for e in combo_error])
    pair_model = None
    if fit_error:
        if fitted.size == 0:
            raise ValueError(f'No combo had {min_cells} fit events. Check the thresholds, or lower the fit '
                             f'fold or minimum events per combo.')
        logger.info(f'{fitted.size}/{n_combo} combos characterised; the rest use the pairwise model')
        tag_on_all = shifted_tags[fit_idx] * design.logical[ids[fit_idx] - 1]
        pair_model = pair_model_fit(tag_on_all, error_of(fit_idx, tag_on_all), thresholds, min_cells)
    else:
        logger.info('Debarcoding only: no FRET error model fitted')

    af_unit = af_mean / max(np.linalg.norm(af_mean), _EPS)
    shape_slope = None
    if unstained_raw is not None and len(unstained_raw) > 1:
        unst_af = ols_unmix(unstained_raw, basis)[:, af_col]
        afc = unst_af - unst_af.mean()
        shape_slope = ((unstained_raw - unstained_raw.mean(axis=0)) * afc[:, None]).sum(axis=0) \
            / np.sum(afc ** 2) - af_mean
    for g in fitted:
        on = design.on_tags(g)
        idx = fit_by[g]
        a_med = np.nan_to_num(dc.combo_median[g, on]) - tag_baseline[on]
        slope = combo_error_slope(combo_error[g], a_med)
        on_spectra = spectra[tag_cols[on]]
        on_norm = np.linalg.norm(on_spectra)
        slope_in = (slope @ coef_map.T) @ basis
        table['in_span'][g] = np.linalg.norm(slope_in) / on_norm
        table['out_span'][g] = np.linalg.norm(slope - slope_in) / on_norm
        a_fit = shifted_tags[np.ix_(idx, on)]
        x = combo_features(a_fit) / _combo_scale(thresholds[on])
        table['kappa'][g] = np.linalg.cond(x.T @ x)
        # how far the fitted error departs from the best line through the
        # origin, outside the AF direction (each event's AF variation is noise)
        tag_on = np.zeros((idx.size, design.n_tags))
        tag_on[:, on] = a_fit
        full = combo_features(a_fit) @ combo_error[g]
        linear, *_ = np.linalg.lstsq(x[:, :3], error_of(idx, tag_on), rcond=None)
        bend = full - x[:, :3] @ linear
        full = full - np.outer(full @ af_unit, af_unit)
        bend = bend - np.outer(bend @ af_unit, af_unit)
        table['curvature'][g] = np.linalg.norm(bend) / max(np.linalg.norm(full), _EPS)
        if shape_slope is not None:
            q, _r = np.linalg.qr(np.vstack([on_spectra, af_mean]).T)
            leak = shape_slope - q @ (q.T @ shape_slope)
            table['af_confound'][g] = abs(table['af_slope'][g]) * np.linalg.norm(leak) \
                / max(np.linalg.norm(slope.mean(axis=0)), _EPS)

    corrected = apply_error_correction(raw, fixed_tags, ids, design, combo_error, pair_model, thresholds,
                                       'combo' if fit_error else 'none', tag_baseline=tag_baseline)
    fixed_corrected = ols_unmix(corrected['raw'], basis) if fit_error else fixed

    tandem_reference = {}
    for tandem in tandems or ():
        exclude = tandem_exclusions(tandems, tandem)
        stat = tandem_statistic(tandem_parent_values(corrected['raw'], spectra, fluor_names, af_mean, tandem),
                                shifted_tags, ids, design, tandem.tag, exclude, min_cells)
        before = tandem_statistic(tandem_parent_values(raw, spectra, fluor_names, af_mean, tandem),
                                  shifted_tags, ids, design, tandem.tag, exclude, min_cells)
        tandem_reference[tandem.fluor] = {
            'tag': design.tags[tandem.tag], 'parent': tandem.parent, 'source': tandem.source,
            **stat, 'beta_uncorrected': before['beta'], 'se_uncorrected': before['se'],
        }

    def off_rms(m, idx, off):
        # off-tag medians relative to the untransduced
        return float(np.sqrt(np.mean((np.median(m[np.ix_(idx, off)], axis=0) - reference[off]) ** 2)))

    def off_spill(m, idx, on, off):
        # largest robust slope of any off channel on the on-tags
        fit = residual_spillover(m[np.ix_(idx, on)], m[np.ix_(idx, off)], min_events=min_cells)
        if fit is None or not fit['converged'].any():
            return float('nan')
        return float(np.abs(fit['slope'][:, fit['converged']]).max())

    validation = []
    for set_name, set_idx in ((('fit', fit_idx), ('held.out', dc.rescued)) if fit_error else ()):
        for g in range(n_combo):
            idx = set_idx[ids[set_idx] == g + 1]
            if idx.size == 0:
                continue
            on_fluors = {design.fluors[t] for t in design.on_tags(g)}
            off = [i for i, f in enumerate(fluor_names) if f not in on_fluors]
            on = tag_cols[design.on_tags(g)]
            validation.append({
                'Id': design.combo_names[g], 'set': set_name, 'n': int(idx.size),
                'off_rms_before': off_rms(fixed, idx, off),
                'off_rms_after': off_rms(fixed_corrected, idx, off),
                'off_spill_before': off_spill(fixed, idx, on, off),
                'off_spill_after': off_spill(fixed_corrected, idx, on, off),
            })

    fit_table = {'Id': list(design.combo_names), 'n_confident': n_conf, 'n_fit': n_fit, **table}
    fc = FlowCodeSpectra(
        design=design,
        thresholds=thresholds.copy(),
        spectra=np.array(spectra, dtype=np.float64, copy=True),
        fluor_names=fluor_names,
        detectors=tuple(detectors),
        af_mean=np.array(af_mean, dtype=np.float64, copy=True),
        combo_raw_median=dc.combo_median,
        combo_error=combo_error,
        pair_model=pair_model,
        rescue_floor=None if rescue_floor is None else np.asarray(rescue_floor, dtype=float).copy(),
        transform_params=dict(transform_params or {}),
        raw_baseline=raw_baseline,
        tag_background=tag_background,
        settings=dict(vars(settings)),
        diagnostics={'fit': fit_table, 'af_slope_reference': af_slope_reference,
                     'af_baseline': float(baseline_fixed[af_col]), 'tag_baseline': tag_baseline,
                     'baseline_events': int(base_pool.size), 'validation': validation, 'rescue': dc.rescue,
                     'rescue_toofew': dc.rescue_toofew, 'tandem': tandem_reference},
        fret_corrected=bool(fit_error),
    )
    return CharacterisationResult(spectra=fc, ids=ids, confident=confident, rescued=dc.rescued,
                                  fit_idx=fit_idx, fixed=fixed, corrected_raw=corrected['raw'],
                                  rescued_toofew=dc.rescued_toofew)


# ---------------------------------------------------------------------------
# Sample unmixing
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class StagedResult:
    unmixed: np.ndarray
    af_scale: np.ndarray
    af_idx: np.ndarray
    flowcode: np.ndarray
    combo_channels: np.ndarray
    ids: np.ndarray
    diagnostics: dict


def unsaturated_mask(raw: np.ndarray, fc: FlowCodeSpectra, frac: float = 0.999) -> np.ndarray:
    """Events with no detector at or above ``frac`` x the magnitude ceiling."""
    ceiling = float(fc.transform_params.get('magnitude_ceiling', 4194304.0))
    return ~np.any(raw >= frac * ceiling, axis=1)


def _debarcode_and_correct(raw: np.ndarray, fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names,
                           af_only_fn, transform, ref_idx, thr: np.ndarray, rescue: bool, flat_threshold: float,
                           gate_frac: float, tag_shift_min_n: int, correction: str, regressor: str,
                           too_few: TooFewSettings | None = None) -> dict:
    """Debarcode *raw* with rescue and subtract the characterised error:
    the ``dc`` result, ``tag_abundances`` (the correction's regressors),
    ``tag_baseline`` and the ``corrected`` record of
    ``apply_error_correction``. Events with a saturated detector are not
    too-few rescue candidates."""
    if regressor not in ('fixed', 'debarcode'):
        raise ValueError(f'Unknown regressor "{regressor}".')
    design = fc.design
    tag_cols = np.array([fluor_names.index(f) for f in design.fluors])
    debarcoded = af_only_fn(raw)
    dc = debarcode_rescue(
        debarcoded['unmixed'][:, tag_cols], thr, design, transform,
        reference_median=fc.combo_raw_median, floor=fc.rescue_floor if rescue else None,
        rescue=rescue, flat_threshold=flat_threshold, gate_frac=gate_frac,
        tag_shift_min_n=tag_shift_min_n, ref_idx=ref_idx, too_few=too_few,
        af_scale=debarcoded.get('af_scale'), usable=unsaturated_mask(raw, fc),
        reference_background=fc.tag_background,
    )
    if regressor == 'fixed':
        tag_abundances = ols_unmix(raw, np.vstack([spectra, fc.af_mean]))[:, tag_cols]
    else:
        tag_abundances = debarcoded['unmixed'][:, tag_cols]
    tag_baseline = fc.tag_baseline(spectra, fluor_names)
    corrected = apply_error_correction(raw, tag_abundances, dc.ids, design, fc.combo_error,
                                       fc.pair_model, thr, correction, tag_baseline=tag_baseline)
    return {'dc': dc, 'tag_abundances': tag_abundances, 'tag_baseline': tag_baseline, 'corrected': corrected}


def estimate_tandem_shifts(raw: np.ndarray, fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names,
                           af_only_fn, transform, tandems, pool=None, thresholds: np.ndarray | None = None,
                           rescue: bool = True, flat_threshold: float = 0.9, gate_frac: float = 0.5,
                           tag_shift_min_n: int = 100, correction: str = 'combo', regressor: str = 'fixed',
                           z_min: float = 3.0, min_combos: int = 5, too_few: TooFewSettings | None = None) -> list:
    """
    The breakdown correction of each tandem tag in one sample, from its
    gated events *raw*: debarcode and correct them as ``unmix_staged`` does,
    then ``tandem_statistic`` on the corrected data and ``tandem_shift``
    against the backbone's statistic (``fc.diagnostics['tandem']``).
    Without a backbone statistic for a tag (a characterisation made before
    its parent was known) nothing is applied, and a characterisation without
    FRET correction estimates nothing.

    tandems : ``TandemTag`` list to estimate.
    pool    : every ``TandemTag`` the backbone reference was computed with
              (default *tandems*); same-parent tags among them are excluded
              as in the reference.

    Returns one record per tandem: ``tandem``, ``fluor``, ``tag``,
    ``parent``, ``source``, ``has_reference``, the sample's ``sample_beta``
    and ``sample_se``, the statistic's counts and dispersions, the
    ``tandem_shift`` fields (``beta`` is applied) and ``at_median`` (the
    parent abundance removed at the median tag abundance).
    """
    if not tandems or not fc.fret_corrected:
        return []
    fluor_names = list(fluor_names)
    design = fc.design
    pool = list(pool) if pool is not None else list(tandems)
    thr = fc.thresholds if thresholds is None else np.asarray(thresholds, dtype=float)
    stage = _debarcode_and_correct(raw, fc, spectra, fluor_names, af_only_fn, transform, None, thr, rescue,
                                   flat_threshold, gate_frac, tag_shift_min_n, correction, regressor, too_few)
    corrected = stage['corrected']['raw']
    shifted = stage['tag_abundances'] - stage['tag_baseline']
    ids = stage['dc'].ids
    reference = (fc.diagnostics or {}).get('tandem') or {}
    min_cells = int(fc.settings.get('min_cells_per_combo', 50))
    out = []
    for tandem in tandems:
        parent = tandem_parent_values(corrected, spectra, fluor_names, fc.af_mean, tandem)
        stat = tandem_statistic(parent, shifted, ids, design, tandem.tag, tandem_exclusions(pool, tandem),
                                min_cells)
        ref = reference.get(tandem.fluor)
        if ref is not None and ref.get('parent') != tandem.parent:
            ref = None
        shift = tandem_shift(stat, ref, z_min, min_combos)
        if ref is None:
            shift.update(beta=0.0, weight=0.0, reason='no backbone statistic: run section 4 again')
        at_median = shift['beta'] * stat['median_tag'] if np.isfinite(stat['median_tag']) else 0.0
        out.append({
            'tandem': tandem, 'fluor': tandem.fluor, 'tag': design.tags[tandem.tag], 'parent': tandem.parent,
            'source': tandem.source, 'has_reference': ref is not None, 'events': int(len(raw)),
            'sample_beta': stat['beta'], 'sample_se': stat['se'],
            **{k: stat[k] for k in ('n_combos', 'n_null', 'gamma', 'sigma_null', 'sigma_tag', 'median_tag')},
            **shift, 'at_median': at_median,
        })
    return out


def staged_corrected_raw(raw: np.ndarray, fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names,
                         af_only_fn, transform, ref_mask: np.ndarray | None = None,
                         thresholds: np.ndarray | None = None, rescue: bool = True, flat_threshold: float = 0.9,
                         gate_frac: float = 0.5, tag_shift_min_n: int = 100, correction: str = 'combo',
                         regressor: str = 'fixed', tandem_shifts=(),
                         too_few: TooFewSettings | None = None) -> tuple[dict, np.ndarray]:
    """
    The FlowCode-corrected raw fluorescence of one block of events, the
    input of ``unmix_staged``'s final unmix: debarcode with rescue, subtract
    the characterised FRET/error model, then the tandem breakdown. Returns
    the ``_debarcode_and_correct`` record and the corrected raw. Arguments
    as ``unmix_staged``. A characterisation without FRET correction
    (``fc.fret_corrected`` False) leaves the raw data unchanged whatever
    *correction* and *tandem_shifts* say.
    """
    if not fc.fret_corrected:
        correction, tandem_shifts = 'none', ()
    thr = fc.thresholds if thresholds is None else np.asarray(thresholds, dtype=float)
    ref_idx = None if ref_mask is None else np.flatnonzero(ref_mask)
    stage = _debarcode_and_correct(raw, fc, spectra, list(fluor_names), af_only_fn, transform, ref_idx, thr,
                                   rescue, flat_threshold, gate_frac, tag_shift_min_n, correction, regressor,
                                   too_few)
    stage['ref_idx'] = ref_idx
    stage['correction'] = correction
    corrected_raw = stage['corrected']['raw']
    tandem_shifts = [(tandem, float(beta)) for tandem, beta in tandem_shifts]
    if any(beta for _t, beta in tandem_shifts):
        corrected_raw = apply_tandem_correction(corrected_raw, stage['tag_abundances'], stage['dc'].ids,
                                                fc.design, thr, stage['tag_baseline'], tandem_shifts)
    stage['tandem_shifts'] = tandem_shifts
    return stage, corrected_raw


def unmix_staged(raw: np.ndarray, fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names,
                 af_only_fn, final_fn, transform, ref_mask: np.ndarray | None = None, thresholds: np.ndarray | None = None,
                 rescue: bool = True, flat_threshold: float = 0.9, gate_frac: float = 0.5,
                 tag_shift_min_n: int = 100, correction: str = 'combo',
                 regressor: str = 'fixed', tandem_shifts=(), too_few: TooFewSettings | None = None) -> StagedResult:
    """
    FlowCode unmixing of one block of events (unmix.flowcode.staged()).

    raw           : events x detectors, ``fc.detectors`` order.
    spectra       : current reference spectra (fluorophores x detectors, no
                    AF), rows in ``fluor_names`` order; the tag rows must
                    match the backbone's (``check_spectra_match``).
    af_only_fn    : raw -> dict(unmixed, af_scale, af_idx); per-cell AF, no
                    variants, with this sample's AF library (debarcoding
                    space).
    final_fn      : corrected raw -> dict(unmixed, af_scale, af_idx); the
                    final unmix, with variants when used.
    ref_mask      : events inside the FlowCode scatter gate; the rescue's
                    per-tag shift is estimated on these (all events when
                    None).
    thresholds    : overrides ``fc.thresholds``.
    tandem_shifts : (``TandemTag``, beta) pairs of this sample
                    (``estimate_tandem_shifts``), removed after the error
                    correction (``apply_tandem_correction``).
    too_few       : too-few-tag rescue settings (off when None); its
                    references come from this block's own events.

    Returns the final unmix, the FlowCode channel (median final on-tag
    abundance for barcoded events, else 0), one channel per combo (the same
    value for that combo's events, else 0), combo codes and diagnostics.
    """
    design = fc.design
    fluor_names = list(fluor_names)
    tag_cols = np.array([fluor_names.index(f) for f in design.fluors])

    stage, corrected_raw = staged_corrected_raw(
        raw, fc, spectra, fluor_names, af_only_fn, transform, ref_mask=ref_mask, thresholds=thresholds,
        rescue=rescue, flat_threshold=flat_threshold, gate_frac=gate_frac, tag_shift_min_n=tag_shift_min_n,
        correction=correction, regressor=regressor, tandem_shifts=tandem_shifts, too_few=too_few)
    dc, corrected = stage['dc'], stage['corrected']
    ids = dc.ids
    ref_idx = stage['ref_idx']
    tandem_shifts = stage['tandem_shifts']
    final = final_fn(corrected_raw)
    unmixed = final['unmixed']

    n = len(raw)
    flowcode = np.zeros(n)
    combo_channels = np.zeros((n, design.n_combos))
    for g in range(design.n_combos):
        idx = np.flatnonzero(ids == g + 1)
        if idx.size == 0:
            continue
        med = np.median(unmixed[np.ix_(idx, tag_cols[design.on_tags(g)])], axis=1)
        flowcode[idx] = med
        combo_channels[idx, g] = med

    return StagedResult(
        unmixed=unmixed, af_scale=final['af_scale'], af_idx=final['af_idx'],
        flowcode=flowcode, combo_channels=combo_channels, ids=ids,
        diagnostics={
            'rescue': dc.rescue,
            'rescue_toofew': dc.rescue_toofew,
            'applied_combo': corrected['applied_combo'],
            'pairwise_cells': corrected['pairwise_cells'],
            'gate_events': None if ref_idx is None else int(ref_idx.size),
            'correction': stage['correction'],
            'regressor': regressor,
            'tandem': {tandem.fluor: beta for tandem, beta in tandem_shifts},
        },
    )


def debarcode_summary(ids: np.ndarray, design: FlowCodeDesign, rescued: int = 0,
                      pairwise_cells: int = 0, rescued_toofew: int = 0) -> dict:
    """Event counts per outcome and per combo (``rescued``: too-many-tag
    rescues, ``rescued_toofew``: too-few-tag rescues)."""
    counts = np.bincount(ids, minlength=design.n_combos + 2)
    return {
        'total_events': int(len(ids)),
        'untransduced': int(counts[UNTRANSDUCED]),
        'unresolved': int(counts[design.unresolved_code]),
        'rescued': int(rescued),
        'rescued_toofew': int(rescued_toofew),
        'pairwise_corrected': int(pairwise_cells),
        'per_combo': {name: int(c) for name, c in zip(design.combo_names, counts[1:design.n_combos + 1])},
    }


def gate_records_mask(raw: np.ndarray, pnn, records) -> np.ndarray | None:
    """Events inside every gate record (raw-unit polygons on two named raw
    channels); None when a gated channel is not among *pnn*."""
    pnn = list(pnn)
    mask = np.ones(len(raw), dtype=bool)
    for rec in records:
        if rec['x'] not in pnn or rec['y'] not in pnn:
            return None
        mask &= points_in_polygon(raw[:, pnn.index(rec['x'])], raw[:, pnn.index(rec['y'])], rec['vertices_raw'])
    return mask


def staged_export_fn(fc: FlowCodeSpectra, spectra: np.ndarray, fluor_names, final_fn, gate_records,
                     pnn_raw, settings: dict, cell_weight: bool = False, on_block=None, tandems=(),
                     tandem_pool=None, on_tandem=None, debarcode_fn=None):
    """
    Per-chunk unmixing function for the FlowCode export
    (``flowcode_export``): ``unmix_fn(raw_fl_chunk, raw_chunk, af,
    sample_key=None)`` runs ``unmix_staged`` with the sample's AF library
    and returns ``unmixed``, ``af_scale``, ``af_idx`` and ``extra`` (the
    FlowCode channel, the FlowCode ID channel, then one channel per combo).
    ``unmix_fn.prepare(raw_fl, raw, af, sample_key=None)``, the exporter's
    per-sample step, estimates the sample's tandem breakdown from its gated
    events (``estimate_tandem_shifts``); the chunks that follow remove it.
    ``unmix_fn.corrected_raw(raw_fl, raw, af, sample_key=None, pnn=None)``
    returns what ``unmix_fn`` hands its final unmix: the FlowCode-corrected
    raw fluorescence (``staged_corrected_raw``), for consumers that unmix it
    themselves (Fix My Unmix). ``pnn`` names the columns of ``raw`` when
    they are not *pnn_raw* (``prepare`` takes it too).

    final_fn     : (corrected raw fluorescence, af, sample key) ->
                   dict(unmixed, af_scale, af_idx); the final unmix.
    debarcode_fn : (raw fluorescence, af, sample key) -> the same dict; the
                   debarcoding space. Default: per-cell AF on *spectra*
                   (``af_only_unmix``).
    gate_records : FlowCode scatter gates in raw units; events inside all of
                   them set the rescue's per-tag shift and the tandem
                   estimate.
    pnn_raw      : column names of ``raw_chunk``.
    settings     : ``tag_shift_min_n``, ``correction``, ``regressor``,
                   ``tandem_z_min``, ``tandem_min_combos``,
                   ``tandem_max_events``. The rescue settings come from
                   the characterisation (``fc.settings``).
    on_block     : called with each chunk's ``staged_summary``.
    tandems      : ``TandemTag`` list to correct; ``tandem_pool`` as in
                   ``estimate_tandem_shifts``.
    on_tandem    : called with each sample's tandem records (without the
                   ``TandemTag``).
    """
    fluor_names = list(fluor_names)
    transform = logicle_transform(float(fc.transform_params.get('magnitude_ceiling', 4194304.0)),
                                  w=float(fc.transform_params.get('w', 0.5)),
                                  m=float(fc.transform_params.get('m', 4.5)),
                                  a=float(fc.transform_params.get('a', 0.0)))
    records = list(gate_records or [])
    tandems = list(tandems or []) if fc.fret_corrected else []
    staged = dict(
        rescue=bool(fc.settings.get('rescue_toomany', True)),
        flat_threshold=float(fc.settings.get('rescue_flat_threshold', 0.9)),
        gate_frac=float(fc.settings.get('rescue_gate_frac', 0.5)),
        tag_shift_min_n=int(settings.get('tag_shift_min_n', 100)),
        correction=str(settings.get('correction', 'combo')),
        regressor=str(settings.get('regressor', 'fixed')),
        too_few=TooFewSettings.from_settings(fc.settings),
    )
    sample = {'shifts': []}

    def af_only_fn(af, sample_key=None):
        if debarcode_fn is not None:
            return lambda raw: debarcode_fn(raw, af, sample_key)
        return lambda raw: af_only_unmix(raw, spectra, fluor_names, af.spectra, cell_weight=cell_weight,
                                         envelope=getattr(af, 'envelope', None))

    def prepare(raw_fl, raw, af, sample_key=None, pnn=None):
        sample['shifts'] = []
        if not tandems:
            return
        raw_fl = np.ascontiguousarray(raw_fl, dtype=np.float64)
        mask = gate_records_mask(raw, pnn_raw if pnn is None else pnn, records) if records else None
        idx = np.flatnonzero(mask) if mask is not None else np.arange(len(raw_fl))
        max_events = int(settings.get('tandem_max_events', 2_000_000))
        if idx.size > max_events:
            idx = np.sort(np.random.default_rng(42).choice(idx, max_events, replace=False))
        estimates = estimate_tandem_shifts(
            raw_fl[idx], fc, spectra, fluor_names, af_only_fn(af, sample_key), transform, tandems, pool=tandem_pool,
            z_min=float(settings.get('tandem_z_min', 3.0)),
            min_combos=int(settings.get('tandem_min_combos', 5)), **staged,
        )
        sample['shifts'] = [(e['tandem'], e['beta']) for e in estimates]
        if on_tandem is not None:
            on_tandem([{k: v for k, v in e.items() if k != 'tandem'} for e in estimates])

    def unmix_fn(raw_fl_chunk, raw_chunk, af, sample_key=None):
        raw_fl = np.ascontiguousarray(raw_fl_chunk, dtype=np.float64)
        ref_mask = gate_records_mask(raw_chunk, pnn_raw, records) if records else None
        result = unmix_staged(
            raw_fl, fc, spectra, fluor_names,
            af_only_fn=af_only_fn(af, sample_key),
            final_fn=lambda raw: final_fn(raw, af, sample_key),
            transform=transform, ref_mask=ref_mask, tandem_shifts=sample['shifts'], **staged,
        )
        if on_block is not None:
            on_block(staged_summary(result, fc.design))
        return {'unmixed': result.unmixed, 'af_scale': result.af_scale, 'af_idx': result.af_idx,
                'extra': np.column_stack([result.flowcode, result.ids.astype(np.float64),
                                          result.combo_channels])}

    def corrected_raw(raw_fl, raw, af, sample_key=None, pnn=None):
        raw_fl = np.ascontiguousarray(raw_fl, dtype=np.float64)
        ref_mask = gate_records_mask(raw, pnn_raw if pnn is None else pnn, records) if records else None
        _stage, out = staged_corrected_raw(raw_fl, fc, spectra, fluor_names, af_only_fn(af, sample_key), transform,
                                           ref_mask=ref_mask, tandem_shifts=sample['shifts'], **staged)
        return out

    unmix_fn.prepare = prepare
    unmix_fn.corrected_raw = corrected_raw
    return unmix_fn


def staged_summary(result: StagedResult, design: FlowCodeDesign) -> dict:
    """``debarcode_summary`` of one ``unmix_staged`` block."""
    rescue = result.diagnostics.get('rescue') or {}
    toofew = result.diagnostics.get('rescue_toofew') or {}
    return debarcode_summary(result.ids, design, rescued=int(rescue.get('n_rescued', 0)),
                             pairwise_cells=int(result.diagnostics.get('pairwise_cells') or 0),
                             rescued_toofew=int(toofew.get('n_rescued', 0)))


def add_summaries(total: dict | None, block: dict) -> dict:
    """Sum two ``debarcode_summary`` records (blocks of one sample)."""
    if total is None:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in block.items()}
    out = {k: total[k] + block[k] for k in total if k != 'per_combo'}
    out['per_combo'] = {name: total['per_combo'].get(name, 0) + n for name, n in block['per_combo'].items()}
    return out


SUMMARY_COLUMNS = ('Sample', 'Total events', 'Untransduced', 'Unresolved', 'Rescued (too many)',
                   'Rescued (too few)', 'Pairwise corrected')


def write_debarcode_summary_csv(path, samples: list, design: FlowCodeDesign, warning: str | None = None):
    """One row per sample, (name, ``debarcode_summary`` record) pairs: event
    totals, then one count column per combo, then, for each tandem tag in
    any record's ``tandem`` list, the parent abundance removed per unit of
    tag, its z and the amount removed at the median tag abundance. A
    *warning* (samples unmixed without FRET correction) is a last column,
    repeated on every row."""
    tandem_keys = []
    for _name, s in samples:
        for rec in s.get('tandem') or ():
            key = (rec['fluor'], rec['parent'], rec['tag'])
            if key not in tandem_keys:
                tandem_keys.append(key)
    tandem_header = []
    for fluor, parent, tag in tandem_keys:
        tandem_header += [f'{fluor} -> {parent} per unit', f'{fluor} -> {parent} z',
                          f'{fluor} -> {parent} at median {tag}']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(list(SUMMARY_COLUMNS) + list(design.combo_names) + tandem_header
                        + (['Warning'] if warning else []))
        for name, s in samples:
            recs = {(r['fluor'], r['parent'], r['tag']): r for r in s.get('tandem') or ()}
            tandem_cells = []
            for key in tandem_keys:
                rec = recs.get(key)
                tandem_cells += ['', '', ''] if rec is None else [_csv_number(rec['beta']), _csv_number(rec['z']),
                                                                 _csv_number(rec['at_median'])]
            writer.writerow([name, s['total_events'], s['untransduced'], s['unresolved'], s['rescued'],
                             s.get('rescued_toofew', 0), s['pairwise_corrected']]
                            + [s['per_combo'].get(c, 0) for c in design.combo_names]
                            + tandem_cells + ([warning] if warning else []))


def _csv_number(value) -> str:
    value = float(value)
    return f'{value:.6g}' if np.isfinite(value) else ''


def rename_combos(fc: FlowCodeSpectra, design: FlowCodeDesign) -> FlowCodeSpectra:
    """
    *fc* with its combos named and ordered as in *design*, matched by tag
    set, so a characterised backbone can be used after combos are renamed.
    Raises ValueError when the tags or the set of combos differ.
    """
    if tuple(design.tags) != tuple(fc.design.tags) or tuple(design.fluors) != tuple(fc.design.fluors):
        raise ValueError('The FlowCode tags changed since the backbone was characterised.')
    position = {combo_key(tags): g for g, tags in enumerate(fc.design.combo_tags)}
    keys = [combo_key(tags) for tags in design.combo_tags]
    if sorted(keys) != sorted(position):
        raise ValueError('The FlowCode combinations changed since the backbone was characterised.')
    order = [position[k] for k in keys]
    return FlowCodeSpectra(
        design=design,
        thresholds=fc.thresholds,
        spectra=fc.spectra,
        fluor_names=fc.fluor_names,
        detectors=fc.detectors,
        af_mean=fc.af_mean,
        combo_raw_median=np.asarray(fc.combo_raw_median)[order],
        combo_error=[fc.combo_error[g] for g in order],
        pair_model=fc.pair_model,
        rescue_floor=fc.rescue_floor,
        transform_params=fc.transform_params,
        raw_baseline=fc.raw_baseline,
        tag_background=fc.tag_background,
        settings=fc.settings,
        diagnostics=fc.diagnostics,
        fret_corrected=fc.fret_corrected,
    )


# ---------------------------------------------------------------------------
# Characterisation summaries (figures and tables)
# ---------------------------------------------------------------------------

def spillover_matrix(fc: FlowCodeSpectra, space: str = 'unmixed') -> dict:
    """
    Unmixing error per unit of on-tag abundance for each characterised
    combo, in percent, at the combo's median on-tag abundances (the slope of
    its error model there, ``combo_error_slope``, measured from the
    untransduced baseline).

    'unmixed'  : 100 x mean over the combo's on-tags k of (D_g U^T)[k, f],
                 with D_g that slope and U = solve(B B^T, B), B = [spectra;
                 AF mean]: the abundance each fluorophore (and AF) gains per
                 unit of on-tag abundance when uncorrected data are unmixed.
                 Columns are ``fluor_names`` plus 'AF'; ``on`` marks the
                 combo's own tag fluorophores.
    'detector' : 100 x mean over k of D_g[k, :] / max|S_k|, the error as a
                 percentage of each on-tag spectrum's peak. Columns are the
                 detectors; ``on`` is all False.

    Returns ``rows`` (combo names), ``combo_index``, ``columns``, ``values``
    (rows x columns), ``on`` and ``pairwise_combos`` (combos that use the
    pairwise model instead).
    """
    if space not in ('unmixed', 'detector'):
        raise ValueError(f'Unknown spillover space "{space}".')
    design = fc.design
    tag_cols = fc.tag_columns()
    fitted = [g for g in range(design.n_combos) if fc.combo_error[g] is not None]
    if space == 'unmixed':
        coef_map = _unmixing_map(fc)
        columns = list(fc.fluor_names) + ['AF']
    else:
        columns = list(fc.detectors)

    medians = np.asarray(fc.combo_raw_median, dtype=float)
    tag_baseline = fc.tag_baseline()
    values = np.zeros((len(fitted), len(columns)))
    on = np.zeros((len(fitted), len(columns)), dtype=bool)
    for r, g in enumerate(fitted):
        tags = design.on_tags(g)
        a_med = (np.nan_to_num(medians[g, tags]) - tag_baseline[tags] if medians.ndim == 2
                 else np.zeros(len(tags)))
        slope = combo_error_slope(np.asarray(fc.combo_error[g], dtype=float), a_med)
        if space == 'unmixed':
            values[r] = 100.0 * (slope @ coef_map.T).mean(axis=0)
            on[r, tag_cols[tags]] = True
        else:
            peaks = np.max(np.abs(fc.spectra[tag_cols[tags]]), axis=1)
            values[r] = 100.0 * (slope / np.maximum(peaks, _EPS)[:, None]).mean(axis=0)
    return {
        'rows': [design.combo_names[g] for g in fitted],
        'combo_index': fitted,
        'columns': columns,
        'values': values,
        'on': on,
        'pairwise_combos': [design.combo_names[g] for g in range(design.n_combos) if fc.combo_error[g] is None],
    }


def _unmixing_map(fc: FlowCodeSpectra) -> np.ndarray:
    """Least-squares unmixing map of the fluorophore spectra plus the mean
    AF spectrum ((fluorophores + 1) x detectors)."""
    basis = np.vstack([fc.spectra, fc.af_mean])
    coef_map = _solve(basis @ basis.T, basis)
    if coef_map is None:
        raise ValueError('The fluorophore spectra plus the mean AF spectrum are collinear.')
    return coef_map


def spillover_spread(fc: FlowCodeSpectra, gains: np.ndarray) -> dict:
    """
    Spillover spread of each combo's tags into every fluorophore: the
    standard deviation an unmixed channel f gains from the combo's on-tags k
    at their median abundances a_k (``combo_raw_median`` above the
    untransduced baseline),

        sqrt( sum_k a_k sum_d U[f, d]^2 g_d S_k[d] ),

    with U the least-squares unmixing map of the fluorophore spectra plus
    the mean AF spectrum (as in ``spillover_matrix``), S_k the tag spectra
    and g the photon gain of each detector (the noise model's variance per
    unit signal). A combo whose tags spread far into a channel is also the
    one in which an error in their spectra moves that channel most.

    Returns ``rows`` (combo names), ``columns`` (``fluor_names``) and
    ``values`` (combos x fluorophores, raw abundance units).
    """
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (len(fc.detectors),):
        raise ValueError('The detector gains do not match the characterisation\'s detectors.')
    design = fc.design
    coef_map = _unmixing_map(fc)[:len(fc.fluor_names)]
    tag_spectra = fc.spectra[fc.tag_columns()]
    per_unit = (coef_map[:, None, :] ** 2 * gains[None, None, :] * np.maximum(tag_spectra, 0.0)[None]).sum(axis=2)
    medians = np.asarray(fc.combo_raw_median, dtype=float)
    tag_baseline = fc.tag_baseline()
    values = np.zeros((design.n_combos, len(fc.fluor_names)))
    for g in range(design.n_combos):
        tags = design.on_tags(g)
        a = (np.maximum(np.nan_to_num(medians[g, tags]) - tag_baseline[tags], 0.0) if medians.ndim == 2
             else np.zeros(len(tags)))
        values[g] = np.sqrt(np.maximum(per_unit[:, tags] @ a, 0.0))
    return {'rows': list(design.combo_names), 'columns': list(fc.fluor_names), 'values': values}


def expression_event_indices(ids: np.ndarray, design: FlowCodeDesign, max_events: int = 20_000,
                             seed: int = 42) -> tuple:
    """The untransduced and barcoded events ``panel_expression_data`` shows:
    seeded subsamples of at most *max_events* each."""
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)

    def pick(mask):
        idx = np.flatnonzero(mask)
        if idx.size > max_events:
            idx = np.sort(rng.choice(idx, max_events, replace=False))
        return idx

    negative = pick(ids == UNTRANSDUCED)
    barcoded = pick((ids >= 1) & (ids <= design.n_combos))
    return negative, barcoded


def panel_expression_data(uncorrected_unmixed: np.ndarray, corrected_unmixed: np.ndarray, ids: np.ndarray,
                          design: FlowCodeDesign, fluor_names, axes: dict, labels: dict | None = None,
                          max_events: int = 20_000, bins: int = 200, smooth_bins: float = 2.0,
                          seed: int = 42) -> dict:
    """
    Smoothed expression of every panel marker in three groups of backbone
    events: untransduced (no positive tag) and barcoded events from
    *uncorrected_unmixed*, and the same barcoded events from
    *corrected_unmixed*. Only the rows of ``expression_event_indices`` are
    read.

    axes : {fluorophore: (forward, limits, ticks)}; ``forward`` maps
           abundances to the display scale, ``ticks`` is [(position, label)].

    Each group is a seeded subsample of at most ``max_events``. Per marker,
    each trace is a ``bins``-bin histogram over the display limits smoothed
    with a Gaussian of ``smooth_bins`` bins and scaled to a peak of 1 (None
    for a group without events).

    Returns ``markers`` (row labels), ``is_tag``, ``centres``, ``limits``,
    ``ticks``, ``traces`` ({group: per-marker list}), ``medians`` ({group:
    per-marker display-scale median}) and ``n`` ({group: events}); groups
    are 'untransduced', 'before' and 'after'.
    """
    names = list(fluor_names)
    negative, barcoded = expression_event_indices(ids, design, max_events, seed)
    groups = {'untransduced': (uncorrected_unmixed, negative), 'before': (uncorrected_unmixed, barcoded),
              'after': (corrected_unmixed, barcoded)}

    half = int(np.ceil(4 * smooth_bins)) if smooth_bins > 0 else 0
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / max(smooth_bins, _EPS)) ** 2)
    kernel /= kernel.sum()

    tag_set = set(design.fluors)
    out = {'markers': [(labels or {}).get(f, f) for f in names], 'is_tag': [f in tag_set for f in names],
           'centres': [], 'limits': [], 'ticks': [],
           'traces': {g: [] for g in groups}, 'medians': {g: [] for g in groups},
           'n': {g: int(idx.size) for g, (_m, idx) in groups.items()}}
    for j, fluor in enumerate(names):
        forward, limits, ticks = axes[fluor]
        lo, hi = (float(v) for v in limits)
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
        out['centres'].append(0.5 * (edges[:-1] + edges[1:]))
        out['limits'].append((lo, hi))
        out['ticks'].append(list(ticks))
        for group, (mat, idx) in groups.items():
            values = np.asarray(forward(mat[idx, j]), dtype=float) if idx.size else np.zeros(0)
            values = values[np.isfinite(values)]
            if values.size == 0:
                out['traces'][group].append(None)
                out['medians'][group].append(float('nan'))
                continue
            counts, _ = np.histogram(np.clip(values, lo, hi), bins=edges)
            trace = np.convolve(counts.astype(float), kernel, mode='same') if half else counts.astype(float)
            peak = trace.max()
            out['traces'][group].append(trace / peak if peak > 0 else trace)
            out['medians'][group].append(float(np.median(values)))
    return out


def _list_value(values, i) -> float:
    """Element *i* of an array or of a JSON list (None for NaN) as a float."""
    if values is None:
        return float('nan')
    v = values[i]
    return float('nan') if v is None else float(v)


def _row_value(row: dict, key: str) -> float:
    v = row.get(key)
    return float('nan') if v is None else float(v)


def fit_table_rows(fc: FlowCodeSpectra) -> list:
    """
    One row per combo: the characterisation diagnostics (support, condition
    number, error slope inside and outside the spectral span, curvature of
    the error, AF slope terms) and
    the off-target validation on fit and held-out (rescued) events. Works on
    the diagnostics of a fresh characterisation or of saved spectra.
    """
    design = fc.design
    fit = fc.diagnostics.get('fit') or {}
    validation = {(row['Id'], row['set']): row for row in fc.diagnostics.get('validation') or []}
    rows = []
    for g, name in enumerate(design.combo_names):
        held = validation.get((name, 'held.out'), {})
        fitted = validation.get((name, 'fit'), {})
        rows.append({
            'Combo': name,
            'Tags': ' '.join(design.combo_tags[g]),
            'Error': ('none' if not fc.fret_corrected else
                      'combo' if fc.combo_error[g] is not None else 'pairwise'),
            'Confident': int(_list_value(fit.get('n_confident'), g)) if fit else 0,
            'Fit': int(_list_value(fit.get('n_fit'), g)) if fit else 0,
            'Kappa': _list_value(fit.get('kappa'), g),
            'In span': _list_value(fit.get('in_span'), g),
            'Out of span': _list_value(fit.get('out_span'), g),
            'Curvature': _list_value(fit.get('curvature'), g),
            'AF slope': _list_value(fit.get('af_slope'), g),
            'AF excess': _list_value(fit.get('af_excess'), g),
            'AF confound': _list_value(fit.get('af_confound'), g),
            'Off RMS before': _row_value(fitted, 'off_rms_before'),
            'Off RMS after': _row_value(fitted, 'off_rms_after'),
            'Off spill before': _row_value(fitted, 'off_spill_before'),
            'Off spill after': _row_value(fitted, 'off_spill_after'),
            'Held-out n': int(held.get('n') or 0),
            'Held-out RMS before': _row_value(held, 'off_rms_before'),
            'Held-out RMS after': _row_value(held, 'off_rms_after'),
        })
    return rows


def write_thresholds_csv(path, design: FlowCodeDesign, thresholds_raw, transform):
    """Thresholds in the FlowCodeUnmix format: Fluor, Tag, Threshold_Raw,
    Threshold_Transformed (``transform`` = the rescue logicle)."""
    raw = np.asarray(thresholds_raw, dtype=float)
    shown = transform(raw)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Fluor', 'Tag', 'Threshold_Raw', 'Threshold_Transformed'])
        for fluor, tag, r, t in zip(design.fluors, design.tags, raw, shown):
            writer.writerow([fluor, tag, repr(float(r)), repr(float(t))])


def read_thresholds_csv(path) -> list:
    """
    Rows of a FlowCodeUnmix thresholds CSV (``Fluor``, ``Tag``,
    ``Threshold_Raw``; column names case-insensitive, ``Fluor`` or ``Tag``
    may be missing) as (fluor, tag, raw threshold). Raises ValueError for a
    file without a ``Threshold_Raw`` column, or naming the lines whose
    threshold is not a number.
    """
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError('The thresholds file is empty.')
    header = [c.strip().lower() for c in rows[0]]
    if 'threshold_raw' not in header:
        raise ValueError('The thresholds file has no Threshold_Raw column.')
    col_raw = header.index('threshold_raw')
    col_fluor = header.index('fluor') if 'fluor' in header else None
    col_tag = header.index('tag') if 'tag' in header else None
    if col_fluor is None and col_tag is None:
        raise ValueError('The thresholds file needs a Fluor or a Tag column.')
    out, bad = [], []
    for line_no, row in enumerate(rows[1:], start=2):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        cells += [''] * (len(header) - len(cells))
        try:
            raw = float(cells[col_raw])
        except ValueError:
            bad.append(line_no)
            continue
        if not np.isfinite(raw):
            bad.append(line_no)
            continue
        out.append(('' if col_fluor is None else cells[col_fluor], '' if col_tag is None else cells[col_tag], raw))
    if bad:
        raise ValueError('Threshold_Raw is not a number on line(s) ' + ', '.join(map(str, bad)) + '.')
    if not out:
        raise ValueError('The thresholds file has no thresholds.')
    return out


def match_thresholds(rows, tags, fluors) -> tuple:
    """
    Match thresholds-file rows to the design's tags: by tag name
    (case-insensitive), otherwise by fluorophore. Returns ({tag: raw},
    [rows that match no tag]); a tag named twice takes its first row.
    """
    by_tag = {str(t).upper(): t for t in tags}
    by_fluor = {str(f).upper(): t for t, f in zip(tags, fluors)}
    matched, unmatched = {}, []
    for fluor, tag, raw in rows:
        target = by_tag.get(str(tag).upper()) or by_fluor.get(str(fluor).upper())
        if target is None:
            unmatched.append(tag or fluor)
            continue
        matched.setdefault(target, float(raw))
    return matched, unmatched


# ---------------------------------------------------------------------------
# Scatter gates and unstained matching
# ---------------------------------------------------------------------------

def singlet_ratio_cutoff(area: np.ndarray, height: np.ndarray, quantile: float = 0.85) -> float:
    """Quantile of Area/Height; singlets lie below it (read.fcs.clean())."""
    ratio = np.asarray(area, dtype=float) / (np.asarray(height, dtype=float) + 1e-9)
    return float(np.quantile(ratio, quantile))


def singlet_polygon(cutoff: float, area_max: float, height_max: float) -> np.ndarray:
    """
    Raw-unit polygon on (Area, Height) holding events with
    Area / Height < cutoff, i.e. Height > Area / cutoff, up to the channel
    maxima.
    """
    if cutoff <= 0:
        raise ValueError('The singlet ratio cutoff must be positive.')
    if area_max / cutoff <= height_max:
        return np.array([[0.0, 0.0], [area_max, area_max / cutoff],
                         [area_max, height_max], [0.0, height_max]])
    return np.array([[0.0, 0.0], [cutoff * height_max, height_max], [0.0, height_max]])


def landmark_indices(ols_tags: np.ndarray, n_per_tag: int = 100, method: str = 'triplet') -> np.ndarray:
    """
    Landmark events for the scatter gate (define.gate()).

    'triplet': the top ``n_per_tag * n_tags`` events by their third-highest
    tag abundance (events clearly positive for three tags). 'per.tag': the
    top ``n_per_tag`` events of each tag.
    """
    tags = np.where(np.isnan(ols_tags), -np.inf, ols_tags)
    n_events, n_tags = tags.shape
    if method == 'triplet':
        if n_tags < 3:
            raise ValueError('Triplet landmarks need at least 3 FlowCode tags.')
        third = -np.sort(-tags, axis=1)[:, 2]
        n_top = min(n_per_tag * n_tags, n_events)
        return np.argsort(-third, kind='stable')[:n_top]
    if method == 'per.tag':
        picks = []
        for j in range(n_tags):
            if np.all(np.isneginf(tags[:, j])):
                continue
            picks.extend(np.argsort(-tags[:, j], kind='stable')[:min(n_per_tag, n_events)])
        return np.array(list(dict.fromkeys(picks)), dtype=np.int64)
    raise ValueError(f'Unknown landmark method "{method}".')


def landmark_gate(scatter: np.ndarray, percentile: float = 0.70, random_state: int = 42) -> np.ndarray:
    """
    Convex hull, in raw scatter units, of the landmark events within the
    ``percentile`` contour of squared robust Mahalanobis distance (minimum
    covariance determinant, without reweighting, as MASS::cov.rob).

    Returns the hull vertices (counter-clockwise). Raises ValueError when the
    landmarks do not span a 2-D region.
    """
    from scipy.spatial import ConvexHull
    from sklearn.covariance import MinCovDet

    scatter = np.asarray(scatter, dtype=float)
    if len(scatter) < 3:
        raise ValueError('Too few landmark events for a scatter gate.')
    centred = scatter - scatter.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[-1] <= sv[0] * max(centred.shape) * _EPS:
        raise ValueError('The landmark events lie on a line; they do not define a scatter gate.')
    try:
        mcd = MinCovDet(random_state=random_state).fit(scatter)
        diff = scatter - mcd.raw_location_
        d2 = np.einsum('ij,jk,ik->i', diff, np.linalg.inv(mcd.raw_covariance_), diff)
        kept = scatter[d2 <= np.quantile(d2, percentile)]
        hull = ConvexHull(kept)
    except Exception as exc:
        raise ValueError(f'Could not define the landmark scatter gate: {exc}') from exc
    return kept[hull.vertices]


def polygon_area(vertices) -> float:
    """Shoelace area."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(vertices) -> np.ndarray:
    """Area centroid of a closed polygon (vertex mean when degenerate)."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if len(v) < 3 or abs(area) <= _EPS * max(1.0, float(np.abs(v).max()) ** 2):
        return v.mean(axis=0)
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6.0 * area)


def expand_polygon(vertices, factor: float) -> np.ndarray:
    """Scale a polygon about its area centroid: with *factor* 1.5 every
    vertex ends up 1.5 times as far from the centroid (area x 2.25)."""
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3 or factor == 1.0:
        return v.copy()
    centre = polygon_centroid(v)
    return centre + float(factor) * (v - centre)


def scatter_histogram(x_t: np.ndarray, y_t: np.ndarray, x_limits, y_limits, bins: int = 64) -> np.ndarray:
    """Normalised 2-D histogram of transformed scatter over the axis limits."""
    hist, _xe, _ye = np.histogram2d(x_t, y_t, bins=bins, range=[list(x_limits), list(y_limits)])
    total = hist.sum()
    return hist / total if total > 0 else hist


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya coefficient of two normalised histograms (1 = identical)."""
    return float(np.sum(np.sqrt(p * q)))


def best_match(scores: dict, order) -> str | None:
    """Key with the highest score; ties go to the earliest key in *order*."""
    best, best_score = None, -np.inf
    for key in order:
        score = scores.get(key)
        if score is None or not np.isfinite(score):
            continue
        if score > best_score:
            best, best_score = key, score
    return best


def clean_events_mask(fl_raw: np.ndarray, scatter: np.ndarray, magnitude_ceiling: float,
                      scatter_ceilings, frac: float = 0.999) -> np.ndarray:
    """
    Events to keep: no spectral detector at or above ``frac`` x the magnitude
    ceiling and no scatter channel at or above ``frac`` x its PnR.

    scatter          : events x scatter channels (e.g. FSC-A, SSC-A).
    scatter_ceilings : PnR of each scatter column.
    """
    keep = ~np.any(fl_raw >= frac * float(magnitude_ceiling), axis=1)
    ceilings = np.asarray(scatter_ceilings, dtype=float)
    if scatter.size:
        keep &= ~np.any(scatter >= frac * ceilings[None, :], axis=1)
    return keep


# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------

def rectangle_vertices(x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    """Corners of an axis-aligned rectangle, counter-clockwise."""
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)


def densify_closed_polygon(vertices, points_per_edge: int = 64) -> np.ndarray:
    """Evenly spaced points along every edge of a closed polygon (closing edge
    included, last vertex not repeated), so the shape survives a non-linear
    change of axis transform."""
    v = np.asarray(vertices, dtype=float)
    if len(v) < 2 or points_per_edge <= 1:
        return v.copy()
    t = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)[:, None]
    ends = np.roll(v, -1, axis=0)
    return np.vstack([a + t * (b - a) for a, b in zip(v, ends)])


def map_vertices(vertices, fx, fy) -> np.ndarray:
    """Apply per-axis functions to polygon vertices."""
    v = np.asarray(vertices, dtype=float)
    return np.column_stack([np.asarray(fx(v[:, 0]), dtype=float), np.asarray(fy(v[:, 1]), dtype=float)])


def points_in_polygon(x: np.ndarray, y: np.ndarray, vertices) -> np.ndarray:
    """Boolean mask of the points inside the closed polygon *vertices*."""
    from matplotlib.path import Path as MplPath

    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        return np.zeros(len(x), dtype=bool)
    path = MplPath(np.vstack([v, v[:1]]), closed=True)
    return path.contains_points(np.column_stack([x, y]))


def auto_singlet_gate(area: np.ndarray, height: np.ndarray, area_max: float, height_max: float,
                      quantile: float = 0.85) -> tuple:
    """Area/Height ratio singlet gate (read.fcs.clean()): returns (cutoff,
    raw-unit polygon)."""
    cutoff = singlet_ratio_cutoff(area, height, quantile)
    return cutoff, singlet_polygon(cutoff, area_max, height_max)


@dataclass(eq=False)
class LandmarkGateResult:
    """Landmark scatter gate, or the reason it could not be built."""
    vertices: np.ndarray | None
    landmark_idx: np.ndarray
    reason: str | None = None


def auto_landmark_gate(raw_fl: np.ndarray, scatter_xy: np.ndarray, spectra: np.ndarray,
                       tag_rows, n_per_tag: int = 100, percentile: float = 0.70,
                       min_landmarks: int = 50, to_display=None,
                       min_display_area: float = 1e-4, random_state: int = 42) -> LandmarkGateResult:
    """
    FSC/SSC gate around events clearly positive for three tags (define.gate()
    with landmark = "triplet").

    raw_fl     : events x detectors (singlet events).
    scatter_xy : the same events' raw (FSC-A, SSC-A).
    spectra    : fluorophores x detectors, no AF; OLS basis for the landmarks.
    tag_rows   : rows of *spectra* that are FlowCode tags.
    to_display : optional raw -> display mapping of polygon vertices, for the
                 minimum-area check.

    Returns a result whose ``vertices`` are None, with a ``reason``, when
    there are fewer than 3 tags or ``min_landmarks`` landmarks, the robust
    fit fails, or the hull is degenerate.
    """
    tag_rows = np.asarray(tag_rows, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    if tag_rows.size < 3:
        return LandmarkGateResult(None, empty, 'fewer than 3 tags')
    if len(raw_fl) == 0:
        return LandmarkGateResult(None, empty, 'no singlet events')
    ols_tags = ols_unmix(raw_fl, spectra)[:, tag_rows]
    idx = landmark_indices(ols_tags, n_per_tag=n_per_tag, method='triplet')
    if idx.size < min_landmarks:
        return LandmarkGateResult(None, idx, f'only {idx.size} landmark events (need {min_landmarks})')
    try:
        hull = landmark_gate(scatter_xy[idx], percentile=percentile, random_state=random_state)
    except ValueError as exc:
        return LandmarkGateResult(None, idx, str(exc))
    shown = to_display(hull) if to_display is not None else hull
    if polygon_area(shown) < min_display_area:
        return LandmarkGateResult(None, idx, 'the landmark gate is too small')
    return LandmarkGateResult(hull, idx, None)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def initial_threshold(x_t: np.ndarray, floor_t: float | None = None, min_delta_bic: float = 10.0,
                      max_events: int = 50_000, seed: int = 42) -> tuple:
    """
    Starting positivity threshold for one tag, in transformed units.

    A 2-component Gaussian mixture crossover when the data are bimodal
    (ΔBIC > ``min_delta_bic``), otherwise the 1% upper tail of the mirrored
    negative population; never below ``floor_t``. Uses a seeded subsample of
    at most ``max_events``.

    Returns (threshold, method, delta_bic); method is 'mixture', 'tail' or
    'floor'.
    """
    _ensure_bundled_plugins_on_path()
    import ag_core

    x = np.asarray(x_t, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) > max_events:
        x = np.random.default_rng(seed).choice(x, max_events, replace=False)
    delta_bic = ag_core.bimodality_delta_bic(x)
    if delta_bic > min_delta_bic:
        threshold, method = ag_core.threshold_mixture(x, n_components=2), 'mixture'
    else:
        threshold, method = ag_core.threshold_tail(x, tail_fraction=0.01), 'tail'
    if floor_t is not None and np.isfinite(floor_t) and floor_t > threshold:
        threshold, method = float(floor_t), 'floor'
    return float(threshold), method, float(delta_bic)


def unstained_floor(unstained_tags: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    """Per-tag positivity floor: the 99.5th percentile of the gated matched
    unstained in debarcoding space (raw units)."""
    if unstained_tags is None or len(unstained_tags) == 0:
        return np.full(0 if unstained_tags is None else unstained_tags.shape[1], np.nan)
    return np.quantile(unstained_tags, quantile, axis=0)


def rescue_floor(pos_thresholds, fluor_names, tag_fluors, unstained_tags: np.ndarray) -> tuple:
    """
    Noise floor per tag for the too-many-tag rescue.

    The AutoSpectral Optimization positivity thresholds (one per fluorophore,
    ``fluor_names`` order) when available and complete; otherwise the gated
    matched unstained floor. Returns (floor, source), source being
    'optimization' or 'unstained'.
    """
    fluor_names = list(fluor_names)
    if pos_thresholds is not None:
        pos = np.asarray(pos_thresholds, dtype=float).ravel()
        if pos.size == len(fluor_names):
            floor = np.array([pos[fluor_names.index(f)] for f in tag_fluors])
            if np.all(np.isfinite(floor)):
                return floor, 'optimization'
    return unstained_floor(unstained_tags), 'unstained'


def initial_thresholds(tag_values: np.ndarray, floor_raw: np.ndarray, magnitude_ceiling: float,
                       min_delta_bic: float = 10.0, max_events: int = 50_000, seed: int = 42) -> list:
    """
    Starting thresholds for every tag column of *tag_values* (debarcoding
    space, raw units), set in the fixed rescue logicle and returned in raw
    units.

    Returns one dict per tag: ``raw``, ``transformed``, ``method``,
    ``delta_bic``.
    """
    forward = logicle_transform(magnitude_ceiling)
    inverse = logicle_inverse(magnitude_ceiling)
    floor_raw = np.asarray(floor_raw, dtype=float)
    out = []
    for j in range(tag_values.shape[1]):
        x_t = _transform_finite(forward, tag_values[:, j])
        floor_t = float(forward(np.array([floor_raw[j]]))[0]) if np.isfinite(floor_raw[j]) else None
        t, method, delta_bic = initial_threshold(x_t, floor_t, min_delta_bic=min_delta_bic,
                                                 max_events=max_events, seed=seed)
        out.append({'raw': float(inverse(np.array([t]))[0]), 'transformed': t,
                    'method': method, 'delta_bic': delta_bic})
    return out


# ---------------------------------------------------------------------------
# Debarcoding space
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class DebarcodeSpace:
    """Gated backbone and matched unstained in debarcoding space (per-cell AF,
    no variants), plus the quantities derived from them."""
    backbone: np.ndarray
    af_scale: np.ndarray
    af_idx: np.ndarray
    unstained: np.ndarray
    af_mean: np.ndarray
    floor: np.ndarray
    rescue_floor: np.ndarray
    rescue_floor_source: str
    engine: str


def debarcoding_space(backbone_fl: np.ndarray, unstained_fl: np.ndarray, spectra: np.ndarray,
                      fluor_names, tag_fluors, af_spectra: np.ndarray, pos_thresholds=None,
                      cell_weight: bool = False, use_kernel: bool = False,
                      n_threads: int | None = None, envelope: dict | None = None,
                      unmix_fn=None, engine: str | None = None) -> DebarcodeSpace:
    """
    Per-cell AF unmixing of the gated backbone and gated matched unstained
    with the matched AF library (unmix.backbone()'s debarcoding unmix), the
    mean AF spectrum of the gated unstained, the unstained positivity floor
    and the rescue floor.

    backbone_fl, unstained_fl : events x detectors, raw.
    spectra                   : fluorophores x detectors, no AF.
    envelope                  : the AF library's negative envelope, or None
                                (``af_only_unmix``).
    unmix_fn                  : raw -> dict(unmixed, af_scale, af_idx), the
                                debarcoding unmix in place of
                                ``af_only_unmix``; *engine* names it.
    """
    fluor_names = list(fluor_names)
    tag_cols = [fluor_names.index(f) for f in tag_fluors]
    if unmix_fn is None:
        def unmix_fn(raw):
            return af_only_unmix(raw, spectra, fluor_names, af_spectra, cell_weight=cell_weight,
                                 n_threads=n_threads, use_kernel=use_kernel, envelope=envelope)
    bb = unmix_fn(backbone_fl)
    if len(unstained_fl):
        un = unmix_fn(unstained_fl)['unmixed']
        af_mean = af_mean_spectrum(unstained_fl)
    else:
        un = np.zeros((0, len(fluor_names)))
        af_mean = np.full(spectra.shape[1], np.nan)
    floor = unstained_floor(un[:, tag_cols]) if len(un) else np.full(len(tag_cols), np.nan)
    rfloor, source = rescue_floor(pos_thresholds, fluor_names, tag_fluors, un[:, tag_cols])
    return DebarcodeSpace(backbone=bb['unmixed'], af_scale=bb['af_scale'], af_idx=bb['af_idx'],
                          unstained=un, af_mean=af_mean, floor=floor, rescue_floor=rfloor,
                          rescue_floor_source=source, engine=engine or bb.get('engine', ''))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_flowcode_spectra(fc: FlowCodeSpectra, npz_path) -> dict:
    """
    Write the arrays to *npz_path* and return JSON-serialisable metadata
    (stored in the experiment) that ``load_flowcode_spectra`` reads back.
    """
    arrays = {
        'thresholds': fc.thresholds,
        'spectra': fc.spectra,
        'af_mean': fc.af_mean,
        'combo_raw_median': fc.combo_raw_median,
    }
    if fc.pair_model is not None:
        arrays['pair_linear'] = fc.pair_model.linear
        arrays['pair_product'] = fc.pair_model.product
        arrays['pair_support'] = fc.pair_model.support
    if fc.rescue_floor is not None:
        arrays['rescue_floor'] = fc.rescue_floor
    if fc.raw_baseline is not None:
        arrays['raw_baseline'] = fc.raw_baseline
    if fc.tag_background is not None:
        arrays['tag_background'] = fc.tag_background
    for g, error in enumerate(fc.combo_error):
        if error is not None:
            arrays[f'combo_error__{g}'] = error
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **arrays)

    diagnostics = {key: value for key, value in fc.diagnostics.items() if key not in ('rescue', 'rescue_toofew')}
    return _jsonable({
        'error_model': ERROR_MODEL,
        'tags': list(fc.design.tags),
        'fluors': list(fc.design.fluors),
        'combo_names': list(fc.design.combo_names),
        'combo_tags': [list(t) for t in fc.design.combo_tags],
        'fluor_names': list(fc.fluor_names),
        'detectors': list(fc.detectors),
        'transform_params': fc.transform_params,
        'settings': fc.settings,
        'diagnostics': diagnostics,
        'fingerprint': fc.fingerprint(),
        'fret_corrected': bool(fc.fret_corrected),
    })


def load_flowcode_spectra(meta: dict, npz_path) -> FlowCodeSpectra:
    """Inverse of ``save_flowcode_spectra``. Raises ValueError for a
    characterisation saved with another error model."""
    if meta.get('error_model') != ERROR_MODEL:
        raise ValueError('The saved characterisation uses an earlier error model: run section 4 again.')
    design = FlowCodeDesign.build(
        dict(zip(meta['tags'], meta['fluors'])),
        list(zip(meta['combo_names'], [tuple(t) for t in meta['combo_tags']])),
    )
    with np.load(str(npz_path)) as data:
        arrays = {key: data[key] for key in data.files}
    pair_model = None
    if 'pair_linear' in arrays:
        pair_model = PairModel(linear=arrays['pair_linear'], product=arrays['pair_product'],
                               support=arrays['pair_support'].astype(bool))
    return FlowCodeSpectra(
        design=design,
        thresholds=arrays['thresholds'],
        spectra=arrays['spectra'],
        fluor_names=tuple(meta['fluor_names']),
        detectors=tuple(meta['detectors']),
        af_mean=arrays['af_mean'],
        combo_raw_median=arrays['combo_raw_median'],
        combo_error=[arrays.get(f'combo_error__{g}') for g in range(design.n_combos)],
        pair_model=pair_model,
        rescue_floor=arrays.get('rescue_floor'),
        transform_params=dict(meta.get('transform_params') or {}),
        raw_baseline=arrays.get('raw_baseline'),
        tag_background=arrays.get('tag_background'),
        settings=dict(meta.get('settings') or {}),
        diagnostics=dict(meta.get('diagnostics') or {}),
        fret_corrected=bool(meta.get('fret_corrected', True)),
    )
