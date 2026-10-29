"""
flowcode_phenotype.py
---------------------
Gate-free phenotype populations for the FlowCode Analysis page.

A phenotype is a list of marker signs (CD3+ CD4+ FoxP3+ CD8-). Instead of a
drawn gate, each event gets a positivity probability per listed marker from
the experiment's own negative envelope: the SD of the negative is the
square root of the sample's AF library null variance (``c0 + c1 k + c2 k^2``
in the AF abundance k) plus the spillover spread at the event's abundances,
the same quantities the positivity-weighted AF assignment uses
(``autospectral_functions``). With z the abundance in units of that SD, the
probability is a logistic soft threshold, ``sigmoid((z - POSITIVE_Z) /
POSITIVE_WIDTH)``: one half at ``POSITIVE_Z`` SDs above the negative, 0.12
one SD below that and 0.88 one SD above it, 0.0025 at the negative's
centre. An event's membership is the product of the probabilities
(positive markers) and their complements (negative markers); a
population's size is the sum of memberships, a soft count, per barcode
combination.

Definitions are written in canonical marker names (``marker_database.csv``)
and resolved against the experiment's spectral model, so they transfer
across panels. A positive marker the panel lacks leaves the definition
unresolved; a negative marker the panel lacks is dropped and reported as
not checked. The panel's viability dye (``is.viability`` in the fluorophore
database) is added as a negative marker when there is one.

Limits. The probability says "above the negative" only, so level
distinctions (CD44 high against intermediate) are out of scope. Scatter has
no probability; singlets and debris stay on the FlowCode gates. The product
treats the markers as independent. The abundances scored are the FlowCode
unmix's final values at the event's chosen AF row, while the envelope
describes the negative at the library's mean AF row; under the penalised
methods the negatives are narrower than the envelope says, so the threshold
sits a little further from the negative than its nominal SDs. A marker whose
positives lie within a few SDs of the negative gets graded probabilities,
not a clean split.

No Qt. Worker-safe.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCORE_VERSION = 'soft-threshold-1'
POSITIVE_Z = 3.0            # SDs above the negative where the probability is one half
POSITIVE_WIDTH = 0.5        # SDs over which it goes from 0.27 to 0.73
DEFINITIONS_FILE = 'flowcode_phenotypes.csv'
BUNDLED_DATABASE = 'drc_cell_type_database.csv'
SOURCE_PLUGIN = 'plugin'
SOURCE_BUNDLED = 'bundled'
AF_ABUNDANCE = 'AF Abundance'


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Definition:
    """One phenotype: canonical marker names, positive and negative."""
    cell_type: str
    species: str
    positive: tuple
    negative: tuple
    source: str = SOURCE_PLUGIN
    notes: str = ''

    @property
    def key(self) -> str:
        return f'{self.cell_type} ({self.species})' if self.species else self.cell_type

    def text(self) -> str:
        """``CD3+ CD4+ FoxP3+ CD8-``."""
        return ' '.join([f'{m}+' for m in self.positive] + [f'{m}-' for m in self.negative])


_canonical_cache: dict = {}


def canonical_marker(name: str, marker_db=None) -> str:
    """The marker database's canonical spelling of *name*, or *name*
    stripped when it knows none. Lookups in the bundled database are
    remembered, since the same marker names recur across definitions."""
    name = str(name or '').strip()
    if not name:
        return ''
    if marker_db is None and name in _canonical_cache:
        return _canonical_cache[name]
    from honeychrome.controller_components.label_matching import get_marker_db, match_marker
    found = match_marker(name, marker_db if marker_db is not None else get_marker_db()) or name
    if marker_db is None:
        _canonical_cache[name] = found
    return found


def _split_markers(text: str, marker_db) -> tuple:
    out = []
    for part in str(text or '').split(';'):
        name = canonical_marker(part, marker_db)
        if name and name not in out:
            out.append(name)
    return tuple(out)


_definitions_cache: dict = {}


def read_definitions(path, source: str, marker_db=None) -> list:
    """Definitions in a CSV with columns ``cell_type, species, positive_markers,
    negative_markers, notes`` (the bundled database's format). Rows without
    any marker are skipped; marker names are canonicalised. A file is read
    again only when it changed on disk."""
    path = Path(path)
    if not path.exists():
        return []
    stat = path.stat()
    key = (str(path), source, stat.st_mtime_ns, stat.st_size)
    if marker_db is None and key in _definitions_cache:
        return list(_definitions_cache[key])
    out = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            positive = _split_markers(row.get('positive_markers'), marker_db)
            negative = _split_markers(row.get('negative_markers'), marker_db)
            if not positive and not negative:
                continue
            out.append(Definition(cell_type=(row.get('cell_type') or '').strip(),
                                  species=(row.get('species') or '').strip(),
                                  positive=positive, negative=negative, source=source,
                                  notes=(row.get('notes') or '').strip()))
    if marker_db is None:
        _definitions_cache[key] = list(out)
    return out


def bundled_database_path() -> Path:
    import honeychrome
    return Path(honeychrome.__file__).resolve().parent / 'bundled_plugins' / BUNDLED_DATABASE


def load_definitions(plugin_dir=None, include_bundled: bool = True, marker_db=None) -> list:
    """The plugin's definitions (``flowcode_phenotypes.csv`` beside this
    module, or in *plugin_dir*) followed by the bundled cell-type database.
    A plugin definition hides a bundled one with the same cell type and
    species."""
    plugin_dir = Path(plugin_dir) if plugin_dir is not None else Path(__file__).resolve().parent
    out = read_definitions(plugin_dir / DEFINITIONS_FILE, SOURCE_PLUGIN, marker_db)
    if include_bundled:
        seen = {d.key for d in out}
        for d in read_definitions(bundled_database_path(), SOURCE_BUNDLED, marker_db):
            if d.key not in seen:
                out.append(d)
                seen.add(d.key)
    return out


# ---------------------------------------------------------------------------
# Resolution against the panel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Panel:
    """The experiment's labelled fluorophores and their canonical markers."""
    fluorophores: tuple                     # fluorophore labels in spectral-model order
    markers: dict                           # canonical marker -> tuple of fluorophore labels
    antigens: dict                          # fluorophore -> antigen text as typed
    unmatched: tuple                        # (fluorophore, antigen) pairs the marker database does not know
    viability: tuple                        # fluorophores that are viability dyes


def panel_from_spectral_model(spectral_model, fluor_names=None, marker_db=None) -> Panel:
    """*spectral_model* is the experiment's list of ``{'label', 'antigen'}``;
    *fluor_names* restricts it to the fluorophores with a reference spectrum
    (spectral-model order is kept)."""
    keep = None if fluor_names is None else set(fluor_names)
    fluorophores, antigens, markers, unmatched = [], {}, {}, []
    from honeychrome.controller_components.label_matching import get_marker_db, match_marker
    db = marker_db if marker_db is not None else get_marker_db()
    for entry in spectral_model or []:
        label = entry.get('label')
        if not label or (keep is not None and label not in keep):
            continue
        fluorophores.append(label)
        antigen = str(entry.get('antigen') or '').strip()
        antigens[label] = antigen
        if not antigen:
            continue
        canonical = match_marker(antigen, db)
        if canonical is None:
            unmatched.append((label, antigen))
            canonical = antigen
        markers.setdefault(canonical, []).append(label)
    return Panel(fluorophores=tuple(fluorophores), markers={k: tuple(v) for k, v in markers.items()},
                 antigens=antigens, unmatched=tuple(unmatched), viability=tuple(viability_fluorophores(fluorophores)))


def viability_fluorophores(fluorophores) -> list:
    """The viability dyes among *fluorophores* (Automated Gating's lookup)."""
    try:
        from flowcode_core import _ensure_bundled_plugins_on_path
        _ensure_bundled_plugins_on_path()
        import ag_core
        found = ag_core.viability_channels(fluorophores)
    except Exception as exc:
        logger.warning('Viability dye lookup unavailable (%s)', exc)
        return []
    return [f for f in fluorophores if f in found]


@dataclass(frozen=True)
class Resolved:
    """A definition on a panel: ``markers`` are ``(canonical marker,
    fluorophore, sign)`` with sign +1 or -1, in definition order with the
    viability dye last; ``not_checked`` the negative markers the panel
    lacks."""
    definition: Definition
    markers: tuple
    not_checked: tuple = ()
    viability: str | None = None

    @property
    def fluorophores(self) -> tuple:
        return tuple(f for _m, f, _s in self.markers)

    @property
    def signs(self) -> np.ndarray:
        return np.array([s for _m, _f, s in self.markers], dtype=np.float64)

    def summary(self) -> str:
        parts = [f'{m}{"+" if s > 0 else "-"} ({f})' for m, f, s in self.markers]
        text = ' '.join(parts)
        if self.not_checked:
            text += '; not checked: ' + ', '.join(self.not_checked)
        return text

    def signature_parts(self) -> dict:
        """What decides the scored values, for a counts signature."""
        return {'phenotype': self.definition.key, 'markers': [list(m) for m in self.markers],
                'not_checked': list(self.not_checked), 'score': SCORE_VERSION}


@dataclass(frozen=True)
class Unresolved:
    definition: Definition
    reason: str
    missing: tuple = ()
    ambiguous: tuple = ()


def resolve(definition: Definition, panel: Panel, viability: bool = True):
    """``Resolved`` or ``Unresolved``. Every positive marker must be on
    exactly one fluorophore; a negative marker on none is dropped into
    ``not_checked``; a marker on two fluorophores is ambiguous. With
    *viability*, the panel's viability dye (the first when there are
    several) is appended as a negative marker."""
    markers, missing, ambiguous, not_checked = [], [], [], []
    for sign, names in ((1, definition.positive), (-1, definition.negative)):
        for name in names:
            found = panel.markers.get(name, ())
            if len(found) == 1:
                markers.append((name, found[0], sign))
            elif len(found) > 1:
                ambiguous.append((name, found))
            elif sign > 0:
                missing.append(name)
            else:
                not_checked.append(name)
    if missing or ambiguous:
        reasons = []
        if missing:
            reasons.append('not in the panel: ' + ', '.join(missing))
        if ambiguous:
            reasons.append('on several fluorophores: ' + ', '.join(f'{m} ({", ".join(f)})' for m, f in ambiguous))
        return Unresolved(definition=definition, reason='; '.join(reasons), missing=tuple(missing),
                          ambiguous=tuple(ambiguous))
    dye = None
    if viability and panel.viability:
        dye = panel.viability[0]
        if dye not in {f for _m, f, _s in markers}:
            markers.append(('viability', dye, -1))
    if not markers:
        return Unresolved(definition=definition, reason='no marker to score')
    return Resolved(definition=definition, markers=tuple(markers), not_checked=tuple(not_checked), viability=dye)


CONGENIC_MARKERS = ('CD45.1', 'CD45.2')


def split_congenic(resolved: Resolved, panel: Panel, markers=CONGENIC_MARKERS) -> dict:
    """
    ``{'<cell type> CD45.1': Resolved, '<cell type> CD45.2': Resolved}``:
    the population split by congenic marker, each variant positive for one
    marker and negative for the other, placed before the viability dye.
    Empty when the panel does not carry both markers on one fluorophore
    each, or the definition already lists one of them.
    """
    listed = {m for m, _f, _s in resolved.markers}
    fluors = [panel.markers.get(m, ()) for m in markers]
    if any(len(f) != 1 for f in fluors) or listed & set(markers):
        return {}
    body = [m for m in resolved.markers if m[0] != 'viability']
    tail = [m for m in resolved.markers if m[0] == 'viability']
    out = {}
    for own, other in ((markers[0], markers[1]), (markers[1], markers[0])):
        extra = [(own, panel.markers[own][0], 1), (other, panel.markers[other][0], -1)]
        definition = Definition(cell_type=f'{resolved.definition.cell_type} {own}', species=resolved.definition.species,
                                positive=resolved.definition.positive + (own,),
                                negative=resolved.definition.negative + (other,),
                                source=resolved.definition.source, notes=resolved.definition.notes)
        out[definition.cell_type] = Resolved(definition=definition, markers=tuple(body + extra + tail),
                                             not_checked=resolved.not_checked, viability=resolved.viability)
    return out


# ---------------------------------------------------------------------------
# The negative model and the score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegativeModel:
    """Variance of the negative per fluorophore: ``null_var`` (3, F) as c0,
    c1, c2 in the AF abundance, ``spread`` (F, F) spillover spread or None,
    in the order of ``fluor_names``."""
    fluor_names: tuple
    null_var: np.ndarray
    spread: np.ndarray | None = None
    origin: str = ''

    def variance(self, abundances: np.ndarray, af_abundance: np.ndarray, columns) -> np.ndarray:
        """(n, len(columns)) variance of the negative at each event:
        *abundances* (n, F) in ``fluor_names`` order, *af_abundance* (n,)."""
        cols = np.asarray(columns, dtype=np.intp)
        k = np.asarray(af_abundance, dtype=np.float64)[:, np.newaxis]
        c0, c1, c2 = self.null_var[:, cols]
        var = c0 + k * (c1 + k * c2)
        if self.spread is not None and np.any(self.spread):
            var = var + np.maximum(np.asarray(abundances, dtype=np.float64), 0.0) @ self.spread[:, cols]
        return var


def negative_model(af, fluor_names, state=None, is_bead: bool = False) -> NegativeModel | None:
    """The negative model of one sample: its AF library's runtime envelope
    (*af* is the ``AFState``; the first mean row's null variance and the
    spread), else the noise model's no-AF variance and spread from the
    experiment's method *state*. None when there is neither."""
    names = tuple(fluor_names)
    envelope = None if af is None or not getattr(af, 'precomputed', None) else af.precomputed.get('envelope')
    if envelope is not None and 'null_var' in envelope:
        null_var = np.asarray(envelope['null_var'], dtype=np.float64)
        if null_var.ndim == 3:
            null_var = null_var[0]
        if null_var.shape != (3, len(names)):
            raise ValueError(f'negative envelope has {null_var.shape[-1]} fluorophores, the panel {len(names)}')
        spread = envelope.get('spread')
        return NegativeModel(names, null_var, None if spread is None else np.asarray(spread, dtype=np.float64),
                             origin='AF envelope')
    noise = getattr(state, 'noise', None)
    if noise is not None:
        var = noise.bead_null_var if is_bead and noise.bead_null_var is not None else noise.cell_null_var
        var = np.asarray(var, dtype=np.float64)
        if var.shape != (len(names),):
            raise ValueError(f'noise model has {var.shape[0]} fluorophores, the panel {len(names)}')
        null_var = np.vstack([var, np.zeros_like(var), np.zeros_like(var)])
        spread = getattr(state, 'spread_no_af', None)
        return NegativeModel(names, null_var, None if spread is None else np.asarray(spread, dtype=np.float64),
                             origin='noise model')
    return None


def sample_negative_model(af, fluor_names, state=None, sample_key=None) -> NegativeModel | None:
    """``negative_model`` for one sample under the experiment's method: a
    sample on a corrected spectra set takes that set's rebuilt envelope, and
    a bead sample the beads' no-AF variance."""
    is_bead = False
    if state is not None and sample_key is not None:
        is_bead = bool(state.is_bead(sample_key))
        set_id = state.set_for(sample_key)
        if set_id and af is not None:
            from honeychrome.controller_components import unmix_method as um
            af = um._set_af(state, set_id, af)
    return negative_model(af, fluor_names, state, is_bead)


def positivity(abundances: np.ndarray, af_abundance: np.ndarray, columns, model: NegativeModel) -> np.ndarray:
    """(n, len(columns)) probability that each event is positive in each
    listed fluorophore column: the logistic soft threshold of the abundance
    in units of the model's negative SD (see the module docstring)."""
    from scipy.special import expit

    cols = np.asarray(columns, dtype=np.intp)
    a = np.asarray(abundances, dtype=np.float64)
    sd = np.sqrt(np.maximum(model.variance(a, af_abundance, cols), 1e-12))
    return expit((a[:, cols] / sd - POSITIVE_Z) / POSITIVE_WIDTH)


def membership(pi: np.ndarray, signs) -> np.ndarray:
    """Product over the listed markers of pi (sign +1) or 1 - pi (sign -1)."""
    signs = np.asarray(signs, dtype=np.float64)
    terms = np.where(signs > 0, pi, 1.0 - pi)
    return np.prod(terms, axis=1)


def soft_counts(weights: np.ndarray, ids: np.ndarray, n_combos: int) -> np.ndarray:
    """Membership summed per combination id (0 = not barcoded, 1..n the
    combinations, n + 1 = unresolved), length n + 2."""
    ids = np.clip(np.asarray(ids, dtype=np.int64), 0, n_combos + 1)
    return np.bincount(ids, weights=np.asarray(weights, dtype=np.float64), minlength=n_combos + 2)


def population_mask(events: np.ndarray, pnn: list, resolved: Resolved, model: NegativeModel, gate_records=(),
                    threshold: float = 0.5) -> np.ndarray:
    """Events whose membership is at least *threshold*, inside the FlowCode
    gates (*gate_records*, raw-unit polygons on the scatter channels; every
    event without them): the phenotype as a hard population, for display
    and comparison with drawn gates."""
    mask = score_events(events, pnn, resolved, model)['membership'] >= threshold
    if gate_records:
        from flowcode_core import gate_records_mask
        inside = gate_records_mask(events, list(pnn), list(gate_records))
        if inside is not None:
            mask &= inside
    return mask


def score_events(events: np.ndarray, pnn: list, resolved: Resolved, model: NegativeModel) -> dict:
    """
    Score one sample's FlowCode-unmixed events (columns *pnn*, as the live
    unmix or an exported file lays them out): ``membership`` (n,),
    ``pi`` (n, markers) and ``columns`` (the fluorophore columns scored).
    """
    pnn = list(pnn)
    fl_cols = [pnn.index(f) for f in model.fluor_names]
    abundances = np.asarray(events, dtype=np.float64)[:, fl_cols]
    af_abundance = np.asarray(events[:, pnn.index(AF_ABUNDANCE)], dtype=np.float64) if AF_ABUNDANCE in pnn \
        else np.zeros(len(events))
    columns = [model.fluor_names.index(f) for f in resolved.fluorophores]
    pi = positivity(abundances, af_abundance, columns, model)
    return {'membership': membership(pi, resolved.signs), 'pi': pi, 'columns': columns}
