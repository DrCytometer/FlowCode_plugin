"""
flowcode_analysis.py
--------------------
Data side of the FlowCode Analysis page. No Qt.

The page takes its events from one of two sources:

* **Live unmix** (``SOURCE_LIVE``): each experiment sample's raw file,
  FlowCode-unmixed in memory exactly as section 5 would export it
  (``flowcode_export.live_sample_events``), with the experiment's unmixing
  method;
* **Exported files** (``SOURCE_EXPORTED``): the files written by section 5,
  each sample's ``<sample> FlowCode (Unmixed).fcs``.

For every sample it

1. takes the FlowCode-corrected, unmixed events;
2. gates them with the experiment's unmixed gates. Only the gate
   boundaries are taken from the experiment: each gate is rasterised onto
   its channels' transform bins (``functions.gate_lookup_tables``, the
   main window's own gating method) and the exported events are binned by
   their own values, so nothing computed on the main window's events is
   reused. Each sample's own gate overrides are used, and per-sample
   (Time) transforms take their range from the file, as in the main
   window;
3. counts events per FlowCode combination inside each selected population
   (each ticked gate on its own), and sums each combination's marker
   intensities on the display scale for the marker test.

A population can also be a phenotype (``flowcode_phenotype``): a marker
definition scored on the live-unmixed events inside the FlowCode gates,
where each event has a membership between 0 and 1 and the counts are the
memberships summed (soft counts), so ``events``, ``in_population`` and
``barcoded`` can be fractional and the count tables hold floats. Phenotype
populations need the live unmix, whose AF library gives the negative model;
exported files carry no such model.

``live_links`` lists the raw files the live unmix can use,
``link_exports`` matches exported files to samples, ``link_folders`` lists the
exported files of other experiments (other acquisition days) to count with them, ``SampleGating`` (taken on the
main thread with ``SampleGating.from_controller``) holds the gates and gates
each file with its own sample's gates, ``count_events`` gates and counts one
sample's events (``count_sample``: of an exported file), ``count_table`` assembles every sample, ``filter_counts`` applies the
inclusion rules and ``qc_table`` summarises each sample. ``count_populations``
counts several populations in one pass; ``units_table`` assembles them with
one row per population of each sample (``unit_id``), and the marker sums are
kept in the experiment's cache (``save_marker_sums``, ``load_marker_sums``).

Several libraries: the files' combinations are tag triplets, the same in
every file, and the counts are kept per triplet. Each sample's triplets are
named by its own library (``library_feature_map``, from the current library
setup) when the tables are made (``library_features``): a guide outside the
sample's library is NaN, not 0, and events on a triplet the library does
not use are counted as ``out_of_library``, outside ``barcoded``. Samples
assigned to Other, or to no library, have no guides and are left out of
the tables.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from honeychrome.controller_components.per_sample_gating_functions import SampleGating

logger = logging.getLogger(__name__)

COMBO_PREFIX = 'FC '
ID_CHANNEL = 'FlowCode ID'

# Link status of an experiment sample's exported file.
LINK_OK = 'ok'
LINK_MISSING = 'missing'
LINK_UNREADABLE = 'unreadable'
LINK_OTHER_COMBOS = 'different combinations'
LINK_OTHER_CHARACTERISATION = 'older characterisation'
LINK_NO_AF = 'no AF library'
LINK_OTHER_CHANNELS = 'different channels'
USABLE_BY_DEFAULT = (LINK_OK,)

# Sample key of a file linked from another experiment's export folder.
LINKED_PREFIX = 'linked:'

# Covariate filled in from the files' acquisition dates ($DATE).
DATE_COVARIATE = 'acquisition date'
DATE_FORMATS = ('%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%d.%m.%Y', '%d %b %Y', '%b %d, %Y')

# Where the page's events come from.
SOURCE_LIVE = 'live'
SOURCE_EXPORTED = 'exported'
SOURCES = (SOURCE_LIVE, SOURCE_EXPORTED)

DEFAULT_MIN_SAMPLE_EVENTS = 1
DEFAULT_MIN_COMBO_EVENTS = 20

# Pairing and split choices that are not covariates.
PAIR_BY_SAMPLE = '\x1fsample'
SPLIT_BY_GROUP = '\x1fgroup'
SPLIT_BY_LIBRARY = '\x1flibrary'

COUNT_COLUMNS = ('events', 'in_population', 'barcoded', 'untransduced', 'unresolved', 'not_barcoded')


ANALYSIS_DEFAULTS = {
    'source': SOURCE_LIVE,
    'folder': None,
    'linked_folders': [],
    'date_covariate': True,
    'gates': [],
    'phenotypes': [],
    'congenic_split': False,
    'exclude_combos': [],
    'min_combo_events': DEFAULT_MIN_COMBO_EVENTS,
    'min_sample_events': DEFAULT_MIN_SAMPLE_EVENTS,
    'use': {},
    'include': {'unstained': False, 'single_stain': False, 'beads': False},
    'group_names': [],
    'sample_groups': {},
    'covariates': {},
    'group_patterns': {},
    'group_colors': {},
    'pairing': '',
    'adjust': [],
    'split_by': '',
    'run_counts': True,
    'run_freq': True,
    'contrast_mode': 'reference',
    'reference': '',
    'pval_threshold': 0.05,
    'fc_threshold': 1.0,
    'use_treat': False,
    'fdr_scope': 'per_comparison',
    'counts_method': 'ql',
    'normalisation': 'robust',
    'freq_weights': 'capped',
    'control_combos': None,
    'remove_clones': False,
    'clone_min_count': 200,
    'clone_min_pct': 10.0,
    'clone_max_levels': 2,
    'heatmap_order': 'effect',
    'heatmap_rows': 1,
    'n_loadings': 10,
    'export_root': None,
    'counts': None,
    'compare': 'groups',
    'reference_population': '',
    'population_contrasts': False,
    'population_pairing': PAIR_BY_SAMPLE,
    'population_split': SPLIT_BY_GROUP,
    'run_markers': True,
    'marker_reference': 'tags',
    'marker_spillover': True,
    'marker_spillover_k': 'auto',
    'marker_channels': None,
    'marker_min_events': 10,
    'marker_threshold': 0.03,
    'run_pathways': True,
    'pathway_min_size': 3,
    'pathway_rotations': 9999,
    'pathway_evidence': 'all',
    'pathway_names': {},
    'run_gene_groups': True,
    'gene_group_table': [],
    'gene_group_file': '',
    'gene_group_areas': {},
    'gene_group_colours': {},
    'gene_group_order': [],
    'gene_group_comparisons': [],
    'shift': {},
}
PATHWAY_EVIDENCE = ('all', 'experimental')


def analysis_settings(stored: dict | None) -> dict:
    """The Analysis page's saved settings with defaults filled in."""
    out = deepcopy(ANALYSIS_DEFAULTS)
    stored = stored or {}
    for key, value in stored.items():
        if key in out:
            out[key] = deepcopy(value)
    if 'source' not in stored and (stored.get('counts') or {}).get('records'):
        out['source'] = SOURCE_EXPORTED
    if out['source'] not in SOURCES:
        out['source'] = SOURCE_LIVE
    folders = out['linked_folders']
    out['linked_folders'] = [str(f) for f in folders if f] if isinstance(folders, list) else []
    k = out['marker_spillover_k']
    if k != 'auto' and not (isinstance(k, (int, float)) and not isinstance(k, bool) and k >= 0):
        out['marker_spillover_k'] = 'auto'
    if out['freq_weights'] not in (None, 'capped'):
        out['freq_weights'] = ANALYSIS_DEFAULTS['freq_weights']
    if out['pathway_evidence'] not in PATHWAY_EVIDENCE:
        out['pathway_evidence'] = ANALYSIS_DEFAULTS['pathway_evidence']
    if not isinstance(out['pathway_names'], dict):
        out['pathway_names'] = {}
    rows = out['gene_group_table']
    out['gene_group_table'] = ([[str(r[0]), str(r[1])] for r in rows if isinstance(r, (list, tuple)) and len(r) >= 2]
                               if isinstance(rows, list) else [])
    for key in ('gene_group_areas', 'gene_group_colours'):
        if not isinstance(out[key], dict):
            out[key] = {}
    for key in ('gene_group_order', 'gene_group_comparisons'):
        if not isinstance(out[key], list):
            out[key] = []
    return out


def resolve_group_colours(settings: dict) -> bool:
    """Give every group in ``settings['group_names']`` a colour in
    ``settings['group_colors']``: the next glasbey colour not yet used, as
    Honeycluster assigns them. Returns True when a colour was added."""
    import colorcet as cc

    colours = settings.setdefault('group_colors', {})
    added = False
    for name in settings.get('group_names') or []:
        if colours.get(name):
            continue
        used = set(colours.values())
        colours[name] = next((c for c in cc.glasbey if c not in used),
                             cc.glasbey[len(colours) % len(cc.glasbey)])
        added = True
    return added


def assign_by_pattern(keys: list, names: dict, group_names: list, patterns: dict) -> tuple:
    """
    Groups from the groups' sample-name patterns: each sample takes the
    first group (in *group_names* order) whose regular expression matches
    its displayed name (``names[key]``), case-insensitive; a sample matching
    none has no group. Returns ({sample: group}, {group: n matched}) and
    raises ValueError naming the group whose pattern is not valid.
    """
    import re

    compiled = []
    for g in group_names:
        pattern = (patterns.get(g) or '').strip()
        if not pattern:
            continue
        try:
            compiled.append((g, re.compile(pattern, re.IGNORECASE)))
        except re.error as exc:
            raise ValueError(f'{g}: {exc}') from exc
    groups, matched = {}, {g: 0 for g, _rx in compiled}
    for k in keys:
        name = names.get(k, Path(str(k)).stem)
        for g, rx in compiled:
            if rx.search(name):
                groups[k] = g
                matched[g] += 1
                break
    return groups, matched


def _digest(value) -> str:
    import hashlib
    import json
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


def file_signature(path) -> list:
    """[path, size, modification time] of a file, or [path] if it is gone."""
    try:
        st = Path(path).stat()
        return [str(path), int(st.st_size), int(st.st_mtime_ns)]
    except OSError:
        return [str(path)]


def counts_signature(files: dict, gate_names: list, gating_signature: str, unmixing=None,
                     phenotypes: list | None = None) -> str:
    """Identifies a set of counts: the files ({sample: path}) as they are on
    disk, the chosen gates and the gating they were counted with, for a
    live unmix everything that decides the unmixed values (*unmixing*), and
    the phenotype populations as resolved and scored (*phenotypes*, their
    ``Resolved.signature_parts()``)."""
    parts = {'files': {k: file_signature(p) for k, p in sorted(files.items())},
             'gates': list(gate_names), 'gating': gating_signature}
    if len(gate_names) > 1:
        parts['populations'] = 'each gate'
    if unmixing is not None:
        parts['unmixing'] = unmixing
    if phenotypes:
        parts['phenotypes'] = list(phenotypes)
    return _digest(parts)


def live_links(sample_keys: list, experiment_dir, af_ok: dict) -> dict:
    """
    The raw files of *sample_keys* for the live unmix, in the shape
    ``link_exports`` returns: ``ok``, ``missing`` (no raw file) or
    ``no AF library`` (*af_ok* {key: bool}: an AF library of at least 2
    spectra, as section 5 needs).
    """
    files = {}
    for key in sample_keys:
        path = Path(experiment_dir) / str(key)
        if not path.exists():
            files[key] = {'path': None, 'status': LINK_MISSING, 'note': 'The raw file was not found.'}
        elif not af_ok.get(key):
            files[key] = {'path': None, 'status': LINK_NO_AF,
                          'note': 'Assign AF profiles in AutoSpectral > Autofluorescence.'}
        else:
            files[key] = {'path': str(path), 'status': LINK_OK, 'note': ''}
    return {'files': files, 'unlinked': []}


# ---------------------------------------------------------------------------
# Linking exported files to experiment samples
# ---------------------------------------------------------------------------

# ``FLOWCODECORRECTION`` value of files exported without FRET correction.
NO_CORRECTION_KEYWORD = 'NONE'


def read_fcs_keywords(path) -> dict:
    """TEXT keywords of an FCS file, keys lower case without '$'."""
    from flowio import FlowData

    data = FlowData(str(path), only_text=True)
    return {str(k).lstrip('$').lower(): v for k, v in data.text.items()}


def expected_export_path(folder, sample_key: str, raw_subdir: str, display_name: str,
                         suffix: str) -> Path:
    """Where section 5 writes *sample_key*'s file: the sample's folder under
    the raw samples folder, mirrored under *folder*, named
    ``<display name><suffix> (Unmixed).fcs``."""
    key = Path(str(sample_key))
    raw = Path(str(raw_subdir)) if raw_subdir else None
    try:
        rel = key.relative_to(raw) if raw is not None else key
    except ValueError:
        rel = key
    return Path(folder) / rel.parent / f'{display_name}{suffix} (Unmixed).fcs'


def link_exports(folder, sample_keys: list, raw_subdir: str, display_names: dict, suffix: str,
                 combos_keyword: str | None, characterisation: str | None,
                 sample_libraries: dict | None = None) -> dict:
    """
    Match the FCS files in *folder* (searched recursively) to experiment
    samples: first by the ``FLOWCODESOURCE`` keyword the export writes, then
    by the export's file naming. Each sample gets a status:

    * ``ok``                     — found, same combinations and characterisation
    * ``missing``                — no file
    * ``unreadable``             — the file's header could not be read
    * ``different combinations`` — ``FLOWCODECOMBOS`` differs from *combos_keyword*
    * ``older characterisation`` — ``FLOWCODECHARACTERISATION`` differs from
      *characterisation*

    Files written before the characterisation keyword existed are ``ok`` with
    a note. ``uncorrected`` is True for a file exported without FRET
    correction (``FLOWCODECORRECTION`` = ``NONE``). With several libraries
    (*sample_libraries*, {sample: library}), a file whose ``FLOWCODELIBRARY``
    differs from its sample's library gets a note and keeps its status: the
    counts are named by the current assignment. Returns
    ``{'files': {key: {'path', 'status', 'note', 'uncorrected'}},
    'unlinked': [paths of files that match no sample]}``.
    """
    folder = Path(folder)
    found = sorted(p for p in folder.rglob('*.fcs') if not p.name.startswith('._')) if folder.is_dir() else []
    keywords, unreadable = {}, {}
    for path in found:
        try:
            keywords[path] = read_fcs_keywords(path)
        except Exception as exc:
            unreadable[path] = str(exc) or type(exc).__name__
    by_source = {}
    for path, kw in keywords.items():
        source = kw.get('flowcodesource')
        if source:
            by_source.setdefault(str(source), path)
    by_name = {p.resolve(): p for p in found}

    files, used = {}, set()
    for key in sample_keys:
        path = by_source.get(str(key))
        if path is None:
            guess = expected_export_path(folder, key, raw_subdir, display_names.get(key, Path(key).stem), suffix)
            path = by_name.get(guess.resolve())
            if path is not None and path in keywords and keywords[path].get('flowcodesource') not in (None, str(key)):
                path = None
        if path is None:
            files[key] = {'path': None, 'status': LINK_MISSING, 'note': 'No exported file.'}
            continue
        used.add(path)
        if path in unreadable:
            files[key] = {'path': str(path), 'status': LINK_UNREADABLE, 'note': unreadable[path]}
            continue
        kw = keywords[path]
        status, note = LINK_OK, ''
        if combos_keyword is not None and kw.get('flowcodecombos', '') != combos_keyword:
            status, note = LINK_OTHER_COMBOS, 'Exported with a different combination table.'
        elif characterisation is not None:
            cid = kw.get('flowcodecharacterisation')
            if cid is None:
                note = 'Exported before characterisation records were kept.'
            elif cid != characterisation:
                status, note = LINK_OTHER_CHARACTERISATION, 'Exported with an earlier characterisation.'
        if sample_libraries is not None and kw.get('flowcodelibrary') not in (None, sample_libraries.get(key)):
            current = sample_libraries.get(key) or 'no library'
            note = ' '.join(filter(None, [note, f"Exported as {kw['flowcodelibrary']}; counted as {current}."]))
        files[key] = {'path': str(path), 'status': status, 'note': note,
                      'uncorrected': str(kw.get('flowcodecorrection', '')).strip().upper() == NO_CORRECTION_KEYWORD,
                      'date': acquisition_date(kw)}
    unlinked = [str(p) for p in found if p not in used]
    return {'files': files, 'unlinked': unlinked}


# ---------------------------------------------------------------------------
# Files of other experiments (other acquisition days)
# ---------------------------------------------------------------------------

def is_linked(key) -> bool:
    """True for a sample that is a file linked from another experiment."""
    return str(key).startswith(LINKED_PREFIX)


def linked_key(path) -> str:
    """Sample key of a linked file: its absolute path behind ``LINKED_PREFIX``."""
    return LINKED_PREFIX + Path(path).resolve().as_posix()


def acquisition_date(keywords: dict) -> str | None:
    """The ``$DATE`` keyword of an FCS file (*keywords* as ``read_fcs_keywords``
    gives them) as YYYY-MM-DD, or as written when no known format reads it;
    None when the file has no date."""
    raw = str(keywords.get('date') or '').strip()
    if not raw:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def file_dates(paths: dict) -> dict:
    """{sample: acquisition date} of the files *paths* ({sample: path});
    files that cannot be read or carry no date are left out."""
    out = {}
    for key, path in paths.items():
        try:
            date = acquisition_date(read_fcs_keywords(path))
        except Exception:
            continue
        if date:
            out[key] = date
    return out


def file_pnn(keywords: dict) -> list:
    """Channel names ($PnN) of an FCS file from its keywords."""
    try:
        n = int(keywords.get('par') or 0)
    except (TypeError, ValueError):
        n = 0
    return [str(keywords.get(f'p{i}n') or '') for i in range(1, n + 1)]


def linked_display_name(path, suffix: str) -> str:
    """The sample name of an exported file: its file name without the
    export's ``<suffix> (Unmixed)`` ending."""
    stem = Path(path).stem
    for tail in (f'{suffix} (Unmixed)', ' (Unmixed)'):
        if tail and stem.endswith(tail):
            return stem[:-len(tail)]
    return stem


def link_folders(folders, combos_keyword: str | None, required_channels, suffix: str, exclude=()) -> dict:
    """
    FlowCode exports of other experiments (e.g. the same cohort acquired on
    other days, each day processed in its own experiment), in the shape
    ``link_exports`` returns, keyed by ``linked_key``. Every FCS file under
    each folder (searched recursively) that has FlowCode combination channels
    is listed with a status:

    * ``ok``
    * ``unreadable``             — the file's header could not be read
    * ``different combinations`` — ``FLOWCODECOMBOS`` differs from *combos_keyword*
    * ``different channels``     — one of *required_channels* (this
      experiment's unmixed channels) is missing, e.g. another panel

    The characterisation is not checked: each experiment has its own. Each
    entry also holds 'name' (the file name without the export's ending),
    'date' (``acquisition_date``), 'folder' and 'library'
    (``FLOWCODELIBRARY``, or None). Files at the paths in *exclude* (this
    experiment's own exports) are skipped; FCS files without combination
    channels are returned in 'unlinked'.
    """
    exclude = {Path(p).resolve() for p in exclude}
    required = [c for c in required_channels if c]
    files, unlinked = {}, []
    for folder in folders:
        root = Path(folder)
        found = sorted(p for p in root.rglob('*.fcs') if not p.name.startswith('._')) if root.is_dir() else []
        for path in found:
            if path.resolve() in exclude:
                continue
            key = linked_key(path)
            if key in files:
                continue
            entry = {'path': str(path), 'status': LINK_OK, 'note': '', 'uncorrected': False,
                     'name': linked_display_name(path, suffix), 'date': None, 'folder': str(root), 'library': None}
            try:
                kw = read_fcs_keywords(path)
            except Exception as exc:
                entry.update(status=LINK_UNREADABLE, note=str(exc) or type(exc).__name__)
                files[key] = entry
                continue
            pnn = file_pnn(kw)
            if not combo_names_in_file(pnn):
                unlinked.append(str(path))
                continue
            entry['date'] = acquisition_date(kw)
            entry['library'] = kw.get('flowcodelibrary') or None
            entry['uncorrected'] = str(kw.get('flowcodecorrection', '')).strip().upper() == NO_CORRECTION_KEYWORD
            missing = [c for c in required if c not in pnn]
            if combos_keyword is not None and kw.get('flowcodecombos', '') != combos_keyword:
                entry.update(status=LINK_OTHER_COMBOS, note='Exported with a different combination table.')
            elif missing:
                more = f' and {len(missing) - 3} more' if len(missing) > 3 else ''
                entry.update(status=LINK_OTHER_CHANNELS,
                             note=f"Lacks this experiment's channel(s) {', '.join(missing[:3])}{more}.")
            else:
                entry['note'] = f"Linked; acquired {entry['date'] or 'on an unknown date'}."
            files[key] = entry
    return {'files': files, 'unlinked': unlinked}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def combo_names_in_file(pnn: list) -> list:
    """Combination names of a file's ``FC <name>`` channels, in file order
    (the order of the ``FlowCode ID`` values)."""
    return [ch[len(COMBO_PREFIX):] for ch in pnn if ch.startswith(COMBO_PREFIX)]


def event_combo_ids(events: np.ndarray, pnn: list) -> tuple:
    """
    Combination of each event: ``(ids, has_id_channel)`` with 0 = not
    barcoded, 1..n = the file's n-th ``FC`` channel, n + 1 = unresolved.
    Files with a ``FlowCode ID`` channel give untransduced (0) and unresolved
    (n + 1) apart; older files only tell barcoded events (the one non-zero
    ``FC`` channel) from the rest, which all read 0.
    """
    names = combo_names_in_file(pnn)
    if ID_CHANNEL in pnn:
        ids = np.rint(events[:, pnn.index(ID_CHANNEL)]).astype(np.int64)
        return ids, True
    cols = [pnn.index(COMBO_PREFIX + n) for n in names]
    if not cols:
        return np.zeros(len(events), dtype=np.int64), False
    nonzero = events[:, cols] != 0
    ids = np.where(nonzero.any(axis=1), np.argmax(nonzero, axis=1) + 1, 0).astype(np.int64)
    return ids, False


def count_sample(path, sample_key: str, snapshot: SampleGating, gate_names: list) -> dict:
    """Gate one exported file and count its events per combination (``count_events``)."""
    from honeychrome.controller_components.functions import sample_from_fcs

    sample = sample_from_fcs(str(path))
    pnn = list(sample.pnn_labels)
    events = np.asarray(sample.get_events(source='raw'), dtype=np.float64)
    if not combo_names_in_file(pnn):
        raise ValueError(f'{Path(path).name} has no FlowCode combination channels.')
    return count_events(events, pnn, sample_key, snapshot, gate_names)


def count_events(events: np.ndarray, pnn: list, sample_key: str, snapshot: SampleGating,
                 gate_names: list) -> dict:
    """
    Gate one sample's FlowCode-unmixed events (columns *pnn*, as an
    exported file's) and count them per combination.

    Returns ``{'counts': {combo name: n}, 'events', 'in_population',
    'barcoded', 'untransduced', 'unresolved', 'not_barcoded'}``;
    untransduced and unresolved are None for files without a
    ``FlowCode ID`` channel.
    """
    pnn = list(pnn)
    names = combo_names_in_file(pnn)
    if not names:
        raise ValueError(f'{sample_key}: no FlowCode combination channels.')
    mask = snapshot.population_mask(sample_key, events, pnn, gate_names)
    ids, has_id = event_combo_ids(events[mask], pnn)
    n = len(names)
    per = np.bincount(np.clip(ids, 0, n + 1), minlength=n + 2)
    barcoded = int(per[1:n + 1].sum())
    return {
        'counts': {name: int(per[i + 1]) for i, name in enumerate(names)},
        'events': int(len(events)),
        'in_population': int(mask.sum()),
        'barcoded': barcoded,
        'untransduced': int(per[0]) if has_id else None,
        'unresolved': int(per[n + 1]) if has_id else None,
        'not_barcoded': int(mask.sum()) - barcoded,
    }


def count_table(records: dict, combo_order: list | None = None) -> tuple:
    """
    ``(counts, meta)`` from ``{sample key: count_sample() record}``: counts
    is samples × combinations (combinations in *combo_order*, then any
    others in first-seen order; a combination missing from a file counts 0),
    integers unless a record holds fractional (soft) counts, meta is
    samples × COUNT_COLUMNS.
    """
    keys = list(records)
    names = list(combo_order or [])
    for rec in records.values():
        for name in rec['counts']:
            if name not in names:
                names.append(name)
    position = {name: j for j, name in enumerate(names)}
    values = np.zeros((len(keys), len(names)), dtype=np.float64)
    for i, rec in enumerate(records.values()):
        for name, n in rec['counts'].items():
            values[i, position[name]] = float(n)
    counts = pd.DataFrame(values, index=pd.Index(keys, name='sample'), columns=names)
    meta = pd.DataFrame([[rec.get(col) for col in COUNT_COLUMNS] for rec in records.values()],
                        index=pd.Index(keys, name='sample'), columns=list(COUNT_COLUMNS), dtype=object)
    if np.all(np.mod(counts.values, 1.0) == 0):
        counts = counts.astype(np.int64)
    return counts, meta


def whole_or_float(value, tolerance: float = 1e-6):
    """An integer when *value* is whole to within *tolerance* (sums of
    memberships carry rounding residuals), else the float (soft counts)."""
    value = float(value)
    nearest = round(value)
    return int(nearest) if abs(value - nearest) <= tolerance else value


# ---------------------------------------------------------------------------
# Several populations, and marker intensities per combination
# ---------------------------------------------------------------------------

UNIT_SEP = '\x1f'
ALL_EVENTS = 'All events'
MARKERS_FILE = 'marker_sums.npz'


def unit_id(sample_key: str, population: str, several: bool) -> str:
    """Row key of one population of one sample: the sample key when one
    population is analysed, ``<sample><UNIT_SEP><population>`` otherwise."""
    return f'{sample_key}{UNIT_SEP}{population}' if several else str(sample_key)


def unit_parts(unit: str, population: str = '') -> tuple:
    """(sample key, population) of a row key (*population* when the key
    is a bare sample key)."""
    sample, sep, pop = str(unit).partition(UNIT_SEP)
    return (sample, pop) if sep else (sample, population)


def count_populations(events: np.ndarray, pnn: list, sample_key: str, snapshot: SampleGating,
                      populations: list, marker_channels=(), phenotypes: dict | None = None,
                      gate_records=()) -> dict:
    """
    Gate one sample's FlowCode-unmixed events (columns *pnn*) and count them
    per combination in each population (each gate of *populations* on its
    own; no gate: every event, as population ``ALL_EVENTS``).

    A population named in *phenotypes* (``{name: (Resolved, NegativeModel)}``,
    ``flowcode_phenotype``) is scored instead of gated: each event inside
    the FlowCode gates (*gate_records*, raw-unit polygons on the scatter
    channels; every event without them) weighs its membership, and the
    record's counts are the memberships summed (soft counts, floats).

    Returns ``{population: record}`` with the ``count_events`` fields, plus
    'marker_sums': per combination (file order) and marker channel, the sum
    of the events' intensities on the experiment's display scale for that
    channel (None without *marker_channels*; membership-weighted for a
    phenotype), and 'marker_channels': the channels summed (those in *pnn*
    with a fixed transform).
    """
    pnn = list(pnn)
    names = combo_names_in_file(pnn)
    if not names:
        raise ValueError(f'{sample_key}: no FlowCode combination channels.')
    pops = list(populations) or [ALL_EVENTS]
    phenotypes = dict(phenotypes or {})
    gates = [p for p in pops if p != ALL_EVENTS and p not in phenotypes]
    membership = snapshot.memberships(sample_key, events, pnn, gates) if gates else {}
    ids, has_id = event_combo_ids(events, pnn)
    n = len(names)
    ids = np.clip(ids, 0, n + 1)
    channels = [ch for ch in marker_channels if ch in pnn and ch in snapshot.transforms
                and getattr(snapshot.transforms[ch], 'id', None) not in (None, 'default')]
    values = None
    if channels:
        values = np.empty((len(events), len(channels)), dtype=np.float64)
        for j, ch in enumerate(channels):
            values[:, j] = snapshot.transforms[ch].xform.apply(events[:, pnn.index(ch)])
    inside = None
    if phenotypes:
        import flowcode_core as fcc
        inside = fcc.gate_records_mask(events, pnn, list(gate_records)) if gate_records else None
        if inside is None:
            inside = np.ones(len(events), dtype=bool)
    out = {}
    for pop in pops:
        weights = None
        if pop == ALL_EVENTS:
            mask = np.ones(len(events), dtype=bool)
        elif pop in phenotypes:
            import flowcode_phenotype as fp
            resolved, model = phenotypes[pop]
            weights = fp.score_events(events, pnn, resolved, model)['membership'] * inside
            mask = weights > 0
        elif pop in membership:
            mask = np.asarray(membership[pop], dtype=bool)
        else:
            raise ValueError(f'Gate "{pop}" could not be applied: it does not exist, or one of its '
                             f'channels (or an ancestor gate\'s) is not in the data.')
        own = ids[mask]
        if weights is None:
            per = np.bincount(own, minlength=n + 2)
            in_population = int(mask.sum())
            number = int
        else:
            per = np.bincount(own, weights=weights[mask], minlength=n + 2)
            in_population = whole_or_float(weights.sum())
            number = whole_or_float
        barcoded = number(per[1:n + 1].sum())
        sums = None
        if values is not None:
            w = values[mask] if weights is None else values[mask] * weights[mask][:, np.newaxis]
            sums = np.stack([np.bincount(own, weights=w[:, j], minlength=n + 2)[1:n + 1]
                             for j in range(len(channels))], axis=1)
        out[pop] = {
            'counts': {name: number(per[i + 1]) for i, name in enumerate(names)},
            'events': int(len(events)),
            'in_population': in_population,
            'barcoded': barcoded,
            'untransduced': number(per[0]) if has_id else None,
            'unresolved': number(per[n + 1]) if has_id else None,
            'not_barcoded': number(in_population - barcoded),
            'marker_sums': sums,
            'marker_channels': channels,
        }
    return out


def read_exported_events(path) -> tuple:
    """(events, pnn) of an exported file; ValueError when it has no FlowCode
    combination channels."""
    from honeychrome.controller_components.functions import sample_from_fcs

    sample = sample_from_fcs(str(path))
    pnn = list(sample.pnn_labels)
    events = np.asarray(sample.get_events(source='raw'), dtype=np.float64)
    if not combo_names_in_file(pnn):
        raise ValueError(f'{Path(path).name} has no FlowCode combination channels.')
    return events, pnn


def count_sample_populations(path, sample_key: str, snapshot: SampleGating, populations: list,
                             marker_channels=()) -> dict:
    """``count_populations`` of an exported file."""
    events, pnn = read_exported_events(path)
    return count_populations(events, pnn, sample_key, snapshot, populations, marker_channels)


def saved_records(saved: dict) -> tuple:
    """``(populations, {population: {sample: record}})`` of saved counts;
    counts saved before several populations were counted hold one
    population (their union of gates)."""
    saved = saved or {}
    records = saved.get('records') or {}
    if 'populations' in saved:
        pops = list(saved['populations'])
        return pops, {p: dict(records.get(p) or {}) for p in pops}
    if not records:
        return [], {}
    pop = saved.get('population') or ALL_EVENTS
    return [pop], {pop: dict(records)}


def units_table(records_by_pop: dict, populations: list, combo_order: list | None = None) -> tuple:
    """
    ``(counts, meta)`` over every population of every sample: rows are
    ``unit_id`` keys (bare sample keys for one population), combinations as
    in ``count_table``; meta adds 'sample' and 'population' to
    COUNT_COLUMNS.
    """
    several = len(populations) > 1
    flat, origin = {}, {}
    for pop in populations:
        for key, rec in (records_by_pop.get(pop) or {}).items():
            unit = unit_id(key, pop, several)
            flat[unit] = rec
            origin[unit] = (key, pop)
    counts, meta = count_table(flat, combo_order)
    meta['sample'] = [origin[u][0] for u in meta.index]
    meta['population'] = [origin[u][1] for u in meta.index]
    return counts, meta


# ---------------------------------------------------------------------------
# Several libraries
# ---------------------------------------------------------------------------

def library_feature_map(state) -> dict:
    """
    {library: {triplet key: feature}} from the current library setup
    (*state*, a ``FlowCodeState`` with several libraries): every included
    combination row, named by its triplet, gives each library that uses it
    that library's guide. With ``merge_same_names`` a guide name used by
    several libraries is one feature; otherwise each of those libraries'
    guides is ``<guide> (<library label>)``.
    """
    from flowcode_state import library_label, shared_guide_names

    shared = set() if state.merge_same_names else set(shared_guide_names(state.libraries))
    labels = {str(lib.get('name', '')): library_label(lib) for lib in state.libraries}
    out = {name: {} for name in labels}
    for row in state.included_combos():
        for lib, guide in (row.get('names') or {}).items():
            if lib in out:
                out[lib][row['name']] = f'{guide} ({labels[lib]})' if guide in shared else str(guide)
    return out


def library_features(counts: pd.DataFrame, meta: pd.DataFrame, sample_library: dict, feature_map: dict) -> tuple:
    """
    Name the triplet counts by each sample's library.

    *counts* is units × triplets and *meta* its ``units_table`` meta;
    *sample_library* is {sample: library}; *feature_map* is
    ``library_feature_map``. Returns (counts, meta, left_out): counts is
    units × features (every library's, library by library), NaN where the
    feature is not in the unit's library; meta adds 'library' and
    'out_of_library' (events on triplets the library does not use), and its
    'barcoded' counts the library's triplets only; left_out is {sample:
    library or ''} of the samples without a library in *feature_map* (Other,
    unassigned, or a library since removed), which are left out.
    """
    features = list(dict.fromkeys(f for fmap in feature_map.values() for f in fmap.values()))
    sample_of = meta['sample'].to_dict()
    keep, left_out = [], {}
    for unit in counts.index:
        sample = sample_of[unit]
        lib = sample_library.get(sample)
        if lib in feature_map:
            keep.append(unit)
        else:
            left_out[sample] = lib or ''
    feature_col = {f: j for j, f in enumerate(features)}
    triplet_col = {t: j for j, t in enumerate(counts.columns)}
    row_of = {u: i for i, u in enumerate(counts.index)}
    raw = counts.to_numpy(dtype=np.float64)
    values_out = np.full((len(keep), len(features)), np.nan)
    libraries, outside, barcoded = [], [], []
    for r, unit in enumerate(keep):
        lib = sample_library[sample_of[unit]]
        row = raw[row_of[unit]]
        values = {}
        for triplet, feature in feature_map[lib].items():
            j = triplet_col.get(triplet)
            values[feature] = values.get(feature, 0.0) + (float(row[j]) if j is not None else 0.0)
        for feature, value in values.items():
            values_out[r, feature_col[feature]] = value
        inside = whole_or_float(sum(values.values()))
        libraries.append(lib)
        outside.append(whole_or_float(row.sum() - inside))
        barcoded.append(inside)
    out = pd.DataFrame(values_out, index=pd.Index(keep, name=counts.index.name), columns=features)
    new_meta = meta.loc[keep].copy()
    new_meta['library'] = ''
    new_meta.loc[:, 'library'] = libraries
    new_meta['out_of_library'] = pd.Series(outside, index=new_meta.index, dtype=object)
    new_meta['barcoded'] = pd.Series(barcoded, index=new_meta.index, dtype=object)
    return out, new_meta, left_out


def marker_cache_path(experiment_dir) -> Path:
    return Path(experiment_dir) / 'cache' / 'flowcode_analysis' / MARKERS_FILE


def save_marker_sums(path, signature: str, records_by_pop: dict, populations: list):
    """
    Write the marker sums of fresh counts (``count_populations`` records)
    to *path*, keyed by the counts' *signature*, and drop them from the
    records (they are too large to keep with the experiment's settings).
    """
    several = len(populations) > 1
    units, combos, channels, blocks = [], [], [], []
    for pop in populations:
        for key, rec in (records_by_pop.get(pop) or {}).items():
            sums = rec.pop('marker_sums', None)
            chans = rec.pop('marker_channels', None) or []
            if sums is None or not chans:
                continue
            if not channels:
                channels = list(chans)
            if list(chans) != channels:
                continue
            units.append(unit_id(key, pop, several))
            combos.append(list(rec['counts']))
            blocks.append(np.asarray(sums, dtype=np.float64))
    if not units:
        return False
    names = list(dict.fromkeys(c for cs in combos for c in cs))
    full = np.zeros((len(units), len(names), len(channels)), dtype=np.float64)
    for i, (cs, block) in enumerate(zip(combos, blocks)):
        full[i, [names.index(c) for c in cs]] = block
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=np.array(signature), units=np.array(units, dtype=object),
                        combos=np.array(names, dtype=object), channels=np.array(channels, dtype=object),
                        sums=full)
    return True


def load_marker_sums(path, signature: str | None) -> dict | None:
    """{'units', 'combos', 'channels', 'sums'} saved for counts with
    *signature*, or None when there are none (or they belong to other
    counts)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if signature is not None and str(data['signature']) != signature:
                return None
            return {'units': [str(u) for u in data['units']], 'combos': [str(c) for c in data['combos']],
                    'channels': [str(c) for c in data['channels']], 'sums': np.array(data['sums'])}
    except Exception:
        logger.exception('Could not read %s', path)
        return None


def marker_sums_for(saved: dict, units: list, combos: list, channels: list) -> np.ndarray:
    """(units, combos, channels) sums from ``load_marker_sums``, zeros where
    a unit, combination or channel is missing."""
    out = np.zeros((len(units), len(combos), len(channels)), dtype=np.float64)
    u_at = {u: i for i, u in enumerate(saved['units'])}
    c_at = {c: i for i, c in enumerate(saved['combos'])}
    k_at = {k: i for i, k in enumerate(saved['channels'])}
    ci = [(j, c_at[c]) for j, c in enumerate(combos) if c in c_at]
    ki = [(j, k_at[k]) for j, k in enumerate(channels) if k in k_at]
    for i, u in enumerate(units):
        src = u_at.get(u)
        if src is None:
            continue
        for j, cj in ci:
            out[i, j, [a for a, _b in ki]] = saved['sums'][src, cj, [b for _a, b in ki]]
    return out


@dataclass
class FilteredCounts:
    """Counts after the inclusion rules: ``counts`` (samples × combinations),
    ``denominator`` (barcoded events over the kept combinations, per kept
    sample), and what was dropped and why."""
    counts: pd.DataFrame
    denominator: pd.Series
    excluded_combos: list
    rare_combos: dict
    small_samples: dict
    notes: list


def filter_counts(counts: pd.DataFrame, exclude_combos=(), min_combo_events: int = DEFAULT_MIN_COMBO_EVENTS,
                  min_sample_events: int = DEFAULT_MIN_SAMPLE_EVENTS) -> FilteredCounts:
    """
    Apply the inclusion rules, in order:

    1. combinations the user left out are dropped;
    2. combinations with fewer than *min_combo_events* events summed over all
       samples are dropped (barcodes that failed or were barely transduced);
    3. samples with fewer than *min_sample_events* barcoded events over the
       remaining combinations are dropped.

    The denominator of each remaining sample is its barcoded events over the
    remaining combinations, so frequencies are among transduced cells of the
    population.
    """
    exclude = [c for c in exclude_combos if c in counts.columns]
    kept = counts.drop(columns=exclude)
    totals = kept.sum(axis=0)
    rare = {c: int(totals[c]) for c in kept.columns if totals[c] < min_combo_events}
    kept = kept.drop(columns=list(rare))
    per_sample = kept.sum(axis=1)
    small = {s: int(per_sample[s]) for s in kept.index if per_sample[s] < min_sample_events}
    kept = kept.drop(index=list(small))
    notes = []
    if exclude:
        notes.append(f"Combinations left out by choice: {', '.join(exclude)}.")
    if rare:
        notes.append(f"Combinations with fewer than {min_combo_events} events in all samples together "
                     "were left out: " + ', '.join(f'{c} ({n})' for c, n in rare.items()) + '.')
    if small:
        notes.append(f"Samples with fewer than {min_sample_events} barcoded events were left out: "
                     + ', '.join(f'{Path(str(s)).stem} ({n})' for s, n in small.items()) + '.')
    return FilteredCounts(counts=kept, denominator=kept.sum(axis=1).astype(float),
                          excluded_combos=exclude, rare_combos=rare, small_samples=small, notes=notes)


# ---------------------------------------------------------------------------
# Per-sample QC
# ---------------------------------------------------------------------------

def gini(values) -> float:
    """Gini coefficient of non-negative values (0 equal, towards 1 one value
    holds everything); R ``ineq::Gini`` without small-sample correction."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return float('nan')
    ranks = np.arange(1, n + 1)
    return float(2.0 * np.sum(ranks * x) / (n * total) - (n + 1.0) / n)


def qc_table(counts: pd.DataFrame, meta: pd.DataFrame, display_names: dict | None = None) -> pd.DataFrame:
    """
    One row per sample: events in the file and the population, barcoded
    events and their share of the population, untransduced and unresolved
    events (new exports only), the Gini coefficient of the barcode
    distribution and the most frequent combination with its share.
    High Gini or a dominant combination points to clonal expansion. With
    several libraries (``library_features``) the library and the events
    outside it follow, and combinations outside the sample's library (NaN)
    take no part.
    """
    rows = []
    several = 'library' in meta.columns
    for key in counts.index:
        row = counts.loc[key].astype(float).dropna()
        barcoded = float(row.sum())
        top = row.idxmax() if barcoded > 0 else ''
        pop = meta.loc[key, 'in_population']
        out = {
            'sample': key,
            'name': (display_names or {}).get(key, Path(str(key)).stem),
            'events': meta.loc[key, 'events'],
            'in_population': pop,
            'barcoded': whole_or_float(barcoded),
            'pct_barcoded': 100.0 * barcoded / pop if pop else float('nan'),
            'untransduced': meta.loc[key, 'untransduced'],
            'unresolved': meta.loc[key, 'unresolved'],
            'gini': gini(row.values),
            'top_combination': top,
            'top_share_pct': 100.0 * float(row.max()) / barcoded if barcoded > 0 else float('nan'),
        }
        if several:
            out.update(library=meta.loc[key, 'library'], out_of_library=meta.loc[key, 'out_of_library'])
        rows.append(out)
    return pd.DataFrame(rows).set_index('sample')
