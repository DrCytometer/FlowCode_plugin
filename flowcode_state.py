"""
flowcode_state.py
-----------------
Experiment state of the FlowCode plugin, and the input rules of its first two
sections. No Qt.

``FlowCodeState`` holds everything the user chooses (backbone, matched
unstained, AF profiles, tags, combos, gates, thresholds, settings) and is
stored as JSON in ``experiment.process['flowcode_unmix']``. Arrays produced
by later sections live in the experiment's cache folder.

Sections that run a computation (3: gates and thresholds, 4: debarcode and
FRET, 5: unmix and export) record a signature of their inputs when they
finish. A section is done while its recorded signature matches the current
inputs; otherwise it needs a re-run, and the differing inputs say why.

Several libraries: when one experiment holds samples transduced with
different guide libraries on the same tags, each library has its own
combinations CSV and each sample is assigned to one library (or to Other,
e.g. an FMO). Debarcoding works on tag triplets, so the design is the union
of every library's triplets, each combination named by its triplet
(``AU1_C_FLAG``); a library only names the guide a triplet carries. Guide
names in the combination table are read-only and come from the CSVs.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PLUGIN_VERSION = '0.4.1'
PROCESS_KEY = 'flowcode_unmix'

SINGLETS_GATE = 'FlowCode Singlets'
CELLS_GATE = 'FlowCode Cells'
GATE_NAMES = (SINGLETS_GATE, CELLS_GATE)

SECTION_TITLES = {
    1: 'Inputs',
    2: 'Tags & Combos',
    3: 'Gates & Thresholds',
    4: 'Debarcode & FRET',
    5: 'Unmix & Export',
    6: 'Analysis',
}

MIN_TAGS = 3
RECOMMENDED_TAGS = 4

# Shown wherever results come from samples debarcoded without FRET correction.
NO_FRET_WARNING = 'WARNING: no spillover performed! Results invalid!'

DEFAULT_SETTINGS = {
    # section 1
    'match_max_events': 50_000,
    'match_bins': 64,
    # section 3
    'backbone_max_events': 2_000_000,
    'unstained_max_events': 100_000,
    'singlet_quantile': 0.85,
    'landmark_n_per_tag': 100,
    'landmark_percentile': 0.70,
    'landmark_min_events': 50,
    'landmark_expand': 1.5,
    'threshold_max_events': 50_000,
    'threshold_min_delta_bic': 10.0,
    'threshold_display_max_events': 200_000,
    'cell_weight': None,                # in the section 3 run record only
    'debarcode_method': 'r1',           # 'r1' (per-cell AF) or 'experiment' (the experiment's method)
    # section 4
    'rescue_toomany': True,
    'rescue_flat_threshold': 0.9,
    'rescue_gate_frac': 0.5,
    'rescue_toofew': True,
    'rescue_toofew_flat_threshold': 0.9,
    'rescue_toofew_gate_frac': 0.5,
    'rescue_toofew_min_z': 2.0,
    'rescue_toofew_min_frac': 0.25,
    'rescue_toofew_af_ratio': 1.0,
    'rescue_toofew_n0': 50.0,
    'fit_min_fold': 3.0,
    'min_cells_per_combo': 50,
    'spill_ridge': 0.0,
    'expression_max_events': 20_000,
    'characterise_display_max_events': 200_000,
    # section 5
    'tag_shift_min_n': 100,
    'correction': 'combo',
    'regressor': 'fixed',
    'export_include_controls': False,
    'tandem_tags': None,
    'tandem_z_min': 3.0,
    'tandem_min_combos': 5,
    'tandem_max_events': 2_000_000,
    'calib_combos': True,
    'calib_brightness': False,
    'calib_z_min': 3.0,
    'calib_min_share': 0.05,
    'calib_min_events': 40,
}

SECTION3_SETTINGS = ('backbone_max_events', 'unstained_max_events', 'cell_weight')
SECTION4_SETTINGS = ('rescue_toomany', 'rescue_flat_threshold', 'rescue_gate_frac', 'rescue_toofew',
                     'rescue_toofew_flat_threshold', 'rescue_toofew_gate_frac', 'rescue_toofew_min_z',
                     'rescue_toofew_min_frac', 'rescue_toofew_af_ratio', 'rescue_toofew_n0', 'fit_min_fold',
                     'min_cells_per_combo', 'spill_ridge')
SECTION5_SETTINGS = ('tag_shift_min_n', 'correction', 'regressor', 'tandem_tags', 'tandem_z_min',
                     'tandem_min_combos', 'tandem_max_events')
# Per-sample combination calibration (section 5): a run input of its own, so
# that exports made before it existed say why they need a re-run.
CALIBRATION_SETTINGS = ('calib_combos', 'calib_brightness', 'calib_z_min', 'calib_min_share', 'calib_min_events')

# Final-unmix settings of earlier versions; the experiment's method (AutoSpectral tab) replaces them.
RETIRED_SETTINGS = ('use_variants', 'n_passes', 'n_af_passes', 'refine_af_quantile', 'noise_floor', 'alpha',
                    'collinear_threshold', 'joint_pair_resolution', 'threads')

_REASONS = {
    'backbone': 'backbone sample changed',
    'unstained': 'matched unstained changed',
    'af_profiles': 'AF profiles changed',
    'tags': 'tag selection changed',
    'combos': 'combinations changed',
    'combo_names': 'combo names changed',
    'fingerprint': 'spectral model changed',
    'settings3': 'section 3 settings changed',
    'gates': 'gates changed',
    'thresholds': 'thresholds changed',
    'settings4': 'section 4 settings changed',
    'settings5': 'section 5 settings changed',
    'calibration': 'combination calibration settings changed',
    'samples': 'sample selection changed',
    'output': 'output folder changed',
    'debarcode': 'debarcoding unmix changed',
    'method': 'unmixing method changed',
    'no_fret': 'FRET correction switched on or off',
    'libraries': 'libraries or library assignments changed',
}

# Library of a sample that belongs to none (e.g. an FMO).
OTHER_LIBRARY = 'Other'
LIBRARY_PREFIX = 'Library'


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, default=_json_default)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serialisable: {type(value).__name__}')


def model_fingerprint(reference_spectra, fluor_names, tag_fluors, detectors) -> str:
    """Fingerprint of the spectral model as FlowCode uses it: the full
    reference spectra, the tag fluorophores' rows and the detector list."""
    spectra = np.round(np.asarray(reference_spectra, dtype=np.float64), 8)
    h = hashlib.sha1(spectra.tobytes())
    names = list(fluor_names)
    for fluor in tag_fluors:
        h.update(str(fluor).encode('utf-8'))
        if fluor in names:
            h.update(spectra[names.index(fluor)].tobytes())
    h.update('|'.join(map(str, detectors)).encode('utf-8'))
    return h.hexdigest()


def _backbone_digest_value(keys: list):
    """A single backbone digests as its key, a pool as its sorted keys."""
    if not keys:
        return None
    return keys[0] if len(keys) == 1 else sorted(keys)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class FlowCodeState:
    """User choices and run records for one experiment."""
    backbone_keys: list | None = None
    unstained_key: str | None = None
    unstained_auto: bool = True
    match_backbone_keys: list | None = None
    unstained_scores: dict = field(default_factory=dict)
    af_profile_names: list = field(default_factory=list)
    af_auto: bool = True
    combos_csv_path: str | None = None
    csv_combos: list | None = None
    tag_selection: list | None = None
    tag_names: dict = field(default_factory=dict)
    combos: list = field(default_factory=list)
    # Several libraries: [{'name', 'label', 'pattern', 'csv_path', 'combos'}]
    # and {sample key: library name or OTHER_LIBRARY}.
    multi_library: bool = False
    libraries: list = field(default_factory=list)
    sample_libraries: dict = field(default_factory=dict)
    merge_same_names: bool = True
    gates: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    y_channels: dict = field(default_factory=dict)
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))
    output_folder: str | None = None
    export_samples: list | None = None
    last_export: dict | None = None
    characterisation: dict | None = None
    backbone_summary: dict | None = None
    fingerprint: str | None = None
    completed: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    # Debarcode without FRET correction (no FRET backbone), and when the user
    # acknowledged that its results are invalid.
    no_fret: bool = False
    no_fret_acknowledged: str | None = None
    plugin_version: str = PLUGIN_VERSION
    # Set by the plugin from the experiment's unmixing method, not saved:
    # 'token' (the method) and 'final' (the method and the export samples' spectra sets).
    unmixing_inputs: dict = field(default_factory=dict, repr=False)

    # -- persistence ---------------------------------------------------------

    def to_json(self) -> dict:
        data = deepcopy(vars(self))
        data.pop('unmixing_inputs', None)
        data['plugin_version'] = PLUGIN_VERSION
        return json.loads(json.dumps(data, default=_json_default))

    @classmethod
    def from_json(cls, data: dict | None) -> 'FlowCodeState':
        state = cls()
        if not data:
            return state
        for key in vars(state):
            if key in data and key != 'unmixing_inputs':
                setattr(state, key, deepcopy(data[key]))
        # states written with a single backbone
        for old, new in (('backbone_key', 'backbone_keys'), ('match_backbone_key', 'match_backbone_keys')):
            if new not in data and data.get(old):
                setattr(state, new, [data[old]])
        merged = dict(DEFAULT_SETTINGS)
        merged.update(state.settings or {})
        # settings of the earlier error model
        merged.pop('af_fret_term', None)
        for key in RETIRED_SETTINGS:
            merged.pop(key, None)
        if merged.get('correction') == 'global':
            merged['correction'] = 'pairwise'
        state.settings = merged
        state.completed = {str(k): v for k, v in (state.completed or {}).items()}
        return state

    # -- backbone ------------------------------------------------------------

    def backbones(self) -> list:
        """Chosen backbone sample keys (pooled), in experiment order."""
        return list(self.backbone_keys or [])

    # -- tags and combos -----------------------------------------------------

    def tag_channels(self) -> list:
        """Ticked tag channels, in selection order."""
        return list(self.tag_selection or [])

    def tag_map(self) -> dict:
        """{tag name: channel} for the ticked channels."""
        return {self.tag_names.get(ch, ch): ch for ch in self.tag_channels()}

    def included_combos(self) -> list:
        return [row for row in self.combos if row.get('include', True)]

    def combo_triplets(self) -> list:
        """Sorted tag triplets of the included combos (names excluded)."""
        return sorted(tuple(sorted(str(t).upper() for t in row['tags'])) for row in self.included_combos())

    # -- libraries -------------------------------------------------------------

    def enable_libraries(self):
        """Switch to several libraries. The first time, Library 1 takes the
        section 1 combinations CSV and an empty Library 2 is added."""
        self.multi_library = True
        if not self.libraries:
            first = new_library([], self.combos_csv_path, self.csv_combos)
            self.libraries = [first, new_library([first])]

    def library_names(self) -> list:
        return [str(lib.get('name', '')) for lib in self.libraries]

    def library_csv_combos(self) -> list:
        """Every library's combos ([guide, tags]), in library order."""
        return [combo for lib in self.libraries for combo in lib.get('combos') or []]

    def csv_source(self) -> list | None:
        """The CSV combos that set the tags and the combination rows: every
        library's with several libraries, else the section 1 CSV's."""
        if self.multi_library:
            return self.library_csv_combos() or None
        return self.csv_combos

    def library_of(self, sample_key: str) -> str | None:
        """The sample's library name, OTHER_LIBRARY, or None when unassigned
        or assigned to a library that no longer exists."""
        name = self.sample_libraries.get(sample_key)
        if name == OTHER_LIBRARY or name in set(self.library_names()):
            return name
        return None

    # -- run records ---------------------------------------------------------

    def section_inputs(self, section: int, fingerprint: str | None) -> dict:
        """Digest of each input a run-based section depends on."""
        tags = [[ch, self.tag_names.get(ch, ch)] for ch in self.tag_channels()]
        inputs = {
            'backbone': _digest(_backbone_digest_value(self.backbones())),
            'unstained': _digest(self.unstained_key),
            'af_profiles': _digest(sorted(self.af_profile_names)),
            'tags': _digest(tags),
            'combos': _digest(self.combo_triplets()),
            'fingerprint': _digest(fingerprint),
            'settings3': _digest({k: self.settings.get(k) for k in SECTION3_SETTINGS}),
            'debarcode': (None if self.settings.get('debarcode_method', 'r1') == 'r1'
                          else _digest(['experiment', self.unmixing_inputs.get('token')])),
        }
        if section == 3:
            inputs['gates'] = _digest({name: self.gates.get(name, {}).get('vertices_raw')
                                       for name in GATE_NAMES})
            return inputs
        if section >= 4:
            inputs.update(self.section_inputs(3, fingerprint))
            inputs['thresholds'] = _digest({t: v.get('raw') for t, v in sorted(self.thresholds.items())})
            inputs['settings4'] = _digest({k: self.settings.get(k) for k in SECTION4_SETTINGS})
            inputs['no_fret'] = _digest('no FRET correction') if self.no_fret else None
        if section >= 5:
            inputs['combo_names'] = _digest([[row['name'], sorted(row['tags'])]
                                             for row in self.included_combos()])
            inputs['settings5'] = _digest({k: self.settings.get(k) for k in SECTION5_SETTINGS})
            inputs['calibration'] = _digest({k: self.settings.get(k) for k in CALIBRATION_SETTINGS})
            inputs['samples'] = _digest(self.export_samples) if not self.settings.get('export_include_controls') \
                else _digest([self.export_samples, 'controls'])
            inputs['output'] = _digest(self.output_folder)
            inputs['method'] = _digest(self.unmixing_inputs.get('final'))
            inputs['libraries'] = (_digest([[lib.get('name'), lib.get('label'), lib.get('combos')]
                                            for lib in self.libraries] + [self.sample_libraries])
                                   if self.multi_library else None)
        return inputs

    def combo_backbone_counts(self) -> dict:
        """{combo name: backbone events} from the last characterisation,
        matched to the current rows by tag set."""
        from flowcode_core import combo_key

        per_key = (self.backbone_summary or {}).get('per_combo') or {}
        return {row['name']: per_key[combo_key(row['tags'])] for row in self.combos
                if combo_key(row['tags']) in per_key}

    def mark_completed(self, section: int, fingerprint: str | None):
        self.completed[str(section)] = self.section_inputs(section, fingerprint)
        if section == 3:
            self.fingerprint = fingerprint

    def clear_completed(self, section: int):
        self.completed.pop(str(section), None)

    def run_status(self, section: int, fingerprint: str | None) -> tuple:
        """('done', ''), ('stale', reasons) or ('pending', '')."""
        recorded = self.completed.get(str(section))
        if not recorded:
            return 'pending', ''
        current = self.section_inputs(section, fingerprint)
        changed = [key for key in current if recorded.get(key) != current[key]]
        if not changed:
            return 'done', ''
        return 'stale', ', '.join('combination calibration added' if key == 'calibration' and key not in recorded
                                  else _REASONS.get(key, key) for key in changed)


# ---------------------------------------------------------------------------
# Section 1: inputs
# ---------------------------------------------------------------------------

def _is_bead(samples: dict, key: str, name: str) -> bool:
    """In the experiment's bead_samples list; by name in Honeychrome versions without it."""
    beads = samples.get('bead_samples')
    if beads is not None:
        return key in beads
    return bool(re.search(r'bead', name, re.IGNORECASE) or re.search(r'bead', key, re.IGNORECASE))


def _is_unstained(key: str, name: str, tagged: set) -> bool:
    return key in tagged or 'unstained' in key.lower() or 'unstained' in str(name).lower()


# AutoSpectral naming of single-stain control files: "<well> <marker> <fluor> (Cells|Beads)"
_CONTROL_NAME = re.compile(r'\((cells|beads)\)', re.IGNORECASE)


def unstained_candidates(samples: dict) -> list:
    """
    Keys of unstained cell samples, in experiment order: tagged as unstained
    or with 'unstained' in the path or name, excluding beads (the AutoSpectral
    Optimization rules).
    """
    all_samples = samples.get('all_samples') or {}
    tagged = set(samples.get('unstained_samples') or [])
    return [key for key, name in all_samples.items()
            if _is_unstained(key, name, tagged) and not _is_bead(samples, key, str(name))]


def control_sample_keys(samples: dict) -> set:
    """
    Keys of single-stain controls and unstained samples (beads included).

    A sample is a single-stain control when it is tagged as one, when its
    file name is that of a tagged control (a copy of the reference group
    elsewhere in the raw folder), or when it is named as AutoSpectral names
    controls, "(Cells)" or "(Beads)" at the end. It is unstained when tagged
    as unstained or 'unstained' is in its path or name.
    """
    all_samples = samples.get('all_samples') or {}
    controls = set(samples.get('single_stain_controls') or [])
    control_files = {Path(key).name.lower() for key in controls}
    unstained = set(samples.get('unstained_samples') or [])
    out = set()
    for key, name in all_samples.items():
        if key in controls or Path(key).name.lower() in control_files \
                or _CONTROL_NAME.search(Path(key).stem) or _CONTROL_NAME.search(str(name)) \
                or _is_unstained(key, name, unstained):
            out.add(key)
    return out


def backbone_candidates(samples: dict) -> list:
    """Every sample key except single-stain controls and unstained samples,
    in experiment order."""
    excluded = control_sample_keys(samples)
    return [key for key in (samples.get('all_samples') or {}) if key not in excluded]


_BACKBONE_NAME = re.compile(r'backbone|fret', re.IGNORECASE)


def default_backbone_keys(samples: dict) -> list:
    """Backbone candidates with 'backbone' or 'FRET' in the file or sample
    name (case-insensitive), in experiment order."""
    all_samples = samples.get('all_samples') or {}
    return [key for key in backbone_candidates(samples)
            if _BACKBONE_NAME.search(Path(key).name) or _BACKBONE_NAME.search(str(all_samples.get(key, '')))]


def _same_sample(source: str, key: str, experiment_dir) -> bool:
    if not source or not key:
        return False
    src = Path(source)
    if src.is_absolute() and experiment_dir is not None:
        try:
            src = src.relative_to(Path(experiment_dir))
        except ValueError:
            return False
    return src.as_posix() == Path(key).as_posix()


def af_profiles_for_sample(af_profiles: dict, sample_key: str, experiment_dir=None) -> list:
    """Names of AF profiles extracted from *sample_key*, in stored order."""
    return [name for name, entry in (af_profiles or {}).items()
            if _same_sample(str(entry.get('source_fcs') or ''), sample_key, experiment_dir)]


def af_library_problem(af_profiles: dict, names, detectors, unstained_name: str = '') -> tuple:
    """
    Check the AF library chosen for debarcoding.

    Returns (number of spectra, problem message or None).
    """
    names = [n for n in (names or []) if n in (af_profiles or {})]
    hint = (f"No AF profile extracted from '{unstained_name}'. Open the Autofluorescence page of the AutoSpectral tab, select "
            f"this sample in Step 1 (Extract AF Profile), and run extraction.")
    if not names:
        return 0, hint
    n_spectra = 0
    for name in names:
        entry = af_profiles[name]
        channels = entry.get('channel_names')
        if channels is not None and list(channels) != list(detectors):
            return 0, (f"AF profile '{name}' was extracted with different detectors. "
                       f"Re-extract it on the Autofluorescence page of the AutoSpectral tab.")
        n_spectra += len(entry.get('spectra') or [])
    if n_spectra < 2:
        return n_spectra, hint
    return n_spectra, None


def af_library_spectra(af_profiles: dict, names) -> np.ndarray:
    """Combined spectra of the named profiles (rows stacked in name order)."""
    rows = [np.asarray(af_profiles[n]['spectra'], dtype=float) for n in names if n in af_profiles]
    return np.vstack(rows) if rows else np.zeros((0, 0))


def export_af_profile_names(assigned, debarcode_names) -> list:
    """AF profiles a sample is exported with: its AutoSpectral AF assignment,
    then the profiles the thresholds were set with (section 3) that the
    assignment lacks."""
    names = [str(n) for n in (assigned or [])]
    return names + [str(n) for n in (debarcode_names or []) if str(n) not in names]


def export_af_assignments(sample_af_profiles: dict, keys, debarcode_names) -> dict:
    """{sample key: ``export_af_profile_names``} for *keys*, other entries kept."""
    out = {k: list(v or []) for k, v in (sample_af_profiles or {}).items()}
    for key in keys:
        out[key] = export_af_profile_names(out.get(key), debarcode_names)
    return out


def af_assignment_gaps(sample_af_profiles: dict, keys, debarcode_names, af_profiles: dict) -> list:
    """Keys whose AutoSpectral AF assignment lacks a stored profile the
    thresholds were set with."""
    needed = [n for n in (debarcode_names or []) if n in (af_profiles or {})]
    return [key for key in keys
            if any(n not in set((sample_af_profiles or {}).get(key) or []) for n in needed)]


# ---------------------------------------------------------------------------
# Section 2: tags and combos
# ---------------------------------------------------------------------------

def tag_name_for_channel(channel: str, antigen: str, recognise) -> str:
    """Tag name of a ticked channel: the recognised FlowCode tag, else the
    antigen label, else the channel name."""
    tag = recognise(antigen) if antigen else None
    if tag:
        return tag
    return antigen.strip() if antigen and antigen.strip() else channel


def auto_tag_channels(channels, antigens: dict, recognise) -> list:
    """Channels whose antigen is a FlowCode tag, in channel order."""
    return [ch for ch in channels if antigens.get(ch) and recognise(antigens[ch])]


def csv_tag_channels(csv_combos, channels, antigens: dict, recognise) -> tuple:
    """
    Channels named by the tags of a combinations CSV.

    Tags are matched to channels by canonical marker name (case-insensitive),
    never by substring. Returns (channels in channel order, {CSV tag upper:
    channel}, tags that match no channel).
    """
    by_upper = {}
    for ch in channels:
        antigen = antigens.get(ch) or ''
        name = tag_name_for_channel(ch, antigen, recognise)
        by_upper.setdefault(name.upper(), ch)
        if antigen.strip():
            by_upper.setdefault(antigen.strip().upper(), ch)
    wanted, unmatched = {}, []
    for _name, tags in csv_combos or []:
        for tag in tags:
            key = str(tag).strip()
            canonical = recognise(key) or key
            ch = by_upper.get(canonical.upper()) or by_upper.get(key.upper())
            if ch is None:
                if key not in unmatched:
                    unmatched.append(key)
            else:
                wanted[key.upper()] = ch
    picked = [ch for ch in channels if ch in set(wanted.values())]
    return picked, wanted, unmatched


def generate_combo_rows(tags, previous=None) -> list:
    """All 3-tag combos of *tags*, keeping Include and Name from *previous*
    rows with the same tag set."""
    from flowcode_core import all_triplets

    kept = {tuple(sorted(str(t).upper() for t in row['tags'])): row for row in previous or []}
    rows = []
    for name, triplet in all_triplets(tags):
        old = kept.get(tuple(sorted(t.upper() for t in triplet)))
        rows.append({
            'include': bool(old['include']) if old else True,
            'name': old['name'] if old else name,
            'tags': list(triplet),
        })
    return rows


def csv_combo_rows(csv_combos, tag_for_csv: dict, previous=None) -> list:
    """Combo rows from a combinations CSV, tags renamed to the tag names of
    their channels; Include and Name are kept from *previous* rows with the
    same tag set."""
    kept = {tuple(sorted(str(t).upper() for t in row['tags'])): row for row in previous or []}
    rows = []
    for name, tags in csv_combos or []:
        tags = [tag_for_csv.get(str(t).strip().upper(), str(t).strip()) for t in tags]
        old = kept.get(tuple(sorted(t.upper() for t in tags)))
        rows.append({
            'include': bool(old['include']) if old else True,
            'name': old['name'] if old else name,
            'tags': tags,
        })
    return rows


def default_library_pattern(name: str) -> str:
    """
    Sample-name pattern of a library: 'Librar' (Library, Libraries), then
    nothing, a space or an underscore, then the library's number, letter or
    name, not followed by another letter or digit. That token is what
    follows 'Library' at the end of *name* ('Library 2' gives '2'), else the
    whole name. 'Library 1' matches 'SILPL_Library1_LSRB' but not
    'Library10'.
    """
    text = str(name).strip()
    m = re.search(r'librar(?:y|ies)?[ _]?([A-Za-z0-9]+)$', text, re.IGNORECASE)
    token = m.group(1) if m else text
    return rf'librar(?:y|ies)?[ _]?{re.escape(token)}(?![A-Za-z0-9])'


def new_library(libraries, csv_path=None, combos=None) -> dict:
    """A library named 'Library <n>' with the first free n, its default
    pattern, and the given combinations CSV."""
    taken = {str(lib.get('name', '')).upper() for lib in libraries or []}
    n = 1
    while f'{LIBRARY_PREFIX} {n}'.upper() in taken:
        n += 1
    name = f'{LIBRARY_PREFIX} {n}'
    return {'name': name, 'label': '', 'pattern': default_library_pattern(name),
            'csv_path': csv_path, 'combos': [[str(g), [str(t) for t in tags]] for g, tags in combos or []]}


def library_label(library: dict) -> str:
    """The name a library is shown by in results: its label, else its name."""
    return str(library.get('label') or '').strip() or str(library.get('name', ''))


def library_sample_keys(samples: dict, backbone_keys) -> list:
    """Samples that need a library, in experiment order: every sample
    except single-stain controls, unstained samples, beads and the
    backbones."""
    all_samples = samples.get('all_samples') or {}
    backbones = set(backbone_keys or [])
    return [key for key in backbone_candidates(samples)
            if key not in backbones and not _is_bead(samples, key, str(all_samples.get(key, '')))]


def assign_libraries(keys, names: dict, libraries, current: dict, only_unassigned: bool = False) -> tuple:
    """
    Library assignments from the libraries' sample-name patterns, by
    ``flowcode_analysis.assign_by_pattern`` (case-insensitive on the
    displayed name, first matching library wins).

    With *only_unassigned*, only samples without a library are filled.
    Otherwise every sample in *keys* is reassigned: one matching no pattern
    loses its library, except that a sample assigned to Other keeps it.
    Returns ({sample: library}, {library: n matched}); raises ValueError
    naming a library whose pattern is not valid.
    """
    from flowcode_analysis import assign_by_pattern

    lib_names = [str(lib.get('name', '')) for lib in libraries]
    patterns = {str(lib.get('name', '')): lib.get('pattern') or '' for lib in libraries}
    found, matched = assign_by_pattern(list(keys), names, lib_names, patterns)
    known = set(lib_names) | {OTHER_LIBRARY}
    out = {k: v for k, v in (current or {}).items() if v in known}
    for key in keys:
        if only_unassigned and out.get(key):
            continue
        if key in found:
            out[key] = found[key]
        elif out.get(key) != OTHER_LIBRARY:
            out.pop(key, None)
    return out, matched


def library_combo_rows(libraries, tag_for_csv: dict, previous=None) -> list:
    """
    Combination rows of several libraries: one row per tag triplet used by
    any library, in order of first use, named by its triplet
    (``combo_key``), with the guide each library gives it in ``names``
    ({library: guide}). Tags are renamed to the tag names of their channels;
    Include is kept from *previous* rows with the same tag set.
    """
    from flowcode_core import combo_key

    kept = {tuple(sorted(str(t).upper() for t in row['tags'])): row for row in previous or []}
    rows, index = [], {}
    for lib in libraries or []:
        for guide, tags in lib.get('combos') or []:
            tags = [tag_for_csv.get(str(t).strip().upper(), str(t).strip()) for t in tags]
            key = tuple(sorted(t.upper() for t in tags))
            if key not in index:
                old = kept.get(key)
                index[key] = len(rows)
                rows.append({'include': bool(old['include']) if old else True,
                             'name': combo_key(tags), 'tags': tags, 'names': {}})
            rows[index[key]]['names'][str(lib.get('name', ''))] = str(guide)
    return rows


def shared_guide_names(libraries) -> dict:
    """{guide name: [library names]} for names used by more than one library."""
    seen: dict = {}
    for lib in libraries or []:
        for guide, _tags in lib.get('combos') or []:
            owners = seen.setdefault(str(guide), [])
            if lib.get('name') not in owners:
                owners.append(lib.get('name'))
    return {g: libs for g, libs in seen.items() if len(libs) > 1}


def library_problems(libraries) -> list:
    """Problems with the libraries (empty if none): names, patterns and
    combinations files."""
    problems, seen = [], set()
    if len(libraries or []) < 1:
        problems.append('Add at least one library.')
    for lib in libraries or []:
        name = str(lib.get('name', '')).strip()
        if not name:
            problems.append('A library has no name.')
            continue
        if name.upper() == OTHER_LIBRARY.upper():
            problems.append(f'"{OTHER_LIBRARY}" is reserved for samples that belong to no library.')
        if name.upper() in seen:
            problems.append(f'Library name "{name}" is used twice.')
        seen.add(name.upper())
        try:
            re.compile(lib.get('pattern') or '')
        except re.error as exc:
            problems.append(f'{name}: the sample-name pattern is not valid ({exc}).')
        if not lib.get('combos'):
            problems.append(f'{name}: load its combinations file.')
    return problems


def tag_problems(tag_map: dict, fluor_profiles) -> tuple:
    """(errors, warnings) for the ticked tags."""
    errors, warnings = [], []
    n = len(tag_map)
    if n < MIN_TAGS:
        errors.append(f'Tick at least {MIN_TAGS} tag channels ({n} ticked).')
    elif n < RECOMMENDED_TAGS:
        warnings.append(f'Only {n} tags are ticked; FlowCodeUnmix expects at least {RECOMMENDED_TAGS}.')
    upper = {}
    for tag in tag_map:
        upper.setdefault(tag.upper(), []).append(tag)
    for dupes in upper.values():
        if len(dupes) > 1:
            errors.append(f'Tags {", ".join(dupes)} differ only in case; rename the antigens.')
    missing = [ch for ch in tag_map.values() if ch not in set(fluor_profiles)]
    if missing:
        errors.append(f'No reference spectrum for {", ".join(missing)}.')
    return errors, warnings


def duplicate_tag_problems(tag_names: dict) -> list:
    """Ticked channels that carry the same tag name (case-insensitive)."""
    by_tag = {}
    for channel, tag in tag_names.items():
        by_tag.setdefault(str(tag).upper(), []).append(channel)
    return [f'{", ".join(chs)} all carry tag {tag_names[chs[0]]}; untick all but one.'
            for chs in by_tag.values() if len(chs) > 1]


def combo_problems(rows, tag_names, existing_channels) -> list:
    """Problems with the combos table (empty if none)."""
    from flowcode_core import combo_key, validate_combo_names

    problems = []
    included = [row for row in rows if row.get('include', True)]
    if not included:
        problems.append('Include at least one combination.')
    known = {t.upper() for t in tag_names}
    seen = {}
    for row in included:
        tags = [str(t) for t in row['tags']]
        unknown = [t for t in tags if t.upper() not in known]
        if unknown:
            problems.append(f'"{row["name"]}": {", ".join(unknown)} is not a ticked tag.')
        if len(tags) != 3 or len({t.upper() for t in tags}) != 3:
            problems.append(f'"{row["name"]}" must have three different tags.')
        key = combo_key(tags)
        if key in seen:
            problems.append(f'"{seen[key]}" and "{row["name"]}" have the same tags.')
        seen.setdefault(key, row['name'])
    problems.extend(validate_combo_names([row['name'] for row in included], existing_channels))
    return problems


def design_from_state(state: FlowCodeState, channel_fluor: dict | None = None):
    """``FlowCodeDesign`` of the ticked tags and included combos. Tag
    fluorophores are the tag channels (unmixed channel = fluorophore label)
    unless *channel_fluor* maps them."""
    from flowcode_core import FlowCodeDesign

    channel_fluor = channel_fluor or {}
    tag_to_fluor = {tag: channel_fluor.get(ch, ch) for tag, ch in state.tag_map().items()}
    combos = [(row['name'], tuple(row['tags'])) for row in state.included_combos()]
    return FlowCodeDesign.build(tag_to_fluor, combos)


# ---------------------------------------------------------------------------
# Section 3: gates and thresholds
# ---------------------------------------------------------------------------

def gate_record(x_channel: str, y_channel: str, vertices_raw, source: str, note: str = '') -> dict:
    """JSON record of a FlowCode scatter gate in raw units."""
    return {
        'x': x_channel,
        'y': y_channel,
        'vertices_raw': [[float(a), float(b)] for a, b in np.asarray(vertices_raw, dtype=float)],
        'source': source,
        'note': note,
    }


# Threshold methods set by the user, kept when the gates are confirmed again.
USER_THRESHOLD_METHODS = ('manual', 'imported')


def threshold_record(raw: float, method: str, delta_bic: float | None, auto_raw: float | None = None,
                     auto_method: str | None = None) -> dict:
    return {
        'raw': float(raw),
        'method': method,
        'delta_bic': None if delta_bic is None or not np.isfinite(delta_bic) else float(delta_bic),
        'auto_raw': float(raw if auto_raw is None else auto_raw),
        'auto_method': auto_method or method,
    }


# ---------------------------------------------------------------------------
# Section 4: debarcode and FRET
# ---------------------------------------------------------------------------

def backbone_summary(summary: dict, design) -> dict:
    """JSON record of the backbone debarcoding counts, per combo keyed by
    tag set so that renaming a combo keeps its count."""
    from flowcode_core import combo_key

    out = {k: int(summary.get(k, 0)) for k in ('total_events', 'untransduced', 'unresolved', 'rescued',
                                                'rescued_toofew')}
    out['per_combo'] = {combo_key(tags): int(summary['per_combo'][name])
                        for name, tags in zip(design.combo_names, design.combo_tags)}
    return out
