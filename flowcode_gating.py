"""
flowcode_gating.py
------------------
Data of the corrected gating viewer (Analysis page, *Gating* tab). No Qt.

The viewer shows the experiment's Unmixed Data plots, with their gates,
over the FlowCode-unmixed events of the sample selected in the main
window's sample browser, as the Analysis counts them. The events come from
the same live unmix as the counts (``flowcode_export.live_sample_events``),
in one of three views:

* ``VIEW_UNCORRECTED``: the experiment's own unmix of the raw file through
  the same plan (no FRET correction, no calibration, no tandem);
* ``VIEW_STANDARD``: the FlowCode correction without the per-sample
  combination calibration;
* ``VIEW_CALIBRATED``: the FlowCode correction with it.

``plot_data`` builds the ``data_for_cytometry_plots`` dictionary that
Honeychrome's ``CytometryPlotWidget`` draws: the experiment's plots and
gating, the sample's transforms, gate membership from
``SampleGating.memberships`` (the sample's own gate overrides included,
exactly as ``flowcode_analysis.count_populations``), statistics and
histograms from ``functions.calc_stats`` and ``functions.calc_hists``.

The tab draws them from its own copy of the experiment's Unmixed Data
gating, axes and plot list (``LocalGating``). Plots, axes and gates edited
on the tab change that copy only, never the experiment or the Analysis
counts. ``gate_view`` gates one sample's events with a copy of it on a
worker thread, and ``bind`` points the result at the copy itself, so that
the plots edit it.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np

from honeychrome.controller_components.per_sample_gating_functions import SampleGating

VIEW_UNCORRECTED = 'uncorrected'
VIEW_STANDARD = 'standard'
VIEW_CALIBRATED = 'calibrated'
VIEWS = {VIEW_UNCORRECTED: 'Uncorrected', VIEW_STANDARD: 'FlowCode standard',
         VIEW_CALIBRATED: 'FlowCode calibrated'}
# The file section 5 wrote, shown when the Analysis source is Exported files.
VIEW_EXPORTED = 'exported'
VIEW_LABELS = {**VIEWS, VIEW_EXPORTED: 'Exported file'}
# Section 5 settings of each view (``PluginWidget.export_job`` overrides).
VIEW_OVERRIDES = {VIEW_UNCORRECTED: {'correction': 'none', 'tandem_tags': [], 'calib_combos': False},
                  VIEW_STANDARD: {'calib_combos': False},
                  VIEW_CALIBRATED: {'calib_combos': True}}

HIGHLIGHT_BARCODED = 'barcoded'
HIGHLIGHT_NONE = 'none'
# A phenotype population (flowcode_phenotype): 'phenotype:<population name>'.
HIGHLIGHT_PHENOTYPE = 'phenotype:'
# Highlighted events drawn per 2D plot (all events are counted).
MAX_HIGHLIGHT_POINTS = 100_000


def phenotype_highlight(name: str) -> str:
    return HIGHLIGHT_PHENOTYPE + str(name)


def highlighted_phenotype(highlight) -> str | None:
    """The population name of a phenotype highlight, else None."""
    text = str(highlight or '')
    return text[len(HIGHLIGHT_PHENOTYPE):] if text.startswith(HIGHLIGHT_PHENOTYPE) else None


def focus_mask(ids: np.ndarray, highlight=HIGHLIGHT_BARCODED, n_combos: int = 0) -> np.ndarray:
    """Events highlighted: none, barcoded (1..*n_combos*), or one combination
    (a phenotype highlight is scored by ``flowcode_phenotype.population_mask``
    instead and gives no events here)."""
    if highlight == HIGHLIGHT_NONE or highlighted_phenotype(highlight) is not None:
        return np.zeros(len(ids), dtype=bool)
    if highlight == HIGHLIGHT_BARCODED:
        return (ids >= 1) & (ids <= n_combos) if n_combos else ids >= 1
    return ids == int(highlight)


def _with_overrides(gating, sample_key: str, overrides: dict):
    """*gating* with the sample's override gates as its custom gates, so
    ``get_gate(name, sample_id=sample_key)`` returns what the sample is gated
    with."""
    for name, gate in (overrides or {}).items():
        try:
            if gating.is_custom_gate(sample_key, name):
                continue
            path = gating.find_matching_gate_paths(name)[0]
            gating.add_gate(gate, gate_path=path, sample_id=sample_key)
        except Exception:
            continue
    return gating


def plot_data(events: np.ndarray, pnn: list, sample_key: str, snapshot, plots: list, fluorescence: list,
              spectral_model: list | None = None) -> dict:
    """
    ``data_for_cytometry_plots`` of one sample's FlowCode-unmixed *events*
    (columns *pnn*): *plots* (an Unmixed Data plot list; plots on channels
    missing from *pnn* are left out, the others are *plots*' own
    dictionaries, at the positions given in 'plot_index'), the snapshot's
    gating and the sample's transforms, the gate membership the counts use,
    the statistics and histograms. Columns with no axis in the experiment
    (FlowCode's own columns) get one from ``own_axes``, so that any column
    can be chosen on a plot. *fluorescence* names the fluorescence
    channels (ribbon plots). *spectral_model* (the experiment's controls)
    gives the axis labels their antigens, as on the Unmixed Data plots.
    """
    from honeychrome.controller_components.functions import build_display_label_map, calc_hists, calc_stats
    import honeychrome.settings as hc_settings

    pnn = list(pnn)
    transforms = snapshot._sample_transforms(events, pnn)
    transforms.update(own_axes(events, pnn, [ch for ch in pnn if ch not in transforms],
                               logicle=transforms.get('ribbon')))
    usable, index = [], []
    for i, plot in enumerate(plots):
        channels = [plot.get(k) for k in ('channel_x', 'channel_y') if plot.get(k)]
        if plot.get('type') == 'ribbon' and 'ribbon' not in transforms:
            continue
        if all(ch in pnn and ch in transforms for ch in channels):
            usable.append(plot)
            index.append(i)
    membership = snapshot.memberships(sample_key, events, pnn)
    data = {
        'pnn': pnn,
        'pnn_labels': build_display_label_map(pnn, spectral_model),
        'fluoro_indices': [pnn.index(ch) for ch in fluorescence if ch in pnn],
        'lookup_tables': {},
        'event_data': events,
        'transformations': transforms,
        'statistics': {},
        'gating': _with_overrides(snapshot.gating, str(sample_key), snapshot.overrides.get(str(sample_key))),
        'sample_id': str(sample_key),
        'plots': usable,
        'plot_index': index,
        'histograms': [],
        'gate_membership': membership,
        'cytometer_db_col': None,
    }
    data['statistics'] = calc_stats(data, initialise=True)
    data['histograms'] = calc_hists(data, density_cutoff=hc_settings.density_cutoff_retrieved)
    return data


def gate_rows(data: dict, reference_membership: dict | None = None, focus: np.ndarray | None = None) -> list:
    """
    One row per gate of *data* (``plot_data``): 'name', 'parent', 'n_in'
    (the population's events, the counts' 'in_population'), 'percent' (of
    its parent), 'n_focus' (the highlighted events inside the gate; None
    without *focus*) and 'changed': the highlighted events inside the gate
    in one view and not in the other (None without *reference_membership*,
    the uncorrected view's).
    """
    gating, stats, membership = data['gating'], data['statistics'], data['gate_membership']
    rows = []
    for gid, gpath in gating.get_gate_ids():
        if gating._get_gate_node(gid, gpath).gate_type == 'QuadrantGate' or gid not in membership:
            continue
        inside = np.asarray(membership[gid], dtype=bool)
        changed = None
        if reference_membership is not None and gid in reference_membership:
            differs = inside != np.asarray(reference_membership[gid], dtype=bool)
            changed = int(np.count_nonzero(differs & focus)) if focus is not None else int(np.count_nonzero(differs))
        parent = gpath[-1] if gpath else 'root'
        if parent != 'root' and gating._get_gate_node(parent, gpath[:-1]).gate_type == 'QuadrantGate':
            parent = gpath[-2] if len(gpath) > 1 else 'root'
        rows.append({'name': gid, 'parent': parent, 'n_in': int(np.count_nonzero(inside)),
                     'percent': 100.0 * float(stats.get(gid, {}).get('p_gate_parent', np.nan)),
                     'n_focus': None if focus is None else int(np.count_nonzero(inside & focus)),
                     'changed': changed})
    return rows


def highlight_overlay(data: dict, focus: np.ndarray, seed: int = 0) -> list:
    """Per plot of *data*, the highlighted events of its source population on
    its display scale: ('points', x, y, n) for a 2D plot (*n* highlighted
    events, of which at most ``MAX_HIGHLIGHT_POINTS`` are drawn), ('counts',
    counts) on the plot's bins for a 1D plot, None for a ribbon plot or
    without highlighted events."""
    from honeychrome.controller_components.functions import calc_hist1d

    pnn, events, transforms = data['pnn'], data['event_data'], data['transformations']
    rng = np.random.default_rng(seed)
    out = []
    for plot in data['plots']:
        source = data['gate_membership'].get(plot['source_gate'])
        if source is None or plot['type'] == 'ribbon' or not focus.any():
            out.append(None)
            continue
        mask = np.asarray(source, dtype=bool) & focus
        if plot['type'] == 'hist1d':
            out.append(('counts', calc_hist1d(events, mask, pnn.index(plot['channel_x']),
                                              transforms[plot['channel_x']])))
            continue
        idx = np.flatnonzero(mask)
        n_total = int(idx.size)
        if idx.size > MAX_HIGHLIGHT_POINTS:
            idx = np.sort(rng.choice(idx, MAX_HIGHLIGHT_POINTS, replace=False))
        xy = []
        for ch in (plot['channel_x'], plot['channel_y']):
            values = events[idx, pnn.index(ch)].astype(np.float64)
            xform = getattr(transforms[ch], 'xform', None)
            xy.append(xform.apply(values) if xform is not None else values)
        out.append(('points', xy[0], xy[1], n_total))
    return out


def own_axes(events: np.ndarray, pnn: list, channels, logicle=None) -> dict:
    """An axis for each of *channels* (columns of *events* named by *pnn*),
    for columns the experiment has no axis for: the FlowCode channel and the
    per-combination channels, which hold tag abundances, get a copy of the
    fluorescence axis *logicle* (the experiment's default logicle axis);
    the others (e.g. the FlowCode ID) a linear axis from 0 to just above the
    column's maximum."""
    from honeychrome.controller_components.transform import Transform
    import flowcode_core as fcc

    out = {}
    for ch in channels:
        if logicle is not None and (ch == fcc.FLOWCODE_CHANNEL or ch.startswith(fcc.COMBO_CHANNEL_PREFIX)):
            out[ch] = deepcopy(logicle)
            continue
        col = events[:, pnn.index(ch)] if len(events) else np.zeros(0)
        top = float(np.nanmax(col)) if col.size else 0.0
        axis = Transform(scale_t=1.05 * top if top > 0 else 1.0, linear_a=0)
        axis.set_transform(id=0, limits=[0, 1])
        out[ch] = axis
    return out


def default_axes(controller, channels, events: np.ndarray, pnn: list) -> dict:
    """
    ``{channel: axis}`` for Reset Axes on the tab: the experiment's default
    axis of each of *channels*, as Reset Axes gives in Unmixed Data
    (per-sample axes span this sample's values), and the linear axis of
    ``own_axes`` for columns the experiment has no axis for.
    """
    from honeychrome.controller_components.functions import assign_default_transforms, generate_transformations
    import honeychrome.settings as hc_settings

    settings = controller.experiment.settings['unmixed']
    known = set(settings.get('event_channels_pnn') or []) | set(hc_settings.af_channels)
    listed = [ch for ch in channels if ch in known]
    fresh = generate_transformations(assign_default_transforms(settings, channels=listed,
                                                               n_af_spectra=controller.n_af_spectra()))
    out = {ch: fresh[ch] for ch in channels if ch in fresh}
    out.update(SampleGating(None, out)._sample_transforms(events, list(pnn)))
    rest = [ch for ch in channels if ch not in out and ch in pnn]
    logicle = generate_transformations(assign_default_transforms(settings, channels=[])).get('ribbon')
    out.update(own_axes(events, list(pnn), rest, logicle=logicle))
    return out


def _unmixed_plots(controller) -> list:
    return list((getattr(controller.experiment, 'cytometry', None) or {}).get('plots') or [])


def unmixed_signature(controller) -> str:
    """Identifies the experiment's Unmixed Data gates (overrides included),
    axes and plot list, to tell when a copy taken from them is out of date."""
    try:
        gating = SampleGating.from_controller(controller, 'unmixed', copy=False).signature()
    except Exception:
        gating = None
    text = json.dumps([gating, _unmixed_plots(controller)], sort_keys=True, default=str)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


class LocalGating:
    """
    The Gating tab's own copy of the experiment's Unmixed Data gating (with
    the per-sample gate overrides), axes and plot list. The tab's plots edit
    these objects in place on the main thread; ``worker_snapshot`` gives a
    worker thread copies of them. ``base`` is the ``unmixed_signature`` the
    copy was taken at; ``edited`` is set once anything is changed on the
    tab.
    """

    def __init__(self, gating, transforms: dict, overrides: dict | None, plots: list, base: str = ''):
        self.gating = gating
        self.transforms = dict(transforms or {})
        self.overrides = {str(k): dict(v) for k, v in (overrides or {}).items()}
        self.plots = list(plots or [])
        self.base = base
        self.edited = False

    @classmethod
    def from_controller(cls, controller) -> 'LocalGating':
        snapshot = SampleGating.from_controller(controller, 'unmixed', copy=True)
        return cls(snapshot.gating, snapshot.transforms, snapshot.overrides, deepcopy(_unmixed_plots(controller)),
                   unmixed_signature(controller))

    def sync_overrides(self, sample_key: str):
        """Record the sample's custom gates as they now are in the gating,
        where edits on the plots land."""
        key = str(sample_key)
        own = {}
        for gid, _path in self.gating.get_gate_ids():
            try:
                if self.gating.is_custom_gate(key, gid):
                    own[gid] = self.gating.get_gate(gid, sample_id=key)
            except Exception:
                continue
        if own:
            self.overrides[key] = {**self.overrides.get(key, {}), **own}

    def worker_snapshot(self, sample_key: str):
        """(SampleGating, plot list): copies of the gating, axes, overrides
        and plots, for gating *sample_key* on a worker thread."""
        self.sync_overrides(sample_key)
        return (SampleGating(deepcopy(self.gating), deepcopy(self.transforms), deepcopy(self.overrides)),
                deepcopy(self.plots))


def gate_view(events: np.ndarray, pnn: list, sample_key: str, snapshot, plots: list, fluorescence: list,
              spectral_model: list | None = None, reference=None, focus: np.ndarray | None = None) -> dict:
    """
    One sample's events gated for the tab: 'data' (``plot_data``), 'rows'
    (``gate_rows``) and 'overlay' (``highlight_overlay``). *reference*,
    (events, pnn) of the uncorrected view of the same events in the same
    order, is gated the same way for the rows' 'changed'; *focus* marks the
    highlighted events.
    """
    data = plot_data(events, pnn, sample_key, snapshot, plots, fluorescence, spectral_model)
    other = snapshot.memberships(sample_key, reference[0], reference[1]) if reference is not None else None
    focus = np.zeros(len(events), dtype=bool) if focus is None else focus
    return {'data': data, 'rows': gate_rows(data, other, focus), 'overlay': highlight_overlay(data, focus)}


def bind(data: dict, local: LocalGating, sample_key: str) -> list:
    """
    Point *data* (``gate_view``'s, gated with a ``worker_snapshot`` of
    *local*) at *local*'s own gating, axes and plots, so that the plots edit
    *local*. Columns given an axis by ``own_axes`` keep it in *local* from
    now on. Returns *local*'s plots left out of *data* (on channels the
    events lack).
    """
    transforms = data['transformations']
    for ch, axis in local.transforms.items():
        if ch in transforms and getattr(axis, 'id', None) != 'default':
            transforms[ch] = axis
    for ch, axis in transforms.items():
        local.transforms.setdefault(ch, axis)
    data['gating'] = _with_overrides(local.gating, str(sample_key), local.overrides.get(str(sample_key)))
    data['plots'] = [local.plots[i] for i in data['plot_index']]
    kept = set(data['plot_index'])
    return [plot for i, plot in enumerate(local.plots) if i not in kept]
