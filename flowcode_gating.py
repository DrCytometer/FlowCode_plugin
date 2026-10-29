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
Gates are only displayed: editing stays in the main window.
"""

from __future__ import annotations

import numpy as np

VIEW_UNCORRECTED = 'uncorrected'
VIEW_STANDARD = 'standard'
VIEW_CALIBRATED = 'calibrated'
VIEWS = {VIEW_UNCORRECTED: 'Uncorrected', VIEW_STANDARD: 'FlowCode standard',
         VIEW_CALIBRATED: 'FlowCode calibrated'}
# Section 5 settings of each view (``PluginWidget.export_job`` overrides).
VIEW_OVERRIDES = {VIEW_UNCORRECTED: {'correction': 'none', 'tandem_tags': [], 'calib_combos': False},
                  VIEW_STANDARD: {'calib_combos': False},
                  VIEW_CALIBRATED: {'calib_combos': True}}

HIGHLIGHT_BARCODED = 'barcoded'
HIGHLIGHT_NONE = 'none'
# Highlighted events drawn per 2D plot (all events are counted).
MAX_HIGHLIGHT_POINTS = 5_000


def focus_mask(ids: np.ndarray, highlight=HIGHLIGHT_BARCODED, n_combos: int = 0) -> np.ndarray:
    """Events highlighted: none, barcoded (1..*n_combos*), or one combination."""
    if highlight == HIGHLIGHT_NONE:
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


def plot_data(events: np.ndarray, pnn: list, sample_key: str, snapshot, plots: list, fluorescence: list) -> dict:
    """
    ``data_for_cytometry_plots`` of one sample's FlowCode-unmixed *events*
    (columns *pnn*): *plots* (the experiment's Unmixed Data plot list; plots
    on channels missing from *pnn* are left out), the snapshot's gating and
    the sample's transforms, the gate membership the counts use, the
    statistics and histograms. *fluorescence* names the fluorescence
    channels (ribbon plots).
    """
    from honeychrome.controller_components.functions import calc_hists, calc_stats
    import honeychrome.settings as hc_settings

    pnn = list(pnn)
    transforms = snapshot._sample_transforms(events, pnn)
    usable = []
    for plot in plots:
        channels = [plot.get(k) for k in ('channel_x', 'channel_y') if plot.get(k)]
        if plot.get('type') == 'ribbon' and 'ribbon' not in transforms:
            continue
        if all(ch in pnn and ch in transforms for ch in channels):
            usable.append(dict(plot))
    membership = snapshot.memberships(sample_key, events, pnn)
    data = {
        'pnn': pnn,
        'pnn_labels': {},
        'fluoro_indices': [pnn.index(ch) for ch in fluorescence if ch in pnn],
        'lookup_tables': {},
        'event_data': events,
        'transformations': transforms,
        'statistics': {},
        'gating': _with_overrides(snapshot.gating, str(sample_key), snapshot.overrides.get(str(sample_key))),
        'sample_id': str(sample_key),
        'plots': usable,
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
    its parent) and 'changed': the highlighted (*focus*) events inside the
    gate in one view and not in the other (None without
    *reference_membership*, the uncorrected view's).
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
                     'changed': changed})
    return rows


def highlight_overlay(data: dict, focus: np.ndarray, seed: int = 0) -> list:
    """Per plot of *data*, the highlighted events of its source population on
    its display scale: ('points', x, y) for a 2D plot (at most
    ``MAX_HIGHLIGHT_POINTS``), ('counts', counts) on the plot's bins for a
    1D plot, None for a ribbon plot or without highlighted events."""
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
        if idx.size > MAX_HIGHLIGHT_POINTS:
            idx = np.sort(rng.choice(idx, MAX_HIGHLIGHT_POINTS, replace=False))
        xy = []
        for ch in (plot['channel_x'], plot['channel_y']):
            values = events[idx, pnn.index(ch)].astype(np.float64)
            xform = getattr(transforms[ch], 'xform', None)
            xy.append(xform.apply(values) if xform is not None else values)
        out.append(('points', xy[0], xy[1]))
    return out
