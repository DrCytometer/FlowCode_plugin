"""
flowcode_fix_unmix.py
---------------------
Fix My Unmix (AutoSpectral > Spectral Correction) on FlowCode-corrected
data, started from section 5. No Qt.

The main app's run (``spectral_correction.run_corrections``) reads every
sample through a loader. ``corrected_loader`` gives it, for the FlowCode
samples, their gated raw events with the characterised FRET/error model
and the tandem breakdown removed: ``staged_export_fn``'s
``corrected_raw``, the same raw that section 5 hands its final unmix. Any
other sample (the unstained sample, the single-stain controls of the spread
recomputed after the run) is read unchanged. ``keep_tags_out`` keeps the
tag fluorophores fixed and out of the fit as target channels
(``FixSettings.exclude_fixed_targets``), so evidence in a tag channel never
changes a marker's spectrum.

The stored corrections carry a ``flowcode`` record (characterisation id,
tags, time; ``mark_corrections``), so section 5 can tell corrections fitted
here from ones fitted on uncorrected data by the Spectral Correction page.

v1 simplifications
    * Each sample is corrected as one block of its events in the Fix Unmix
      gate, not in section 5's 2,000,000-event chunks of the whole file, so
      the too-few rescue's references and the tandem estimate come from
      those events.
    * The Spectral Correction page's own Fix Unmix still reads uncorrected
      data; section 5 only warns when the corrections in use came from it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

STORE_MARK = 'flowcode'
JOB_LABEL = 'FlowCode: Fix Unmix (FRET-corrected)'


def keep_tags_out(settings, tags) -> None:
    """Add *tags* to the run's fixed fluorophores and leave fixed rows out as
    target channels. Raises ValueError on a Honeychrome without the option."""
    if not hasattr(settings, 'exclude_fixed_targets'):
        raise ValueError('This Honeychrome\'s Fix My Unmix cannot leave the tag channels out of the fit: '
                         'update Honeychrome.')
    settings.fixed_fluorophores = tuple(dict.fromkeys([str(f) for f in settings.fixed_fluorophores or ()]
                                                      + [str(t) for t in tags]))
    settings.exclude_fixed_targets = True


def corrected_loader(unmix_fn, af_by_sample: dict, detectors, scatter_names, base_loader=None):
    """
    A ``spectral_correction.run_corrections`` loader: ``(experiment_dir,
    path, gate, layout, gating) -> (fluorescence, scatter)``, the FlowCode-
    corrected raw fluorescence for the samples of *af_by_sample* ({sample
    key: AFState of its export library}) and the raw events for any other.

    unmix_fn      : ``PluginWidget.export_job``'s per-chunk function
                    (``staged_export_fn``); its ``prepare`` debarcodes the
                    sample and estimates its calibration and tandem
                    breakdown.
    detectors     : the fluorescence columns the base loader returns.
    scatter_names : its scatter columns (FlowCode scatter gates are found
                    by name among both).
    """
    if base_loader is None:
        from honeychrome.controller_components.spread_model import load_events as base_loader
    pnn = list(detectors) + list(scatter_names)

    def loader(experiment_dir, path, gate, layout, gating):
        fl, sc = base_loader(experiment_dir, path, gate, layout, gating)
        af = af_by_sample.get(path)
        if af is None or len(fl) == 0:
            return fl, sc
        fl = np.asarray(fl, dtype=np.float64)
        raw = np.hstack([fl, np.asarray(sc, dtype=np.float64).reshape(len(fl), -1)])
        if getattr(unmix_fn, 'prepare', None) is not None:
            unmix_fn.prepare(fl, raw, af, path, pnn=pnn)
        return unmix_fn.corrected_raw(fl, raw, af, path, pnn=pnn), sc

    return loader


def mark_corrections(process: dict, store_key: str, characterisation: str | None, tags) -> None:
    """Record on the stored corrections that FlowCode fitted them."""
    entry = process.get(store_key)
    if isinstance(entry, dict):
        entry[STORE_MARK] = {'characterisation': characterisation, 'tags': [str(t) for t in tags],
                             'fitted': datetime.now().strftime('%Y-%m-%d %H:%M')}


def correction_record(process: dict, store_key: str) -> dict | None:
    """The ``flowcode`` record of the stored corrections, or None."""
    entry = process.get(store_key)
    return entry.get(STORE_MARK) if isinstance(entry, dict) else None
