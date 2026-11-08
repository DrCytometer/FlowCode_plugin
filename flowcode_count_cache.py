"""
flowcode_count_cache.py
-----------------------
Compact per-sample events for *Count events* on the FlowCode Analysis
page, so that counting again does not unmix (or read) every sample again.
No Qt.

In plain terms
    Counting needs far less than a whole sample. Every counted population
    sits inside one gate drawn only on scatter, Time or the viability dye
    (usually the viable gate); outside it nothing is ever counted. Inside
    it, only the channels that gates are drawn on, the FlowCode ID and, for
    barcoded events alone, the marker channels (for the marker sums) are
    used. The first count reads (with Live unmix: unmixes) each sample once
    and keeps just that; later counts read only the kept part, however the
    gates below the storing gate are redrawn or which populations under it
    are ticked. On OTB01 (CD3, CD4, CD8 and Treg counted) the storing gate
    was the viable gate, holding 61 % of the events, and 14 of the 106
    columns were kept: 12 MB per sample.

What is kept, per sample
    * every event inside the *storing gate*: the deepest gate above every
      counted population drawn on no fluorophore other than a viability
      dye (every event when there is none, e.g. counting all events);
    * for those events, the channels of the storing gate, of the gates
      above it and of every gate below it, each per-sample axis (Time) and
      the FlowCode ID (the combination channels for files without one);
    * for the barcoded ones, the marker channels as well;
    * the event with the largest value on each per-sample axis, so that
      per-sample gates are scaled exactly as on the whole sample; it lies
      outside the storing gate unless it was kept anyway, so it is never
      counted;
    * the sample's total number of events.

    Counts made from the kept events are the counts of the whole sample,
    exactly: every counted population lies inside the storing gate, each
    event is gated on its own values with the same axis ranges, and the
    marker sums are over barcoded events only.

When a sample is read in full again
    When its file or the unmixing changes, when the storing gate or a gate
    above it is redrawn (or overridden for that sample), when a gate is
    drawn on a channel that is not kept, or when the marker channels
    change. One file is kept per sample, in ``cache/flowcode_analysis/
    count_events``; an older file of the same sample is removed when a new
    one is written.

Phenotype populations are scored on every fluorophore of every event inside
the FlowCode gates, so counts with phenotypes ticked do not use this cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import flowcode_analysis as fa
import flowcode_phenotype_shift as fsh

VERSION = 1


def cache_folder(experiment_dir) -> Path:
    """Folder of the compact events, in the experiment's cache."""
    return Path(experiment_dir) / 'cache' / 'flowcode_analysis' / 'count_events'


def storing_gate(snapshot, gates, fluors, viability) -> str | None:
    """The deepest gate above (or at) every gate of *gates* drawn, like all
    its ancestors, on no fluorophore of *fluors* other than the viability
    dyes; None (every event) when there is none."""
    leave = [f for f in fluors if f not in set(viability or ())]
    return fsh.selection_gate(snapshot, list(gates), leave)


def kept_channels(snapshot, gate: str | None) -> list:
    """Channels the counts can gate on: those of *gate*, of the gates above
    it and of every gate below it (every gate when *gate* is None), and
    each per-sample axis."""
    out = []
    names = snapshot.gate_names()
    for name in names:
        chain = fsh.gate_chain(snapshot, name)
        if gate is None or gate in chain or name in fsh.gate_chain(snapshot, gate):
            out.extend(fsh.gate_channels(snapshot, name))
    out.extend(ch for ch, tr in snapshot.transforms.items() if getattr(tr, 'id', None) == 'default')
    return list(dict.fromkeys(out))


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def file_for(folder, sample_key: str, source_key: str, gate: str | None, chain_signature: str, channels: list,
             markers: list) -> Path:
    """Path of one sample's compact events: the sample's prefix, then a
    digest of everything that decides them."""
    stem = _digest(str(sample_key))[:16]
    return Path(folder) / f'{stem}_{_digest([VERSION, source_key, gate, chain_signature, channels, markers])[:24]}.npz'


def compact(events: np.ndarray, pnn: list, sample_key: str, snapshot, gate: str | None, channels: list,
            markers: list) -> dict:
    """The kept part of one sample (see the module docstring), as arrays
    ready for ``numpy.savez``."""
    pnn = list(pnn)
    names = fa.combo_names_in_file(pnn)
    if not names:
        raise ValueError(f'{sample_key}: no FlowCode combination channels.')
    ids, has_id = fa.event_combo_ids(events, pnn)
    if gate is None:
        inside = np.ones(len(events), dtype=bool)
    else:
        membership = snapshot.memberships(sample_key, events, pnn, [gate])
        if gate not in membership:
            raise ValueError(f'Gate "{gate}" could not be applied to {sample_key}.')
        inside = np.asarray(membership[gate], dtype=bool)
    ranges = [int(np.argmax(events[:, pnn.index(ch)])) for ch in fsh.per_sample_channels(snapshot, pnn)
              if len(events)]
    rows = np.union1d(np.flatnonzero(inside), np.asarray(ranges, dtype=np.int64))
    ids_kept = ids[rows]
    columns = [c for c in channels if c in pnn]
    columns += [fa.ID_CHANNEL] if has_id else [fa.COMBO_PREFIX + n for n in names]
    columns = list(dict.fromkeys(columns))
    marker_columns = [m for m in markers if m in pnn and m not in columns]
    barcoded = np.flatnonzero((ids_kept >= 1) & (ids_kept <= len(names)))
    values = np.asarray(events[np.ix_(rows, [pnn.index(c) for c in columns])])
    marker_values = np.asarray(events[np.ix_(rows[barcoded], [pnn.index(m) for m in marker_columns])])
    out = []
    for block in (values, marker_values):
        small = block.astype(np.float32)
        out.append(small if np.array_equal(small.astype(block.dtype), block) else block)
    return {'pnn': np.array(pnn, dtype=str), 'columns': np.array(columns, dtype=str), 'values': out[0],
            'markers': np.array(marker_columns, dtype=str), 'marker_values': out[1],
            'barcoded': barcoded.astype(np.int64), 'n_events': np.array(len(events), dtype=np.int64)}


def expand(stored) -> tuple:
    """(events, pnn) rebuilt from compact events: kept rows, every column of
    the sample, zero where a value was not kept."""
    pnn = [str(c) for c in stored['pnn']]
    values = np.asarray(stored['values'], dtype=np.float64)
    events = np.zeros((len(values), len(pnn)), dtype=np.float64)
    events[:, [pnn.index(str(c)) for c in stored['columns']]] = values
    markers = [pnn.index(str(m)) for m in stored['markers']]
    if markers:
        events[np.ix_(np.asarray(stored['barcoded']), markers)] = np.asarray(stored['marker_values'],
                                                                            dtype=np.float64)
    return events, pnn


def _write(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)
    for old in path.parent.glob(f"{path.name.split('_')[0]}_*.npz"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass


def count_sample(sample_key: str, load, snapshot, gates: list, markers: list, folder, source_key: str,
                 fluors, viability) -> tuple:
    """
    ``flowcode_analysis.count_populations`` of one sample from its compact
    events, made first from the whole sample (*load*: () → (events, pnn))
    when they are not in *folder*. *source_key* identifies the sample's
    unmixed values (file and unmixing). Returns (records, read in full).
    """
    gate = storing_gate(snapshot, gates, fluors, viability)
    channels = kept_channels(snapshot, gate)
    chain = snapshot.chain_signature(sample_key, [gate]) if gate else ''
    path = file_for(folder, sample_key, source_key, gate, chain, channels, list(markers))
    read = not path.exists()
    if read:
        events, pnn = load()
        arrays = compact(events, pnn, sample_key, snapshot, gate, channels, list(markers))
        del events
        _write(path, arrays)
    else:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    events, pnn = expand(arrays)
    counted = fa.count_populations(events, pnn, sample_key, snapshot, gates, markers)
    n_events = int(arrays['n_events'])
    for record in counted.values():
        record['events'] = n_events
    return counted, read
