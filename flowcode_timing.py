"""
flowcode_timing.py
------------------
Timing logs for the FlowCode live unmix (Analysis > Count events) and
export. ``install()`` wraps the steps below in flowcode_core,
flowcode_export, flowcode_method, flowcode_analysis and the raw-event
reader (``FcsEventReader``: opening a file, and each block read, whose
events are the rows it returns); every call,
including those on worker threads, is added to its sample's totals, and
one block of rows per sample is appended to ``~/flowcode_timing.csv`` when
the sample is done (the gating and counting, ``count_populations``, after
the unmix, as a block of its own; the export is timed only in its
per-sample preparation):

    run, sample, step, calls, total_s, max_s, events

Times are inclusive (a step's time contains the steps it calls). The row
``sample total`` is the whole sample; ``cached`` is 1 when the sample's
FlowCode columns came from the live-unmix cache.
"""

from __future__ import annotations

import csv
import functools
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path.home() / 'flowcode_timing.csv'
TARGETS = {
    'flowcode_export': ('live_sample_events', 'prepare_sample', 'apply_transfer_matrix'),
    'flowcode_core': ('_debarcode', '_debarcode_and_correct', 'estimate_combination_scales', 'estimate_tandem_shifts',
                      'staged_corrected_raw', 'unmix_staged', 'apply_error_correction', 'apply_combination_scales',
                      'apply_tandem_correction'),
    'flowcode_analysis': ('count_populations',),
}
METHODS = {
    'flowcode_method': {'UnmixPlan': ('final', 'debarcode')},
    'honeychrome.controller_components.functions': {'FcsEventReader': ('__init__', 'read')},
}
SAMPLE_ARGS = {'live_sample_events': 3, 'prepare_sample': 4, 'count_populations': 2}
EVENT_ARGS = {'apply_transfer_matrix': 1}

_lock = threading.Lock()
_local = threading.local()
_run = datetime.now().strftime('%H:%M:%S')
_installed = False


def _events(args) -> int:
    for a in args:
        if isinstance(a, np.ndarray) and a.ndim == 2:
            return int(a.shape[0])
    return 0


def _record(name: str, seconds: float, n: int):
    totals = getattr(_local, 'totals', None)
    if totals is None:
        return
    row = totals.setdefault(name, [0, 0.0, 0.0, 0])
    row[0] += 1
    row[1] += seconds
    row[2] = max(row[2], seconds)
    row[3] += n


def _write(sample: str, outer: str, totals: dict, seconds: float, n: int):
    rows = [[_run, sample, 'sample total' if outer == 'live_sample_events' else f'{outer} total', 1,
             f'{seconds:.3f}', f'{seconds:.3f}', n or '']]
    if outer == 'live_sample_events':
        rows.append([_run, sample, 'cached', int('prepare_sample' not in totals and 'unmix_staged' not in totals),
                     '', '', ''])
    for name, (calls, total, longest, n) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        rows.append([_run, sample, name, calls, f'{total:.3f}', f'{longest:.3f}', n])
    with _lock:
        new = not OUT.exists()
        with open(OUT, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(['run', 'sample', 'step', 'calls', 'total_s', 'max_s', 'events'])
            writer.writerows(rows)


def _wrap(fn, name: str):
    position = SAMPLE_ARGS.get(name)
    event_arg = EVENT_ARGS.get(name)

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        outer = position is not None and getattr(_local, 'totals', None) is None
        if outer:
            _local.totals = {}
        start = time.perf_counter()
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        finally:
            seconds = time.perf_counter() - start
            n = _events(args[event_arg:event_arg + 1] if event_arg is not None else args) or _events((result,))
            if outer:
                sample = args[position] if len(args) > position else kwargs.get('sample_key', '')
                totals, _local.totals = _local.totals, None
                _write(str(sample), name, totals, seconds, n)
            else:
                _record(name, seconds, n)
    timed._flowcode_timed = True
    return timed


def install():
    """Wrap the timed steps once (module functions and plan methods)."""
    global _installed
    if _installed:
        return
    import importlib
    for module_name, names in TARGETS.items():
        module = importlib.import_module(module_name)
        for name in names:
            fn = getattr(module, name, None)
            if fn is not None and not getattr(fn, '_flowcode_timed', False):
                setattr(module, name, _wrap(fn, name))
    for module_name, classes in METHODS.items():
        module = importlib.import_module(module_name)
        for cls_name, names in classes.items():
            cls = getattr(module, cls_name)
            for name in names:
                fn = getattr(cls, name)
                if not getattr(fn, '_flowcode_timed', False):
                    setattr(cls, name, _wrap(fn, f'{cls_name}.{name}'))
    _installed = True
    print(f'[FlowCode timing] installed; writing {OUT}')
