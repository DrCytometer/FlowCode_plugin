"""
flowcode_export.py
------------------
Reading, FlowCode unmixing and writing of whole samples: the section 5
export and the Analysis page's live unmix.

``build_export_layout(controller)`` (main thread) snapshots the channel
layout. ``sample_blocks`` reads one sample a chunk at a time, runs the
FlowCode per-chunk function (``flowcode_core.staged_export_fn``) and yields
the export columns: the unmixed channels (scatter, time and FACSDiscover
imaging channels through the transfer matrix, fluorophores from the
FlowCode unmix), AF Abundance, AF Index, then the FlowCode channels. The
export writes those blocks (``FlowCodeExporter``); the Analysis page gates
them in memory (``live_sample_events``), so a live count and a count of the
exported file see the same events.

``FlowCodeExportRun`` times the export first when the experiment's method
is slow (``unmix_method.needs_time_probe``: spectral variants, or the noise
model without the compiled kernel), asks for confirmation, then exports.
It holds the application's heavy-job gate (``controller.job_gate``, when
this Honeychrome has one) from the estimate to the end of the export.

v1 simplification: the live unmix's cache keeps the fluorescence, AF and
FlowCode columns (float32, as written to FCS); the scatter and time columns
are taken from the raw file again.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import honeychrome.settings as hc_settings
from honeychrome.controller_components.functions import (
    FcsEventReader, apply_transfer_matrix, export_unmixed_sample_stream, iter_event_blocks,
)

logger = logging.getLogger(__name__)

# Events per chunk (also the block the too-few rescue and M2pg's sample-level
# AF weights are computed on, so a live count equals the exported file's).
UNMIX_CHUNK_SIZE = 2_000_000
PROBE_EVENTS = 5000
JOB_LABEL = 'FlowCode export'

LIVE_CACHE_SUBDIR = Path('cache') / 'flowcode_unmixed'
LIVE_CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ExportLayout:
    """The export's channel layout, snapshotted from the controller.

    The raw channel space is the whitelisted raw channels plus any
    FACSDiscover imaging channels, carried to the output by an identity
    block in ``transfer_matrix`` (raw -> unmixed, applied as ``raw @
    transfer_matrix``, uncompensated; files carry ``$SPILLOVER``).
    """
    experiment_dir: Path
    raw_subdir_abs: Path
    unmixed_subdirectory: str
    pnn_raw_export: tuple
    fl_ids_raw: tuple
    pnn_unmixed: tuple
    fl_ids_unmixed: np.ndarray
    transfer_matrix: np.ndarray
    spillover: np.ndarray
    fluor_names: tuple
    raw_settings: dict
    unmixed_settings: dict
    spectral_model: list
    all_samples: dict
    n_af_index: int | None


def _imaging_channels(raw_settings: dict, pnn_raw: list, pnn_raw_full: list) -> list:
    """FACSDiscover imaging channels not already in the whitelist."""
    if 'FACSDiscover' not in raw_settings.get('cytometer', ''):
        return []
    import re
    from honeychrome.controller_components.cytometer_whitelist import _CYTOMETER_PARAMS

    params = _CYTOMETER_PARAMS.get('FACSDiscover')
    if params is None:
        return []
    exclude = {'FSC', 'SSC', 'Time'}
    prefixes = [p for p in params.non_spectral_pat if not p.startswith('-') and p not in exclude]
    pattern = re.compile('|'.join(rf'(?:^|\b){re.escape(p)}' for p in prefixes))
    whitelisted = set(pnn_raw)
    if raw_settings.get('event_id_channel_id') is not None:
        whitelisted |= {pnn_raw_full[raw_settings['event_id_channel_id']]}
    return [ch for ch in pnn_raw_full if re.search(pattern, ch) and ch not in whitelisted]


def build_export_layout(controller) -> ExportLayout:
    """Snapshot the export layout from *controller*. Main thread only."""
    exp = controller.experiment
    raw_settings = deepcopy(exp.settings['raw'])
    unmixed_settings = deepcopy(exp.settings['unmixed'])
    experiment_dir = Path(controller.experiment_dir)

    pnn_raw_full = raw_settings['event_channels_pnn']
    pnn_raw = list(raw_settings.get('whitelisted_pnn') or pnn_raw_full)
    imaging_pnn = _imaging_channels(raw_settings, pnn_raw, pnn_raw_full)
    pnn_raw_export = pnn_raw + imaging_pnn
    pnn_unmixed = list(unmixed_settings['event_channels_pnn'])
    n_before_imaging = len(pnn_unmixed)
    pnn_unmixed = pnn_unmixed + imaging_pnn

    fl_ids_raw = [pnn_raw_export.index(pnn_raw_full[i]) for i in controller.filtered_raw_fluorescence_channel_ids]
    sc_ids_raw = [pnn_raw_export.index(pnn_raw_full[i]) for i in raw_settings['scatter_channel_ids']]
    fl_ids_unmixed = np.array(unmixed_settings['fluorescence_channel_ids'])
    sc_ids_unmixed = np.array(unmixed_settings['scatter_channel_ids'])
    n_scatter = unmixed_settings['n_scatter_channels']

    transfer = np.zeros((len(pnn_unmixed), len(pnn_raw_export)))
    transfer[np.ix_(fl_ids_unmixed, fl_ids_raw)] = np.array(exp.process['unmixing_matrix'])
    transfer[np.ix_(sc_ids_unmixed, sc_ids_raw)] = np.eye(n_scatter)
    if raw_settings.get('time_channel_id') is not None:
        raw_time_id = pnn_raw_export.index(pnn_raw_full[raw_settings['time_channel_id']])
        transfer[unmixed_settings['time_channel_id'], raw_time_id] = 1
    for k in range(len(imaging_pnn)):
        transfer[n_before_imaging + k, len(pnn_raw) + k] = 1.0

    profiles = exp.process.get('profiles', {})
    fluor_names = [c['label'] for c in exp.process.get('spectral_model', []) if c['label'] in profiles]
    return ExportLayout(
        experiment_dir=experiment_dir,
        raw_subdir_abs=(experiment_dir / raw_settings['raw_samples_subdirectory']).resolve(),
        unmixed_subdirectory=unmixed_settings['unmixed_samples_subdirectory'],
        pnn_raw_export=tuple(pnn_raw_export),
        fl_ids_raw=tuple(fl_ids_raw),
        pnn_unmixed=tuple(pnn_unmixed),
        fl_ids_unmixed=fl_ids_unmixed,
        transfer_matrix=transfer.T,
        spillover=np.array(exp.process['spillover']),
        fluor_names=tuple(fluor_names),
        raw_settings=raw_settings,
        unmixed_settings=unmixed_settings,
        spectral_model=deepcopy(exp.process.get('spectral_model', [])),
        all_samples=dict(exp.samples.get('all_samples', {})),
        n_af_index=controller.n_af_spectra(),
    )


def export_pnn(layout: ExportLayout, extra_channels) -> list:
    """Column names of an exported file: unmixed channels, AF channels, extra channels."""
    out = list(layout.pnn_unmixed)
    for label in hc_settings.af_channels:
        if label not in out:
            out.append(label)
    for name in extra_channels:
        if name in out:
            raise ValueError(f'Extra channel "{name}" has the same name as an existing channel.')
        out.append(name)
    return out


def export_relative_path(sample_path: str, layout: ExportLayout) -> Path:
    """Path of *sample_path* relative to the raw samples subdirectory."""
    p = Path(sample_path)
    sample_abs = p.resolve() if p.is_absolute() else (layout.experiment_dir / p).resolve()
    for parent in [sample_abs] + list(sample_abs.parents):
        try:
            if parent.samefile(layout.raw_subdir_abs):
                return sample_abs.relative_to(parent)
        except OSError:
            pass
    return Path(*sample_abs.parts[len(layout.raw_subdir_abs.parts):])


# ---------------------------------------------------------------------------
# One sample (worker-safe)
# ---------------------------------------------------------------------------

def open_reader(layout: ExportLayout, sample_key: str, bus=None) -> FcsEventReader:
    return FcsEventReader(layout.experiment_dir / sample_key, layout.pnn_raw_export, bus)


def prepare_sample(reader: FcsEventReader, layout: ExportLayout, af, unmix_fn, sample_key: str):
    """The per-sample step (``unmix_fn.prepare``) on every event of the sample, when there is one."""
    prepare = getattr(unmix_fn, 'prepare', None)
    if prepare is None or reader.n_events == 0:
        return
    raw = reader.read(0, reader.n_events)
    prepare(raw[:, list(layout.fl_ids_raw)], raw, af, sample_key)
    del raw


def sample_blocks(reader: FcsEventReader, layout: ExportLayout, af, unmix_fn, sample_key: str,
                  pnn: list, n_extra: int, chunk_size: int = UNMIX_CHUNK_SIZE, status=None):
    """Read, FlowCode-unmix and yield one sample's export columns (*pnn*) chunk by chunk."""
    n_events = reader.n_events
    n_unmixed = len(layout.pnn_unmixed)
    af_abundance_col = pnn.index(hc_settings.af_abundance_channel)
    af_index_col = pnn.index(hc_settings.af_index_channel)
    extra_start = len(pnn) - n_extra
    fl_ids_raw = list(layout.fl_ids_raw)
    n_chunks = math.ceil(n_events / chunk_size) if n_events else 0
    for chunk_i, (s_row, e_row, _) in enumerate(iter_event_blocks(n_events, chunk_events=chunk_size)):
        if n_chunks > 1 and status is not None:
            status(f'{Path(sample_key).stem}: events {s_row:,}-{e_row:,} of {n_events:,}')
        raw = reader.read(s_row, e_row)
        result = unmix_fn(raw[:, fl_ids_raw], raw, af, sample_key)
        unmixed = apply_transfer_matrix(layout.transfer_matrix, raw)
        unmixed[:, layout.fl_ids_unmixed] = result['unmixed']
        block = np.zeros((e_row - s_row, len(pnn)), dtype=np.float64)
        block[:, :n_unmixed] = unmixed
        block[:, af_abundance_col] = result['af_scale']
        idx = np.asarray(result['af_idx'], dtype=np.int64)
        block[:, af_index_col] = np.where(idx > 0, np.asarray(af.index_map)[np.clip(idx, 1, None) - 1], 0)
        if n_extra:
            block[:, extra_start:] = result['extra']
        del raw, result, unmixed
        yield block
        del block
        if n_chunks > 1 and chunk_i % 5 == 4:
            gc.collect()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _qt():
    from PySide6.QtCore import QObject, QThread, Signal
    return QObject, QThread, Signal


QObject, QThread, Signal = _qt()


class FlowCodeExporter(QObject):
    """Unmix and write each sample with the FlowCode per-chunk function.

    ``af_by_sample`` is {sample key: AFState} (samples without one, or with
    fewer than 2 AF spectra, are skipped); ``method_label(sample key, af)``
    names the sample's unmixing for the ``UNMIXINGMETHOD`` keyword and
    ``spectra_for(sample key)`` gives the spectra written to the file.
    ``sample_done(sample key, n_events)`` runs on the export thread after
    each file is written; ``sample_keywords(sample key)`` adds keywords to
    that file only.
    """
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, sample_keys, layout: ExportLayout, unmix_fn, af_by_sample: dict, bus=None,
                 extra_channels=(), output_root=None, name_suffix: str = '', method_label=None,
                 spectra_for=None, extra_keywords: dict | None = None, chunk_size: int = UNMIX_CHUNK_SIZE,
                 sample_done=None, sample_keywords=None):
        super().__init__()
        self.sample_keys = list(sample_keys)
        self.layout = layout
        self.unmix_fn = unmix_fn
        self.af_by_sample = dict(af_by_sample)
        self.bus = bus
        self.extra_channels = list(extra_channels)
        self.output_root = Path(output_root) if output_root is not None else None
        self.name_suffix = name_suffix
        self.method_label = method_label or (lambda _key, _af: 'FlowCode')
        self.spectra_for = spectra_for
        self.extra_keywords = dict(extra_keywords or {})
        self.chunk_size = int(chunk_size)
        self.sample_done = sample_done
        self.sample_keywords = sample_keywords

    def _output_folder(self, sample_key: str) -> Path:
        rel = export_relative_path(sample_key, self.layout)
        root = self.output_root or (self.layout.experiment_dir / self.layout.unmixed_subdirectory)
        return (root / rel).parent

    def _status(self, text: str):
        if self.bus is not None:
            self.bus.statusMessage.emit(f'FlowCode export: {text}')

    def run(self):
        try:
            from honeychrome import __version__

            layout = self.layout
            pnn = export_pnn(layout, self.extra_channels)
            total = len(self.sample_keys)
            n_exported = n_skipped = 0
            for n, key in enumerate(self.sample_keys):
                self.progress.emit(n, total)
                af = self.af_by_sample.get(key)
                if af is None or af.n_spectra < 2:
                    logger.warning(f'FlowCode export: "{key}" has no AF library with at least 2 spectra; skipped.')
                    n_skipped += 1
                    continue
                reader = open_reader(layout, key, self.bus)
                try:
                    n_events = reader.n_events
                    if n_events == 0:
                        n_skipped += 1
                        continue
                    prepare_sample(reader, layout, af, self.unmix_fn, key)
                    folder = self._output_folder(key)
                    folder.mkdir(parents=True, exist_ok=True)
                    name = layout.all_samples.get(key, Path(key).stem)
                    keywords = dict(self.extra_keywords)
                    if self.sample_keywords is not None:
                        keywords.update(self.sample_keywords(key) or {})
                    spectra = self.spectra_for(key) if self.spectra_for is not None else None
                    export_unmixed_sample_stream(
                        sample_name=f'{name}{self.name_suffix}',
                        unmixed_folder=folder,
                        chunks=sample_blocks(reader, layout, af, self.unmix_fn, key, pnn,
                                             len(self.extra_channels), self.chunk_size, self._status),
                        n_events=n_events,
                        export_pnn=pnn,
                        spillover=layout.spillover,
                        raw_keywords=reader.keywords,
                        spectral_model=layout.spectral_model,
                        unmixed_settings=layout.unmixed_settings,
                        raw_settings=layout.raw_settings,
                        af_spectra=af.spectra,
                        unmixing_spectra=None if spectra is None else np.asarray(spectra),
                        version=__version__,
                        extra_null_channels=None,
                        unmixing_method=self.method_label(key, af),
                        unmixing_weights=None,
                        af_index_map=af.index_map,
                        n_af_index=layout.n_af_index,
                        extra_keywords=keywords or None,
                    )
                finally:
                    reader.close()
                n_exported += 1
                if self.sample_done is not None:
                    self.sample_done(key, n_events)
            self.progress.emit(total, total)
            if self.bus is not None:
                destination = (str(self.output_root) if self.output_root is not None
                               else f'"{layout.unmixed_subdirectory}" folder in experiment folder')
                message = f'Exported {n_exported} sample(s) with FlowCode unmixing, to \n{destination}'
                if n_skipped:
                    message += f'\n\n{n_skipped} sample(s) skipped (no AF library of 2 or more spectra, or no events).'
                self.bus.popupMessage.emit(message)
            self.finished.emit()
        except Exception as exc:
            logger.exception('FlowCode export failed')
            self.error.emit(str(exc) or type(exc).__name__)


def estimate_time(sample_keys, layout: ExportLayout, af_by_sample: dict, probe_fn,
                  sample_n: int = PROBE_EVENTS) -> dict | None:
    """
    Estimated unmixing time of the export: *probe_fn* (raw fluorescence, af,
    sample key) timed on ``sample_n`` events taken from the start of the
    samples, in proportion to their event counts. File reading and writing
    are not included. None when there is nothing to export; ``probe_events``
    0 when the export is no larger than the probe.
    """
    from honeychrome.controller_components.unmix_method import allocate_probe_rows

    usable = []
    for key in sample_keys:
        af = af_by_sample.get(key)
        if af is None or af.n_spectra < 2:
            continue
        reader = open_reader(layout, key)
        try:
            n = int(reader.n_events)
        finally:
            reader.close()
        if n:
            usable.append((key, af, n))
    if not usable:
        return None
    total = sum(n for _k, _a, n in usable)
    out = {'samples': len(usable), 'events': int(total), 'probe_events': 0, 'seconds': 0.0,
           'events_per_second': float('nan')}
    if total <= sample_n:
        return out
    elapsed, probed = 0.0, 0
    fl_ids = list(layout.fl_ids_raw)
    for (key, af, _n), k in zip(usable, allocate_probe_rows([n for _k, _a, n in usable], sample_n)):
        if k == 0:
            continue
        reader = open_reader(layout, key)
        try:
            block = reader.read(0, k)
        finally:
            reader.close()
        start = time.perf_counter()
        probe_fn(np.ascontiguousarray(block[:, fl_ids], dtype=np.float64), af, key)
        elapsed += time.perf_counter() - start
        probed += len(block)
    rate = probed / max(elapsed, 1e-9)
    out.update(probe_events=probed, events_per_second=rate, seconds=total / rate)
    return out


def estimate_message(estimate: dict, method_label: str) -> str:
    from honeychrome.controller_components.unmix_method import format_duration
    return '\n'.join([
        f'Estimated FlowCode unmixing time with the {method_label}: about '
        f'{format_duration(estimate["seconds"])} for {estimate["samples"]} sample(s), '
        f'{estimate["events"]:,} events in total.',
        '',
        f'Based on a {estimate["probe_events"]:,}-event probe ({estimate["events_per_second"]:,.0f} events/sec, '
        'debarcoding and final unmix). Reading and writing files is not included; the real time can differ '
        'by a factor of 2 or more.',
        '',
        'Start the export?',
    ])


class _EstimateWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            logger.exception('FlowCode export: time estimate failed')
            self.error.emit(str(exc) or type(exc).__name__)


class FlowCodeExportRun(QObject):
    """Estimate (when *probe_fn* is given), confirm, export. Create and
    start on the main thread; ``done`` is emitted once at the end, whatever
    the outcome. ``start()`` returns False when the heavy-job gate is held
    by another job (nothing is started)."""
    status = Signal(str)
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)
    cancelled = Signal()
    done = Signal()

    def __init__(self, exporter: FlowCodeExporter, controller=None, parent_widget=None, probe_fn=None,
                 method_label: str = '', sample_n: int = PROBE_EVENTS):
        super().__init__()
        self._exporter = exporter
        self._controller = controller
        self._parent_widget = parent_widget
        self._probe_fn = probe_fn
        self._method_label = method_label
        self._sample_n = sample_n
        self._token = None
        self._est_thread = self._est_worker = None
        self._exp_thread = None

    def busy_text(self) -> str | None:
        gate = getattr(self._controller, 'job_gate', None)
        if gate is None or gate.holder is None:
            return None
        busy = getattr(gate, 'busy_text', None)
        return busy() if callable(busy) else 'Another job is running.'

    def start(self) -> bool:
        gate = getattr(self._controller, 'job_gate', None)
        if gate is not None:
            self._token = gate.try_acquire(JOB_LABEL)
            if self._token is None:
                self.status.emit(self.busy_text() or 'Another job is running.')
                return False
        if self._probe_fn is None:
            self._start_export()
            return True
        self.status.emit('Estimating the unmixing time…')
        exporter = self._exporter
        fn = lambda: estimate_time(exporter.sample_keys, exporter.layout, exporter.af_by_sample,  # noqa: E731
                                   self._probe_fn, self._sample_n)
        self._est_thread = QThread()
        self._est_worker = _EstimateWorker(fn)
        self._est_worker.moveToThread(self._est_thread)
        self._est_thread.started.connect(self._est_worker.run)
        self._est_worker.finished.connect(self._on_estimate)
        self._est_worker.error.connect(self._on_estimate_error)
        self._est_worker.finished.connect(self._est_thread.quit)
        self._est_worker.error.connect(self._est_thread.quit)
        self._est_thread.finished.connect(self._est_thread.deleteLater)
        self._est_thread.finished.connect(self._on_est_thread_finished)
        self._est_thread.start()
        return True

    def _on_est_thread_finished(self):
        self._est_thread = None
        self._est_worker = None

    def _confirm(self, text: str) -> bool:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self._parent_widget, 'FlowCode unmixing time', text,
                                     QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
        return reply == QMessageBox.Yes

    def _release(self):
        token, self._token = self._token, None
        if token is not None:
            token.release()

    def _finish(self):
        self._release()
        self.done.emit()

    def _on_estimate(self, estimate):
        try:
            if estimate is None:
                self.error.emit('None of the selected samples has an AF library with at least 2 spectra.')
                self._finish()
                return
            if estimate['probe_events'] == 0:
                self._start_export()
                return
            from honeychrome.controller_components.unmix_method import format_duration
            self.status.emit(f'Estimated unmixing time: about {format_duration(estimate["seconds"])}')
            if self._confirm(estimate_message(estimate, self._method_label)):
                self._start_export()
            else:
                self.status.emit('Export cancelled.')
                self.cancelled.emit()
                self._finish()
        except Exception:
            self._finish()
            raise

    def _on_estimate_error(self, message: str):
        if self._confirm(f'The unmixing time could not be estimated:\n{message}\n\nStart the export anyway?'):
            self._start_export()
        else:
            self.status.emit('Export cancelled.')
            self.cancelled.emit()
            self._finish()

    def _start_export(self):
        self._exp_thread = QThread()
        self._exporter.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exporter.run)
        self._exporter.progress.connect(self.progress)
        self._exporter.finished.connect(self.finished)
        self._exporter.error.connect(self.error)
        self._exporter.finished.connect(self._exp_thread.quit)
        self._exporter.error.connect(self._exp_thread.quit)
        self._exp_thread.finished.connect(self._exp_thread.deleteLater)
        self._exp_thread.finished.connect(self._on_export_thread_finished)
        self._exp_thread.start()

    def _on_export_thread_finished(self):
        self._exp_thread = None
        self._finish()


# ---------------------------------------------------------------------------
# Live unmix for the Analysis page (worker-safe)
# ---------------------------------------------------------------------------

def file_signature(path) -> list:
    try:
        st = Path(path).stat()
        return [int(st.st_size), int(st.st_mtime_ns)]
    except OSError:
        return []


def live_cache_key(*parts) -> str:
    """Digest naming a sample's live-unmix cache file: everything that decides its FlowCode columns."""
    from honeychrome.controller_components.unmix_method import digest
    return digest(LIVE_CACHE_VERSION, *parts)


def live_cache_file(experiment_dir, key: str) -> Path:
    return Path(experiment_dir) / LIVE_CACHE_SUBDIR / f'{key}.npz'


def live_sample_events(layout: ExportLayout, af, unmix_fn, sample_key: str, pnn: list, n_extra: int,
                       cache_key: str | None = None, chunk_size: int = UNMIX_CHUNK_SIZE,
                       max_cache_bytes: int | None = None) -> np.ndarray:
    """
    One sample's export columns (*pnn*), as the export would write them
    (float32 values), without writing a file. With *cache_key* the columns
    that come from the FlowCode unmix (fluorophores, AF, FlowCode channels)
    are read from or written to ``cache/flowcode_unmixed/<cache_key>.npz``;
    the others are recomputed from the raw file.
    """
    reader = open_reader(layout, sample_key)
    try:
        n = reader.n_events
        fl_cols = np.concatenate([np.asarray(layout.fl_ids_unmixed, dtype=np.int64),
                                  [pnn.index(c) for c in hc_settings.af_channels],
                                  np.arange(len(pnn) - n_extra, len(pnn))]).astype(np.int64)
        cache_path = live_cache_file(layout.experiment_dir, cache_key) if cache_key else None
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path) as z:
                    cached = z['values']
                if cached.shape == (n, len(fl_cols)):
                    os.utime(cache_path)
                    out = np.zeros((n, len(pnn)), dtype=np.float32)
                    rows = 0
                    for s_row, e_row, _ in iter_event_blocks(n, chunk_events=chunk_size):
                        raw = reader.read(s_row, e_row)
                        out[s_row:e_row, :len(layout.pnn_unmixed)] = apply_transfer_matrix(layout.transfer_matrix,
                                                                                         raw)
                        rows = e_row
                    out[:rows, fl_cols] = cached
                    return out
            except Exception as exc:
                logger.warning(f'FlowCode live unmix: cache {cache_path.name} unreadable ({exc})')
        prepare_sample(reader, layout, af, unmix_fn, sample_key)
        out = np.empty((n, len(pnn)), dtype=np.float32)
        start = 0
        for block in sample_blocks(reader, layout, af, unmix_fn, sample_key, pnn, n_extra, chunk_size):
            out[start:start + len(block)] = block
            start += len(block)
    finally:
        reader.close()
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix('.tmp.npz')
            np.savez(tmp, values=out[:, fl_cols])
            os.replace(tmp, cache_path)
            from honeychrome.controller_components.unmix_method import prune_display_cache
            prune_display_cache(cache_path.parent, max_cache_bytes)
        except OSError as exc:
            logger.warning(f'FlowCode live unmix: could not write the cache ({exc})')
    return out
