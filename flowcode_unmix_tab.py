"""
flowcode_unmix_tab.py
---------------------
Honeychrome plugin: FlowCode debarcoding with FRET and spillover correction,
a port of the FlowCodeUnmix 0.6.0 AF-matched staged pipeline.

Pages
    1. Inputs              backbone samples (pooled), matched unstained,
                           AF library, optional combinations CSV
    2. Tags & Combos       FlowCode tag channels and the combinations table
    3. Gates & Thresholds  FlowCode scatter gates on the backbone, the
                           debarcoding space and per-tag positivity thresholds
    4. Debarcode & FRET    backbone characterisation: combo frequencies,
                           FRET/spillover heatmap, panel expression before
                           and after correction, fit diagnostics
    5. Unmix & Export      FlowCode unmixing and FCS export of the selected
                           samples

Numerics live in flowcode_core.py, the experiment state in flowcode_state.py
and the section 4 figures in flowcode_plots.py, next to this file. The state
is saved with the experiment in ``process['flowcode_unmix']`` and the
characterised backbone in ``cache/flowcode_spectra.npz``; event data are kept
for the session only and reloaded on demand.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt, QObject, QSettings, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMenu, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
    QTableWidgetItem, QToolBox, QVBoxLayout, QWidget,
)
from flowkit import GatingStrategy, gates as fk_gates

import honeychrome.settings as hc_settings
from honeychrome.controller_components.autospectral_functions import af_index_lookup
from honeychrome.controller_components.cytometer_whitelist import singlet_channels
from honeychrome.controller_components.functions import calc_hists, calc_stats, define_polygon_gate
from honeychrome.controller_components.sample_loader import read_raw_events, seeded_subsample_indices
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.help_toggle_widget import HelpToggleWidget, WheelBlocker
from honeychrome.view_components.icon_loader import icon

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import flowcode_core as fcc  # noqa: E402
import flowcode_plots as fcp  # noqa: E402
import flowcode_state as fcs  # noqa: E402

logger = logging.getLogger(__name__)

plugin_name = 'FlowCode'

# QSettings store for UI toggles only; experiment state lives in the .kit file.
_SETTINGS_APP_KEY = 'plugin_FlowCodeUnmix'

_GATE_MODE = 'flowcode'

# Characterised backbone arrays, in the experiment's cache folder.
SPECTRA_FILE = 'flowcode_spectra.npz'
# R-compatible thresholds and combinations, in the experiment folder.
OUTPUT_SUBDIR = 'FlowCode'

# Square plot sizes (pixels): FlowCode gates, threshold biplot, before/after biplots.
GATE_PLOT_SIZE = 360
THRESHOLD_PLOT_SIZE = 440
ASSESS_PLOT_SIZE = 400

# Default export folder (in the experiment folder) and file name suffix.
EXPORT_SUBDIR = 'FlowCode Unmixed'
EXPORT_NAME_SUFFIX = ' FlowCode'
SUMMARY_FILE = 'FlowCode_debarcode_summary.csv'
RUN_FILE = 'FlowCode_run.json'

_ERROR_STYLE = 'color: #d62728;'
_WARN_STYLE = 'color: #cc6600;'
_DONE_STYLE = 'color: #2ca02c;'
_HINT_STYLE = 'color: grey; font-style: italic;'
_BANNER_STYLE = ('QLabel { color: #856404; background: #fff3cd; border: 1px solid #ffc107; '
                 'border-radius: 4px; padding: 8px; }')

SETUP_BANNER = ('Run Setup in the AutoSpectral Optimization tab first. It computes the spectral variants '
                'used for per-cell optimisation in the final unmix and the unstained-derived positivity '
                'thresholds used as the rescue noise floor.')

KERNEL_BANNER = ('<b>Compiled AutoSpectral Optimization kernel not found.</b> Per-cell AF unmixing uses '
                 "Honeychrome's Python implementation and the final unmix runs without per-cell spectral "
                 'variants. Run <tt>build_autospectral_opt_kernel.py</tt> (in bundled_plugins/) to enable them.')

_HELP = {
    1: ('<p><b>Backbone</b>: the samples used to characterise the FlowCode unmixing error, usually pools '
        'of all FlowCode-transduced cells. Tick several to pool their events: every combination needs '
        'enough brightly barcoded events to be characterised. Samples with "backbone" or "FRET" in the '
        'name are ticked at first; single-stain controls and unstained samples are not listed. '
        '<b>Matched unstained</b>: the unstained sample whose FSC/SSC '
        'distribution is closest to the pooled backbone (Bhattacharyya coefficient of the scatter histograms, '
        '1 = identical); click a row to choose a different one. Its AF profiles, extracted in the '
        'AutoSpectral AF tab, are used for per-cell AF unmixing during debarcoding.</p>'
        '<p><b>Combinations CSV</b> (optional): the FlowCodeUnmix file with an <tt>Id</tt> column and '
        'three tag columns. Without it, every 3-tag combination of the ticked tags is used.</p>'),
    2: ('<p>Tick the unmixed channels that carry a FlowCode tag. Channels whose antigen is a known tag '
        '(HA, V5, AU1, AU5, S, C, Flag, NWS, VSVg, Ollas) are ticked automatically. '
        '<b>Nearest tag</b> is the other tag with the most similar reference spectrum; it is the '
        'default y-axis of the threshold plot.</p>'
        '<p>Untick <b>Include</b> to drop a combination: its events are then reported as unresolved. '
        'Names become exported channels <tt>FC &lt;name&gt;</tt>.</p>'),
    3: ('<p><b>FlowCode Singlets</b> keeps events whose FSC area/height ratio is below the 85th '
        'percentile. <b>FlowCode Cells</b> surrounds the scatter of the events most clearly positive '
        'for three tags (70% robust Mahalanobis contour), enlarged 1.5 times about its centre so it '
        'does not clip barcoded cells. Drag the gates to edit them, or drag the round handle on FlowCode '
        'Cells to enlarge or shrink the whole gate about its centre; <b>Use experiment gate</b> copies '
        'the Cells or Singlets gate from the Raw Data tab.</p>'
        '<p><b>Confirm gates</b> unmixes the gated backbone and matched unstained with per-cell AF '
        '(the debarcoding space) and sets a starting threshold per tag: a two-component mixture '
        'crossover when the tag is bimodal, otherwise the upper 1% tail of the negatives, never below '
        'the 99.5th percentile of the unstained. Drag the red line or edit the raw value to change '
        'a threshold. <b>Import thresholds CSV</b> reads a FlowCodeUnmix thresholds file (Fluor, Tag, '
        'Threshold_Raw), matching rows by tag name, then by fluorophore.</p>'),
    4: ('<p><b>Debarcode and identify FRET spillover</b> debarcodes the gated backbone in the debarcoding '
        'space and rescues events with more than three positive tags whose tag pattern clearly matches one '
        'combination, and events with one or two positive tags that match a combination at a lower '
        'expression level (see section 5). For each combination it then fits the unmixing error (FRET plus residual spillover) '
        'on events whose every on-tag is at least 3x its threshold, as a line through the origin in its '
        'three tag abundances plus the products of each tag pair. FRET grows with donor x acceptor, so '
        'the error curves upward; a straight line through the middle of the data would leave a constant '
        'offset after correction. The error is measured from the untransduced events (their median raw '
        'spectrum and its tag abundances), so the line runs through an untransduced cell: AF and '
        'non-specific background staining that every cell carries in the other tag channels are kept, '
        'and corrected cells sit with the untransduced cells rather than at zero, while error that looks '
        'like autofluorescence stays in the error rather than being taken for the cells\' own AF. '
        'Combinations with fewer than 50 fit events use the <b>pairwise '
        'model</b>, fitted on the fit events of all combinations together: the error of each tag given '
        'each partner, plus a product term per tag pair. It is also applied, through the above-threshold '
        'tags, to events without a valid barcode.</p>'
        '<p>The heatmap shows how much each channel gains per unit of on-tag abundance at the '
        "combination's median expression (outlined: the combination's own tags); <b>Curvature</b> in "
        'the table is how far the fitted error bends away from a straight line. The panel expression '
        'plot and the biplots use the final unmix of section 5 (the compiled kernel with per-cell '
        'variants and cell.weight when they are set) on untransduced and barcoded events, without and '
        'with the fitted error removed; barcoded events are drawn as larger points. Results are saved '
        'with the experiment; <tt>FlowCode/thresholds.csv</tt> and <tt>FlowCode/combinations.csv</tt> can '
        'be read by FlowCodeUnmix in R.</p>'),
    5: ('<p>Unmixes the selected samples with FlowCode error correction (unmix.flowcode.staged()): '
        'per-cell AF unmixing with the sample\'s own AF library, debarcoding with the backbone '
        'thresholds and too-many and too-few-tag rescue, subtraction of the characterised error from the '
        'raw data, then the final unmix, with per-cell spectral variants when ticked. Each file gets a '
        '<tt>FlowCode</tt> channel (the median final abundance of a barcoded event\'s three tags, 0 '
        'otherwise) and one <tt>FC &lt;name&gt;</tt> channel per combination.</p>'
        '<p>Files are written as <tt>&lt;sample&gt; FlowCode (Unmixed).fcs</tt>, mirroring the raw folder '
        'structure, with <tt>FlowCode_debarcode_summary.csv</tt> (event counts per sample and '
        'combination) and <tt>FlowCode_run.json</tt> (thresholds and settings) beside them.</p>'
        '<p>Each sample is unmixed with the AF profiles assigned to it in the AutoSpectral AF tab plus the '
        'AF profiles the thresholds were set with (section 3), so debarcoding uses at least the AF '
        'spectra the thresholds were set in. Samples whose library has fewer than 2 spectra cannot be '
        'selected.</p>'
        '<p><b>Too-few-tag rescue</b>: the thresholds are the same for every sample, so a cell whose tags '
        'are all dimmer in a sample can lose one or two of them below threshold. An unresolved event '
        'with one or two positive tags is assigned to the combination whose tag pattern it shares when, '
        'measured from the sample\'s untransduced median and in units of each tag\'s threshold, its '
        'pattern is at least 0.9 cosine-similar to that combination\'s confident events in the same block '
        '(shrunk towards the backbone\'s when they are few), clearly closer than to the next combination '
        'that also carries its positive tags, and its missing tags are each at least 2 untransduced SDs and '
        'a quarter of the way to threshold. Events with a saturated detector, or whose AF abundance '
        'exceeds their largest positive tag (large autofluorescent events), are not rescued. The settings '
        'are in section 4.</p>'
        '<p><b>Tandem tags</b> (Advanced settings): a tag on a tandem fluorophore that breaks down in a '
        'sample emits part of its signal as its parent (PE-Cy7 as PE), which lands on the parent\'s marker '
        'in every combination carrying the tag. For each ticked tag, the sample\'s gated events give the '
        'parent\'s rise with expression in combinations with the tag beyond that in combinations without '
        'it (combination means removed), compared combination by combination with the same statistic on '
        'the backbone. A difference with |z| above the threshold is removed along the parent\'s spectrum '
        'in proportion to the tag\'s abundance, weighted by 1 - (threshold / z)<sup>2</sup>. Tags whose '
        'fluorophore class is "Protein tandem" are ticked at first; the parent comes from the '
        '<tt>tandem.parent</tt> column of the fluorophore database, and its spectrum from the panel or, '
        'when it is not in the panel, from the cytometer\'s reference library. The backbone statistic is '
        'computed in section 4.</p>'),
}

# Filled by _ensure_qt_imports() on the main thread: these modules create Qt
# objects on import, and plugin modules are loaded on a background thread.
pg = None
cc = None
CytometryPlotWidget = None
AfComparisonPlotWidget = None
_GatePlot = None
_ThresholdBiplot = None
_TagBiplot = None
_qt_imports_done = False


def _ensure_qt_imports():
    global pg, cc, CytometryPlotWidget, AfComparisonPlotWidget, _qt_imports_done
    if _qt_imports_done:
        return
    import colorcet as _cc
    import pyqtgraph as _pg
    from honeychrome.view_components.cytometry_plot_widget import CytometryPlotWidget as _CPW
    from honeychrome.view_components.autospectral_tab import AfComparisonPlotWidget as _ACPW
    cc, pg = _cc, _pg
    CytometryPlotWidget, AfComparisonPlotWidget = _CPW, _ACPW
    _build_plot_classes()
    _qt_imports_done = True


def _build_plot_classes():
    """Plot subclasses, defined once their base classes are imported."""
    global _GatePlot, _ThresholdBiplot, _TagBiplot

    class GatePlot(CytometryPlotWidget):
        """A CytometryPlotWidget showing one fixed FlowCode gate: no source-gate
        or channel menus, no new plots, and deleting the gate asks the page to
        reset it."""
        gateResetRequested = Signal(str)

        def select_plot_on_parent_grid(self):
            pass

        def mouseDoubleClickEvent(self, event):
            event.accept()

        def set_source_gate(self, n, parent):
            pass

        def new_plot_on_gate(self, source_gate):
            pass

        def configure_title(self):
            super().configure_title()
            self.plot_title.leftClickMenuItems = []

        def right_click_menu(self, ev):
            menu = QMenu()
            menu.addAction('Fit Axes to Data', self.fit_axes_to_data)
            menu.addAction('Reset Axes', self.reset_axes_transforms)
            menu.addAction('Export Image', self.export_image)
            menu.exec(ev.screenPos().toPoint())

        def remove_gate_and_roi(self, *args):
            roi = args[0] if args and hasattr(args[0], 'label') else None
            name = roi.label.gate_name if roi is not None else ''
            self.gateResetRequested.emit(name)

    class ThresholdBiplot(AfComparisonPlotWidget):
        """Untitled debarcoding-space biplot of one tag with a draggable
        threshold line. The x channel is the tag; the y channel can be changed
        from the axis label."""
        thresholdMoved = Signal(float)

        def __init__(self, controller, parent=None):
            super().__init__('', controller, parent=parent)
            self.graphics_widget.removeItem(self.plot_title)
            self._threshold_raw = None
            self._line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ff3b30', width=2),
                                         hoverPen=pg.mkPen('#ff9500', width=3))
            self._line.setZValue(50)
            self._line.sigPositionChangeFinished.connect(self._on_line_moved)
            self.vb.addItem(self._line)

        def _configure_title(self):
            self.plot_title.setText('')
            self.plot_title.leftClickMenuItems = []

        def _configure_axes(self):
            super()._configure_axes()
            self.label_x.leftClickMenuItems = []
            self._place_line()

        def _compute_gate_mask(self, event_data):
            return None

        def _draw(self):
            super()._draw()
            self._place_line()

        def set_threshold(self, raw):
            self._threshold_raw = None if raw is None else float(raw)
            self._place_line()

        def _x_transform(self):
            return self._transformations.get(self._channel_x)

        def _place_line(self):
            tr = self._x_transform()
            if tr is None or self._threshold_raw is None:
                self._line.setVisible(False)
                return
            self._line.setVisible(True)
            self._line.blockSignals(True)
            self._line.setValue(_forward(tr)(np.array([self._threshold_raw]))[0])
            self._line.blockSignals(False)

        def _on_line_moved(self):
            tr = self._x_transform()
            if tr is None:
                return
            raw = float(_inverse(tr)(np.array([self._line.value()]))[0])
            self._threshold_raw = raw
            self.thresholdMoved.emit(raw)

    class TagBiplot(AfComparisonPlotWidget):
        """Biplot of locally held events, without a source gate, whose axis
        menus offer the channels given to ``set_menu_channels``. Events picked
        by ``set_highlight`` are also drawn as points over the density, so
        that rare events stay visible whatever the axis scaling."""

        def __init__(self, title, controller, parent=None):
            super().__init__(title, controller, parent=parent)
            self._menu_channels = []
            self._menu_labels = []
            self._highlight = None
            self._highlight_size = 0
            self._highlight_item = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(255, 46, 196, 220))
            self._highlight_item.setZValue(30)
            self.vb.addItem(self._highlight_item)

        def set_highlight(self, mask, size: int):
            """Draw the events in *mask* as *size*-pixel points (none when 0)."""
            self._highlight = None if mask is None else np.asarray(mask, dtype=bool)
            self._highlight_size = int(size)
            self._draw_highlight()

        def set_highlight_size(self, size: int):
            self._highlight_size = int(size)
            self._draw_highlight()

        def highlighted_count(self) -> int:
            return 0 if self._highlight is None else int(self._highlight.sum())

        def clear(self):
            super().clear()
            self._highlight = None
            self._highlight_item.setData([], [])

        def _draw(self):
            super()._draw()
            self._draw_highlight()

        def _draw_highlight(self):
            data = self._event_data
            mask = self._highlight
            pnn = self.controller.experiment.settings['unmixed'].get('event_channels_pnn', [])
            cx, cy = self._channel_x, self._channel_y
            if data is None or mask is None or self._highlight_size <= 0 or len(mask) != len(data) \
                    or cx not in pnn or cy not in pnn \
                    or cx not in self._transformations or cy not in self._transformations:
                self._highlight_item.setData([], [])
                return
            x = _forward(self._transformations[cx])(data[mask, pnn.index(cx)])
            y = _forward(self._transformations[cy])(data[mask, pnn.index(cy)])
            self._highlight_item.setData(x, y, size=self._highlight_size)

        def set_menu_channels(self, channels, labels):
            self._menu_channels = list(channels)
            self._menu_labels = list(labels)

        def channels(self) -> tuple:
            return self._channel_x, self._channel_y

        def _configure_title(self):
            self.plot_title.setText(self._title_text)
            self.plot_title.leftClickMenuItems = []

        def _compute_gate_mask(self, event_data):
            return None

        def _configure_axes(self):
            super()._configure_axes()
            if not self._menu_channels:
                return
            for label, channel in ((self.label_x, self._channel_x), (self.label_y, self._channel_y)):
                label.leftClickMenuItems = list(self._menu_labels)
                label.leftItemSelected = self._menu_channels.index(channel) if channel in self._menu_channels else 0

        def _menu_channel(self, n):
            return self._menu_channels[n] if 0 <= n < len(self._menu_channels) else None

        def _set_channel_x(self, n, _parent):
            channel = self._menu_channel(n)
            if channel is None:
                return
            self._channel_x = channel
            self._configure_axes()
            self._draw()
            self.channelChanged.emit(self._channel_x, self._channel_y)

        def _set_channel_y(self, n, _parent):
            channel = self._menu_channel(n)
            if channel is None:
                return
            self._channel_y = channel
            self._configure_axes()
            self._draw()
            self.channelChanged.emit(self._channel_x, self._channel_y)

    _GatePlot, _ThresholdBiplot, _TagBiplot = GatePlot, ThresholdBiplot, TagBiplot


# ---------------------------------------------------------------------------
# Controller helpers (main thread)
# ---------------------------------------------------------------------------

def _process(controller) -> dict:
    return controller.experiment.process


def _fluor_names(controller) -> list:
    """Fluorophore labels with a reference spectrum, in spectral-model order."""
    profiles = _process(controller).get('profiles') or {}
    return [c['label'] for c in _process(controller).get('spectral_model') or [] if c['label'] in profiles]


def _reference_spectra(controller):
    spectra = controller._build_fluor_spectra()
    return None if spectra is None else np.asarray(spectra, dtype=np.float64)


def _detectors(controller) -> list:
    raw = controller.experiment.settings['raw']
    full = raw['event_channels_pnn']
    pnn = raw.get('whitelisted_pnn') or full
    return [full[i] for i in controller.filtered_raw_fluorescence_channel_ids if full[i] in pnn]


def _antigens(controller) -> dict:
    return {c.get('label'): (c.get('antigen') or '') for c in _process(controller).get('spectral_model') or []}


def _tag_channel_options(controller) -> list:
    """Unmixed fluorophore channels that can carry a tag."""
    unmixed = controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn') or []
    return [name for name in _fluor_names(controller) if name in unmixed]


def _channel_label(channel: str, antigens: dict) -> str:
    antigen = antigens.get(channel, '')
    return f'{antigen} - {channel}' if antigen else channel


def _cytometer_db_col(controller):
    return controller.experiment.settings.get('raw', {}).get('cytometer_db_col')


def _optimization_setup_done(controller) -> bool:
    return bool(getattr(controller, 'autospectral_variants', None)) and \
        getattr(controller, 'autospectral_unmixed_pos_thresholds', None) is not None


def _sample_name(controller, key) -> str:
    return str((controller.experiment.samples.get('all_samples') or {}).get(key, key))


def _is_dark() -> bool:
    """Dark figure styling when the application palette is dark."""
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    palette = app.palette() if app is not None else QPalette()
    return palette.color(QPalette.ColorRole.Base).value() < 128


def _rescue_transform_params(magnitude_ceiling: float) -> dict:
    """Parameters of the fixed rescue logicle (flowcode_core.logicle_transform)."""
    return {'type': 'logicle', 'magnitude_ceiling': float(magnitude_ceiling), 'w': 0.5, 'm': 4.5, 'a': 0.0}


def _forward(tr):
    """Raw -> display for a Honeychrome Transform (identity without xform)."""
    if tr is None or tr.xform is None:
        return lambda v: np.asarray(v, dtype=float)
    return lambda v: np.asarray(tr.xform.apply(np.asarray(v, dtype=float)), dtype=float)


def _inverse(tr):
    if tr is None or tr.xform is None:
        return lambda v: np.asarray(v, dtype=float)
    return lambda v: np.asarray(tr.xform.inverse(np.asarray(v, dtype=float)), dtype=float)


def _is_linear(tr) -> bool:
    return tr is None or tr.xform is None or tr.id == 0


@dataclass(frozen=True, eq=False)
class RawLayout:
    """Raw channel layout of the experiment, snapshotted on the main thread."""
    pnn_raw: tuple
    whitelisted: bool
    fl_cols: tuple
    detectors: tuple
    scatter_x: str | None
    scatter_y: str | None
    singlet_x: str | None
    singlet_y: str | None
    singlet_auto: bool
    pnr: dict
    magnitude_ceiling: float

    def col(self, channel: str) -> int:
        return self.pnn_raw.index(channel)

    def reader(self):
        return SimpleNamespace(whitelisted=self.whitelisted, pnn_raw=self.pnn_raw)

    def ceiling(self, channel: str) -> float:
        return float(self.pnr.get(channel) or self.magnitude_ceiling)


def _gate_dims(gating, name):
    try:
        return list(gating.get_gate(name).get_dimension_ids())
    except Exception:
        return None


def _raw_layout(controller) -> RawLayout:
    raw = controller.experiment.settings['raw']
    full = list(raw['event_channels_pnn'])
    whitelist = raw.get('whitelisted_pnn') or None
    pnn = list(whitelist or full)
    fl_full = controller.filtered_raw_fluorescence_channel_ids
    fl_cols = tuple(pnn.index(full[i]) for i in fl_full if full[i] in pnn)
    detectors = tuple(pnn[i] for i in fl_cols)

    pnr_values = raw.get('channel_pnr') or []
    ceiling = float(raw.get('magnitude_ceiling') or hc_settings.magnitude_ceiling)
    pnr = {ch: float(v) for ch, v in zip(full, pnr_values)}

    scatter = [ch for ch in (raw.get('scatter_param') or []) if ch in pnn][:2]
    if len(scatter) < 2:
        dims = _gate_dims(controller.raw_gating, 'Cells') if controller.raw_gating else None
        scatter = [ch for ch in (dims or []) if ch in pnn][:2]
    scatter_x, scatter_y = (scatter + [None, None])[:2]

    pair = singlet_channels(raw.get('cytometer', ''))
    singlet_x = singlet_y = None
    auto = False
    if pair is not None and pair[0] in pnn:
        singlet_x = pair[0]
        if pair[1] == 'FSC-H' and 'FSC-H' in pnn:
            singlet_y, auto = 'FSC-H', True
    if not auto:
        dims = _gate_dims(controller.raw_gating, 'Singlets') if controller.raw_gating else None
        if dims and len(dims) == 2 and all(ch in pnn for ch in dims):
            singlet_x, singlet_y = dims
    return RawLayout(pnn_raw=tuple(pnn), whitelisted=whitelist is not None, fl_cols=fl_cols,
                     detectors=detectors, scatter_x=scatter_x, scatter_y=scatter_y,
                     singlet_x=singlet_x, singlet_y=singlet_y, singlet_auto=auto,
                     pnr=pnr, magnitude_ceiling=ceiling)


def _experiment_gate_record(controller, gate_name: str):
    """A Raw Data tab gate as a FlowCode gate record in raw units, or None."""
    gating = controller.raw_gating
    if gating is None:
        return None
    try:
        gate = gating.get_gate(gate_name)
    except Exception:
        return None
    dims = list(gate.get_dimension_ids())
    if len(dims) != 2:
        return None
    transforms = controller.raw_transformations or {}

    def axis_inverse(dim, channel):
        ref = getattr(dim, 'transformation_ref', None)
        if ref is None:
            return lambda v: np.asarray(v, dtype=float), True
        xform = gating.transformations.get(ref)
        if xform is None and channel in transforms:
            xform = transforms[channel].xform
        if xform is None:
            return lambda v: np.asarray(v, dtype=float), True
        tr = transforms.get(channel)
        return (lambda v: np.asarray(xform.inverse(np.asarray(v, dtype=float)), dtype=float)), _is_linear(tr)

    inv_x, lin_x = axis_inverse(gate.dimensions[0], dims[0])
    inv_y, lin_y = axis_inverse(gate.dimensions[1], dims[1])
    if gate.gate_type == 'PolygonGate':
        display = np.asarray(gate.vertices, dtype=float)
    elif gate.gate_type == 'RectangleGate':
        lims = []
        for dim, channel in zip(gate.dimensions, dims):
            tr = transforms.get(channel)
            lo_default, hi_default = (tr.limits if tr is not None else (0.0, 1.0))
            lims.append((lo_default if dim.min is None else dim.min, hi_default if dim.max is None else dim.max))
        display = fcc.rectangle_vertices(lims[0][0], lims[0][1], lims[1][0], lims[1][1])
    else:
        return None
    if not (lin_x and lin_y):
        display = fcc.densify_closed_polygon(display, 64)
    raw = fcc.map_vertices(display, inv_x, inv_y)
    return fcs.gate_record(dims[0], dims[1], raw, 'experiment', f"Copied from the Raw Data tab's {gate_name} gate.")


def _densify_record(record: dict, transforms: dict) -> dict:
    """Densify a straight-edged raw polygon when either axis is non-linear."""
    if _is_linear(transforms.get(record['x'])) and _is_linear(transforms.get(record['y'])):
        return record
    out = dict(record)
    out['vertices_raw'] = fcc.densify_closed_polygon(record['vertices_raw'], 64).tolist()
    return out


# ---------------------------------------------------------------------------
# Background work (no controller access; inputs are snapshotted)
# ---------------------------------------------------------------------------

class _TaskWorker(QObject):
    """Runs ``fn(progress)`` on a worker thread."""
    finished = Signal(str, object)
    error = Signal(str, str)
    progress = Signal(str, int, int, str)

    def __init__(self, key: str, fn):
        super().__init__()
        self._key = key
        self._fn = fn

    def run(self):
        try:
            result = self._fn(lambda n, total, text='': self.progress.emit(self._key, n, total, text))
        except Exception as exc:
            logger.exception(f'FlowCode: {self._key} failed')
            self.error.emit(self._key, str(exc) or type(exc).__name__)
            return
        self.finished.emit(self._key, result)


def match_unstained(experiment_dir, backbone_keys, candidates: list, layout: RawLayout,
                    tr_x, tr_y, max_events: int, bins: int, progress) -> tuple:
    """Bhattacharyya coefficient of each candidate's FSC/SSC histogram with
    the pooled backbones' (*max_events* from each file). Candidates that
    cannot be read score None. Returns (backbone keys, scores)."""
    col_x, col_y = layout.col(layout.scatter_x), layout.col(layout.scatter_y)
    ceilings = np.array([layout.ceiling(layout.scatter_x), layout.ceiling(layout.scatter_y)])
    fx, fy = _forward(tr_x), _forward(tr_y)

    def scatter(key):
        raw = read_raw_events(Path(experiment_dir) / key, layout.reader())
        idx = seeded_subsample_indices(len(raw), max_events, key, seed=42)
        sc = raw[idx][:, [col_x, col_y]]
        return sc[~np.any(sc >= 0.999 * ceilings, axis=1)]

    def histogram(sc):
        return fcc.scatter_histogram(fx(sc[:, 0]), fy(sc[:, 1]), tr_x.limits, tr_y.limits, bins)

    backbone_keys = list(backbone_keys)
    total = len(backbone_keys) + len(candidates)
    pooled = []
    for i, key in enumerate(backbone_keys):
        progress(i, total, 'backbone')
        pooled.append(scatter(key))
    reference = histogram(np.vstack(pooled))
    scores = {}
    for i, key in enumerate(candidates, start=len(backbone_keys)):
        progress(i, total, key)
        try:
            scores[key] = fcc.bhattacharyya(reference, histogram(scatter(key)))
        except Exception as exc:
            logger.warning(f'FlowCode: could not read {key} for unstained matching: {exc}')
            scores[key] = None
    progress(total, total, '')
    return backbone_keys, scores


@dataclass(eq=False)
class BackboneData:
    """Cleaned, pooled backbone and matched unstained events for the
    session; ``n_file`` counts the events in all backbone files."""
    backbone_keys: tuple
    unstained_key: str
    scatter_channels: tuple
    scatter: np.ndarray
    fl: np.ndarray
    n_file: int
    un_scatter: np.ndarray
    un_fl: np.ndarray
    un_n_file: int

    @property
    def n_clean(self) -> int:
        return len(self.scatter)

    @property
    def un_n_clean(self) -> int:
        return len(self.un_scatter)

    def scatter_column(self, channel: str, unstained: bool = False) -> np.ndarray:
        data = self.un_scatter if unstained else self.scatter
        return data[:, self.scatter_channels.index(channel)]


def _read_cleaned(experiment_dir, key: str, layout: RawLayout, scatter_channels: tuple,
                  max_events: int) -> tuple:
    raw = read_raw_events(Path(experiment_dir) / key, layout.reader())
    n_file = len(raw)
    sc_cols = [layout.col(ch) for ch in scatter_channels]
    clean_channels = [ch for ch in (layout.scatter_x, layout.scatter_y) if ch in scatter_channels]
    clean_cols = [scatter_channels.index(ch) for ch in clean_channels]
    fl = raw[:, list(layout.fl_cols)]
    sc = raw[:, sc_cols]
    keep = fcc.clean_events_mask(fl, sc[:, clean_cols], layout.magnitude_ceiling,
                                 [layout.ceiling(ch) for ch in clean_channels])
    kept = np.flatnonzero(keep)
    sub = kept[seeded_subsample_indices(len(kept), max_events, key, seed=42)]
    return np.ascontiguousarray(sc[sub]), np.ascontiguousarray(fl[sub], dtype=np.float32), n_file


def load_backbone(experiment_dir, backbone_keys, unstained_key: str, layout: RawLayout,
                  scatter_channels: tuple, max_events: int, progress) -> BackboneData:
    """Read and clean the backbones (at most *max_events* cleaned events from
    each file, pooled) and the matched unstained."""
    backbone_keys = tuple(backbone_keys)
    total = len(backbone_keys) + 1
    scatter, fl, n_file = [], [], 0
    for i, key in enumerate(backbone_keys):
        progress(i, total, 'backbone' if len(backbone_keys) == 1 else f'backbone {i + 1} of {len(backbone_keys)}')
        sc_k, fl_k, n_k = _read_cleaned(experiment_dir, key, layout, scatter_channels, max_events)
        scatter.append(sc_k)
        fl.append(fl_k)
        n_file += n_k
    sc = scatter[0] if len(scatter) == 1 else np.concatenate(scatter)
    fl = fl[0] if len(fl) == 1 else np.concatenate(fl)
    progress(total - 1, total, 'matched unstained')
    un_sc, un_fl, un_n = _read_cleaned(experiment_dir, unstained_key, layout, scatter_channels, max_events)
    progress(total, total, '')
    return BackboneData(backbone_keys=backbone_keys, unstained_key=unstained_key,
                        scatter_channels=tuple(scatter_channels), scatter=sc, fl=fl, n_file=n_file,
                        un_scatter=un_sc, un_fl=un_fl, un_n_file=un_n)


def auto_singlets_gate(data: BackboneData, layout: RawLayout, quantile: float, transforms: dict) -> dict | None:
    """Area/height ratio singlet gate, or None without an FSC height channel."""
    if not layout.singlet_auto or data.n_clean == 0:
        return None
    a = data.scatter_column(layout.singlet_x)
    h = data.scatter_column(layout.singlet_y)
    cutoff, poly = fcc.auto_singlet_gate(a, h, layout.ceiling(layout.singlet_x),
                                         layout.ceiling(layout.singlet_y), quantile)
    return _densify_record(fcs.gate_record(layout.singlet_x, layout.singlet_y, poly, 'auto',
                                           f'Area/height ratio below {cutoff:.3f} ({quantile:.0%} quantile).'),
                           transforms)


def gate_mask(data: BackboneData, records: dict, gate: str, unstained: bool = False) -> np.ndarray:
    """Events inside *gate* and its parent (FlowCode Cells lies inside
    FlowCode Singlets)."""
    n = data.un_n_clean if unstained else data.n_clean
    mask = np.ones(n, dtype=bool)
    order = fcs.GATE_NAMES[:fcs.GATE_NAMES.index(gate) + 1]
    for name in order:
        rec = records.get(name)
        if rec is None:
            continue
        x = data.scatter_column(rec['x'], unstained)
        y = data.scatter_column(rec['y'], unstained)
        mask &= fcc.points_in_polygon(x, y, rec['vertices_raw'])
    return mask


def auto_cells_gate(data: BackboneData, singlet_record: dict | None, layout: RawLayout,
                    spectra: np.ndarray, tag_rows, settings: dict, transforms: dict) -> tuple:
    """Landmark FSC/SSC gate on the singlets. Returns (record or None,
    landmark raw scatter, reason)."""
    singlets = gate_mask(data, {fcs.SINGLETS_GATE: singlet_record}, fcs.SINGLETS_GATE)
    xy = np.column_stack([data.scatter_column(layout.scatter_x), data.scatter_column(layout.scatter_y)])
    fx, fy = _forward(transforms.get(layout.scatter_x)), _forward(transforms.get(layout.scatter_y))
    result = fcc.auto_landmark_gate(
        np.asarray(data.fl[singlets], dtype=np.float64), xy[singlets], spectra, tag_rows,
        n_per_tag=int(settings['landmark_n_per_tag']), percentile=float(settings['landmark_percentile']),
        min_landmarks=int(settings['landmark_min_events']),
        to_display=lambda v: fcc.map_vertices(v, fx, fy),
    )
    landmarks = xy[singlets][result.landmark_idx]
    if result.vertices is None:
        return None, landmarks, result.reason
    factor = float(settings.get('landmark_expand', 1.0))
    tr_x, tr_y = transforms.get(layout.scatter_x), transforms.get(layout.scatter_y)
    if _is_linear(tr_x) and _is_linear(tr_y):
        vertices = fcc.expand_polygon(result.vertices, factor)
    else:
        # expand on the display scale, keeping its straight edges
        shown = fcc.densify_closed_polygon(fcc.map_vertices(result.vertices, fx, fy), 64)
        vertices = fcc.map_vertices(fcc.expand_polygon(shown, factor), _inverse(tr_x), _inverse(tr_y))
    note = f'{len(result.landmark_idx)} landmark events, {settings["landmark_percentile"]:.0%} contour'
    note += f', enlarged {factor:g}x.' if factor != 1.0 else '.'
    record = fcs.gate_record(layout.scatter_x, layout.scatter_y, vertices, 'auto', note)
    return record, landmarks, None


def compute_debarcode_space(backbone_fl: np.ndarray, unstained_fl: np.ndarray, spectra: np.ndarray,
                            fluor_names, tag_fluors, af_spectra: np.ndarray, pos_thresholds,
                            cell_weight: bool, magnitude_ceiling: float, settings: dict, progress) -> tuple:
    """Debarcoding space of the gated events and the automatic thresholds."""
    progress(0, 2, 'per-cell AF unmixing')
    space = fcc.debarcoding_space(backbone_fl, unstained_fl, spectra, fluor_names, tag_fluors, af_spectra,
                                  pos_thresholds=pos_thresholds, cell_weight=cell_weight)
    progress(1, 2, 'thresholds')
    names = list(fluor_names)
    tag_cols = [names.index(f) for f in tag_fluors]
    thresholds = fcc.initial_thresholds(space.backbone[:, tag_cols], space.floor, magnitude_ceiling,
                                        min_delta_bic=float(settings['threshold_min_delta_bic']),
                                        max_events=int(settings['threshold_max_events']))
    progress(2, 2, '')
    return space, thresholds


@dataclass(eq=False)
class CharacteriseOutput:
    """A section 4 run: the characterisation, the backbone through the final
    unmix without and with correction (rows not shown are NaN), the
    debarcoding counts and the figure data."""
    result: object
    before: dict
    after: dict
    summary: dict
    display_idx: np.ndarray
    expression: dict
    engine: str = ''


def _display_axes(fluor_names, transforms: dict, magnitude_ceiling: float) -> dict:
    """{fluorophore: (forward, limits, major ticks)} from the display
    transforms; the fixed rescue logicle where a channel has none."""
    axes = {}
    for fluor in fluor_names:
        tr = transforms.get(fluor)
        if tr is None or tr.xform is None:
            axes[fluor] = (fcc.logicle_transform(magnitude_ceiling), (0.0, 1.0), [])
            continue
        try:
            ticks = [(float(p), str(label)) for p, label in tr.ticks()[1]]
        except Exception:
            ticks = []
        axes[fluor] = (_forward(tr), tuple(float(v) for v in tr.limits), ticks)
    return axes


def _unmix_rows(final_fn, raw: np.ndarray, rows: np.ndarray, n_fluors: int) -> dict:
    """*final_fn* on *rows* of *raw*, scattered into full-length arrays (NaN
    unmixed and AF abundance, AF index 0 elsewhere)."""
    out = {'unmixed': np.full((len(raw), n_fluors), np.nan), 'af_scale': np.full(len(raw), np.nan),
           'af_idx': np.zeros(len(raw), dtype=np.int64)}
    if rows.size:
        part = final_fn(np.ascontiguousarray(raw[rows], dtype=np.float64))
        out['unmixed'][rows] = part['unmixed']
        out['af_scale'][rows] = part['af_scale']
        out['af_idx'][rows] = np.asarray(part['af_idx'], dtype=np.int64)
    return out


def run_characterisation(raw_fl: np.ndarray, debarcode_unmixed: np.ndarray, unstained_fl: np.ndarray, design,
                         thresholds: np.ndarray, spectra: np.ndarray, fluor_names, detectors,
                         af_mean: np.ndarray, rescue_floor, magnitude_ceiling: float, settings: dict,
                         final_fn, engine: str, transforms: dict, labels: dict, progress,
                         tandems=(), debarcode_af_scale=None) -> CharacteriseOutput:
    """Characterise the gated backbone, then unmix the events the figures
    show (the display subsample and the panel expression groups) with the
    export's final unmix, without and with the FlowCode correction.

    final_fn           : raw fluorescence -> dict(unmixed, af_scale, af_idx).
    tandems            : every usable ``TandemTag``; the backbone statistic of
                         each is the reference for the export.
    debarcode_af_scale : per-cell AF abundance in debarcoding space, for the
                         too-few rescue.
    """
    progress(0, 3, 'debarcoding and fitting the FlowCode error')
    char_settings = fcc.CharacteriseSettings(**{f.name: settings[f.name]
                                                for f in fields(fcc.CharacteriseSettings)})
    result = fcc.characterise_backbone(
        raw_fl, debarcode_unmixed, design, thresholds, spectra, fluor_names, detectors, af_mean,
        fcc.logicle_transform(magnitude_ceiling), _rescue_transform_params(magnitude_ceiling),
        rescue_floor=rescue_floor, unstained_raw=unstained_fl, settings=char_settings, tandems=list(tandems),
        debarcode_af_scale=debarcode_af_scale,
    )
    display_idx = seeded_subsample_indices(len(raw_fl), int(settings['characterise_display_max_events']),
                                           'flowcode_characterise_display', seed=42)
    max_events = int(settings['expression_max_events'])
    negative, barcoded = fcc.expression_event_indices(result.ids, design, max_events)
    rows = np.union1d(np.union1d(display_idx, negative), barcoded)
    progress(1, 3, f'unmixing {rows.size:,} backbone events before and after correction ({engine})')
    before = _unmix_rows(final_fn, raw_fl, rows, len(fluor_names))
    after = _unmix_rows(final_fn, result.corrected_raw, rows, len(fluor_names))
    progress(2, 3, 'preparing the figures')
    summary = fcc.debarcode_summary(result.ids, design, rescued=len(result.rescued),
                                    rescued_toofew=len(result.rescued_toofew))
    expression = fcc.panel_expression_data(before['unmixed'], after['unmixed'], result.ids, design, fluor_names,
                                           _display_axes(fluor_names, transforms, magnitude_ceiling), labels,
                                           max_events=max_events)
    progress(3, 3, '')
    return CharacteriseOutput(result=result, before=before, after=after, summary=summary,
                              display_idx=display_idx, expression=expression, engine=engine)


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

def _fmt_beta(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    return f'{value:+.2e}' if np.isfinite(value) else 'n/a'


def _status_label() -> QLabel:
    label = QLabel('')
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    return label


def _message_label(style: str = _ERROR_STYLE) -> QLabel:
    label = QLabel('')
    label.setWordWrap(True)
    label.setStyleSheet(style)
    label.setVisible(False)
    return label


def _set_message(label: QLabel, lines):
    lines = [str(line) for line in (lines or []) if line]
    label.setText('<br>'.join(lines))
    label.setVisible(bool(lines))


def _no_wheel(*widgets):
    for w in widgets:
        w.installEventFilter(WheelBlocker(w))
        w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)


def _int_spin(lo, hi, step=1) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    _no_wheel(spin)
    return spin


def _float_spin(lo, hi, decimals=2, step=0.05) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    _no_wheel(spin)
    return spin


class _SideTabWidget(QWidget):
    """Vertical navigation list beside a page stack (as in Honeycluster)."""

    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QStackedWidget

        self._nav = QListWidget()
        self._nav.setIconSize(QSize(20, 20))
        self._nav.setSelectionMode(QAbstractItemView.SingleSelection)
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._nav.setFrameShape(QFrame.NoFrame)
        self._nav.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._nav.setStyleSheet('QListWidget::item { padding: 8px 14px 8px 8px; }')
        self._stack = QStackedWidget()

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._nav)
        layout.addWidget(divider)
        layout.addWidget(self._stack, stretch=1)
        self._nav.currentRowChanged.connect(self._on_row_changed)

    def addTab(self, widget: QWidget, label: str, icon_name: str | None = None) -> int:
        index = self._stack.addWidget(widget)
        item = QListWidgetItem(label)
        if icon_name:
            item.setIcon(icon(icon_name))
        self._nav.addItem(item)
        if self._nav.currentRow() < 0:
            self._nav.setCurrentRow(0)
        self._nav.ensurePolished()
        self._nav.setFixedWidth(self._nav.sizeHintForColumn(0) + 2 * self._nav.frameWidth() + 12)
        return index

    def setTabText(self, index: int, text: str):
        item = self._nav.item(index)
        if item is not None:
            item.setText(text)

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int):
        self._nav.setCurrentRow(index)

    def currentWidget(self):
        return self._stack.currentWidget()

    def widget(self, index: int):
        return self._stack.widget(index)

    def count(self) -> int:
        return self._stack.count()

    def _on_row_changed(self, row: int):
        if row < 0:
            self._nav.setCurrentRow(self._stack.currentIndex())
            return
        if row == self._stack.currentIndex():
            return
        self._stack.setCurrentIndex(row)
        self.currentChanged.emit(row)


class _Page(QWidget):
    """
    Common page frame: title, status line and help above either one
    scrollable body or, with ``concertina``, a fixed header (``body``) and
    collapsible sections that fill the rest of the page, one open at a time
    (``add_section``). The open section is remembered between sessions.
    """

    def __init__(self, plugin: 'PluginWidget', section: int, parent=None, concertina: bool = False):
        super().__init__(parent)
        self.plugin = plugin
        self.section = section
        self.toolbox = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        self.body = QVBoxLayout(body)
        self.body.setAlignment(Qt.AlignTop)
        self.body.setSpacing(10)
        if concertina:
            outer.setSpacing(6)
            outer.addWidget(body)
            self.toolbox = QToolBox()
            outer.addWidget(self.toolbox, stretch=1)
            self.toolbox.currentChanged.connect(
                lambda index: plugin.qsettings.setValue(f'open_section_{section}', int(index)))
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(body)
            outer.addWidget(scroll)

        title = QLabel(f'{section}. {fcs.SECTION_TITLES[section]}')
        title.setStyleSheet(hc_settings.heading_style)
        self.body.addWidget(title)
        self.status = _status_label()
        self.body.addWidget(self.status)
        self.body.addWidget(HelpToggleWidget(text=_HELP[section]))

    def add_section(self, title: str) -> QVBoxLayout:
        """A new concertina section, scrolling on its own; returns its layout."""
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        self.toolbox.addItem(scroll, title)
        return lay

    def restore_open_section(self, default: int = 0):
        """Open the section that was open last time (or *default*)."""
        try:
            index = int(self.plugin.qsettings.value(f'open_section_{self.section}', default))
        except (TypeError, ValueError):
            index = default
        if not 0 <= index < self.toolbox.count():
            index = default
        self.toolbox.setCurrentIndex(index)

    def open_section(self, index: int):
        if self.toolbox is not None and 0 <= index < self.toolbox.count():
            self.toolbox.setCurrentIndex(index)

    def set_status(self, kind: str, text: str):
        style = {'done': _DONE_STYLE, 'error': _ERROR_STYLE, 'stale': _WARN_STYLE}.get(kind, _HINT_STYLE)
        self.status.setStyleSheet(style)
        self.status.setText(text)

    def refresh(self):
        pass


# ---------------------------------------------------------------------------
# Page 1: inputs
# ---------------------------------------------------------------------------

class InputsPage(_Page):
    def __init__(self, plugin, parent=None):
        super().__init__(plugin, 1, parent)
        self._filling = False

        self.setup_banner = QLabel(SETUP_BANNER)
        self.setup_banner.setWordWrap(True)
        self.setup_banner.setStyleSheet(_BANNER_STYLE)
        self.body.addWidget(self.setup_banner)

        grp = QGroupBox('Backbone')
        lay = QVBoxLayout(grp)
        hint = QLabel('Tick one or more samples; their events are pooled.')
        hint.setStyleSheet(_HINT_STYLE)
        lay.addWidget(hint)
        self.backbone_list = QListWidget()
        self.backbone_list.setMaximumHeight(160)
        self.backbone_list.itemChanged.connect(self._on_backbone_item_changed)
        lay.addWidget(self.backbone_list)
        self.backbone_msg = _message_label()
        lay.addWidget(self.backbone_msg)
        self.body.addWidget(grp)

        grp = QGroupBox('Matched unstained')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.match_btn = QPushButton('Match again')
        self.match_btn.clicked.connect(lambda: self.plugin.start_matching(force=True))
        self.best_btn = QPushButton('Use best match')
        self.best_btn.clicked.connect(self._on_use_best)
        self.match_status = QLabel('')
        self.match_status.setStyleSheet(_HINT_STYLE)
        row.addWidget(self.match_btn)
        row.addWidget(self.best_btn)
        row.addWidget(self.match_status, stretch=1)
        lay.addLayout(row)
        self.unstained_table = QTableWidget(0, 3)
        self.unstained_table.setHorizontalHeaderLabels(['Sample', 'Similarity (BC)', 'AF profile'])
        self.unstained_table.verticalHeader().setVisible(False)
        self.unstained_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.unstained_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.unstained_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.unstained_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.unstained_table.setMaximumHeight(220)
        self.unstained_table.cellClicked.connect(self._on_unstained_clicked)
        lay.addWidget(self.unstained_table)
        self.unstained_msg = _message_label()
        lay.addWidget(self.unstained_msg)
        self.body.addWidget(grp)

        grp = QGroupBox('AF library for debarcoding')
        lay = QVBoxLayout(grp)
        hint = QLabel('Profiles extracted from the matched unstained are ticked automatically.')
        hint.setStyleSheet(_HINT_STYLE)
        lay.addWidget(hint)
        self.af_list = QListWidget()
        self.af_list.setMaximumHeight(160)
        self.af_list.itemChanged.connect(self._on_af_item_changed)
        lay.addWidget(self.af_list)
        self.af_msg = _message_label()
        lay.addWidget(self.af_msg)
        self.body.addWidget(grp)

        grp = QGroupBox('Combinations CSV (optional)')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        load_btn = QPushButton('Load combinations.csv…')
        load_btn.clicked.connect(self._on_load_csv)
        self.clear_csv_btn = QPushButton('Clear')
        self.clear_csv_btn.clicked.connect(self._on_clear_csv)
        row.addWidget(load_btn)
        row.addWidget(self.clear_csv_btn)
        row.addStretch()
        lay.addLayout(row)
        self.csv_label = QLabel('')
        self.csv_label.setWordWrap(True)
        lay.addWidget(self.csv_label)
        self.csv_msg = _message_label()
        lay.addWidget(self.csv_msg)
        self.body.addWidget(grp)
        self.body.addStretch()

    # -- refresh -------------------------------------------------------------

    def refresh(self):
        c = self.plugin.controller
        state = self.plugin.state
        self._filling = True
        try:
            self.setup_banner.setVisible(not _optimization_setup_done(c))

            chosen = set(state.backbones())
            self.backbone_list.clear()
            for key in fcs.backbone_candidates(c.experiment.samples):
                item = QListWidgetItem(_sample_name(c, key))
                item.setData(Qt.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if key in chosen else Qt.Unchecked)
                self.backbone_list.addItem(item)
            _set_message(self.backbone_msg, [] if self.backbone_list.count() else
                         ['No samples to use as the backbone: every sample is a single-stain control or '
                          'unstained.'])

            self._fill_unstained_table()
            self._fill_af_list()

            if state.csv_combos:
                self.csv_label.setText(f'{Path(state.combos_csv_path or "").name}: '
                                       f'{len(state.csv_combos)} combination(s).')
            else:
                self.csv_label.setText('No file loaded: every 3-tag combination of the ticked tags is used.')
            self.clear_csv_btn.setEnabled(bool(state.csv_combos))
        finally:
            self._filling = False

    def _fill_unstained_table(self):
        c = self.plugin.controller
        state = self.plugin.state
        candidates = fcs.unstained_candidates(c.experiment.samples)
        af_profiles = _process(c).get('af_profiles') or {}
        self.unstained_table.setRowCount(len(candidates))
        selected_row = -1
        for row, key in enumerate(candidates):
            score = state.unstained_scores.get(key) if state.match_backbone_keys == state.backbones() else None
            has_af = bool(fcs.af_profiles_for_sample(af_profiles, key, c.experiment_dir))
            cells = [_sample_name(c, key), '' if score is None else f'{score:.3f}', '✓' if has_af else '✗']
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, key)
                if col > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.unstained_table.setItem(row, col, item)
            if key == state.unstained_key:
                selected_row = row
        self.unstained_table.clearSelection()
        if selected_row >= 0:
            self.unstained_table.selectRow(selected_row)
        problems = []
        if not candidates:
            problems.append('No unstained samples found. Tag the unstained sample in the sample browser, '
                            "or include 'unstained' in its name.")
        elif state.unstained_key is None:
            problems.append('Choose the matched unstained sample.')
        _set_message(self.unstained_msg, problems)

        task = self.plugin.task_running('match')
        self.match_btn.setEnabled(bool(state.backbones()) and bool(candidates) and not task)
        self.best_btn.setEnabled(bool(state.unstained_scores) and not task)
        if task:
            self.match_status.setText('Matching scatter…')
        elif state.unstained_key and not state.unstained_auto:
            self.match_status.setText('Chosen by hand.')
        elif state.unstained_key:
            self.match_status.setText('Best scatter match.')
        else:
            self.match_status.setText('')

    def _fill_af_list(self):
        c = self.plugin.controller
        state = self.plugin.state
        af_profiles = _process(c).get('af_profiles') or {}
        self.af_list.clear()
        for name, entry in af_profiles.items():
            n = len(entry.get('spectra') or [])
            source = Path(str(entry.get('source_fcs') or '')).name
            item = QListWidgetItem(f'{name}  ({n} spectra{", from " + source if source else ""})')
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in state.af_profile_names else Qt.Unchecked)
            self.af_list.addItem(item)
        _set_message(self.af_msg, [self.plugin.af_library_problem()])

    # -- handlers --------------------------------------------------------------

    def _on_backbone_item_changed(self, _item):
        if self._filling:
            return
        keys = []
        for i in range(self.backbone_list.count()):
            item = self.backbone_list.item(i)
            if item.checkState() == Qt.Checked:
                keys.append(item.data(Qt.UserRole))
        state = self.plugin.state
        if keys == state.backbones() and state.backbone_keys is not None:
            return
        state.backbone_keys = keys
        QTimer.singleShot(0, self.plugin.backbone_changed)

    def _on_unstained_clicked(self, row, _col):
        item = self.unstained_table.item(row, 0)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        state = self.plugin.state
        if key == state.unstained_key:
            return
        state.unstained_key = key
        state.unstained_auto = False
        self.plugin.apply_auto_af_profiles(force=True)
        self.plugin.state_changed_later()

    def _on_use_best(self):
        self.plugin.choose_best_unstained()

    def _on_af_item_changed(self, _item):
        if self._filling:
            return
        names = []
        for i in range(self.af_list.count()):
            item = self.af_list.item(i)
            if item.checkState() == Qt.Checked:
                names.append(item.data(Qt.UserRole))
        state = self.plugin.state
        state.af_profile_names = names
        state.af_auto = False
        self.plugin.state_changed_later()

    def _on_load_csv(self):
        c = self.plugin.controller
        start = str(Path(self.plugin.state.combos_csv_path).parent) if self.plugin.state.combos_csv_path \
            else str(c.experiment_dir)
        path, _ = QFileDialog.getOpenFileName(self, 'Load FlowCode combinations', start, 'CSV files (*.csv)')
        if path:
            self.load_csv(path)

    def load_csv(self, path):
        try:
            combos = fcc.read_combinations_csv(path)
            problems = _csv_structure_problems(combos)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            problems = [str(exc)]
        if problems:
            _set_message(self.csv_msg, problems)
            return False
        _set_message(self.csv_msg, [])
        state = self.plugin.state
        state.csv_combos = [[name, list(tags)] for name, tags in combos]
        state.combos_csv_path = str(path)
        state.combos = []
        state.tag_selection = None
        self.plugin.state_changed()
        return True

    def _on_clear_csv(self):
        state = self.plugin.state
        state.csv_combos = None
        state.combos_csv_path = None
        state.combos = []
        state.tag_selection = None
        _set_message(self.csv_msg, [])
        self.plugin.state_changed()


def _csv_structure_problems(combos) -> list:
    """Duplicate Ids or tag triplets in a combinations CSV."""
    problems, names, triplets = [], {}, {}
    for name, tags in combos:
        if len({str(t).upper() for t in tags}) != 3:
            problems.append(f'"{name}" does not have three different tags.')
        if name in names:
            problems.append(f'Id "{name}" is used twice.')
        names[name] = True
        key = fcc.combo_key(tags)
        if key in triplets:
            problems.append(f'"{triplets[key]}" and "{name}" have the same tags.')
        triplets.setdefault(key, name)
    return problems


# ---------------------------------------------------------------------------
# Page 2: tags and combos
# ---------------------------------------------------------------------------

_COMBO_HEADERS = ['Include', 'Name', 'Tag 1', 'Tag 2', 'Tag 3', 'Backbone events']


class TagsPage(_Page):
    def __init__(self, plugin, parent=None):
        super().__init__(plugin, 2, parent)
        self._filling = False
        self._grid_channels = []
        self._checkboxes = {}
        self._combo_signature = None

        grp = QGroupBox('Tag channels')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        auto_btn = QPushButton('Detect tags')
        auto_btn.setToolTip('Tick the channels whose antigen is a FlowCode tag (or that the combinations CSV names).')
        auto_btn.clicked.connect(self._on_auto_tags)
        row.addWidget(auto_btn)
        row.addStretch()
        lay.addLayout(row)
        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setSpacing(4)
        lay.addWidget(grid_widget)
        self.tag_table_holder = QVBoxLayout()
        lay.addLayout(self.tag_table_holder)
        self.tag_table = None
        self.tag_msg = _message_label()
        self.tag_warn = _message_label(_WARN_STYLE)
        lay.addWidget(self.tag_msg)
        lay.addWidget(self.tag_warn)
        self.body.addWidget(grp)

        grp = QGroupBox('Combinations')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        all_btn = QPushButton('Include all')
        all_btn.clicked.connect(lambda: self._set_all_included(True))
        none_btn = QPushButton('Include none')
        none_btn.clicked.connect(lambda: self._set_all_included(False))
        save_btn = QPushButton('Save combinations CSV…')
        save_btn.clicked.connect(self._on_save_csv)
        self.combo_count = QLabel('')
        self.combo_count.setStyleSheet(_HINT_STYLE)
        for w in (all_btn, none_btn, save_btn):
            row.addWidget(w)
        row.addWidget(self.combo_count, stretch=1)
        lay.addLayout(row)
        self.combo_table = QTableWidget(0, len(_COMBO_HEADERS))
        self.combo_table.setHorizontalHeaderLabels(_COMBO_HEADERS)
        self.combo_table.verticalHeader().setVisible(False)
        self.combo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.combo_table.setMinimumHeight(260)
        self.combo_table.itemChanged.connect(self._on_combo_item_changed)
        lay.addWidget(self.combo_table)
        self.combo_msg = _message_label()
        lay.addWidget(self.combo_msg)
        self.body.addWidget(grp)
        self.body.addStretch()

    # -- refresh -------------------------------------------------------------

    def refresh(self):
        self.plugin.sync_derived_state()
        c = self.plugin.controller
        state = self.plugin.state
        channels = _tag_channel_options(c)
        antigens = _antigens(c)
        self._filling = True
        try:
            if channels != self._grid_channels:
                self._rebuild_grid(channels, antigens)
            ticked = set(state.tag_channels())
            for ch, cb in self._checkboxes.items():
                cb.setText(_channel_label(ch, antigens))
                cb.setChecked(ch in ticked)
            self._rebuild_tag_table()
            self._fill_combo_table()
        finally:
            self._filling = False
        errors, warnings = self.plugin.tag_problems()
        _set_message(self.tag_msg, errors)
        _set_message(self.tag_warn, warnings)
        _set_message(self.combo_msg, self.plugin.combo_problems())

    def _rebuild_grid(self, channels, antigens):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes = {}
        for i, ch in enumerate(channels):
            cb = QCheckBox(_channel_label(ch, antigens))
            cb.toggled.connect(lambda checked, channel=ch: self._on_tag_toggled(channel, checked))
            self._checkboxes[ch] = cb
            row, col = divmod(i, 4)
            self.grid.addWidget(cb, row, col)
        self._grid_channels = list(channels)

    def _rebuild_tag_table(self):
        rows = self.plugin.tag_table_rows()
        old = self.tag_table
        self.tag_table = CopyableTableWidget(rows, ['Tag', 'Fluorophore', 'Nearest tag'])
        self.tag_table.setMaximumHeight(min(60 + 28 * len(rows), 320))
        if old is not None:
            self.tag_table_holder.replaceWidget(old, self.tag_table)
            old.setParent(None)
            old.deleteLater()
        else:
            self.tag_table_holder.addWidget(self.tag_table)
        self.tag_table.setVisible(bool(rows))

    def _fill_combo_table(self):
        state = self.plugin.state
        counts = self.plugin.combo_backbone_counts()
        signature = (tuple((row['include'], row['name'], tuple(row['tags'])) for row in state.combos),
                     tuple(sorted(counts.items())))
        if signature != self._combo_signature:
            self.combo_table.setRowCount(len(state.combos))
            for r, row in enumerate(state.combos):
                inc = QTableWidgetItem()
                inc.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                inc.setCheckState(Qt.Checked if row['include'] else Qt.Unchecked)
                self.combo_table.setItem(r, 0, inc)
                name = QTableWidgetItem(row['name'])
                name.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.combo_table.setItem(r, 1, name)
                for k in range(3):
                    tag = QTableWidgetItem(str(row['tags'][k]) if k < len(row['tags']) else '')
                    tag.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    self.combo_table.setItem(r, 2 + k, tag)
                n = counts.get(row['name'])
                cnt = QTableWidgetItem('' if n is None else f'{n:,}')
                cnt.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                cnt.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.combo_table.setItem(r, 5, cnt)
            self._combo_signature = signature
        n_inc = len(state.included_combos())
        self.combo_count.setText(f'{n_inc} of {len(state.combos)} combination(s) included'
                                 + (' (from the combinations CSV).' if state.csv_combos else '.'))

    # -- handlers --------------------------------------------------------------

    def _on_tag_toggled(self, channel, checked):
        if self._filling:
            return
        state = self.plugin.state
        ticked = set(state.tag_channels())
        if checked:
            ticked.add(channel)
        else:
            ticked.discard(channel)
        state.tag_selection = [ch for ch in self._grid_channels if ch in ticked]
        self.plugin.state_changed_later()

    def _on_auto_tags(self):
        self.plugin.state.tag_selection = None
        self.plugin.state_changed()

    def _set_all_included(self, include: bool):
        for row in self.plugin.state.combos:
            row['include'] = include
        self.plugin.state_changed()

    def _on_combo_item_changed(self, item):
        if self._filling:
            return
        state = self.plugin.state
        r = item.row()
        if r >= len(state.combos):
            return
        row = state.combos[r]
        if item.column() == 0:
            row['include'] = item.checkState() == Qt.Checked
        elif item.column() == 1:
            row['name'] = item.text().strip()
        else:
            return
        self.plugin.state_changed_later()

    def _on_save_csv(self):
        try:
            design = self.plugin.design()
        except ValueError as exc:
            _set_message(self.combo_msg, [str(exc)])
            return
        start = str(Path(self.plugin.controller.experiment_dir) / 'combinations.csv')
        path, _ = QFileDialog.getSaveFileName(self, 'Save FlowCode combinations', start, 'CSV files (*.csv)')
        if path:
            fcc.write_combinations_csv(path, design)


# ---------------------------------------------------------------------------
# Page 3: gates and thresholds
# ---------------------------------------------------------------------------

class ScatterPreview(QWidget):
    """Read-only density plot with a polygon outline (the matched unstained
    in FlowCode Cells)."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFixedHeight(24)
        lay.addWidget(self.title)
        self.plot = pg.PlotWidget()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.viewport().installEventFilter(WheelBlocker(self))
        colors = cc.palette[hc_settings.colourmap_name_retrieved]
        cmap = pg.ColorMap(pos=0.9 * np.linspace(0.0, 1.0, len(colors)) ** 2
                           + 0.1 * np.linspace(0.0, 1.0, len(colors)), color=colors)
        lut = cmap.getLookupTable(alpha=True)
        lut[0, 3] = 0
        self.img = pg.ImageItem()
        self.img.setLookupTable(lut)
        self.plot.addItem(self.img)
        self.outline = pg.PlotDataItem(pen=pg.mkPen('#2ca02c', width=2))
        self.plot.addItem(self.outline)
        lay.addWidget(self.plot)
        self.plot.setFixedSize(GATE_PLOT_SIZE, GATE_PLOT_SIZE - 24)
        self.setFixedSize(GATE_PLOT_SIZE, GATE_PLOT_SIZE)

    def show_events(self, x, y, tr_x, tr_y, polygon_display, x_label, y_label):
        fx, fy = _forward(tr_x), _forward(tr_y)
        lim_x, lim_y = list(tr_x.limits), list(tr_y.limits)
        hist, _xe, _ye = np.histogram2d(fx(x), fy(y), bins=256, range=[lim_x, lim_y])
        self.img.setImage(hist)
        from PySide6.QtCore import QRectF
        self.img.setRect(QRectF(lim_x[0], lim_y[0], lim_x[1] - lim_x[0], lim_y[1] - lim_y[0]))
        if polygon_display is not None and len(polygon_display):
            closed = np.vstack([polygon_display, polygon_display[:1]])
            self.outline.setData(closed[:, 0], closed[:, 1])
        else:
            self.outline.setData([], [])
        self.plot.setXRange(*lim_x, padding=0)
        self.plot.setYRange(*lim_y, padding=0)
        self.plot.getAxis('bottom').setTicks(tr_x.ticks())
        self.plot.getAxis('left').setTicks(tr_y.ticks())
        self.plot.setLabel('bottom', x_label)
        self.plot.setLabel('left', y_label)

    def clear(self):
        self.img.clear()
        self.outline.setData([], [])


class GatePanel(QWidget):
    """
    FlowCode Singlets and FlowCode Cells on the cleaned backbone, drawn with
    Honeychrome's CytometryPlotWidget on a private event bus and data
    dictionary, plus the matched unstained in FlowCode Cells.

    Gates are kept in raw units (``records``) so they keep their meaning when
    an axis transform changes; the display vertices are re-derived from them.
    A round handle on FlowCode Cells scales the whole gate about its area
    centroid on the display scale: the gate grows or shrinks by the handle's
    distance from the centre along its starting direction.
    """
    gateEdited = Signal(str)
    gateResetRequested = Signal(str)

    SCALE_LIMITS = (0.05, 20.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bus = None
        self._data = None
        self._plots = []
        self._records = {}
        self._backbone = None
        self._transforms = {}
        self._base_transforms = {}
        self._landmarks = None
        self._landmark_item = None
        self._scale_handle = None
        self._scale_preview = None
        self._scale_anchor = None
        self._placing_handle = False
        self._updating = False
        self._mask_cache = {}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._columns = []
        for _ in range(3):
            col = QVBoxLayout()
            col.setAlignment(Qt.AlignTop)
            holder = QVBoxLayout()
            col.addLayout(holder)
            lay.addLayout(col)
            self._columns.append((col, holder))
        lay.addStretch()
        self.preview = ScatterPreview('Matched unstained in FlowCode Cells')
        self._columns[2][1].addWidget(self.preview)
        self._placeholders = []
        for i in range(2):
            ph = QLabel('Load the backbone to show the gates.')
            ph.setAlignment(Qt.AlignCenter)
            ph.setFixedSize(GATE_PLOT_SIZE, GATE_PLOT_SIZE)
            ph.setStyleSheet(_HINT_STYLE)
            self._columns[i][1].addWidget(ph)
            self._placeholders.append(ph)

    def control_layout(self, index: int) -> QVBoxLayout:
        return self._columns[index][0]

    @property
    def records(self) -> dict:
        return self._records

    # -- data ------------------------------------------------------------------

    def set_backbone(self, data: BackboneData, records: dict, transforms: dict, landmarks=None):
        self._backbone = data
        self._base_transforms = {ch: deepcopy(tr) for ch, tr in transforms.items()}
        self._transforms = {ch: deepcopy(tr) for ch, tr in transforms.items()}
        self._records = {name: dict(rec) for name, rec in records.items()}
        self._mask_cache = {}
        self._landmarks = landmarks
        self._rebuild()

    def set_records(self, records: dict, landmarks=None):
        """Replace the gates; landmark events are kept unless new ones are given."""
        self._records = {name: dict(rec) for name, rec in records.items()}
        self._mask_cache = {}
        if landmarks is not None:
            self._landmarks = landmarks
        self._rebuild()

    def has_backbone(self, data) -> bool:
        return data is not None and self._backbone is data

    def clear(self):
        self._backbone = None
        self._records = {}
        self._mask_cache = {}
        self._clear_plots()
        for ph in self._placeholders:
            ph.setVisible(True)
        self.preview.clear()

    def _cached_mask(self, gate: str, unstained: bool) -> np.ndarray:
        key = (gate, unstained)
        if key not in self._mask_cache:
            self._mask_cache[key] = gate_mask(self._backbone, self._records, gate, unstained=unstained)
        return self._mask_cache[key]

    def masks(self) -> dict:
        """Backbone events in each gate (and its parent)."""
        if self._backbone is None:
            return {}
        return {name: self._cached_mask(name, False) for name in fcs.GATE_NAMES}

    def unstained_mask(self, gate: str = fcs.CELLS_GATE) -> np.ndarray | None:
        if self._backbone is None:
            return None
        return self._cached_mask(gate, True)

    # -- plots -----------------------------------------------------------------

    def _clear_plots(self):
        for plot in self._plots:
            plot.setParent(None)
            plot.deleteLater()
        self._plots = []
        self._landmark_item = None
        self._scale_handle = None
        self._scale_preview = None
        self._scale_anchor = None
        self._data = None
        self._bus = None

    def _rebuild(self):
        self._clear_plots()
        if self._backbone is None or not all(name in self._records for name in fcs.GATE_NAMES):
            return
        data = self._backbone
        channels = list(data.scatter_channels)
        self._bus = EventBus()
        gating = GatingStrategy()
        for ch in channels:
            tr = self._transforms.get(ch)
            if tr is not None and tr.xform is not None:
                gating.transformations[ch] = tr.xform
        path = ('root',)
        for name in fcs.GATE_NAMES:
            rec = self._records[name]
            display = self._display_vertices(rec)
            vertices, dim_x, dim_y = define_polygon_gate([tuple(v) for v in display], rec['x'], rec['y'],
                                                         self._transforms)
            gating.add_gate(fk_gates.PolygonGate(name, [dim_x, dim_y], vertices, use_complement=False),
                            gate_path=path)
            path = path + (name,)
        singlets, cells = self._records[fcs.SINGLETS_GATE], self._records[fcs.CELLS_GATE]
        self._data = {
            'pnn': channels,
            'pnn_labels': {},
            'fluoro_indices': [],
            'lookup_tables': {},
            'event_data': data.scatter,
            'transformations': self._transforms,
            'statistics': {},
            'gating': gating,
            'plots': [
                {'type': 'hist2d', 'channel_x': singlets['x'], 'channel_y': singlets['y'],
                 'source_gate': 'root', 'child_gates': [fcs.SINGLETS_GATE]},
                {'type': 'hist2d', 'channel_x': cells['x'], 'channel_y': cells['y'],
                 'source_gate': fcs.SINGLETS_GATE, 'child_gates': [fcs.CELLS_GATE]},
            ],
            'histograms': [],
            'gate_membership': {},
            'cytometer_db_col': None,
        }
        self._recompute(emit=False)
        for i in range(2):
            plot = _GatePlot(bus=self._bus, mode=_GATE_MODE, n_in_plot_sequence=i,
                             plot=self._data['plots'][i], data_for_cytometry_plots=self._data)
            plot.setFixedSize(GATE_PLOT_SIZE, GATE_PLOT_SIZE)
            plot.gateResetRequested.connect(self.gateResetRequested)
            self._placeholders[i].setVisible(False)
            self._columns[i][1].addWidget(plot)
            self._plots.append(plot)
        self._landmark_item = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(255, 59, 48, 170))
        self._landmark_item.setZValue(40)
        self._plots[1].vb.addItem(self._landmark_item)
        self._scale_preview = pg.PlotDataItem(pen=pg.mkPen('#ffcc00', width=1.5, style=Qt.DashLine))
        self._scale_preview.setZValue(55)
        self._plots[1].vb.addItem(self._scale_preview)
        self._scale_handle = pg.TargetItem(size=13, symbol='o', movable=True, pen=pg.mkPen('#1a1a1a', width=1.5),
                                           brush=pg.mkBrush('#ffcc00'), hoverBrush=pg.mkBrush('#ffe680'))
        self._scale_handle.setZValue(60)
        self._scale_handle.setToolTip('Drag to enlarge or shrink the whole gate about its centre.')
        self._scale_handle.sigPositionChanged.connect(self._on_scale_handle_moved)
        self._scale_handle.sigPositionChangeFinished.connect(self._on_scale_handle_released)
        self._plots[1].vb.addItem(self._scale_handle)
        self._bus.changedGatingHierarchy.connect(self._on_gating_changed)
        self._bus.axisTransformed.connect(self._on_axis_transformed)
        self._bus.axesReset.connect(self._on_axes_reset)
        self._bus.histsStatsRecalculated.emit(_GATE_MODE, [0, 1])
        self._update_overlays()

    def _display_vertices(self, rec: dict) -> np.ndarray:
        return fcc.map_vertices(rec['vertices_raw'], _forward(self._transforms.get(rec['x'])),
                                _forward(self._transforms.get(rec['y'])))

    def _recompute(self, emit: bool = True):
        data = self._data
        if data is None:
            return
        masks = self.masks()
        data['gate_membership'] = {'root': np.ones(self._backbone.n_clean, dtype=bool), **masks}
        data['statistics'] = calc_stats(data, initialise=True)
        data['histograms'] = calc_hists(data, density_cutoff=hc_settings.density_cutoff_retrieved)
        if emit and self._bus is not None:
            self._bus.histsStatsRecalculated.emit(_GATE_MODE, [0, 1])
            self._update_overlays()

    def _update_overlays(self):
        self._place_scale_handle()
        if self._landmark_item is not None:
            cells = self._records[fcs.CELLS_GATE]
            if self._landmarks is not None and len(self._landmarks):
                pts = fcc.map_vertices(self._landmarks, _forward(self._transforms.get(cells['x'])),
                                       _forward(self._transforms.get(cells['y'])))
                self._landmark_item.setData(pts[:, 0], pts[:, 1])
            else:
                self._landmark_item.setData([], [])
        self._update_preview()

    def _update_preview(self):
        data = self._backbone
        if data is None or data.un_n_clean == 0:
            self.preview.clear()
            return
        cells = self._records[fcs.CELLS_GATE]
        singlets = self.unstained_mask(fcs.SINGLETS_GATE)
        x = data.scatter_column(cells['x'], unstained=True)[singlets]
        y = data.scatter_column(cells['y'], unstained=True)[singlets]
        self.preview.show_events(x, y, self._transforms[cells['x']], self._transforms[cells['y']],
                                 self._display_vertices(cells), cells['x'], cells['y'])

    # -- scale handle ----------------------------------------------------------

    def _axis_spans(self, rec: dict) -> np.ndarray:
        """Display-axis spans of a gate's channels, to measure distances as seen."""
        spans = []
        for ch in (rec['x'], rec['y']):
            tr = self._transforms.get(ch)
            lo, hi = (tr.limits if tr is not None else (0.0, 1.0))
            spans.append(max(float(hi) - float(lo), 1e-12))
        return np.array(spans)

    def _place_scale_handle(self):
        """Put the handle on the FlowCode Cells vertex farthest from the gate's
        centre (as displayed) and forget any preview."""
        if self._scale_handle is None or fcs.CELLS_GATE not in self._records:
            return
        rec = self._records[fcs.CELLS_GATE]
        shown = self._display_vertices(rec)
        if len(shown) < 3 or not np.all(np.isfinite(shown)):
            self._scale_handle.setVisible(False)
            self._scale_anchor = None
            return
        centre = fcc.polygon_centroid(shown)
        spans = self._axis_spans(rec)
        far = shown[int(np.argmax(np.linalg.norm((shown - centre) / spans, axis=1)))]
        self._scale_anchor = (centre, far, spans)
        self._placing_handle = True
        try:
            self._scale_handle.setPos(float(far[0]), float(far[1]))
        finally:
            self._placing_handle = False
        self._scale_handle.setVisible(True)
        if self._scale_preview is not None:
            self._scale_preview.setData([], [])

    def _scale_factor(self) -> float | None:
        """Gate scale implied by the handle: its distance from the centre
        along the starting direction, relative to the starting distance."""
        if self._scale_anchor is None or self._scale_handle is None:
            return None
        centre, far, spans = self._scale_anchor
        pos = self._scale_handle.pos()
        ray = (far - centre) / spans
        moved = (np.array([pos.x(), pos.y()]) - centre) / spans
        length = float(ray @ ray)
        if length <= 0:
            return None
        return float(np.clip((moved @ ray) / length, *self.SCALE_LIMITS))

    def _on_scale_handle_moved(self, *_args):
        if self._placing_handle or self._scale_preview is None:
            return
        factor = self._scale_factor()
        if factor is None:
            return
        centre = self._scale_anchor[0]
        shown = self._display_vertices(self._records[fcs.CELLS_GATE])
        preview = centre + factor * (shown - centre)
        closed = np.vstack([preview, preview[:1]])
        self._scale_preview.setData(closed[:, 0], closed[:, 1])

    def _on_scale_handle_released(self, *_args):
        if self._placing_handle:
            return
        factor = self._scale_factor()
        if factor is None or abs(factor - 1.0) < 1e-3:
            self._place_scale_handle()
            return
        self.scale_cells_gate(factor)

    def scale_cells_gate(self, factor: float):
        """Scale FlowCode Cells about its centroid on the display scale."""
        rec = self._records.get(fcs.CELLS_GATE)
        if rec is None or self._data is None:
            return
        scaled = fcc.expand_polygon(self._display_vertices(rec), factor)
        raw = fcc.map_vertices(scaled, _inverse(self._transforms.get(rec['x'])),
                               _inverse(self._transforms.get(rec['y'])))
        self._records[fcs.CELLS_GATE] = fcs.gate_record(rec['x'], rec['y'], raw, 'manual',
                                                        f'Scaled {factor:.2f}x by hand.')
        self._mask_cache = {}
        self._reproject()
        self.gateEdited.emit(fcs.CELLS_GATE)

    # -- private bus -----------------------------------------------------------

    def _on_gating_changed(self, mode, gate_name):
        if mode != _GATE_MODE or self._updating or self._data is None:
            return
        gating = self._data['gating']
        names = {g[0] for g in gating.get_gate_ids()}
        if not set(fcs.GATE_NAMES) <= names:
            QTimer.singleShot(0, self._rebuild)
            return
        if gate_name not in fcs.GATE_NAMES:
            return
        rec = self._records[gate_name]
        display = np.asarray(gating.get_gate(gate_name).vertices, dtype=float)
        raw = fcc.map_vertices(display, _inverse(self._transforms.get(rec['x'])),
                               _inverse(self._transforms.get(rec['y'])))
        self._records[gate_name] = fcs.gate_record(rec['x'], rec['y'], raw, 'manual', 'Edited by hand.')
        self._mask_cache = {}
        self._recompute()
        self.gateEdited.emit(gate_name)

    def _reproject(self):
        gating = self._data['gating']
        for ch, tr in self._transforms.items():
            if tr is not None and tr.xform is not None:
                gating.transformations[ch] = tr.xform
        for name in fcs.GATE_NAMES:
            gating.get_gate(name).vertices = [tuple(v) for v in self._display_vertices(self._records[name])]
        self._updating = True
        try:
            for i in range(len(self._plots)):
                self._bus.updateRois.emit(_GATE_MODE, i)
        finally:
            self._updating = False
        self._recompute()

    def _on_axis_transformed(self, channel):
        if self._data is None or channel is None:
            return
        self._reproject()

    def _on_axes_reset(self, channels):
        if self._data is None:
            return
        for ch in channels:
            if ch in self._base_transforms:
                self._transforms[ch] = deepcopy(self._base_transforms[ch])
        self._reproject()


class ThresholdPanel(QWidget):
    """Tag list, debarcoding-space biplot with a threshold line, and the
    threshold table."""
    thresholdChanged = Signal(str, float)
    yChannelChanged = Signal(str, str)
    autoAllRequested = Signal()
    resetRequested = Signal(str)
    importRequested = Signal()

    HEADERS = ['Tag', 'Fluorophore', 'Y channel', 'Threshold (raw)', 'Threshold (logicle)', 'Method', 'ΔBIC']

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._tags = []
        self._thresholds = {}
        self._y = {}
        self._forward_logicle = None
        self._filling = False
        self._initialised = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.tag_list = QListWidget()
        self.tag_list.setFixedWidth(150)
        self.tag_list.setMaximumHeight(THRESHOLD_PLOT_SIZE)
        self.tag_list.currentRowChanged.connect(self._on_tag_selected)
        top.addWidget(self.tag_list, alignment=Qt.AlignTop)
        self.biplot = _ThresholdBiplot(controller)
        self.biplot.setFixedSize(THRESHOLD_PLOT_SIZE, THRESHOLD_PLOT_SIZE)
        self.biplot.thresholdMoved.connect(self._on_line_moved)
        self.biplot.channelChanged.connect(self._on_channels_changed)
        top.addWidget(self.biplot, alignment=Qt.AlignTop)
        top.addStretch()
        lay.addLayout(top)

        row = QHBoxLayout()
        auto_btn = QPushButton('Auto-set all')
        auto_btn.clicked.connect(self.autoAllRequested)
        reset_btn = QPushButton('Reset selected to auto')
        reset_btn.clicked.connect(lambda: self.resetRequested.emit(self.current_tag() or ''))
        import_btn = QPushButton('Import thresholds CSV…')
        import_btn.setToolTip('Set the thresholds from a FlowCodeUnmix thresholds file (Fluor, Tag, Threshold_Raw).')
        import_btn.clicked.connect(self.importRequested)
        row.addWidget(auto_btn)
        row.addWidget(reset_btn)
        row.addWidget(import_btn)
        row.addStretch()
        lay.addLayout(row)
        self.import_msg = QLabel('')
        self.import_msg.setWordWrap(True)
        self.import_msg.setVisible(False)
        lay.addWidget(self.import_msg)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(200)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.currentCellChanged.connect(lambda r, *_: self.tag_list.setCurrentRow(r))
        lay.addWidget(self.table)

    def current_tag(self) -> str | None:
        row = self.tag_list.currentRow()
        return self._tags[row][0] if 0 <= row < len(self._tags) else None

    def set_data(self, event_data: np.ndarray, tags: list, thresholds: dict, y_channels: dict,
                 magnitude_ceiling: float):
        """tags: [(tag, channel)], channel = fluorophore label."""
        self._tags = list(tags)
        self._thresholds = thresholds
        self._y = y_channels
        self._forward_logicle = fcc.logicle_transform(magnitude_ceiling)
        self.biplot.set_event_data(event_data)
        current = self.current_tag()
        self.tag_list.blockSignals(True)
        self.tag_list.clear()
        for tag, channel in self._tags:
            self.tag_list.addItem(f'{tag}  ({channel})')
        self.tag_list.blockSignals(False)
        names = [t for t, _ in self._tags]
        row = names.index(current) if current in names else 0
        self._initialised = False
        self.tag_list.blockSignals(True)
        self.tag_list.setCurrentRow(row)
        self.tag_list.blockSignals(False)
        self._on_tag_selected(row)
        self.update_table()

    def clear(self):
        self.biplot.clear()
        self.tag_list.clear()
        self.table.setRowCount(0)
        self._tags = []

    def update_table(self):
        self._filling = True
        try:
            self.table.setRowCount(len(self._tags))
            for r, (tag, channel) in enumerate(self._tags):
                rec = self._thresholds.get(tag) or {}
                raw = rec.get('raw')
                shown = '' if raw is None or self._forward_logicle is None \
                    else f'{float(self._forward_logicle(np.array([raw]))[0]):.3f}'
                dbic = rec.get('delta_bic')
                cells = [tag, channel, self._y.get(tag, ''), '' if raw is None else f'{raw:.1f}', shown,
                         rec.get('method', ''), '' if dbic is None else f'{dbic:.1f}']
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                    if col == 3:
                        flags |= Qt.ItemIsEditable
                    item.setFlags(flags)
                    if col >= 3:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(r, col, item)
        finally:
            self._filling = False
        tag = self.current_tag()
        if tag is not None:
            self.biplot.set_threshold((self._thresholds.get(tag) or {}).get('raw'))

    def _on_tag_selected(self, row):
        if not (0 <= row < len(self._tags)):
            return
        tag, channel = self._tags[row]
        y = self._y.get(tag) or channel
        if not self._initialised:
            self.biplot.initialise_from_controller(channel, y)
            self._initialised = True
        else:
            self.biplot.blockSignals(True)
            self.biplot.set_channels(channel, y)
            self.biplot.blockSignals(False)
        self.biplot.set_threshold((self._thresholds.get(tag) or {}).get('raw'))
        self.biplot.redraw()
        self.table.blockSignals(True)
        self.table.selectRow(row)
        self.table.blockSignals(False)

    def _on_line_moved(self, raw):
        tag = self.current_tag()
        if tag is not None:
            self.thresholdChanged.emit(tag, float(raw))

    def _on_channels_changed(self, _x, y):
        tag = self.current_tag()
        if tag is not None:
            self.yChannelChanged.emit(tag, y)

    def _on_table_item_changed(self, item):
        if self._filling or item.column() != 3:
            return
        tag = self._tags[item.row()][0]
        try:
            value = float(item.text().replace(',', ''))
        except ValueError:
            QTimer.singleShot(0, self.update_table)
            return
        QTimer.singleShot(0, lambda: self.thresholdChanged.emit(tag, value))


class GatesPage(_Page):
    SECTION_GATES, SECTION_THRESHOLDS, SECTION_ADVANCED = range(3)

    def __init__(self, plugin, parent=None):
        super().__init__(plugin, 3, parent, concertina=True)
        _ensure_qt_imports()

        grp = QGroupBox('Backbone and matched unstained')
        lay = QHBoxLayout(grp)
        self.load_btn = QPushButton('Load events')
        self.load_btn.clicked.connect(lambda: self.plugin.load_backbone())
        self.load_status = QLabel('')
        self.load_status.setWordWrap(True)
        lay.addWidget(self.load_btn)
        lay.addWidget(self.load_status, stretch=1)
        self.body.addWidget(grp)

        lay = self.add_section('FlowCode scatter gates')
        self.panel = GatePanel()
        self.panel.gateEdited.connect(self.plugin.on_gate_edited)
        self.panel.gateResetRequested.connect(self.plugin.on_gate_reset_requested)
        lay.addWidget(self.panel)
        self.gate_controls = {}
        for i, name in enumerate(fcs.GATE_NAMES):
            box = self.panel.control_layout(i)
            row = QHBoxLayout()
            auto_btn = QPushButton('Auto')
            auto_btn.clicked.connect(lambda _=False, n=name: self.plugin.auto_gate(n))
            exp_btn = QPushButton('Use experiment gate')
            exp_btn.clicked.connect(lambda _=False, n=name: self.plugin.use_experiment_gate(n))
            row.addWidget(auto_btn)
            row.addWidget(exp_btn)
            row.addStretch()
            box.addLayout(row)
            note = QLabel('')
            note.setWordWrap(True)
            note.setMaximumWidth(GATE_PLOT_SIZE)
            note.setStyleSheet(_HINT_STYLE)
            box.addWidget(note)
            self.gate_controls[name] = (auto_btn, exp_btn, note)
        self.counts = QLabel('')
        self.counts.setWordWrap(True)
        self.counts.setMaximumWidth(GATE_PLOT_SIZE)
        self.panel.control_layout(2).addWidget(self.counts)
        row = QHBoxLayout()
        self.confirm_btn = QPushButton('Confirm gates')
        self.confirm_btn.setToolTip('Unmix the gated events in the debarcoding space and set starting thresholds.')
        self.confirm_btn.clicked.connect(lambda: self.plugin.compute_debarcode_space(confirm=True))
        self.confirm_status = QLabel('')
        self.confirm_status.setWordWrap(True)
        row.addWidget(self.confirm_btn)
        row.addWidget(self.confirm_status, stretch=1)
        lay.addLayout(row)

        lay = self.add_section('Thresholds')
        self.threshold_msg = QLabel('Confirm the gates to set thresholds.')
        self.threshold_msg.setStyleSheet(_HINT_STYLE)
        lay.addWidget(self.threshold_msg)
        self.thresholds = ThresholdPanel(plugin.controller)
        self.thresholds.thresholdChanged.connect(self.plugin.on_threshold_changed)
        self.thresholds.yChannelChanged.connect(self.plugin.on_y_channel_changed)
        self.thresholds.autoAllRequested.connect(self.plugin.on_thresholds_auto_all)
        self.thresholds.resetRequested.connect(self.plugin.on_threshold_reset)
        self.thresholds.importRequested.connect(self._on_import_thresholds)
        lay.addWidget(self.thresholds)

        self._build_advanced(self.add_section('Advanced settings'))
        self.restore_open_section()

    def _build_advanced(self, lay):
        form = QFormLayout()
        lay.addLayout(form)
        self.adv = {
            'backbone_max_events': _int_spin(10_000, 20_000_000, 100_000),
            'unstained_max_events': _int_spin(1_000, 2_000_000, 10_000),
            'singlet_quantile': _float_spin(0.5, 0.999, 3, 0.01),
            'landmark_n_per_tag': _int_spin(10, 10_000, 10),
            'landmark_percentile': _float_spin(0.1, 0.99, 2, 0.05),
            'landmark_expand': _float_spin(1.0, 3.0, 2, 0.1),
            'threshold_max_events': _int_spin(1_000, 1_000_000, 10_000),
            'threshold_min_delta_bic': _float_spin(0.0, 1000.0, 1, 1.0),
        }
        labels = {
            'backbone_max_events': 'Backbone events read per file (max):',
            'unstained_max_events': 'Gated unstained events (max):',
            'singlet_quantile': 'Singlet ratio quantile:',
            'landmark_n_per_tag': 'Landmark events per tag:',
            'landmark_percentile': 'Landmark contour:',
            'landmark_expand': 'Enlarge the landmark gate (x):',
            'threshold_max_events': 'Threshold events (max):',
            'threshold_min_delta_bic': 'Bimodality ΔBIC:',
        }
        tips = {
            'backbone_max_events': 'Cleaned events read from each backbone file before gating; the files are '
                                   'pooled. Every combination needs enough gated events with all three tags '
                                   'clearly positive to be characterised, so read whole files when memory '
                                   'allows (about 300 MB per million events on a 64-detector cytometer).',
            'landmark_expand': 'Each vertex of the automatic FlowCode Cells gate is moved this many times as far '
                               'from the gate centre. Applies the next time Auto is pressed.',
        }
        for key, widget in self.adv.items():
            form.addRow(labels[key], widget)
            if key in tips:
                widget.setToolTip(tips[key])
            signal = widget.valueChanged
            signal.connect(lambda value, k=key: self.plugin.on_setting_changed(k, value))
        self.cell_weight = QCheckBox('cell.weight')
        self.cell_weight.toggled.connect(lambda c: self.plugin.on_setting_changed('cell_weight', bool(c)))
        form.addRow(self.cell_weight)

    def _on_import_thresholds(self):
        c = self.plugin.controller
        start = str(Path(c.experiment_dir) / OUTPUT_SUBDIR)
        path, _ = QFileDialog.getOpenFileName(self, 'Import FlowCode thresholds', start, 'CSV files (*.csv)')
        if path:
            self.plugin.import_thresholds(path)

    def refresh(self):
        plugin = self.plugin
        state = plugin.state
        for key, widget in self.adv.items():
            widget.blockSignals(True)
            widget.setValue(state.settings[key])
            widget.blockSignals(False)
        self.cell_weight.blockSignals(True)
        self.cell_weight.setChecked(plugin.cell_weight())
        self.cell_weight.blockSignals(False)

        ready = plugin.section_ready(3)
        busy = plugin.task_running('load') or plugin.task_running('space') or plugin.task_running('auto_cells')
        self.load_btn.setEnabled(ready and not busy)
        data = plugin.backbone_data
        if busy:
            pass
        elif data is None:
            self.load_status.setText('' if ready else 'Complete sections 1 and 2 first.')
        else:
            names = ', '.join(_sample_name(plugin.controller, key) for key in data.backbone_keys)
            self.load_status.setText(
                f'{names}: {data.n_clean:,} cleaned events (of {data.n_file:,}). '
                f'{_sample_name(plugin.controller, data.unstained_key)}: '
                f'{data.un_n_clean:,} (of {data.un_n_file:,}).')
        have_gates = data is not None and all(n in state.gates for n in fcs.GATE_NAMES)
        for name, (auto_btn, exp_btn, note) in self.gate_controls.items():
            rec = state.gates.get(name) or {}
            auto_ok = name != fcs.SINGLETS_GATE or plugin.raw_layout().singlet_auto
            auto_btn.setEnabled(have_gates and auto_ok and not busy)
            exp_btn.setEnabled(have_gates and not busy)
            note.setText(rec.get('note', '') if have_gates else '')
        self.confirm_btn.setEnabled(have_gates and not busy)
        self.update_counts()
        space_ok = plugin.debarcode_space_current()
        self.thresholds.setEnabled(space_ok)
        if space_ok:
            self.threshold_msg.setText(plugin.space_summary())
        elif plugin.debarcode_space is not None:
            self.threshold_msg.setText('The gates changed: confirm them again to update the debarcoding space.')
        else:
            self.threshold_msg.setText('Confirm the gates to set thresholds.')

    def update_counts(self):
        data = self.plugin.backbone_data
        masks = self.panel.masks()
        if data is None or not masks:
            self.counts.setText('')
            return
        n_s, n_c = int(masks[fcs.SINGLETS_GATE].sum()), int(masks[fcs.CELLS_GATE].sum())
        un = self.panel.unstained_mask()
        n_u = int(un.sum()) if un is not None else 0
        pct = lambda a, b: f'{100.0 * a / b:.1f}%' if b else '–'  # noqa: E731
        self.counts.setText(
            f'Backbone: {n_s:,} singlets ({pct(n_s, data.n_clean)}), {n_c:,} in FlowCode Cells '
            f'({pct(n_c, data.n_clean)}).<br>Matched unstained: {n_u:,} in FlowCode Cells '
            f'({pct(n_u, data.un_n_clean)}).')
        self.counts.setTextFormat(Qt.RichText)


# ---------------------------------------------------------------------------
# Pages 4 and 5
# ---------------------------------------------------------------------------

def _figure_canvas(fig):
    """Qt canvas for *fig* that passes wheel events on to the page's scroll area."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class _Canvas(FigureCanvasQTAgg):
        def wheelEvent(self, event):
            event.ignore()

    return _Canvas(fig)


class FigurePanel(QWidget):
    """A matplotlib figure shown at its own size, with an export button that
    saves it in the experiment's FlowCode folder."""

    def __init__(self, name: str, plugin: 'PluginWidget', parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.name = name
        self.figure = None
        self.canvas = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.holder = QVBoxLayout()
        lay.addLayout(self.holder)
        row = QHBoxLayout()
        self.export_btn = QPushButton('Export graphic')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        self.export_msg = QLabel('')
        self.export_msg.setStyleSheet(_HINT_STYLE)
        row.addWidget(self.export_btn)
        row.addWidget(self.export_msg, stretch=1)
        lay.addLayout(row)

    def set_figure(self, fig, exportable: bool = True):
        self._drop_canvas()
        self.figure = fig
        self.canvas = _figure_canvas(fig)
        width, height = fig.get_size_inches() * fig.dpi
        self.canvas.setFixedSize(int(round(width)), int(round(height)))
        self.holder.addWidget(self.canvas, alignment=Qt.AlignLeft)
        self.canvas.draw_idle()
        self.export_btn.setEnabled(exportable)
        self.export_msg.setText('')

    def clear(self):
        self._drop_canvas()
        self.figure = None
        self.export_btn.setEnabled(False)
        self.export_msg.setText('')

    def _drop_canvas(self):
        if self.canvas is not None:
            self.holder.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None

    def export(self):
        if self.figure is None:
            return
        folder = Path(self.plugin.controller.experiment_dir) / OUTPUT_SUBDIR
        path = folder / f'{self.name}.{hc_settings.graphics_export_format_retrieved}'
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(path, bbox_inches='tight', facecolor=self.figure.get_facecolor())
        except (OSError, ValueError) as exc:
            self.export_msg.setText(f'Could not export: {exc}')
            return
        self.export_msg.setText(f'Saved {path}')


def _table_value(value):
    """Fit-table cell: numbers to 4 significant digits, NaN blank."""
    if isinstance(value, (float, np.floating)):
        return '' if not np.isfinite(value) else float(f'{float(value):.4g}')
    return value


FIT_TABLE_HEADERS = ['Combo', 'Tags', 'Error', 'Confident', 'Fit', 'Kappa', 'In span', 'Out of span',
                     'Curvature', 'AF slope', 'AF excess', 'AF confound', 'Off RMS before', 'Off RMS after',
                     'Off spill before', 'Off spill after', 'Held-out n', 'Held-out RMS before',
                     'Held-out RMS after']


class CharacterisePage(_Page):
    """Section 4: backbone characterisation and its figures."""

    SECTION_FREQUENCY, SECTION_HEATMAP, SECTION_EXPRESSION, SECTION_ASSESS, SECTION_FIT, SECTION_ADVANCED = range(6)

    def __init__(self, plugin, parent=None):
        super().__init__(plugin, 4, parent, concertina=True)
        _ensure_qt_imports()
        self._shown_key = None
        self._heat_key = None
        self._expression_key = None

        grp = QGroupBox('Characterisation')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.run_btn = QPushButton('Debarcode and identify FRET spillover')
        self.run_btn.clicked.connect(lambda: self.plugin.characterise())
        self.run_status = QLabel('')
        self.run_status.setWordWrap(True)
        row.addWidget(self.run_btn)
        row.addWidget(self.run_status, stretch=1)
        lay.addLayout(row)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        lay.addWidget(self.summary)
        self.body.addWidget(grp)

        lay = self.add_section('Combo frequencies')
        self.frequency = FigurePanel('FlowCode combo frequencies', plugin)
        lay.addWidget(self.frequency)

        lay = self.add_section('FRET and spillover error')
        row = QHBoxLayout()
        row.addWidget(QLabel('Show:'))
        self.heat_space = QComboBox()
        self.heat_space.addItem('Unmixed space', 'unmixed')
        self.heat_space.addItem('Detector space', 'detector')
        _no_wheel(self.heat_space)
        self.heat_space.currentIndexChanged.connect(lambda _i: self.show_results())
        row.addWidget(self.heat_space)
        row.addStretch()
        lay.addLayout(row)
        self.heatmap = FigurePanel('FlowCode spillover', plugin)
        lay.addWidget(self.heatmap)

        lay = self.add_section('Panel expression before and after correction')
        self.expression = FigurePanel('FlowCode panel expression', plugin)
        lay.addWidget(self.expression)

        lay = self.add_section('Assess correction')
        self.biplot_msg = QLabel('')
        self.biplot_msg.setWordWrap(True)
        self.biplot_msg.setStyleSheet(_HINT_STYLE)
        lay.addWidget(self.biplot_msg)
        row = QHBoxLayout()
        self.highlight_check = QCheckBox('Draw barcoded events as points')
        self.highlight_check.setToolTip('Barcoded events are often rare in the backbone; points keep them '
                                        'visible whatever the axis scaling.')
        self.highlight_size = _int_spin(1, 12)
        self.highlight_size.setSuffix(' px')
        settings = plugin.qsettings
        self.highlight_check.setChecked(str(settings.value('highlight_barcoded', 'true')).lower() == 'true')
        try:
            self.highlight_size.setValue(int(settings.value('highlight_size', 4)))
        except (TypeError, ValueError):
            self.highlight_size.setValue(4)
        self.highlight_check.toggled.connect(self._on_highlight_changed)
        self.highlight_size.valueChanged.connect(self._on_highlight_changed)
        row.addWidget(self.highlight_check)
        row.addWidget(QLabel('Size:'))
        row.addWidget(self.highlight_size)
        row.addStretch()
        lay.addLayout(row)
        row = QHBoxLayout()
        self.biplot_before = _TagBiplot('Uncorrected', plugin.controller)
        self.biplot_after = _TagBiplot('Corrected', plugin.controller)
        for source, target in ((self.biplot_before, self.biplot_after), (self.biplot_after, self.biplot_before)):
            source.setFixedSize(ASSESS_PLOT_SIZE, ASSESS_PLOT_SIZE)
            source.channelChanged.connect(lambda x, y, t=target: self._mirror_channels(t, x, y))
            source.scalingChanged.connect(lambda axis, tr, t=target: t.set_scaling(axis, tr))
            row.addWidget(source, alignment=Qt.AlignTop)
        row.addStretch()
        lay.addLayout(row)

        lay = self.add_section('Fit diagnostics')
        hint = QLabel('Kappa: condition number of the on-tag fit. In/Out of span: error inside and outside the '
                      'space the reference spectra can unmix, relative to the on-tag spectra. Off RMS/spill: '
                      'median off-target abundance and largest off-target slope before and after correction, '
                      'on the fit events and on held-out (rescued) events.')
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_STYLE)
        lay.addWidget(hint)
        self.table_holder = QVBoxLayout()
        lay.addLayout(self.table_holder)
        self.table = None

        self._build_advanced(self.add_section('Advanced settings'))
        self.restore_open_section(self.SECTION_FREQUENCY)

    def _build_advanced(self, lay):
        form = QFormLayout()
        lay.addLayout(form)
        self.checks = {
            'rescue_toomany': QCheckBox('Rescue events with more than three positive tags (rescue.toomany)'),
            'rescue_toofew': QCheckBox('Rescue events with one or two positive tags'),
        }
        self.adv = {
            'rescue_flat_threshold': _float_spin(0.0, 1.0, 2, 0.05),
            'rescue_gate_frac': _float_spin(0.0, 1.0, 2, 0.05),
            'rescue_toofew_flat_threshold': _float_spin(0.0, 1.0, 2, 0.05),
            'rescue_toofew_gate_frac': _float_spin(0.0, 1.0, 2, 0.05),
            'rescue_toofew_min_z': _float_spin(0.0, 50.0, 1, 0.5),
            'rescue_toofew_min_frac': _float_spin(0.0, 1.0, 2, 0.05),
            'rescue_toofew_af_ratio': _float_spin(0.0, 1000.0, 2, 0.25),
            'fit_min_fold': _float_spin(1.0, 100.0, 1, 0.5),
            'min_cells_per_combo': _int_spin(5, 1_000_000, 10),
            'spill_ridge': _float_spin(0.0, 1.0, 3, 0.01),
        }
        labels = {
            'rescue_flat_threshold': 'Rescue minimum similarity (rescue.flat.threshold):',
            'rescue_gate_frac': 'Rescue margin over the runner-up (rescue.gate.frac):',
            'rescue_toofew_flat_threshold': 'Too-few rescue minimum similarity:',
            'rescue_toofew_gate_frac': 'Too-few rescue margin over the runner-up:',
            'rescue_toofew_min_z': 'Missing tags: at least x untransduced SDs:',
            'rescue_toofew_min_frac': 'Missing tags: at least x of the way to threshold:',
            'rescue_toofew_af_ratio': 'Largest AF abundance per unit of the largest positive tag:',
            'fit_min_fold': 'Fit events: every on-tag at least x threshold (fit.min.fold):',
            'min_cells_per_combo': 'Minimum fit events per combination (min.cells.per.combo):',
            'spill_ridge': 'Ridge penalty (spill.ridge):',
        }
        rows = ('rescue_toomany', 'rescue_flat_threshold', 'rescue_gate_frac', 'rescue_toofew',
                'rescue_toofew_flat_threshold', 'rescue_toofew_gate_frac', 'rescue_toofew_min_z',
                'rescue_toofew_min_frac', 'rescue_toofew_af_ratio', 'fit_min_fold', 'min_cells_per_combo',
                'spill_ridge')
        for key in rows:
            if key in self.checks:
                form.addRow(self.checks[key])
            else:
                form.addRow(labels[key], self.adv[key])
        for key, widget in self.adv.items():
            widget.valueChanged.connect(lambda value, k=key: self.plugin.on_setting_changed(k, value))
        for key, box in self.checks.items():
            box.toggled.connect(lambda checked, k=key: self.plugin.on_setting_changed(k, bool(checked)))

    def highlight_point_size(self) -> int:
        return self.highlight_size.value() if self.highlight_check.isChecked() else 0

    def _on_highlight_changed(self, *_args):
        settings = self.plugin.qsettings
        settings.setValue('highlight_barcoded', 'true' if self.highlight_check.isChecked() else 'false')
        settings.setValue('highlight_size', int(self.highlight_size.value()))
        self.highlight_size.setEnabled(self.highlight_check.isChecked())
        size = self.highlight_point_size()
        for plot in (self.biplot_before, self.biplot_after):
            plot.set_highlight_size(size)

    @staticmethod
    def _mirror_channels(target, x, y):
        target.blockSignals(True)
        target.set_channels(x, y)
        target.blockSignals(False)

    # -- refresh -------------------------------------------------------------

    def refresh(self):
        plugin = self.plugin
        settings = plugin.state.settings
        for key, widget in self.adv.items():
            widget.blockSignals(True)
            widget.setValue(settings[key])
            widget.blockSignals(False)
        for key, box in self.checks.items():
            box.blockSignals(True)
            box.setChecked(bool(settings[key]))
            box.blockSignals(False)
        ready = plugin.section_ready(4)
        busy = plugin.characterise_busy()
        self.run_btn.setEnabled(ready and not busy)
        if not ready and not busy:
            self.run_status.setText('Confirm the gates and thresholds in section 3 first.')
        elif ready and not busy and self.run_status.text() == 'Confirm the gates and thresholds in section 3 first.':
            self.run_status.setText('')
        self.show_results()

    def clear(self):
        self._shown_key = self._heat_key = self._expression_key = None
        self.summary.setText('')
        self.frequency.clear()
        self.heatmap.clear()
        self.expression.clear()
        self.biplot_before.clear()
        self.biplot_after.clear()
        self.biplot_msg.setText('')
        self._set_table([])

    def _set_table(self, rows):
        old = self.table
        self.table = CopyableTableWidget([{k: _table_value(v) for k, v in row.items()} for row in rows],
                                         FIT_TABLE_HEADERS)
        self.table.setMinimumHeight(min(80 + 26 * len(rows), 420))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if old is not None:
            self.table_holder.replaceWidget(old, self.table)
            old.setParent(None)
            old.deleteLater()
        else:
            self.table_holder.addWidget(self.table)
        self.table.setVisible(bool(rows))

    def show_results(self):
        """Show the session run, or the saved characterisation without its
        event plots; rebuild only the figures whose inputs changed."""
        view = self.plugin.characterisation_view()
        if view is None:
            if self._shown_key is not None:
                self.clear()
            return
        fc, summary, output = view
        key = (id(fc), id(output))
        is_dark = _is_dark()
        if key != self._shown_key:
            self._shown_key = key
            self._heat_key = self._expression_key = None
            self.summary.setText(self.plugin.characterisation_summary(fc, summary))
            names = list(fc.design.combo_names)
            per_key = summary.get('per_combo') or {}
            counts = [per_key.get(fcc.combo_key(tags), 0) for tags in fc.design.combo_tags]
            self.frequency.set_figure(fcp.combo_frequency_figure(names, counts, summary, is_dark))
            self._set_table(fcc.fit_table_rows(fc))
            self._show_biplots(output)

        space = self.heat_space.currentData()
        if (key, space) != self._heat_key:
            self._heat_key = (key, space)
            try:
                matrix = fcc.spillover_matrix(fc, space)
                fig = fcp.spillover_heatmap_figure(matrix, self.plugin.channel_labels(), space, is_dark)
                self.heatmap.set_figure(fig)
            except ValueError as exc:
                self.heatmap.set_figure(fcp.message_figure(str(exc), is_dark), exportable=False)

        busy = output is None and self.plugin.characterise_busy()
        if (key, busy) != self._expression_key:
            self._expression_key = (key, busy)
            if output is None:
                text = ('Reloading the backbone to show the panel expression…' if busy
                        else 'Run the characterisation to show the panel expression.')
                self.expression.set_figure(fcp.message_figure(text, is_dark), exportable=False)
            else:
                self.expression.set_figure(fcp.panel_expression_figure(output.expression, is_dark))

    def _show_biplots(self, output):
        plots = (self.biplot_before, self.biplot_after)
        if output is None:
            for plot in plots:
                plot.clear()
            self.biplot_msg.setText('Run the characterisation to show the biplots.')
            return
        before, after, channels, labels, barcoded = self.plugin.biplot_data(output)
        if before is None or len(channels) < 2:
            for plot in plots:
                plot.clear()
            self.biplot_msg.setText('')
            return
        x = self.biplot_before.channels()[0]
        y = self.biplot_before.channels()[1]
        if x not in channels or y not in channels:
            tags = [f for f in output.result.spectra.design.fluors if f in channels]
            x = tags[0] if tags else channels[0]
            y = self.plugin.partner_channel(x) or next(ch for ch in channels if ch != x)
        self.highlight_size.setEnabled(self.highlight_check.isChecked())
        for plot, data in zip(plots, (before, after)):
            plot.set_event_data(data)
            plot.set_menu_channels(channels, labels)
            plot.initialise_from_controller(x, y)
            plot.set_highlight(barcoded, self.highlight_point_size())
            plot.redraw()
        engine = output.engine or 'final unmix'
        self.biplot_msg.setText(f'{len(before):,} gated backbone events ({int(barcoded.sum()):,} barcoded) without '
                                f'(left) and with (right) the FlowCode correction, both through the final unmix '
                                f'({engine}). Click an axis label to show any other parameter.')


_CORRECTION_ITEMS = (('Per combination; pairwise model for the rest (combo)', 'combo'),
                     ('Pairwise model for every event (pairwise)', 'pairwise'),
                     ('No correction (none)', 'none'))
_REGRESSOR_ITEMS = (('Fixed basis: spectra plus mean AF (fixed)', 'fixed'),
                    ('Debarcoding space (debarcode)', 'debarcode'))


class ExportPage(_Page):
    """Section 5: FlowCode unmixing and FCS export of the selected samples."""

    SECTION_SAMPLES, SECTION_OUTPUT, SECTION_ADVANCED = range(3)

    def __init__(self, plugin, parent=None):
        super().__init__(plugin, 5, parent, concertina=True)
        self._filling = False
        self._status_is_problem = False
        self.setup_banner = QLabel(SETUP_BANNER)
        self.setup_banner.setWordWrap(True)
        self.setup_banner.setStyleSheet(_BANNER_STYLE)
        self.body.addWidget(self.setup_banner)

        grp = QGroupBox('Unmix and export')
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.run_btn = QPushButton('Unmix and export')
        self.run_btn.clicked.connect(lambda: self.plugin.start_export())
        self.run_status = QLabel('')
        self.run_status.setWordWrap(True)
        row.addWidget(self.run_btn)
        row.addWidget(self.run_status, stretch=1)
        lay.addLayout(row)
        self.result = QLabel('')
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.RichText)
        lay.addWidget(self.result)
        self.body.addWidget(grp)

        lay = self.add_section('Samples')
        row = QHBoxLayout()
        all_btn = QPushButton('Select all')
        all_btn.clicked.connect(lambda: self._select_all(True))
        none_btn = QPushButton('Select none')
        none_btn.clicked.connect(lambda: self._select_all(False))
        self.sample_count = QLabel('')
        self.sample_count.setStyleSheet(_HINT_STYLE)
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addWidget(self.sample_count, stretch=1)
        lay.addLayout(row)
        self.controls_check = QCheckBox('Include single-stain controls and unstained samples')
        self.controls_check.toggled.connect(self._on_controls_toggled)
        lay.addWidget(self.controls_check)
        self.sample_list = QListWidget()
        self.sample_list.setMinimumHeight(240)
        self.sample_list.itemChanged.connect(self._on_sample_item_changed)
        lay.addWidget(self.sample_list)
        self.af_warn = _message_label(_WARN_STYLE)
        lay.addWidget(self.af_warn)

        lay = self.add_section('Output')
        row = QHBoxLayout()
        row.addWidget(QLabel('Folder:'))
        self.folder_label = QLabel('')
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.folder_label, stretch=1)
        choose_btn = QPushButton('Choose…')
        choose_btn.clicked.connect(self._on_choose_folder)
        self.default_btn = QPushButton('Use default')
        self.default_btn.clicked.connect(self._on_default_folder)
        row.addWidget(choose_btn)
        row.addWidget(self.default_btn)
        lay.addLayout(row)
        naming = QLabel('Files are named "<sample> FlowCode (Unmixed).fcs" and keep the folder structure of the '
                        'raw samples.')
        naming.setWordWrap(True)
        naming.setStyleSheet(_HINT_STYLE)
        lay.addWidget(naming)
        self.variants_check = QCheckBox('Use per-cell spectral variants in the final unmix')
        self.variants_check.toggled.connect(lambda c: self._on_value('use_variants', bool(c)))
        lay.addWidget(self.variants_check)
        self.variants_note = QLabel('')
        self.variants_note.setWordWrap(True)
        self.variants_note.setStyleSheet(_HINT_STYLE)
        lay.addWidget(self.variants_note)

        self._build_advanced(self.add_section('Advanced settings'))
        self.restore_open_section(self.SECTION_SAMPLES)

    def _build_advanced(self, lay):
        form = QFormLayout()
        lay.addLayout(form)
        self.correction = QComboBox()
        for text, value in _CORRECTION_ITEMS:
            self.correction.addItem(text, value)
        self.regressor = QComboBox()
        for text, value in _REGRESSOR_ITEMS:
            self.regressor.addItem(text, value)
        _no_wheel(self.correction, self.regressor)
        self.correction.currentIndexChanged.connect(
            lambda _i: self._on_choice('correction', self.correction))
        self.regressor.currentIndexChanged.connect(
            lambda _i: self._on_choice('regressor', self.regressor))
        form.addRow('Correction (correction):', self.correction)
        form.addRow('Correction regressors (regressor):', self.regressor)
        cpu = os.cpu_count() or 2
        self.adv = {
            'tag_shift_min_n': _int_spin(0, 10_000_000, 10),
            'n_passes': _int_spin(1, 20),
            'n_af_passes': _int_spin(1, 20),
            'refine_af_quantile': _float_spin(0.0, 1.0, 2, 0.05),
            'noise_floor': _float_spin(0.0, 1_000_000.0, 1, 5.0),
            'alpha': _float_spin(0.0, 1.0, 2, 0.05),
            'collinear_threshold': _float_spin(0.0, 1.0, 2, 0.05),
            'threads': _int_spin(1, cpu),
        }
        labels = {
            'tag_shift_min_n': 'Events per tag for the rescue shift (tag.shift.min.n):',
            'n_passes': 'n.passes:',
            'n_af_passes': 'n.af.passes:',
            'refine_af_quantile': 'refine.af.quantile:',
            'noise_floor': 'noise.floor:',
            'alpha': 'alpha:',
            'collinear_threshold': 'collinear.threshold:',
            'threads': 'threads:',
        }
        for key, widget in self.adv.items():
            form.addRow(labels[key], widget)
            widget.valueChanged.connect(lambda value, k=key: self._on_value(k, value))
        self.joint_pair = QCheckBox('joint.pair.resolution')
        self.joint_pair.toggled.connect(lambda c: self._on_value('joint_pair_resolution', bool(c)))
        form.addRow(self.joint_pair)
        note = QLabel('The kernel settings (n.passes to joint.pair.resolution) apply to the final unmix with the '
                      'compiled AutoSpectral Optimization kernel; cell.weight is set in section 3.')
        note.setWordWrap(True)
        note.setStyleSheet(_HINT_STYLE)
        form.addRow(note)

        grp = QGroupBox('Tandem tags: sample-specific breakdown')
        tlay = QVBoxLayout(grp)
        hint = QLabel('Ticked tags are checked in each sample for breakdown into their parent beyond the '
                      'backbone\'s, and corrected when it is found (see Help).')
        hint.setWordWrap(True)
        hint.setStyleSheet(_HINT_STYLE)
        tlay.addWidget(hint)
        self.tandem_checks_lay = QVBoxLayout()
        tlay.addLayout(self.tandem_checks_lay)
        self.tandem_checks = {}
        self._tandem_keys = None
        self.tandem_none = QLabel('No tag fluorophore is a tandem in the fluorophore database.')
        self.tandem_none.setStyleSheet(_HINT_STYLE)
        tlay.addWidget(self.tandem_none)
        self.tandem_status = QLabel('')
        self.tandem_status.setWordWrap(True)
        self.tandem_status.setTextFormat(Qt.RichText)
        tlay.addWidget(self.tandem_status)
        tform = QFormLayout()
        tlay.addLayout(tform)
        self.tandem_adv = {
            'tandem_z_min': _float_spin(0.0, 100.0, 1, 0.5),
            'tandem_min_combos': _int_spin(2, 10_000),
            'tandem_max_events': _int_spin(10_000, 1_000_000_000, 100_000),
        }
        tandem_labels = {
            'tandem_z_min': 'z threshold:',
            'tandem_min_combos': 'Combinations needed with and without the tag:',
            'tandem_max_events': 'Gated events per sample for the estimate:',
        }
        for key, widget in self.tandem_adv.items():
            tform.addRow(tandem_labels[key], widget)
            widget.valueChanged.connect(lambda value, k=key: self._on_value(k, value))
        lay.addWidget(grp)

    # -- refresh -------------------------------------------------------------

    def refresh(self):
        plugin = self.plugin
        state = plugin.state
        self.setup_banner.setVisible(not _optimization_setup_done(plugin.controller))
        self._filling = True
        try:
            self.controls_check.setChecked(bool(state.settings.get('export_include_controls')))
            self._fill_samples()
            _set_message(self.af_warn, [plugin.export_af_note()])
            self.folder_label.setText(str(plugin.export_folder()))
            self.default_btn.setEnabled(bool(state.output_folder))
            available = plugin.variants_available()
            self.variants_check.setEnabled(available)
            self.variants_check.setChecked(plugin.use_variants())
            self.variants_note.setText(plugin.variants_note())
            for widget, key in ((self.correction, 'correction'), (self.regressor, 'regressor')):
                widget.setCurrentIndex(max(widget.findData(state.settings[key]), 0))
            for key, widget in self.adv.items():
                widget.blockSignals(True)
                widget.setValue(plugin.kernel_threads() if key == 'threads' else state.settings[key])
                widget.blockSignals(False)
            self.joint_pair.setChecked(bool(state.settings['joint_pair_resolution']))
            for key, widget in self.tandem_adv.items():
                widget.blockSignals(True)
                widget.setValue(state.settings[key])
                widget.blockSignals(False)
            self._fill_tandems()
        finally:
            self._filling = False
        running = plugin.export_running()
        ready = plugin.section_ready(5)
        problem = plugin.export_problem() if ready else None
        self.run_btn.setEnabled(ready and not running and problem is None)
        if not running:
            text = 'Complete section 4 first.' if not ready else problem
            if text:
                self.set_run_status(text, problem=True)
            elif self._status_is_problem:
                self.set_run_status('')
        self.result.setText(plugin.export_result_text())

    def set_run_status(self, text: str, problem: bool = False):
        self.run_status.setText(text)
        self._status_is_problem = problem

    def _fill_tandems(self):
        plugin = self.plugin
        candidates = plugin.tandem_candidates()
        ticked = set(plugin.tandem_selection(candidates))
        keys = [(c['fluor'], c['parent'], c['problem']) for c in candidates]
        if keys != self._tandem_keys:
            while self.tandem_checks_lay.count():
                widget = self.tandem_checks_lay.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()
            self.tandem_checks = {}
            for c in candidates:
                text = f'{c["tag"]} ({c["fluor"]} \u2192 {c["parent"]})' if c['parent'] \
                    else f'{c["tag"]} ({c["fluor"]})'
                box = QCheckBox(text + (f': {c["problem"]}' if c['problem'] else ''))
                box.setEnabled(c['tandem'] is not None)
                box.toggled.connect(lambda checked, f=c['fluor']: self._on_tandem_toggled(f, checked))
                self.tandem_checks_lay.addWidget(box)
                self.tandem_checks[c['fluor']] = box
            self._tandem_keys = keys
        for fluor, box in self.tandem_checks.items():
            box.setChecked(fluor in ticked)
        self.tandem_none.setVisible(not candidates)
        self.tandem_status.setText(plugin.tandem_reference_text(candidates))

    def _on_tandem_toggled(self, fluor: str, checked: bool):
        if not self._filling:
            self.plugin.set_tandem_tag(fluor, checked)

    def _fill_samples(self):
        plugin = self.plugin
        candidates = plugin.export_candidates()
        selected = set(plugin.export_selection())
        self.sample_list.clear()
        for key, name, eligible in candidates:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, key)
            if eligible:
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked if key in selected else Qt.Unchecked)
            else:
                item.setFlags(Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setToolTip('Assign AF profiles in the AutoSpectral AF tab.')
            self.sample_list.addItem(item)
        n_eligible = sum(1 for _k, _n, ok in candidates if ok)
        self.sample_count.setText(f'{len(selected)} of {n_eligible} sample(s) selected'
                                  + (f'; {len(candidates) - n_eligible} without an AF profile.'
                                     if n_eligible < len(candidates) else '.'))

    # -- handlers --------------------------------------------------------------

    def _checked_keys(self) -> list:
        keys = []
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            if item.checkState() == Qt.Checked:
                keys.append(item.data(Qt.UserRole))
        return keys

    def _on_sample_item_changed(self, _item):
        if self._filling:
            return
        self.plugin.state.export_samples = self._checked_keys()
        self.plugin.state_changed_later()

    def _on_controls_toggled(self, checked: bool):
        if not self._filling:
            self.plugin.set_export_include_controls(checked)

    def _select_all(self, checked: bool):
        eligible = [k for k, _n, ok in self.plugin.export_candidates() if ok]
        self.plugin.state.export_samples = list(eligible) if checked else []
        self.plugin.state_changed()

    def _on_choose_folder(self):
        start = str(self.plugin.export_folder())
        path = QFileDialog.getExistingDirectory(self, 'FlowCode export folder', start)
        if path:
            self.plugin.state.output_folder = str(path)
            self.plugin.state_changed()

    def _on_default_folder(self):
        self.plugin.state.output_folder = None
        self.plugin.state_changed()

    def _on_choice(self, key, widget):
        if not self._filling:
            self.plugin.on_setting_changed(key, widget.currentData())

    def _on_value(self, key, value):
        if not self._filling:
            self.plugin.on_setting_changed(key, value)


# ---------------------------------------------------------------------------
# Plugin widget
# ---------------------------------------------------------------------------

class PluginWidget(QWidget):
    def __init__(self, bus=None, controller=None, parent=None):
        super().__init__(parent)
        _ensure_qt_imports()
        self.bus = bus
        self.controller = controller
        self.qsettings = QSettings('honeychrome', _SETTINGS_APP_KEY)
        self.state = fcs.FlowCodeState()
        self._experiment_key = None
        self._active = False
        self._tasks = {}
        self._task_experiment = {}
        self._threads = []
        self._change_pending = False
        self._pending_load = None
        self._pending_auto = None
        self._pending_space = None
        self._pending_char = None
        self._after_space = None
        self._saved_cache = None
        self._export_run = None
        self._export_job = None
        self._reset_session()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self.label_disabled = QLabel(f'{plugin_name}: calculate the unmixing matrix (Spectral Process tab) '
                                     f'before using this plugin.')
        self.label_disabled.setWordWrap(True)
        self.label_disabled.setStyleSheet(_BANNER_STYLE)
        outer.addWidget(self.label_disabled)

        self.content = QWidget()
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        self.kernel_banner = QLabel(KERNEL_BANNER)
        self.kernel_banner.setWordWrap(True)
        self.kernel_banner.setTextFormat(Qt.RichText)
        self.kernel_banner.setStyleSheet(_BANNER_STYLE)
        self.kernel_banner.setVisible(not fcc.kernel_available())
        content.addWidget(self.kernel_banner)

        self.pages = {
            1: InputsPage(self),
            2: TagsPage(self),
            3: GatesPage(self),
            4: CharacterisePage(self),
            5: ExportPage(self),
        }
        self.tabs = _SideTabWidget()
        icons = {1: 'file-import', 2: 'table', 3: 'polygon', 4: 'sparkles', 5: 'file-export'}
        for n, page in self.pages.items():
            self.tabs.addTab(page, f'{n}. {fcs.SECTION_TITLES[n]}', icons[n])
        self.tabs.currentChanged.connect(self._on_page_changed)
        content.addWidget(self.tabs, stretch=1)
        outer.addWidget(self.content, stretch=1)
        self.content.setVisible(False)

        if self.bus is not None:
            self.bus.modeChangeRequested.connect(self._on_mode_change)
            self.bus.spectralProcessRefreshed.connect(self._on_process_refreshed)
            self.bus.sampleTreeUpdated.connect(self._on_sample_tree_updated)

    # -- experiment and state ------------------------------------------------

    def _reset_session(self):
        self.backbone_data = None
        self.debarcode_space = None
        self.characterisation = None
        self._load_signature = None
        self._space_inputs = None
        self._space_tags = None
        self._space_ceiling = None
        self._char_inputs = None

    def _available(self) -> bool:
        c = self.controller
        return bool(c is not None and c.experiment is not None
                    and _process(c).get('unmixing_matrix') is not None)

    def _sync_experiment(self):
        key = getattr(self.controller.experiment, 'experiment_path', None)
        if key == self._experiment_key:
            return
        self._experiment_key = key
        self.state = fcs.FlowCodeState.from_json(_process(self.controller).get(fcs.PROCESS_KEY))
        self._reset_session()
        self._after_space = None
        self._saved_cache = None
        self.pages[3].panel.clear()
        self.pages[3].thresholds.clear()
        self.pages[4].clear()

    def save_state(self):
        """Write the state into the experiment (saved with the .kit file)."""
        if self.controller is None or self.controller.experiment is None or self._experiment_key is None:
            return
        if getattr(self.controller.experiment, 'experiment_path', None) != self._experiment_key:
            return
        _process(self.controller)[fcs.PROCESS_KEY] = self.state.to_json()

    def state_changed(self, save: bool = True):
        """Bring derived state up to date, refresh the pages and autosave."""
        self._change_pending = False
        self.sync_derived_state()
        if save:
            self.save_state()
            if self.bus is not None:
                self.bus.autoSaveRequested.emit()
        self.refresh()

    def state_changed_later(self):
        """state_changed() once control returns to the event loop, so a widget
        is never rebuilt inside its own signal handler."""
        if not self._change_pending:
            self._change_pending = True
            QTimer.singleShot(0, self.state_changed)

    # -- bus -------------------------------------------------------------------

    def _on_mode_change(self, mode):
        if mode != plugin_name:
            if self._active:
                self.save_state()
                self._active = False
            return
        self._active = True
        available = self._available()
        self.label_disabled.setVisible(not available)
        self.content.setVisible(available)
        if not available:
            return
        self._sync_experiment()
        self.state_changed(save=False)
        QTimer.singleShot(0, self._on_page_shown)

    def _on_process_refreshed(self):
        if self._active and self._available():
            self.refresh()

    def _on_sample_tree_updated(self):
        if self._active and self._available():
            self.refresh()

    def _on_page_changed(self, _index):
        self.refresh()
        self._on_page_shown()

    def _on_page_shown(self):
        page = self.tabs.currentWidget()
        if page is self.pages[1]:
            self.start_matching()
        elif page is self.pages[3]:
            self._restore_session()
        elif page is self.pages[4]:
            self._restore_characterisation()

    # -- refresh and status ------------------------------------------------------

    def refresh(self):
        if not self._available():
            return
        if self.backbone_data is not None and self._load_signature != self.load_signature():
            self._reset_session()
            self.pages[3].panel.clear()
            self.pages[3].thresholds.clear()
        for n, page in self.pages.items():
            kind, text = self.section_status(n)
            page.set_status(kind, text)
        page = self.tabs.currentWidget()
        if page is not None:
            page.refresh()

    def section_status(self, n: int) -> tuple:
        if n == 1:
            problems = self.section1_problems()
            return ('error', problems[0]) if problems else ('done', 'Done')
        if n == 2:
            if self.section1_problems():
                return 'waiting', 'Waiting for section 1'
            errors = self.tag_problems()[0] + self.combo_problems()
            return ('error', errors[0]) if errors else ('done', 'Done')
        if not self.section_ready(n):
            return 'waiting', f'Waiting for section {n - 1}'
        status, reason = self.state.run_status(n, self.fingerprint())
        if status == 'done' and n == 4 and self.saved_spectra() is None:
            return 'stale', 'Needs re-run: the saved characterisation was not found.'
        if status == 'done':
            return 'done', 'Done'
        if status == 'stale':
            if 'spectral model changed' in reason and n >= 3:
                return 'stale', ('The spectral model changed since the backbone was characterised. '
                                 'Re-run sections 3 and 4.')
            return 'stale', f'Needs re-run: {reason}'
        return 'pending', 'Not run yet'

    def section_ready(self, n: int) -> bool:
        """True when every section before *n* is complete."""
        for k in range(1, n):
            if self.section_status(k)[0] != 'done':
                return False
        return True

    # -- section 1 ---------------------------------------------------------------

    def raw_layout(self) -> RawLayout:
        return _raw_layout(self.controller)

    def section1_problems(self) -> list:
        c = self.controller
        state = self.state
        problems = []
        if not state.backbones():
            problems.append('Choose at least one backbone sample.')
        if not state.unstained_key or state.unstained_key not in fcs.unstained_candidates(c.experiment.samples):
            problems.append('Choose the matched unstained sample.')
        else:
            problem = self.af_library_problem()
            if problem:
                problems.append(problem)
        layout = self.raw_layout()
        if layout.scatter_x is None or layout.scatter_y is None:
            problems.append('No FSC/SSC area channels found for this cytometer.')
        return problems

    def af_library_problem(self) -> str | None:
        state = self.state
        if not state.unstained_key:
            return None
        _n, problem = fcs.af_library_problem(_process(self.controller).get('af_profiles') or {},
                                             state.af_profile_names, _detectors(self.controller),
                                             _sample_name(self.controller, state.unstained_key))
        return problem

    def apply_auto_af_profiles(self, force: bool = False):
        state = self.state
        if not (force or state.af_auto) or not state.unstained_key:
            return
        state.af_profile_names = fcs.af_profiles_for_sample(_process(self.controller).get('af_profiles') or {},
                                                            state.unstained_key, self.controller.experiment_dir)
        state.af_auto = True

    def apply_matching_scores(self, backbone_keys: list, scores: dict):
        state = self.state
        if list(backbone_keys) != state.backbones():
            self.start_matching()
            return
        state.unstained_scores = dict(scores)
        state.match_backbone_keys = list(backbone_keys)
        if state.unstained_auto or state.unstained_key is None:
            self.choose_best_unstained(save=False)
        self.state_changed()

    def choose_best_unstained(self, save: bool = True):
        state = self.state
        order = fcs.unstained_candidates(self.controller.experiment.samples)
        best = fcc.best_match(state.unstained_scores, order)
        if best is None:
            return
        if best != state.unstained_key:
            state.unstained_key = best
            self.apply_auto_af_profiles(force=True)
        state.unstained_auto = True
        if save:
            self.state_changed()

    def backbone_changed(self):
        self.state_changed()
        self.start_matching()

    def start_matching(self, force: bool = False):
        state = self.state
        c = self.controller
        candidates = fcs.unstained_candidates(c.experiment.samples)
        if not state.backbones() or not candidates:
            return
        if not force and state.match_backbone_keys == state.backbones() \
                and set(state.unstained_scores) == set(candidates):
            return
        layout = self.raw_layout()
        if layout.scatter_x is None or layout.scatter_y is None:
            return
        transforms = c.raw_transformations or {}
        tr_x, tr_y = deepcopy(transforms.get(layout.scatter_x)), deepcopy(transforms.get(layout.scatter_y))
        if tr_x is None or tr_y is None:
            return
        experiment_dir = Path(c.experiment_dir)
        backbone = state.backbones()
        max_events = int(state.settings['match_max_events'])
        bins = int(state.settings['match_bins'])
        started = self._start_task('match', lambda progress: match_unstained(
            experiment_dir, backbone, candidates, layout, tr_x, tr_y, max_events, bins, progress))
        if started:
            self.refresh()

    # -- section 2 ---------------------------------------------------------------

    def sync_derived_state(self):
        """Resolve the automatic AF profiles, the tag selection, tag names and
        combo rows from the current experiment and combinations CSV."""
        if not self._available():
            return
        c = self.controller
        state = self.state
        if state.backbone_keys is None:
            state.backbone_keys = fcs.default_backbone_keys(c.experiment.samples)
        else:
            allowed = set(fcs.backbone_candidates(c.experiment.samples))
            state.backbone_keys = [key for key in state.backbone_keys if key in allowed]
        if state.af_auto:
            self.apply_auto_af_profiles()
        channels = _tag_channel_options(c)
        antigens = _antigens(c)
        recognise = fcc.recognise_tag
        csv_picked, csv_map = [], {}
        if state.csv_combos:
            csv_picked, csv_map, _unmatched = fcs.csv_tag_channels(state.csv_combos, channels, antigens, recognise)
        if state.tag_selection is None:
            state.tag_selection = csv_picked if state.csv_combos else fcs.auto_tag_channels(channels, antigens,
                                                                                            recognise)
        state.tag_selection = [ch for ch in channels if ch in set(state.tag_selection)]
        state.tag_names = {ch: fcs.tag_name_for_channel(ch, antigens.get(ch, ''), recognise)
                           for ch in state.tag_selection}
        if state.csv_combos:
            tag_for_csv = {k: state.tag_names.get(ch, ch) for k, ch in csv_map.items()}
            state.combos = fcs.csv_combo_rows(state.csv_combos, tag_for_csv, previous=state.combos)
        else:
            state.combos = fcs.generate_combo_rows(list(state.tag_map()), previous=state.combos)

    def tag_problems(self) -> tuple:
        state = self.state
        errors, warnings = fcs.tag_problems(state.tag_map(), _fluor_names(self.controller))
        errors = fcs.duplicate_tag_problems(state.tag_names) + errors
        if state.csv_combos:
            channels = _tag_channel_options(self.controller)
            _p, _m, unmatched = fcs.csv_tag_channels(state.csv_combos, channels, _antigens(self.controller),
                                                     fcc.recognise_tag)
            if unmatched:
                errors.append('Tags in the combinations CSV that match no channel: ' + ', '.join(unmatched))
        return errors, warnings

    def combo_problems(self) -> list:
        unmixed = self.controller.experiment.settings.get('unmixed', {}).get('event_channels_pnn') or []
        return fcs.combo_problems(self.state.combos, list(self.state.tag_map()), unmixed)

    def design(self):
        return fcs.design_from_state(self.state)

    def tag_table_rows(self) -> list:
        state = self.state
        tag_map = state.tag_map()
        spectra = _reference_spectra(self.controller)
        names = _fluor_names(self.controller)
        partners = self.tag_partners()
        return [{'Tag': tag, 'Fluorophore': ch, 'Nearest tag': partners.get(tag, '')}
                for tag, ch in tag_map.items() if spectra is not None and ch in names]

    def tag_partners(self) -> dict:
        """{tag: the other tag with the most similar reference spectrum}."""
        tag_map = self.state.tag_map()
        spectra = _reference_spectra(self.controller)
        names = _fluor_names(self.controller)
        tags = [t for t, ch in tag_map.items() if ch in names]
        if spectra is None or len(tags) < 2:
            return {}
        rows = spectra[[names.index(tag_map[t]) for t in tags]]
        nearest = fcc.cosine_partners(rows)
        return {t: tags[j] for t, j in zip(tags, nearest)}

    def combo_backbone_counts(self) -> dict:
        return self.state.combo_backbone_counts()

    def partner_channel(self, channel: str) -> str | None:
        """Tag channel with the reference spectrum most similar to *channel*'s."""
        tag_map = self.state.tag_map()
        by_channel = {ch: tag for tag, ch in tag_map.items()}
        partner = self.tag_partners().get(by_channel.get(channel))
        return tag_map.get(partner)

    def channel_labels(self) -> dict:
        """{fluorophore: 'Antigen - Fluorophore'} for figure labels."""
        antigens = _antigens(self.controller)
        return {name: _channel_label(name, antigens) for name in _fluor_names(self.controller)}

    # -- section 3 ---------------------------------------------------------------

    def fingerprint(self) -> str | None:
        spectra = _reference_spectra(self.controller)
        if spectra is None:
            return None
        return fcs.model_fingerprint(spectra, _fluor_names(self.controller), self.state.tag_channels(),
                                     _detectors(self.controller))

    def cell_weight(self) -> bool:
        value = self.state.settings.get('cell_weight')
        return _cytometer_db_col(self.controller) == 'ID7000' if value is None else bool(value)

    def on_setting_changed(self, key, value):
        self.state.settings[key] = value
        self.state_changed()

    def _scatter_transforms(self, channels) -> dict:
        transforms = self.controller.raw_transformations or {}
        return {ch: deepcopy(transforms[ch]) for ch in channels if ch in transforms}

    def _gate_channels(self, layout: RawLayout, fallbacks: dict) -> tuple:
        channels = [layout.scatter_x, layout.scatter_y, layout.singlet_x, layout.singlet_y]
        for rec in list(fallbacks.values()) + list(self.state.gates.values()):
            if rec:
                channels += [rec['x'], rec['y']]
        return tuple(dict.fromkeys(ch for ch in channels if ch and ch in layout.pnn_raw))

    def load_signature(self) -> tuple:
        """Inputs that determine the loaded events."""
        s = self.state
        return tuple(s.backbones()), s.unstained_key, s.settings.get('backbone_max_events')

    def load_backbone(self, then_restore: bool = False):
        if not self.section_ready(3) or self.task_running('load'):
            return
        c = self.controller
        state = self.state
        layout = self.raw_layout()
        fallbacks = {name: _experiment_gate_record(c, name.replace('FlowCode ', '')) for name in fcs.GATE_NAMES}
        channels = self._gate_channels(layout, fallbacks)
        transforms = self._scatter_transforms(channels)
        missing = [ch for ch in channels if ch not in transforms]
        if missing:
            self.pages[3].load_status.setText(f'No display transform for {", ".join(missing)}.')
            return
        experiment_dir = Path(c.experiment_dir)
        keys = (tuple(state.backbones()), state.unstained_key)
        max_events = int(state.settings['backbone_max_events'])
        self._pending_load = {'fallbacks': fallbacks, 'layout': layout, 'restore': then_restore,
                              'signature': self.load_signature()}
        self.pages[3].load_status.setText('Reading the backbone and matched unstained…')
        if self._start_task('load', lambda progress: load_backbone(
                experiment_dir, keys[0], keys[1], layout, channels, max_events, progress)):
            self.refresh()

    def _usable_record(self, record: dict | None) -> dict | None:
        data = self.backbone_data
        if record and data is not None and record['x'] in data.scatter_channels \
                and record['y'] in data.scatter_channels:
            return record
        return None

    def _on_load_finished(self, data: BackboneData):
        pending = self._pending_load
        if pending['signature'] != self.load_signature():
            return
        self._reset_session()
        self.backbone_data = data
        self._load_signature = pending['signature']
        self.pages[3].panel.clear()
        self.pages[3].thresholds.clear()
        state = self.state
        layout = pending['layout']
        singlets = self._usable_record(state.gates.get(fcs.SINGLETS_GATE))
        if singlets is None:
            fallback = self._usable_record(pending['fallbacks'].get(fcs.SINGLETS_GATE))
            auto = self._auto_singlets()
            if auto is not None:
                singlets = auto
            elif fallback is not None:
                singlets = dict(fallback, note='No FSC height channel on this cytometer: the singlet gate '
                                               'is copied from the Raw Data tab.')
            else:
                singlets = self._full_range_gate(layout.singlet_x or layout.scatter_x,
                                                 layout.singlet_y or layout.scatter_y,
                                                 'No singlet gate available: all events are kept.')
            state.gates[fcs.SINGLETS_GATE] = singlets
        cells = self._usable_record(state.gates.get(fcs.CELLS_GATE))
        if cells is None:
            self.auto_gate(fcs.CELLS_GATE, restore=pending['restore'])
            return
        self._show_gates()
        self._after_gates_changed(save=not pending['restore'])
        if pending['restore']:
            self.compute_debarcode_space(confirm=False)

    def _auto_singlets(self) -> dict | None:
        data = self.backbone_data
        if data is None:
            return None
        return auto_singlets_gate(data, self.raw_layout(), float(self.state.settings['singlet_quantile']),
                                  self._scatter_transforms(data.scatter_channels))

    def _show_gates(self, landmarks=None):
        panel = self.pages[3].panel
        data = self.backbone_data
        records = {name: self.state.gates[name] for name in fcs.GATE_NAMES}
        if panel.has_backbone(data):
            panel.set_records(records, landmarks=landmarks)
        else:
            panel.set_backbone(data, records, self._scatter_transforms(data.scatter_channels), landmarks)

    def _full_range_gate(self, x, y, note: str) -> dict:
        layout = self.raw_layout()
        cx, cy = layout.ceiling(x), layout.ceiling(y)
        return fcs.gate_record(x, y, fcc.rectangle_vertices(-cx, 2 * cx, -cy, 2 * cy), 'none', note)

    def auto_gate(self, name: str, restore: bool = False):
        data = self.backbone_data
        if data is None or self.task_running('auto_cells'):
            return
        state = self.state
        if name == fcs.SINGLETS_GATE:
            auto = self._auto_singlets()
            if auto is None:
                return
            state.gates[fcs.SINGLETS_GATE] = auto
            self._show_gates()
            self._after_gates_changed()
            return
        layout = self.raw_layout()
        transforms = self._scatter_transforms(data.scatter_channels)
        spectra = _reference_spectra(self.controller)
        names = _fluor_names(self.controller)
        tag_rows = [names.index(ch) for ch in state.tag_channels() if ch in names]
        singlet = dict(state.gates[fcs.SINGLETS_GATE])
        settings = dict(state.settings)
        self._pending_auto = {'fallback': self._usable_record(_experiment_gate_record(self.controller, 'Cells')),
                              'restore': restore, 'data': data}
        self.pages[3].gate_controls[fcs.CELLS_GATE][2].setText('Finding landmark events…')
        if self._start_task('auto_cells', lambda progress: auto_cells_gate(
                data, singlet, layout, spectra, tag_rows, settings, transforms)):
            self.refresh()

    def _on_auto_cells_finished(self, result):
        record, landmarks, reason = result
        pending = self._pending_auto
        if pending['data'] is not self.backbone_data:
            return
        state = self.state
        if record is None:
            fallback = pending['fallback']
            if fallback is not None:
                record = dict(fallback, note=f'Automatic gate not possible ({reason}); copied from the Raw Data tab.')
            else:
                layout = self.raw_layout()
                record = self._full_range_gate(layout.scatter_x, layout.scatter_y,
                                               f'Automatic gate not possible ({reason}); all singlets kept.')
        state.gates[fcs.CELLS_GATE] = record
        self._show_gates(landmarks=landmarks)
        self._after_gates_changed(save=not pending['restore'])
        if pending['restore']:
            self.compute_debarcode_space(confirm=False)

    def use_experiment_gate(self, name: str):
        base = name.replace('FlowCode ', '')
        record = self._usable_record(_experiment_gate_record(self.controller, base))
        if record is None:
            self.pages[3].gate_controls[name][2].setText(f'The Raw Data tab has no usable {base} gate to copy.')
            return
        self.state.gates[name] = record
        self._show_gates()
        self._after_gates_changed()

    def on_gate_edited(self, name: str):
        self.state.gates[name] = dict(self.pages[3].panel.records[name])
        self._after_gates_changed()

    def on_gate_reset_requested(self, name: str):
        if name in fcs.GATE_NAMES:
            record = self._usable_record(_experiment_gate_record(self.controller, name.replace('FlowCode ', '')))
            if record is not None:
                self.state.gates[name] = record
        QTimer.singleShot(0, self._show_gates)
        self.state_changed_later()

    def _after_gates_changed(self, save: bool = True):
        self.state_changed(save=save)
        self.pages[3].update_counts()

    def _gated_inputs(self):
        """Gated backbone and capped gated unstained detector values."""
        data = self.backbone_data
        panel = self.pages[3].panel
        cells = panel.masks()[fcs.CELLS_GATE]
        backbone_fl = np.asarray(data.fl[cells], dtype=np.float64)
        un_mask = np.flatnonzero(panel.unstained_mask())
        cap = int(self.state.settings['unstained_max_events'])
        un_idx = un_mask[seeded_subsample_indices(len(un_mask), cap, data.unstained_key, seed=42)]
        return backbone_fl, np.asarray(data.un_fl[un_idx], dtype=np.float64)

    def compute_debarcode_space(self, confirm: bool):
        data = self.backbone_data
        if data is None:
            return
        c = self.controller
        state = self.state
        backbone_fl, unstained_fl = self._gated_inputs()
        problem = None
        if len(backbone_fl) == 0:
            problem = 'No backbone events in FlowCode Cells.'
        elif len(unstained_fl) < 2:
            problem = 'Too few matched unstained events in FlowCode Cells.'
        if problem:
            self.pages[3].confirm_status.setText(problem)
            if self._after_space is not None:
                self._after_space = None
                self.pages[4].run_status.setText(problem)
            return
        spectra = _reference_spectra(c)
        names = _fluor_names(c)
        tag_fluors = [ch for ch in state.tag_channels() if ch in names]
        af = fcs.af_library_spectra(_process(c).get('af_profiles') or {}, state.af_profile_names)
        pos = getattr(c, 'autospectral_unmixed_pos_thresholds', None)
        pos = None if pos is None else np.asarray(pos, dtype=float).copy()
        cell_weight = self.cell_weight()
        ceiling = float(c.experiment.settings['raw'].get('magnitude_ceiling') or hc_settings.magnitude_ceiling)
        settings = dict(state.settings)
        if self.task_running('space'):
            return
        self._pending_space = {'confirm': confirm, 'inputs': state.section_inputs(3, self.fingerprint()),
                               'tags': list(zip([state.tag_names[ch] for ch in tag_fluors], tag_fluors)),
                               'ceiling': ceiling}
        self.pages[3].confirm_status.setText('Unmixing the gated events with per-cell AF…')
        if self._start_task('space', lambda progress: compute_debarcode_space(
                backbone_fl, unstained_fl, spectra, names, tag_fluors, af, pos, cell_weight, ceiling, settings,
                progress)):
            self.refresh()

    def _on_space_finished(self, result):
        space, auto = result
        pending = self._pending_space
        state = self.state
        follow, self._after_space = self._after_space, None
        if pending['inputs'] != state.section_inputs(3, self.fingerprint()):
            self.pages[3].confirm_status.setText('The inputs changed while unmixing: confirm the gates again.')
            return
        self.debarcode_space = space
        self._space_inputs = pending['inputs']
        self._space_tags = pending['tags']
        self._space_ceiling = pending['ceiling']
        for (tag, _ch), item in zip(pending['tags'], auto):
            old = state.thresholds.get(tag)
            if pending['confirm'] or old is None:
                if old is not None and old.get('method') in fcs.USER_THRESHOLD_METHODS:
                    state.thresholds[tag] = fcs.threshold_record(old['raw'], old['method'], item['delta_bic'],
                                                                 item['raw'], item['method'])
                else:
                    state.thresholds[tag] = fcs.threshold_record(item['raw'], item['method'], item['delta_bic'])
        tags = {t for t, _ in pending['tags']}
        state.thresholds = {t: v for t, v in state.thresholds.items() if t in tags}
        partners = self.tag_partners()
        tag_map = state.tag_map()
        for tag in tags:
            if state.y_channels.get(tag) not in _fluor_names(self.controller):
                state.y_channels[tag] = tag_map.get(partners.get(tag), tag_map[tag])
        if pending['confirm']:
            state.mark_completed(3, self.fingerprint())
        n_un = len(space.unstained)
        self.pages[3].confirm_status.setText(
            f'Debarcoding space: {len(space.backbone):,} backbone and {n_un:,} unstained events '
            f'({space.engine} per-cell AF).')
        self._show_thresholds()
        if pending['confirm']:
            self.pages[3].open_section(GatesPage.SECTION_THRESHOLDS)
        self.state_changed(save=pending['confirm'])
        if follow is not None:
            QTimer.singleShot(0, lambda: self.characterise(restore=follow == 'restore'))

    def debarcode_space_current(self) -> bool:
        return self.debarcode_space is not None and \
            self._space_inputs == self.state.section_inputs(3, self.fingerprint())

    def space_summary(self) -> str:
        space = self.debarcode_space
        source = {'optimization': 'AutoSpectral Optimization thresholds',
                  'unstained': '99.5th percentile of the matched unstained'}[space.rescue_floor_source]
        return f'Rescue noise floor: {source}.'

    def unmixed_display(self, unmixed: np.ndarray, af_scale: np.ndarray, af_idx: np.ndarray,
                        idx: np.ndarray, extra: dict | None = None) -> np.ndarray:
        """Events *idx* of a per-cell AF unmix in the unmixed channel layout
        (fluorophores, AF Abundance, AF Index and the *extra* {channel:
        values} columns; other columns zero)."""
        c = self.controller
        layout = list(c.experiment.settings['unmixed']['event_channels_pnn'])
        names = _fluor_names(c)
        display = np.zeros((len(idx), len(layout)))
        for j, name in enumerate(names):
            if name in layout:
                display[:, layout.index(name)] = unmixed[idx, j]
        if hc_settings.af_abundance_channel in layout:
            display[:, layout.index(hc_settings.af_abundance_channel)] = af_scale[idx]
        if hc_settings.af_index_channel in layout:
            index_map = af_index_lookup(_process(c).get('af_profiles') or {}, self.state.af_profile_names)
            sub = np.asarray(af_idx[idx], dtype=np.int64)
            values = sub if index_map is None else np.asarray(index_map)[np.clip(sub, 1, len(index_map)) - 1]
            display[:, layout.index(hc_settings.af_index_channel)] = values
        for channel, values in (extra or {}).items():
            if channel in layout:
                display[:, layout.index(channel)] = np.asarray(values)[idx]
        return display

    def _show_thresholds(self):
        space = self.debarcode_space
        if space is None:
            return
        n = len(space.backbone)
        cap = int(self.state.settings['threshold_display_max_events'])
        idx = seeded_subsample_indices(n, cap, 'flowcode_threshold_display', seed=42)
        display = self.unmixed_display(space.backbone, space.af_scale, space.af_idx, idx)
        self.pages[3].thresholds.set_data(display, self._space_tags, self.state.thresholds,
                                          self.state.y_channels, self._space_ceiling)

    def on_threshold_changed(self, tag: str, raw: float):
        rec = self.state.thresholds.get(tag)
        if rec is None:
            return
        rec.update(raw=float(raw), method='manual')
        self.pages[3].thresholds.update_table()
        self.state_changed()

    def on_y_channel_changed(self, tag: str, channel: str):
        self.state.y_channels[tag] = channel
        self.pages[3].thresholds.update_table()
        self.state_changed()

    def import_thresholds(self, path) -> bool:
        """Set thresholds from a FlowCodeUnmix thresholds CSV; tags the file
        does not name keep theirs."""
        panel = self.pages[3].thresholds
        state = self.state
        try:
            rows = fcc.read_thresholds_csv(path)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            panel.import_msg.setStyleSheet(_ERROR_STYLE)
            _set_message(panel.import_msg, [f'Could not import {Path(path).name}: {exc}'])
            return False
        tags = [t for t, _ch in (self._space_tags or []) if t in state.thresholds]
        fluors = [ch for t, ch in (self._space_tags or []) if t in state.thresholds]
        matched, unmatched = fcc.match_thresholds(rows, tags, fluors)
        for tag, raw in matched.items():
            state.thresholds[tag].update(raw=raw, method='imported')
        lines = [f'Imported {len(matched)} of {len(tags)} thresholds from {Path(path).name}.']
        missing = [t for t in tags if t not in matched]
        if missing:
            lines.append('Not in the file (unchanged): ' + ', '.join(missing) + '.')
        if unmatched:
            lines.append('Rows matching no tag: ' + ', '.join(str(u) for u in unmatched) + '.')
        panel.import_msg.setStyleSheet(_WARN_STYLE if (missing or unmatched or not matched) else _DONE_STYLE)
        _set_message(panel.import_msg, lines)
        panel.update_table()
        if matched:
            self.state_changed()
        return bool(matched)

    def on_thresholds_auto_all(self):
        for rec in self.state.thresholds.values():
            rec.update(raw=rec['auto_raw'], method=rec['auto_method'])
        self.pages[3].thresholds.update_table()
        self.state_changed()

    def on_threshold_reset(self, tag: str):
        rec = self.state.thresholds.get(tag)
        if rec is None:
            return
        rec.update(raw=rec['auto_raw'], method=rec['auto_method'])
        self.pages[3].thresholds.update_table()
        self.state_changed()

    def _restore_session(self):
        """Reload session data for section 3 when its saved gates exist but the
        events are not in memory."""
        state = self.state
        if self.backbone_data is not None or not self.section_ready(3):
            return
        if self.task_running('load'):
            return
        if all(name in state.gates for name in fcs.GATE_NAMES):
            self.load_backbone(then_restore=state.run_status(3, self.fingerprint())[0] == 'done')

    # -- section 4 ---------------------------------------------------------------

    def characterise_busy(self) -> bool:
        return any(self.task_running(k) for k in ('characterise', 'load', 'space', 'auto_cells')) \
            or self._after_space is not None

    def characterise(self, restore: bool = False):
        """Run section 4 on the gated backbone. Without a current debarcoding
        space, the events are reloaded and unmixed first. *restore* only
        rebuilds the session figures of a completed run: nothing is saved."""
        if not self.section_ready(4) or self.task_running('characterise'):
            return
        page = self.pages[4]
        if not self.debarcode_space_current():
            self._after_space = 'restore' if restore else 'run'
            page.run_status.setText('Loading the gated backbone and its debarcoding space…')
            if self.backbone_data is None:
                if not self.task_running('load'):
                    self.load_backbone(then_restore=True)
                started = self.task_running('load')
            else:
                if not self.task_running('space'):
                    self.compute_debarcode_space(confirm=False)
                started = self.task_running('space')
            if not started:
                self._after_space = None
                page.run_status.setText('Could not reload the gated backbone: open section 3 and confirm '
                                        'the gates.')
            self.refresh()
            return

        c = self.controller
        state = self.state
        space = self.debarcode_space
        try:
            design = self.design()
        except ValueError as exc:
            page.run_status.setText(str(exc))
            return
        backbone_fl, unstained_fl = self._gated_inputs()
        if len(backbone_fl) != len(space.backbone):
            page.run_status.setText('The gated events changed: confirm the gates in section 3 again.')
            return
        spectra = _reference_spectra(c)
        names = _fluor_names(c)
        thresholds = np.array([state.thresholds[tag]['raw'] for tag in design.tags], dtype=float)
        af = fcs.af_library_spectra(_process(c).get('af_profiles') or {}, state.af_profile_names)
        transforms = c.unmixed_transformations or {}
        transforms = {f: deepcopy(transforms[f]) for f in names if f in transforms}
        labels = self.channel_labels()
        settings = dict(state.settings)
        ceiling = float(self._space_ceiling)
        final, engine = self.final_unmix(spectra, names)
        detectors = _detectors(c)
        fingerprint = self.fingerprint()
        self._pending_char = {'restore': restore, 'inputs': state.section_inputs(4, fingerprint),
                              'design': design, 'ceiling': ceiling}
        page.run_status.setText('Characterising the backbone…' if not restore
                                else 'Rebuilding the figures of the saved characterisation…')
        tandems = [cand['tandem'] for cand in self.tandem_candidates() if cand['tandem'] is not None]
        args = (backbone_fl, space.backbone, unstained_fl, design, thresholds, spectra, names, detectors,
                space.af_mean, space.rescue_floor, ceiling, settings, lambda raw: final(raw, af), engine,
                transforms, labels)
        af_scale = space.af_scale
        if self._start_task('characterise', lambda progress: run_characterisation(
                *args, progress, tandems, debarcode_af_scale=af_scale)):
            self.refresh()

    def _on_characterise_finished(self, output: CharacteriseOutput):
        pending = self._pending_char
        state = self.state
        page = self.pages[4]
        fingerprint = self.fingerprint()
        if pending['inputs'] != state.section_inputs(4, fingerprint):
            page.run_status.setText('The inputs changed during the run: run it again.')
            return
        self.characterisation = output
        self._char_inputs = pending['inputs']
        if not pending['restore']:
            try:
                self._save_characterisation(output, pending['design'], pending['ceiling'])
            except OSError as exc:
                page.run_status.setText(f'Could not save the characterisation: {exc}')
                return
            state.mark_completed(4, fingerprint)
        fc = output.result.spectra
        n_char = sum(e is not None for e in fc.combo_error)
        page.run_status.setText(f'Done: {n_char} of {fc.design.n_combos} combinations characterised.')
        self.state_changed(save=not pending['restore'])

    def _save_characterisation(self, output: CharacteriseOutput, design, ceiling: float):
        """Arrays to the cache folder, metadata and counts to the state, and
        the R-compatible thresholds and combinations files."""
        experiment_dir = Path(self.controller.experiment_dir)
        fc = output.result.spectra
        meta = fcc.save_flowcode_spectra(fc, experiment_dir / 'cache' / SPECTRA_FILE)
        out_dir = experiment_dir / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fcc.write_thresholds_csv(out_dir / 'thresholds.csv', design, fc.thresholds, fcc.logicle_transform(ceiling))
        fcc.write_combinations_csv(out_dir / 'combinations.csv', design)
        self.state.characterisation = meta
        self.state.backbone_summary = fcs.backbone_summary(output.summary, design)
        self._saved_cache = None

    def saved_spectra(self):
        """The FlowCodeSpectra of the last completed run, read from the cache
        folder, or None when missing or not matching the saved metadata."""
        meta = self.state.characterisation
        if not meta or self.controller is None:
            return None
        path = Path(self.controller.experiment_dir) / 'cache' / SPECTRA_FILE
        try:
            stamp = (str(path), path.stat().st_mtime_ns, meta.get('fingerprint'))
        except OSError:
            return None
        if self._saved_cache is not None and self._saved_cache[0] == stamp:
            return self._saved_cache[1]
        try:
            fc = fcc.load_flowcode_spectra(meta, path)
            if fc.fingerprint() != meta.get('fingerprint'):
                fc = None
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(f'FlowCode: could not read {path}: {exc}')
            fc = None
        self._saved_cache = (stamp, fc)
        return fc

    def characterisation_view(self):
        """(FlowCodeSpectra, summary, session output or None) to display, or
        None. The session run wins; otherwise the saved run, without event
        data."""
        output = self.characterisation
        if output is not None:
            fc = output.result.spectra
            return fc, fcs.backbone_summary(output.summary, fc.design), output
        fc = self.saved_spectra()
        if fc is None or not self.state.backbone_summary:
            return None
        return fc, self.state.backbone_summary, None

    def characterisation_summary(self, fc, summary: dict) -> str:
        design = fc.design
        n_char = sum(e is not None for e in fc.combo_error)
        total = int(summary.get('total_events') or 0)
        valid = int(sum((summary.get('per_combo') or {}).values()))
        rescued = int(summary.get('rescued') or 0)
        rescued_toofew = int(summary.get('rescued_toofew') or 0)
        pct = f' ({100.0 * valid / total:.1f}%)' if total else ''
        lines = [f'{n_char} of {design.n_combos} combinations characterised'
                 + (f'; {design.n_combos - n_char} use the pairwise model.' if n_char < design.n_combos else '.'),
                 f'{valid:,} of {total:,} gated backbone events barcoded{pct}; {rescued:,} rescued from more '
                 f'than three positive tags and {rescued_toofew:,} from one or two.']
        if not fc.settings.get('rescue_toomany', True) or fc.rescue_floor is None:
            lines.append('Too-many-tag rescue was off.')
        if not fc.settings.get('rescue_toofew', True):
            lines.append('Too-few-tag rescue was off.')
        for fluor, ref in ((fc.diagnostics or {}).get('tandem') or {}).items():
            lines.append(f'Tandem {ref.get("tag", "")} ({fluor} \u2192 {ref.get("parent", "")}): '
                         f'{ref.get("parent", "")} per unit tag {_fmt_beta(ref.get("beta_uncorrected"))} before the '
                         f'correction, against the reference spectra; {_fmt_beta(ref.get("beta"))} after it '
                         f'(the reference for samples).')
        if self.characterisation is None:
            lines.append('<i>Saved run: the tag expression and biplots need the gated backbone and are '
                         'rebuilt when it is loaded.</i>')
        return '<br>'.join(lines)

    def biplot_data(self, output: CharacteriseOutput) -> tuple:
        """Uncorrected and corrected backbone (display subsample) in the
        unmixed layout, the channels that hold data, their labels and which
        displayed events carry a valid barcode."""
        space = self.debarcode_space
        c = self.controller
        layout = list(c.experiment.settings['unmixed']['event_channels_pnn'])
        if space is None or len(space.backbone) != len(output.after['unmixed']):
            return None, None, [], [], None
        scatter = {}
        data = self.backbone_data
        cells = self.pages[3].panel.masks().get(fcs.CELLS_GATE)
        if data is not None and cells is not None and int(cells.sum()) == len(space.backbone):
            scatter = {ch: data.scatter_column(ch)[cells] for ch in data.scatter_channels if ch in layout}
        filled = set(_fluor_names(c)) | set(scatter) | {hc_settings.af_abundance_channel, hc_settings.af_index_channel}
        transforms = c.unmixed_transformations or {}
        channels = [ch for ch in layout if ch in filled and ch in transforms]
        names = self.channel_labels()
        labels = [names.get(ch, ch) for ch in channels]
        idx = output.display_idx
        before = self.unmixed_display(output.before['unmixed'], output.before['af_scale'], output.before['af_idx'],
                                      idx, scatter)
        after = self.unmixed_display(output.after['unmixed'], output.after['af_scale'], output.after['af_idx'],
                                     idx, scatter)
        ids = np.asarray(output.result.ids)[idx]
        barcoded = (ids >= 1) & (ids <= output.result.spectra.design.n_combos)
        return before, after, channels, labels, barcoded

    def _restore_characterisation(self):
        """Rebuild the event figures of a completed run when page 4 is shown."""
        if self.characterisation is not None or self.characterise_busy():
            return
        if self.section_status(4)[0] == 'done':
            self.characterise(restore=True)

    # -- section 5 ---------------------------------------------------------------

    def export_candidates(self) -> list:
        """(key, name, has an AF library of at least 2 spectra) for every
        sample except single-stain controls and unstained samples, which are
        listed too when the section 5 setting includes them. The library is
        ``fcs.export_af_profile_names``: the sample's assignment plus the
        profiles the thresholds were set with."""
        c = self.controller
        samples = c.experiment.samples
        af_profiles = _process(c).get('af_profiles') or {}
        assigned = samples.get('sample_af_profiles') or {}
        keys = list(samples.get('all_samples') or {}) if self.state.settings.get('export_include_controls') \
            else fcs.backbone_candidates(samples)
        out = []
        for key in keys:
            names = fcs.export_af_profile_names(assigned.get(key), self.state.af_profile_names)
            n = sum(len(af_profiles[name].get('spectra') or []) for name in names if name in af_profiles)
            out.append((key, _sample_name(c, key), n >= 2))
        return out

    def export_af_note(self) -> str:
        """Warning for selected samples whose AutoSpectral AF assignment lacks
        the AF profiles the thresholds were set with ('' when none)."""
        c = self.controller
        if c is None or c.experiment is None:
            return ''
        keys = self.export_selection()
        gaps = fcs.af_assignment_gaps(c.experiment.samples.get('sample_af_profiles') or {}, keys,
                                      self.state.af_profile_names, _process(c).get('af_profiles') or {})
        if not gaps:
            return ''
        names = [_sample_name(c, key) for key in gaps]
        shown = ', '.join(names[:5]) + (f' and {len(names) - 5} more' if len(names) > 5 else '')
        return (f'{len(gaps)} selected sample(s) are not assigned the AF profile(s) the thresholds were set with '
                f'({", ".join(self.state.af_profile_names)}): {shown}. They are added to these samples\' AF '
                f'library for the export; check the assignments in the AutoSpectral AF tab.')

    def export_selection(self) -> list:
        """Selected samples that can be exported (all of them by default)."""
        eligible = [key for key, _name, ok in self.export_candidates() if ok]
        chosen = self.state.export_samples
        if chosen is None:
            return eligible
        chosen = set(chosen)
        return [key for key in eligible if key in chosen]

    def set_export_include_controls(self, include: bool):
        """List (and select) the single-stain controls and unstained samples
        for export, or hide them."""
        state = self.state
        state.settings['export_include_controls'] = bool(include)
        if include and state.export_samples is not None:
            controls = fcs.control_sample_keys(self.controller.experiment.samples)
            eligible = [key for key, _name, ok in self.export_candidates() if ok and key in controls]
            state.export_samples = list(dict.fromkeys(list(state.export_samples) + eligible))
        self.state_changed()

    def export_folder(self) -> Path:
        folder = self.state.output_folder
        return Path(folder) if folder else Path(self.controller.experiment_dir) / EXPORT_SUBDIR

    def variants_available(self) -> bool:
        """Kernel built and at least one Active fluorophore with variants."""
        if not fcc.kernel_available():
            return False
        fcc._ensure_bundled_plugins_on_path()
        from autospectral_optimization_functions import active_variant_labels, has_active_variants
        variants = getattr(self.controller, 'autospectral_variants', None) or {}
        return has_active_variants(variants, active_variant_labels(_process(self.controller)))

    def use_variants(self) -> bool:
        value = self.state.settings.get('use_variants')
        available = self.variants_available()
        return available if value is None else bool(value) and available

    def variants_note(self) -> str:
        if not fcc.kernel_available():
            return ('The compiled kernel is not built: the final unmix uses per-cell AF only '
                    "(Honeychrome's Python implementation).")
        if not self.variants_available():
            return ('No Active fluorophore has spectral variants: run Setup in the AutoSpectral Optimization '
                    'tab to use them. The final unmix uses per-cell AF only.')
        return ('Variants of the Active fluorophores in the AutoSpectral Optimization tab, with its positivity '
                'thresholds. The run time is estimated before the export starts.')

    def tandem_candidates(self) -> list:
        """``fcc.tandem_candidates`` of the current tags and spectral model
        ([] while the tags are incomplete)."""
        c = self.controller
        if c is None:
            return []
        try:
            design = self.design()
            spectra = _reference_spectra(c)
            if spectra is None:
                return []
            return fcc.tandem_candidates(design, _fluor_names(c), spectra, _detectors(c), _cytometer_db_col(c))
        except (ValueError, KeyError, OSError) as exc:
            logger.warning(f'FlowCode: tandem tags unavailable: {exc}')
            return []

    def tandem_selection(self, candidates=None) -> list:
        """Ticked tandem tag fluorophores: the saved choice, or by default
        those of class "Protein tandem" with a usable parent."""
        candidates = self.tandem_candidates() if candidates is None else candidates
        usable = [c['fluor'] for c in candidates if c['tandem'] is not None]
        chosen = self.state.settings.get('tandem_tags')
        if chosen is None:
            return [c['fluor'] for c in candidates
                    if c['tandem'] is not None and c['fluor_class'].lower() == 'protein tandem']
        return [f for f in usable if f in chosen]

    def set_tandem_tag(self, fluor: str, checked: bool):
        candidates = self.tandem_candidates()
        ticked = set(self.tandem_selection(candidates))
        if checked:
            ticked.add(fluor)
        else:
            ticked.discard(fluor)
        self.on_setting_changed('tandem_tags', [c['fluor'] for c in candidates if c['fluor'] in ticked])

    def export_tandems(self, candidates=None) -> tuple:
        """(ticked ``TandemTag`` list, every usable ``TandemTag``)."""
        candidates = self.tandem_candidates() if candidates is None else candidates
        pool = [c['tandem'] for c in candidates if c['tandem'] is not None]
        ticked = set(self.tandem_selection(candidates))
        return [t for t in pool if t.fluor in ticked], pool

    def tandem_reference_text(self, candidates=None) -> str:
        """The backbone statistic of each ticked tandem tag, from the saved
        characterisation."""
        candidates = self.tandem_candidates() if candidates is None else candidates
        ticked, _pool = self.export_tandems(candidates)
        if not ticked:
            return ''
        fc = self.saved_spectra()
        reference = ((fc.diagnostics if fc is not None else None) or {}).get('tandem') or {}
        lines = []
        for t in ticked:
            tag = next(c['tag'] for c in candidates if c['fluor'] == t.fluor)
            ref = reference.get(t.fluor)
            if ref is None or ref.get('parent') != t.parent:
                lines.append(f'{tag}: no backbone statistic; run section 4 again.')
                continue
            lines.append(f'{tag}: backbone {t.parent} per unit {tag} {_fmt_beta(ref.get("beta"))} after the '
                         f'correction ({ref.get("n_combos", 0)} combinations with the tag, '
                         f'{ref.get("n_null", 0)} without); {_fmt_beta(ref.get("beta_uncorrected"))} before it, '
                         f'against the reference spectra.')
        return '<br>'.join(lines)

    def kernel_threads(self) -> int:
        value = self.state.settings.get('threads')
        return int(value) if value else max(1, (os.cpu_count() or 2) - 1)

    def kernel_kwargs(self) -> dict:
        """Settings of the final kernel unmix (unmix_autospectral_optimization())."""
        s = self.state.settings
        return dict(n_passes=int(s['n_passes']), n_threads=self.kernel_threads(), cell_weight=self.cell_weight(),
                    noise_floor=float(s['noise_floor']), alpha=float(s['alpha']),
                    collinear_thresh=float(s['collinear_threshold']),
                    joint_pair_resolution=bool(s['joint_pair_resolution']), n_af_passes=int(s['n_af_passes']),
                    refine_af_quantile=float(s['refine_af_quantile']))

    def final_unmix(self, spectra: np.ndarray, names) -> tuple:
        """The export's final unmix as ``fn(raw fluorescence, AF spectra)`` and
        a short description of it: the compiled kernel, with the Active
        fluorophores' variants when used and cell.weight when ticked; without
        the kernel, Honeychrome's Python per-cell AF. Reads the controller,
        so call it on the main thread."""
        names = list(names)
        cell_weight = self.cell_weight()
        if not fcc.kernel_available():
            def python_fn(raw_fl, af_spectra):
                return fcc.af_only_unmix(raw_fl, spectra, names, af_spectra, cell_weight=cell_weight)
            return python_fn, 'per-cell AF, Python'

        fcc._ensure_bundled_plugins_on_path()
        from autospectral_optimization_functions import (
            active_variant_labels, cached_unmixed_thresholds, make_optimization_unmix_fn,
        )
        c = self.controller
        use_variants = self.use_variants()
        variants = dict(getattr(c, 'autospectral_variants', None) or {}) if use_variants else {}
        active = active_variant_labels(_process(c)) if use_variants else set()
        positivity = cached_unmixed_thresholds(c, spectra) if use_variants else np.zeros(len(names))
        kernel_fn = make_optimization_unmix_fn(spectra, names, variants, active, positivity, self.kernel_kwargs())

        def kernel_final_fn(raw_fl, af_spectra):
            return kernel_fn(raw_fl, None, SimpleNamespace(spectra=af_spectra))
        engine = 'kernel, ' + ('per-cell variants' if use_variants else 'no variants')
        return kernel_final_fn, engine + (', cell.weight' if cell_weight else '')

    def export_spectra(self):
        """The saved characterisation with the current combo names, checked
        against the current spectral model. Raises ValueError."""
        fc = self.saved_spectra()
        if fc is None:
            raise ValueError('The saved characterisation was not found: run section 4 again.')
        fc = fcc.rename_combos(fc, self.design())
        fcc.check_spectra_match(fc, _reference_spectra(self.controller), _fluor_names(self.controller),
                                _detectors(self.controller))
        return fc

    def export_problem(self) -> str | None:
        """Why the export cannot start now, or None."""
        try:
            self.export_spectra()
        except ValueError as exc:
            return str(exc)
        if not self.export_selection():
            return 'Select at least one sample.'
        return None

    def export_running(self) -> bool:
        return self._export_run is not None

    def export_result_text(self) -> str:
        last = self.state.last_export
        if not last:
            return ''
        text = (f'Last export ({last.get("time", "")}): {last.get("n_samples", 0)} sample(s), '
                f'{last.get("barcoded", 0):,} of {last.get("events", 0):,} events barcoded, to '
                f'<tt>{last.get("folder", "")}</tt>. Event counts per sample and combination are in '
                f'<tt>{SUMMARY_FILE}</tt>.')
        if last.get('tandem_tags'):
            text += (f' Tandem breakdown ({", ".join(last["tandem_tags"])}) was corrected in '
                     f'{last.get("tandem_corrected", 0)} sample(s); the estimates are in the same file.')
        return text

    def start_export(self):
        """Estimate, confirm and run the FlowCode export of the selected
        samples (JointUnmixExporter), writing the summary files when done."""
        if self.export_running() or not self.section_ready(5):
            return
        page = self.pages[5]
        problem = self.export_problem()
        if problem:
            page.set_run_status(problem, problem=True)
            return
        fcc._ensure_bundled_plugins_on_path()
        import joint_unmix_export as jue

        c = self.controller
        state = self.state
        try:
            fc = self.export_spectra()
            layout = jue.build_export_layout(c)
        except (ValueError, KeyError) as exc:
            page.set_run_status(f'Could not prepare the export: {exc}', problem=True)
            return
        sample_paths = self.export_selection()
        layout = replace(layout, sample_af_profiles=fcs.export_af_assignments(
            layout.sample_af_profiles, sample_paths, state.af_profile_names))
        design = fc.design
        names = list(layout.fluor_names)
        spectra = np.asarray(layout.reference_spectra, dtype=np.float64)
        cell_weight = self.cell_weight()
        use_variants = self.use_variants()
        kernel = fcc.kernel_available()
        final, _engine = self.final_unmix(spectra, names)

        def final_fn(raw_fl, af):
            return final(raw_fl, af.spectra)

        tandems, tandem_pool = self.export_tandems()
        job = {'design': design, 'folder': self.export_folder(), 'current': None, 'samples': [],
               'inputs': state.section_inputs(5, self.fingerprint()), 'use_variants': use_variants,
               'engine': 'kernel' if kernel else 'python', 'fc': fc, 'tandems': tandems, 'tandem': None,
               'af_profiles': {p: list(layout.sample_af_profiles.get(p) or []) for p in sample_paths}}

        def on_block(summary):
            job['current'] = fcc.add_summaries(job['current'], summary)

        def on_tandem(records):
            job['tandem'] = records

        def sample_done(sample_path, n_events):
            empty = fcc.debarcode_summary(np.zeros(0, dtype=np.int64), design)
            summary = job['current'] or empty
            if tandems:
                summary = dict(summary, tandem=job['tandem'] or [])
            job['samples'].append((sample_path, summary))
            job['current'] = None
            job['tandem'] = None

        records = [state.gates[name] for name in fcs.GATE_NAMES if name in state.gates]
        unmix_fn = fcc.staged_export_fn(fc, spectra, names, final_fn, records, layout.pnn_raw_export,
                                        dict(state.settings), cell_weight=cell_weight, on_block=on_block,
                                        tandems=tandems, tandem_pool=tandem_pool, on_tandem=on_tandem)
        extra_channels = [fcc.FLOWCODE_CHANNEL] + [fcc.combo_channel_name(n) for n in design.combo_names]
        keywords = {
            'FLOWCODEUNMIX': fcs.PLUGIN_VERSION,
            'FLOWCODECOMBOS': ';'.join(f'{fcc.combo_channel_name(n)}={fcc.combo_key(t)}'
                                       for n, t in zip(design.combo_names, design.combo_tags)),
        }
        exporter = jue.JointUnmixExporter(
            sample_paths, layout, unmix_fn, bus=self.bus, extra_channels=extra_channels,
            output_root=job['folder'], name_suffix=EXPORT_NAME_SUFFIX, unmixing_method='FlowCode',
            extra_keywords=keywords, sample_done=sample_done,
            prepare_fn=unmix_fn.prepare if tandems else None,
        )
        probe_fn = final_fn if kernel and use_variants else None
        run = jue.JointUnmixRun(exporter, parent_widget=self, probe_fn=probe_fn, bus=self.bus)
        run.status.connect(page.set_run_status)
        run.progress.connect(self._on_export_progress)
        run.finished.connect(self._on_export_finished)
        run.error.connect(self._on_export_error)
        run.done.connect(self._on_export_done)
        self._export_run = run
        self._export_job = job
        page.set_run_status(f'Exporting {len(sample_paths)} sample(s)…')
        self.refresh()
        run.start()

    def _on_export_progress(self, n, total):
        self.pages[5].set_run_status(f'Unmixing and exporting… ({n}/{total} samples)')
        if self.bus is not None and total:
            self.bus.progress.emit(n, total)

    def _on_export_finished(self):
        job = self._export_job
        page = self.pages[5]
        if job is None:
            return
        state = self.state
        c = self.controller
        rows = [(_sample_name(c, key), summary) for key, summary in job['samples']]
        folder = job['folder']
        try:
            folder.mkdir(parents=True, exist_ok=True)
            fcc.write_debarcode_summary_csv(folder / SUMMARY_FILE, rows, job['design'])
            with open(folder / RUN_FILE, 'w', encoding='utf-8') as fh:
                json.dump(self._run_record(job), fh, indent=2,
                          default=lambda v: v.item() if hasattr(v, 'item') else str(v))
        except OSError as exc:
            page.set_run_status(f'Exported, but the summary files could not be written: {exc}')
            return
        events = sum(s['total_events'] for _n, s in rows)
        barcoded = sum(sum(s['per_combo'].values()) for _n, s in rows)
        tandem_corrected = sum(1 for _n, s in rows if any(r['beta'] for r in s.get('tandem') or ()))
        state.last_export = {'time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'folder': str(folder),
                             'n_samples': len(rows), 'events': int(events), 'barcoded': int(barcoded),
                             'tandem_tags': [t.fluor for t in job['tandems']],
                             'tandem_corrected': int(tandem_corrected)}
        if job['inputs'] == state.section_inputs(5, self.fingerprint()):
            state.mark_completed(5, self.fingerprint())
        page.set_run_status(f'Done: {len(rows)} sample(s) exported.')
        self.state_changed()

    def _run_record(self, job) -> dict:
        """FlowCode_run.json: what the export used."""
        from honeychrome import __version__ as hc_version

        state = self.state
        fc = job['fc']
        design = job['design']
        settings = {k: state.settings.get(k) for k in fcs.SECTION4_SETTINGS + fcs.SECTION5_SETTINGS}
        settings['cell_weight'] = self.cell_weight()
        settings['threads'] = self.kernel_threads()
        return {
            'plugin_version': fcs.PLUGIN_VERSION,
            'honeychrome_version': hc_version,
            'time': datetime.now().isoformat(timespec='seconds'),
            'backbone': state.backbones(),
            'matched_unstained': state.unstained_key,
            'thresholds': [{'Fluor': f, 'Tag': t, 'Threshold_Raw': float(r)}
                           for f, t, r in zip(design.fluors, design.tags, fc.thresholds)],
            'combinations': [{'Id': n, 'tags': list(t), 'channel': fcc.combo_channel_name(n),
                              'error': 'combo' if fc.combo_error[g] is not None else 'pairwise'}
                             for g, (n, t) in enumerate(zip(design.combo_names, design.combo_tags))],
            'settings': settings,
            'use_variants': job['use_variants'],
            'final_unmix': job['engine'],
            'fingerprint': fc.fingerprint(),
            'tandem_tags': [{'tag': design.tags[t.tag], 'fluor': t.fluor, 'parent': t.parent, 'source': t.source,
                             'reference': {k: v for k, v in ((fc.diagnostics or {}).get('tandem') or {})
                                           .get(t.fluor, {}).items() if k not in ('combo_beta', 'combo_null')}}
                            for t in job['tandems']],
            'af_profiles': job.get('af_profiles') or {},
            'samples': [{'sample': key, **summary} for key, summary in job['samples']],
        }

    def _on_export_error(self, message: str):
        self.pages[5].set_run_status(f'Error: {message}')
        if self.bus is not None:
            self.bus.statusMessage.emit(f'FlowCode export: {message}')

    def _on_export_done(self):
        self._export_run = None
        self._export_job = None
        self.refresh()

    # -- background tasks ----------------------------------------------------------

    def task_running(self, key: str) -> bool:
        return key in self._tasks

    def _start_task(self, key: str, fn) -> bool:
        if key in self._tasks:
            return False
        thread = QThread(self)
        worker = _TaskWorker(key, fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_task_finished)
        worker.error.connect(self._on_task_error)
        worker.progress.connect(self._on_task_progress)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._reap_threads)
        self._tasks[key] = worker
        self._task_experiment[key] = self._experiment_key
        self._threads.append(thread)
        thread.start()
        return True

    def _reap_threads(self):
        for thread in [t for t in self._threads if t.isFinished()]:
            self._threads.remove(thread)
            thread.deleteLater()

    def _on_task_progress(self, key, n, total, text):
        if self.bus is not None and total:
            self.bus.progress.emit(n, total)
        if key == 'load' and text:
            self.pages[3].load_status.setText(f'Reading the {text}…')
        elif key == 'match' and text:
            self.pages[1].match_status.setText(f'Matching scatter… ({n}/{total})')
        elif key == 'characterise' and text:
            self.pages[4].run_status.setText(f'Characterising the backbone: {text}…')

    def _on_task_finished(self, key, result):
        self._tasks.pop(key, None)
        if self._task_experiment.pop(key, None) != self._experiment_key:
            return
        if key == 'match':
            self.apply_matching_scores(*result)
        elif key == 'load':
            self._on_load_finished(result)
        elif key == 'auto_cells':
            self._on_auto_cells_finished(result)
        elif key == 'space':
            self._on_space_finished(result)
        elif key == 'characterise':
            self._on_characterise_finished(result)
        self.refresh()

    def _on_task_error(self, key, message):
        self._tasks.pop(key, None)
        if self._task_experiment.pop(key, None) != self._experiment_key:
            return
        if key == 'match':
            self.pages[1].match_status.setText(f'Matching failed: {message}')
        elif key == 'load':
            self.pages[3].load_status.setText(f'Could not load the events: {message}')
        elif key == 'auto_cells':
            self.pages[3].gate_controls[fcs.CELLS_GATE][2].setText(f'Automatic gate failed: {message}')
        elif key == 'space':
            self.pages[3].confirm_status.setText(f'Error: {message}')
        elif key == 'characterise':
            self.pages[4].run_status.setText(f'Error: {message}')
        if key in ('load', 'space', 'auto_cells') and self._after_space is not None:
            self._after_space = None
            self.pages[4].run_status.setText(f'Could not reload the gated backbone: {message}')
        if self.bus is not None:
            self.bus.statusMessage.emit(f'FlowCode: {message}')
        self.refresh()
