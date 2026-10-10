"""
flowcode_plots.py
-----------------
Matplotlib figures for the FlowCode plugin's Debarcode & FRET section. No Qt:
each builder returns a ``matplotlib.figure.Figure`` without a canvas.

* ``spillover_heatmap_figure``  combos x channels unmixing error per unit of
                                on-tag abundance (``flowcode_core.spillover_matrix``)
* ``panel_expression_figure``   every panel marker on untransduced events and
                                on barcoded events before and after the
                                FlowCode correction
* ``combo_frequency_figure``    backbone events per combo
* ``message_figure``            a short text in place of a plot
"""

from __future__ import annotations

import numpy as np

ON_COLOUR = '#E07B39'
BAR_COLOUR = '#4477AA'
GROUP_COLOURS = {'untransduced': '#8C8C8C', 'before': '#CC6677', 'after': '#4477AA'}
GROUP_TITLES = {'untransduced': 'Combo\u2212 (untransduced)', 'before': 'Combo+ uncorrected',
                'after': 'Combo+ corrected'}


def _theme(fig, is_dark: bool, axes=None) -> str:
    """Honeychrome's light/dark figure styling; returns the foreground colour."""
    try:
        from honeychrome.view_components.differential_plots import style_figure_theme
    except ImportError:
        style_figure_theme = None
    if style_figure_theme is not None:
        return style_figure_theme(fig, is_dark, axes=axes)
    fg = 'white' if is_dark else 'black'
    fig.patch.set_facecolor('#1e1e1e' if is_dark else 'white')
    return fg


def message_figure(text: str, is_dark: bool = False, width: float = 6.0):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, 1.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    fg = _theme(fig, is_dark, axes=[ax])
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, color=fg, wrap=True)
    return fig


# ---------------------------------------------------------------------------
# Spillover heatmap
# ---------------------------------------------------------------------------

def symmetric_limit(values: np.ndarray, quantile: float = 0.99) -> float:
    """Colour limit: the *quantile* of |values| (finite), never zero."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    lim = float(np.quantile(v, quantile))
    return lim if lim > 0 else max(float(v.max()), 1e-6)


def spillover_heatmap_figure(matrix: dict, column_labels: dict | None = None, space: str = 'unmixed',
                             is_dark: bool = False):
    """
    Heatmap of ``flowcode_core.spillover_matrix`` output (the error slope at
    each combo's median abundances): diverging colours
    centred on 0, limits at the 99th percentile of |value|, the combo's own
    tag channels outlined. Values are printed in the cells when the grid is
    small enough to read.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    rows, columns = matrix['rows'], matrix['columns']
    values = np.asarray(matrix['values'], dtype=float)
    if not rows:
        return message_figure('No combo had enough events to characterise; every combo uses the '
                              'pairwise model.', is_dark)
    labels = [(column_labels or {}).get(c, c) for c in columns]
    n_rows, n_cols = values.shape
    cell = 0.32 if n_rows <= 40 else 0.2
    width = min(max(6.0, 0.34 * n_cols + 3.0), 30.0)
    height = max(3.0, cell * n_rows + 2.2)
    fig = Figure(figsize=(width, height), dpi=100, constrained_layout=True)
    ax = fig.add_subplot(111)
    lim = symmetric_limit(values)
    image = ax.imshow(values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
    fg = _theme(fig, is_dark, axes=[ax])

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=7 if n_rows <= 40 else 5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    on = np.asarray(matrix.get('on'), dtype=bool) if matrix.get('on') is not None else np.zeros_like(values, bool)
    for r, c in zip(*np.nonzero(on)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=fg, linewidth=1.2))

    if n_rows * n_cols <= 400:
        for r in range(n_rows):
            for c in range(n_cols):
                v = values[r, c]
                if not np.isfinite(v):
                    continue
                text = f'{v:.1f}'
                if text == '-0.0':
                    text = '0.0'
                strong = abs(v) > 0.6 * lim
                ax.text(c, r, text, ha='center', va='center', fontsize=6,
                        color='white' if strong else 'black')

    unit = ('% of on-tag abundance' if space == 'unmixed' else '% of on-tag spectrum peak')
    bar = fig.colorbar(image, ax=ax, shrink=min(1.0, 4.0 / height + 0.2), pad=0.01)
    bar.set_label(f'Error per unit on-tag at the median ({unit})', color=fg, fontsize=8)
    bar.ax.tick_params(colors=fg, labelsize=7)
    title = 'Unmixed-space error' if space == 'unmixed' else 'Detector-space error'
    ax.set_title(title, fontsize=10)
    if matrix.get('pairwise_combos'):
        names = matrix['pairwise_combos']
        shown = ', '.join(names[:12]) + (f' and {len(names) - 12} more' if len(names) > 12 else '')
        ax.set_xlabel(f'Pairwise model used for: {shown}', fontsize=7, color=fg)
    return fig


# ---------------------------------------------------------------------------
# Panel expression before and after correction
# ---------------------------------------------------------------------------

def panel_expression_figure(data: dict, is_dark: bool = False):
    """
    One row per panel marker and three columns: untransduced events,
    barcoded events before correction and the same events after it
    (``flowcode_core.panel_expression_data``). Each panel is a smoothed
    density scaled to a peak of 1 on the marker's display scale; the dashed
    line marks the untransduced median, so a shift of the barcoded events
    away from it is FlowCode signal reaching that marker. Tag rows are
    labelled in orange.
    """
    from matplotlib.figure import Figure

    markers = list(data['markers'])
    groups = [g for g in ('untransduced', 'before', 'after') if g in data['traces']]
    if not markers or not groups:
        return message_figure('No events to plot.', is_dark)
    n_rows = len(markers)
    row_h = 0.42
    height = n_rows * row_h + 1.0
    width = 1.7 + 2.3 * len(groups)
    fig = Figure(figsize=(width, height), dpi=100)
    fig.subplots_adjust(left=1.6 / width, right=1.0 - 0.15 / width, top=1.0 - 0.55 / height,
                        bottom=0.35 / height, hspace=0.9, wspace=0.08)
    axes = fig.subplots(n_rows, len(groups), squeeze=False)
    fg = _theme(fig, is_dark, axes=list(axes.ravel()))
    neutral = '#BBBBBB' if is_dark else '#555555'

    for r, marker in enumerate(markers):
        lo, hi = data['limits'][r]
        centres = data['centres'][r]
        reference = data['medians']['untransduced'][r] if 'untransduced' in data['medians'] else float('nan')
        ticks = data['ticks'][r]
        for c, group in enumerate(groups):
            ax = axes[r, c]
            trace = data['traces'][group][r]
            if trace is not None:
                colour = GROUP_COLOURS[group]
                ax.fill_between(centres, 0.0, trace, color=colour, alpha=0.45, linewidth=0)
                ax.plot(centres, trace, color=colour, linewidth=0.8)
            if np.isfinite(reference):
                ax.axvline(reference, color=neutral, linewidth=0.6, linestyle='--')
            ax.set_xlim(lo, hi)
            ax.set_ylim(0.0, 1.08)
            ax.set_yticks([])
            ax.grid(False)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            if ticks:
                ax.set_xticks([p for p, _l in ticks])
                ax.set_xticklabels([lab for _p, lab in ticks], fontsize=5)
            ax.tick_params(axis='x', length=2, pad=1, labelsize=5, colors=fg)
            if r == 0:
                ax.set_title(f'{GROUP_TITLES[group]}\n{data["n"][group]:,} events', fontsize=8, color=fg)
        label_ax = axes[r, 0]
        label_ax.set_ylabel(marker, rotation=0, ha='right', va='center', fontsize=7, labelpad=6,
                            color=ON_COLOUR if data['is_tag'][r] else fg)
    fig.suptitle('Panel expression: dashed line = untransduced median; orange labels = FlowCode tags',
                 fontsize=8, color=fg, y=1.0 - 0.12 / height)
    return fig


# ---------------------------------------------------------------------------
# Combo frequencies
# ---------------------------------------------------------------------------

def combo_frequency_figure(names: list, counts: list, totals: dict, is_dark: bool = False):
    """
    Bars of each combo's share of the validly barcoded backbone events (%),
    labelled with the count; horizontal above 20 combos. The footer gives
    untransduced, unresolved and rescued events.

    totals : ``total_events``, ``untransduced``, ``unresolved``, ``rescued``
             (too many tags), ``rescued_toofew``.
    """
    from matplotlib.figure import Figure

    counts = np.asarray(counts, dtype=float)
    valid = counts.sum()
    total = float(totals.get('total_events') or 0)
    if len(names) == 0:
        return message_figure('No combinations.', is_dark)
    pct = 100.0 * counts / valid if valid > 0 else np.zeros_like(counts)
    horizontal = len(names) > 20
    if horizontal:
        fig = Figure(figsize=(7.5, max(3.0, 0.2 * len(names) + 1.6)), dpi=100, constrained_layout=True)
    else:
        fig = Figure(figsize=(max(5.0, 0.5 * len(names) + 2.0), 4.2), dpi=100, constrained_layout=True)
    ax = fig.add_subplot(111)
    fg = _theme(fig, is_dark, axes=[ax])
    positions = np.arange(len(names))
    if horizontal:
        ax.barh(positions, pct, color=BAR_COLOUR)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel('% of barcoded events', fontsize=8)
        for p, v, c in zip(positions, pct, counts):
            ax.text(v, p, f' {int(c):,}', va='center', ha='left', fontsize=5, color=fg)
        ax.set_xlim(0, max(pct.max() * 1.25, 1.0))
    else:
        ax.bar(positions, pct, color=BAR_COLOUR)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('% of barcoded events', fontsize=8)
        for p, v, c in zip(positions, pct, counts):
            ax.text(p, v, f'{int(c):,}', va='bottom', ha='center', fontsize=6, color=fg)
        ax.set_ylim(0, max(pct.max() * 1.2, 1.0))
    ax.tick_params(labelsize=7)

    def share(n):
        return f'{int(n):,} ({100.0 * n / total:.1f}%)' if total > 0 else f'{int(n):,}'

    ax.set_title(f'Combo frequencies: {int(valid):,} barcoded of {int(total):,} events', fontsize=10)
    fig.supxlabel(f'Untransduced {share(totals.get("untransduced", 0))}; unresolved '
                  f'{share(totals.get("unresolved", 0))}; rescued {int(totals.get("rescued", 0)):,} from too '
                  f'many and {int(totals.get("rescued_toofew", 0)):,} from too few tags',
                  fontsize=8, color=fg)
    return fig
