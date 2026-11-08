"""
flowcode_shift_plots.py
-----------------------
Figures of the FlowCode phenotype-shift readout (flowcode_phenotype_shift.py),
one population and sample group at a time. No Qt.

    shift_figure        guides by signed shift angle (x) and −log10 p (y),
                        coloured by the cluster each guide gains most, sized
                        by its cells per mouse
    direction_figure    guides on the first two principal components of
                        their mean displacements, with an arrow per cluster:
                        guides pointing the same way share a phenotype effect
    heatmap_figure      significant guides × clusters (log2 ratio of the
                        guide's share to the reference's), above each
                        cluster's marker medians so clusters read as
                        phenotypes
    composition_figure  one guide's cluster shares in each mouse beside its
                        reference's
    summary_heatmap_figure
                        every guide's shift angle in every group (and
                        population), as the Analysis page's comparison
                        heatmap

Figures that can be double-clicked carry ``fig._points`` (x, y, guide) and
``fig._point_axes``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flowcode_phenotype_shift as fsh

MAX_LABELS = 25
MAX_HEATMAP_ROWS = 60


def cluster_colours(k: int) -> dict:
    """{cluster name: colour} for *k* clusters (tab20, then tab20b)."""
    from matplotlib import colormaps

    base = list(colormaps['tab20'].colors) + list(colormaps['tab20b'].colors)
    return {fsh.cluster_name(i): base[i % len(base)] for i in range(k)}


def cluster_descriptions(medians: pd.DataFrame, labels: dict | None = None, top: int = 3,
                         margin: float = 0.1) -> dict:
    """{cluster: 'C3: CD44 hi, Ly-6C hi'}: up to *top* channels whose median
    in the cluster is above the median over clusters by more than *margin*
    (display scale), brightest first; 'C3: low' when none."""
    labels = labels or {}
    centre = medians.median(axis=0)
    out = {}
    for cluster, row in medians.iterrows():
        diff = (row - centre).sort_values(ascending=False)
        hi = [labels.get(ch, ch) for ch, v in diff.items() if v > margin][:top]
        out[cluster] = f"{cluster}: {', '.join(f'{h} hi' for h in hi) if hi else 'low'}"
    return out


def _message(text: str, title: str, is_dark: bool, run_label: str):
    from honeychrome.view_components import differential_plots as dp
    return dp.message_figure(text, title, is_dark, run_label)


def _subset(table: pd.DataFrame, population: str, group: str) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=getattr(table, 'columns', []))
    return table[(table['population'] == population) & (table['group'] == group)]


def _finish(fig, is_dark: bool, run_label: str):
    from honeychrome.view_components import differential_plots as dp
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def shift_figure(results: pd.DataFrame, population: str, group: str, k: int, is_dark: bool = False,
                 run_label: str = '', descriptions: dict | None = None, fdr: float = 0.1):
    """Signed shift angle against −log10 p; significant guides (q below
    *fdr*) are outlined and labelled (the strongest first, at most
    ``MAX_LABELS``)."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    title = f'Phenotype shift: {population}, {group}'
    sub = _subset(results, population, group)
    if sub.empty:
        return _message('No guide was tested in this group.', title, is_dark, run_label)
    colours = cluster_colours(k)
    fig = Figure(figsize=(7.5, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = sub['angle_deg'].to_numpy(float)
    y = -np.log10(np.maximum(sub['p'].to_numpy(float), 1e-300))
    cells = sub['median_cells'].to_numpy(float)
    size = 12 + 60 * np.sqrt(cells / max(cells.max(), 1.0))
    sig = sub['significant'].to_numpy(bool)
    fg = 'white' if is_dark else 'black'
    ax.scatter(x, y, s=size, c=[colours.get(c, '#7f7f7f') for c in sub['gain_cluster']],
               edgecolors=np.where(sig, fg, 'none'), linewidths=0.9, alpha=0.9)
    ax.axvline(0.0, color='#7f7f7f', linewidth=0.8)
    order = np.argsort(-y)
    labelled = [i for i in order if sig[i]][:MAX_LABELS]
    for i in labelled:
        ax.annotate(str(sub['guide'].iloc[i]), (x[i], y[i]), xytext=(3, 3), textcoords='offset points',
                    fontsize=7, color=fg)
    ax.set_xlabel('Shift angle (°; noise centres on 0, a shift is positive)', fontsize=9)
    ax.set_ylabel('−log10 p', fontsize=9)
    ax.grid(False)
    present = list(dict.fromkeys(sub.loc[sig, 'gain_cluster'])) or list(dict.fromkeys(sub['gain_cluster']))[:8]
    names = descriptions or {}
    handles = [Line2D([], [], marker='o', linestyle='', color=colours.get(c, '#7f7f7f'), label=names.get(c, c))
               for c in sorted(present, key=lambda c: int(c[1:]))]
    if handles:
        ax.legend(handles=handles, title='Cluster gained most', fontsize=7, title_fontsize=7, loc='upper left',
                  frameon=False)
    n_sig = int(sig.sum())
    ax.set_title(f'{title}\n{len(sub)} guides tested, {n_sig} with q < {fdr:g} (outlined); size = cells per mouse',
                 fontsize=10)
    fig._points = list(zip(x, y, sub['guide'].astype(str)))
    fig._point_axes = ax
    return _finish(fig, is_dark, run_label)


def _matrix(direction: pd.DataFrame, population: str, group: str, value: str, k: int) -> pd.DataFrame:
    sub = _subset(direction, population, group)
    if sub.empty:
        return pd.DataFrame()
    m = sub.pivot(index='guide', columns='cluster', values=value)
    return m.reindex(columns=[fsh.cluster_name(i) for i in range(k)])


def direction_figure(direction: pd.DataFrame, results: pd.DataFrame, population: str, group: str, k: int,
                     is_dark: bool = False, run_label: str = '', descriptions: dict | None = None):
    """Guides on the first two principal components of their mean
    displacements (uncentred: the origin is no shift), with each cluster's
    direction as an arrow."""
    from matplotlib.figure import Figure

    title = f'Shift directions: {population}, {group}'
    m = _matrix(direction, population, group, 'mean_displacement', k)
    if m.shape[0] < 3:
        return _message('Too few guides were tested to map their directions.', title, is_dark, run_label)
    v = np.nan_to_num(m.to_numpy(float))
    u, s, vt = np.linalg.svd(v, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = s[:2] ** 2 / max(float((s ** 2).sum()), 1e-12)
    sub = _subset(results, population, group).set_index('guide').reindex(m.index)
    sig = sub['significant'].fillna(False).to_numpy(bool)
    colours = cluster_colours(k)
    fg = 'white' if is_dark else 'black'
    fig = Figure(figsize=(7.0, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.scatter(scores[:, 0], scores[:, 1], s=np.where(sig, 40, 14),
               c=[colours.get(c, '#7f7f7f') if g else '#9a9a9a' for c, g in zip(sub['gain_cluster'], sig)],
               edgecolors=np.where(sig, fg, 'none'), linewidths=0.8, alpha=0.9, zorder=3)
    reach = float(np.abs(scores).max()) or 1.0
    load = vt[:2].T
    scale = 0.9 * reach / max(float(np.abs(load).max()), 1e-12)
    names = descriptions or {}
    for j, cluster in enumerate(m.columns):
        dx, dy = load[j] * scale
        if np.hypot(dx, dy) < 0.15 * reach:
            continue
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops={'arrowstyle': '->', 'color': colours.get(cluster, '#7f7f7f'), 'lw': 1.2})
        ax.text(dx * 1.06, dy * 1.06, names.get(cluster, cluster), fontsize=7, color=colours.get(cluster, fg),
                ha='center', va='center')
    order = np.argsort(-np.hypot(scores[:, 0], scores[:, 1]))
    for i in [i for i in order if sig[i]][:MAX_LABELS]:
        ax.annotate(str(m.index[i]), scores[i], xytext=(3, 3), textcoords='offset points', fontsize=7, color=fg)
    ax.axhline(0, color='#7f7f7f', linewidth=0.6)
    ax.axvline(0, color='#7f7f7f', linewidth=0.6)
    ax.set_xlim(-1.2 * reach, 1.2 * reach)
    ax.set_ylim(-1.2 * reach, 1.2 * reach)
    ax.set_xlabel(f'PC1 ({100 * explained[0]:.0f}%)', fontsize=9)
    ax.set_ylabel(f'PC2 ({100 * explained[1]:.0f}%)' if len(explained) > 1 else 'PC2', fontsize=9)
    ax.grid(False)
    ax.set_title(f'{title}\nguides pointing the same way share a phenotype effect; significant guides in colour',
                 fontsize=10)
    fig._points = list(zip(scores[:, 0], scores[:, 1], m.index.astype(str)))
    fig._point_axes = ax
    return _finish(fig, is_dark, run_label)


def heatmap_figure(direction: pd.DataFrame, results: pd.DataFrame, medians: pd.DataFrame, population: str,
                   group: str, k: int, is_dark: bool = False, run_label: str = '', labels: dict | None = None):
    """Significant guides (else the 20 with the smallest p) × clusters, the
    log2 ratio of the guide's mean share to the reference's, above the
    clusters' marker medians on the display scale."""
    from matplotlib.figure import Figure

    title = f'Shift heatmap: {population}, {group}'
    m = _matrix(direction, population, group, 'log2fc', k)
    sub = _subset(results, population, group)
    if m.empty or sub.empty:
        return _message('No guide was tested in this group.', title, is_dark, run_label)
    chosen = sub[sub['significant']].sort_values('p')
    note = 'significant guides'
    if chosen.empty:
        chosen = sub.sort_values('p').head(20)
        note = 'no significant guide; the 20 with the smallest p'
    chosen = chosen.head(MAX_HEATMAP_ROWS)
    m = m.reindex(chosen['guide'])
    med = medians.reindex(index=m.columns)
    labels = labels or {}
    rows_top, rows_bottom = len(m), med.shape[1]
    height = 1.6 + 0.18 * (rows_top + rows_bottom)
    fig = Figure(figsize=(max(5.5, 0.45 * k + 2.8), max(4.0, height)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[max(rows_top, 1), max(rows_bottom, 1)], width_ratios=[30, 1])
    ax = fig.add_subplot(grid[0, 0])
    v = m.to_numpy(float)
    lim = float(np.nanquantile(np.abs(v), 0.98)) if np.isfinite(v).any() else 1.0
    lim = lim if lim > 0 else 1.0
    im = ax.imshow(v, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax.set_yticks(range(rows_top))
    ax.set_yticklabels(m.index.astype(str), fontsize=7)
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([])
    ax.set_title(f'{title}\n{note}; log2 (guide share / reference share)', fontsize=10)
    fig.colorbar(im, cax=fig.add_subplot(grid[0, 1])).set_label('log2 ratio', fontsize=7)
    ax2 = fig.add_subplot(grid[1, 0])
    im2 = ax2.imshow(med.to_numpy(float).T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_yticks(range(rows_bottom))
    ax2.set_yticklabels([labels.get(c, c) for c in med.columns], fontsize=7)
    ax2.set_xticks(range(len(med.index)))
    ax2.set_xticklabels(list(med.index), fontsize=7)
    ax2.set_xlabel('Cluster (marker medians, display scale)', fontsize=8)
    fig.colorbar(im2, cax=fig.add_subplot(grid[1, 1])).set_label('median', fontsize=7)
    for a in (ax, ax2):
        a.grid(False)
    fig._points = None
    return _finish(fig, is_dark, run_label)


def composition_figure(comp: pd.DataFrame, guide: str, population: str, group: str, k: int,
                       sample_labels: dict | None = None, is_dark: bool = False, run_label: str = '',
                       descriptions: dict | None = None):
    """Per mouse, two stacked bars of cluster shares: the guide's (left,
    solid) and its reference's (right, faint)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    title = f'{guide}: cluster composition per mouse ({population}, {group})'
    if comp is None or comp.empty:
        return _message('This guide has no mouse with enough cells.', title, is_dark, run_label)
    colours = cluster_colours(k)
    clusters = [fsh.cluster_name(i) for i in range(k)]
    samples = list(dict.fromkeys(comp['sample']))
    sample_labels = sample_labels or {}
    fig = Figure(figsize=(max(5.5, 0.9 * len(samples) + 2.5), 4.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    width = 0.38
    for i, s in enumerate(samples):
        rows = comp[comp['sample'] == s].set_index('cluster').reindex(clusters)
        for offset, column, alpha in ((-width / 2, 'guide_share', 1.0), (width / 2, 'reference_share', 0.45)):
            bottom = 0.0
            for c in clusters:
                h = max(float(rows.loc[c, column]), 0.0) if np.isfinite(rows.loc[c, column]) else 0.0
                ax.bar(i + offset, h, width=width * 0.95, bottom=bottom, color=colours[c], alpha=alpha,
                       linewidth=0)
                bottom += h
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([f"{sample_labels.get(s, s)}\n({int(comp.loc[comp['sample'] == s, 'cells'].iloc[0])} cells)"
                        for s in samples], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Share of cells (left: guide; right, faint: reference)', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    names = descriptions or {}
    ax.legend(handles=[Patch(color=colours[c], label=names.get(c, c)) for c in clusters], fontsize=6,
              loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_title(title, fontsize=10)
    fig._points = None
    return _finish(fig, is_dark, run_label)


def summary_heatmap_figure(results: pd.DataFrame, populations: list, groups: list, is_dark: bool = False,
                           run_label: str = '', order: str = 'effect', n_rows: int = 1, fdr: float = 0.1):
    """The size of every guide's shift in every group: one row per group
    (with the population in brackets when there are several), one column per
    guide, coloured by the shift angle where the guide is significant (q
    below *fdr*) and left blank otherwise, drawn as the Analysis page's
    comparison heatmap (*order* 'effect' or 'alphabetical', *n_rows*
    panels)."""
    from honeychrome.view_components import differential_plots as dp

    title = f'Phenotype shift, all groups: shift angle of guides with q < {fdr:g}'
    if results is None or results.empty:
        return _message('No guide was tested.', title, is_dark, run_label)
    several = len(populations) > 1

    def row_name(pop, group):
        return f'{group} [{pop}]' if several else str(group)
    df = results[results['population'].isin(populations)]
    table = pd.DataFrame({'feature': df['guide'].astype(str),
                          'comparison': [row_name(p, g) for p, g in zip(df['population'], df['group'])],
                          'logFC': df['angle_deg'].astype(float), 'significant': df['significant'].astype(bool)})
    rows = [row_name(p, g) for p in populations for g in groups]
    fig = dp.make_comparison_heatmap_figure(table, title, is_dark=is_dark, run_label=run_label, order=order,
                                            n_rows=n_rows, value_label='shift angle (°)', comparison_order=rows)
    fig._points = None
    return fig


def calibration_text(calibration: pd.DataFrame | None) -> str:
    """One line per population and group of the label-shuffle check."""
    if calibration is None or calibration.empty:
        return ''
    lines = []
    for r in calibration.itertuples(index=False):
        lines.append(f'{r.population}, {r.group}: {r.pct_p_below_0_05:.1f}% of {r.tests} shuffled tests at '
                     f'p < 0.05; {r.shuffles_with_a_hit} of {r.shuffles} shuffles with a significant guide')
    return '\n'.join(lines)
